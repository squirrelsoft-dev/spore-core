"""Tests for the Ralph loop strategy (issue #58).

Mirrors ``rust/crates/spore-core/src/harness.rs`` Ralph unit tests and the
shared fixture at ``fixtures/harness/ralph.json``. Ralph is a
multi-context-window continuation loop: each window is a FRESH session
re-seeded with the instruction plus reloaded checkpoint state, driven by a
registered ``Stop`` hook reading the Ralph progress checkpoint (B1), bounded by
the ``max_resets`` outer cap (B3). Each test maps to one rule; the rule lives in
the docstring.

#142: the Ralph checkpoint MOVED off the ``.spore/`` filesystem onto the durable
project-id :class:`RunStore`. The test writer and the harness reader must SHARE
one store + project id so a write is what the read sees: ``_config`` pins
``project_id`` from the sandbox root, ``_write_progress`` writes into the SAME
store at the project namespace, and the agents hold that store so what they
write is what the harness reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spore_core import (
    AgentId,
    AlwaysContinuePolicy,
    BudgetLimits,
    Context,
    FinalResponse,
    HaltReasonRalphCompletionUnmet,
    HarnessConfig,
    HarnessRunOptions,
    RalphConfig,
    ReactConfig,
    RunResultFailure,
    RunResultSuccess,
    SandboxViolation,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    TokenUsage,
    ToolCall,
    ToolOutputSuccess,
    VcsError,
    VcsLogArgs,
)
from spore_core.agent import ToolCallRequested
from spore_core.harness import (
    BaseSandboxProvider,
    CommandOutput,
    FixtureVcsProvider,
    GitVcsProvider,
    NoopContextManager,
)
from spore_core.model import ModelParams
from spore_core.storage import (
    RALPH_FEATURE_LIST_KEY,
    RALPH_PROGRESS_KEY,
    InMemoryStorageProvider,
    RunStore,
    StorageProvider,
    project_id_from_canonical_path,
    project_namespace,
)

INCOMPLETE = '{"complete": false, "remaining": ["task A"]}'
COMPLETE = '{"complete": true, "remaining": []}'


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class WorkspaceSandbox(BaseSandboxProvider):
    """Allow-all sandbox whose ``workspace_root`` is a real tempdir. #142: the
    Ralph reload + completion check now read the project-id RunStore (not
    ``.spore/`` files), but the root is still the derivation source for the
    project id."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None

    def workspace_root(self) -> Path:
        return self._root


def _ralph_ns(root: Path):  # type: ignore[no-untyped-def]
    """The project namespace the Ralph checkpoint lives under (#142): the project
    id ``_config`` pins from ``root`` (matching the harness reader), projected
    onto the ``RunStore`` session-id axis. The test writer and the harness reader
    MUST agree on this so a write is what the read sees."""
    return project_namespace(project_id_from_canonical_path(str(root)))


async def _write_progress(store: RunStore, root: Path, body: str) -> None:
    """Write the Ralph progress checkpoint into the SHARED run store under the
    project namespace derived from ``root`` (#142 relocated this off the
    ``.spore/`` filesystem). ``body`` is the legacy JSON body string."""
    await store.put(_ralph_ns(root), RALPH_PROGRESS_KEY, json.loads(body))


async def _write_feature_list(store: RunStore, root: Path, body: str) -> None:
    await store.put(_ralph_ns(root), RALPH_FEATURE_LIST_KEY, json.loads(body))


class ProgressWritingAgent:
    """Agent that, on each turn, pops the next progress body from a queue and
    writes it to the SHARED run store under the project namespace BEFORE returning
    a ``FinalResponse`` — modelling "the agent did work this window and updated
    progress." Records the contexts it saw so tests can assert fresh-state /
    reload. #142: the Ralph worker holds the shared run store so what it writes is
    what the harness reads. Mirrors Rust's ``ProgressWritingAgent``."""

    def __init__(self, store: RunStore, root: Path, bodies: list[str]) -> None:
        self._id = AgentId("ralph-build")
        self._store = store
        self._ns = _ralph_ns(root)
        self._queue = list(bodies)
        self.seen: list[Context] = []

    @property
    def call_count(self) -> int:
        return len(self.seen)

    def seen_text(self) -> list[str]:
        out: list[str] = []
        for ctx in self.seen:
            texts: list[str] = []
            for m in ctx.messages:
                content = m.content
                texts.append(getattr(content, "text", ""))
            out.append(" | ".join(texts))
        return out

    async def turn(self, context: Context) -> FinalResponse:
        self.seen.append(context)
        if self._queue:
            await self._store.put(self._ns, RALPH_PROGRESS_KEY, json.loads(self._queue.pop(0)))
        return FinalResponse(
            content="window done",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def id(self) -> AgentId:
        return self._id


class ToolLoopingAgent:
    """An agent that NEVER returns a ``FinalResponse`` — every turn requests a
    tool call, so a bounded ReAct window can only ever exhaust its turn budget (it
    never terminates cleanly). On its FIRST turn it writes ``COMPLETE`` to the
    SHARED run store, so any code path that DID consult Ralph's external
    completion after the window would (wrongly) see "complete" and Success.
    Mirrors Rust's ``ToolLoopingAgent`` (#125 F5)."""

    def __init__(self, store: RunStore, root: Path) -> None:
        self._id = AgentId("ralph-tool-loop")
        self._store = store
        self._ns = _ralph_ns(root)
        self._calls = 0

    @property
    def call_count(self) -> int:
        return self._calls

    async def turn(self, context: Context) -> ToolCallRequested:
        if self._calls == 0:
            # Mark COMPLETE up front — if Ralph (wrongly) consulted completion
            # after a budget-exhausted window it would Success.
            await self._store.put(self._ns, RALPH_PROGRESS_KEY, json.loads(COMPLETE))
        n = self._calls
        self._calls += 1
        return ToolCallRequested(
            calls=[ToolCall(id=f"c{n}", name="x", input={})],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def id(self) -> AgentId:
        return self._id


class PassThroughContextManager:
    """Context manager that appends seeded user messages as real text messages
    so the agent's recorded context reflects what the window was re-seeded
    with (instruction + reloaded ``.spore/`` state)."""

    async def assemble(self, session: SessionState, task: Task, sources: object) -> Context:
        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: Any) -> None:
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        from spore_core.model import Message, Role, TextContent

        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    def should_compact(self, session: SessionState) -> bool:
        return False


def _new_storage() -> StorageProvider:
    """A fresh in-memory provider the test, the agent, and the harness all share
    (#142): the agent writes the checkpoint into it and the harness reads from the
    same store at the same project namespace."""
    return StorageProvider.single(InMemoryStorageProvider())


def _config(
    root: Path,
    agent: Any,
    storage: StorageProvider,
    *,
    max_resets: int = 3,
    context_manager: Any = None,
    vcs_provider: Any = None,
) -> HarnessConfig:
    """Build a Ralph harness config whose checkpoint store + project namespace are
    SHARED with ``storage`` (#142). The harness reads the checkpoint from
    ``cfg.storage`` at ``cfg.project_id``; this pins ``project_id`` from ``root``
    (the same root ``WorkspaceSandbox`` exposes), so a write the test/agent makes
    against ``storage`` at ``_ralph_ns(root)`` is exactly what the harness reads.
    ``agent`` already holds ``storage.run()``."""
    return HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=WorkspaceSandbox(root),
        context_manager=(context_manager if context_manager is not None else NoopContextManager()),
        termination_policy=AlwaysContinuePolicy(),
        max_resets=max_resets,
        vcs_provider=vcs_provider,
        storage=storage,
        # #142: pin the project id from the SAME root the sandbox exposes so the
        # harness reads back what the test/agent writes under ``_ralph_ns(root)``.
        project_id=project_id_from_canonical_path(str(root)),
    )


def _ralph_task() -> Task:
    # One ReAct turn per context window keeps the per-window sub-loop bounded so
    # the OUTER reset loop drives the test deterministically.
    return Task.new(
        "implement the thing",
        SessionId("ralph-session"),
        RalphConfig.simple(),
        budget=BudgetLimits(max_turns=1),
    )


# ---------------------------------------------------------------------------
# R0: Ralph is implemented — no longer StrategyNotYetImplemented.
# ---------------------------------------------------------------------------


async def test_r0_ralph_implemented(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, COMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [COMPLETE])
    h = StandardHarness(_config(tmp_path, agent, storage))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultSuccess)


# ---------------------------------------------------------------------------
# R4: incomplete,incomplete,complete → Success at iteration 3.
# ---------------------------------------------------------------------------


async def test_r4_resets_until_complete(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, INCOMPLETE, COMPLETE])
    h = StandardHarness(_config(tmp_path, agent, storage, max_resets=3))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultSuccess)
    # Exactly three context windows ran (one agent turn each).
    assert agent.call_count == 3


# ---------------------------------------------------------------------------
# R5: always-incomplete → exactly max_resets windows → RalphCompletionUnmet.
# ---------------------------------------------------------------------------


async def test_r5_exhausts_max_resets(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(
        storage.run(), tmp_path, [INCOMPLETE, INCOMPLETE, INCOMPLETE, INCOMPLETE]
    )
    h = StandardHarness(_config(tmp_path, agent, storage, max_resets=3))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonRalphCompletionUnmet)
    assert r.reason.iterations == 3
    assert "task A" in r.reason.last_reason
    # Exactly max_resets windows ran.
    assert agent.call_count == 3


async def test_r5_single_window_cap(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, INCOMPLETE])
    h = StandardHarness(_config(tmp_path, agent, storage, max_resets=1))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonRalphCompletionUnmet)
    assert r.reason.iterations == 1
    assert agent.call_count == 1


# ---------------------------------------------------------------------------
# F5 (#125): a Ralph window whose INNER LEAF exhausts its OWN budget mid-window
# (the leaf's ``PerLoop`` policy is the binding cap, no smaller global backstop).
# The window surfaces ``StrategyOutcomeBudgetExhausted``; Ralph must treat it as
# "window incomplete → RESET and retry" — it must NOT consult external completion
# (which is COMPLETE on disk here) and must NOT cascade the child's exhaustion
# into its own terminal. So the run reaches ``max_resets`` windows and ends
# ``RalphCompletionUnmet``, NOT Success.
#
# NOTE: the other Ralph tests use an unbounded leaf + a global ``max_turns`` cap
# (Deviation #14c) to AVOID this path; this test adds explicit coverage of the
# BOUNDED-leaf path.
# ---------------------------------------------------------------------------


async def test_ralph_budget_exhausted_window_resets_no_completion_no_cascade(
    tmp_path: Path,
) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ToolLoopingAgent(storage.run(), tmp_path)
    # Provide tool outputs for the looping calls across all windows.
    reg = ScriptedToolRegistry()
    for _ in range(32):
        reg.push(ToolOutputSuccess(content="ok", truncated=False))
    cfg = _config(tmp_path, agent, storage, max_resets=3)
    cfg.tool_registry = reg
    h = StandardHarness(cfg)

    # Inner leaf carries its OWN binding cap (PerLoop{2}); NO global max_turns
    # backstop, so the leaf policy — not the global cap — exhausts the window and
    # the new BudgetExhausted path is taken.
    task = Task.new(
        "implement the thing",
        SessionId("ralph-exhaust"),
        RalphConfig(inner=ReactConfig.per_loop(2), agent=""),
        budget=BudgetLimits(max_turns=None),
    )

    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure), (
        f"expected RalphCompletionUnmet (reset-on-exhaust, no completion, no cascade), got {r!r}"
    )
    assert isinstance(r.reason, HaltReasonRalphCompletionUnmet)
    # Reached max_resets — completion was NEVER consulted (despite COMPLETE on
    # disk) and the child's exhaustion did NOT cascade into Ralph's own terminal.
    assert r.reason.iterations == 3, "exactly max_resets windows ran"
    # Three windows × two leaf turns each = six agent turns total — proving each
    # exhausted window fully reset and re-ran, not short-circuited.
    assert agent.call_count == 6, "3 windows × 2 leaf turns each"


# ---------------------------------------------------------------------------
# R2: each reset builds a FRESH SessionState — no message carryover.
# ---------------------------------------------------------------------------


async def test_r2_fresh_session_per_reset(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, COMPLETE])
    h = StandardHarness(
        _config(tmp_path, agent, storage, max_resets=3, context_manager=PassThroughContextManager())
    )
    await h.run(HarnessRunOptions(_ralph_task()))
    texts = agent.seen_text()
    assert len(texts) == 2
    # Window 2's fresh context does NOT carry window 1's "window done" output.
    assert "window done" not in texts[1]


# ---------------------------------------------------------------------------
# R3: the filesystem reload injects `.spore/` state into the fresh seed.
# ---------------------------------------------------------------------------


async def test_r3_reload_injects_filesystem_state(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    await _write_feature_list(storage.run(), tmp_path, '[{"name":"login","passes":false}]')
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, COMPLETE])
    h = StandardHarness(
        _config(tmp_path, agent, storage, max_resets=3, context_manager=PassThroughContextManager())
    )
    await h.run(HarnessRunOptions(_ralph_task()))
    texts = agent.seen_text()
    # Window 1's fresh seed contains the reloaded progress + feature list. The
    # "Reloaded .spore/…" prefix is retained (#142) even though the checkpoint now
    # lives in the project-id RunStore.
    assert "Reloaded .spore/progress.json" in texts[0]
    assert "Reloaded .spore/feature_list.json" in texts[0]
    assert "login" in texts[0]


# ---------------------------------------------------------------------------
# R6: budgets fold across ALL context windows.
# ---------------------------------------------------------------------------


async def test_r6_budgets_fold_across_windows(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, INCOMPLETE, COMPLETE])
    h = StandardHarness(_config(tmp_path, agent, storage, max_resets=3))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultSuccess)
    # Three windows × one turn × (1 in, 1 out) folded.
    assert r.usage.input_tokens == 3
    assert r.usage.output_tokens == 3


# ---------------------------------------------------------------------------
# R7: the registered Stop hook is inert without a progress file — a plain
# ReAct run on a `.spore/`-free workspace terminates in one turn.
# ---------------------------------------------------------------------------


async def test_r7_stop_hook_inert_without_progress_file(tmp_path: Path) -> None:
    from spore_core import MockAgent

    a = MockAgent(AgentId("react"))
    a.push(FinalResponse(content="done", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    cfg = HarnessConfig(
        agent=a,
        tool_registry=ScriptedToolRegistry(),
        sandbox=WorkspaceSandbox(tmp_path),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )
    h = StandardHarness(cfg)
    task = Task.new("do", SessionId("s1"), ReactConfig.per_loop(5))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 1


# ---------------------------------------------------------------------------
# Completion-status helper: progress complete but a feature fails ⇒ still
# incomplete (the feature list corroborates).
# ---------------------------------------------------------------------------


async def test_completion_status_feature_list_gate() -> None:
    # #142: the checkpoint lives in the project-id RunStore, not on disk. The
    # completion-status helper reads ``(run_store, project)``; the test writes the
    # same checkpoint into the same store under the project namespace.
    from spore_core.harness import _ralph_completion_status

    storage = _new_storage()
    project = project_id_from_canonical_path("/feature-gate")
    ns = project_namespace(project)
    run = storage.run()
    await run.put(ns, RALPH_PROGRESS_KEY, json.loads(COMPLETE))
    await run.put(ns, RALPH_FEATURE_LIST_KEY, [{"name": "login", "passes": False}])
    status = await _ralph_completion_status(run, project)
    assert status is not None and "login" in status
    assert "incomplete features" in status
    # Now mark it passing — complete.
    await run.put(ns, RALPH_FEATURE_LIST_KEY, [{"name": "login", "passes": True}])
    assert await _ralph_completion_status(run, project) is None


async def test_completion_status_remaining_nonempty_is_incomplete() -> None:
    from spore_core.harness import _ralph_completion_status

    # complete:true but a non-empty remaining list ⇒ incomplete (fixture case
    # ``complete_but_remaining_nonempty_is_incomplete``).
    storage = _new_storage()
    project = project_id_from_canonical_path("/remaining-gate")
    run = storage.run()
    await run.put(
        project_namespace(project),
        RALPH_PROGRESS_KEY,
        {"complete": True, "remaining": ["leftover"]},
    )
    status = await _ralph_completion_status(run, project)
    assert status is not None and "leftover" in status


# ---------------------------------------------------------------------------
# Cross-language fixture replay against fixtures/harness/ralph.json.
# ---------------------------------------------------------------------------


def _fixture_path() -> Path:
    # python/packages/spore_core/tests/ -> repo root /fixtures/harness/ralph.json
    return Path(__file__).resolve().parents[4] / "fixtures" / "harness" / "ralph.json"


async def test_ralph_fixture_replay(tmp_path: Path) -> None:
    suite = json.loads(_fixture_path().read_text())
    for i, case in enumerate(suite["cases"]):
        case_dir = tmp_path / f"case_{i}"
        case_dir.mkdir()
        storage = _new_storage()
        # Seed an initial incomplete progress checkpoint so window 1 reloads state.
        await _write_progress(storage.run(), case_dir, INCOMPLETE)
        bodies = [
            json.dumps({"complete": w["complete"], "remaining": w.get("remaining", [])})
            for w in case["windows"]
        ]
        agent = ProgressWritingAgent(storage.run(), case_dir, bodies)
        # issue #58 v2: when the case carries a ``vcs_log``, wire a
        # FixtureVcsProvider seeded with it; absent ⇒ None ⇒ no git section. The
        # PassThroughContextManager records seeds so we can assert injection.
        vcs_log = case.get("vcs_log")
        vcs_provider = FixtureVcsProvider(vcs_log, "") if vcs_log is not None else None
        h = StandardHarness(
            _config(
                case_dir,
                agent,
                storage,
                max_resets=case["max_resets"],
                context_manager=PassThroughContextManager(),
                vcs_provider=vcs_provider,
            )
        )
        r = await h.run(HarnessRunOptions(_ralph_task()))
        # When a vcs_log is present, the first fresh window must include it.
        if vcs_log is not None:
            texts = agent.seen_text()
            assert any("Recent VCS history:" in t and vcs_log.strip() in t for t in texts), (
                f"case {case['name']}: vcs_log not injected into reload: {texts}"
            )
        expected = case["expected"]
        name = case["name"]
        if expected["kind"] == "success":
            assert isinstance(r, RunResultSuccess), f"case {name}: expected success, got {r}"
            assert agent.call_count == expected["iterations"], f"case {name}: window count"
        elif expected["kind"] == "completion_unmet":
            assert isinstance(r, RunResultFailure), f"case {name}: expected failure, got {r}"
            assert isinstance(r.reason, HaltReasonRalphCompletionUnmet), f"case {name}"
            assert r.reason.iterations == expected["iterations"], f"case {name}: iteration count"
        else:  # pragma: no cover - fixture schema guard
            raise AssertionError(f"case {name}: unknown expected kind {expected['kind']}")


# ---------------------------------------------------------------------------
# VcsProvider seam (issue #58 v2) — git-log reload for Ralph.
# ---------------------------------------------------------------------------


class CommandCapturingSandbox(BaseSandboxProvider):
    """Mock sandbox that records the (command, args, working_dir) of the last
    ``execute_command`` call and returns a canned :class:`CommandOutput`, so
    :class:`GitVcsProvider`'s argv construction is asserted without spawning a
    process."""

    def __init__(self, root: Path, output: CommandOutput | None = None) -> None:
        self._root = root
        self._output = output if output is not None else CommandOutput(stdout="ok", exit_code=0)
        self.command: str | None = None
        self.args: list[str] | None = None
        self.working_dir: Path | None = None

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None

    def workspace_root(self) -> Path:
        return self._root

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        self.command = command
        self.args = list(args)
        self.working_dir = working_dir
        return self._output


# (a) FixtureVcsProvider.log returns the seeded string verbatim; status() too.
async def test_vcs_fixture_log_verbatim() -> None:
    log = "cafe123 implement login\nbeef456 add login tests"
    provider = FixtureVcsProvider(log, "clean")
    out = await provider.log(VcsLogArgs(max_entries=20, since_ref="HEAD~5", format="%h %s"))
    assert out == log
    # status() round-trips its seeded string verbatim (test (e)).
    assert await provider.status() == "clean"


# (b) GitVcsProvider builds the correct `git log` command from VcsLogArgs.
async def test_vcs_git_log_command_built(tmp_path: Path) -> None:
    sandbox = CommandCapturingSandbox(tmp_path, CommandOutput(stdout="LOG", exit_code=0))
    git = GitVcsProvider(sandbox, "/work")
    out = await git.log(VcsLogArgs(max_entries=20, since_ref="abc123", format="%h %s"))
    assert out == "LOG"
    assert sandbox.command == "git"
    # -n N, then --format=, then <ref>.. — mirrors Rust's log_args ordering.
    assert sandbox.args == ["log", "-n", "20", "--format=%h %s", "abc123.."]
    assert sandbox.working_dir == Path("/work")


async def test_vcs_git_log_command_minimal_args(tmp_path: Path) -> None:
    sandbox = CommandCapturingSandbox(tmp_path, CommandOutput(stdout="LOG", exit_code=0))
    git = GitVcsProvider(sandbox, "/work")
    await git.log(VcsLogArgs(max_entries=5))
    assert sandbox.args == ["log", "-n", "5"]


# (e) status() runs `git status` and round-trips stdout.
async def test_vcs_git_status_roundtrip(tmp_path: Path) -> None:
    sandbox = CommandCapturingSandbox(
        tmp_path, CommandOutput(stdout="nothing to commit", exit_code=0)
    )
    git = GitVcsProvider(sandbox, "/work")
    out = await git.status()
    assert out == "nothing to commit"
    assert sandbox.command == "git"
    assert sandbox.args == ["status"]


# A non-zero exit raises VcsError carrying stderr.
async def test_vcs_git_nonzero_exit_raises(tmp_path: Path) -> None:
    sandbox = CommandCapturingSandbox(
        tmp_path, CommandOutput(stderr="not a git repo", exit_code=128)
    )
    git = GitVcsProvider(sandbox, "/work")
    try:
        await git.status()
    except VcsError as e:
        assert "not a git repo" in str(e)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected VcsError on non-zero exit")


# (c) Ralph with a FixtureVcsProvider injects the vcs_log into reloaded context
# across a reset.
async def test_vcs_ralph_injects_log_into_reload(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, COMPLETE])
    vcs_log = "cafe123 implement login\nbeef456 add login tests"
    h = StandardHarness(
        _config(
            tmp_path,
            agent,
            storage,
            max_resets=3,
            context_manager=PassThroughContextManager(),
            vcs_provider=FixtureVcsProvider(vcs_log, ""),
        )
    )
    await h.run(HarnessRunOptions(_ralph_task()))
    texts = agent.seen_text()
    assert "Recent VCS history:" in texts[0]
    assert "cafe123 implement login" in texts[0]


# (d) Ralph with vcs_provider=None omits any git section (v1 unchanged).
async def test_vcs_ralph_none_omits_git_section(tmp_path: Path) -> None:
    storage = _new_storage()
    await _write_progress(storage.run(), tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(storage.run(), tmp_path, [INCOMPLETE, COMPLETE])
    h = StandardHarness(
        _config(
            tmp_path,
            agent,
            storage,
            max_resets=3,
            context_manager=PassThroughContextManager(),
            vcs_provider=None,
        )
    )
    await h.run(HarnessRunOptions(_ralph_task()))
    texts = agent.seen_text()
    assert all("Recent VCS history:" not in t for t in texts)
