"""Tests for the Ralph loop strategy (issue #58).

Mirrors ``rust/crates/spore-core/src/harness.rs`` Ralph unit tests and the
shared fixture at ``fixtures/harness/ralph.json``. Ralph is a
multi-context-window continuation loop: each window is a FRESH session
re-seeded with the instruction plus reloaded ``.spore/`` state, driven by a
registered ``Stop`` hook reading ``.spore/progress.json`` (B1), bounded by the
``max_resets`` outer cap (B3). Each test maps to one rule; the rule lives in
the docstring.
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
    LoopStrategyRalph,
    LoopStrategyReAct,
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
)
from spore_core.harness import BaseSandboxProvider, NoopContextManager
from spore_core.model import ModelParams

INCOMPLETE = '{"complete": false, "remaining": ["task A"]}'
COMPLETE = '{"complete": true, "remaining": []}'


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class WorkspaceSandbox(BaseSandboxProvider):
    """Allow-all sandbox whose ``workspace_root`` is a real tempdir, so the
    Ralph filesystem reload + completion check read real ``.spore/`` files."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None

    def workspace_root(self) -> Path:
        return self._root


def _write_progress(root: Path, body: str) -> None:
    (root / ".spore").mkdir(exist_ok=True)
    (root / ".spore" / "progress.json").write_text(body)


def _write_feature_list(root: Path, body: str) -> None:
    (root / ".spore").mkdir(exist_ok=True)
    (root / ".spore" / "feature_list.json").write_text(body)


class ProgressWritingAgent:
    """Agent that, on each turn, pops the next progress-file body from a queue
    and writes it to ``.spore/progress.json`` BEFORE returning a
    ``FinalResponse`` — modelling "the agent did work this window and updated
    progress." Records the contexts it saw so tests can assert fresh-state /
    reload. Mirrors Rust's ``ProgressWritingAgent``."""

    def __init__(self, root: Path, bodies: list[str]) -> None:
        self._id = AgentId("ralph-build")
        self._root = root
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
            _write_progress(self._root, self._queue.pop(0))
        return FinalResponse(
            content="window done",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def id(self) -> AgentId:
        return self._id


class PassThroughContextManager:
    """Context manager that appends seeded user messages as real text messages
    so the agent's recorded context reflects what the window was re-seeded
    with (instruction + reloaded ``.spore/`` state)."""

    async def assemble(self, session: SessionState, task: Task) -> Context:
        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: Any) -> None:
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        from spore_core.model import Message, Role, TextContent

        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    def should_compact(self, session: SessionState) -> bool:
        return False


def _config(
    root: Path,
    agent: Any,
    *,
    max_resets: int = 3,
    context_manager: Any = None,
) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=WorkspaceSandbox(root),
        context_manager=(context_manager if context_manager is not None else NoopContextManager()),
        termination_policy=AlwaysContinuePolicy(),
        max_resets=max_resets,
    )


def _ralph_task() -> Task:
    # One ReAct turn per context window keeps the per-window sub-loop bounded so
    # the OUTER reset loop drives the test deterministically.
    return Task.new(
        "implement the thing",
        SessionId("ralph-session"),
        LoopStrategyRalph(),
        budget=BudgetLimits(max_turns=1),
    )


# ---------------------------------------------------------------------------
# R0: Ralph is implemented — no longer StrategyNotYetImplemented.
# ---------------------------------------------------------------------------


async def test_r0_ralph_implemented(tmp_path: Path) -> None:
    _write_progress(tmp_path, COMPLETE)
    agent = ProgressWritingAgent(tmp_path, [COMPLETE])
    h = StandardHarness(_config(tmp_path, agent))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultSuccess)


# ---------------------------------------------------------------------------
# R4: incomplete,incomplete,complete → Success at iteration 3.
# ---------------------------------------------------------------------------


async def test_r4_resets_until_complete(tmp_path: Path) -> None:
    _write_progress(tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(tmp_path, [INCOMPLETE, INCOMPLETE, COMPLETE])
    h = StandardHarness(_config(tmp_path, agent, max_resets=3))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultSuccess)
    # Exactly three context windows ran (one agent turn each).
    assert agent.call_count == 3


# ---------------------------------------------------------------------------
# R5: always-incomplete → exactly max_resets windows → RalphCompletionUnmet.
# ---------------------------------------------------------------------------


async def test_r5_exhausts_max_resets(tmp_path: Path) -> None:
    _write_progress(tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(tmp_path, [INCOMPLETE, INCOMPLETE, INCOMPLETE, INCOMPLETE])
    h = StandardHarness(_config(tmp_path, agent, max_resets=3))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonRalphCompletionUnmet)
    assert r.reason.iterations == 3
    assert "task A" in r.reason.last_reason
    # Exactly max_resets windows ran.
    assert agent.call_count == 3


async def test_r5_single_window_cap(tmp_path: Path) -> None:
    _write_progress(tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(tmp_path, [INCOMPLETE, INCOMPLETE])
    h = StandardHarness(_config(tmp_path, agent, max_resets=1))
    r = await h.run(HarnessRunOptions(_ralph_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonRalphCompletionUnmet)
    assert r.reason.iterations == 1
    assert agent.call_count == 1


# ---------------------------------------------------------------------------
# R2: each reset builds a FRESH SessionState — no message carryover.
# ---------------------------------------------------------------------------


async def test_r2_fresh_session_per_reset(tmp_path: Path) -> None:
    _write_progress(tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(tmp_path, [INCOMPLETE, COMPLETE])
    h = StandardHarness(
        _config(tmp_path, agent, max_resets=3, context_manager=PassThroughContextManager())
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
    _write_progress(tmp_path, INCOMPLETE)
    _write_feature_list(tmp_path, '[{"name":"login","passes":false}]')
    agent = ProgressWritingAgent(tmp_path, [INCOMPLETE, COMPLETE])
    h = StandardHarness(
        _config(tmp_path, agent, max_resets=3, context_manager=PassThroughContextManager())
    )
    await h.run(HarnessRunOptions(_ralph_task()))
    texts = agent.seen_text()
    # Window 1's fresh seed contains the reloaded progress + feature list.
    assert "Reloaded .spore/progress.json" in texts[0]
    assert "Reloaded .spore/feature_list.json" in texts[0]
    assert "login" in texts[0]


# ---------------------------------------------------------------------------
# R6: budgets fold across ALL context windows.
# ---------------------------------------------------------------------------


async def test_r6_budgets_fold_across_windows(tmp_path: Path) -> None:
    _write_progress(tmp_path, INCOMPLETE)
    agent = ProgressWritingAgent(tmp_path, [INCOMPLETE, INCOMPLETE, COMPLETE])
    h = StandardHarness(_config(tmp_path, agent, max_resets=3))
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
    task = Task.new("do", SessionId("s1"), LoopStrategyReAct(max_iterations=5))
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 1


# ---------------------------------------------------------------------------
# Completion-status helper: progress complete but a feature fails ⇒ still
# incomplete (the feature list corroborates).
# ---------------------------------------------------------------------------


async def test_completion_status_feature_list_gate(tmp_path: Path) -> None:
    from spore_core.harness import _ralph_completion_status

    _write_progress(tmp_path, COMPLETE)
    _write_feature_list(tmp_path, '[{"name":"login","passes":false}]')
    status = _ralph_completion_status(tmp_path)
    assert status is not None and "login" in status
    # Now mark it passing — complete.
    _write_feature_list(tmp_path, '[{"name":"login","passes":true}]')
    assert _ralph_completion_status(tmp_path) is None


async def test_completion_status_remaining_nonempty_is_incomplete(tmp_path: Path) -> None:
    from spore_core.harness import _ralph_completion_status

    # complete:true but a non-empty remaining list ⇒ incomplete (fixture case
    # ``complete_but_remaining_nonempty_is_incomplete``).
    _write_progress(tmp_path, '{"complete": true, "remaining": ["leftover"]}')
    status = _ralph_completion_status(tmp_path)
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
        # Seed an initial incomplete progress file so window 1 reloads state.
        _write_progress(case_dir, INCOMPLETE)
        bodies = [
            json.dumps({"complete": w["complete"], "remaining": w.get("remaining", [])})
            for w in case["windows"]
        ]
        agent = ProgressWritingAgent(case_dir, bodies)
        h = StandardHarness(_config(case_dir, agent, max_resets=case["max_resets"]))
        r = await h.run(HarnessRunOptions(_ralph_task()))
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
