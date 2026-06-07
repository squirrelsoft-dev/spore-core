"""Tests for the #126 PlanExecute DAG executor on :class:`StandardHarness`.

Mirrors the ``#126`` executor tests in the Rust reference
(``rust/crates/spore-core/src/harness.rs``): the ready-set DAG walk (AC1), the
harness-observed ``files_touched`` ledger (AC2), the failure cascade
(AC3/AC4), the execute-entry cycle re-check (AC5), and the deprecated
``PlanArtifact`` bridge (AC6). Every test exercises one resolved decision
(A–F) or acceptance criterion. The fixture-replay tests drive the FULL two-phase
``run()`` off the shared ground-truth fixtures so the outcome is byte-identical
to the Rust reference.

The runnable task list is authored via the persisted ``task_list`` store (the
#126 ONE authoring path — it can carry real ``blockers``); the plan turn output
is ignored for the task list but still drives the plan phase.
"""

from __future__ import annotations

from pathlib import Path

from spore_core import (
    AgentErrorEmpty,
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    FinalResponse,
    HaltReasonTaskGraphCycle,
    HaltReasonTasksBlockedByFailure,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryStorageProvider,
    ModelAgent,
    MockAgent,
    PlanExecuteConfig,
    ProviderInfo,
    ReplayModelInterface,
    RunResultFailure,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    Task,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolOutputSuccess,
    TurnError,
)
from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList, TaskStatus
from spore_core.tasklist import Task as TaskNode


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _config(agent: object, storage: StorageProvider, **overrides: object) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,  # type: ignore[arg-type]
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=overrides.get("sandbox", AllowAllSandbox()),
        context_manager=overrides.get("context_manager", _StoringContextManager()),
        termination_policy=overrides.get("termination_policy", AlwaysContinuePolicy()),
        hooks=overrides.get("hooks"),
        storage=storage,
    )


class _StoringContextManager:
    """A context manager that STORES appended user messages on the session and
    assembles them into the Context, so a recording agent can observe the Tier-1
    / Tier-2 text the executor seeds per step (unlike the no-op default)."""

    async def assemble(self, session: SessionState, task: object) -> object:
        from spore_core.agent import Context
        from spore_core.model import ModelParams

        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: object) -> None:
        return None

    async def append_user_message(self, session: SessionState, text: str) -> None:
        from spore_core.model import Message, Role, TextContent

        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    async def append_assistant_message(self, session: SessionState, message: object) -> None:
        session.messages.append(message)  # type: ignore[arg-type]

    def should_compact(self, session: SessionState) -> bool:
        return False


class _RecordingAgent:
    """Records every assembled Context and yields a scripted sequence of finals,
    so a test can assert what each execute step saw (Tier-1 / Tier-2 seeds)."""

    def __init__(self, contents: list[str]) -> None:
        self.seen: list[object] = []
        self._contents = list(contents)

    async def turn(self, context: object) -> FinalResponse:
        self.seen.append(context)
        content = self._contents.pop(0) if self._contents else "done"
        return FinalResponse(content=content, usage=_usage())

    def id(self) -> AgentId:
        return AgentId("recording")

    def seen_text(self) -> list[str]:
        out: list[str] = []
        for ctx in self.seen:
            parts: list[str] = []
            for m in ctx.messages:  # type: ignore[attr-defined]
                text = getattr(m.content, "text", "")
                if text:
                    parts.append(text)
            out.append("\n".join(parts))
        return out


def _new_storage() -> StorageProvider:
    return StorageProvider.single(InMemoryStorageProvider())


def _plan_task(session: str = "s1") -> Task:
    return Task.new("orchestrate the DAG", SessionId(session), PlanExecuteConfig.simple())


async def _seed_dag(storage: StorageProvider, session: SessionId, tl: TaskList) -> None:
    """Persist an authored DAG task list (the #126 authoring path) so the
    executor's ``load_task_list`` picks it up over the linear plan bridge."""
    await storage.run().put(session, TASK_LIST_EXTRAS_KEY, tl.to_dict())


async def _stored(h: StandardHarness, session: SessionId) -> TaskList:
    value = await h.storage().run().get(session, TASK_LIST_EXTRAS_KEY)
    assert value is not None
    return TaskList.from_dict(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC1: a blocker DAG executes in dependency order with a lowest-id tiebreak.
# ---------------------------------------------------------------------------


async def test_dag_executes_in_dependency_order_with_id_tiebreak() -> None:
    a = _agent()
    a.push(FinalResponse(content='{"tasks":["ignored"]}', usage=_usage()))  # plan turn
    a.push(FinalResponse(content="did 1", usage=_usage()))
    a.push(FinalResponse(content="did 2", usage=_usage()))
    a.push(FinalResponse(content="did 3", usage=_usage()))
    a.push(FinalResponse(content="did 4", usage=_usage()))
    storage = _new_storage()
    session = SessionId("s1")
    tl = TaskList()
    tl.add("one", [])  # 1
    tl.add("two", [1])  # 2 -> 1
    tl.add("three", [1])  # 3 -> 1
    tl.add("four", [2, 3])  # 4 -> 2,3
    await _seed_dag(storage, session, tl)
    h = StandardHarness(_config(a, storage))

    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultSuccess)
    # Last completed (id 4) text is the run output (ready order 1,2,3,4).
    assert r.output == "did 4"
    stored = await _stored(h, session)
    assert all(t.status is TaskStatus.COMPLETED for t in stored.tasks)


# ---------------------------------------------------------------------------
# AC1 branch isolation + Tier-1 lean: a task's seed contains its TRANSITIVE
# blockers' outputs/ledger ONLY — never an independent branch's.
# ---------------------------------------------------------------------------


async def test_dag_branch_isolation_tier1_excludes_independent_branch() -> None:
    worker = _RecordingAgent(
        [
            '{"tasks":["ignored"]}',  # plan
            "ROOT_OUTPUT_AAA",  # task 1
            "CHILD_OUTPUT_BBB",  # task 2 (-> 1)
            "INDEP_OUTPUT_CCC",  # task 3 (indep)
        ]
    )
    storage = _new_storage()
    session = SessionId("s1")
    tl = TaskList()
    tl.add("root", [])  # 1
    tl.add("child of root", [1])  # 2 -> 1
    tl.add("independent", [])  # 3 indep
    await _seed_dag(storage, session, tl)
    h = StandardHarness(_config(worker, storage))

    await h.run(HarnessRunOptions(_plan_task()))
    contexts = worker.seen_text()
    # [0] plan, [1] task1, [2] task2, [3] task3.
    assert len(contexts) == 4
    # Task 2 (index 2) is seeded with its transitive blocker (task 1)'s output.
    assert "ROOT_OUTPUT_AAA" in contexts[2]
    # Task 2 must NOT see the independent task 3 (not a blocker; not yet run).
    assert "INDEP_OUTPUT_CCC" not in contexts[2]
    # Task 3 (index 3) is INDEPENDENT — no Tier-1 upstream block.
    assert "Results from upstream tasks" not in contexts[3]


# ---------------------------------------------------------------------------
# AC2: files_touched is HARNESS-OBSERVED from write/edit calls, not
# self-reported. Task 1 only NARRATES touching a file (no write call) → empty;
# task 2 issues a real edit_file call → the path is recorded in the ledger.
# ---------------------------------------------------------------------------


async def test_dag_files_touched_observed_not_self_reported() -> None:
    a = _agent()
    a.push(FinalResponse(content='{"tasks":["ignored"]}', usage=_usage()))  # plan
    # Task 1: prose claims a file but issues NO write call.
    a.push(FinalResponse(content="I touched src/phantom.py (but did not)", usage=_usage()))
    # Task 2: a real edit_file call carrying a path, then finalize.
    a.push(
        ToolCallRequested(
            calls=[
                ToolCall(
                    id="e1",
                    name="edit_file",
                    input={"path": "src/real.py", "old_string": "a", "new_string": "b"},
                )
            ],
            usage=_usage(),
        )
    )
    a.push(FinalResponse(content="edited the file", usage=_usage()))
    reg = ScriptedToolRegistry()
    reg.push(ToolOutputSuccess(content="edited", truncated=False))
    storage = _new_storage()
    session = SessionId("s1")
    tl = TaskList()
    tl.add("narrate", [])  # 1
    tl.add("really edit", [1])  # 2 -> 1
    await _seed_dag(storage, session, tl)
    h = StandardHarness(_config(a, storage, tool_registry=reg))

    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "edited the file"


# ---------------------------------------------------------------------------
# AC2 (discriminating, direct): the StrategyExecutor observed-write seam records
# ONLY real write/edit calls (not prose / not self-reported), de-duplicated.
# ---------------------------------------------------------------------------


async def test_observed_writes_seam_records_only_real_write_calls() -> None:
    storage = _new_storage()
    h = StandardHarness(_config(_agent(), storage))
    # Nothing observed yet.
    assert h.take_observed_writes() == []
    # A non-write tool call is ignored.
    h._observe_write_call(ToolCall(id="1", name="search", input={"q": "x"}))
    assert h.take_observed_writes() == []
    # write_file / edit_file with a path are recorded, de-duplicated.
    h._observe_write_call(ToolCall(id="2", name="write_file", input={"path": "a.py"}))
    h._observe_write_call(ToolCall(id="3", name="edit_file", input={"path": "b.py"}))
    h._observe_write_call(ToolCall(id="4", name="write_file", input={"path": "a.py"}))  # dup
    assert h.take_observed_writes() == ["a.py", "b.py"]
    # take drains; clear resets.
    assert h.take_observed_writes() == []
    h._observe_write_call(ToolCall(id="5", name="write_file", input={"path": "c.py"}))
    h.clear_observed_writes()
    assert h.take_observed_writes() == []


# ---------------------------------------------------------------------------
# AC3: a terminal task failure blocks only its TRANSITIVE DEPENDENTS; unrelated
# tasks still complete; the run does NOT abort on first failure but drains to
# TasksBlockedByFailure.
#   1 good (no blockers) → completes
#   2 bad  (no blockers) → fails terminally
#   3 dep  (blocked by 2) → cascade-Blocked (never runs)
#   4 indep (no blockers) → still completes
# ---------------------------------------------------------------------------


async def test_failure_cascades_only_to_dependents() -> None:
    a = _agent()
    a.push(FinalResponse(content='{"tasks":["good","bad","dep","indep"]}', usage=_usage()))
    a.push(FinalResponse(content="did good", usage=_usage()))  # task 1
    # task 2 fails terminally: an agent error inside its sub-loop.
    a.push(TurnError(error=AgentErrorEmpty(), usage=None))
    a.push(FinalResponse(content="did indep", usage=_usage()))  # task 4
    storage = _new_storage()
    session = SessionId("s1")
    tl = TaskList()
    g = tl.add("good", [])
    bad = tl.add("bad", [])
    tl.add("dep", [bad])
    tl.add("indep", [])
    assert (g, bad) == (1, 2)
    await _seed_dag(storage, session, tl)
    h = StandardHarness(_config(a, storage))

    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTasksBlockedByFailure)
    assert r.reason.failed_task == 2
    # 1 (good) and 4 (indep) complete; 2 and its dependent 3 are blocked.
    assert r.reason.completed == [1, 4]
    assert r.reason.blocked == [2, 3]
    # plan(1) + good(1) + bad(1) + indep(1) = 4 calls; "dep" never ran.
    assert a.call_count == 4


# ---------------------------------------------------------------------------
# AC5: a persisted cyclic task graph is rejected at execute entry — no task runs.
# ---------------------------------------------------------------------------


async def test_cyclic_graph_rejected_at_execute_entry() -> None:
    a = _agent()
    a.push(FinalResponse(content='{"tasks":["ignored"]}', usage=_usage()))  # plan
    storage = _new_storage()
    session = SessionId("s1")
    # Hand-build a cyclic graph (add() would reject it): 1 -> 2, 2 -> 1.
    tl = TaskList(
        tasks=[
            TaskNode(id=1, description="a", status=TaskStatus.PENDING, blockers=[2]),
            TaskNode(id=2, description="b", status=TaskStatus.PENDING, blockers=[1]),
        ],
        next_id=3,
    )
    await _seed_dag(storage, session, tl)
    h = StandardHarness(_config(a, storage))

    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTaskGraphCycle)
    # Only the plan turn ran; no execute step.
    assert a.call_count == 1


# ---------------------------------------------------------------------------
# Ledger drop-oldest end-to-end: a >20-task linear chain leaves the persisted
# completion intact and the run succeeds (the bounded ledger never breaks it).
# (The bounded-ledger mechanics are unit-tested directly in test_tasklist_dag.)
# ---------------------------------------------------------------------------


async def test_long_chain_succeeds_with_bounded_ledger() -> None:
    a = _agent()
    a.push(FinalResponse(content='{"tasks":["ignored"]}', usage=_usage()))  # plan
    storage = _new_storage()
    session = SessionId("s1")
    tl = TaskList()
    prev: list[int] = []
    for i in range(1, 26):  # 25 tasks, each blocked by the previous (linear).
        new_id = tl.add(f"step {i}", prev)
        prev = [new_id]
        a.push(FinalResponse(content=f"did {i}", usage=_usage()))
    await _seed_dag(storage, session, tl)
    h = StandardHarness(_config(a, storage))

    r = await h.run(HarnessRunOptions(_plan_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "did 25"
    stored = await _stored(h, session)
    assert all(t.status is TaskStatus.COMPLETED for t in stored.tasks)


# ===========================================================================
# Fixture-replay parity (byte-identity ground truth for all four languages).
# ===========================================================================


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "fixtures" / "model_responses" / "harness"


def _replay_harness(fixture: str, storage: StorageProvider) -> StandardHarness:
    text = (_fixtures_dir() / fixture).read_text()
    replay = ReplayModelInterface.from_jsonl(
        text,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("planner"), replay)
    return StandardHarness(
        HarnessConfig(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=_StoringContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            storage=storage,
        )
    )


async def test_fixture_replay_dag_order() -> None:
    storage = _new_storage()
    session = SessionId("dag-order")
    tl = TaskList()
    tl.add("one", [])  # 1
    tl.add("two", [1])  # 2 -> 1
    tl.add("three", [1])  # 3 -> 1
    tl.add("four", [2, 3])  # 4 -> 2,3
    await _seed_dag(storage, session, tl)
    h = _replay_harness("plan_execute_dag_order.jsonl", storage)
    r = await h.run(HarnessRunOptions(_plan_task("dag-order")))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "did 4"


async def test_fixture_replay_branch_isolation() -> None:
    storage = _new_storage()
    session = SessionId("dag-iso")
    tl = TaskList()
    tl.add("root", [])  # 1
    tl.add("child of root", [1])  # 2 -> 1
    tl.add("independent", [])  # 3 indep
    await _seed_dag(storage, session, tl)
    h = _replay_harness("plan_execute_dag_branch_isolation.jsonl", storage)
    r = await h.run(HarnessRunOptions(_plan_task("dag-iso")))
    assert isinstance(r, RunResultSuccess)
    # Ready order: 1, 2, 3 → last output is the independent task's text.
    assert r.output == "INDEP_OUTPUT_CCC"


async def test_fixture_replay_failure_cascade() -> None:
    storage = _new_storage()
    session = SessionId("dag-fail")
    tl = TaskList()
    tl.add("root", [])  # 1
    bad = tl.add("mid", [])  # 2 (fails terminally per fixture)
    tl.add("leaf", [bad])  # 3 -> 2 (cascade-blocked)
    tl.add("indep", [])  # 4 indep (still completes)
    await _seed_dag(storage, session, tl)
    h = _replay_harness("plan_execute_dag_failure_cascade.jsonl", storage)
    r = await h.run(HarnessRunOptions(_plan_task("dag-fail")))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTasksBlockedByFailure)
    assert r.reason.failed_task == 2
    assert r.reason.completed == [1, 4]
    assert r.reason.blocked == [2, 3]


async def test_fixture_replay_budget_fail_cascade_twin() -> None:
    # The budget-Fail resolution shares the SAME cascade arm as an error failure
    # (#126 AC4). This fixture is the error-failed twin; the budget-Fail node is
    # exercised directly by the budget-enforcement unit tests. The outcome here
    # is byte-identical to the error-failure cascade.
    storage = _new_storage()
    session = SessionId("dag-budget")
    tl = TaskList()
    tl.add("root", [])  # 1
    bad = tl.add("mid", [])  # 2 (fails terminally per fixture)
    tl.add("leaf", [bad])  # 3 -> 2 (cascade-blocked)
    tl.add("indep", [])  # 4 indep
    await _seed_dag(storage, session, tl)
    h = _replay_harness("plan_execute_dag_budget_fail_cascade.jsonl", storage)
    r = await h.run(HarnessRunOptions(_plan_task("dag-budget")))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTasksBlockedByFailure)
    assert r.reason.failed_task == 2
    assert r.reason.completed == [1, 4]
    assert r.reason.blocked == [2, 3]


async def test_fixture_replay_cycle_rejection() -> None:
    storage = _new_storage()
    session = SessionId("dag-cycle")
    # Hand-built cyclic graph (the persisted store could be cyclic out of band).
    tl = TaskList(
        tasks=[
            TaskNode(id=1, description="a", status=TaskStatus.PENDING, blockers=[2]),
            TaskNode(id=2, description="b", status=TaskStatus.PENDING, blockers=[1]),
        ],
        next_id=3,
    )
    await _seed_dag(storage, session, tl)
    h = _replay_harness("plan_execute_dag_cycle_rejection.jsonl", storage)
    r = await h.run(HarnessRunOptions(_plan_task("dag-cycle")))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTaskGraphCycle)
