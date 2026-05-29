"""Tests for the PlanExecute execute loop on :class:`StandardHarness` (issue #59).

Mirrors the execute-phase unit + fixture-replay tests in the Rust reference
(``rust/crates/spore-core/src/harness.rs``). Each test exercises one resolved
spec decision (Q1-Q5) or wiring rule; the rule lives in the test docstring.

The plan phase (#70) is reused unchanged; here we drive the FULL two-phase
``run()`` so the execute loop actually drains the parsed task list.
"""

from __future__ import annotations

from pathlib import Path

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    FinalResponse,
    HaltReasonBudgetExceeded,
    HaltReasonEmptyPlan,
    HaltReasonStepFailed,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryObservabilityProvider,
    InMemoryStorageProvider,
    LoopStrategyPlanExecute,
    MockAgent,
    ModelAgent,
    NoopContextManager,
    PLAN_EXECUTE_EXTRAS_KEY,
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
)
from spore_core.hooks import (
    FunctionHook,
    HookContinue,
    HookEvent,
    OnTaskAdvanceContext,
    StandardHookChain,
)
from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _config(agent: MockAgent, **overrides: object) -> HarnessConfig:
    # #76: the task list now lives on the RunStore seam (not SessionState.extras),
    # so the test harness defaults to a real (in-memory) run store for the
    # readback assertions below to observe what the harness wrote.
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=overrides.get("sandbox", AllowAllSandbox()),
        context_manager=overrides.get("context_manager", NoopContextManager()),
        termination_policy=overrides.get("termination_policy", AlwaysContinuePolicy()),
        observability=overrides.get("observability"),
        hooks=overrides.get("hooks"),
        planner_agent=overrides.get("planner_agent"),
        storage=overrides.get("storage", StorageProvider.single(InMemoryStorageProvider())),
    )


async def _stored_task_list(h: StandardHarness, session_id: SessionId) -> object:
    """Read the persisted task list back through the harness's RunStore (#76)."""
    return await h.storage().run().get(session_id, TASK_LIST_EXTRAS_KEY)


def _task(*, max_turns: int | None = None) -> Task:
    return Task.new(
        "build a CLI",
        SessionId("pe-s1"),
        LoopStrategyPlanExecute(plan_model=None),
        budget=BudgetLimits(max_turns=max_turns),
    )


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


# A two-task plan + the two step responses, in agent-queue order.
_PLAN_2 = '{"tasks":["task one","task two"],"rationale":"r"}'


def _seed_two_tasks(a: MockAgent) -> MockAgent:
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    a.push(FinalResponse(content="did task one", usage=_usage()))
    a.push(FinalResponse(content="did task two", usage=_usage()))
    return a


# ---------------------------------------------------------------------------
# Happy path: plan → drain the task list → Success.
# ---------------------------------------------------------------------------


async def test_happy_path_drains_task_list() -> None:
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    # plan turn + one turn per task = 3 agent calls.
    assert a.call_count == 3


# ---------------------------------------------------------------------------
# Q2: success output is the LAST completed step's final text (not concat, not
# the plan rationale).
# ---------------------------------------------------------------------------


async def test_success_output_is_last_step_final_text() -> None:
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "did task two"
    assert "did task one" not in r.output
    assert "r" != r.output  # not the rationale


# ---------------------------------------------------------------------------
# Lifecycle: every task ends Completed (Pending → InProgress → Completed),
# observable via the persisted task list.
# ---------------------------------------------------------------------------


async def test_all_tasks_completed_in_run_store() -> None:
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a))
    task = _task()
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultSuccess)
    stored = await _stored_task_list(h, task.session_id)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert [t.description for t in tl.tasks] == ["task one", "task two"]
    assert all(t.status is TaskStatus.COMPLETED for t in tl.tasks)
    # #76: not mirrored into SessionState.extras anymore.
    assert TASK_LIST_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# Q3: an empty plan (tasks: []) fails with EmptyPlan, not a silent success.
# ---------------------------------------------------------------------------


async def test_empty_plan_fails_with_empty_plan() -> None:
    a = _agent()
    a.push(FinalResponse(content='{"tasks":[],"rationale":"nothing"}', usage=_usage()))
    h = StandardHarness(_config(a))
    task = _task()
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonEmptyPlan)
    # Only the plan turn ran; no execute steps.
    assert a.call_count == 1
    # No task list persisted for an empty plan.
    assert await _stored_task_list(h, task.session_id) is None
    assert TASK_LIST_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# Q5: a step that fails (agent error / blocked) aborts the run with StepFailed
# and does NOT run the next task.
# ---------------------------------------------------------------------------


async def test_step_failure_aborts_with_step_failed() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    # task one errors: a tool call with an empty registry → unrecoverable tool
    # error inside the sub-loop. (MockAgent then has nothing left for task two.)
    a.push(ToolCallRequested(calls=[ToolCall(id="c", name="missing", input={})], usage=_usage()))
    h = StandardHarness(_config(a))
    task = _task()
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStepFailed)
    assert r.reason.task_index == 0
    assert r.reason.task == "task one"
    assert r.reason.reason  # carries the underlying reason rendering
    # task two never ran: the failing task is marked blocked; the next stays
    # pending (never advanced to in_progress).
    stored = await _stored_task_list(h, task.session_id)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert tl.tasks[0].status is TaskStatus.BLOCKED
    assert tl.tasks[1].status is TaskStatus.PENDING


# ---------------------------------------------------------------------------
# Q5: an agent error in a step also yields StepFailed (not BudgetExceeded).
# ---------------------------------------------------------------------------


async def test_step_agent_error_is_step_failed() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    # task one: no response queued → MockAgent returns AgentErrorEmpty.
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonStepFailed)
    assert r.reason.task_index == 0


# ---------------------------------------------------------------------------
# Q1: per-task turn allocation + shared budget. With a global cap of 5 and a
# plan turn already spent, two tasks each get floor(remaining/remaining_tasks).
# ---------------------------------------------------------------------------


async def test_per_task_turn_allocation_and_shared_budget() -> None:
    # Plan turn spends 1 → 4 turns remain over 2 tasks → 2 turns each.
    # Each task here finishes in a single turn, so the run completes well under
    # the cap and uses exactly 3 turns total (1 plan + 1 per task).
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_task(max_turns=5)))
    assert isinstance(r, RunResultSuccess)
    assert r.turns == 3


# ---------------------------------------------------------------------------
# Q1 / budget exhaustion mid-execute: a tight global cap exhausts the budget
# during the execute phase → BudgetExceeded (distinct from StepFailed).
# ---------------------------------------------------------------------------


async def test_budget_exhausted_mid_execute() -> None:
    # Global cap of 1: the plan turn alone consumes it, so the first execute
    # step's sub-loop hits the turn gate immediately → BudgetExceeded.
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a))
    state = SessionState()
    r = await h.run(HarnessRunOptions(_task(max_turns=1), session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonBudgetExceeded)
    assert r.reason.limit_type == "turns"
    # Plan turn ran; the first step's sub-loop made no agent call (gated first).
    assert a.call_count == 1


# ---------------------------------------------------------------------------
# Q1: the shared budget is carried — cumulative usage reflects plan + all steps.
# ---------------------------------------------------------------------------


async def test_usage_accumulates_across_plan_and_steps() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage(in_t=4, out_t=2)))
    a.push(FinalResponse(content="one", usage=_usage(in_t=3, out_t=1)))
    a.push(FinalResponse(content="two", usage=_usage(in_t=5, out_t=2)))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.usage.input_tokens == 4 + 3 + 5
    assert r.usage.output_tokens == 2 + 1 + 2


# ---------------------------------------------------------------------------
# OnTaskAdvance fires once per task with the correct index/total, and the hook
# may rewrite the step instruction.
# ---------------------------------------------------------------------------


async def test_on_task_advance_fires_per_task_with_indices() -> None:
    seen: list[tuple[int, int, str]] = []

    def handler(ctx: object) -> HookContinue:
        assert isinstance(ctx, OnTaskAdvanceContext)
        seen.append((ctx.task_index, ctx.total_tasks, ctx.task.instruction))
        return HookContinue()

    chain = StandardHookChain()
    chain.register(FunctionHook("advance", [HookEvent.ON_TASK_ADVANCE], handler))
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a, hooks=chain))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert seen == [
        (0, 2, "task one"),
        (1, 2, "task two"),
    ]


# ---------------------------------------------------------------------------
# Observability: one turn span per turn (plan + one per step).
# ---------------------------------------------------------------------------


async def test_observability_span_count() -> None:
    obs = InMemoryObservabilityProvider()
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a, observability=obs))
    task = _task()
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    metrics = await obs.get_session_metrics(task.session_id)
    assert metrics is not None
    # 1 plan turn + 1 per task = 3.
    assert metrics.total_turns == 3


# ---------------------------------------------------------------------------
# Compaction-in-loop: a should_compact context manager triggers compaction
# inside the per-step sub-loop without breaking the run.
# ---------------------------------------------------------------------------


class _CompactOnceContextManager(NoopContextManager):
    """Triggers compaction exactly once across the whole run, to prove the
    per-step sub-loop participates in the shared compaction machinery."""

    def __init__(self) -> None:
        self._fired = False

    def should_compact(self, session: SessionState) -> bool:
        if self._fired:
            return False
        self._fired = True
        return True

    def prepare_compaction_turn(self, session: SessionState) -> None:
        # Nothing to compact in this minimal stub → run_compaction is a no-op.
        return None


async def test_compaction_in_loop_does_not_break_run() -> None:
    a = _seed_two_tasks(_agent())
    cm = _CompactOnceContextManager()
    h = StandardHarness(_config(a, context_manager=cm))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "did task two"


# ---------------------------------------------------------------------------
# Q4: RunStore persistence. The task list is written through the storage seam.
# ---------------------------------------------------------------------------


async def test_run_store_persistence() -> None:
    backend = InMemoryStorageProvider()
    provider = StorageProvider.single(backend)
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a, storage=provider))
    task = _task()
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    stored = await provider.run().get(task.session_id, TASK_LIST_EXTRAS_KEY)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    # Final durable write reflects both tasks completed.
    assert all(t.status is TaskStatus.COMPLETED for t in tl.tasks)


# ---------------------------------------------------------------------------
# #76: after a plan/execute run, BOTH persistence keys live on the RunStore
# seam and NEITHER is mirrored into SessionState.extras. The ephemeral extras
# keys (``__rich_state``, ``subagent_handoff_summary``) are owned by other
# components and are untouched here.
# ---------------------------------------------------------------------------


async def test_persistence_lives_on_run_store_not_extras() -> None:
    backend = InMemoryStorageProvider()
    provider = StorageProvider.single(backend)
    a = _seed_two_tasks(_agent())
    h = StandardHarness(_config(a, storage=provider))
    task = _task()
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultSuccess)

    # Both keys are durable in the RunStore.
    assert await provider.run().get(task.session_id, PLAN_EXECUTE_EXTRAS_KEY) is not None
    assert await provider.run().get(task.session_id, TASK_LIST_EXTRAS_KEY) is not None

    # Neither key is mirrored into SessionState.extras anymore.
    assert PLAN_EXECUTE_EXTRAS_KEY not in state.extras
    assert TASK_LIST_EXTRAS_KEY not in state.extras


# ---------------------------------------------------------------------------
# Fixture-replay parity: drive the full two-phase loop off the shared fixture
# and assert the SAME outcome as the Rust reference.
# ---------------------------------------------------------------------------


def _fixture_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[4] / "fixtures" / "model_responses" / "harness" / "plan_execute_loop.jsonl"


async def test_fixture_replay_matches_rust() -> None:
    text = _fixture_path().read_text()
    replay = ReplayModelInterface.from_jsonl(
        text,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    # The fixture has 3 exchanges (plan + 2 steps) consumed positionally.
    agent = ModelAgent(AgentId("planner"), replay)
    h = StandardHarness(
        HarnessConfig(
            agent=agent,
            tool_registry=ScriptedToolRegistry(),
            sandbox=AllowAllSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            storage=StorageProvider.single(InMemoryStorageProvider()),
        )
    )
    state = SessionState()
    task = Task.new(
        "build a CLI",
        SessionId("pe-fixture"),
        LoopStrategyPlanExecute(plan_model=None),
    )
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultSuccess)
    # Q2: output is the LAST completed step's final text.
    assert r.output == "wrote the integration tests"
    # 1 plan turn + 1 per task (2 tasks) = 3 turns.
    assert r.turns == 3
    # Both fixture tasks completed (read back through the RunStore seam, #76).
    stored = await _stored_task_list(h, task.session_id)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert [t.description for t in tl.tasks] == [
        "scaffold the project",
        "write the integration tests",
    ]
    assert all(t.status is TaskStatus.COMPLETED for t in tl.tasks)
