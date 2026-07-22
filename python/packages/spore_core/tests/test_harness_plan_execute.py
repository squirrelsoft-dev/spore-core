"""Tests for the PlanExecute execute loop on :class:`StandardHarness` (issue #59).

Mirrors the execute-phase unit + fixture-replay tests in the Rust reference
(``rust/crates/spore-core/src/harness.rs``). Each test exercises one resolved
spec decision (Q1-Q5) or wiring rule; the rule lives in the test docstring.

The plan phase (#70) is reused unchanged; here we drive the FULL two-phase
``run()`` so the execute loop actually drains the parsed task list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    FinalResponse,
    HaltReasonBudgetExceeded,
    HaltReasonEmptyPlan,
    HaltReasonTasksBlockedByFailure,
    HarnessConfig,
    HarnessRunOptions,
    InMemoryObservabilityProvider,
    InMemoryStorageProvider,
    PlanExecuteConfig,
    PromptToolCallFlag,
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
    ToolOutput,
    ToolOutputSuccess,
    ToolSchema,
    TurnResult,
)
from spore_core.hooks import (
    FunctionHook,
    HookContinue,
    HookEvent,
    OnTaskAdvanceContext,
    StandardHookChain,
)
from spore_core.storage import project_namespace
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
        tool_call_nudge=overrides.get("tool_call_nudge", False),
        plan_empty_max_retries=overrides.get("plan_empty_max_retries", 0),
        prompt_tool_call_flag=overrides.get("prompt_tool_call_flag"),
    )


async def _stored_task_list(h: StandardHarness, session_id: SessionId) -> object:
    """Read the persisted task list back through the harness's RunStore (#76).
    #142: durable artifacts are keyed by the project namespace, not the run
    session id — read it back under ``project_namespace(h.project_id())``."""
    _ = session_id  # #142: durable readback keys by the project namespace.
    return await h.storage().run().get(project_namespace(h.project_id()), TASK_LIST_EXTRAS_KEY)


def _task(*, max_turns: int | None = None) -> Task:
    return Task.new(
        "build a CLI",
        SessionId("pe-s1"),
        PlanExecuteConfig.simple(),
        budget=BudgetLimits(max_turns=max_turns),
    )


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


class _RecordingAgent:
    def __init__(self, agent_id: str = "rec") -> None:
        self._id = AgentId(agent_id)
        self._results: list[TurnResult] = []
        self.call_count = 0
        self.tools_seen: list[list[str]] = []
        self.seen: list[Any] = []

    def push(self, result: TurnResult) -> _RecordingAgent:
        self._results.append(result)
        return self

    async def turn(self, context: Any) -> TurnResult:
        self.call_count += 1
        self.seen.append(context)
        self.tools_seen.append([t.name for t in context.tools])
        if not self._results:
            raise AssertionError("recording agent ran out of scripted turns")
        return self._results.pop(0)

    def id(self) -> AgentId:
        return self._id


class _SchemaToolRegistry:
    def __init__(self, *names: str) -> None:
        self.names = list(names) or ["inspect"]
        self.call_count = 0

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.call_count += 1
        return ToolOutputSuccess.success(f"ran {call.name}")

    def is_always_halt(self, tool_name: str) -> bool:
        _ = tool_name
        return False

    def schemas(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name=name,
                description=f"{name} tool",
                input_schema={"type": "object", "properties": {}},
            )
            for name in self.names
        ]


def _texts_seen(context: Any) -> str:
    parts: list[str] = []
    for m in context.messages:
        content = m.content
        if isinstance(content, str):
            parts.append(content)
            continue
        text = getattr(content, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


class _RecordingContextManager:
    async def assemble(self, session: SessionState, task: Task, sources: Any) -> Any:
        _ = (task, sources)
        from spore_core.agent import Context
        from spore_core.model import ModelParams

        return Context(messages=list(session.messages), tools=[], params=ModelParams())

    async def append_user_message(self, session: SessionState, text: str) -> None:
        from spore_core.model import Message, Role, TextContent

        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    async def append_assistant_message(self, session: SessionState, msg: Any) -> None:
        session.messages.append(msg)

    async def append_tool_result(self, session: SessionState, result: Any) -> None:
        _ = (session, result)

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


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
    h = StandardHarness(_config(a, plan_empty_max_retries=0))
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


async def test_plan_empty_max_retries_reasks_with_tools_suppressed() -> None:
    a = _RecordingAgent()
    # First plan attempt is empty. The configured re-ask forces the canonical
    # plan schema and suppresses tools, then succeeds and executes the one task.
    a.push(FinalResponse(content='{"tasks":[],"rationale":"hidden"}', usage=_usage(1, 2)))
    a.push(FinalResponse(content='{"tasks":["retry task"],"rationale":"r"}', usage=_usage(3, 4)))
    a.push(FinalResponse(content="done after retry", usage=_usage(5, 6)))
    reg = _SchemaToolRegistry("set_plan", "inspect")
    h = StandardHarness(
        _config(
            a,
            tool_registry=reg,
            plan_empty_max_retries=1,
            context_manager=_RecordingContextManager(),
        )
    )

    r = await h.run(HarnessRunOptions(_task()))

    assert isinstance(r, RunResultSuccess)
    assert r.output == "done after retry"
    assert a.call_count == 3
    # Normal first attempt keeps the plan tool available and directs the model
    # to author through set_plan, while retry suppresses all tools and asks for
    # JSON message content because set_plan is unavailable.
    assert a.tools_seen == [["set_plan", "inspect"], [], ["set_plan", "inspect"]]
    first_plan_text = _texts_seen(a.seen[0])
    assert "Record the plan ONLY through `set_plan`" in first_plan_text
    assert "single JSON object" not in first_plan_text
    retry_plan_text = _texts_seen(a.seen[1])
    assert "single JSON object" in retry_plan_text
    assert "do NOT call any tools" in retry_plan_text
    # Usage from the discarded empty attempt is folded into the terminal run.
    assert r.usage.input_tokens == 1 + 3 + 5
    assert r.usage.output_tokens == 2 + 4 + 6


async def test_tool_call_nudge_default_off_accepts_tool_free_execute_final() -> None:
    a = _RecordingAgent()
    a.push(FinalResponse(content='{"tasks":["inspect repo"],"rationale":"r"}', usage=_usage()))
    a.push(FinalResponse(content="guessed without tools", usage=_usage()))
    reg = _SchemaToolRegistry("inspect")
    h = StandardHarness(_config(a, tool_registry=reg))

    r = await h.run(HarnessRunOptions(_task()))

    assert isinstance(r, RunResultSuccess)
    assert r.output == "guessed without tools"
    assert a.call_count == 2
    assert reg.call_count == 0


async def test_tool_call_nudge_enabled_reasks_tool_free_execute_final() -> None:
    a = _RecordingAgent()
    a.push(FinalResponse(content='{"tasks":["inspect repo"],"rationale":"r"}', usage=_usage()))
    # First execute turn returns a terminal answer without having called tools:
    # with ``tool_call_nudge`` enabled this is recorded + nudged instead of
    # accepted. The next execute turn calls the tool, so the final answer is
    # grounded and no further structural nudge fires.
    a.push(FinalResponse(content="guessed without tools", usage=_usage()))
    a.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name="inspect", input={})],
            usage=_usage(),
        )
    )
    a.push(FinalResponse(content="grounded", usage=_usage()))
    reg = _SchemaToolRegistry("inspect")
    flag = PromptToolCallFlag(False)
    h = StandardHarness(
        _config(
            a,
            tool_registry=reg,
            tool_call_nudge=True,
            prompt_tool_call_flag=flag,
        )
    )

    r = await h.run(HarnessRunOptions(_task()))

    assert isinstance(r, RunResultSuccess)
    assert r.output == "grounded"
    assert a.call_count == 4
    assert reg.call_count == 1
    # Rust's adaptive branch flips the shared prompt-tool flag when either the
    # phrase detector OR the structural zero-tool nudge fires, so the next turn
    # switches conversational/local models into prompt-based tool calling.
    assert flag.value is True


# ---------------------------------------------------------------------------
# #126 AC3 (REPLACES the Q5 blanket abort): a terminal step failure no longer
# raises StepFailed/aborts on the first failure — it cascade-Blocks only its
# transitive dependents and KEEPS scheduling unrelated tasks, draining to
# TasksBlockedByFailure. A linear bridge has NO blocker edges, so the failing
# task blocks no one; the next (independent) task still completes.
# ---------------------------------------------------------------------------


async def test_terminal_failure_cascades_not_blanket_abort() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    # task one (independent — bridge has no blockers) completes.
    a.push(FinalResponse(content="did task one", usage=_usage()))
    # task two: no response queued → dry → AgentErrorEmpty terminal failure. The
    # run does NOT abort before task one ran; it drains to TasksBlockedByFailure.
    h = StandardHarness(_config(a))
    task = _task()
    state = SessionState()
    r = await h.run(HarnessRunOptions(task, session_state=state))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTasksBlockedByFailure)
    # task 2 is the failing task; it has no dependents → only itself is blocked;
    # task 1 completed (proving no blanket abort and DAG-ordered draining).
    assert r.reason.failed_task == 2
    assert r.reason.blocked == [2]
    assert r.reason.completed == [1]
    assert r.reason.reason  # carries the underlying reason rendering
    stored = await _stored_task_list(h, task.session_id)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert tl.tasks[0].status is TaskStatus.COMPLETED
    assert tl.tasks[1].status is TaskStatus.BLOCKED


# ---------------------------------------------------------------------------
# #126 AC3: an agent error in a step also cascades (not BudgetExceeded). With
# both tasks running dry (no responses queued), both fail terminally; the first
# failure (task 1) is reported as the trigger.
# ---------------------------------------------------------------------------


async def test_step_agent_error_cascades() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    # task one: no response queued → MockAgent returns AgentErrorEmpty.
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_task()))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonTasksBlockedByFailure)
    assert r.reason.failed_task == 1


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
    # #142: durable artifacts are keyed by the project namespace.
    stored = await provider.run().get(project_namespace(h.project_id()), TASK_LIST_EXTRAS_KEY)
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

    # Both keys are durable in the RunStore. #142: keyed by the project namespace.
    durable_ns = project_namespace(h.project_id())
    assert await provider.run().get(durable_ns, PLAN_EXECUTE_EXTRAS_KEY) is not None
    assert await provider.run().get(durable_ns, TASK_LIST_EXTRAS_KEY) is not None

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
        PlanExecuteConfig.simple(),
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
