"""Tests for the per-variant ``RunStrategy.run`` recursive executor (issue #124,
Composable Execution A.1/A.5/A.6).

The central dispatch ``match`` in ``_run_inner`` is GONE — the only ``match`` is
the enum→config delegation in :func:`run_strategy`, and each config owns its
loop (recursion = ``run_strategy(self.inner, cx)``). These tests cover the
#124-specific behaviors that the per-strategy suites do not:

* the recursive entry drives ReAct / PlanExecute end-to-end and propagates the
  strategy's verbatim terminal;
* A.6 deep-resume — an already-Completed task on the durable checkpoint is not
  re-run;
* A.5 output-contracts — a bare ``ReAct`` in a structured slot (plan / propose /
  worker) is rejected at startup validation;
* the recursive executor threads the shared :class:`ExecutionContext` through a
  stub strategy tree (mirrors the Rust recursion-stub test);
* StrategyOutcome→RunResult mapping (Q5).

Mirrors the new ``#124`` tests in ``rust/crates/spore-core/src/harness.rs`` and
``execution_registry.rs``.
"""

from __future__ import annotations

import inspect

import pytest

from spore_core import (
    AgentId,
    AggregateUsage,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BudgetLimits,
    BudgetSnapshot,
    ExecutionContext,
    ExecutionRegistry,
    FinalResponse,
    HaltReasonConfigurationError,
    HarnessConfig,
    HarnessErrorException,
    HarnessErrorInvalidConfiguration,
    HarnessRunOptions,
    InMemoryStorageProvider,
    MockAgent,
    NoopContextManager,
    PlanExecuteConfig,
    RalphConfig,
    ReactConfig,
    RunResultFailure,
    RunResultSuccess,
    ScriptedToolRegistry,
    SelfVerifyingConfig,
    SessionId,
    SessionState,
    StandardHarness,
    StorageProvider,
    StrategyOutcomeComplete,
    StrategyOutcomeFailed,
    Task,
    TokenUsage,
    run_strategy,
)
from spore_core.harness import HillClimbingConfig, new_session_id
from spore_core.plan import PlanArtifact
from spore_core.storage import project_namespace
from spore_core.tasklist import (
    TASK_LIST_EXTRAS_KEY,
    TaskList,
    TaskStatus,
    plan_artifact_to_task_list,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent() -> MockAgent:
    return MockAgent(AgentId("test"))


def _config(agent: MockAgent, **overrides: object) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=overrides.get("sandbox", AllowAllSandbox()),
        context_manager=overrides.get("context_manager", NoopContextManager()),
        termination_policy=overrides.get("termination_policy", AlwaysContinuePolicy()),
        storage=overrides.get("storage", StorageProvider.single(InMemoryStorageProvider())),
        registry=overrides.get("registry"),
    )


def _usage(in_t: int = 1, out_t: int = 1) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _react_task() -> Task:
    return Task.new("do something", SessionId("react-s1"), ReactConfig.per_loop(5))


def _plan_task() -> Task:
    return Task.new(
        "build a CLI",
        SessionId("pe-s1"),
        PlanExecuteConfig.simple(),
        budget=BudgetLimits(max_turns=None),
    )


_PLAN_2 = '{"tasks":["task one","task two"],"rationale":"r"}'


# ---------------------------------------------------------------------------
# AC1: the only `match` is the enum→config delegation in run_strategy.
# ---------------------------------------------------------------------------


def test_only_dispatch_match_is_in_run_strategy() -> None:
    """``_run_inner`` no longer contains a central per-strategy dispatch chain —
    it collapses to a single ``_drive_strategy`` call. The only place a
    strategy is matched on its variant is :func:`run_strategy`."""
    inner_src = inspect.getsource(StandardHarness._run_inner)
    # The old chain dispatched via isinstance on each *Config; the new entry must
    # not branch into the per-strategy ``_run_*`` orchestration methods directly.
    assert "self._run_react(" not in inner_src
    assert "self._run_plan_execute(" not in inner_src
    assert "self._run_self_verifying(" not in inner_src
    assert "self._run_ralph(" not in inner_src
    assert "self._run_hill_climbing(" not in inner_src
    assert "_drive_strategy(" in inner_src
    # run_strategy keeps the single delegation match.
    assert "match strategy:" in inspect.getsource(run_strategy)


# ---------------------------------------------------------------------------
# AC2: ReAct leaf executes end-to-end through the recursive trait call, and the
# strategy's verbatim terminal (output / session) propagates up.
# ---------------------------------------------------------------------------


async def test_react_leaf_e2e_through_recursive_executor() -> None:
    a = _agent()
    a.push(FinalResponse(content="done", usage=_usage()))
    h = StandardHarness(_config(a))
    r = await h.run(HarnessRunOptions(_react_task()))
    # The leaf's verbatim terminal (output / turns / session) propagated up
    # through the recursive entry unchanged.
    assert isinstance(r, RunResultSuccess)
    assert r.output == "done"
    assert r.turns == 1
    assert a.call_count == 1


# ---------------------------------------------------------------------------
# AC2: PlanExecute[ReAct, ReAct] executes end-to-end through recursive calls.
# ---------------------------------------------------------------------------


async def test_plan_execute_e2e_through_recursive_executor() -> None:
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    a.push(FinalResponse(content="did task one", usage=_usage()))
    a.push(FinalResponse(content="did task two", usage=_usage()))
    h = StandardHarness(_config(a))
    task = _plan_task()
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess)
    # plan turn + one turn per task = 3 agent calls.
    assert a.call_count == 3
    # Q2: output is the last completed step's final text.
    assert r.output == "did task two"
    # Both tasks Completed in the persisted list. #142: keyed by the project ns.
    stored = await h.storage().run().get(project_namespace(h.project_id()), TASK_LIST_EXTRAS_KEY)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert all(t.status is TaskStatus.COMPLETED for t in tl.tasks)


# ---------------------------------------------------------------------------
# #124 REGRESSION-PROOF: a NON-ReAct execute child genuinely runs per task.
#
# ``PlanExecute[plan: ReAct, execute: SelfVerifying[ReAct]]`` over a 2-task plan.
# The execute child is SelfVerifying — its build↔evaluate loop runs ONCE PER TASK,
# so the scripted verifier is invoked exactly twice (once per task). The pre-#124
# implementation hardcoded a flat ReAct sub-loop in the execute phase and silently
# DROPPED the SelfVerifying child — the verifier would have been invoked ZERO
# times. This proves ``self.execute`` is dispatched via ``run_strategy`` and a
# non-ReAct child genuinely executes. Mirrors Rust's
# ``plan_execute_runs_non_react_execute_child_per_task``.
# ---------------------------------------------------------------------------


async def test_plan_execute_runs_non_react_execute_child_per_task() -> None:
    from spore_core import (
        VerifierInput,
        VerifierVerdict,
        VerifierVerdictPassed,
    )

    class _RecordingVerifier:
        """Records every invocation; PASS each time."""

        def __init__(self) -> None:
            self.seen: list[VerifierInput] = []

        async def verify(self, input: VerifierInput) -> VerifierVerdict:
            self.seen.append(input)
            return VerifierVerdictPassed()

        def max_iterations(self) -> int:
            return 3

    # #124 Q1c: the single worker agent (default ``""`` key) runs the plan turn
    # (JSON), then per task the build turn, then the evaluate-phase turn. 2 tasks
    # ⇒ plan + 2×(build + eval) = 5 turns.
    a = _agent()
    a.push(FinalResponse(content=_PLAN_2, usage=_usage()))
    a.push(FinalResponse(content="built t0", usage=_usage()))
    a.push(FinalResponse(content="PASS", usage=_usage()))
    a.push(FinalResponse(content="built t1", usage=_usage()))
    a.push(FinalResponse(content="PASS", usage=_usage()))
    verifier = _RecordingVerifier()
    # #124: the verifier resolves from the SelfVerifying ``evaluator`` key — here
    # the default empty key, folded via the ``verifier=`` constructor param.
    cfg = _config(a, registry=ExecutionRegistry.builder().verifier("", verifier).build())
    h = StandardHarness(cfg)

    # The execute child is a genuine SelfVerifying combinator (NOT a ReAct).
    task = Task.new(
        "build a CLI",
        SessionId("pe-nonreact"),
        PlanExecuteConfig(
            plan=ReactConfig(
                budget=ReactConfig.per_loop(2**31 - 1).budget, agent="", toolset="", output=""
            ),
            execute=SelfVerifyingConfig(
                inner=ReactConfig(
                    budget=ReactConfig.per_loop(2**31 - 1).budget,
                    agent="",
                    toolset="",
                    output="",
                ),
                evaluator="",
            ),
        ),
        budget=BudgetLimits(max_turns=None),
    )

    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    # Q2: output is the last step's final output.
    assert r.output == "built t1"

    # The smoking gun: the SelfVerifying evaluator ran ONCE PER TASK (2x). A
    # dropped execute child would record ZERO verifier invocations.
    assert len(verifier.seen) == 2, (
        "the SelfVerifying execute child must run its verifier once per task"
    )


# ---------------------------------------------------------------------------
# AC4: A.6 deep-resume — an already-Completed task on the durable checkpoint is
# NOT re-run (serialize→reset→resume, not the old shallow no-op).
# ---------------------------------------------------------------------------


async def test_deep_resume_skips_already_completed_task() -> None:
    a = _agent()
    # Only ONE response: if task 0 were re-run the loop would starve; one
    # response is enough iff task 0 is skipped.
    a.push(FinalResponse(content="done two", usage=_usage()))
    h = StandardHarness(_config(a))
    t = _plan_task()

    # Pre-seed the durable checkpoint: task 0 Completed, task 1 Pending.
    checkpoint = plan_artifact_to_task_list(PlanArtifact(tasks=["one", "two"], rationale=""))
    first_id = checkpoint.tasks[0].id
    checkpoint.complete(first_id)
    await h.persist_task_list(t.session_id, checkpoint)

    # The freshly-parsed list (as the plan phase would produce) is all-Pending.
    fresh = plan_artifact_to_task_list(PlanArtifact(tasks=["one", "two"], rationale=""))
    result = await h._run_execute_phase(
        t,
        SessionState(),
        fresh,
        BudgetSnapshot(turns=1),
        AggregateUsage(),
        None,
    )
    assert isinstance(result, RunResultSuccess)
    assert result.output == "done two"
    # Exactly ONE agent turn: task 0 was resumed from checkpoint, not re-run.
    assert a.call_count == 1
    # Both tasks Completed in the final persisted list. #142: keyed by project ns.
    stored = await h.storage().run().get(project_namespace(h.project_id()), TASK_LIST_EXTRAS_KEY)
    final_list = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert all(x.status is TaskStatus.COMPLETED for x in final_list.tasks)


# ---------------------------------------------------------------------------
# AC5: A.5 output-contract enforcement — a bare ReAct in a STRUCTURED slot
# (plan / worker / propose) is rejected by registry validation.
# ---------------------------------------------------------------------------


def _wired_registry() -> ExecutionRegistry:
    return (
        ExecutionRegistry.builder()
        .agent("a1", _agent())
        .toolset("t1", ScriptedToolRegistry())
        .schema("plan-schema", {})
        .schema("worker-schema", {})
        # #124 Q1: the SelfVerifying ``evaluator`` resolves as a VERIFIER key.
        .verifier("eval-schema", object())
        # #124 Q2: the HillClimbing ``evaluator`` resolves as a metric key.
        .metric_evaluator("metric-1", object())
        .build()
    )


def _react_leaf(output: str | None = None) -> ReactConfig:
    return ReactConfig(
        budget=ReactConfig.per_loop(4).budget, agent="a1", toolset="t1", output=output
    )


@pytest.mark.parametrize(
    ("tree", "slot_name"),
    [
        (
            PlanExecuteConfig(plan=_react_leaf(), execute=_react_leaf()),
            "plan",
        ),
        (
            SelfVerifyingConfig(inner=_react_leaf(), evaluator="eval-schema"),
            "worker",
        ),
        (
            HillClimbingConfig(
                inner=_react_leaf(),
                direction="maximize",
                max_stagnation=3,
                revert_on_no_improvement=False,
                min_improvement_delta=0.1,
                evaluator="metric-1",
            ),
            "propose",
        ),
    ],
)
def test_structured_slot_rejects_bare_react_without_output_schema(
    tree: object, slot_name: str
) -> None:
    reg = _wired_registry()
    task = Task.new("contract", new_session_id(), tree)  # type: ignore[arg-type]
    with pytest.raises(HarnessErrorException) as ei:
        reg.validate(task)
    assert isinstance(ei.value.error, HarnessErrorInvalidConfiguration)
    assert slot_name in ei.value.error.reason


def test_structured_slot_accepts_react_with_output_schema() -> None:
    reg = _wired_registry()
    tree = PlanExecuteConfig(plan=_react_leaf(output="plan-schema"), execute=_react_leaf())
    task = Task.new("contract", new_session_id(), tree)
    reg.validate(task)  # no raise


def test_structured_slot_accepts_combinator_child() -> None:
    """A non-leaf child in a structured slot carries its own contract; the
    bare-ReAct check applies only to a leaf, so a PlanExecute plan slot holding
    another combinator is accepted."""
    reg = _wired_registry()
    inner_sv = SelfVerifyingConfig(
        inner=_react_leaf(output="worker-schema"), evaluator="eval-schema"
    )
    tree = PlanExecuteConfig(plan=inner_sv, execute=_react_leaf())
    task = Task.new("contract", new_session_id(), tree)
    reg.validate(task)  # no raise


async def test_output_contract_rejected_at_harness_startup() -> None:
    """The harness runs registry validation at the entry of ``run`` — a bare
    ReAct in the structured plan slot is a startup ``ConfigurationError`` BEFORE
    any agent turn."""
    a = _agent()
    reg = _wired_registry()
    h = StandardHarness(_config(a, registry=reg))
    tree = PlanExecuteConfig(plan=_react_leaf(), execute=_react_leaf())
    task = Task.new("contract", SessionId("startup-s1"), tree)
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultFailure)
    assert isinstance(r.reason, HaltReasonConfigurationError)
    assert a.call_count == 0  # rejected before the first turn


# ---------------------------------------------------------------------------
# The recursive executor threads the shared ExecutionContext + budget stack
# through a stub strategy tree (mirrors the Rust recursion-stub test).
# ---------------------------------------------------------------------------


async def test_recursive_stub_threads_execution_context() -> None:
    from spore_core.harness import BudgetContext, BudgetExhaustedFail, BudgetPolicyPerLoop

    depths: list[int] = []

    class RecursiveStub:
        def __init__(self, child: object | None) -> None:
            self.child = child

        async def run(self, cx: ExecutionContext) -> object:
            cx.budgets.push(
                BudgetContext(
                    policy=BudgetPolicyPerLoop(value=1),
                    behavior=BudgetExhaustedFail(),
                    phase="stub",
                )
            )
            depths.append(cx.budgets.depth())
            if self.child is not None:
                await self.child.run(cx)
            cx.budgets.pop()
            return StrategyOutcomeComplete(output="")

    cx = ExecutionContext(registry=ExecutionRegistry.empty())
    leaf = RecursiveStub(None)
    root = RecursiveStub(RecursiveStub(leaf))
    outcome = await root.run(cx)
    assert isinstance(outcome, StrategyOutcomeComplete)
    # Three nested frames were active at peak; the stack unwinds to empty.
    assert depths == [1, 2, 3]
    assert cx.budgets.depth() == 0


# ---------------------------------------------------------------------------
# Q5: a leaf failure maps to RunResultFailure; a wired-but-missing executor is a
# TYPED Failed, never a raise.
# ---------------------------------------------------------------------------


async def test_no_executor_is_typed_failure() -> None:
    cx = ExecutionContext(registry=ExecutionRegistry.empty())
    cx.scratch.task = Task.new("t", new_session_id(), ReactConfig.per_loop(1))
    outcome = await run_strategy(ReactConfig.per_loop(1), cx)
    assert isinstance(outcome, StrategyOutcomeFailed)
    assert isinstance(outcome.error, HarnessErrorInvalidConfiguration)


# ---------------------------------------------------------------------------
# #124 REGRESSION-PROOF: the three combinators GENUINELY recurse into a NON-ReAct
# inner. Each asserts the inner's distinctive work fires (a hardcoded-ReAct loop
# would record ZERO). Mirrors the Rust ``*_runs_non_react_inner_*`` tests.
# ---------------------------------------------------------------------------


class _StoringContextManager:
    """A context manager that actually STORES appended user messages on the
    session (unlike the no-op default) and assembles them into the Context, so a
    recording agent can observe the plan directive text the harness seeds."""

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


def _react_structured() -> ReactConfig:
    """A bare ReAct leaf carrying the default output schema handle, for the
    STRUCTURED slots (PlanExecute plan / SelfVerifying worker / HillClimbing
    propose) A.5 requires to declare an output schema."""
    return ReactConfig(
        budget=ReactConfig.per_loop(2**31 - 1).budget, agent="", toolset="", output=""
    )


class _RecordingAgent:
    """Records every ``Context`` it is handed and yields a scripted sequence of
    finals, so a test can assert what the worker saw (e.g. the plan directive)."""

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


async def test_self_verifying_runs_non_react_inner_worker() -> None:
    """SelfVerifying[inner: PlanExecute[ReAct, ReAct]] — each build iteration runs
    the inner PlanExecute's WHOLE loop, which fires a plan turn (parsing JSON)
    before the execute step. A hardcoded-ReAct build would NEVER fire the plan
    phase. Mirrors Rust's ``self_verifying_runs_non_react_inner_worker``."""
    from spore_core import VerifierInput, VerifierVerdict, VerifierVerdictPassed

    class _RecordingVerifier:
        def __init__(self) -> None:
            self.seen: list[VerifierInput] = []

        async def verify(self, input: VerifierInput) -> VerifierVerdict:
            self.seen.append(input)
            return VerifierVerdictPassed()

        def max_iterations(self) -> int:
            return 3

    # One SelfVerifying iteration of PlanExecute[ReAct,ReAct] over a 1-task plan:
    # plan JSON, execute step, then the evaluate-phase turn.
    worker = _RecordingAgent(['{"tasks":["only"],"rationale":"r"}', "did the step", "PASS"])
    verifier = _RecordingVerifier()
    cfg = _config(
        worker,
        context_manager=_StoringContextManager(),
        registry=ExecutionRegistry.builder().verifier("", verifier).build(),
    )
    h = StandardHarness(cfg)

    # inner is a genuine PlanExecute combinator (NOT a ReAct).
    task = Task.new(
        "build it",
        SessionId("sv-nonreact"),
        SelfVerifyingConfig(
            inner=PlanExecuteConfig(
                plan=_react_structured(), execute=ReactConfig.per_loop(2**31 - 1)
            ),
            evaluator="",
        ),
        budget=BudgetLimits(max_turns=None),
    )
    r = await h.run(HarnessRunOptions(task))
    assert isinstance(r, RunResultSuccess), f"expected Success, got {r!r}"
    # The inner PlanExecute fired its plan turn ⇒ the worker saw the plan
    # directive. A hardcoded-ReAct build would record ZERO plan turns.
    plan_turns = sum(1 for c in worker.seen_text() if "step-by-step plan" in c)
    assert plan_turns >= 1, (
        f"the inner PlanExecute plan phase must fire >=1 per iteration; saw {worker.seen_text()!r}"
    )
    # The verifier fired (the SelfVerifying loop ran its evaluate phase).
    assert len(verifier.seen) == 1


async def test_ralph_runs_non_react_inner_per_window() -> None:
    """Ralph[inner: SelfVerifying[ReAct]] over always-incomplete progress — each
    window runs its inner SelfVerifying's full build↔evaluate loop (firing its
    verifier once), Ralph reads incomplete and resets until ``max_resets`` is
    exhausted. With max_resets=2 the verifier fires >=1 per window (>=2 total). A
    hardcoded-ReAct window would record ZERO verifier invocations. Mirrors Rust's
    ``ralph_runs_non_react_inner_per_window``."""
    import tempfile
    from pathlib import Path

    from spore_core import VerifierInput, VerifierVerdict, VerifierVerdictPassed
    from spore_core.harness import BaseSandboxProvider

    class _RecordingVerifier:
        def __init__(self) -> None:
            self.seen: list[VerifierInput] = []

        async def verify(self, input: VerifierInput) -> VerifierVerdict:
            self.seen.append(input)
            return VerifierVerdictPassed()

        def max_iterations(self) -> int:
            return 3

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        incomplete = '{"complete": false, "remaining": ["task A"]}'
        spore = root / ".spore"
        spore.mkdir()
        (spore / "progress.json").write_text(incomplete)

        class _WorkspaceSandbox(BaseSandboxProvider):
            async def validate(self, call: object) -> None:
                return None

            def workspace_root(self) -> Path:
                return root

        # The worker keeps progress incomplete on every turn it writes.
        class _ProgressAgent(_RecordingAgent):
            async def turn(self, context: object) -> FinalResponse:
                (spore / "progress.json").write_text(incomplete)
                return await super().turn(context)

        worker = _ProgressAgent(["window done"] * 16)
        verifier = _RecordingVerifier()
        cfg = HarnessConfig(
            agent=worker,
            tool_registry=ScriptedToolRegistry(),
            sandbox=_WorkspaceSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            max_resets=2,
            registry=ExecutionRegistry.builder().verifier("", verifier).build(),
        )
        h = StandardHarness(cfg)

        # inner is a genuine SelfVerifying combinator (NOT a ReAct).
        task = Task.new(
            "keep going",
            SessionId("ralph-nonreact"),
            RalphConfig(
                inner=SelfVerifyingConfig(inner=_react_structured(), evaluator=""),
                agent="",
            ),
            budget=BudgetLimits(max_turns=8),
        )
        r = await h.run(HarnessRunOptions(task))
        from spore_core import HaltReasonRalphCompletionUnmet

        assert isinstance(r, RunResultFailure), f"expected Failure, got {r!r}"
        assert isinstance(r.reason, HaltReasonRalphCompletionUnmet)
        assert r.reason.iterations == 2, "exactly max_resets windows ran"
        # The inner SelfVerifying verifier fired at least once per window (>=2). A
        # hardcoded-ReAct window would record ZERO verifier invocations.
        assert len(verifier.seen) >= 2, (
            f"inner SelfVerifying verifier must fire >=1 per window; got {len(verifier.seen)}"
        )


async def test_hill_climbing_runs_non_react_inner_per_iteration() -> None:
    """HillClimbing[inner: PlanExecute[ReAct, ReAct]] improve-then-stagnate — each
    iteration recurses the inner PlanExecute's WHOLE loop (firing a plan turn +
    execute step) before the metric eval. baseline 1.0, iter1 2.0 (improve→keep),
    iter2 0.5 (regress→discard, stagnation hits cap 1 ⇒ halt).

    #138 AC1: the durable task_list is project-scoped, so iteration 1 authors it and
    iteration 2 SKIPS re-planning (the list already exists). The plan phase fires
    EXACTLY ONCE across the whole climb — which still proves the recursion (a ReAct
    proposer would fire it zero times).
    Mirrors Rust's ``hill_climbing_runs_non_react_inner_per_iteration``."""
    import tempfile
    from pathlib import Path
    from typing import Any

    from spore_core.harness import BaseSandboxProvider, CommandOutput
    from spore_core.metric import MetricResult
    from spore_core.termination import SessionStateSnapshot

    class _ScriptedMetric:
        def __init__(self, seq: list[float]) -> None:
            self._seq = list(seq)
            self._i = 0

        async def evaluate(self, sandbox: Any, snapshot: SessionStateSnapshot) -> MetricResult:
            v = self._seq[self._i] if self._i < len(self._seq) else self._seq[-1]
            self._i += 1
            return MetricResult(value=v, raw_output="", duration=0.0)

        def direction(self) -> str:
            return "maximize"

        def description(self) -> str:
            return "scripted metric"

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        class _Sandbox(BaseSandboxProvider):
            async def validate(self, call: object) -> None:
                return None

            async def execute_command(
                self, command: str, args: list[str], working_dir: Any = None, timeout: Any = None
            ) -> CommandOutput:
                return CommandOutput(stdout="", stderr="", exit_code=0, timed_out=False)

            def workspace_root(self) -> Path:
                return root

        # Iteration 1 fires a plan JSON turn + execute step; #138 AC1 makes
        # iteration 2 skip the (now-durable) plan phase.
        worker = _RecordingAgent(
            [
                '{"tasks":["only"],"rationale":"r"}',
                "changed iter1",
            ]
        )
        evaluator = _ScriptedMetric([1.0, 2.0, 0.5])
        cfg = HarnessConfig(
            agent=worker,
            tool_registry=ScriptedToolRegistry(),
            sandbox=_Sandbox(),
            context_manager=_StoringContextManager(),
            termination_policy=AlwaysContinuePolicy(),
            metric_evaluator=evaluator,
            # #138 AC1: a real durable store so iteration 1's task_list survives
            # into iteration 2 (which then skips re-planning). The no-op default
            # would drop the write and force a re-plan every iteration.
            storage=StorageProvider.single(InMemoryStorageProvider()),
        )
        h = StandardHarness(cfg)

        # inner is a genuine PlanExecute combinator (NOT a ReAct).
        task = Task.new(
            "optimize it",
            SessionId("hc-nonreact"),
            HillClimbingConfig(
                inner=PlanExecuteConfig(
                    plan=_react_structured(), execute=ReactConfig.per_loop(2**31 - 1)
                ),
                direction="maximize",
                max_stagnation=1,
                revert_on_no_improvement=False,
                min_improvement_delta=0.0,
                evaluator="",
            ),
            budget=BudgetLimits(max_turns=50),
        )
        r = await h.run(HarnessRunOptions(task))
        from spore_core import HaltReasonStagnationLimitReached

        assert isinstance(r, RunResultFailure), f"expected Failure, got {r!r}"
        assert isinstance(r.reason, HaltReasonStagnationLimitReached)
        # #138 AC1: the inner PlanExecute fired its plan turn EXACTLY ONCE — the
        # first iteration authored the durable task_list; later iterations skip
        # re-planning. A hardcoded-ReAct proposer would record ZERO plan turns, so
        # a single plan turn still proves the genuine recursion into PlanExecute.
        plan_turns = sum(1 for c in worker.seen_text() if "step-by-step plan" in c)
        assert plan_turns == 1, (
            f"inner PlanExecute plan phase fires once (then #138 AC1 skips "
            f"re-planning); saw {worker.seen_text()!r}"
        )
