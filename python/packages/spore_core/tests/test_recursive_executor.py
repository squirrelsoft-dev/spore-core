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
    # Both tasks Completed in the persisted list.
    stored = await h.storage().run().get(task.session_id, TASK_LIST_EXTRAS_KEY)
    assert stored is not None
    tl = TaskList.from_dict(stored)  # type: ignore[arg-type]
    assert all(t.status is TaskStatus.COMPLETED for t in tl.tasks)


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
    result = await h.execute_phase(
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
    # Both tasks Completed in the final persisted list.
    stored = await h.storage().run().get(t.session_id, TASK_LIST_EXTRAS_KEY)
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
        .schema("eval-schema", {})
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
                evaluator="a1",
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
