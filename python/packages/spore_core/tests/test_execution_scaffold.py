"""Tests for the Composable Execution runtime scaffold (issue #123):
``StrategyOutcome`` + ``ExecutionContext`` / ``BudgetContext`` / ``BudgetStack``
/ ``SpanStack``.

SCAFFOLD ONLY — ``BudgetContext.charge`` is pure arithmetic against a per-scope
step allowance; the behavior-chain walk, continue-consumption, and persistence
land in the budget-enforcement slice (#124+). These are runtime-only types
(plain dataclasses, never serialized), so there are NO fixtures.
"""

from __future__ import annotations

import pytest

from spore_core.execution_registry import ExecutionRegistry
from spore_core.harness import (
    AggregateUsage,
    BudgetContext,
    BudgetExhausted,
    BudgetExhaustedContinue,
    BudgetExhaustedEscalate,
    BudgetExhaustedFail,
    BudgetPolicyPerAttempt,
    BudgetPolicyPerLoop,
    BudgetPolicyTotalSteps,
    BudgetPolicyUnlimited,
    BudgetStack,
    ExecutionContext,
    HarnessErrorStrategyNotFound,
    SessionState,
    SpanStack,
    StrategyOutcome,
    StrategyOutcomeBudgetExhausted,
    StrategyOutcomeComplete,
    StrategyOutcomeFailed,
)
from spore_core.observability import new_span_id


def _continue(max_continues: int) -> BudgetExhaustedContinue:
    return BudgetExhaustedContinue(
        max_continues=max_continues,
        on_exhausted=BudgetExhaustedFail(),
    )


# ---------------------------------------------------------------------------
# charge: within allowance increments steps_taken
# ---------------------------------------------------------------------------


def test_charge_within_allowance_increments_steps() -> None:
    cx = BudgetContext(
        policy=BudgetPolicyPerLoop(value=5),
        behavior=BudgetExhaustedFail(),
        phase="loop",
    )
    cx.charge(2)
    assert cx.steps_taken == 2
    cx.charge(3)  # exactly to the cap
    assert cx.steps_taken == 5
    assert cx.remaining() == 0


# ---------------------------------------------------------------------------
# charge: overflow raises with correct fields and does NOT mutate
# ---------------------------------------------------------------------------


def test_charge_overflow_raises_with_state_and_no_mutation() -> None:
    cx = BudgetContext(
        policy=BudgetPolicyTotalSteps(value=3),
        behavior=_continue(2),
        phase="phase-x",
    )
    cx.charge(3)
    with pytest.raises(BudgetExhausted) as exc:
        cx.charge(1)
    err = exc.value
    assert err.policy == BudgetPolicyTotalSteps(value=3)
    assert err.behavior == _continue(2)
    assert err.steps_taken == 3
    assert err.continues_used == 0
    assert err.phase == "phase-x"
    # state unchanged on failure
    assert cx.steps_taken == 3


# ---------------------------------------------------------------------------
# charge: Unlimited never exhausts
# ---------------------------------------------------------------------------


def test_charge_unlimited_never_exhausts() -> None:
    cx = BudgetContext(
        policy=BudgetPolicyUnlimited(),
        behavior=BudgetExhaustedFail(),
        phase="root",
    )
    cx.charge(2**31 - 1)
    cx.charge(100)
    assert cx.remaining() is None


# ---------------------------------------------------------------------------
# remaining(): capped + Unlimited
# ---------------------------------------------------------------------------


def test_remaining_capped_and_unlimited() -> None:
    capped = BudgetContext(
        policy=BudgetPolicyPerAttempt(value=10),
        behavior=BudgetExhaustedFail(),
        phase="attempt",
    )
    assert capped.remaining() == 10
    capped.charge(4)
    assert capped.remaining() == 6

    unlimited = BudgetContext(
        policy=BudgetPolicyUnlimited(),
        behavior=BudgetExhaustedFail(),
        phase="u",
    )
    assert unlimited.remaining() is None


# ---------------------------------------------------------------------------
# continues_remaining(): Continue / Escalate / Fail
# ---------------------------------------------------------------------------


def test_continues_remaining_per_behavior() -> None:
    cont = BudgetContext(
        policy=BudgetPolicyUnlimited(),
        behavior=_continue(3),
        phase="c",
    )
    assert cont.continues_remaining() == 3
    cont.continues_used = 2
    assert cont.continues_remaining() == 1
    cont.continues_used = 5  # saturates to 0, never negative
    assert cont.continues_remaining() == 0

    esc = BudgetContext(
        policy=BudgetPolicyUnlimited(),
        behavior=BudgetExhaustedEscalate(),
        phase="e",
    )
    assert esc.continues_remaining() == 0

    fail = BudgetContext(
        policy=BudgetPolicyUnlimited(),
        behavior=BudgetExhaustedFail(),
        phase="f",
    )
    assert fail.continues_remaining() == 0


# ---------------------------------------------------------------------------
# StrategyOutcome variant discrimination (BudgetExhausted vs Failed)
# ---------------------------------------------------------------------------


def test_strategy_outcome_budget_exhausted_distinct_from_failed() -> None:
    exhausted: StrategyOutcome = StrategyOutcomeBudgetExhausted(
        policy=BudgetPolicyPerLoop(value=1),
        behavior=BudgetExhaustedFail(),
        steps_taken=1,
        continues_used=0,
        phase="p",
        partial_output="partial",
    )
    failed: StrategyOutcome = StrategyOutcomeFailed(
        error=HarnessErrorStrategyNotFound(key="missing")
    )
    complete: StrategyOutcome = StrategyOutcomeComplete("done")

    # A BudgetExhausted is NOT a Failed — callers can distinguish via isinstance.
    assert isinstance(exhausted, StrategyOutcomeBudgetExhausted)
    assert not isinstance(exhausted, StrategyOutcomeFailed)
    assert exhausted.partial_output == "partial"
    assert isinstance(failed, StrategyOutcomeFailed)
    assert isinstance(complete, StrategyOutcomeComplete)
    assert complete.output == "done"


def test_strategy_outcome_budget_exhausted_partial_output_defaults_none() -> None:
    exhausted = StrategyOutcomeBudgetExhausted(
        policy=BudgetPolicyPerLoop(value=1),
        behavior=BudgetExhaustedFail(),
        steps_taken=1,
        continues_used=0,
        phase="p",
    )
    assert exhausted.partial_output is None


# ---------------------------------------------------------------------------
# BudgetExhausted promotes to StrategyOutcomeBudgetExhausted at the boundary
# ---------------------------------------------------------------------------


def test_budget_exhausted_promotes_to_outcome() -> None:
    cx = BudgetContext(
        policy=BudgetPolicyPerLoop(value=1),
        behavior=_continue(2),
        phase="leaf",
    )
    cx.charge(1)
    try:
        cx.charge(1)
    except BudgetExhausted as err:
        outcome = StrategyOutcomeBudgetExhausted(
            policy=err.policy,
            behavior=err.behavior,
            steps_taken=err.steps_taken,
            continues_used=err.continues_used,
            phase=err.phase,
            partial_output="so far",
        )
    assert outcome.steps_taken == 1
    assert outcome.partial_output == "so far"


# ---------------------------------------------------------------------------
# SpanStack push/pop depth
# ---------------------------------------------------------------------------


def test_span_stack_push_pop_depth() -> None:
    spans = SpanStack()
    assert spans.depth() == 0
    spans.push(new_span_id("a"))
    spans.push(new_span_id("b"))
    assert spans.depth() == 2
    assert spans.pop() == new_span_id("b")
    assert spans.depth() == 1
    spans.pop()
    assert spans.pop() is None  # empty pop is a no-op None


# ---------------------------------------------------------------------------
# BudgetStack push/pop/current/depth
# ---------------------------------------------------------------------------


def test_budget_stack_push_pop_current_depth() -> None:
    budgets = BudgetStack()
    assert budgets.depth() == 0
    assert budgets.current() is None
    assert budgets.pop() is None

    budgets.push(
        BudgetContext(
            policy=BudgetPolicyPerLoop(value=5),
            behavior=BudgetExhaustedFail(),
            phase="x",
        )
    )
    assert budgets.depth() == 1
    current = budgets.current()
    assert current is not None
    current.charge(1)  # mutate in place via current()
    assert budgets.current().steps_taken == 1  # type: ignore[union-attr]
    popped = budgets.pop()
    assert popped is not None and popped.steps_taken == 1
    assert budgets.depth() == 0


# ---------------------------------------------------------------------------
# ExecutionContext shape: one shared context, runtime-only fields
# ---------------------------------------------------------------------------


def test_execution_context_fresh_defaults() -> None:
    registry = ExecutionRegistry.empty()
    cx = ExecutionContext(registry=registry)
    assert cx.registry is registry
    assert isinstance(cx.budgets, BudgetStack)
    assert isinstance(cx.usage, AggregateUsage)
    assert isinstance(cx.session, SessionState)
    assert isinstance(cx.spans, SpanStack)
    assert cx.stream is None
    assert cx.budgets.depth() == 0
    assert cx.spans.depth() == 0


# ---------------------------------------------------------------------------
# Recursive stub strategy threads ExecutionContext + BudgetStack push/pop,
# returning to baseline depth (one shared context for the whole tree)
# ---------------------------------------------------------------------------


class _RecursiveStub:
    """A recursive stub: pushes its own scope, optionally recurses, then pops —
    modeling how a combinator threads the shared context and gives each node
    (incl. siblings) its OWN ``BudgetContext``."""

    def __init__(self, value: int, children: list[_RecursiveStub]) -> None:
        self.value = value
        self.children = children

    async def run(self, cx: ExecutionContext) -> StrategyOutcome:
        baseline = cx.budgets.depth()
        cx.budgets.push(
            BudgetContext(
                policy=BudgetPolicyPerLoop(value=self.value),
                behavior=BudgetExhaustedFail(),
                phase="node",
            )
        )
        # charge against this node's OWN scope
        scope = cx.budgets.current()
        assert scope is not None
        scope.charge(1)
        for child in self.children:
            await child.run(cx)
        # pop our scope; depth returns to baseline
        cx.budgets.pop()
        assert cx.budgets.depth() == baseline
        return StrategyOutcomeComplete("")


async def test_recursive_stub_threads_context_stack_returns_to_baseline() -> None:
    registry = ExecutionRegistry.empty()
    cx = ExecutionContext(registry=registry)
    assert cx.budgets.depth() == 0

    tree = _RecursiveStub(
        value=4,
        children=[
            _RecursiveStub(value=2, children=[]),
            _RecursiveStub(value=3, children=[]),
        ],
    )

    outcome = await tree.run(cx)
    assert isinstance(outcome, StrategyOutcomeComplete)
    # stack fully unwound after the recursive run
    assert cx.budgets.depth() == 0
    # the shared usage/session/spans are reachable through one context
    assert cx.spans.depth() == 0


# ---------------------------------------------------------------------------
# Siblings get DISTINCT BudgetContexts (no shared state)
# ---------------------------------------------------------------------------


def test_siblings_get_distinct_budget_contexts() -> None:
    budgets = BudgetStack()
    budgets.push(
        BudgetContext(
            policy=BudgetPolicyPerLoop(value=5),
            behavior=BudgetExhaustedFail(),
            phase="sib-a",
        )
    )
    # mutate sibling A
    budgets.current().charge(2)  # type: ignore[union-attr]
    a = budgets.pop()
    assert a is not None and a.steps_taken == 2

    # sibling B starts fresh — no shared state with A
    budgets.push(
        BudgetContext(
            policy=BudgetPolicyPerLoop(value=5),
            behavior=BudgetExhaustedFail(),
            phase="sib-b",
        )
    )
    b = budgets.current()
    assert b is not None
    assert b.steps_taken == 0
    assert a.phase != b.phase
