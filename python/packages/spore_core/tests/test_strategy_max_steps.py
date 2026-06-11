"""Tests for ``loop_strategy_max_steps`` / ``strategy_ref_max_steps`` — the
advisory worst-case **turn** bound for a fully-bounded strategy tree (issue
#122, Composable Execution B.8).

The bound is a pre-run advisory figure, NOT an enforcement mechanism. It is
Option-monadic: any ``Unlimited`` node (or a ``Custom`` ref) collapses it to
``None``. PlanExecute is a PER-TASK bound and Ralph is a PER-WINDOW bound;
neither folds in the runtime-data-dependent task_count / max_windows.

This table mirrors the Rust ``strategy_tests`` ``max_steps_*`` suite so the
figure is byte-for-byte cross-language identical.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from spore_core.harness import (
    BudgetPolicyPerAttempt,
    BudgetPolicyPerLoop,
    BudgetPolicyTotalSteps,
    BudgetPolicyUnlimited,
    HillClimbingConfig,
    LoopStrategy,
    PlanExecuteConfig,
    RalphConfig,
    ReactConfig,
    SelfVerifyingConfig,
    StrategyRefBuiltIn,
    StrategyRefCustom,
    loop_strategy_max_steps,
    strategy_ref_max_steps,
)

_STRATEGY = TypeAdapter(LoopStrategy)

# The unbounded-windows sentinel for ``max_stagnation`` (Python's i32-compatible
# stand-in for Rust's ``u32::MAX``).
_MAX_STAGNATION_UNBOUNDED = 2**31 - 1


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "fixtures").is_dir() and (parent / "rust").is_dir():
            return parent
    raise RuntimeError("could not locate spore-core repo root from test file")


def _react(budget) -> ReactConfig:  # type: ignore[no-untyped-def]
    return ReactConfig(budget=budget, agent="a", toolset="t")


def _self_verifying(inner: LoopStrategy) -> SelfVerifyingConfig:
    return SelfVerifyingConfig(inner=inner, evaluator="judge")


def _plan_execute(plan: LoopStrategy, execute: LoopStrategy) -> PlanExecuteConfig:
    return PlanExecuteConfig(plan=plan, execute=execute)


def _ralph(inner: LoopStrategy) -> RalphConfig:
    return RalphConfig(inner=inner, agent="ralph")


def _hill_climbing(inner: LoopStrategy, max_stagnation: int) -> HillClimbingConfig:
    return HillClimbingConfig(
        inner=inner,
        direction="maximize",
        max_stagnation=max_stagnation,
        revert_on_no_improvement=False,
        min_improvement_delta=0.0,
        evaluator="m",
    )


# ---------------------------------------------------------------------------
# ReAct leaf ⇒ budget allowance value (Unlimited ⇒ None)
# ---------------------------------------------------------------------------


def test_react_leaf() -> None:
    assert loop_strategy_max_steps(_react(BudgetPolicyPerLoop(value=4))) == 4
    assert loop_strategy_max_steps(_react(BudgetPolicyTotalSteps(value=7))) == 7
    assert loop_strategy_max_steps(_react(BudgetPolicyPerAttempt(value=5))) == 5
    assert loop_strategy_max_steps(_react(BudgetPolicyUnlimited())) is None


# ---------------------------------------------------------------------------
# SelfVerifying ⇒ inner + 1 (the single evaluator turn)
# ---------------------------------------------------------------------------


def test_self_verifying_adds_one() -> None:
    s = _self_verifying(_react(BudgetPolicyPerLoop(value=12)))
    assert loop_strategy_max_steps(s) == 13


# ---------------------------------------------------------------------------
# PlanExecute ⇒ plan + execute (PER-TASK)
# ---------------------------------------------------------------------------


def test_plan_execute_is_per_task_sum() -> None:
    s = _plan_execute(
        _react(BudgetPolicyPerLoop(value=4)),
        _react(BudgetPolicyPerLoop(value=6)),
    )
    assert loop_strategy_max_steps(s) == 10


# ---------------------------------------------------------------------------
# HillClimbing ⇒ inner × (max_stagnation + 1); sentinel ⇒ None
# ---------------------------------------------------------------------------


def test_hill_climbing() -> None:
    # inner=5, max_stagnation=2 ⇒ 5 * (2 + 1) = 15.
    s = _hill_climbing(_react(BudgetPolicyPerLoop(value=5)), 2)
    assert loop_strategy_max_steps(s) == 15
    # Unbounded sentinel ⇒ unbounded windows ⇒ None.
    unbounded = _hill_climbing(_react(BudgetPolicyPerLoop(value=5)), _MAX_STAGNATION_UNBOUNDED)
    assert loop_strategy_max_steps(unbounded) is None


# ---------------------------------------------------------------------------
# Ralph ⇒ inner (PER-WINDOW)
# ---------------------------------------------------------------------------


def test_ralph_is_per_window() -> None:
    assert loop_strategy_max_steps(_ralph(_react(BudgetPolicyPerLoop(value=9)))) == 9
    # Canonical cordyceps subtree wrapped in Ralph ⇒ per-window 17.
    s = _ralph(
        _plan_execute(
            _react(BudgetPolicyPerLoop(value=4)),
            _self_verifying(_react(BudgetPolicyPerLoop(value=12))),
        )
    )
    assert loop_strategy_max_steps(s) == 17


# ---------------------------------------------------------------------------
# Canonical cordyceps subtree ⇒ 4 + (12 + 1) = 17
# ---------------------------------------------------------------------------


def test_canonical_cordyceps_subtree() -> None:
    subtree = _plan_execute(
        _react(BudgetPolicyPerLoop(value=4)),
        _self_verifying(_react(BudgetPolicyPerLoop(value=12))),
    )
    assert loop_strategy_max_steps(subtree) == 17


# ---------------------------------------------------------------------------
# The whole shared fixture tree (Ralph wraps PlanExecute) ⇒ per-window 17
# ---------------------------------------------------------------------------


def test_cordyceps_fixture_is_17() -> None:
    raw = (_repo_root() / "fixtures/strategy/cordyceps_tree.json").read_text()
    deserialized = _STRATEGY.validate_json(raw)
    assert loop_strategy_max_steps(deserialized) == 17


# ---------------------------------------------------------------------------
# Unlimited anywhere ⇒ None (Option-monadic propagation)
# ---------------------------------------------------------------------------


def test_unlimited_anywhere_is_none() -> None:
    # Plan leaf unlimited ⇒ None.
    assert (
        loop_strategy_max_steps(
            _plan_execute(
                _react(BudgetPolicyUnlimited()),
                _self_verifying(_react(BudgetPolicyPerLoop(value=12))),
            )
        )
        is None
    )
    # Execute's inner ReAct unlimited ⇒ None.
    assert (
        loop_strategy_max_steps(
            _plan_execute(
                _react(BudgetPolicyPerLoop(value=4)),
                _self_verifying(_react(BudgetPolicyUnlimited())),
            )
        )
        is None
    )
    # HillClimbing-wrapped unlimited inner ⇒ None (propagates to root).
    assert (
        loop_strategy_max_steps(_ralph(_hill_climbing(_react(BudgetPolicyUnlimited()), 2))) is None
    )


# ---------------------------------------------------------------------------
# StrategyRef — Custom ⇒ None, BuiltIn ⇒ delegate
# ---------------------------------------------------------------------------


def test_strategy_ref() -> None:
    assert strategy_ref_max_steps(StrategyRefCustom(value="x")) is None
    assert (
        strategy_ref_max_steps(StrategyRefBuiltIn(value=_react(BudgetPolicyPerLoop(value=4)))) == 4
    )


# ---------------------------------------------------------------------------
# Overflow past the u32 ceiling ⇒ None (mirrors Rust's checked_*)
# ---------------------------------------------------------------------------


def test_overflow_is_none() -> None:
    # inner = u32::MAX/2, max_stagnation = 3 ⇒ (u32::MAX/2) * 4 overflows ⇒ None.
    s = _hill_climbing(_react(BudgetPolicyPerLoop(value=(2**32 - 1) // 2)), 3)
    assert loop_strategy_max_steps(s) is None
