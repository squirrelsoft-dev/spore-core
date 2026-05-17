"""Tests for :class:`StandardTerminationPolicy` — issue #13.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/termination.rs`` ``tests`` module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spore_core.agent import AgentErrorEmpty
from spore_core.harness import BudgetLimits, BudgetSnapshot, SessionId, TaskId
from spore_core.memory import Timestamp
from spore_core.middleware import HookPoint
from spore_core.sensor import SensorId, SensorOutcome, SensorResult
from spore_core.termination import (
    FixedCompletionCheck,
    SessionStateSnapshot,
    StandardTerminationPolicy,
    TerminationContinueDecision,
    TerminationDecision,
    TerminationFailureAgentError,
    TerminationFailureCompletionCheckFailed,
    TerminationFailureHumanHalted,
    TerminationFailureMaxRetriesExhausted,
    TerminationFailureMiddlewareHalt,
    TerminationFailurePolicyViolation,
    TerminationFailureReason,
    TerminationFailureUnrecoverableSensorHalt,
    TerminationHaltBudgetExceeded,
    TerminationHaltFailure,
    TerminationHaltSuccess,
    TerminationInput,
    check_budget_default,
)


def _snapshot() -> SessionStateSnapshot:
    return SessionStateSnapshot(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
    )


def _input_at(turn: int, done: bool) -> TerminationInput:
    return TerminationInput(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        turn_number=turn,
        agent_claims_done=done,
        agent_response="ok",
        budget_used=BudgetSnapshot(),
        budget_limits=BudgetLimits(),
        sensor_results=[],
        session_state=_snapshot(),
    )


def _sensor_result(sid: str, outcome: SensorOutcome) -> SensorResult:
    return SensorResult(
        sensor_id=SensorId(sid),
        outcome=outcome,
        observation=None,
        detail=outcome.value,
        fired_at=Timestamp("2026-05-17T00:00:00Z"),
    )


# ── Rule: budget is always checked first ─────────────────────────────────


@pytest.mark.asyncio
async def test_budget_hard_stop_when_done() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.budget_used.turns = 5
    inp.budget_limits.max_turns = 5
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltBudgetExceeded)
    assert decision.limit_type == "turns"


@pytest.mark.asyncio
async def test_budget_hard_stop_when_not_done() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, False)
    inp.budget_used.turns = 5
    inp.budget_limits.max_turns = 5
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltBudgetExceeded)


def test_budget_check_covers_every_limit_type() -> None:
    cases: list[tuple[BudgetSnapshot, BudgetLimits, str]] = [
        (BudgetSnapshot(turns=3), BudgetLimits(max_turns=3), "turns"),
        (
            BudgetSnapshot(input_tokens=10),
            BudgetLimits(max_input_tokens=10),
            "input_tokens",
        ),
        (
            BudgetSnapshot(output_tokens=10),
            BudgetLimits(max_output_tokens=10),
            "output_tokens",
        ),
        (
            BudgetSnapshot(wall_time=10),
            BudgetLimits(max_wall_time=10),
            "wall_time",
        ),
        (BudgetSnapshot(cost_usd=1.0), BudgetLimits(max_cost_usd=1.0), "cost_usd"),
    ]
    for snap, lim, want in cases:
        got = check_budget_default(snap, lim)
        assert isinstance(got, TerminationHaltBudgetExceeded)
        assert got.limit_type == want
    assert check_budget_default(BudgetSnapshot(), BudgetLimits()) is None


# ── Rule: not-done always continues (after budget) ───────────────────────


@pytest.mark.asyncio
async def test_not_done_continues() -> None:
    p = StandardTerminationPolicy.with_null_check()
    decision = await p.evaluate(_input_at(1, False))
    assert isinstance(decision, TerminationContinueDecision)


# ── Rule: sensor halt becomes UnrecoverableSensorHalt ────────────────────


@pytest.mark.asyncio
async def test_sensor_halt_overrides_completion_success() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.sensor_results.append(_sensor_result("guardrail", SensorOutcome.HALT))
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltFailure)
    assert isinstance(decision.reason, TerminationFailureUnrecoverableSensorHalt)
    assert decision.reason.sensor_id == "guardrail"


@pytest.mark.asyncio
async def test_sensor_warn_does_not_halt() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.sensor_results.append(_sensor_result("guardrail", SensorOutcome.WARN))
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltSuccess)


# ── Rule: completion check returning a reason ⇒ Continue ─────────────────


@pytest.mark.asyncio
async def test_incomplete_check_continues_with_agent_claimed_done() -> None:
    p = StandardTerminationPolicy(FixedCompletionCheck.incomplete("feature B not implemented"))
    decision = await p.evaluate(_input_at(1, True))
    assert isinstance(decision, TerminationContinueDecision)


# ── Rule: completion check returning None ⇒ HaltSuccess(summary) ─────────


@pytest.mark.asyncio
async def test_complete_check_halts_success_with_summary() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.agent_response = "all green"
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltSuccess)
    assert decision.summary == "all green"


@pytest.mark.asyncio
async def test_halt_success_summary_empty_when_no_response() -> None:
    p = StandardTerminationPolicy.with_null_check()
    inp = _input_at(1, True)
    inp.agent_response = None
    decision = await p.evaluate(inp)
    assert isinstance(decision, TerminationHaltSuccess)
    assert decision.summary == ""


# ── Rule: HaltFailure carries typed reason ────────────────────────────────


def test_halt_failure_reason_is_typed() -> None:
    """Round-trip every variant through JSON to prove the wire format
    matches the cross-language fixture schema."""
    reason_adapter: TypeAdapter[TerminationFailureReason] = TypeAdapter(TerminationFailureReason)
    cases: list[TerminationFailureReason] = [
        TerminationFailureCompletionCheckFailed(detail="nope"),
        TerminationFailureMaxRetriesExhausted(tool="bash", attempts=3),
        TerminationFailureUnrecoverableSensorHalt(sensor_id=SensorId("g"), detail="tripped"),
        TerminationFailureMiddlewareHalt(hook=HookPoint.BEFORE_TURN, reason="veto"),
        TerminationFailureAgentError(error=AgentErrorEmpty()),
        TerminationFailurePolicyViolation(detail="policy"),
        TerminationFailureHumanHalted(),
    ]
    for c in cases:
        as_json = c.model_dump_json()
        back = reason_adapter.validate_json(as_json)
        assert back == c


# ── Fixture replay ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fixture_replay_basic() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4] / "fixtures" / "termination_policy" / "basic.json"
    )
    suite = json.loads(fixture_path.read_text())
    decision_adapter: TypeAdapter[TerminationDecision] = TypeAdapter(TerminationDecision)
    for case in suite["cases"]:
        cc_spec = case["completion_check"]
        if cc_spec["kind"] == "complete":
            check = FixedCompletionCheck.complete()
        elif cc_spec["kind"] == "incomplete":
            check = FixedCompletionCheck.incomplete(cc_spec["reason"])
        else:
            raise AssertionError(f"unknown completion check: {cc_spec!r}")

        policy = StandardTerminationPolicy(check)
        sensor_results = [SensorResult.model_validate(r) for r in case.get("sensor_results", [])]
        inp = TerminationInput(
            session_id=SessionId("fixture"),
            task_id=TaskId("fixture-task"),
            turn_number=1,
            agent_claims_done=case["agent_claims_done"],
            agent_response=case.get("agent_response"),
            budget_used=BudgetSnapshot.model_validate(case["budget_used"]),
            budget_limits=BudgetLimits.model_validate(case["budget_limits"]),
            sensor_results=sensor_results,
            session_state=_snapshot(),
        )
        got = await policy.evaluate(inp)
        got_json = json.loads(decision_adapter.dump_json(got).decode("utf-8"))
        assert got_json == case["expected"], (
            f"fixture case `{case['name']}` produced unexpected decision: {got_json!r}"
        )
