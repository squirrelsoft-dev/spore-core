"""Tests for ``BudgetPolicy`` + ``BudgetExhaustedBehavior`` (issue #117).

These are pure, serializable value types from the Composable Execution PRD
(Part B). The wire format is internally tagged on ``kind`` (snake_case) and must
be byte-identical across Rust / TS / Python / Go. ``fixtures/budget_policy/
cases.json`` is the ground-truth corpus for that byte-identity contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from spore_core.harness import (
    BudgetExhaustedBehavior,
    BudgetExhaustedContinue,
    BudgetExhaustedEscalate,
    BudgetExhaustedFail,
    BudgetPolicy,
    BudgetPolicyPerAttempt,
    BudgetPolicyPerLoop,
    BudgetPolicyTotalSteps,
    BudgetPolicyUnlimited,
)

_POLICY = TypeAdapter(BudgetPolicy)
_BEHAVIOR = TypeAdapter(BudgetExhaustedBehavior)


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "fixtures").is_dir() and (parent / "rust").is_dir():
            return parent
    raise RuntimeError("could not locate spore-core repo root from test file")


def _fixture_path() -> Path:
    return _repo_root() / "fixtures/budget_policy/cases.json"


# ---------------------------------------------------------------------------
# BudgetPolicy — per-variant round-trip + exact serialized bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "wire"),
    [
        (BudgetPolicyUnlimited(), '{"kind":"unlimited"}'),
        (BudgetPolicyTotalSteps(value=100), '{"kind":"total_steps","value":100}'),
        (BudgetPolicyPerLoop(value=10), '{"kind":"per_loop","value":10}'),
        (BudgetPolicyPerAttempt(value=3), '{"kind":"per_attempt","value":3}'),
    ],
)
def test_budget_policy_roundtrip_and_bytes(policy: object, wire: str) -> None:
    serialized = _POLICY.dump_json(policy).decode()
    assert serialized == wire
    back = _POLICY.validate_json(serialized)
    assert back == policy


# ---------------------------------------------------------------------------
# BudgetExhaustedBehavior — per-variant round-trip + exact serialized bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("behavior", "wire"),
    [
        (BudgetExhaustedEscalate(), '{"kind":"escalate"}'),
        (BudgetExhaustedFail(), '{"kind":"fail"}'),
        (
            BudgetExhaustedContinue(max_continues=2, on_exhausted=BudgetExhaustedFail()),
            '{"kind":"continue","max_continues":2,"on_exhausted":{"kind":"fail"}}',
        ),
    ],
)
def test_budget_behavior_roundtrip_and_bytes(behavior: object, wire: str) -> None:
    serialized = _BEHAVIOR.dump_json(behavior).decode()
    assert serialized == wire
    back = _BEHAVIOR.validate_json(serialized)
    assert back == behavior


def test_nested_continue_continue_fail_roundtrip() -> None:
    """The deepest case: Continue -> Continue -> Fail. Recursive nesting must
    serialize byte-identically and survive a full round-trip."""
    nested = BudgetExhaustedContinue(
        max_continues=1,
        on_exhausted=BudgetExhaustedContinue(
            max_continues=2,
            on_exhausted=BudgetExhaustedFail(),
        ),
    )
    wire = (
        '{"kind":"continue","max_continues":1,'
        '"on_exhausted":{"kind":"continue","max_continues":2,'
        '"on_exhausted":{"kind":"fail"}}}'
    )
    serialized = _BEHAVIOR.dump_json(nested).decode()
    assert serialized == wire
    back = _BEHAVIOR.validate_json(serialized)
    assert back == nested
    # The recursive node really is a Continue carrying a Continue carrying a Fail.
    assert isinstance(back, BudgetExhaustedContinue)
    assert isinstance(back.on_exhausted, BudgetExhaustedContinue)
    assert isinstance(back.on_exhausted.on_exhausted, BudgetExhaustedFail)


# ---------------------------------------------------------------------------
# Negative tests — no silent default to Continue; unknown/missing kind rejected
# ---------------------------------------------------------------------------


def test_unknown_policy_kind_is_rejected() -> None:
    # PerGoal is intentionally excluded in v1.
    with pytest.raises(ValidationError):
        _POLICY.validate_python({"kind": "per_goal", "value": 1})


def test_missing_policy_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _POLICY.validate_python({"value": 1})


def test_unknown_behavior_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _BEHAVIOR.validate_python({"kind": "retry"})


def test_missing_behavior_kind_does_not_default_to_continue() -> None:
    # A payload with only ``max_continues`` must NOT be silently coerced into a
    # ``Continue`` — there is deliberately no default behavior.
    with pytest.raises(ValidationError):
        _BEHAVIOR.validate_python({"max_continues": 1})


# ---------------------------------------------------------------------------
# Fixture replay — byte-identity against the shared ground-truth corpus
# ---------------------------------------------------------------------------


def test_fixture_byte_identity_roundtrip() -> None:
    raw = json.loads(_fixture_path().read_text())

    for entry in raw["policies"]:
        model = _POLICY.validate_python(entry)
        # ``json.dumps`` with compact separators reproduces the canonical wire
        # form; compare structurally against the round-tripped model.
        round_tripped = json.loads(_POLICY.dump_json(model))
        assert round_tripped == entry

    for entry in raw["behaviors"]:
        model = _BEHAVIOR.validate_python(entry)
        round_tripped = json.loads(_BEHAVIOR.dump_json(model))
        assert round_tripped == entry
