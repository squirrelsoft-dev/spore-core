"""Tests for the PlanExecute capture grammar (issue #70).

Mirrors the unit tests in ``rust/crates/spore-core/src/plan.rs``. Each test
exercises one rule of the Q3 capture grammar; :func:`capture_plan_artifact` must
be deterministic and total (R9) and byte-identical across all four languages.
"""

from __future__ import annotations

import pytest

from spore_core import PlanArtifact, PlanPhaseError, capture_plan_artifact
from spore_core.plan import PLAN_EXECUTE_EXTRAS_KEY


# R3 / R9: a known JSON object captures to exact tasks + rationale.
def test_captures_plain_json_object() -> None:
    artifact = capture_plan_artifact('{"tasks":["a","b","c"],"rationale":"because"}')
    assert artifact.tasks == ["a", "b", "c"]
    assert artifact.rationale == "because"


# Q3: surrounding ASCII whitespace is trimmed.
def test_trims_surrounding_whitespace() -> None:
    artifact = capture_plan_artifact('\n\t  {"tasks":["x"]}  \r\n')
    assert artifact.tasks == ["x"]
    assert artifact.rationale == ""


# Q3 fence-strip: ```json … ``` is stripped before parsing.
def test_strips_json_fence() -> None:
    text = '```json\n{"tasks":["step 1","step 2"],"rationale":"r"}\n```'
    artifact = capture_plan_artifact(text)
    assert artifact.tasks == ["step 1", "step 2"]
    assert artifact.rationale == "r"


# Q3 fence-strip: a bare ``` fence (no language tag) is also stripped.
def test_strips_bare_fence() -> None:
    artifact = capture_plan_artifact('```\n{"tasks":["only"]}\n```')
    assert artifact.tasks == ["only"]


# Q3 fence-strip: uppercase ```JSON tag is stripped (language-tag agnostic).
def test_strips_uppercase_json_fence() -> None:
    artifact = capture_plan_artifact('```JSON\n{"tasks":["u"]}\n```')
    assert artifact.tasks == ["u"]


# Q3: rationale is optional and defaults to "".
def test_rationale_defaults_to_empty() -> None:
    artifact = capture_plan_artifact('{"tasks":["a"]}')
    assert artifact.rationale == ""


# Q3: an empty tasks array is ALLOWED (degenerate-plan handling is #72).
def test_empty_tasks_array_is_allowed() -> None:
    artifact = capture_plan_artifact('{"tasks":[]}')
    assert artifact.tasks == []


# Q3: task strings are kept verbatim — no trimming or filtering.
def test_tasks_kept_verbatim() -> None:
    artifact = capture_plan_artifact('{"tasks":["  spaced  ",""]}')
    assert artifact.tasks == ["  spaced  ", ""]


# R9: malformed inputs raise UnparseablePlan, never an unexpected error.
def test_invalid_json_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact("not json at all")
    assert ei.value.kind == "unparseable_plan"


def test_non_object_top_level_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact("[1,2,3]")
    assert ei.value.kind == "unparseable_plan"


def test_missing_tasks_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact('{"rationale":"x"}')
    assert ei.value.kind == "unparseable_plan"


def test_tasks_not_array_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact('{"tasks":"a"}')
    assert ei.value.kind == "unparseable_plan"


def test_non_string_task_element_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact('{"tasks":["a",2]}')
    assert ei.value.kind == "unparseable_plan"


def test_non_string_rationale_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact('{"tasks":["a"],"rationale":5}')
    assert ei.value.kind == "unparseable_plan"


def test_empty_input_is_unparseable() -> None:
    with pytest.raises(PlanPhaseError) as ei:
        capture_plan_artifact("   \n  ")
    assert ei.value.kind == "unparseable_plan"


# R9: deterministic — identical input yields an identical artifact.
def test_capture_is_deterministic() -> None:
    text = '```json\n{"tasks":["a","b"],"rationale":"r"}\n```'
    a1 = capture_plan_artifact(text)
    a2 = capture_plan_artifact(text)
    assert a1 == a2
    assert isinstance(a1, PlanArtifact)


def test_extras_key_is_stable() -> None:
    # The cross-language extras key must stay "plan_execute".
    assert PLAN_EXECUTE_EXTRAS_KEY == "plan_execute"
