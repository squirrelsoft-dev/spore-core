"""Tests for the PlanArtifact -> TaskList parser (issue #72).

Mirrors the unit + fixture-replay tests in
``rust/crates/spore-core/src/tasklist.rs`` (``plan_artifact_to_task_list``).
Outcomes MUST be byte-identical across all four languages — the shared fixtures
under ``fixtures/plan_to_tasklist/`` are ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from spore_core import PlanArtifact, plan_artifact_to_task_list
from spore_core.tasklist import TaskList, TaskStatus

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "fixtures" / "plan_to_tasklist"


def _artifact(tasks: list[str], rationale: str = "") -> PlanArtifact:
    return PlanArtifact(tasks=tasks, rationale=rationale)


# ============================================================================
# Unit tests — one per rule
# ============================================================================


def test_one_task_per_step_in_order_all_pending() -> None:
    task_list = plan_artifact_to_task_list(_artifact(["first", "second", "third"]))
    assert [t.description for t in task_list.tasks] == ["first", "second", "third"]
    assert all(t.status is TaskStatus.PENDING for t in task_list.tasks)


def test_assigns_sequential_ids() -> None:
    task_list = plan_artifact_to_task_list(_artifact(["a", "b", "c"]))
    assert [t.id for t in task_list.tasks] == [1, 2, 3]
    assert task_list.next_id == 4


def test_is_deterministic() -> None:
    artifact = _artifact(["x", "y"], "why")
    assert plan_artifact_to_task_list(artifact) == plan_artifact_to_task_list(artifact)


def test_keeps_descriptions_verbatim() -> None:
    task_list = plan_artifact_to_task_list(_artifact(["  spaced  ", ""]))
    assert task_list.tasks[0].description == "  spaced  "
    assert task_list.tasks[1].description == ""
    assert task_list.tasks[0].id == 1
    assert task_list.tasks[1].id == 2


def test_empty_plan_yields_default_list() -> None:
    task_list = plan_artifact_to_task_list(_artifact([]))
    assert task_list == TaskList()
    assert task_list.tasks == []
    assert task_list.next_id == 1


def test_empty_plan_serializes_canonically() -> None:
    task_list = plan_artifact_to_task_list(_artifact([]))
    assert task_list.to_json() == '{"tasks":[],"next_id":1}'


def test_drops_rationale() -> None:
    task_list = plan_artifact_to_task_list(_artifact(["do thing"], "SECRET_RATIONALE_TOKEN"))
    serialized = task_list.to_json()
    assert "SECRET_RATIONALE_TOKEN" not in serialized
    assert "rationale" not in serialized


def test_result_serializes_byte_identical_canonical() -> None:
    task_list = plan_artifact_to_task_list(_artifact(["alpha", "beta"], "r"))
    serialized = task_list.to_json()
    assert serialized == (
        '{"tasks":['
        '{"id":1,"description":"alpha","status":"pending"},'
        '{"id":2,"description":"beta","status":"pending"}'
        '],"next_id":3}'
    )
    # Round-trip is byte-identical.
    reparsed = TaskList.from_json(serialized)
    assert reparsed.to_json() == serialized
    assert reparsed == task_list


def test_builds_fresh_list_no_merge() -> None:
    # Two distinct artifacts each start from a fresh default list — ids restart
    # at 1, never continuing from a prior parse (no replanning / merge).
    first = plan_artifact_to_task_list(_artifact(["a", "b"]))
    second = plan_artifact_to_task_list(_artifact(["c"]))
    assert [t.id for t in first.tasks] == [1, 2]
    assert [t.id for t in second.tasks] == [1]
    assert second.next_id == 2


# ============================================================================
# Fixture replay (ground truth — fixtures/plan_to_tasklist/cases.json)
# ============================================================================


def test_fixture_replay_plan_to_tasklist() -> None:
    raw = (FIXTURES / "cases.json").read_text(encoding="utf-8")
    cases = json.loads(raw)
    assert len(cases) >= 1, "expected >=1 case"

    for case in cases:
        name = case["name"]
        artifact = PlanArtifact(**case["input"])
        got = plan_artifact_to_task_list(artifact)

        # Byte-for-byte canonical serialization equality against `expected`.
        got_json = got.to_json()
        want_json = TaskList.from_dict(case["expected"]).to_json()
        assert got_json == want_json, f"case {name}: canonical bytes"
