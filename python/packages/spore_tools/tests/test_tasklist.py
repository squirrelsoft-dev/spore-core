"""Tests for the TaskList primitive, tool, and disk persistence (issue #71).

Mirrors the unit + fixture-replay tests in
``rust/crates/spore-core/src/tasklist.rs`` and
``rust/crates/spore-core/src/tools/tasklist.rs``. Outcomes MUST be byte-identical
across all four languages — the shared fixtures under ``fixtures/tasklist/`` are
ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core.harness import (
    BaseSandboxProvider,
    Operation,
    SandboxViolation,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.tasklist import (
    TASK_LIST_PATH,
    Task,
    TaskList,
    TaskListError,
    TaskStatus,
    load_task_list,
    store_task_list,
    validate_transition,
)
from spore_tools.tools.tasklist import TaskListTool

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES = REPO_ROOT / "fixtures" / "tasklist"


# ============================================================================
# helpers
# ============================================================================


class _TempRootSandbox(BaseSandboxProvider):
    """Roots ``.spore/task_list.json`` inside a tempdir so the read-modify-write
    hits a real (isolated) file. Mirrors the Rust ``TempRootSandbox``: no
    boundary checks, the parent dir is created on write by ``store_task_list``."""

    def __init__(self, root: Path) -> None:
        self._root = root

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None

    async def resolve_path(self, path: str, operation: Operation = "read") -> Path:
        return self._root / path


def _sandbox(root: Path) -> _TempRootSandbox:
    return _TempRootSandbox(root)


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=TaskListTool.NAME, input=input_)


def _list_with(*statuses: TaskStatus) -> TaskList:
    tl = TaskList()
    for _ in statuses:
        tl.add("t")
    for i, s in enumerate(statuses):
        tl.tasks[i].status = s
    return tl


def _parse_list(out: object) -> TaskList:
    assert isinstance(out, ToolOutputSuccess), f"expected Success, got {out!r}"
    return TaskList.from_json(out.content)


# ============================================================================
# TaskList primitive (unit)
# ============================================================================


# R1: ids are assigned 1, 2, 3, … sequentially.
def test_ids_are_sequential_from_one() -> None:
    tl = TaskList()
    assert tl.add("a") == 1
    assert tl.add("b") == 2
    assert tl.add("c") == 3
    assert tl.next_id == 4
    assert [t.id for t in tl.tasks] == [1, 2, 3]


# R2: add appends to the end, preserving positional order; new tasks are pending.
def test_add_appends_in_order() -> None:
    tl = TaskList()
    tl.add("first")
    tl.add("second")
    tl.add("third")
    assert [t.description for t in tl.tasks] == ["first", "second", "third"]
    assert all(t.status == TaskStatus.PENDING for t in tl.tasks)


# R3: serializing for a list_tasks action leaves state untouched.
def test_serialize_does_not_mutate() -> None:
    tl = _list_with(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
    before = tl.to_json()
    _ = tl.to_json()
    assert tl.to_json() == before


def test_update_status_valid() -> None:
    tl = _list_with(TaskStatus.PENDING)
    tl.update(1, TaskStatus.IN_PROGRESS)
    assert tl.tasks[0].status == TaskStatus.IN_PROGRESS


def test_update_description_only() -> None:
    tl = _list_with(TaskStatus.PENDING)
    tl.update(1, None, "rewritten")
    assert tl.tasks[0].description == "rewritten"
    assert tl.tasks[0].status == TaskStatus.PENDING


def test_update_status_and_description() -> None:
    tl = _list_with(TaskStatus.PENDING)
    tl.update(1, TaskStatus.BLOCKED, "blocked on x")
    assert tl.tasks[0].status == TaskStatus.BLOCKED
    assert tl.tasks[0].description == "blocked on x"


# update with neither field → no-op success.
def test_update_no_fields_is_noop_success() -> None:
    tl = _list_with(TaskStatus.IN_PROGRESS)
    before = tl.to_json()
    tl.update(1)
    assert tl.to_json() == before


def test_complete_marks_completed() -> None:
    tl = _list_with(TaskStatus.IN_PROGRESS)
    tl.complete(1)
    assert tl.tasks[0].status == TaskStatus.COMPLETED


# R4: unknown id on update/complete → task_not_found.
def test_unknown_id_is_task_not_found() -> None:
    tl = _list_with(TaskStatus.PENDING)
    with pytest.raises(TaskListError) as ei:
        tl.update(99, TaskStatus.COMPLETED)
    assert ei.value.kind == "task_not_found"
    assert ei.value.id == 99
    with pytest.raises(TaskListError) as ei2:
        tl.complete(99)
    assert ei2.value.kind == "task_not_found"


# R5/R6: a rejected transition leaves the task untouched.
def test_rejected_transition_does_not_mutate() -> None:
    tl = _list_with(TaskStatus.COMPLETED)
    before = tl.to_json()
    with pytest.raises(TaskListError) as ei:
        tl.update(1, TaskStatus.IN_PROGRESS)
    assert ei.value.kind == "invalid_transition"
    assert tl.to_json() == before


# DECISION 1: every ALLOWED transition (incl. idempotent self-transitions).
def test_allowed_transitions() -> None:
    allowed = [
        (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
        (TaskStatus.PENDING, TaskStatus.COMPLETED),
        (TaskStatus.PENDING, TaskStatus.BLOCKED),
        (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED),
        (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),
        (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),
        (TaskStatus.BLOCKED, TaskStatus.COMPLETED),
        (TaskStatus.PENDING, TaskStatus.PENDING),
        (TaskStatus.IN_PROGRESS, TaskStatus.IN_PROGRESS),
        (TaskStatus.COMPLETED, TaskStatus.COMPLETED),
        (TaskStatus.BLOCKED, TaskStatus.BLOCKED),
    ]
    for frm, to in allowed:
        validate_transition(1, frm, to)  # must not raise


# DECISION 1 / R6: every transition OUT of completed (except self) rejected.
def test_out_of_completed_rejected() -> None:
    for to in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
        with pytest.raises(TaskListError) as ei:
            validate_transition(7, TaskStatus.COMPLETED, to)
        assert ei.value.kind == "invalid_transition"
        assert ei.value.id == 7
        assert ei.value.from_status == TaskStatus.COMPLETED
        assert ei.value.to_status == to


# The remaining rejected (non-completed-origin) transitions.
def test_other_rejected_transitions() -> None:
    with pytest.raises(TaskListError):
        validate_transition(1, TaskStatus.IN_PROGRESS, TaskStatus.PENDING)
    with pytest.raises(TaskListError):
        validate_transition(1, TaskStatus.BLOCKED, TaskStatus.PENDING)


def test_pending_to_completed_allowed() -> None:
    tl = _list_with(TaskStatus.PENDING)
    tl.complete(1)
    assert tl.tasks[0].status == TaskStatus.COMPLETED


def test_blocked_to_in_progress_and_completed_allowed() -> None:
    tl = _list_with(TaskStatus.BLOCKED, TaskStatus.BLOCKED)
    tl.update(1, TaskStatus.IN_PROGRESS)
    tl.complete(2)
    assert tl.tasks[0].status == TaskStatus.IN_PROGRESS
    assert tl.tasks[1].status == TaskStatus.COMPLETED


# R7: idempotent self-transition is a success and a no-op on state.
def test_idempotent_self_transition() -> None:
    tl = _list_with(TaskStatus.COMPLETED)
    tl.update(1, TaskStatus.COMPLETED)
    tl.complete(1)  # completed -> completed via complete().
    assert tl.tasks[0].status == TaskStatus.COMPLETED


# Reload preserves next_id (ids never reused after a round-trip).
def test_reload_preserves_next_id() -> None:
    tl = TaskList()
    tl.add("a")
    tl.add("b")
    reloaded = TaskList.from_json(tl.to_json())
    assert reloaded.next_id == 3
    assert reloaded.add("c") == 3  # continues from 3, not 1.


# Serde round-trip is byte-identical (re-serializing the parsed form).
def test_serde_round_trip_byte_identical() -> None:
    tl = TaskList()
    tl.add("alpha")
    tl.add("beta")
    tl.update(2, TaskStatus.IN_PROGRESS)
    json1 = tl.to_json()
    parsed = TaskList.from_json(json1)
    assert parsed.to_json() == json1
    assert parsed == tl


# Status snake_case spellings are exact.
def test_status_snake_case_spellings() -> None:
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.IN_PROGRESS.value == "in_progress"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.BLOCKED.value == "blocked"
    assert TaskStatus("in_progress") == TaskStatus.IN_PROGRESS


# Canonical empty-list serialization.
def test_default_serializes_canonically() -> None:
    assert TaskList().to_json() == '{"tasks":[],"next_id":1}'


# Canonical populated-list serialization (exact spelling + field order).
def test_populated_serializes_canonically() -> None:
    tl = TaskList()
    tl.add("write tests")
    tl.update(1, TaskStatus.IN_PROGRESS)
    assert tl.to_json() == (
        '{"tasks":[{"id":1,"description":"write tests","status":"in_progress"}],"next_id":2}'
    )


# ============================================================================
# Disk persistence (unit)
# ============================================================================


async def test_load_absent_file_yields_default(tmp_path: Path) -> None:
    tl = await load_task_list(_sandbox(tmp_path))
    assert tl == TaskList()


async def test_store_then_load_identical(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tl = TaskList()
    tl.add("one")
    tl.update(1, TaskStatus.BLOCKED)
    await store_task_list(tl, sb)
    # File exists at the canonical path under the workspace root.
    assert (tmp_path / TASK_LIST_PATH).exists()
    reloaded = await load_task_list(sb)
    assert reloaded == tl


async def test_load_malformed_raises(tmp_path: Path) -> None:
    target = tmp_path / TASK_LIST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        await load_task_list(_sandbox(tmp_path))


# ============================================================================
# Tool (integration over a real sandbox-backed file)
# ============================================================================


async def test_add_then_list_persists_and_assigns_ids(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tool = TaskListTool()

    r1 = await tool.execute(_call({"action": "add_task", "description": "a"}), sb)
    l1 = _parse_list(r1)
    assert len(l1.tasks) == 1
    assert l1.tasks[0].id == 1
    assert l1.next_id == 2

    r2 = await tool.execute(_call({"action": "add_task", "description": "b"}), sb)
    l2 = _parse_list(r2)
    assert [t.id for t in l2.tasks] == [1, 2]

    assert (tmp_path / TASK_LIST_PATH).exists()

    r3 = await tool.execute(_call({"action": "list_tasks"}), sb)
    assert _parse_list(r3) == l2


async def test_tool_update_status_and_complete(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tool = TaskListTool()
    await tool.execute(_call({"action": "add_task", "description": "x"}), sb)

    r = await tool.execute(_call({"action": "update_task", "id": 1, "status": "in_progress"}), sb)
    assert _parse_list(r).tasks[0].status == TaskStatus.IN_PROGRESS

    r = await tool.execute(_call({"action": "complete_task", "id": 1}), sb)
    assert _parse_list(r).tasks[0].status == TaskStatus.COMPLETED


async def test_tool_update_description(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tool = TaskListTool()
    await tool.execute(_call({"action": "add_task", "description": "x"}), sb)
    r = await tool.execute(_call({"action": "update_task", "id": 1, "description": "y"}), sb)
    assert _parse_list(r).tasks[0].description == "y"


async def test_tool_unknown_id_is_recoverable_error(tmp_path: Path) -> None:
    r = await TaskListTool().execute(
        _call({"action": "complete_task", "id": 42}), _sandbox(tmp_path)
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable


async def test_tool_invalid_transition_out_of_completed_is_recoverable(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tool = TaskListTool()
    await tool.execute(_call({"action": "add_task", "description": "x"}), sb)
    await tool.execute(_call({"action": "complete_task", "id": 1}), sb)
    r = await tool.execute(_call({"action": "update_task", "id": 1, "status": "pending"}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable


async def test_tool_bad_params_is_recoverable_error(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tool = TaskListTool()
    # Unknown action.
    r = await tool.execute(_call({"action": "nope"}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable
    # Missing required field for add_task.
    r2 = await tool.execute(_call({"action": "add_task"}), sb)
    assert isinstance(r2, ToolOutputError)
    assert r2.recoverable
    # Extra/unknown field is rejected (extra=forbid).
    r3 = await tool.execute(_call({"action": "list_tasks", "bogus": 1}), sb)
    assert isinstance(r3, ToolOutputError)
    assert r3.recoverable


def test_schema_is_not_read_only() -> None:
    s = TaskListTool.schema()
    assert not s.annotations.read_only
    assert not s.annotations.destructive
    assert not s.annotations.open_world


async def test_persist_then_reload_yields_identical_list(tmp_path: Path) -> None:
    sb = _sandbox(tmp_path)
    tool = TaskListTool()
    await tool.execute(_call({"action": "add_task", "description": "one"}), sb)
    r = await tool.execute(_call({"action": "add_task", "description": "two"}), sb)
    from_tool = _parse_list(r)
    reloaded = await load_task_list(sb)
    assert from_tool == reloaded


def test_tool_registers_through_standard_registry() -> None:
    from spore_core.tool_registry import StandardToolRegistry

    registry = StandardToolRegistry()
    tool = TaskListTool()
    registry.register(tool, TaskListTool.schema())
    schemas = registry.active_schemas(None)
    assert any(s.name == TaskListTool.NAME for s in schemas)


# ============================================================================
# Fixture replay (ground truth — fixtures/tasklist/*.json)
# ============================================================================


async def test_fixture_replay_operations(tmp_path: Path) -> None:
    scenarios = json.loads((FIXTURES / "operations.json").read_text())
    assert scenarios, "expected >= 1 scenario"
    tool = TaskListTool()
    for sc in scenarios:
        # Fresh isolated workspace per scenario.
        root = tmp_path / sc["name"]
        root.mkdir()
        sb = _sandbox(root)
        for i, step in enumerate(sc["steps"]):
            out = await tool.execute(_call(step["action"]), sb)
            expected = step["expected"]
            if expected["ok"]:
                assert isinstance(out, ToolOutputSuccess), f"{sc['name']} step {i}: {out!r}"
                got = TaskList.from_json(out.content)
                want = TaskList.from_dict(expected["list"])
                assert got == want, f"{sc['name']} step {i}"
            else:
                assert isinstance(out, ToolOutputError), f"{sc['name']} step {i}: {out!r}"
                assert out.recoverable, f"{sc['name']} step {i}: errors must be recoverable"
                if "not found" in out.message:
                    kind = "task_not_found"
                elif "invalid transition" in out.message:
                    kind = "invalid_transition"
                else:
                    kind = "other"
                assert kind == expected["error"], f"{sc['name']} step {i}: {out.message}"


def test_fixture_replay_transitions() -> None:
    cases = json.loads((FIXTURES / "transitions.json").read_text())
    assert cases, "expected >= 1 case"
    for c in cases:
        frm = TaskStatus(c["from"])
        to = TaskStatus(c["to"])
        try:
            validate_transition(1, frm, to)
            got = "ok"
        except TaskListError:
            got = "invalid_transition"
        assert got == c["expected"], f"{c['from']} -> {c['to']}"


def test_fixture_replay_serialization() -> None:
    cases = json.loads((FIXTURES / "serialization.json").read_text())
    assert cases, "expected >= 1 case"
    for c in cases:
        tl = TaskList.from_dict(c["list"])
        # serialize(list) must equal the pinned JSON (byte-identical).
        assert tl.to_json() == c["json"], f"serialize {c['name']}"
        # parse(json) must equal the list.
        parsed = TaskList.from_json(c["json"])
        assert parsed == tl, f"parse {c['name']}"


def test_task_to_dict_field_order() -> None:
    # Field order id, description, status is wire-significant.
    t = Task(id=1, description="d", status=TaskStatus.PENDING)
    assert list(t.to_dict().keys()) == ["id", "description", "status"]
