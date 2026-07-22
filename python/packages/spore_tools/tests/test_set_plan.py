"""Tests for the ``set_plan`` tool (parity plan §5.16).

Mirrors ``rust/crates/spore-core/src/tools/set_plan.rs``.
"""

from __future__ import annotations


from spore_core.harness import (
    BaseSandboxProvider,
    SandboxViolation,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.storage import (
    InMemoryStorageProvider,
    NoOpStorageProvider,
    RunStore,
    project_id_from_canonical_path,
    project_namespace,
)
from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList
from spore_core.tool_registry import ToolContext
from spore_tools.tools.set_plan import SetPlanTool


class _AllowAllSandbox(BaseSandboxProvider):
    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None


def _ctx(run_store: RunStore, key: str = "test-project") -> ToolContext:
    from spore_core import SessionId

    return ToolContext(
        session_id=SessionId("ephemeral"),
        project_id=project_id_from_canonical_path(f"/proj/{key}"),
        run_store=run_store,
        memory_store=NoOpStorageProvider(),
    )


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name="set_plan", input=input_)


async def test_schema_requires_tasks_array() -> None:
    s = SetPlanTool.schema()
    assert s.name == "set_plan"
    assert "tasks" in s.parameters["required"]
    assert s.parameters["properties"]["tasks"]["type"] == "array"


async def test_set_plan_persists_linear_task_list() -> None:
    store = InMemoryStorageProvider()
    ctx = _ctx(store)
    tool = SetPlanTool()
    out = await tool.execute(
        _call({"tasks": ["step one", "step two"], "rationale": "because"}),
        _AllowAllSandbox(),
        ctx,
    )
    assert isinstance(out, ToolOutputSuccess)
    assert "2-step plan" in out.content
    # The blob is readable under the SAME key task_list uses.
    ns = project_namespace(project_id_from_canonical_path("/proj/test-project"))
    blob = await store.get(ns, TASK_LIST_EXTRAS_KEY)
    assert blob is not None
    lst = TaskList.from_dict(blob) if hasattr(TaskList, "from_dict") else None
    if lst is None:
        # The persisted shape is the canonical TaskList dict.
        assert blob["tasks"]
        assert len(blob["tasks"]) == 2
    else:
        assert len(lst.tasks) == 2


async def test_set_plan_rejects_empty_tasks() -> None:
    tool = SetPlanTool()
    out = await tool.execute(
        _call({"tasks": []}), _AllowAllSandbox(), _ctx(InMemoryStorageProvider())
    )
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True
    assert "non-empty array" in out.message


async def test_set_plan_rejects_non_string_element() -> None:
    tool = SetPlanTool()
    out = await tool.execute(
        _call({"tasks": ["ok", 42]}), _AllowAllSandbox(), _ctx(InMemoryStorageProvider())
    )
    assert isinstance(out, ToolOutputError)
    assert "task 1 must be a non-empty string" in out.message


async def test_set_plan_rejects_missing_tasks() -> None:
    tool = SetPlanTool()
    out = await tool.execute(_call({}), _AllowAllSandbox(), _ctx(InMemoryStorageProvider()))
    assert isinstance(out, ToolOutputError)
    assert "non-empty array" in out.message


async def test_set_plan_preset_in_standard_tools() -> None:
    from spore_tools import StandardTools

    bundle = StandardTools.set_plan()
    assert bundle.schema.name == "set_plan"
    assert bundle.implementation.name() == "set_plan"
