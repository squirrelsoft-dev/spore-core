"""SetPlan tool — the planner authors its whole plan in ONE atomic call.

Mirrors ``rust/crates/spore-core/src/tools/set_plan.rs``.

``set_plan(tasks=[...], rationale="...")`` replaces the persisted task list with
the given ordered steps. Because the plan arrives as a TOOL CALL — consumed by
the harness, never rendered as a message — it cannot leak as a raw
``{"tasks":...}`` assistant message, and the execute phase (which is not told to
author plans and is not offered this tool) won't emit one either.

It writes the SAME RunStore blob the ``task_list`` tool authors (durable,
project-scoped, under :data:`~spore_core.tasklist.TASK_LIST_EXTRAS_KEY`), so the
existing plan-capture fallback (``load_task_list``) and the execute-phase
runnable list read it unchanged. Like ``task_list`` it is NOT ``read_only`` — it
mutates that shared blob and must dispatch sequentially.
"""

from __future__ import annotations

from typing import Any

from spore_core.harness import SandboxProvider, ToolOutput, ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.storage import StorageError, project_namespace
from spore_core.tasklist import TASK_LIST_EXTRAS_KEY, TaskList
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

__all__ = ["SetPlanTool"]


class SetPlanTool:
    """``set_plan`` — atomic plan authoring."""

    NAME = "set_plan"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @staticmethod
    def schema() -> ToolSchema:
        return ToolSchema(
            name=SetPlanTool.NAME,
            description=(
                "Record your plan in ONE call: `tasks` = ordered step strings. "
                "Use this instead of writing the plan as text."
            ),
            parameters={
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered, concrete step strings.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Optional one-line reason for the plan.",
                    },
                },
                "required": ["tasks"],
                "type": "object",
            },
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        # Parse `tasks`: a non-empty array of non-empty strings (bad → recoverable).
        tasks_val = call.input.get("tasks")
        if not isinstance(tasks_val, list) or not tasks_val:
            return ToolOutputError(
                message="`tasks` must be a non-empty array of step strings",
                recoverable=True,
            )
        tasks: list[str] = []
        for i, el in enumerate(tasks_val):
            if isinstance(el, str) and el.strip():
                tasks.append(el)
            else:
                return ToolOutputError(
                    message=f"task {i} must be a non-empty string",
                    recoverable=True,
                )

        # Build a linear TaskList (no blockers) from the ordered steps.
        task_list = TaskList()
        for desc in tasks:
            try:
                task_list.add(desc, [])
            except Exception as e:  # noqa: BLE001 — domain error → recoverable
                return ToolOutputError(
                    message=f"could not record task {desc!r}: {e}",
                    recoverable=True,
                )

        # Persist under the SAME durable, project-scoped key `task_list` uses.
        session_id = project_namespace(ctx.project_id)
        value: Any = task_list.to_dict()
        try:
            await ctx.run_store.put(session_id, TASK_LIST_EXTRAS_KEY, value)
        except StorageError as e:
            return ToolOutputError(
                message=f"could not persist plan: {e}",
                recoverable=True,
            )
        return ToolOutputSuccess(content=f"Recorded a {len(tasks)}-step plan.")
