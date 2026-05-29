"""TaskList tool (#71): the single mutating tool over the persisted task list.

Mirrors ``rust/crates/spore-core/src/tools/tasklist.rs``.

One tool, :class:`TaskListTool` (``NAME = "task_list"``), dispatched on an
``action`` discriminator (``add_task``, ``update_task``, ``complete_task``,
``list_tasks``). See :mod:`spore_core.tasklist` for the types, the transition
matrix, and the disk-persistence helpers this tool drives.

The tool is read-modify-write over the on-disk
:data:`spore_core.tasklist.TASK_LIST_PATH`:

1. parse params (bad input → recoverable error),
2. load the current list (absent → default),
3. apply the action (domain errors → recoverable),
4. persist the (possibly mutated) list,
5. return the serialized current list as success content.

CRITICAL: this tool is NOT annotated ``read_only``. ``read_only`` tools are run
CONCURRENTLY by ``dispatch_all``, and a concurrent read-modify-write over the
same file would race. Leaving ``read_only`` false makes the registry dispatch it
sequentially. ``destructive`` / ``open_world`` are also left false so it is not
treated as an irreversible side effect.
"""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.sandbox import SandboxViolationException
from spore_core.tasklist import (
    TaskListError,
    load_task_list,
    store_task_list,
)
from spore_core.tool_registry import ToolAnnotations, ToolSchema

from .error import SandboxViolationError, ToolExecutionError
from .params import (
    AddTaskParams,
    CompleteTaskParams,
    UpdateTaskParams,
    parse_task_list_params,
)


class TaskListTool:
    """Manage the persisted task list: add, update, complete, or list tasks."""

    NAME = "task_list"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        # Fields kept sorted/stable for cache stability: `action` (required) plus
        # the union of per-action fields.
        return ToolSchema(
            name=cls.NAME,
            description="Manage the persisted task list: add, update, complete, or list tasks",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add_task", "complete_task", "list_tasks", "update_task"],
                    },
                    "description": {"type": "string"},
                    "id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "completed", "in_progress", "pending"],
                    },
                },
                "required": ["action"],
            },
            # Intentionally NOT read_only: this tool mutates shared on-disk state
            # and must dispatch sequentially. See module docs.
            annotations=ToolAnnotations(),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        # 1. Parse params (bad input → recoverable).
        try:
            action, params = parse_task_list_params(call)
        except ToolExecutionError as e:
            return e.to_tool_output()

        # 2. Load the current list (absent → default; malformed → recoverable).
        try:
            task_list = await load_task_list(sandbox)
        except SandboxViolationException as e:
            return SandboxViolationError(violation=e.violation).to_tool_output()
        except (ValueError, OSError) as e:
            return ToolOutputError(message=f"could not parse task list: {e}", recoverable=True)

        # 3. Apply the action. Domain errors → recoverable. `list_tasks` does not
        #    mutate.
        mutated = False
        try:
            if isinstance(params, AddTaskParams):
                task_list.add(params.description)
                mutated = True
            elif isinstance(params, UpdateTaskParams):
                task_list.update(params.id, params.status, params.description)
                mutated = True
            elif isinstance(params, CompleteTaskParams):
                task_list.complete(params.id)
                mutated = True
            else:  # ListTasksParams — no mutation.
                assert action == "list_tasks"  # noqa: S101 — invariant
        except TaskListError as e:
            return ToolOutputError(message=e.message, recoverable=True)

        # 4. Persist the (possibly mutated) list. list_tasks skips the write.
        if mutated:
            try:
                await store_task_list(task_list, sandbox)
            except SandboxViolationException as e:
                return SandboxViolationError(violation=e.violation).to_tool_output()
            except OSError as e:
                return ToolOutputError(
                    message=f"could not persist task list: {e}", recoverable=True
                )

        # 5. Return the serialized current list.
        return ToolOutputSuccess(content=task_list.to_json(), truncated=False)


__all__ = ["TaskListTool"]
