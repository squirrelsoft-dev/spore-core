"""TaskList tool (#71, storage seam #75): the single mutating tool over the
persisted task list.

Mirrors ``rust/crates/spore-core/src/tools/tasklist.rs``.

One tool, :class:`TaskListTool` (``NAME = "task_list"``), dispatched on an
``action`` discriminator (``add_task``, ``update_task``, ``complete_task``,
``list_tasks``). See :mod:`spore_core.tasklist` for the types and the transition
matrix.

Storage seam (#75)
------------------
The tool persists via the :class:`ToolContext`'s
:class:`~spore_core.storage.RunStore` — NOT the sandbox filesystem. It is
read-modify-write keyed by the run's :class:`~spore_core.harness.SessionId`
under :data:`~spore_core.tasklist.TASK_LIST_EXTRAS_KEY` (``"task_list"``):

1. parse params (bad input → recoverable error),
2. ``ctx.run_store.get(session_id, "task_list")`` (absent → default empty
   :class:`~spore_core.tasklist.TaskList`),
3. apply the action (domain errors → recoverable),
4. on a mutating action, ``ctx.run_store.put(session_id, "task_list", value)``,
5. return the serialized current list as success content.

Shared key
~~~~~~~~~~
This standalone tool and the harness-side PlanExecute execute loop persist
under the SAME :class:`~spore_core.storage.RunStore` key (``"task_list"``),
keyed by :class:`~spore_core.harness.SessionId`. A standalone tool call and a
PlanExecute run on the same session intentionally share one blob. The JSON
shape is the canonical serialized :class:`~spore_core.tasklist.TaskList`
(``{"tasks":[...],"next_id":N}``), unchanged.

Behavior change vs the retired sandbox path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously the tool persisted to ``.spore/task_list.json`` via the sandbox.
That path is GONE. With the library's default storage (the no-op provider) a
standalone tool call persists NOTHING across processes — the no-op run store
silently discards writes and returns ``None`` on read. This is an accepted
behavior change: durable cross-process persistence now requires configuring a
real ``StorageProvider``. There is NO migration shim for old on-disk
``.spore/task_list.json`` files.

Storage-error mapping
~~~~~~~~~~~~~~~~~~~~~~
A :class:`~spore_core.storage.StorageError` from a get/put maps to a recoverable
:class:`~spore_core.harness.ToolOutputError`. A present-but-malformed blob
(parse failure) is likewise recoverable. ``list_tasks`` never writes.

CRITICAL: this tool is NOT annotated ``read_only``. ``read_only`` tools are run
CONCURRENTLY by ``dispatch_all``, and a concurrent read-modify-write over the
same key would race. Leaving ``read_only`` false makes the registry dispatch it
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
from spore_core.storage import StorageError
from spore_core.tasklist import (
    TASK_LIST_EXTRAS_KEY,
    TaskList,
    TaskListError,
)
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .error import ToolExecutionError
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
                    "blockers": {"type": "array", "items": {"type": "integer"}},
                    "description": {"type": "string"},
                    "id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["blocked", "completed", "in_progress", "pending"],
                    },
                },
                "required": ["action"],
            },
            # Intentionally NOT read_only: this tool mutates shared persisted
            # state and must dispatch sequentially. See module docs.
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        session_id = ctx.session_id
        run_store = ctx.run_store

        # 1. Parse params (bad input → recoverable).
        try:
            action, params = parse_task_list_params(call)
        except ToolExecutionError as e:
            return e.to_tool_output()

        # 2. Load the current list from the run store (absent → default). A
        #    storage error or a malformed blob is recoverable.
        try:
            value = await run_store.get(session_id, TASK_LIST_EXTRAS_KEY)
        except StorageError as e:
            return ToolOutputError(message=f"could not load task list: {e}", recoverable=True)
        if value is None:
            task_list = TaskList()
        else:
            try:
                task_list = TaskList.from_dict(value)
            except (ValueError, KeyError, TypeError) as e:
                return ToolOutputError(message=f"could not parse task list: {e}", recoverable=True)

        # 3. Apply the action. Domain errors → recoverable. `list_tasks` does not
        #    mutate.
        mutated = False
        try:
            if isinstance(params, AddTaskParams):
                task_list.add(params.description, params.blockers)
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

        # 4. Persist the (possibly mutated) list to the run store, keyed by
        #    SessionId under the shared TASK_LIST_EXTRAS_KEY. We always persist
        #    on a mutating action; list_tasks skips the write.
        if mutated:
            try:
                await run_store.put(session_id, TASK_LIST_EXTRAS_KEY, task_list.to_dict())
            except StorageError as e:
                return ToolOutputError(
                    message=f"could not persist task list: {e}", recoverable=True
                )

        # 5. Return the serialized current list.
        return ToolOutputSuccess(content=task_list.to_json(), truncated=False)


__all__ = ["TaskListTool"]
