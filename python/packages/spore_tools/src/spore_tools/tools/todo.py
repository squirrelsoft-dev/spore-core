"""TodoWrite tool (#81, net-new Tier-2 storage tool).

Mirrors ``rust/crates/spore-core/src/tools/todo.rs``.

``todo_write`` persists an agent-managed todo list via the :class:`ToolContext`'s
:class:`~spore_core.storage.RunStore` under the key :data:`TODO_STORE_KEY`
(``"todo"``), keyed by the run's :class:`~spore_core.harness.SessionId`. The
agent supplies the FULL desired list on every call; it REPLACES the persisted
list wholesale (no per-item diffing). The current list is returned as JSON
success content.

Like :class:`~spore_tools.tools.tasklist.TaskListTool` it is NOT annotated
``read_only``: it mutates shared persisted state and must dispatch sequentially
(a concurrent read-modify-write would race).
"""

from __future__ import annotations

import json

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.storage import StorageError
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .error import ToolExecutionError
from .params import TodoWriteParams, parse_params

#: RunStore key under which the todo list is persisted (issue #81, Q5).
TODO_STORE_KEY = "todo"


class TodoWriteTool:
    NAME = "todo_write"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Replace the persisted todo list with the supplied full list",
            parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["completed", "in_progress", "pending"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
            # Intentionally NOT read_only — mutates shared persisted state and
            # must dispatch sequentially. See module docs / TaskListTool.
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(TodoWriteParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()

        # Serialize the todos to plain JSON-compatible dicts for both the store
        # and the returned content.
        value = [item.model_dump(mode="json") for item in params.todos]
        try:
            await ctx.run_store.put(ctx.session_id, TODO_STORE_KEY, value)
        except StorageError as e:
            return ToolOutputError(message=f"could not persist todos: {e}", recoverable=True)
        return ToolOutputSuccess(content=json.dumps(value), truncated=False)


__all__ = ["TODO_STORE_KEY", "TodoWriteTool"]
