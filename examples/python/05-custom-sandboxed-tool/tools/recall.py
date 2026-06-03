"""``recall(key)`` — read a previously-remembered fact back out of the run store.

This is the **read** half of the pair. Unlike :mod:`remember` it is annotated
``read_only`` + ``idempotent``: it only reads shared state, so the harness may
dispatch it concurrently with other read-only tools.

Looking up a key that was never stored is a *recoverable* error — the agent can
adapt (try a different key, or remember the fact first) rather than halting the
run.
"""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.storage import JsonValue, StorageError
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .remember import FACT_PREFIX


class RecallTool:
    """Return the string stored under ``fact:{key}`` by :class:`RememberTool`."""

    NAME = "recall"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        """The registry-side schema. ``name`` MUST equal :meth:`name`."""
        return ToolSchema(
            name=cls.NAME,
            description="Recall a fact previously stored with `remember`, by its key.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
            # Pure read of shared state: safe to mark read_only + idempotent.
            annotations=ToolAnnotations(read_only=True, idempotent=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        key = call.input.get("key")
        if not isinstance(key, str):
            return ToolOutputError.error("recall: missing or non-string 'key'")

        store_key = f"{FACT_PREFIX}{key}"
        try:
            value = await ctx.run_store.get(ctx.session_id, store_key)
        except StorageError as e:
            return ToolOutputError.error(f"recall: could not read '{key}': {e}")
        if value is None:
            return ToolOutputError.error(f"no fact stored under '{key}'")
        return ToolOutputSuccess.success(_value_to_string(value))


def _value_to_string(value: JsonValue) -> str:
    """``remember`` always stores a JSON string, so render that back as plain
    text. Fall back to ``str`` for anything unexpected."""
    if isinstance(value, str):
        return value
    return str(value)


__all__ = ["RecallTool"]
