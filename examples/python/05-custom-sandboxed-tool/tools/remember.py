"""``remember(key, value)`` — persist a fact into the run store.

This is the **write** half of the custom-tool pair. It demonstrates the storage
seam: :attr:`ToolContext.run_store` + :attr:`ToolContext.session_id` are the
only path to durable, per-run state. The ``sandbox`` parameter is part of the
:meth:`Tool.execute` signature but unused here — these tools never touch the
filesystem, so they ignore it.

Keys are namespaced under ``fact:{key}`` so the example cannot collide with
reserved store keys the catalogue uses (``todo``, ``task``, ``memory``).
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
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

#: Prefix applied to every key so this example's facts live in their own
#: namespace inside the run store.
FACT_PREFIX = "fact:"


class RememberTool:
    """Store a string ``value`` under ``fact:{key}`` in the run store."""

    NAME = "remember"

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
            description=(
                "Store a fact under a short key so it can be recalled later. "
                "Use a stable, memorable key (e.g. 'habitat', 'lifespan')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
            # Intentionally NOT read_only: this mutates shared persisted state.
            annotations=ToolAnnotations(),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        key = call.input.get("key")
        if not isinstance(key, str):
            return ToolOutputError.error("remember: missing or non-string 'key'")
        value = call.input.get("value")
        if not isinstance(value, str):
            return ToolOutputError.error("remember: missing or non-string 'value'")

        store_key = f"{FACT_PREFIX}{key}"
        try:
            await ctx.run_store.put(ctx.session_id, store_key, value)
        except StorageError as e:
            return ToolOutputError.error(f"remember: could not persist '{key}': {e}")
        return ToolOutputSuccess.success(f"remembered {key}")


__all__ = ["FACT_PREFIX", "RememberTool"]
