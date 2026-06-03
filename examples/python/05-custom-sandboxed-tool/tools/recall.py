"""``recall(key)`` — read a previously-remembered fact back out of the run store.

This is the **read** half of the pair, also defined with the
:func:`~spore_tools.define_tool` helper. Unlike :mod:`remember` it passes
``annotations`` to mark itself ``read_only`` + ``idempotent``: it only reads
shared state, so the harness may dispatch it concurrently with other read-only
tools.

Looking up a key that was never stored is a *recoverable* error — the agent can
adapt (try a different key, or remember the fact first) rather than halting the
run.
"""

from __future__ import annotations

from pydantic import BaseModel

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.storage import JsonValue, StorageError
from spore_core.tool_registry import ToolAnnotations, ToolContext
from spore_tools import StandardTool, define_tool

from .remember import FACT_PREFIX

#: Tool name.
NAME = "recall"


class RecallInput(BaseModel):
    """Validated input for ``recall``."""

    key: str


async def _recall(input: RecallInput, sandbox: SandboxProvider, ctx: ToolContext) -> ToolOutput:
    store_key = f"{FACT_PREFIX}{input.key}"
    try:
        value = await ctx.run_store.get(ctx.session_id, store_key)
    except StorageError as e:
        return ToolOutputError.error(f"recall: could not read '{input.key}': {e}")
    if value is None:
        return ToolOutputError.error(f"no fact stored under '{input.key}'")
    return ToolOutputSuccess.success(_value_to_string(value))


def recall_tool() -> StandardTool:
    """Build the ``recall`` tool. ``annotations`` marks it ``read_only`` +
    ``idempotent`` (a pure read of shared state), in contrast to ``remember``."""
    return define_tool(
        name=NAME,
        description="Recall a fact previously stored with `remember`, by its key.",
        input_model=RecallInput,
        execute=_recall,
        annotations=ToolAnnotations(read_only=True, idempotent=True),
    )


def _value_to_string(value: JsonValue) -> str:
    """``remember`` always stores a JSON string, so render that back as plain
    text. Fall back to ``str`` for anything unexpected."""
    if isinstance(value, str):
        return value
    return str(value)


__all__ = ["NAME", "RecallInput", "recall_tool"]
