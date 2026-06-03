"""``remember(key, value)`` — persist a fact into the run store.

This is the **write** half of the custom-tool pair, defined with the
:func:`~spore_tools.define_tool` helper: a typed pydantic input model plus an
async ``execute`` body, and the helper derives the advertised JSON schema from
the model (via ``model_json_schema()``) so the schema and the validation can
never drift.

It demonstrates the storage seam: :attr:`ToolContext.run_store` +
:attr:`ToolContext.session_id` are the only path to durable, per-run state. The
``sandbox`` parameter is part of the ``execute`` signature but unused here —
these tools never touch the filesystem, so they ignore it.

Keys are namespaced under ``fact:{key}`` so the example cannot collide with
reserved store keys the catalogue uses (``todo``, ``task``, ``memory``).
"""

from __future__ import annotations

from pydantic import BaseModel

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.storage import StorageError
from spore_core.tool_registry import ToolContext
from spore_tools import StandardTool, define_tool

#: Prefix applied to every key so this example's facts live in their own
#: namespace inside the run store.
FACT_PREFIX = "fact:"

#: Tool name — also used by tests and ``recall`` cross-checks.
NAME = "remember"


class RememberInput(BaseModel):
    """Validated input for ``remember``. ``define_tool`` derives the advertised
    JSON schema from exactly this model."""

    key: str
    value: str


async def _remember(input: RememberInput, sandbox: SandboxProvider, ctx: ToolContext) -> ToolOutput:
    store_key = f"{FACT_PREFIX}{input.key}"
    try:
        await ctx.run_store.put(ctx.session_id, store_key, input.value)
    except StorageError as e:
        return ToolOutputError.error(f"remember: could not persist '{input.key}': {e}")
    return ToolOutputSuccess.success(f"remembered {input.key}")


def remember_tool() -> StandardTool:
    """Build the ``remember`` tool. :func:`~spore_tools.define_tool` generates the
    ``Tool`` impl, derives the schema from :class:`RememberInput`, and bundles
    them into a :class:`~spore_tools.StandardTool` ready for ``.tool(...)``.

    Annotations are omitted, so they default to all-``False`` — ``remember``
    MUTATES shared persisted state, so (unlike ``recall``) it is intentionally
    not ``read_only``."""
    return define_tool(
        name=NAME,
        description=(
            "Store a fact under a short key so it can be recalled later. "
            "Use a stable, memorable key (e.g. 'habitat', 'lifespan')."
        ),
        input_model=RememberInput,
        execute=_remember,
    )


__all__ = ["FACT_PREFIX", "NAME", "RememberInput", "remember_tool"]
