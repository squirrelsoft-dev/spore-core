"""Internal helpers shared by the standard tool catalogue."""

from __future__ import annotations

from spore_core.harness import SandboxProvider, ToolOutput, ToolOutputSuccess

# Threshold (in characters) above which tool output is routed through
# ``SandboxProvider.handle_large_output`` instead of returned inline.
LARGE_OUTPUT_THRESHOLD: int = 64 * 1024

# Head/tail token budgets passed to ``handle_large_output`` by default.
DEFAULT_HEAD_TOKENS: int = 2000
DEFAULT_TAIL_TOKENS: int = 2000


async def finish_with_possible_truncation(
    content: str,
    call_id: str,
    sandbox: SandboxProvider,
) -> ToolOutput:
    """Return :class:`ToolOutputSuccess`, routing through the sandbox if
    ``content`` exceeds :data:`LARGE_OUTPUT_THRESHOLD`."""

    if len(content) > LARGE_OUTPUT_THRESHOLD:
        truncated = await sandbox.handle_large_output(
            content, call_id, DEFAULT_HEAD_TOKENS, DEFAULT_TAIL_TOKENS
        )
        return ToolOutputSuccess(content=truncated.summary, truncated=True)
    return ToolOutputSuccess(content=content, truncated=False)


__all__ = [
    "DEFAULT_HEAD_TOKENS",
    "DEFAULT_TAIL_TOKENS",
    "LARGE_OUTPUT_THRESHOLD",
    "finish_with_possible_truncation",
]
