"""``send_user_message(message)`` — let an agent narrate its plan to the human.

Pure side-effecting tool: it prints the message to stdout (prefixed with a
per-agent marker emoji so you can tell who is speaking — 🤖 for the worker) and
returns a short confirmation so the model keeps going. It closes over the
marker, so it is built with a small closure handed to :func:`define_tool`.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, Field

from spore_core.harness import SandboxProvider, ToolOutput, ToolOutputError, ToolOutputSuccess
from spore_core.tool_registry import ToolContext
from spore_tools import StandardTool, define_tool

#: The registered name of the tool.
NAME = "send_user_message"


class SendMessageInput(BaseModel):
    """One short sentence telling the watching human what you are about to do."""

    message: str = Field(
        description="What you are about to do and why, in one short sentence.",
    )


def send_user_message_tool(marker: str) -> StandardTool:
    """Build a ``send_user_message`` :class:`StandardTool` that prefixes its
    output with ``marker`` (e.g. "🤖" for the worker)."""

    async def _execute(
        input: SendMessageInput, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        _ = (sandbox, ctx)
        message = input.message.strip()
        if not message:
            return ToolOutputError(
                message="invalid parameters: `message` (non-empty string) is required",
                recoverable=True,
            )
        # Leading newline to break from the stream banners, the marker, then a
        # trailing blank line to give the message room.
        print(f"\n{marker} {message}\n", flush=True, file=sys.stdout)
        return ToolOutputSuccess(content="Message shown to the user.", truncated=False)

    return define_tool(
        name=NAME,
        description=(
            "Tell the watching human what you are about to do and why, in one short sentence, "
            "BEFORE you act. Call this at the start of each step so your plan is visible. Pass a "
            "single `message` string. This does not pause the run."
        ),
        input_model=SendMessageInput,
        execute=_execute,
    )


__all__ = ["NAME", "SendMessageInput", "send_user_message_tool"]
