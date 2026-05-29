"""SendMessage tool (#81, net-new Tier-1 tool).

Mirrors ``rust/crates/spore-core/src/tools/message.rs``.

``send_message`` surfaces an out-of-band message to the user. The TOOL itself
is trivial: it echoes the ``content`` back as a :class:`ToolOutputSuccess`. The
harness loop is what gives it meaning — it recognizes the ``send_message`` tool
name (:data:`~spore_core.harness.SEND_MESSAGE_TOOL_NAME`), emits a
:class:`~spore_core.harness.StreamUserMessage` with the content, and records a
minimal success tool result so the loop continues. The tool does NOT touch the
sandbox or storage.
"""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .error import ToolExecutionError
from .params import SendMessageParams, parse_params


class SendMessageTool:
    NAME = "send_message"

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
            description="Send a message to the user",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(SendMessageParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        # The content is returned verbatim; the harness loop reads it off this
        # Success and emits a StreamUserMessage event.
        return ToolOutputSuccess(content=params.content, truncated=False)


__all__ = ["SendMessageTool"]
