"""Tests for the net-new ``send_message`` tool (#81)."""

from __future__ import annotations

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.message import SendMessageTool

_CTX = make_test_ctx()


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=SendMessageTool.NAME, input=input_)


async def test_send_message_echoes_content() -> None:
    r = await SendMessageTool().execute(_call({"content": "hi user"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "hi user"


async def test_send_message_empty_content() -> None:
    r = await SendMessageTool().execute(_call({"content": ""}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == ""


async def test_send_message_bad_params_is_recoverable_error() -> None:
    r = await SendMessageTool().execute(_call({}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


def test_schema_read_only() -> None:
    assert SendMessageTool.schema().annotations.read_only is True
