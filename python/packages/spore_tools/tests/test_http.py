"""Tests for HTTP tools — uses :mod:`pytest_httpx` to mock the network."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.http import HttpGetTool, HttpPostTool

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_http_get_returns_body(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url="https://example.test/hello", text="world")
    sb = AllowAllSandbox()
    r = await HttpGetTool().execute(
        _call("http_get", {"url": "https://example.test/hello"}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "world"


async def test_http_post_sends_json_body(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url="https://example.test/echo", text="ok")
    sb = AllowAllSandbox()
    r = await HttpPostTool().execute(
        _call("http_post", {"url": "https://example.test/echo", "body": {"x": 1}}),
        sb,
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "ok"


async def test_http_get_invalid_url_is_recoverable() -> None:
    sb = AllowAllSandbox()
    r = await HttpGetTool().execute(_call("http_get", {"url": "not-a-url://////"}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


@pytest.mark.parametrize("tool", [HttpGetTool, HttpPostTool])
async def test_http_invalid_params_recoverable(tool: type) -> None:
    sb = AllowAllSandbox()
    r = await tool().execute(_call(tool.NAME, {}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
