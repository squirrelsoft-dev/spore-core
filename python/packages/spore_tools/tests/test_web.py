"""Tests for the net-new web tools (#81) — mocked HTTP via :mod:`pytest_httpx`.
NEVER hits the live network."""

from __future__ import annotations

from pytest_httpx import HTTPXMock

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.web import WebFetchTool, WebSearchTool

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_web_fetch_returns_body(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url="https://example.test/page", text="page-body")
    r = await WebFetchTool().execute(
        _call("web_fetch", {"url": "https://example.test/page"}), AllowAllSandbox(), _CTX
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "page-body"


async def test_web_fetch_bad_url_is_recoverable() -> None:
    r = await WebFetchTool().execute(
        _call("web_fetch", {"url": "not-a-url://////"}), AllowAllSandbox(), _CTX
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_web_fetch_bad_params_is_recoverable() -> None:
    r = await WebFetchTool().execute(_call("web_fetch", {}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_web_search_returns_structured_results_from_mock(httpx_mock: HTTPXMock) -> None:
    results = '{"results":[{"title":"t","url":"u"}]}'
    httpx_mock.add_response(method="POST", url="https://search.test/search", text=results)
    tool = WebSearchTool.with_endpoint("https://search.test/search")
    r = await tool.execute(_call("web_search", {"query": "python"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == results


async def test_web_search_without_backend_is_recoverable_error() -> None:
    r = await WebSearchTool().execute(_call("web_search", {"query": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_web_search_bad_params_is_recoverable() -> None:
    tool = WebSearchTool.with_endpoint("https://search.test/search")
    r = await tool.execute(_call("web_search", {}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


def test_web_search_with_endpoint_is_named_web_search() -> None:
    from spore_tools.tools.catalogue import StandardTools

    t = StandardTools.web_search_with_endpoint("http://localhost:9/search")
    assert t.implementation.name() == "web_search"
    assert t.schema.name == "web_search"
