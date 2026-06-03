"""Tests for the net-new web tools (#81) — mocked HTTP via :mod:`pytest_httpx`.
NEVER hits the live network."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.web import (
    EnvVarEmpty,
    EnvVarNotSet,
    SearchMethod,
    WebFetchTool,
    WebSearchConfig,
    WebSearchTool,
)

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


# ── #108: GET / auth headers / in-body auth / env resolution ──────────────────


def test_search_method_default_is_post() -> None:
    assert WebSearchConfig(endpoint="http://x/").method is SearchMethod.POST


async def test_get_method_url_encodes_query_into_query_string(httpx_mock: HTTPXMock) -> None:
    # Brave-style: GET with the query under ``q``, special chars encoded.
    httpx_mock.add_response(
        method="GET",
        url="https://search.test/search?q=rust+%26+go",
        text="get-results",
    )
    cfg = WebSearchConfig(
        endpoint="https://search.test/search",
        method=SearchMethod.GET,
        query_param="q",
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "rust & go"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "get-results"


async def test_auth_header_attached_on_get(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAVE_API_KEY_GET", "brave-secret")
    httpx_mock.add_response(
        method="GET",
        url="https://search.test/search?q=x",
        match_headers={"X-Subscription-Token": "brave-secret"},
        text="ok",
    )
    cfg = WebSearchConfig(
        endpoint="https://search.test/search",
        method=SearchMethod.GET,
        auth_headers=[("X-Subscription-Token", "BRAVE_API_KEY_GET")],
        query_param="q",
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)


async def test_auth_header_attached_on_post(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BRAVE_API_KEY_POST", "brave-secret")
    httpx_mock.add_response(
        method="POST",
        url="https://search.test/search",
        match_headers={"X-Subscription-Token": "brave-secret"},
        text="ok",
    )
    cfg = WebSearchConfig(
        endpoint="https://search.test/search",
        method=SearchMethod.POST,
        auth_headers=[("X-Subscription-Token", "BRAVE_API_KEY_POST")],
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)


async def test_multiple_auth_headers_all_attached(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MULTI_A", "aaa")
    monkeypatch.setenv("MULTI_B", "bbb")
    httpx_mock.add_response(
        method="POST",
        url="https://search.test/search",
        match_headers={"X-A": "aaa", "X-B": "bbb"},
        text="ok",
    )
    cfg = WebSearchConfig(
        endpoint="https://search.test/search",
        method=SearchMethod.POST,
        auth_headers=[("X-A", "MULTI_A"), ("X-B", "MULTI_B")],
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)


async def test_in_body_auth_param_tavily_shape(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tavily POST body: {"api_key": <secret>, "query": <q>}.
    monkeypatch.setenv("TAVILY_API_KEY", "tav-secret")
    httpx_mock.add_response(
        method="POST",
        url="https://search.test/search",
        match_json={"api_key": "tav-secret", "query": "rust"},
        text="tavily-results",
    )
    cfg = WebSearchConfig(
        endpoint="https://search.test/search",
        method=SearchMethod.POST,
        body_auth_params=[("api_key", "TAVILY_API_KEY")],
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "rust"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "tavily-results"


def test_missing_env_var_is_construction_error_no_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WEB_MISSING_KEY", raising=False)
    # No mock server registered: if a request were attempted, pytest-httpx would
    # error; we assert construction raises first, so no request is ever made.
    cfg = WebSearchConfig(
        endpoint="http://127.0.0.1:1/never",
        method=SearchMethod.POST,
        auth_headers=[("X-Key", "WEB_MISSING_KEY")],
    )
    with pytest.raises(EnvVarNotSet) as exc:
        WebSearchTool.with_config(cfg)
    assert exc.value.env_var == "WEB_MISSING_KEY"


def test_empty_env_var_is_construction_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_EMPTY_KEY", "   ")
    cfg = WebSearchConfig(
        endpoint="http://127.0.0.1:1/never",
        method=SearchMethod.POST,
        body_auth_params=[("api_key", "WEB_EMPTY_KEY")],
    )
    with pytest.raises(EnvVarEmpty) as exc:
        WebSearchTool.with_config(cfg)
    assert exc.value.env_var == "WEB_EMPTY_KEY"


async def test_no_auth_post_carries_only_content_type(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="POST", url="https://search.test/search", text="ok")
    cfg = WebSearchConfig(endpoint="https://search.test/search", method=SearchMethod.POST)
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    # No auth headers were configured; only httpx's own headers (Content-Type) sent.
    req = httpx_mock.get_request()
    assert req is not None
    assert req.headers.get("content-type") == "application/json"
    assert "x-subscription-token" not in req.headers
    assert "authorization" not in req.headers


async def test_post_default_query_shape_unchanged_via_config(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://search.test/search",
        match_json={"query": "rust"},
        text="res",
    )
    tool = WebSearchTool.with_config(WebSearchConfig(endpoint="https://search.test/search"))
    r = await tool.execute(_call("web_search", {"query": "rust"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "res"


async def test_get_preserves_existing_endpoint_query_string(httpx_mock: HTTPXMock) -> None:
    # SearXNG shape: the endpoint already carries ``?format=json`` and the query
    # is appended under ``q``. The outbound request must carry BOTH params —
    # httpx's ``params=`` would otherwise replace the pre-existing query string.
    httpx_mock.add_response(method="GET", text="searxng-results")
    cfg = WebSearchConfig(
        endpoint="http://testserver/search?format=json",
        method=SearchMethod.GET,
        query_param="q",
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "rust"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "searxng-results"
    req = httpx_mock.get_request()
    assert req is not None
    assert req.url.params.get("format") == "json"
    assert req.url.params.get("q") == "rust"


async def test_get_returns_body_verbatim(httpx_mock: HTTPXMock) -> None:
    raw = '{"web":{"results":[{"title":"t"}]}}'
    httpx_mock.add_response(
        method="GET",
        url="https://search.test/search?q=t",
        text=raw,
    )
    cfg = WebSearchConfig(
        endpoint="https://search.test/search",
        method=SearchMethod.GET,
        query_param="q",
    )
    tool = WebSearchTool.with_config(cfg)
    r = await tool.execute(_call("web_search", {"query": "t"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == raw
