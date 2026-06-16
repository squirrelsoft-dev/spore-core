"""Tests for the net-new web tools (#81) — mocked HTTP via :mod:`pytest_httpx`.
NEVER hits the live network."""

from __future__ import annotations

import json
import pathlib

import pytest
from pytest_httpx import HTTPXMock

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.web import (
    EnvVarEmpty,
    EnvVarNotSet,
    SearchMethod,
    UrlPolicy,
    WebFetchTool,
    WebSearchConfig,
    WebSearchTool,
    _apply_web_fetch_range,
    validate_fetch_url,
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


# ── SSRF guard: validate_fetch_url (#145) ─────────────────────────────────────

_DENY_URLS = [
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost/",
    "file:///etc/passwd",
    "http://127.0.0.1/",
    "http://10.1.2.3/",
    "http://192.168.0.1/",
    "http://172.16.5.5/",
    "http://[::1]/",
    "ftp://example.com/x",
]

_ALLOW_URLS = [
    "https://example.com/",
    "http://93.184.216.34/",
]


@pytest.mark.parametrize("url", _DENY_URLS + _ALLOW_URLS)
def test_permissive_policy_allows_everything(url: str) -> None:
    assert validate_fetch_url(url, UrlPolicy.permissive()) is None


@pytest.mark.parametrize("url", _DENY_URLS + _ALLOW_URLS)
def test_default_policy_is_permissive(url: str) -> None:
    # The dataclass default and ``permissive()`` are equivalent — both allow all.
    assert UrlPolicy() == UrlPolicy.permissive()
    assert validate_fetch_url(url, UrlPolicy()) is None


@pytest.mark.parametrize("url", _DENY_URLS)
def test_deny_private_rejects_blocked_urls(url: str) -> None:
    result = validate_fetch_url(url, UrlPolicy.deny_private())
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True


@pytest.mark.parametrize("url", _ALLOW_URLS)
def test_deny_private_allows_public_hosts(url: str) -> None:
    assert validate_fetch_url(url, UrlPolicy.deny_private()) is None


def test_deny_private_rejects_ipv4_mapped_metadata() -> None:
    # ``::ffff:169.254.169.254`` must unwrap to the v4 and be blocked.
    result = validate_fetch_url("http://[::ffff:169.254.169.254]/", UrlPolicy.deny_private())
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True


def test_deny_private_rejects_broadcast_and_unspecified() -> None:
    for url in ("http://255.255.255.255/", "http://0.0.0.0/"):
        result = validate_fetch_url(url, UrlPolicy.deny_private())
        assert isinstance(result, ToolOutputError), url


async def test_web_fetch_deny_private_blocks_metadata_endpoint(
    httpx_mock: HTTPXMock,
) -> None:
    tool = WebFetchTool().with_url_policy(UrlPolicy.deny_private())
    r = await tool.execute(
        _call("web_fetch", {"url": "http://169.254.169.254/"}), AllowAllSandbox(), _CTX
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    # The outbound request was never issued.
    assert httpx_mock.get_requests() == []


async def test_web_search_deny_private_blocks_localhost_endpoint() -> None:
    tool = WebSearchTool.with_endpoint("http://localhost:8080/search").with_url_policy(
        UrlPolicy.deny_private()
    )
    r = await tool.execute(_call("web_search", {"query": "x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


# ── _apply_web_fetch_range unit tests (#135) ──────────────────────────────────


def test_range_start_zero_no_header() -> None:
    result = _apply_web_fetch_range("hello world", 0)
    assert result == "hello world"


def test_range_start_zero_empty_body_no_header() -> None:
    result = _apply_web_fetch_range("", 0)
    assert result == ""


def test_range_start_mid_prepends_header() -> None:
    result = _apply_web_fetch_range("hello world", 6)
    assert result == "[starting at byte 6 of 11]\nworld"


def test_range_start_at_last_byte() -> None:
    result = _apply_web_fetch_range("hello", 4)
    assert result == "[starting at byte 4 of 5]\no"


def test_range_start_past_end_is_recoverable_error() -> None:
    result = _apply_web_fetch_range("hello", 10)
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True
    assert result.message == "start_byte 10 exceeds response length 5"


def test_range_start_at_body_len_is_recoverable_error() -> None:
    result = _apply_web_fetch_range("hello", 5)
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True
    assert result.message == "start_byte 5 exceeds response length 5"


def test_range_empty_body_nonzero_start_is_recoverable_error() -> None:
    result = _apply_web_fetch_range("", 1)
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True
    assert result.message == "start_byte 1 exceeds response length 0"


# ── fixture replay: web_fetch_range.json (#135) ───────────────────────────────

_FIXTURE_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent
    / "fixtures"
    / "tools"
    / "web_fetch_range.json"
)


def test_fixture_replay_web_fetch_range() -> None:
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"fixture not found: {_FIXTURE_PATH}")
    cases = json.loads(_FIXTURE_PATH.read_text())
    assert len(cases) >= 1, "expected ≥1 case in fixture"
    for case in cases:
        name = case["name"]
        body = case["body"]
        start_byte = case["start_byte"]
        result = _apply_web_fetch_range(body, start_byte)
        if "expected" in case:
            assert not isinstance(result, ToolOutputError), (
                f"case {name!r}: expected success but got error: {result.message}"
            )
            assert result == case["expected"], f"case {name!r}: output mismatch"
        elif "expected_error" in case:
            assert isinstance(result, ToolOutputError), (
                f"case {name!r}: expected error but got: {result!r}"
            )
            assert result.message == case["expected_error"], (
                f"case {name!r}: error message mismatch"
            )
        else:
            pytest.fail(f"case {name!r}: fixture row has neither 'expected' nor 'expected_error'")


# ── integration: web_fetch with start_byte via mock server (#135) ─────────────


async def test_web_fetch_start_byte_zero_no_header(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url="https://example.test/page", text="hello world")
    r = await WebFetchTool().execute(
        _call("web_fetch", {"url": "https://example.test/page", "start_byte": 0}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "hello world"


async def test_web_fetch_start_byte_mid_slices_with_header(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(method="GET", url="https://example.test/page", text="hello world")
    r = await WebFetchTool().execute(
        _call("web_fetch", {"url": "https://example.test/page", "start_byte": 6}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "[starting at byte 6 of 11]\nworld"


async def test_web_fetch_start_byte_past_end_is_recoverable_error(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(method="GET", url="https://example.test/page", text="hello")
    r = await WebFetchTool().execute(
        _call("web_fetch", {"url": "https://example.test/page", "start_byte": 99}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "start_byte 99 exceeds response length 5" in r.message
