"""Tests for :mod:`spore_core.openai` — issue #40.

Mirrors the Rust reference at ``rust/crates/spore-core/src/openai.rs``
test module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from spore_core.cache_provider import OpenAICacheProvider
from spore_core.model import (
    Message,
    MessageStart,
    MessageStop,
    ModelParams,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ProviderInfo,
    RateLimited,
    ReplayMode,
    ReplayModelInterface,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    ThinkingBlock,
    TimeoutError as ModelTimeoutError,
    TokenUsage,
    ToolCallContent,
    ToolResultContent,
    ToolSchema,
    ToolUseBlock,
    ToolUseDelta,
)
from spore_core.openai import (
    OpenAIModelInterface,
    _backoff_delay,
    build_request_body,
    context_window,
    is_reasoning_model,
    parse_response_body,
    parse_stop_reason,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=TextContent(text=text))


def _sys(text: str) -> Message:
    return Message(role=Role.SYSTEM, content=TextContent(text=text))


def _req(messages: list[Message], **kwargs: Any) -> ModelRequest:
    return ModelRequest(messages=messages, **kwargs)


def _mock_client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


# ---------------------------------------------------------------------------
# build_request_body
# ---------------------------------------------------------------------------


def test_build_request_keeps_system_in_messages() -> None:
    body = build_request_body("gpt-4o", _req([_sys("be helpful"), _user("hi")]), stream=False)
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "be helpful"
    assert body["messages"][1]["role"] == "user"


def test_build_request_sets_max_tokens_for_chat_models() -> None:
    body = build_request_body(
        "gpt-4o",
        _req([_user("hi")], params=ModelParams(max_tokens=256)),
        stream=False,
    )
    assert body["max_tokens"] == 256
    assert "max_completion_tokens" not in body


def test_build_request_o_series_uses_max_completion_tokens_and_no_temperature() -> None:
    body = build_request_body(
        "o3",
        _req([_user("hi")], params=ModelParams(max_tokens=512, temperature=0.7)),
        stream=False,
    )
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 512
    assert "temperature" not in body


def test_is_reasoning_model_detection() -> None:
    assert is_reasoning_model("o4-mini")
    assert is_reasoning_model("o3")
    assert is_reasoning_model("o1-pro")
    assert not is_reasoning_model("gpt-4o")


def test_build_request_maps_tool_call_to_assistant_tool_calls() -> None:
    body = build_request_body(
        "gpt-4o",
        _req(
            [
                Message(
                    role=Role.ASSISTANT,
                    content=ToolCallContent(id="call-1", name="fetch", input={"url": "x"}),
                )
            ]
        ),
        stream=False,
    )
    msg = body["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "call-1"
    assert msg["tool_calls"][0]["type"] == "function"
    assert msg["tool_calls"][0]["function"]["name"] == "fetch"
    # arguments must be a JSON-encoded STRING, not a nested object.
    args = msg["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"url": "x"}


def test_build_request_maps_tool_result_to_tool_role_message() -> None:
    body = build_request_body(
        "gpt-4o",
        _req(
            [
                Message(
                    role=Role.TOOL,
                    content=ToolResultContent(tool_use_id="call-1", content="ok"),
                )
            ]
        ),
        stream=False,
    )
    msg = body["messages"][0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-1"
    assert msg["content"] == "ok"


def test_build_request_streaming_sets_include_usage() -> None:
    body = build_request_body("gpt-4o", _req([_user("hi")]), stream=True)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_build_request_emits_tools_when_present() -> None:
    req = _req(
        [_user("hi")],
        tools=[
            ToolSchema(name="search", description="search the web", input_schema={"type": "object"})
        ],
    )
    body = build_request_body("gpt-4o", req, stream=False)
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "search"
    assert body["tools"][0]["function"]["description"] == "search the web"


# ---------------------------------------------------------------------------
# parse_response_body
# ---------------------------------------------------------------------------


def test_parse_response_extracts_text_and_usage() -> None:
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    r = parse_response_body(body)
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "hi there"
    assert r.usage.input_tokens == 4
    assert r.usage.output_tokens == 2
    assert r.stop_reason is StopReason.END_TURN


def test_parse_response_extracts_tool_calls() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"rust"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    r = parse_response_body(body)
    assert r.stop_reason is StopReason.TOOL_USE
    assert isinstance(r.content[0], ToolUseBlock)
    assert r.content[0].id == "c1"
    assert r.content[0].name == "search"
    assert r.content[0].input == {"q": "rust"}


def test_parse_response_extracts_reasoning_as_thinking() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning": "let me think...",
                    "content": "the answer is 4",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    r = parse_response_body(body)
    assert isinstance(r.content[0], ThinkingBlock)
    assert isinstance(r.content[1], TextBlock)


def test_parse_response_extracts_cache_read_only() -> None:
    body = {
        "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 50},
        },
    }
    r = parse_response_body(body)
    assert r.usage.cache_read_tokens == 50
    assert r.usage.cache_write_tokens is None


def test_stop_reason_mapping() -> None:
    assert parse_stop_reason("stop") is StopReason.END_TURN
    assert parse_stop_reason("tool_calls") is StopReason.TOOL_USE
    assert parse_stop_reason("function_call") is StopReason.TOOL_USE
    assert parse_stop_reason("length") is StopReason.MAX_TOKENS
    assert parse_stop_reason(None) is StopReason.END_TURN
    assert parse_stop_reason("???") is StopReason.END_TURN


# ---------------------------------------------------------------------------
# context_window / backoff / provider() / from_env / repr
# ---------------------------------------------------------------------------


def test_context_window_known_and_unknown() -> None:
    assert context_window("gpt-4o") == 128_000
    assert context_window("gpt-4o-mini") == 128_000
    assert context_window("gpt-4.1") == 1_000_000
    assert context_window("o3") == 200_000
    assert context_window("o4-mini") == 200_000
    assert context_window("o1-pro") == 128_000
    assert context_window("claude-x") == 0


def test_backoff_grows_then_caps() -> None:
    d0 = _backoff_delay(0)
    d3 = _backoff_delay(3)
    dmax = _backoff_delay(20)
    assert d3 > d0
    assert dmax <= 30.0
    assert d0 == pytest.approx(0.5)


def test_provider_info_uses_model_id() -> None:
    c = OpenAIModelInterface("test-key", "gpt-4o")
    p = c.provider()
    assert p.name == "openai"
    assert p.model_id == "gpt-4o"
    assert p.context_window == 128_000


def test_repr_redacts_api_key() -> None:
    c = OpenAIModelInterface("super-secret-key", "gpt-4o")
    r = repr(c)
    assert "super-secret-key" not in r
    assert "<redacted>" in r


def test_from_env_errors_when_unset() -> None:
    var = "__SPORE_TEST_OPENAI_KEY_UNSET__"
    os.environ.pop(var, None)
    with pytest.raises(ProviderError) as exc:
        OpenAIModelInterface.from_env(var, "gpt-4o")
    assert "not set" in exc.value.message


def test_from_env_errors_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("__SPORE_TEST_OPENAI_KEY_EMPTY__", "   ")
    with pytest.raises(ProviderError) as exc:
        OpenAIModelInterface.from_env("__SPORE_TEST_OPENAI_KEY_EMPTY__", "gpt-4o")
    assert "empty" in exc.value.message


# ---------------------------------------------------------------------------
# End-to-end mocked HTTP — call()
# ---------------------------------------------------------------------------


async def test_call_against_mock_returns_response() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hello there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface(
        "test-key", "gpt-4o", base_url="https://example.test", http_client=client
    )
    r = await iface.call(_req([_user("hi")]))
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "hello there"
    assert r.usage.input_tokens == 5
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "gpt-4o"


async def test_call_maps_429_to_rate_limited() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, text="rate limited")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface(
        "k", "gpt-4o", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(RateLimited) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.retry_after == 7.0


async def test_call_maps_400_to_provider_error_with_openai_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": "max_tokens must be > 0",
                }
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface(
        "k", "gpt-4o", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(ProviderError) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.code == 400
    assert "max_tokens" in exc.value.message


async def test_call_retries_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="rate limited")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "after retry"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface("k", "gpt-4o", base_url="https://x.test", http_client=client)
    r = await iface.call(_req([_user("hi")]))
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "after retry"
    assert calls["n"] == 2


async def test_call_timeout_surfaces_as_model_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface(
        "k", "gpt-4o", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(ModelTimeoutError):
        await iface.call(_req([_user("hi")]))


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


async def test_count_tokens_uses_bytes_over_four_heuristic() -> None:
    iface = OpenAIModelInterface("k", "gpt-4o")
    n = await iface.count_tokens(_req([_user("a" * 40)]))
    assert n == 10


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


_SSE_TEXT = (
    'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{"content":" world"}}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":3,"completion_tokens":5}}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)


async def test_streaming_emits_text_delta_then_stop() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_SSE_TEXT.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface("k", "gpt-4o", base_url="https://x.test", http_client=client)
    events: list[Any] = []
    async for ev in iface.call_streaming(_req([_user("hi")])):
        events.append(ev)
    assert isinstance(events[0], MessageStart)
    text_deltas = [e.delta for e in events if hasattr(e, "delta") and isinstance(e.delta, str)]
    assert text_deltas == ["hello", " world"]
    last = events[-1]
    assert isinstance(last, MessageStop)
    assert last.usage.input_tokens == 3
    assert last.usage.output_tokens == 5
    assert last.stop_reason is StopReason.END_TURN


# Three partial chunks for the same tool call (index=0): the first
# carries id+name; subsequent chunks carry incremental arguments
# fragments. Consumer joins ToolUseDelta.partial_json to reconstruct
# the full JSON.
_SSE_TOOL = (
    'data: {"choices":[{"index":0,"delta":{"tool_calls":'
    '[{"index":0,"id":"call-1","function":{"name":"fetch","arguments":"{\\"u"}}]}}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{"tool_calls":'
    '[{"index":0,"function":{"arguments":"rl\\":\\"x\\""}}]}}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{"tool_calls":'
    '[{"index":0,"function":{"arguments":"}"}}]}}]}\n'
    "\n"
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],'
    '"usage":{"prompt_tokens":1,"completion_tokens":1}}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)


async def test_streaming_accumulates_tool_call_deltas() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_SSE_TOOL.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface("k", "gpt-4o", base_url="https://x.test", http_client=client)
    tool_fragments: list[str] = []
    final_stop: StopReason = StopReason.END_TURN
    async for ev in iface.call_streaming(_req([_user("hi")])):
        if isinstance(ev, ToolUseDelta):
            tool_fragments.append(ev.partial_json)
        elif isinstance(ev, MessageStop):
            final_stop = ev.stop_reason
    joined = "".join(tool_fragments)
    parsed = json.loads(joined)
    assert parsed == {"url": "x"}
    assert final_stop is StopReason.TOOL_USE


async def test_streaming_maps_status_error_eagerly() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "3"}, text="rate limited")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OpenAIModelInterface(
        "k", "gpt-4o", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(RateLimited) as exc:
        async for _ in iface.call_streaming(_req([_user("hi")])):
            pass
    assert exc.value.retry_after == 3.0


# ---------------------------------------------------------------------------
# OpenAICacheProvider — pricing extension
# ---------------------------------------------------------------------------


def _resp_with_cache(read: int | None) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text="x")],
        usage=TokenUsage(input_tokens=10, output_tokens=1, cache_read_tokens=read),
        stop_reason=StopReason.END_TURN,
    )


def test_openai_cache_default_pricing() -> None:
    p = OpenAICacheProvider()
    assert p.cache_read_usd_per_million == pytest.approx(1.25)
    s = p.parse_cache_stats(_resp_with_cache(1_000_000))
    assert s is not None
    assert s.cache_read_tokens == 1_000_000
    assert s.cache_write_tokens == 0
    assert s.cache_read_cost_usd == pytest.approx(1.25)


def test_openai_cache_with_model_pricing() -> None:
    cases = [
        ("gpt-4o-mini", 0.075),
        ("gpt-4o", 1.25),
        ("o4-mini", 0.275),
        ("o3", 2.50),
        ("o1", 7.50),
        ("unknown-model", 1.25),  # default
    ]
    for model_id, expected in cases:
        p = OpenAICacheProvider().with_model_pricing(model_id)
        s = p.parse_cache_stats(_resp_with_cache(1_000_000))
        assert s is not None
        assert s.cache_read_cost_usd == pytest.approx(expected), f"{model_id}: {s}"


def test_openai_cache_no_metadata_returns_none() -> None:
    p = OpenAICacheProvider()
    assert p.parse_cache_stats(_resp_with_cache(None)) is None


# ---------------------------------------------------------------------------
# Fixture replay (shared cross-language fixture)
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = (
    _REPO_ROOT / "fixtures" / "model_responses" / "model_interface" / "openai_basic_text.jsonl"
)


async def test_openai_basic_text_fixture_replays() -> None:
    """Round-trip the shared OpenAI fixture through ReplayModelInterface."""

    assert _FIXTURE.exists(), f"missing fixture: {_FIXTURE}"
    text = _FIXTURE.read_text(encoding="utf-8")
    provider = ProviderInfo(name="openai", model_id="gpt-4o", context_window=128_000)
    replay = ReplayModelInterface.from_jsonl(text, provider, mode=ReplayMode.POSITIONAL)
    # Build a request matching the recorded fixture and verify the
    # response round-trips byte-for-byte at the level of types.
    request = ModelRequest(
        messages=[_user("hello")],
        params=ModelParams(max_tokens=1024),
    )
    response = await replay.call(request)
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == "Hello! How can I help you today?"
    assert response.usage.input_tokens == 8
    assert response.usage.output_tokens == 11
    assert response.stop_reason is StopReason.END_TURN


# ---------------------------------------------------------------------------
# Live-API integration tests (skipped by default)
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; live-API test skipped",
)


@LIVE
async def test_openai_live_call_returns_response() -> None:
    iface = OpenAIModelInterface.from_env("OPENAI_API_KEY", "gpt-4o-mini")
    try:
        r = await iface.call(_req([_user("Reply with the word 'pong'.")]))
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0
    finally:
        await iface.aclose()


@LIVE
async def test_openai_live_streaming_emits_events() -> None:
    iface = OpenAIModelInterface.from_env("OPENAI_API_KEY", "gpt-4o-mini")
    saw_stop = False
    try:
        async for ev in iface.call_streaming(_req([_user("Reply with the word 'pong'.")])):
            if isinstance(ev, MessageStop):
                saw_stop = True
        assert saw_stop
    finally:
        await iface.aclose()
