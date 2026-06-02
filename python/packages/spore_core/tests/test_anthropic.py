"""Tests for :mod:`spore_core.anthropic` — issue #39.

Mirrors the Rust reference at ``rust/crates/spore-core/src/anthropic.rs``
test module.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from spore_core.anthropic import (
    ANTHROPIC_VERSION,
    AnthropicModelInterface,
    _backoff_delay,
    _parse_sse_event,
    build_request_body,
    context_window,
    parse_response_body,
    parse_stop_reason,
)
from spore_core.model import (
    ContentBlock,
    Message,
    MessageStart,
    MessageStop,
    ModelParams,
    ModelRequest,
    ProviderError,
    RateLimited,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    ThinkingBlock,
    TimeoutError as ModelTimeoutError,
    ToolCallContent,
    ToolResultContent,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
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


def test_build_request_extracts_system_message() -> None:
    body = build_request_body(
        "claude-sonnet-4-6",
        _req([_sys("be helpful"), _user("hi")]),
        stream=False,
    )
    assert body["system"] == "be helpful"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"


def test_build_request_joins_multiple_system_messages() -> None:
    body = build_request_body(
        "claude-sonnet-4-6",
        _req([_sys("first"), _sys("second"), _user("hi")]),
        stream=False,
    )
    assert body["system"] == "first\n\nsecond"


def test_build_request_defaults_max_tokens_when_unset() -> None:
    body = build_request_body("claude-sonnet-4-6", _req([_user("hi")]), stream=False)
    assert body["max_tokens"] == 4096


def test_build_request_respects_max_tokens() -> None:
    body = build_request_body(
        "claude-sonnet-4-6",
        _req([_user("hi")], params=ModelParams(max_tokens=256)),
        stream=False,
    )
    assert body["max_tokens"] == 256


def test_build_request_maps_tool_call_message() -> None:
    body = build_request_body(
        "claude-sonnet-4-6",
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
    s = json.dumps(body)
    assert '"type": "tool_use"' in s
    assert '"id": "call-1"' in s


def test_build_request_maps_tool_result_to_user_role() -> None:
    body = build_request_body(
        "claude-sonnet-4-6",
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
    assert body["messages"][0]["role"] == "user"
    s = json.dumps(body["messages"][0]["content"])
    assert '"type": "tool_result"' in s


def test_build_request_stream_flag_only_serialized_when_true() -> None:
    body_off = build_request_body("claude-sonnet-4-6", _req([_user("hi")]), stream=False)
    body_on = build_request_body("claude-sonnet-4-6", _req([_user("hi")]), stream=True)
    assert "stream" not in body_off
    assert body_on["stream"] is True


# ---------------------------------------------------------------------------
# parse_response_body / parse_stop_reason
# ---------------------------------------------------------------------------


def test_parse_response_extracts_text_and_usage() -> None:
    body = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi there"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    r = parse_response_body(body)
    block: ContentBlock = r.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "hi there"
    assert r.usage.input_tokens == 4
    assert r.usage.output_tokens == 2
    assert r.stop_reason is StopReason.END_TURN


def test_parse_response_extracts_tool_use() -> None:
    r = parse_response_body(
        {
            "content": [{"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "rust"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert isinstance(r.content[0], ToolUseBlock)
    assert r.content[0].id == "c1"
    assert r.content[0].name == "search"
    assert r.stop_reason is StopReason.TOOL_USE


def test_parse_response_extracts_thinking_block() -> None:
    r = parse_response_body(
        {
            "content": [
                {"type": "thinking", "thinking": "let me reason..."},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert isinstance(r.content[0], ThinkingBlock)
    assert isinstance(r.content[1], TextBlock)


def test_parse_response_extracts_cache_usage() -> None:
    r = parse_response_body(
        {
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 2,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 30,
            },
        }
    )
    assert r.usage.cache_read_tokens == 50
    assert r.usage.cache_write_tokens == 30


def test_stop_reason_mapping() -> None:
    assert parse_stop_reason("end_turn") is StopReason.END_TURN
    assert parse_stop_reason("tool_use") is StopReason.TOOL_USE
    assert parse_stop_reason("max_tokens") is StopReason.MAX_TOKENS
    assert parse_stop_reason("stop_sequence") is StopReason.STOP_SEQUENCE
    assert parse_stop_reason(None) is StopReason.END_TURN
    assert parse_stop_reason("???") is StopReason.END_TURN


# ---------------------------------------------------------------------------
# context_window / backoff / SSE parsing
# ---------------------------------------------------------------------------


def test_context_window_known_and_unknown() -> None:
    assert context_window("claude-sonnet-4-6") == 200_000
    assert context_window("claude-opus-4-7") == 200_000
    assert context_window("claude-imaginary-9") == 200_000
    assert context_window("gpt-4o") == 0


def test_backoff_grows_then_caps() -> None:
    d0 = _backoff_delay(0)
    d3 = _backoff_delay(3)
    dmax = _backoff_delay(20)
    assert d3 > d0
    assert dmax <= 30.0
    assert d0 == pytest.approx(0.5)


def test_parse_sse_event_basic() -> None:
    raw = 'event: message_start\ndata: {"type":"message_start"}'
    parsed = _parse_sse_event(raw)
    assert parsed is not None
    name, data = parsed
    assert name == "message_start"
    assert data == '{"type":"message_start"}'


def test_parse_sse_event_multiline_data() -> None:
    raw = 'event: message_delta\ndata: {"first":1}\ndata: continuation'
    parsed = _parse_sse_event(raw)
    assert parsed is not None
    name, data = parsed
    assert name == "message_delta"
    assert data == '{"first":1}\ncontinuation'


# ---------------------------------------------------------------------------
# provider()
# ---------------------------------------------------------------------------


def test_provider_info_uses_model_id() -> None:
    c = AnthropicModelInterface("test-key", "claude-sonnet-4-6")
    p = c.provider()
    assert p.name == "anthropic"
    assert p.model_id == "claude-sonnet-4-6"
    assert p.context_window == 200_000


def test_repr_redacts_api_key() -> None:
    c = AnthropicModelInterface("super-secret-key", "claude-sonnet-4-6")
    r = repr(c)
    assert "super-secret-key" not in r
    assert "<redacted>" in r


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def test_from_env_errors_when_unset() -> None:
    var = "__SPORE_TEST_ANTHROPIC_KEY_UNSET__"
    os.environ.pop(var, None)
    with pytest.raises(ProviderError) as exc:
        AnthropicModelInterface.from_env(var, "claude-sonnet-4-6")
    assert "not set" in exc.value.message


def test_from_env_errors_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("__SPORE_TEST_KEY_EMPTY__", "   ")
    with pytest.raises(ProviderError) as exc:
        AnthropicModelInterface.from_env("__SPORE_TEST_KEY_EMPTY__", "claude-sonnet-4-6")
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
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello there"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "test-key", "claude-sonnet-4-6", base_url="https://example.test", http_client=client
    )
    r = await iface.call(_req([_user("hi")]))
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "hello there"
    assert r.usage.input_tokens == 5
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert captured["body"]["model"] == "claude-sonnet-4-6"


async def test_call_maps_429_to_rate_limited() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, text="rate limited")

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(RateLimited) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.retry_after == 7.0


async def test_call_maps_529_to_rate_limited_no_retry_after() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, text="overloaded")

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(RateLimited) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.retry_after is None


async def test_call_maps_400_to_provider_error_with_anthropic_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "max_tokens must be > 0",
                },
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", max_retries=0, http_client=client
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
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "after retry"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", http_client=client
    )
    r = await iface.call(_req([_user("hi")]))
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "after retry"
    assert calls["n"] == 2


async def test_call_timeout_surfaces_as_model_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(ModelTimeoutError):
        await iface.call(_req([_user("hi")]))


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


async def test_count_tokens_uses_real_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"input_tokens": 42})

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", http_client=client
    )
    n = await iface.count_tokens(_req([_user("hi")]))
    assert n == 42
    assert captured["url"].endswith("/v1/messages/count_tokens")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


_SSE_PAYLOAD = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"index":0,"delta":{"type":"text_delta","text":"hello"}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"index":0,"delta":{"type":"text_delta","text":" world"}}\n'
    "\n"
    "event: content_block_stop\n"
    'data: {"index":0}\n'
    "\n"
    "event: message_delta\n"
    'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n'
    "\n"
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n'
    "\n"
)


async def test_streaming_emits_text_delta_then_stop() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_SSE_PAYLOAD.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", http_client=client
    )
    events: list[Any] = []
    async for ev in iface.call_streaming(_req([_user("hi")])):
        events.append(ev)
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "MessageStart",
        "ContentBlockDelta",
        "ContentBlockDelta",
        "ContentBlockStop",
        "MessageStop",
    ]
    last = events[-1]
    assert isinstance(last, MessageStop)
    assert last.usage.input_tokens == 3
    assert last.usage.output_tokens == 5
    assert last.stop_reason is StopReason.END_TURN
    # First event is MessageStart
    assert isinstance(events[0], MessageStart)


_SSE_TOOL_PAYLOAD = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}\n'
    "\n"
    "event: content_block_start\n"
    'data: {"index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup"}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":"}}\n'
    "\n"
    "event: content_block_delta\n"
    'data: {"index":0,"delta":{"type":"input_json_delta","partial_json":"\\"rust\\"}"}}\n'
    "\n"
    "event: content_block_stop\n"
    'data: {"index":0}\n'
    "\n"
    "event: message_delta\n"
    'data: {"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}\n'
    "\n"
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n'
    "\n"
)


async def test_streaming_tool_use_emits_start_then_args() -> None:
    """content_block_start (tool_use) → ToolUseStart carrying id + name, then
    input_json_delta fragments → ToolUseDelta."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_SSE_TOOL_PAYLOAD.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", http_client=client
    )
    starts: list[ToolUseStart] = []
    fragments: list[str] = []
    final_stop = StopReason.END_TURN
    async for ev in iface.call_streaming(_req([_user("hi")])):
        if isinstance(ev, ToolUseStart):
            starts.append(ev)
        elif isinstance(ev, ToolUseDelta):
            fragments.append(ev.partial_json)
        elif isinstance(ev, MessageStop):
            final_stop = ev.stop_reason
    assert len(starts) == 1
    assert starts[0].index == 0
    assert starts[0].id == "toolu_1"
    assert starts[0].name == "lookup"
    assert json.loads("".join(fragments)) == {"q": "rust"}
    assert final_stop is StopReason.TOOL_USE


async def test_streaming_maps_status_error_eagerly() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "3"}, text="rate limited")

    client = _mock_client(httpx.MockTransport(handler))
    iface = AnthropicModelInterface(
        "k", "claude-sonnet-4-6", base_url="https://x.test", max_retries=0, http_client=client
    )
    with pytest.raises(RateLimited) as exc:
        async for _ in iface.call_streaming(_req([_user("hi")])):
            pass
    assert exc.value.retry_after == 3.0


# ---------------------------------------------------------------------------
# Live-API integration tests (skipped by default)
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; live-API test skipped",
)


@LIVE
async def test_anthropic_live_call_returns_response() -> None:
    iface = AnthropicModelInterface.from_env("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
    try:
        r = await iface.call(_req([_user("Reply with the word 'pong'.")]))
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0
    finally:
        await iface.aclose()


@LIVE
async def test_anthropic_live_count_tokens_is_nonzero() -> None:
    iface = AnthropicModelInterface.from_env("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
    try:
        n = await iface.count_tokens(_req([_user("count my tokens please")]))
        assert n > 0
    finally:
        await iface.aclose()


@LIVE
async def test_anthropic_live_streaming_emits_events() -> None:
    iface = AnthropicModelInterface.from_env("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
    saw_stop = False
    try:
        async for ev in iface.call_streaming(_req([_user("Reply with the word 'pong'.")])):
            if isinstance(ev, MessageStop):
                saw_stop = True
        assert saw_stop
    finally:
        await iface.aclose()
