"""Tests for :mod:`spore_core.ollama` — issue #41.

Mirrors the Rust reference at ``rust/crates/spore-core/src/ollama.rs``
test module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from spore_core.model import (
    Message,
    MessageStart,
    MessageStop,
    ModelParams,
    ModelRequest,
    ProviderError,
    ProviderInfo,
    ReplayMode,
    ReplayModelInterface,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    TimeoutError as ModelTimeoutError,
    ToolCallContent,
    ToolResultContent,
    ToolSchema,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
)
from spore_core.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_TIMEOUT_SECONDS,
    OllamaModelInterface,
    build_request_body,
    context_window,
    parse_response_body,
    parse_stop_reason,
    parse_structured_content,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _user(text: str) -> Message:
    return Message(role=Role.USER, content=TextContent(text=text))


def _req(messages: list[Message], **kwargs: Any) -> ModelRequest:
    return ModelRequest(messages=messages, **kwargs)


def _mock_client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


def _tags_handler(model: str = "llama3.2") -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": f"{model}:latest"}]})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return handle


# ---------------------------------------------------------------------------
# Constructors / defaults
# ---------------------------------------------------------------------------


def test_new_uses_localhost_defaults() -> None:
    c = OllamaModelInterface("llama3.2")
    assert c.base_url == "http://localhost:11434"
    assert c.model_id == "llama3.2"
    assert c.keep_alive == "5m"


def test_with_base_url_overrides() -> None:
    c = OllamaModelInterface.with_base_url("mistral", "http://remote:9999")
    assert c.base_url == "http://remote:9999"
    assert c.model_id == "mistral"


def test_defaults_match_spec() -> None:
    assert DEFAULT_BASE_URL == "http://localhost:11434"
    assert DEFAULT_TIMEOUT_SECONDS == 300.0
    assert DEFAULT_KEEP_ALIVE == "5m"


# ---------------------------------------------------------------------------
# build_request_body
# ---------------------------------------------------------------------------


def test_build_request_serializes_options_and_keep_alive() -> None:
    body = build_request_body(
        "llama3.2",
        "10m",
        _req(
            [_user("hi")],
            params=ModelParams(max_tokens=256, temperature=0.7, top_p=0.9, stop_sequences=["END"]),
        ),
        stream=False,
    )
    assert body["keep_alive"] == "10m"
    assert body["options"]["num_predict"] == 256
    assert body["options"]["temperature"] == 0.7
    assert body["options"]["top_p"] == 0.9
    assert body["options"]["stop"] == ["END"]
    assert body["stream"] is False


def test_build_request_serializes_tools() -> None:
    body = build_request_body(
        "llama3.2",
        None,
        _req(
            [_user("hi")],
            tools=[
                ToolSchema(
                    name="search", description="search the web", input_schema={"type": "object"}
                )
            ],
        ),
        stream=False,
    )
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "search"


def test_build_request_tool_call_uses_object_arguments() -> None:
    body = build_request_body(
        "llama3.2",
        None,
        _req(
            [
                Message(
                    role=Role.ASSISTANT,
                    content=ToolCallContent(id="call-0", name="fetch", input={"url": "x"}),
                )
            ]
        ),
        stream=False,
    )
    msg = body["messages"][0]
    args = msg["tool_calls"][0]["function"]["arguments"]
    # arguments must be a JSON object (dict), NOT a JSON-encoded string.
    assert isinstance(args, dict)
    assert args == {"url": "x"}


def test_build_request_tool_result_maps_to_tool_role() -> None:
    body = build_request_body(
        "llama3.2",
        None,
        _req(
            [
                Message(
                    role=Role.TOOL,
                    content=ToolResultContent(tool_use_id="call-0", content="ok"),
                )
            ]
        ),
        stream=False,
    )
    msg = body["messages"][0]
    assert msg["role"] == "tool"
    assert msg["content"] == "ok"
    assert msg["tool_call_id"] == "call-0"


def test_thinking_block_omitted_in_request() -> None:
    body = build_request_body("llama3.2", None, _req([_user("hi")]), stream=False)
    s = json.dumps(body)
    # No "thinking" key ever appears on the Ollama wire.
    assert "thinking" not in s


def test_build_request_omits_keep_alive_when_none() -> None:
    body = build_request_body("llama3.2", None, _req([_user("hi")]), stream=False)
    assert "keep_alive" not in body


def test_build_request_omits_options_when_empty() -> None:
    body = build_request_body("llama3.2", "5m", _req([_user("hi")]), stream=False)
    assert "options" not in body


# ---------------------------------------------------------------------------
# parse_stop_reason / parse_response_body
# ---------------------------------------------------------------------------


def test_stop_reason_mapping_stop() -> None:
    assert parse_stop_reason("stop") is StopReason.END_TURN
    assert parse_stop_reason(None) is StopReason.END_TURN
    assert parse_stop_reason("???") is StopReason.END_TURN


def test_stop_reason_mapping_tool_calls_and_length() -> None:
    assert parse_stop_reason("tool_calls") is StopReason.TOOL_USE
    assert parse_stop_reason("length") is StopReason.MAX_TOKENS


def test_parse_response_extracts_usage() -> None:
    r = parse_response_body(
        {
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 2,
        }
    )
    assert r.usage.input_tokens == 7
    assert r.usage.output_tokens == 2
    assert r.stop_reason is StopReason.END_TURN
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "hi"


def test_parse_response_cache_fields_none() -> None:
    r = parse_response_body(
        {
            "message": {"role": "assistant", "content": "x"},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
    )
    assert r.usage.cache_read_tokens is None
    assert r.usage.cache_write_tokens is None


def test_parse_response_tool_call_synthesizes_id() -> None:
    r = parse_response_body(
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "fetch", "arguments": {"url": "x"}}},
                    {"function": {"name": "search", "arguments": {"q": "y"}}},
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
    )
    assert r.stop_reason is StopReason.TOOL_USE
    assert isinstance(r.content[0], ToolUseBlock)
    assert r.content[0].id == "call-0"
    assert r.content[0].name == "fetch"
    assert r.content[0].input == {"url": "x"}
    assert isinstance(r.content[1], ToolUseBlock)
    assert r.content[1].id == "call-1"


# ---------------------------------------------------------------------------
# context_window / provider
# ---------------------------------------------------------------------------


def test_context_window_table() -> None:
    assert context_window("llama3.2") == 128_000
    assert context_window("llama3.2:3b") == 128_000
    assert context_window("qwen2.5-coder-7b") == 128_000
    assert context_window("mistral") == 32_000
    assert context_window("gemma") == 8_192
    assert context_window("unknown") == 0


def test_provider_info_uses_table() -> None:
    p = OllamaModelInterface("llama3.2").provider()
    assert p.name == "ollama"
    assert p.model_id == "llama3.2"
    assert p.context_window == 128_000


# ---------------------------------------------------------------------------
# End-to-end mocked HTTP — call()
# ---------------------------------------------------------------------------


async def test_call_against_mock_returns_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "hello there"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                },
            )
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    r = await iface.call(_req([_user("hi")]))
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "hello there"
    assert r.usage.input_tokens == 5
    assert r.usage.output_tokens == 2


async def test_connection_refused_helpful_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://127.0.0.1:1", http_client=client)
    with pytest.raises(ProviderError) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.code == 0
    assert "Ollama not running" in exc.value.message


async def test_connection_error_does_not_retry() -> None:
    """Fail-fast: a connect error returns immediately, not after backoff."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://127.0.0.1:1", http_client=client)
    start = time.monotonic()
    with pytest.raises(ProviderError):
        await iface.call(_req([_user("hi")]))
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"expected fail-fast (<500ms); took {elapsed:.3f}s"


async def test_model_not_found_suggests_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "mistral:latest"}]})
        raise AssertionError("should not POST /api/chat")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    with pytest.raises(ProviderError) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.code == 404
    assert "ollama pull llama3.2" in exc.value.message


async def test_chat_404_maps_to_pull_suggestion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(
                404,
                text='{"error":"model \'llama3.2\' not found, try pulling it first"}',
            )
        raise AssertionError("unexpected")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    with pytest.raises(ProviderError) as exc:
        await iface.call(_req([_user("hi")]))
    assert exc.value.code == 404
    assert "ollama pull llama3.2" in exc.value.message


async def test_timeout_maps_to_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    with pytest.raises(ModelTimeoutError):
        await iface.call(_req([_user("hi")]))


async def test_model_check_cached_after_first_call() -> None:
    calls = {"tags": 0, "chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            calls["tags"] += 1
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/chat":
            calls["chat"] += 1
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 1,
                    "eval_count": 1,
                },
            )
        raise AssertionError("unexpected")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    await iface.call(_req([_user("a")]))
    await iface.call(_req([_user("b")]))
    assert calls["tags"] == 1
    assert calls["chat"] == 2


# ---------------------------------------------------------------------------
# Streaming (NDJSON)
# ---------------------------------------------------------------------------


_NDJSON_TEXT = (
    '{"message":{"role":"assistant","content":"hello"},"done":false}\n'
    '{"message":{"role":"assistant","content":" world"},"done":false}\n'
    '{"message":{"role":"assistant","content":""},"done":true,'
    '"done_reason":"stop","prompt_eval_count":3,"eval_count":5}\n'
)


async def test_streaming_emits_text_delta_then_stop() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        return httpx.Response(
            200,
            content=_NDJSON_TEXT.encode("utf-8"),
            headers={"content-type": "application/x-ndjson"},
        )

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
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


async def test_streaming_parses_ndjson_lines() -> None:
    """Multiple JSON objects in one chunk are both parsed."""

    ndjson = (
        '{"message":{"role":"assistant","content":"ab"},"done":false}\n'
        '{"message":{"role":"assistant","content":"cd"},"done":false}\n'
        '{"message":{"role":"assistant","content":""},"done":true,'
        '"done_reason":"stop","prompt_eval_count":1,"eval_count":1}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    deltas: list[str] = []
    async for ev in iface.call_streaming(_req([_user("hi")])):
        if hasattr(ev, "delta") and isinstance(ev.delta, str):
            deltas.append(ev.delta)
    assert deltas == ["ab", "cd"]


async def test_streaming_done_carries_usage() -> None:
    ndjson = (
        '{"message":{"role":"assistant","content":"x"},"done":true,'
        '"done_reason":"stop","prompt_eval_count":42,"eval_count":7}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    final_usage = None
    async for ev in iface.call_streaming(_req([_user("hi")])):
        if isinstance(ev, MessageStop):
            final_usage = ev.usage
    assert final_usage is not None
    assert final_usage.input_tokens == 42
    assert final_usage.output_tokens == 7


async def test_streaming_accumulates_tool_calls() -> None:
    """Ollama returns full arguments objects per chunk (not partial
    strings). Verify we emit a ToolUseStart carrying the name + id, then one
    ToolUseDelta with the serialized object and MessageStop.stop_reason=ToolUse."""

    ndjson = (
        '{"message":{"role":"assistant","tool_calls":'
        '[{"function":{"name":"fetch","arguments":{"url":"x"}}}]},"done":false}\n'
        '{"message":{"role":"assistant","content":""},"done":true,'
        '"done_reason":"tool_calls","prompt_eval_count":1,"eval_count":1}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    tool_jsons: list[str] = []
    starts: list[ToolUseStart] = []
    final_stop = StopReason.END_TURN
    async for ev in iface.call_streaming(_req([_user("hi")])):
        if isinstance(ev, ToolUseStart):
            starts.append(ev)
        elif isinstance(ev, ToolUseDelta):
            tool_jsons.append(ev.partial_json)
        elif isinstance(ev, MessageStop):
            final_stop = ev.stop_reason
    assert len(starts) == 1
    assert starts[0].name == "fetch"
    assert starts[0].id == "call_1"
    assert len(tool_jsons) == 1
    assert json.loads(tool_jsons[0]) == {"url": "x"}
    assert final_stop is StopReason.TOOL_USE


async def test_streaming_keeps_multiple_tool_calls_distinct() -> None:
    """A response with three tool calls streams them in SEPARATE chunks, each a
    one-element tool_calls array distinguished only by ``function.index``. Each
    call must land on its own stream index so its argument JSON stays
    well-formed — keying off the array position would collapse all three onto
    index 1 and concatenate their args into invalid JSON."""

    ndjson = (
        '{"message":{"role":"assistant","tool_calls":[{"id":"call_a",'
        '"function":{"index":0,"name":"calculator",'
        '"arguments":{"a":"144","b":"12","op":"/"}}}]},"done":false}\n'
        '{"message":{"role":"assistant","tool_calls":[{"id":"call_b",'
        '"function":{"index":1,"name":"get_current_time","arguments":{}}}]},'
        '"done":false}\n'
        '{"message":{"role":"assistant","tool_calls":[{"id":"call_c",'
        '"function":{"index":2,"name":"reverse_string",'
        '"arguments":{"text":"harness"}}}]},"done":false}\n'
        '{"message":{"role":"assistant","content":""},"done":true,'
        '"done_reason":"tool_calls","prompt_eval_count":1,"eval_count":1}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    names: dict[int, str] = {}
    jsons: dict[int, str] = {}
    async for ev in iface.call_streaming(_req([_user("hi")])):
        if isinstance(ev, ToolUseStart):
            names[ev.index] = ev.name
        elif isinstance(ev, ToolUseDelta):
            jsons[ev.index] = jsons.get(ev.index, "") + ev.partial_json
    assert len(names) == 3
    assert len(jsons) == 3
    for blob in jsons.values():
        assert isinstance(json.loads(blob), dict)
    assert [names[i] for i in sorted(names)] == [
        "calculator",
        "get_current_time",
        "reverse_string",
    ]


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


async def test_count_tokens_uses_embed_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"prompt_eval_count": 123})
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    n = await iface.count_tokens(_req([_user("hello world")]))
    assert n == 123


async def test_count_tokens_falls_back_to_heuristic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(500)
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    # 40 chars + 1 newline = 41 → 41/4 = 10
    n = await iface.count_tokens(_req([_user("a" * 40)]))
    assert n == 10


# ---------------------------------------------------------------------------
# /api/show discovery + tool-capability guard
# ---------------------------------------------------------------------------


def _tool_req(model_msg: str = "use a tool") -> ModelRequest:
    return _req(
        [_user(model_msg)],
        tools=[
            ToolSchema(name="search", description="search the web", input_schema={"type": "object"})
        ],
    )


def _chat_ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    )


async def test_provider_reflects_discovered_context_window() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={"model_info": {"llama.context_length": 16_384}, "capabilities": ["tools"]},
            )
        if request.url.path == "/api/chat":
            return _chat_ok_response()
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    # Before the probe runs, provider() falls back to the static table.
    assert iface.provider().context_window == 128_000
    await iface.call(_req([_user("hi")]))
    # After the probe, provider() reflects the discovered value.
    assert iface.provider().context_window == 16_384


async def test_provider_falls_back_when_show_404s() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            return httpx.Response(404)
        if request.url.path == "/api/chat":
            return _chat_ok_response()
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    await iface.call(_req([_user("hi")]))
    assert iface.provider().context_window == 128_000


async def test_provider_falls_back_when_context_length_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            # Succeeds but has no *.context_length entry.
            return httpx.Response(
                200,
                json={"model_info": {"general.architecture": "llama"}, "capabilities": ["tools"]},
            )
        if request.url.path == "/api/chat":
            return _chat_ok_response()
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    await iface.call(_req([_user("hi")]))
    assert iface.provider().context_window == 128_000


async def test_tool_request_rejected_when_capability_absent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma:latest"}]})
        if request.url.path == "/api/show":
            # capabilities lacks "tools".
            return httpx.Response(
                200,
                json={
                    "model_info": {"gemma.context_length": 8_192},
                    "capabilities": ["completion"],
                },
            )
        raise AssertionError(f"should not POST {request.url.path}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("gemma", base_url="http://x.test", http_client=client)
    with pytest.raises(ProviderError) as exc:
        await iface.call(_tool_req())
    assert exc.value.code == 0
    assert "does not support tool calling" in exc.value.message


async def test_tool_request_proceeds_when_capability_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "model_info": {"llama.context_length": 128_000},
                    "capabilities": ["completion", "tools"],
                },
            )
        if request.url.path == "/api/chat":
            return _chat_ok_response()
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    r = await iface.call(_tool_req())
    assert isinstance(r.content[0], TextBlock)
    assert r.content[0].text == "ok"


async def test_tool_request_rejected_when_capabilities_empty() -> None:
    # ``/api/show`` returns an empty ``capabilities`` array. With the static
    # whitelist removed, empty capabilities fail closed — even for a model id
    # (``llama3.2``) that the old prefix table would have allowed.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "model_info": {"llama.context_length": 128_000},
                    "capabilities": [],
                },
            )
        raise AssertionError(f"should not POST {request.url.path}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    with pytest.raises(ProviderError) as exc:
        await iface.call(_tool_req())
    assert exc.value.code == 0
    assert "does not support tool calling" in exc.value.message


async def test_show_fetched_at_most_once() -> None:
    calls = {"tags": 0, "show": 0, "chat": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            calls["tags"] += 1
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            calls["show"] += 1
            return httpx.Response(
                200,
                json={"model_info": {"llama.context_length": 32_000}, "capabilities": ["tools"]},
            )
        if request.url.path == "/api/chat":
            calls["chat"] += 1
            return _chat_ok_response()
        raise AssertionError(f"unexpected: {request.url}")

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    await iface.call(_req([_user("a")]))
    await iface.call(_req([_user("b")]))
    assert calls["tags"] == 1
    assert calls["show"] == 1
    assert calls["chat"] == 2


# ---------------------------------------------------------------------------
# Fixture replay (shared cross-language fixture)
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = (
    _REPO_ROOT / "fixtures" / "model_responses" / "model_interface" / "ollama_basic_text.jsonl"
)


async def test_ollama_basic_text_fixture_replays() -> None:
    """Round-trip the shared Ollama fixture through ReplayModelInterface."""

    assert _FIXTURE.exists(), f"missing fixture: {_FIXTURE}"
    text = _FIXTURE.read_text(encoding="utf-8")
    provider = ProviderInfo(name="ollama", model_id="llama3.2", context_window=128_000)
    replay = ReplayModelInterface.from_jsonl(text, provider, mode=ReplayMode.POSITIONAL)
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
# Live integration tests (skipped by default; require local Ollama)
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skip(reason="live-API; needs local ollama with llama3.2 pulled")


@LIVE
async def test_ollama_live_call() -> None:
    iface = OllamaModelInterface("llama3.2")
    try:
        r = await iface.call(_req([_user("Reply with the word 'pong'.")]))
        assert r.usage.input_tokens > 0
        assert r.usage.output_tokens > 0
    finally:
        await iface.aclose()


@LIVE
async def test_ollama_live_streaming() -> None:
    iface = OllamaModelInterface("llama3.2")
    saw_stop = False
    try:
        async for ev in iface.call_streaming(_req([_user("Reply with the word 'pong'.")])):
            if isinstance(ev, MessageStop):
                saw_stop = True
        assert saw_stop
    finally:
        await iface.aclose()


@LIVE
async def test_ollama_live_tool_call() -> None:
    iface = OllamaModelInterface("llama3.2")
    try:
        r = await iface.call(
            _req(
                [_user("Use the echo tool with text='hi'.")],
                tools=[
                    ToolSchema(
                        name="echo",
                        description="echoes the given text",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    )
                ],
            )
        )
        assert r.stop_reason in (StopReason.TOOL_USE, StopReason.END_TURN)
    finally:
        await iface.aclose()


# ---------------------------------------------------------------------------
# structured tool calls (opt-in constrained decoding) — mirrors Rust
# ---------------------------------------------------------------------------


def _structured_tool_req() -> ModelRequest:
    return ModelRequest(
        messages=[_user("write a summary file")],
        params=ModelParams(structured_tool_calls=True),
        tools=[
            ToolSchema(
                name="write_file",
                description="write a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            ),
            ToolSchema(
                name="read_file",
                description="read a file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            ),
        ],
    )


def test_build_request_structured_sets_format_drops_tools_adds_system() -> None:
    body = build_request_body("llama3.2", None, _structured_tool_req(), stream=False)
    # Native tools dropped in structured mode.
    assert "tools" not in body
    # format schema present with tool enum = tool names + "final".
    fmt = body["format"]
    enum_vals = fmt["properties"]["tool"]["enum"]
    assert "write_file" in enum_vals
    assert "read_file" in enum_vals
    assert "final" in enum_vals
    assert fmt["required"] == ["tool"]
    # A system message describing the tools is prepended.
    assert body["messages"][0]["role"] == "system"
    assert "write_file" in body["messages"][0]["content"]
    assert "read_file" in body["messages"][0]["content"]
    assert "SINGLE JSON object" in body["messages"][0]["content"]


def test_build_request_structured_merges_into_existing_system_message() -> None:
    req = _structured_tool_req()
    req.messages.insert(0, Message(role=Role.SYSTEM, content=TextContent(text="You are terse.")))
    body = build_request_body("llama3.2", None, req, stream=False)
    system_count = sum(1 for m in body["messages"] if m["role"] == "system")
    assert system_count == 1
    assert "You are terse." in body["messages"][0]["content"]
    assert "write_file" in body["messages"][0]["content"]


def test_build_request_structured_off_when_no_tools() -> None:
    # Flag on but no tools → unchanged behavior, no format.
    req = ModelRequest(
        messages=[_user("hi")],
        params=ModelParams(structured_tool_calls=True),
    )
    body = build_request_body("llama3.2", None, req, stream=False)
    assert "format" not in body


def test_build_request_structured_off_by_default() -> None:
    # Flag default off with tools present → native tools, no format.
    req = ModelRequest(
        messages=[_user("hi")],
        tools=[ToolSchema(name="search", description="search the web", input_schema={})],
    )
    body = build_request_body("llama3.2", None, req, stream=False)
    assert "format" not in body
    assert len(body["tools"]) == 1


def test_structured_flag_omitted_from_serialization_when_false() -> None:
    # Hash parity: a False flag must NOT appear in the serialized request.
    off = ModelParams()
    assert "structured_tool_calls" not in off.model_dump(mode="json")
    assert "structured_tool_calls" not in json.loads(off.model_dump_json())
    on = ModelParams(structured_tool_calls=True)
    assert on.model_dump(mode="json")["structured_tool_calls"] is True


def test_parse_response_structured_tool_call() -> None:
    r = parse_response_body(
        {
            "message": {
                "role": "assistant",
                "content": (
                    '{"tool":"write_file","arguments":{"path":"SUMMARY.md","content":"hi"}}'
                ),
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
        structured=True,
    )
    assert r.stop_reason is StopReason.TOOL_USE
    assert len(r.content) == 1
    block = r.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.name == "write_file"
    assert block.input == {"path": "SUMMARY.md", "content": "hi"}


def test_parse_response_structured_final() -> None:
    r = parse_response_body(
        {
            "message": {"role": "assistant", "content": '{"tool":"final","content":"all done"}'},
            "done": True,
            "done_reason": "stop",
        },
        structured=True,
    )
    assert r.stop_reason is StopReason.END_TURN
    assert len(r.content) == 1
    block = r.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "all done"


def test_parse_response_structured_malformed_falls_back_to_text() -> None:
    # Weak model violates constrained decoding: not valid JSON. We must not
    # raise — fall back to a Text block with the raw content and END_TURN.
    r = parse_response_body(
        {
            "message": {"role": "assistant", "content": "oops not json"},
            "done": True,
            "done_reason": "stop",
        },
        structured=True,
    )
    assert r.stop_reason is StopReason.END_TURN
    assert len(r.content) == 1
    block = r.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "oops not json"


def test_parse_structured_content_missing_tool_falls_back() -> None:
    blocks, stop = parse_structured_content('{"arguments":{}}', 0)
    assert stop is StopReason.END_TURN
    assert isinstance(blocks[0], TextBlock)


# ── structured fence stripping (capable/cloud models wrap JSON) ─────────────


def test_parse_structured_json_fenced_tool_call_dispatches() -> None:
    # Regression for the exact gemma-cloud output: the constrained JSON tool
    # call arrives inside a ```json fence. Must dispatch, not fall back to Text.
    raw = '```json\n{"tool":"web_search","arguments":{"query":"x"}}\n```'
    blocks, stop = parse_structured_content(raw, 0)
    assert stop is StopReason.TOOL_USE
    assert isinstance(blocks[0], ToolUseBlock)
    assert blocks[0].name == "web_search"
    assert blocks[0].input == {"query": "x"}


def test_parse_structured_bare_fenced_tool_call_dispatches() -> None:
    # A bare ``` fence (no language tag) also strips and dispatches.
    raw = '```\n{"tool":"web_search","arguments":{"query":"y"}}\n```'
    blocks, stop = parse_structured_content(raw, 0)
    assert stop is StopReason.TOOL_USE
    assert isinstance(blocks[0], ToolUseBlock)
    assert blocks[0].name == "web_search"


def test_parse_structured_fenced_final_is_text() -> None:
    # A fenced ``final`` envelope still resolves to a Text/END_TURN answer.
    raw = '```json\n{"tool":"final","content":"done"}\n```'
    blocks, stop = parse_structured_content(raw, 0)
    assert stop is StopReason.END_TURN
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "done"


def test_parse_structured_raw_tool_call_still_dispatches() -> None:
    # Un-fenced tool calls (grammar-honoring models) still dispatch — no regression.
    raw = '{"tool":"web_search","arguments":{"query":"z"}}'
    blocks, stop = parse_structured_content(raw, 0)
    assert stop is StopReason.TOOL_USE
    assert isinstance(blocks[0], ToolUseBlock)
    assert blocks[0].name == "web_search"


def test_parse_structured_garbage_falls_back_to_text() -> None:
    # Genuine garbage still falls back to a Text block with END_TURN.
    raw = "not json at all"
    blocks, stop = parse_structured_content(raw, 0)
    assert stop is StopReason.END_TURN
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "not json at all"


async def test_streaming_structured_buffers_json_then_reconstructs_tool_call() -> None:
    """In structured mode the constrained JSON object arrives as content
    deltas spread across chunks. We buffer it (never surfacing raw JSON as
    answer text) and at ``done`` reconstruct a write_file tool call with
    valid argument JSON."""

    ndjson = (
        '{"message":{"role":"assistant","content":"{\\"tool\\":\\"write"},"done":false}\n'
        '{"message":{"role":"assistant","content":"_file\\",\\"arguments\\":{\\"path\\""},'
        '"done":false}\n'
        '{"message":{"role":"assistant","content":":\\"a.txt\\",\\"content\\":\\"hi\\"}}"},'
        '"done":false}\n'
        '{"message":{"role":"assistant","content":""},"done":true,'
        '"done_reason":"stop","prompt_eval_count":2,"eval_count":3}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2:latest"}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["tools"]})
        return httpx.Response(200, content=ndjson.encode("utf-8"))

    client = _mock_client(httpx.MockTransport(handler))
    iface = OllamaModelInterface("llama3.2", base_url="http://x.test", http_client=client)
    starts: list[ToolUseStart] = []
    tool_jsons: list[str] = []
    raw_deltas: list[str] = []
    final_stop = StopReason.END_TURN
    try:
        async for ev in iface.call_streaming(_structured_tool_req()):
            if isinstance(ev, ToolUseStart):
                starts.append(ev)
            elif isinstance(ev, ToolUseDelta):
                tool_jsons.append(ev.partial_json)
            elif hasattr(ev, "delta") and isinstance(ev.delta, str):
                raw_deltas.append(ev.delta)
            elif isinstance(ev, MessageStop):
                final_stop = ev.stop_reason
    finally:
        await iface.aclose()
    # Raw JSON fragments must NOT be surfaced as content deltas.
    assert raw_deltas == []
    assert len(starts) == 1
    assert starts[0].name == "write_file"
    assert len(tool_jsons) == 1
    assert json.loads(tool_jsons[0]) == {"path": "a.txt", "content": "hi"}
    assert final_stop is StopReason.TOOL_USE


# ── output-schema constrained decoding (issue #139) ────────────────────────


def test_build_request_output_schema_populates_format_channel() -> None:
    # #139 AC1: ``params.output_schema`` (set by the harness for an
    # output-schema-enforced terminal turn) routes into the Ollama ``format``
    # constrained-decoding channel verbatim, even with NO tools.
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok", "error"]}},
        "required": ["status"],
    }
    req = _req([_user("answer")], params=ModelParams(output_schema=schema))
    body = build_request_body("llama3.2", None, req, stream=False)
    assert body["format"] == schema


def test_build_request_no_output_schema_leaves_format_absent() -> None:
    # Absent output_schema (the default) keeps ``format`` unset — byte-identical
    # to pre-#139.
    body = build_request_body("llama3.2", None, _req([_user("hi")]), stream=False)
    assert "format" not in body
