"""Tests for ModelInterface (issue #1).

The fixture-replay test mirrors the Rust integration test at
``rust/crates/spore-core/tests/model_fixture_replay.rs`` byte-for-byte —
both consume the same shared JSONL fixture under
``fixtures/model_responses/model_interface/basic_text.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spore_core import (
    AlwaysHaltError,
    BudgetExceeded,
    ContentBlockDelta,
    ContentBlockStop,
    ContextLimitExceeded,
    Message,
    MessageStart,
    MessageStop,
    MockModelInterface,
    ModelError,
    ModelInterface,
    ModelParams,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ProviderInfo,
    RateLimited,
    ReplayModelInterface,
    Role,
    SporeError,
    StopReason,
    TextBlock,
    TextContent,
    TimeoutError,
    TokenUsage,
    ToolCall,
    ToolSchema,
    ToolUseBlock,
    enforce_budget,
    enforce_context_limit,
)


def _provider() -> ProviderInfo:
    return ProviderInfo(name="test", model_id="test-1", context_window=1000)


def _empty_request() -> ModelRequest:
    return ModelRequest()


def _text_response(text: str, in_tok: int, out_tok: int) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=text)],
        usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok),
        stop_reason=StopReason.END_TURN,
    )


# ---------------------------------------------------------------------------
# Mock-driven rules
# ---------------------------------------------------------------------------


def test_mock_satisfies_protocol() -> None:
    assert isinstance(MockModelInterface(_provider()), ModelInterface)


async def test_call_returns_queued_response() -> None:
    m = MockModelInterface(_provider())
    m.push_response(_text_response("hi", 3, 1))
    r = await m.call(_empty_request())
    assert len(r.content) == 1
    assert r.stop_reason is StopReason.END_TURN


async def test_token_usage_reported_on_every_call() -> None:
    m = MockModelInterface(_provider())
    m.push_response(_text_response("a", 5, 7)).push_response(_text_response("b", 11, 13))
    r1 = await m.call(_empty_request())
    r2 = await m.call(_empty_request())
    assert (r1.usage.input_tokens, r1.usage.output_tokens) == (5, 7)
    assert (r2.usage.input_tokens, r2.usage.output_tokens) == (11, 13)
    assert m.call_count == 2


def test_provider_identity_reported() -> None:
    m = MockModelInterface(_provider())
    p = m.provider()
    assert (p.name, p.model_id, p.context_window) == ("test", "test-1", 1000)


async def test_streaming_yields_message_stop_with_usage() -> None:
    m = MockModelInterface(_provider())
    m.push_response(_text_response("hello", 4, 2))
    saw_start = False
    final_usage: TokenUsage | None = None
    async for ev in m.call_streaming(_empty_request()):
        if isinstance(ev, MessageStart):
            saw_start = True
        elif isinstance(ev, MessageStop):
            final_usage = ev.usage
    assert saw_start
    assert final_usage is not None
    assert (final_usage.input_tokens, final_usage.output_tokens) == (4, 2)


async def test_provider_errors_surface_typed() -> None:
    m = MockModelInterface(_provider())
    m.push_response(ProviderError(code=503, message="unavailable"))
    with pytest.raises(ProviderError) as exc:
        await m.call(_empty_request())
    assert exc.value.code == 503


async def test_rate_limit_surface_with_retry_after() -> None:
    m = MockModelInterface(_provider())
    m.push_response(RateLimited(retry_after=2.0))
    with pytest.raises(RateLimited) as exc:
        await m.call(_empty_request())
    assert exc.value.retry_after == 2.0


async def test_timeout_surface() -> None:
    m = MockModelInterface(_provider())
    m.push_response(TimeoutError())
    with pytest.raises(TimeoutError):
        await m.call(_empty_request())


# ---------------------------------------------------------------------------
# Context / budget guards
# ---------------------------------------------------------------------------


def test_context_limit_enforced_pre_call() -> None:
    with pytest.raises(ContextLimitExceeded) as exc:
        enforce_context_limit(1500, 1000)
    assert (exc.value.limit, exc.value.actual) == (1000, 1500)


def test_context_limit_passes_when_under_or_equal() -> None:
    enforce_context_limit(999, 1000)
    enforce_context_limit(1000, 1000)


def test_budget_enforced_against_max_tokens() -> None:
    with pytest.raises(BudgetExceeded) as exc:
        enforce_budget(101, 100)
    assert (exc.value.budget, exc.value.used) == (100, 101)


def test_budget_passes_when_under_or_unset() -> None:
    enforce_budget(99, 100)
    enforce_budget(100, 100)
    enforce_budget(1_000_000, None)


def test_always_halt_marker_on_context_and_budget() -> None:
    assert issubclass(ContextLimitExceeded, AlwaysHaltError)
    assert issubclass(BudgetExceeded, AlwaysHaltError)
    assert issubclass(ContextLimitExceeded, ModelError)
    assert issubclass(BudgetExceeded, ModelError)
    assert issubclass(ModelError, SporeError)


# ---------------------------------------------------------------------------
# Error variant coverage
# ---------------------------------------------------------------------------


def test_every_error_variant_is_constructible() -> None:
    errs: list[ModelError] = [
        ProviderError(code=500, message="boom"),
        RateLimited(retry_after=5.0),
        RateLimited(retry_after=None),
        ContextLimitExceeded(limit=1, actual=2),
        BudgetExceeded(budget=1, used=2),
        TimeoutError(),
    ]
    for e in errs:
        assert str(e)
        assert e.kind  # class attribute set on every subclass


# ---------------------------------------------------------------------------
# JSON round-trips (cross-language wire format)
# ---------------------------------------------------------------------------


def test_model_request_roundtrips_json() -> None:
    req = ModelRequest(
        messages=[Message(role=Role.USER, content=TextContent(text="hi"))],
        tools=[
            ToolSchema(
                name="echo",
                description="echoes input",
                input_schema={"type": "object"},
            )
        ],
        params=ModelParams(temperature=0.7, max_tokens=1024),
    )
    s = req.model_dump_json()
    back = ModelRequest.model_validate_json(s)
    assert back == req


def test_model_response_roundtrips_json_with_tool_use() -> None:
    resp = ModelResponse(
        content=[
            TextBlock(text="ok"),
            ToolUseBlock(id="1", name="x", input={"a": 1}),
        ],
        usage=TokenUsage(
            input_tokens=3, output_tokens=4, cache_read_tokens=1, cache_write_tokens=2
        ),
        stop_reason=StopReason.TOOL_USE,
    )
    s = resp.model_dump_json()
    back = ModelResponse.model_validate_json(s)
    assert back == resp
    # Spot-check the wire shape — the tool_use block flattens the ToolCall
    # alongside the type tag so it stays portable with the Rust serialization.
    assert '"type":"tool_use"' in s
    assert '"name":"x"' in s


def test_tool_call_dataclass_constructible() -> None:
    # ToolCall is the canonical inner struct shared with #4 (ToolRegistry).
    tc = ToolCall(id="t1", name="echo", input={"text": "hi"})
    assert tc.name == "echo"
    assert tc.input["text"] == "hi"


# ---------------------------------------------------------------------------
# Fixture replay
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "fixtures").is_dir() and (parent / "rust").is_dir():
            return parent
    raise RuntimeError("could not locate spore-core repo root from test file")


def _fixture_path() -> Path:
    return _repo_root() / "fixtures/model_responses/model_interface/basic_text.jsonl"


async def test_basic_text_fixture_replays_in_order() -> None:
    jsonl = _fixture_path().read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    assert replay.remaining() == 3

    r1 = await replay.call(_empty_request())
    assert r1.stop_reason is StopReason.END_TURN
    assert r1.usage.input_tokens == 8
    assert r1.usage.output_tokens == 11
    assert isinstance(r1.content[0], TextBlock)
    assert r1.content[0].text == "Hello! How can I help you today?"

    r2 = await replay.call(_empty_request())
    assert r2.usage.input_tokens == 10
    assert r2.usage.output_tokens == 1

    r3 = await replay.call(_empty_request())
    assert r3.stop_reason is StopReason.TOOL_USE
    assert isinstance(r3.content[0], ToolUseBlock)
    assert r3.content[0].name == "echo"
    assert r3.content[0].input["text"] == "hi"


async def test_replay_exhaustion_raises_provider_error() -> None:
    replay = ReplayModelInterface.from_jsonl(
        _fixture_path().read_text(),
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    for _ in range(3):
        await replay.call(_empty_request())
    with pytest.raises(ProviderError) as exc:
        await replay.call(_empty_request())
    assert exc.value.code == 0


async def test_replay_streaming_emits_blocks_and_message_stop() -> None:
    jsonl = _fixture_path().read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    events: list[type] = []
    final_usage: TokenUsage | None = None
    async for ev in replay.call_streaming(_empty_request()):
        events.append(type(ev))
        if isinstance(ev, MessageStop):
            final_usage = ev.usage
    assert events[0] is MessageStart
    assert ContentBlockDelta in events
    assert ContentBlockStop in events
    assert events[-1] is MessageStop
    assert final_usage is not None
    assert final_usage.input_tokens == 8


async def test_replay_count_tokens_is_deterministic() -> None:
    replay = ReplayModelInterface([], ProviderInfo(name="x", model_id="y", context_window=100))
    req = ModelRequest(messages=[Message(role=Role.USER, content=TextContent(text="a" * 40))])
    assert await replay.count_tokens(req) == 10
