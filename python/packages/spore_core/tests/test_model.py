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
    RecordedExchange,
    RecordingMode,
    RecordingModelInterface,
    ReplayMode,
    ReplayModelInterface,
    Role,
    SporeError,
    StopReason,
    StreamInterrupted,
    TextBlock,
    TextContent,
    TimeoutError,
    TokenUsage,
    ToolCall,
    ToolSchema,
    ToolUseBlock,
    Transport,
    enforce_budget,
    enforce_context_limit,
    request_hash,
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
        Transport("conn refused"),
        StreamInterrupted("stream chunk error: eof"),
    ]
    for e in errs:
        assert str(e)
        assert e.kind  # class attribute set on every subclass


# ---------------------------------------------------------------------------
# SC-3: typed retryable model errors
# ---------------------------------------------------------------------------


def test_retryable_classifies_transient_vs_deterministic() -> None:
    """SC-3: transport drops, mid-stream interruptions, timeouts and rate
    limits are transient; provider/context/budget errors are deterministic.
    Mirrors Rust's ``retryable_classifies_transient_vs_deterministic``."""
    assert Transport("x").retryable()
    assert StreamInterrupted("x").retryable()
    assert TimeoutError().retryable()
    assert RateLimited(retry_after=None).retryable()
    assert not ProviderError(code=0, message="x").retryable()
    assert not ContextLimitExceeded(limit=1, actual=2).retryable()
    assert not BudgetExceeded(budget=1, used=2).retryable()


def test_typed_transport_errors_serde_tag_shape() -> None:
    """The two SC-3 variants serialize to the ``kind``-tagged shape the
    cross-language ports mirror byte-for-byte:
    ``{"kind": "Transport", "message": ...}`` /
    ``{"kind": "StreamInterrupted", "message": ...}``."""
    assert Transport("m").to_dict() == {"kind": "Transport", "message": "m"}
    assert StreamInterrupted("m").to_dict() == {
        "kind": "StreamInterrupted",
        "message": "m",
    }
    # The raw `message` field is preserved verbatim (not the display string).
    assert Transport("m").message == "m"
    assert Transport("m").kind == "Transport"
    assert StreamInterrupted("m").message == "m"
    assert StreamInterrupted("m").kind == "StreamInterrupted"
    # The serialized form round-trips back to an equivalent typed error.
    for cls, kind in ((Transport, "Transport"), (StreamInterrupted, "StreamInterrupted")):
        d = cls("boom").to_dict()
        assert d["kind"] == kind
        back = cls(d["message"])
        assert back.to_dict() == d


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


async def test_replay_count_tokens_uses_recorded_input_tokens_in_hash_mode() -> None:
    # Carry-over from #39: replace the bytes/4 heuristic with the recorded
    # input_tokens when we can match by request_hash.
    q = ModelRequest(
        messages=[Message(role=Role.USER, content=TextContent(text="the quick brown fox"))]
    )
    recorded = RecordedExchange(
        request_hash=request_hash(q),
        request=q,
        response=ModelResponse(
            content=[TextBlock(text="ok")],
            usage=TokenUsage(input_tokens=137, output_tokens=4),
            stop_reason=StopReason.END_TURN,
        ),
        provider="anthropic",
    )
    r = ReplayModelInterface([recorded], _provider())
    assert r.mode() is ReplayMode.HASH_MATCHED
    n = await r.count_tokens(q)
    # 137 came from the recorded usage. bytes/4 would have produced
    # floor(19/4) = 4, so this proves the recorded value wins.
    assert n == 137


async def test_replay_count_tokens_falls_back_to_heuristic_when_no_match() -> None:
    q1 = _req_text("xx")
    recorded = RecordedExchange(
        request_hash=request_hash(q1),
        request=q1,
        response=_resp_text("r"),
        provider="fixture",
    )
    r = ReplayModelInterface([recorded], _provider())
    unrecorded = _req_text("never seen before, a long string indeed")
    n = await r.count_tokens(unrecorded)
    # Length 39 → floor(39/4) = 9.
    assert n == 9


# ---------------------------------------------------------------------------
# #37: request_hash + ReplayMode
# ---------------------------------------------------------------------------


def _req_text(s: str) -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=TextContent(text=s))])


def _resp_text(s: str) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=s)],
        usage=TokenUsage(),
        stop_reason=StopReason.END_TURN,
    )


def test_request_hash_is_stable() -> None:
    assert request_hash(_req_text("hello world")) == request_hash(_req_text("hello world"))


def test_request_hash_changes_when_messages_change() -> None:
    assert request_hash(_req_text("hello")) != request_hash(_req_text("hello!"))


def test_request_hash_is_16_hex_chars() -> None:
    h = request_hash(_req_text("x"))
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


async def test_replay_auto_detects_positional_when_no_hashes() -> None:
    exchanges = [
        RecordedExchange(
            request=_req_text("q1"),
            response=_resp_text("r1"),
            provider="fixture",
        )
    ]
    r = ReplayModelInterface(exchanges, _provider())
    assert r.mode() is ReplayMode.POSITIONAL
    got = await r.call(_req_text("any"))
    assert got == _resp_text("r1")


async def test_replay_auto_detects_hash_matched_when_all_have_hashes() -> None:
    q1, q2 = _req_text("q1"), _req_text("q2")
    exchanges = [
        RecordedExchange(
            request_hash=request_hash(q1),
            request=q1,
            response=_resp_text("r1"),
            provider="fixture",
        ),
        RecordedExchange(
            request_hash=request_hash(q2),
            request=q2,
            response=_resp_text("r2"),
            provider="fixture",
        ),
    ]
    r = ReplayModelInterface(exchanges, _provider())
    assert r.mode() is ReplayMode.HASH_MATCHED
    # Out-of-order calls return the right response.
    assert await r.call(q2) == _resp_text("r2")
    assert await r.call(q1) == _resp_text("r1")
    assert await r.call(q2) == _resp_text("r2")


async def test_replay_hash_matched_no_match_returns_provider_error() -> None:
    q1 = _req_text("q1")
    exchanges = [
        RecordedExchange(
            request_hash=request_hash(q1),
            request=q1,
            response=_resp_text("r1"),
            provider="fixture",
        )
    ]
    r = ReplayModelInterface(exchanges, _provider())
    with pytest.raises(ProviderError) as exc:
        await r.call(_req_text("unrecorded"))
    assert "no matching fixture" in exc.value.message


# ---------------------------------------------------------------------------
# #37: cross-language hash stability fixture
# ---------------------------------------------------------------------------


def test_fixture_request_hash_stability() -> None:
    import json

    path = _repo_root() / "fixtures/model_hashing/cases.json"
    suite = json.loads(path.read_text())
    for case in suite["cases"]:
        req = ModelRequest.model_validate(case["request"])
        got = request_hash(req)
        assert got == case["expected_hash"], (
            f"case `{case['name']}`: hash mismatch (got {got}, expected {case['expected_hash']})"
        )


# ---------------------------------------------------------------------------
# #38: RecordingModelInterface
# ---------------------------------------------------------------------------


async def test_recording_appends_request_response_pair(tmp_path) -> None:
    path = tmp_path / "recorded.jsonl"
    inner = MockModelInterface(_provider())
    inner.push_response(_resp_text("hello back")).push_response(_resp_text("hello again"))
    r = RecordingModelInterface(inner, path, RecordingMode.RECORD)
    await r.call(_req_text("hello"))
    await r.call(_req_text("hello2"))
    raw = path.read_text()
    lines = [ln for ln in raw.splitlines() if ln]
    assert len(lines) == 2
    for line in lines:
        entry = RecordedExchange.model_validate_json(line)
        assert entry.request_hash is not None
        assert entry.provider == "test"
        assert entry.model_id == "test-1"


async def test_recording_record_if_new_skips_when_file_exists(tmp_path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text("preexisting line\n")
    inner = MockModelInterface(_provider())
    inner.push_response(_resp_text("ok"))
    r = RecordingModelInterface(inner, path, RecordingMode.RECORD_IF_NEW)
    await r.call(_req_text("q"))
    assert path.read_text() == "preexisting line\n"


async def test_recording_record_if_new_writes_when_file_absent(tmp_path) -> None:
    path = tmp_path / "new.jsonl"
    inner = MockModelInterface(_provider())
    inner.push_response(_resp_text("ok"))
    r = RecordingModelInterface(inner, path, RecordingMode.RECORD_IF_NEW)
    await r.call(_req_text("q"))
    assert path.exists()
    assert len([ln for ln in path.read_text().splitlines() if ln]) == 1


async def test_recording_passthrough_does_not_write(tmp_path) -> None:
    path = tmp_path / "nope.jsonl"
    inner = MockModelInterface(_provider())
    inner.push_response(_resp_text("ok"))
    r = RecordingModelInterface(inner, path, RecordingMode.PASSTHROUGH)
    await r.call(_req_text("q"))
    assert not path.exists()


async def test_recording_then_replay_roundtrip(tmp_path) -> None:
    path = tmp_path / "roundtrip.jsonl"
    inner = MockModelInterface(_provider())
    inner.push_response(_resp_text("answer1")).push_response(_resp_text("answer2"))
    recorder = RecordingModelInterface(inner, path, RecordingMode.RECORD)
    q1, q2 = _req_text("question 1"), _req_text("question 2")
    await recorder.call(q1)
    await recorder.call(q2)
    jsonl = path.read_text()
    replay = ReplayModelInterface.from_jsonl(jsonl, _provider())
    assert replay.mode() is ReplayMode.HASH_MATCHED
    # Replay out-of-order to confirm hash matching end-to-end.
    assert await replay.call(q2) == _resp_text("answer2")
    assert await replay.call(q1) == _resp_text("answer1")
