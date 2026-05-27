"""Unit tests for LLM-native content capture — issue #64.

Covers every rule of the cross-language ground truth: ``truncate_field``
(within budget, byte-boundary clip + mark, multibyte back-off),
``ContentCaptureConfig`` env parsing (default OFF), ``GenAiRole`` → event-name
mapping, content on/off serialization, ``gen_ai.*`` attributes presence, and the
per-message GenAI span events the OTLP forwarder emits.

Mirrors the Rust reference in ``rust/crates/spore-core/src/observability.rs`` and
``observability_outbox.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core import (
    AllowAllSandbox,
    AlwaysContinuePolicy,
    FinalResponse,
    HarnessBuilder,
    HarnessRunOptions,
    LoopStrategyReAct,
    MockAgent,
    NoopContextManager,
    ScriptedToolRegistry,
    Task,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolOutputSuccess,
)
from spore_core.agent import AgentId
from spore_core.harness import SessionId, TaskId
from spore_core.memory import Timestamp
from spore_core.model import StopReason
from spore_core.observability import (
    TRUNCATION_MARKER,
    ContentCaptureConfig,
    GenAiMessage,
    GenAiRole,
    SpanBase,
    SpanId,
    SpanKind,
    SpanStatusOk,
    ToolCallContent,
    ToolCallSpan,
    ToolResultContent,
    TurnSpan,
    truncate_field,
)
from spore_core.observability_outbox import TraceLine, _genai_events


def _base(span_id: str, kind: SpanKind) -> SpanBase:
    return SpanBase(
        span_id=SpanId(span_id),
        parent_span_id=None,
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        kind=kind,
        started_at=Timestamp("2026-05-26T18:00:00.0Z"),
        ended_at=Timestamp("2026-05-26T18:00:02.1Z"),
        duration_ms=2100,
        status=SpanStatusOk(),
    )


# ── truncate_field ─────────────────────────────────────────────────────────


def test_truncate_within_budget_unchanged() -> None:
    assert truncate_field("hello", 8192) == ("hello", False)
    # Exactly at the budget is NOT truncated.
    assert truncate_field("abcd", 4) == ("abcd", False)


def test_truncate_clips_and_marks_ascii() -> None:
    out, truncated = truncate_field("abcdefghij", 4)
    assert truncated is True
    assert out == "abcd" + TRUNCATION_MARKER
    # The payload before the marker is bounded by the byte budget.
    assert out[: -len(TRUNCATION_MARKER)].encode("utf-8") == b"abcd"


def test_truncate_byte_measured_not_char_measured() -> None:
    # "é" is 2 UTF-8 bytes. Six é = 12 bytes; budget 5 bytes → 2 full é (4 bytes)
    # fit, the 3rd would split, so it backs off to a boundary at byte 4.
    s = "é" * 6
    out, truncated = truncate_field(s, 5)
    assert truncated is True
    assert out == "éé" + TRUNCATION_MARKER


def test_truncate_multibyte_back_off_never_splits() -> None:
    # Budget lands mid-multibyte-char: a 4-byte emoji. Budget 2 forces a back-off
    # to byte 0 (no whole char fits) → empty payload + marker.
    out, truncated = truncate_field("😀abc", 2)
    assert truncated is True
    assert out == TRUNCATION_MARKER
    # And a budget that fits exactly the emoji (4 bytes) keeps it whole.
    out2, trunc2 = truncate_field("😀abc", 4)
    assert trunc2 is True
    assert out2 == "😀" + TRUNCATION_MARKER


# ── ContentCaptureConfig ────────────────────────────────────────────────────


def test_config_default_off() -> None:
    cfg = ContentCaptureConfig()
    assert cfg.enabled is False
    assert cfg.max_field_len == 8192


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", " On "])
def test_config_from_env_enabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SPORE_TRACE_CONTENT", value)
    monkeypatch.delenv("SPORE_TRACE_CONTENT_MAX_LEN", raising=False)
    assert ContentCaptureConfig.from_env().enabled is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "nonsense"])
def test_config_from_env_disabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SPORE_TRACE_CONTENT", value)
    assert ContentCaptureConfig.from_env().enabled is False


def test_config_from_env_unset_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPORE_TRACE_CONTENT", raising=False)
    cfg = ContentCaptureConfig.from_env()
    assert cfg.enabled is False
    assert cfg.max_field_len == 8192


def test_config_from_env_max_len_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPORE_TRACE_CONTENT", "1")
    monkeypatch.setenv("SPORE_TRACE_CONTENT_MAX_LEN", "256")
    assert ContentCaptureConfig.from_env().max_field_len == 256


def test_config_from_env_max_len_unparseable_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPORE_TRACE_CONTENT", "1")
    monkeypatch.setenv("SPORE_TRACE_CONTENT_MAX_LEN", "lots")
    assert ContentCaptureConfig.from_env().max_field_len == 8192


# ── GenAiRole → event name ──────────────────────────────────────────────────


def test_role_event_names() -> None:
    assert GenAiRole.SYSTEM.event_name() == "gen_ai.system.message"
    assert GenAiRole.USER.event_name() == "gen_ai.user.message"
    assert GenAiRole.ASSISTANT.event_name() == "gen_ai.assistant.message"
    assert GenAiRole.TOOL.event_name() == "gen_ai.tool.message"


def test_role_bare_string_values() -> None:
    assert GenAiRole.ASSISTANT.value == "assistant"
    assert GenAiRole.TOOL.value == "tool"


# ── content on/off serialization ────────────────────────────────────────────


def test_turn_content_off_has_no_genai_keys() -> None:
    span = TurnSpan(
        base=_base("b1", SpanKind.TURN),
        turn_number=1,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=0.0,
        stop_reason=StopReason.END_TURN,
        tool_calls_requested=0,
    )
    attrs = TraceLine.from_turn(span, "tid").attributes
    assert not any(k.startswith("gen_ai.") for k in attrs)


def test_turn_content_on_emits_output_and_tool_calls() -> None:
    span = TurnSpan(
        base=_base("b1", SpanKind.TURN),
        turn_number=1,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=0.0,
        stop_reason=StopReason.TOOL_USE,
        tool_calls_requested=1,
        output_text=GenAiMessage(role=GenAiRole.ASSISTANT, content="hi", truncated=False),
        tool_calls=[
            ToolCallContent(name="shell", arguments={"command": "ls"}, arguments_truncated=False)
        ],
    )
    attrs = TraceLine.from_turn(span, "tid").attributes
    assert attrs["gen_ai.response.role"] == "assistant"
    assert attrs["gen_ai.response.content"] == "hi"
    assert attrs["gen_ai.response.content_truncated"] is False
    assert attrs["gen_ai.response.tool_calls"] == [
        {"name": "shell", "arguments": {"command": "ls"}, "arguments_truncated": False}
    ]


def test_tool_call_content_on_emits_args_and_result() -> None:
    span = ToolCallSpan(
        base=_base("tc1", SpanKind.TOOL_CALL),
        tool_name="shell",
        call_id="c1",
        parameters_size_bytes=10,
        output_size_bytes=20,
        truncated=False,
        sandbox_mode="",
        sandbox_violations=[],
        arguments=ToolCallContent(
            name="shell", arguments={"command": "ls"}, arguments_truncated=False
        ),
        result=ToolResultContent(content="ok", truncated=False),
    )
    attrs = TraceLine.from_tool_call(span, "tid").attributes
    assert attrs["gen_ai.tool.name"] == "shell"
    assert attrs["gen_ai.tool.call.arguments"] == {"command": "ls"}
    assert attrs["gen_ai.tool.call.arguments_truncated"] is False
    assert attrs["gen_ai.tool.message.content"] == "ok"
    assert attrs["gen_ai.tool.message.content_truncated"] is False


def test_tool_call_content_off_has_no_genai_keys() -> None:
    span = ToolCallSpan(
        base=_base("tc1", SpanKind.TOOL_CALL),
        tool_name="shell",
        call_id="c1",
        parameters_size_bytes=10,
        output_size_bytes=20,
        truncated=False,
        sandbox_mode="",
        sandbox_violations=[],
    )
    attrs = TraceLine.from_tool_call(span, "tid").attributes
    assert not any(k.startswith("gen_ai.") for k in attrs)


def test_truncated_arguments_stored_as_json_string() -> None:
    # When clipped, ToolCallContent.arguments is a JSON string carrying the marker.
    span = TurnSpan(
        base=_base("b1", SpanKind.TURN),
        turn_number=1,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=0.0,
        stop_reason=StopReason.TOOL_USE,
        tool_calls_requested=1,
        tool_calls=[
            ToolCallContent(
                name="shell",
                arguments='{"command":"ls -la /ver...[truncated]',
                arguments_truncated=True,
            )
        ],
    )
    attrs = TraceLine.from_turn(span, "tid").attributes
    call = attrs["gen_ai.response.tool_calls"][0]
    assert call["arguments_truncated"] is True
    assert isinstance(call["arguments"], str)
    assert call["arguments"].endswith(TRUNCATION_MARKER)


# ── GenAI span events (per message) ─────────────────────────────────────────


def test_genai_events_empty_when_content_off() -> None:
    span = TurnSpan(
        base=_base("b1", SpanKind.TURN),
        turn_number=1,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=0.0,
        stop_reason=StopReason.END_TURN,
        tool_calls_requested=0,
    )
    line = TraceLine.from_turn(span, "tid")
    assert _genai_events(line) == []


def test_genai_events_turn_output_and_tool_calls() -> None:
    span = TurnSpan(
        base=_base("b1", SpanKind.TURN),
        turn_number=1,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=None,
        cache_write_tokens=None,
        cost_usd=0.0,
        stop_reason=StopReason.TOOL_USE,
        tool_calls_requested=1,
        output_text=GenAiMessage(role=GenAiRole.ASSISTANT, content="thinking", truncated=False),
        tool_calls=[
            ToolCallContent(name="shell", arguments={"command": "ls"}, arguments_truncated=False)
        ],
    )
    events = _genai_events(TraceLine.from_turn(span, "tid"))
    # One assistant.message for the output text, one per tool call.
    assert events[0][0] == "gen_ai.assistant.message"
    assert events[0][1]["gen_ai.message.content"] == "thinking"
    assert events[1][0] == "gen_ai.assistant.message"
    assert events[1][1]["gen_ai.tool.name"] == "shell"
    assert events[1][1]["gen_ai.tool.call.arguments"] == '{"command":"ls"}'


def test_genai_events_tool_result_message() -> None:
    span = ToolCallSpan(
        base=_base("tc1", SpanKind.TOOL_CALL),
        tool_name="shell",
        call_id="c1",
        parameters_size_bytes=10,
        output_size_bytes=20,
        truncated=False,
        sandbox_mode="",
        sandbox_violations=[],
        arguments=ToolCallContent(
            name="shell", arguments={"command": "ls"}, arguments_truncated=False
        ),
        result=ToolResultContent(content="total 0", truncated=False),
    )
    events = _genai_events(TraceLine.from_tool_call(span, "tid"))
    tool_events = [e for e in events if e[0] == "gen_ai.tool.message"]
    assert len(tool_events) == 1
    assert tool_events[0][1]["gen_ai.message.role"] == "tool"
    assert tool_events[0][1]["gen_ai.message.content"] == "total 0"


# ── harness e2e: content threaded through the builder + emission sites ───────


def _read_lines(root: Path, session: str) -> list[dict]:
    path = root / "sessions" / session / "trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def test_harness_content_on_writes_genai_content_to_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPORE_OTLP_ENDPOINT", raising=False)
    agent = MockAgent(AgentId("test"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name="shell", input={"command": "ls -la"})],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    agent.push(FinalResponse(content="all done", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="total 0"))
    harness = (
        HarnessBuilder(
            agent,
            reg,
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .with_observability_outbox(tmp_path)
        .content_capture(ContentCaptureConfig(enabled=True))
        .build()
    )
    task = Task.new("do it", SessionId("s1"), LoopStrategyReAct(max_iterations=5))
    await harness.run(HarnessRunOptions(task))

    lines = _read_lines(tmp_path, "s1")
    turn_lines = [line for line in lines if line["kind"] == "turn"]
    tool_lines = [line for line in lines if line["kind"] == "tool_call"]
    # The tool-requesting turn carries the requested tool calls.
    assert any(
        "gen_ai.response.tool_calls" in line["attributes"] for line in turn_lines
    )
    # The final-response turn carries the assistant output text.
    assert any(
        line["attributes"].get("gen_ai.response.content") == "all done" for line in turn_lines
    )
    # The tool-call span carries args + result.
    assert tool_lines
    tc = tool_lines[0]["attributes"]
    assert tc["gen_ai.tool.name"] == "shell"
    assert tc["gen_ai.tool.call.arguments"] == {"command": "ls -la"}
    assert tc["gen_ai.tool.message.content"] == "total 0"


async def test_harness_content_off_writes_no_genai_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPORE_OTLP_ENDPOINT", raising=False)
    agent = MockAgent(AgentId("test"))
    agent.push(
        ToolCallRequested(
            calls=[ToolCall(id="c1", name="shell", input={"command": "ls -la"})],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
        )
    )
    agent.push(FinalResponse(content="all done", usage=TokenUsage(input_tokens=1, output_tokens=1)))
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="total 0"))
    # Default builder → content capture OFF.
    harness = (
        HarnessBuilder(
            agent,
            reg,
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .with_observability_outbox(tmp_path)
        .build()
    )
    task = Task.new("do it", SessionId("s1"), LoopStrategyReAct(max_iterations=5))
    await harness.run(HarnessRunOptions(task))

    lines = _read_lines(tmp_path, "s1")
    for line in lines:
        assert not any(k.startswith("gen_ai.") for k in line["attributes"]), (
            f"content leaked into {line['kind']} with guard OFF"
        )
