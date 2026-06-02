"""Tests for delta-level streaming through the harness (issue #103).

Mirrors the ``#103`` streaming tests in
``rust/crates/spore-core/src/harness.rs``. Drives a ``ModelAgent`` backed by a
``ReplayModelInterface`` (which synthesizes deltas from a recorded
``ModelResponse``) through the harness ReAct loop with a stream sink attached,
and asserts the emitted harness ``StreamEvent`` ordering.

The fixture-replay test reads the shared ground-truth fixtures
``fixtures/model_responses/harness/streaming_turn.jsonl`` and
``fixtures/harness/streaming_events.json``.

Known limitation (replicated, the golden fixture depends on it): model-layer
tool-argument deltas do NOT carry the tool name / id; the streamed coarse
``StreamToolCall`` therefore has an empty name and a synthesized ``call_{index}``
id, while the args round-trip faithfully.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    BlockKind,
    HarnessConfig,
    HarnessRunOptions,
    HarnessStreamEvent,
    LoopStrategyReAct,
    ModelAgent,
    ModelResponse,
    NoopContextManager,
    ProviderInfo,
    RecordedExchange,
    ReplayModelInterface,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    StopReason,
    StreamBlockStart,
    StreamBlockStop,
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolArgsDelta,
    StreamToolCall,
    StreamToolCallStart,
    StreamToolResult,
    Task,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolOutputSuccess,
    ToolUseBlock,
    TurnStreamState,
)
from spore_core.model import (
    ContentBlockStop,
    MessageStart,
    MessageStop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider() -> ProviderInfo:
    return ProviderInfo(name="anthropic", model_id="replay", context_window=200_000)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=3, output_tokens=4)


def _replay_agent(responses: list[ModelResponse]) -> ModelAgent:
    # Positional mode (no hashes): returns responses in call order regardless
    # of request content.
    from spore_core import ModelRequest

    exchanges = [
        RecordedExchange(request=ModelRequest(stream=True), response=r, provider="anthropic")
        for r in responses
    ]
    replay = ReplayModelInterface(exchanges, _provider())
    return ModelAgent(AgentId("stream-agent"), replay)


def _config(agent: ModelAgent, **overrides: Any) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )


def _react_task(max_iter: int = 5) -> Task:
    return Task.new("do something", SessionId("s1"), LoopStrategyReAct(max_iterations=max_iter))


def _collect() -> tuple[list[HarnessStreamEvent], Any]:
    events: list[HarnessStreamEvent] = []
    return events, events.append


def _index_of(events: list[HarnessStreamEvent], pred: Any) -> int | None:
    for i, e in enumerate(events):
        if pred(e):
            return i
    return None


def _precedes(events: list[HarnessStreamEvent], first: Any, second: Any) -> bool:
    fi = _index_of(events, first)
    si = _index_of(events, second)
    return fi is not None and si is not None and fi < si


def _block_index_for(events: list[HarnessStreamEvent], kind: BlockKind) -> int:
    for e in events:
        if isinstance(e, StreamBlockStart) and e.block == kind:
            return e.index
    raise AssertionError(f"no BlockStart for {kind}")


def _has_block_stop(events: list[HarnessStreamEvent], index: int) -> bool:
    return any(isinstance(e, StreamBlockStop) and e.index == index for e in events)


# ---------------------------------------------------------------------------
# TextDelta concatenation, ReasoningDelta, BlockStart/BlockStop bracketing,
# MessageStart/Stop dropped (Q3).
# ---------------------------------------------------------------------------


async def test_text_and_reasoning_deltas_bracketed_by_blocks() -> None:
    agent = _replay_agent(
        [
            ModelResponse(
                content=[ThinkingBlock(text="thinking aloud"), TextBlock(text="hello world")],
                usage=_usage(),
                stop_reason=StopReason.END_TURN,
            )
        ]
    )
    events, sink = _collect()
    h = StandardHarness(_config(agent))
    r = await h.run(HarnessRunOptions(_react_task(), on_stream=sink))
    assert isinstance(r, RunResultSuccess)

    # ReasoningDelta + TextDelta present with the right content.
    assert any(
        isinstance(e, StreamReasoningDelta) and e.content == "thinking aloud" for e in events
    )
    assert any(isinstance(e, StreamTextDelta) and e.content == "hello world" for e in events)

    # Q3: no empty deltas leaked from message-level markers.
    assert not any(isinstance(e, StreamTextDelta) and e.content == "" for e in events)

    # Every delta is bracketed: matching BlockStop, and BlockStart precedes its delta.
    reasoning_idx = _block_index_for(events, BlockKind.REASONING)
    text_idx = _block_index_for(events, BlockKind.TEXT)
    assert _has_block_stop(events, reasoning_idx)
    assert _has_block_stop(events, text_idx)
    assert _precedes(
        events,
        lambda e: isinstance(e, StreamBlockStart) and e.block == BlockKind.REASONING,
        lambda e: isinstance(e, StreamReasoningDelta),
    )


# ---------------------------------------------------------------------------
# Tool lifecycle ordering + enriched coarse events (Q5).
# ---------------------------------------------------------------------------


async def test_tool_lifecycle_ordering_and_enriched_coarse_events() -> None:
    agent = _replay_agent(
        [
            ModelResponse(
                content=[ToolUseBlock(id="toolu_1", name="lookup", input={"q": "rust"})],
                usage=_usage(),
                stop_reason=StopReason.TOOL_USE,
            ),
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=_usage(),
                stop_reason=StopReason.END_TURN,
            ),
        ]
    )
    reg = ScriptedToolRegistry()
    reg.push(ToolOutputSuccess(content="lookup result body"))
    events, sink = _collect()
    h = StandardHarness(_config(agent, tool_registry=reg))
    r = await h.run(HarnessRunOptions(_react_task(), on_stream=sink))
    assert isinstance(r, RunResultSuccess)
    assert r.output == "done"

    # ToolCallStart → ToolArgsDelta → coarse ToolCall.
    assert _precedes(
        events,
        lambda e: isinstance(e, StreamToolCallStart),
        lambda e: isinstance(e, StreamToolArgsDelta),
    )
    assert _precedes(
        events,
        lambda e: isinstance(e, StreamToolArgsDelta),
        lambda e: isinstance(e, StreamToolCall),
    )

    # ToolArgsDelta carries the args JSON.
    assert any(isinstance(e, StreamToolArgsDelta) and "rust" in e.partial_json for e in events)

    # Q5: coarse ToolCall carries the final accumulated args; the name is
    # recovered from the ToolUseStart model event every provider emits.
    coarse = next(e for e in events if isinstance(e, StreamToolCall))
    assert coarse.args == {"q": "rust"}
    assert coarse.name == "lookup"

    # Q5: coarse ToolResult carries the result content.
    result_ev = next(e for e in events if isinstance(e, StreamToolResult))
    assert result_ev.content == "lookup result body"

    # The ToolCallStart and ToolArgsDelta call_ids correlate.
    start_id = next(e.call_id for e in events if isinstance(e, StreamToolCallStart))
    delta_id = next(e.call_id for e in events if isinstance(e, StreamToolArgsDelta))
    assert start_id == delta_id == "toolu_1"


# ---------------------------------------------------------------------------
# Back-compat: no sink yields an identical RunResult to the streaming run.
# ---------------------------------------------------------------------------


async def test_no_sink_matches_non_streaming_baseline() -> None:
    def resp() -> ModelResponse:
        return ModelResponse(
            content=[TextBlock(text="identical")],
            usage=_usage(),
            stop_reason=StopReason.END_TURN,
        )

    events, sink = _collect()
    h1 = StandardHarness(_config(_replay_agent([resp()])))
    r_stream = await h1.run(HarnessRunOptions(_react_task(), on_stream=sink))

    h2 = StandardHarness(_config(_replay_agent([resp()])))
    r_plain = await h2.run(HarnessRunOptions(_react_task()))

    assert isinstance(r_stream, RunResultSuccess)
    assert isinstance(r_plain, RunResultSuccess)
    assert r_stream.output == r_plain.output
    assert r_stream.turns == r_plain.turns
    # The streaming run produced delta events; the plain run had no sink.
    assert events


# ---------------------------------------------------------------------------
# map_model_stream_event unit rules.
# ---------------------------------------------------------------------------


def test_message_markers_map_to_nothing() -> None:
    st = TurnStreamState()
    assert StandardHarness._map_model_stream_event(MessageStart(), st) == []
    assert (
        StandardHarness._map_model_stream_event(
            MessageStop(usage=TokenUsage(), stop_reason=StopReason.END_TURN), st
        )
        == []
    )


def test_block_start_emitted_once_per_index() -> None:
    from spore_core.model import ContentBlockDelta

    st = TurnStreamState()
    first = StandardHarness._map_model_stream_event(ContentBlockDelta(index=0, delta="a"), st)
    second = StandardHarness._map_model_stream_event(ContentBlockDelta(index=0, delta="b"), st)
    assert any(isinstance(e, StreamBlockStart) for e in first)
    assert not any(isinstance(e, StreamBlockStart) for e in second)
    assert all(isinstance(e, StreamTextDelta) for e in second)


def test_content_block_stop_maps_to_block_stop() -> None:
    st = TurnStreamState()
    out = StandardHarness._map_model_stream_event(ContentBlockStop(index=2), st)
    assert len(out) == 1
    assert isinstance(out[0], StreamBlockStop)
    assert out[0].index == 2


# ---------------------------------------------------------------------------
# Coarse-event round-trip: pre-#103 serialized events (without the new
# args / content fields) still deserialize (back-compat).
# ---------------------------------------------------------------------------


def test_coarse_events_roundtrip_without_new_fields() -> None:
    adapter = TypeAdapter(HarnessStreamEvent)
    back = adapter.validate_json('{"kind":"tool_call","call_id":"c1","name":"x"}')
    assert isinstance(back, StreamToolCall)
    assert back.args == {}

    back = adapter.validate_json('{"kind":"tool_result","call_id":"c1","is_error":false}')
    assert isinstance(back, StreamToolResult)
    assert back.content == ""


# ---------------------------------------------------------------------------
# Fixture-replay against the shared golden (issue #103).
# ---------------------------------------------------------------------------


def _fixtures_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "fixtures"
        if candidate.is_dir():
            return candidate
    raise AssertionError("fixtures/ directory not found")


async def test_fixture_replay_matches_golden_event_order() -> None:
    root = _fixtures_root()
    jsonl = (root / "model_responses" / "harness" / "streaming_turn.jsonl").read_text(
        encoding="utf-8"
    )
    replay = ReplayModelInterface.from_jsonl(jsonl, _provider())
    agent = ModelAgent(AgentId("fixture-agent"), replay)

    reg = ScriptedToolRegistry()
    # The recorded turn ends on tool_use; provide a tool result so dispatch
    # does not error. The run then attempts a 2nd turn and exhausts the
    # single-entry fixture — we only assert on the FIRST turn's delta events.
    reg.push(ToolOutputSuccess(content="result"))

    events, sink = _collect()
    h = StandardHarness(_config(agent, tool_registry=reg))
    await h.run(HarnessRunOptions(_react_task(), on_stream=sink))

    # Filter to delta / frame / tool_call_start events of the FIRST turn
    # (everything up to the first coarse ToolCall).
    cutoff = _index_of(events, lambda e: isinstance(e, StreamToolCall))
    if cutoff is None:
        cutoff = len(events)
    delta_kinds = (
        StreamBlockStart,
        StreamBlockStop,
        StreamTextDelta,
        StreamReasoningDelta,
        StreamToolArgsDelta,
        StreamToolCallStart,
    )
    delta_events = [e for e in events[:cutoff] if isinstance(e, delta_kinds)]

    golden = json.loads((root / "harness" / "streaming_events.json").read_text(encoding="utf-8"))
    adapter = TypeAdapter(HarnessStreamEvent)
    expected = [adapter.validate_python(ev) for ev in golden["events"]]

    assert len(delta_events) == len(expected), (
        f"event count mismatch:\n got: {delta_events}\n exp: {expected}"
    )
    for got, exp in zip(delta_events, expected, strict=True):
        assert got == exp, f"event mismatch: {got} != {exp}"
