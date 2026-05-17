"""Tests for :class:`InMemoryObservabilityProvider` — issue #12.

Mirrors the Rust unit tests in
``rust/crates/spore-core/src/observability.rs`` ``tests`` module.
"""

from __future__ import annotations

import json
from pathlib import Path

from spore_core.guide_registry import (
    SessionOutcomeFailure,
    SessionOutcomePartial,
    SessionOutcomeSuccess,
)
from spore_core.harness import SessionId, TaskId
from spore_core.memory import Timestamp
from spore_core.middleware import HookPoint, MiddlewareContinue
from spore_core.model import StopReason
from spore_core.observability import (
    ContextOperationAssembly,
    ContextOperationCompaction,
    ContextSpan,
    InMemoryObservabilityProvider,
    MiddlewareSpan,
    PricingTable,
    SensorSpan,
    SpanBase,
    SpanId,
    SpanKind,
    SpanStatusOk,
    ToolCallSpan,
    TurnSpan,
)
from spore_core.sensor import (
    SensorId,
    SensorKind,
    SensorOutcome,
    SensorTriggerPostTurn,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _ts(s: str) -> Timestamp:
    return Timestamp(s)


def _sid(s: str) -> SessionId:
    return SessionId(s)


def _tid(s: str) -> TaskId:
    return TaskId(s)


def _turn_span(
    session: str,
    span_id: str,
    turn: int,
    inp: int,
    out: int,
    started_at: str = "2026-05-16T00:00:00Z",
    duration_ms: int = 1000,
) -> TurnSpan:
    return TurnSpan(
        base=SpanBase(
            span_id=SpanId(span_id),
            session_id=_sid(session),
            task_id=_tid("t1"),
            kind=SpanKind.TURN,
            started_at=_ts(started_at),
            ended_at=_ts("2026-05-16T00:00:01Z"),
            duration_ms=duration_ms,
            status=SpanStatusOk(),
        ),
        turn_number=turn,
        input_tokens=inp,
        output_tokens=out,
        cost_usd=0.0,
        stop_reason=StopReason.END_TURN,
        tool_calls_requested=0,
    )


# ── Rule: emit_turn is fire-and-forget; metrics aggregate ──────────────────


async def test_emit_turn_recorded_and_metrics_aggregate() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "sp1", 1, 100, 50))
    obs.emit_turn(_turn_span("s1", "sp2", 2, 200, 80))
    obs.set_session_outcome(_sid("s1"), SessionOutcomeSuccess())
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert m.total_turns == 2
    assert m.total_input_tokens == 300
    assert m.total_output_tokens == 130
    assert isinstance(m.outcome, SessionOutcomeSuccess)


# ── Rule: emit_tool_call counted in metrics ────────────────────────────────


async def test_emit_tool_call_counted_in_metrics() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "t1", 1, 10, 5))
    obs.emit_tool_call(
        ToolCallSpan(
            base=SpanBase(
                span_id=SpanId("tc1"),
                session_id=_sid("s1"),
                task_id=_tid("t1"),
                kind=SpanKind.TOOL_CALL,
                started_at=_ts("2026-05-16T00:00:00Z"),
                ended_at=_ts("2026-05-16T00:00:00Z"),
                duration_ms=250,
                status=SpanStatusOk(),
            ),
            tool_name="shell",
            call_id="c1",
            parameters_size_bytes=12,
            output_size_bytes=42,
            truncated=False,
            sandbox_mode="workspace_scoped",
        )
    )
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert m.tool_calls == 1
    assert m.total_duration_ms == 1250


# ── Rule: sensor metrics — fires and halts ─────────────────────────────────


async def test_sensor_metrics_count_fires_and_halts() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "t1", 1, 10, 5))

    def mk(span_id: str, fired: bool, outcome: SensorOutcome) -> SensorSpan:
        return SensorSpan(
            base=SpanBase(
                span_id=SpanId(span_id),
                session_id=_sid("s1"),
                task_id=_tid("t1"),
                kind=SpanKind.SENSOR_EVALUATION,
                started_at=_ts("2026-05-16T00:00:00Z"),
                ended_at=_ts("2026-05-16T00:00:00Z"),
                duration_ms=1,
                status=SpanStatusOk(),
            ),
            sensor_id=SensorId("lint"),
            sensor_kind=SensorKind.COMPUTATIONAL,
            trigger=SensorTriggerPostTurn(),
            outcome=outcome,
            fired=fired,
        )

    obs.emit_sensor(mk("sn1", True, SensorOutcome.WARN))
    obs.emit_sensor(mk("sn2", True, SensorOutcome.HALT))
    obs.emit_sensor(mk("sn3", False, SensorOutcome.PASS))
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert m.sensor_fires == 2
    assert m.sensor_halts == 1


# ── Rule: compaction counted; assembly not ─────────────────────────────────


async def test_compaction_counted_in_metrics() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "t1", 1, 100, 50))

    def mk_ctx(op_kind: str) -> ContextSpan:
        if op_kind == "compaction":
            op = ContextOperationCompaction(messages_removed=5, tokens_reclaimed=5000)
        else:
            op = ContextOperationAssembly(guides_loaded=2, memory_items_loaded=3, tools_loaded=5)
        return ContextSpan(
            base=SpanBase(
                span_id=SpanId(f"c-{op_kind}"),
                session_id=_sid("s1"),
                task_id=_tid("t1"),
                kind=SpanKind.COMPACTION if op_kind == "compaction" else SpanKind.CONTEXT_ASSEMBLY,
                started_at=_ts("2026-05-16T00:00:00Z"),
                ended_at=_ts("2026-05-16T00:00:00Z"),
                duration_ms=1,
                status=SpanStatusOk(),
            ),
            operation=op,
            tokens_before=10_000,
            tokens_after=5_000,
            utilization_before=0.9,
            utilization_after=0.5,
        )

    obs.emit_context(mk_ctx("compaction"))
    obs.emit_context(mk_ctx("assembly"))
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert m.compactions == 1


# ── Rule: PricingTable computes cost_usd ───────────────────────────────────


def test_pricing_table_computes_cost() -> None:
    table = PricingTable(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.3,
        cache_write_per_million=3.75,
    )
    cost = table.cost_for(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    # 3 + 15 + 0.3 + 3.75 = 22.05
    assert abs(cost - 22.05) < 1e-9


def test_pricing_table_default_is_zero() -> None:
    cost = PricingTable.DEFAULT.cost_for(1_000, 1_000, 1_000, 1_000)
    assert cost == 0.0


# ── Rule: flush_session is idempotent ──────────────────────────────────────


async def test_flush_session_is_idempotent() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "t1", 1, 10, 5))
    await obs.flush_session(_sid("s1"))
    await obs.flush_session(_sid("s1"))  # second flush is a no-op
    # Spans remain queryable after flush.
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert m.total_turns == 1


# ── Rule: get_trace returns spans in insertion order ───────────────────────


async def test_get_trace_preserves_insertion_order() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "sp1", 1, 10, 5))
    obs.emit_tool_call(
        ToolCallSpan(
            base=SpanBase(
                span_id=SpanId("sp2"),
                session_id=_sid("s1"),
                task_id=_tid("t1"),
                kind=SpanKind.TOOL_CALL,
                started_at=_ts("2026-05-16T00:00:00Z"),
                ended_at=_ts("2026-05-16T00:00:00Z"),
                duration_ms=1,
                status=SpanStatusOk(),
                parent_span_id=SpanId("sp1"),
            ),
            tool_name="shell",
            call_id="c1",
            parameters_size_bytes=0,
            output_size_bytes=0,
            truncated=False,
            sandbox_mode="none",
        )
    )
    trace = await obs.get_trace(_sid("s1"))
    assert len(trace) == 2
    assert trace[0].base.span_id == "sp1"
    assert trace[1].base.span_id == "sp2"
    # Parent linkage preserved.
    assert trace[1].base.parent_span_id == "sp1"


# ── Rule: middleware spans recorded ────────────────────────────────────────


async def test_middleware_span_recorded_in_trace() -> None:
    obs = InMemoryObservabilityProvider()
    span = MiddlewareSpan(
        base=SpanBase(
            span_id=SpanId("mw1"),
            session_id=_sid("s1"),
            task_id=_tid("t1"),
            kind=SpanKind.MIDDLEWARE_HOOK,
            started_at=_ts("2026-05-16T00:00:00Z"),
            ended_at=_ts("2026-05-16T00:00:00Z"),
            duration_ms=0,
            status=SpanStatusOk(),
        ),
        hook=HookPoint.BEFORE_TURN,
        decision=MiddlewareContinue(),
    )
    obs.emit_middleware(span)
    trace = await obs.get_trace(_sid("s1"))
    assert len(trace) == 1
    assert trace[0].base.kind is SpanKind.MIDDLEWARE_HOOK


# ── Rule: get_sessions filters by outcome ──────────────────────────────────


async def test_get_sessions_filters_by_outcome() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("good", "sp1", 1, 10, 5))
    obs.emit_turn(_turn_span("bad", "sp2", 1, 10, 5))
    obs.set_session_outcome(_sid("good"), SessionOutcomeSuccess())
    obs.set_session_outcome(_sid("bad"), SessionOutcomeFailure(reason="x"))
    success_only = await obs.get_sessions(
        _ts("2026-05-16T00:00:00Z"), outcome=SessionOutcomeSuccess()
    )
    assert len(success_only) == 1
    assert success_only[0].session_id == "good"


# ── Rule: get_sessions filters by since timestamp ──────────────────────────


async def test_get_sessions_filters_by_since() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("old", "sp1", 1, 10, 5, started_at="2026-01-01T00:00:00Z"))
    obs.emit_turn(_turn_span("new", "sp2", 1, 10, 5))
    recent = await obs.get_sessions(_ts("2026-05-15T00:00:00Z"))
    ids = [m.session_id for m in recent]
    assert "new" in ids
    assert "old" not in ids


# ── Rule: outcome defaults to Partial when unset ───────────────────────────


async def test_session_metrics_defaults_outcome_partial() -> None:
    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "sp1", 1, 1, 1))
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert isinstance(m.outcome, SessionOutcomePartial)


# ── Rule: SpanBase root/child helpers ──────────────────────────────────────


def test_span_base_new_root_and_child() -> None:
    root = SpanBase.new_root(
        SpanId("r"),
        _sid("s"),
        _tid("t"),
        SpanKind.SESSION,
        _ts("2026-05-16T00:00:00Z"),
    )
    child = SpanBase.new_child(
        SpanId("c"),
        root,
        SpanKind.TURN,
        _ts("2026-05-16T00:00:01Z"),
    )
    assert child.parent_span_id == "r"
    assert child.session_id == "s"


def test_span_base_finish_updates_fields() -> None:
    sb = SpanBase.new_root(
        SpanId("a"),
        _sid("s"),
        _tid("t"),
        SpanKind.TURN,
        _ts("2026-05-16T00:00:00Z"),
    )
    sb.finish(_ts("2026-05-16T00:00:02Z"), SpanStatusOk(), 2000)
    assert sb.duration_ms == 2000
    assert sb.ended_at == "2026-05-16T00:00:02Z"


# ── Rule: get_session_metrics returns None for unknown session ─────────────


async def test_get_session_metrics_returns_none_for_unknown_session() -> None:
    obs = InMemoryObservabilityProvider()
    assert await obs.get_session_metrics(_sid("missing")) is None


# ── Rule: guides_used surfaced in metrics ──────────────────────────────────


async def test_guides_used_surfaced_in_metrics() -> None:
    from spore_core.guide_registry import GuideId

    obs = InMemoryObservabilityProvider()
    obs.emit_turn(_turn_span("s1", "t1", 1, 1, 1))
    obs.record_guides_used(_sid("s1"), [GuideId("g1"), GuideId("g2")])
    m = await obs.get_session_metrics(_sid("s1"))
    assert m is not None
    assert m.guides_used == ["g1", "g2"]


# ── Fixture replay ─────────────────────────────────────────────────────────


async def test_fixture_replay_session_metrics() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "fixtures"
        / "observability"
        / "session_metrics_basic.json"
    )
    case = json.loads(fixture_path.read_text())

    obs = InMemoryObservabilityProvider()
    for t in case["turns"]:
        obs.emit_turn(
            _turn_span(
                case["session_id"],
                t["span_id"],
                t["turn"],
                t["input"],
                t["output"],
            )
        )
    outcome_str = case["outcome"]
    if outcome_str == "success":
        outcome = SessionOutcomeSuccess()
    elif outcome_str == "partial":
        outcome = SessionOutcomePartial()
    else:
        outcome = SessionOutcomeFailure(reason=outcome_str)
    obs.set_session_outcome(_sid(case["session_id"]), outcome)

    m = await obs.get_session_metrics(_sid(case["session_id"]))
    assert m is not None
    expected = case["expected"]
    assert m.total_turns == expected["total_turns"]
    assert m.total_input_tokens == expected["total_input_tokens"]
    assert m.total_output_tokens == expected["total_output_tokens"]
