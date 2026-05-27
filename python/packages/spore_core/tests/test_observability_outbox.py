"""Tests for :class:`OutboxObservabilityProvider` — issue #33.

Mirrors the Rust unit + fixture-replay tests in
``rust/crates/spore-core/src/observability_outbox.rs``. Fully hermetic: uses
``tmp_path`` and never touches a live OTLP / network stack
(``SPORE_OTLP_ENDPOINT`` is unset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spore_core.guide_registry import SessionOutcomeSuccess
from spore_core.harness import SessionId, TaskId
from spore_core.memory import Timestamp
from spore_core.middleware import HookPoint, MiddlewareContinue
from spore_core.model import StopReason
from spore_core.observability import (
    ContextOperationAssembly,
    ContextOperationCompaction,
    ContextSpan,
    MiddlewareSpan,
    PatchSpan,
    PatchTypeParameterCoercion,
    SensorSpan,
    SessionMetrics,
    SpanBase,
    SpanId,
    SpanKind,
    SpanStatusError,
    SpanStatusHalted,
    SpanStatusOk,
    ToolCallSpan,
    TurnSpan,
)
from spore_core.observability_outbox import (
    OutboxConfig,
    OutboxObservabilityProvider,
    SessionNotFound,
    TraceLine,
)
from spore_core.sensor import (
    SensorId,
    SensorKind,
    SensorOutcome,
    SensorTriggerPostTool,
)

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures" / "observability"


# ── helpers ──────────────────────────────────────────────────────────────────


def _ts(s: str) -> Timestamp:
    return Timestamp(s)


def _sid(s: str) -> SessionId:
    return SessionId(s)


def _base(
    session: str,
    span_id: str,
    kind: SpanKind,
    status: Any = None,
) -> SpanBase:
    return SpanBase(
        span_id=SpanId(span_id),
        session_id=_sid(session),
        task_id=TaskId("task1"),
        kind=kind,
        started_at=_ts("2026-05-26T18:00:00.000Z"),
        ended_at=_ts("2026-05-26T18:00:02.100Z"),
        duration_ms=2100,
        status=status if status is not None else SpanStatusOk(),
    )


def _turn(session: str, span_id: str) -> TurnSpan:
    return TurnSpan(
        base=_base(session, span_id, SpanKind.TURN),
        turn_number=1,
        input_tokens=1820,
        output_tokens=140,
        cache_read_tokens=1600,
        cache_write_tokens=0,
        cost_usd=0.0123,
        stop_reason=StopReason.TOOL_USE,
        tool_calls_requested=1,
    )


def _read_lines(root: Path, session: str) -> list[dict[str, Any]]:
    path = root / "sessions" / session / "trace.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture(autouse=True)
def _no_otlp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPORE_OTLP_ENDPOINT", raising=False)


def _provider(root: Path) -> OutboxObservabilityProvider:
    return OutboxObservabilityProvider(OutboxConfig(root=root))


# ── one line per emit ──────────────────────────────────────────────────────


def test_one_line_per_emit(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_turn(_turn("s1", "sp1"))
    obs.emit_turn(_turn("s1", "sp2"))
    assert len(_read_lines(tmp_path, "s1")) == 2


# ── schema-conformant turn line ────────────────────────────────────────────


def test_turn_line_matches_schema(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_turn(_turn("s1", "sp1"))
    line = _read_lines(tmp_path, "s1")[0]
    assert line["kind"] == "turn"
    assert line["level"] == "info"
    assert line["span_id"] == "sp1"
    assert line["parent_span_id"] is None
    assert line["session_id"] == "s1"
    assert line["task_id"] == "task1"
    assert line["timestamp"] == "2026-05-26T18:00:02.100Z"
    assert line["started_at"] == "2026-05-26T18:00:00.000Z"
    assert line["duration_ms"] == 2100
    assert line["status"] == "ok"
    assert line["status_detail"] is None
    assert line["attributes"]["turn_number"] == 1
    assert line["attributes"]["input_tokens"] == 1820
    assert line["attributes"]["cache_read_tokens"] == 1600
    assert line["attributes"]["stop_reason"] == "tool_use"
    assert line["attributes"]["tool_calls_requested"] == 1
    assert len(line["trace_id"]) == 32


# ── patch line is warn ─────────────────────────────────────────────────────


def test_patch_line_is_warn(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    span = PatchSpan(
        base=_base("s1", "p1", SpanKind.PATCH),
        call_id="c1",
        tool_name="shell",
        original_parameters={"a": "1"},
        patched_parameters={"a": 1},
        patch_type=PatchTypeParameterCoercion(field="a", **{"from": "string"}, to="number"),
    )
    obs.emit_patch(span)
    line = _read_lines(tmp_path, "s1")[0]
    assert line["kind"] == "patch"
    assert line["level"] == "warn"
    assert line["attributes"]["patch_type"]["kind"] == "parameter_coercion"
    assert line["attributes"]["patch_type"]["from"] == "string"
    assert line["attributes"]["original_parameters"]["a"] == "1"
    assert line["attributes"]["patched_parameters"]["a"] == 1


# ── status ok / error / halted ─────────────────────────────────────────────


def test_status_error_and_halted_map_to_detail(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    t1 = _turn("s1", "err")
    t1.base.status = SpanStatusError(message="boom")
    obs.emit_turn(t1)
    t2 = _turn("s1", "halt")
    t2.base.status = SpanStatusHalted(reason="stop")
    obs.emit_turn(t2)
    lines = _read_lines(tmp_path, "s1")
    assert lines[0]["status"] == "error"
    assert lines[0]["status_detail"] == "boom"
    assert lines[1]["status"] == "halted"
    assert lines[1]["status_detail"] == "stop"


# ── context vs compaction kind ─────────────────────────────────────────────


def test_context_vs_compaction_kind(tmp_path: Path) -> None:
    obs = _provider(tmp_path)

    def mk(span_id: str, op: Any) -> ContextSpan:
        return ContextSpan(
            base=_base("s1", span_id, SpanKind.CONTEXT_ASSEMBLY),
            operation=op,
            tokens_before=100,
            tokens_after=50,
            utilization_before=0.9,
            utilization_after=0.5,
        )

    obs.emit_context(
        mk("asm", ContextOperationAssembly(guides_loaded=1, memory_items_loaded=2, tools_loaded=3))
    )
    obs.emit_context(
        mk("comp", ContextOperationCompaction(messages_removed=5, tokens_reclaimed=50))
    )
    lines = _read_lines(tmp_path, "s1")
    assert lines[0]["kind"] == "context_assembly"
    assert lines[0]["attributes"]["operation"]["kind"] == "assembly"
    assert lines[1]["kind"] == "compaction"
    assert lines[1]["attributes"]["operation"]["kind"] == "compaction"


# ── sensor and middleware lines ────────────────────────────────────────────


def test_sensor_and_middleware_lines(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_sensor(
        SensorSpan(
            base=_base("s1", "sn1", SpanKind.SENSOR_EVALUATION),
            sensor_id=SensorId("test-runner"),
            sensor_kind=SensorKind.COMPUTATIONAL,
            trigger=SensorTriggerPostTool(tool_name="shell"),
            outcome=SensorOutcome.PASS,
            fired=True,
        )
    )
    obs.emit_middleware(
        MiddlewareSpan(
            base=_base("s1", "mw1", SpanKind.MIDDLEWARE_HOOK),
            hook=HookPoint.BEFORE_TURN,
            decision=MiddlewareContinue(),
        )
    )
    lines = _read_lines(tmp_path, "s1")
    assert lines[0]["kind"] == "sensor_evaluation"
    assert lines[0]["attributes"]["sensor_id"] == "test-runner"
    assert lines[0]["attributes"]["trigger"]["kind"] == "post_tool"
    assert lines[0]["attributes"]["fired"] is True
    assert lines[1]["kind"] == "middleware_hook"
    assert lines[1]["attributes"]["hook"] == "before_turn"
    assert lines[1]["attributes"]["decision"]["kind"] == "continue"


# ── tool_call line ─────────────────────────────────────────────────────────


def test_tool_call_line(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_tool_call(
        ToolCallSpan(
            base=_base("s1", "tc1", SpanKind.TOOL_CALL),
            tool_name="shell",
            call_id="call_1",
            parameters_size_bytes=64,
            output_size_bytes=4096,
            truncated=True,
            sandbox_mode="workspace_scoped",
            sandbox_violations=[],
        )
    )
    line = _read_lines(tmp_path, "s1")[0]
    assert line["kind"] == "tool_call"
    assert line["attributes"]["tool_name"] == "shell"
    assert line["attributes"]["truncated"] is True
    assert line["attributes"]["sandbox_violations"] == []


# ── flush writes session summary + marker ──────────────────────────────────


async def test_flush_writes_session_summary_and_marker(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_turn(_turn("s1", "sp1"))
    obs.inner.set_session_outcome(_sid("s1"), SessionOutcomeSuccess())
    await obs.flush_session(_sid("s1"))
    lines = _read_lines(tmp_path, "s1")
    last = lines[-1]
    assert last["kind"] == "session"
    assert last["attributes"]["outcome"] == "success"
    assert last["attributes"]["total_turns"] == 1
    assert (tmp_path / "sessions" / "s1" / ".flushed").exists()


# ── flush summary disabled ─────────────────────────────────────────────────


async def test_flush_no_summary_when_disabled(tmp_path: Path) -> None:
    obs = OutboxObservabilityProvider(OutboxConfig(root=tmp_path, flush_on_session_end=False))
    obs.emit_turn(_turn("s1", "sp1"))
    obs.inner.set_session_outcome(_sid("s1"), SessionOutcomeSuccess())
    await obs.flush_session(_sid("s1"))
    lines = _read_lines(tmp_path, "s1")
    assert len(lines) == 1
    assert lines[0]["kind"] == "turn"
    assert (tmp_path / "sessions" / "s1" / ".flushed").exists()


# ── rotation at tiny max_size_bytes ────────────────────────────────────────


def test_rotation_at_tiny_max_size(tmp_path: Path) -> None:
    obs = OutboxObservabilityProvider(OutboxConfig(root=tmp_path, max_size_bytes=10))
    obs.emit_turn(_turn("s1", "sp1"))
    obs.emit_turn(_turn("s1", "sp2"))
    obs.emit_turn(_turn("s1", "sp3"))
    session_dir = tmp_path / "sessions" / "s1"
    rotated = [
        p for p in session_dir.iterdir() if p.name.startswith("trace-") and p.suffix == ".jsonl"
    ]
    assert rotated, "expected at least one rotated segment"
    assert (session_dir / "trace-001.jsonl").exists()


# ── JSONL only when env unset ──────────────────────────────────────────────


def test_jsonl_only_when_env_unset(tmp_path: Path) -> None:
    obs = OutboxObservabilityProvider(OutboxConfig(root=tmp_path))
    obs.emit_turn(_turn("s1", "sp1"))
    assert len(_read_lines(tmp_path, "s1")) == 1


# ── list_unflushed before/after flush ──────────────────────────────────────


async def test_list_unflushed_before_and_after_flush(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_turn(_turn("s1", "sp1"))
    before = await obs.list_unflushed_sessions()
    assert before == [_sid("s1")]
    obs.inner.set_session_outcome(_sid("s1"), SessionOutcomeSuccess())
    await obs.flush_session(_sid("s1"))
    after = await obs.list_unflushed_sessions()
    assert after == []


# ── cleanup_session success + not found ────────────────────────────────────


async def test_cleanup_session_success_and_not_found(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_turn(_turn("s1", "sp1"))
    await obs.cleanup_session(_sid("s1"))
    assert not (tmp_path / "sessions" / "s1").exists()
    with pytest.raises(SessionNotFound) as exc:
        await obs.cleanup_session(_sid("missing"))
    assert exc.value.session_id == "missing"


# ── trace_id stable per session / distinct across sessions ─────────────────


def test_trace_id_stable_per_session_and_differs_across_sessions(tmp_path: Path) -> None:
    obs = _provider(tmp_path)
    obs.emit_turn(_turn("s1", "a"))
    obs.emit_turn(_turn("s1", "b"))
    obs.emit_turn(_turn("s2", "c"))
    s1 = _read_lines(tmp_path, "s1")
    s2 = _read_lines(tmp_path, "s2")
    assert s1[0]["trace_id"] == s1[1]["trace_id"]
    assert s1[0]["trace_id"] != s2[0]["trace_id"]


# ── fixture replay (cross-language ground truth) ───────────────────────────


def _build_line(fixture_name: str, span: dict[str, Any], trace_id: str) -> TraceLine:
    if fixture_name == "trace_line_turn.json":
        return TraceLine.from_turn(_load_turn(span), trace_id)
    if fixture_name == "trace_line_tool_call.json":
        return TraceLine.from_tool_call(_load_tool_call(span), trace_id)
    if fixture_name == "trace_line_sensor.json":
        return TraceLine.from_sensor(_load_sensor(span), trace_id)
    if fixture_name in ("trace_line_context_assembly.json", "trace_line_compaction.json"):
        return TraceLine.from_context(_load_context(span), trace_id)
    if fixture_name == "trace_line_middleware.json":
        return TraceLine.from_middleware(_load_middleware(span), trace_id)
    if fixture_name == "trace_line_patch.json":
        return TraceLine.from_patch(_load_patch(span), trace_id)
    if fixture_name == "trace_line_session_summary.json":
        metrics = SessionMetrics(
            **{
                **span["metrics"],
                "outcome": _load_outcome(span["metrics"]["outcome"]),
            }
        )
        return TraceLine.session_summary(metrics, trace_id, _load_base(span["root"]))
    raise AssertionError(f"unknown fixture {fixture_name}")


def _load_outcome(d: dict[str, Any]) -> Any:
    from spore_core.guide_registry import (
        SessionOutcomeFailure,
        SessionOutcomePartial,
        SessionOutcomeSuccess as _Succ,
    )

    kind = d["kind"]
    if kind == "success":
        return _Succ()
    if kind == "failure":
        return SessionOutcomeFailure(reason=d.get("reason", ""))
    return SessionOutcomePartial()


def _load_base(d: dict[str, Any]) -> SpanBase:
    status = d["status"]
    if status["kind"] == "error":
        st: Any = SpanStatusError(message=status["message"])
    elif status["kind"] == "halted":
        st = SpanStatusHalted(reason=status["reason"])
    else:
        st = SpanStatusOk()
    return SpanBase(
        span_id=SpanId(d["span_id"]),
        parent_span_id=(SpanId(d["parent_span_id"]) if d.get("parent_span_id") else None),
        session_id=SessionId(d["session_id"]),
        task_id=TaskId(d["task_id"]),
        kind=SpanKind(d["kind"]),
        started_at=Timestamp(d["started_at"]),
        ended_at=Timestamp(d["ended_at"]),
        duration_ms=d["duration_ms"],
        status=st,
    )


def _load_turn(d: dict[str, Any]) -> TurnSpan:
    return TurnSpan(
        base=_load_base(d["base"]),
        turn_number=d["turn_number"],
        input_tokens=d["input_tokens"],
        output_tokens=d["output_tokens"],
        cache_read_tokens=d.get("cache_read_tokens"),
        cache_write_tokens=d.get("cache_write_tokens"),
        cost_usd=d["cost_usd"],
        stop_reason=StopReason(d["stop_reason"]),
        tool_calls_requested=d["tool_calls_requested"],
    )


def _load_tool_call(d: dict[str, Any]) -> ToolCallSpan:
    return ToolCallSpan(
        base=_load_base(d["base"]),
        tool_name=d["tool_name"],
        call_id=d["call_id"],
        parameters_size_bytes=d["parameters_size_bytes"],
        output_size_bytes=d["output_size_bytes"],
        truncated=d["truncated"],
        sandbox_mode=d["sandbox_mode"],
        sandbox_violations=list(d["sandbox_violations"]),
    )


def _load_sensor(d: dict[str, Any]) -> SensorSpan:
    return SensorSpan(
        base=_load_base(d["base"]),
        sensor_id=SensorId(d["sensor_id"]),
        sensor_kind=SensorKind(d["sensor_kind"]),
        trigger=SensorTriggerPostTool(tool_name=d["trigger"].get("tool_name", "")),
        outcome=SensorOutcome(d["outcome"]),
        fired=d["fired"],
    )


def _load_context(d: dict[str, Any]) -> ContextSpan:
    op = d["operation"]
    if op["kind"] == "assembly":
        operation: Any = ContextOperationAssembly(
            guides_loaded=op["guides_loaded"],
            memory_items_loaded=op["memory_items_loaded"],
            tools_loaded=op["tools_loaded"],
        )
    else:
        operation = ContextOperationCompaction(
            messages_removed=op["messages_removed"],
            tokens_reclaimed=op["tokens_reclaimed"],
        )
    return ContextSpan(
        base=_load_base(d["base"]),
        operation=operation,
        tokens_before=d["tokens_before"],
        tokens_after=d["tokens_after"],
        utilization_before=d["utilization_before"],
        utilization_after=d["utilization_after"],
    )


def _load_middleware(d: dict[str, Any]) -> MiddlewareSpan:
    return MiddlewareSpan(
        base=_load_base(d["base"]),
        hook=HookPoint(d["hook"]),
        decision=MiddlewareContinue(),
    )


def _load_patch(d: dict[str, Any]) -> PatchSpan:
    pt = d["patch_type"]
    return PatchSpan(
        base=_load_base(d["base"]),
        call_id=d["call_id"],
        tool_name=d["tool_name"],
        original_parameters=d["original_parameters"],
        patched_parameters=d["patched_parameters"],
        patch_type=PatchTypeParameterCoercion(
            field=pt["field"], **{"from": pt["from"]}, to=pt["to"]
        ),
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "trace_line_turn.json",
        "trace_line_tool_call.json",
        "trace_line_sensor.json",
        "trace_line_context_assembly.json",
        "trace_line_compaction.json",
        "trace_line_middleware.json",
        "trace_line_patch.json",
        "trace_line_session_summary.json",
    ],
)
def test_fixture_replay(fixture_name: str) -> None:
    raw = json.loads((FIXTURES / fixture_name).read_text())
    line = _build_line(fixture_name, raw["span"], raw["trace_id"])
    assert line.to_dict() == raw["expected_line"], f"mismatch in {fixture_name}"
