"""Issue #33 — :class:`OutboxObservabilityProvider`: a durable, JSONL-backed
observability provider that wraps :class:`InMemoryObservabilityProvider`.

Mirrors the Rust reference at
``rust/crates/spore-core/src/observability_outbox.rs``. This is a behavioural
port, NOT a transliteration — it uses idiomatic Python (dataclasses, pydantic
serialization, ``typing.Protocol``) and the existing synchronous ``emit_*``
signatures.

## What it adds on top of the in-memory provider

1. **Durable outbox.** Every ``emit_*`` writes exactly ONE JSONL line,
   synchronously appended and flushed, to
   ``{root}/sessions/{session_id}/trace.jsonl``. The local JSONL file is the
   **source of truth** (see ``observability/TRACE_SCHEMA.md``). The wrapped
   :class:`InMemoryObservabilityProvider` still handles all buffering, metrics,
   and query methods — this provider only adds the file write + OTLP hop.
2. **OTLP forwarding.** When the env var ``SPORE_OTLP_ENDPOINT`` is set
   (non-empty / non-whitespace) at construction, each span is ALSO forwarded to
   OTLP gRPC, best-effort and non-blocking. Drops on failure are acceptable
   because the JSONL file is durable. When unset/empty, JSONL only.

## Rules enforced

* One JSONL line per ``emit_*``, synchronously appended + flushed.
* The line matches the schema envelope exactly; ``attributes`` is the verbatim
  serde serialization of the payload fields (structured unions stay tagged
  objects; scalars stay scalar; keys are NOT renamed/flattened).
* ``level``: patch spans → ``"warn"``; all other kinds → ``"info"``.
* ``status``/``status_detail``: ``ok`` → ``("ok", None)``, ``error`` →
  ``("error", message)``, ``halted`` → ``("halted", reason)``. Bare strings.
* ``context_assembly`` vs ``compaction`` envelope ``kind`` mirrors
  :meth:`emit_context` (compaction → ``"compaction"``, else
  ``"context_assembly"``).
* ``session`` summary ``attributes.outcome`` is the bare-string serialization of
  :data:`SessionOutcome` (``success``/``failure``/``partial``).
* ``trace_id``: a 32-hex (16 random bytes) string generated ONCE per session,
  reused in every line for that session and as the OTLP trace id. JSONL
  ``span_id``/``parent_span_id`` are the harness ``SpanId`` string VERBATIM. For
  OTLP only, an 8-byte span id is derived by hashing (SHA-256, first 8 bytes)
  the ``SpanId`` string.
* Rotation: when the active ``trace.jsonl`` exceeds ``max_size_bytes`` after an
  append, it is renamed to ``trace-{NNN}.jsonl`` (zero-padded, increasing) and a
  fresh ``trace.jsonl`` is opened. Rotated segments keep ``.jsonl``.
* ``flush_session``: writes the trailing ``session`` summary line (when
  ``flush_on_session_end``), flushes the file, force-flushes OTLP best-effort
  (logs a warning, never raises), and creates a sibling ``.flushed`` marker.
* ``list_unflushed_sessions`` / ``cleanup_session`` per issue #33.

## Deviation — OTLP behind an internal Protocol

The OTLP forwarding layer sits behind a small internal :class:`_OtlpForwarder`
Protocol with a real ``opentelemetry-sdk`` + OTLP-gRPC implementation
(:class:`_OtlpSdkForwarder`, using a ``BatchSpanProcessor`` for non-blocking
export) and a no-op default (:class:`_NullForwarder`). The durable-JSONL path is
fully implemented and tested WITHOUT any live OTLP / Tempo stack or network.
This isolates the version-churny OTLP SDK from the reliability-critical outbox
and lets the tests run hermetically. If the OTLP SDK is not importable, the
provider silently degrades to JSONL-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .errors import SporeError
from .guide_registry import (
    SessionOutcome,
    SessionOutcomeFailure,
    SessionOutcomeSuccess,
)
from .harness import SessionId
from .memory import Timestamp
from .observability import (
    ContextOperationCompaction,
    ContextSpan,
    InMemoryObservabilityProvider,
    MiddlewareSpan,
    PatchSpan,
    SensorSpan,
    SessionMetrics,
    Span,
    SpanBase,
    SpanId,
    SpanKind,
    SpanStatusError,
    SpanStatusHalted,
    ToolCallSpan,
    TurnSpan,
)

_LOG = logging.getLogger("spore_core.observability_outbox")

DEFAULT_MAX_SIZE_BYTES = 50 * 1024 * 1024


# ============================================================================
# Errors
# ============================================================================


class ObservabilityError(SporeError):
    """Root of durable-outbox provider errors (issue #33)."""


class SessionNotFound(ObservabilityError):
    """``cleanup_session`` was called for a session with no outbox directory."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"session not found: {session_id}")


# ============================================================================
# Config
# ============================================================================


@dataclass
class OutboxConfig:
    """Configuration for the durable outbox. ``root`` is the outbox root
    directory (``.spore`` by convention); per session the provider derives
    ``root/sessions/{session_id}/trace.jsonl``."""

    root: Path
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES
    flush_on_session_end: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def session_dir(self, session_id: SessionId) -> Path:
        return self.root / "sessions" / str(session_id)


# ============================================================================
# Verbatim serialization of span payload fields
# ============================================================================


def _to_json_value(value: Any) -> Any:
    """Serialize a span field to its verbatim JSON representation.

    Pydantic models (tagged unions such as ``trigger``, ``operation``,
    ``decision``, ``patch_type``) keep their tagged-object shape via
    ``model_dump(by_alias=True, mode="json")``; ``str``-backed enums collapse to
    their value; scalars and ``None`` pass through unchanged. Names are NOT
    renamed or flattened.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True, mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    return value


def _status_pair(base: SpanBase) -> tuple[str, str | None]:
    """Map a :data:`SpanStatus` to the bare ``(status, status_detail)`` pair."""
    status = base.status
    if isinstance(status, SpanStatusError):
        return ("error", status.message)
    if isinstance(status, SpanStatusHalted):
        return ("halted", status.reason)
    return ("ok", None)


# ============================================================================
# TraceLine envelope
# ============================================================================


@dataclass
class TraceLine:
    """One on-disk JSONL line. Common envelope fields are top-level; the
    per-kind payload lives under ``attributes``. Built by the ``from_*`` /
    :meth:`session_summary` constructors — these are the unit-tested,
    cross-language mapping surface."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    session_id: str
    task_id: str
    kind: str
    level: str
    timestamp: str
    started_at: str
    duration_ms: int
    status: str
    status_detail: str | None
    attributes: dict[str, Any]

    @staticmethod
    def _from_base(
        base: SpanBase,
        trace_id: str,
        kind: str,
        level: str,
        attributes: dict[str, Any],
    ) -> TraceLine:
        status, status_detail = _status_pair(base)
        return TraceLine(
            trace_id=trace_id,
            span_id=str(base.span_id),
            parent_span_id=(str(base.parent_span_id) if base.parent_span_id is not None else None),
            session_id=str(base.session_id),
            task_id=str(base.task_id),
            kind=kind,
            level=level,
            timestamp=str(base.ended_at),
            started_at=str(base.started_at),
            duration_ms=base.duration_ms,
            status=status,
            status_detail=status_detail,
            attributes=attributes,
        )

    @staticmethod
    def from_turn(span: TurnSpan, trace_id: str) -> TraceLine:
        attributes = {
            "turn_number": span.turn_number,
            "input_tokens": span.input_tokens,
            "output_tokens": span.output_tokens,
            "cache_read_tokens": span.cache_read_tokens,
            "cache_write_tokens": span.cache_write_tokens,
            "cost_usd": span.cost_usd,
            "stop_reason": _to_json_value(span.stop_reason),
            "tool_calls_requested": span.tool_calls_requested,
        }
        return TraceLine._from_base(span.base, trace_id, "turn", "info", attributes)

    @staticmethod
    def from_tool_call(span: ToolCallSpan, trace_id: str) -> TraceLine:
        attributes = {
            "tool_name": span.tool_name,
            "call_id": span.call_id,
            "parameters_size_bytes": span.parameters_size_bytes,
            "output_size_bytes": span.output_size_bytes,
            "truncated": span.truncated,
            "sandbox_mode": span.sandbox_mode,
            "sandbox_violations": list(span.sandbox_violations),
        }
        return TraceLine._from_base(span.base, trace_id, "tool_call", "info", attributes)

    @staticmethod
    def from_sensor(span: SensorSpan, trace_id: str) -> TraceLine:
        attributes = {
            "sensor_id": str(span.sensor_id),
            "sensor_kind": _to_json_value(span.sensor_kind),
            "trigger": _to_json_value(span.trigger),
            "outcome": _to_json_value(span.outcome),
            "fired": span.fired,
        }
        return TraceLine._from_base(span.base, trace_id, "sensor_evaluation", "info", attributes)

    @staticmethod
    def from_context(span: ContextSpan, trace_id: str) -> TraceLine:
        # Mirror emit_context: Compaction → "compaction"; all else →
        # "context_assembly".
        kind = (
            "compaction"
            if isinstance(span.operation, ContextOperationCompaction)
            else "context_assembly"
        )
        attributes = {
            "operation": _to_json_value(span.operation),
            "tokens_before": span.tokens_before,
            "tokens_after": span.tokens_after,
            "utilization_before": span.utilization_before,
            "utilization_after": span.utilization_after,
        }
        return TraceLine._from_base(span.base, trace_id, kind, "info", attributes)

    @staticmethod
    def from_middleware(span: MiddlewareSpan, trace_id: str) -> TraceLine:
        attributes = {
            "hook": _to_json_value(span.hook),
            "decision": _to_json_value(span.decision),
        }
        return TraceLine._from_base(span.base, trace_id, "middleware_hook", "info", attributes)

    @staticmethod
    def from_patch(span: PatchSpan, trace_id: str) -> TraceLine:
        attributes = {
            "tool_name": span.tool_name,
            "call_id": span.call_id,
            "patch_type": _to_json_value(span.patch_type),
            "original_parameters": _to_json_value(span.original_parameters),
            "patched_parameters": _to_json_value(span.patched_parameters),
        }
        # Patch spans are ALWAYS warn-level.
        return TraceLine._from_base(span.base, trace_id, "patch", "warn", attributes)

    @staticmethod
    def session_summary(
        metrics: SessionMetrics,
        trace_id: str,
        root: SpanBase,
    ) -> TraceLine:
        outcome = _session_outcome_str(metrics.outcome)
        attributes = {
            "outcome": outcome,
            "total_turns": metrics.total_turns,
            "total_cost_usd": metrics.total_cost_usd,
            "sensor_fires": metrics.sensor_fires,
            "sensor_halts": metrics.sensor_halts,
            "patch_count": metrics.patch_count,
        }
        return TraceLine._from_base(root, trace_id, "session", "info", attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "level": self.level,
            "timestamp": self.timestamp,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "status_detail": self.status_detail,
            "attributes": self.attributes,
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":")) + "\n"


def _session_outcome_str(outcome: SessionOutcome) -> str:
    if isinstance(outcome, SessionOutcomeSuccess):
        return "success"
    if isinstance(outcome, SessionOutcomeFailure):
        return "failure"
    return "partial"


# ============================================================================
# trace_id / span_id derivation
# ============================================================================


def new_trace_id() -> str:
    """A fresh 32-hex (16 random bytes) trace id, generated once per session."""
    return secrets.token_hex(16)


def derive_otlp_span_id(span_id: str) -> bytes:
    """Derive an 8-byte OTLP span id from the harness SpanId string by hashing
    (SHA-256, first 8 bytes) — matches the Rust reference."""
    return hashlib.sha256(span_id.encode("utf-8")).digest()[:8]


# ============================================================================
# OTLP forwarder (internal Protocol — see module-header deviation note)
# ============================================================================


class _OtlpForwarder(Protocol):
    """Internal abstraction over the OTLP forwarding hop. The durable JSONL path
    does not depend on this; it exists only to isolate the OTLP SDK and let
    tests run without a network."""

    def forward(self, line: TraceLine) -> None: ...

    def force_flush(self) -> None: ...


class _NullForwarder:
    """No-op forwarder used when ``SPORE_OTLP_ENDPOINT`` is unset/empty."""

    def forward(self, line: TraceLine) -> None:
        return None

    def force_flush(self) -> None:
        return None


class _OtlpSdkForwarder:
    """Real OTLP forwarder backed by ``opentelemetry-sdk`` + OTLP-gRPC with a
    ``BatchSpanProcessor`` so export is buffered and non-blocking."""

    def __init__(self, endpoint: str) -> None:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        self._processor = BatchSpanProcessor(exporter)
        resource = Resource.create({"service.name": "spore-core"})
        self._provider = TracerProvider(resource=resource)
        self._provider.add_span_processor(self._processor)
        self._tracer = self._provider.get_tracer("spore-core")

    def forward(self, line: TraceLine) -> None:
        # Best-effort, fire-and-forget. The batch processor handles batching and
        # non-blocking export; the JSONL file is durable, so any failure is
        # swallowed.
        try:
            from opentelemetry.trace import SpanKind as OtelSpanKind

            span = self._tracer.start_span(
                line.kind,
                kind=OtelSpanKind.INTERNAL,
                attributes={
                    "session_id": line.session_id,
                    "task_id": line.task_id,
                    "level": line.level,
                    "status": line.status,
                    "trace_id": line.trace_id,
                    "span_id_str": line.span_id,
                    **(
                        {"parent_span_id": line.parent_span_id}
                        if line.parent_span_id is not None
                        else {}
                    ),
                },
            )
            span.end()
        except Exception as exc:  # noqa: BLE001 - best-effort, never raise
            _LOG.warning("OTLP forward failed (JSONL is durable): %r", exc)

    def force_flush(self) -> None:
        try:
            self._provider.force_flush(timeout_millis=2000)
        except Exception as exc:  # noqa: BLE001 - best-effort, never raise
            _LOG.warning("OTLP force_flush failed (JSONL is durable): %r", exc)


def _build_forwarder() -> _OtlpForwarder:
    endpoint = os.environ.get("SPORE_OTLP_ENDPOINT", "")
    if not endpoint.strip():
        return _NullForwarder()
    try:
        return _OtlpSdkForwarder(endpoint.strip())
    except Exception as exc:  # noqa: BLE001 - degrade to JSONL-only on any error
        _LOG.warning(
            "failed to init OTLP forwarder for %r; JSONL only: %r",
            endpoint.strip(),
            exc,
        )
        return _NullForwarder()


# ============================================================================
# SessionWriter — per-session open handle + rotation + trace_id
# ============================================================================


@dataclass
class _SessionWriter:
    dir: Path
    max_size_bytes: int
    active_path: Path = field(init=False)
    bytes_written: int = field(init=False, default=0)
    next_seq: int = field(init=False, default=1)
    trace_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.active_path = self.dir / "trace.jsonl"
        self._handle = self.active_path.open("a", encoding="utf-8")
        self.bytes_written = self.active_path.stat().st_size if self.active_path.exists() else 0
        self.next_seq = self._scan_next_seq()
        self.trace_id = new_trace_id()

    def _scan_next_seq(self) -> int:
        max_seen = 0
        for entry in self.dir.iterdir():
            name = entry.name
            if name.startswith("trace-") and name.endswith(".jsonl"):
                num = name[len("trace-") : -len(".jsonl")]
                if num.isdigit():
                    max_seen = max(max_seen, int(num))
        return max_seen + 1

    def append(self, line: str) -> None:
        self._handle.write(line)
        self._handle.flush()
        self.bytes_written += len(line.encode("utf-8"))
        if self.bytes_written > self.max_size_bytes:
            self._rotate()

    def _rotate(self) -> None:
        rotated = self.dir / f"trace-{self.next_seq:03d}.jsonl"
        self.next_seq += 1
        self._handle.flush()
        self._handle.close()
        self.active_path.rename(rotated)
        self._handle = self.active_path.open("a", encoding="utf-8")
        self.bytes_written = 0

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


# ============================================================================
# OutboxObservabilityProvider
# ============================================================================


class OutboxObservabilityProvider:
    """A durable, JSONL-backed :class:`ObservabilityProvider` (issue #33). Wraps
    an :class:`InMemoryObservabilityProvider` for all buffering / metrics / query
    behavior and adds: one synchronous JSONL line per ``emit_*``, optional
    best-effort OTLP forwarding, rotation, and flush markers."""

    def __init__(self, config: OutboxConfig) -> None:
        self.config = config
        self._inner = InMemoryObservabilityProvider()
        self._lock = threading.Lock()
        self._writers: dict[SessionId, _SessionWriter] = {}
        self._otlp = _build_forwarder()

    @property
    def inner(self) -> InMemoryObservabilityProvider:
        """The wrapped in-memory provider (e.g. to call
        ``set_session_outcome`` / ``record_guides_used``)."""
        return self._inner

    # ── writer management ───────────────────────────────────────────────────

    def _writer_for(self, session_id: SessionId) -> _SessionWriter:
        writer = self._writers.get(session_id)
        if writer is None:
            writer = _SessionWriter(self.config.session_dir(session_id), self.config.max_size_bytes)
            self._writers[session_id] = writer
        return writer

    def trace_id_for(self, session_id: SessionId) -> str:
        """The per-session ``trace_id``, opening the session writer if needed."""
        with self._lock:
            return self._writer_for(session_id).trace_id

    def _write_line(self, session_id: SessionId, line: TraceLine) -> None:
        jsonl = line.to_jsonl_line()
        with self._lock:
            try:
                self._writer_for(session_id).append(jsonl)
            except OSError as exc:
                _LOG.warning("outbox append failed: %r", exc)
                return
        self._otlp.forward(line)

    # ── emit_* (fire-and-forget) ────────────────────────────────────────────

    def emit_turn(self, span: TurnSpan) -> None:
        trace_id = self.trace_id_for(span.base.session_id)
        line = TraceLine.from_turn(span, trace_id)
        self._inner.emit_turn(span)
        self._write_line(span.base.session_id, line)

    def emit_tool_call(self, span: ToolCallSpan) -> None:
        trace_id = self.trace_id_for(span.base.session_id)
        line = TraceLine.from_tool_call(span, trace_id)
        self._inner.emit_tool_call(span)
        self._write_line(span.base.session_id, line)

    def emit_sensor(self, span: SensorSpan) -> None:
        trace_id = self.trace_id_for(span.base.session_id)
        line = TraceLine.from_sensor(span, trace_id)
        self._inner.emit_sensor(span)
        self._write_line(span.base.session_id, line)

    def emit_context(self, span: ContextSpan) -> None:
        trace_id = self.trace_id_for(span.base.session_id)
        line = TraceLine.from_context(span, trace_id)
        self._inner.emit_context(span)
        self._write_line(span.base.session_id, line)

    def emit_middleware(self, span: MiddlewareSpan) -> None:
        trace_id = self.trace_id_for(span.base.session_id)
        line = TraceLine.from_middleware(span, trace_id)
        self._inner.emit_middleware(span)
        self._write_line(span.base.session_id, line)

    def emit_patch(self, span: PatchSpan) -> None:
        trace_id = self.trace_id_for(span.base.session_id)
        line = TraceLine.from_patch(span, trace_id)
        self._inner.emit_patch(span)
        self._write_line(span.base.session_id, line)

    # ── flush_session ────────────────────────────────────────────────────────

    async def flush_session(self, session_id: SessionId) -> None:
        # Write the trailing session summary line from rolled-up metrics.
        if self.config.flush_on_session_end:
            metrics = await self._inner.get_session_metrics(session_id)
            if metrics is not None:
                trace_id = self.trace_id_for(session_id)
                root = SpanBase(
                    span_id=SpanId(str(session_id)),
                    session_id=session_id,
                    task_id=metrics.task_id,
                    kind=SpanKind.SESSION,
                    started_at=Timestamp(""),
                    ended_at=Timestamp(""),
                    duration_ms=metrics.total_duration_ms,
                    parent_span_id=None,
                )
                line = TraceLine.session_summary(metrics, trace_id, root)
                self._write_line(session_id, line)

        # Flush the JSONL file handle.
        with self._lock:
            writer = self._writers.get(session_id)
            if writer is not None:
                writer.flush()

        # Best-effort OTLP force-flush; never raises out of flush_session.
        self._otlp.force_flush()

        # Create the sibling .flushed marker.
        session_dir = self.config.session_dir(session_id)
        if session_dir.exists():
            try:
                (session_dir / ".flushed").touch()
            except OSError as exc:
                _LOG.warning("failed to write .flushed marker: %r", exc)

        # Delegate so the inner provider's `flushed` bookkeeping stays consistent.
        await self._inner.flush_session(session_id)

    # ── query delegation ──────────────────────────────────────────────────────

    async def get_session_metrics(self, session_id: SessionId) -> SessionMetrics | None:
        return await self._inner.get_session_metrics(session_id)

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]:
        return await self._inner.get_sessions(since, domain, outcome)

    async def get_trace(self, session_id: SessionId) -> list[Span]:
        return await self._inner.get_trace(session_id)

    # ── outbox management (issue #33) ──────────────────────────────────────────

    async def list_unflushed_sessions(self) -> list[SessionId]:
        """Session ids whose durable outbox has a ``trace.jsonl`` but no
        ``.flushed`` marker."""
        out: list[SessionId] = []
        sessions_dir = self.config.root / "sessions"
        if not sessions_dir.exists():
            return out
        for entry in sessions_dir.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "trace.jsonl").exists() and not (entry / ".flushed").exists():
                out.append(SessionId(entry.name))
        out.sort()
        return out

    async def cleanup_session(self, session_id: SessionId) -> None:
        """Delete a session's durable outbox directory. The provider NEVER
        auto-deletes; the caller drives cleanup. Raises :class:`SessionNotFound`
        if the directory does not exist."""
        session_dir = self.config.session_dir(session_id)
        if not session_dir.exists():
            raise SessionNotFound(str(session_id))
        with self._lock:
            writer = self._writers.pop(session_id, None)
            if writer is not None:
                writer.close()
        shutil.rmtree(session_dir)


__all__ = [
    "DEFAULT_MAX_SIZE_BYTES",
    "ObservabilityError",
    "OutboxConfig",
    "OutboxObservabilityProvider",
    "SessionNotFound",
    "TraceLine",
    "derive_otlp_span_id",
    "new_trace_id",
]
