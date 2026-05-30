"""ObservabilityProvider — structured recording of all harness activity
(issue #12).

Mirrors the Rust reference at ``rust/crates/spore-core/src/observability.rs``.

Every observable harness operation emits one span. Spans carry identity
(session, task, parent span), timing, status, and operation-specific
payload. Aggregates roll up to :class:`SessionMetrics` for the
improvement loop.

See ``docs/harness-engineering-concepts.md`` § "ObservabilityProvider"
for the authoritative rules. This module ships:

* All span payload types from the spec (:class:`TurnSpan`,
  :class:`ToolCallSpan`, :class:`SensorSpan`, :class:`ContextSpan`,
  :class:`MiddlewareSpan`).
* The full :class:`ObservabilityProvider` Protocol with per-span-kind
  ``emit_*`` methods, :meth:`flush_session`, and query methods.
* :class:`InMemoryObservabilityProvider` — buffered in-memory backend
  used for tests and short-lived processes.
* :class:`PricingTable` — provider-specific token → USD lookup,
  injected at construction so ``cost_usd`` is a first-class span field.

Rules enforced:

* ``emit_*`` methods are **fire-and-forget** (synchronous, return ``None``).
* :meth:`InMemoryObservabilityProvider.flush_session` is idempotent.
* :meth:`InMemoryObservabilityProvider.get_trace` returns spans in
  insertion order; ``parent_span_id`` linkage is preserved on each span.
* ``cost_usd`` on :class:`TurnSpan` is computed via :class:`PricingTable`;
  :attr:`PricingTable.DEFAULT` reports zero cost for any token counts.
* :meth:`InMemoryObservabilityProvider.get_session_metrics` aggregates
  token counts, cost, durations, tool_calls, sensor_fires (``fired==True``),
  sensor_halts (``outcome.kind == "halt"``), and compactions
  (``operation.kind == "compaction"``).
* :meth:`InMemoryObservabilityProvider.get_sessions` filters by ``since``
  (lexical compare of timestamps) and by ``outcome``.
* Observability is **passive** — documented, not enforced.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .guide_registry import (
    GuideId,
    SessionOutcome,
    SessionOutcomeFailure,
    SessionOutcomePartial,
)
from .harness import SessionId, TaskId
from .memory import Timestamp
from .middleware import HookPoint, MiddlewareDecision
from .model import StopReason
from .sensor import SensorId, SensorKind, SensorOutcome, SensorTrigger

# ============================================================================
# Identity
# ============================================================================

SpanId = NewType("SpanId", str)
"""Stable identifier for a single emitted span."""


def new_span_id(s: str) -> SpanId:
    return SpanId(s)


# ============================================================================
# Pydantic base
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# SpanKind
# ============================================================================


class SpanKind(str, Enum):
    SESSION = "session"
    TURN = "turn"
    TOOL_CALL = "tool_call"
    SENSOR_EVALUATION = "sensor_evaluation"
    CONTEXT_ASSEMBLY = "context_assembly"
    COMPACTION = "compaction"
    MIDDLEWARE_HOOK = "middleware_hook"
    GUIDE_SELECTION = "guide_selection"
    MEMORY_QUERY = "memory_query"
    MEMORY_WRITE = "memory_write"
    # Emitted by ``PatchToolCallsMiddleware`` whenever it mutates a tool call
    # (issue #28). Always carries a :class:`PatchSpan` at
    # :attr:`SpanLevel.WARN`.
    PATCH = "patch"
    # Emitted when a compaction summary is accepted despite failing verification
    # (issue #46). Always carries a :class:`WarnSpan` at :attr:`SpanLevel.WARN`.
    WARN = "warn"


# ============================================================================
# SpanStatus (discriminated union on ``kind``)
# ============================================================================


class SpanStatusOk(_Model):
    kind: Literal["ok"] = "ok"


class SpanStatusError(_Model):
    kind: Literal["error"] = "error"
    message: str


class SpanStatusHalted(_Model):
    kind: Literal["halted"] = "halted"
    reason: str


SpanStatus = Annotated[
    SpanStatusOk | SpanStatusError | SpanStatusHalted,
    Field(discriminator="kind"),
]


# ============================================================================
# SpanBase
# ============================================================================


@dataclass
class SpanBase:
    """Identity, timing, and status shared by every span payload."""

    span_id: SpanId
    session_id: SessionId
    task_id: TaskId
    kind: SpanKind
    started_at: Timestamp
    ended_at: Timestamp
    duration_ms: int = 0
    status: SpanStatus = field(default_factory=SpanStatusOk)
    parent_span_id: SpanId | None = None

    @staticmethod
    def new_root(
        span_id: SpanId,
        session_id: SessionId,
        task_id: TaskId,
        kind: SpanKind,
        started_at: Timestamp,
    ) -> SpanBase:
        return SpanBase(
            span_id=span_id,
            session_id=session_id,
            task_id=task_id,
            kind=kind,
            started_at=started_at,
            ended_at=started_at,
            duration_ms=0,
            status=SpanStatusOk(),
            parent_span_id=None,
        )

    @staticmethod
    def new_child(
        span_id: SpanId,
        parent: SpanBase,
        kind: SpanKind,
        started_at: Timestamp,
    ) -> SpanBase:
        return SpanBase(
            span_id=span_id,
            session_id=parent.session_id,
            task_id=parent.task_id,
            kind=kind,
            started_at=started_at,
            ended_at=started_at,
            duration_ms=0,
            status=SpanStatusOk(),
            parent_span_id=parent.span_id,
        )

    def finish(
        self,
        ended_at: Timestamp,
        status: SpanStatus,
        duration_ms: int,
    ) -> None:
        self.ended_at = ended_at
        self.status = status
        self.duration_ms = duration_ms


# ============================================================================
# LLM-native content capture (issue #64)
# ============================================================================
#
# Opt-in capture of conversation/tool-call **content** following the
# OpenTelemetry GenAI semantic conventions. OFF by default. Plumbed into the
# existing span payloads (no Protocol-signature changes) and serialized into the
# durable JSONL only when present.
#
# Resolved maintainer decisions (mirror the Rust reference byte-for-byte):
#   1. Canonical convention is pure OTel ``gen_ai.*`` events (no OpenInference).
#   2. Routing is the single configurable ``SPORE_OTLP_ENDPOINT`` (no fan-out).
#   3. Truncation default is 8192 UTF-8 bytes, marker ``...[truncated]``,
#      clipped at a char boundary; override ``SPORE_TRACE_CONTENT_MAX_LEN``;
#      guard ``SPORE_TRACE_CONTENT`` (default OFF).
#   4. The turn span carries the model output + requested tool calls only; the
#      tool-call span carries the tool args + result. No assembled input history.

TRUNCATION_MARKER = "...[truncated]"
"""The exact ASCII marker appended to any captured field that was clipped by
:func:`truncate_field`. Cross-language ground truth — never change the bytes."""

_DEFAULT_CONTENT_MAX_LEN = 8192
"""Default content-field cap, in UTF-8 bytes (maintainer decision 3)."""


def truncate_field(s: str, max_len: int) -> tuple[str, bool]:
    """Clip ``s`` to at most ``max_len`` UTF-8 **bytes**, appending
    :data:`TRUNCATION_MARKER` when (and only when) a clip occurred. Returns
    ``(clipped, was_truncated)``.

    Pure and deterministic — this is the cross-language ground truth:

    * Measurement is in UTF-8 bytes, not characters.
    * When ``s`` fits the budget, returns ``s`` unchanged with ``False``.
    * Otherwise clips to the largest valid UTF-8 char boundary ``<= max_len``
      (never splitting a multibyte char — backs off to the previous boundary)
      and appends the marker. The marker is appended AFTER the byte budget, so
      the returned string may exceed ``max_len`` bytes by the marker's length;
      the budget bounds the *captured payload*, not the marker.
    """
    encoded = s.encode("utf-8")
    if len(encoded) <= max_len:
        return (s, False)
    # Back off to the largest char boundary <= max_len. A UTF-8 continuation
    # byte has its top two bits set to ``10`` (``0b10xxxxxx``); a boundary is any
    # index that is not a continuation byte.
    boundary = max_len
    while boundary > 0 and (encoded[boundary] & 0xC0) == 0x80:
        boundary -= 1
    clipped = encoded[:boundary].decode("utf-8")
    return (clipped + TRUNCATION_MARKER, True)


@dataclass(frozen=True)
class ContentCaptureConfig:
    """Opt-in guard + truncation limit for LLM-native content capture
    (issue #64).

    Content capture is OFF by default. When :attr:`enabled` is ``False`` the
    harness populates none of the ``gen_ai.*`` content fields, so the durable
    JSONL stays byte-identical to the pre-#64 metrics-only output.
    """

    enabled: bool = False
    """Whether to capture message / tool-call content at all. Default ``False``."""
    max_field_len: int = _DEFAULT_CONTENT_MAX_LEN
    """Maximum UTF-8 byte length of any single captured field before
    :func:`truncate_field` clips it. Default ``8192``."""

    @classmethod
    def from_env(cls) -> ContentCaptureConfig:
        """Read the config from the environment:

        * ``SPORE_TRACE_CONTENT`` — ``1``/``true``/``yes``/``on``
          (case-insensitive) enables capture; anything else (or unset) leaves
          it OFF.
        * ``SPORE_TRACE_CONTENT_MAX_LEN`` — parsed as ``int``; falls back to the
          8192-byte default when unset or unparseable.
        """
        raw = os.environ.get("SPORE_TRACE_CONTENT")
        enabled = raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")
        max_field_len = _DEFAULT_CONTENT_MAX_LEN
        raw_max = os.environ.get("SPORE_TRACE_CONTENT_MAX_LEN")
        if raw_max is not None:
            try:
                max_field_len = int(raw_max.strip())
            except ValueError:
                max_field_len = _DEFAULT_CONTENT_MAX_LEN
        return cls(enabled=enabled, max_field_len=max_field_len)


class GenAiRole(str, Enum):
    """Role of a captured message. The bare-string values map onto the
    conventional GenAI span-event names (``gen_ai.<role>.message``)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def event_name(self) -> str:
        """The conventional OTel GenAI span-event name for this role, e.g.
        ``gen_ai.assistant.message``."""
        return f"gen_ai.{self.value}.message"


@dataclass
class GenAiMessage:
    """One captured conversation message (issue #64)."""

    role: GenAiRole
    content: str
    truncated: bool
    """``True`` when ``content`` was clipped by :func:`truncate_field`."""


@dataclass
class ToolCallContent:
    """A requested tool call captured on a :class:`TurnSpan` (issue #64).

    When clipped, the arguments are stored as a JSON string value carrying the
    truncation marker (a structured value cannot be clipped in place), and
    :attr:`arguments_truncated` is ``True``."""

    name: str
    arguments: Any
    arguments_truncated: bool


@dataclass
class ToolResultContent:
    """A tool result body captured on a :class:`ToolCallSpan` (issue #64)."""

    content: str
    truncated: bool
    """``True`` when ``content`` was clipped by :func:`truncate_field`."""


# ============================================================================
# Span payload dataclasses
# ============================================================================


@dataclass
class TurnSpan:
    base: SpanBase
    turn_number: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: StopReason
    tool_calls_requested: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    # The model's output text for this turn (issue #64). Captured only when
    # ``ContentCaptureConfig.enabled``; ``None`` keeps the line pre-#64-identical.
    output_text: GenAiMessage | None = None
    # The tool calls the model requested this turn (issue #64). Captured only
    # when content capture is enabled.
    tool_calls: list[ToolCallContent] | None = None
    # The assembled INPUT messages the model saw this turn — the full prompt
    # (issue #64). First element is the system prompt; the rest mirror the
    # assembled conversation history in order. Captured only when content
    # capture is enabled; ``None`` keeps the line pre-#64-identical.
    input_messages: list[GenAiMessage] | None = None


@dataclass
class ToolCallSpan:
    base: SpanBase
    tool_name: str
    call_id: str
    parameters_size_bytes: int
    output_size_bytes: int
    truncated: bool
    sandbox_mode: str
    sandbox_violations: list[str] = field(default_factory=list)
    # The tool-call arguments (issue #64). Captured only when content capture is
    # enabled; ``None`` keeps the line pre-#64-identical.
    arguments: ToolCallContent | None = None
    # The tool result body (issue #64). Captured only when content capture is
    # enabled.
    result: ToolResultContent | None = None


@dataclass
class SensorSpan:
    base: SpanBase
    sensor_id: SensorId
    sensor_kind: SensorKind
    trigger: SensorTrigger
    outcome: SensorOutcome
    fired: bool


# ============================================================================
# ContextOperation (discriminated union on ``kind``)
# ============================================================================


class ContextOperationAssembly(_Model):
    kind: Literal["assembly"] = "assembly"
    guides_loaded: int
    memory_items_loaded: int
    tools_loaded: int


class ContextOperationToolResultAppended(_Model):
    kind: Literal["tool_result_appended"] = "tool_result_appended"
    tool_name: str
    truncated: bool


class ContextOperationCompaction(_Model):
    kind: Literal["compaction"] = "compaction"
    messages_removed: int
    tokens_reclaimed: int


class ContextOperationSkillInjected(_Model):
    kind: Literal["skill_injected"] = "skill_injected"
    guide_id: GuideId


ContextOperation = Annotated[
    ContextOperationAssembly
    | ContextOperationToolResultAppended
    | ContextOperationCompaction
    | ContextOperationSkillInjected,
    Field(discriminator="kind"),
]


@dataclass
class ContextSpan:
    base: SpanBase
    operation: ContextOperation
    tokens_before: int
    tokens_after: int
    utilization_before: float
    utilization_after: float


@dataclass
class MiddlewareSpan:
    base: SpanBase
    hook: HookPoint
    decision: MiddlewareDecision


# ============================================================================
# Patch observability (issue #28)
# ============================================================================
#
# ``PatchToolCallsMiddleware`` is an always-on, highest-priority ``BeforeTool``
# action mutator that silently rewrites malformed or dangling tool calls before
# the sandbox and sensors see them. An always-on mutator with no observability
# is a footgun: the trace would show the patched call as if the model had sent
# it. Issue #28 closes that gap with the types below.
#
# Rules enforced (mirrored by the unit tests):
#   R1  every patch emits exactly one ``Warn``-level patch span.
#   R2  no patch → no span emitted.
#   R3  the span records BOTH the original and the patched parameters.
#   R4  ``patch_type`` is classified correctly for each variant.
#   R5  the trace (:meth:`get_trace`) contains the patch event.
#   R6  :attr:`SessionMetrics.patch_count` counts patch spans for the session.
#   R7  :attr:`SessionMetrics.patch_rate` = patches / total tool calls
#       (0.0 when there are no tool calls — never divides by zero).
#   R8  :attr:`SessionMetrics.patches_by_tool` breaks the count down per tool.
#   R9  a batch of N patched calls emits N patch spans.


class SpanLevel(str, Enum):
    """Severity of an emitted span. Patch spans are always
    :attr:`SpanLevel.WARN` per issue #28; this enum keeps the level
    orthogonal to :data:`SpanStatus` so a successful (``ok``) trace can still
    surface warn-level patch events."""

    INFO = "info"
    WARN = "warn"


# PatchType — discriminated union on ``kind`` (snake_case tags).


class PatchTypeMalformedJson(_Model):
    """The raw tool-call arguments failed to parse as JSON; a repair was
    attempted. ``error`` is the parse error that was recovered from."""

    kind: Literal["malformed_json"] = "malformed_json"
    error: str


class PatchTypeDanglingToolCall(_Model):
    """The call was structurally incomplete (e.g. empty tool name) and was
    completed with defaults. ``reason`` explains what was missing."""

    kind: Literal["dangling_tool_call"] = "dangling_tool_call"
    reason: str


class PatchTypeParameterCoercion(_Model):
    """A parameter value was coerced from one type to another to satisfy the
    tool schema."""

    kind: Literal["parameter_coercion"] = "parameter_coercion"
    field: str
    from_: str = Field(alias="from")
    to: str


PatchType = Annotated[
    PatchTypeMalformedJson | PatchTypeDanglingToolCall | PatchTypeParameterCoercion,
    Field(discriminator="kind"),
]


@dataclass
class PatchSpan:
    """One observability event per tool-call patch (issue #28).

    Carries both the original parameters (what the model sent) and the
    patched parameters (what was dispatched) so the trace shows the diff,
    never just the patched call. :attr:`level` is always
    :attr:`SpanLevel.WARN`.
    """

    base: SpanBase
    call_id: str
    tool_name: str
    original_parameters: dict[str, Any]
    patched_parameters: dict[str, Any]
    patch_type: PatchType
    level: SpanLevel = SpanLevel.WARN


# ============================================================================
# Warn observability (issue #46)
# ============================================================================
#
# A compaction summary may fail post-compaction verification on every attempt.
# A blocked compaction is worse than an imperfect one, so the harness accepts
# the failing summary anyway and emits exactly one warn-level event recording
# what was still missing. The shape mirrors :class:`PatchSpan` (always
# :attr:`SpanLevel.WARN`) but keeps warns that are not tied to a single tool
# call in their own type.
#
# Rules enforced (mirrored by the unit tests):
#   W1  exhausting ``max_compaction_attempts`` emits exactly one warn span.
#   W2  the warn span carries a :class:`WarnEventCompactionVerificationFailed`
#       with the still-missing items and ``accepted_anyway=True``.
#   W3  :attr:`SessionMetrics.compaction_verification_failures` counts these
#       spans for the session (mirrors how ``compactions`` is derived).
#   W4  :meth:`ObservabilityProvider.emit_warn` has a default no-op body so
#       providers predating #46 keep compiling and behave unchanged.


class WarnEventCompactionVerificationFailed(_Model):
    """A compaction summary failed verification on every attempt and was
    accepted as-is (issue #46). ``missing_items`` are the preservation-list
    terms still absent from the final summary; ``accepted_anyway`` is always
    ``True`` for this variant (the harness never blocks on compaction)."""

    warn: Literal["compaction_verification_failed"] = "compaction_verification_failed"
    missing_items: list[str]
    accepted_anyway: bool


class WarnEventHillClimbingIteration(_Model):
    """One iteration of a ``HillClimbing`` loop strategy run (issue #60). Emitted
    fire-and-forget after each iteration's metric evaluation so the run is
    traceable per-iteration with its metric value and delta. ``iteration`` is the
    0-based iteration index (``0`` = the pure baseline). ``metric_value`` and
    ``delta`` are ``None`` on crashed/timeout iterations (no comparable metric);
    ``delta`` is also ``None`` for the baseline. ``status`` is the snake_case
    :data:`~spore_core.metric.IterationStatus` string the harness recorded
    (``kept``/``discarded``/``crashed``/``timeout``). Mirrors Rust's
    ``WarnEvent::HillClimbingIteration``."""

    warn: Literal["hill_climbing_iteration"] = "hill_climbing_iteration"
    iteration: int
    metric_value: float | None = None
    delta: float | None = None
    status: str
    reverted: bool


WarnEvent = Annotated[
    WarnEventCompactionVerificationFailed | WarnEventHillClimbingIteration,
    Field(discriminator="warn"),
]
"""A warn-level, fire-and-forget observability event. Future warn classes add
members; consumers must keep a fallthrough."""


@dataclass
class WarnSpan:
    """One warn-level observability span (issue #46). Carries a
    :class:`SpanBase` for trace correlation, the classified :data:`WarnEvent`,
    and a hardcoded :attr:`SpanLevel.WARN`."""

    base: SpanBase
    event: WarnEvent
    level: SpanLevel = SpanLevel.WARN


Span = TurnSpan | ToolCallSpan | SensorSpan | ContextSpan | MiddlewareSpan | PatchSpan | WarnSpan
"""Heterogeneous span type returned by :meth:`ObservabilityProvider.get_trace`."""


# ============================================================================
# SessionMetrics
# ============================================================================


@dataclass
class SessionMetrics:
    session_id: SessionId
    task_id: TaskId
    total_turns: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    total_duration_ms: int
    tool_calls: int
    sensor_fires: int
    sensor_halts: int
    compactions: int
    outcome: SessionOutcome
    guides_used: list[GuideId] = field(default_factory=list)
    # Number of tool-call patches in the session (issue #28).
    patch_count: int = 0
    # ``patch_count / tool_calls``. ``0.0`` when there are no tool calls.
    patch_rate: float = 0.0
    # Patch count broken down by tool name.
    patches_by_tool: dict[str, int] = field(default_factory=dict)
    # Number of compactions whose summary failed verification on every attempt
    # and was accepted anyway (issue #46). Derived by counting
    # :class:`WarnSpan`s carrying
    # :class:`WarnEventCompactionVerificationFailed`, mirroring how
    # ``compactions`` is derived from compaction context spans.
    compaction_verification_failures: int = 0


# ============================================================================
# PricingTable
# ============================================================================


@dataclass(frozen=True)
class PricingTable:
    """Provider-specific token → USD lookup. Production callers inject a
    real table; :attr:`DEFAULT` is a conservative zero-cost pass-through."""

    input_per_million: float
    output_per_million: float
    cache_read_per_million: float
    cache_write_per_million: float

    DEFAULT: ClassVar[PricingTable]

    def cost_for(
        self,
        input: int,
        output: int,
        cache_read: int | None = None,
        cache_write: int | None = None,
    ) -> float:
        per = 1_000_000.0
        return (
            self.input_per_million * input / per
            + self.output_per_million * output / per
            + self.cache_read_per_million * (cache_read or 0) / per
            + self.cache_write_per_million * (cache_write or 0) / per
        )


# Cannot assign DEFAULT on a frozen dataclass post-definition without
# bypassing __setattr__; use object.__setattr__ trick via __init_subclass__,
# or just attach after class body via the descriptor protocol.
PricingTable.DEFAULT = PricingTable(  # type: ignore[misc]
    input_per_million=0.0,
    output_per_million=0.0,
    cache_read_per_million=0.0,
    cache_write_per_million=0.0,
)


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class ObservabilityProvider(Protocol):
    """Structured observability surface. All ``emit_*`` methods are
    fire-and-forget; they must never block the harness loop.
    Implementations buffer internally and flush asynchronously via
    :meth:`flush_session`."""

    def emit_turn(self, span: TurnSpan) -> None: ...

    def emit_tool_call(self, span: ToolCallSpan) -> None: ...

    def emit_sensor(self, span: SensorSpan) -> None: ...

    def emit_context(self, span: ContextSpan) -> None: ...

    def emit_middleware(self, span: MiddlewareSpan) -> None: ...

    def emit_patch(self, span: PatchSpan) -> None: ...

    def emit_warn(self, span: WarnSpan) -> None:
        """Record a warn-level event (issue #46). Default no-op so providers
        predating #46 keep satisfying the Protocol and behave unchanged;
        :class:`InMemoryObservabilityProvider` and the durable outbox override
        it."""
        _ = span

    def set_session_outcome(self, session_id: SessionId, outcome: SessionOutcome) -> None:
        """Record the terminal outcome for a session before flush. Default
        no-op so providers that do not track outcomes still satisfy the
        Protocol; :class:`InMemoryObservabilityProvider` overrides it."""
        _ = (session_id, outcome)

    async def flush_session(self, session_id: SessionId) -> None: ...

    async def get_session_metrics(self, session_id: SessionId) -> SessionMetrics | None: ...

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]: ...

    async def get_trace(self, session_id: SessionId) -> list[Span]: ...

    async def list_unflushed_sessions(self) -> list[SessionId]:
        """Session ids whose durable outbox has a ``trace.jsonl`` but no
        ``.flushed`` marker (issue #33). Only the durable-outbox provider has
        unflushed on-disk sessions; the default returns an empty list."""
        return []

    async def cleanup_session(self, session_id: SessionId) -> None:
        """Delete a session's durable outbox (issue #33). The provider NEVER
        auto-deletes; the caller drives cleanup. Default: no-op."""
        _ = session_id


# ============================================================================
# InMemoryObservabilityProvider — reference implementation
# ============================================================================


def _outcomes_equal(a: SessionOutcome, b: SessionOutcome) -> bool:
    """Structural equality across pydantic outcome variants (kind + reason)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, SessionOutcomeFailure) and isinstance(b, SessionOutcomeFailure):
        return a.reason == b.reason
    return True


class InMemoryObservabilityProvider:
    """Reference :class:`ObservabilityProvider` implementation. In-memory;
    suitable for tests and short-lived processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: list[TurnSpan] = []
        self._tool_calls: list[ToolCallSpan] = []
        self._sensors: list[SensorSpan] = []
        self._contexts: list[ContextSpan] = []
        self._middlewares: list[MiddlewareSpan] = []
        self._patches: list[PatchSpan] = []
        self._warns: list[WarnSpan] = []
        # Per-session insertion-ordered (kind, span_id) tuples — the
        # trace-analyzer feed.
        self._trace_order: dict[SessionId, list[tuple[SpanKind, SpanId]]] = {}
        self._flushed: dict[SessionId, bool] = {}
        # Per-session terminal outcome, set by the harness after AfterSession.
        self._outcomes: dict[SessionId, SessionOutcome] = {}
        # Per-session guides used, populated at session start.
        self._guides_used: dict[SessionId, list[GuideId]] = {}

    # ── post-hoc recorders ─────────────────────────────────────────────────

    def set_session_outcome(self, session_id: SessionId, outcome: SessionOutcome) -> None:
        """Record the terminal outcome so :meth:`get_session_metrics` can
        surface it. The harness calls this once, after ``fire_after_session``.
        """
        with self._lock:
            self._outcomes[session_id] = outcome

    def record_guides_used(self, session_id: SessionId, guides: list[GuideId]) -> None:
        """Record the guides selected for a session. Called once at session
        start."""
        with self._lock:
            self._guides_used[session_id] = list(guides)

    def patch_spans(self, session_id: SessionId) -> list[PatchSpan]:
        """All recorded patch spans for a session, in insertion order
        (issue #28). Lets callers inspect the original/patched diff and the
        classified :data:`PatchType` without reconstructing them from the
        heterogeneous trace."""
        with self._lock:
            return [p for p in self._patches if p.base.session_id == session_id]

    def warn_spans(self, session_id: SessionId) -> list[WarnSpan]:
        """All recorded warn spans for a session, in insertion order
        (issue #46)."""
        with self._lock:
            return [w for w in self._warns if w.base.session_id == session_id]

    def context_spans(self, session_id: SessionId) -> list[ContextSpan]:
        """All recorded context spans (assembly + compaction) for a session, in
        insertion order. Lets callers spot-check a ``Compaction`` operation and
        its reclaimed-token accounting without reconstructing it from the
        heterogeneous trace (#57)."""
        with self._lock:
            return [c for c in self._contexts if c.base.session_id == session_id]

    # ── helpers ────────────────────────────────────────────────────────────

    def _push_order(self, sid: SessionId, kind: SpanKind, span_id: SpanId) -> None:
        self._trace_order.setdefault(sid, []).append((kind, span_id))

    # ── emit_* (fire-and-forget) ───────────────────────────────────────────

    def emit_turn(self, span: TurnSpan) -> None:
        with self._lock:
            self._push_order(span.base.session_id, SpanKind.TURN, span.base.span_id)
            self._turns.append(span)

    def emit_tool_call(self, span: ToolCallSpan) -> None:
        with self._lock:
            self._push_order(span.base.session_id, SpanKind.TOOL_CALL, span.base.span_id)
            self._tool_calls.append(span)

    def emit_sensor(self, span: SensorSpan) -> None:
        with self._lock:
            self._push_order(span.base.session_id, SpanKind.SENSOR_EVALUATION, span.base.span_id)
            self._sensors.append(span)

    def emit_context(self, span: ContextSpan) -> None:
        with self._lock:
            kind = (
                SpanKind.COMPACTION
                if isinstance(span.operation, ContextOperationCompaction)
                else SpanKind.CONTEXT_ASSEMBLY
            )
            self._push_order(span.base.session_id, kind, span.base.span_id)
            self._contexts.append(span)

    def emit_middleware(self, span: MiddlewareSpan) -> None:
        with self._lock:
            self._push_order(span.base.session_id, SpanKind.MIDDLEWARE_HOOK, span.base.span_id)
            self._middlewares.append(span)

    def emit_patch(self, span: PatchSpan) -> None:
        with self._lock:
            self._push_order(span.base.session_id, SpanKind.PATCH, span.base.span_id)
            self._patches.append(span)

    def emit_warn(self, span: WarnSpan) -> None:
        with self._lock:
            self._push_order(span.base.session_id, SpanKind.WARN, span.base.span_id)
            self._warns.append(span)

    # ── flush_session (idempotent) ─────────────────────────────────────────

    async def flush_session(self, session_id: SessionId) -> None:
        with self._lock:
            self._flushed[session_id] = True

    # ── get_session_metrics ────────────────────────────────────────────────

    async def get_session_metrics(self, session_id: SessionId) -> SessionMetrics | None:
        with self._lock:
            turns = [t for t in self._turns if t.base.session_id == session_id]
            tool_calls = [c for c in self._tool_calls if c.base.session_id == session_id]
            sensors = [s for s in self._sensors if s.base.session_id == session_id]
            contexts = [c for c in self._contexts if c.base.session_id == session_id]
            patches = [p for p in self._patches if p.base.session_id == session_id]
            warns = [w for w in self._warns if w.base.session_id == session_id]
            outcome = self._outcomes.get(session_id)
            guides = list(self._guides_used.get(session_id, []))

        if not turns and outcome is None:
            return None

        task_id = turns[0].base.task_id if turns else TaskId("")
        input_tokens = sum(t.input_tokens for t in turns)
        output_tokens = sum(t.output_tokens for t in turns)
        cost = sum(t.cost_usd for t in turns)
        duration = sum(t.base.duration_ms for t in turns) + sum(
            c.base.duration_ms for c in tool_calls
        )
        sensor_fires = sum(1 for s in sensors if s.fired)
        sensor_halts = sum(1 for s in sensors if s.outcome == SensorOutcome.HALT)
        compactions = sum(
            1 for c in contexts if isinstance(c.operation, ContextOperationCompaction)
        )
        patch_count = len(patches)
        # R7: guard divide-by-zero; denominator is all tool-call spans.
        patch_rate = patch_count / len(tool_calls) if tool_calls else 0.0
        patches_by_tool: dict[str, int] = {}
        for p in patches:
            patches_by_tool[p.tool_name] = patches_by_tool.get(p.tool_name, 0) + 1
        compaction_verification_failures = sum(
            1 for w in warns if isinstance(w.event, WarnEventCompactionVerificationFailed)
        )

        return SessionMetrics(
            session_id=session_id,
            task_id=task_id,
            total_turns=len(turns),
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cost_usd=cost,
            total_duration_ms=duration,
            tool_calls=len(tool_calls),
            sensor_fires=sensor_fires,
            sensor_halts=sensor_halts,
            compactions=compactions,
            outcome=outcome if outcome is not None else SessionOutcomePartial(),
            guides_used=guides,
            patch_count=patch_count,
            patch_rate=patch_rate,
            patches_by_tool=patches_by_tool,
            compaction_verification_failures=compaction_verification_failures,
        )

    # ── get_sessions ───────────────────────────────────────────────────────

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]:
        # ``domain`` is part of the spec surface; the in-memory reference
        # has no domain index so it is accepted but not used.
        _ = domain
        with self._lock:
            ids_in_order: list[SessionId] = []
            seen: set[SessionId] = set()
            for t in self._turns:
                if str(t.base.started_at) < str(since):
                    continue
                sid = t.base.session_id
                if sid in seen:
                    continue
                seen.add(sid)
                ids_in_order.append(sid)
            ids_in_order.sort()

        out: list[SessionMetrics] = []
        for sid in ids_in_order:
            m = await self.get_session_metrics(sid)
            if m is None:
                continue
            if outcome is not None and not _outcomes_equal(m.outcome, outcome):
                continue
            out.append(m)
        return out

    # ── get_trace ──────────────────────────────────────────────────────────

    async def get_trace(self, session_id: SessionId) -> list[Span]:
        with self._lock:
            order = list(self._trace_order.get(session_id, []))
            turns = {t.base.span_id: t for t in self._turns}
            tool_calls = {c.base.span_id: c for c in self._tool_calls}
            sensors = {s.base.span_id: s for s in self._sensors}
            contexts = {c.base.span_id: c for c in self._contexts}
            middlewares = {m.base.span_id: m for m in self._middlewares}
            patches = {p.base.span_id: p for p in self._patches}
            warns = {w.base.span_id: w for w in self._warns}

        out: list[Span] = []
        for kind, span_id in order:
            if kind is SpanKind.TURN and span_id in turns:
                out.append(turns[span_id])
            elif kind is SpanKind.TOOL_CALL and span_id in tool_calls:
                out.append(tool_calls[span_id])
            elif kind is SpanKind.SENSOR_EVALUATION and span_id in sensors:
                out.append(sensors[span_id])
            elif kind in (SpanKind.CONTEXT_ASSEMBLY, SpanKind.COMPACTION) and span_id in contexts:
                out.append(contexts[span_id])
            elif kind is SpanKind.MIDDLEWARE_HOOK and span_id in middlewares:
                out.append(middlewares[span_id])
            elif kind is SpanKind.PATCH and span_id in patches:
                out.append(patches[span_id])
            elif kind is SpanKind.WARN and span_id in warns:
                out.append(warns[span_id])
        return out

    # ── outbox stubs (issue #33) ────────────────────────────────────────────
    # The in-memory provider keeps no durable outbox; these satisfy the
    # :class:`ObservabilityProvider` Protocol with the documented defaults.

    async def list_unflushed_sessions(self) -> list[SessionId]:
        return []

    async def cleanup_session(self, session_id: SessionId) -> None:
        _ = session_id


__all__ = [
    "TRUNCATION_MARKER",
    "ContentCaptureConfig",
    "ContextOperation",
    "ContextOperationAssembly",
    "ContextOperationCompaction",
    "ContextOperationSkillInjected",
    "ContextOperationToolResultAppended",
    "ContextSpan",
    "GenAiMessage",
    "GenAiRole",
    "InMemoryObservabilityProvider",
    "MiddlewareSpan",
    "ObservabilityProvider",
    "PatchSpan",
    "PatchType",
    "PatchTypeDanglingToolCall",
    "PatchTypeMalformedJson",
    "PatchTypeParameterCoercion",
    "PricingTable",
    "SensorSpan",
    "SessionMetrics",
    "Span",
    "SpanBase",
    "SpanId",
    "SpanKind",
    "SpanLevel",
    "SpanStatus",
    "SpanStatusError",
    "SpanStatusHalted",
    "SpanStatusOk",
    "ToolCallContent",
    "ToolCallSpan",
    "ToolResultContent",
    "TurnSpan",
    "WarnEvent",
    "WarnEventCompactionVerificationFailed",
    "WarnEventHillClimbingIteration",
    "WarnSpan",
    "new_span_id",
    "truncate_field",
]
