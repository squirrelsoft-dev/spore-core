//! Issue #12 — `ObservabilityProvider`: structured recording of all harness
//! activity.
//!
//! Every observable harness operation emits one [`Span`]. Spans carry
//! identity (session, task, parent span), timing, status, and operation-
//! specific payload. Aggregates roll up to [`SessionMetrics`] for the
//! improvement loop.
//!
//! See `docs/harness-engineering-concepts.md` § "ObservabilityProvider" for
//! the authoritative rules. This module ships:
//!   - The full [`ObservabilityProvider`] trait with per-span-kind
//!     `emit_*` methods, `flush_session`, and query methods.
//!   - All span payload types from the spec ([`TurnSpan`], [`ToolCallSpan`],
//!     [`SensorSpan`], [`ContextSpan`], [`MiddlewareSpan`]).
//!   - [`InMemoryObservabilityProvider`] — buffered in-memory backend used
//!     for tests and short-lived processes. The OTLP/JSONL backends live in
//!     sibling crates; they implement the same trait.
//!   - [`PricingTable`] — provider-specific token → USD lookup, injected at
//!     construction so `cost_usd` is a first-class span field (per spec).
//!
//! ## Rules enforced
//!   - `emit_*` methods are **fire-and-forget**. The standard implementation
//!     pushes to an `Arc<Mutex<...>>` buffer and returns synchronously.
//!     Async flush moves spans to permanent storage.
//!   - Every harness operation type has a corresponding `emit_*` method —
//!     nothing is exempt.
//!   - `cost_usd` on [`TurnSpan`] is computed at emit time from the
//!     pricing table; the harness does not estimate it. Cache read/write
//!     tokens count against cache pricing, never input/output pricing.
//!   - Observability is **passive**: the trait has no mutator that affects
//!     harness behavior. Calling `emit_*` cannot change a `TurnResult` or
//!     a `ToolOutput`.
//!   - `get_trace` returns the full [`Span`] tree for a session in
//!     insertion order — the trace analyzer reconstructs hierarchy via
//!     `parent_span_id`.
//!   - `flush_session` is idempotent — calling it twice for the same
//!     session is a no-op the second time.
//!
//! ## Implementor notes
//!   - The new-span helpers ([`SpanBase::new_root`], [`SpanBase::new_child`])
//!     stamp `started_at` and leave `ended_at` to the caller via
//!     [`SpanBase::finish`]. This matches the harness pattern of "open a
//!     span at the start of an operation, finish it at the end."
//!   - [`InMemoryObservabilityProvider`] uses RFC 3339 timestamps so spans
//!     compare lexically; production OTLP backends use nanosecond
//!     monotonic clocks.
//!   - [`PricingTable::DEFAULT`] is a conservative pass-through that
//!     reports zero cost — production callers inject a real table.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::guide_registry::{GuideId, SessionOutcome};
use crate::harness::{BoxFut, SessionId, TaskId};
use crate::memory::Timestamp;
use crate::middleware::{HookPoint, MiddlewareDecision};
use crate::model::StopReason;
use crate::sensor::{SensorId, SensorKind, SensorOutcome, SensorTrigger};

// ============================================================================
// Identity
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(pub String);

impl SpanId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// Span enums and base
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind {
    Session,
    Turn,
    ToolCall,
    SensorEvaluation,
    ContextAssembly,
    Compaction,
    MiddlewareHook,
    GuideSelection,
    MemoryQuery,
    MemoryWrite,
    /// Emitted by `PatchToolCallsMiddleware` whenever it mutates a tool call
    /// (issue #28). Always carries a [`PatchSpan`] at [`SpanLevel::Warn`].
    Patch,
    /// Emitted by the harness compaction loop when a summary is accepted
    /// despite failing verification (issue #46). Always carries a [`WarnSpan`]
    /// at [`SpanLevel::Warn`].
    Warn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpanStatus {
    Ok,
    Error { message: String },
    Halted { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanBase {
    pub span_id: SpanId,
    #[serde(default)]
    pub parent_span_id: Option<SpanId>,
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub kind: SpanKind,
    pub started_at: Timestamp,
    pub ended_at: Timestamp,
    pub duration_ms: u64,
    pub status: SpanStatus,
}

impl SpanBase {
    pub fn new_root(
        span_id: SpanId,
        session_id: SessionId,
        task_id: TaskId,
        kind: SpanKind,
        started_at: Timestamp,
    ) -> Self {
        Self {
            span_id,
            parent_span_id: None,
            session_id,
            task_id,
            kind,
            started_at: started_at.clone(),
            ended_at: started_at,
            duration_ms: 0,
            status: SpanStatus::Ok,
        }
    }

    pub fn new_child(
        span_id: SpanId,
        parent: &SpanBase,
        kind: SpanKind,
        started_at: Timestamp,
    ) -> Self {
        Self {
            span_id,
            parent_span_id: Some(parent.span_id.clone()),
            session_id: parent.session_id.clone(),
            task_id: parent.task_id.clone(),
            kind,
            started_at: started_at.clone(),
            ended_at: started_at,
            duration_ms: 0,
            status: SpanStatus::Ok,
        }
    }

    pub fn finish(&mut self, ended_at: Timestamp, status: SpanStatus, duration_ms: u64) {
        self.ended_at = ended_at;
        self.status = status;
        self.duration_ms = duration_ms;
    }
}

// ============================================================================
// Span payload types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnSpan {
    pub base: SpanBase,
    pub turn_number: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_tokens: Option<u32>,
    #[serde(default)]
    pub cache_write_tokens: Option<u32>,
    pub cost_usd: f64,
    pub stop_reason: StopReason,
    pub tool_calls_requested: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallSpan {
    pub base: SpanBase,
    pub tool_name: String,
    pub call_id: String,
    pub parameters_size_bytes: usize,
    pub output_size_bytes: usize,
    pub truncated: bool,
    pub sandbox_mode: String,
    #[serde(default)]
    pub sandbox_violations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorSpan {
    pub base: SpanBase,
    pub sensor_id: SensorId,
    pub sensor_kind: SensorKind,
    pub trigger: SensorTrigger,
    pub outcome: SensorOutcome,
    pub fired: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextOperation {
    Assembly {
        guides_loaded: u32,
        memory_items_loaded: u32,
        tools_loaded: u32,
    },
    ToolResultAppended {
        tool_name: String,
        truncated: bool,
    },
    Compaction {
        messages_removed: u32,
        tokens_reclaimed: u32,
    },
    SkillInjected {
        guide_id: GuideId,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextSpan {
    pub base: SpanBase,
    pub operation: ContextOperation,
    pub tokens_before: u32,
    pub tokens_after: u32,
    pub utilization_before: f32,
    pub utilization_after: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiddlewareSpan {
    pub base: SpanBase,
    pub hook: HookPoint,
    pub decision: MiddlewareDecision,
}

// ============================================================================
// Patch observability (issue #28)
// ============================================================================
//
// `PatchToolCallsMiddleware` is an always-on, highest-priority `BeforeTool`
// action mutator that silently rewrites malformed or dangling tool calls
// before the sandbox and sensors see them. An always-on mutator with no
// observability is a footgun: the trace would show the patched call as if the
// model had sent it. Issue #28 closes that gap.
//
// Types added here:
//   - [`SpanLevel`]  — severity tag; patch spans are ALWAYS `Warn`, never
//     `Info`. Distinct from [`SpanStatus`] so the surrounding trace stays
//     `Ok` while the patch event itself is flagged.
//   - [`PatchType`]  — what kind of repair happened (`MalformedJson`,
//     `DanglingToolCall`, `ParameterCoercion`). `#[non_exhaustive]` so new
//     repair classes can be added without breaking downstream matches.
//   - [`PatchSpan`]  — the full event: identity (`base`), the call id and
//     tool name, the original parameters as the model sent them, the patched
//     parameters that were actually dispatched, the classified `patch_type`,
//     and the hardcoded `level: SpanLevel::Warn`.
//
// Trait method added:
//   - [`ObservabilityProvider::emit_patch`] — fire-and-forget, synchronous,
//     mirrors the other `emit_*` methods.
//
// Rules enforced (mirrored by the inline tests):
//   R1  every patch emits exactly one `Warn`-level patch span.
//   R2  no patch → no span emitted.
//   R3  the span records BOTH the original and the patched parameters.
//   R4  `patch_type` is classified correctly for each variant.
//   R5  the trace (`get_trace`) contains the patch event.
//   R6  `SessionMetrics::patch_count` counts patch spans for the session.
//   R7  `SessionMetrics::patch_rate` = patches / total tool calls
//       (0.0 when there are no tool calls — never divides by zero).
//   R8  `SessionMetrics::patches_by_tool` breaks the count down per tool.
//   R9  a batch of N patched calls emits N patch spans.

/// Severity of an emitted span. Patch spans are always [`SpanLevel::Warn`]
/// per issue #28; this enum keeps the level orthogonal to [`SpanStatus`] so a
/// successful (`Ok`) trace can still surface warn-level patch events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanLevel {
    Info,
    Warn,
}

/// Classification of a tool-call patch. `#[non_exhaustive]`: production
/// JSON-repair routines add variants; downstream matches must keep a wildcard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PatchType {
    /// The raw tool-call arguments failed to parse as JSON; a repair was
    /// attempted. `error` is the parse error that was recovered from.
    MalformedJson { error: String },
    /// The call was structurally incomplete (e.g. empty tool name) and was
    /// completed with defaults. `reason` explains what was missing.
    DanglingToolCall { reason: String },
    /// A parameter value was coerced from one type to another to satisfy the
    /// tool schema.
    ParameterCoercion {
        field: String,
        from: String,
        to: String,
    },
}

/// One observability event per tool-call patch (issue #28). Carries both the
/// original parameters (what the model sent) and the patched parameters (what
/// was dispatched) so the trace shows the diff, never just the patched call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatchSpan {
    pub base: SpanBase,
    pub call_id: String,
    pub tool_name: String,
    pub original_parameters: serde_json::Value,
    pub patched_parameters: serde_json::Value,
    pub patch_type: PatchType,
    /// Always [`SpanLevel::Warn`]. Constructed via [`PatchSpan::new`].
    pub level: SpanLevel,
}

impl PatchSpan {
    /// Build a patch span. The level is forced to [`SpanLevel::Warn`]; callers
    /// cannot emit an `Info`-level patch.
    pub fn new(
        base: SpanBase,
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        original_parameters: serde_json::Value,
        patched_parameters: serde_json::Value,
        patch_type: PatchType,
    ) -> Self {
        Self {
            base,
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            original_parameters,
            patched_parameters,
            patch_type,
            level: SpanLevel::Warn,
        }
    }
}

// ============================================================================
// Span trait (for get_trace's heterogeneous return)
// ============================================================================

pub trait Span: std::fmt::Debug + Send + Sync {
    fn base(&self) -> &SpanBase;
    fn kind(&self) -> SpanKind {
        self.base().kind
    }
}

impl Span for TurnSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}
impl Span for ToolCallSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}
impl Span for SensorSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}
impl Span for ContextSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}
impl Span for MiddlewareSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}
impl Span for PatchSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}

// ============================================================================
// Compaction-verification warn (issue #46)
// ============================================================================
//
// The harness compaction loop (issue #29 pseudocode, wired in #46) verifies
// every agent-produced summary with a [`CompactionVerifier`] before accepting
// it. After `max_compaction_attempts` failed verifications the harness accepts
// the summary anyway — a blocked compaction is worse than an imperfect one —
// and emits exactly one warn-level [`WarnSpan::CompactionVerificationFailed`]
// recording the still-missing items and `accepted_anyway: true`.
//
// Rules (mirrored by inline tests in `harness`):
//   W1  a successful (or first-try-passing) compaction emits NO warn span.
//   W2  exhausting attempts emits EXACTLY ONE warn span carrying the final
//       `missing_items` and `accepted_anyway = true`.
//   W3  `SessionMetrics::compaction_verification_failures` counts these spans
//       for the session (mirrors how `compactions` is derived from spans).
//   W4  `emit_warn` has a default no-op body so providers predating #46 keep
//       compiling and behave unchanged.

/// A warn-level, fire-and-forget observability event. `#[non_exhaustive]`:
/// future warn classes add variants; downstream matches must keep a wildcard.
/// The enum-as-event shape mirrors [`PatchSpan`] (always [`SpanLevel::Warn`])
/// but keeps warns that are not tied to a single tool call in their own type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "warn", rename_all = "snake_case")]
#[non_exhaustive]
pub enum WarnEvent {
    /// A compaction summary failed verification on every attempt and was
    /// accepted as-is (issue #46). `missing_items` are the preservation-list
    /// terms still absent from the final summary; `accepted_anyway` is always
    /// `true` for this variant (the harness never blocks on compaction).
    CompactionVerificationFailed {
        missing_items: Vec<String>,
        accepted_anyway: bool,
    },
}

/// One warn-level observability span (issue #46). Carries a [`SpanBase`] for
/// trace correlation, the classified [`WarnEvent`], and a hardcoded
/// `level: SpanLevel::Warn` (constructed via [`WarnSpan::new`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarnSpan {
    pub base: SpanBase,
    pub event: WarnEvent,
    /// Always [`SpanLevel::Warn`]. Constructed via [`WarnSpan::new`].
    pub level: SpanLevel,
}

impl WarnSpan {
    /// Build a warn span. The level is forced to [`SpanLevel::Warn`].
    pub fn new(base: SpanBase, event: WarnEvent) -> Self {
        Self {
            base,
            event,
            level: SpanLevel::Warn,
        }
    }
}

impl Span for WarnSpan {
    fn base(&self) -> &SpanBase {
        &self.base
    }
}

// ============================================================================
// SessionMetrics
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub total_turns: u32,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub total_cost_usd: f64,
    pub total_duration_ms: u64,
    pub tool_calls: u32,
    pub sensor_fires: u32,
    pub sensor_halts: u32,
    pub compactions: u32,
    pub outcome: SessionOutcome,
    #[serde(default)]
    pub guides_used: Vec<GuideId>,
    /// Number of tool-call patches in the session (issue #28).
    #[serde(default)]
    pub patch_count: u32,
    /// `patch_count / tool_calls`. `0.0` when there are no tool calls.
    #[serde(default)]
    pub patch_rate: f32,
    /// Patch count broken down by tool name.
    #[serde(default)]
    pub patches_by_tool: HashMap<String, u32>,
    /// Number of compactions whose summary failed verification on every
    /// attempt and was accepted anyway (issue #46). Derived from
    /// [`WarnSpan`]s carrying [`WarnEvent::CompactionVerificationFailed`],
    /// mirroring how `compactions` is derived from compaction spans.
    #[serde(default)]
    pub compaction_verification_failures: u32,
}

// ============================================================================
// Pricing table
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PricingTable {
    /// USD per 1M input tokens.
    pub input_per_million: f64,
    /// USD per 1M output tokens.
    pub output_per_million: f64,
    /// USD per 1M cache-read tokens (typically 0.1× input price).
    pub cache_read_per_million: f64,
    /// USD per 1M cache-write tokens (typically 1.25× input price).
    pub cache_write_per_million: f64,
}

impl PricingTable {
    /// Conservative zero-cost default. Production callers inject a real table.
    pub const DEFAULT: PricingTable = PricingTable {
        input_per_million: 0.0,
        output_per_million: 0.0,
        cache_read_per_million: 0.0,
        cache_write_per_million: 0.0,
    };

    pub fn cost_for(
        &self,
        input: u32,
        output: u32,
        cache_read: Option<u32>,
        cache_write: Option<u32>,
    ) -> f64 {
        let per_token = |total_per_million: f64| total_per_million / 1_000_000.0;
        per_token(self.input_per_million) * f64::from(input)
            + per_token(self.output_per_million) * f64::from(output)
            + per_token(self.cache_read_per_million) * f64::from(cache_read.unwrap_or(0))
            + per_token(self.cache_write_per_million) * f64::from(cache_write.unwrap_or(0))
    }
}

// ============================================================================
// Trait
// ============================================================================

/// Structured observability surface. All `emit_*` methods are fire-and-forget;
/// they must never block the harness loop. Implementations buffer internally
/// and flush asynchronously via [`flush_session`](Self::flush_session).
pub trait ObservabilityProvider: Send + Sync {
    fn emit_turn(&self, span: TurnSpan);
    fn emit_tool_call(&self, span: ToolCallSpan);
    fn emit_sensor(&self, span: SensorSpan);
    fn emit_context(&self, span: ContextSpan);
    fn emit_middleware(&self, span: MiddlewareSpan);
    /// Record a warn-level tool-call patch event (issue #28). Fire-and-forget
    /// like the other `emit_*` methods.
    fn emit_patch(&self, span: PatchSpan);

    /// Record a warn-level event not tied to a single tool call (issue #46) —
    /// e.g. an accepted-anyway compaction-verification failure. Fire-and-forget
    /// like the other `emit_*` methods. Default: no-op, so providers predating
    /// #46 keep compiling and behave unchanged.
    fn emit_warn(&self, span: WarnSpan) {
        let _ = span;
    }

    /// Record the terminal [`SessionOutcome`] for a session so the trailing
    /// `session` summary line (and [`SessionMetrics`]) reflect it. The harness
    /// calls this once, immediately before [`flush_session`](Self::flush_session),
    /// at every terminal run outcome. Default: no-op — providers that do not
    /// roll up per-session metrics can ignore it.
    fn set_session_outcome(&self, session_id: &SessionId, outcome: SessionOutcome) {
        let _ = (session_id, outcome);
    }

    fn flush_session<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()>;

    fn get_session_metrics<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<SessionMetrics>>;

    fn get_sessions<'a>(
        &'a self,
        since: Timestamp,
        domain: Option<String>,
        outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Vec<SessionMetrics>>;

    fn get_trace<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, Vec<Box<dyn Span>>>;

    /// Session ids whose durable outbox has a `trace.jsonl` but no `.flushed`
    /// marker (issue #33). Default: empty — only the durable-outbox provider
    /// has unflushed on-disk sessions.
    fn list_unflushed_sessions<'a>(&'a self) -> BoxFut<'a, Vec<SessionId>> {
        Box::pin(async { Vec::new() })
    }

    /// Delete a session's durable outbox (issue #33). The provider NEVER
    /// auto-deletes; the caller drives cleanup. Default: no-op `Ok`.
    fn cleanup_session<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<(), ObservabilityError>> {
        let _ = session_id;
        Box::pin(async { Ok(()) })
    }
}

/// Errors surfaced by the durable-outbox provider (issue #33).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ObservabilityError {
    /// An I/O error from the durable outbox.
    #[error("observability I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// `cleanup_session` was called for a session with no outbox directory.
    #[error("session not found: {session_id}")]
    SessionNotFound { session_id: String },
}

// ============================================================================
// Standard in-memory implementation
// ============================================================================

#[derive(Default)]
pub struct InMemoryObservabilityProvider {
    inner: Mutex<Store>,
}

#[derive(Default)]
struct Store {
    turns: Vec<TurnSpan>,
    tool_calls: Vec<ToolCallSpan>,
    sensors: Vec<SensorSpan>,
    contexts: Vec<ContextSpan>,
    middlewares: Vec<MiddlewareSpan>,
    patches: Vec<PatchSpan>,
    warns: Vec<WarnSpan>,
    // Per-session insertion-ordered span ids — the trace-analyzer feed.
    trace_order: HashMap<SessionId, Vec<(SpanKind, SpanId)>>,
    flushed: HashMap<SessionId, bool>,
    // Per-session terminal outcome, recorded by the harness via
    // `set_session_outcome` after AfterSession.
    outcomes: HashMap<SessionId, SessionOutcome>,
    // Per-session guides used, populated by the harness via record_guides_used.
    guides_used: HashMap<SessionId, Vec<GuideId>>,
}

impl InMemoryObservabilityProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// All recorded patch spans for a session, in insertion order (issue #28).
    /// Lets callers inspect the original/patched diff and classified
    /// [`PatchType`] without reconstructing them from the heterogeneous trace.
    pub fn patch_spans(&self, session_id: &SessionId) -> Vec<PatchSpan> {
        self.inner
            .lock()
            .unwrap()
            .patches
            .iter()
            .filter(|p| p.base.session_id == *session_id)
            .cloned()
            .collect()
    }

    /// All recorded warn spans for a session, in insertion order (issue #46).
    /// Lets callers inspect compaction-verification failures without
    /// reconstructing them from the heterogeneous trace.
    pub fn warn_spans(&self, session_id: &SessionId) -> Vec<WarnSpan> {
        self.inner
            .lock()
            .unwrap()
            .warns
            .iter()
            .filter(|w| w.base.session_id == *session_id)
            .cloned()
            .collect()
    }

    /// All recorded context spans for a session, in insertion order. Includes
    /// both `ContextAssembly` and `Compaction` operations — filter on
    /// [`ContextOperation`] to inspect compaction reclamation (issue #57).
    pub fn context_spans(&self, session_id: &SessionId) -> Vec<ContextSpan> {
        self.inner
            .lock()
            .unwrap()
            .contexts
            .iter()
            .filter(|c| c.base.session_id == *session_id)
            .cloned()
            .collect()
    }

    /// Record the guides selected for a session. Called once at session start.
    pub fn record_guides_used(&self, session_id: &SessionId, guides: Vec<GuideId>) {
        self.inner
            .lock()
            .unwrap()
            .guides_used
            .insert(session_id.clone(), guides);
    }
}

fn push_order(store: &mut Store, sid: &SessionId, kind: SpanKind, span_id: SpanId) {
    store
        .trace_order
        .entry(sid.clone())
        .or_default()
        .push((kind, span_id));
}

impl ObservabilityProvider for InMemoryObservabilityProvider {
    fn emit_turn(&self, span: TurnSpan) {
        let mut s = self.inner.lock().unwrap();
        push_order(
            &mut s,
            &span.base.session_id,
            SpanKind::Turn,
            span.base.span_id.clone(),
        );
        s.turns.push(span);
    }
    fn emit_tool_call(&self, span: ToolCallSpan) {
        let mut s = self.inner.lock().unwrap();
        push_order(
            &mut s,
            &span.base.session_id,
            SpanKind::ToolCall,
            span.base.span_id.clone(),
        );
        s.tool_calls.push(span);
    }
    fn emit_sensor(&self, span: SensorSpan) {
        let mut s = self.inner.lock().unwrap();
        push_order(
            &mut s,
            &span.base.session_id,
            SpanKind::SensorEvaluation,
            span.base.span_id.clone(),
        );
        s.sensors.push(span);
    }
    fn emit_context(&self, span: ContextSpan) {
        let mut s = self.inner.lock().unwrap();
        let kind = match span.operation {
            ContextOperation::Compaction { .. } => SpanKind::Compaction,
            _ => SpanKind::ContextAssembly,
        };
        push_order(
            &mut s,
            &span.base.session_id,
            kind,
            span.base.span_id.clone(),
        );
        s.contexts.push(span);
    }
    fn emit_middleware(&self, span: MiddlewareSpan) {
        let mut s = self.inner.lock().unwrap();
        push_order(
            &mut s,
            &span.base.session_id,
            SpanKind::MiddlewareHook,
            span.base.span_id.clone(),
        );
        s.middlewares.push(span);
    }
    fn emit_patch(&self, span: PatchSpan) {
        let mut s = self.inner.lock().unwrap();
        push_order(
            &mut s,
            &span.base.session_id,
            SpanKind::Patch,
            span.base.span_id.clone(),
        );
        s.patches.push(span);
    }
    fn emit_warn(&self, span: WarnSpan) {
        let mut s = self.inner.lock().unwrap();
        push_order(
            &mut s,
            &span.base.session_id,
            SpanKind::Warn,
            span.base.span_id.clone(),
        );
        s.warns.push(span);
    }

    fn set_session_outcome(&self, session_id: &SessionId, outcome: SessionOutcome) {
        self.inner
            .lock()
            .unwrap()
            .outcomes
            .insert(session_id.clone(), outcome);
    }

    fn flush_session<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let mut s = self.inner.lock().unwrap();
            s.flushed.insert(session_id.clone(), true);
        })
    }

    fn get_session_metrics<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<SessionMetrics>> {
        Box::pin(async move {
            let s = self.inner.lock().unwrap();
            let turns: Vec<&TurnSpan> = s
                .turns
                .iter()
                .filter(|t| t.base.session_id == *session_id)
                .collect();
            if turns.is_empty() && !s.outcomes.contains_key(session_id) {
                return None;
            }
            let task_id = turns
                .first()
                .map(|t| t.base.task_id.clone())
                .unwrap_or_else(|| TaskId::new(""));
            let input: u32 = turns.iter().map(|t| t.input_tokens).sum();
            let output: u32 = turns.iter().map(|t| t.output_tokens).sum();
            let cost: f64 = turns.iter().map(|t| t.cost_usd).sum();
            let duration: u64 = turns.iter().map(|t| t.base.duration_ms).sum::<u64>()
                + s.tool_calls
                    .iter()
                    .filter(|c| c.base.session_id == *session_id)
                    .map(|c| c.base.duration_ms)
                    .sum::<u64>();
            let tool_calls = s
                .tool_calls
                .iter()
                .filter(|c| c.base.session_id == *session_id)
                .count() as u32;
            let session_sensors: Vec<&SensorSpan> = s
                .sensors
                .iter()
                .filter(|s| s.base.session_id == *session_id)
                .collect();
            let sensor_fires = session_sensors.iter().filter(|s| s.fired).count() as u32;
            let sensor_halts = session_sensors
                .iter()
                .filter(|s| s.outcome == SensorOutcome::Halt)
                .count() as u32;
            let compactions = s
                .contexts
                .iter()
                .filter(|c| {
                    c.base.session_id == *session_id
                        && matches!(c.operation, ContextOperation::Compaction { .. })
                })
                .count() as u32;
            let session_patches: Vec<&PatchSpan> = s
                .patches
                .iter()
                .filter(|p| p.base.session_id == *session_id)
                .collect();
            let patch_count = session_patches.len() as u32;
            // R7: guard divide-by-zero; denominator is all tool-call spans.
            let patch_rate = if tool_calls == 0 {
                0.0
            } else {
                patch_count as f32 / tool_calls as f32
            };
            let mut patches_by_tool: HashMap<String, u32> = HashMap::new();
            for p in &session_patches {
                *patches_by_tool.entry(p.tool_name.clone()).or_insert(0) += 1;
            }
            let compaction_verification_failures = s
                .warns
                .iter()
                .filter(|w| {
                    w.base.session_id == *session_id
                        && matches!(w.event, WarnEvent::CompactionVerificationFailed { .. })
                })
                .count() as u32;
            Some(SessionMetrics {
                session_id: session_id.clone(),
                task_id,
                total_turns: turns.len() as u32,
                total_input_tokens: input,
                total_output_tokens: output,
                total_cost_usd: cost,
                total_duration_ms: duration,
                tool_calls,
                sensor_fires,
                sensor_halts,
                compactions,
                outcome: s
                    .outcomes
                    .get(session_id)
                    .cloned()
                    .unwrap_or(SessionOutcome::Partial),
                guides_used: s.guides_used.get(session_id).cloned().unwrap_or_default(),
                patch_count,
                patch_rate,
                patches_by_tool,
                compaction_verification_failures,
            })
        })
    }

    fn get_sessions<'a>(
        &'a self,
        since: Timestamp,
        _domain: Option<String>,
        outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Vec<SessionMetrics>> {
        Box::pin(async move {
            // Collect distinct session ids that have spans with started_at >= since.
            let mut session_ids: Vec<SessionId> = {
                let s = self.inner.lock().unwrap();
                let mut ids: Vec<SessionId> = s
                    .turns
                    .iter()
                    .filter(|t| t.base.started_at.as_str() >= since.as_str())
                    .map(|t| t.base.session_id.clone())
                    .collect();
                ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
                ids.dedup();
                ids
            };
            // Drop the lock before awaiting on get_session_metrics.
            let mut out = Vec::new();
            for sid in session_ids.drain(..) {
                if let Some(m) = self.get_session_metrics(&sid).await {
                    if let Some(want) = &outcome {
                        if &m.outcome != want {
                            continue;
                        }
                    }
                    out.push(m);
                }
            }
            out
        })
    }

    fn get_trace<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, Vec<Box<dyn Span>>> {
        Box::pin(async move {
            let s = self.inner.lock().unwrap();
            let order = match s.trace_order.get(session_id) {
                Some(v) => v.clone(),
                None => return Vec::new(),
            };
            let mut out: Vec<Box<dyn Span>> = Vec::with_capacity(order.len());
            for (kind, id) in order {
                match kind {
                    SpanKind::Turn => {
                        if let Some(sp) = s.turns.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    SpanKind::ToolCall => {
                        if let Some(sp) = s.tool_calls.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    SpanKind::SensorEvaluation => {
                        if let Some(sp) = s.sensors.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    SpanKind::ContextAssembly | SpanKind::Compaction => {
                        if let Some(sp) = s.contexts.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    SpanKind::MiddlewareHook => {
                        if let Some(sp) = s.middlewares.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    SpanKind::Patch => {
                        if let Some(sp) = s.patches.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    SpanKind::Warn => {
                        if let Some(sp) = s.warns.iter().find(|t| t.base.span_id == id) {
                            out.push(Box::new(sp.clone()));
                        }
                    }
                    _ => {}
                }
            }
            out
        })
    }
}

// Wrapper so Arc<InMemoryObservabilityProvider> satisfies the trait when
// callers want shared ownership.
impl<T: ObservabilityProvider + ?Sized> ObservabilityProvider for Arc<T> {
    fn emit_turn(&self, span: TurnSpan) {
        (**self).emit_turn(span)
    }
    fn emit_tool_call(&self, span: ToolCallSpan) {
        (**self).emit_tool_call(span)
    }
    fn emit_sensor(&self, span: SensorSpan) {
        (**self).emit_sensor(span)
    }
    fn emit_context(&self, span: ContextSpan) {
        (**self).emit_context(span)
    }
    fn emit_middleware(&self, span: MiddlewareSpan) {
        (**self).emit_middleware(span)
    }
    fn emit_patch(&self, span: PatchSpan) {
        (**self).emit_patch(span)
    }
    fn emit_warn(&self, span: WarnSpan) {
        (**self).emit_warn(span)
    }
    fn set_session_outcome(&self, session_id: &SessionId, outcome: SessionOutcome) {
        (**self).set_session_outcome(session_id, outcome)
    }
    fn flush_session<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()> {
        (**self).flush_session(session_id)
    }
    fn get_session_metrics<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<SessionMetrics>> {
        (**self).get_session_metrics(session_id)
    }
    fn get_sessions<'a>(
        &'a self,
        since: Timestamp,
        domain: Option<String>,
        outcome: Option<SessionOutcome>,
    ) -> BoxFut<'a, Vec<SessionMetrics>> {
        (**self).get_sessions(since, domain, outcome)
    }
    fn get_trace<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, Vec<Box<dyn Span>>> {
        (**self).get_trace(session_id)
    }
    fn list_unflushed_sessions<'a>(&'a self) -> BoxFut<'a, Vec<SessionId>> {
        (**self).list_unflushed_sessions()
    }
    fn cleanup_session<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Result<(), ObservabilityError>> {
        (**self).cleanup_session(session_id)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> Timestamp {
        Timestamp::new(s)
    }
    fn sid(s: &str) -> SessionId {
        SessionId::new(s)
    }
    fn tid(s: &str) -> TaskId {
        TaskId::new(s)
    }

    fn turn_span(session: &str, span_id: &str, turn: u32, input: u32, output: u32) -> TurnSpan {
        TurnSpan {
            base: SpanBase {
                span_id: SpanId::new(span_id),
                parent_span_id: None,
                session_id: sid(session),
                task_id: tid("t1"),
                kind: SpanKind::Turn,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:01Z"),
                duration_ms: 1000,
                status: SpanStatus::Ok,
            },
            turn_number: turn,
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: None,
            cache_write_tokens: None,
            cost_usd: 0.0,
            stop_reason: StopReason::EndTurn,
            tool_calls_requested: 0,
        }
    }

    // ── Rule: emit_turn is fire-and-forget (no async) and span is queryable ─

    #[tokio::test]
    async fn emit_turn_recorded_and_metrics_aggregate() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "sp1", 1, 100, 50));
        obs.emit_turn(turn_span("s1", "sp2", 2, 200, 80));
        obs.set_session_outcome(&sid("s1"), SessionOutcome::Success);
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.total_turns, 2);
        assert_eq!(m.total_input_tokens, 300);
        assert_eq!(m.total_output_tokens, 130);
        assert_eq!(m.outcome, SessionOutcome::Success);
    }

    // ── Rule: emit_tool_call counted in metrics ─────────────────────────────

    #[tokio::test]
    async fn emit_tool_call_counted_in_metrics() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "t1", 1, 10, 5));
        let base = SpanBase {
            span_id: SpanId::new("tc1"),
            parent_span_id: None,
            session_id: sid("s1"),
            task_id: tid("t1"),
            kind: SpanKind::ToolCall,
            started_at: ts("2026-05-16T00:00:00Z"),
            ended_at: ts("2026-05-16T00:00:00Z"),
            duration_ms: 250,
            status: SpanStatus::Ok,
        };
        obs.emit_tool_call(ToolCallSpan {
            base,
            tool_name: "shell".into(),
            call_id: "c1".into(),
            parameters_size_bytes: 12,
            output_size_bytes: 42,
            truncated: false,
            sandbox_mode: "workspace_scoped".into(),
            sandbox_violations: vec![],
        });
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.tool_calls, 1);
        assert_eq!(m.total_duration_ms, 1250);
    }

    // ── Rule: sensor metrics — fires and halts ──────────────────────────────

    #[tokio::test]
    async fn sensor_metrics_count_fires_and_halts() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "t1", 1, 10, 5));
        let make = |id: &str, fired: bool, outcome: SensorOutcome| SensorSpan {
            base: SpanBase {
                span_id: SpanId::new(id),
                parent_span_id: None,
                session_id: sid("s1"),
                task_id: tid("t1"),
                kind: SpanKind::SensorEvaluation,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:00Z"),
                duration_ms: 1,
                status: SpanStatus::Ok,
            },
            sensor_id: SensorId::new("lint"),
            sensor_kind: SensorKind::Computational,
            trigger: SensorTrigger::PostTurn,
            outcome,
            fired,
        };
        obs.emit_sensor(make("sn1", true, SensorOutcome::Warn));
        obs.emit_sensor(make("sn2", true, SensorOutcome::Halt));
        obs.emit_sensor(make("sn3", false, SensorOutcome::Pass));
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.sensor_fires, 2);
        assert_eq!(m.sensor_halts, 1);
    }

    // ── Rule: compaction counted ────────────────────────────────────────────

    #[tokio::test]
    async fn compaction_counted_in_metrics() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "t1", 1, 100, 50));
        let mk_ctx = |op: ContextOperation| ContextSpan {
            base: SpanBase {
                span_id: SpanId::new("c1"),
                parent_span_id: None,
                session_id: sid("s1"),
                task_id: tid("t1"),
                kind: SpanKind::Compaction,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:00Z"),
                duration_ms: 1,
                status: SpanStatus::Ok,
            },
            operation: op,
            tokens_before: 10_000,
            tokens_after: 5_000,
            utilization_before: 0.9,
            utilization_after: 0.5,
        };
        obs.emit_context(mk_ctx(ContextOperation::Compaction {
            messages_removed: 5,
            tokens_reclaimed: 5_000,
        }));
        obs.emit_context(mk_ctx(ContextOperation::Assembly {
            guides_loaded: 2,
            memory_items_loaded: 3,
            tools_loaded: 5,
        }));
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.compactions, 1);
    }

    // ── Rule: pricing table computes cost_usd ────────────────────────────────

    #[test]
    fn pricing_table_computes_cost() {
        let table = PricingTable {
            input_per_million: 3.0,
            output_per_million: 15.0,
            cache_read_per_million: 0.3,
            cache_write_per_million: 3.75,
        };
        let cost = table.cost_for(1_000_000, 1_000_000, Some(1_000_000), Some(1_000_000));
        // 3 + 15 + 0.3 + 3.75 = 22.05
        assert!((cost - 22.05).abs() < 1e-9);
    }

    #[test]
    fn pricing_table_default_is_zero() {
        let cost = PricingTable::DEFAULT.cost_for(1_000, 1_000, Some(1_000), Some(1_000));
        assert_eq!(cost, 0.0);
    }

    // ── Rule: flush_session idempotent ──────────────────────────────────────

    #[tokio::test]
    async fn flush_session_is_idempotent() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "t1", 1, 10, 5));
        obs.flush_session(&sid("s1")).await;
        obs.flush_session(&sid("s1")).await; // second flush is a no-op
                                             // Spans remain queryable after flush.
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.total_turns, 1);
    }

    // ── Rule: get_trace returns spans in insertion order ────────────────────

    #[tokio::test]
    async fn get_trace_preserves_insertion_order() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "sp1", 1, 10, 5));
        obs.emit_tool_call(ToolCallSpan {
            base: SpanBase {
                span_id: SpanId::new("sp2"),
                parent_span_id: Some(SpanId::new("sp1")),
                session_id: sid("s1"),
                task_id: tid("t1"),
                kind: SpanKind::ToolCall,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:00Z"),
                duration_ms: 1,
                status: SpanStatus::Ok,
            },
            tool_name: "shell".into(),
            call_id: "c1".into(),
            parameters_size_bytes: 0,
            output_size_bytes: 0,
            truncated: false,
            sandbox_mode: "none".into(),
            sandbox_violations: vec![],
        });
        let trace = obs.get_trace(&sid("s1")).await;
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].base().span_id.as_str(), "sp1");
        assert_eq!(trace[1].base().span_id.as_str(), "sp2");
        // Parent linkage preserved.
        assert_eq!(
            trace[1].base().parent_span_id.as_ref().unwrap().as_str(),
            "sp1"
        );
    }

    // ── Rule: middleware spans recorded with hook and decision ──────────────

    #[tokio::test]
    async fn middleware_span_recorded_in_trace() {
        let obs = InMemoryObservabilityProvider::new();
        let span = MiddlewareSpan {
            base: SpanBase {
                span_id: SpanId::new("mw1"),
                parent_span_id: None,
                session_id: sid("s1"),
                task_id: tid("t1"),
                kind: SpanKind::MiddlewareHook,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:00Z"),
                duration_ms: 0,
                status: SpanStatus::Ok,
            },
            hook: HookPoint::BeforeTurn,
            decision: MiddlewareDecision::Continue,
        };
        obs.emit_middleware(span);
        let trace = obs.get_trace(&sid("s1")).await;
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].base().kind, SpanKind::MiddlewareHook);
    }

    // ── Rule: get_sessions filters by outcome ───────────────────────────────

    #[tokio::test]
    async fn get_sessions_filters_by_outcome() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("good", "sp1", 1, 10, 5));
        obs.emit_turn(turn_span("bad", "sp2", 1, 10, 5));
        obs.set_session_outcome(&sid("good"), SessionOutcome::Success);
        obs.set_session_outcome(&sid("bad"), SessionOutcome::Failure { reason: "x".into() });
        let success_only = obs
            .get_sessions(
                ts("2026-05-16T00:00:00Z"),
                None,
                Some(SessionOutcome::Success),
            )
            .await;
        assert_eq!(success_only.len(), 1);
        assert_eq!(success_only[0].session_id.as_str(), "good");
    }

    // ── Rule: get_sessions filters by since timestamp ───────────────────────

    #[tokio::test]
    async fn get_sessions_filters_by_since() {
        let obs = InMemoryObservabilityProvider::new();
        let mut early = turn_span("old", "sp1", 1, 10, 5);
        early.base.started_at = ts("2026-01-01T00:00:00Z");
        obs.emit_turn(early);
        obs.emit_turn(turn_span("new", "sp2", 1, 10, 5));
        let recent = obs
            .get_sessions(ts("2026-05-15T00:00:00Z"), None, None)
            .await;
        let ids: Vec<_> = recent
            .iter()
            .map(|m| m.session_id.as_str().to_string())
            .collect();
        assert!(ids.contains(&"new".to_string()));
        assert!(!ids.contains(&"old".to_string()));
    }

    // ── Rule: passive observer — no mutator on trait surface ────────────────
    // (compile-time: the trait has no &mut self method that affects behavior.)

    #[tokio::test]
    async fn observability_provider_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let obs: Arc<dyn ObservabilityProvider> = Arc::new(InMemoryObservabilityProvider::new());
        assert_send_sync(&obs);
    }

    // ── SpanBase helpers ─────────────────────────────────────────────────────

    #[test]
    fn span_base_new_root_and_child() {
        let root = SpanBase::new_root(
            SpanId::new("r"),
            sid("s"),
            tid("t"),
            SpanKind::Session,
            ts("2026-05-16T00:00:00Z"),
        );
        let child = SpanBase::new_child(
            SpanId::new("c"),
            &root,
            SpanKind::Turn,
            ts("2026-05-16T00:00:01Z"),
        );
        assert_eq!(child.parent_span_id.unwrap().as_str(), "r");
        assert_eq!(child.session_id.as_str(), "s");
    }

    // ── Fixture replay ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn fixture_replay_session_metrics() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/observability/session_metrics_basic.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let case: FixtureCase = serde_json::from_str(&raw).unwrap();

        let obs = InMemoryObservabilityProvider::new();
        for t in case.turns {
            obs.emit_turn(turn_span(
                &case.session_id,
                &t.span_id,
                t.turn,
                t.input,
                t.output,
            ));
        }
        obs.set_session_outcome(
            &sid(&case.session_id),
            match case.outcome.as_str() {
                "success" => SessionOutcome::Success,
                "partial" => SessionOutcome::Partial,
                _ => SessionOutcome::Failure {
                    reason: case.outcome.clone(),
                },
            },
        );
        let m = obs
            .get_session_metrics(&sid(&case.session_id))
            .await
            .unwrap();
        assert_eq!(m.total_turns, case.expected.total_turns);
        assert_eq!(m.total_input_tokens, case.expected.total_input_tokens);
        assert_eq!(m.total_output_tokens, case.expected.total_output_tokens);
    }

    // ── Patch spans (issue #28) ──────────────────────────────────────────────

    fn patch_span(session: &str, span_id: &str, call_id: &str, tool: &str) -> PatchSpan {
        PatchSpan::new(
            SpanBase {
                span_id: SpanId::new(span_id),
                parent_span_id: None,
                session_id: sid(session),
                task_id: tid("t1"),
                kind: SpanKind::Patch,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:00Z"),
                duration_ms: 0,
                status: SpanStatus::Ok,
            },
            call_id,
            tool,
            serde_json::json!({"a": "1"}),
            serde_json::json!({"a": 1}),
            PatchType::ParameterCoercion {
                field: "a".into(),
                from: "string".into(),
                to: "number".into(),
            },
        )
    }

    fn tool_call_span(session: &str, span_id: &str, tool: &str) -> ToolCallSpan {
        ToolCallSpan {
            base: SpanBase {
                span_id: SpanId::new(span_id),
                parent_span_id: None,
                session_id: sid(session),
                task_id: tid("t1"),
                kind: SpanKind::ToolCall,
                started_at: ts("2026-05-16T00:00:00Z"),
                ended_at: ts("2026-05-16T00:00:00Z"),
                duration_ms: 0,
                status: SpanStatus::Ok,
            },
            tool_name: tool.into(),
            call_id: span_id.into(),
            parameters_size_bytes: 0,
            output_size_bytes: 0,
            truncated: false,
            sandbox_mode: "none".into(),
            sandbox_violations: vec![],
        }
    }

    // R1/R5: emit_patch records a warn-level span that appears in the trace.
    #[tokio::test]
    async fn emit_patch_appears_in_trace_as_warn() {
        let obs = InMemoryObservabilityProvider::new();
        let sp = patch_span("s1", "p1", "c1", "shell");
        assert_eq!(sp.level, SpanLevel::Warn);
        obs.emit_patch(sp);
        let trace = obs.get_trace(&sid("s1")).await;
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].base().kind, SpanKind::Patch);
        // R3: both original and patched present and different.
        let patches = obs.inner.lock().unwrap();
        let p = &patches.patches[0];
        assert_ne!(p.original_parameters, p.patched_parameters);
    }

    // R6/R7/R8: patch metrics roll up.
    #[tokio::test]
    async fn patch_metrics_count_rate_and_by_tool() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "t1", 1, 10, 5));
        // 4 tool calls, 2 of which were patched (shell twice).
        obs.emit_tool_call(tool_call_span("s1", "tc1", "shell"));
        obs.emit_tool_call(tool_call_span("s1", "tc2", "shell"));
        obs.emit_tool_call(tool_call_span("s1", "tc3", "edit"));
        obs.emit_tool_call(tool_call_span("s1", "tc4", "edit"));
        obs.emit_patch(patch_span("s1", "p1", "tc1", "shell"));
        obs.emit_patch(patch_span("s1", "p2", "tc2", "shell"));
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.patch_count, 2);
        assert!((m.patch_rate - 0.5).abs() < 1e-6); // 2 / 4
        assert_eq!(m.patches_by_tool.get("shell"), Some(&2));
        assert_eq!(m.patches_by_tool.get("edit"), None);
    }

    // R7: zero tool calls → patch_rate is 0.0, never a divide-by-zero.
    #[tokio::test]
    async fn patch_rate_zero_when_no_tool_calls() {
        let obs = InMemoryObservabilityProvider::new();
        obs.emit_turn(turn_span("s1", "t1", 1, 10, 5));
        obs.emit_patch(patch_span("s1", "p1", "c1", "shell"));
        let m = obs.get_session_metrics(&sid("s1")).await.unwrap();
        assert_eq!(m.patch_count, 1);
        assert_eq!(m.patch_rate, 0.0);
    }

    // SessionMetrics with patch fields deserializes from a legacy fixture
    // lacking them (serde default).
    #[test]
    fn session_metrics_patch_fields_default() {
        let json = serde_json::json!({
            "session_id": "s1",
            "task_id": "t1",
            "total_turns": 1,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "total_duration_ms": 0,
            "tool_calls": 0,
            "sensor_fires": 0,
            "sensor_halts": 0,
            "compactions": 0,
            "outcome": {"kind": "partial"}
        });
        let m: SessionMetrics = serde_json::from_value(json).unwrap();
        assert_eq!(m.patch_count, 0);
        assert_eq!(m.patch_rate, 0.0);
        assert!(m.patches_by_tool.is_empty());
    }

    #[derive(serde::Deserialize)]
    struct FixtureCase {
        session_id: String,
        turns: Vec<FixtureTurn>,
        outcome: String,
        expected: FixtureExpected,
    }
    #[derive(serde::Deserialize)]
    struct FixtureTurn {
        span_id: String,
        turn: u32,
        input: u32,
        output: u32,
    }
    #[derive(serde::Deserialize)]
    struct FixtureExpected {
        total_turns: u32,
        total_input_tokens: u32,
        total_output_tokens: u32,
    }
}
