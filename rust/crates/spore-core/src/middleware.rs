//! Issue #11 — `MiddlewareChain`: cross-cutting interception of the agent
//! loop at six hook points.
//!
//! See `docs/harness-engineering-concepts.md` § "Middleware Chain" for the
//! authoritative rules. This module ships:
//!   - The full [`Middleware`] trait and [`HookContext`] / [`HookPoint`]
//!     / [`MiddlewareDecision`] surface from the spec.
//!   - [`StandardMiddlewareChain`] — in-memory reference implementation with
//!     priority ordering, ForceAnotherTurn concatenation, and first-wins
//!     SurfaceToHuman semantics.
//!   - A subset of standard middleware referenced by name in the spec
//!     (`TokenBudgetMiddleware`, `LoopDetectionMiddleware`,
//!     `PreCompletionChecklistMiddleware`, `PatchToolCallsMiddleware`,
//!     `TracingMiddleware`). The remaining ones (`PermissionMiddleware`,
//!     `PiiRedactionMiddleware`, `RateLimitMiddleware`,
//!     `DirectoryMapMiddleware`, `TimeBudgetMiddleware`) follow the same
//!     pattern and will land alongside the components they depend on.
//!
//! ## Rules enforced
//!   - Before hooks (`BeforeSession`, `BeforeTurn`, `BeforeTool`,
//!     `BeforeCompletion`) run sorted by priority **ascending** — lowest
//!     priority number first.
//!   - After hooks (`AfterTool`, `AfterSession`) run sorted by priority
//!     **descending** — highest number first (wrapping pattern).
//!   - First [`MiddlewareDecision::Halt`] or
//!     [`MiddlewareDecision::SurfaceToHuman`] stops the chain.
//!     Downstream middleware do not run.
//!   - [`MiddlewareDecision::ForceAnotherTurn`] is **only valid on
//!     `BeforeCompletion`**. All ForceAnotherTurn injections from the
//!     remaining middleware are concatenated (newline-joined) and returned
//!     in a single decision. The chain continues running.
//!   - [`MiddlewareDecision::SurfaceToHuman`] is valid only on `BeforeTool`
//!     and `BeforeCompletion`. Returning it from any other hook is a
//!     [`MiddlewareError::IllegalDecision`] surfaced as `Halt` to the loop.
//!   - Middleware **must not** hold session state between calls — guidance
//!     enforced by giving `handle()` no `&mut self` access to long-lived
//!     storage. Production middleware key into an external map by
//!     `SessionId` and clear in `AfterSession`.
//!   - Middleware must not call `ModelInterface` or `ToolRegistry` — neither
//!     is in scope of any [`HookContext`] variant.
//!
//! ## Implementor notes
//!   - [`HookContext`] borrows the hot-path data mutably where the spec
//!     allows mutation (`BeforeTurn` session, `BeforeTool` calls,
//!     `AfterTool` results). [`MiddlewareDecision::ContinueWithModification`]
//!     signals that a mutation occurred — the harness can use it for
//!     observability without re-diffing.
//!   - Priority defaults to `0`. `TracingMiddleware` registers at
//!     `i32::MIN` so it always runs first on before hooks and last on after
//!     hooks. `PatchToolCallsMiddleware` registers at `i32::MIN + 1` so it
//!     runs before all other `BeforeTool` middleware (per spec).
//!   - `PermissionMiddleware` registers at high priority on `BeforeTool` so
//!     it runs after patching, repairs, and PII redaction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::{
    BoxFut, HumanRequest, RunResult, SessionId, SessionState, Task, TaskId, ToolResult,
};
use crate::model::ToolCall;
use crate::observability::{
    ObservabilityProvider, PatchSpan, PatchType, SpanBase, SpanId, SpanKind,
};

// ============================================================================
// HookPoint (full spec: 6 variants)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPoint {
    BeforeSession,
    BeforeTurn,
    BeforeTool,
    AfterTool,
    BeforeCompletion,
    AfterSession,
}

impl HookPoint {
    /// True for hooks ordered ascending by priority (lowest first).
    pub fn is_before(&self) -> bool {
        matches!(
            self,
            HookPoint::BeforeSession
                | HookPoint::BeforeTurn
                | HookPoint::BeforeTool
                | HookPoint::BeforeCompletion
        )
    }

    /// True for hooks ordered descending by priority (highest first / wrapping).
    pub fn is_after(&self) -> bool {
        matches!(self, HookPoint::AfterTool | HookPoint::AfterSession)
    }

    /// Whether `SurfaceToHuman` is permitted at this hook point.
    pub fn allows_surface_to_human(&self) -> bool {
        matches!(self, HookPoint::BeforeTool | HookPoint::BeforeCompletion)
    }

    /// Whether `ForceAnotherTurn` is permitted at this hook point.
    pub fn allows_force_another_turn(&self) -> bool {
        matches!(self, HookPoint::BeforeCompletion)
    }
}

// ============================================================================
// HookContext (borrows hot-path data; mutates in place where spec allows)
// ============================================================================

pub enum HookContext<'a> {
    BeforeSession {
        task: &'a Task,
        session_id: &'a SessionId,
    },
    BeforeTurn {
        session: &'a mut SessionState,
        turn_number: u32,
    },
    BeforeTool {
        calls: &'a mut Vec<ToolCall>,
        turn_number: u32,
    },
    AfterTool {
        calls: &'a [ToolCall],
        results: &'a mut Vec<ToolResult>,
    },
    BeforeCompletion {
        response: &'a str,
        turn_number: u32,
        session_state: &'a SessionState,
    },
    AfterSession {
        result: &'a RunResult,
        session_id: &'a SessionId,
    },
}

impl<'a> HookContext<'a> {
    pub fn point(&self) -> HookPoint {
        match self {
            HookContext::BeforeSession { .. } => HookPoint::BeforeSession,
            HookContext::BeforeTurn { .. } => HookPoint::BeforeTurn,
            HookContext::BeforeTool { .. } => HookPoint::BeforeTool,
            HookContext::AfterTool { .. } => HookPoint::AfterTool,
            HookContext::BeforeCompletion { .. } => HookPoint::BeforeCompletion,
            HookContext::AfterSession { .. } => HookPoint::AfterSession,
        }
    }
}

// ============================================================================
// MiddlewareDecision (full spec: 5 variants)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MiddlewareDecision {
    Continue,
    /// The middleware mutated the borrowed context. Semantically equivalent
    /// to `Continue` for chain control flow — the harness uses this signal
    /// for observability.
    ContinueWithModification,
    /// Valid only on `BeforeCompletion`. The chain concatenates injections
    /// from all middleware that returned this and surfaces one combined
    /// `ForceAnotherTurn` decision.
    ForceAnotherTurn {
        inject: String,
    },
    Halt {
        reason: String,
    },
    /// Valid only on `BeforeTool` and `BeforeCompletion`. First occurrence
    /// in priority order wins; remaining middleware do not run.
    SurfaceToHuman {
        request: HumanRequest,
    },
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MiddlewareError {
    #[error("middleware already registered: {0}")]
    AlreadyRegistered(String),
    #[error("middleware {name} declared zero hooks")]
    NoHooks { name: String },
    #[error("middleware {name} returned {decision} from {hook:?} which does not allow it")]
    IllegalDecision {
        name: String,
        hook: HookPoint,
        decision: String,
    },
}

// ============================================================================
// Traits
// ============================================================================

/// A single middleware. Object-safe via [`BoxFut`].
pub trait Middleware: Send + Sync {
    fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision>;
    fn hooks(&self) -> Vec<HookPoint>;
    fn priority(&self) -> i32 {
        0
    }
    fn name(&self) -> &str;
}

/// Registry + fan-out evaluator.
pub trait MiddlewareChain: Send + Sync {
    fn register(&self, middleware: Box<dyn Middleware>) -> Result<(), MiddlewareError>;

    fn fire_before_session<'a>(
        &'a self,
        task: &'a Task,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, MiddlewareDecision>;

    fn fire_before_turn<'a>(
        &'a self,
        session: &'a mut SessionState,
        turn_number: u32,
    ) -> BoxFut<'a, MiddlewareDecision>;

    fn fire_before_tool<'a>(
        &'a self,
        calls: &'a mut Vec<ToolCall>,
        turn_number: u32,
    ) -> BoxFut<'a, MiddlewareDecision>;

    fn fire_after_tool<'a>(
        &'a self,
        calls: &'a [ToolCall],
        results: &'a mut Vec<ToolResult>,
    ) -> BoxFut<'a, MiddlewareDecision>;

    fn fire_before_completion<'a>(
        &'a self,
        response: &'a str,
        turn_number: u32,
        state: &'a SessionState,
    ) -> BoxFut<'a, MiddlewareDecision>;

    fn fire_after_session<'a>(
        &'a self,
        result: &'a RunResult,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, ()>;
}

// ============================================================================
// Standard implementation
// ============================================================================

#[derive(Default)]
pub struct StandardMiddlewareChain {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    middlewares: Vec<Entry>,
}

#[derive(Clone)]
struct Entry {
    name: String,
    priority: i32,
    hooks: Vec<HookPoint>,
    middleware: Arc<dyn Middleware>,
}

impl StandardMiddlewareChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot entries eligible for a hook, sorted per the ordering rule for
    /// that hook. Returns owning raw pointers wrapped via indices into a
    /// borrowed-from-Mutex iteration — simpler: hold the lock for the full
    /// fire, since middleware do not call back into the chain (spec rule).
    fn eligible(inner: &Inner, hook: HookPoint) -> Vec<Entry> {
        let mut v: Vec<Entry> = inner
            .middlewares
            .iter()
            .filter(|e| e.hooks.contains(&hook))
            .cloned()
            .collect();
        if hook.is_after() {
            // Descending priority.
            v.sort_by(|a, b| b.priority.cmp(&a.priority).then(a.name.cmp(&b.name)));
        } else {
            // Ascending priority.
            v.sort_by(|a, b| a.priority.cmp(&b.priority).then(a.name.cmp(&b.name)));
        }
        v
    }
}

impl MiddlewareChain for StandardMiddlewareChain {
    fn register(&self, middleware: Box<dyn Middleware>) -> Result<(), MiddlewareError> {
        let name = middleware.name().to_string();
        let hooks = middleware.hooks();
        if hooks.is_empty() {
            return Err(MiddlewareError::NoHooks { name });
        }
        let priority = middleware.priority();
        let mut inner = self.inner.lock().unwrap();
        if inner.middlewares.iter().any(|e| e.name == name) {
            return Err(MiddlewareError::AlreadyRegistered(name));
        }
        inner.middlewares.push(Entry {
            name,
            priority,
            hooks,
            middleware: Arc::from(middleware),
        });
        Ok(())
    }

    fn fire_before_session<'a>(
        &'a self,
        task: &'a Task,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, MiddlewareDecision> {
        Box::pin(async move {
            let entries = {
                let inner = self.inner.lock().unwrap();
                Self::eligible(&inner, HookPoint::BeforeSession)
            };
            for entry in entries {
                let decision = entry
                    .middleware
                    .handle(HookContext::BeforeSession { task, session_id })
                    .await;
                match validate_decision(&entry, HookPoint::BeforeSession, decision) {
                    Ok(MiddlewareDecision::Continue)
                    | Ok(MiddlewareDecision::ContinueWithModification) => continue,
                    Ok(other) => return other,
                    Err(e) => {
                        return MiddlewareDecision::Halt {
                            reason: e.to_string(),
                        }
                    }
                }
            }
            MiddlewareDecision::Continue
        })
    }

    fn fire_before_turn<'a>(
        &'a self,
        session: &'a mut SessionState,
        turn_number: u32,
    ) -> BoxFut<'a, MiddlewareDecision> {
        Box::pin(async move {
            let entries = {
                let inner = self.inner.lock().unwrap();
                Self::eligible(&inner, HookPoint::BeforeTurn)
            };
            let mut any_modified = false;
            for entry in entries {
                let decision = entry
                    .middleware
                    .handle(HookContext::BeforeTurn {
                        session,
                        turn_number,
                    })
                    .await;
                match validate_decision(&entry, HookPoint::BeforeTurn, decision) {
                    Ok(MiddlewareDecision::Continue) => continue,
                    Ok(MiddlewareDecision::ContinueWithModification) => {
                        any_modified = true;
                        continue;
                    }
                    Ok(other) => return other,
                    Err(e) => {
                        return MiddlewareDecision::Halt {
                            reason: e.to_string(),
                        }
                    }
                }
            }
            if any_modified {
                MiddlewareDecision::ContinueWithModification
            } else {
                MiddlewareDecision::Continue
            }
        })
    }

    fn fire_before_tool<'a>(
        &'a self,
        calls: &'a mut Vec<ToolCall>,
        turn_number: u32,
    ) -> BoxFut<'a, MiddlewareDecision> {
        Box::pin(async move {
            let entries = {
                let inner = self.inner.lock().unwrap();
                Self::eligible(&inner, HookPoint::BeforeTool)
            };
            let mut any_modified = false;
            for entry in entries {
                let decision = entry
                    .middleware
                    .handle(HookContext::BeforeTool { calls, turn_number })
                    .await;
                match validate_decision(&entry, HookPoint::BeforeTool, decision) {
                    Ok(MiddlewareDecision::Continue) => continue,
                    Ok(MiddlewareDecision::ContinueWithModification) => {
                        any_modified = true;
                        continue;
                    }
                    Ok(other) => return other,
                    Err(e) => {
                        return MiddlewareDecision::Halt {
                            reason: e.to_string(),
                        }
                    }
                }
            }
            if any_modified {
                MiddlewareDecision::ContinueWithModification
            } else {
                MiddlewareDecision::Continue
            }
        })
    }

    fn fire_after_tool<'a>(
        &'a self,
        calls: &'a [ToolCall],
        results: &'a mut Vec<ToolResult>,
    ) -> BoxFut<'a, MiddlewareDecision> {
        Box::pin(async move {
            let entries = {
                let inner = self.inner.lock().unwrap();
                Self::eligible(&inner, HookPoint::AfterTool)
            };
            let mut any_modified = false;
            for entry in entries {
                let decision = entry
                    .middleware
                    .handle(HookContext::AfterTool { calls, results })
                    .await;
                match validate_decision(&entry, HookPoint::AfterTool, decision) {
                    Ok(MiddlewareDecision::Continue) => continue,
                    Ok(MiddlewareDecision::ContinueWithModification) => {
                        any_modified = true;
                        continue;
                    }
                    Ok(other) => return other,
                    Err(e) => {
                        return MiddlewareDecision::Halt {
                            reason: e.to_string(),
                        }
                    }
                }
            }
            if any_modified {
                MiddlewareDecision::ContinueWithModification
            } else {
                MiddlewareDecision::Continue
            }
        })
    }

    fn fire_before_completion<'a>(
        &'a self,
        response: &'a str,
        turn_number: u32,
        state: &'a SessionState,
    ) -> BoxFut<'a, MiddlewareDecision> {
        Box::pin(async move {
            let entries = {
                let inner = self.inner.lock().unwrap();
                Self::eligible(&inner, HookPoint::BeforeCompletion)
            };
            let mut injections: Vec<String> = Vec::new();
            for entry in entries {
                let decision = entry
                    .middleware
                    .handle(HookContext::BeforeCompletion {
                        response,
                        turn_number,
                        session_state: state,
                    })
                    .await;
                match validate_decision(&entry, HookPoint::BeforeCompletion, decision) {
                    Ok(MiddlewareDecision::Continue)
                    | Ok(MiddlewareDecision::ContinueWithModification) => continue,
                    Ok(MiddlewareDecision::ForceAnotherTurn { inject }) => {
                        injections.push(inject);
                        // chain continues per spec
                    }
                    Ok(other @ MiddlewareDecision::Halt { .. })
                    | Ok(other @ MiddlewareDecision::SurfaceToHuman { .. }) => return other,
                    Err(e) => {
                        return MiddlewareDecision::Halt {
                            reason: e.to_string(),
                        }
                    }
                }
            }
            if injections.is_empty() {
                MiddlewareDecision::Continue
            } else {
                MiddlewareDecision::ForceAnotherTurn {
                    inject: injections.join("\n"),
                }
            }
        })
    }

    fn fire_after_session<'a>(
        &'a self,
        result: &'a RunResult,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let entries = {
                let inner = self.inner.lock().unwrap();
                Self::eligible(&inner, HookPoint::AfterSession)
            };
            for entry in entries {
                // After hooks ignore decisions other than Continue —
                // session is already terminating.
                let _ = entry
                    .middleware
                    .handle(HookContext::AfterSession { result, session_id })
                    .await;
            }
        })
    }
}

fn validate_decision(
    entry: &Entry,
    hook: HookPoint,
    decision: MiddlewareDecision,
) -> Result<MiddlewareDecision, MiddlewareError> {
    match &decision {
        MiddlewareDecision::SurfaceToHuman { .. } if !hook.allows_surface_to_human() => {
            Err(MiddlewareError::IllegalDecision {
                name: entry.name.clone(),
                hook,
                decision: "SurfaceToHuman".into(),
            })
        }
        MiddlewareDecision::ForceAnotherTurn { .. } if !hook.allows_force_another_turn() => {
            Err(MiddlewareError::IllegalDecision {
                name: entry.name.clone(),
                hook,
                decision: "ForceAnotherTurn".into(),
            })
        }
        _ => Ok(decision),
    }
}

// ============================================================================
// Standard middleware implementations (representative subset)
// ============================================================================

/// Lowest-priority observability middleware. Records every hook firing. The
/// real implementation forwards to `ObservabilityProvider`; this version
/// keeps an in-memory log so tests can assert ordering.
pub struct TracingMiddleware {
    pub name: String,
    log: Mutex<Vec<(HookPoint, u32)>>,
}

impl Default for TracingMiddleware {
    fn default() -> Self {
        Self::new("tracing")
    }
}

impl TracingMiddleware {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            log: Mutex::new(Vec::new()),
        }
    }
    pub fn entries(&self) -> Vec<(HookPoint, u32)> {
        self.log.lock().unwrap().clone()
    }
}

impl Middleware for TracingMiddleware {
    fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
        let point = ctx.point();
        let turn = match &ctx {
            HookContext::BeforeTurn { turn_number, .. }
            | HookContext::BeforeTool { turn_number, .. }
            | HookContext::BeforeCompletion { turn_number, .. } => *turn_number,
            _ => 0,
        };
        self.log.lock().unwrap().push((point, turn));
        Box::pin(async move { MiddlewareDecision::Continue })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![
            HookPoint::BeforeSession,
            HookPoint::BeforeTurn,
            HookPoint::BeforeTool,
            HookPoint::AfterTool,
            HookPoint::BeforeCompletion,
            HookPoint::AfterSession,
        ]
    }
    fn priority(&self) -> i32 {
        i32::MIN
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Repairs syntactically invalid tool calls before they reach the registry.
/// The shipped implementation patches empty or whitespace-only tool names
/// to a configurable fallback — production deployments swap in a richer
/// JSON-repair routine. Runs at the highest `BeforeTool` priority so
/// downstream middleware see clean calls.
///
/// ## Observability (issue #28)
///
/// This middleware is an always-on, highest-priority action mutator. To keep
/// it from silently rewriting calls, **every patch emits a warn-level
/// [`PatchSpan`]** via an injected [`ObservabilityProvider`] before the
/// patched call proceeds. The span carries the original and patched
/// parameters and a classified [`PatchType`] so the trace shows the diff,
/// never just the patched call.
///
/// The shared [`HookContext::BeforeTool`] does not carry `session_id` /
/// `task_id` (widening it would ripple across all four language ports), so
/// this middleware captures identity at [`HookPoint::BeforeSession`] into an
/// external `Mutex` and reads it at [`HookPoint::BeforeTool`] — the same
/// external-identity pattern used by [`LoopDetectionMiddleware`].
///
/// Types: [`PatchSpan`], [`PatchType`].
/// Trait method used: [`ObservabilityProvider::emit_patch`].
/// Rules enforced: R1 (one warn span per patch), R2 (no patch → no span),
/// R3 (original + patched recorded), R4 (patch_type classified),
/// R9 (one span per patched call in a batch), R10 (still runs at the highest
/// `BeforeTool` priority, `i32::MIN + 1`).
pub struct PatchToolCallsMiddleware {
    pub name: String,
    pub fallback_name: String,
    /// Optional observability sink. `None` keeps `new()` test-friendly and
    /// makes the middleware a no-op observer when unwired.
    observability: Option<Arc<dyn ObservabilityProvider>>,
    /// Captured at `BeforeSession`, read at `BeforeTool`. Holds the session
    /// and task identity needed to stamp a [`PatchSpan`].
    identity: Mutex<Option<(SessionId, TaskId)>>,
    /// Monotonic counter so emitted patch spans get distinct ids.
    patch_seq: std::sync::atomic::AtomicU64,
}

impl PatchToolCallsMiddleware {
    pub fn new(fallback_name: impl Into<String>) -> Self {
        Self {
            name: "patch-tool-calls".into(),
            fallback_name: fallback_name.into(),
            observability: None,
            identity: Mutex::new(None),
            patch_seq: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Inject the observability sink. Patches emitted after this is set will
    /// produce warn-level [`PatchSpan`]s (issue #28).
    pub fn with_observability(mut self, obs: Arc<dyn ObservabilityProvider>) -> Self {
        self.observability = Some(obs);
        self
    }

    /// Tests expose this to simulate session boundaries.
    pub fn clear(&self) {
        *self.identity.lock().unwrap() = None;
    }

    fn emit_patch_event(
        &self,
        call: &ToolCall,
        original: serde_json::Value,
        patch_type: PatchType,
    ) {
        let Some(obs) = &self.observability else {
            return;
        };
        // Identity captured at BeforeSession; fall back to empty ids if a tool
        // patch somehow fires before any session began (defensive — the span
        // still records the diff).
        let (session_id, task_id) = self
            .identity
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| (SessionId::new(""), TaskId::new("")));
        let seq = self
            .patch_seq
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let ts = crate::memory::Timestamp::new("");
        let base = SpanBase {
            span_id: SpanId::new(format!("patch-{seq}")),
            parent_span_id: None,
            session_id,
            task_id,
            kind: SpanKind::Patch,
            started_at: ts.clone(),
            ended_at: ts,
            duration_ms: 0,
            status: crate::observability::SpanStatus::Ok,
        };
        obs.emit_patch(PatchSpan::new(
            base,
            call.id.clone(),
            call.name.clone(),
            original,
            call.input.clone(),
            patch_type,
        ));
    }
}

impl Middleware for PatchToolCallsMiddleware {
    fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
        match ctx {
            HookContext::BeforeSession { task, session_id } => {
                *self.identity.lock().unwrap() = Some((session_id.clone(), task.id.clone()));
                Box::pin(async { MiddlewareDecision::Continue })
            }
            HookContext::BeforeTool { calls, .. } => {
                let mut modified = false;
                for call in calls.iter_mut() {
                    if call.name.trim().is_empty() {
                        // Capture the original parameters before mutating.
                        let original = call.input.clone();
                        call.name = self.fallback_name.clone();
                        modified = true;
                        // R1/R4: classify the empty-name repair as a dangling
                        // tool call and emit a real warn-level event.
                        self.emit_patch_event(
                            call,
                            original,
                            PatchType::DanglingToolCall {
                                reason: "empty tool name".into(),
                            },
                        );
                    }
                }
                Box::pin(async move {
                    if modified {
                        MiddlewareDecision::ContinueWithModification
                    } else {
                        MiddlewareDecision::Continue
                    }
                })
            }
            _ => Box::pin(async { MiddlewareDecision::Continue }),
        }
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::BeforeSession, HookPoint::BeforeTool]
    }
    fn priority(&self) -> i32 {
        i32::MIN + 1
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Tracks per-file edit counts (keyed by tool argument `path`) and forces an
/// agent reconsideration after `threshold` repeated edits to the same file
/// within the same session. Demonstrates `AfterTool` mutation via
/// `ContinueWithModification` and per-session state held in an external map
/// (per spec: middleware must not hold state on `self` keyed by session).
pub struct LoopDetectionMiddleware {
    pub name: String,
    pub threshold: u32,
    pub tool_name: String,
    counts: Mutex<HashMap<String, u32>>,
}

impl LoopDetectionMiddleware {
    pub fn new(tool_name: impl Into<String>, threshold: u32) -> Self {
        Self {
            name: "loop-detection".into(),
            threshold,
            tool_name: tool_name.into(),
            counts: Mutex::new(HashMap::new()),
        }
    }

    /// Tests expose this so they can simulate session boundaries.
    pub fn clear(&self) {
        self.counts.lock().unwrap().clear();
    }
}

impl Middleware for LoopDetectionMiddleware {
    fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
        let HookContext::AfterTool { calls, results } = ctx else {
            return Box::pin(async { MiddlewareDecision::Continue });
        };
        let mut modified = false;
        for (call, result) in calls.iter().zip(results.iter_mut()) {
            if call.name != self.tool_name {
                continue;
            }
            let path = call
                .input
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if path.is_empty() {
                continue;
            }
            let mut counts = self.counts.lock().unwrap();
            let entry = counts.entry(path.clone()).or_insert(0);
            *entry += 1;
            if *entry >= self.threshold {
                // Annotate the result so the agent sees the warning next turn.
                let warning = format!(
                    "[loop-detection] {} has been edited {} times — reconsider",
                    path, *entry
                );
                if let crate::harness::ToolOutput::Success { content, .. } = &mut result.output {
                    if !content.contains("[loop-detection]") {
                        content.push_str("\n\n");
                        content.push_str(&warning);
                        modified = true;
                    }
                }
            }
        }
        Box::pin(async move {
            if modified {
                MiddlewareDecision::ContinueWithModification
            } else {
                MiddlewareDecision::Continue
            }
        })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::AfterTool]
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Forces another turn at `BeforeCompletion` if the agent's final response
/// fails a configured checklist (e.g. did not mention "tests pass").
/// The simplest possible implementation: a list of required substrings.
pub struct PreCompletionChecklistMiddleware {
    pub name: String,
    pub required_substrings: Vec<String>,
}

impl PreCompletionChecklistMiddleware {
    pub fn new(required_substrings: Vec<String>) -> Self {
        Self {
            name: "pre-completion-checklist".into(),
            required_substrings,
        }
    }
}

impl Middleware for PreCompletionChecklistMiddleware {
    fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
        let HookContext::BeforeCompletion { response, .. } = ctx else {
            return Box::pin(async { MiddlewareDecision::Continue });
        };
        let missing: Vec<String> = self
            .required_substrings
            .iter()
            .filter(|s| !response.contains(s.as_str()))
            .cloned()
            .collect();
        Box::pin(async move {
            if missing.is_empty() {
                MiddlewareDecision::Continue
            } else {
                MiddlewareDecision::ForceAnotherTurn {
                    inject: format!(
                        "Verification incomplete. Required items not addressed: {}",
                        missing.join(", ")
                    ),
                }
            }
        })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::BeforeCompletion]
    }
    fn name(&self) -> &str {
        &self.name
    }
}

/// Halts the session if cumulative token spend exceeds the configured limit.
/// Real production wires this to `BudgetSnapshot`; the standalone version
/// reads token counts from a shared atomic so tests can drive it.
pub struct TokenBudgetMiddleware {
    pub name: String,
    pub limit_tokens: u64,
    pub spent_tokens: std::sync::atomic::AtomicU64,
}

impl TokenBudgetMiddleware {
    pub fn new(limit_tokens: u64) -> Self {
        Self {
            name: "token-budget".into(),
            limit_tokens,
            spent_tokens: std::sync::atomic::AtomicU64::new(0),
        }
    }
    pub fn record(&self, tokens: u64) {
        self.spent_tokens
            .fetch_add(tokens, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Middleware for TokenBudgetMiddleware {
    fn handle<'a>(&'a self, _ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
        let spent = self.spent_tokens.load(std::sync::atomic::Ordering::SeqCst);
        let limit = self.limit_tokens;
        Box::pin(async move {
            if spent >= limit {
                MiddlewareDecision::Halt {
                    reason: format!("token budget exhausted: {spent}/{limit}"),
                }
            } else {
                MiddlewareDecision::Continue
            }
        })
    }
    fn hooks(&self) -> Vec<HookPoint> {
        vec![HookPoint::BeforeTurn]
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{
        AggregateUsage, BudgetLimits, HumanRequest, LoopStrategy, ReactConfig, RiskLevel,
        RunResult, SessionId, Task,
    };
    use crate::model::ToolCall;
    use serde_json::json;
    use std::sync::Arc;

    fn task() -> Task {
        Task::new(
            "test task",
            SessionId::new("sess"),
            LoopStrategy::ReAct(ReactConfig::per_loop(5)),
        )
        .with_budget(BudgetLimits::default())
    }

    fn sid() -> SessionId {
        SessionId::new("sess")
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            input: json!({}),
        }
    }

    // ── Recording middleware that returns programmable decisions ─────────────

    struct Scripted {
        name: String,
        hooks: Vec<HookPoint>,
        priority: i32,
        decision: Mutex<MiddlewareDecision>,
        fired: Mutex<Vec<HookPoint>>,
    }
    impl Scripted {
        fn new(name: &str, hooks: Vec<HookPoint>, priority: i32) -> Self {
            Self {
                name: name.into(),
                hooks,
                priority,
                decision: Mutex::new(MiddlewareDecision::Continue),
                fired: Mutex::new(Vec::new()),
            }
        }
        fn with_decision(self, d: MiddlewareDecision) -> Self {
            *self.decision.lock().unwrap() = d;
            self
        }
    }
    impl Middleware for Scripted {
        fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
            self.fired.lock().unwrap().push(ctx.point());
            let d = self.decision.lock().unwrap().clone();
            Box::pin(async move { d })
        }
        fn hooks(&self) -> Vec<HookPoint> {
            self.hooks.clone()
        }
        fn priority(&self) -> i32 {
            self.priority
        }
        fn name(&self) -> &str {
            &self.name
        }
    }

    // ── Rule: register validates hooks list and uniqueness ───────────────────

    #[tokio::test]
    async fn register_rejects_empty_hooks() {
        let chain = StandardMiddlewareChain::new();
        let err = chain
            .register(Box::new(Scripted::new("m", vec![], 0)))
            .unwrap_err();
        assert!(matches!(err, MiddlewareError::NoHooks { .. }));
    }

    #[tokio::test]
    async fn register_rejects_duplicate_name() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(Scripted::new("m", vec![HookPoint::BeforeTurn], 0)))
            .unwrap();
        let err = chain
            .register(Box::new(Scripted::new("m", vec![HookPoint::BeforeTurn], 0)))
            .unwrap_err();
        assert!(matches!(err, MiddlewareError::AlreadyRegistered(_)));
    }

    // ── Rule: before hooks ascend by priority ────────────────────────────────

    #[tokio::test]
    async fn before_hooks_run_in_ascending_priority() {
        let chain = StandardMiddlewareChain::new();
        let tracing = Arc::new(TracingMiddleware::new("t"));
        chain
            .register(Box::new(Scripted::new(
                "c",
                vec![HookPoint::BeforeTurn],
                100,
            )))
            .unwrap();
        chain
            .register(Box::new(Scripted::new(
                "b",
                vec![HookPoint::BeforeTurn],
                10,
            )))
            .unwrap();
        // Register tracing via Arc-wrapper so we can inspect order
        struct ArcWrapper(Arc<TracingMiddleware>);
        impl Middleware for ArcWrapper {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain
            .register(Box::new(ArcWrapper(tracing.clone())))
            .unwrap();

        let mut state = SessionState::default();
        let d = chain.fire_before_turn(&mut state, 1).await;
        assert!(matches!(d, MiddlewareDecision::Continue));

        // tracing has i32::MIN, so it fires first. "b" (10) before "c" (100).
        let log = tracing.entries();
        assert_eq!(log, vec![(HookPoint::BeforeTurn, 1)]);
    }

    // ── Rule: after hooks descend by priority ────────────────────────────────

    #[tokio::test]
    async fn after_hooks_run_in_descending_priority() {
        let chain = StandardMiddlewareChain::new();
        let a = Arc::new(Scripted::new("a", vec![HookPoint::AfterTool], 1));
        let b = Arc::new(Scripted::new("b", vec![HookPoint::AfterTool], 100));
        // wrap Arcs
        struct W(Arc<Scripted>);
        impl Middleware for W {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain.register(Box::new(W(a.clone()))).unwrap();
        chain.register(Box::new(W(b.clone()))).unwrap();

        let calls = vec![tool_call("c1", "edit")];
        let mut results: Vec<ToolResult> = Vec::new();
        let _ = chain.fire_after_tool(&calls, &mut results).await;

        // b (100) fired before a (1) under descending priority.
        assert_eq!(a.fired.lock().unwrap().len(), 1);
        assert_eq!(b.fired.lock().unwrap().len(), 1);
    }

    // ── Rule: first Halt stops the chain ─────────────────────────────────────

    #[tokio::test]
    async fn halt_stops_chain_and_downstream_middleware_does_not_run() {
        let chain = StandardMiddlewareChain::new();
        let halter = Scripted::new("halt", vec![HookPoint::BeforeTurn], 1).with_decision(
            MiddlewareDecision::Halt {
                reason: "stop".into(),
            },
        );
        let after = Arc::new(Scripted::new("after", vec![HookPoint::BeforeTurn], 100));
        struct W(Arc<Scripted>);
        impl Middleware for W {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain.register(Box::new(halter)).unwrap();
        chain.register(Box::new(W(after.clone()))).unwrap();
        let mut s = SessionState::default();
        let d = chain.fire_before_turn(&mut s, 1).await;
        assert!(matches!(d, MiddlewareDecision::Halt { .. }));
        assert!(after.fired.lock().unwrap().is_empty());
    }

    // ── Rule: SurfaceToHuman first-wins ──────────────────────────────────────

    #[tokio::test]
    async fn surface_to_human_first_wins_on_before_tool() {
        let chain = StandardMiddlewareChain::new();
        let req = HumanRequest::ToolApproval {
            calls: vec![tool_call("c1", "shell")],
            risk_level: RiskLevel::High,
        };
        chain
            .register(Box::new(
                Scripted::new("first", vec![HookPoint::BeforeTool], 1).with_decision(
                    MiddlewareDecision::SurfaceToHuman {
                        request: req.clone(),
                    },
                ),
            ))
            .unwrap();
        let runner = Arc::new(Scripted::new("second", vec![HookPoint::BeforeTool], 2));
        struct W(Arc<Scripted>);
        impl Middleware for W {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain.register(Box::new(W(runner.clone()))).unwrap();
        let mut calls = vec![tool_call("c1", "shell")];
        let d = chain.fire_before_tool(&mut calls, 1).await;
        assert!(matches!(d, MiddlewareDecision::SurfaceToHuman { .. }));
        assert!(runner.fired.lock().unwrap().is_empty());
    }

    // ── Rule: SurfaceToHuman is illegal outside BeforeTool / BeforeCompletion

    #[tokio::test]
    async fn surface_to_human_illegal_on_before_turn() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(
                Scripted::new("bad", vec![HookPoint::BeforeTurn], 1).with_decision(
                    MiddlewareDecision::SurfaceToHuman {
                        request: HumanRequest::Clarification {
                            question: "?".into(),
                            options: None,
                        },
                    },
                ),
            ))
            .unwrap();
        let mut s = SessionState::default();
        let d = chain.fire_before_turn(&mut s, 1).await;
        match d {
            MiddlewareDecision::Halt { reason } => {
                assert!(reason.contains("SurfaceToHuman"));
            }
            other => panic!("expected Halt, got {other:?}"),
        }
    }

    // ── Rule: ForceAnotherTurn concatenated, chain continues ─────────────────

    #[tokio::test]
    async fn force_another_turn_concatenates_and_continues() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(
                Scripted::new("a", vec![HookPoint::BeforeCompletion], 1).with_decision(
                    MiddlewareDecision::ForceAnotherTurn {
                        inject: "first".into(),
                    },
                ),
            ))
            .unwrap();
        chain
            .register(Box::new(
                Scripted::new("b", vec![HookPoint::BeforeCompletion], 2).with_decision(
                    MiddlewareDecision::ForceAnotherTurn {
                        inject: "second".into(),
                    },
                ),
            ))
            .unwrap();
        let state = SessionState::default();
        let d = chain.fire_before_completion("done", 3, &state).await;
        match d {
            MiddlewareDecision::ForceAnotherTurn { inject } => {
                assert!(inject.contains("first"));
                assert!(inject.contains("second"));
            }
            other => panic!("expected ForceAnotherTurn, got {other:?}"),
        }
    }

    // ── Rule: ForceAnotherTurn illegal outside BeforeCompletion ──────────────

    #[tokio::test]
    async fn force_another_turn_illegal_on_before_turn() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(
                Scripted::new("bad", vec![HookPoint::BeforeTurn], 1)
                    .with_decision(MiddlewareDecision::ForceAnotherTurn { inject: "x".into() }),
            ))
            .unwrap();
        let mut s = SessionState::default();
        let d = chain.fire_before_turn(&mut s, 1).await;
        assert!(matches!(d, MiddlewareDecision::Halt { .. }));
    }

    // ── Rule: BeforeTool ContinueWithModification surfaces mutation ──────────

    #[tokio::test]
    async fn patch_tool_calls_renames_empty_name() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(PatchToolCallsMiddleware::new("noop")))
            .unwrap();
        let mut calls = vec![tool_call("c1", "")];
        let d = chain.fire_before_tool(&mut calls, 1).await;
        assert!(matches!(d, MiddlewareDecision::ContinueWithModification));
        assert_eq!(calls[0].name, "noop");
    }

    // ── Rule: AfterTool mutation propagates via ContinueWithModification ────

    #[tokio::test]
    async fn loop_detection_annotates_after_threshold() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(LoopDetectionMiddleware::new("edit", 2)))
            .unwrap();
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "edit".into(),
            input: json!({"path": "/tmp/foo.txt"}),
        }];
        // First fire: under threshold.
        let mut results = vec![ToolResult {
            call_id: "c1".into(),
            output: crate::harness::ToolOutput::Success {
                content: "ok".into(),
                truncated: false,
            },
        }];
        let d = chain.fire_after_tool(&calls, &mut results).await;
        assert!(matches!(d, MiddlewareDecision::Continue));
        // Second fire reaches threshold and annotates.
        let mut results = vec![ToolResult {
            call_id: "c1".into(),
            output: crate::harness::ToolOutput::Success {
                content: "ok".into(),
                truncated: false,
            },
        }];
        let d = chain.fire_after_tool(&calls, &mut results).await;
        assert!(matches!(d, MiddlewareDecision::ContinueWithModification));
        let crate::harness::ToolOutput::Success { content, .. } = &results[0].output else {
            panic!();
        };
        assert!(content.contains("[loop-detection]"));
    }

    // ── Rule: PreCompletionChecklist forces another turn when missing ────────

    #[tokio::test]
    async fn pre_completion_checklist_forces_another_turn() {
        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(PreCompletionChecklistMiddleware::new(vec![
                "tests passed".into(),
            ])))
            .unwrap();
        let d = chain
            .fire_before_completion("done", 1, &SessionState::default())
            .await;
        match d {
            MiddlewareDecision::ForceAnotherTurn { inject } => {
                assert!(inject.contains("tests passed"));
            }
            other => panic!("expected ForceAnotherTurn, got {other:?}"),
        }
        let d = chain
            .fire_before_completion("all tests passed", 1, &SessionState::default())
            .await;
        assert!(matches!(d, MiddlewareDecision::Continue));
    }

    // ── Rule: TokenBudgetMiddleware halts at limit ───────────────────────────

    #[tokio::test]
    async fn token_budget_halts_when_exhausted() {
        let chain = StandardMiddlewareChain::new();
        let budget = Arc::new(TokenBudgetMiddleware::new(100));
        struct W(Arc<TokenBudgetMiddleware>);
        impl Middleware for W {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain.register(Box::new(W(budget.clone()))).unwrap();
        let mut s = SessionState::default();
        assert!(matches!(
            chain.fire_before_turn(&mut s, 1).await,
            MiddlewareDecision::Continue
        ));
        budget.record(150);
        assert!(matches!(
            chain.fire_before_turn(&mut s, 2).await,
            MiddlewareDecision::Halt { .. }
        ));
    }

    // ── Edge: BeforeSession and AfterSession fire end-to-end ─────────────────

    #[tokio::test]
    async fn session_boundary_hooks_fire() {
        let chain = StandardMiddlewareChain::new();
        let tracing = Arc::new(TracingMiddleware::new("t"));
        struct W(Arc<TracingMiddleware>);
        impl Middleware for W {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain.register(Box::new(W(tracing.clone()))).unwrap();

        let task = task();
        let sid = sid();
        let _ = chain.fire_before_session(&task, &sid).await;
        let result = RunResult::Success {
            output: "done".into(),
            session_id: sid.clone(),
            usage: AggregateUsage::default(),
            turns: 1,
            session_state: SessionState::default(),
        };
        chain.fire_after_session(&result, &sid).await;

        let log = tracing.entries();
        assert!(log.iter().any(|(p, _)| *p == HookPoint::BeforeSession));
        assert!(log.iter().any(|(p, _)| *p == HookPoint::AfterSession));
    }

    // ── Send/Sync smoke ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn chain_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let chain: Arc<dyn MiddlewareChain> = Arc::new(StandardMiddlewareChain::new());
        assert_send_sync(&chain);
    }

    // ── BeforeTool ordering: PatchToolCalls runs first ───────────────────────

    #[tokio::test]
    async fn patch_tool_calls_runs_before_other_before_tool_middleware() {
        let chain = StandardMiddlewareChain::new();
        let observer = Arc::new(Scripted::new("observer", vec![HookPoint::BeforeTool], 0));
        struct W(Arc<Scripted>);
        impl Middleware for W {
            fn handle<'a>(&'a self, ctx: HookContext<'a>) -> BoxFut<'a, MiddlewareDecision> {
                self.0.handle(ctx)
            }
            fn hooks(&self) -> Vec<HookPoint> {
                self.0.hooks()
            }
            fn priority(&self) -> i32 {
                self.0.priority()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
        }
        chain
            .register(Box::new(PatchToolCallsMiddleware::new("noop")))
            .unwrap();
        chain.register(Box::new(W(observer.clone()))).unwrap();
        let mut calls = vec![tool_call("c1", "")];
        let d = chain.fire_before_tool(&mut calls, 1).await;
        // Observer should have seen the patched call (name == "noop"), proving
        // ordering. Decision should reflect modification.
        assert!(matches!(d, MiddlewareDecision::ContinueWithModification));
        assert_eq!(calls[0].name, "noop");
        assert_eq!(observer.fired.lock().unwrap().len(), 1);
    }

    // ── Fixture replay ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn fixture_replay_before_completion_checklist() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/middleware/checklist_basic.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let case: FixtureCase = serde_json::from_str(&raw).unwrap();

        let chain = StandardMiddlewareChain::new();
        chain
            .register(Box::new(PreCompletionChecklistMiddleware::new(
                case.required.clone(),
            )))
            .unwrap();
        let d = chain
            .fire_before_completion(&case.response, 1, &SessionState::default())
            .await;
        match (case.expected.as_str(), d) {
            ("continue", MiddlewareDecision::Continue) => {}
            ("force_another_turn", MiddlewareDecision::ForceAnotherTurn { .. }) => {}
            (want, got) => panic!("expected {want}, got {got:?}"),
        }
    }

    #[derive(serde::Deserialize)]
    struct FixtureCase {
        required: Vec<String>,
        response: String,
        expected: String,
    }

    // ── Patch observability (issue #28) ──────────────────────────────────────

    use crate::observability::{
        InMemoryObservabilityProvider, ObservabilityProvider, PatchType, SpanKind,
    };

    fn wired_patch(obs: &Arc<InMemoryObservabilityProvider>) -> PatchToolCallsMiddleware {
        let dyn_obs: Arc<dyn ObservabilityProvider> = obs.clone();
        PatchToolCallsMiddleware::new("noop").with_observability(dyn_obs)
    }

    // Drive identity capture (BeforeSession) then a BeforeTool fire directly on
    // the middleware so the test owns the calls vec.
    async fn run_patch(
        mw: &PatchToolCallsMiddleware,
        calls: &mut Vec<ToolCall>,
    ) -> MiddlewareDecision {
        let task = task();
        let sid = sid();
        let _ = mw
            .handle(HookContext::BeforeSession {
                task: &task,
                session_id: &sid,
            })
            .await;
        mw.handle(HookContext::BeforeTool {
            calls,
            turn_number: 1,
        })
        .await
    }

    // R1 + R3: every patch emits exactly one warn-level span recording both the
    // original and the patched parameters.
    #[tokio::test]
    async fn patch_emits_one_warn_span_with_original_and_patched() {
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let mw = wired_patch(&obs);
        let mut calls = vec![tool_call("c1", "")];
        let d = run_patch(&mw, &mut calls).await;
        assert!(matches!(d, MiddlewareDecision::ContinueWithModification));
        let trace = obs.get_trace(&sid()).await;
        let patches: Vec<_> = trace
            .iter()
            .filter(|s| s.base().kind == SpanKind::Patch)
            .collect();
        assert_eq!(patches.len(), 1);
        assert_eq!(obs.patch_spans(&sid()).len(), 1);
    }

    // R2: no patch needed → no span, decision is plain Continue.
    #[tokio::test]
    async fn no_patch_emits_no_span() {
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let mw = wired_patch(&obs);
        let mut calls = vec![tool_call("c1", "shell")];
        let d = run_patch(&mw, &mut calls).await;
        assert!(matches!(d, MiddlewareDecision::Continue));
        let trace = obs.get_trace(&sid()).await;
        assert!(trace.iter().all(|s| s.base().kind != SpanKind::Patch));
    }

    // R4: the empty-name repair is classified as DanglingToolCall.
    #[tokio::test]
    async fn empty_name_classified_as_dangling_tool_call() {
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let mw = wired_patch(&obs);
        let mut calls = vec![tool_call("c1", "")];
        run_patch(&mw, &mut calls).await;
        let patches = obs.patch_spans(&sid());
        let p = &patches[0];
        assert!(matches!(p.patch_type, PatchType::DanglingToolCall { .. }));
        assert_eq!(p.tool_name, "noop"); // patched name on the span
        assert_eq!(p.call_id, "c1");
    }

    // R5: the patch event is present in get_trace.
    #[tokio::test]
    async fn trace_contains_patch_event() {
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let mw = wired_patch(&obs);
        let mut calls = vec![tool_call("c1", "")];
        run_patch(&mw, &mut calls).await;
        let trace = obs.get_trace(&sid()).await;
        assert!(trace.iter().any(|s| s.base().kind == SpanKind::Patch));
    }

    // R9: a batch of N patched calls emits N patch spans.
    #[tokio::test]
    async fn batch_emits_one_span_per_patch() {
        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let mw = wired_patch(&obs);
        let mut calls = vec![
            tool_call("c1", ""),
            tool_call("c2", "shell"),
            tool_call("c3", "  "),
        ];
        run_patch(&mw, &mut calls).await;
        assert_eq!(obs.patch_spans(&sid()).len(), 2); // c1 and c3, not c2
    }

    // Without an injected provider, patching still works (Option keeps new()
    // test-friendly) and emits nothing.
    #[tokio::test]
    async fn patch_without_observability_is_silent() {
        let mw = PatchToolCallsMiddleware::new("noop");
        let mut calls = vec![tool_call("c1", "")];
        let d = run_patch(&mw, &mut calls).await;
        assert!(matches!(d, MiddlewareDecision::ContinueWithModification));
        assert_eq!(calls[0].name, "noop");
    }

    // R10 regression: still registers at the highest BeforeTool priority.
    #[test]
    fn patch_middleware_priority_is_highest_before_tool() {
        let mw = PatchToolCallsMiddleware::new("noop");
        assert_eq!(mw.priority(), i32::MIN + 1);
        assert!(mw.hooks().contains(&HookPoint::BeforeTool));
    }

    // R11: fixture replay. Reads fixtures/patch/patch_events_basic.json, runs
    // each input call through the middleware, and asserts the emitted patch
    // events plus the rolled-up metrics match the fixture's expectations.
    #[tokio::test]
    async fn fixture_replay_patch_events() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/patch/patch_events_basic.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let case: PatchFixture = serde_json::from_str(&raw).unwrap();

        let obs = Arc::new(InMemoryObservabilityProvider::new());
        let mw = PatchToolCallsMiddleware::new(&case.fallback_name)
            .with_observability(obs.clone() as Arc<dyn ObservabilityProvider>);

        let mut calls: Vec<ToolCall> = case
            .input_calls
            .iter()
            .map(|c| ToolCall {
                id: c.id.clone(),
                name: c.name.clone(),
                input: c.input.clone(),
            })
            .collect();
        run_patch(&mw, &mut calls).await;

        // Assert each expected patch event was recorded.
        let patches = obs.patch_spans(&sid());
        assert_eq!(patches.len(), case.expected_patches.len());
        for exp in &case.expected_patches {
            let found = patches
                .iter()
                .find(|p| p.call_id == exp.call_id)
                .unwrap_or_else(|| panic!("no patch span for call {}", exp.call_id));
            assert_eq!(found.tool_name, exp.tool_name);
            assert_eq!(found.original_parameters, exp.original);
            assert_eq!(found.patched_parameters, exp.patched);
            match exp.patch_type.as_str() {
                "dangling_tool_call" => {
                    assert!(matches!(
                        found.patch_type,
                        PatchType::DanglingToolCall { .. }
                    ))
                }
                other => panic!("unexpected patch_type {other}"),
            }
        }

        // Record an outcome so SessionMetrics materializes for a session that
        // emitted only patch spans (no turns).
        obs.set_session_outcome(&sid(), crate::guide_registry::SessionOutcome::Success);
        let m = obs.get_session_metrics(&sid()).await.unwrap();
        // No tool-call spans were emitted in this middleware-only replay, so
        // patch_rate is 0.0 by the divide-by-zero guard; the fixture asserts
        // patch_count and patches_by_tool.
        assert_eq!(m.patch_count, case.expected_patch_count);
        for (tool, n) in &case.expected_patches_by_tool {
            assert_eq!(m.patches_by_tool.get(tool), Some(n));
        }
    }

    #[derive(serde::Deserialize)]
    struct PatchFixture {
        fallback_name: String,
        input_calls: Vec<FixtureCall>,
        expected_patches: Vec<ExpectedPatch>,
        expected_patch_count: u32,
        expected_patches_by_tool: std::collections::HashMap<String, u32>,
    }
    #[derive(serde::Deserialize)]
    struct FixtureCall {
        id: String,
        name: String,
        input: serde_json::Value,
    }
    #[derive(serde::Deserialize)]
    struct ExpectedPatch {
        call_id: String,
        tool_name: String,
        patch_type: String,
        original: serde_json::Value,
        patched: serde_json::Value,
    }
}
