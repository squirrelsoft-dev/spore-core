//! Harness — the agent runtime loop.
//!
//! Implements issue #3. The harness owns execution lifecycle and wires
//! all components together. It is stateless between `run()` calls; everything
//! the harness needs comes in via [`HarnessRunOptions`] or [`PausedState`],
//! and everything it produces goes out via [`RunResult`].
//!
//! ## `dangerous` feature gate
//!
//! `IsolationMode::None` (no path enforcement) is a named safety footgun and is
//! only compiled when the `dangerous` Cargo feature is enabled (issue #34). The
//! default build omits it, so using it is a compile error rather than a runtime
//! warning. Consequently the [`SandboxProvider::isolation_mode`] default body is
//! now `IsolationMode::WorkspaceScoped` (safe-by-default) instead of `None`. The
//! wire tag for the gated variant stays `"none"`.
//!
//! ## What this component does
//!
//! - Assemble context (via `ContextManager`) before each turn
//! - Call the agent for one turn
//! - Dispatch tool calls to `ToolRegistry`
//! - Evaluate `TerminationPolicy` after each turn
//! - Fire middleware lifecycle hooks
//! - Track iterations, token spend, elapsed time
//! - Pause and resume for human-in-the-loop interactions
//!
//! ## What this component does NOT do
//!
//! - Touch the filesystem, execute commands, or call the model directly
//! - Persist `PausedState` — the caller owns persistence
//! - Implement individual tools, sandbox policy, or context assembly
//!
//! ## Rules enforced here
//!
//! 1. Harness owns the loop — the agent only executes one turn at a time.
//! 2. Termination is evaluated against external state via `TerminationPolicy`.
//! 3. Any budget overrun terminates the loop with an explicit `HaltReason`.
//! 4. A turn that yields neither a tool call nor a final response is an error
//!    (surfaced via `AgentError`, routed through error-propagation rules).
//! 5. All components are injected at construction; the harness never builds
//!    them itself.
//! 6. Stateless between pause and resume — the caller owns `PausedState`.
//! 7. `WaitingForHuman` returns immediately; no internal timeout.
//! 8. `approved_results` prevents double-execution on resume.
//! 9. Subagents cannot spawn their own subagents — [`ChildPausedState`] has
//!    no `child_state` field (compile-time depth-1 enforcement).
//! 10. Tool Escalation Protocol (issue #80): a tool may return
//!     [`ToolOutput::Escalate`] carrying a [`HarnessSignal`]. The harness is a
//!     pure intermediary — it does NOT act on the signal. It terminates
//!     cleanly WITHOUT appending the escalation to message history (it is a
//!     control signal, not a turn), preserves remaining tool calls into
//!     `pending_tool_calls`, finalizes observability with
//!     [`SessionOutcome::Escalated`](crate::guide_registry::SessionOutcome::Escalated),
//!     and returns [`RunResult::Escalate`]. The signal is NOT stored in
//!     [`PausedState`], so it is discarded on `resume` — the harness never
//!     re-acts on it. `HarnessSignal::Abort` surfaces as `RunResult::Escalate`,
//!     NOT `RunResult::Failure`. (No `// SPEC QUESTION:` markers remain — all
//!     four pre-implementation ambiguities were resolved.)
//!
//! ## Component dependencies (forward declarations)
//!
//! Many of the trait dependencies of the harness (`ToolRegistry`,
//! `SandboxProvider`, `ContextManager`, …) ship in their own component
//! issues (#4–#13). Until those land, this module defines minimal forward
//! declarations of the trait surface the loop actually consumes. Each is
//! tagged with the owning issue. When a sibling issue lands its canonical
//! definition will replace the stub here.
//!
//! ## Cross-language note
//!
//! The shape of `Task`, `BudgetLimits`, `RunResult`, `HaltReason`,
//! `PausedState`, `ChildPausedState`, and `HumanRequest` / `HumanResponse`
//! mirrors byte-for-byte across TypeScript, Python, and Go.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::agent::{Agent, AgentError, Context, TurnResult};
use crate::context::{
    CompactionPreserveHints, CompactionVerifier, ContextError, KeyTermVerifier,
    SessionState as ContextSessionState,
};
use crate::guide_registry::SessionOutcome;
use crate::memory::Timestamp;
use crate::model::{Content, Message, Role, StopReason, TokenUsage, ToolCall, ToolSchema};
use crate::observability::{
    truncate_field, ContentCaptureConfig, GenAiMessage, GenAiRole, PricingTable, SpanBase, SpanId,
    SpanKind, SpanStatus, ToolCallContent, ToolCallSpan, ToolResultContent, TurnSpan, WarnEvent,
    WarnSpan,
};
use crate::observability_outbox::{OutboxConfig, OutboxObservabilityProvider};
use crate::tool_call_repair::ToolCallRepair;
use crate::verifier::{VerifierInput, VerifierVerdict};

/// Boxed future alias used to make the component traits `dyn`-compatible.
/// `trait_variant::make(Send)` generates RPITIT which is not dyn-safe; the
/// harness needs `Arc<dyn Trait>` everywhere, so we hand-roll the future
/// return shape.
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

// ============================================================================
// Identity newtypes
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Generate a fresh, opaque session id. Used internally where the spec
    /// requires guaranteed-fresh sessions (e.g. the SelfVerifying evaluator).
    pub fn generate() -> Self {
        Self(format!("sess-{}", random_id()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn generate() -> Self {
        Self(format!("task-{}", random_id()))
    }
}

fn random_id() -> String {
    // Deterministic-enough opaque id for tests / non-crypto identity.
    // Real binaries can wrap with uuid; spore-core stays dep-light.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{n:016x}")
}

// ============================================================================
// Budget tracking
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BudgetLimits {
    #[serde(default)]
    pub max_turns: Option<u32>,
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default, with = "duration_secs_opt")]
    pub max_wall_time: Option<Duration>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
}

mod duration_secs_opt {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(v: &Option<Duration>, s: S) -> Result<S::Ok, S::Error> {
        v.map(|d| d.as_secs()).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(d)?.map(Duration::from_secs))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetLimitType {
    Turns,
    InputTokens,
    OutputTokens,
    WallTime,
    CostUsd,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BudgetSnapshot {
    pub turns: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, with = "duration_secs_opt")]
    pub wall_time: Option<Duration>,
    #[serde(default)]
    pub cost_usd: f64,
}

/// Aggregated usage reported on every [`RunResult`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AggregateUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
    #[serde(default)]
    pub cost_usd: f64,
}

impl AggregateUsage {
    pub fn add_turn(&mut self, u: &TokenUsage) {
        self.input_tokens += u.input_tokens as u64;
        self.output_tokens += u.output_tokens as u64;
        self.cache_read_tokens += u.cache_read_tokens.unwrap_or(0) as u64;
        self.cache_write_tokens += u.cache_write_tokens.unwrap_or(0) as u64;
    }
}

// ============================================================================
// Task + loop strategy
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
}

/// Loop strategy. The data shape is canonical; concrete strategy traits
/// (`CompletionCheck`, `Verifier`, `MetricEvaluator`) are owned by their
/// respective component issues. [`LoopStrategy::ReAct`] (issue #57) and
/// [`LoopStrategy::PlanExecute`] (issues #70/#59) are fully executable in
/// [`StandardHarness`]; the remaining variants still halt with
/// [`HaltReason::StrategyNotYetImplemented`] until their component issues land.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoopStrategy {
    ReAct {
        max_iterations: u32,
    },
    PlanExecute {
        plan_model: Option<ModelConfig>,
    },
    Ralph,
    SelfVerifying,
    HillClimbing {
        direction: OptimizationDirection,
        max_stagnation: Option<u32>,
        revert_on_no_improvement: bool,
        min_improvement_delta: Option<f64>,
    },
}

/// Lightweight placeholder for an alternate planner model selection. Full
/// shape lives wherever provider routing lands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub instruction: String,
    pub session_id: SessionId,
    pub budget: BudgetLimits,
    pub loop_strategy: LoopStrategy,
}

impl Task {
    pub fn new(
        instruction: impl Into<String>,
        session_id: SessionId,
        loop_strategy: LoopStrategy,
    ) -> Self {
        Self {
            id: TaskId::generate(),
            instruction: instruction.into(),
            session_id,
            budget: BudgetLimits::default(),
            loop_strategy,
        }
    }
    pub fn with_budget(mut self, budget: BudgetLimits) -> Self {
        self.budget = budget;
        self
    }
}

// ============================================================================
// Streaming events emitted by the harness
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    TurnStart {
        turn: u32,
    },
    TurnEnd {
        turn: u32,
    },
    ToolCall {
        call_id: String,
        name: String,
    },
    ToolResult {
        call_id: String,
        is_error: bool,
    },
    FinalResponse {
        content: String,
    },
    BudgetWarning {
        limit_type: BudgetLimitType,
    },
    /// Emitted when a tool wants to surface a message to the user out-of-band
    /// (issue #81 — `SendMessageTool`). The harness loop recognizes the
    /// `send_message` tool, emits this event instead of collapsing the content
    /// into a normal tool result, and records a minimal success tool result so
    /// the loop continues.
    UserMessage {
        content: String,
    },
}

pub type StreamSink = Box<dyn Fn(StreamEvent) + Send + Sync>;

// ============================================================================
// Forward-declared sibling component types
// (full surfaces live in their owning issues)
// ============================================================================

/// Tool Escalation Protocol — the typed channel by which a tool signals the
/// harness to terminate cleanly and pass a *structural* state change up to its
/// caller (issue #80).
///
/// The harness is a pure intermediary: it never acts on a signal itself. Mode
/// switching, plan approval, and graceful abort are the caller's concern. The
/// harness terminates cleanly, surfaces the signal via
/// [`RunResult::Escalate`], and the caller (CLI, chat UI, REST API, parent
/// harness) owns the orchestration. This mirrors the [`RunResult::WaitingForHuman`]
/// model — the harness does not resume itself either.
///
/// ## Variants
/// - [`HarnessSignal::EnterPlanMode`] — agent requests entry into plan mode,
///   carrying accumulated context as a seed for the planning harness.
/// - [`HarnessSignal::ExitPlanMode`] — planning agent's terminal signal,
///   carrying the produced [`PlanArtifact`](crate::plan::PlanArtifact) for
///   human approval before an execution harness is instantiated.
/// - [`HarnessSignal::SwitchMode`] — agent requests a mode switch; carries the
///   target [`Mode`](crate::prompt_chunk_registry::Mode) (the EXISTING mode
///   enum — there is no separate `HarnessMode` type).
/// - [`HarnessSignal::Abort`] — agent requests a graceful, intentional stop
///   with a reason. Distinct from `HaltReason::AgentError` — it surfaces as
///   `RunResult::Escalate`, NOT `RunResult::Failure`.
///
/// Wire format: serde-tagged on `kind`, snake_case, byte-identical across the
/// four language implementations. Round-tripped by
/// `fixtures/harness/escalation_signals.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HarnessSignal {
    /// Agent requests entry into plan mode. Carries context the agent has
    /// accumulated so far as seed for the planning harness.
    EnterPlanMode { context: String },
    /// Planning agent has produced a plan and requests exit from plan mode.
    /// Carries the plan artifact for human approval before the execution
    /// harness is instantiated. This is the planning agent's terminal signal.
    ExitPlanMode { plan: crate::plan::PlanArtifact },
    /// Agent requests a mode switch. The caller instantiates the appropriate
    /// harness for the new mode.
    SwitchMode {
        mode: crate::prompt_chunk_registry::Mode,
    },
    /// Agent requests a graceful abort with a reason surfaced to the user.
    /// Distinct from `HaltReason::AgentError` — this is an intentional,
    /// agent-initiated stop and surfaces as `RunResult::Escalate`.
    Abort { reason: String },
}

/// Output of a single tool dispatch. Full type lives in issue #4 (ToolRegistry)
/// / #5 (Tool). The variants below cover what the harness loop needs to
/// route; richer payloads are additive.
///
/// The [`ToolOutput::Escalate`] variant is the tool-side entry point of the
/// Tool Escalation Protocol (issue #80): when a dispatched tool returns it, the
/// harness terminates cleanly (NOT appending the escalation to message
/// history — it is a control signal, not a conversation turn) and returns
/// [`RunResult::Escalate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolOutput {
    Success {
        content: String,
        #[serde(default)]
        truncated: bool,
    },
    Error {
        message: String,
        recoverable: bool,
    },
    /// Boxed because `ChildPausedState` is significantly larger than the other
    /// variants; keeps `ToolOutput` cheap to clone on the happy path.
    WaitingForHuman {
        child_state: Box<ChildPausedState>,
        request: HumanRequest,
    },
    /// Tool requests a structural state change from the harness's parent
    /// (issue #80). The harness terminates cleanly and passes the signal to
    /// the caller via [`RunResult::Escalate`]. The escalation is NOT appended
    /// to message history.
    Escalate {
        signal: HarnessSignal,
    },
    /// Tool requests a clarifying answer from the human before it can produce a
    /// result (issue #81, Q4b — `AskUserQuestionTool`). UNLIKE
    /// [`ToolOutput::WaitingForHuman`], there is NO [`ChildPausedState`]: the
    /// harness loop builds a [`PausedState`] directly, sets its `human_request`
    /// to [`HumanRequest::Clarification`], and returns
    /// [`RunResult::WaitingForHuman`]. The clarifying tool call is preserved in
    /// `pending_tool_calls`; on resume the human's [`HumanResponse::Answer`]
    /// text is injected as the tool RESULT for that pending call (not appended
    /// as a free-standing user message) and the loop continues.
    AwaitingClarification {
        question: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        options: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub output: ToolOutput,
}

/// Sandbox-side violation. Full enum lives in issue #6 (SandboxProvider).
/// `PathEscape` and `NetworkViolation` are Layer-1 always-halt per the spec;
/// the remaining variants are middleware-routable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SandboxViolation {
    PathEscape {
        path: String,
    },
    NetworkViolation {
        host: String,
    },
    PathDenied {
        path: String,
        #[serde(default)]
        matched_rule: String,
    },
    ReadOnlyViolation {
        path: String,
    },
    ExtensionDenied {
        path: String,
        extension: String,
    },
    FileSizeExceeded {
        path: String,
        size: u64,
        limit: u64,
    },
    DisallowedCommand {
        command: String,
    },
}

impl SandboxViolation {
    /// Layer-1 violations cannot be overridden — they always halt.
    pub fn is_always_halt(&self) -> bool {
        matches!(
            self,
            SandboxViolation::PathEscape { .. } | SandboxViolation::NetworkViolation { .. }
        )
    }
}

/// Where in the lifecycle a middleware hook fired. Full enum lives in #11.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPoint {
    BeforeTurn,
    BeforeTool,
    AfterTool,
    BeforeCompletion,
}

/// Termination-policy decision. Full enum lives in #13.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerminationDecision {
    Continue,
    Halt { reason: String },
}

/// Session state handed back and forth across pause/resume. The harness does
/// not interpret its contents; it round-trips opaquely so that #7
/// (ContextManager) and #8 (MemoryProvider) own the schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SessionState {
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

// ============================================================================
// Forward-declared sibling component traits
// ============================================================================

/// Issue #4 — ToolRegistry: dispatches tool calls.
pub trait ToolRegistry: Send + Sync {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput>;

    /// Whether this tool is annotated `always_halt` (e.g. dangerous external
    /// effects). The harness short-circuits on the first such call in a batch.
    fn is_always_halt(&self, tool_name: &str) -> bool {
        let _ = tool_name;
        false
    }

    /// Tool schemas to feed into context assembly.
    fn schemas(&self) -> Vec<ToolSchema> {
        Vec::new()
    }
}

/// Issue #6 — SandboxProvider: validates tool calls against sandbox policy.
///
/// Issue #5 adds default implementations for `execute_command`,
/// `handle_large_output`, and `resolve_path` so the standard tool catalogue
/// can be built before #6 lands its canonical sandbox. **These defaults are
/// NOT production-safe**: `execute_command` spawns processes directly with no
/// sandboxing, `resolve_path` returns the input as-is, and
/// `handle_large_output` truncates inline without offloading. Issue #6 must
/// override these.
pub trait SandboxProvider: Send + Sync {
    fn validate<'a>(&'a self, call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>>;

    /// Run a subprocess. **Default impl spawns directly with no sandboxing —
    /// production sandboxes must override.**
    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        working_dir: Option<&'a std::path::Path>,
        timeout: Option<std::time::Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        Box::pin(async move {
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args);
            if let Some(dir) = working_dir {
                cmd.current_dir(dir);
            }
            let fut = cmd.output();
            let output_res = if let Some(t) = timeout {
                match tokio::time::timeout(t, fut).await {
                    Ok(r) => r,
                    Err(_) => {
                        return Ok(CommandOutput {
                            stdout: String::new(),
                            stderr: format!("command timed out after {}s", t.as_secs()),
                            exit_code: -1,
                            timed_out: true,
                            truncated: false,
                        });
                    }
                }
            } else {
                fut.await
            };
            match output_res {
                Ok(out) => Ok(CommandOutput {
                    stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&out.stderr).to_string(),
                    exit_code: out.status.code().unwrap_or(-1),
                    timed_out: false,
                    truncated: false,
                }),
                Err(e) => Ok(CommandOutput {
                    stdout: String::new(),
                    stderr: format!("spawn failed: {e}"),
                    exit_code: -1,
                    timed_out: false,
                    truncated: false,
                }),
            }
        })
    }

    /// Truncate large output head+tail. Default treats `head_tokens * 4` as
    /// an approximate char budget per half. Production sandboxes should
    /// offload the full content to a file and return a `FileRef`.
    fn handle_large_output<'a>(
        &'a self,
        content: String,
        _call_id: &'a str,
        head_tokens: u32,
        tail_tokens: u32,
    ) -> BoxFut<'a, TruncatedOutput> {
        Box::pin(async move {
            let head_chars = (head_tokens as usize).saturating_mul(4);
            let tail_chars = (tail_tokens as usize).saturating_mul(4);
            let total = content.chars().count();
            let original_size = content.len() as u64;
            if total <= head_chars + tail_chars {
                return TruncatedOutput {
                    content,
                    truncated: false,
                    full_ref: None,
                    original_size,
                };
            }
            let head: String = content.chars().take(head_chars).collect();
            let tail: String = content.chars().skip(total - tail_chars).collect();
            let elided = total - head_chars - tail_chars;
            let snippet = format!("{head}\n... [{elided} chars elided] ...\n{tail}");
            TruncatedOutput {
                content: snippet,
                truncated: true,
                full_ref: None,
                original_size,
            }
        })
    }

    /// Resolve / canonicalize a path against the sandbox root. Default is
    /// an identity pass-through; production sandboxes must enforce roots.
    ///
    /// `operation` lets implementations distinguish reads from writes so they
    /// can apply `read_only`-style policies and skip canonicalization for
    /// not-yet-existing files on writes.
    fn resolve_path<'a>(
        &'a self,
        path: &'a str,
        operation: Operation,
    ) -> BoxFut<'a, Result<std::path::PathBuf, SandboxViolation>> {
        let _ = operation;
        Box::pin(async move { Ok(std::path::PathBuf::from(path)) })
    }

    /// Active isolation mode. Used by observability and middleware. Default
    /// returns `IsolationMode::WorkspaceScoped` (safe-by-default, issue #34).
    /// A provider that genuinely wants no isolation must override this and opt
    /// in to `IsolationMode::None`, which only exists under the `dangerous`
    /// feature.
    fn isolation_mode(&self) -> IsolationMode {
        IsolationMode::WorkspaceScoped
    }

    /// Workspace root used by `ContextManager` for directory-map injection.
    /// Default returns `/`, which is **not** safe — production sandboxes
    /// must override.
    fn workspace_root(&self) -> &std::path::Path {
        std::path::Path::new("/")
    }
}

/// Filesystem operation being performed on a path. Passed to
/// [`SandboxProvider::resolve_path`] so the sandbox can apply read-only
/// policies and adjust canonicalization behavior for not-yet-existing files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Read,
    Write,
    Execute,
}

/// Isolation strategy for sandboxed subprocess execution.
///
/// Lives on the trait so middleware and observability can route on it
/// without depending on the concrete sandbox impl.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IsolationMode {
    /// No isolation. Trusted-dev use only; the sandbox must emit a warning
    /// at construction time. Gated behind the `dangerous` feature (issue #34);
    /// absent from the default build. The wire tag stays `"none"`.
    #[cfg(feature = "dangerous")]
    None,
    /// Path enforcement only — no process or network isolation. Default for
    /// the canonical `WorkspaceScopedSandbox`.
    WorkspaceScoped,
    /// Linux process isolation via bubblewrap. Reserved; not implemented in
    /// the v1 reference sandbox.
    Bubblewrap { profile: BwrapProfile },
    /// Full Docker isolation including network policy. Reserved; not
    /// implemented in the v1 reference sandbox.
    Docker {
        image: String,
        network: NetworkPolicy,
    },
}

/// Network egress policy for `IsolationMode::Docker`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NetworkPolicy {
    /// No network access at all.
    None,
    /// Allow egress to the listed hosts only.
    Allowlist { hosts: Vec<String> },
    /// Unrestricted egress. Use with caution.
    Full,
}

/// Bubblewrap profile descriptor. Opaque in v1 — the bubblewrap backend is
/// not yet implemented, but the shape lives on the public surface so the
/// `IsolationMode` enum can be exhaustive across all four implementations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BwrapProfile {}

/// Read-only decorator over any [`SandboxProvider`] (issue #61). Wraps an
/// inner sandbox and additionally rejects every WRITE/EXECUTE-classified tool
/// call in [`validate`](SandboxProvider::validate) with
/// [`SandboxViolation::ReadOnlyViolation`], while delegating reads (and all the
/// other `SandboxProvider` methods) to the inner sandbox unchanged.
///
/// The `SelfVerifying` strategy derives one of these INTERNALLY for its
/// evaluate phase (the Default-FAIL contract: the evaluator must not be able to
/// mutate the workspace it is reviewing). Write-intent is classified by tool
/// name against [`Self::DEFAULT_WRITE_TOOLS`] (the standard catalogue's
/// mutating tools: `write_file`, `edit_file`, `delete_file`, `move_file`,
/// `exec`, `bash_command`, `run_tests`). Callers with bespoke tool names can
/// supply their own set via [`with_write_tools`](Self::with_write_tools).
///
/// `ReadOnlyViolation` is a Layer-2 (recoverable) violation, so in the harness
/// loop a blocked write surfaces to the evaluator agent as a recoverable tool
/// error — it does NOT halt the evaluate run.
pub struct ReadOnlySandbox {
    inner: Arc<dyn SandboxProvider>,
    write_tools: std::collections::HashSet<String>,
}

impl ReadOnlySandbox {
    /// Standard-catalogue tool names that MUTATE the workspace and are therefore
    /// blocked by a read-only sandbox.
    pub const DEFAULT_WRITE_TOOLS: &'static [&'static str] = &[
        "write_file",
        "edit_file",
        "delete_file",
        "move_file",
        "exec",
        "bash_command",
        "run_tests",
    ];

    /// Wrap `inner`, blocking the standard mutating tools.
    pub fn new(inner: Arc<dyn SandboxProvider>) -> Self {
        Self {
            inner,
            write_tools: Self::DEFAULT_WRITE_TOOLS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// Wrap `inner`, blocking exactly the supplied tool names (overrides the
    /// default set).
    pub fn with_write_tools(
        inner: Arc<dyn SandboxProvider>,
        write_tools: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            inner,
            write_tools: write_tools.into_iter().collect(),
        }
    }

    fn is_write(&self, tool_name: &str) -> bool {
        self.write_tools.contains(tool_name)
    }
}

impl SandboxProvider for ReadOnlySandbox {
    fn validate<'a>(&'a self, call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async move {
            if self.is_write(&call.name) {
                return Err(SandboxViolation::ReadOnlyViolation {
                    path: call.name.clone(),
                });
            }
            self.inner.validate(call).await
        })
    }

    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        _args: &'a [String],
        _working_dir: Option<&'a std::path::Path>,
        _timeout: Option<std::time::Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        // A read-only sandbox forbids subprocess execution outright (commands
        // may have arbitrary write side effects).
        Box::pin(async move {
            Err(SandboxViolation::ReadOnlyViolation {
                path: command.to_string(),
            })
        })
    }

    fn resolve_path<'a>(
        &'a self,
        path: &'a str,
        operation: Operation,
    ) -> BoxFut<'a, Result<std::path::PathBuf, SandboxViolation>> {
        Box::pin(async move {
            if matches!(operation, Operation::Write | Operation::Execute) {
                return Err(SandboxViolation::ReadOnlyViolation {
                    path: path.to_string(),
                });
            }
            self.inner.resolve_path(path, operation).await
        })
    }

    fn isolation_mode(&self) -> IsolationMode {
        self.inner.isolation_mode()
    }

    fn workspace_root(&self) -> &std::path::Path {
        self.inner.workspace_root()
    }
}

// ============================================================================
// VcsProvider seam (issue #58 v2) — git-log reload for the Ralph loop strategy
// ============================================================================

/// Read-only VCS abstraction the `Ralph` loop strategy uses to reload git
/// history between context windows (issue #58 v2, decision B4).
///
/// The v1 Ralph reload (commit `927cc57`) re-seeded each fresh context window
/// from `.spore/progress.json` + `.spore/feature_list.json` only; the spec's
/// "reload git log" step was deferred (B4) because there was no hermetic,
/// cross-language-testable seam for VCS reads. This trait IS that seam.
///
/// It mirrors how [`SandboxProvider`] abstracts filesystem/shell access:
/// define a trait, ship a real implementation ([`GitVcsProvider`]) and a
/// deterministic fixture double ([`FixtureVcsProvider`]), and inject the chosen
/// one at construction via [`HarnessBuilder::vcs_provider`]. The harness owns an
/// `Arc<dyn VcsProvider>` clone exactly as it does for every other component.
///
/// `Ralph` calls [`log`](Self::log) during its reload phase and injects the
/// output into the next window's seed as a clearly delimited
/// "Recent VCS history:" section. When NO provider is wired
/// (`vcs_provider == None`, the default) the git-log section is OMITTED and
/// Ralph behaves byte-for-byte like v1 — this is the B4→None decision.
pub trait VcsProvider: Send + Sync {
    /// Return the project's commit log, shaped by `args`. The returned string
    /// is the verbatim VCS output (e.g. `git log` stdout); the caller does not
    /// parse it, it is injected into the reloaded context block as-is.
    fn log<'a>(&'a self, args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>>;

    /// Return the working-tree status (e.g. `git status` stdout), verbatim.
    fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>>;
}

/// Parameters shaping a [`VcsProvider::log`] read. Each field maps to a
/// `git log` flag in [`GitVcsProvider`]:
///   - `max_entries` → `-n <N>` (cap the number of commits returned),
///   - `since_ref`   → `<ref>..` (only commits AFTER `<ref>`),
///   - `format`      → `--format=<fmt>` (custom pretty format).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VcsLogArgs {
    /// Maximum number of commits to return (`git log -n <max_entries>`).
    pub max_entries: usize,
    /// Only commits reachable after this ref (`git log <since_ref>..`). `None`
    /// returns the full history (subject to `max_entries`).
    pub since_ref: Option<String>,
    /// Custom `git log --format=<format>` string. `None` uses git's default
    /// formatting.
    pub format: Option<String>,
}

/// Error raised by a [`VcsProvider`]. Mirrors the per-component
/// `<Component>Error` convention (`thiserror`, `#[non_exhaustive]`).
#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VcsError {
    /// The underlying VCS command failed (non-zero exit), carrying the captured
    /// stderr.
    #[error("vcs command failed: {message}")]
    CommandFailed { message: String },
    /// The VCS command was blocked or could not be spawned by the sandbox.
    #[error("vcs command blocked by sandbox: {0:?}")]
    Sandbox(SandboxViolation),
}

/// Real [`VcsProvider`] that shells out to `git` THROUGH a [`SandboxProvider`]
/// (issue #58 v2). It wraps the sandbox and calls
/// [`SandboxProvider::execute_command`] — it never bypasses sandboxing to spawn
/// `git` directly. The command line is built from [`VcsLogArgs`] (see that
/// type for the flag mapping); [`status`](VcsProvider::status) runs
/// `git status`. All commands run in `workspace_root`.
pub struct GitVcsProvider {
    sandbox: Arc<dyn SandboxProvider>,
    workspace_root: std::path::PathBuf,
}

impl GitVcsProvider {
    /// Wrap `sandbox`, running `git` invocations in `workspace_root`.
    pub fn new(
        sandbox: Arc<dyn SandboxProvider>,
        workspace_root: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            sandbox,
            workspace_root: workspace_root.into(),
        }
    }

    /// Build the `git log` argument vector from `args` (visible for testing the
    /// flag mapping independently of process execution).
    fn log_args(args: &VcsLogArgs) -> Vec<String> {
        let mut out = vec![
            "log".to_string(),
            "-n".to_string(),
            args.max_entries.to_string(),
        ];
        if let Some(fmt) = &args.format {
            out.push(format!("--format={fmt}"));
        }
        if let Some(since) = &args.since_ref {
            out.push(format!("{since}.."));
        }
        out
    }
}

impl VcsProvider for GitVcsProvider {
    fn log<'a>(&'a self, args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move {
            let argv = Self::log_args(args);
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(out.stdout)
        })
    }

    fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move {
            let argv = vec!["status".to_string()];
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(out.stdout)
        })
    }
}

/// Output of a subprocess executed through the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    #[serde(default)]
    pub timed_out: bool,
    /// `true` if `stdout`/`stderr` were truncated by `handle_large_output`
    /// before being returned. Kept alongside `timed_out` so existing callers
    /// continue to compile.
    #[serde(default)]
    pub truncated: bool,
}

/// Head+tail-truncated output. `full_ref` is `Some` when the sandbox offloads
/// the original content to a backing file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncatedOutput {
    /// The (possibly truncated) head+tail content surfaced to the agent.
    pub content: String,
    /// `true` if the original content was truncated.
    #[serde(default)]
    pub truncated: bool,
    /// Backing file holding the full original content, if offloaded.
    #[serde(default)]
    pub full_ref: Option<FileRef>,
    /// Original size of the input content in bytes.
    #[serde(default)]
    pub original_size: u64,
}

/// Reference to a file holding offloaded tool output.
///
/// `path` is a string and `byte_len` is a `u64` to keep the wire format
/// portable across the four reference implementations — `PathBuf` serde is
/// platform-specific and `usize` width differs by target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRef {
    pub path: String,
    pub byte_len: u64,
}

/// Inputs the harness compaction loop (issue #46) needs to run one
/// compaction turn and verify its result.
///
/// The harness loop operates on the opaque [`SessionState`] above; the rich
/// compaction/verification API ([`crate::context::ContextManager`],
/// [`CompactionVerifier`]) operates on [`crate::context::SessionState`]. This
/// struct is the bridge: a [`ContextManager`] that supports compaction
/// projects everything the loop needs into one value, so the loop never has
/// to know which concrete state type its manager uses internally.
///
/// `context` is fed straight to `Agent::turn` to produce the summary;
/// `preserve_hints` and `verification_state` are passed to
/// [`CompactionVerifier::verify`]. On a verification failure the loop re-runs
/// the turn with [`ContextManager::inject_missing_items`] applied to `context`.
pub struct CompactionTurn {
    /// Context to feed `Agent::turn` to elicit the summary.
    pub context: Context,
    /// Preservation hints to hand the verifier.
    pub preserve_hints: CompactionPreserveHints,
    /// Verifier-facing session state (rich `context::SessionState`).
    pub verification_state: ContextSessionState,
    /// Messages about to be removed — used to stamp the compaction span.
    pub messages_removed: u32,
}

/// Issue #7 — ContextManager: assembles per-turn context.
///
/// Issue #46 adds the optional compaction-loop surface
/// ([`prepare_compaction_turn`](Self::prepare_compaction_turn),
/// [`inject_missing_items`](Self::inject_missing_items),
/// [`apply_compaction`](Self::apply_compaction)). All three have defaults so
/// managers that do not compact (the default `should_compact` returns `false`)
/// need not implement them.
pub trait ContextManager: Send + Sync {
    fn assemble<'a>(&'a self, session: &'a SessionState, task: &'a Task) -> BoxFut<'a, Context>;

    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()>;

    /// Append the assistant's turn (model output: text and/or the tool calls it
    /// requested) to the conversation so the next assemble() reflects what the
    /// agent already did. Without this the model loses track of its own actions.
    fn append_assistant_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        message: &'a Message,
    ) -> BoxFut<'a, ()> {
        let _ = (session, message);
        Box::pin(async {})
    }

    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()>;

    fn should_compact(&self, session: &SessionState) -> bool {
        let _ = session;
        false
    }

    /// Build the inputs for one compaction turn (issue #46). Returns `None`
    /// when there is nothing to compact (e.g. history shorter than the
    /// preserve window), in which case the harness skips compaction entirely.
    ///
    /// Default: `None` — managers that never compact need not override this.
    fn prepare_compaction_turn(&self, session: &SessionState) -> Option<CompactionTurn> {
        let _ = session;
        None
    }

    /// Mutate a compaction [`Context`] in place to request a revised summary
    /// on retry (issue #46). The harness calls this with the items the prior
    /// summary failed to preserve. Default: append the standard "missing
    /// these items … please revise" instruction as a user message.
    fn inject_missing_items(&self, context: &mut Context, missing: &[String]) {
        context.messages.push(Message {
            role: crate::model::Role::User,
            content: crate::model::Content::Text {
                text: format!(
                    "Your summary is missing these items: {}. Please revise.",
                    missing.join(", ")
                ),
            },
        });
    }

    /// Accept a verified (or accepted-anyway) summary into the session,
    /// replacing the compacted span (issue #46). Default: no-op — only
    /// compaction-capable managers implement it.
    fn apply_compaction(&self, session: &mut SessionState, summary: String) {
        let _ = (session, summary);
    }

    /// Current token budget used for this session, if the manager tracks one
    /// (issue #57 — Known Deviation #2 token-accounting fix). The harness reads
    /// this after [`apply_compaction`](Self::apply_compaction) to stamp the
    /// compaction span with the real post-compaction utilization. Default:
    /// `None` — managers that do not track tokens fall back to the pre-value.
    fn token_budget_used(&self, session: &SessionState) -> Option<u32> {
        let _ = session;
        None
    }
}

/// Issue #13 — TerminationPolicy.
pub trait TerminationPolicy: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        session: &'a SessionState,
        budget_used: &'a BudgetSnapshot,
    ) -> BoxFut<'a, TerminationDecision>;
}

/// Issue #11 — Middleware chain. Full shape (BeforeTool modification,
/// SurfaceToHuman payload) lives in #11; this stub covers what ReAct needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MiddlewareDecision {
    Continue,
    ContinueWithModification { calls: Vec<ToolCall> },
    Halt { reason: String },
    SurfaceToHuman { request: HumanRequest },
}

pub trait MiddlewareChain: Send + Sync {
    fn fire<'a>(
        &'a self,
        hook: HookPoint,
        session: &'a SessionState,
    ) -> BoxFut<'a, MiddlewareDecision>;
}

// Issue #12 — `ObservabilityProvider` is no longer a no-op stub here. The
// canonical trait lives in [`crate::observability`]; the harness loop emits
// real spans through it (see `run_react`). Re-exported below for ergonomics.
pub use crate::observability::ObservabilityProvider;

// ============================================================================
// Human-in-the-loop
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HumanRequest {
    ToolApproval {
        calls: Vec<ToolCall>,
        risk_level: RiskLevel,
    },
    Clarification {
        question: String,
        /// Optional fixed-choice options offered to the human (issue #81, Q4b).
        /// `None` for a free-form clarification. `#[serde(default)]` keeps older
        /// `Clarification` blobs (no `options` field) deserializing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        options: Option<Vec<String>>,
    },
    Review {
        content: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HumanResponse {
    Allow,
    AllowWithModification { calls: Vec<ToolCall> },
    Deny { reason: String },
    Halt,
    Answer { text: String },
    ApproveWithFeedback { feedback: String },
    Reject { reason: String },
}

// ============================================================================
// PausedState / ChildPausedState
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PausedState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub session_state: SessionState,
    pub pending_tool_calls: Vec<ToolCall>,
    pub approved_results: Vec<ToolResult>,
    /// `None` for an escalation-derived state (issue #80) — an escalation has
    /// no human request. `WaitingForHuman` construction paths always set
    /// `Some(..)`. `#[serde(default)]` keeps old `WaitingForHuman` blobs
    /// deserializing (field present) while escalation blobs omit it.
    #[serde(default)]
    pub human_request: Option<HumanRequest>,
    pub task: Task,
    pub budget_used: BudgetSnapshot,
    #[serde(default)]
    pub child_state: Option<ChildPausedState>,
}

/// Child paused state. **Deliberately has no `child_state` field** — the type
/// system enforces a one-level subagent depth (spec rule).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildPausedState {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub session_state: SessionState,
    pub pending_tool_calls: Vec<ToolCall>,
    pub approved_results: Vec<ToolResult>,
    /// `None` for an escalation-derived state (issue #80). `WaitingForHuman`
    /// construction paths always set `Some(..)`.
    #[serde(default)]
    pub human_request: Option<HumanRequest>,
    pub task: Task,
    pub budget_used: BudgetSnapshot,
    pub parent_tool_call_id: String,
}

// ============================================================================
// Halt reasons / RunResult
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HaltReason {
    BudgetExceeded {
        limit_type: BudgetLimitType,
    },
    TerminationPolicyHalt {
        reason: String,
    },
    MiddlewareHalt {
        hook: HookPoint,
        reason: String,
    },
    AgentError {
        error: AgentError,
    },
    /// A [`ContextError`] surfaced by the [`crate::context::ContextManager`]
    /// during assembly halts the run (e.g. a cache-hash mismatch — both Block 1
    /// `Static` and, as of #32, Block 2 `PerSession` halt mid-session). This is
    /// the routing type; mirrors [`HaltReason::AgentError`]. The live
    /// [`StandardHarness`] loop does not yet trigger it because its placeholder
    /// `ContextManager::assemble` is infallible pending the #7 migration.
    ContextError {
        error: ContextError,
    },
    SandboxViolation {
        violation: SandboxViolation,
    },
    UnrecoverableToolError {
        tool: String,
        error: String,
    },
    HumanHalted,
    StagnationLimitReached {
        iterations: u32,
        best_metric: f64,
    },
    /// Returned by [`StandardHarness`] for [`LoopStrategy`] variants whose
    /// concrete trait dependencies (e.g. `CompletionCheck`, `Verifier`,
    /// `MetricEvaluator`) ship with later component issues. Marker only; will
    /// be removed once those issues land.
    StrategyNotYetImplemented {
        strategy: String,
    },
    /// Returned by [`StandardHarness`] for the `PlanExecute` strategy (issue
    /// #59) when the accepted plan parsed into an EMPTY task list (`tasks: []`).
    /// Per Q3, an empty plan is a failure — the run does NOT silently succeed.
    EmptyPlan,
    /// Returned by [`StandardHarness`] for the `PlanExecute` strategy (issue
    /// #59) when an execute step's bounded ReAct sub-loop errored or the agent
    /// returned a blocked/failed outcome (Q5). A plan is a dependency chain by
    /// assumption, so the whole run aborts at the failing step — execution does
    /// NOT continue to the next task. Carries the failing step's positional
    /// index, its instruction, and a human-readable reason derived from the
    /// underlying [`HaltReason`].
    StepFailed {
        task_index: usize,
        task: String,
        reason: String,
    },
    /// The PlanExecute plan phase (issue #70) failed before producing an
    /// artifact: the planner's response was unparseable, the planner requested
    /// a tool call in the one-shot turn, or the artifact could not be
    /// serialized for storage. Carries the underlying
    /// [`PlanPhaseError`](crate::plan::PlanPhaseError).
    PlanPhaseFailed {
        error: crate::plan::PlanPhaseError,
    },
    /// Returned by [`StandardHarness`] for the `SelfVerifying` strategy (issue
    /// #61, D4) when the build↔evaluate loop ran out of the verifier's
    /// `max_iterations` round-trips without an explicit `Passed` verdict. A
    /// RUNTIME limit — the work was attempted in good faith but never verified;
    /// a caller might retry with a different task decomposition. Carries the
    /// number of round-trips run and the last failure reason the verifier gave.
    /// PEER to [`SelfVerifyMisconfigured`](Self::SelfVerifyMisconfigured) (NOT a
    /// sub-case of it).
    SelfVerifyExhausted {
        iterations: u32,
        last_reason: String,
    },
    /// Returned by [`StandardHarness`] for the `SelfVerifying` strategy (issue
    /// #61, D4) when the strategy cannot run because it is misconfigured — e.g.
    /// `config.verifier` is `None`. Likely a BUILD-TIME bug in the caller's
    /// wiring. Surfaced as a typed halt, NOT a panic. PEER to
    /// [`SelfVerifyExhausted`](Self::SelfVerifyExhausted) (NOT a sub-case of it).
    SelfVerifyMisconfigured {
        reason: String,
    },
    /// Returned by [`StandardHarness`] for the `Ralph` strategy (issue #58, B3)
    /// when the multi-context-window continuation loop reached its `max_resets`
    /// cap with tasks still incomplete (the Ralph analogue of
    /// [`SelfVerifyExhausted`](Self::SelfVerifyExhausted)). A RUNTIME limit — the
    /// work was attempted across `iterations` context windows but the
    /// filesystem-backed completion check (the registered `Stop` hook reading
    /// `.spore/progress.json`) never reported done. Carries the number of
    /// context-window resets performed and the last incompletion reason.
    RalphCompletionUnmet {
        iterations: u32,
        last_reason: String,
    },
    /// Returned by [`StandardHarness`] for the `HillClimbing` strategy (issue
    /// #60) when the strategy cannot run because it is misconfigured — i.e.
    /// `config.metric_evaluator` is `None`. Likely a BUILD-TIME bug in the
    /// caller's wiring. Surfaced as a typed halt, NOT a panic. PEER to
    /// [`SelfVerifyMisconfigured`](Self::SelfVerifyMisconfigured).
    HillClimbingMisconfigured {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunResult {
    Success {
        output: String,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
    },
    Failure {
        reason: HaltReason,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
    },
    /// Boxed because `PausedState` is significantly larger than the other
    /// variants; keeps `RunResult` cheap to clone on the happy path.
    WaitingForHuman {
        state: Box<PausedState>,
        request: HumanRequest,
    },
    /// Harness terminated cleanly due to a tool escalation signal (issue #80).
    /// The caller handles the `signal` and decides whether to resume the
    /// original harness, instantiate a new one, or present UI to the user. The
    /// `state` preserves the full [`PausedState`] (with `human_request: None`)
    /// so `harness.resume(state, ..)` continues the original session; the
    /// signal is NOT stored in the state, so it is naturally discarded on
    /// resume — the harness does not re-act on it.
    Escalate {
        signal: HarnessSignal,
        state: Box<PausedState>,
        session_id: SessionId,
        usage: AggregateUsage,
        turns: u32,
    },
}

/// Internal result of a successful PlanExecute plan phase (issue #70). Carries
/// the produced artifact plus the run accounting so the caller can build the
/// terminal `RunResult`. Not part of the public surface — the artifact itself
/// is observable via `SessionState.extras["plan_execute"]`.
#[derive(Debug)]
struct PlanPhaseOutcome {
    artifact: crate::plan::PlanArtifact,
    usage: AggregateUsage,
    turns: u32,
}

// ============================================================================
// Ralph loop strategy support types (issue #58)
// ============================================================================

/// Deserialized `.spore/progress.json` for the Ralph loop strategy (issue #58,
/// B1/B2). The handoff artifact between context windows: `complete` is the
/// primary completion signal, `remaining` lists outstanding work so an
/// incompletion reason can name what is left. Tolerant by default (`#[serde]`
/// defaults) so a partially-written file deserializes to "incomplete" rather
/// than erroring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RalphProgress {
    #[serde(default)]
    complete: bool,
    #[serde(default)]
    remaining: Vec<String>,
}

/// One `.spore/feature_list.json` entry, mirroring
/// [`FeatureListCheck`](crate::termination::FeatureListCheck)'s schema so the
/// two sources agree (issue #58, B2).
#[derive(Debug, Clone, Deserialize)]
struct RalphFeatureEntry {
    name: String,
    passes: bool,
}

/// The `Stop` lifecycle hook (issue #69) the Ralph loop strategy registers at
/// construction (issue #58, B1). Drives multi-context-window continuation off
/// `.spore/progress.json`: while tasks remain incomplete it returns
/// [`HookDecision::Block`](crate::hooks::HookDecision::Block) (the reason
/// describes what is left) so the harness loops into a new context window; when
/// complete it returns [`Continue`](crate::hooks::HookDecision::Continue) so the
/// loop terminates with success.
///
/// Registration is harmless for non-Ralph strategies: when `.spore/progress.json`
/// is ABSENT the hook returns `Continue` and does not interfere with ReAct /
/// PlanExecute / SelfVerifying runs. It only blocks when a progress file is
/// PRESENT and reports incomplete tasks — the Ralph contract.
struct RalphStopHook {
    workspace_root: std::path::PathBuf,
}

impl crate::hooks::Hook for RalphStopHook {
    fn handle<'a>(
        &'a self,
        ctx: &'a mut crate::hooks::HookContext<'a>,
    ) -> BoxFut<'a, Result<crate::hooks::HookDecision, crate::hooks::HookError>> {
        Box::pin(async move {
            // Only act on `Stop`; any other event is a no-op `Continue`.
            if !matches!(ctx, crate::hooks::HookContext::Stop { .. }) {
                return Ok(crate::hooks::HookDecision::Continue);
            }
            // Absent progress file ⇒ do not interfere with non-Ralph runs.
            let progress_path = self.workspace_root.join(".spore/progress.json");
            if !progress_path.exists() {
                return Ok(crate::hooks::HookDecision::Continue);
            }
            match StandardHarness::ralph_completion_status(&self.workspace_root) {
                None => Ok(crate::hooks::HookDecision::Continue),
                Some(reason) => Ok(crate::hooks::HookDecision::Block { reason }),
            }
        })
    }
    fn events(&self) -> Vec<crate::hooks::HookEvent> {
        vec![crate::hooks::HookEvent::Stop]
    }
    fn name(&self) -> String {
        "ralph-stop".into()
    }
}

#[derive(Debug, Clone, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
#[non_exhaustive]
pub enum HarnessError {
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
}

// ============================================================================
// HarnessRunOptions
// ============================================================================

pub struct HarnessRunOptions {
    pub task: Task,
    pub on_stream: Option<StreamSink>,
    /// Optional starting session state (e.g. resumed conversation history).
    pub session_state: Option<SessionState>,
}

impl HarnessRunOptions {
    pub fn new(task: Task) -> Self {
        Self {
            task,
            on_stream: None,
            session_state: None,
        }
    }

    /// Carry an existing session state into the run (e.g. a resumed
    /// conversation history for multi-turn scenarios — issue #57 S2).
    pub fn with_session_state(mut self, session_state: SessionState) -> Self {
        self.session_state = Some(session_state);
        self
    }

    /// Attach a streaming sink for `StreamEvent`s emitted during the run.
    pub fn with_stream(mut self, on_stream: StreamSink) -> Self {
        self.on_stream = Some(on_stream);
        self
    }
}

// ============================================================================
// The trait
// ============================================================================

pub trait Harness: Send + Sync {
    fn run<'a>(&'a self, options: HarnessRunOptions) -> BoxFut<'a, RunResult>;

    fn resume<'a>(
        &'a self,
        state: PausedState,
        response: HumanResponse,
        on_stream: Option<StreamSink>,
    ) -> BoxFut<'a, RunResult>;
}

// ============================================================================
// StandardHarness — the canonical implementation
// ============================================================================

/// Components injected at construction. Mirrors `HarnessConfig` in the spec.
/// Components not yet covered by their own issue are optional here so the
/// harness can be exercised before #4–#13 land.
///
/// Every component, including the agent, is held erased behind `Arc<dyn _>`.
/// The `Agent` trait (issue #2) is dyn-compatible via the hand-rolled `BoxFut`
/// pattern, so no type parameter is needed (issue #45).
pub struct HarnessConfig {
    pub agent: Arc<dyn Agent>,
    pub tool_registry: Arc<dyn ToolRegistry>,
    pub sandbox: Arc<dyn SandboxProvider>,
    pub context_manager: Arc<dyn ContextManager>,
    pub termination_policy: Arc<dyn TerminationPolicy>,
    pub middleware: Option<Arc<dyn MiddlewareChain>>,
    pub observability: Option<Arc<dyn ObservabilityProvider>>,
    /// Post-compaction verifier (issue #29/#46). The harness runs it after
    /// each compaction turn and retries up to `max_compaction_attempts` before
    /// accepting a failing summary. Defaults to [`KeyTermVerifier`].
    pub compaction_verifier: Arc<dyn CompactionVerifier>,
    /// Maximum compaction-summary attempts before accepting a failing summary
    /// anyway (issue #46). Defaults to `2` (mirrors `CompactionConfig`).
    pub max_compaction_attempts: u32,
    /// Token → USD pricing used to stamp `cost_usd` on emitted [`TurnSpan`]s.
    /// Defaults to [`PricingTable::DEFAULT`] (zero cost) when unset.
    pub pricing: PricingTable,
    /// LLM-native content capture guard + truncation limit (issue #64).
    /// Defaults to [`ContentCaptureConfig::default`] (OFF). When disabled the
    /// harness populates no `gen_ai.*` content and the JSONL stays
    /// byte-identical to the pre-#64 metrics-only output.
    pub content_capture: ContentCaptureConfig,
    /// Optional deterministic tool-call repair provider. When set, recoverable
    /// tool-dispatch errors trigger an argument-coercion + re-dispatch attempt
    /// before falling back to error feedback. `None` (the default) preserves
    /// today's behaviour byte-for-byte.
    pub tool_call_repair: Option<Arc<dyn ToolCallRepair>>,
    /// Maximum number of repair-and-re-dispatch attempts per tool call.
    /// Defaults to `1`. Ignored when `tool_call_repair` is `None`.
    pub max_repair_attempts: u32,
    /// Maximum number of consecutive Stop-hook blocks within a single `run()`
    /// before the loop terminates anyway (issue #69, Decision 4/5). The
    /// counter is PER-RUN — it resets on every `run()` call, so a resumed
    /// session starts fresh. Defaults to `8` (matching Claude Code).
    pub max_stop_blocks: u32,
    /// Lifecycle hook chain (issue #69). When set, the harness fires the wired
    /// lifecycle events (`PreTurn`, `Stop`, `OnError`, …) through it.
    /// `None` (the default) preserves today's behaviour byte-for-byte.
    pub hooks: Option<Arc<dyn crate::hooks::HookChain>>,
    /// Optional alternate agent used for the PlanExecute plan phase (issue #70,
    /// Q1). When the loop strategy is `PlanExecute` and this is `Some`, the
    /// one-shot plan turn runs on this agent; otherwise it runs on
    /// [`agent`](Self::agent). `plan_model` on the strategy is descriptive
    /// metadata only — there is no `ModelConfig`→agent factory.
    pub planner_agent: Option<Arc<dyn Agent>>,
    /// The verification oracle for the `SelfVerifying` loop strategy (issue
    /// #61). When the loop strategy is `SelfVerifying` and this is `Some`, the
    /// strategy consults it after each build↔evaluate round-trip and its
    /// [`max_iterations`](crate::verifier::Verifier::max_iterations) governs the
    /// round-trip cap (D3). When `None`, `SelfVerifying` is MISCONFIGURED: the
    /// run returns `Failure { SelfVerifyMisconfigured }` (D4) — it does NOT
    /// panic. Irrelevant for every other loop strategy. Defaults to `None`.
    pub verifier: Option<Arc<dyn crate::verifier::Verifier>>,
    /// Optional alternate agent used for the `SelfVerifying` evaluate phase
    /// (issue #61, D2). Follows the EXACT same defaulting contract as
    /// [`planner_agent`](Self::planner_agent): when `None`, the evaluate phase
    /// runs on [`agent`](Self::agent) (the canonical fallback is
    /// `config.evaluator_agent.as_ref().unwrap_or(&config.agent)`). The
    /// evaluate phase always runs on a FRESH [`SessionId::generate`] session and
    /// a READ-ONLY sandbox derived internally (no `evaluator_sandbox` field) so
    /// the evaluator cannot be biased by the build session nor mutate the
    /// workspace it is reviewing (the Default-FAIL contract). Defaults to `None`.
    pub evaluator_agent: Option<Arc<dyn Agent>>,
    /// Pluggable per-domain persistence layer (issue #73). Defaults to an
    /// all-no-op [`StorageProvider`] so existing callers/tests compile and
    /// behave unchanged. v1 is expose-only — the run/resume loop is NOT
    /// modified to read/write sessions internally.
    pub storage: Arc<crate::storage::StorageProvider>,
    /// Source of conditional prompt chunks (issue #79). Defaults to an empty
    /// [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider).
    /// The harness loads chunks from it at construction and feeds them through
    /// a [`ContextSourcesBuilder`](crate::prompt_assembly::ContextSourcesBuilder).
    pub chunk_provider: Arc<dyn crate::prompt_assembly::ChunkProvider>,
    /// Outer-loop cap for the `Ralph` loop strategy (issue #58, B3): the maximum
    /// number of context-window RESETS the continuation loop performs before
    /// halting with [`HaltReason::RalphCompletionUnmet`] when tasks are still
    /// incomplete. Independent from `max_turns` (the per-context-window ReAct
    /// turn budget) and from `max_stop_blocks` (the per-sub-loop Stop-block cap).
    /// Defaults to `3` (matching the `SelfVerifying` verifier `max_iterations`
    /// precedent). Irrelevant for every other loop strategy.
    pub max_resets: u32,
    /// Optional VCS read seam for the `Ralph` loop strategy (issue #58 v2). When
    /// `Some`, Ralph's per-window reload phase ALSO calls
    /// [`VcsProvider::log`] and injects the output into the fresh context window
    /// as a delimited "Recent VCS history:" section — alongside the reloaded
    /// `.spore/progress.json` + `.spore/feature_list.json` content. When `None`
    /// (the default) the git-log section is OMITTED and Ralph behaves
    /// byte-for-byte like v1 (the B4→None decision). Irrelevant for every other
    /// loop strategy.
    pub vcs_provider: Option<Arc<dyn VcsProvider>>,
    /// The scoring oracle for the `HillClimbing` loop strategy (issue #60).
    /// When the loop strategy is `HillClimbing` and this is `Some`, the harness
    /// evaluates the workspace metric at iteration 0 (the pure baseline, no
    /// agent turn) and after every subsequent agent turn, feeding each result
    /// through [`should_keep`](crate::metric::should_keep). When `None`,
    /// `HillClimbing` is MISCONFIGURED: the run returns
    /// `Failure { HillClimbingMisconfigured }` — it does NOT panic. Irrelevant
    /// for every other loop strategy. Defaults to `None`.
    pub metric_evaluator: Option<Arc<dyn crate::metric::MetricEvaluator>>,
}

impl Clone for HarnessConfig {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            tool_registry: self.tool_registry.clone(),
            sandbox: self.sandbox.clone(),
            context_manager: self.context_manager.clone(),
            termination_policy: self.termination_policy.clone(),
            middleware: self.middleware.clone(),
            observability: self.observability.clone(),
            compaction_verifier: self.compaction_verifier.clone(),
            max_compaction_attempts: self.max_compaction_attempts,
            pricing: self.pricing.clone(),
            content_capture: self.content_capture,
            tool_call_repair: self.tool_call_repair.clone(),
            max_repair_attempts: self.max_repair_attempts,
            max_stop_blocks: self.max_stop_blocks,
            hooks: self.hooks.clone(),
            planner_agent: self.planner_agent.clone(),
            verifier: self.verifier.clone(),
            evaluator_agent: self.evaluator_agent.clone(),
            storage: self.storage.clone(),
            chunk_provider: self.chunk_provider.clone(),
            max_resets: self.max_resets,
            vcs_provider: self.vcs_provider.clone(),
            metric_evaluator: self.metric_evaluator.clone(),
        }
    }
}

/// Fluent assembler for a [`HarnessConfig`] / [`StandardHarness`].
///
/// The harness follows strict inversion of control: every component is
/// injected. `HarnessBuilder` is the canonical assembly point — it takes the
/// required components up front and exposes fluent setters for the optional
/// ones (middleware, observability, pricing). It is the intended caller that
/// wires the [`ObservabilityProvider`] into the loop, including the durable
/// outbox via [`with_observability_outbox`](Self::with_observability_outbox).
///
/// ```no_run
/// # use std::sync::Arc;
/// # use spore_core::{HarnessBuilder, PricingTable};
/// # fn demo(
/// #     agent: Arc<dyn spore_core::Agent>,
/// #     tools: Arc<dyn spore_core::HarnessToolRegistry>,
/// #     sandbox: Arc<dyn spore_core::SandboxProvider>,
/// #     ctx: Arc<dyn spore_core::HarnessContextManager>,
/// #     term: Arc<dyn spore_core::TerminationPolicy>,
/// # ) {
/// let harness = HarnessBuilder::new(agent, tools, sandbox, ctx, term)
///     .with_observability_outbox(".spore")
///     .pricing(PricingTable::DEFAULT)
///     .build();
/// # }
/// ```
pub struct HarnessBuilder {
    agent: Arc<dyn Agent>,
    tool_registry: Arc<dyn ToolRegistry>,
    sandbox: Arc<dyn SandboxProvider>,
    context_manager: Arc<dyn ContextManager>,
    termination_policy: Arc<dyn TerminationPolicy>,
    middleware: Option<Arc<dyn MiddlewareChain>>,
    observability: Option<Arc<dyn ObservabilityProvider>>,
    compaction_verifier: Arc<dyn CompactionVerifier>,
    max_compaction_attempts: u32,
    pricing: PricingTable,
    content_capture: ContentCaptureConfig,
    tool_call_repair: Option<Arc<dyn ToolCallRepair>>,
    max_repair_attempts: u32,
    max_stop_blocks: u32,
    hooks: Option<Arc<dyn crate::hooks::HookChain>>,
    planner_agent: Option<Arc<dyn Agent>>,
    verifier: Option<Arc<dyn crate::verifier::Verifier>>,
    evaluator_agent: Option<Arc<dyn Agent>>,
    storage: Option<Arc<crate::storage::StorageProvider>>,
    /// Conditional prompt-chunk source (issue #79). `None` resolves to an empty
    /// [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider)
    /// at build time.
    chunk_provider: Option<Arc<dyn crate::prompt_assembly::ChunkProvider>>,
    /// Outer-loop reset cap for the `Ralph` loop strategy (issue #58, B3).
    /// Defaults to `3`.
    max_resets: u32,
    /// Optional VCS read seam for the `Ralph` loop strategy (issue #58 v2).
    /// `None` (the default) omits the git-log reload section (v1 behavior).
    vcs_provider: Option<Arc<dyn VcsProvider>>,
    /// Scoring oracle for the `HillClimbing` loop strategy (issue #60). `None`
    /// (the default) makes a `HillClimbing` run halt with
    /// `HillClimbingMisconfigured`.
    metric_evaluator: Option<Arc<dyn crate::metric::MetricEvaluator>>,
    /// Standard catalogue tools accumulated via [`HarnessBuilder::tool`] /
    /// [`HarnessBuilder::tools`] (issue #81). They are drained into a populated
    /// [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry) by
    /// [`HarnessBuilder::build_tool_registry`], applying last-wins upsert.
    standard_tools: Vec<crate::tools::StandardTool>,
}

impl HarnessBuilder {
    /// Start a builder from the five required components. Optional components
    /// default to `None`/[`PricingTable::DEFAULT`] until set.
    pub fn new(
        agent: Arc<dyn Agent>,
        tool_registry: Arc<dyn ToolRegistry>,
        sandbox: Arc<dyn SandboxProvider>,
        context_manager: Arc<dyn ContextManager>,
        termination_policy: Arc<dyn TerminationPolicy>,
    ) -> Self {
        Self {
            agent,
            tool_registry,
            sandbox,
            context_manager,
            termination_policy,
            middleware: None,
            observability: None,
            compaction_verifier: Arc::new(KeyTermVerifier),
            max_compaction_attempts: 2,
            pricing: PricingTable::DEFAULT,
            content_capture: ContentCaptureConfig::default(),
            tool_call_repair: None,
            max_repair_attempts: 1,
            max_stop_blocks: 8,
            hooks: None,
            planner_agent: None,
            verifier: None,
            evaluator_agent: None,
            storage: None,
            chunk_provider: None,
            max_resets: 3,
            vcs_provider: None,
            metric_evaluator: None,
            standard_tools: Vec::new(),
        }
    }

    /// Add a single [`StandardTool`](crate::tools::StandardTool) to the
    /// catalogue accumulated for this harness (issue #81, Q1/Q2). The bundled
    /// implementation + schema are destructured when the registry is built via
    /// [`build_tool_registry`](Self::build_tool_registry). Registration applies
    /// LAST-WINS upsert: a later `.tool()` with the same name overrides an
    /// earlier one (e.g. a custom tool registered after a preset).
    pub fn tool(mut self, tool: crate::tools::StandardTool) -> Self {
        self.standard_tools.push(tool);
        self
    }

    /// Add many [`StandardTool`](crate::tools::StandardTool)s at once (e.g. a
    /// preset like [`StandardTools::coding_set`](crate::tools::StandardTools::coding_set)).
    /// Order is preserved, so last-wins upsert still applies across the batch.
    pub fn tools(mut self, tools: impl IntoIterator<Item = crate::tools::StandardTool>) -> Self {
        self.standard_tools.extend(tools);
        self
    }

    /// Drain the accumulated catalogue tools into a populated
    /// [`StandardToolRegistry`](crate::tool_registry::StandardToolRegistry),
    /// registering each with last-wins upsert (issue #81, Q1/Q2). Consumes the
    /// accumulated set. The harness-loop [`tool_registry`](Self) seam is
    /// separate (it is the bridged dispatch trait); callers wire the returned
    /// registry into a [`RealToolRegistry`](crate::scenarios::RealToolRegistry)
    /// per run. Returns an empty registry if no catalogue tools were added.
    pub fn drain_tools_into_registry(&mut self) -> Arc<crate::tool_registry::StandardToolRegistry> {
        use crate::tool_registry::ToolRegistry as _;
        let reg = crate::tool_registry::StandardToolRegistry::new();
        for t in self.standard_tools.drain(..) {
            // Upsert: a duplicate name overwrites (issue #81, Q1). Schema/name
            // validation errors are ignored here since catalogue schemas are
            // well-formed by construction.
            let _ = reg.register(t.implementation, t.schema);
        }
        Arc::new(reg)
    }

    /// Inject a [`StorageProvider`](crate::storage::StorageProvider) (issue #73).
    /// Mirrors [`observability`](Self::observability) — independent injection
    /// point. Defaults to an all-no-op provider so existing callers/tests
    /// compile and behave unchanged.
    pub fn storage(mut self, storage: Arc<crate::storage::StorageProvider>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Set the conditional prompt-chunk provider (issue #79). Defaults to an
    /// empty [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider).
    pub fn chunk_provider(
        mut self,
        provider: Arc<dyn crate::prompt_assembly::ChunkProvider>,
    ) -> Self {
        self.chunk_provider = Some(provider);
        self
    }

    /// Convenience: register chunks inline without constructing a provider
    /// (issue #79). Resolves to an
    /// [`InMemoryChunkProvider`](crate::prompt_assembly::InMemoryChunkProvider).
    pub fn chunks(mut self, chunks: Vec<crate::prompt_assembly::PromptChunk>) -> Self {
        self.chunk_provider = Some(Arc::new(
            crate::prompt_assembly::InMemoryChunkProvider::new(chunks),
        ));
        self
    }

    /// Inject an alternate agent for the PlanExecute plan phase (issue #70,
    /// Q1). When set and the loop strategy is `PlanExecute`, the one-shot plan
    /// turn runs on this agent; otherwise the plan turn runs on the default
    /// agent. Defaults to `None`.
    pub fn planner_agent(mut self, planner_agent: Arc<dyn Agent>) -> Self {
        self.planner_agent = Some(planner_agent);
        self
    }

    /// Inject the verification oracle for the `SelfVerifying` loop strategy
    /// (issue #61). Required for that strategy: without it a `SelfVerifying`
    /// run halts with `SelfVerifyMisconfigured` (D4). Its
    /// [`max_iterations`](crate::verifier::Verifier::max_iterations) caps the
    /// build↔evaluate round-trips (D3). Defaults to `None`.
    pub fn verifier(mut self, verifier: Arc<dyn crate::verifier::Verifier>) -> Self {
        self.verifier = Some(verifier);
        self
    }

    /// Inject an alternate agent for the `SelfVerifying` evaluate phase (issue
    /// #61, D2). Mirrors [`planner_agent`](Self::planner_agent): when set and
    /// the loop strategy is `SelfVerifying`, the evaluate phase runs on this
    /// agent; otherwise it runs on the default agent. Defaults to `None`.
    pub fn evaluator_agent(mut self, evaluator_agent: Arc<dyn Agent>) -> Self {
        self.evaluator_agent = Some(evaluator_agent);
        self
    }

    /// Inject a lifecycle hook chain (issue #69).
    pub fn hooks(mut self, hooks: Arc<dyn crate::hooks::HookChain>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set the per-run cap on consecutive Stop-hook blocks (issue #69).
    /// Defaults to `8`.
    pub fn max_stop_blocks(mut self, max: u32) -> Self {
        self.max_stop_blocks = max;
        self
    }

    /// Set the outer-loop context-window reset cap for the `Ralph` loop strategy
    /// (issue #58, B3). Defaults to `3`.
    pub fn max_resets(mut self, max: u32) -> Self {
        self.max_resets = max;
        self
    }

    /// Inject a [`VcsProvider`] for the `Ralph` loop strategy (issue #58 v2).
    /// When set, Ralph's per-window reload phase also calls
    /// [`VcsProvider::log`] and injects a delimited "Recent VCS history:"
    /// section into the fresh context window. Defaults to `None`, which omits
    /// the git-log section and preserves v1 Ralph behavior byte-for-byte (the
    /// B4→None decision).
    pub fn vcs_provider(mut self, provider: Arc<dyn VcsProvider>) -> Self {
        self.vcs_provider = Some(provider);
        self
    }

    /// Inject the scoring oracle for the `HillClimbing` loop strategy (issue
    /// #60). Required for that strategy: without it a `HillClimbing` run halts
    /// with `HillClimbingMisconfigured` (no panic). The evaluator is called once
    /// at the iteration-0 baseline (no agent turn) and after every subsequent
    /// agent turn; each result is routed through
    /// [`should_keep`](crate::metric::should_keep). Defaults to `None`.
    pub fn metric_evaluator(mut self, evaluator: Arc<dyn crate::metric::MetricEvaluator>) -> Self {
        self.metric_evaluator = Some(evaluator);
        self
    }

    /// Inject a deterministic tool-call repair provider (e.g.
    /// [`StandardToolCallRepair`](crate::tool_call_repair::StandardToolCallRepair)).
    /// When set, recoverable tool-dispatch errors trigger an argument-coercion
    /// and re-dispatch attempt before falling back to enriched error feedback.
    /// Defaults to `None` (behaviour byte-identical to today).
    pub fn tool_call_repair(mut self, repair: Arc<dyn ToolCallRepair>) -> Self {
        self.tool_call_repair = Some(repair);
        self
    }

    /// Set the maximum number of repair-and-re-dispatch attempts per tool call.
    /// Defaults to `1`. Ignored when no repair provider is set.
    pub fn max_repair_attempts(mut self, attempts: u32) -> Self {
        self.max_repair_attempts = attempts;
        self
    }

    /// Inject a post-compaction verifier (issue #46). Defaults to
    /// [`KeyTermVerifier`].
    pub fn compaction_verifier(mut self, verifier: Arc<dyn CompactionVerifier>) -> Self {
        self.compaction_verifier = verifier;
        self
    }

    /// Set the maximum number of compaction-summary attempts before accepting
    /// a failing summary anyway (issue #46). Defaults to `2`.
    pub fn max_compaction_attempts(mut self, attempts: u32) -> Self {
        self.max_compaction_attempts = attempts;
        self
    }

    /// Inject a middleware chain.
    pub fn middleware(mut self, middleware: Arc<dyn MiddlewareChain>) -> Self {
        self.middleware = Some(middleware);
        self
    }

    /// Inject an observability provider. The harness loop emits real spans
    /// through it (turn spans, tool-call spans) and flushes on terminal outcomes.
    pub fn observability(mut self, observability: Arc<dyn ObservabilityProvider>) -> Self {
        self.observability = Some(observability);
        self
    }

    /// Convenience: construct and inject a durable-outbox observability provider
    /// rooted at `root` (typically the `.spore` directory). Honors the
    /// `SPORE_OTLP_ENDPOINT` env var for OTLP forwarding (issue #33).
    pub fn with_observability_outbox(self, root: impl Into<std::path::PathBuf>) -> Self {
        let provider = Arc::new(OutboxObservabilityProvider::new(OutboxConfig::new(root)));
        self.observability(provider)
    }

    /// Set the token→USD pricing table used to stamp `cost_usd` on turn spans.
    pub fn pricing(mut self, pricing: PricingTable) -> Self {
        self.pricing = pricing;
        self
    }

    /// Set the LLM-native content-capture config (issue #64). Defaults to OFF.
    /// Use [`ContentCaptureConfig::from_env`] to honor `SPORE_TRACE_CONTENT` /
    /// `SPORE_TRACE_CONTENT_MAX_LEN`.
    pub fn content_capture(mut self, content_capture: ContentCaptureConfig) -> Self {
        self.content_capture = content_capture;
        self
    }

    /// Assemble the [`HarnessConfig`] without wrapping it in a harness.
    pub fn build_config(self) -> HarnessConfig {
        HarnessConfig {
            agent: self.agent,
            tool_registry: self.tool_registry,
            sandbox: self.sandbox,
            context_manager: self.context_manager,
            termination_policy: self.termination_policy,
            middleware: self.middleware,
            observability: self.observability,
            compaction_verifier: self.compaction_verifier,
            max_compaction_attempts: self.max_compaction_attempts,
            pricing: self.pricing,
            content_capture: self.content_capture,
            tool_call_repair: self.tool_call_repair,
            max_repair_attempts: self.max_repair_attempts,
            max_stop_blocks: self.max_stop_blocks,
            hooks: self.hooks,
            planner_agent: self.planner_agent,
            verifier: self.verifier,
            evaluator_agent: self.evaluator_agent,
            storage: self
                .storage
                .unwrap_or_else(|| Arc::new(crate::storage::StorageProvider::no_op())),
            chunk_provider: self.chunk_provider.unwrap_or_else(|| {
                Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty())
            }),
            max_resets: self.max_resets,
            vcs_provider: self.vcs_provider,
            metric_evaluator: self.metric_evaluator,
        }
    }

    /// Assemble a ready-to-run [`StandardHarness`].
    pub fn build(self) -> StandardHarness {
        StandardHarness::new(self.build_config())
    }
}

pub struct StandardHarness {
    config: HarnessConfig,
}

impl StandardHarness {
    pub fn new(config: HarnessConfig) -> Self {
        // Ralph completion mechanism (issue #58, B1): register a `Stop` hook
        // that drives multi-context-window continuation off `.spore/progress.json`.
        // Registration is harmless for non-Ralph runs — the hook only BLOCKS when
        // a progress file is PRESENT and reports incomplete tasks; when the file
        // is absent (the common case for ReAct / other strategies) it returns
        // `Continue`, so existing strategies are unaffected byte-for-byte.
        let workspace_root = config.sandbox.workspace_root().to_path_buf();
        let chain: Arc<dyn crate::hooks::HookChain> = match config.hooks.clone() {
            Some(c) => c,
            None => Arc::new(crate::hooks::StandardHookChain::new()),
        };
        // Best-effort: a duplicate/invalid registration must never panic the
        // constructor. The hook subscribes only to the can-block `Stop` event.
        let _ = chain.register(Arc::new(RalphStopHook { workspace_root }));
        let mut config = config;
        config.hooks = Some(chain);
        Self { config }
    }

    /// Read-only access to the assembled config (used by builder tests).
    pub fn config(&self) -> &HarnessConfig {
        &self.config
    }

    /// The injected [`StorageProvider`](crate::storage::StorageProvider)
    /// (issue #73). Defaults to an all-no-op provider when `.storage(...)` was
    /// never set.
    pub fn storage(&self) -> &Arc<crate::storage::StorageProvider> {
        &self.config.storage
    }

    /// Convenience accessor for the storage layer's
    /// [`SessionStore`](crate::storage::SessionStore) (issue #73, expose-only).
    pub fn session_store(&self) -> &Arc<dyn crate::storage::SessionStore> {
        self.config.storage.session()
    }

    /// Test-only hook: drive the post-compaction verify→retry→warn loop
    /// directly, bypassing the full ReAct loop. Lets out-of-module tests (e.g.
    /// the #55 compaction-adapter tests) exercise the seam end-to-end without
    /// scripting a complete model conversation. `#[doc(hidden)]`: not part of
    /// the stable surface.
    #[doc(hidden)]
    pub async fn run_compaction_for_test(
        &self,
        session_state: &mut SessionState,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
        usage: &mut AggregateUsage,
    ) {
        self.run_compaction(session_state, session_id, task_id, span_seq, usage)
            .await;
    }

    fn emit(stream: &Option<StreamSink>, event: StreamEvent) {
        if let Some(s) = stream.as_ref() {
            s(event);
        }
    }

    /// Capture a requested tool call as [`ToolCallContent`], truncating its
    /// arguments to `max` UTF-8 bytes (issue #64). The arguments are measured by
    /// their canonical JSON serialization; when over budget they are clipped and
    /// stored as a JSON string carrying the truncation marker (a structured
    /// `Value` cannot be clipped in place), with `arguments_truncated = true`.
    fn capture_tool_call_args(call: &ToolCall, max: usize) -> ToolCallContent {
        let serialized = call.input.to_string();
        let (clipped, truncated) = truncate_field(&serialized, max);
        let arguments = if truncated {
            serde_json::Value::String(clipped)
        } else {
            call.input.clone()
        };
        ToolCallContent {
            name: call.name.clone(),
            arguments,
            arguments_truncated: truncated,
        }
    }

    /// Snapshot the assembled INPUT messages (the full prompt the model saw)
    /// into [`GenAiMessage`]s for LLM-native tracing (issue #64). Each message's
    /// [`Role`] maps to the conventional [`GenAiRole`]; the [`Content`] is
    /// rendered to a plain string and truncated to `max` bytes:
    ///   - `Text { text }`        → the text verbatim
    ///   - `ToolResult(tr)`       → `tr.content` (role stays `Tool`)
    ///   - `ToolCall(tc)`         → `"<name> <compact-json-args>"` (assistant)
    ///   - `Image { media_type }` → `"[image <media_type>]"` — NEVER the base64
    ///
    /// System-first, then history order is preserved because the assembled
    /// `messages` already lead with the `Role::System` prompt.
    fn capture_input_messages(messages: &[Message], max: usize) -> Vec<GenAiMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => GenAiRole::System,
                    Role::User => GenAiRole::User,
                    Role::Assistant => GenAiRole::Assistant,
                    Role::Tool => GenAiRole::Tool,
                };
                let rendered = match &m.content {
                    Content::Text { text } => text.clone(),
                    Content::ToolResult(tr) => tr.content.clone(),
                    Content::ToolCall(tc) => {
                        format!("{} {}", tc.name, tc.input)
                    }
                    // NEVER dump the base64 `data` — placeholder only.
                    Content::Image { media_type, .. } => format!("[image {media_type}]"),
                };
                let (content, truncated) = truncate_field(&rendered, max);
                GenAiMessage {
                    role,
                    content,
                    truncated,
                }
            })
            .collect()
    }

    /// Enrich a recoverable tool-error message that survived (or was not
    /// eligible for) repair, so the model gets actionable feedback instead of a
    /// bare serde message. Appends the tool's parameter schema (when available)
    /// plus a short hint about supplying correctly-typed JSON arguments.
    fn enrich_tool_error(message: &str, schema: Option<&ToolSchema>) -> ToolOutput {
        let mut enriched = message.to_string();
        if let Some(schema) = schema {
            if let Ok(schema_json) = serde_json::to_string(&schema.input_schema) {
                enriched.push_str("\n\nExpected parameter schema: ");
                enriched.push_str(&schema_json);
            }
        }
        enriched.push_str(
            "\n\nHint: provide arguments as correctly-typed JSON \
             (e.g. true/false as a bool, 42 as a number, [\"a\"] as an array) \
             rather than as quoted strings.",
        );
        ToolOutput::Error {
            message: enriched,
            recoverable: true,
        }
    }

    fn budget_exceeded(
        budget: &BudgetLimits,
        used: &BudgetSnapshot,
        started_at: Instant,
    ) -> Option<BudgetLimitType> {
        if let Some(max) = budget.max_turns {
            if used.turns >= max {
                return Some(BudgetLimitType::Turns);
            }
        }
        if let Some(max) = budget.max_input_tokens {
            if used.input_tokens > max as u64 {
                return Some(BudgetLimitType::InputTokens);
            }
        }
        if let Some(max) = budget.max_output_tokens {
            if used.output_tokens > max as u64 {
                return Some(BudgetLimitType::OutputTokens);
            }
        }
        if let Some(max) = budget.max_wall_time {
            if started_at.elapsed() >= max {
                return Some(BudgetLimitType::WallTime);
            }
        }
        if let Some(max) = budget.max_cost_usd {
            if used.cost_usd > max {
                return Some(BudgetLimitType::CostUsd);
            }
        }
        None
    }

    /// Record the terminal outcome and flush the observability session. Called
    /// at every terminal `run_react` outcome (success or any halt) — never on a
    /// `WaitingForHuman` pause, which is not terminal. No-op when no provider is
    /// configured.
    async fn finalize_observability(&self, session_id: &SessionId, outcome: SessionOutcome) {
        if let Some(obs) = self.config.observability.as_ref() {
            obs.set_session_outcome(session_id, outcome);
            obs.flush_session(session_id).await;
        }
    }

    /// Fire registered `Stop` hooks (issue #69, Decision 5/6). Returns
    /// `Some(reason)` when the loop should continue (a hook blocked and the
    /// per-run `max_stop_blocks` cap has not yet been hit), incrementing
    /// `stop_blocks`. Returns `None` to allow normal termination — either no
    /// hooks blocked, no hook chain is configured, or the cap was reached.
    ///
    /// A Stop-hook error (e.g. a failing command handler) is treated as a
    /// non-blocking outcome: the loop terminates normally rather than looping
    /// forever on a broken hook.
    async fn fire_stop_hooks(
        &self,
        session_id: &SessionId,
        task: &Task,
        turn_number: u32,
        last_output_text: &str,
        stop_blocks: &mut u32,
    ) -> Option<String> {
        let chain = self.config.hooks.as_ref()?;
        // Build the rich SessionState the Stop contract requires from the data
        // the ReAct loop has on hand.
        let rich_state = ContextSessionState::new(
            session_id.clone(),
            task.id.clone(),
            task.instruction.clone(),
        );
        let last_output = crate::hooks::TurnOutput {
            text: last_output_text.to_string(),
            had_tool_calls: false,
        };
        let mut ctx = crate::hooks::HookContext::Stop {
            session_id,
            turn_number,
            last_output: &last_output,
            task_instruction: &task.instruction,
            session_state: &rich_state,
        };
        match chain.fire(&mut ctx).await {
            Ok(crate::hooks::FireOutcome::Block { reason }) => {
                if *stop_blocks >= self.config.max_stop_blocks {
                    // R14: cap reached — terminate anyway.
                    None
                } else {
                    *stop_blocks += 1;
                    Some(reason)
                }
            }
            // Continue / Inject / Deny / errors → allow normal termination.
            _ => None,
        }
    }

    /// Drive the ReAct loop, then finalize observability for terminal outcomes.
    /// A `WaitingForHuman` pause is not terminal, so it is never flushed here —
    /// the eventual `resume` path reaches a terminal outcome and flushes then.
    async fn run_react(
        &self,
        task: Task,
        max_iterations: u32,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let result = self
            .run_react_inner(task, max_iterations, session_state, budget_used, on_stream)
            .await;
        match &result {
            RunResult::Success { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Success)
                    .await;
            }
            RunResult::Failure {
                session_id, reason, ..
            } => {
                self.finalize_observability(
                    session_id,
                    SessionOutcome::Failure {
                        reason: format!("{reason:?}"),
                    },
                )
                .await;
            }
            RunResult::WaitingForHuman { .. } => {}
            // Escalation (issue #80) IS a terminal outcome — the harness has
            // run to a clean stop and handed a signal up. Finalize
            // observability with the dedicated `Escalated` outcome (NOT
            // `Partial`). Contrast WaitingForHuman, which is NOT terminal.
            RunResult::Escalate { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Escalated)
                    .await;
            }
        }
        result
    }

    async fn run_react_inner(
        &self,
        task: Task,
        max_iterations: u32,
        mut session_state: SessionState,
        mut budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let session_id = task.session_id.clone();
        let started_at = Instant::now();
        let mut usage = AggregateUsage::default();
        // Per-run Stop-hook block counter (issue #69, Decision 4). Resets on
        // every run() — resume starts fresh. After `max_stop_blocks` consecutive
        // blocks the loop terminates anyway (R14).
        let mut stop_blocks: u32 = 0;
        // Monotonic per-run span counter for turn / tool-call span ids, and the
        // most recent turn span — parent for the tool-call spans of that turn.
        let mut span_seq: u64 = 0;
        // Set on every turn before any tool dispatch; the initial `None` only
        // covers the (unreachable) case of a tool span with no preceding turn.
        #[allow(unused_assignments)]
        let mut current_turn_base: Option<SpanBase> = None;
        let effective_turn_cap = match task.budget.max_turns {
            Some(t) => t.min(max_iterations),
            None => max_iterations,
        };

        loop {
            // Layer-1 budget gates before the turn.
            if budget_used.turns >= effective_turn_cap {
                return RunResult::Failure {
                    reason: HaltReason::BudgetExceeded {
                        limit_type: BudgetLimitType::Turns,
                    },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                };
            }
            if let Some(limit_type) = Self::budget_exceeded(&task.budget, &budget_used, started_at)
            {
                return RunResult::Failure {
                    reason: HaltReason::BudgetExceeded { limit_type },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                };
            }

            // Middleware: BeforeTurn.
            if let Some(mw) = self.config.middleware.as_ref() {
                match mw.fire(HookPoint::BeforeTurn, &session_state).await {
                    MiddlewareDecision::Continue => {}
                    MiddlewareDecision::ContinueWithModification { .. } => {}
                    MiddlewareDecision::Halt { reason } => {
                        return RunResult::Failure {
                            reason: HaltReason::MiddlewareHalt {
                                hook: HookPoint::BeforeTurn,
                                reason,
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                        };
                    }
                    MiddlewareDecision::SurfaceToHuman { request } => {
                        let state = PausedState {
                            session_id: session_id.clone(),
                            task_id: task.id.clone(),
                            turn_number: budget_used.turns,
                            session_state,
                            pending_tool_calls: vec![],
                            approved_results: vec![],
                            human_request: Some(request.clone()),
                            task,
                            budget_used,
                            child_state: None,
                        };
                        return RunResult::WaitingForHuman {
                            state: Box::new(state),
                            request,
                        };
                    }
                }
            }

            // Assemble + invoke agent for one turn.
            let context = self
                .config
                .context_manager
                .assemble(&session_state, &task)
                .await;
            Self::emit(
                &on_stream,
                StreamEvent::TurnStart {
                    turn: budget_used.turns + 1,
                },
            );
            let turn_started_at = Timestamp::now();
            let turn_clock = Instant::now();
            // LLM-native content capture (issue #64): snapshot the assembled
            // INPUT messages (the full prompt the model saw) BEFORE `context`
            // is moved into `agent.turn`. Zero-cost when the guard is off (no
            // clone of the message history).
            let input_messages = if self.config.content_capture.enabled {
                Some(Self::capture_input_messages(
                    &context.messages,
                    self.config.content_capture.max_field_len,
                ))
            } else {
                None
            };
            let result = self.config.agent.turn(context).await;
            budget_used.turns += 1;
            // Emit a turn span for every model call (issue #12). Fire-and-forget;
            // it never affects control flow. The span base is retained as the
            // parent for any tool-call spans dispatched this turn.
            {
                let zero = TokenUsage::default();
                let u = match &result {
                    TurnResult::ToolCallRequested { usage, .. }
                    | TurnResult::FinalResponse { usage, .. } => usage,
                    TurnResult::Error { usage, .. } => usage.as_ref().unwrap_or(&zero),
                };
                let (stop_reason, tool_calls_requested) = match &result {
                    TurnResult::FinalResponse { .. } => (StopReason::EndTurn, 0),
                    TurnResult::ToolCallRequested { calls, .. } => {
                        (StopReason::ToolUse, calls.len() as u32)
                    }
                    TurnResult::Error { .. } => (StopReason::EndTurn, 0),
                };
                let mut base = SpanBase::new_root(
                    SpanId::new(format!(
                        "{}-turn-{}",
                        session_id.as_str(),
                        budget_used.turns
                    )),
                    session_id.clone(),
                    task.id.clone(),
                    SpanKind::Turn,
                    turn_started_at,
                );
                let status = match &result {
                    TurnResult::Error { error, .. } => SpanStatus::Error {
                        message: format!("{error:?}"),
                    },
                    _ => SpanStatus::Ok,
                };
                base.finish(
                    Timestamp::now(),
                    status,
                    turn_clock.elapsed().as_millis() as u64,
                );
                if let Some(obs) = self.config.observability.as_ref() {
                    // LLM-native content capture (issue #64): output text +
                    // requested tool calls, ONLY when the guard is enabled.
                    // Decision 4: turn span carries output + tool calls; no
                    // assembled input-message history.
                    let cc = &self.config.content_capture;
                    let (output_text, tool_calls) = if cc.enabled {
                        let output_text = match &result {
                            TurnResult::FinalResponse { content, .. } => {
                                let (content, truncated) =
                                    truncate_field(content, cc.max_field_len);
                                Some(GenAiMessage {
                                    role: GenAiRole::Assistant,
                                    content,
                                    truncated,
                                })
                            }
                            _ => None,
                        };
                        let tool_calls = match &result {
                            TurnResult::ToolCallRequested { calls, .. } => Some(
                                calls
                                    .iter()
                                    .map(|c| Self::capture_tool_call_args(c, cc.max_field_len))
                                    .collect(),
                            ),
                            _ => None,
                        };
                        (output_text, tool_calls)
                    } else {
                        (None, None)
                    };
                    obs.emit_turn(TurnSpan {
                        base: base.clone(),
                        turn_number: budget_used.turns,
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_read_tokens: u.cache_read_tokens,
                        cache_write_tokens: u.cache_write_tokens,
                        cost_usd: self.config.pricing.cost_for(
                            u.input_tokens,
                            u.output_tokens,
                            u.cache_read_tokens,
                            u.cache_write_tokens,
                        ),
                        stop_reason,
                        tool_calls_requested,
                        output_text,
                        tool_calls,
                        input_messages,
                    });
                }
                span_seq += 1;
                current_turn_base = Some(base);
            }
            Self::emit(
                &on_stream,
                StreamEvent::TurnEnd {
                    turn: budget_used.turns,
                },
            );

            match result {
                TurnResult::FinalResponse { content, usage: u } => {
                    usage.add_turn(&u);
                    budget_used.input_tokens += u.input_tokens as u64;
                    budget_used.output_tokens += u.output_tokens as u64;

                    if let Some(mw) = self.config.middleware.as_ref() {
                        match mw.fire(HookPoint::BeforeCompletion, &session_state).await {
                            MiddlewareDecision::Continue => {}
                            MiddlewareDecision::ContinueWithModification { .. } => {}
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeCompletion,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                };
                            }
                            MiddlewareDecision::SurfaceToHuman { request } => {
                                let state = PausedState {
                                    session_id: session_id.clone(),
                                    task_id: task.id.clone(),
                                    turn_number: budget_used.turns,
                                    session_state,
                                    pending_tool_calls: vec![],
                                    approved_results: vec![],
                                    human_request: Some(request.clone()),
                                    task,
                                    budget_used,
                                    child_state: None,
                                };
                                return RunResult::WaitingForHuman {
                                    state: Box::new(state),
                                    request,
                                };
                            }
                        }
                    }

                    match self
                        .config
                        .termination_policy
                        .evaluate(&session_state, &budget_used)
                        .await
                    {
                        TerminationDecision::Halt { reason } => {
                            return RunResult::Failure {
                                reason: HaltReason::TerminationPolicyHalt { reason },
                                session_id,
                                usage,
                                turns: budget_used.turns,
                            };
                        }
                        TerminationDecision::Continue => {
                            // Record the assistant's final text in history so a
                            // continued session reflects what the agent said.
                            let msg = Message {
                                role: Role::Assistant,
                                content: Content::Text {
                                    text: content.clone(),
                                },
                            };
                            self.config
                                .context_manager
                                .append_assistant_message(&mut session_state, &msg)
                                .await;

                            // Stop hook (issue #69, Decision 5). The strategy
                            // believes the task is done; fire registered Stop
                            // hooks synchronously. If any blocks (and we are
                            // under `max_stop_blocks`), inject the reason via
                            // the same path `ForceAnotherTurn` uses and continue
                            // the loop instead of terminating.
                            if let Some(reason) = self
                                .fire_stop_hooks(
                                    &session_id,
                                    &task,
                                    budget_used.turns,
                                    &content,
                                    &mut stop_blocks,
                                )
                                .await
                            {
                                self.config
                                    .context_manager
                                    .append_user_message(&mut session_state, &reason)
                                    .await;
                                continue;
                            }

                            Self::emit(
                                &on_stream,
                                StreamEvent::FinalResponse {
                                    content: content.clone(),
                                },
                            );
                            return RunResult::Success {
                                output: content,
                                session_id,
                                usage,
                                turns: budget_used.turns,
                            };
                        }
                    }
                }

                TurnResult::ToolCallRequested { calls, usage: u } => {
                    usage.add_turn(&u);
                    budget_used.input_tokens += u.input_tokens as u64;
                    budget_used.output_tokens += u.output_tokens as u64;

                    // Always-halt short circuit (Layer 1).
                    if let Some(c) = calls
                        .iter()
                        .find(|c| self.config.tool_registry.is_always_halt(&c.name))
                    {
                        return RunResult::Failure {
                            reason: HaltReason::UnrecoverableToolError {
                                tool: c.name.clone(),
                                error: "tool is annotated always_halt".into(),
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                        };
                    }

                    // Record the assistant's turn (the tool calls the model
                    // requested) as soon as the calls are known — BEFORE the
                    // BeforeTool middleware (which may pause via SurfaceToHuman)
                    // and before any tool result. This keeps the conversation
                    // well-formed (assistant tool_use precedes its tool result)
                    // on every path, including human-in-the-loop resume, so
                    // resume_inner never has to append it. The recorded turn
                    // reflects the model's original request; a middleware or
                    // human modification changes only what is dispatched.
                    for call in &calls {
                        let msg = Message {
                            role: Role::Assistant,
                            content: Content::ToolCall(call.clone()),
                        };
                        self.config
                            .context_manager
                            .append_assistant_message(&mut session_state, &msg)
                            .await;
                    }

                    // Middleware: BeforeTool.
                    let calls = match self.config.middleware.as_ref() {
                        None => calls,
                        Some(mw) => match mw.fire(HookPoint::BeforeTool, &session_state).await {
                            MiddlewareDecision::Continue => calls,
                            MiddlewareDecision::ContinueWithModification { calls: modified } => {
                                modified
                            }
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeTool,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                };
                            }
                            MiddlewareDecision::SurfaceToHuman { request } => {
                                let state = PausedState {
                                    session_id: session_id.clone(),
                                    task_id: task.id.clone(),
                                    turn_number: budget_used.turns,
                                    session_state,
                                    pending_tool_calls: calls,
                                    approved_results: vec![],
                                    human_request: Some(request.clone()),
                                    task,
                                    budget_used,
                                    child_state: None,
                                };
                                return RunResult::WaitingForHuman {
                                    state: Box::new(state),
                                    request,
                                };
                            }
                        },
                    };

                    let mut approved_results: Vec<ToolResult> = Vec::new();
                    for (i, call) in calls.iter().enumerate() {
                        // Sandbox validation.
                        if let Err(violation) = self.config.sandbox.validate(call).await {
                            if violation.is_always_halt() {
                                return RunResult::Failure {
                                    reason: HaltReason::SandboxViolation { violation },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                };
                            }
                            // Layer-2 default: recoverable — append as tool error.
                            let tr = ToolResult {
                                call_id: call.id.clone(),
                                output: ToolOutput::Error {
                                    message: format!("sandbox: {violation:?}"),
                                    recoverable: true,
                                },
                            };
                            Self::emit(
                                &on_stream,
                                StreamEvent::ToolResult {
                                    call_id: call.id.clone(),
                                    is_error: true,
                                },
                            );
                            self.config
                                .context_manager
                                .append_tool_result(&mut session_state, &tr)
                                .await;
                            approved_results.push(tr);
                            continue;
                        }

                        Self::emit(
                            &on_stream,
                            StreamEvent::ToolCall {
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                            },
                        );
                        let tool_started_at = Timestamp::now();
                        let tool_clock = Instant::now();
                        // `effective_call` is the call actually dispatched. It
                        // starts as the agent's call and may be replaced by a
                        // repaired variant below. Spans/results are stamped from it.
                        let mut effective_call = call.clone();
                        let mut output = self
                            .config
                            .tool_registry
                            .dispatch(effective_call.clone())
                            .await;

                        // Tool-call repair (additive): if the dispatch returned a
                        // recoverable error and a repair provider is configured,
                        // try to coerce the arguments and re-dispatch, bounded by
                        // `max_repair_attempts`. If repair gives up or the retry
                        // still errors, fall through to the existing behaviour but
                        // with an enriched error message (schema + hint).
                        if let Some(repair) = self.config.tool_call_repair.as_ref() {
                            let mut attempts_left = self.config.max_repair_attempts;
                            while attempts_left > 0 {
                                let ToolOutput::Error {
                                    ref message,
                                    recoverable: true,
                                } = output
                                else {
                                    break;
                                };
                                let schema = self
                                    .config
                                    .tool_registry
                                    .schemas()
                                    .into_iter()
                                    .find(|s| s.name == effective_call.name);
                                let Some(repaired_call) =
                                    repair.repair(&effective_call, message, schema.as_ref())
                                else {
                                    // Repair gave up: enrich the error fed back to
                                    // the model with the parameter schema + a hint.
                                    output = Self::enrich_tool_error(message, schema.as_ref());
                                    break;
                                };
                                attempts_left -= 1;
                                effective_call = repaired_call;
                                output = self
                                    .config
                                    .tool_registry
                                    .dispatch(effective_call.clone())
                                    .await;
                                if !matches!(
                                    output,
                                    ToolOutput::Error {
                                        recoverable: true,
                                        ..
                                    }
                                ) {
                                    break;
                                }
                                // Still a recoverable error and no budget left:
                                // enrich before giving up.
                                if attempts_left == 0 {
                                    if let ToolOutput::Error {
                                        message: ref m,
                                        recoverable: true,
                                    } = output
                                    {
                                        let schema = self
                                            .config
                                            .tool_registry
                                            .schemas()
                                            .into_iter()
                                            .find(|s| s.name == effective_call.name);
                                        output = Self::enrich_tool_error(m, schema.as_ref());
                                    }
                                }
                            }
                        }
                        let call = &effective_call;

                        // Pause propagation: WaitingForHuman from a subagent tool.
                        if let ToolOutput::WaitingForHuman {
                            child_state,
                            request,
                        } = output
                        {
                            let remaining = calls[i + 1..].to_vec();
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: remaining,
                                approved_results,
                                human_request: Some(request.clone()),
                                task,
                                budget_used,
                                child_state: Some(*child_state),
                            };
                            return RunResult::WaitingForHuman {
                                state: Box::new(state),
                                request,
                            };
                        }

                        // Escalation propagation (issue #80): a tool requests a
                        // structural state change. The harness is a pure
                        // intermediary — it does NOT act on the signal. It
                        // terminates cleanly, preserves session state for a
                        // possible resume, and returns RunResult::Escalate.
                        // The escalation is a control signal, NOT a conversation
                        // turn: it is never appended to message history, and the
                        // remaining tool calls in this batch are preserved into
                        // pending_tool_calls (mirroring WaitingForHuman). The
                        // signal is NOT stored in PausedState, so it is
                        // discarded on resume — the harness never re-acts on it.
                        if let ToolOutput::Escalate { signal } = output {
                            let remaining = calls[i + 1..].to_vec();
                            let turns = budget_used.turns;
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: remaining,
                                approved_results,
                                human_request: None,
                                task,
                                budget_used,
                                child_state: None,
                            };
                            return RunResult::Escalate {
                                signal,
                                state: Box::new(state),
                                session_id,
                                usage,
                                turns,
                            };
                        }

                        // Clarification pause (issue #81, Q4b): a tool (e.g.
                        // `ask_user_question`) needs a human answer before it can
                        // produce a result. UNLIKE the subagent `WaitingForHuman`
                        // path, there is NO `ChildPausedState`: the loop builds a
                        // `PausedState` directly with `human_request` set to
                        // `Clarification`. The CLARIFYING call itself is preserved
                        // as the head of `pending_tool_calls` (followed by the
                        // remaining batch) so that, on resume, the human's answer
                        // is injected as the tool RESULT for that pending call.
                        if let ToolOutput::AwaitingClarification { question, options } = output {
                            let mut pending = vec![call.clone()];
                            pending.extend_from_slice(&calls[i + 1..]);
                            let request = HumanRequest::Clarification { question, options };
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: pending,
                                approved_results,
                                human_request: Some(request.clone()),
                                task,
                                budget_used,
                                child_state: None,
                            };
                            return RunResult::WaitingForHuman {
                                state: Box::new(state),
                                request,
                            };
                        }

                        // SendMessage (issue #81): the `send_message` tool surfaces
                        // an out-of-band message to the user. The loop emits a
                        // `StreamEvent::UserMessage` rather than collapsing the
                        // content into a normal tool result, then records a minimal
                        // success result so the loop continues.
                        let output = if call.name == crate::tools::SendMessageTool::NAME {
                            if let ToolOutput::Success { content, .. } = &output {
                                Self::emit(
                                    &on_stream,
                                    StreamEvent::UserMessage {
                                        content: content.clone(),
                                    },
                                );
                            }
                            output
                        } else {
                            output
                        };

                        let is_error = matches!(output, ToolOutput::Error { .. });
                        // Layer-2: unrecoverable tool error halts immediately.
                        if let ToolOutput::Error {
                            ref message,
                            recoverable: false,
                        } = output
                        {
                            return RunResult::Failure {
                                reason: HaltReason::UnrecoverableToolError {
                                    tool: call.name.clone(),
                                    error: message.clone(),
                                },
                                session_id,
                                usage,
                                turns: budget_used.turns,
                            };
                        }

                        // Tool-call span (issue #12), child of the current turn.
                        if let Some(obs) = self.config.observability.as_ref() {
                            // LLM-native content capture (issue #64): tool args +
                            // tool result, ONLY when the guard is enabled.
                            let cc = &self.config.content_capture;
                            let (tool_args_content, tool_result_content) = if cc.enabled {
                                let args = Self::capture_tool_call_args(call, cc.max_field_len);
                                let result = match &output {
                                    ToolOutput::Success { content, .. } => {
                                        let (content, t) =
                                            truncate_field(content, cc.max_field_len);
                                        Some(ToolResultContent {
                                            content,
                                            truncated: t,
                                        })
                                    }
                                    ToolOutput::Error { message, .. } => {
                                        let (content, t) =
                                            truncate_field(message, cc.max_field_len);
                                        Some(ToolResultContent {
                                            content,
                                            truncated: t,
                                        })
                                    }
                                    ToolOutput::WaitingForHuman { .. } => None,
                                    // Escalate / AwaitingClarification are handled
                                    // and returned above; they never reach the
                                    // capture path.
                                    ToolOutput::Escalate { .. } => None,
                                    ToolOutput::AwaitingClarification { .. } => None,
                                };
                                (Some(args), result)
                            } else {
                                (None, None)
                            };
                            let (output_size_bytes, truncated) = match &output {
                                ToolOutput::Success { content, truncated } => {
                                    (content.len(), *truncated)
                                }
                                ToolOutput::Error { message, .. } => (message.len(), false),
                                ToolOutput::WaitingForHuman { .. } => (0, false),
                                ToolOutput::Escalate { .. } => (0, false),
                                ToolOutput::AwaitingClarification { .. } => (0, false),
                            };
                            let span_id =
                                SpanId::new(format!("{}-tool-{}", session_id.as_str(), span_seq));
                            let mut base = match &current_turn_base {
                                Some(parent) => SpanBase::new_child(
                                    span_id,
                                    parent,
                                    SpanKind::ToolCall,
                                    tool_started_at,
                                ),
                                None => SpanBase::new_root(
                                    span_id,
                                    session_id.clone(),
                                    task.id.clone(),
                                    SpanKind::ToolCall,
                                    tool_started_at,
                                ),
                            };
                            let status = if is_error {
                                SpanStatus::Error {
                                    message: "tool returned a recoverable error".into(),
                                }
                            } else {
                                SpanStatus::Ok
                            };
                            base.finish(
                                Timestamp::now(),
                                status,
                                tool_clock.elapsed().as_millis() as u64,
                            );
                            obs.emit_tool_call(ToolCallSpan {
                                base,
                                tool_name: call.name.clone(),
                                call_id: call.id.clone(),
                                parameters_size_bytes: call.input.to_string().len(),
                                output_size_bytes,
                                truncated,
                                sandbox_mode: String::new(),
                                sandbox_violations: Vec::new(),
                                arguments: tool_args_content,
                                result: tool_result_content,
                            });
                            span_seq += 1;
                        }

                        let tr = ToolResult {
                            call_id: call.id.clone(),
                            output,
                        };
                        Self::emit(
                            &on_stream,
                            StreamEvent::ToolResult {
                                call_id: call.id.clone(),
                                is_error,
                            },
                        );
                        self.config
                            .context_manager
                            .append_tool_result(&mut session_state, &tr)
                            .await;
                        approved_results.push(tr);
                    }

                    // Middleware: AfterTool.
                    if let Some(mw) = self.config.middleware.as_ref() {
                        if let MiddlewareDecision::Halt { reason } =
                            mw.fire(HookPoint::AfterTool, &session_state).await
                        {
                            return RunResult::Failure {
                                reason: HaltReason::MiddlewareHalt {
                                    hook: HookPoint::AfterTool,
                                    reason,
                                },
                                session_id,
                                usage,
                                turns: budget_used.turns,
                            };
                        }
                    }

                    // Compaction (issue #46): after tool results are appended,
                    // before the loop restarts — matches the concepts-doc loop
                    // diagram's "compact if should_compact()" placement. Runs the
                    // verify→retry→warn loop; never halts the run.
                    if self.config.context_manager.should_compact(&session_state) {
                        self.run_compaction(
                            &mut session_state,
                            &session_id,
                            &task.id,
                            &mut span_seq,
                            &mut usage,
                        )
                        .await;
                    }

                    continue;
                }

                TurnResult::Error { error, usage: u } => {
                    if let Some(u) = u.as_ref() {
                        usage.add_turn(u);
                        budget_used.input_tokens += u.input_tokens as u64;
                        budget_used.output_tokens += u.output_tokens as u64;
                    }
                    return RunResult::Failure {
                        reason: HaltReason::AgentError { error },
                        session_id,
                        usage,
                        turns: budget_used.turns,
                    };
                }
            }
        }
    }

    // ========================================================================
    // SelfVerifying loop strategy (issue #61)
    // ========================================================================
    //
    // Loop-within-a-loop. Each round-trip runs a bounded BUILD ReAct sub-loop
    // (the agent does work until it claims done), then a fresh EVALUATE run
    // (a separate evaluator agent on a read-only sandbox in a never-shared
    // session), then asks the injected [`Verifier`] to translate
    // `(build_result, eval_result)` into a verdict. `Passed` ⇒ Success;
    // `Failed { reason }` ⇒ inject `reason` into the build context and loop.
    //
    // ## Config fields this strategy reads (both default `None`):
    //   - `config.verifier` — the oracle. REQUIRED: `None` ⇒
    //     `Failure { SelfVerifyMisconfigured }` (D4) — a typed halt, NOT a panic.
    //     Its `max_iterations()` (default 3) caps the round-trips (D3); per-run
    //     `max_stop_blocks` does NOT enter the picture for this strategy.
    //   - `config.evaluator_agent` — the evaluate-phase agent. Defaulting
    //     contract (D2): `config.evaluator_agent.as_ref().unwrap_or(&config.agent)`
    //     (identical to `planner_agent`). The read-only sandbox and the fresh
    //     `SessionId::generate()` for the evaluate phase are derived INTERNALLY.
    //
    // ## Terminal `HaltReason` variants this strategy produces (peers, D4):
    //   - `SelfVerifyExhausted { iterations, last_reason }` — ran out of
    //     `max_iterations` round-trips without a pass (clean exhaustion).
    //   - `SelfVerifyMisconfigured { reason }` — `config.verifier` is `None`.
    //
    // ## Rules enforced (each maps to a test):
    //   - R1  build phase runs a ReAct loop until the agent claims done.
    //   - R2  evaluate phase uses a FRESH SessionId never shared with build.
    //   - R3  evaluate phase uses a read-only sandbox (writes ⇒ ReadOnlyViolation;
    //         the build sandbox is unaffected).
    //   - R4  the evaluator carries the `role-evaluator` chunk. PRESENCE-ONLY:
    //         the chunk's content (if the provider supplies it) is prepended to
    //         the evaluate seed message; the chunk-condition machinery (#79) is
    //         NOT otherwise wired into the harness loop.
    //   - R5  Default-FAIL: an indeterminate evaluator verdict keeps looping.
    //   - R6  on findings, the verdict reason is injected into the build context
    //         and the build loop resumes (same path the Stop-block injection uses).
    //   - R7  stagnation guard: always-Failed ⇒ exactly `max_iterations` cycles ⇒
    //         `Failure { SelfVerifyExhausted }`.
    //   - R8  budgets fold BOTH phases across ALL iterations.
    //   - R9  build vs evaluate are distinguishable in traces (distinct session
    //         ids: the build session vs each evaluate's generated session).
    //   - R11 `verifier == None` ⇒ `Failure { SelfVerifyMisconfigured }`.
    //
    // No `// SPEC QUESTION:` markers — all four forks (D1–D4) are resolved.
    async fn run_self_verifying(
        &self,
        task: Task,
        mut session_state: SessionState,
        budget_used: BudgetSnapshot,
        _on_stream: Option<StreamSink>,
    ) -> RunResult {
        let build_session_id = task.session_id.clone();

        // D4/R11: a missing verifier is a typed halt, not a panic.
        let verifier = match self.config.verifier.as_ref() {
            Some(v) => v.clone(),
            None => {
                let result = RunResult::Failure {
                    reason: HaltReason::SelfVerifyMisconfigured {
                        reason: "SelfVerifying requires `config.verifier`, but it is None"
                            .to_string(),
                    },
                    session_id: build_session_id,
                    usage: AggregateUsage::default(),
                    turns: 0,
                };
                self.finalize_self_verifying(&result).await;
                return result;
            }
        };

        let max_iterations = verifier.max_iterations();
        // Shared budget threaded across every build + evaluate sub-run (R8).
        let mut carried = budget_used;
        // Cumulative usage across ALL build + evaluate runs of ALL iterations.
        let mut total_usage = AggregateUsage::default();
        // The most recent verifier failure reason (for SelfVerifyExhausted).
        let mut last_reason = String::new();

        for iteration in 0..max_iterations {
            // ── Build phase (R1): bounded ReAct sub-loop carrying the shared
            //    budget. The first iteration's seed instruction is already in
            //    `session_state` (from `run_inner`); later iterations already
            //    have the prior verdict reason injected as a user message (R6).
            let build_task = Task {
                id: task.id.clone(),
                instruction: task.instruction.clone(),
                session_id: build_session_id.clone(),
                budget: task.budget.clone(),
                loop_strategy: task.loop_strategy.clone(),
            };
            let build_cap = task.budget.max_turns.unwrap_or(u32::MAX);
            let build_result = self
                .run_react_inner(
                    build_task,
                    build_cap,
                    session_state.clone(),
                    carried.clone(),
                    // Sub-loops run with a suppressed sink (mirrors PlanExecute);
                    // terminal observability is finalized by this strategy.
                    None,
                )
                .await;
            Self::fold_usage(&mut total_usage, &mut carried, &build_result);

            // A build run that paused / escalated is propagated up unchanged —
            // the caller must handle it before verification can resume.
            match &build_result {
                RunResult::WaitingForHuman { state, request } => {
                    return RunResult::WaitingForHuman {
                        state: state.clone(),
                        request: request.clone(),
                    };
                }
                RunResult::Escalate { .. } => {
                    self.finalize_self_verifying(&build_result).await;
                    return build_result;
                }
                _ => {}
            }

            // ── Evaluate phase (R2/R3/R4): a fresh evaluator RUN. Distinct
            //    generated session id (never shared with build — R2/R9), a
            //    read-only sandbox derived internally (R3), the evaluator agent
            //    (D2 defaulting), and the `role-evaluator` chunk (R4).
            let eval_result = self
                .run_evaluate_phase(&task, &mut carried, &mut total_usage)
                .await;

            // ── Verdict.
            let input = VerifierInput {
                build_result,
                eval_result,
                workspace: self.config.sandbox.workspace_root().to_path_buf(),
                iteration,
            };
            match verifier.verify(&input).await {
                VerifierVerdict::Passed => {
                    // Reuse the build run's output as the run's output.
                    let (output, turns) = match input.build_result {
                        RunResult::Success { output, turns, .. } => (output, turns),
                        // A non-Success build can still be `Passed` only if a
                        // bespoke verifier says so; fall back to a generic handle.
                        _ => (String::new(), carried.turns),
                    };
                    let result = RunResult::Success {
                        output,
                        session_id: build_session_id,
                        usage: total_usage,
                        turns,
                    };
                    self.finalize_self_verifying(&result).await;
                    return result;
                }
                VerifierVerdict::Failed { reason } => {
                    // R5/R6: Default-FAIL keeps looping; inject the reason into
                    // the build context via the SAME path the Stop-block uses
                    // (append a user message) so the next build iteration sees it.
                    last_reason = reason.clone();
                    self.config
                        .context_manager
                        .append_user_message(&mut session_state, &reason)
                        .await;
                }
            }
        }

        // R7: ran out of round-trips without a pass — clean exhaustion.
        let result = RunResult::Failure {
            reason: HaltReason::SelfVerifyExhausted {
                iterations: max_iterations,
                last_reason,
            },
            session_id: build_session_id,
            usage: total_usage,
            turns: carried.turns,
        };
        self.finalize_self_verifying(&result).await;
        result
    }

    /// Run the SelfVerifying evaluate phase (issue #61): a fresh evaluator RUN
    /// over a read-only sandbox in a never-shared session.
    ///
    /// Builds a child [`StandardHarness`] from a clone of `self.config` with the
    /// `agent` swapped to the evaluator agent (D2 defaulting) and the `sandbox`
    /// wrapped in a [`ReadOnlySandbox`] (R3). The evaluator runs a fresh ReAct
    /// loop seeded with the `role-evaluator` chunk (R4, presence-only) plus a
    /// review directive, in a freshly [`generate`](SessionId::generate)d session
    /// (R2/R9). Folds the evaluate run's usage into `total_usage`/`carried` (R8)
    /// and returns its terminal [`RunResult`].
    async fn run_evaluate_phase(
        &self,
        task: &Task,
        carried: &mut BudgetSnapshot,
        total_usage: &mut AggregateUsage,
    ) -> RunResult {
        // D2: evaluator agent defaulting — identical contract to `planner_agent`.
        let evaluator = self
            .config
            .evaluator_agent
            .as_ref()
            .unwrap_or(&self.config.agent)
            .clone();

        // R3: derive a read-only sandbox internally from the build sandbox.
        let read_only_sandbox: Arc<dyn SandboxProvider> =
            Arc::new(ReadOnlySandbox::new(self.config.sandbox.clone()));

        // R2/R9: fresh, never-shared session id for the evaluate run.
        let eval_session_id = SessionId::generate();

        // R4 (presence-only): prepend the `role-evaluator` chunk content (if the
        // configured provider supplies it) to the review directive.
        let role_chunk = self.role_evaluator_chunk().await;
        let directive = match role_chunk {
            Some(content) => format!(
                "{content}\n\nReview the work produced for the following task and \
                 report whether it is correct. You did NOT write this code; default \
                 to FAIL unless you can confirm it is right.\n\nTask:\n{}",
                task.instruction
            ),
            None => format!(
                "Review the work produced for the following task and report whether \
                 it is correct. You did NOT write this code; default to FAIL unless \
                 you can confirm it is right.\n\nTask:\n{}",
                task.instruction
            ),
        };

        let eval_task = Task {
            id: TaskId::generate(),
            instruction: directive.clone(),
            session_id: eval_session_id.clone(),
            budget: task.budget.clone(),
            loop_strategy: LoopStrategy::ReAct {
                max_iterations: task.budget.max_turns.unwrap_or(u32::MAX),
            },
        };

        // Child harness: clone the config, swap agent + sandbox. Cloning shares
        // the same observability/storage seams so the evaluate run's spans land
        // in the SAME trace stream (distinguished by its distinct session id).
        let mut eval_config = self.config.clone();
        eval_config.agent = evaluator;
        eval_config.sandbox = read_only_sandbox;
        let eval_harness = StandardHarness::new(eval_config);

        let mut eval_state = SessionState::default();
        eval_harness
            .config
            .context_manager
            .append_user_message(&mut eval_state, &directive)
            .await;

        let cap = task.budget.max_turns.unwrap_or(u32::MAX);
        let eval_result = eval_harness
            .run_react(eval_task, cap, eval_state, BudgetSnapshot::default(), None)
            .await;

        Self::fold_usage(total_usage, carried, &eval_result);
        eval_result
    }

    /// Look up the `role-evaluator` chunk content from the configured
    /// [`ChunkProvider`](crate::prompt_assembly::ChunkProvider) (R4,
    /// presence-only). Returns `None` if the provider has no such chunk or fails
    /// to load.
    async fn role_evaluator_chunk(&self) -> Option<String> {
        let chunks = self.config.chunk_provider.load().await.ok()?;
        chunks
            .into_iter()
            .find(|c| c.id == "role-evaluator")
            .map(|c| c.content)
    }

    /// Fold a sub-run's token usage / turn count into the cumulative
    /// `total_usage` and the shared `carried` budget snapshot (R8). Mirrors the
    /// PlanExecute budget fold. `carried.turns` becomes the sub-run's ABSOLUTE
    /// turn count (the build sub-loop already gates on cumulative turns); the
    /// evaluate run reports its own fresh-session turns, which are added in.
    fn fold_usage(total_usage: &mut AggregateUsage, carried: &mut BudgetSnapshot, r: &RunResult) {
        let (usage, turns) = match r {
            RunResult::Success { usage, turns, .. }
            | RunResult::Failure { usage, turns, .. }
            | RunResult::Escalate { usage, turns, .. } => (usage, *turns),
            RunResult::WaitingForHuman { .. } => return,
        };
        total_usage.input_tokens += usage.input_tokens;
        total_usage.output_tokens += usage.output_tokens;
        total_usage.cache_read_tokens += usage.cache_read_tokens;
        total_usage.cache_write_tokens += usage.cache_write_tokens;
        total_usage.cost_usd += usage.cost_usd;
        carried.input_tokens += usage.input_tokens;
        carried.output_tokens += usage.output_tokens;
        carried.turns = carried.turns.max(turns);
    }

    /// Finalize observability for a terminal SelfVerifying outcome. Mirrors the
    /// tail of [`run_react`](Self::run_react) / [`finalize_plan_execute`].
    /// `WaitingForHuman` is not terminal and is never flushed here.
    async fn finalize_self_verifying(&self, result: &RunResult) {
        match result {
            RunResult::Success { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Success)
                    .await;
            }
            RunResult::Failure {
                session_id, reason, ..
            } => {
                self.finalize_observability(
                    session_id,
                    SessionOutcome::Failure {
                        reason: format!("{reason:?}"),
                    },
                )
                .await;
            }
            RunResult::WaitingForHuman { .. } => {}
            RunResult::Escalate { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Escalated)
                    .await;
            }
        }
    }

    // ========================================================================
    // Ralph loop strategy (issue #58)
    // ========================================================================
    //
    // Multi-context-window continuation loop. The model's exit attempt is
    // intercepted — instead of terminating, the harness RESETS the context
    // window (discards the prior `SessionState`, builds a FRESH
    // `SessionState::default()`), RELOADS state from the filesystem (the
    // deterministic `.spore/progress.json` + `.spore/feature_list.json` files —
    // B4: no git-log read in v1), and resumes — until an external completion
    // check passes. The filesystem is what makes multi-context-window work
    // possible: each window starts from nothing but the instruction plus the
    // reloaded `.spore/` state.
    //
    // ## Completion mechanism (B1): driven off the `Stop` hook (issue #69).
    //   At construction [`StandardHarness::new`] registers a [`RalphStopHook`]
    //   that reads `.spore/progress.json` (+ `.spore/feature_list.json`): while
    //   tasks remain incomplete it returns [`HookDecision::Block`] (the reason
    //   describes what is left) and the harness loops into a new context window;
    //   when all tasks are complete it returns [`HookDecision::Continue`] and the
    //   loop terminates with success. There is NO dedicated `completion_check`
    //   config field and the deprecated `CompletionCheck` trait is NOT reused.
    //   The harness fires the Stop hook (`fire_stop_hooks` on `FinalResponse`);
    //   this strategy ALSO evaluates the SAME filesystem check between windows
    //   ([`ralph_completion_status`]) to drive the OUTER reset loop and to decide
    //   the terminal outcome.
    //
    // ## Config fields this strategy reads:
    //   - `config.max_resets` (B3, default 3) — the OUTER loop cap: the maximum
    //     number of context-window RESETS. Independent from `max_turns` (the
    //     per-window ReAct turn budget) and `max_stop_blocks` (the per-sub-loop
    //     Stop-block cap).
    //
    // ## Canonical filesystem paths (B2, `.spore/`-prefixed):
    //   - `{workspace_root}/.spore/progress.json`
    //   - `{workspace_root}/.spore/feature_list.json`
    //   (matches [`FeatureListCheck::new`](crate::termination::FeatureListCheck)).
    //
    // ## Terminal `HaltReason` variant this strategy produces (peer to
    //    [`SelfVerifyExhausted`](HaltReason::SelfVerifyExhausted)):
    //   - [`RalphCompletionUnmet`](HaltReason::RalphCompletionUnmet)
    //     `{ iterations, last_reason }` — reached `max_resets` context windows
    //     with tasks still incomplete.
    //
    // ## Rules enforced (each maps to a test):
    //   - R1  the model's exit attempt RESETS the context window instead of
    //         terminating, while tasks remain incomplete.
    //   - R2  each reset builds a FRESH `SessionState::default()` — no message
    //         carryover between context windows.
    //   - R3  the filesystem reload injects the reloaded `.spore/progress.json` +
    //         `.spore/feature_list.json` content into the fresh seed.
    //   - R4  completion pattern `incomplete,incomplete,complete` ⇒ `Success` at
    //         iteration 3.
    //   - R5  always-incomplete ⇒ exactly `max_resets` iterations ⇒
    //         `Failure { RalphCompletionUnmet }`.
    //   - R6  budgets fold across ALL context windows (`fold_usage`).
    //   - R7  each reset is traceable (a distinct generated session id per
    //         window, finalized via observability).
    //
    // No `// SPEC QUESTION:` markers — B1–B4 are resolved.
    async fn run_ralph(
        &self,
        task: Task,
        budget_used: BudgetSnapshot,
        _on_stream: Option<StreamSink>,
    ) -> RunResult {
        let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
        let max_resets = self.config.max_resets;
        // Ralph's incoming budget snapshot is irrelevant — each context window is
        // a fresh start with its own per-window turn budget (the reset discards
        // the turn budget along with the SessionState). Token/turn accounting is
        // accumulated separately for terminal reporting (R6).
        let _ = budget_used;

        // Cumulative usage + turns across ALL context windows (R6).
        let mut total_usage = AggregateUsage::default();
        let mut cumulative_turns: u32 = 0;
        // The most recent incompletion reason (for RalphCompletionUnmet).
        let mut last_reason = String::from(".spore/progress.json missing");
        // Session id of the most recent context window (terminal accounting).
        let mut last_session_id = task.session_id.clone();

        // The OUTER loop: each iteration is ONE context window. `max_resets`
        // caps the number of windows (B3). Iteration 0 is the first window; a
        // reset is the transition into the next iteration (R1).
        for iteration in 0..max_resets.max(1) {
            // R7: a fresh, distinct session id per context window so each reset
            // is independently traceable.
            let window_session_id = if iteration == 0 {
                task.session_id.clone()
            } else {
                SessionId::generate()
            };
            last_session_id = window_session_id.clone();

            // R2: a FRESH SessionState per window — discard the prior one. No
            // message carryover; the window is re-seeded from scratch.
            let mut session_state = SessionState::default();

            // Seed the instruction (R2) then R3: reload the deterministic
            // `.spore/` state from the filesystem and inject it as context so the
            // fresh window knows what is already done / still outstanding.
            self.config
                .context_manager
                .append_user_message(&mut session_state, &task.instruction)
                .await;
            if let Some(reload) = Self::ralph_reload_context(&workspace_root) {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &reload)
                    .await;
            }
            // R3 (issue #58 v2): when a `VcsProvider` is wired, ALSO reload git
            // history and inject it as a delimited "Recent VCS history:" section,
            // exactly as the `.spore/` reload content is injected. When the
            // provider is `None` (the default), this section is omitted entirely
            // — Ralph's reloaded context is then byte-for-byte the v1 behavior
            // (the B4→None decision).
            if let Some(vcs) = &self.config.vcs_provider {
                let args = VcsLogArgs {
                    max_entries: 20,
                    since_ref: None,
                    format: None,
                };
                if let Ok(log) = vcs.log(&args).await {
                    let trimmed = log.trim();
                    if !trimmed.is_empty() {
                        let block = format!("Recent VCS history:\n{trimmed}");
                        self.config
                            .context_manager
                            .append_user_message(&mut session_state, &block)
                            .await;
                    }
                }
            }

            // The per-window bounded ReAct sub-loop. The registered Stop hook
            // (B1) fires inside it on each `FinalResponse`: while incomplete it
            // blocks (capped by `max_stop_blocks`), forcing more work within the
            // window; this strategy's OUTER loop then decides reset vs success.
            let window_task = Task {
                id: task.id.clone(),
                instruction: task.instruction.clone(),
                session_id: window_session_id.clone(),
                budget: task.budget.clone(),
                loop_strategy: task.loop_strategy.clone(),
            };
            let window_cap = task.budget.max_turns.unwrap_or(u32::MAX);
            // FRESH per-window budget: the context-window reset resets the turn
            // budget too. Token fold is accumulated separately via `total_usage`.
            let mut carried = BudgetSnapshot::default();
            let window_result = self
                .run_react_inner(
                    window_task,
                    window_cap,
                    session_state,
                    carried.clone(),
                    None,
                )
                .await;
            Self::fold_usage(&mut total_usage, &mut carried, &window_result);
            cumulative_turns += carried.turns;

            // A window that paused / escalated is propagated up unchanged.
            match &window_result {
                RunResult::WaitingForHuman { state, request } => {
                    return RunResult::WaitingForHuman {
                        state: state.clone(),
                        request: request.clone(),
                    };
                }
                RunResult::Escalate { .. } => {
                    self.finalize_self_verifying(&window_result).await;
                    return window_result;
                }
                _ => {}
            }

            // External completion check (B1): consult the SAME filesystem state
            // the Stop hook reads. `None` ⇒ done ⇒ Success; `Some(reason)` ⇒
            // tasks remain ⇒ reset into the next window (R1) unless the cap is
            // reached (R5).
            match Self::ralph_completion_status(&workspace_root) {
                None => {
                    let output = match window_result {
                        RunResult::Success { output, .. } => output,
                        _ => String::new(),
                    };
                    let result = RunResult::Success {
                        output,
                        session_id: window_session_id,
                        usage: total_usage,
                        turns: cumulative_turns,
                    };
                    self.finalize_self_verifying(&result).await;
                    return result;
                }
                Some(reason) => {
                    last_reason = reason;
                    let _ = iteration;
                }
            }
        }

        // R5: ran out of context-window resets without completion.
        let result = RunResult::Failure {
            reason: HaltReason::RalphCompletionUnmet {
                iterations: max_resets.max(1),
                last_reason,
            },
            session_id: last_session_id,
            usage: total_usage,
            turns: cumulative_turns,
        };
        self.finalize_self_verifying(&result).await;
        result
    }

    /// Ralph external completion check (issue #58, B1). Reads the deterministic
    /// `.spore/` files under `workspace_root` and reports whether the task is
    /// complete. Returns `None` when complete, `Some(reason)` when tasks remain
    /// (the reason describes what is left). This is the SAME logic the registered
    /// [`RalphStopHook`] applies — one source of truth for the completion
    /// mechanism.
    ///
    /// Contract (both files are written by the strategy/initializer and fixture
    /// cleanly — B4, no git):
    ///   - `.spore/progress.json`: `{ "complete": bool, "remaining": [string] }`.
    ///     `complete: true` with an empty `remaining` ⇒ progress satisfied.
    ///     Missing/unreadable/invalid ⇒ incomplete (so the agent learns to write
    ///     it).
    ///   - `.spore/feature_list.json`: a JSON array of `{ "name", "passes" }`
    ///     (the [`FeatureListCheck`](crate::termination::FeatureListCheck) schema).
    ///     Any `passes: false` ⇒ incomplete. A MISSING feature list is tolerated
    ///     here (progress.json is the primary signal); an invalid one is not.
    fn ralph_completion_status(workspace_root: &std::path::Path) -> Option<String> {
        let progress_path = workspace_root.join(".spore/progress.json");
        let raw = match std::fs::read_to_string(&progress_path) {
            Ok(s) => s,
            Err(_) => return Some(".spore/progress.json missing".to_string()),
        };
        let progress: RalphProgress = match serde_json::from_str(&raw) {
            Ok(p) => p,
            Err(e) => return Some(format!(".spore/progress.json invalid JSON: {e}")),
        };
        if !progress.complete {
            let detail = if progress.remaining.is_empty() {
                "task not marked complete".to_string()
            } else {
                format!("remaining: {}", progress.remaining.join(", "))
            };
            return Some(detail);
        }
        if !progress.remaining.is_empty() {
            return Some(format!("remaining: {}", progress.remaining.join(", ")));
        }

        // Progress says done — corroborate against the feature list when present.
        let feature_path = workspace_root.join(".spore/feature_list.json");
        if let Ok(raw) = std::fs::read_to_string(&feature_path) {
            let entries: Vec<RalphFeatureEntry> = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => return Some(format!(".spore/feature_list.json invalid JSON: {e}")),
            };
            let incomplete: Vec<String> = entries
                .into_iter()
                .filter(|e| !e.passes)
                .map(|e| e.name)
                .collect();
            if !incomplete.is_empty() {
                return Some(format!("incomplete features: {}", incomplete.join(", ")));
            }
        }
        None
    }

    /// Build the filesystem-reload context block injected into each fresh
    /// context window (issue #58, R3). Returns the verbatim `.spore/progress.json`
    /// and `.spore/feature_list.json` contents (when present) so the re-seeded
    /// window knows what is already done and what remains. Returns `None` when
    /// neither file exists (nothing to reload).
    fn ralph_reload_context(workspace_root: &std::path::Path) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Ok(raw) = std::fs::read_to_string(workspace_root.join(".spore/progress.json")) {
            parts.push(format!("Reloaded .spore/progress.json:\n{}", raw.trim()));
        }
        if let Ok(raw) = std::fs::read_to_string(workspace_root.join(".spore/feature_list.json")) {
            parts.push(format!(
                "Reloaded .spore/feature_list.json:\n{}",
                raw.trim()
            ));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    // ========================================================================
    // HillClimbing loop strategy (issue #60)
    // ========================================================================
    //
    // Iterative optimization loop. Establishes a baseline metric over the
    // starting workspace, then repeatedly: runs ONE agent turn proposing a
    // change, evaluates the metric, and KEEPS or REVERTS based on whether the
    // new value strictly beats the running best (per `should_keep`). Generalizes
    // the autoresearch pattern.
    //
    // ## Loop semantics
    //   - Iteration 0 is a PURE baseline measurement of the starting workspace —
    //     NO agent turn. Its `ResultsEntry` has `status: kept`, and its
    //     `duration_secs` is the wall-clock time of the baseline evaluator call
    //     only. It seeds `current_best`. Agent turns begin at iteration 1.
    //   - Iterations 1.. : run one bounded ReAct agent turn (proposes a change),
    //     then evaluate. On a successful evaluation, route through
    //     `should_keep(new, current_best, payload_direction, min_improvement_delta)`:
    //       * keep   → status Kept, update `current_best`, reset the stagnation
    //                  counter to 0.
    //       * no keep → status Discarded; if `revert_on_no_improvement`, run
    //                  `git reset --hard HEAD` via the sandbox; increment the
    //                  stagnation counter.
    //     On a MetricError (Crashed/Timeout/etc.), the iteration status is
    //     `IterationStatus::from_error` (Crashed/Timeout), the metric value is
    //     EMPTY in the TSV, the iteration counts as a non-improvement (increment
    //     the stagnation counter), and the loop continues.
    //   - `max_stagnation: Some(N)` ⇒ after N consecutive non-improvements, halt
    //     `StagnationLimitReached { iterations, best_metric }`. Improvement resets
    //     the counter. `max_stagnation: None` ⇒ run until the budget/turn limit
    //     (no stagnation halt).
    //   - Budget gates (turns/tokens/wall/cost) are honored per iteration exactly
    //     as the other run_* methods do; each iteration emits a fire-and-forget
    //     observability span carrying the metric value/delta.
    //   - The harness writes the TSV after the run to
    //     `{workspace_root}/.spore/results/{task_id}.tsv`.
    //
    // ## Six pinned spec decisions (resolved with the user — NOT relitigated)
    //   1. REVERT: `revert_on_no_improvement: true` reverts via the sandbox
    //      `git reset --hard HEAD` directly from the harness. The harness NEVER
    //      git commits; `commit_hash` in the TSV stays empty unless a VcsProvider
    //      supplies it (default empty). Revert discards uncommitted working-tree
    //      changes back to current HEAD.
    //   2. TSV FLOAT FORMAT: both `metric_value` and `duration_secs` are
    //      formatted with exactly 6 decimal places (`format!("{:.6}", x)`) for
    //      cross-language byte-identity.
    //   3. TSV SCHEMA: written to `{workspace_root}/.spore/results/{task_id}.tsv`,
    //      tab-separated, REQUIRED header then one row per iteration ascending.
    //      Columns: iteration, commit_hash, metric_value, direction, status,
    //      duration_secs, description. `commit_hash` empty when no VCS wired;
    //      `metadata` EXCLUDED; `metric_value` EMPTY on crashed/timeout rows;
    //      `direction` snake_case; `status` snake_case; `duration_secs` is
    //      `MetricResult.duration.as_secs_f64()` formatted `{:.6}`.
    //   4. DIRECTION: the strategy payload `direction` is authoritative for the
    //      keep/revert decision via `should_keep`; the evaluator's `direction()`
    //      is descriptive only. The TSV `direction` column records the payload
    //      direction.
    //   5. BASELINE: iteration 0 is a pure baseline (no agent turn); its row has
    //      `status: kept` and `duration_secs` = the baseline evaluator-call time.
    //   6. MISCONFIGURATION: a `None` `metric_evaluator` ⇒ halt
    //      `HillClimbingMisconfigured { reason }` (peer to SelfVerifyMisconfigured).
    //      No panic.
    //
    // No `// SPEC QUESTION:` markers — all six decisions are resolved.
    #[allow(clippy::too_many_arguments)]
    async fn run_hill_climbing(
        &self,
        task: Task,
        direction: OptimizationDirection,
        max_stagnation: Option<u32>,
        revert_on_no_improvement: bool,
        min_improvement_delta: Option<f64>,
        budget_used: BudgetSnapshot,
        _on_stream: Option<StreamSink>,
    ) -> RunResult {
        use crate::metric::{should_keep, IterationStatus, MetricResult, ResultsEntry};
        use crate::termination::SessionStateSnapshot;

        let session_id = task.session_id.clone();
        let workspace_root = self.config.sandbox.workspace_root().to_path_buf();

        // Decision 6: a missing evaluator is a typed halt, not a panic.
        let evaluator = match self.config.metric_evaluator.as_ref() {
            Some(e) => e.clone(),
            None => {
                let result = RunResult::Failure {
                    reason: HaltReason::HillClimbingMisconfigured {
                        reason: "HillClimbing requires `config.metric_evaluator`, but it is None"
                            .to_string(),
                    },
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: 0,
                };
                self.finalize_self_verifying(&result).await;
                return result;
            }
        };

        let description = evaluator.description();
        // Per-iteration observability span counter.
        let mut span_seq: u64 = 0;
        // Cumulative usage + turns across ALL agent-turn iterations.
        let mut total_usage = AggregateUsage::default();
        // Shared budget threaded across every agent sub-run.
        let mut carried = budget_used;
        // The TSV rows, in iteration order.
        let mut rows: Vec<ResultsEntry> = Vec::new();

        // A snapshot for the evaluator. HillClimbing keeps no carried message
        // state of its own (each iteration is a fresh sub-run), so a default
        // SessionState is the right snapshot to hand the evaluator.
        let snapshot = SessionStateSnapshot::new(
            session_id.clone(),
            task.id.clone(),
            SessionState::default(),
            workspace_root.clone(),
        );

        // ── Iteration 0: pure baseline. No agent turn (Decision 5).
        let baseline = evaluator
            .evaluate(self.config.sandbox.as_ref(), &snapshot)
            .await;
        let mut current_best = match baseline {
            Ok(MetricResult {
                value, duration, ..
            }) => {
                rows.push(ResultsEntry {
                    iteration: 0,
                    commit_hash: self.hill_climbing_commit_hash().await,
                    metric_value: value,
                    direction,
                    status: IterationStatus::Kept,
                    duration,
                    description: description.clone(),
                    metadata: Default::default(),
                });
                self.emit_hill_climbing_span(
                    &session_id,
                    &task.id,
                    &mut span_seq,
                    0,
                    Some(value),
                    None,
                    IterationStatus::Kept,
                    false,
                );
                value
            }
            Err(err) => {
                // A baseline that cannot even be measured is a misconfiguration of
                // the experiment, not a non-improvement to climb away from — there
                // is no `current_best` to compare against. Record the failed row,
                // write the TSV, and halt.
                let status = IterationStatus::from_error(&err);
                rows.push(ResultsEntry {
                    iteration: 0,
                    commit_hash: self.hill_climbing_commit_hash().await,
                    // Sentinel; excluded from the TSV (crashed/timeout ⇒ empty).
                    metric_value: f64::NAN,
                    direction,
                    status,
                    duration: Duration::ZERO,
                    description: description.clone(),
                    metadata: Default::default(),
                });
                self.emit_hill_climbing_span(
                    &session_id,
                    &task.id,
                    &mut span_seq,
                    0,
                    None,
                    None,
                    status,
                    false,
                );
                self.write_hill_climbing_tsv(&workspace_root, &task.id, &rows)
                    .await;
                let result = RunResult::Failure {
                    reason: HaltReason::HillClimbingMisconfigured {
                        reason: format!("baseline evaluation failed: {err}"),
                    },
                    session_id,
                    usage: total_usage,
                    turns: carried.turns,
                };
                self.finalize_self_verifying(&result).await;
                return result;
            }
        };

        // Consecutive non-improvement counter (Decision-driven stagnation halt).
        let mut stagnation: u32 = 0;
        // The 0-based iteration index; agent turns begin at 1.
        let mut iteration: u32 = 1;

        loop {
            // Budget gate before the iteration's agent turn (mirrors run_react).
            let turn_cap = task.budget.max_turns.unwrap_or(u32::MAX);
            if carried.turns >= turn_cap {
                break;
            }
            if let Some(limit_type) = Self::budget_exceeded(&task.budget, &carried, Instant::now())
            {
                // A wall-time/cost/token cap reached BEFORE any iteration work is a
                // clean budget halt — but only surface it as a failure when no
                // iteration has produced an improvement to report. We mirror
                // run_react's contract: budget gates terminate with BudgetExceeded.
                self.write_hill_climbing_tsv(&workspace_root, &task.id, &rows)
                    .await;
                let result = RunResult::Failure {
                    reason: HaltReason::BudgetExceeded { limit_type },
                    session_id,
                    usage: total_usage,
                    turns: carried.turns,
                };
                self.finalize_self_verifying(&result).await;
                return result;
            }

            // ── One bounded agent turn proposes a change. The sub-run carries the
            //    shared budget so per-iteration turns count toward the cap.
            let iter_task = Task {
                id: task.id.clone(),
                instruction: task.instruction.clone(),
                session_id: session_id.clone(),
                budget: task.budget.clone(),
                loop_strategy: task.loop_strategy.clone(),
            };
            let mut iter_state = SessionState::default();
            self.config
                .context_manager
                .append_user_message(&mut iter_state, &task.instruction)
                .await;
            let turn_result = self
                .run_react_inner(iter_task, turn_cap, iter_state, carried.clone(), None)
                .await;
            Self::fold_usage(&mut total_usage, &mut carried, &turn_result);

            // A turn that paused / escalated is propagated up unchanged.
            match &turn_result {
                RunResult::WaitingForHuman { state, request } => {
                    return RunResult::WaitingForHuman {
                        state: state.clone(),
                        request: request.clone(),
                    };
                }
                RunResult::Escalate { .. } => {
                    self.write_hill_climbing_tsv(&workspace_root, &task.id, &rows)
                        .await;
                    self.finalize_self_verifying(&turn_result).await;
                    return turn_result;
                }
                _ => {}
            }

            // ── Evaluate the metric after the change.
            let eval = evaluator
                .evaluate(self.config.sandbox.as_ref(), &snapshot)
                .await;
            match eval {
                Ok(MetricResult {
                    value, duration, ..
                }) => {
                    let kept = should_keep(value, current_best, direction, min_improvement_delta);
                    let delta = match direction {
                        OptimizationDirection::Minimize => current_best - value,
                        OptimizationDirection::Maximize => value - current_best,
                    };
                    if kept {
                        current_best = value;
                        stagnation = 0;
                        rows.push(ResultsEntry {
                            iteration,
                            commit_hash: self.hill_climbing_commit_hash().await,
                            metric_value: value,
                            direction,
                            status: IterationStatus::Kept,
                            duration,
                            description: description.clone(),
                            metadata: Default::default(),
                        });
                        self.emit_hill_climbing_span(
                            &session_id,
                            &task.id,
                            &mut span_seq,
                            iteration,
                            Some(value),
                            Some(delta),
                            IterationStatus::Kept,
                            false,
                        );
                    } else {
                        // No improvement (Decision 1: optionally revert).
                        let reverted = if revert_on_no_improvement {
                            self.hill_climbing_revert().await;
                            true
                        } else {
                            false
                        };
                        stagnation += 1;
                        rows.push(ResultsEntry {
                            iteration,
                            commit_hash: self.hill_climbing_commit_hash().await,
                            metric_value: value,
                            direction,
                            status: IterationStatus::Discarded,
                            duration,
                            description: description.clone(),
                            metadata: Default::default(),
                        });
                        self.emit_hill_climbing_span(
                            &session_id,
                            &task.id,
                            &mut span_seq,
                            iteration,
                            Some(value),
                            Some(delta),
                            IterationStatus::Discarded,
                            reverted,
                        );
                    }
                }
                Err(err) => {
                    // Crash/timeout/etc.: counts as a non-improvement. Optionally
                    // revert, increment stagnation, record an empty-metric row.
                    let status = IterationStatus::from_error(&err);
                    let reverted = if revert_on_no_improvement {
                        self.hill_climbing_revert().await;
                        true
                    } else {
                        false
                    };
                    stagnation += 1;
                    rows.push(ResultsEntry {
                        iteration,
                        commit_hash: self.hill_climbing_commit_hash().await,
                        // Sentinel; excluded from the TSV (crashed/timeout ⇒ empty).
                        metric_value: f64::NAN,
                        direction,
                        status,
                        duration: Duration::ZERO,
                        description: description.clone(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        &session_id,
                        &task.id,
                        &mut span_seq,
                        iteration,
                        None,
                        None,
                        status,
                        reverted,
                    );
                }
            }

            // ── Stagnation halt (only when a cap is configured).
            if let Some(limit) = max_stagnation {
                if stagnation >= limit {
                    self.write_hill_climbing_tsv(&workspace_root, &task.id, &rows)
                        .await;
                    let result = RunResult::Failure {
                        reason: HaltReason::StagnationLimitReached {
                            iterations: stagnation,
                            best_metric: current_best,
                        },
                        session_id,
                        usage: total_usage,
                        turns: carried.turns,
                    };
                    self.finalize_self_verifying(&result).await;
                    return result;
                }
            }

            iteration = iteration.saturating_add(1);
        }

        // Budget/turn cap reached without a stagnation halt — clean budget halt.
        self.write_hill_climbing_tsv(&workspace_root, &task.id, &rows)
            .await;
        let result = RunResult::Failure {
            reason: HaltReason::BudgetExceeded {
                limit_type: BudgetLimitType::Turns,
            },
            session_id,
            usage: total_usage,
            turns: carried.turns,
        };
        self.finalize_self_verifying(&result).await;
        result
    }

    /// Revert the working tree to current HEAD for a no-improvement iteration
    /// (issue #60, Decision 1). Runs `git reset --hard HEAD` THROUGH the sandbox;
    /// the harness NEVER spawns git directly. A sandbox rejection / non-zero exit
    /// is best-effort: the loop continues (the next agent turn re-derives state).
    async fn hill_climbing_revert(&self) {
        let args = [
            "reset".to_string(),
            "--hard".to_string(),
            "HEAD".to_string(),
        ];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &args, None, None)
            .await;
    }

    /// Resolve the `commit_hash` recorded on a HillClimbing TSV row (issue #60,
    /// Decision 1). The harness never commits, so this is the EMPTY string unless
    /// a [`VcsProvider`] is wired to supply a hash. v1 has no per-keep commit, so
    /// we always return `None` (serialized as the empty string in the TSV); the
    /// `VcsProvider` seam is reserved for a later revision.
    async fn hill_climbing_commit_hash(&self) -> Option<String> {
        None
    }

    /// Emit one fire-and-forget per-iteration observability span for a
    /// HillClimbing run (issue #60). No-op when no provider is configured.
    #[allow(clippy::too_many_arguments)]
    fn emit_hill_climbing_span(
        &self,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
        iteration: u32,
        metric_value: Option<f64>,
        delta: Option<f64>,
        status: crate::metric::IterationStatus,
        reverted: bool,
    ) {
        if let Some(obs) = self.config.observability.as_ref() {
            let base = SpanBase::new_root(
                SpanId::new(format!("{}-hill-{}", session_id.as_str(), *span_seq)),
                session_id.clone(),
                task_id.clone(),
                SpanKind::Warn,
                Timestamp::now(),
            );
            let status_str = serde_json::to_value(status)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("{status:?}").to_lowercase());
            obs.emit_warn(WarnSpan::new(
                base,
                WarnEvent::HillClimbingIteration {
                    iteration,
                    metric_value,
                    delta,
                    status: status_str,
                    reverted,
                },
            ));
            *span_seq += 1;
        }
    }

    /// Serialize the HillClimbing results log and write it to
    /// `{workspace_root}/.spore/results/{task_id}.tsv` (issue #60, Decisions 2/3).
    /// Tab-separated, REQUIRED header, one row per iteration in ascending order.
    /// Floats use exactly 6 decimal places for cross-language byte-identity.
    /// `metric_value` is the empty string on crashed/timeout rows. `metadata` is
    /// excluded. Best-effort: a filesystem error is swallowed (the run outcome is
    /// authoritative, the TSV is a diagnostic artifact).
    async fn write_hill_climbing_tsv(
        &self,
        workspace_root: &std::path::Path,
        task_id: &TaskId,
        rows: &[crate::metric::ResultsEntry],
    ) {
        let body = Self::render_hill_climbing_tsv(rows);
        let dir = workspace_root.join(".spore/results");
        let _ = tokio::fs::create_dir_all(&dir).await;
        let path = dir.join(format!("{}.tsv", task_id.as_str()));
        let _ = tokio::fs::write(&path, body).await;
    }

    /// Render the HillClimbing results-log TSV body (issue #60, Decisions 2/3).
    /// Pure function over the rows so the exact byte content is unit-testable and
    /// cross-language-comparable. Trailing newline after every row (including the
    /// last) so appends and diffs stay line-oriented.
    fn render_hill_climbing_tsv(rows: &[crate::metric::ResultsEntry]) -> String {
        use crate::metric::IterationStatus;
        let mut out = String::from(
            "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n",
        );
        for r in rows {
            // Decision 3: metric_value is EMPTY on crashed/timeout rows.
            let metric_value = match r.status {
                IterationStatus::Crashed | IterationStatus::Timeout => String::new(),
                _ => format!("{:.6}", r.metric_value),
            };
            let commit_hash = r.commit_hash.clone().unwrap_or_default();
            let direction = match r.direction {
                OptimizationDirection::Minimize => "minimize",
                OptimizationDirection::Maximize => "maximize",
            };
            let status = match r.status {
                IterationStatus::Kept => "kept",
                IterationStatus::Discarded => "discarded",
                IterationStatus::Crashed => "crashed",
                IterationStatus::Timeout => "timeout",
            };
            let duration_secs = format!("{:.6}", r.duration.as_secs_f64());
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                r.iteration,
                commit_hash,
                metric_value,
                direction,
                status,
                duration_secs,
                r.description
            ));
        }
        out
    }

    // (private outcome type for `run_plan_phase` is defined at module scope.)

    /// Drive the PlanExecute strategy (issue #59) — the two-phase loop.
    ///
    /// # Phases
    /// 1. **Plan phase** (issue #70, runs EXACTLY ONCE):
    ///    [`run_plan_phase`](Self::run_plan_phase) seeds a planning directive,
    ///    runs one constrained planner turn, captures a
    ///    [`PlanArtifact`](crate::plan::PlanArtifact), fires `OnPlanCreated`,
    ///    and counts the turn against the shared budget.
    /// 2. **Execute phase** (issue #59, loops):
    ///    [`run_execute_phase`](Self::run_execute_phase) drains the task list,
    ///    giving each task its own bounded ReAct sub-loop.
    ///
    /// Between the phases the artifact is parsed into a
    /// [`TaskList`](crate::tasklist::TaskList) via
    /// [`plan_artifact_to_task_list`](crate::tasklist::plan_artifact_to_task_list)
    /// and persisted through the storage seam (Q4) and the `extras` mirror.
    ///
    /// # HaltReason variants
    /// - NEW [`HaltReason::EmptyPlan`] — the plan parsed to `tasks: []` (Q3).
    /// - NEW [`HaltReason::StepFailed`] — an execute step errored / blocked (Q5).
    /// - REMOVED `HaltReason::ExecutePhaseNotImplemented` — the execute phase is
    ///   now implemented; the old "plan produced, execute not implemented" halt
    ///   no longer exists.
    /// - Plan-phase failures still surface as
    ///   [`HaltReason::PlanPhaseFailed`] / [`HaltReason::AgentError`] /
    ///   [`HaltReason::BudgetExceeded`] unchanged.
    ///
    /// # Resolved spec decisions (issue #59 — all five FINAL)
    /// - **Q1 (execute step model):** each task gets its OWN bounded ReAct
    ///   sub-loop, fully isolated and SEQUENTIAL (task N completes before N+1).
    ///   The per-task turn cap is derived at the START of each step:
    ///   `per_task_turns = remaining_turns / remaining_tasks`, floored at 1
    ///   (integer division; `remaining_tasks` = not-yet-started tasks including
    ///   the current one). The shared/parent budget — turns, tokens,
    ///   observability spans, compaction — is carried across EVERY step exactly
    ///   as ReAct does, and the global budget is the hard stop.
    /// - **Q2 (success output):** `RunResult::Success.output` is the LAST
    ///   completed step's `FinalResponse` text — not a concatenation, not the
    ///   plan rationale. Full per-step history stays in session state / traces.
    /// - **Q3 (empty plan):** an empty task list ⇒ [`HaltReason::EmptyPlan`].
    /// - **Q4 (persistence):** the task list and plan are persisted through the
    ///   [`StorageProvider`](crate::storage::StorageProvider) /
    ///   [`RunStore`](crate::storage::RunStore) seam ONLY. The #71
    ///   sandbox-filesystem path (`.spore/task_list.json`) is NOT used by the
    ///   execute loop — one source of truth. The `extras` mirror is kept.
    /// - **Q5 (per-task failure):** a step's ReAct sub-loop erroring or returning
    ///   a blocked/failed outcome ABORTS the whole run with
    ///   [`HaltReason::StepFailed`]; execution does NOT continue to the next
    ///   task (a plan is a dependency chain by assumption).
    ///
    /// Like `run_react`, this finalizes observability for the terminal outcome.
    async fn run_plan_execute(
        &self,
        task: Task,
        mut session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let session_id = task.session_id.clone();

        // ── Phase 1: plan (runs exactly once) ──────────────────────────────
        let outcome = match self
            .run_plan_phase(&task, &mut session_state, budget_used.clone(), &on_stream)
            .await
        {
            Ok(outcome) => outcome,
            // Plan-phase failure: propagate unchanged (no task list persisted).
            Err(failure) => {
                self.finalize_plan_execute(&failure).await;
                return failure;
            }
        };

        // Bridge: parse the accepted plan into a TaskList (#72).
        let task_list = crate::tasklist::plan_artifact_to_task_list(&outcome.artifact);

        // Q3: an empty plan is a failure, not a silent success.
        if task_list.tasks.is_empty() {
            let result = RunResult::Failure {
                reason: HaltReason::EmptyPlan,
                session_id,
                usage: outcome.usage,
                turns: outcome.turns,
            };
            self.finalize_plan_execute(&result).await;
            return result;
        }

        // Q4: persist the task list through the storage seam (RunStore) ONLY.
        // The #71 sandbox path is intentionally unused.
        self.persist_task_list(&session_id, &task_list).await;

        // Carry the shared budget forward: the plan turn already consumed
        // `outcome.turns` turns and `outcome.usage` tokens (Q1 — shared budget).
        let mut carried = budget_used;
        carried.turns = outcome.turns;
        carried.input_tokens += outcome.usage.input_tokens;
        carried.output_tokens += outcome.usage.output_tokens;

        // ── Phase 2: execute (loops over the task list) ────────────────────
        let result = self
            .run_execute_phase(
                &task,
                &mut session_state,
                task_list,
                carried,
                outcome.usage,
                &on_stream,
            )
            .await;
        self.finalize_plan_execute(&result).await;
        result
    }

    /// Finalize observability for a terminal PlanExecute outcome. Mirrors the
    /// tail of [`run_react`](Self::run_react): `WaitingForHuman` is not terminal
    /// and is never flushed here.
    async fn finalize_plan_execute(&self, result: &RunResult) {
        match result {
            RunResult::Success { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Success)
                    .await;
            }
            RunResult::Failure {
                session_id, reason, ..
            } => {
                self.finalize_observability(
                    session_id,
                    SessionOutcome::Failure {
                        reason: format!("{reason:?}"),
                    },
                )
                .await;
            }
            RunResult::WaitingForHuman { .. } => {}
            // Escalation (issue #80) is a terminal outcome here too.
            RunResult::Escalate { session_id, .. } => {
                self.finalize_observability(session_id, SessionOutcome::Escalated)
                    .await;
            }
        }
    }

    /// Persist the parsed [`TaskList`](crate::tasklist::TaskList) for the run
    /// (Q4). The DURABLE write goes through the
    /// [`RunStore`](crate::storage::RunStore) seam under
    /// [`TASK_LIST_EXTRAS_KEY`](crate::tasklist::TASK_LIST_EXTRAS_KEY); the #71
    /// sandbox-filesystem path (`.spore/task_list.json`) is intentionally NOT
    /// used — one source of truth. The RunStore write is the single source of
    /// truth (#76 removed the redundant `SessionState.extras` mirror).
    /// Serialization failures are swallowed: a successful plan must not be lost
    /// to a storage hiccup (the default no-op/in-memory provider never fails).
    async fn persist_task_list(
        &self,
        session_id: &SessionId,
        task_list: &crate::tasklist::TaskList,
    ) {
        if let Ok(value) = serde_json::to_value(task_list) {
            // Durable write through the storage seam (RunStore).
            let _ = self
                .config
                .storage
                .run()
                .put(session_id, crate::tasklist::TASK_LIST_EXTRAS_KEY, value)
                .await;
        }
    }

    /// Drive the PlanExecute execute phase (issue #59), draining `task_list`.
    ///
    /// Per Q1 each task gets its own bounded, fully-isolated, SEQUENTIAL ReAct
    /// sub-loop. The per-task turn cap is derived at the START of each step from
    /// the shared budget: `per_task_turns = remaining_turns / remaining_tasks`,
    /// floored at 1 (integer division; `remaining_tasks` counts the not-yet-
    /// started tasks including the current one). The shared budget snapshot
    /// (`carried`) is threaded through every step so early tasks cannot starve
    /// later ones and the global budget stays the hard stop.
    ///
    /// Before each step the task is marked `InProgress` (and `Completed` after),
    /// the list is re-persisted (Q4), and `OnTaskAdvance` fires with the correct
    /// `task_index` / `total_tasks` (the hook may rewrite the step instruction).
    ///
    /// Q2: on success `output` is the LAST completed step's `FinalResponse`.
    /// Q5: a step that errors / blocks aborts the run with
    /// [`HaltReason::StepFailed`] — no further tasks run.
    ///
    /// `plan_usage` seeds the cumulative [`AggregateUsage`] with the plan turn's
    /// usage so the terminal `RunResult` reflects the WHOLE run.
    async fn run_execute_phase(
        &self,
        task: &Task,
        session_state: &mut SessionState,
        mut task_list: crate::tasklist::TaskList,
        mut carried: BudgetSnapshot,
        plan_usage: AggregateUsage,
        on_stream: &Option<StreamSink>,
    ) -> RunResult {
        use crate::tasklist::TaskStatus;

        let session_id = task.session_id.clone();
        let total_tasks = task_list.tasks.len();
        // Cumulative usage across the plan turn + every execute step (Q1).
        let mut total_usage = plan_usage;
        // Q2: the success handle is the LAST completed step's final text.
        let mut last_output = String::new();
        // Global turn cap (the hard stop). `None` ⇒ no global turn ceiling.
        let global_max_turns = task.budget.max_turns;

        for index in 0..total_tasks {
            let task_id = task_list.tasks[index].id;
            let instruction = task_list.tasks[index].description.clone();

            // Q1: per-task turn allocation, derived at the START of this step.
            // remaining_tasks = not-yet-started tasks including this one.
            let remaining_tasks = (total_tasks - index) as u32;
            let per_task_turns = match global_max_turns {
                Some(max) => {
                    let remaining_turns = max.saturating_sub(carried.turns);
                    (remaining_turns / remaining_tasks).max(1)
                }
                // No global turn cap: each step's sub-loop is bounded only by
                // the other (token / wall / cost) budget gates.
                None => u32::MAX,
            };
            // The sub-loop's effective cap is RELATIVE to the carried turns:
            // run_react_inner gates on the cumulative `budget_used.turns`, so a
            // per-task cap of K means "stop K turns from now" while the global
            // budget (carried forward) remains the hard stop.
            let sub_loop_cap = carried.turns.saturating_add(per_task_turns);

            // Mark InProgress (Pending -> InProgress) and re-persist (Q4).
            let _ = task_list.update(task_id, Some(TaskStatus::InProgress), None);
            self.persist_task_list(&session_id, &task_list).await;

            // Fire OnTaskAdvance (pre, mutable). The hook may rewrite the step's
            // instruction via the carried harness::Task; the (possibly mutated)
            // instruction seeds the sub-loop.
            let mut step_task = Task {
                id: task.id.clone(),
                instruction,
                session_id: session_id.clone(),
                budget: task.budget.clone(),
                loop_strategy: task.loop_strategy.clone(),
            };
            if let Some(chain) = self.config.hooks.as_ref() {
                let mut ctx = crate::hooks::HookContext::OnTaskAdvance {
                    session_id: &session_id,
                    task: &mut step_task,
                    task_index: index,
                    total_tasks,
                };
                let _ = chain.fire(&mut ctx).await;
            }

            // Seed the step instruction as a user message, then run the bounded
            // ReAct sub-loop carrying the shared budget (Q1).
            self.config
                .context_manager
                .append_user_message(session_state, &step_task.instruction)
                .await;

            let sub_result = self
                .run_react_inner(
                    step_task,
                    sub_loop_cap,
                    session_state.clone(),
                    carried.clone(),
                    None,
                )
                .await;

            match sub_result {
                RunResult::Success {
                    output,
                    usage,
                    turns,
                    ..
                } => {
                    // Carry the shared budget forward (Q1): cumulative turns are
                    // the sub-loop's absolute count; fold in its token usage.
                    carried.turns = turns;
                    carried.input_tokens += usage.input_tokens;
                    carried.output_tokens += usage.output_tokens;
                    total_usage.input_tokens += usage.input_tokens;
                    total_usage.output_tokens += usage.output_tokens;
                    total_usage.cache_read_tokens += usage.cache_read_tokens;
                    total_usage.cache_write_tokens += usage.cache_write_tokens;
                    total_usage.cost_usd += usage.cost_usd;
                    last_output = output;

                    // Mark Completed and re-persist (Q4).
                    let _ = task_list.complete(task_id);
                    self.persist_task_list(&session_id, &task_list).await;
                    // Surface the completed step's final text to the caller's
                    // sink. The per-step sub-loop runs with its own (suppressed)
                    // sink, so this is the parent-visible step boundary.
                    Self::emit(
                        on_stream,
                        StreamEvent::FinalResponse {
                            content: last_output.clone(),
                        },
                    );
                }
                // Q5: any non-success step aborts the whole run. A budget halt
                // surfaces as BudgetExceeded (mid-execute exhaustion); other
                // failures surface as StepFailed carrying the step context.
                RunResult::Failure {
                    reason,
                    usage,
                    turns,
                    ..
                } => {
                    total_usage.input_tokens += usage.input_tokens;
                    total_usage.output_tokens += usage.output_tokens;
                    total_usage.cache_read_tokens += usage.cache_read_tokens;
                    total_usage.cache_write_tokens += usage.cache_write_tokens;
                    total_usage.cost_usd += usage.cost_usd;

                    let _ = task_list.update(task_id, Some(TaskStatus::Blocked), None);
                    self.persist_task_list(&session_id, &task_list).await;

                    let terminal_reason = match reason {
                        // Budget exhaustion mid-execute is its own halt — keep
                        // it distinct from a step's own failure.
                        HaltReason::BudgetExceeded { .. } => reason,
                        other => HaltReason::StepFailed {
                            task_index: index,
                            task: task_list.tasks[index].description.clone(),
                            reason: format!("{other:?}"),
                        },
                    };
                    return RunResult::Failure {
                        reason: terminal_reason,
                        session_id,
                        usage: total_usage,
                        turns,
                    };
                }
                // A step surfacing to a human pauses the whole run; propagate.
                RunResult::WaitingForHuman { state, request } => {
                    return RunResult::WaitingForHuman { state, request };
                }
                // A step escalating (issue #80) terminates the whole run
                // cleanly; propagate the signal and preserved state up.
                RunResult::Escalate {
                    signal,
                    state,
                    session_id,
                    usage,
                    turns,
                } => {
                    return RunResult::Escalate {
                        signal,
                        state,
                        session_id,
                        usage,
                        turns,
                    };
                }
            }
        }

        // Q2: success output is the LAST completed step's FinalResponse text.
        RunResult::Success {
            output: last_output,
            session_id,
            usage: total_usage,
            turns: carried.turns,
        }
    }

    /// Run the one-shot PlanExecute plan phase (issue #70).
    ///
    /// Selects the planner agent (Q1: `config.planner_agent` if set, else the
    /// default agent), seeds a planning directive as a user message, runs
    /// EXACTLY ONE constrained turn (R1), expects a `FinalResponse` (a tool call
    /// is a planning failure — R2 — never a dispatch loop), captures the
    /// response via [`capture_plan_artifact`](crate::plan::capture_plan_artifact)
    /// (R3), fires `OnPlanCreated` (which may rewrite the artifact — R11),
    /// stores the result in `extras["plan_execute"]` (R4), emits the turn span
    /// (R8), and counts the turn against the shared budget (R7). A budget
    /// exhausted before the turn returns a budget-exceeded `Failure` with no
    /// artifact stored (R10).
    ///
    /// On success returns the produced [`PlanArtifact`](crate::plan::PlanArtifact)
    /// plus the run accounting. On any failure returns the terminal
    /// `RunResult::Failure` to propagate.
    async fn run_plan_phase(
        &self,
        task: &Task,
        session_state: &mut SessionState,
        mut budget_used: BudgetSnapshot,
        on_stream: &Option<StreamSink>,
    ) -> Result<PlanPhaseOutcome, RunResult> {
        let session_id = task.session_id.clone();
        let started_at = Instant::now();
        let mut usage = AggregateUsage::default();

        // R10: Layer-1 budget gate BEFORE the plan turn. Mirrors run_react_inner.
        let effective_turn_cap = task.budget.max_turns.unwrap_or(u32::MAX).max(1);
        if budget_used.turns >= effective_turn_cap {
            return Err(RunResult::Failure {
                reason: HaltReason::BudgetExceeded {
                    limit_type: BudgetLimitType::Turns,
                },
                session_id,
                usage,
                turns: budget_used.turns,
            });
        }
        if let Some(limit_type) = Self::budget_exceeded(&task.budget, &budget_used, started_at) {
            return Err(RunResult::Failure {
                reason: HaltReason::BudgetExceeded { limit_type },
                session_id,
                usage,
                turns: budget_used.turns,
            });
        }

        // Q1: select the planner agent (alternate if configured, else default).
        let planner = self
            .config
            .planner_agent
            .as_ref()
            .unwrap_or(&self.config.agent);

        // Seed the planning directive as a user message (reuse ContextManager).
        let directive = format!(
            "Produce a step-by-step plan for the following task. Respond with a \
             single JSON object: {{\"tasks\": [<ordered step strings>], \
             \"rationale\": <string>}}.\n\nTask:\n{}",
            task.instruction
        );
        self.config
            .context_manager
            .append_user_message(session_state, &directive)
            .await;

        // Assemble + invoke the planner for exactly ONE turn (R1).
        let context = self
            .config
            .context_manager
            .assemble(session_state, task)
            .await;
        Self::emit(
            on_stream,
            StreamEvent::TurnStart {
                turn: budget_used.turns + 1,
            },
        );
        let turn_started_at = Timestamp::now();
        let turn_clock = Instant::now();
        let result = planner.turn(context).await;
        budget_used.turns += 1; // R7: the plan turn counts against the budget.

        // R8: emit exactly one TurnSpan for the plan turn. Mirrors the metrics
        // path of run_react_inner; content capture intentionally omitted (the
        // plan turn carries no tool calls and #64 content capture is wired in
        // the ReAct loop only).
        {
            let zero = TokenUsage::default();
            let u = match &result {
                TurnResult::ToolCallRequested { usage, .. }
                | TurnResult::FinalResponse { usage, .. } => usage,
                TurnResult::Error { usage, .. } => usage.as_ref().unwrap_or(&zero),
            };
            let (stop_reason, tool_calls_requested) = match &result {
                TurnResult::FinalResponse { .. } => (StopReason::EndTurn, 0),
                TurnResult::ToolCallRequested { calls, .. } => {
                    (StopReason::ToolUse, calls.len() as u32)
                }
                TurnResult::Error { .. } => (StopReason::EndTurn, 0),
            };
            let mut base = SpanBase::new_root(
                SpanId::new(format!(
                    "{}-turn-{}",
                    session_id.as_str(),
                    budget_used.turns
                )),
                session_id.clone(),
                task.id.clone(),
                SpanKind::Turn,
                turn_started_at,
            );
            let status = match &result {
                TurnResult::Error { error, .. } => SpanStatus::Error {
                    message: format!("{error:?}"),
                },
                _ => SpanStatus::Ok,
            };
            base.finish(
                Timestamp::now(),
                status,
                turn_clock.elapsed().as_millis() as u64,
            );
            if let Some(obs) = self.config.observability.as_ref() {
                obs.emit_turn(TurnSpan {
                    base,
                    turn_number: budget_used.turns,
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    cache_read_tokens: u.cache_read_tokens,
                    cache_write_tokens: u.cache_write_tokens,
                    cost_usd: self.config.pricing.cost_for(
                        u.input_tokens,
                        u.output_tokens,
                        u.cache_read_tokens,
                        u.cache_write_tokens,
                    ),
                    stop_reason,
                    tool_calls_requested,
                    output_text: None,
                    tool_calls: None,
                    input_messages: None,
                });
            }
        }
        Self::emit(
            on_stream,
            StreamEvent::TurnEnd {
                turn: budget_used.turns,
            },
        );

        // Classify the one-shot turn. R2: a tool call is a planning failure,
        // NOT a dispatch loop.
        let final_text = match result {
            TurnResult::FinalResponse { content, usage: u } => {
                usage.add_turn(&u);
                budget_used.input_tokens += u.input_tokens as u64;
                budget_used.output_tokens += u.output_tokens as u64;
                content
            }
            TurnResult::ToolCallRequested { usage: u, .. } => {
                usage.add_turn(&u);
                return Err(RunResult::Failure {
                    reason: HaltReason::PlanPhaseFailed {
                        error: crate::plan::PlanPhaseError::PlanningTurnFailed {
                            message: "planner requested a tool call in the one-shot plan turn"
                                .into(),
                        },
                    },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                });
            }
            TurnResult::Error { error, usage: u } => {
                if let Some(u) = u.as_ref() {
                    usage.add_turn(u);
                }
                return Err(RunResult::Failure {
                    reason: HaltReason::AgentError { error },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                });
            }
        };

        // R3: capture the artifact from the response text.
        let mut artifact = match crate::plan::capture_plan_artifact(&final_text) {
            Ok(a) => a,
            Err(e) => {
                return Err(RunResult::Failure {
                    reason: HaltReason::PlanPhaseFailed { error: e },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                });
            }
        };

        // R11: fire OnPlanCreated synchronously; the hook may rewrite `artifact`
        // in place. The stored artifact reflects any mutation.
        if let Some(chain) = self.config.hooks.as_ref() {
            let mut ctx = crate::hooks::HookContext::OnPlanCreated {
                session_id: &session_id,
                plan: &mut artifact,
            };
            // Errors are non-fatal: an observability/handler error must not lose
            // a successfully-captured plan. The (possibly mutated) artifact is
            // still stored.
            let _ = chain.fire(&mut ctx).await;
        }

        // R4: persist the produced artifact to the RunStore seam under
        // PLAN_EXECUTE_EXTRAS_KEY (#76 — the durable single source of truth;
        // no longer mirrored into SessionState.extras). The put result is
        // swallowed (matching the execute-phase persist): a successfully
        // captured plan must not be lost to a storage hiccup.
        match serde_json::to_value(&artifact) {
            Ok(value) => {
                let _ = self
                    .config
                    .storage
                    .run()
                    .put(&session_id, crate::plan::PLAN_EXECUTE_EXTRAS_KEY, value)
                    .await;
            }
            Err(e) => {
                return Err(RunResult::Failure {
                    reason: HaltReason::PlanPhaseFailed {
                        error: crate::plan::PlanPhaseError::UnparseablePlan {
                            message: format!("failed to serialize plan artifact: {e}"),
                        },
                    },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                });
            }
        }

        Ok(PlanPhaseOutcome {
            artifact,
            usage,
            turns: budget_used.turns,
        })
    }

    /// Run the post-compaction verify→retry→warn loop (issue #46/#29).
    ///
    /// Drives one compaction turn through the agent, verifies the summary,
    /// and either accepts it, retries with the missing items injected, or —
    /// after `max_compaction_attempts` — emits a warn event and accepts the
    /// summary anyway. A blocked compaction is worse than an imperfect one, so
    /// this method NEVER returns an error or halts the run; the worst case is
    /// an accepted-anyway summary plus one warn span.
    ///
    /// Token usage from compaction turns folds into the run-level
    /// [`AggregateUsage`]; each compaction turn that produces a summary is
    /// surfaced as a `Compaction` [`ContextSpan`]. The
    /// `compaction_verification_failures` metric is derived from the emitted
    /// [`WarnSpan`].
    async fn run_compaction(
        &self,
        session_state: &mut SessionState,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
        usage: &mut AggregateUsage,
    ) {
        let Some(mut turn) = self
            .config
            .context_manager
            .prepare_compaction_turn(session_state)
        else {
            // Nothing to compact (e.g. history shorter than preserve window).
            return;
        };
        let tokens_before = turn.verification_state.token_budget_used;
        let max_attempts = self.config.max_compaction_attempts.max(1);
        let mut attempt: u32 = 0;

        loop {
            attempt += 1;
            // Run one compaction turn through the agent to produce a summary.
            let result = self.config.agent.turn(turn.context.clone()).await;
            let summary = match result {
                TurnResult::FinalResponse { content, usage: u } => {
                    usage.add_turn(&u);
                    content
                }
                TurnResult::ToolCallRequested { usage: u, .. } => {
                    // A compaction turn is expected to yield a summary, not a
                    // tool call. Treat the (empty) response as the summary so
                    // verification can run and the loop terminates predictably.
                    usage.add_turn(&u);
                    String::new()
                }
                TurnResult::Error { usage: u, .. } => {
                    if let Some(u) = u.as_ref() {
                        usage.add_turn(u);
                    }
                    String::new()
                }
            };

            let verification = self.config.compaction_verifier.verify(
                &summary,
                &turn.preserve_hints,
                &turn.verification_state,
            );

            if verification.passed {
                self.accept_compaction(
                    session_state,
                    summary,
                    turn.messages_removed,
                    tokens_before,
                    session_id,
                    task_id,
                    span_seq,
                );
                return;
            }

            if attempt < max_attempts {
                // Inject the missing items and retry.
                self.config
                    .context_manager
                    .inject_missing_items(&mut turn.context, &verification.missing_items);
                continue;
            }

            // Exhausted attempts: warn, then accept anyway.
            if let Some(obs) = self.config.observability.as_ref() {
                let base = SpanBase::new_root(
                    SpanId::new(format!("{}-warn-{}", session_id.as_str(), *span_seq)),
                    session_id.clone(),
                    task_id.clone(),
                    SpanKind::Warn,
                    Timestamp::now(),
                );
                obs.emit_warn(WarnSpan::new(
                    base,
                    WarnEvent::CompactionVerificationFailed {
                        missing_items: verification.missing_items.clone(),
                        accepted_anyway: true,
                    },
                ));
                *span_seq += 1;
            }
            self.accept_compaction(
                session_state,
                summary,
                turn.messages_removed,
                tokens_before,
                session_id,
                task_id,
                span_seq,
            );
            return;
        }
    }

    /// Apply an accepted summary and emit the `Compaction` context span.
    #[allow(clippy::too_many_arguments)]
    fn accept_compaction(
        &self,
        session_state: &mut SessionState,
        summary: String,
        messages_removed: u32,
        tokens_before: u32,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
    ) {
        self.config
            .context_manager
            .apply_compaction(session_state, summary);

        // Real token accounting (issue #57): read the post-compaction budget the
        // manager tracks. Managers that do not track tokens report `None`, in
        // which case we fall back to the pre-compaction value (the old behavior).
        let tokens_after = self
            .config
            .context_manager
            .token_budget_used(session_state)
            .unwrap_or(tokens_before);
        let tokens_reclaimed = tokens_before.saturating_sub(tokens_after);

        if let Some(obs) = self.config.observability.as_ref() {
            let base = SpanBase::new_root(
                SpanId::new(format!("{}-compaction-{}", session_id.as_str(), *span_seq)),
                session_id.clone(),
                task_id.clone(),
                SpanKind::Compaction,
                Timestamp::now(),
            );
            obs.emit_context(crate::observability::ContextSpan {
                base,
                operation: crate::observability::ContextOperation::Compaction {
                    messages_removed,
                    tokens_reclaimed,
                },
                tokens_before,
                tokens_after,
                utilization_before: 0.0,
                utilization_after: 0.0,
            });
            *span_seq += 1;
        }
    }
}

impl Harness for StandardHarness {
    fn run<'a>(&'a self, options: HarnessRunOptions) -> BoxFut<'a, RunResult> {
        Box::pin(self.run_inner(options))
    }

    fn resume<'a>(
        &'a self,
        state: PausedState,
        response: HumanResponse,
        on_stream: Option<StreamSink>,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(self.resume_inner(state, response, on_stream))
    }
}

impl StandardHarness {
    async fn run_inner(&self, options: HarnessRunOptions) -> RunResult {
        let HarnessRunOptions {
            task,
            on_stream,
            session_state,
        } = options;
        let mut session_state = session_state.unwrap_or_default();
        let budget_used = BudgetSnapshot::default();

        match task.loop_strategy.clone() {
            LoopStrategy::ReAct { max_iterations } => {
                // Seed the task instruction as the initial user message of this
                // FRESH run only. The compaction adapter intentionally mirrors
                // `session.messages` and ignores `task` on `assemble`, so the
                // harness must own delivering the prompt. On a fresh run this
                // turns an otherwise-empty conversation into a real user turn;
                // on multi-turn runs over a carried `session_state` each `run()`
                // call appends its own follow-up instruction. This lives on the
                // `run()` entry — NOT in the shared `run_react_inner` — so that
                // `resume_inner` (which also calls `run_react`) does not
                // re-append the instruction after the human's response.
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &task.instruction)
                    .await;
                self.run_react(task, max_iterations, session_state, budget_used, on_stream)
                    .await
            }
            LoopStrategy::PlanExecute { .. } => {
                self.run_plan_execute(task, session_state, budget_used, on_stream)
                    .await
            }
            LoopStrategy::Ralph => {
                // Ralph (issue #58) re-seeds a FRESH SessionState per context
                // window INSIDE `run_ralph` (the context-window reset). The
                // incoming `session_state` is intentionally discarded — the first
                // window is built from scratch like every subsequent reset, so
                // the prompt is NOT seeded here.
                let _ = session_state;
                self.run_ralph(task, budget_used, on_stream).await
            }
            LoopStrategy::SelfVerifying => {
                // Seed the build task instruction as the initial user message of
                // this fresh run (mirrors the ReAct entry above): the build
                // sub-loop reuses `run_react_inner`, which does NOT itself seed
                // the prompt.
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &task.instruction)
                    .await;
                self.run_self_verifying(task, session_state, budget_used, on_stream)
                    .await
            }
            LoopStrategy::HillClimbing {
                direction,
                max_stagnation,
                revert_on_no_improvement,
                min_improvement_delta,
            } => {
                // HillClimbing (issue #60) re-seeds a FRESH SessionState per
                // agent-turn iteration INSIDE `run_hill_climbing` — the iteration-0
                // baseline runs NO agent turn at all, and every subsequent
                // iteration is its own bounded sub-run. The incoming
                // `session_state` is intentionally discarded; the prompt is NOT
                // seeded here.
                let _ = session_state;
                self.run_hill_climbing(
                    task,
                    direction,
                    max_stagnation,
                    revert_on_no_improvement,
                    min_improvement_delta,
                    budget_used,
                    on_stream,
                )
                .await
            }
        }
    }

    async fn resume_inner(
        &self,
        state: PausedState,
        response: HumanResponse,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let PausedState {
            session_id,
            task_id: _,
            turn_number,
            mut session_state,
            pending_tool_calls,
            approved_results: _approved_results,
            human_request: hr,
            task,
            budget_used,
            child_state,
        } = state;

        // Clarification resume (issue #81, Q4b): if this pause came from
        // `ToolOutput::AwaitingClarification`, the human's answer is injected as
        // the tool RESULT for the clarifying call (the head of
        // `pending_tool_calls`) — NOT appended as a free-standing user message.
        // Any remaining pending calls after the clarifying one are then
        // dispatched normally before the loop resumes.
        if matches!(hr, Some(HumanRequest::Clarification { .. })) {
            if let HumanResponse::Answer { text }
            | HumanResponse::ApproveWithFeedback { feedback: text } = &response
            {
                let mut pending = pending_tool_calls.into_iter();
                if let Some(clarifying_call) = pending.next() {
                    let tr = ToolResult {
                        call_id: clarifying_call.id.clone(),
                        output: ToolOutput::Success {
                            content: text.clone(),
                            truncated: false,
                        },
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
                // Dispatch any remaining pending calls from the same batch.
                for call in pending {
                    let output = self.config.tool_registry.dispatch(call.clone()).await;
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
                let max_iterations = match task.loop_strategy {
                    LoopStrategy::ReAct { max_iterations } => max_iterations,
                    _ => u32::MAX,
                };
                return self
                    .run_react(task, max_iterations, session_state, budget_used, on_stream)
                    .await;
            }
        }

        // Subagent depth: if there's a child, route the response through it.
        // The actual child harness is owned by the caller-installed
        // `SubagentTool`; without a tool registry hook for `resume_child`,
        // the harness surfaces a tool result indicating the child's outcome
        // is the caller's responsibility. Wiring lands with #4/#5.
        if child_state.is_some() {
            // Append a synthetic tool result and continue the parent loop.
            // The full child.resume() dispatch lives with SubagentTool (#5).
        }

        match response {
            HumanResponse::Halt => {
                return RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: turn_number,
                };
            }
            HumanResponse::Deny { reason } => {
                // Append failed tool results for each pending call, continue loop.
                for call in &pending_tool_calls {
                    let tr = ToolResult {
                        call_id: call.id.clone(),
                        output: ToolOutput::Error {
                            message: reason.clone(),
                            recoverable: true,
                        },
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
            }
            HumanResponse::Reject { reason } => {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &reason)
                    .await;
            }
            HumanResponse::Answer { text }
            | HumanResponse::ApproveWithFeedback { feedback: text } => {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &text)
                    .await;
            }
            HumanResponse::Allow => {
                // Dispatch remaining pending calls before resuming the loop.
                for call in pending_tool_calls {
                    let output = self.config.tool_registry.dispatch(call.clone()).await;
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
            }
            HumanResponse::AllowWithModification { calls } => {
                for call in calls {
                    let output = self.config.tool_registry.dispatch(call.clone()).await;
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
            }
        }

        // Resume the ReAct loop from where we left off.
        let max_iterations = match task.loop_strategy {
            LoopStrategy::ReAct { max_iterations } => max_iterations,
            _ => u32::MAX,
        };
        self.run_react(task, max_iterations, session_state, budget_used, on_stream)
            .await
    }
}

// ============================================================================
// Test-only stubs of the sibling traits (cover ReAct paths in the unit tests
// before the canonical impls land in #4–#13).
// ============================================================================

#[cfg(any(test, feature = "test-utils"))]
pub mod testing {
    use super::*;
    use std::sync::Mutex;

    pub struct NoopContextManager;
    impl ContextManager for NoopContextManager {
        fn assemble<'a>(
            &'a self,
            session: &'a SessionState,
            _task: &'a Task,
        ) -> BoxFut<'a, Context> {
            let messages = session.messages.clone();
            Box::pin(async move {
                Context {
                    messages,
                    tools: vec![],
                    params: crate::model::ModelParams::default(),
                }
            })
        }
        fn append_tool_result<'a>(
            &'a self,
            session: &'a mut SessionState,
            result: &'a ToolResult,
        ) -> BoxFut<'a, ()> {
            Box::pin(async move {
                let text = match &result.output {
                    ToolOutput::Success { content, .. } => content.clone(),
                    ToolOutput::Error { message, .. } => format!("[error] {message}"),
                    ToolOutput::WaitingForHuman { .. } => "[waiting]".into(),
                    ToolOutput::Escalate { .. } => "[escalate]".into(),
                    ToolOutput::AwaitingClarification { .. } => "[clarification]".into(),
                };
                session.messages.push(Message {
                    role: crate::model::Role::Tool,
                    content: crate::model::Content::Text { text },
                });
            })
        }
        fn append_assistant_message<'a>(
            &'a self,
            session: &'a mut SessionState,
            message: &'a Message,
        ) -> BoxFut<'a, ()> {
            let message = message.clone();
            Box::pin(async move {
                session.messages.push(message);
            })
        }
        fn append_user_message<'a>(
            &'a self,
            session: &'a mut SessionState,
            text: &'a str,
        ) -> BoxFut<'a, ()> {
            Box::pin(async move {
                session.messages.push(Message {
                    role: crate::model::Role::User,
                    content: crate::model::Content::Text { text: text.into() },
                });
            })
        }
    }

    pub struct AllowAllSandbox;
    impl SandboxProvider for AllowAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
    }

    /// Deterministic [`VcsProvider`] double for tests and fixture replay (issue
    /// #58 v2). It returns pre-loaded strings VERBATIM with no process spawning,
    /// so multi-context-window Ralph continuation can be exercised hermetically.
    /// [`log`](VcsProvider::log) ignores its [`VcsLogArgs`] and yields
    /// `log_output`; [`status`](VcsProvider::status) yields `status_output`.
    pub struct FixtureVcsProvider {
        log_output: String,
        status_output: String,
    }

    impl FixtureVcsProvider {
        /// Construct a fixture provider returning `log_output` from `log` and
        /// `status_output` from `status`.
        pub fn new(log_output: impl Into<String>, status_output: impl Into<String>) -> Self {
            Self {
                log_output: log_output.into(),
                status_output: status_output.into(),
            }
        }
    }

    impl VcsProvider for FixtureVcsProvider {
        fn log<'a>(&'a self, _args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>> {
            Box::pin(async move { Ok(self.log_output.clone()) })
        }

        fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
            Box::pin(async move { Ok(self.status_output.clone()) })
        }
    }

    pub struct ScriptedSandbox {
        outcomes: Mutex<std::collections::VecDeque<Result<(), SandboxViolation>>>,
    }
    impl Default for ScriptedSandbox {
        fn default() -> Self {
            Self::new()
        }
    }
    impl ScriptedSandbox {
        pub fn new() -> Self {
            Self {
                outcomes: Mutex::new(Default::default()),
            }
        }
        pub fn push(&self, r: Result<(), SandboxViolation>) -> &Self {
            self.outcomes.lock().unwrap().push_back(r);
            self
        }
    }
    impl SandboxProvider for ScriptedSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            let r = self.outcomes.lock().unwrap().pop_front().unwrap_or(Ok(()));
            Box::pin(async move { r })
        }
    }

    pub struct ScriptedToolRegistry {
        outputs: Mutex<std::collections::VecDeque<ToolOutput>>,
        pub call_count: std::sync::atomic::AtomicUsize,
        always_halt: Mutex<Vec<String>>,
    }
    impl Default for ScriptedToolRegistry {
        fn default() -> Self {
            Self::new()
        }
    }
    impl ScriptedToolRegistry {
        pub fn new() -> Self {
            Self {
                outputs: Mutex::new(Default::default()),
                call_count: Default::default(),
                always_halt: Mutex::new(vec![]),
            }
        }
        pub fn push(&self, o: ToolOutput) -> &Self {
            self.outputs.lock().unwrap().push_back(o);
            self
        }
        pub fn mark_always_halt(&self, name: &str) {
            self.always_halt.lock().unwrap().push(name.into());
        }
    }
    impl ToolRegistry for ScriptedToolRegistry {
        fn dispatch<'a>(&'a self, _call: ToolCall) -> BoxFut<'a, ToolOutput> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let out = self
                .outputs
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(ToolOutput::Success {
                    content: "ok".into(),
                    truncated: false,
                });
            Box::pin(async move { out })
        }
        fn is_always_halt(&self, tool_name: &str) -> bool {
            self.always_halt
                .lock()
                .unwrap()
                .iter()
                .any(|n| n == tool_name)
        }
    }

    /// Tool registry that models a single tool taking one `flag: bool`
    /// parameter. Dispatch deserializes `flag` strictly: a real JSON bool
    /// succeeds, anything else (e.g. the string `"false"`) returns a
    /// recoverable error — exactly the weak-model failure tool-call repair
    /// targets. Exposes the tool's schema via [`ToolRegistry::schemas`] so the
    /// repair provider can read expected types.
    pub struct BoolToolRegistry {
        pub call_count: std::sync::atomic::AtomicUsize,
    }
    impl Default for BoolToolRegistry {
        fn default() -> Self {
            Self::new()
        }
    }
    impl BoolToolRegistry {
        pub fn new() -> Self {
            Self {
                call_count: Default::default(),
            }
        }
    }
    impl ToolRegistry for BoolToolRegistry {
        fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let out = match call.input.get("flag") {
                Some(serde_json::Value::Bool(b)) => ToolOutput::Success {
                    content: format!("flag={b}"),
                    truncated: false,
                },
                other => ToolOutput::Error {
                    message: format!("invalid parameters: expected bool `flag`, got {other:?}"),
                    recoverable: true,
                },
            };
            Box::pin(async move { out })
        }
        fn schemas(&self) -> Vec<ToolSchema> {
            vec![ToolSchema {
                name: "set_flag".into(),
                description: "set a boolean flag".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "flag": { "type": "boolean" } },
                    "required": ["flag"],
                }),
            }]
        }
    }

    pub struct AlwaysContinuePolicy;
    impl TerminationPolicy for AlwaysContinuePolicy {
        fn evaluate<'a>(
            &'a self,
            _session: &'a SessionState,
            _budget_used: &'a BudgetSnapshot,
        ) -> BoxFut<'a, TerminationDecision> {
            Box::pin(async { TerminationDecision::Continue })
        }
    }

    pub struct ScriptedTerminationPolicy {
        decisions: Mutex<std::collections::VecDeque<TerminationDecision>>,
    }
    impl Default for ScriptedTerminationPolicy {
        fn default() -> Self {
            Self::new()
        }
    }
    impl ScriptedTerminationPolicy {
        pub fn new() -> Self {
            Self {
                decisions: Mutex::new(Default::default()),
            }
        }
        pub fn push(&self, d: TerminationDecision) -> &Self {
            self.decisions.lock().unwrap().push_back(d);
            self
        }
    }
    impl TerminationPolicy for ScriptedTerminationPolicy {
        fn evaluate<'a>(
            &'a self,
            _session: &'a SessionState,
            _budget_used: &'a BudgetSnapshot,
        ) -> BoxFut<'a, TerminationDecision> {
            let d = self
                .decisions
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(TerminationDecision::Continue);
            Box::pin(async move { d })
        }
    }

    pub struct ScriptedMiddleware {
        decisions: Mutex<std::collections::VecDeque<(HookPoint, MiddlewareDecision)>>,
    }
    impl Default for ScriptedMiddleware {
        fn default() -> Self {
            Self::new()
        }
    }
    impl ScriptedMiddleware {
        pub fn new() -> Self {
            Self {
                decisions: Mutex::new(Default::default()),
            }
        }
        pub fn push(&self, h: HookPoint, d: MiddlewareDecision) -> &Self {
            self.decisions.lock().unwrap().push_back((h, d));
            self
        }
    }
    impl MiddlewareChain for ScriptedMiddleware {
        fn fire<'a>(
            &'a self,
            hook: HookPoint,
            _session: &'a SessionState,
        ) -> BoxFut<'a, MiddlewareDecision> {
            let mut q = self.decisions.lock().unwrap();
            let d = if let Some(front) = q.front() {
                if front.0 == hook {
                    q.pop_front().unwrap().1
                } else {
                    MiddlewareDecision::Continue
                }
            } else {
                MiddlewareDecision::Continue
            };
            Box::pin(async move { d })
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::agent::{mock::MockAgent, AgentId};
    use crate::hooks::HookChain as _;
    use crate::model::{ModelError, ToolCall};

    fn make_agent() -> Arc<MockAgent> {
        Arc::new(MockAgent::new(AgentId::new("test")))
    }

    fn standard_config(agent: Arc<MockAgent>) -> HarnessConfig {
        HarnessConfig {
            agent,
            tool_registry: Arc::new(ScriptedToolRegistry::new()),
            sandbox: Arc::new(AllowAllSandbox),
            context_manager: Arc::new(NoopContextManager),
            termination_policy: Arc::new(AlwaysContinuePolicy),
            middleware: None,
            observability: None,
            compaction_verifier: Arc::new(KeyTermVerifier),
            max_compaction_attempts: 2,
            pricing: PricingTable::DEFAULT,
            content_capture: ContentCaptureConfig::default(),
            tool_call_repair: None,
            max_repair_attempts: 1,
            max_stop_blocks: 8,
            hooks: None,
            planner_agent: None,
            verifier: None,
            evaluator_agent: None,
            // #76: plan_execute + task_list persistence now lives on the
            // RunStore seam (not SessionState.extras), so the test harness
            // needs a real (in-memory) run store for the readback helpers and
            // assertions below to observe what the harness wrote.
            storage: Arc::new(crate::storage::StorageProvider::single(Arc::new(
                crate::storage::InMemoryStorageProvider::new(),
            ))),
            chunk_provider: Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty()),
            max_resets: 3,
            vcs_provider: None,
            metric_evaluator: None,
        }
    }

    fn task(strategy: LoopStrategy) -> Task {
        Task::new("do something", SessionId::new("s1"), strategy)
    }

    fn react(max: u32) -> Task {
        task(LoopStrategy::ReAct {
            max_iterations: max,
        })
    }

    fn usage() -> TokenUsage {
        TokenUsage {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }

    // Rule: Harness owns the loop; final response on first turn returns Success.
    #[tokio::test]
    async fn final_response_returns_success() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let h = StandardHarness::new(standard_config(a));
        let r = h.run(HarnessRunOptions::new(react(5))).await;
        match r {
            RunResult::Success { output, turns, .. } => {
                assert_eq!(output, "done");
                assert_eq!(turns, 1);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // Rule: tool calls are dispatched, then loop continues to a final response.
    #[tokio::test]
    async fn tool_call_then_final_response_loops() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "after-tool".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "tool-ok".into(),
            truncated: false,
        });
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { turns, output, .. } => {
                assert_eq!(output, "after-tool");
                assert_eq!(turns, 2);
                assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // Issue #12 — the harness emits real spans through an injected
    // ObservabilityProvider and flushes a terminal session summary. Hermetic:
    // SPORE_OTLP_ENDPOINT is left unset so the outbox writes JSONL only.
    #[tokio::test]
    async fn harness_emits_spans_through_outbox_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "do_thing".into(),
                input: serde_json::json!({ "k": "v" }),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "tool-ok".into(),
            truncated: false,
        });

        let harness = HarnessBuilder::new(
            a,
            reg,
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .with_observability_outbox(tmp.path())
        .build();

        match harness.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { turns, .. } => assert_eq!(turns, 2),
            other => panic!("expected Success, got {other:?}"),
        }

        let path = tmp.path().join("sessions/s1/trace.jsonl");
        let body = std::fs::read_to_string(&path).expect("trace.jsonl written");
        let lines: Vec<serde_json::Value> = body
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let kinds: Vec<&str> = lines.iter().map(|l| l["kind"].as_str().unwrap()).collect();
        assert!(
            kinds.contains(&"turn"),
            "expected a turn span, got {kinds:?}"
        );
        assert!(
            kinds.contains(&"tool_call"),
            "expected a tool_call span, got {kinds:?}"
        );
        // The terminal summary line is written last.
        assert_eq!(kinds.last(), Some(&"session"), "kinds: {kinds:?}");
        let summary = lines.last().unwrap();
        assert_eq!(summary["attributes"]["outcome"], "success");
        assert_eq!(summary["attributes"]["total_turns"], 2);
        let tool = lines.iter().find(|l| l["kind"] == "tool_call").unwrap();
        assert_eq!(tool["attributes"]["tool_name"], "do_thing");
        assert_eq!(tool["attributes"]["call_id"], "c1");
        // Turn spans parent the tool-call spans within a shared trace.
        let trace_ids: std::collections::HashSet<&str> = lines
            .iter()
            .map(|l| l["trace_id"].as_str().unwrap())
            .collect();
        assert_eq!(trace_ids.len(), 1, "all spans share one trace id");
        assert!(tmp.path().join("sessions/s1/.flushed").exists());
    }

    // Issue #64 — content capture ON: the harness populates GenAI content on
    // the turn (output text + tool calls) and tool-call (args + result) spans,
    // and the outbox writes it into the JSONL as `gen_ai.*` attributes.
    #[tokio::test]
    async fn content_capture_on_writes_genai_content_to_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "do_thing".into(),
                input: serde_json::json!({ "k": "v" }),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "all finished".into(),
            usage: usage(),
        });
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "tool result body".into(),
            truncated: false,
        });

        let harness = HarnessBuilder::new(
            a,
            reg,
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .with_observability_outbox(tmp.path())
        .content_capture(ContentCaptureConfig {
            enabled: true,
            max_field_len: 8192,
        })
        .build();

        harness.run(HarnessRunOptions::new(react(5))).await;

        let body = std::fs::read_to_string(tmp.path().join("sessions/s1/trace.jsonl")).unwrap();
        let lines: Vec<serde_json::Value> = body
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        // The tool-requesting turn carries the requested tool call.
        let turn_with_calls = lines
            .iter()
            .find(|l| {
                l["kind"] == "turn" && !l["attributes"]["gen_ai.response.tool_calls"].is_null()
            })
            .expect("turn with captured tool calls");
        assert_eq!(
            turn_with_calls["attributes"]["gen_ai.response.tool_calls"][0]["name"],
            "do_thing"
        );

        // The final turn carries the assistant output text.
        let final_turn = lines
            .iter()
            .find(|l| {
                l["kind"] == "turn" && l["attributes"]["gen_ai.response.content"] == "all finished"
            })
            .expect("turn with captured output text");
        assert_eq!(
            final_turn["attributes"]["gen_ai.response.role"],
            "assistant"
        );

        // The tool-call span carries args + result.
        let tool = lines.iter().find(|l| l["kind"] == "tool_call").unwrap();
        assert_eq!(tool["attributes"]["gen_ai.tool.name"], "do_thing");
        assert_eq!(tool["attributes"]["gen_ai.tool.call.arguments"]["k"], "v");
        assert_eq!(
            tool["attributes"]["gen_ai.tool.message.content"],
            "tool result body"
        );
    }

    // Issue #64 — content capture OFF (default): no `gen_ai.*` content reaches
    // the JSONL. Byte-for-byte the same metrics-only output as pre-#64.
    #[tokio::test]
    async fn content_capture_off_writes_no_genai_content() {
        let tmp = tempfile::tempdir().unwrap();
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "do_thing".into(),
                input: serde_json::json!({ "secret": "value" }),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "sensitive output".into(),
            usage: usage(),
        });
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "sensitive result".into(),
            truncated: false,
        });

        // Default builder → content_capture OFF.
        let harness = HarnessBuilder::new(
            a,
            reg,
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .with_observability_outbox(tmp.path())
        .build();

        harness.run(HarnessRunOptions::new(react(5))).await;

        let body = std::fs::read_to_string(tmp.path().join("sessions/s1/trace.jsonl")).unwrap();
        // No `gen_ai.` keys and no leaked content strings anywhere in the file.
        assert!(!body.contains("gen_ai."), "no gen_ai content keys when off");
        assert!(!body.contains("sensitive output"));
        assert!(!body.contains("sensitive result"));
        assert!(!body.contains("\"value\""));
        // No assembled-input prompt leaked either (issue #64).
        assert!(!body.contains("gen_ai.prompt"));
    }

    // ── Input-message capture (issue #64) ──────────────────────────────────

    /// `capture_input_messages` maps each role and renders each Content variant:
    /// system + user + assistant tool-call + tool result, in order.
    #[test]
    fn capture_input_messages_maps_roles_and_renders_content() {
        use crate::model::{Content, Role, ToolCall as MToolCall, ToolResult as MToolResult};
        let msgs = vec![
            Message {
                role: Role::System,
                content: Content::Text {
                    text: "be helpful".into(),
                },
            },
            Message {
                role: Role::User,
                content: Content::Text {
                    text: "list files".into(),
                },
            },
            Message {
                role: Role::Assistant,
                content: Content::ToolCall(MToolCall {
                    id: "c1".into(),
                    name: "shell".into(),
                    input: serde_json::json!({ "command": "ls" }),
                }),
            },
            Message {
                role: Role::Tool,
                content: Content::ToolResult(MToolResult {
                    tool_use_id: "c1".into(),
                    content: "file.txt".into(),
                    is_error: false,
                }),
            },
        ];
        let out = StandardHarness::capture_input_messages(&msgs, 8192);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].role, GenAiRole::System);
        assert_eq!(out[0].content, "be helpful");
        assert_eq!(out[1].role, GenAiRole::User);
        assert_eq!(out[1].content, "list files");
        assert_eq!(out[2].role, GenAiRole::Assistant);
        assert_eq!(out[2].content, "shell {\"command\":\"ls\"}");
        assert_eq!(out[3].role, GenAiRole::Tool);
        assert_eq!(out[3].content, "file.txt");
        assert!(out.iter().all(|m| !m.truncated));
    }

    /// Image content renders as a placeholder — the base64 `data` is NEVER dumped.
    #[test]
    fn capture_input_messages_image_renders_placeholder_not_base64() {
        use crate::model::{Content, Role};
        let msgs = vec![Message {
            role: Role::User,
            content: Content::Image {
                media_type: "image/png".into(),
                data: "AAAABBBBCCCCDDDD_secret_base64".into(),
            },
        }];
        let out = StandardHarness::capture_input_messages(&msgs, 8192);
        assert_eq!(out[0].content, "[image image/png]");
        assert!(!out[0].content.contains("secret_base64"));
    }

    /// Truncation applies to rendered input content (byte budget + marker).
    #[test]
    fn capture_input_messages_truncates_long_content() {
        use crate::model::{Content, Role};
        let long = "x".repeat(100);
        let msgs = vec![Message {
            role: Role::User,
            content: Content::Text { text: long },
        }];
        let out = StandardHarness::capture_input_messages(&msgs, 10);
        assert!(out[0].truncated);
        assert!(out[0].content.ends_with("...[truncated]"));
        assert!(out[0].content.starts_with("xxxxxxxxxx"));
    }

    /// End-to-end: guard ON → the assembled prompt rides as `gen_ai.prompt`
    /// with the user message present and correct roles.
    #[tokio::test]
    async fn content_capture_on_writes_input_messages_to_jsonl() {
        let tmp = tempfile::tempdir().unwrap();
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let reg = Arc::new(ScriptedToolRegistry::new());
        let harness = HarnessBuilder::new(
            a,
            reg,
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .with_observability_outbox(tmp.path())
        .content_capture(ContentCaptureConfig {
            enabled: true,
            max_field_len: 8192,
        })
        .build();

        harness.run(HarnessRunOptions::new(react(5))).await;

        let body = std::fs::read_to_string(tmp.path().join("sessions/s1/trace.jsonl")).unwrap();
        let lines: Vec<serde_json::Value> = body
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let turn = lines
            .iter()
            .find(|l| l["kind"] == "turn" && !l["attributes"]["gen_ai.prompt"].is_null())
            .expect("turn with captured input messages");
        let prompt = turn["attributes"]["gen_ai.prompt"].as_array().unwrap();
        assert!(!prompt.is_empty());
        // The seeded user request is present as a user message.
        assert!(prompt
            .iter()
            .any(|m| m["role"] == "user" && m["content"].as_str().is_some()));
    }

    // Issue #73: the builder defaults storage to an all-no-op StorageProvider,
    // and `.storage(...)` / `storage()` / `session_store()` round-trip.
    #[tokio::test]
    async fn default_storage_is_no_op_and_setter_round_trips() {
        use crate::storage::RunStore as _;
        let a = make_agent();
        let reg = Arc::new(ScriptedToolRegistry::new());
        // Default: no .storage() — must be no-op (reads empty).
        let h = HarnessBuilder::new(
            a.clone(),
            reg.clone(),
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .build();
        let sess = SessionId::new("s1");
        assert!(h
            .session_store()
            .get_session(&sess)
            .await
            .unwrap()
            .is_none());
        assert!(h.storage().run().get(&sess, "k").await.unwrap().is_none());

        // Explicit injection round-trips through the accessor.
        let backend = Arc::new(crate::storage::InMemoryStorageProvider::new());
        backend
            .put(&sess, "plan", serde_json::json!({"ok": true}))
            .await
            .unwrap();
        let storage = Arc::new(crate::storage::StorageProvider::single(backend));
        let h2 = HarnessBuilder::new(
            a,
            reg,
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .storage(storage)
        .build();
        assert_eq!(
            h2.storage().run().get(&sess, "plan").await.unwrap(),
            Some(serde_json::json!({"ok": true}))
        );
    }

    // Issue #79, R25: the builder defaults the chunk provider to an empty
    // InMemoryChunkProvider, and `.chunks(...)` resolves to an
    // InMemoryChunkProvider carrying the registered chunks.
    #[tokio::test]
    async fn default_chunk_provider_empty_and_chunks_setter_round_trips() {
        use crate::prompt_assembly::PromptChunk;
        let a = make_agent();
        let reg = Arc::new(ScriptedToolRegistry::new());

        // Default: empty provider.
        let h = HarnessBuilder::new(
            a.clone(),
            reg.clone(),
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .build();
        let loaded = h.config().chunk_provider.load().await.unwrap();
        assert!(loaded.is_empty());

        // `.chunks(...)` resolves to an InMemoryChunkProvider with those chunks.
        let h2 = HarnessBuilder::new(
            a,
            reg,
            Arc::new(AllowAllSandbox),
            Arc::new(NoopContextManager),
            Arc::new(AlwaysContinuePolicy),
        )
        .chunks(vec![
            PromptChunk::new("core", "rules"),
            PromptChunk::new("style", "be concise"),
        ])
        .build();
        let loaded2 = h2.config().chunk_provider.load().await.unwrap();
        assert_eq!(loaded2.len(), 2);
        assert_eq!(loaded2[0].id, "core");
    }

    // Rule: parallel tool calls all dispatched in one turn.
    #[tokio::test]
    async fn parallel_tool_calls_all_dispatched() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![
                ToolCall {
                    id: "a".into(),
                    name: "x".into(),
                    input: serde_json::json!({}),
                },
                ToolCall {
                    id: "b".into(),
                    name: "y".into(),
                    input: serde_json::json!({}),
                },
            ],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "ok".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "1".into(),
            truncated: false,
        });
        reg.push(ToolOutput::Success {
            content: "2".into(),
            truncated: false,
        });
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(react(5))).await;
        assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    // Rule: budget overrun terminates with explicit reason.
    #[tokio::test]
    async fn budget_max_turns_exceeded() {
        let a = make_agent();
        for _ in 0..10 {
            a.push(TurnResult::ToolCallRequested {
                calls: vec![ToolCall {
                    id: "c".into(),
                    name: "x".into(),
                    input: serde_json::json!({}),
                }],
                usage: usage(),
            });
        }
        let h = StandardHarness::new(standard_config(a));
        let mut t = react(100);
        t.budget.max_turns = Some(2);
        match h.run(HarnessRunOptions::new(t)).await {
            RunResult::Failure {
                reason:
                    HaltReason::BudgetExceeded {
                        limit_type: BudgetLimitType::Turns,
                    },
                turns,
                ..
            } => {
                assert_eq!(turns, 2);
            }
            other => panic!("expected BudgetExceeded(Turns), got {other:?}"),
        }
    }

    // Rule: A turn with neither tool call nor response is an error.
    #[tokio::test]
    async fn agent_error_terminates_with_agent_error_halt_reason() {
        let a = make_agent();
        a.push(TurnResult::Error {
            error: AgentError::EmptyResponse,
            usage: Some(usage()),
        });
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason: HaltReason::AgentError { .. },
                ..
            } => {}
            other => panic!("expected AgentError halt, got {other:?}"),
        }
    }

    // Rule: Layer-1 SandboxViolation::PathEscape halts unconditionally.
    #[tokio::test]
    async fn layer1_path_escape_halts() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "read".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let sb = Arc::new(ScriptedSandbox::new());
        sb.push(Err(SandboxViolation::PathEscape {
            path: "/etc/passwd".into(),
        }));
        cfg.sandbox = sb;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason:
                    HaltReason::SandboxViolation {
                        violation: SandboxViolation::PathEscape { .. },
                    },
                ..
            } => {}
            other => panic!("expected SandboxViolation halt, got {other:?}"),
        }
    }

    // Rule: Layer-2 recoverable sandbox violation → appended as tool error, loop continues.
    #[tokio::test]
    async fn layer2_path_denied_continues_as_tool_error() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "read".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "ack".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let sb = Arc::new(ScriptedSandbox::new());
        sb.push(Err(SandboxViolation::PathDenied {
            path: "/p".into(),
            matched_rule: "test".into(),
        }));
        cfg.sandbox = sb;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { turns, .. } => assert_eq!(turns, 2),
            other => panic!("expected Success after recoverable violation, got {other:?}"),
        }
    }

    // Rule: TerminationPolicy::Halt overrides final response.
    #[tokio::test]
    async fn termination_policy_halt_overrides_success() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let tp = Arc::new(ScriptedTerminationPolicy::new());
        tp.push(TerminationDecision::Halt {
            reason: "not yet".into(),
        });
        cfg.termination_policy = tp;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason: HaltReason::TerminationPolicyHalt { reason },
                ..
            } => assert_eq!(reason, "not yet"),
            other => panic!("expected TerminationPolicyHalt, got {other:?}"),
        }
    }

    // Rule: Middleware::Halt at BeforeTurn halts before model call.
    #[tokio::test]
    async fn middleware_halt_before_turn() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "unused".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let mw = Arc::new(ScriptedMiddleware::new());
        mw.push(
            HookPoint::BeforeTurn,
            MiddlewareDecision::Halt {
                reason: "blocked".into(),
            },
        );
        cfg.middleware = Some(mw);
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason:
                    HaltReason::MiddlewareHalt {
                        hook: HookPoint::BeforeTurn,
                        reason,
                    },
                turns,
                ..
            } => {
                assert_eq!(reason, "blocked");
                assert_eq!(turns, 0);
            }
            other => panic!("expected MiddlewareHalt, got {other:?}"),
        }
    }

    // Rule: Middleware::SurfaceToHuman at BeforeTool returns WaitingForHuman with pending calls.
    #[tokio::test]
    async fn middleware_surface_to_human_before_tool() {
        let a = make_agent();
        let calls = vec![ToolCall {
            id: "c".into(),
            name: "x".into(),
            input: serde_json::json!({}),
        }];
        a.push(TurnResult::ToolCallRequested {
            calls: calls.clone(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let mw = Arc::new(ScriptedMiddleware::new());
        mw.push(
            HookPoint::BeforeTool,
            MiddlewareDecision::SurfaceToHuman {
                request: HumanRequest::ToolApproval {
                    calls: calls.clone(),
                    risk_level: RiskLevel::Medium,
                },
            },
        );
        cfg.middleware = Some(mw);
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::WaitingForHuman { state, .. } => {
                assert_eq!(state.pending_tool_calls.len(), 1);
                assert!(state.child_state.is_none());
            }
            other => panic!("expected WaitingForHuman, got {other:?}"),
        }
    }

    // Rule: always_halt tool annotation halts the loop.
    #[tokio::test]
    async fn always_halt_tool_halts() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "danger".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.mark_always_halt("danger");
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason: HaltReason::UnrecoverableToolError { tool, .. },
                ..
            } => assert_eq!(tool, "danger"),
            other => panic!("expected UnrecoverableToolError, got {other:?}"),
        }
    }

    // Rule: Unrecoverable tool error halts loop immediately.
    #[tokio::test]
    async fn unrecoverable_tool_error_halts() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Error {
            message: "boom".into(),
            recoverable: false,
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason: HaltReason::UnrecoverableToolError { error, .. },
                ..
            } => assert_eq!(error, "boom"),
            other => panic!("expected UnrecoverableToolError, got {other:?}"),
        }
    }

    // Tool-call repair: a `flag: "false"` string argument is coerced to a real
    // bool and re-dispatched, so the tool succeeds and the run completes.
    #[tokio::test]
    async fn tool_call_repair_fixes_bad_bool_arg() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let a = make_agent();
        // Turn 1: call the tool with a stringified bool (weak-model behaviour).
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "set_flag".into(),
                input: serde_json::json!({ "flag": "false" }),
            }],
            usage: usage(),
        });
        // Turn 2: finish.
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });

        let mut cfg = standard_config(a);
        let reg = Arc::new(BoolToolRegistry::new());
        cfg.tool_registry = reg.clone();
        cfg.tool_call_repair = Some(Arc::new(crate::tool_call_repair::StandardToolCallRepair));

        let last_tool_error = Arc::new(AtomicBool::new(false));
        let sink_flag = last_tool_error.clone();
        let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(move |ev| {
            if let StreamEvent::ToolResult { is_error, .. } = ev {
                sink_flag.store(is_error, Ordering::SeqCst);
            }
        }));

        let h = StandardHarness::new(cfg);
        match h.run(opts).await {
            RunResult::Success { output, .. } => assert_eq!(output, "done"),
            other => panic!("expected Success, got {other:?}"),
        }
        // Two dispatches: the failed original + the repaired retry.
        assert_eq!(reg.call_count.load(Ordering::SeqCst), 2);
        // Final tool result was a success, not an error.
        assert!(!last_tool_error.load(Ordering::SeqCst));
    }

    // Without a repair provider, the same bad `flag: "false"` argument yields a
    // recoverable tool error fed back to the model (today's behaviour).
    #[tokio::test]
    async fn without_repair_bad_bool_arg_errors() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "set_flag".into(),
                input: serde_json::json!({ "flag": "false" }),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });

        let mut cfg = standard_config(a);
        let reg = Arc::new(BoolToolRegistry::new());
        cfg.tool_registry = reg.clone();
        // No repair provider configured.

        let last_tool_error = Arc::new(AtomicBool::new(false));
        let sink_flag = last_tool_error.clone();
        let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(move |ev| {
            if let StreamEvent::ToolResult { is_error, .. } = ev {
                sink_flag.store(is_error, Ordering::SeqCst);
            }
        }));

        let h = StandardHarness::new(cfg);
        let _ = h.run(opts).await;
        // Exactly one dispatch, and it errored.
        assert_eq!(reg.call_count.load(Ordering::SeqCst), 1);
        assert!(last_tool_error.load(Ordering::SeqCst));
    }

    // Rule: WaitingForHuman from a tool dispatch propagates to RunResult.
    #[tokio::test]
    async fn tool_waiting_for_human_propagates() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "subagent".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        let child_task = task(LoopStrategy::ReAct { max_iterations: 1 });
        reg.push(ToolOutput::WaitingForHuman {
            child_state: Box::new(ChildPausedState {
                session_id: SessionId::new("child"),
                task_id: TaskId::new("ct"),
                turn_number: 1,
                session_state: SessionState::default(),
                pending_tool_calls: vec![],
                approved_results: vec![],
                human_request: Some(HumanRequest::Clarification {
                    question: "?".into(),
                    options: None,
                }),
                task: child_task,
                budget_used: BudgetSnapshot::default(),
                parent_tool_call_id: "c".into(),
            }),
            request: HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            },
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::WaitingForHuman { state, .. } => {
                assert!(state.child_state.is_some());
            }
            other => panic!("expected WaitingForHuman from tool, got {other:?}"),
        }
    }

    // ====================================================================
    // Tool Escalation Protocol (issue #80) — rules R1–R9
    // ====================================================================

    /// Build a `MockAgent` whose first turn requests `n` tool calls (named
    /// `t0..t{n-1}` with ids `c0..c{n-1}`), then a `FinalResponse` so a
    /// resumed loop can terminate.
    fn agent_with_tool_calls(n: usize) -> Arc<MockAgent> {
        let a = make_agent();
        let calls: Vec<ToolCall> = (0..n)
            .map(|k| ToolCall {
                id: format!("c{k}"),
                name: format!("t{k}"),
                input: serde_json::json!({}),
            })
            .collect();
        a.push(TurnResult::ToolCallRequested {
            calls,
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "resumed-done".into(),
            usage: usage(),
        });
        a
    }

    fn abort_signal() -> HarnessSignal {
        HarnessSignal::Abort {
            reason: "agent requested stop".into(),
        }
    }

    // R1 + R8: a dispatched `Escalate { Abort }` terminates the run and
    // returns `RunResult::Escalate`, NOT `RunResult::Failure`.
    #[tokio::test]
    async fn escalate_abort_terminates_with_escalate_not_failure() {
        let a = agent_with_tool_calls(1);
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Escalate {
            signal: abort_signal(),
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Escalate { signal, .. } => {
                assert_eq!(signal, abort_signal());
            }
            RunResult::Failure { .. } => {
                panic!("Abort must NOT surface as Failure (R8)")
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // R2: the escalation is NOT appended to message history. With one
    // escalating call, the only appended message is the assistant tool-call
    // turn — there is no `Role::Tool` result message.
    #[tokio::test]
    async fn escalate_is_not_appended_to_history() {
        let a = agent_with_tool_calls(1);
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Escalate {
            signal: abort_signal(),
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Escalate { state, .. } => {
                let tool_results = state
                    .session_state
                    .messages
                    .iter()
                    .filter(|m| m.role == Role::Tool)
                    .count();
                assert_eq!(tool_results, 0, "escalation must not append a tool result");
                // The assistant tool-call turn IS recorded (well-formed history).
                let assistant_turns = state
                    .session_state
                    .messages
                    .iter()
                    .filter(|m| m.role == Role::Assistant)
                    .count();
                assert_eq!(assistant_turns, 1);
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // R3: observability is finalized with `SessionOutcome::Escalated`.
    #[tokio::test]
    async fn escalate_finalizes_observability_as_escalated() {
        let a = agent_with_tool_calls(1);
        let mut cfg = standard_config(a);
        let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
        cfg.observability = Some(obs.clone());
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Escalate {
            signal: abort_signal(),
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        let sid = match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Escalate { session_id, .. } => session_id,
            other => panic!("expected Escalate, got {other:?}"),
        };
        let metrics = obs
            .get_session_metrics(&sid)
            .await
            .expect("escalation must flush a finalized session");
        assert_eq!(metrics.outcome, SessionOutcome::Escalated);
    }

    // R3 (contrast): `WaitingForHuman` does NOT finalize observability.
    #[tokio::test]
    async fn waiting_for_human_does_not_finalize_observability() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "subagent".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
        cfg.observability = Some(obs.clone());
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::WaitingForHuman {
            child_state: Box::new(ChildPausedState {
                session_id: SessionId::new("child"),
                task_id: TaskId::new("ct"),
                turn_number: 1,
                session_state: SessionState::default(),
                pending_tool_calls: vec![],
                approved_results: vec![],
                human_request: Some(HumanRequest::Clarification {
                    question: "?".into(),
                    options: None,
                }),
                task: task(LoopStrategy::ReAct { max_iterations: 1 }),
                budget_used: BudgetSnapshot::default(),
                parent_tool_call_id: "c".into(),
            }),
            request: HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            },
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        let sid = match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::WaitingForHuman { state, .. } => state.session_id.clone(),
            other => panic!("expected WaitingForHuman, got {other:?}"),
        };
        // Not finalized: `finalize_observability` was never called, so no
        // terminal `SessionOutcome` was recorded. Metrics may still exist
        // (a turn span was emitted) but the outcome defaults to `Partial` —
        // crucially it is NOT `Escalated` (contrast the escalation path).
        if let Some(m) = obs.get_session_metrics(&sid).await {
            assert_ne!(
                m.outcome,
                SessionOutcome::Escalated,
                "WaitingForHuman is not terminal — it must not finalize as Escalated"
            );
        }
    }

    // R4: `RunResult::Escalate` carries all five fields populated, and
    // `turns == budget_used.turns`.
    #[tokio::test]
    async fn escalate_carries_all_five_fields() {
        let a = agent_with_tool_calls(1);
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Escalate {
            signal: abort_signal(),
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Escalate {
                signal,
                state,
                session_id,
                usage: agg,
                turns,
            } => {
                assert_eq!(signal, abort_signal());
                assert_eq!(session_id, SessionId::new("s1"));
                assert_eq!(state.session_id, session_id);
                assert_eq!(turns, state.budget_used.turns, "turns == budget_used.turns");
                // One turn was consumed before the escalating dispatch.
                assert_eq!(turns, 1);
                // Usage is populated from the consumed turn.
                assert_eq!(agg.input_tokens, 1);
                assert_eq!(agg.output_tokens, 1);
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
    }

    // R5 + R6: the preserved `state` is resumable, and the signal is
    // discarded on resume — the harness just continues the original session.
    #[tokio::test]
    async fn escalate_state_resumes_and_discards_signal() {
        let a = agent_with_tool_calls(1);
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Escalate {
            signal: HarnessSignal::SwitchMode {
                mode: crate::prompt_chunk_registry::Mode::Plan,
            },
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        let state = match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Escalate { state, .. } => *state,
            other => panic!("expected Escalate, got {other:?}"),
        };
        // Escalation-derived state has no human request (R5 shape).
        assert!(state.human_request.is_none());
        // R6: resume continues the ORIGINAL session — it does NOT switch mode,
        // enter plan mode, or abort. The MockAgent's next turn is a
        // FinalResponse, so resume runs to Success.
        match h.resume(state, HumanResponse::Allow, None).await {
            RunResult::Success {
                output, session_id, ..
            } => {
                assert_eq!(output, "resumed-done");
                assert_eq!(session_id, SessionId::new("s1"));
            }
            other => panic!("expected Success on resume (signal discarded), got {other:?}"),
        }
    }

    // Issue #81: a tool returning AwaitingClarification pauses the loop with a
    // PausedState whose human_request is a Clarification (NO child_state) and
    // returns WaitingForHuman. The clarifying call is preserved as the head of
    // pending_tool_calls.
    #[tokio::test]
    async fn awaiting_clarification_pauses_without_child_state() {
        let a = agent_with_tool_calls(2);
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::AwaitingClarification {
            question: "which option?".into(),
            options: Some(vec!["a".into(), "b".into()]),
        });
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::WaitingForHuman { state, request } => {
                // No subagent child state on a clarification pause.
                assert!(state.child_state.is_none());
                match &request {
                    HumanRequest::Clarification { question, options } => {
                        assert_eq!(question, "which option?");
                        assert_eq!(
                            options.as_ref().unwrap(),
                            &vec!["a".to_string(), "b".to_string()]
                        );
                    }
                    other => panic!("expected Clarification, got {other:?}"),
                }
                // The clarifying call (c0/t0) is the head of pending; c1 trails.
                assert_eq!(state.pending_tool_calls.len(), 2);
                assert_eq!(state.pending_tool_calls[0].id, "c0");
                assert_eq!(state.pending_tool_calls[1].id, "c1");
            }
            other => panic!("expected WaitingForHuman, got {other:?}"),
        }
        // Exactly one dispatch (the clarifying call); c1 was preserved, not run.
        assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    // Issue #81: resuming a Clarification pause injects the human's Answer as
    // the TOOL RESULT for the clarifying call, then continues to Success.
    #[tokio::test]
    async fn clarification_resume_injects_answer_as_tool_result() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "clarified-done".into(),
            usage: usage(),
        });
        let cfg = standard_config(a);
        let h = StandardHarness::new(cfg);
        let state = PausedState {
            session_id: SessionId::new("s"),
            task_id: TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![ToolCall {
                id: "ask".into(),
                name: "ask_user_question".into(),
                input: serde_json::json!({"question": "?"}),
            }],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            }),
            task: react(5),
            budget_used: BudgetSnapshot::default(),
            child_state: None,
        };
        match h
            .resume(
                state,
                HumanResponse::Answer {
                    text: "use a".into(),
                },
                None,
            )
            .await
        {
            RunResult::Success { output, .. } => assert_eq!(output, "clarified-done"),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // Issue #81: the send_message tool surfaces a StreamEvent::UserMessage and
    // the loop records a success tool result (continuing, not pausing).
    #[tokio::test]
    async fn send_message_emits_user_message_event() {
        use std::sync::Mutex;
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: crate::tools::SendMessageTool::NAME.into(),
                input: serde_json::json!({"content": "hello human"}),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "hello human".into(),
            truncated: false,
        });
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);

        let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = captured.clone();
        let opts = HarnessRunOptions::new(react(5)).with_stream(Box::new(move |ev| {
            if let StreamEvent::UserMessage { content } = ev {
                sink.lock().unwrap().push(content);
            }
        }));
        match h.run(opts).await {
            RunResult::Success { output, .. } => assert_eq!(output, "done"),
            other => panic!("expected Success, got {other:?}"),
        }
        let msgs = captured.lock().unwrap();
        assert_eq!(msgs.as_slice(), &["hello human".to_string()]);
    }

    // R7-adjacent: each `HarnessSignal` variant round-trips through serde
    // wrapped in `ToolOutput::Escalate` with the documented wire shape.
    // (Full byte fixture lives in `fixtures/harness/escalation_signals.json`.)
    #[test]
    fn harness_signal_wire_format_round_trips() {
        #[cfg_attr(not(feature = "dangerous"), allow(unused_mut))]
        let mut cases = vec![
            ToolOutput::Escalate {
                signal: HarnessSignal::EnterPlanMode {
                    context: "ctx".into(),
                },
            },
            ToolOutput::Escalate {
                signal: HarnessSignal::ExitPlanMode {
                    plan: crate::plan::PlanArtifact {
                        tasks: vec!["a".into(), "b".into()],
                        rationale: "why".into(),
                    },
                },
            },
            ToolOutput::Escalate {
                signal: HarnessSignal::SwitchMode {
                    mode: crate::prompt_chunk_registry::Mode::SafeAuto,
                },
            },
            ToolOutput::Escalate {
                signal: abort_signal(),
            },
        ];
        // Yolo only exists under the `dangerous` feature (issue #34).
        #[cfg(feature = "dangerous")]
        cases.push(ToolOutput::Escalate {
            signal: HarnessSignal::SwitchMode {
                mode: crate::prompt_chunk_registry::Mode::Yolo,
            },
        });
        for case in cases {
            let json = serde_json::to_string(&case).unwrap();
            let back: ToolOutput = serde_json::from_str(&json).unwrap();
            assert_eq!(case, back);
        }
        // Spot-check the tag shape: snake_case `kind` on both layers.
        let json = serde_json::to_value(&ToolOutput::Escalate {
            signal: abort_signal(),
        })
        .unwrap();
        assert_eq!(json["kind"], "escalate");
        assert_eq!(json["signal"]["kind"], "abort");
        assert_eq!(json["signal"]["reason"], "agent requested stop");
    }

    // R7: `SwitchMode` carries the EXISTING `Mode` enum (no `HarnessMode`).
    #[test]
    fn switch_mode_uses_existing_mode_enum() {
        let json = serde_json::to_value(&HarnessSignal::SwitchMode {
            mode: crate::prompt_chunk_registry::Mode::SafeAuto,
        })
        .unwrap();
        assert_eq!(json["kind"], "switch_mode");
        assert_eq!(json["mode"], "safe_auto");
    }

    // R9: remaining tool calls after the escalating call land in
    // `state.pending_tool_calls` (escalate on call[0] of a 2-call batch →
    // pending == [call_1]).
    #[tokio::test]
    async fn escalate_preserves_remaining_calls_as_pending() {
        let a = agent_with_tool_calls(2);
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        // First dispatched call (c0) escalates.
        reg.push(ToolOutput::Escalate {
            signal: abort_signal(),
        });
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Escalate { state, .. } => {
                assert_eq!(state.pending_tool_calls.len(), 1);
                assert_eq!(state.pending_tool_calls[0].id, "c1");
                assert_eq!(state.pending_tool_calls[0].name, "t1");
            }
            other => panic!("expected Escalate, got {other:?}"),
        }
        // Exactly one dispatch happened — c1 was preserved, not executed.
        assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    // Rule: resume() with Halt returns Failure(HumanHalted).
    #[tokio::test]
    async fn resume_with_halt_returns_human_halted() {
        let a = make_agent();
        let h = StandardHarness::new(standard_config(a));
        let state = PausedState {
            session_id: SessionId::new("s"),
            task_id: TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            }),
            task: react(5),
            budget_used: BudgetSnapshot::default(),
            child_state: None,
        };
        match h.resume(state, HumanResponse::Halt, None).await {
            RunResult::Failure {
                reason: HaltReason::HumanHalted,
                ..
            } => {}
            other => panic!("expected HumanHalted, got {other:?}"),
        }
    }

    // Rule: resume() with Allow dispatches pending tool calls then continues loop.
    #[tokio::test]
    async fn resume_with_allow_executes_pending_and_continues() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "tool-ok".into(),
            truncated: false,
        });
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);
        let state = PausedState {
            session_id: SessionId::new("s"),
            task_id: TaskId::new("t"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            approved_results: vec![],
            human_request: Some(HumanRequest::ToolApproval {
                calls: vec![],
                risk_level: RiskLevel::Low,
            }),
            task: react(5),
            budget_used: BudgetSnapshot::default(),
            child_state: None,
        };
        match h.resume(state, HumanResponse::Allow, None).await {
            RunResult::Success { output, .. } => assert_eq!(output, "done"),
            other => panic!("expected Success on resume, got {other:?}"),
        }
        assert_eq!(reg.call_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    // Rule (issue #60): HillClimbing is now implemented — it no longer returns
    // StrategyNotYetImplemented. With no `metric_evaluator` wired it instead
    // halts with the typed `HillClimbingMisconfigured` (Decision 6), exercised in
    // the HillClimbing test block below. PlanExecute (#70), SelfVerifying (#61),
    // and Ralph (#58) are likewise implemented and covered in their own blocks,
    // so no loop strategy returns the generic stub anymore.
    #[tokio::test]
    async fn hill_climbing_no_longer_not_yet_implemented() {
        let a = make_agent();
        let h = StandardHarness::new(standard_config(a));
        let t = task(LoopStrategy::HillClimbing {
            direction: OptimizationDirection::Maximize,
            max_stagnation: None,
            revert_on_no_improvement: false,
            min_improvement_delta: None,
        });
        match h.run(HarnessRunOptions::new(t)).await {
            RunResult::Failure { reason, .. } => {
                assert!(
                    !matches!(reason, HaltReason::StrategyNotYetImplemented { .. }),
                    "HillClimbing must not return StrategyNotYetImplemented; got {reason:?}"
                );
                assert!(
                    matches!(reason, HaltReason::HillClimbingMisconfigured { .. }),
                    "expected HillClimbingMisconfigured with no evaluator wired; got {reason:?}"
                );
            }
            other => panic!("expected Failure, got {other:?}"),
        }
    }

    // ── PlanExecute plan phase (issue #70) ──────────────────────────────────

    use crate::plan::{PlanArtifact, PlanPhaseError, PLAN_EXECUTE_EXTRAS_KEY};

    fn plan_task() -> Task {
        task(LoopStrategy::PlanExecute { plan_model: None })
    }

    fn final_resp(text: &str) -> TurnResult {
        TurnResult::FinalResponse {
            content: text.into(),
            usage: usage(),
        }
    }

    // #76: the plan artifact now lives on the RunStore seam (not extras). Read
    // it back through the harness's storage under PLAN_EXECUTE_EXTRAS_KEY.
    async fn stored_artifact(h: &StandardHarness, session_id: &SessionId) -> PlanArtifact {
        let v = h
            .storage()
            .run()
            .get(session_id, PLAN_EXECUTE_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .expect("plan_execute present in run store");
        serde_json::from_value(v).expect("artifact deserializes")
    }

    // Issue #59 happy path: plan turn (2 tasks) then 2 execute completions. The
    // full PlanExecute run() now SUCCEEDS (proving ExecutePhaseNotImplemented is
    // gone) and `output` is the LAST step's FinalResponse (Q2). turns == 3 (one
    // plan + one per step).
    #[tokio::test]
    async fn plan_execute_full_run_succeeds() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["one","two"],"rationale":"why"}"#));
        a.push(final_resp("did one"));
        a.push(final_resp("did two"));
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Success { output, turns, .. } => {
                // Q2: the success handle is the LAST completed step's final text.
                assert_eq!(output, "did two");
                assert_eq!(turns, 3, "one plan turn + one turn per task");
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // R1/R3/R4: the plan turn runs exactly once and stores the exact artifact.
    // Uses run_plan_phase directly so we can inspect the mutated session_state.
    #[tokio::test]
    async fn plan_phase_runs_once_and_stores_exact_artifact() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["a","b","c"],"rationale":"r"}"#));
        let agent_for_count = a.clone();
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task();
        let mut state = SessionState::default();
        let outcome = h
            .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect("plan phase succeeds");
        // R1: exactly one planner turn.
        assert_eq!(
            agent_for_count
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        // R7: counted against the budget.
        assert_eq!(outcome.turns, 1);
        // R3/R4: exact artifact stored.
        let stored = stored_artifact(&h, &t.session_id).await;
        assert_eq!(stored.tasks, vec!["a", "b", "c"]);
        assert_eq!(stored.rationale, "r");
        assert_eq!(stored, outcome.artifact);
    }

    // R2: a tool call in the one-shot plan turn is a planning failure — no
    // dispatch loop runs (tool registry is never hit).
    #[tokio::test]
    async fn plan_phase_tool_call_is_planning_failure() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let reg = Arc::new(ScriptedToolRegistry::new());
        let mut cfg = standard_config(a);
        cfg.tool_registry = reg.clone();
        let h = StandardHarness::new(cfg);
        let t = plan_task();
        let mut state = SessionState::default();
        let err = h
            .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect_err("tool call fails the plan phase");
        match err {
            RunResult::Failure {
                reason:
                    HaltReason::PlanPhaseFailed {
                        error: PlanPhaseError::PlanningTurnFailed { .. },
                    },
                ..
            } => {}
            other => panic!("expected PlanningTurnFailed, got {other:?}"),
        }
        // No dispatch loop: registry never called, nothing stored.
        assert_eq!(
            reg.call_count.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "tool registry must not be dispatched in the plan turn"
        );
        assert!(h
            .storage()
            .run()
            .get(&t.session_id, PLAN_EXECUTE_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .is_none());
    }

    // R5: when planner_agent is set, the PLANNER runs the plan turn and the
    // default agent does not.
    #[tokio::test]
    async fn plan_phase_routes_to_planner_agent() {
        let default_agent = make_agent();
        default_agent.push(final_resp(r#"{"tasks":["default"]}"#));
        let planner = make_agent();
        planner.push(final_resp(r#"{"tasks":["planner"]}"#));

        let mut cfg = standard_config(default_agent.clone());
        cfg.planner_agent = Some(planner.clone());
        let h = StandardHarness::new(cfg);
        let t = plan_task();
        let mut state = SessionState::default();
        h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect("plan phase succeeds");

        assert_eq!(
            planner.call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "planner agent ran the plan turn"
        );
        assert_eq!(
            default_agent
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "default agent did not run the plan turn"
        );
        assert_eq!(
            stored_artifact(&h, &t.session_id).await.tasks,
            vec!["planner"]
        );
    }

    // R6: with no planner_agent, the plan turn runs on the default agent.
    #[tokio::test]
    async fn plan_phase_routes_to_default_agent_when_unset() {
        let default_agent = make_agent();
        default_agent.push(final_resp(r#"{"tasks":["default"]}"#));
        let h = StandardHarness::new(standard_config(default_agent.clone()));
        let t = plan_task();
        let mut state = SessionState::default();
        h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect("plan phase succeeds");
        assert_eq!(
            default_agent
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            stored_artifact(&h, &t.session_id).await.tasks,
            vec!["default"]
        );
    }

    // R8 (#70): the plan turn records exactly one TurnSpan, and it is the FIRST
    // span of the run (turn 1). With the execute phase (#59) wired in, the full
    // run() emits additional spans for the execute steps; this asserts the plan
    // turn's span specifically. A dedicated #59 test
    // (`execute_phase_span_count`) covers the per-step span accounting.
    #[tokio::test]
    async fn plan_phase_records_one_turn_span() {
        let a = make_agent();
        // Plan turn + one execute completion so the run progresses cleanly.
        a.push(final_resp(r#"{"tasks":["a"]}"#));
        a.push(final_resp("did a"));
        let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
        let mut cfg = standard_config(a);
        cfg.observability = Some(obs.clone());
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(plan_task())).await;
        let spans = obs.turn_spans(&SessionId::new("s1"));
        // Plan turn (1) + the single execute step turn (1) = 2 spans total.
        assert_eq!(spans.len(), 2, "plan turn + one execute step turn");
        // The plan turn is span 0, turn_number 1.
        assert_eq!(spans[0].turn_number, 1, "first span is the plan turn");
    }

    // R10: budget exhausted before the plan turn → budget-exceeded Failure with
    // no artifact stored and no planner turn.
    #[tokio::test]
    async fn plan_phase_budget_exhausted_before_turn() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["a"]}"#));
        let agent_for_count = a.clone();
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task().with_budget(BudgetLimits {
            max_turns: Some(1),
            ..Default::default()
        });
        let mut state = SessionState::default();
        // Pre-consume the only allowed turn.
        let used = BudgetSnapshot {
            turns: 1,
            ..Default::default()
        };
        let err = h
            .run_plan_phase(&t, &mut state, used, &None)
            .await
            .expect_err("budget exhausted");
        match err {
            RunResult::Failure {
                reason:
                    HaltReason::BudgetExceeded {
                        limit_type: BudgetLimitType::Turns,
                    },
                ..
            } => {}
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
        assert_eq!(
            agent_for_count
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "no planner turn ran"
        );
        assert!(h
            .storage()
            .run()
            .get(&t.session_id, PLAN_EXECUTE_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .is_none());
    }

    // R11: an OnPlanCreated hook can rewrite the plan before storage; the
    // stored artifact reflects the mutation.
    #[tokio::test]
    async fn plan_phase_on_plan_created_rewrites_before_storage() {
        use crate::hooks::{FunctionHook, HookDecision, HookEvent, StandardHookChain};

        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["original"],"rationale":"orig"}"#));

        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(FunctionHook::new(
                "rewrite-plan",
                vec![HookEvent::OnPlanCreated],
                |ctx| {
                    if let crate::hooks::HookContext::OnPlanCreated { plan, .. } = ctx {
                        plan.tasks = vec!["rewritten".to_string()];
                        plan.rationale = "mutated".to_string();
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();

        let mut cfg = standard_config(a);
        cfg.hooks = Some(Arc::new(chain));
        let h = StandardHarness::new(cfg);
        let t = plan_task();
        let mut state = SessionState::default();
        let outcome = h
            .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect("plan phase succeeds");

        let stored = stored_artifact(&h, &t.session_id).await;
        assert_eq!(stored.tasks, vec!["rewritten"]);
        assert_eq!(stored.rationale, "mutated");
        assert_eq!(stored, outcome.artifact);
    }

    // R3 fence variant: a fenced ```json plan is captured through the harness.
    #[tokio::test]
    async fn plan_phase_captures_fenced_json() {
        let a = make_agent();
        a.push(final_resp("```json\n{\"tasks\":[\"f1\",\"f2\"]}\n```"));
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task();
        let mut state = SessionState::default();
        h.run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect("plan phase succeeds");
        assert_eq!(
            stored_artifact(&h, &t.session_id).await.tasks,
            vec!["f1", "f2"]
        );
    }

    // R3: an unparseable plan surfaces PlanPhaseFailed/UnparseablePlan and
    // stores nothing.
    #[tokio::test]
    async fn plan_phase_unparseable_response_fails() {
        let a = make_agent();
        a.push(final_resp("this is not json"));
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task();
        let mut state = SessionState::default();
        let err = h
            .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect_err("unparseable plan fails");
        match err {
            RunResult::Failure {
                reason:
                    HaltReason::PlanPhaseFailed {
                        error: PlanPhaseError::UnparseablePlan { .. },
                    },
                ..
            } => {}
            other => panic!("expected UnparseablePlan, got {other:?}"),
        }
        assert!(h
            .storage()
            .run()
            .get(&t.session_id, PLAN_EXECUTE_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .is_none());
    }

    // ── PlanExecute execute phase (issue #59) ───────────────────────────────

    use crate::tasklist::{TaskList, TaskStatus, TASK_LIST_EXTRAS_KEY};

    // #76: the task list now lives on the RunStore seam (not extras). Read it
    // back through the harness's storage under TASK_LIST_EXTRAS_KEY.
    async fn run_store_task_list(h: &StandardHarness, session_id: &SessionId) -> TaskList {
        let v = h
            .storage()
            .run()
            .get(session_id, TASK_LIST_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .expect("task_list present in run store");
        serde_json::from_value(v).expect("task list deserializes")
    }

    // Build a PlanExecute task carrying an explicit budget (e.g. a turn cap).
    fn plan_task_with_budget(budget: BudgetLimits) -> Task {
        plan_task().with_budget(budget)
    }

    // Q1 happy path + plan-then-execute + task drain. Plan produces K=3 tasks,
    // then 3 execute completions. The run succeeds; turns == 4 (plan + 3).
    #[tokio::test]
    async fn execute_phase_happy_path_drains_all_tasks() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["t1","t2","t3"],"rationale":"r"}"#));
        a.push(final_resp("done t1"));
        a.push(final_resp("done t2"));
        a.push(final_resp("done t3"));
        let agent_count = a.clone();
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Success { output, turns, .. } => {
                assert_eq!(output, "done t3", "Q2: output is the last step's final");
                assert_eq!(turns, 4, "one plan turn + one per task");
            }
            other => panic!("expected Success, got {other:?}"),
        }
        // 1 plan turn + 3 execute turns = 4 agent calls.
        assert_eq!(
            agent_count
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            4
        );
    }

    // Task drain Pending -> InProgress -> Completed: inspect the persisted list
    // after a run via run_execute_phase directly so we can read the final state.
    #[tokio::test]
    async fn execute_phase_drains_pending_inprogress_completed() {
        let a = make_agent();
        a.push(final_resp("done one"));
        a.push(final_resp("done two"));
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task();
        let mut state = SessionState::default();
        let list = plan_artifact_to_tasklist_helper(&["one", "two"]);
        // All start Pending.
        assert!(list.tasks.iter().all(|x| x.status == TaskStatus::Pending));
        let result = h
            .run_execute_phase(
                &t,
                &mut state,
                list,
                BudgetSnapshot {
                    turns: 1,
                    ..Default::default()
                },
                AggregateUsage::default(),
                &None,
            )
            .await;
        match result {
            RunResult::Success { output, .. } => assert_eq!(output, "done two"),
            other => panic!("expected Success, got {other:?}"),
        }
        // Final persisted list: every task Completed.
        let final_list = run_store_task_list(&h, &t.session_id).await;
        assert!(
            final_list
                .tasks
                .iter()
                .all(|x| x.status == TaskStatus::Completed),
            "all tasks Completed after drain"
        );
    }

    fn plan_artifact_to_tasklist_helper(steps: &[&str]) -> TaskList {
        crate::tasklist::plan_artifact_to_task_list(&PlanArtifact {
            tasks: steps.iter().map(|s| s.to_string()).collect(),
            rationale: String::new(),
        })
    }

    // Q1 per-task turn allocation + shared budget: with a global cap of 7 turns
    // and a plan turn already spent (1), 3 tasks split the remaining 6 turns
    // (2 each). A task that needs >2 turns must be cut off by its per-task cap,
    // proving the allocation is enforced and the shared budget carries forward.
    #[tokio::test]
    async fn execute_phase_per_task_turn_allocation() {
        let a = make_agent();
        // Plan: 3 tasks.
        a.push(final_resp(r#"{"tasks":["a","b","c"]}"#));
        // Task a: 2 tool calls then would need a 3rd turn — but per_task = 2, so
        // it is cut off at 2 turns inside its sub-loop. Push 2 tool-call turns.
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "1".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "2".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "ok".into(),
            truncated: false,
        });
        reg.push(ToolOutput::Success {
            content: "ok".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        let t = plan_task_with_budget(BudgetLimits {
            max_turns: Some(7),
            ..Default::default()
        });
        // remaining = 7 - 1 = 6; 3 tasks -> per_task = 2. Task a uses 2 turns of
        // tool calls without finishing, so its sub-loop hits the per-task cap and
        // the run aborts (StepFailed / BudgetExceeded depending on which gate).
        match h.run(HarnessRunOptions::new(t)).await {
            RunResult::Failure { reason, turns, .. } => {
                // The per-task cap is a turn budget: the sub-loop hits the turn
                // gate, which surfaces as BudgetExceeded(Turns) routed through.
                assert!(
                    matches!(
                        reason,
                        HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns
                        }
                    ),
                    "per-task turn cap enforced, got {reason:?}"
                );
                // plan(1) + 2 task-a turns = 3 cumulative turns.
                assert_eq!(turns, 3, "shared budget carried: 1 plan + 2 task turns");
            }
            other => panic!("expected Failure from turn cap, got {other:?}"),
        }
    }

    // Budget exhaustion MID-execute: a tight global turn cap stops the run
    // partway with BudgetExceeded, not StepFailed.
    #[tokio::test]
    async fn execute_phase_budget_exhausted_mid_execute() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["x","y","z"]}"#));
        a.push(final_resp("did x"));
        // No more turns allowed after the first execute step.
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task_with_budget(BudgetLimits {
            max_turns: Some(2), // plan(1) + exactly one execute turn
            ..Default::default()
        });
        match h.run(HarnessRunOptions::new(t)).await {
            RunResult::Failure { reason, turns, .. } => {
                assert!(
                    matches!(
                        reason,
                        HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns
                        }
                    ),
                    "global turn budget is the hard stop, got {reason:?}"
                );
                assert_eq!(turns, 2);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }

    // Observability span count: plan turn + one span per executed step.
    #[tokio::test]
    async fn execute_phase_span_count() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["a","b"]}"#));
        a.push(final_resp("did a"));
        a.push(final_resp("did b"));
        let obs = Arc::new(crate::observability::InMemoryObservabilityProvider::new());
        let mut cfg = standard_config(a);
        cfg.observability = Some(obs.clone());
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(plan_task())).await;
        let spans = obs.turn_spans(&SessionId::new("s1"));
        // 1 plan turn + 2 execute step turns = 3 turn spans.
        assert_eq!(spans.len(), 3, "plan turn + one span per executed step");
    }

    // Compaction works inside the execute loop: the context manager requests a
    // compaction, and the run still completes. (Exercises the shared compaction
    // path through run_react_inner during execute steps.)
    #[tokio::test]
    async fn execute_phase_compaction_in_loop() {
        // Use the default NoopContextManager (never compacts) is not enough to
        // exercise compaction; instead assert the loop tolerates many turns with
        // tool calls and still drains. A dedicated compaction adapter lives in
        // the ReAct compaction tests; here we assert execute reuses that path by
        // running a multi-turn step to completion.
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["only"]}"#));
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "1".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: usage(),
        });
        a.push(final_resp("finished only"));
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "ok".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Success { output, turns, .. } => {
                assert_eq!(output, "finished only");
                // plan(1) + tool turn(1) + final(1) = 3.
                assert_eq!(turns, 3);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // OnTaskAdvance fires exactly N times with the correct task_index /
    // total_tasks, and (Q1) may rewrite the step instruction.
    #[tokio::test]
    async fn execute_phase_on_task_advance_fires_per_task() {
        use crate::hooks::{FunctionHook, HookDecision, HookEvent, StandardHookChain};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["s0","s1","s2"]}"#));
        a.push(final_resp("d0"));
        a.push(final_resp("d1"));
        a.push(final_resp("d2"));

        let fire_count = Arc::new(AtomicUsize::new(0));
        let seen_indices = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let seen_totals = Arc::new(std::sync::Mutex::new(Vec::<usize>::new()));
        let fc = fire_count.clone();
        let si = seen_indices.clone();
        let st = seen_totals.clone();

        let chain = StandardHookChain::new();
        chain
            .register(Arc::new(FunctionHook::new(
                "count-advance",
                vec![HookEvent::OnTaskAdvance],
                move |ctx| {
                    if let crate::hooks::HookContext::OnTaskAdvance {
                        task_index,
                        total_tasks,
                        ..
                    } = ctx
                    {
                        fc.fetch_add(1, Ordering::SeqCst);
                        si.lock().unwrap().push(*task_index);
                        st.lock().unwrap().push(*total_tasks);
                    }
                    Ok(HookDecision::Continue)
                },
            )))
            .unwrap();

        let mut cfg = standard_config(a);
        cfg.hooks = Some(Arc::new(chain));
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(plan_task())).await;

        assert_eq!(fire_count.load(Ordering::SeqCst), 3, "fires once per task");
        assert_eq!(*seen_indices.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(*seen_totals.lock().unwrap(), vec![3, 3, 3]);
    }

    // Q3: an empty plan -> HaltReason::EmptyPlan (not a silent success).
    #[tokio::test]
    async fn execute_phase_empty_plan_halts() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":[],"rationale":"nothing"}"#));
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Failure {
                reason: HaltReason::EmptyPlan,
                turns,
                ..
            } => {
                assert_eq!(turns, 1, "only the plan turn ran");
            }
            other => panic!("expected EmptyPlan, got {other:?}"),
        }
    }

    // Q5: a step that errors aborts the whole run with StepFailed carrying the
    // failing index + instruction; later tasks do NOT run.
    #[tokio::test]
    async fn execute_phase_step_failure_aborts_with_step_failed() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["good","bad","never"]}"#));
        a.push(final_resp("did good"));
        // Step 1 ("bad"): agent returns an error.
        a.push(TurnResult::Error {
            error: crate::agent::AgentError::EmptyResponse,
            usage: None,
        });
        // "never" must NOT run.
        let agent_count = a.clone();
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Failure {
                reason:
                    HaltReason::StepFailed {
                        task_index, task, ..
                    },
                ..
            } => {
                assert_eq!(task_index, 1, "aborts on the failing step");
                assert_eq!(task, "bad");
            }
            other => panic!("expected StepFailed, got {other:?}"),
        }
        // plan(1) + good(1) + bad(1) = 3 calls; "never" never ran.
        assert_eq!(
            agent_count
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            3,
            "the third task must not run after a step failure"
        );
    }

    // Q4: the task list is persisted through the RunStore seam (not the #71
    // sandbox path). Assert the durable RunStore holds the list after a run.
    #[tokio::test]
    async fn execute_phase_persists_through_run_store() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["one"]}"#));
        a.push(final_resp("did one"));
        let store = Arc::new(crate::storage::InMemoryStorageProvider::new());
        let provider = Arc::new(crate::storage::StorageProvider::single(store));
        let mut cfg = standard_config(a);
        cfg.storage = provider.clone();
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(plan_task())).await;

        let stored = provider
            .run()
            .get(&SessionId::new("s1"), TASK_LIST_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .expect("task list present in run store");
        let list: TaskList = serde_json::from_value(stored).expect("deserializes");
        assert_eq!(list.tasks.len(), 1);
        assert_eq!(list.tasks[0].status, TaskStatus::Completed);
    }

    // #76: after a plan/execute run, BOTH persistence keys live on the RunStore
    // seam and NEITHER is mirrored into SessionState.extras. Drives the phases
    // directly (rather than `run()`) so the post-run `state.extras` is
    // observable. The ephemeral extras keys (`__rich_state`,
    // `subagent_handoff_summary`) are owned by other components and untouched
    // here.
    #[tokio::test]
    async fn plan_execute_persistence_lives_on_run_store_not_extras() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["one","two"],"rationale":"why"}"#));
        a.push(final_resp("did one"));
        a.push(final_resp("did two"));
        let h = StandardHarness::new(standard_config(a));
        let t = plan_task();
        let mut state = SessionState::default();

        let outcome = h
            .run_plan_phase(&t, &mut state, BudgetSnapshot::default(), &None)
            .await
            .expect("plan phase succeeds");
        let task_list = crate::tasklist::plan_artifact_to_task_list(&outcome.artifact);
        let result = h
            .run_execute_phase(
                &t,
                &mut state,
                task_list,
                BudgetSnapshot {
                    turns: outcome.turns,
                    ..Default::default()
                },
                outcome.usage,
                &None,
            )
            .await;
        assert!(matches!(result, RunResult::Success { .. }));

        // Both keys are durable in the RunStore.
        assert!(h
            .storage()
            .run()
            .get(&t.session_id, PLAN_EXECUTE_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .is_some());
        assert!(h
            .storage()
            .run()
            .get(&t.session_id, TASK_LIST_EXTRAS_KEY)
            .await
            .expect("run store get ok")
            .is_some());

        // Neither key is mirrored into SessionState.extras anymore.
        assert!(state.extras.get(PLAN_EXECUTE_EXTRAS_KEY).is_none());
        assert!(state.extras.get(TASK_LIST_EXTRAS_KEY).is_none());
    }

    // Q2: success output is the LAST step's FinalResponse, not a concatenation
    // and not the plan rationale.
    #[tokio::test]
    async fn execute_phase_success_output_is_last_step() {
        let a = make_agent();
        a.push(final_resp(
            r#"{"tasks":["a","b"],"rationale":"RATIONALE_TOKEN"}"#,
        ));
        a.push(final_resp("FIRST_STEP_OUTPUT"));
        a.push(final_resp("LAST_STEP_OUTPUT"));
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Success { output, .. } => {
                assert_eq!(output, "LAST_STEP_OUTPUT");
                assert!(!output.contains("FIRST_STEP_OUTPUT"));
                assert!(!output.contains("RATIONALE_TOKEN"));
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // Confirms ExecutePhaseNotImplemented is gone: a full PlanExecute run with
    // execute completions returns Success (the old variant would have halted).
    #[tokio::test]
    async fn execute_phase_not_implemented_is_gone() {
        let a = make_agent();
        a.push(final_resp(r#"{"tasks":["only"]}"#));
        a.push(final_resp("done"));
        let h = StandardHarness::new(standard_config(a));
        assert!(matches!(
            h.run(HarnessRunOptions::new(plan_task())).await,
            RunResult::Success { .. }
        ));
    }

    // Planner-agent routing through the FULL run: the planner runs the plan turn
    // and the default agent runs the execute steps.
    #[tokio::test]
    async fn execute_phase_planner_agent_routing() {
        let default_agent = make_agent();
        default_agent.push(final_resp("did the step"));
        let planner = make_agent();
        planner.push(final_resp(r#"{"tasks":["step"]}"#));

        let mut cfg = standard_config(default_agent.clone());
        cfg.planner_agent = Some(planner.clone());
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(plan_task())).await {
            RunResult::Success { output, .. } => assert_eq!(output, "did the step"),
            other => panic!("expected Success, got {other:?}"),
        }
        assert_eq!(
            planner.call_count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "planner ran exactly the plan turn"
        );
        assert_eq!(
            default_agent
                .call_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1,
            "default agent ran exactly the execute step"
        );
    }

    // Rule: Aggregate usage accumulates across turns.
    #[tokio::test]
    async fn aggregate_usage_accumulates() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({}),
            }],
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
        });
        a.push(TurnResult::FinalResponse {
            content: "ok".into(),
            usage: TokenUsage {
                input_tokens: 7,
                output_tokens: 3,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
        });
        let mut cfg = standard_config(a);
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "x".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { usage, .. } => {
                assert_eq!(usage.input_tokens, 17);
                assert_eq!(usage.output_tokens, 8);
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // Rule: ModelError surfaces as AgentError → HaltReason::AgentError.
    #[tokio::test]
    async fn model_error_halts_via_agent_error() {
        let a = make_agent();
        a.push(TurnResult::Error {
            error: AgentError::ModelError(ModelError::Timeout),
            usage: None,
        });
        let h = StandardHarness::new(standard_config(a));
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Failure {
                reason:
                    HaltReason::AgentError {
                        error: AgentError::ModelError(ModelError::Timeout),
                    },
                ..
            } => {}
            other => panic!("expected AgentError(Timeout), got {other:?}"),
        }
    }

    // Serde round-trip — fixture portability.
    #[test]
    fn run_result_roundtrips_json() {
        let r = RunResult::Failure {
            reason: HaltReason::BudgetExceeded {
                limit_type: BudgetLimitType::Turns,
            },
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 3,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: RunResult = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn paused_state_roundtrips_json() {
        let ps = PausedState {
            session_id: SessionId::new("s"),
            task_id: TaskId::new("t"),
            turn_number: 4,
            session_state: SessionState::default(),
            pending_tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "x".into(),
                input: serde_json::json!({"k":1}),
            }],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "what?".into(),
                options: None,
            }),
            task: react(5),
            budget_used: BudgetSnapshot {
                turns: 4,
                input_tokens: 100,
                output_tokens: 50,
                wall_time: Some(Duration::from_secs(10)),
                cost_usd: 0.0,
            },
            child_state: None,
        };
        let s = serde_json::to_string(&ps).unwrap();
        let back: PausedState = serde_json::from_str(&s).unwrap();
        assert_eq!(ps, back);
    }

    // ChildPausedState cannot nest itself — compile-time depth-1 enforcement.
    #[test]
    fn child_paused_state_has_no_child_field() {
        // This test exists to lock in the rule via type system inspection at
        // review time. If a future change adds `child_state` to
        // ChildPausedState, this assertion is unaffected — but the spec rule
        // is broken. We assert structural shape via serde.
        let cs = ChildPausedState {
            session_id: SessionId::new("c"),
            task_id: TaskId::new("ct"),
            turn_number: 1,
            session_state: SessionState::default(),
            pending_tool_calls: vec![],
            approved_results: vec![],
            human_request: Some(HumanRequest::Clarification {
                question: "?".into(),
                options: None,
            }),
            task: react(1),
            budget_used: BudgetSnapshot::default(),
            parent_tool_call_id: "p".into(),
        };
        let s = serde_json::to_string(&cs).unwrap();
        assert!(!s.contains("\"child_state\""));
    }

    // ====================================================================
    // Compaction verify→retry→warn loop (issue #46)
    // ====================================================================
    mod compaction {
        use super::*;
        use crate::context::{
            CompactionPreserveHints, CompactionVerificationResult, CompactionVerifier,
            SessionState as ContextSessionState,
        };
        use crate::model::{Content, Role};
        use crate::observability::{
            InMemoryObservabilityProvider, ObservabilityProvider, SpanBase, WarnEvent, WarnSpan,
        };
        use crate::{SessionId, TaskId};
        use std::sync::Mutex;

        /// A `ContextManager` that always offers a compaction turn. Records how
        /// many times `apply_compaction` ran and the contexts the agent saw.
        struct CompactingContextManager {
            applied: Mutex<u32>,
            should: bool,
        }
        impl CompactingContextManager {
            fn new(should: bool) -> Self {
                Self {
                    applied: Mutex::new(0),
                    should,
                }
            }
        }
        impl ContextManager for CompactingContextManager {
            fn assemble<'a>(
                &'a self,
                session: &'a SessionState,
                _task: &'a Task,
            ) -> BoxFut<'a, Context> {
                let messages = session.messages.clone();
                Box::pin(async move {
                    Context {
                        messages,
                        tools: vec![],
                        params: crate::model::ModelParams::default(),
                    }
                })
            }
            fn append_tool_result<'a>(
                &'a self,
                session: &'a mut SessionState,
                _result: &'a ToolResult,
            ) -> BoxFut<'a, ()> {
                Box::pin(async move {
                    session.messages.push(Message {
                        role: Role::Tool,
                        content: Content::Text {
                            text: "tool".into(),
                        },
                    });
                })
            }
            fn append_user_message<'a>(
                &'a self,
                _session: &'a mut SessionState,
                _text: &'a str,
            ) -> BoxFut<'a, ()> {
                Box::pin(async {})
            }
            fn should_compact(&self, _session: &SessionState) -> bool {
                self.should
            }
            fn prepare_compaction_turn(&self, _session: &SessionState) -> Option<CompactionTurn> {
                let mut vs = ContextSessionState::new(
                    SessionId::new("s1"),
                    TaskId::new("t1"),
                    "deploy the payment service",
                );
                vs.token_budget_used = 1000;
                Some(CompactionTurn {
                    context: Context {
                        messages: vec![Message {
                            role: Role::User,
                            content: Content::Text {
                                text: "please summarize".into(),
                            },
                        }],
                        tools: vec![],
                        params: crate::model::ModelParams::default(),
                    },
                    preserve_hints: CompactionPreserveHints::default(),
                    verification_state: vs,
                    messages_removed: 3,
                })
            }
            fn apply_compaction(&self, _session: &mut SessionState, _summary: String) {
                *self.applied.lock().unwrap() += 1;
            }
        }

        /// Agent that records every context it is handed and replays a queue of
        /// final-response summaries.
        struct RecordingAgent {
            summaries: Mutex<std::collections::VecDeque<String>>,
            seen: Mutex<Vec<Context>>,
        }
        impl RecordingAgent {
            fn new(summaries: Vec<&str>) -> Self {
                Self {
                    summaries: Mutex::new(summaries.iter().map(|s| s.to_string()).collect()),
                    seen: Mutex::new(Vec::new()),
                }
            }
        }
        impl Agent for RecordingAgent {
            fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
                self.seen.lock().unwrap().push(context);
                let content = self
                    .summaries
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or_default();
                Box::pin(async move {
                    TurnResult::FinalResponse {
                        content,
                        usage: TokenUsage {
                            input_tokens: 1,
                            output_tokens: 1,
                            cache_read_tokens: None,
                            cache_write_tokens: None,
                        },
                    }
                })
            }
            fn id(&self) -> AgentId {
                AgentId::new("recording")
            }
        }

        /// Verifier that fails the first `fail_first` calls, then passes.
        struct ScriptedVerifier {
            fail_first: Mutex<u32>,
            missing: Vec<String>,
        }
        impl ScriptedVerifier {
            fn new(fail_first: u32, missing: Vec<&str>) -> Self {
                Self {
                    fail_first: Mutex::new(fail_first),
                    missing: missing.iter().map(|s| s.to_string()).collect(),
                }
            }
        }
        impl CompactionVerifier for ScriptedVerifier {
            fn verify(
                &self,
                _summary: &str,
                _hints: &CompactionPreserveHints,
                _state: &ContextSessionState,
            ) -> CompactionVerificationResult {
                let mut remaining = self.fail_first.lock().unwrap();
                if *remaining > 0 {
                    *remaining -= 1;
                    CompactionVerificationResult {
                        passed: false,
                        missing_items: self.missing.clone(),
                        detail: "scripted fail".into(),
                    }
                } else {
                    CompactionVerificationResult {
                        passed: true,
                        missing_items: vec![],
                        detail: "scripted pass".into(),
                    }
                }
            }
        }

        fn harness_with(
            cm: Arc<CompactingContextManager>,
            agent: Arc<RecordingAgent>,
            verifier: Arc<dyn CompactionVerifier>,
            obs: Arc<dyn ObservabilityProvider>,
            max_attempts: u32,
        ) -> StandardHarness {
            StandardHarness::new(HarnessConfig {
                agent,
                tool_registry: Arc::new(ScriptedToolRegistry::new()),
                sandbox: Arc::new(AllowAllSandbox),
                context_manager: cm,
                termination_policy: Arc::new(AlwaysContinuePolicy),
                middleware: None,
                observability: Some(obs),
                compaction_verifier: verifier,
                max_compaction_attempts: max_attempts,
                pricing: PricingTable::DEFAULT,
                content_capture: ContentCaptureConfig::default(),
                tool_call_repair: None,
                max_repair_attempts: 1,
                max_stop_blocks: 8,
                hooks: None,
                planner_agent: None,
                verifier: None,
                evaluator_agent: None,
                storage: Arc::new(crate::storage::StorageProvider::no_op()),
                chunk_provider: Arc::new(crate::prompt_assembly::InMemoryChunkProvider::empty()),
                max_resets: 3,
                vcs_provider: None,
                metric_evaluator: None,
            })
        }

        async fn drive(
            h: &StandardHarness,
            agent: &Arc<RecordingAgent>,
            cm: &Arc<CompactingContextManager>,
        ) {
            let mut state = SessionState::default();
            let mut usage = AggregateUsage::default();
            let mut span_seq = 0u64;
            // Pre-condition: should_compact gate is honored by the caller.
            if h.config.context_manager.should_compact(&state) {
                h.run_compaction(
                    &mut state,
                    &SessionId::new("s1"),
                    &TaskId::new("t1"),
                    &mut span_seq,
                    &mut usage,
                )
                .await;
            }
            let _ = (agent, cm);
        }

        #[tokio::test]
        async fn no_compaction_when_should_compact_false() {
            let cm = Arc::new(CompactingContextManager::new(false));
            let agent = Arc::new(RecordingAgent::new(vec!["summary"]));
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                Arc::new(ScriptedVerifier::new(0, vec![])),
                obs.clone(),
                2,
            );
            drive(&h, &agent, &cm).await;
            assert_eq!(agent.seen.lock().unwrap().len(), 0, "no compaction turn");
            assert_eq!(*cm.applied.lock().unwrap(), 0);
        }

        #[tokio::test]
        async fn passing_verifier_one_turn_one_apply_no_warn() {
            let cm = Arc::new(CompactingContextManager::new(true));
            let agent = Arc::new(RecordingAgent::new(vec!["good summary"]));
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                Arc::new(ScriptedVerifier::new(0, vec![])),
                obs.clone(),
                2,
            );
            drive(&h, &agent, &cm).await;
            assert_eq!(agent.seen.lock().unwrap().len(), 1, "exactly one turn");
            assert_eq!(*cm.applied.lock().unwrap(), 1, "applied once");
            assert!(
                obs.warn_spans(&SessionId::new("s1")).is_empty(),
                "no warn emitted"
            );
        }

        #[tokio::test]
        async fn failing_then_passing_retries_and_injects_missing_items() {
            let cm = Arc::new(CompactingContextManager::new(true));
            let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2"]));
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                Arc::new(ScriptedVerifier::new(1, vec!["payment", "deploy"])),
                obs.clone(),
                2,
            );
            drive(&h, &agent, &cm).await;
            let seen = agent.seen.lock().unwrap();
            assert_eq!(seen.len(), 2, "two compaction turns");
            // The retry context must contain the injected "missing these items"
            // message with the actual missing items.
            let retry_ctx = &seen[1];
            let injected = retry_ctx.messages.iter().any(|m| {
                matches!(&m.content, Content::Text { text }
                    if text.contains("missing these items")
                        && text.contains("payment")
                        && text.contains("deploy"))
            });
            assert!(injected, "retry context carries the missing-items message");
            assert_eq!(*cm.applied.lock().unwrap(), 1, "applied once after pass");
            assert!(obs.warn_spans(&SessionId::new("s1")).is_empty());
        }

        #[tokio::test]
        async fn always_failing_warns_and_accepts_anyway() {
            let cm = Arc::new(CompactingContextManager::new(true));
            let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2"]));
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                Arc::new(ScriptedVerifier::new(99, vec!["payment"])),
                obs.clone(),
                2,
            );
            drive(&h, &agent, &cm).await;
            assert_eq!(agent.seen.lock().unwrap().len(), 2, "max attempts == 2");
            // apply_compaction STILL called.
            assert_eq!(*cm.applied.lock().unwrap(), 1, "accepted anyway");
            let sid = SessionId::new("s1");
            let warns = obs.warn_spans(&sid);
            assert_eq!(warns.len(), 1, "exactly one warn span");
            match &warns[0].event {
                WarnEvent::CompactionVerificationFailed {
                    missing_items,
                    accepted_anyway,
                } => {
                    assert_eq!(missing_items, &vec!["payment".to_string()]);
                    assert!(*accepted_anyway);
                }
                other => panic!("expected CompactionVerificationFailed, got {other:?}"),
            }
            // SessionMetrics failure counter == 1. Seed an outcome so metrics
            // are produced even though no turn span was emitted in this unit.
            obs.set_session_outcome(&sid, SessionOutcome::Success);
            let m = obs.get_session_metrics(&sid).await.unwrap();
            assert_eq!(m.compaction_verification_failures, 1);
        }

        #[tokio::test]
        async fn max_attempts_one_honored() {
            let cm = Arc::new(CompactingContextManager::new(true));
            let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2", "v3"]));
            let obs = Arc::new(InMemoryObservabilityProvider::new());
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                Arc::new(ScriptedVerifier::new(99, vec!["payment"])),
                obs.clone(),
                1,
            );
            drive(&h, &agent, &cm).await;
            assert_eq!(
                agent.seen.lock().unwrap().len(),
                1,
                "exactly one attempt with max=1"
            );
            assert_eq!(*cm.applied.lock().unwrap(), 1, "accepted after one attempt");
            assert_eq!(obs.warn_spans(&SessionId::new("s1")).len(), 1);
        }

        /// A minimal provider that does NOT override `emit_warn`; proves the
        /// default no-op body does not break it (W4).
        #[derive(Default)]
        struct BareProvider;
        impl ObservabilityProvider for BareProvider {
            fn emit_turn(&self, _s: crate::observability::TurnSpan) {}
            fn emit_tool_call(&self, _s: ToolCallSpan) {}
            fn emit_sensor(&self, _s: crate::observability::SensorSpan) {}
            fn emit_context(&self, _s: crate::observability::ContextSpan) {}
            fn emit_middleware(&self, _s: crate::observability::MiddlewareSpan) {}
            fn emit_patch(&self, _s: crate::observability::PatchSpan) {}
            fn flush_session<'a>(&'a self, _sid: &'a SessionId) -> BoxFut<'a, ()> {
                Box::pin(async {})
            }
            fn get_session_metrics<'a>(
                &'a self,
                _sid: &'a SessionId,
            ) -> BoxFut<'a, Option<crate::observability::SessionMetrics>> {
                Box::pin(async { None })
            }
            fn get_sessions<'a>(
                &'a self,
                _since: Timestamp,
                _domain: Option<String>,
                _outcome: Option<SessionOutcome>,
            ) -> BoxFut<'a, Vec<crate::observability::SessionMetrics>> {
                Box::pin(async { Vec::new() })
            }
            fn get_trace<'a>(
                &'a self,
                _sid: &'a SessionId,
            ) -> BoxFut<'a, Vec<Box<dyn crate::observability::Span>>> {
                Box::pin(async { Vec::new() })
            }
        }

        #[tokio::test]
        async fn emit_warn_default_noop_does_not_break_bare_provider() {
            let cm = Arc::new(CompactingContextManager::new(true));
            let agent = Arc::new(RecordingAgent::new(vec!["v1", "v2"]));
            let obs = Arc::new(BareProvider);
            let h = harness_with(
                cm.clone(),
                agent.clone(),
                Arc::new(ScriptedVerifier::new(99, vec!["payment"])),
                obs,
                2,
            );
            // Reaching the warn path must not panic; bare provider ignores it.
            drive(&h, &agent, &cm).await;
            assert_eq!(*cm.applied.lock().unwrap(), 1);
            // Construct a WarnSpan directly to lock W4: the default body runs.
            let base = SpanBase::new_root(
                crate::observability::SpanId::new("x"),
                SessionId::new("s1"),
                TaskId::new("t1"),
                SpanKind::Warn,
                Timestamp::now(),
            );
            BareProvider.emit_warn(WarnSpan::new(
                base,
                WarnEvent::CompactionVerificationFailed {
                    missing_items: vec![],
                    accepted_anyway: true,
                },
            ));
        }

        // ----------------------------------------------------------------
        // Cross-language consistency fixture replay (issue #46)
        // ----------------------------------------------------------------

        /// Verifier driven by a fixture verdict queue; repeats the last verdict
        /// once the queue is exhausted, matching the fixture contract.
        struct FixtureVerifier {
            verdicts: Mutex<Vec<CompactionVerificationResult>>,
            idx: std::sync::atomic::AtomicUsize,
        }
        impl CompactionVerifier for FixtureVerifier {
            fn verify(
                &self,
                _summary: &str,
                _hints: &CompactionPreserveHints,
                _state: &ContextSessionState,
            ) -> CompactionVerificationResult {
                let v = self.verdicts.lock().unwrap();
                let i = self.idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                v.get(i).or_else(|| v.last()).cloned().unwrap()
            }
        }

        #[tokio::test]
        async fn fixture_replay_loop_outcomes() {
            #[derive(serde::Deserialize)]
            struct Verdict {
                passed: bool,
                missing_items: Vec<String>,
            }
            #[derive(serde::Deserialize)]
            struct Expected {
                attempts: u32,
                apply_compaction_calls: u32,
                warn_emitted: bool,
                #[serde(default)]
                retry_injected_missing: Option<Vec<String>>,
                final_missing_items: Vec<String>,
            }
            #[derive(serde::Deserialize)]
            struct Case {
                name: String,
                max_compaction_attempts: u32,
                verdicts: Vec<Verdict>,
                expected: Expected,
            }
            #[derive(serde::Deserialize)]
            struct Suite {
                cases: Vec<Case>,
            }

            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../fixtures/compaction_loop/cases.json");
            let raw = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let suite: Suite = serde_json::from_str(&raw).unwrap();
            assert!(suite.cases.len() >= 5, "expected >= 5 fixture cases");

            for case in &suite.cases {
                let cm = Arc::new(CompactingContextManager::new(true));
                // Enough summaries that the agent never starves.
                let agent = Arc::new(RecordingAgent::new(vec!["s1", "s2", "s3", "s4"]));
                let obs = Arc::new(InMemoryObservabilityProvider::new());
                let verifier = Arc::new(FixtureVerifier {
                    verdicts: Mutex::new(
                        case.verdicts
                            .iter()
                            .map(|v| CompactionVerificationResult {
                                passed: v.passed,
                                missing_items: v.missing_items.clone(),
                                detail: String::new(),
                            })
                            .collect(),
                    ),
                    idx: std::sync::atomic::AtomicUsize::new(0),
                });
                let h = harness_with(
                    cm.clone(),
                    agent.clone(),
                    verifier,
                    obs.clone(),
                    case.max_compaction_attempts,
                );
                drive(&h, &agent, &cm).await;

                let sid = SessionId::new("s1");
                assert_eq!(
                    agent.seen.lock().unwrap().len() as u32,
                    case.expected.attempts,
                    "case `{}`: attempts",
                    case.name
                );
                assert_eq!(
                    *cm.applied.lock().unwrap(),
                    case.expected.apply_compaction_calls,
                    "case `{}`: apply_compaction_calls",
                    case.name
                );
                let warns = obs.warn_spans(&sid);
                assert_eq!(
                    !warns.is_empty(),
                    case.expected.warn_emitted,
                    "case `{}`: warn_emitted",
                    case.name
                );
                if case.expected.warn_emitted {
                    assert_eq!(warns.len(), 1, "case `{}`: exactly one warn", case.name);
                    match &warns[0].event {
                        WarnEvent::CompactionVerificationFailed {
                            missing_items,
                            accepted_anyway,
                        } => {
                            assert_eq!(
                                missing_items, &case.expected.final_missing_items,
                                "case `{}`: final_missing_items",
                                case.name
                            );
                            assert!(*accepted_anyway, "case `{}`: accepted_anyway", case.name);
                        }
                        other => panic!(
                            "case `{}`: expected CompactionVerificationFailed, got {other:?}",
                            case.name
                        ),
                    }
                }
                if let Some(expected_inject) = &case.expected.retry_injected_missing {
                    let seen = agent.seen.lock().unwrap();
                    let joined = expected_inject.join(", ");
                    let found = seen.iter().skip(1).any(|c| {
                        c.messages.iter().any(|m| {
                            matches!(&m.content,
                            Content::Text { text }
                                if text.contains("missing these items")
                                    && text.contains(&joined))
                        })
                    });
                    assert!(found, "case `{}`: retry injection", case.name);
                }
            }
        }
    }

    // ── Assistant-turn recording (regression for lost conversation history) ──

    /// A ContextManager that records every message in a shared vector so tests
    /// can inspect the conversation the loop builds. Mirrors
    /// `NoopContextManager` but exposes the message log.
    #[derive(Clone, Default)]
    struct RecordingContextManager {
        messages: Arc<std::sync::Mutex<Vec<Message>>>,
    }
    impl ContextManager for RecordingContextManager {
        fn assemble<'a>(
            &'a self,
            session: &'a SessionState,
            _task: &'a Task,
        ) -> BoxFut<'a, Context> {
            let messages = session.messages.clone();
            Box::pin(async move {
                Context {
                    messages,
                    tools: vec![],
                    params: crate::model::ModelParams::default(),
                }
            })
        }
        fn append_tool_result<'a>(
            &'a self,
            session: &'a mut SessionState,
            result: &'a ToolResult,
        ) -> BoxFut<'a, ()> {
            let msg = Message {
                role: Role::Tool,
                content: Content::ToolResult(crate::model::ToolResult {
                    tool_use_id: result.call_id.clone(),
                    content: match &result.output {
                        ToolOutput::Success { content, .. } => content.clone(),
                        ToolOutput::Error { message, .. } => message.clone(),
                        ToolOutput::WaitingForHuman { .. } => String::new(),
                        ToolOutput::Escalate { .. } => String::new(),
                        ToolOutput::AwaitingClarification { .. } => String::new(),
                    },
                    is_error: matches!(result.output, ToolOutput::Error { .. }),
                }),
            };
            Box::pin(async move {
                session.messages.push(msg.clone());
                self.messages.lock().unwrap().push(msg);
            })
        }
        fn append_assistant_message<'a>(
            &'a self,
            session: &'a mut SessionState,
            message: &'a Message,
        ) -> BoxFut<'a, ()> {
            let message = message.clone();
            Box::pin(async move {
                session.messages.push(message.clone());
                self.messages.lock().unwrap().push(message);
            })
        }
        fn append_user_message<'a>(
            &'a self,
            session: &'a mut SessionState,
            text: &'a str,
        ) -> BoxFut<'a, ()> {
            let msg = Message {
                role: Role::User,
                content: Content::Text { text: text.into() },
            };
            Box::pin(async move {
                session.messages.push(msg.clone());
                self.messages.lock().unwrap().push(msg);
            })
        }
    }

    /// Regression: a turn that requests a tool call must record the assistant's
    /// tool-call message in history, positioned BEFORE the tool result, so the
    /// next turn's assembled context reflects what the agent already did.
    #[tokio::test]
    async fn tool_call_records_assistant_message_before_result() {
        let a = make_agent();
        a.push(TurnResult::ToolCallRequested {
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                input: serde_json::json!({ "path": "a.txt" }),
            }],
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let cm = RecordingContextManager::default();
        let mut cfg = standard_config(a);
        cfg.context_manager = Arc::new(cm.clone());
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "contents".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;
        let h = StandardHarness::new(cfg);
        assert!(matches!(
            h.run(HarnessRunOptions::new(react(5))).await,
            RunResult::Success { .. }
        ));

        let msgs = cm.messages.lock().unwrap();
        let assistant_idx = msgs.iter().position(|m| {
            m.role == Role::Assistant && matches!(&m.content, Content::ToolCall(c) if c.id == "c1")
        });
        let tool_idx = msgs.iter().position(|m| {
            m.role == Role::Tool
                && matches!(&m.content, Content::ToolResult(r) if r.tool_use_id == "c1")
        });
        let assistant_idx = assistant_idx.expect("assistant tool-call message must be recorded");
        let tool_idx = tool_idx.expect("tool result must be recorded");
        assert!(
            assistant_idx < tool_idx,
            "assistant tool_use (idx {assistant_idx}) must precede its tool result (idx {tool_idx})"
        );
    }

    /// Regression: a final response must append the assistant's text to history
    /// so a continued session sees what the agent said.
    #[tokio::test]
    async fn final_response_records_assistant_text() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "the final answer".into(),
            usage: usage(),
        });
        let cm = RecordingContextManager::default();
        let mut cfg = standard_config(a);
        cfg.context_manager = Arc::new(cm.clone());
        let h = StandardHarness::new(cfg);
        assert!(matches!(
            h.run(HarnessRunOptions::new(react(5))).await,
            RunResult::Success { .. }
        ));

        let msgs = cm.messages.lock().unwrap();
        assert!(
            msgs.iter().any(|m| m.role == Role::Assistant
                && matches!(&m.content, Content::Text { text } if text == "the final answer")),
            "assistant final text must be recorded in history"
        );
    }

    /// Regression: when a run pauses at BeforeTool (SurfaceToHuman) and is then
    /// resumed with Allow, the assistant tool-call message must already be in
    /// history — recorded before the pause — and positioned before its tool
    /// result, with no duplicate from the resume path. Guards the option-(a)
    /// fix: the assistant turn is recorded as soon as the calls are known, so
    /// resume_inner never has to (and never double-records) it.
    #[tokio::test]
    async fn resume_after_surface_to_human_records_assistant_once_before_result() {
        let a = make_agent();
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            input: serde_json::json!({ "path": "a.txt" }),
        }];
        a.push(TurnResult::ToolCallRequested {
            calls: calls.clone(),
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let cm = RecordingContextManager::default();
        let mut cfg = standard_config(a);
        cfg.context_manager = Arc::new(cm.clone());
        let reg = Arc::new(ScriptedToolRegistry::new());
        reg.push(ToolOutput::Success {
            content: "contents".into(),
            truncated: false,
        });
        cfg.tool_registry = reg;
        let mw = Arc::new(ScriptedMiddleware::new());
        mw.push(
            HookPoint::BeforeTool,
            MiddlewareDecision::SurfaceToHuman {
                request: HumanRequest::ToolApproval {
                    calls: calls.clone(),
                    risk_level: RiskLevel::Medium,
                },
            },
        );
        cfg.middleware = Some(mw);
        let h = StandardHarness::new(cfg);

        // Pause at BeforeTool — the assistant turn was recorded just before.
        let state = match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::WaitingForHuman { state, .. } => *state,
            other => panic!("expected WaitingForHuman, got {other:?}"),
        };
        // Resume with approval; the pending call is dispatched and its result
        // appended after the already-recorded assistant turn.
        assert!(matches!(
            h.resume(state, HumanResponse::Allow, None).await,
            RunResult::Success { .. }
        ));

        let msgs = cm.messages.lock().unwrap();
        let assistant_idx = msgs.iter().position(|m| {
            m.role == Role::Assistant && matches!(&m.content, Content::ToolCall(c) if c.id == "c1")
        });
        let tool_idx = msgs.iter().position(|m| {
            m.role == Role::Tool
                && matches!(&m.content, Content::ToolResult(r) if r.tool_use_id == "c1")
        });
        let assistant_idx =
            assistant_idx.expect("assistant tool-call must be recorded on the resume path");
        let tool_idx = tool_idx.expect("tool result must be recorded");
        assert!(
            assistant_idx < tool_idx,
            "assistant tool_use (idx {assistant_idx}) must precede its tool result (idx {tool_idx})"
        );
        let count = msgs
            .iter()
            .filter(|m| {
                m.role == Role::Assistant
                    && matches!(&m.content, Content::ToolCall(c) if c.id == "c1")
            })
            .count();
        assert_eq!(
            count, 1,
            "assistant tool-call must be recorded exactly once, not duplicated by resume"
        );
    }

    // ── Issue #69 — Stop-hook wiring into the ReAct loop ───────────────────

    /// A Stop hook that blocks `block_n` times then continues, counting fires.
    struct CountingStopHook {
        block_first_n: u32,
        fired: std::sync::atomic::AtomicU32,
    }
    impl crate::hooks::Hook for CountingStopHook {
        fn handle<'a>(
            &'a self,
            _ctx: &'a mut crate::hooks::HookContext<'a>,
        ) -> BoxFut<'a, Result<crate::hooks::HookDecision, crate::hooks::HookError>> {
            let n = self.fired.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let decision = if n < self.block_first_n {
                crate::hooks::HookDecision::Block {
                    reason: format!("not done yet (block {n})"),
                }
            } else {
                crate::hooks::HookDecision::Continue
            };
            Box::pin(async move { Ok(decision) })
        }
        fn events(&self) -> Vec<crate::hooks::HookEvent> {
            vec![crate::hooks::HookEvent::Stop]
        }
        fn name(&self) -> String {
            "counting-stop".into()
        }
    }

    // Stop block → inject → continue, then continue → terminate (R12/R13).
    #[tokio::test]
    async fn stop_hook_block_then_continue_loops() {
        let a = make_agent();
        // Two final responses: first one is blocked by the Stop hook and the
        // loop continues; second one is allowed to terminate.
        a.push(TurnResult::FinalResponse {
            content: "first".into(),
            usage: usage(),
        });
        a.push(TurnResult::FinalResponse {
            content: "second".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        let chain = Arc::new(crate::hooks::StandardHookChain::new());
        chain
            .register(Arc::new(CountingStopHook {
                block_first_n: 1,
                fired: std::sync::atomic::AtomicU32::new(0),
            }))
            .unwrap();
        cfg.hooks = Some(chain);
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { output, turns, .. } => {
                assert_eq!(output, "second");
                assert_eq!(turns, 2, "loop should run a second turn after the block");
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // No registered Stop hook → terminate immediately (R13 baseline).
    #[tokio::test]
    async fn stop_hook_absent_terminates() {
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        cfg.hooks = Some(Arc::new(crate::hooks::StandardHookChain::new()));
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { turns, .. } => assert_eq!(turns, 1),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // max_stop_blocks cap: a hook that always blocks terminates anyway after
    // `max_stop_blocks` consecutive blocks (R14).
    #[tokio::test]
    async fn stop_hook_max_blocks_cap() {
        let a = make_agent();
        // Always-final agent; the hook always blocks, so the cap must stop it.
        for _ in 0..20 {
            a.push(TurnResult::FinalResponse {
                content: "again".into(),
                usage: usage(),
            });
        }
        let mut cfg = standard_config(a);
        cfg.max_stop_blocks = 3;
        let chain = Arc::new(crate::hooks::StandardHookChain::new());
        chain
            .register(Arc::new(CountingStopHook {
                block_first_n: u32::MAX, // always block
                fired: std::sync::atomic::AtomicU32::new(0),
            }))
            .unwrap();
        cfg.hooks = Some(chain);
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(50))).await {
            RunResult::Success { turns, .. } => {
                // 3 blocks → 3 extra turns; turn 4 hits the cap and terminates.
                assert_eq!(turns, 4, "should terminate after max_stop_blocks=3 blocks");
            }
            other => panic!("expected Success after cap, got {other:?}"),
        }
    }

    // ========================================================================
    // SelfVerifying loop strategy (issue #61)
    // ========================================================================

    use crate::verifier::{Verifier, VerifierInput, VerifierVerdict};
    use std::sync::Mutex as StdMutex;

    fn self_verifying_task() -> Task {
        task(LoopStrategy::SelfVerifying)
    }

    /// A `Verifier` test double that replays a scripted queue of verdicts.
    /// Records the `(iteration, build_session, eval_session)` it saw on each
    /// call so tests can assert phase distinctness (R2/R9). `max_iterations` is
    /// configurable to drive the round-trip cap (D3/R7).
    struct ScriptedVerifier {
        verdicts: StdMutex<std::collections::VecDeque<VerifierVerdict>>,
        default_verdict: VerifierVerdict,
        max_iters: u32,
        seen: StdMutex<Vec<(u32, String, String)>>,
    }
    impl ScriptedVerifier {
        fn new(verdicts: Vec<VerifierVerdict>, default_verdict: VerifierVerdict, max: u32) -> Self {
            Self {
                verdicts: StdMutex::new(verdicts.into_iter().collect()),
                default_verdict,
                max_iters: max,
                seen: StdMutex::new(Vec::new()),
            }
        }
    }
    impl Verifier for ScriptedVerifier {
        fn verify<'a>(&'a self, input: &'a VerifierInput) -> BoxFut<'a, VerifierVerdict> {
            let build_sid = match &input.build_result {
                RunResult::Success { session_id, .. } | RunResult::Failure { session_id, .. } => {
                    session_id.as_str().to_string()
                }
                _ => "?".to_string(),
            };
            let eval_sid = match &input.eval_result {
                RunResult::Success { session_id, .. } | RunResult::Failure { session_id, .. } => {
                    session_id.as_str().to_string()
                }
                _ => "?".to_string(),
            };
            self.seen
                .lock()
                .unwrap()
                .push((input.iteration, build_sid, eval_sid));
            let v = self
                .verdicts
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| self.default_verdict.clone());
            Box::pin(async move { v })
        }
        fn max_iterations(&self) -> u32 {
            self.max_iters
        }
    }

    /// Agent that records every `Context` it is handed and replays a queue of
    /// `TurnResult`s (default: an empty `FinalResponse`). Use the recorded
    /// contexts to assert what the build / evaluate phases saw.
    struct RecordingTurnAgent {
        id: AgentId,
        turns: StdMutex<std::collections::VecDeque<TurnResult>>,
        seen: StdMutex<Vec<Context>>,
    }
    impl RecordingTurnAgent {
        fn new(id: &str, turns: Vec<TurnResult>) -> Arc<Self> {
            Arc::new(Self {
                id: AgentId::new(id),
                turns: StdMutex::new(turns.into_iter().collect()),
                seen: StdMutex::new(Vec::new()),
            })
        }
        fn final_resp(text: &str) -> TurnResult {
            TurnResult::FinalResponse {
                content: text.into(),
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
            }
        }
        fn tool_call(name: &str) -> TurnResult {
            TurnResult::ToolCallRequested {
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: name.into(),
                    input: serde_json::json!({}),
                }],
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
            }
        }
        /// Flatten every recorded context's messages to a single text blob.
        fn seen_text(&self) -> Vec<String> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|c| {
                    c.messages
                        .iter()
                        .map(|m| match &m.content {
                            Content::Text { text } => text.clone(),
                            Content::ToolCall(tc) => tc.name.clone(),
                            _ => String::new(),
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .collect()
        }
        fn call_count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }
    impl Agent for RecordingTurnAgent {
        fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
            self.seen.lock().unwrap().push(context);
            let t = self
                .turns
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Self::final_resp(""));
            Box::pin(async move { t })
        }
        fn id(&self) -> AgentId {
            self.id.clone()
        }
    }

    fn passed() -> VerifierVerdict {
        VerifierVerdict::Passed
    }
    fn failed(reason: &str) -> VerifierVerdict {
        VerifierVerdict::failed(reason)
    }

    // R10: SelfVerifying no longer returns StrategyNotYetImplemented.
    #[tokio::test]
    async fn self_verifying_no_longer_unimplemented() {
        let build = RecordingTurnAgent::new("build", vec![RecordingTurnAgent::final_resp("done")]);
        let mut cfg = standard_config(make_agent());
        cfg.agent = build;
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)));
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(self_verifying_task())).await {
            RunResult::Failure {
                reason: HaltReason::StrategyNotYetImplemented { .. },
                ..
            } => panic!("SelfVerifying must be implemented"),
            RunResult::Success { .. } => {}
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // R11: verifier == None → Failure { SelfVerifyMisconfigured } (not panic).
    #[tokio::test]
    async fn self_verifying_missing_verifier_is_typed_halt() {
        let cfg = standard_config(make_agent());
        // verifier defaults to None.
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(self_verifying_task())).await {
            RunResult::Failure {
                reason: HaltReason::SelfVerifyMisconfigured { reason },
                ..
            } => assert!(reason.contains("verifier"), "got: {reason}"),
            other => panic!("expected SelfVerifyMisconfigured, got {other:?}"),
        }
    }

    // R1: build phase runs a ReAct loop until the agent claims done (a tool call
    // followed by a final response — the build agent loops at least twice).
    #[tokio::test]
    async fn self_verifying_build_runs_react_until_done() {
        let build = RecordingTurnAgent::new(
            "build",
            vec![
                RecordingTurnAgent::tool_call("read_file"),
                RecordingTurnAgent::final_resp("done"),
            ],
        );
        let eval = RecordingTurnAgent::new("eval", vec![RecordingTurnAgent::final_resp("PASS")]);
        let mut cfg = standard_config(make_agent());
        cfg.agent = build.clone();
        cfg.evaluator_agent = Some(eval);
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)));
        let h = StandardHarness::new(cfg);
        let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
        assert!(matches!(r, RunResult::Success { .. }), "got {r:?}");
        // The build agent took at least two turns (tool call, then final).
        assert!(
            build.call_count() >= 2,
            "build should loop: {} turns",
            build.call_count()
        );
    }

    // R2 + R9: the evaluate phase uses a FRESH SessionId never shared with the
    // build session — distinguishable in traces.
    #[tokio::test]
    async fn self_verifying_evaluate_uses_fresh_distinct_session() {
        let build = RecordingTurnAgent::new("build", vec![RecordingTurnAgent::final_resp("done")]);
        let eval = RecordingTurnAgent::new("eval", vec![RecordingTurnAgent::final_resp("PASS")]);
        let verifier = Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3));
        let mut cfg = standard_config(make_agent());
        cfg.agent = build;
        cfg.evaluator_agent = Some(eval);
        cfg.verifier = Some(verifier.clone());
        let h = StandardHarness::new(cfg);
        let t = self_verifying_task();
        let build_session = t.session_id.as_str().to_string();
        let _ = h.run(HarnessRunOptions::new(t)).await;
        let seen = verifier.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let (_iter, b, e) = &seen[0];
        assert_eq!(b, &build_session, "build session id is the run's session");
        assert_ne!(e, &build_session, "evaluate session must be fresh");
        assert!(e.starts_with("sess-"), "evaluate uses generate(): {e}");
    }

    // R3: the evaluate phase uses a read-only sandbox — an evaluator Write
    // attempt is rejected as a (recoverable) ReadOnlyViolation, while the build
    // sandbox is unaffected (the build phase writes freely).
    #[tokio::test]
    async fn self_verifying_evaluate_sandbox_is_read_only() {
        // Build agent writes (allowed by AllowAllSandbox), then claims done.
        let build = RecordingTurnAgent::new(
            "build",
            vec![
                RecordingTurnAgent::tool_call("write_file"),
                RecordingTurnAgent::final_resp("done"),
            ],
        );
        // Evaluator tries to write (must be blocked), then claims a verdict.
        let eval = RecordingTurnAgent::new(
            "eval",
            vec![
                RecordingTurnAgent::tool_call("write_file"),
                RecordingTurnAgent::final_resp("PASS"),
            ],
        );
        let mut cfg = standard_config(make_agent());
        cfg.agent = build.clone();
        cfg.evaluator_agent = Some(eval.clone());
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)));
        let h = StandardHarness::new(cfg);
        let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
        assert!(matches!(r, RunResult::Success { .. }), "got {r:?}");
        // The evaluator's second context (after its blocked write) must carry the
        // recoverable read-only sandbox error fed back as a tool result.
        let eval_seen = eval.seen_text();
        assert!(
            eval_seen.iter().any(|c| c.contains("ReadOnlyViolation")),
            "evaluator write must be rejected read-only; saw: {eval_seen:?}"
        );
        // The build phase's write was NOT rejected (no read-only error fed back).
        let build_seen = build.seen_text();
        assert!(
            !build_seen.iter().any(|c| c.contains("ReadOnlyViolation")),
            "build sandbox must be unaffected; saw: {build_seen:?}"
        );
    }

    // R4: the evaluator carries the `role-evaluator` chunk (presence assertion).
    #[tokio::test]
    async fn self_verifying_evaluator_carries_role_chunk() {
        let build = RecordingTurnAgent::new("build", vec![RecordingTurnAgent::final_resp("done")]);
        let eval = RecordingTurnAgent::new("eval", vec![RecordingTurnAgent::final_resp("PASS")]);
        let mut cfg = standard_config(make_agent());
        cfg.agent = build;
        cfg.evaluator_agent = Some(eval.clone());
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(vec![passed()], passed(), 3)));
        // Register the role-evaluator chunk through the provider.
        cfg.chunk_provider = Arc::new(crate::prompt_assembly::InMemoryChunkProvider::new(vec![
            crate::prompt_assembly::PromptChunk::new(
                "role-evaluator",
                "You are a fresh evaluator. You did not write this code.",
            ),
        ]));
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(self_verifying_task())).await;
        let eval_seen = eval.seen_text();
        assert!(
            eval_seen.iter().any(|c| c.contains("fresh evaluator")),
            "evaluator context must carry the role-evaluator chunk; saw: {eval_seen:?}"
        );
    }

    // R5: Default-FAIL — an indeterminate evaluator verdict keeps looping and
    // does not succeed. (The verifier returns Failed for an indeterminate eval.)
    #[tokio::test]
    async fn self_verifying_default_fail_does_not_succeed() {
        let build = RecordingTurnAgent::new(
            "build",
            vec![
                RecordingTurnAgent::final_resp("attempt 1"),
                RecordingTurnAgent::final_resp("attempt 2"),
            ],
        );
        let eval = RecordingTurnAgent::new(
            "eval",
            vec![
                RecordingTurnAgent::final_resp("indeterminate"),
                RecordingTurnAgent::final_resp("indeterminate"),
            ],
        );
        let mut cfg = standard_config(make_agent());
        cfg.agent = build;
        cfg.evaluator_agent = Some(eval);
        // Default-FAIL: every verdict is Failed (no explicit pass).
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(
            vec![],
            failed("indeterminate — default fail"),
            2,
        )));
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(self_verifying_task())).await {
            RunResult::Failure {
                reason: HaltReason::SelfVerifyExhausted { .. },
                ..
            } => {}
            other => panic!("expected SelfVerifyExhausted (no success), got {other:?}"),
        }
    }

    // R6: on findings, the verdict reason is injected into the build context and
    // the build loop resumes (fail iter 0 with reason X, pass iter 1; iter-1
    // build context contains X, final Success).
    #[tokio::test]
    async fn self_verifying_injects_reason_and_resumes() {
        const FINDING: &str = "MISSING_NULL_CHECK_IN_HANDLER";
        let build = RecordingTurnAgent::new(
            "build",
            vec![
                RecordingTurnAgent::final_resp("v1"),
                RecordingTurnAgent::final_resp("v2"),
            ],
        );
        let eval = RecordingTurnAgent::new(
            "eval",
            vec![
                RecordingTurnAgent::final_resp("FAIL"),
                RecordingTurnAgent::final_resp("PASS"),
            ],
        );
        let mut cfg = standard_config(make_agent());
        cfg.agent = build.clone();
        cfg.evaluator_agent = Some(eval);
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(
            vec![failed(FINDING), passed()],
            passed(),
            3,
        )));
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(self_verifying_task())).await {
            RunResult::Success { .. } => {}
            other => panic!("expected Success on iter 1, got {other:?}"),
        }
        // The build agent's LAST context (iteration 1) must contain the injected
        // finding from iteration 0.
        let build_seen = build.seen_text();
        let last = build_seen.last().expect("build ran");
        assert!(
            last.contains(FINDING),
            "iter-1 build context must contain the injected finding; saw: {last}"
        );
    }

    // R7: stagnation guard — an always-Failed verifier runs exactly
    // max_iterations cycles, then Failure { SelfVerifyExhausted }.
    #[tokio::test]
    async fn self_verifying_stagnation_guard_caps_iterations() {
        let build = RecordingTurnAgent::new(
            "build",
            vec![
                RecordingTurnAgent::final_resp("a"),
                RecordingTurnAgent::final_resp("b"),
                RecordingTurnAgent::final_resp("c"),
            ],
        );
        let eval = RecordingTurnAgent::new(
            "eval",
            vec![
                RecordingTurnAgent::final_resp("x"),
                RecordingTurnAgent::final_resp("y"),
                RecordingTurnAgent::final_resp("z"),
            ],
        );
        let verifier = Arc::new(ScriptedVerifier::new(
            vec![],
            failed("never good enough"),
            3,
        ));
        let mut cfg = standard_config(make_agent());
        cfg.agent = build;
        cfg.evaluator_agent = Some(eval);
        cfg.verifier = Some(verifier.clone());
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(self_verifying_task())).await {
            RunResult::Failure {
                reason:
                    HaltReason::SelfVerifyExhausted {
                        iterations,
                        last_reason,
                    },
                ..
            } => {
                assert_eq!(iterations, 3, "should run exactly max_iterations cycles");
                assert!(
                    last_reason.contains("never good enough"),
                    "got {last_reason}"
                );
            }
            other => panic!("expected SelfVerifyExhausted, got {other:?}"),
        }
        // Exactly max_iterations verifier calls (one per build↔evaluate cycle).
        assert_eq!(verifier.seen.lock().unwrap().len(), 3);
    }

    // R8: budgets fold both phases across all iterations (sum-check). Each turn
    // reports 1 input + 1 output token; with 2 iterations × (1 build + 1 eval
    // turn) the cumulative usage sums to 4 input + 4 output tokens.
    #[tokio::test]
    async fn self_verifying_budgets_fold_both_phases() {
        let build = RecordingTurnAgent::new(
            "build",
            vec![
                RecordingTurnAgent::final_resp("a"),
                RecordingTurnAgent::final_resp("b"),
            ],
        );
        let eval = RecordingTurnAgent::new(
            "eval",
            vec![
                RecordingTurnAgent::final_resp("x"),
                RecordingTurnAgent::final_resp("PASS"),
            ],
        );
        let mut cfg = standard_config(make_agent());
        cfg.agent = build;
        cfg.evaluator_agent = Some(eval);
        cfg.verifier = Some(Arc::new(ScriptedVerifier::new(
            vec![failed("retry"), passed()],
            passed(),
            3,
        )));
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(self_verifying_task())).await {
            RunResult::Success { usage, .. } => {
                // 2 build turns + 2 eval turns, each 1 in / 1 out.
                assert_eq!(usage.input_tokens, 4, "input tokens fold both phases");
                assert_eq!(usage.output_tokens, 4, "output tokens fold both phases");
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // R12: fixture replay — scripted (build verdict sequence, evaluator output
    // sequence) → expected terminal RunResult kind + iteration count.
    #[derive(serde::Deserialize)]
    struct SvFixtureCase {
        name: String,
        /// Per-iteration verifier verdicts. Each entry is `"pass"` or a failure
        /// reason string (anything else).
        verdicts: Vec<String>,
        max_iterations: u32,
        expected: SvFixtureExpected,
    }
    #[derive(serde::Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum SvFixtureExpected {
        Success { iterations: u32 },
        Exhausted { iterations: u32 },
        Misconfigured,
    }
    #[derive(serde::Deserialize)]
    struct SvFixtureSuite {
        cases: Vec<SvFixtureCase>,
    }

    #[tokio::test]
    async fn self_verifying_fixture_replay() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/harness/self_verifying.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: SvFixtureSuite = serde_json::from_str(&raw).expect("fixture parses");
        for case in suite.cases {
            // Build an agent with one "done" turn per possible iteration.
            let n = case.max_iterations.max(1) as usize;
            let build = RecordingTurnAgent::new(
                "build",
                (0..n)
                    .map(|_| RecordingTurnAgent::final_resp("done"))
                    .collect(),
            );
            let eval = RecordingTurnAgent::new(
                "eval",
                (0..n)
                    .map(|_| RecordingTurnAgent::final_resp("out"))
                    .collect(),
            );
            let verdicts: Vec<VerifierVerdict> = case
                .verdicts
                .iter()
                .map(|v| if v == "pass" { passed() } else { failed(v) })
                .collect();
            let mut cfg = standard_config(make_agent());
            cfg.agent = build;
            cfg.evaluator_agent = Some(eval);
            let misconfigured = matches!(case.expected, SvFixtureExpected::Misconfigured);
            if !misconfigured {
                cfg.verifier = Some(Arc::new(ScriptedVerifier::new(
                    verdicts,
                    failed("default fail"),
                    case.max_iterations,
                )));
            }
            let h = StandardHarness::new(cfg);
            let r = h.run(HarnessRunOptions::new(self_verifying_task())).await;
            match (case.expected, r) {
                (SvFixtureExpected::Success { iterations }, RunResult::Success { turns, .. }) => {
                    // `turns` is the final build sub-loop's absolute turn count;
                    // assert the run succeeded (iteration count is asserted via
                    // dedicated unit tests). `iterations` documents the case.
                    let _ = (iterations, turns);
                }
                (
                    SvFixtureExpected::Exhausted { iterations },
                    RunResult::Failure {
                        reason:
                            HaltReason::SelfVerifyExhausted {
                                iterations: got, ..
                            },
                        ..
                    },
                ) => {
                    assert_eq!(got, iterations, "case `{}` iteration count", case.name);
                }
                (
                    SvFixtureExpected::Misconfigured,
                    RunResult::Failure {
                        reason: HaltReason::SelfVerifyMisconfigured { .. },
                        ..
                    },
                ) => {}
                (_, other) => panic!("case `{}`: unexpected result {other:?}", case.name),
            }
        }
    }

    // ========================================================================
    // Ralph loop strategy (issue #58)
    // ========================================================================

    fn ralph_task() -> Task {
        let mut t = task(LoopStrategy::Ralph);
        // One ReAct turn per context window keeps the per-window sub-loop
        // bounded so the OUTER reset loop drives the test deterministically.
        t.budget.max_turns = Some(1);
        t
    }

    /// A `SandboxProvider` whose `workspace_root` is a real tempdir, so the
    /// Ralph filesystem reload + completion check read real `.spore/` files.
    struct WorkspaceSandbox {
        root: std::path::PathBuf,
    }
    impl SandboxProvider for WorkspaceSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
        fn workspace_root(&self) -> &std::path::Path {
            &self.root
        }
    }

    /// Write `.spore/progress.json` under `root` (creating `.spore/`).
    fn write_progress(root: &std::path::Path, body: &str) {
        std::fs::create_dir_all(root.join(".spore")).unwrap();
        std::fs::write(root.join(".spore/progress.json"), body).unwrap();
    }
    fn write_feature_list(root: &std::path::Path, body: &str) {
        std::fs::create_dir_all(root.join(".spore")).unwrap();
        std::fs::write(root.join(".spore/feature_list.json"), body).unwrap();
    }
    const INCOMPLETE: &str = r#"{"complete":false,"remaining":["task A"]}"#;
    const COMPLETE: &str = r#"{"complete":true,"remaining":[]}"#;

    /// Agent that, on each turn, pops the next progress-file body from a queue
    /// and writes it to `.spore/progress.json` BEFORE returning a `FinalResponse`
    /// — modelling "the agent did work this window and updated progress." Also
    /// records the messages it saw so tests can assert fresh-state / reload.
    struct ProgressWritingAgent {
        id: AgentId,
        root: std::path::PathBuf,
        progress_queue: StdMutex<std::collections::VecDeque<String>>,
        seen: StdMutex<Vec<Context>>,
    }
    impl ProgressWritingAgent {
        fn new(root: &std::path::Path, bodies: Vec<&str>) -> Arc<Self> {
            Arc::new(Self {
                id: AgentId::new("ralph-build"),
                root: root.to_path_buf(),
                progress_queue: StdMutex::new(bodies.into_iter().map(String::from).collect()),
                seen: StdMutex::new(Vec::new()),
            })
        }
        fn call_count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
        fn seen_text(&self) -> Vec<String> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|c| {
                    c.messages
                        .iter()
                        .map(|m| match &m.content {
                            Content::Text { text } => text.clone(),
                            _ => String::new(),
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .collect()
        }
    }
    impl Agent for ProgressWritingAgent {
        fn turn<'a>(&'a self, context: Context) -> BoxFut<'a, TurnResult> {
            self.seen.lock().unwrap().push(context);
            if let Some(body) = self.progress_queue.lock().unwrap().pop_front() {
                write_progress(&self.root, &body);
            }
            Box::pin(async move {
                TurnResult::FinalResponse {
                    content: "window done".into(),
                    usage: TokenUsage {
                        input_tokens: 1,
                        output_tokens: 1,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                    },
                }
            })
        }
        fn id(&self) -> AgentId {
            self.id.clone()
        }
    }

    fn ralph_config(root: &std::path::Path, agent: Arc<dyn Agent>) -> HarnessConfig {
        let mut cfg = standard_config(make_agent());
        cfg.agent = agent;
        cfg.sandbox = Arc::new(WorkspaceSandbox {
            root: root.to_path_buf(),
        });
        // Use a real context manager so reloaded context lands in messages.
        cfg.context_manager = Arc::new(NoopContextManager);
        cfg
    }

    // R0: Ralph is implemented — no longer StrategyNotYetImplemented.
    #[tokio::test]
    async fn ralph_no_longer_unimplemented() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), COMPLETE);
        let agent = ProgressWritingAgent::new(dir.path(), vec![COMPLETE]);
        let h = StandardHarness::new(ralph_config(dir.path(), agent));
        match h.run(HarnessRunOptions::new(ralph_task())).await {
            RunResult::Failure {
                reason: HaltReason::StrategyNotYetImplemented { .. },
                ..
            } => panic!("Ralph must be implemented"),
            RunResult::Success { .. } => {}
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // R4: completion pattern incomplete,incomplete,complete → Success at it. 3.
    #[tokio::test]
    async fn ralph_resets_until_complete() {
        let dir = tempfile::tempdir().unwrap();
        // Start incomplete so the first window's reload sees prior state.
        write_progress(dir.path(), INCOMPLETE);
        let agent = ProgressWritingAgent::new(dir.path(), vec![INCOMPLETE, INCOMPLETE, COMPLETE]);
        let mut cfg = ralph_config(dir.path(), agent.clone());
        cfg.max_resets = 3;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(ralph_task())).await {
            RunResult::Success { .. } => {}
            other => panic!("expected Success at the 3rd window, got {other:?}"),
        }
        // Exactly three context windows ran (one agent turn each).
        assert_eq!(agent.call_count(), 3, "should reset twice, run 3 windows");
    }

    // R5: always-incomplete → exactly max_resets windows → RalphCompletionUnmet.
    #[tokio::test]
    async fn ralph_exhausts_max_resets() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), INCOMPLETE);
        let agent = ProgressWritingAgent::new(
            dir.path(),
            vec![INCOMPLETE, INCOMPLETE, INCOMPLETE, INCOMPLETE],
        );
        let mut cfg = ralph_config(dir.path(), agent.clone());
        cfg.max_resets = 3;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(ralph_task())).await {
            RunResult::Failure {
                reason:
                    HaltReason::RalphCompletionUnmet {
                        iterations,
                        last_reason,
                    },
                ..
            } => {
                assert_eq!(iterations, 3, "exactly max_resets windows");
                assert!(last_reason.contains("task A"), "got: {last_reason}");
            }
            other => panic!("expected RalphCompletionUnmet, got {other:?}"),
        }
        assert_eq!(agent.call_count(), 3, "exactly max_resets windows ran");
    }

    // R2: each reset builds a FRESH SessionState — no message carryover. The
    // agent's recorded context for window 2 must NOT contain window 1's content.
    #[tokio::test]
    async fn ralph_fresh_session_per_reset() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), INCOMPLETE);
        let agent = ProgressWritingAgent::new(dir.path(), vec![INCOMPLETE, COMPLETE]);
        let mut cfg = ralph_config(dir.path(), agent.clone());
        cfg.max_resets = 3;
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
        let texts = agent.seen_text();
        assert_eq!(texts.len(), 2, "two windows");
        // The assistant's "window done" output from window 1 is NOT present in
        // window 2's fresh context (no carryover). Each window's context is
        // re-seeded from instruction + reloaded `.spore/` state only.
        assert!(
            !texts[1].contains("window done"),
            "window 2 carried over window 1 messages: {}",
            texts[1]
        );
    }

    // R3: the filesystem reload injects `.spore/` state into the fresh seed.
    #[tokio::test]
    async fn ralph_reload_injects_filesystem_state() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), INCOMPLETE);
        write_feature_list(dir.path(), r#"[{"name":"login","passes":false}]"#);
        // Agent leaves progress incomplete on window 1, complete on window 2.
        let agent = ProgressWritingAgent::new(dir.path(), vec![INCOMPLETE, COMPLETE]);
        let mut cfg = ralph_config(dir.path(), agent.clone());
        cfg.max_resets = 3;
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
        let texts = agent.seen_text();
        // Window 1's fresh seed contains the reloaded progress + feature list.
        assert!(
            texts[0].contains("Reloaded .spore/progress.json")
                && texts[0].contains("Reloaded .spore/feature_list.json")
                && texts[0].contains("login"),
            "reload not injected: {}",
            texts[0]
        );
    }

    // R6: budgets fold across ALL context windows (each window adds usage).
    #[tokio::test]
    async fn ralph_budgets_fold_across_windows() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), INCOMPLETE);
        let agent = ProgressWritingAgent::new(dir.path(), vec![INCOMPLETE, INCOMPLETE, COMPLETE]);
        let mut cfg = ralph_config(dir.path(), agent);
        cfg.max_resets = 3;
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(ralph_task())).await {
            RunResult::Success { usage, .. } => {
                // Three windows × one turn × (1 in, 1 out) folded.
                assert_eq!(usage.input_tokens, 3, "input tokens fold across windows");
                assert_eq!(usage.output_tokens, 3, "output tokens fold across windows");
            }
            other => panic!("expected Success, got {other:?}"),
        }
    }

    // R7: missing progress file is treated as incomplete by the OUTER loop, and
    // the registered Stop hook does NOT interfere with non-Ralph runs.
    #[tokio::test]
    async fn ralph_stop_hook_inert_without_progress_file() {
        // A plain ReAct run on a workspace with no `.spore/progress.json`: the
        // registered RalphStopHook must return Continue (terminate in one turn).
        let dir = tempfile::tempdir().unwrap();
        let a = make_agent();
        a.push(TurnResult::FinalResponse {
            content: "done".into(),
            usage: usage(),
        });
        let mut cfg = standard_config(a);
        cfg.sandbox = Arc::new(WorkspaceSandbox {
            root: dir.path().to_path_buf(),
        });
        let h = StandardHarness::new(cfg);
        match h.run(HarnessRunOptions::new(react(5))).await {
            RunResult::Success { turns, .. } => assert_eq!(turns, 1, "stop hook must be inert"),
            other => panic!("expected Success in one turn, got {other:?}"),
        }
    }

    // Completion-status helper: progress complete but a feature fails ⇒ still
    // incomplete (the feature list corroborates).
    #[tokio::test]
    async fn ralph_completion_status_feature_list_gate() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), COMPLETE);
        write_feature_list(dir.path(), r#"[{"name":"login","passes":false}]"#);
        let status = StandardHarness::ralph_completion_status(dir.path());
        assert!(
            status.as_deref().unwrap_or("").contains("login"),
            "got: {status:?}"
        );
        // Now mark it passing — complete.
        write_feature_list(dir.path(), r#"[{"name":"login","passes":true}]"#);
        assert_eq!(StandardHarness::ralph_completion_status(dir.path()), None);
    }

    // ── VcsProvider seam (issue #58 v2) ─────────────────────────────────────

    // (a) FixtureVcsProvider::log returns the seeded string verbatim.
    #[tokio::test]
    async fn fixture_vcs_log_verbatim() {
        let log = "abc123 first\ndef456 second\n";
        let provider = FixtureVcsProvider::new(log, "clean");
        let args = VcsLogArgs {
            max_entries: 5,
            since_ref: Some("HEAD~3".into()),
            format: Some("%h %s".into()),
        };
        let out = provider.log(&args).await.unwrap();
        assert_eq!(out, log, "log output must be verbatim, args ignored");
    }

    // (e) status() round-trips the seeded string verbatim.
    #[tokio::test]
    async fn fixture_vcs_status_round_trips() {
        let status = "On branch main\nnothing to commit\n";
        let provider = FixtureVcsProvider::new("", status);
        assert_eq!(provider.status().await.unwrap(), status);
    }

    /// A `SandboxProvider` that captures the `execute_command` invocation so the
    /// `git log`/`git status` command lines can be asserted, and returns a
    /// scripted stdout.
    struct CapturingSandbox {
        captured: StdMutex<Vec<(String, Vec<String>)>>,
        stdout: String,
    }
    impl CapturingSandbox {
        fn new(stdout: &str) -> Self {
            Self {
                captured: StdMutex::new(Vec::new()),
                stdout: stdout.to_string(),
            }
        }
        fn last(&self) -> (String, Vec<String>) {
            self.captured.lock().unwrap().last().cloned().unwrap()
        }
    }
    impl SandboxProvider for CapturingSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
        fn execute_command<'a>(
            &'a self,
            command: &'a str,
            args: &'a [String],
            _working_dir: Option<&'a std::path::Path>,
            _timeout: Option<std::time::Duration>,
        ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
            self.captured
                .lock()
                .unwrap()
                .push((command.to_string(), args.to_vec()));
            let stdout = self.stdout.clone();
            Box::pin(async move {
                Ok(CommandOutput {
                    stdout,
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    truncated: false,
                })
            })
        }
    }

    // (b) GitVcsProvider builds the correct `git log` command from VcsLogArgs,
    // shelling out THROUGH the sandbox (never bypassing it).
    #[tokio::test]
    async fn git_vcs_builds_log_command() {
        let sandbox = Arc::new(CapturingSandbox::new("log-output"));
        let git = GitVcsProvider::new(sandbox.clone(), "/work");
        let args = VcsLogArgs {
            max_entries: 7,
            since_ref: Some("main".into()),
            format: Some("%h %s".into()),
        };
        let out = git.log(&args).await.unwrap();
        assert_eq!(out, "log-output");
        let (cmd, argv) = sandbox.last();
        assert_eq!(cmd, "git");
        assert_eq!(
            argv,
            vec![
                "log".to_string(),
                "-n".to_string(),
                "7".to_string(),
                "--format=%h %s".to_string(),
                "main..".to_string(),
            ],
            "git log command line built from VcsLogArgs"
        );

        // status() runs `git status`.
        let _ = git.status().await.unwrap();
        let (cmd, argv) = sandbox.last();
        assert_eq!(cmd, "git");
        assert_eq!(argv, vec!["status".to_string()]);
    }

    // (b') Minimal args produce just `git log -n <N>` (no format/range flags).
    #[tokio::test]
    async fn git_vcs_log_command_minimal_args() {
        let sandbox = Arc::new(CapturingSandbox::new(""));
        let git = GitVcsProvider::new(sandbox.clone(), "/work");
        let args = VcsLogArgs {
            max_entries: 3,
            since_ref: None,
            format: None,
        };
        let _ = git.log(&args).await.unwrap();
        let (_, argv) = sandbox.last();
        assert_eq!(
            argv,
            vec!["log".to_string(), "-n".to_string(), "3".to_string()]
        );
    }

    // (c) Ralph with a FixtureVcsProvider set injects the vcs_log string into the
    // reloaded context across a reset.
    #[tokio::test]
    async fn ralph_injects_vcs_log_into_reload() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), INCOMPLETE);
        let agent = ProgressWritingAgent::new(dir.path(), vec![INCOMPLETE, COMPLETE]);
        let mut cfg = ralph_config(dir.path(), agent.clone());
        cfg.max_resets = 3;
        cfg.vcs_provider = Some(Arc::new(FixtureVcsProvider::new(
            "cafe123 implement login\nbeef456 add tests\n",
            "clean",
        )));
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
        let texts = agent.seen_text();
        assert!(
            texts[0].contains("Recent VCS history:")
                && texts[0].contains("cafe123 implement login"),
            "vcs log not injected: {}",
            texts[0]
        );
    }

    // (d) Ralph with vcs_provider = None omits any git section (v1 unchanged).
    #[tokio::test]
    async fn ralph_none_vcs_omits_git_section() {
        let dir = tempfile::tempdir().unwrap();
        write_progress(dir.path(), INCOMPLETE);
        let agent = ProgressWritingAgent::new(dir.path(), vec![COMPLETE]);
        let cfg = ralph_config(dir.path(), agent.clone());
        // vcs_provider defaults to None.
        assert!(cfg.vcs_provider.is_none());
        let h = StandardHarness::new(cfg);
        let _ = h.run(HarnessRunOptions::new(ralph_task())).await;
        let texts = agent.seen_text();
        assert!(
            !texts[0].contains("Recent VCS history:"),
            "no git section expected when provider is None: {}",
            texts[0]
        );
    }

    // ── Ralph cross-language fixture replay ─────────────────────────────────
    #[derive(serde::Deserialize)]
    struct RalphFixtureCase {
        name: String,
        /// Per-window progress-file body the agent writes (the window's state).
        windows: Vec<RalphWindow>,
        max_resets: u32,
        expected: RalphFixtureExpected,
        /// Optional git-log string (issue #58 v2). When present a
        /// [`FixtureVcsProvider`] seeded with it is wired into the harness and
        /// the reloaded context is asserted to contain it. Absent ⇒ no provider
        /// (None) ⇒ no git section (v1 behavior).
        #[serde(default)]
        vcs_log: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct RalphWindow {
        complete: bool,
        #[serde(default)]
        remaining: Vec<String>,
    }
    #[derive(serde::Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum RalphFixtureExpected {
        Success { iterations: u32 },
        CompletionUnmet { iterations: u32 },
    }
    #[derive(serde::Deserialize)]
    struct RalphFixtureSuite {
        cases: Vec<RalphFixtureCase>,
    }

    #[tokio::test]
    async fn ralph_fixture_replay() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/harness/ralph.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: RalphFixtureSuite = serde_json::from_str(&raw).expect("fixture parses");
        for case in suite.cases {
            let dir = tempfile::tempdir().unwrap();
            // Seed an initial incomplete progress file so window 1 reloads state.
            write_progress(dir.path(), INCOMPLETE);
            let bodies: Vec<String> = case
                .windows
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "complete": w.complete,
                        "remaining": w.remaining,
                    })
                    .to_string()
                })
                .collect();
            let agent =
                ProgressWritingAgent::new(dir.path(), bodies.iter().map(|s| s.as_str()).collect());
            let mut cfg = ralph_config(dir.path(), agent.clone());
            cfg.max_resets = case.max_resets;
            // issue #58 v2: when the case carries a `vcs_log`, wire a
            // FixtureVcsProvider seeded with it; absent ⇒ None ⇒ no git section.
            if let Some(log) = &case.vcs_log {
                cfg.vcs_provider = Some(Arc::new(FixtureVcsProvider::new(log.clone(), "")));
            }
            let h = StandardHarness::new(cfg);
            let r = h.run(HarnessRunOptions::new(ralph_task())).await;
            // When a vcs_log is present, the first fresh window must include it.
            if let Some(log) = &case.vcs_log {
                let texts = agent.seen_text();
                assert!(
                    texts
                        .first()
                        .map(|t| t.contains("Recent VCS history:") && t.contains(log.trim()))
                        .unwrap_or(false),
                    "case `{}`: vcs_log not injected into reload: {:?}",
                    case.name,
                    texts.first()
                );
            }
            match (case.expected, r) {
                (RalphFixtureExpected::Success { iterations }, RunResult::Success { .. }) => {
                    assert_eq!(
                        agent.call_count() as u32,
                        iterations,
                        "case `{}` window count",
                        case.name
                    );
                }
                (
                    RalphFixtureExpected::CompletionUnmet { iterations },
                    RunResult::Failure {
                        reason:
                            HaltReason::RalphCompletionUnmet {
                                iterations: got, ..
                            },
                        ..
                    },
                ) => {
                    assert_eq!(got, iterations, "case `{}` iteration count", case.name);
                }
                (_, other) => panic!("case `{}`: unexpected result {other:?}", case.name),
            }
        }
    }

    // ========================================================================
    // HillClimbing loop strategy (issue #60)
    // ========================================================================

    use crate::metric::{
        IterationStatus, MetricError, MetricEvaluator, MetricResult, ResultsEntry,
    };
    use crate::termination::SessionStateSnapshot as HcSnapshot;

    fn hill_task(
        direction: OptimizationDirection,
        max_stagnation: Option<u32>,
        revert: bool,
        min_delta: Option<f64>,
    ) -> Task {
        // Give a generous turn budget so the loop is bounded by stagnation, not
        // turns, unless a test overrides it.
        let mut t = task(LoopStrategy::HillClimbing {
            direction,
            max_stagnation,
            revert_on_no_improvement: revert,
            min_improvement_delta: min_delta,
        });
        t.budget.max_turns = Some(50);
        t
    }

    /// A `MetricEvaluator` test double returning a pre-programmed sequence of
    /// `Result<MetricResult, MetricError>` — one per `evaluate` call (call 0 is
    /// the baseline). Records how many times it was evaluated.
    struct ScriptedMetricEvaluator {
        outcomes: StdMutex<std::collections::VecDeque<Result<MetricResult, MetricError>>>,
        calls: std::sync::atomic::AtomicUsize,
        direction: OptimizationDirection,
        description: String,
    }
    impl ScriptedMetricEvaluator {
        fn new(values: Vec<Result<f64, MetricError>>, direction: OptimizationDirection) -> Self {
            let outcomes = values
                .into_iter()
                .map(|v| v.map(MetricResult::new))
                .collect();
            Self {
                outcomes: StdMutex::new(outcomes),
                calls: std::sync::atomic::AtomicUsize::new(0),
                direction,
                description: "scripted metric".to_string(),
            }
        }
        fn call_count(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    impl MetricEvaluator for ScriptedMetricEvaluator {
        fn evaluate<'a>(
            &'a self,
            _sandbox: &'a dyn SandboxProvider,
            _session_state: &'a HcSnapshot,
        ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let next = self.outcomes.lock().unwrap().pop_front().unwrap_or(Err(
                MetricError::ExecutionFailed {
                    reason: "scripted evaluator exhausted".into(),
                },
            ));
            Box::pin(async move { next })
        }
        fn direction(&self) -> OptimizationDirection {
            self.direction
        }
        fn description(&self) -> String {
            self.description.clone()
        }
    }

    /// A `SandboxProvider` rooted at a tempdir that spies on `git reset --hard
    /// HEAD` revert calls (issue #60, Decision 1). Counts every `execute_command`
    /// whose argv is exactly the revert command and returns a clean exit.
    struct HcSpySandbox {
        root: tempfile::TempDir,
        reverts: std::sync::atomic::AtomicUsize,
    }
    impl HcSpySandbox {
        fn new() -> Self {
            Self {
                root: tempfile::tempdir().unwrap(),
                reverts: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn revert_count(&self) -> usize {
            self.reverts.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    impl SandboxProvider for HcSpySandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
        fn execute_command<'a>(
            &'a self,
            command: &'a str,
            args: &'a [String],
            _working_dir: Option<&'a std::path::Path>,
            _timeout: Option<std::time::Duration>,
        ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
            if command == "git" && args == ["reset", "--hard", "HEAD"] {
                self.reverts
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            Box::pin(async {
                Ok(CommandOutput {
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: 0,
                    timed_out: false,
                    truncated: false,
                })
            })
        }
        fn workspace_root(&self) -> &std::path::Path {
            self.root.path()
        }
    }

    /// Build a HillClimbing config wired with the scripted evaluator and spy
    /// sandbox. The agent always finishes a turn immediately (one final response
    /// per iteration), so the loop is driven entirely by the metric sequence.
    fn hc_config(
        evaluator: Arc<ScriptedMetricEvaluator>,
        sandbox: Arc<HcSpySandbox>,
    ) -> HarnessConfig {
        // A fresh "done" turn for every possible iteration (the agent is re-seeded
        // per iteration, so each iteration pops exactly one turn).
        let agent = RecordingTurnAgent::new(
            "hc",
            (0..64)
                .map(|_| RecordingTurnAgent::final_resp("changed"))
                .collect(),
        );
        let mut cfg = standard_config(make_agent());
        cfg.agent = agent;
        cfg.sandbox = sandbox;
        cfg.metric_evaluator = Some(evaluator);
        cfg
    }

    // Decision 6: a None evaluator halts with HillClimbingMisconfigured (no panic).
    #[tokio::test]
    async fn hill_climbing_missing_evaluator_is_typed_halt() {
        let cfg = standard_config(make_agent());
        let h = StandardHarness::new(cfg);
        let t = hill_task(OptimizationDirection::Maximize, None, false, None);
        match h.run(HarnessRunOptions::new(t)).await {
            RunResult::Failure {
                reason: HaltReason::HillClimbingMisconfigured { reason },
                ..
            } => assert!(reason.contains("metric_evaluator"), "got: {reason}"),
            other => panic!("expected HillClimbingMisconfigured, got {other:?}"),
        }
    }

    // Decision 5: iteration 0 is a pure baseline (no agent turn) recorded Kept,
    // and it seeds current_best. With max_stagnation = 1 and an immediate regress
    // at iteration 1, the loop halts; we confirm the baseline row is Kept and the
    // agent ran exactly once (only iteration 1's turn, never the baseline).
    #[tokio::test]
    async fn hill_climbing_baseline_first_status_kept_no_agent_turn() {
        let sandbox = Arc::new(HcSpySandbox::new());
        // baseline 1.0, then a worse value (regress) for maximize.
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(1.0), Ok(0.5)],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval.clone(), sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        let h = StandardHarness::new(cfg);
        let t = hill_task(OptimizationDirection::Maximize, Some(1), false, None);
        let task_id = t.id.as_str().to_string();
        let r = h.run(HarnessRunOptions::new(t)).await;
        assert!(
            matches!(
                r,
                RunResult::Failure {
                    reason: HaltReason::StagnationLimitReached { .. },
                    ..
                }
            ),
            "got {r:?}"
        );
        // Two evaluator calls: baseline + iteration 1.
        assert_eq!(eval.call_count(), 2);
        // The TSV baseline row is Kept.
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .expect("tsv written");
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(
            lines[0],
            "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription"
        );
        assert!(lines[1].starts_with("0\t"), "baseline is iteration 0");
        assert!(
            lines[1].contains("\tkept\t"),
            "baseline status kept: {}",
            lines[1]
        );
    }

    // keep-on-improve: a strictly better value updates current_best and resets
    // stagnation. With maximize, baseline 1.0 then 2.0 (improve) then 1.5 (regress
    // vs 2.0) then 3.0 (improve). max_stagnation None ⇒ runs to the turn budget.
    #[tokio::test]
    async fn hill_climbing_keeps_on_improvement() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(1.0), Ok(2.0), Ok(1.5), Ok(3.0)],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval.clone(), sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        // Cap turns at 3 so exactly iterations 1,2,3 run (then budget halt).
        let mut t = hill_task(OptimizationDirection::Maximize, None, false, None);
        t.budget.max_turns = Some(3);
        let task_id = t.id.as_str().to_string();
        let _ = h_run(&cfg, t).await;
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .unwrap();
        let lines: Vec<&str> = tsv.lines().collect();
        // header + baseline(0) + iter1(2.0 kept) + iter2(1.5 discarded) + iter3(3.0 kept)
        assert_eq!(lines.len(), 5, "tsv:\n{tsv}");
        assert!(lines[1].contains("\t1.000000\t") && lines[1].contains("\tkept\t"));
        assert!(lines[2].contains("\t2.000000\t") && lines[2].contains("\tkept\t"));
        assert!(lines[3].contains("\t1.500000\t") && lines[3].contains("\tdiscarded\t"));
        assert!(lines[4].contains("\t3.000000\t") && lines[4].contains("\tkept\t"));
        let _ = eval;
    }

    // discard-on-regress with revert OFF: no git reset issued.
    #[tokio::test]
    async fn hill_climbing_discards_on_regress_no_revert() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(2.0), Ok(3.0)], // minimize: baseline 2.0, then 3.0 (worse)
            OptimizationDirection::Minimize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        let t = hill_task(OptimizationDirection::Minimize, Some(1), false, None);
        let _ = h_run(&cfg, t).await;
        assert_eq!(sandbox.revert_count(), 0, "revert OFF must not git reset");
    }

    // revert ON: a no-improvement iteration triggers exactly one git reset.
    #[tokio::test]
    async fn hill_climbing_reverts_on_no_improvement() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(2.0), Ok(3.0)], // minimize: baseline 2.0, then 3.0 (worse)
            OptimizationDirection::Minimize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        let t = hill_task(OptimizationDirection::Minimize, Some(1), true, None);
        let _ = h_run(&cfg, t).await;
        assert_eq!(sandbox.revert_count(), 1, "revert ON must git reset once");
    }

    // strict min_delta boundary: an improvement of exactly min_delta is NOT
    // progress (discarded). Minimize, baseline 2.0, then 1.5 with min_delta 0.5.
    #[tokio::test]
    async fn hill_climbing_strict_min_delta_boundary() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(2.0), Ok(1.5)],
            OptimizationDirection::Minimize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        let t = hill_task(OptimizationDirection::Minimize, Some(1), false, Some(0.5));
        let task_id = t.id.as_str().to_string();
        let r = h_run(&cfg, t).await;
        // Exactly-min_delta improvement is discarded ⇒ stagnation hits 1 ⇒ halt.
        match r {
            RunResult::Failure {
                reason: HaltReason::StagnationLimitReached { best_metric, .. },
                ..
            } => assert!((best_metric - 2.0).abs() < 1e-9, "best stays baseline 2.0"),
            other => panic!("expected StagnationLimitReached, got {other:?}"),
        }
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .unwrap();
        assert!(
            tsv.lines().nth(2).unwrap().contains("\tdiscarded\t"),
            "exactly-min_delta is discarded: {tsv}"
        );
    }

    // stagnation halt after N consecutive non-improvements.
    #[tokio::test]
    async fn hill_climbing_stagnation_halt() {
        let sandbox = Arc::new(HcSpySandbox::new());
        // maximize, baseline 5.0, then three regresses.
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(5.0), Ok(4.0), Ok(4.0), Ok(4.0)],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        let t = hill_task(OptimizationDirection::Maximize, Some(3), false, None);
        match h_run(&cfg, t).await {
            RunResult::Failure {
                reason:
                    HaltReason::StagnationLimitReached {
                        iterations,
                        best_metric,
                    },
                ..
            } => {
                assert_eq!(iterations, 3);
                assert!((best_metric - 5.0).abs() < 1e-9);
            }
            other => panic!("expected StagnationLimitReached, got {other:?}"),
        }
    }

    // stagnation counter resets on improvement: regress, regress, IMPROVE, regress
    // never reaches a stagnation cap of 3.
    #[tokio::test]
    async fn hill_climbing_stagnation_resets_on_improve() {
        let sandbox = Arc::new(HcSpySandbox::new());
        // maximize: baseline 1.0; iters: 0.5(regress) 0.5(regress) 2.0(improve!) 1.0(regress)
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(1.0), Ok(0.5), Ok(0.5), Ok(2.0), Ok(1.0)],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        // Cap turns at 4 so the improve at iter 3 resets the counter and the run
        // ends on the turn budget rather than stagnation.
        let mut t = hill_task(OptimizationDirection::Maximize, Some(3), false, None);
        t.budget.max_turns = Some(4);
        match h_run(&cfg, t).await {
            RunResult::Failure {
                reason:
                    HaltReason::BudgetExceeded {
                        limit_type: BudgetLimitType::Turns,
                    },
                ..
            } => {}
            other => panic!("expected Turns budget halt (counter reset), got {other:?}"),
        }
    }

    // crash counts as a non-improvement: a Crashed evaluation increments
    // stagnation and writes an EMPTY metric_value row.
    #[tokio::test]
    async fn hill_climbing_crash_counts_as_non_improvement() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(1.0), Err(MetricError::Crashed { log: "boom".into() })],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        let t = hill_task(OptimizationDirection::Maximize, Some(1), false, None);
        let task_id = t.id.as_str().to_string();
        match h_run(&cfg, t).await {
            RunResult::Failure {
                reason: HaltReason::StagnationLimitReached { .. },
                ..
            } => {}
            other => panic!("expected StagnationLimitReached, got {other:?}"),
        }
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .unwrap();
        let crash_row = tsv.lines().nth(2).unwrap();
        // metric_value column (3rd, index 2) is EMPTY on a crashed row.
        let cols: Vec<&str> = crash_row.split('\t').collect();
        assert_eq!(cols[2], "", "crashed row metric_value empty: {crash_row}");
        assert_eq!(cols[4], "crashed", "crashed status: {crash_row}");
    }

    // timeout counts as non-improvement and maps to the `timeout` status.
    #[tokio::test]
    async fn hill_climbing_timeout_status() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![
                Ok(1.0),
                Err(MetricError::Timeout {
                    after: std::time::Duration::from_secs(1),
                }),
            ],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval, sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        let t = hill_task(OptimizationDirection::Maximize, Some(1), false, None);
        let task_id = t.id.as_str().to_string();
        let _ = h_run(&cfg, t).await;
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .unwrap();
        let cols: Vec<&str> = tsv.lines().nth(2).unwrap().split('\t').collect();
        assert_eq!(cols[4], "timeout");
        assert_eq!(cols[2], "", "timeout row metric_value empty");
    }

    // budget gate: max_turns = 0 halts before any agent turn (only the baseline
    // runs). The result is a Turns budget halt with a baseline-only TSV.
    #[tokio::test]
    async fn hill_climbing_budget_turn_gate() {
        let sandbox = Arc::new(HcSpySandbox::new());
        let eval = Arc::new(ScriptedMetricEvaluator::new(
            vec![Ok(1.0)],
            OptimizationDirection::Maximize,
        ));
        let cfg = hc_config(eval.clone(), sandbox.clone());
        let workspace = sandbox.root.path().to_path_buf();
        let mut t = hill_task(OptimizationDirection::Maximize, None, false, None);
        t.budget.max_turns = Some(0);
        let task_id = t.id.as_str().to_string();
        match h_run(&cfg, t).await {
            RunResult::Failure {
                reason:
                    HaltReason::BudgetExceeded {
                        limit_type: BudgetLimitType::Turns,
                    },
                ..
            } => {}
            other => panic!("expected Turns budget halt, got {other:?}"),
        }
        // Only the baseline evaluation ran.
        assert_eq!(eval.call_count(), 1);
        let tsv = std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
            .unwrap();
        assert_eq!(tsv.lines().count(), 2, "header + baseline only: {tsv}");
    }

    // Exact TSV byte content for a representative keep/discard/crash scenario.
    #[tokio::test]
    async fn hill_climbing_exact_tsv_bytes() {
        let rows = vec![
            ResultsEntry {
                iteration: 0,
                commit_hash: None,
                metric_value: 1.0,
                direction: OptimizationDirection::Maximize,
                status: IterationStatus::Kept,
                duration: std::time::Duration::from_millis(1500),
                description: "scripted metric".into(),
                metadata: Default::default(),
            },
            ResultsEntry {
                iteration: 1,
                commit_hash: None,
                metric_value: 2.5,
                direction: OptimizationDirection::Maximize,
                status: IterationStatus::Kept,
                duration: std::time::Duration::from_secs(0),
                description: "scripted metric".into(),
                metadata: Default::default(),
            },
            ResultsEntry {
                iteration: 2,
                commit_hash: None,
                metric_value: f64::NAN, // crashed ⇒ empty in TSV
                direction: OptimizationDirection::Maximize,
                status: IterationStatus::Crashed,
                duration: std::time::Duration::from_secs(0),
                description: "scripted metric".into(),
                metadata: Default::default(),
            },
        ];
        let tsv = StandardHarness::render_hill_climbing_tsv(&rows);
        let expected =
            "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n\
0\t\t1.000000\tmaximize\tkept\t1.500000\tscripted metric\n\
1\t\t2.500000\tmaximize\tkept\t0.000000\tscripted metric\n\
2\t\t\tmaximize\tcrashed\t0.000000\tscripted metric\n";
        assert_eq!(tsv, expected, "TSV byte mismatch");
    }

    /// Helper: build a StandardHarness from `cfg` and run `task` to completion.
    async fn h_run(cfg: &HarnessConfig, t: Task) -> RunResult {
        let h = StandardHarness::new(cfg.clone());
        h.run(HarnessRunOptions::new(t)).await
    }

    // ── HillClimbing cross-language fixture replay ──────────────────────────
    #[derive(serde::Deserialize)]
    struct HcFixtureSuite {
        scenarios: Vec<HcScenario>,
    }
    #[derive(serde::Deserialize)]
    struct HcScenario {
        name: String,
        /// Scripted metric sequence; element 0 is the baseline. `null` = a crash.
        metric_sequence: Vec<Option<f64>>,
        payload: HcPayload,
        expected: HcExpected,
        /// Optional exact-TSV byte assertion for this scenario.
        #[serde(default)]
        expected_tsv: Option<String>,
        /// Optional turn-budget override (defaults to a high cap).
        #[serde(default)]
        max_turns: Option<u32>,
    }
    #[derive(serde::Deserialize)]
    struct HcPayload {
        direction: OptimizationDirection,
        max_stagnation: Option<u32>,
        revert_on_no_improvement: bool,
        min_improvement_delta: Option<f64>,
    }
    #[derive(serde::Deserialize)]
    struct HcExpected {
        /// `"stagnation"` | `"budget_turns"` | `"misconfigured"`.
        halt_reason: String,
        kept_iterations: u32,
        revert_count: usize,
        #[serde(default)]
        best_metric: Option<f64>,
    }

    #[tokio::test]
    async fn hill_climbing_fixture_replay() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/metric_evaluator/hill_climbing_sequences.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: HcFixtureSuite = serde_json::from_str(&raw).expect("fixture parses");
        for sc in suite.scenarios {
            let sandbox = Arc::new(HcSpySandbox::new());
            let values: Vec<Result<f64, MetricError>> = sc
                .metric_sequence
                .iter()
                .map(|v| match v {
                    Some(x) => Ok(*x),
                    None => Err(MetricError::Crashed {
                        log: "scripted crash".into(),
                    }),
                })
                .collect();
            let eval = Arc::new(ScriptedMetricEvaluator::new(values, sc.payload.direction));
            let cfg = hc_config(eval, sandbox.clone());
            let workspace = sandbox.root.path().to_path_buf();
            let mut t = hill_task(
                sc.payload.direction,
                sc.payload.max_stagnation,
                sc.payload.revert_on_no_improvement,
                sc.payload.min_improvement_delta,
            );
            if let Some(mt) = sc.max_turns {
                t.budget.max_turns = Some(mt);
            }
            let task_id = t.id.as_str().to_string();
            let r = h_run(&cfg, t).await;

            // Halt reason.
            match (sc.expected.halt_reason.as_str(), &r) {
                (
                    "stagnation",
                    RunResult::Failure {
                        reason: HaltReason::StagnationLimitReached { best_metric, .. },
                        ..
                    },
                ) => {
                    if let Some(exp) = sc.expected.best_metric {
                        assert!(
                            (best_metric - exp).abs() < 1e-9,
                            "scenario `{}` best_metric: got {best_metric}, want {exp}",
                            sc.name
                        );
                    }
                }
                (
                    "budget_turns",
                    RunResult::Failure {
                        reason:
                            HaltReason::BudgetExceeded {
                                limit_type: BudgetLimitType::Turns,
                            },
                        ..
                    },
                ) => {}
                (other_exp, other_got) => panic!(
                    "scenario `{}`: expected halt `{other_exp}`, got {other_got:?}",
                    sc.name
                ),
            }

            // Kept-iteration count and revert count from the written TSV.
            let tsv =
                std::fs::read_to_string(workspace.join(format!(".spore/results/{task_id}.tsv")))
                    .unwrap_or_else(|e| panic!("scenario `{}` tsv: {e}", sc.name));
            let kept = tsv
                .lines()
                .skip(1)
                .filter(|l| l.contains("\tkept\t"))
                .count() as u32;
            assert_eq!(
                kept, sc.expected.kept_iterations,
                "scenario `{}` kept count; tsv:\n{tsv}",
                sc.name
            );
            assert_eq!(
                sandbox.revert_count(),
                sc.expected.revert_count,
                "scenario `{}` revert count",
                sc.name
            );

            // Optional exact-TSV byte assertion.
            if let Some(expected_tsv) = &sc.expected_tsv {
                assert_eq!(&tsv, expected_tsv, "scenario `{}` TSV bytes", sc.name);
            }
        }
    }
}
