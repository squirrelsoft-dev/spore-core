//! Harness — the agent runtime loop.
//!
//! Implements issue #3. The harness owns execution lifecycle and wires
//! all components together. It is stateless between `run()` calls; everything
//! the harness needs comes in via [`HarnessRunOptions`] or [`PausedState`],
//! and everything it produces goes out via [`RunResult`].
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
    CompactionPreserveHints, CompactionVerifier, KeyTermVerifier,
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
/// respective component issues. Until those land, only [`LoopStrategy::ReAct`]
/// is fully executable in [`StandardHarness`].
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
    TurnStart { turn: u32 },
    TurnEnd { turn: u32 },
    ToolCall { call_id: String, name: String },
    ToolResult { call_id: String, is_error: bool },
    FinalResponse { content: String },
    BudgetWarning { limit_type: BudgetLimitType },
}

pub type StreamSink = Box<dyn Fn(StreamEvent) + Send + Sync>;

// ============================================================================
// Forward-declared sibling component types
// (full surfaces live in their owning issues)
// ============================================================================

/// Output of a single tool dispatch. Full type lives in issue #4 (ToolRegistry)
/// / #5 (Tool). The variants below cover what the harness loop needs to
/// route; richer payloads are additive.
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
    /// returns `IsolationMode::None`; production sandboxes should override.
    fn isolation_mode(&self) -> IsolationMode {
        IsolationMode::None
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
    /// at construction time.
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
    pub human_request: HumanRequest,
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
    pub human_request: HumanRequest,
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
        }
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
        Self { config }
    }

    /// Read-only access to the assembled config (used by builder tests).
    pub fn config(&self) -> &HarnessConfig {
        &self.config
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
                            human_request: request.clone(),
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
                                    human_request: request.clone(),
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
                                    human_request: request.clone(),
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
                                human_request: request.clone(),
                                task,
                                budget_used,
                                child_state: Some(*child_state),
                            };
                            return RunResult::WaitingForHuman {
                                state: Box::new(state),
                                request,
                            };
                        }

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
            LoopStrategy::PlanExecute { .. } => RunResult::Failure {
                reason: HaltReason::StrategyNotYetImplemented {
                    strategy: "plan_execute".into(),
                },
                session_id: task.session_id,
                usage: AggregateUsage::default(),
                turns: 0,
            },
            LoopStrategy::Ralph => RunResult::Failure {
                reason: HaltReason::StrategyNotYetImplemented {
                    strategy: "ralph".into(),
                },
                session_id: task.session_id,
                usage: AggregateUsage::default(),
                turns: 0,
            },
            LoopStrategy::SelfVerifying => RunResult::Failure {
                reason: HaltReason::StrategyNotYetImplemented {
                    strategy: "self_verifying".into(),
                },
                session_id: task.session_id,
                usage: AggregateUsage::default(),
                turns: 0,
            },
            LoopStrategy::HillClimbing { .. } => RunResult::Failure {
                reason: HaltReason::StrategyNotYetImplemented {
                    strategy: "hill_climbing".into(),
                },
                session_id: task.session_id,
                usage: AggregateUsage::default(),
                turns: 0,
            },
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
            human_request: _hr,
            task,
            budget_used,
            child_state,
        } = state;

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
                human_request: HumanRequest::Clarification {
                    question: "?".into(),
                },
                task: child_task,
                budget_used: BudgetSnapshot::default(),
                parent_tool_call_id: "c".into(),
            }),
            request: HumanRequest::Clarification {
                question: "?".into(),
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
            human_request: HumanRequest::Clarification {
                question: "?".into(),
            },
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
            human_request: HumanRequest::ToolApproval {
                calls: vec![],
                risk_level: RiskLevel::Low,
            },
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

    // Rule: non-ReAct strategies are explicitly marked NotYetImplemented.
    #[tokio::test]
    async fn non_react_strategies_marked_not_yet_implemented() {
        let a = make_agent();
        let h = StandardHarness::new(standard_config(a));
        for s in [
            LoopStrategy::Ralph,
            LoopStrategy::SelfVerifying,
            LoopStrategy::PlanExecute { plan_model: None },
            LoopStrategy::HillClimbing {
                direction: OptimizationDirection::Maximize,
                max_stagnation: None,
                revert_on_no_improvement: false,
                min_improvement_delta: None,
            },
        ] {
            let t = task(s);
            match h.run(HarnessRunOptions::new(t)).await {
                RunResult::Failure {
                    reason: HaltReason::StrategyNotYetImplemented { .. },
                    ..
                } => {}
                other => panic!("expected StrategyNotYetImplemented, got {other:?}"),
            }
        }
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
            human_request: HumanRequest::Clarification {
                question: "what?".into(),
            },
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
            human_request: HumanRequest::Clarification {
                question: "?".into(),
            },
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
}
