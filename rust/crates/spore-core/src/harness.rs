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
use crate::model::{Message, TokenUsage, ToolCall, ToolSchema};

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

/// Issue #7 — ContextManager: assembles per-turn context.
pub trait ContextManager: Send + Sync {
    fn assemble<'a>(&'a self, session: &'a SessionState, task: &'a Task) -> BoxFut<'a, Context>;

    fn append_tool_result<'a>(
        &'a self,
        session: &'a mut SessionState,
        result: &'a ToolResult,
    ) -> BoxFut<'a, ()>;

    fn append_user_message<'a>(
        &'a self,
        session: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()>;

    fn should_compact(&self, session: &SessionState) -> bool {
        let _ = session;
        false
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

/// Issue #12 — ObservabilityProvider. Stubbed to a no-op by default.
pub trait ObservabilityProvider: Send + Sync {
    fn record_turn<'a>(&'a self, turn: u32, usage: &'a TokenUsage) -> BoxFut<'a, ()> {
        let _ = (turn, usage);
        Box::pin(async {})
    }
}

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
        }
    }
}

pub struct StandardHarness {
    config: HarnessConfig,
}

impl StandardHarness {
    pub fn new(config: HarnessConfig) -> Self {
        Self { config }
    }

    fn emit(stream: &Option<StreamSink>, event: StreamEvent) {
        if let Some(s) = stream.as_ref() {
            s(event);
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

    async fn run_react(
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
            let result = self.config.agent.turn(context).await;
            budget_used.turns += 1;
            if let Some(obs) = self.config.observability.as_ref() {
                let zero = TokenUsage::default();
                let u = match &result {
                    TurnResult::ToolCallRequested { usage, .. }
                    | TurnResult::FinalResponse { usage, .. } => usage,
                    TurnResult::Error { usage, .. } => usage.as_ref().unwrap_or(&zero),
                };
                obs.record_turn(budget_used.turns, u).await;
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
                        let output = self.config.tool_registry.dispatch(call.clone()).await;

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
        let session_state = session_state.unwrap_or_default();
        let budget_used = BudgetSnapshot::default();

        match task.loop_strategy.clone() {
            LoopStrategy::ReAct { max_iterations } => {
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
}
