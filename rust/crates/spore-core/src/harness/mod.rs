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
//! 11. Conversation-history threading (issue #102): every terminal
//!     [`RunResult::Success`] and [`RunResult::Failure`] carries the post-run
//!     [`SessionState`] (`messages` + `extras`) so an in-process caller resumes
//!     LOSSLESSLY — tool-call and tool-result turns included — without
//!     reconstructing history from `output`. Opt-in
//!     [`auto_persist_sessions`](HarnessConfig::auto_persist_sessions) layers
//!     [`SessionStore`](crate::storage::SessionStore) auto-load (ReAct /
//!     SelfVerifying only; explicit `with_session_state` wins) + terminal
//!     auto-persist on top for cross-process continuity by `session_id`.
//!     OFF by default: zero session-store I/O and byte-identical message flow.
//! 12. Continue cross-process checkpoint (issue #129). Three parts:
//!     - **Serialized `behavior` field.** All FIVE strategy config structs
//!       (`ReactConfig`, `PlanExecuteConfig`, `SelfVerifyingConfig`, `RalphConfig`,
//!       `HillClimbingConfig`) carry a serialized `behavior:
//!       BudgetExhaustedBehavior` for a UNIFORM wire shape (Q1). CANONICAL
//!       POSITION: on `ReactConfig` it is IMMEDIATELY after `budget`; on every
//!       combinator it is the LAST field. SERDE TREATMENT: `#[serde(default =
//!       "default_budget_behavior")]` (default `Escalate`) so a tree serialized
//!       before #129 still deserializes, but the field is NOT `skip_serializing_if`
//!       — it ALWAYS serializes, so every node carries an explicit
//!       `"behavior":{"kind":...}`. The parallel-language ports MUST emit the
//!       field in these exact positions for byte-identity.
//!     - **`ExhaustedResolution::Continue` is now live (no longer dead).** Each
//!       combinator's `Continue` arm loops IN-PROCESS (the scope's
//!       `resolve_current` reset `steps_taken` + bumped `continues_used`); the
//!       BARE-LEAF resolution site is `drive_strategy`, which honors a top-level
//!       leaf's `behavior` (a NESTED leaf still PROPAGATES — rule 6 — Q1). The
//!       in-process Continue path performs NO serialization (AC3).
//!     - **Resume-side `continues_used` fix (AC2, load-bearing).** A `Continue`
//!       that spans a process pause resumes with the correct count:
//!       `resume_inner` reads `continues_used` from the
//!       `HumanRequest::BudgetExhausted` payload (the SOLE carrier across a pause
//!       — Q3, NO new serialized `PausedState`/`BudgetSnapshot` field) and seeds
//!       it back via `BudgetContext::resumed` (a runtime-only constructor) +
//!       `RunScratch::resume_continues`. Pre-#129 the resume rebuilt the scope
//!       from `steps_taken` only, ZEROING the counter and over-granting continues.
//!     - **Shared checkpoint utility (AC1, Q2).** `PausedState::serialize_checkpoint`
//!       / `load_checkpoint` is the durable pause/resume round-trip reused by BOTH
//!       the cross-process Continue path AND Ralph's pause-propagation. It is JUST
//!       the `PausedState` round-trip — it does NOT unify the two paths' CONTEXT
//!       policies: Continue PRESERVES `session_state.messages`; Ralph DISCARDS +
//!       re-seeds from its own filesystem `.spore/progress.json` checkpoint
//!       (`RalphProgress`/`ralph_seed_session`), which stays Ralph-specific.
//!
//!     No `// SPEC QUESTION:` markers — all three #129 design forks are resolved.
//! 13. Resume seeding + skip re-planning (issue #138). A budget/consult resume of
//!     a composed `PlanExecute` tree now (a) SEEDS the stalled worker — the full
//!     worker [`SessionState`] rides the pause's `PausedState.session_state` (AC2-a)
//!     and is re-attached to the single `InProgress` task on resume via the phase-
//!     agnostic [`RunScratch::resume_seed`] (the old `consult_resume`, generalized);
//!     and (b) SKIPS re-planning when the durable, project-scoped `task_list` is
//!     non-empty (AC1) — the plan phase is not re-run on a re-entry (a Ralph window,
//!     a budget grant, or a consult answer). Plan-phase exhaustion (empty list, no
//!     `InProgress` task) instead resumes the PLAN session from the carried
//!     conversation (AC3). The budget pause carries the worker leaf's toolset handle
//!     (AC4-a, #140 parity). NO new SERIALIZED field — the seed is runtime-only,
//!     reconstructed on resume from `PausedState.session_state`.
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
    CompactionPreserveHints, CompactionVerifier, ContextError, ContextSources, KeyTermVerifier,
    SessionState as ContextSessionState,
};
use crate::execution_registry::{AutoGrantInfo, EscalationMode, ExecutionRegistry};
use crate::guide_registry::{Guide, SessionOutcome};
use crate::memory::Timestamp;
use crate::skills::SkillCatalog;
// Issue #11 — the canonical (rich) middleware surface lives in `crate::middleware`.
// The harness loop wires it directly (Q2 / Phase 3): the former harness-local
// `MiddlewareChain` / `MiddlewareDecision` / `HookPoint` stubs were deleted in
// favour of these. `HookContext` is intentionally NOT imported here — it would
// collide with the harness's own (#69) `HookContext`, and the loop fires through
// the chain's `fire_*` methods rather than constructing contexts itself.
use crate::middleware::{HookPoint, MiddlewareChain, MiddlewareDecision};
use crate::model::{
    Content, Message, ModelParams, Role, StopReason, TokenUsage, ToolCall, ToolSchema,
};
use crate::observability::{
    truncate_field, ContentCaptureConfig, ContextOperation, ContextSpan, GenAiMessage, GenAiRole,
    PricingTable, SpanBase, SpanId, SpanKind, SpanStatus, ToolCallContent, ToolCallSpan,
    ToolResultContent, TurnSpan, WarnEvent, WarnSpan,
};
use crate::observability_outbox::{OutboxConfig, OutboxObservabilityProvider};
use crate::tool_call_repair::ToolCallRepair;
use crate::verifier::{VerifierInput, VerifierVerdict};

/// Boxed future alias used to make the component traits `dyn`-compatible.
/// `trait_variant::make(Send)` generates RPITIT which is not dyn-safe; the
/// harness needs `Arc<dyn Trait>` everywhere, so we hand-roll the future
/// return shape.
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

mod react;
mod self_verifying;
mod ralph;
mod hill_climbing;
mod plan_execute;
mod compaction;
mod strategy_executor;
mod run;

mod budget;
pub use budget::*;

mod types;
pub use types::*;

mod config;
pub use config::*;

mod strategies;
pub(crate) use strategies::*;

mod providers;
pub use providers::*;

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
// Task + loop strategy
// ============================================================================

/// HillClimbing optimization direction (formerly `HillClimbingDirection`).
/// Renamed in #119 (Composable Execution Part A) to live alongside the other
/// strategy-config types. Wire format is unchanged: snake_case `"minimize"` /
/// `"maximize"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HillClimbingDirection {
    Minimize,
    Maximize,
}

// ============================================================================
// Composable Execution Part A (issue #119): recursive LoopStrategy config
// newtypes + per-node collaborator handles + StrategyRef + RunStrategy trait.
// ============================================================================
//
// Types defined in this section:
//   - Handle newtypes: AgentRef, ToolsetRef, SchemaRef (transparent strings).
//   - Config structs: ReactConfig, PlanExecuteConfig, SelfVerifyingConfig,
//     RalphConfig, HillClimbingConfig.
//   - LoopStrategy: closed serde enum of config newtypes (recursive via
//     Box<LoopStrategy> children on the combinator variants).
//   - StrategyRef: adjacently-tagged BuiltIn(LoopStrategy) | Custom(String).
//   - RunStrategy trait + delegation impl on LoopStrategy (the single match
//     site) + stub impls on each *Config.
//   - Placeholder runtime types: ExecutionContext, StrategyOutcome (#123 owns
//     their real shapes).
//
// Trait method: `RunStrategy::run(&self, &mut ExecutionContext) ->
// StrategyOutcome` (native async).
//
// Rules enforced:
//   - `LoopStrategy` is `#[serde(tag = "kind", rename_all = "snake_case")]`;
//     the `ReAct` variant overrides its tag to `"react"` (not `"re_act"`).
//   - Newtype-variant inner structs flatten next to `"kind"` (flat wire form);
//     recursive `Box<LoopStrategy>` children nest as tagged objects.
//   - JSON field order follows struct declaration order (cross-language
//     byte-identity target).
//   - Built-ins stay a closed enum (versioning + `max_steps()`, #122);
//     `StrategyRef::Custom` is the opaque escape hatch (registry: #120).
//   - Per-variant run bodies are REAL (#124): each `*Config::run` owns its loop
//     and recurses via `self.inner.run(cx)` (or `self.plan`/`self.execute`); the
//     model-touching machinery stays on the harness behind [`StrategyExecutor`].
//     The harness entry collapses to `task.loop_strategy.run(cx).await` — the
//     central dispatch `match` is GONE (the only `match` left is the enum→config
//     delegation in `LoopStrategy::run`).
//
// No `// SPEC QUESTION:` markers — the plan is fully pinned.

/// Per-node handle to a named agent definition. Serializes as a bare JSON
/// string. Resolution to a concrete agent lands with the registry slice (#120).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentRef(pub String);

/// Per-node handle to a named toolset. Serializes as a bare JSON string.
/// Resolution lands with the registry slice (#120). `Default` yields the empty
/// handle (`ToolsetRef("")`) — the global-catalogue fallback — which backs the
/// `#[serde(default)]` on [`PausedState::toolset`] (#140).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolsetRef(pub String);

/// Reserved [`ToolsetRef`] handle that resolves to [`EmptyToolRegistry`] — the
/// model is offered NO tools. Used by the PlanExecute plan-empty re-ask to force
/// the planner to answer in message content instead of derailing into a tool
/// call. The double-underscore makes a real consumer toolset name collision
/// implausible; [`effective_tool_registry`](StandardHarness::effective_tool_registry)
/// intercepts it before the catalogue lookup.
pub const NO_TOOLS_TOOLSET: &str = "__none__";

/// Per-node handle to a named output/evaluator schema. Serializes as a bare
/// JSON string. Resolution lands with the registry slice (#120).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaRef(pub String);

/// Leaf ReAct node config. `budget` is the renamed `max_iterations`
/// (semantically `PerLoop(n)`). Serializes flat next to `"kind":"react"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactConfig {
    pub budget: BudgetPolicy,
    /// What this node does when its `budget` is spent (#129). Canonical position:
    /// IMMEDIATELY after `budget`. A leaf honors its `behavior` ONLY at the
    /// top-level/bare-leaf resolution site (`drive_strategy`); in the normal
    /// NESTED case the leaf still PROPAGATES exhaustion to its parent (#125 rule
    /// 6 — a nested leaf never self-resolves). Always serialized (Q1).
    #[serde(default = "default_budget_behavior")]
    pub behavior: BudgetExhaustedBehavior,
    pub agent: AgentRef,
    pub toolset: ToolsetRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<SchemaRef>,
    /// Per-leaf operating system-prompt OVERRIDE (SC-10). `None` (the default)
    /// preserves today's behaviour: the leaf's turn window uses the global
    /// [`HarnessConfig::system_prompt`]. When `Some`, THIS prompt REPLACES the
    /// global one for every turn of this leaf's window — the leaf sees ONLY its
    /// own prompt, nothing leaks from the configured global prompt. This is the
    /// per-leaf prompt half of SC-10; the per-leaf TOOLSET half is the existing
    /// [`toolset`](Self::toolset) handle. In a [`PlanExecuteConfig`] this lets the
    /// plan and execute phases run under DISTINCT system prompts (each phase's
    /// leaf carries its own). Omitted from the wire when unset, so existing
    /// strategy configs serialize byte-identically. Canonical position: LAST
    /// field (after [`output`](Self::output)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

impl ReactConfig {
    /// A bare ReAct leaf with a `PerLoop { value }` budget and empty agent /
    /// toolset handles (resolution lands with the registry slice, #120). This
    /// is the migration shim for the old `ReAct { max_iterations }` shape.
    pub fn per_loop(value: u32) -> Self {
        Self {
            budget: BudgetPolicy::PerLoop { value },
            behavior: default_budget_behavior(),
            agent: AgentRef(String::new()),
            toolset: ToolsetRef(String::new()),
            output: None,
            system_prompt: None,
        }
    }

    /// The `max_iterations` value extracted from a `PerLoop` budget; any other
    /// budget shape yields `u32::MAX` (matching the legacy executor's
    /// "unbounded" fall-through for non-ReAct/non-`PerLoop` strategies).
    pub fn max_iterations(&self) -> u32 {
        match self.budget {
            BudgetPolicy::PerLoop { value } => value,
            _ => u32::MAX,
        }
    }
}

/// PlanExecute combinator: a `plan` sub-strategy feeds an `execute`
/// sub-strategy. `plan_model` stays optional/omittable (as today).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanExecuteConfig {
    pub plan: Box<LoopStrategy>,
    pub execute: Box<LoopStrategy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_model: Option<ModelConfig>,
    /// What this combinator does when its execute-phase budget is spent (#129).
    /// Canonical position on a combinator: the LAST field. Always serialized (Q1).
    #[serde(default = "default_budget_behavior")]
    pub behavior: BudgetExhaustedBehavior,
}

impl PlanExecuteConfig {
    /// A PlanExecute whose plan and execute phases are both bare ReAct leaves
    /// (migration shim for the old `PlanExecute { plan_model }` shape; the live
    /// executor does not read the children yet — #124).
    pub fn simple(plan_model: Option<ModelConfig>) -> Self {
        Self {
            plan: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
            execute: Box::new(LoopStrategy::ReAct(ReactConfig::per_loop(u32::MAX))),
            plan_model,
            behavior: default_budget_behavior(),
        }
    }
}

/// SelfVerifying combinator: run `inner`, then judge it against `evaluator`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelfVerifyingConfig {
    pub inner: Box<LoopStrategy>,
    pub evaluator: SchemaRef,
    /// Optional dedicated reviewer agent for the evaluate phase. `None` ⇒ the
    /// evaluate phase reuses the inner worker's resolved agent (Q1c default).
    /// Set this to a SEPARATE agent so the reviewer is not the same model that
    /// wrote the code (a builder reviewing its own work tends to rubber-stamp).
    /// Omitted from the wire when unset (existing configs stay byte-identical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_agent: Option<AgentRef>,
    /// Optional read-only/inspection toolset for the evaluate phase. `None` ⇒
    /// the empty handle (global-catalogue fallback), byte-identical to prior
    /// behavior. The evaluate phase already runs on a read-only sandbox; scoping
    /// the toolset to read-only tools additionally prevents the reviewer from
    /// reaching non-filesystem side-effecting tools (web/MCP) that the sandbox
    /// does not gate. Omitted from the wire when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_toolset: Option<ToolsetRef>,
    /// What this combinator does when its build↔evaluate budget is spent (#129).
    /// Canonical position on a combinator: the LAST field. Always serialized (Q1).
    #[serde(default = "default_budget_behavior")]
    pub behavior: BudgetExhaustedBehavior,
}

/// Ralph combinator: re-run `inner` under a fixed `agent` across context-window
/// resets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RalphConfig {
    pub inner: Box<LoopStrategy>,
    pub agent: AgentRef,
    /// What Ralph does when its own scope is spent (#129). Canonical position on
    /// a combinator: the LAST field. Always serialized (Q1). NOTE: Ralph's window
    /// recovery (reset + retry) is independent of this field — it governs Ralph's
    /// OWN budget scope, not the per-window child exhaustion (which Ralph already
    /// absorbs as "window incomplete" and retries).
    #[serde(default = "default_budget_behavior")]
    pub behavior: BudgetExhaustedBehavior,
}

/// HillClimbing combinator: iterate `inner`, keeping/reverting per the metric
/// `evaluator` and `direction`. `max_stagnation` and `min_improvement_delta`
/// are now required (#119).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HillClimbingConfig {
    pub inner: Box<LoopStrategy>,
    pub direction: HillClimbingDirection,
    pub max_stagnation: u32,
    pub revert_on_no_improvement: bool,
    pub min_improvement_delta: f64,
    pub evaluator: AgentRef,
    /// What this combinator does when its optimization-loop budget is spent
    /// (#129). Canonical position on a combinator: the LAST field. Always
    /// serialized (Q1).
    #[serde(default = "default_budget_behavior")]
    pub behavior: BudgetExhaustedBehavior,
}

/// Durable candidate pointers threaded through a `HillClimbing` run
/// (`hill-climbing-loop-correctness`). The pre-fix loop tracked only a bare
/// `f64` high-water score and never committed the accepted candidate, so a
/// revert snapped the tree all the way back to the original start rather than
/// to the last KEPT state — destroying accepted progress. This struct adds the
/// missing primitive: a real per-accepted-candidate reference.
///
/// `current_*` is where the NEXT neighbor branches from (the last accepted
/// state); `best_*` is the best artifact ever accepted. They are always equal
/// under strict greedy climbing, but are tracked separately from day one (D2)
/// so that annealing / random-restart can diverge them later without a ledger
/// refactor.
///
/// The `*_sha` fields are `None` when the workspace has no usable VCS (e.g. the
/// no-op default provider, or a sandbox with no git): in that degenerate world
/// the loop cannot commit per-candidate and gracefully falls back to the
/// pre-fix behaviour (revert targets `HEAD`). With a real git workspace (or a
/// wired [`VcsProvider`]) the SHAs are populated and the climb is correct.
#[derive(Debug, Clone, PartialEq)]
pub struct HillCandidate {
    /// SHA of the last ACCEPTED candidate — the base the next neighbor branches
    /// from and the target a no-improvement revert resets to. `None` ⇒ no usable
    /// VCS (revert falls back to `HEAD`).
    pub current_sha: Option<String>,
    /// SHA of the best artifact ever accepted (winner applied at loop exit).
    pub best_sha: Option<String>,
    /// Score of the current accepted candidate (the keep/revert comparison
    /// baseline — pre-fix `current_best`).
    pub current_best: f64,
    /// Best score ever accepted. Equal to `current_best` under strict greedy.
    pub best_score: f64,
}

/// Git pathspec excluding the harness's own `.spore/` directory from a
/// HillClimbing candidate commit (`hill-climbing-loop-correctness`). The results
/// ledger lives at `{workspace_root}/.spore/results/` — it must stay UNTRACKED so
/// a `git reset --hard` revert/winner-apply never reverts or clobbers it (reset
/// leaves untracked files alone). Used by both the harness git-through-sandbox
/// fallback and [`GitVcsProvider::commit_checkpoint`]; kept byte-identical across
/// the four reference implementations (per `consumer-friction-port-patterns`).
const HILL_CLIMB_COMMIT_EXCLUDE: &str = ":(exclude).spore";

/// Loop strategy — a closed, recursive serde enum of config newtypes. The
/// `ReAct` variant is the leaf; the rest are combinators holding
/// `Box<LoopStrategy>` children. See the section header above for the full
/// type/wire-format/rule documentation.
///
/// Concrete strategy traits (`CompletionCheck`, `Verifier`, `MetricEvaluator`)
/// are owned by their respective component issues. The live `StandardHarness`
/// executes ReAct/PlanExecute via its own dispatch (migrating to the
/// per-`*Config` `run` bodies in #124); the remaining variants still halt with
/// [`HaltReason::StrategyNotYetImplemented`] until then.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoopStrategy {
    #[serde(rename = "react")]
    ReAct(ReactConfig),
    PlanExecute(PlanExecuteConfig),
    SelfVerifying(SelfVerifyingConfig),
    Ralph(RalphConfig),
    HillClimbing(HillClimbingConfig),
}

impl LoopStrategy {
    /// Advisory worst-case **turn** bound for this strategy tree, computed
    /// statically before a run (#122).
    ///
    /// This is an **advisory pre-run figure**, logged at startup — it is NOT
    /// the enforcement mechanism. The actual safety ceiling is the per-node
    /// [`BudgetPolicy`] charged by each `*Config::run` against its own
    /// [`BudgetContext`]; `max_steps` never gates execution.
    ///
    /// The bound is derived **multiplicatively/additively down the tree** and
    /// is Option-monadic: any `Unlimited` node anywhere in the tree collapses
    /// the whole figure to `None` ("no finite advisory bound"). It is a
    /// runtime-only computation and is **never serialized**.
    ///
    /// Per-variant rules:
    /// - `ReAct(c)` ⇒ `c.budget.allowance_value()` (`Unlimited ⇒ None`).
    /// - `SelfVerifying(c)` ⇒ `inner + 1` — the single read-only evaluator
    ///   turn is exactly one extra turn.
    /// - `PlanExecute(c)` ⇒ `plan + execute`. **PER-TASK bound**: a full run's
    ///   total is `plan + execute_per_task × task_count`, where `task_count` is
    ///   data-dependent (the plan phase builds the task graph at runtime), so
    ///   it is intentionally NOT part of this static figure.
    /// - `Ralph(c)` ⇒ `inner` — **PER-WINDOW bound**, mirroring PlanExecute's
    ///   per-task treatment. A full run's total is `per_window × max_windows`,
    ///   where `max_windows` derives from [`HarnessConfig::max_resets`]
    ///   (default 3) at runtime and is intentionally NOT part of this static
    ///   figure.
    /// - `HillClimbing(c)` ⇒ `inner × (max_stagnation + 1)` — the `+1` is the
    ///   one productive pass; `max_stagnation` non-improving passes follow.
    ///   The `max_stagnation == u32::MAX` sentinel means unbounded windows ⇒
    ///   `None` (a semantic unbounded rule, distinct from arithmetic overflow).
    ///
    /// Arithmetic uses `checked_add`/`checked_mul`; an unrepresentable bound
    /// (overflow) yields `None` — an unrepresentable figure is "no finite
    /// advisory bound".
    pub fn max_steps(&self) -> Option<u32> {
        match self {
            LoopStrategy::ReAct(c) => c.budget.allowance_value(),
            LoopStrategy::SelfVerifying(c) => c.inner.max_steps()?.checked_add(1),
            LoopStrategy::PlanExecute(c) => c.plan.max_steps()?.checked_add(c.execute.max_steps()?),
            LoopStrategy::Ralph(c) => c.inner.max_steps(),
            LoopStrategy::HillClimbing(c) => {
                if c.max_stagnation == u32::MAX {
                    return None;
                }
                c.inner
                    .max_steps()?
                    .checked_mul(c.max_stagnation.checked_add(1)?)
            }
        }
    }
}

/// Serializable identity of a strategy: either a closed built-in
/// [`LoopStrategy`] tree or an opaque `Custom` string key resolved at runtime
/// (registry: #120). Adjacently tagged on `kind`/`value` to avoid a tag
/// collision with the nested `LoopStrategy`'s own `kind`:
///   - `{"kind":"built_in","value":{"kind":"react",...}}`
///   - `{"kind":"custom","value":"my-harness::DoubleVerify"}`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum StrategyRef {
    BuiltIn(LoopStrategy),
    Custom(String),
}

impl StrategyRef {
    /// Advisory worst-case turn bound for this strategy reference (#122).
    /// `Custom` is opaque to the framework (it cannot be introspected), so it
    /// yields `None`; `BuiltIn` delegates to [`LoopStrategy::max_steps`].
    pub fn max_steps(&self) -> Option<u32> {
        match self {
            StrategyRef::Custom(_) => None,
            StrategyRef::BuiltIn(s) => s.max_steps(),
        }
    }
}

// ============================================================================
// Composable Execution runtime scaffold (issue #123): StrategyOutcome +
// ExecutionContext / BudgetContext / BudgetStack / SpanStack.
// ============================================================================
//
// SCAFFOLD ONLY. This slice establishes the typed strategy outcome and the
// shared, mutable runtime context that threads through a nested strategy tree.
// `charge` here is PURE ARITHMETIC against a per-scope step allowance — the
// behavior-chain walk, continue-consumption, and persistence are the later
// budget-enforcement slice (#124+).
//
// Types defined in this section:
//   - `StrategyOutcome` — typed result a strategy node returns.
//   - `BudgetExhausted` — the error type `BudgetContext::charge` returns.
//   - `BudgetContext` — one budget scope (`charge`/`remaining`/
//     `continues_remaining`).
//   - `BudgetStack` — runtime push/pop stack of `BudgetContext`, one node per
//     recursion frame (siblings do NOT share).
//   - `SpanStack` — runtime push/pop stack of `observability::SpanId`.
//   - `ExecutionContext<'a>` — the one shared, mutable context for a whole
//     nested tree; borrows the [`ExecutionRegistry`].
//
// Trait methods (updated this slice):
//   - `RunStrategy::run<'a>(&'a self, &'a mut ExecutionContext<'_>) ->
//     BoxFut<'a, StrategyOutcome>` — the `ExecutionContext` now carries the
//     real runtime scaffold and a borrowed registry lifetime.
//
// Rules enforced:
//   - A child's `StrategyOutcome::BudgetExhausted` is an INSPECTABLE value the
//     parent reads; it does NOT auto-propagate.
//   - `charge` is pure arithmetic: it debits `turns` steps; on success
//     increments `steps_taken` and returns `Ok(())`; on overflow returns
//     `Err(BudgetExhausted{..})` from current state. It does NOT walk the
//     behavior chain or consume continues. `BudgetPolicy::Unlimited` never
//     exhausts.
//   - Each `BudgetContext` represents ONE scope; the allowance is the policy's
//     own `value` (Unlimited = no cap → `None`).
//   - All runtime types here are NEVER serialized (no serde derives). Rust
//     cannot assert the *absence* of a `Serialize` impl via a negative trait
//     bound, so this is structural-only, not test-asserted.
//
// Resolved spec ambiguities (DECIDED — see issue #123 plan):
//   1. `ExecutionContext<'a>` borrows the registry (`&'a ExecutionRegistry`);
//      the `RunStrategy` trait + all impls thread the context lifetime.
//      `Arc<dyn RunStrategy>` stays dyn-compatible.
//   2. `charge` is pure arithmetic (above); `BudgetExhausted` is a dedicated
//      struct used as charge's error; `StrategyOutcome::BudgetExhausted` is a
//      struct variant that mirrors those fields and adds `partial_output`
//      (modeled as a struct variant rather than wrapping, so callers can match
//      and read all fields uniformly). `Output` maps to `String`.
//   3. `continues_used` is an IN-MEMORY field ONLY in this slice. Its "rides a
//      Continue/Escalate checkpoint" persistence is DEFERRED to the
//      enforcement slice — it is intentionally NOT added to
//      `BudgetSnapshot`/`PausedState` here.
//   4. `SpanStack` reuses [`observability::SpanId`](crate::observability::SpanId)
//      as its element type.
//
// No `// SPEC QUESTION:` markers — the plan is fully pinned.

/// The typed result a strategy node returns. Runtime-only (NOT serialized): a
/// strategy outcome is an in-process value, never persisted.
///
/// `BudgetExhausted` is distinguishable from `Failed` by callers — a child's
/// `BudgetExhausted` is a value the parent INSPECTS (e.g. to grant a continue
/// or escalate); it does NOT auto-propagate as a failure. `Output` maps to
/// `String` in this codebase (mirrors [`RunResult`]).
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyOutcome {
    /// The strategy completed and produced its final output.
    Complete(String),
    /// The strategy's budget scope ran out of allowance — OR (issue #137) the
    /// ReAct consecutive-tool-error breaker tripped (see `cause`). Mirrors the
    /// [`BudgetExhausted`] charge-error fields and adds `partial_output` — any
    /// output produced before exhaustion. Inspectable, NOT auto-propagating.
    ///
    /// The `cause` discriminant (#137) lets the SINGLE budget-exhaustion
    /// resolution site (`drive_strategy_with_resume_seed`) pick the right
    /// terminal [`HaltReason`]: genuine budget exhaustion →
    /// [`HaltReason::BudgetExceeded`]; an error loop →
    /// [`HaltReason::ToolErrorLoop`]. Both flow through the node's
    /// [`BudgetExhaustedBehavior`] (Fail / Escalate / Continue) identically.
    BudgetExhausted {
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        steps_taken: u32,
        continues_used: u32,
        phase: String,
        partial_output: Option<String>,
        /// What caused this exhaustion (#137): genuine budget vs. an error loop.
        /// Defaults to [`ExhaustionCause::Budget`] for every pre-#137 path.
        cause: ExhaustionCause,
    },
    /// The strategy halted with a harness error.
    Failed(HarnessError),
}


/// The one shared, mutable runtime context threaded through a whole nested
/// strategy tree. Borrows the [`ExecutionRegistry`] for the duration of the
/// run (`'a`). Runtime-only — NOT serialized.
///
/// `stream` holds an optional boxed `dyn Fn` [`StreamSink`], which is neither
/// `Debug` nor `Clone`; like [`HarnessRunOptions`] this type therefore does not
/// derive `Debug`.
///
/// ## Recursive executor wiring (#124)
///
/// The per-variant [`RunStrategy::run`] bodies are LOOP OWNERS: a combinator
/// recurses by calling `self.inner.run(cx).await` (or `self.plan`/`self.execute`
/// as applicable). For PlanExecute (#124) this recursion is GENUINE: the plan
/// phase dispatches `self.plan.run(cx)` and the execute phase dispatches
/// `self.execute.run(cx)` once per task, so a non-ReAct execute child really
/// executes its loop per task. The leaf model-touching primitives (the ReAct
/// turn-loop window, the SelfVerifying evaluate phase, the HillClimbing metric
/// machinery, the Ralph `.spore/` reload/completion checks, and the PlanExecute
/// artifact-capture / deep-resume / hook helpers) stay on the harness behind the
/// [`StrategyExecutor`] trait, reachable through `executor` — but the per-task
/// ORCHESTRATION loop lives in `PlanExecuteConfig::run`, where it can call the
/// child `.run(cx)`. The per-run mutable orchestration state (`task`, `on_stream`,
/// `run_session`, `run_budget`) lives in the [`RunScratch`] so it threads across
/// recursion through this one shared context — `ExecutionContext::run_strategy`
/// is the single entry the harness calls instead of the old central dispatch
/// `match`.
pub struct ExecutionContext<'a> {
    /// Borrowed handle registry (agents/toolsets/schemas/custom strategies).
    pub registry: &'a ExecutionRegistry,
    /// Per-scope budget stack, pushed/popped through recursion.
    pub budgets: BudgetStack,
    /// Aggregated token/cost usage across the whole tree.
    pub usage: AggregateUsage,
    /// Conversation/session state round-tripped across the tree.
    pub session: SessionState,
    /// Span stack for observability nesting.
    pub spans: SpanStack,
    /// Optional streaming sink for emitted events.
    pub stream: Option<StreamSink>,
    /// The harness primitives the per-variant run bodies delegate to (#124).
    /// `None` only for the scaffold/unit fixtures that exercise the runtime
    /// context without a real harness (the recursion stub tests).
    pub executor: Option<&'a dyn StrategyExecutor>,
    /// Per-run orchestration scratch threaded across recursion (#124).
    pub scratch: RunScratch,
}

/// Per-run mutable orchestration state threaded through the recursive strategy
/// tree (#124). Runtime-only (NOT serialized). The combinator bodies set up the
/// sub-phase `task` here before recursing, and the leaf ([`ReactConfig::run`])
/// reads it to drive the ReAct window. `on_stream` is moved out of the context
/// only at the harness entry (it is `!Clone`), so it lives here rather than
/// being re-threaded per call.
#[derive(Default)]
pub struct RunScratch {
    /// The task whose strategy is currently executing. Combinators swap in a
    /// per-phase sub-task before recursing and restore it after.
    pub task: Option<Task>,
    /// The conversation/session state the current leaf turn-loop builds on.
    pub run_session: SessionState,
    /// The shared budget snapshot threaded across every sub-loop of this run.
    pub run_budget: BudgetSnapshot,
    /// A non-terminal pause (`WaitingForHuman`/`Consult`/`Escalate`) or a
    /// fully-formed terminal that must propagate up the recursion VERBATIM as a
    /// [`RunResult`] rather than being collapsed into a [`StrategyOutcome`]. The
    /// harness entry returns this directly when set, preserving the pause/escalate
    /// contract through the recursive executor (#124).
    pub terminal_override: Option<RunResult>,
    /// Cross-process Continue checkpoint seed (#129): `(phase, continues_used)`
    /// carried from a resumed [`HumanRequest::BudgetExhausted`]. The FIRST
    /// [`push_budget`](ExecutionContext::push_budget) whose `phase` matches seeds
    /// the reconstructed scope's `continues_used` (via [`BudgetContext::resumed`])
    /// and CLEARS this seed — so a `Continue` spanning a process pause resumes
    /// with the correct continue count (AC2). Runtime-only; the value rides the
    /// request payload, NOT a serialized `BudgetContext`/`PausedState` field (Q3).
    /// `None` on a fresh run and after the seed is consumed (an in-process
    /// Continue never sets it → AC3: no serialization on the in-process path).
    pub resume_continues: Option<(String, u32)>,
    /// Phase-agnostic resume seed (#131 consult re-drive + #138 budget resume):
    /// the stalled worker conversation carried from a resumed pause. For a
    /// CONSULT resume the consult answer is already injected as the pending
    /// call's tool result; for a BUDGET resume (#138) it is the worker's full
    /// post-exhaustion session. When set, a [`PlanExecuteConfig`] walk resumes
    /// its single `InProgress` task from THIS session (instead of a fresh
    /// instruction-seeded session) so the stalled worker continues mid-loop, its
    /// SelfVerifying evaluator still runs, and the ready-set walk proceeds.
    ///
    /// #138 AC3 (plan-phase exhaustion): when the durable task_list has NO
    /// `InProgress` task, the exhaustion happened in the PLAN phase before any
    /// task was authored — the walk seeds the PLAN session from this carried
    /// conversation instead of cloning a fresh base session.
    ///
    /// The FIRST PlanExecute walk consumes and CLEARS it. Runtime-only — the
    /// session itself rides the serialized [`PausedState::session_state`], so a
    /// cross-process resume reconstructs this seed in
    /// [`resume_consult`](Harness::resume_consult) (consult) or
    /// [`resume`](Harness::resume)'s `ContinueWithBudget` arm (#138 budget).
    /// `None` on a fresh run / after the seed is consumed.
    pub resume_seed: Option<SessionState>,
    /// Explicit "continue the durable task list" signal for a [`PlanExecuteConfig`]
    /// re-entry that carries NO `resume_seed` (there is no stalled worker
    /// conversation to thread in — the continuation is purely "keep walking the
    /// already-authored DAG"). Set by **Ralph** (each window continues the one
    /// logical task across the per-window session reset) and by the **`Skip`**
    /// budget-escalation resolution (advance past the skipped task). A `resume_seed`
    /// already implies continuation, so the seed-carrying resume paths leave this
    /// `false`. A fresh `run()` and a HillClimbing neighbor leave it `false`, so
    /// PlanExecute plans FRESH by default (it no longer infers "resume" from the
    /// mere presence of a project-scoped durable list). Consumed (reset to `false`)
    /// by the FIRST PlanExecute that reads it, so it never leaks into a nested
    /// per-task PlanExecute, which must plan fresh.
    pub continue_task_list: bool,
    /// One-shot FORCED output schema for the immediate sub-run (the plan-empty
    /// REPAIR re-ask, #139). When `Some`, the next [`ReactConfig::run`] uses it as
    /// the leaf's `output_schema` REGARDLESS of the global `enforce_output_schemas`
    /// flag — so the default path "trusts the model" (unconstrained), and only the
    /// repair forces a schema-constrained (`response_format`) retry. Consumed
    /// (`take`) by the leaf that reads it. `None` everywhere else → byte-identical.
    pub forced_output_schema: Option<serde_json::Value>,
    /// One-shot marker that the immediate sub-run is a PlanExecute EXECUTE step.
    /// Set by the execute walk before `self.execute.run(cx)` and consumed by the
    /// next [`ReactConfig::run`]. Scopes the behavioral tool-call nudge
    /// ([`HarnessConfig::tool_call_nudge`]) to execute steps, so the plan-JSON turn
    /// is never nudged for "no tool call". `false` everywhere else.
    pub in_execute_step: bool,
}

impl<'a> ExecutionContext<'a> {
    /// A fresh context bound to `registry`, with empty stacks and default
    /// usage/session and no stream sink.
    pub fn new(registry: &'a ExecutionRegistry) -> Self {
        Self {
            registry,
            budgets: BudgetStack::default(),
            usage: AggregateUsage::default(),
            session: SessionState::default(),
            spans: SpanStack::default(),
            stream: None,
            executor: None,
            scratch: RunScratch::default(),
        }
    }

    /// The current per-run task (panics only in misuse — the harness always sets
    /// it before driving a strategy). Cloned because the recursive bodies borrow
    /// the context mutably while reading the task.
    fn current_task(&self) -> Task {
        self.scratch
            .task
            .clone()
            .expect("ExecutionContext::scratch.task must be set before running a strategy")
    }

    /// The executor primitives, or a typed failure outcome when absent (the
    /// scaffold-only contexts). Real harness runs always wire one.
    fn executor(&self) -> Result<&'a dyn StrategyExecutor, StrategyOutcome> {
        self.executor.ok_or_else(|| {
            StrategyOutcome::Failed(HarnessError::InvalidConfiguration(
                "ExecutionContext has no StrategyExecutor wired".to_string(),
            ))
        })
    }
}

/// The runtime composition seam: every strategy node knows how to run itself
/// given an [`ExecutionContext`]. Implemented once on [`LoopStrategy`] as the
/// ONLY `match` site — one-line delegation per arm to the inner config's `run`
/// (#124 — the central dispatch `match` in `run_inner` is gone). Each `*Config`
/// impl OWNS its loop: a combinator recurses via `self.inner.run(cx)` (or
/// `self.plan`/`self.execute`), and the leaf [`ReactConfig::run`] drives one
/// bounded ReAct window through the [`StrategyExecutor`] primitive. Without a
/// wired executor (the scaffold-only contexts) every body returns a TYPED
/// [`StrategyOutcome::Failed`] — never a panic/unimplemented/todo (CONVENTIONS).
///
/// Uses the hand-rolled [`BoxFut`] return shape (NOT `trait_variant::make`) so
/// that `Arc<dyn RunStrategy>` is dyn-compatible — the
/// [`ExecutionRegistry`](crate::ExecutionRegistry) custom-strategy escape hatch
/// (#120) stores trait objects keyed by `StrategyRef::Custom` key. This trait is
/// never serialized, so the dyn-safe shape has no wire impact.
pub trait RunStrategy: Send + Sync {
    fn run<'a>(&'a self, cx: &'a mut ExecutionContext<'_>) -> BoxFut<'a, StrategyOutcome>;
}

/// The harness-side primitives the per-variant [`RunStrategy::run`] bodies
/// delegate to (#124). Implemented by [`StandardHarness`]. This is the seam that
/// lets the recursive config bodies own their loops while the model-touching
/// machinery (ReAct turn-loop window, evaluate phase, plan phase, metric
/// machinery, `.spore/` checks) stays where it is tested.
///
/// Every primitive returns a terminal [`RunResult`] for its phase; the config
/// bodies translate the terminal into a [`StrategyOutcome`] (or recurse). The
/// trait is dyn-compatible via [`BoxFut`] so [`ExecutionContext`] can hold
/// `&dyn StrategyExecutor`.
pub trait StrategyExecutor: Send + Sync {
    /// Run ONE bounded ReAct turn-loop window over `session_state`, carrying the
    /// shared `budget_used`. This is the leaf primitive (the body of the legacy
    /// `run_react_inner`). Does NOT finalize observability — the caller does.
    #[allow(clippy::too_many_arguments)]
    fn react_window<'a>(
        &'a self,
        task: Task,
        max_iterations: u32,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        agent: Arc<dyn Agent>,
        // Issue 2 (per-node toolset scoping): the leaf's RESOLVED `toolset`
        // handle, threaded alongside `agent`. Empty ⇒ global-catalogue fallback;
        // a non-empty handle with its own catalogue ⇒ strict per-node scoping.
        toolset: ToolsetRef,
        // Issue #139: the leaf's RESOLVED output schema (canonical compact
        // key-sorted JSON) when `enforce_output_schemas` is ON and the leaf
        // carries `output = Some(..)`; `None` otherwise (no delivery, no
        // validation). When `Some`, the window VALIDATES its terminal against it
        // and retries up to `output_schema_max_retries` extra turns, and sets
        // [`ModelParams::output_schema`] on every turn so the Ollama `format`
        // channel constrains decoding.
        output_schema: Option<serde_json::Value>,
        // Issue #139: the `N` extra validation-retry turns (total attempts =
        // `1 + N`). Ignored when `output_schema` is `None`.
        output_schema_max_retries: u32,
        // SC-10: the leaf's per-node system-prompt OVERRIDE (`ReactConfig::system_prompt`).
        // `None` ⇒ the window uses the global `config.system_prompt` (byte-identical
        // to pre-SC-10). `Some` ⇒ this prompt REPLACES the global one for every turn
        // of this window, so the leaf sees ONLY its own prompt. Mirrors `toolset`.
        system_prompt: Option<String>,
        // Whether this window is a PlanExecute EXECUTE step (scopes the behavioral
        // tool-call nudge). `false` for plan/other leaves.
        execute_step: bool,
    ) -> BoxFut<'a, RunResult>;

    /// Resolve an [`AgentRef`] to its registered agent (#124). The leaf and the
    /// combinators resolve their worker agent through this so a missing handle is
    /// a typed terminal `Failure` rather than a panic. Returns `Err(RunResult)`
    /// carrying the `UnresolvedHandle` failure when the key is absent.
    fn resolve_agent_ref(
        &self,
        r: &AgentRef,
        session_id: &SessionId,
    ) -> Result<Arc<dyn Agent>, Box<RunResult>>;

    /// Run the SelfVerifying evaluate phase (#124): a fresh evaluator RUN over a
    /// read-only sandbox in a never-shared session, on `eval_agent`, scoped to
    /// `eval_toolset` (empty handle ⇒ global-catalogue fallback). Folds the
    /// run's usage into `total_usage`/`carried`; returns its terminal.
    fn evaluate_phase<'a>(
        &'a self,
        task: &'a Task,
        eval_agent: Arc<dyn Agent>,
        eval_toolset: ToolsetRef,
        carried: &'a mut BudgetSnapshot,
        total_usage: &'a mut AggregateUsage,
    ) -> BoxFut<'a, RunResult>;

    /// Append `reason` as a user message on `session_state` (Default-FAIL).
    fn append_user_message<'a>(
        &'a self,
        session_state: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()>;

    /// The configured sandbox workspace root (#124, for `VerifierInput`).
    fn workspace_root(&self) -> std::path::PathBuf;

    /// The configured HITL-vs-AFK escalation knob (#130, SC-5). The `*Config::run`
    /// bodies only hold `&dyn StrategyExecutor`, so they consult the
    /// [`EscalationMode`] through this accessor at each
    /// [`ExhaustedResolution::Escalate`] site: `SurfaceToHuman` pauses with a
    /// [`HumanRequest::BudgetExhausted`]; `Autonomous` keeps the existing
    /// propagate-up behavior; `AutoContinue` auto-grants in-process (via
    /// [`ExecutionContext::try_auto_continue`]) up to its cap, then falls through
    /// to the `Autonomous` terminal. Mirrors
    /// [`ralph_max_resets`](Self::ralph_max_resets) /
    /// [`workspace_root`](Self::workspace_root).
    fn escalation_mode(&self) -> EscalationMode;

    /// The output-schema enforcement MIGRATION GATE (issue #139). `false` (the
    /// default) means [`ReactConfig::run`] never resolves/delivers/validates a
    /// leaf's `output` schema — byte-identical to pre-#139. The `*Config::run`
    /// bodies only hold `&dyn StrategyExecutor`, so they read it through this
    /// accessor (mirrors [`escalation_mode`](Self::escalation_mode)).
    fn enforce_output_schemas(&self) -> bool;

    /// The number of EXTRA terminal-validation retry turns `N` granted under
    /// output-schema enforcement (issue #139; total attempts = `1 + N`). Read by
    /// [`ReactConfig::run`] to thread into the window. Default `2`.
    fn output_schema_max_retries(&self) -> u32;

    /// Seed a user message onto `session_state` (the `ContextManager` seam).
    fn seed_user_message<'a>(
        &'a self,
        session_state: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()>;

    /// Capture + persist a [`PlanArtifact`] from the plan sub-strategy's final
    /// output text (#124). The leaf model work (the planner turn) now runs through
    /// `self.plan.run(cx)`; this primitive owns only the harness-side artifact
    /// machinery: parse the response (R3), fire `OnPlanCreated` (R11), and persist
    /// to the RunStore under `PLAN_EXECUTE_EXTRAS_KEY` (R4). Returns the captured
    /// artifact or a terminal failure to propagate.
    fn capture_plan_artifact<'a>(
        &'a self,
        session_id: &'a SessionId,
        plan_output: &'a str,
        usage: AggregateUsage,
        turns: u32,
    ) -> BoxFut<'a, Result<PlanPhaseOutcome, RunResult>>;

    /// The planning directive seeded before the plan sub-strategy runs (R1) — the
    /// "respond with a single JSON plan" instruction wrapped around the task.
    fn plan_directive(&self, instruction: &str) -> String;

    /// Dispatch the plan sub-strategy `plan` for `plan_task` over `plan_session`,
    /// returning its terminal [`RunResult`] (#124). Genuinely recursive — the
    /// child's `.run(cx)` drives its whole loop. Routes the configured
    /// `planner_agent` (R5/R6) by running the child against an agent-swapped child
    /// harness when one is set; otherwise the default agent runs the plan turn.
    fn run_plan_subtree<'a>(
        &'a self,
        plan: &'a LoopStrategy,
        plan_task: Task,
        plan_session: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        // #139: a one-shot FORCED output schema for this sub-run (the plan-empty
        // repair re-ask). `None` on the trusted first attempt — byte-identical.
        forced_output_schema: Option<serde_json::Value>,
    ) -> BoxFut<'a, Option<RunResult>>;

    /// Reconcile a freshly-parsed task list against the DURABLE RunStore
    /// checkpoint (A.6 deep-resume): any task already `Completed` on the
    /// checkpoint is marked `Completed` in `task_list` so it is NOT re-run.
    fn reconcile_completed_tasks<'a>(
        &'a self,
        session_id: &'a SessionId,
        task_list: &'a mut crate::tasklist::TaskList,
    ) -> BoxFut<'a, ()>;

    /// Fire the `OnTaskAdvance` hook (pre, mutable) for an execute step. The hook
    /// may rewrite `step_task.instruction`; the (possibly mutated) instruction is
    /// what the execute sub-strategy then runs.
    fn fire_task_advance<'a>(
        &'a self,
        session_id: &'a SessionId,
        step_task: &'a mut Task,
        task_index: usize,
        total_tasks: usize,
    ) -> BoxFut<'a, ()>;

    /// Persist a parsed task list through the RunStore seam (legacy
    /// `persist_task_list`).
    fn persist_task_list<'a>(
        &'a self,
        session_id: &'a SessionId,
        task_list: &'a crate::tasklist::TaskList,
    ) -> BoxFut<'a, ()>;

    /// Load the persisted [`TaskList`](crate::tasklist::TaskList) from the
    /// RunStore seam under
    /// [`TASK_LIST_EXTRAS_KEY`](crate::tasklist::TASK_LIST_EXTRAS_KEY), the one
    /// authoring path (#126, decision C). `None` when no list is persisted for
    /// the session. This is how the DAG executor obtains its task list (with
    /// real blockers) instead of the linear `plan_artifact_to_task_list` bridge.
    fn load_task_list<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<crate::tasklist::TaskList>>;

    /// Clear the persisted [`TaskList`](crate::tasklist::TaskList) from the
    /// RunStore seam (the durable, project-scoped store under
    /// [`TASK_LIST_EXTRAS_KEY`](crate::tasklist::TASK_LIST_EXTRAS_KEY)). Called at
    /// the start of a FRESH plan (a non-resume PlanExecute run) so a prior run's
    /// stale list — which `#142` keys by the stable project namespace, NOT the
    /// per-run instruction — does not survive into and suppress the new plan. The
    /// ephemeral `session_id` is unused (mirrors `persist_task_list`/`load_task_list`).
    fn clear_task_list<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()>;

    /// Drain and reset the harness-observed write/edit file-path accumulator
    /// (#126, AC2). The dispatch seam records the `path` of every `write_file` /
    /// `edit_file` tool call here as it happens; the DAG executor calls this on
    /// task completion to attach the OBSERVED `files_touched` to the task's
    /// [`StepLedgerEntry`](crate::tasklist::StepLedgerEntry). Paths are returned
    /// in observation order, de-duplicated (a file written twice in one task
    /// appears once), and the accumulator is cleared so the next task starts
    /// empty. NEVER model-self-reported — observed from the actual tool call.
    fn take_observed_writes(&self) -> Vec<String>;

    /// Clear the harness-observed write accumulator WITHOUT reading it (#126).
    /// Called before each execute step so a step's `files_touched` reflects only
    /// the writes that step issued (no leakage from a prior step).
    fn clear_observed_writes(&self);

    /// Build the per-window Ralph seed session (#124): a fresh
    /// `SessionState::default()` seeded with `instruction`, the `.spore/` reload
    /// context (R3), and the optional VCS history block — exactly the legacy
    /// `run_ralph` window setup, minus the model loop (which now recurses).
    fn ralph_seed_session<'a>(&'a self, instruction: &'a str) -> BoxFut<'a, SessionState>;

    /// Ralph external completion check (#124, legacy `ralph_completion_status`):
    /// `None` ⇒ complete; `Some(reason)` ⇒ tasks remain. ASYNC since #142: the
    /// Ralph checkpoint moved off the `.spore/` filesystem onto the durable
    /// project-id [`RunStore`](crate::storage::RunStore).
    fn ralph_completion_status<'a>(&'a self) -> BoxFut<'a, Option<String>>;

    /// The Ralph outer-loop reset cap (`config.max_resets`, #124).
    fn ralph_max_resets(&self) -> u32;

    /// PlanExecute plan-empty re-ask cap: how many ADDITIONAL times the plan
    /// phase re-asks the model for a usable plan — nudging it to emit the plan as
    /// message CONTENT — before halting with [`HaltReason::EmptyPlan`]. `0` (the
    /// default) preserves the historical behaviour: the first empty plan halts
    /// immediately. Reasoning models that route their plan into the reasoning
    /// channel (e.g. Qwen behind vLLM) leave the message content empty, so the
    /// strict + prose capture yields no tasks; a small cap lets them recover.
    fn plan_empty_max_retries(&self) -> u32 {
        0
    }

    /// Resolve the HillClimbing metric evaluator for `key` (#124, Q2), or
    /// `Err(RunResult)` with the misconfiguration failure when absent.
    fn resolve_metric_evaluator(
        &self,
        key: &str,
        session_id: &SessionId,
    ) -> Result<Arc<dyn crate::metric::MetricEvaluator>, Box<RunResult>>;

    /// HillClimbing iteration-0 baseline (#124): evaluate the metric (no agent
    /// turn), record the row + span, and return the baseline value — or
    /// `Err(RunResult)` on a baseline-evaluation failure (already records the
    /// failed row + writes the TSV). Leaf primitive of the legacy
    /// `run_hill_climbing`.
    ///
    /// `hill-climbing-loop-correctness` D4: also commits the starting tree so
    /// the climb has a well-defined baseline candidate, and returns its SHA
    /// (`None` when the workspace has no usable VCS). The returned SHA seeds
    /// `current_sha`/`best_sha`.
    #[allow(clippy::too_many_arguments)]
    fn hill_baseline<'a>(
        &'a self,
        evaluator: &'a Arc<dyn crate::metric::MetricEvaluator>,
        session_id: &'a SessionId,
        task_id: &'a TaskId,
        direction: HillClimbingDirection,
        rows: &'a mut Vec<crate::metric::ResultsEntry>,
        span_seq: &'a mut u64,
        total_usage: AggregateUsage,
        turns: u32,
    ) -> BoxFut<'a, Result<(f64, Option<String>), RunResult>>;

    /// HillClimbing per-iteration metric eval + keep/revert decision (#124): the
    /// agent turn already ran (recursively); this evaluates the metric, applies
    /// `should_keep`, and updates the durable candidate pointers in `cand`
    /// (`hill-climbing-loop-correctness`):
    ///   - KEEP ⇒ commit the working tree, advance `current_sha`/`current_best`
    ///     (and `best_sha`/`best_score` if it beats the best), record the row
    ///     with the real accepted SHA;
    ///   - DISCARD / error ⇒ reset the tree to `cand.current_sha` (the last
    ///     ACCEPTED state, NOT the original HEAD) when `revert_on_no_improvement`,
    ///     record the row, leave the pointers unchanged.
    ///
    /// Records the row + span and returns `true` iff this iteration was a
    /// non-improvement (the caller's stagnation increment).
    #[allow(clippy::too_many_arguments)]
    fn hill_iteration<'a>(
        &'a self,
        evaluator: &'a Arc<dyn crate::metric::MetricEvaluator>,
        session_id: &'a SessionId,
        task_id: &'a TaskId,
        iteration: u32,
        direction: HillClimbingDirection,
        revert_on_no_improvement: bool,
        min_improvement_delta: Option<f64>,
        cand: &'a mut HillCandidate,
        rows: &'a mut Vec<crate::metric::ResultsEntry>,
        span_seq: &'a mut u64,
    ) -> BoxFut<'a, bool>;

    /// Apply the best accepted candidate at loop exit (winner application,
    /// `hill-climbing-loop-correctness` §4.6): `reset --hard <best_sha>` so the
    /// working tree holds the winner before a terminal return. A no-op when
    /// `best_sha` is `None`.
    fn hill_apply_winner<'a>(&'a self, best_sha: Option<&'a str>) -> BoxFut<'a, ()>;

    /// Write the HillClimbing results TSV (#124, leaf primitive).
    fn hill_write_tsv<'a>(
        &'a self,
        task_id: &'a TaskId,
        rows: &'a [crate::metric::ResultsEntry],
    ) -> BoxFut<'a, ()>;

    /// Finalize observability for a terminal outcome (legacy
    /// `finalize_observability` routing). No-op for non-terminal results.
    fn finalize<'a>(&'a self, result: &'a RunResult) -> BoxFut<'a, ()>;
}


// ============================================================================
// StandardHarness — the canonical implementation
// ============================================================================


pub struct StandardHarness {
    config: HarnessConfig,
    /// #126: harness-observed write/edit file paths for the CURRENTLY-running
    /// execute step. The tool-dispatch seam pushes the `path` of every
    /// `write_file` / `edit_file` call here as it dispatches; the DAG executor
    /// drains it (`take_observed_writes`) on task completion to build the
    /// observed `files_touched` for that task's
    /// [`StepLedgerEntry`](crate::tasklist::StepLedgerEntry). Interior-mutable
    /// because the executor sees the harness only as `&dyn StrategyExecutor`.
    /// Steps run SEQUENTIALLY (v1 ready-set is sequential), so a single shared
    /// accumulator cleared per step is sufficient and deterministic.
    observed_writes: std::sync::Mutex<Vec<String>>,
}

impl StandardHarness {
    pub fn new(mut config: HarnessConfig) -> Self {
        // Ralph completion mechanism (issue #58, B1): register a `Stop` hook
        // that drives multi-context-window continuation off the Ralph checkpoint.
        // #142: the checkpoint moved off `.spore/progress.json` onto the durable
        // project-id RunStore. Registration is harmless for non-Ralph runs — the
        // hook only BLOCKS when the progress checkpoint key is PRESENT and reports
        // incomplete tasks; when it is absent (the common case for ReAct / other
        // strategies) it returns `Continue`, so existing strategies are unaffected.
        let chain: Arc<dyn crate::hooks::HookChain> = config
            .hooks
            .clone()
            .unwrap_or_else(|| Arc::new(crate::hooks::StandardHookChain::new()));
        // Best-effort: a duplicate/invalid registration must never panic the
        // constructor. The hook subscribes only to the can-block `Stop` event.
        let _ = chain.register(Arc::new(RalphStopHook {
            run_store: config.storage.run().clone(),
            project_id: config.project_id.clone(),
        }));
        config.hooks = Some(chain);
        Self {
            config,
            observed_writes: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// #126: record a harness-OBSERVED write/edit at the tool-dispatch seam. Only
    /// `write_file` / `edit_file` calls with a `path` are recorded; de-duplicated
    /// against what is already accumulated for the current step. Called from
    /// `run_react_inner` for the call ACTUALLY dispatched (`effective_call`), so
    /// the path comes from the real tool call — never a model-self-reported field.
    fn observe_write_call(&self, call: &ToolCall) {
        let is_write_edit = matches!(call.name.as_str(), "write_file" | "edit_file");
        if !is_write_edit {
            return;
        }
        if let Some(serde_json::Value::String(path)) = call.input.get("path") {
            if let Ok(mut acc) = self.observed_writes.lock() {
                if !acc.iter().any(|p| p == path) {
                    acc.push(path.clone());
                }
            }
        }
    }

    /// Build the per-turn [`ContextSources`] threaded into the structural
    /// `ContextManager::assemble` seam (issue #115 / SC-26). Supplies the tool
    /// schemas, guides (guide source + active skills), and merged memory (#160);
    /// the composed static prompt populates once the chunk-provider path is
    /// wired. An empty result renders to nothing, so a harness with no sources
    /// stays byte-identical.
    async fn build_context_sources(
        &self,
        tool_schemas: Vec<ToolSchema>,
        task_instruction: &str,
    ) -> ContextSources {
        // #115 / SC-26 / #9: configured guides reach the model structurally through
        // the assemble seam, not as an ad-hoc User-message prepend. Skills (the
        // manifest + active bodies, progressive disclosure) are appended as guides
        // from the shared catalog, so loading a skill via `load_skill` makes its
        // body sticky in the System block on the next turn.
        let mut guides = self.config.guides.clone();
        if let Some(skills) = self.config.skills.as_ref() {
            guides.extend(skills.active_guides());
        }
        // #160 / SC-26 follow-up: query the configured memory provider and inject
        // the relevant items structurally (rendered into the same System block as
        // guides). The query text defaults to the task instruction, so retrieved
        // memory tracks the current work; a configured `query` overrides it. A
        // query error is swallowed (empty memory) — memory is best-effort context,
        // never a halt. `None` leaves memory empty (byte-identical pre-#160 path).
        let memory = match self.config.memory.as_ref() {
            Some(mem) => {
                let query = crate::memory::MemoryQuery {
                    task_instruction: mem
                        .query
                        .clone()
                        .unwrap_or_else(|| task_instruction.to_string()),
                    domain: mem.domain.clone(),
                    session_id: None,
                    min_relevance: mem.min_relevance,
                    max_items: mem.max_items,
                };
                mem.provider.query(query).await.unwrap_or_default()
            }
            None => Vec::new(),
        };
        ContextSources {
            guides,
            tool_schemas,
            memory,
            // The composed static prompt is empty until the chunk-provider path
            // is wired.
            ..Default::default()
        }
    }

    /// Whether a sandbox violation should HALT the run under the configured
    /// [`SandboxViolationPolicy`]: only when the policy is `Halt` AND the
    /// violation is halt-eligible (`PathEscape` / `NetworkViolation`). Layer-2
    /// violations are never halt-eligible, so they stay recoverable regardless.
    fn sandbox_violation_halts(&self, violation: &SandboxViolation) -> bool {
        self.config.sandbox_violation_policy == SandboxViolationPolicy::Halt
            && violation.is_always_halt()
    }

    /// Flatten a [`ToolOutput::SandboxViolation`] into a plain
    /// [`ToolOutput::Error`] whose `recoverable` flag reflects the policy. Used
    /// on tool-result drain paths (resume / human-approval flushes) that append
    /// results to history but have no place to issue a typed halt; the main loop
    /// handles the variant directly so it can return the typed
    /// [`HaltReason::SandboxViolation`]. A non-violation output passes through.
    fn normalize_sandbox_violation(&self, output: ToolOutput) -> ToolOutput {
        match output {
            ToolOutput::SandboxViolation { violation } => {
                let recoverable = !self.sandbox_violation_halts(&violation);
                ToolOutput::Error {
                    message: format!("sandbox violation: {violation}"),
                    recoverable,
                }
            }
            other => other,
        }
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
        agent: Arc<dyn Agent>,
    ) {
        self.run_compaction(session_state, session_id, task_id, span_seq, usage, &agent)
            .await;
    }

    fn emit(stream: &Option<StreamSink>, event: StreamEvent) {
        if let Some(s) = stream.as_ref() {
            s(event);
        }
    }

    /// Map one raw [`crate::model::StreamEvent`] to zero or one harness
    /// [`StreamEvent`] (issue #103), threading [`TurnStreamState`] so blocks and
    /// tool calls are correlated across events.
    ///
    /// Rules enforced here:
    /// - Q2: a block's [`StreamEvent::BlockStart`] is emitted exactly once, the
    ///   first time a delta for that index is observed; `ContentBlockStop` maps
    ///   to [`StreamEvent::BlockStop`].
    /// - Q3: `MessageStart` / `MessageStop` map to `None` (dropped).
    /// - A tool-use block additionally emits [`StreamEvent::ToolCallStart`] on
    ///   open, then each fragment as [`StreamEvent::ToolArgsDelta`] keyed by the
    ///   derived `call_id`.
    ///
    /// Returns a `Vec` because opening a block emits both a `BlockStart` and (for
    /// tool-use blocks) a `ToolCallStart` plus the delta itself.
    fn map_model_stream_event(
        ev: crate::model::StreamEvent,
        state: &mut TurnStreamState,
    ) -> Vec<StreamEvent> {
        use crate::model::StreamEvent as M;
        use std::collections::hash_map::Entry;
        match ev {
            // Q3: dropped at the harness boundary.
            M::MessageStart => Vec::new(),
            M::MessageStop { .. } => Vec::new(),
            M::ContentBlockDelta { index, delta } => {
                let mut out = Vec::new();
                if let Entry::Vacant(e) = state.open_blocks.entry(index) {
                    e.insert(BlockKind::Text);
                    out.push(StreamEvent::BlockStart {
                        index,
                        block: BlockKind::Text,
                        node: None,
                    });
                }
                out.push(StreamEvent::TextDelta {
                    content: delta,
                    node: None,
                });
                out
            }
            M::ThinkingDelta { index, delta } => {
                let mut out = Vec::new();
                if let Entry::Vacant(e) = state.open_blocks.entry(index) {
                    e.insert(BlockKind::Reasoning);
                    out.push(StreamEvent::BlockStart {
                        index,
                        block: BlockKind::Reasoning,
                        node: None,
                    });
                }
                out.push(StreamEvent::ReasoningDelta {
                    content: delta,
                    node: None,
                });
                out
            }
            M::ToolUseStart { index, id, name } => {
                let mut out = Vec::new();
                if let Entry::Vacant(e) = state.open_blocks.entry(index) {
                    e.insert(BlockKind::ToolUse);
                    // Use the real call id from the model; consumers correlate
                    // subsequent ToolArgsDelta by it.
                    state.tool_calls.insert(index, id.clone());
                    out.push(StreamEvent::BlockStart {
                        index,
                        block: BlockKind::ToolUse,
                        node: None,
                    });
                    out.push(StreamEvent::ToolCallStart {
                        index,
                        call_id: id,
                        name,
                        node: None,
                    });
                }
                out
            }
            M::ToolUseDelta {
                index,
                partial_json,
            } => {
                let mut out = Vec::new();
                // Fallback: if a stream omitted ToolUseStart, open the block here
                // with a synthesized id and empty name so args still surface.
                if let Entry::Vacant(e) = state.open_blocks.entry(index) {
                    e.insert(BlockKind::ToolUse);
                    let call_id = TurnStreamState::call_id_for(index);
                    state.tool_calls.insert(index, call_id.clone());
                    out.push(StreamEvent::BlockStart {
                        index,
                        block: BlockKind::ToolUse,
                        node: None,
                    });
                    out.push(StreamEvent::ToolCallStart {
                        index,
                        call_id,
                        name: String::new(),
                        node: None,
                    });
                }
                let call_id = state
                    .tool_calls
                    .get(&index)
                    .cloned()
                    .unwrap_or_else(|| TurnStreamState::call_id_for(index));
                out.push(StreamEvent::ToolArgsDelta {
                    call_id,
                    partial_json,
                    node: None,
                });
                out
            }
            M::ContentBlockStop { index } => {
                state.open_blocks.remove(&index);
                vec![StreamEvent::BlockStop { index, node: None }]
            }
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
    /// Resolve the worker agent for a [`LoopStrategy`] tree from the
    /// [`ExecutionRegistry`] (#124). The worker is the agent on the LEAF reached
    /// by descending the recursion: a `ReAct` leaf's `agent`; a combinator
    /// descends into its primary worker child (`inner` / `execute`). A `Ralph`
    /// with a non-empty `agent` override resolves THAT (Q3). Returns the resolved
    /// agent, or a typed `UnresolvedHandle` failure carried as `Err`.
    fn resolve_worker_agent(&self, ls: &LoopStrategy) -> Result<Arc<dyn Agent>, HarnessError> {
        let key = self.worker_agent_key(ls);
        self.config
            .registry
            .resolve_agent(&AgentRef(key.clone()))
            .cloned()
            .ok_or(HarnessError::UnresolvedHandle {
                kind: "agent".to_string(),
                key,
            })
    }

    /// The agent key the worker leaf of `ls` references (see
    /// [`resolve_worker_agent`](Self::resolve_worker_agent)).
    fn worker_agent_key(&self, ls: &LoopStrategy) -> String {
        match ls {
            LoopStrategy::ReAct(c) => c.agent.0.clone(),
            LoopStrategy::PlanExecute(c) => self.worker_agent_key(&c.execute),
            LoopStrategy::SelfVerifying(c) => self.worker_agent_key(&c.inner),
            LoopStrategy::Ralph(c) => {
                if c.agent.0.is_empty() {
                    self.worker_agent_key(&c.inner)
                } else {
                    c.agent.0.clone()
                }
            }
            LoopStrategy::HillClimbing(c) => self.worker_agent_key(&c.inner),
        }
    }

}

impl Harness for StandardHarness {
    fn run<'a>(&'a self, options: HarnessRunOptions) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            let result = self.run_inner(options).await;
            self.auto_persist_terminal(&result).await;
            result
        })
    }

    fn resume<'a>(
        &'a self,
        state: PausedState,
        response: HumanResponse,
        on_stream: Option<StreamSink>,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            let result = self.resume_inner(state, response, on_stream).await;
            self.auto_persist_terminal(&result).await;
            result
        })
    }

    fn resume_consult<'a>(
        &'a self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: Option<StreamSink>,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            let result = self.resume_consult_inner(state, response, on_stream).await;
            self.auto_persist_terminal(&result).await;
            result
        })
    }
}

/// Test-only seams (#124) that drive the GENUINE recursive plan / execute
/// phases of [`PlanExecuteConfig::run`] in isolation, so the granular plan/
/// execute regression tests exercise the real per-task `self.plan.run(cx)` /
/// `self.execute.run(cx)` dispatch — not a stale parallel copy. The phase logic
/// lives ONLY in [`PlanExecuteConfig::run`]; these helpers reproduce the minimal
/// driver around a single phase using the same scratch wiring + leaf primitives.
#[cfg(test)]
impl StandardHarness {
    /// Drive the recursive plan phase for `task` (whose strategy must be a
    /// `PlanExecute`): seed the directive, dispatch `plan.run(cx)` under the
    /// plan sub-strategy's OWN declared budget (the global `max_turns` is only
    /// the outer backstop), then capture + persist the artifact. Mirrors the
    /// plan half of [`PlanExecuteConfig::run`] — it must NOT clamp the planner to
    /// a single turn or it becomes a stale parallel copy of production.
    async fn run_plan_phase(
        &self,
        task: &Task,
        session_state: &mut SessionState,
        budget_used: BudgetSnapshot,
        on_stream: &Option<StreamSink>,
    ) -> Result<PlanPhaseOutcome, RunResult> {
        let LoopStrategy::PlanExecute(cfg) = &task.loop_strategy else {
            panic!("run_plan_phase test helper requires a PlanExecute strategy");
        };
        let plan_strategy = (*cfg.plan).clone();
        let session_id = task.session_id.clone();

        let directive = Self::plan_directive(&task.instruction);
        let mut plan_session = session_state.clone();
        self.config
            .context_manager
            .append_user_message(&mut plan_session, &directive)
            .await;
        let _ = on_stream;
        // The plan phase runs under the plan sub-strategy's OWN declared budget;
        // the global `max_turns` is only the outer backstop. (Previously clamped
        // to `turns + 1` — a single turn — which starved multi-turn `task_list`
        // authoring and diverged this test seam from production.)
        let plan_task = Task {
            id: task.id.clone(),
            instruction: directive,
            session_id: session_id.clone(),
            budget: task.budget.clone(),
            loop_strategy: plan_strategy.clone(),
        };

        let plan_result = self
            .run_plan_subtree(
                &plan_strategy,
                plan_task,
                plan_session,
                budget_used.clone(),
                None,
                None,
            )
            .await;
        let (plan_output, usage, turns) = match plan_result {
            Some(RunResult::Success {
                output,
                usage,
                turns,
                ..
            }) => (output, usage, turns),
            Some(other) => return Err(other),
            None => {
                return Err(RunResult::Failure {
                    reason: HaltReason::PlanPhaseFailed {
                        error: crate::plan::PlanPhaseError::PlanningTurnFailed {
                            message: "plan sub-strategy produced no terminal".into(),
                        },
                    },
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: budget_used.turns,
                    session_state: SessionState::default(),
                })
            }
        };
        self.capture_and_persist_plan(&session_id, &plan_output, usage, turns)
            .await
    }

    /// Drive the recursive execute phase for `task` (whose strategy must be a
    /// `PlanExecute`), draining `task_list` by dispatching `execute.run(cx)` per
    /// task. Mirrors the execute half of [`PlanExecuteConfig::run`].
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
        let LoopStrategy::PlanExecute(cfg) = &task.loop_strategy else {
            panic!("run_execute_phase test helper requires a PlanExecute strategy");
        };
        let execute_strategy = (*cfg.execute).clone();
        let session_id = task.session_id.clone();

        self.reconcile_deep_resume(&session_id, &mut task_list)
            .await;

        let total_tasks = task_list.tasks.len();
        let mut total_usage = plan_usage;
        let mut last_output = String::new();
        let mut last_state = SessionState::default();

        for index in 0..total_tasks {
            let task_id = task_list.tasks[index].id;
            let instruction = task_list.tasks[index].description.clone();

            if task_list.tasks[index].status == TaskStatus::Completed {
                last_output = instruction.clone();
                continue;
            }

            // #125: the per-task `remaining_turns / remaining_tasks / step_cap`
            // derivation is REMOVED (dead) — enforcement is now `charge`-based on
            // the PlanExecute scope. This helper mirrors the live body's structure.
            let _ = task_list.update(task_id, Some(TaskStatus::InProgress), None);
            self.persist_task_list(&session_id, &task_list).await;

            let mut step_task = Task {
                id: task.id.clone(),
                instruction,
                session_id: session_id.clone(),
                budget: task.budget.clone(),
                loop_strategy: execute_strategy.clone(),
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

            self.config
                .context_manager
                .append_user_message(session_state, &step_task.instruction)
                .await;

            let mut cx = ExecutionContext::new(&self.config.registry);
            cx.executor = Some(self);
            cx.scratch.run_session = session_state.clone();
            cx.scratch.run_budget = carried.clone();
            cx.scratch.task = Some(step_task);
            let _ = execute_strategy.run(&mut cx).await;
            let sub_result = cx.take_child_override();

            match sub_result {
                Some(RunResult::Success {
                    output,
                    usage,
                    turns,
                    session_state: step_state,
                    ..
                }) => {
                    carried.turns = turns;
                    *session_state = step_state;
                    last_state = session_state.clone();
                    carried.input_tokens += usage.input_tokens;
                    carried.output_tokens += usage.output_tokens;
                    total_usage.input_tokens += usage.input_tokens;
                    total_usage.output_tokens += usage.output_tokens;
                    total_usage.cache_read_tokens += usage.cache_read_tokens;
                    total_usage.cache_write_tokens += usage.cache_write_tokens;
                    total_usage.cost_usd += usage.cost_usd;
                    last_output = output;
                    let _ = task_list.complete(task_id);
                    self.persist_task_list(&session_id, &task_list).await;
                    Self::emit(
                        on_stream,
                        StreamEvent::FinalResponse {
                            content: last_output.clone(),
                            node: None,
                        },
                    );
                }
                Some(RunResult::Failure {
                    reason,
                    usage,
                    turns,
                    ..
                }) => {
                    total_usage.input_tokens += usage.input_tokens;
                    total_usage.output_tokens += usage.output_tokens;
                    total_usage.cache_read_tokens += usage.cache_read_tokens;
                    total_usage.cache_write_tokens += usage.cache_write_tokens;
                    total_usage.cost_usd += usage.cost_usd;
                    let _ = task_list.update(task_id, Some(TaskStatus::Blocked), None);
                    self.persist_task_list(&session_id, &task_list).await;
                    let terminal_reason = match reason {
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
                        session_state: last_state,
                    };
                }
                Some(pause) => return pause,
                None => {
                    return RunResult::Failure {
                        reason: HaltReason::StepFailed {
                            task_index: index,
                            task: task_list.tasks[index].description.clone(),
                            reason: "execute sub-strategy produced no terminal".into(),
                        },
                        session_id,
                        usage: total_usage,
                        turns: carried.turns,
                        session_state: last_state,
                    }
                }
            }
        }

        RunResult::Success {
            output: last_output,
            session_id,
            usage: total_usage,
            turns: carried.turns,
            session_state: last_state,
        }
    }
}

// ---- Submodules (extracted from the former single-file harness.rs) ----
#[cfg(test)]
mod budget_enforcement_tests;
#[cfg(test)]
mod budget_policy_tests;
#[cfg(test)]
mod execution_scaffold_tests;
#[cfg(test)]
mod strategy_tests;
#[cfg(any(test, feature = "test-utils"))]
pub mod testing;
#[cfg(test)]
mod tests;
