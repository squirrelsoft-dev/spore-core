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

    #[allow(clippy::too_many_arguments)]
    async fn run_react(
        &self,
        task: Task,
        max_iterations: u32,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        agent: Arc<dyn Agent>,
        // Issue 2: the RESOLVED per-node toolset handle. Most `run_react`
        // wrappers (PlanExecute/DAG execute steps, single-shot run) pass the
        // empty handle (`ToolsetRef("")`) ⇒ global-catalogue fallback, byte-for-
        // byte with pre-Issue-2 behaviour. The SelfVerifying evaluate phase
        // passes its `eval_toolset` so the reviewer is scoped to read-only tools.
        toolset: ToolsetRef,
    ) -> RunResult {
        let result = self
            .run_react_inner(
                task,
                max_iterations,
                session_state,
                budget_used,
                on_stream,
                agent,
                toolset,
                // Issue #139: the legacy `run_react` wrapper does NOT enforce
                // output schemas (the recursive `ReactConfig::run` is the single
                // enforcement seam). `None`/`0` ⇒ byte-for-byte pre-#139.
                None,
                0,
                // SC-10: the legacy `run_react` wrapper carries no per-leaf prompt
                // override (only the recursive `ReactConfig::run` does) ⇒ the
                // window uses the global `config.system_prompt`, byte-identical.
                None,
                // Legacy wrapper is not a PlanExecute execute step.
                false,
            )
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
            // Consult (issue #114) is NOT terminal — like WaitingForHuman, the
            // worker is paused awaiting a resume. Do not finalize observability.
            RunResult::Consult { .. } => {}
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

    /// The harness-loop tool registry to use for a run keyed by `session_id`.
    ///
    /// When catalogue tools were added via [`HarnessBuilder::tool`] / `tools`,
    /// this bridges the folded [`StandardToolRegistry`] through
    /// [`RealToolRegistry`](crate::tool_registry::RealToolRegistry) — built
    /// fresh per run so the run's `SessionId` + storage thread into every tool
    /// dispatch. Otherwise it returns the injected
    /// [`tool_registry`](HarnessConfig::tool_registry) seam unchanged.
    /// Issue 2 (per-node toolset scoping): `toolset` is the resolving leaf's
    /// `ToolsetRef` handle. When it is NON-EMPTY and a per-key catalogue was
    /// registered via [`HarnessBuilder::toolset_tools`], THAT catalogue is
    /// bridged (so the node dispatches ONLY its own tools — strict scoping).
    /// Otherwise (empty handle, or non-empty handle with no per-key catalogue)
    /// the existing global-catalogue / `tool_registry` seam fallback applies, so
    /// examples 01–11 that use `.tools()` keep working byte-for-byte.
    fn effective_tool_registry(
        &self,
        session_id: &SessionId,
        toolset: &ToolsetRef,
    ) -> Arc<dyn ToolRegistry> {
        // Reserved handle: advertise NO tools (plan-empty re-ask). Intercepted
        // before the catalogue lookup so it never falls through to the full
        // registry. Dispatch is unreachable — the model is offered nothing.
        if toolset.0 == NO_TOOLS_TOOLSET {
            return Arc::new(EmptyToolRegistry);
        }
        // Strict per-node scoping: a non-empty handle with its own catalogue.
        if !toolset.0.is_empty() {
            if let Some(catalogue) = self.config.toolset_catalogues.get(&toolset.0) {
                return self.maybe_tool_search(Arc::new(
                    crate::tool_registry::RealToolRegistry::new(
                        catalogue.clone(),
                        self.config.sandbox.clone(),
                        session_id.clone(),
                        self.config.project_id.clone(),
                        self.config.storage.run().clone(),
                        self.config.storage.memory().clone(),
                    ),
                ));
            }
        }
        // Fallback: empty handle (back-compat) or unregistered non-empty handle.
        match &self.config.catalogue_registry {
            Some(catalogue) => {
                self.maybe_tool_search(Arc::new(crate::tool_registry::RealToolRegistry::new(
                    catalogue.clone(),
                    self.config.sandbox.clone(),
                    session_id.clone(),
                    self.config.project_id.clone(),
                    self.config.storage.run().clone(),
                    self.config.storage.memory().clone(),
                )))
            }
            None => self.config.tool_registry.clone(),
        }
    }

    /// Wrap a resolved catalogue registry in a
    /// [`ToolSearchRegistry`](crate::tool_registry::ToolSearchRegistry) when
    /// [`HarnessConfig::tool_search`] is configured; otherwise return it unchanged.
    /// Built per turn-loop window (alongside the registry it wraps), so a searched
    /// tool stays loaded for the rest of the window and resets on the next.
    fn maybe_tool_search(&self, reg: Arc<dyn ToolRegistry>) -> Arc<dyn ToolRegistry> {
        match &self.config.tool_search {
            Some(cfg) => Arc::new(crate::tool_registry::ToolSearchRegistry::new(
                reg,
                cfg.clone(),
            )),
            None => reg,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    async fn run_react_inner(
        &self,
        task: Task,
        max_iterations: u32,
        mut session_state: SessionState,
        mut budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        // #124: the worker agent is RESOLVED by the caller (the recursing
        // `ReactConfig::run` resolves `self.agent` from the registry; Ralph may
        // override it per window). The leaf no longer reads `config.agent`.
        agent: Arc<dyn Agent>,
        // Issue 2: the leaf's RESOLVED `toolset` handle (mirrors `agent`). Empty
        // (`""`) ⇒ global-catalogue fallback; a non-empty handle with its own
        // per-key catalogue ⇒ strict per-node scoping.
        toolset: ToolsetRef,
        // Issue #139: the leaf's RESOLVED output schema (`None` ⇒ no enforcement,
        // identical to pre-#139 behavior). When `Some`, the terminal is validated
        // against it (frozen validator subset) and a validation failure feeds the
        // error back + retries up to `output_schema_max_retries` extra turns
        // WITHIN budget; on exhaustion WITH budget remaining the window returns
        // `HaltReason::OutputSchemaViolation`. The schema is also set on every
        // turn's `ModelParams.output_schema` so the Ollama `format` channel
        // constrains decoding (Anthropic/OpenAI ignore it).
        output_schema: Option<serde_json::Value>,
        output_schema_max_retries: u32,
        // SC-10: the leaf's per-node system-prompt OVERRIDE. `None` ⇒ the global
        // `config.system_prompt` is used (byte-identical to pre-SC-10). `Some` ⇒
        // it REPLACES the global prompt for every turn of this window, so the leaf
        // sees ONLY its own prompt (the per-leaf prompt half of SC-10; the toolset
        // half is the `toolset` arg above).
        system_prompt: Option<String>,
        // Whether this window is a PlanExecute EXECUTE step — scopes the behavioral
        // tool-call nudge so the plan-JSON / synthesis turns are not nudged.
        execute_step: bool,
    ) -> RunResult {
        let session_id = task.session_id.clone();
        // Resolve the effective tool registry once per turn-loop window (all
        // strategies funnel through here). Bridges the per-node toolset catalogue
        // (or the global catalogue when the handle is empty) per-run.
        let tool_registry = self.effective_tool_registry(&session_id, &toolset);
        // Reset the adaptive prompt-based-tool-calling escalation flag at the
        // start of this turn-loop window so detection is scoped to the window
        // and does not leak across `run()` calls (the flag is shared with the
        // model wrapper for the harness's lifetime). No-op unless a
        // `conversational` harness installed the adaptive wrapper.
        if let Some(flag) = self.config.prompt_tool_call_flag.as_ref() {
            flag.store(false, std::sync::atomic::Ordering::Release);
        }
        let started_at = Instant::now();
        let mut usage = AggregateUsage::default();
        // Per-run Stop-hook block counter (issue #69, Decision 4). Resets on
        // every run() — resume starts fresh. After `max_stop_blocks` consecutive
        // blocks the loop terminates anyway (R14).
        let mut stop_blocks: u32 = 0;
        // Consecutive-recoverable-tool-error breaker state (issue #137), keyed
        // by tool name. Loop-local to this window: a fresh `run()`/re-driven
        // `Continue` window starts with a clean N/2N allowance (AC3 Continue
        // reset). `N` is `error_loop_threshold`; the hard stop is at `2 * N`.
        let error_loop_n = self.config.error_loop_threshold;
        let mut error_runs: std::collections::HashMap<String, ErrorRun> =
            std::collections::HashMap::new();
        // Repeated-identical-SUCCESSFUL-call breaker state (the no-progress twin
        // of the error breaker above), loop-local to this window like `error_runs`.
        // Tracks the LAST successful (tool, args) signature and how many times in a
        // row it has recurred; a different successful call resets the run. Shares
        // `error_loop_n`'s `N`/`2N` thresholds. The error breaker owns ERRORS; this
        // owns SUCCESSES, so the two partition cleanly and never double-count.
        let mut last_success_sig: Option<(String, serde_json::Value)> = None;
        let mut repeat_count: u32 = 0;
        let mut repeat_injected = false;
        // Output-schema enforcement state (issue #139), loop-local to this
        // window. `output_schema_retries_used` counts the EXTRA retry turns spent
        // on validation feedback; the budget for them is `output_schema_max_retries`
        // (the `N`). `last_schema_error` holds the most recent frozen validator
        // error so a final exhaustion can report it. Both are inert when
        // `output_schema` is `None` (enforcement OFF / no schema).
        let mut output_schema_retries_used: u32 = 0;
        // Set on every validation failure before it is read in the violation
        // terminal; the initial empty value is never observed.
        #[allow(unused_assignments)]
        let mut last_schema_error: String = String::new();
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
        // Per-window cap for the behavioral tool-call nudge (see
        // `HarnessConfig::tool_call_nudge`) so a model that keeps answering without
        // acting can't loop forever, plus a count of tools actually called this
        // window — the signal is "execute turn finished having called ZERO tools".
        const MAX_TOOL_CALL_NUDGES: u32 = 2;
        let mut tool_call_nudges_used: u32 = 0;
        let mut tools_called_this_window: u32 = 0;

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
                    session_state: session_state.clone(),
                };
            }
            if let Some(limit_type) = Self::budget_exceeded(&task.budget, &budget_used, started_at)
            {
                return RunResult::Failure {
                    reason: HaltReason::BudgetExceeded { limit_type },
                    session_id,
                    usage,
                    turns: budget_used.turns,
                    session_state: session_state.clone(),
                };
            }

            // Middleware: BeforeTurn (rich chain, issue #11). The chain may mutate
            // `session_state` in place (priority-ordered fan-out); a
            // `ContinueWithModification` is the modified-but-proceed signal.
            if let Some(mw) = self.config.middleware.as_ref() {
                match mw
                    .fire_before_turn(&mut session_state, budget_used.turns)
                    .await
                {
                    MiddlewareDecision::Continue | MiddlewareDecision::ContinueWithModification => {
                    }
                    MiddlewareDecision::Halt { reason } => {
                        return RunResult::Failure {
                            reason: HaltReason::MiddlewareHalt {
                                hook: HookPoint::BeforeTurn,
                                reason,
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                            session_state: session_state.clone(),
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
                            // #140: carry this leaf's toolset handle so resume
                            // routes through its scoped catalogue.
                            toolset: toolset.clone(),
                        };
                        return RunResult::WaitingForHuman {
                            state: Box::new(state),
                            request,
                        };
                    }
                    // `ForceAnotherTurn` is valid only at `BeforeCompletion`; the
                    // StandardMiddlewareChain converts it to `Halt` here. Handle it
                    // defensively as a halt for any custom chain that emits it.
                    MiddlewareDecision::ForceAnotherTurn { inject } => {
                        return RunResult::Failure {
                            reason: HaltReason::MiddlewareHalt {
                                hook: HookPoint::BeforeTurn,
                                reason: format!(
                                    "ForceAnotherTurn is not valid at BeforeTurn: {inject}"
                                ),
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                            session_state: session_state.clone(),
                        };
                    }
                }
            }

            // Assemble + invoke agent for one turn. `sources` (issue #115 / SC-26)
            // carries the rich ContextSources into the structural assemble seam.
            // Slice 1 threads the seam with tool schemas only; guides/memory/skills
            // populate in later slices. An empty `sources` renders to nothing, so
            // the no-source path stays byte-identical.
            let sources = self
                .build_context_sources(tool_registry.schemas(), &task.instruction)
                .await;
            let mut context = self
                .config
                .context_manager
                .assemble(&session_state, &task, &sources)
                .await;
            // Deliver the registry's tool schemas to the model. Tool schemas are
            // owned by the `ToolRegistry`; the harness wires them into the
            // assembled context so the model knows what it can call. Only fill
            // when the context manager left `tools` empty (the standard
            // compaction adapter does), so a context manager that deliberately
            // sets a phase-specific tool subset is preserved.
            if context.tools.is_empty() {
                context.tools = tool_registry.schemas();
            }
            // Prepend the configured operating system prompt (issue #91), MERGING
            // it with any structural #115 context block the manager already placed
            // as a leading System message (guides/memory/composed prompt). The
            // system prompt always leads. When the manager produced NO System
            // message (the common case — empty sources, or a test double), this
            // inserts a fresh System message exactly as before, so the
            // no-structural-block path stays byte-identical. The `starts_with`
            // guard keeps a resumed/seeded session that already leads with the
            // system prompt from being given it twice.
            //
            // SC-10: a leaf's per-node `system_prompt` override WINS over the
            // global `config.system_prompt`. When this window's leaf carries one,
            // it is used EXCLUSIVELY (the global prompt does not leak in), so the
            // plan and execute phases of a `PlanExecuteConfig` can run under
            // distinct prompts. With no leaf override (`None`) the global prompt
            // applies exactly as before — byte-identical.
            let effective_system_prompt = system_prompt
                .as_ref()
                .or(self.config.system_prompt.as_ref());
            if let Some(system_prompt) = effective_system_prompt {
                match context.messages.first_mut() {
                    Some(Message {
                        role: Role::System,
                        content: Content::Text { text },
                    }) if !text.starts_with(system_prompt.as_str()) => {
                        *text = format!("{system_prompt}\n\n{text}");
                    }
                    Some(Message {
                        role: Role::System, ..
                    }) => {
                        // Already leads with the system prompt (or a non-text
                        // System block) — leave it.
                    }
                    _ => {
                        context.messages.insert(
                            0,
                            Message {
                                role: Role::System,
                                content: Content::Text {
                                    text: system_prompt.clone(),
                                },
                            },
                        );
                    }
                }
            }
            // Per-run model params win unconditionally (issue #93). The agent
            // copies `Context.params` verbatim into the `ModelRequest`, so this
            // is the single seam that delivers configured params (e.g.
            // structured tool calls) to every tool-requesting ReAct/execute/
            // streaming turn.
            context.params = self.config.model_params.clone();
            // Issue #139: route the enforced output schema into this turn's
            // constrained-decoding channel (`ModelParams.output_schema`). Ollama
            // honors it via `format`; Anthropic/OpenAI ignore it (no-op, like
            // `structured_tool_calls`). `None` leaves the params byte-identical.
            if output_schema.is_some() {
                context.params.output_schema = output_schema.clone();
            }
            // Whether tools were advertised to the model this turn — a
            // precondition for classifying a prose final response as a missed
            // tool call (adaptive prompt-based escalation). Captured before
            // `context` is moved into the turn.
            let tools_advertised = !context.tools.is_empty();
            Self::emit(
                &on_stream,
                StreamEvent::TurnStart {
                    turn: budget_used.turns + 1,
                    node: None,
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
            // Delta-level streaming (issue #103): when a user-facing stream sink
            // is attached, drive the turn through `turn_streaming` and forward each
            // raw `model::StreamEvent` mapped to harness `StreamEvent`s.
            //
            // #stream: the sink is now an `Arc` (cloneable), so the adapter forwards
            // each mapped event to it LIVE — the instant the model emits it — rather
            // than buffering and flushing post-turn (the old `Box`-can't-be-shared
            // workaround). This is what makes token-by-token SSE / mid-turn cancel
            // observable. `TurnStreamState` (block + call-id correlation) lives
            // behind a Mutex shared with the adapter; we map UNDER the lock, RELEASE
            // it, then forward — never running the user sink under the lock (a
            // re-entrant sink would deadlock). Order is preserved: TurnStart (emitted
            // below, before the turn) → live deltas → TurnEnd → coarse events.
            // When no sink is attached we keep the plain `turn` path so the baseline
            // RunResult is byte-identical (back-compat).
            let result = if let Some(sink) = on_stream.clone() {
                let state = std::sync::Arc::new(std::sync::Mutex::new(TurnStreamState::default()));
                let adapter: crate::agent::AgentStreamSink = Box::new(move |ev| {
                    let mapped = {
                        let mut guard = state.lock().expect("stream state poisoned");
                        Self::map_model_stream_event(ev, &mut guard)
                    };
                    for m in mapped {
                        sink(m);
                    }
                });
                agent.turn_streaming(context, adapter).await
            } else {
                agent.turn(context).await
            };
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
                tools_called_this_window += tool_calls_requested;
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
                    node: None,
                },
            );

            match result {
                TurnResult::FinalResponse {
                    content, usage: u, ..
                } => {
                    usage.add_turn(&u);
                    budget_used.input_tokens += u.input_tokens as u64;
                    budget_used.output_tokens += u.output_tokens as u64;

                    // Adaptive prompt-based tool-calling escalation. Two signals,
                    // both meaning "tools were available but the model produced NO
                    // native tool call":
                    //   (a) PHRASE — the prose carries action-intent language ("I'll
                    //       read …"), so it *meant* to act. Always active; cheap and
                    //       conservative (biased to false negatives). This is the
                    //       original trigger and is unchanged for every consumer.
                    //   (b) STRUCTURAL — an EXECUTE step reached a terminal answer
                    //       having called ZERO tools while tools were advertised and
                    //       no output schema constrained it. This is a behavioral
                    //       signal (no phrase-matching) that catches the failure modes
                    //       the phrase list misses: markdown command fences
                    //       (```bash\ncat …```), bare narration, and empty content —
                    //       exactly how Qwen drops out of native tool-call mode at the
                    //       cold start of an isolated execute sub-session. Gated on
                    //       `tool_call_nudge` so it ONLY affects consumers that opted
                    //       into behavioral repair (local models); native-tool models
                    //       (Claude, …) keep phrase-only behavior and never flip the
                    //       flag on a legitimately tool-free turn.
                    // On either signal: set the session flag so the wrapped model
                    // switches to prompt-based tool calling (the next request describes
                    // the tools in the system prompt and clears the native channel),
                    // nudge the model, and force another turn instead of completing.
                    // Guarded on the flag being unset so it fires at most once per
                    // window (bounded — one extra turn) and only on the
                    // `conversational` adaptive path.
                    if let Some(flag) = self.config.prompt_tool_call_flag.as_ref() {
                        let already_escalated = flag.load(std::sync::atomic::Ordering::Acquire);
                        let phrase = crate::tool_call_repair::detect_prose_response(
                            &content,
                            tools_advertised,
                        )
                        .is_some();
                        let structural = self.config.tool_call_nudge
                            && execute_step
                            && tools_advertised
                            && tools_called_this_window == 0
                            && output_schema.is_none();
                        if !already_escalated && (phrase || structural) {
                            flag.store(true, std::sync::atomic::Ordering::Release);
                            // Record the model's last (failed) output, then a
                            // corrective nudge, so the next turn has coherent context.
                            let assistant = Message {
                                role: Role::Assistant,
                                content: Content::Text {
                                    text: content.clone(),
                                },
                            };
                            self.config
                                .context_manager
                                .append_assistant_message(&mut session_state, &assistant)
                                .await;
                            let nudge = "No tool call was detected in your last \
response. The tools available to you and the exact tool-call format are described in \
the system prompt — use that format now to actually invoke a tool, not prose or a code \
block."
                                .to_string();
                            self.config
                                .context_manager
                                .append_user_message(&mut session_state, &nudge)
                                .await;
                            continue;
                        }
                    }

                    // Behavioral tool-call nudge (opt-in via `tool_call_nudge`). This
                    // does NOT inspect the response text for phrases — it keys on
                    // ACTION: an EXECUTE-phase task window finished (a terminal
                    // answer) having called ZERO tools while tools were advertised,
                    // i.e. the model "reported findings" without ever reading
                    // anything. Record that answer, nudge it to actually use a tool
                    // before reporting, and force another turn. Scoped to execute
                    // steps so the legitimately tool-free plan-JSON turn is untouched,
                    // skips schema-constrained turns, and bounded per window so a
                    // model that refuses to act can't loop. Off (default) →
                    // byte-identical.
                    if self.config.tool_call_nudge
                        && execute_step
                        && tools_advertised
                        && tools_called_this_window == 0
                        && output_schema.is_none()
                        && tool_call_nudges_used < MAX_TOOL_CALL_NUDGES
                    {
                        tool_call_nudges_used += 1;
                        let assistant = Message {
                            role: Role::Assistant,
                            content: Content::Text {
                                text: content.clone(),
                            },
                        };
                        self.config
                            .context_manager
                            .append_assistant_message(&mut session_state, &assistant)
                            .await;
                        let nudge = "You have not called any tool yet, so you have not \
actually inspected anything — your answer would be guessed, not grounded. Use the \
available tools (list_dir, read_file, grep, …) to gather the real data NOW, then \
report only what the tools returned."
                            .to_string();
                        self.config
                            .context_manager
                            .append_user_message(&mut session_state, &nudge)
                            .await;
                        continue;
                    }

                    if let Some(mw) = self.config.middleware.as_ref() {
                        match mw
                            .fire_before_completion(&content, budget_used.turns, &session_state)
                            .await
                        {
                            MiddlewareDecision::Continue
                            | MiddlewareDecision::ContinueWithModification => {}
                            // The chain concatenates every middleware's injection
                            // into one `ForceAnotherTurn` (issue #11). Record the
                            // model's final text, then the injection as a user
                            // message, and force another turn instead of completing
                            // — the same channel the Stop-block breaker uses, so the
                            // conversation stays well-formed (assistant final text →
                            // user injection).
                            MiddlewareDecision::ForceAnotherTurn { inject } => {
                                let assistant = Message {
                                    role: Role::Assistant,
                                    content: Content::Text {
                                        text: content.clone(),
                                    },
                                };
                                self.config
                                    .context_manager
                                    .append_assistant_message(&mut session_state, &assistant)
                                    .await;
                                self.config
                                    .context_manager
                                    .append_user_message(&mut session_state, &inject)
                                    .await;
                                continue;
                            }
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeCompletion,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
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
                                    // #140: carry this leaf's toolset handle so
                                    // resume routes through its scoped catalogue.
                                    toolset: toolset.clone(),
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
                                session_state: session_state.clone(),
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

                            // Output-schema enforcement gate (issue #139).
                            // Additive to and EARLIER than the Success terminal:
                            // when a schema is active, validate this terminal
                            // `FinalResponse`. On a validation FAILURE, feed the
                            // frozen error back as a USER message and retry one
                            // more turn (up to `output_schema_max_retries` extra
                            // turns). The retried turn re-enters the loop top,
                            // where the budget gate fires FIRST — so a retry that
                            // would exceed the turn/budget backstop surfaces the
                            // existing `BudgetExceeded` terminal, NOT
                            // `OutputSchemaViolation` (budget-cap-wins, Q2). Only
                            // when the N retries are exhausted WITH budget
                            // remaining does the window terminate with
                            // `OutputSchemaViolation`. `None` schema ⇒ no
                            // validation (migration gate OFF / no `output`), so
                            // the terminal is byte-identical to pre-#139.
                            if let Some(schema) = output_schema.as_ref() {
                                if let Err(verr) =
                                    crate::output_schema::validate_output(&content, schema)
                                {
                                    last_schema_error = verr.clone();
                                    if output_schema_retries_used < output_schema_max_retries {
                                        // Grant one more turn: feed the validator
                                        // error back as a USER message (the
                                        // assistant's invalid text is already in
                                        // history above) and loop. The budget gate
                                        // at the loop top enforces Q2 precedence.
                                        output_schema_retries_used += 1;
                                        Self::emit(
                                            &on_stream,
                                            StreamEvent::OutputSchemaRetry {
                                                attempt: output_schema_retries_used,
                                                error: verr.clone(),
                                                node: None,
                                            },
                                        );
                                        if let Some(obs) = self.config.observability.as_ref() {
                                            let base = SpanBase::new_root(
                                                SpanId::new(format!(
                                                    "{}-output-schema-retry-{}",
                                                    session_id.as_str(),
                                                    span_seq
                                                )),
                                                session_id.clone(),
                                                task.id.clone(),
                                                SpanKind::ContextAssembly,
                                                Timestamp::now(),
                                            );
                                            obs.emit_context(ContextSpan {
                                                base,
                                                operation: ContextOperation::OutputSchemaRetry {
                                                    attempt: output_schema_retries_used,
                                                    error: verr.clone(),
                                                },
                                                tokens_before: 0,
                                                tokens_after: 0,
                                                utilization_before: 0.0,
                                                utilization_after: 0.0,
                                            });
                                        }
                                        let feedback =
                                            crate::output_schema::feedback_message(&verr);
                                        self.config
                                            .context_manager
                                            .append_user_message(&mut session_state, &feedback)
                                            .await;
                                        continue;
                                    }
                                    // Retries exhausted WITH budget remaining
                                    // (the budget gate did not fire) → the typed
                                    // schema-violation terminal (Q2/AC3). Total
                                    // attempts = 1 + max_retries.
                                    let attempts = output_schema_max_retries + 1;
                                    Self::emit(
                                        &on_stream,
                                        StreamEvent::OutputSchemaViolation {
                                            attempts,
                                            error: last_schema_error.clone(),
                                            node: None,
                                        },
                                    );
                                    if let Some(obs) = self.config.observability.as_ref() {
                                        let base = SpanBase::new_root(
                                            SpanId::new(format!(
                                                "{}-output-schema-violation-{}",
                                                session_id.as_str(),
                                                span_seq
                                            )),
                                            session_id.clone(),
                                            task.id.clone(),
                                            SpanKind::ContextAssembly,
                                            Timestamp::now(),
                                        );
                                        obs.emit_context(ContextSpan {
                                            base,
                                            operation: ContextOperation::OutputSchemaViolation {
                                                attempts,
                                                error: last_schema_error.clone(),
                                            },
                                            tokens_before: 0,
                                            tokens_after: 0,
                                            utilization_before: 0.0,
                                            utilization_after: 0.0,
                                        });
                                    }
                                    return RunResult::Failure {
                                        reason: HaltReason::OutputSchemaViolation {
                                            schema: crate::model::canonicalize_json(schema),
                                            attempts,
                                            last_error: last_schema_error.clone(),
                                        },
                                        session_id,
                                        usage,
                                        turns: budget_used.turns,
                                        session_state: session_state.clone(),
                                    };
                                }
                            }

                            Self::emit(
                                &on_stream,
                                StreamEvent::FinalResponse {
                                    content: content.clone(),
                                    node: None,
                                },
                            );
                            return RunResult::Success {
                                output: content,
                                session_id,
                                usage,
                                turns: budget_used.turns,
                                session_state: session_state.clone(),
                            };
                        }
                    }
                }

                TurnResult::ToolCallRequested {
                    mut calls,
                    usage: u,
                    ..
                } => {
                    usage.add_turn(&u);
                    budget_used.input_tokens += u.input_tokens as u64;
                    budget_used.output_tokens += u.output_tokens as u64;

                    // Always-halt short circuit (Layer 1).
                    if let Some(c) = calls.iter().find(|c| tool_registry.is_always_halt(&c.name)) {
                        return RunResult::Failure {
                            reason: HaltReason::UnrecoverableToolError {
                                tool: c.name.clone(),
                                error: "tool is annotated always_halt".into(),
                            },
                            session_id,
                            usage,
                            turns: budget_used.turns,
                            session_state: session_state.clone(),
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

                    // Middleware: BeforeTool (rich chain, issue #11 / SC-11). The
                    // chain mutates `calls` IN PLACE via a priority-ordered
                    // fan-out; `ContinueWithModification` is the modified-but-proceed
                    // signal. The assistant turn recorded just above keeps the
                    // model's ORIGINAL request — only what is dispatched changes.
                    if let Some(mw) = self.config.middleware.as_ref() {
                        match mw.fire_before_tool(&mut calls, budget_used.turns).await {
                            MiddlewareDecision::Continue
                            | MiddlewareDecision::ContinueWithModification => {}
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeTool,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
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
                                    // #140: carry this leaf's toolset handle so
                                    // resume routes through its scoped catalogue.
                                    toolset: toolset.clone(),
                                };
                                return RunResult::WaitingForHuman {
                                    state: Box::new(state),
                                    request,
                                };
                            }
                            // `ForceAnotherTurn` is valid only at `BeforeCompletion`;
                            // the StandardMiddlewareChain converts it to `Halt` here.
                            // Defensive for a custom chain that emits it.
                            MiddlewareDecision::ForceAnotherTurn { inject } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::BeforeTool,
                                        reason: format!(
                                            "ForceAnotherTurn is not valid at BeforeTool: {inject}"
                                        ),
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                        }
                    }

                    let mut approved_results: Vec<ToolResult> = Vec::new();
                    // SC-9: the `session_state.messages` index of each appended
                    // tool result, recorded 1:1 with `approved_results`. The
                    // AfterTool middleware hook uses these to re-render any result
                    // it rewrites in place (via `ContextManager::replace_tool_result`).
                    let mut result_msg_indices: Vec<usize> = Vec::new();
                    for (i, call) in calls.iter().enumerate() {
                        // Sandbox validation. The pre-dispatch `validate` seam
                        // honours the SAME policy as tool-surfaced violations
                        // (`sandbox_violation_halts`): halt only under `Halt` for
                        // an always-halt-eligible violation; otherwise recoverable.
                        if let Err(violation) = self.config.sandbox.validate(call).await {
                            if self.sandbox_violation_halts(&violation) {
                                return RunResult::Failure {
                                    reason: HaltReason::SandboxViolation { violation },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            // Recoverable — append as tool error.
                            let tr = ToolResult {
                                call_id: call.id.clone(),
                                output: ToolOutput::Error {
                                    message: format!("sandbox: {violation}"),
                                    recoverable: true,
                                },
                            };
                            Self::emit(
                                &on_stream,
                                StreamEvent::ToolResult {
                                    call_id: call.id.clone(),
                                    is_error: true,
                                    // Q5: carry the result content.
                                    content: format!("sandbox: {violation}"),
                                    revised: false,
                                    node: None,
                                },
                            );
                            self.config
                                .context_manager
                                .append_tool_result(&mut session_state, &tr)
                                .await;
                            result_msg_indices.push(session_state.messages.len().saturating_sub(1));
                            approved_results.push(tr);
                            continue;
                        }

                        Self::emit(
                            &on_stream,
                            StreamEvent::ToolCall {
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                // Q5: carry the final tool-call arguments.
                                args: call.input.clone(),
                                node: None,
                            },
                        );
                        let tool_started_at = Timestamp::now();
                        let tool_clock = Instant::now();
                        // `effective_call` is the call actually dispatched. It
                        // starts as the agent's call and may be replaced by a
                        // repaired variant below. Spans/results are stamped from it.
                        let mut effective_call = call.clone();
                        let mut output = tool_registry.dispatch(effective_call.clone()).await;

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
                                let schema = tool_registry
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
                                output = tool_registry.dispatch(effective_call.clone()).await;
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
                                        let schema = tool_registry
                                            .schemas()
                                            .into_iter()
                                            .find(|s| s.name == effective_call.name);
                                        output = Self::enrich_tool_error(m, schema.as_ref());
                                    }
                                }
                            }
                        }

                        // Tool-surfaced sandbox violation: the HARNESS decides
                        // halt-vs-recoverable per `sandbox_violation_policy`, not
                        // the tool. Under `Halt` an always-halt-eligible violation
                        // ends the run with a typed `HaltReason::SandboxViolation`;
                        // otherwise (the default) it becomes a recoverable error
                        // the model retries from — uniform across every tool.
                        if let ToolOutput::SandboxViolation { violation } = &output {
                            if self.sandbox_violation_halts(violation) {
                                return RunResult::Failure {
                                    reason: HaltReason::SandboxViolation {
                                        violation: violation.clone(),
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            let message = format!("sandbox violation: {violation}");
                            output = ToolOutput::Error {
                                message,
                                recoverable: true,
                            };
                        }
                        let call = &effective_call;

                        // #126 (AC2): record harness-OBSERVED write/edit paths
                        // from the call ACTUALLY dispatched. This is the single
                        // file-observation point the PlanExecute DAG executor reads
                        // via `take_observed_writes` to build a task's ledger
                        // `files_touched` — never a model-self-reported field.
                        self.observe_write_call(call);

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
                                // #140: the parent leaf's toolset handle (the
                                // child carries its own inside `child_state`).
                                toolset: toolset.clone(),
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
                                // #140: carry this leaf's toolset handle so resume
                                // routes pending per-node calls through its catalogue.
                                toolset: toolset.clone(),
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
                                // #140: carry this leaf's toolset handle so the
                                // preserved clarifying call resumes against its
                                // scoped catalogue.
                                toolset: toolset.clone(),
                            };
                            return RunResult::WaitingForHuman {
                                state: Box::new(state),
                                request,
                            };
                        }

                        // Consult pause (issue #114, R1/R10): a worker-side tool
                        // returns `ToolOutput::Consult { child_state: None, .. }`
                        // to ask for mid-loop help. Like the
                        // `AwaitingClarification` arm, there is NO
                        // `ChildPausedState` at this level — the loop builds a
                        // `PausedState` directly with `human_request: None` and
                        // preserves the CONSULTING call as the head of
                        // `pending_tool_calls` (followed by the remaining batch),
                        // so that on `resume_consult` the helper's answer is
                        // injected as the tool RESULT for that pending call. The
                        // consult is a control signal, NOT a conversation turn:
                        // it is never appended to message history here (R10).
                        if let ToolOutput::Consult { request, .. } = output {
                            // Observability: lightweight consult-spawn event
                            // alongside `SkillInjected`.
                            if let Some(obs) = self.config.observability.as_ref() {
                                let base = SpanBase::new_root(
                                    SpanId::new(format!(
                                        "{}-consult-spawn-{}",
                                        session_id.as_str(),
                                        span_seq
                                    )),
                                    session_id.clone(),
                                    task.id.clone(),
                                    SpanKind::ContextAssembly,
                                    Timestamp::now(),
                                );
                                obs.emit_context(ContextSpan {
                                    base,
                                    operation: ContextOperation::ConsultSpawned {
                                        consult_kind: request.kind.clone(),
                                    },
                                    tokens_before: 0,
                                    tokens_after: 0,
                                    utilization_before: 0.0,
                                    utilization_after: 0.0,
                                });
                                // No `span_seq += 1` here: we return immediately
                                // below, so the increment would be dead.
                            }
                            let mut pending = vec![call.clone()];
                            pending.extend_from_slice(&calls[i + 1..]);
                            let turns = budget_used.turns;
                            let state = PausedState {
                                session_id: session_id.clone(),
                                task_id: task.id.clone(),
                                turn_number: budget_used.turns,
                                session_state,
                                pending_tool_calls: pending,
                                approved_results,
                                human_request: None,
                                task,
                                budget_used,
                                child_state: None,
                                // #140 (THE load-bearing path): carry this leaf's
                                // toolset handle so `resume_consult` routes the
                                // preserved consulting call through its scoped
                                // catalogue instead of the global fallback.
                                toolset: toolset.clone(),
                            };
                            return RunResult::Consult {
                                request,
                                state: Box::new(state),
                                session_id,
                                usage,
                                turns,
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
                                        node: None,
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
                                session_state: session_state.clone(),
                            };
                        }

                        // ── Consecutive-recoverable-tool-error breaker (#137) ──
                        // `output` here is either a Success or a RECOVERABLE
                        // Error (every other variant early-returned above). On a
                        // success the tool's error run resets (AC1); on a
                        // recoverable error we increment the identical-args run
                        // (AC1 args-variant) and check the N / 2N thresholds.
                        // At N we stash the corrective USER message here and
                        // append it AFTER the tool result below (well-formed
                        // conversation: assistant tool_use → tool result →
                        // corrective user message).
                        let mut pending_corrective: Option<String> = None;
                        if let ToolOutput::Error {
                            recoverable: true, ..
                        } = &output
                        {
                            let run = match error_runs.get_mut(&call.name) {
                                // Same tool, structurally-identical args → extend
                                // the run.
                                Some(existing) if existing.args == call.input => {
                                    existing.count += 1;
                                    existing
                                }
                                // First error, or the args changed → fresh run.
                                _ => error_runs
                                    .entry(call.name.clone())
                                    .insert_entry(ErrorRun {
                                        args: call.input.clone(),
                                        count: 1,
                                        injected: false,
                                    })
                                    .into_mut(),
                            };
                            let count = run.count;
                            let two_n = error_loop_n.saturating_mul(2);

                            // 2N: HARD STOP (AC3). Do NOT append this last tool
                            // result or continue — return a typed terminal that
                            // `ReactConfig::run` routes through the node's
                            // `BudgetExhaustedBehavior` WITHOUT burning the rest
                            // of the budget. The breaker is disabled when N == 0.
                            if error_loop_n > 0 && count >= two_n {
                                // AC4: emit the "broken" pair (stream + obs).
                                Self::emit(
                                    &on_stream,
                                    StreamEvent::ToolErrorLoopBroken {
                                        tool: call.name.clone(),
                                        consecutive_errors: count,
                                        node: None,
                                    },
                                );
                                if let Some(obs) = self.config.observability.as_ref() {
                                    let base = SpanBase::new_root(
                                        SpanId::new(format!(
                                            "{}-tool-error-loop-broken-{}",
                                            session_id.as_str(),
                                            span_seq
                                        )),
                                        session_id.clone(),
                                        task.id.clone(),
                                        SpanKind::ContextAssembly,
                                        Timestamp::now(),
                                    );
                                    obs.emit_context(ContextSpan {
                                        base,
                                        operation: ContextOperation::ToolErrorLoopBroken {
                                            tool_name: call.name.clone(),
                                            consecutive_errors: count,
                                        },
                                        tokens_before: 0,
                                        tokens_after: 0,
                                        utilization_before: 0.0,
                                        utilization_after: 0.0,
                                    });
                                }
                                return RunResult::Failure {
                                    reason: HaltReason::ToolErrorLoop {
                                        tool: call.name.clone(),
                                        consecutive_errors: count,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }

                            // N: inject ONE corrective message (AC2). Render the
                            // schema+hint via `enrich_tool_error` (reused
                            // verbatim) and inject it as a USER-role message —
                            // the SAME channel the Stop-block breaker uses — once
                            // per run (do not re-inject between N and 2N).
                            if error_loop_n > 0 && count >= error_loop_n && !run.injected {
                                run.injected = true;
                                let schema = tool_registry
                                    .schemas()
                                    .into_iter()
                                    .find(|s| s.name == call.name);
                                // `enrich_tool_error` renders the bare message +
                                // schema + hint; the bare message is this run's
                                // own error text.
                                let bare = match &output {
                                    ToolOutput::Error { message, .. } => message.as_str(),
                                    _ => "",
                                };
                                let corrective =
                                    match Self::enrich_tool_error(bare, schema.as_ref()) {
                                        ToolOutput::Error { message, .. } => message,
                                        // enrich_tool_error always returns Error.
                                        _ => bare.to_string(),
                                    };
                                // AC4: emit the "detected/warning" pair (stream +
                                // obs) BEFORE the corrective message is appended.
                                Self::emit(
                                    &on_stream,
                                    StreamEvent::ToolErrorLoopDetected {
                                        tool: call.name.clone(),
                                        consecutive_errors: count,
                                        node: None,
                                    },
                                );
                                if let Some(obs) = self.config.observability.as_ref() {
                                    let base = SpanBase::new_root(
                                        SpanId::new(format!(
                                            "{}-tool-error-loop-detected-{}",
                                            session_id.as_str(),
                                            span_seq
                                        )),
                                        session_id.clone(),
                                        task.id.clone(),
                                        SpanKind::ContextAssembly,
                                        Timestamp::now(),
                                    );
                                    obs.emit_context(ContextSpan {
                                        base,
                                        operation: ContextOperation::ToolErrorLoopDetected {
                                            tool_name: call.name.clone(),
                                            consecutive_errors: count,
                                        },
                                        tokens_before: 0,
                                        tokens_after: 0,
                                        utilization_before: 0.0,
                                        utilization_after: 0.0,
                                    });
                                }
                                // Append after the normal tool-result append below
                                // so the conversation stays well-formed (assistant
                                // tool_use → tool result → corrective user msg).
                                // Stash it for injection just after the tool
                                // result is recorded.
                                pending_corrective = Some(corrective);
                            }
                        } else {
                            // AC1: ANY success for this tool resets its error run.
                            error_runs.remove(&call.name);

                            // ── Repeated-identical-successful-call breaker ──
                            // A model that calls the SAME tool with the SAME args
                            // and keeps SUCCEEDING is spinning: it makes no
                            // progress and never reaches a terminal answer (the
                            // plan-phase loop that silently burned the budget).
                            // Track consecutive identical successes; nudge once at
                            // `N`, hard-stop at `2N`. `send_message` is exempt
                            // (benign narration a model may legitimately repeat).
                            if error_loop_n > 0 && call.name != crate::tools::SendMessageTool::NAME
                            {
                                let sig = (call.name.clone(), call.input.clone());
                                if last_success_sig.as_ref() == Some(&sig) {
                                    repeat_count += 1;
                                } else {
                                    last_success_sig = Some(sig);
                                    repeat_count = 1;
                                    repeat_injected = false;
                                }
                                let two_n = error_loop_n.saturating_mul(2);
                                // 2N: HARD STOP. Do NOT append this result or
                                // continue — end with a typed terminal so the
                                // caller/operator can redirect. The run does NOT
                                // auto-continue (more budget would just feed the
                                // same loop), and no remaining budget is burned.
                                if repeat_count >= two_n {
                                    return RunResult::Failure {
                                        reason: HaltReason::RepeatedToolCall {
                                            tool: call.name.clone(),
                                            count: repeat_count,
                                        },
                                        session_id,
                                        usage,
                                        turns: budget_used.turns,
                                        session_state: session_state.clone(),
                                    };
                                }
                                // N: inject ONE corrective USER message (the same
                                // channel the error breaker / Stop-block breaker
                                // use), once per run, appended after this result.
                                if repeat_count >= error_loop_n && !repeat_injected {
                                    repeat_injected = true;
                                    pending_corrective = Some(format!(
                                        "You have called the `{tool}` tool with the same arguments \
{count} times in a row, each returning the same result. Repeating an identical call will not make \
progress. Take a DIFFERENT action now — or, if you already have what you need, produce your final \
response (finalize the plan / answer the task).",
                                        tool = call.name,
                                        count = repeat_count,
                                    ));
                                }
                            }
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
                                    // Escalate / AwaitingClarification / Consult
                                    // are handled and returned above; they never
                                    // reach the capture path.
                                    ToolOutput::Escalate { .. } => None,
                                    ToolOutput::AwaitingClarification { .. } => None,
                                    ToolOutput::Consult { .. } => None,
                                    // Normalized into an `Error` (or a halt)
                                    // before this point; never reaches here.
                                    ToolOutput::SandboxViolation { .. } => None,
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
                                ToolOutput::Consult { .. } => (0, false),
                                // Normalized away before this point.
                                ToolOutput::SandboxViolation { .. } => (0, false),
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
                        // Q5: surface the result content on the coarse event.
                        let result_content = match &tr.output {
                            ToolOutput::Success { content, .. } => content.clone(),
                            ToolOutput::Error { message, .. } => message.clone(),
                            _ => String::new(),
                        };
                        Self::emit(
                            &on_stream,
                            StreamEvent::ToolResult {
                                call_id: call.id.clone(),
                                is_error,
                                content: result_content,
                                revised: false,
                                node: None,
                            },
                        );
                        self.config
                            .context_manager
                            .append_tool_result(&mut session_state, &tr)
                            .await;
                        result_msg_indices.push(session_state.messages.len().saturating_sub(1));
                        approved_results.push(tr);

                        // #137 (AC2): inject the ONE corrective user message at
                        // the N threshold, AFTER this call's tool result is
                        // recorded — same `append_user_message` channel the
                        // Stop-block breaker uses.
                        if let Some(corrective) = pending_corrective.take() {
                            self.config
                                .context_manager
                                .append_user_message(&mut session_state, &corrective)
                                .await;
                        }
                    }

                    // Middleware: AfterTool (rich chain, issue #11 / SC-9). The
                    // chain receives the batch's results as `&mut` and may rewrite
                    // any of them in place (priority-ordered, descending). On
                    // `ContinueWithModification`, re-render the affected tool-result
                    // messages so the rewrite reaches the next model turn — this is
                    // what lets an after-tool middleware turn a landed write into a
                    // model-visible error (or vice versa) without the tool itself
                    // having to invert its output.
                    if let Some(mw) = self.config.middleware.as_ref() {
                        // Snapshot each result's stream view BEFORE the chain runs,
                        // so on `ContinueWithModification` we re-emit ONLY the ones
                        // it actually rewrote.
                        let pre_views: Vec<(bool, String)> = approved_results
                            .iter()
                            .map(|r| stream_view(&r.output))
                            .collect();
                        match mw.fire_after_tool(&calls, &mut approved_results).await {
                            MiddlewareDecision::Continue => {}
                            MiddlewareDecision::ContinueWithModification => {
                                for ((res, &idx), pre) in approved_results
                                    .iter()
                                    .zip(result_msg_indices.iter())
                                    .zip(pre_views.iter())
                                {
                                    self.config
                                        .context_manager
                                        .replace_tool_result(&mut session_state, idx, res)
                                        .await;
                                    // Symmetric stream-side correction: the model
                                    // side was patched above, but the `ToolResult`
                                    // already on the stream still shows the stale
                                    // content. Re-emit with `revised: true` ONLY for
                                    // results the chain actually changed, so consumers
                                    // correct in place rather than double-count by
                                    // `call_id`.
                                    let (is_error, content) = stream_view(&res.output);
                                    if is_error != pre.0 || content != pre.1 {
                                        Self::emit(
                                            &on_stream,
                                            StreamEvent::ToolResult {
                                                call_id: res.call_id.clone(),
                                                is_error,
                                                content,
                                                revised: true,
                                                node: None,
                                            },
                                        );
                                    }
                                }
                            }
                            MiddlewareDecision::Halt { reason } => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::AfterTool,
                                        reason,
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
                            // `SurfaceToHuman` / `ForceAnotherTurn` are not valid at
                            // `AfterTool`; the StandardMiddlewareChain converts them
                            // to `Halt`. Defensive for a custom chain that emits one.
                            other => {
                                return RunResult::Failure {
                                    reason: HaltReason::MiddlewareHalt {
                                        hook: HookPoint::AfterTool,
                                        reason: format!("illegal AfterTool decision: {other:?}"),
                                    },
                                    session_id,
                                    usage,
                                    turns: budget_used.turns,
                                    session_state: session_state.clone(),
                                };
                            }
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
                            &agent,
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
                        session_state: session_state.clone(),
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
        evaluator: Arc<dyn Agent>,
        eval_toolset: ToolsetRef,
        carried: &mut BudgetSnapshot,
        total_usage: &mut AggregateUsage,
    ) -> RunResult {
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
            loop_strategy: LoopStrategy::ReAct(ReactConfig::per_loop(
                task.budget.max_turns.unwrap_or(u32::MAX),
            )),
        };

        // Child harness: clone the config, swap the sandbox to read-only. Cloning
        // shares the same observability/storage seams so the evaluate run's spans
        // land in the SAME trace stream (distinguished by its distinct session
        // id). The evaluate agent is passed to `run_react` directly (#124 — no
        // `config.agent`).
        let mut eval_config = self.config.clone();
        eval_config.sandbox = read_only_sandbox;
        // The evaluate phase is a NON-INTERACTIVE nested review run: the caller
        // never sees its (freshly generated) session id, so an approval/HITL
        // middleware that resolves `SurfaceToHuman` would pause this run with no
        // human able to resume it — the reviewer's first `read_file` would yield
        // `WaitingForHuman`, which the verifier reads as a misconfiguration, and
        // the review half would silently never run. The read-only sandbox already
        // enforces the only safety property a reviewer needs (no writes, no
        // command execution), so the caller's approval middleware is both
        // redundant and harmful here — drop it for the evaluate phase. (Pair with
        // an `eval_toolset` scoped to read-only tools so non-filesystem
        // side-effecting tools the sandbox does not gate stay out of reach.)
        eval_config.middleware = None;
        // SC-30: when the eval phase falls back to the GLOBAL catalogue (empty
        // `eval_toolset`), auto-derive a read-only VIEW of it — advertise +
        // dispatch only the intersection with `StandardTools::readonly_set()`, so
        // the reviewer cannot reach write / exec / side-effecting tools (incl.
        // web/MCP the read-only sandbox does not gate) WITHOUT the consumer
        // registering a scoped read-only toolset. A non-empty `eval_toolset` is an
        // explicit opt-in and is left untouched (it resolves via its own catalogue).
        if eval_toolset.0.is_empty() {
            if let Some(catalogue) = eval_config.catalogue_registry.take() {
                let allow: std::collections::HashSet<String> =
                    crate::tools::StandardTools::readonly_set()
                        .into_iter()
                        .map(|t| t.schema.name)
                        .collect();
                let real: Arc<dyn ToolRegistry> =
                    Arc::new(crate::tool_registry::RealToolRegistry::new(
                        catalogue,
                        eval_config.sandbox.clone(),
                        eval_session_id.clone(),
                        eval_config.project_id.clone(),
                        eval_config.storage.run().clone(),
                        eval_config.storage.memory().clone(),
                    ));
                // `catalogue_registry` is now None (taken), so the empty-handle
                // path of `effective_tool_registry` returns this filtered
                // `tool_registry` for the reviewer's dispatch + schema advertising.
                eval_config.tool_registry = Arc::new(ReadOnlyToolView::new(real, allow));
            }
        }
        let eval_harness = StandardHarness::new(eval_config);

        let mut eval_state = SessionState::default();
        eval_harness
            .config
            .context_manager
            .append_user_message(&mut eval_state, &directive)
            .await;

        let cap = task.budget.max_turns.unwrap_or(u32::MAX);
        let eval_result = eval_harness
            .run_react(
                eval_task,
                cap,
                eval_state,
                BudgetSnapshot::default(),
                None,
                evaluator,
                eval_toolset,
            )
            .await;

        fold_usage(total_usage, carried, &eval_result);
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
    /// Ralph external completion check (issue #58, B1). Reads the deterministic
    /// `.spore/` files under `workspace_root` and reports whether the task is
    /// complete. Returns `None` when complete, `Some(reason)` when tasks remain
    /// (the reason describes what is left). This is the SAME logic the registered
    /// [`RalphStopHook`] applies — one source of truth for the completion
    /// mechanism.
    ///
    /// Contract (#142, decision 3 — the checkpoint MOVED off the `.spore/`
    /// filesystem onto the durable project-id [`RunStore`](crate::storage::RunStore)):
    ///   - [`RALPH_PROGRESS_KEY`](crate::storage::RALPH_PROGRESS_KEY) →
    ///     `{ "complete": bool, "remaining": [string] }`. `complete: true` with an
    ///     empty `remaining` ⇒ progress satisfied. Absent/unreadable/invalid ⇒
    ///     incomplete (so the agent learns to write it).
    ///   - [`RALPH_FEATURE_LIST_KEY`](crate::storage::RALPH_FEATURE_LIST_KEY) → a
    ///     JSON array of `{ "name", "passes" }` (the
    ///     [`FeatureListCheck`](crate::termination::FeatureListCheck) schema). Any
    ///     `passes: false` ⇒ incomplete. An ABSENT feature list is tolerated here
    ///     (progress is the primary signal); an invalid one is not.
    ///
    /// The reason strings stay byte-identical to the legacy `.spore/`-path
    /// messages so the surfaced [`HaltReason::RalphCompletionUnmet`] text is
    /// stable across the storage relocation.
    async fn ralph_completion_status(
        run_store: &Arc<dyn crate::storage::RunStore>,
        project: &crate::storage::ProjectId,
    ) -> Option<String> {
        let ns = project.namespace();
        let progress_value = match run_store.get(&ns, crate::storage::RALPH_PROGRESS_KEY).await {
            Ok(Some(v)) => v,
            // Absent OR a storage error ⇒ incomplete (the agent must write it).
            _ => return Some(".spore/progress.json missing".to_string()),
        };
        let progress: RalphProgress = match serde_json::from_value(progress_value) {
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
        if let Ok(Some(value)) = run_store
            .get(&ns, crate::storage::RALPH_FEATURE_LIST_KEY)
            .await
        {
            let entries: Vec<RalphFeatureEntry> = match serde_json::from_value(value) {
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

    /// Build the reload context block injected into each fresh context window
    /// (issue #58, R3). #142: reads the checkpoint from the durable project-id
    /// [`RunStore`](crate::storage::RunStore) (not the `.spore/` filesystem) and
    /// returns the verbatim progress + feature-list JSON (when present) so the
    /// re-seeded window knows what is done and what remains. Returns `None` when
    /// neither checkpoint key is set. The "Reloaded .spore/…" prefix is retained
    /// so the seeded prompt text is byte-stable across the relocation.
    async fn ralph_reload_context(
        run_store: &Arc<dyn crate::storage::RunStore>,
        project: &crate::storage::ProjectId,
    ) -> Option<String> {
        let ns = project.namespace();
        let mut parts: Vec<String> = Vec::new();
        if let Ok(Some(v)) = run_store.get(&ns, crate::storage::RALPH_PROGRESS_KEY).await {
            if let Ok(raw) = serde_json::to_string(&v) {
                parts.push(format!("Reloaded .spore/progress.json:\n{}", raw.trim()));
            }
        }
        if let Ok(Some(v)) = run_store
            .get(&ns, crate::storage::RALPH_FEATURE_LIST_KEY)
            .await
        {
            if let Ok(raw) = serde_json::to_string(&v) {
                parts.push(format!(
                    "Reloaded .spore/feature_list.json:\n{}",
                    raw.trim()
                ));
            }
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    /// Write the Ralph progress checkpoint to the durable project-id
    /// [`RunStore`](crate::storage::RunStore) (issue #142, decision 3 — the WRITE
    /// path the relocated checkpoint needs; nothing wrote `progress.json` before).
    /// `complete: true` + empty `remaining` ⇒
    /// [`ralph_completion_status`](Self::ralph_completion_status) reports done.
    pub async fn write_ralph_progress(
        &self,
        complete: bool,
        remaining: Vec<String>,
    ) -> Result<(), crate::storage::StorageError> {
        let progress = RalphProgress {
            complete,
            remaining,
        };
        let value = serde_json::to_value(&progress)
            .map_err(|e| crate::storage::StorageError::Serialization(e.to_string()))?;
        self.config
            .storage
            .run()
            .put(
                &self.config.project_id.namespace(),
                crate::storage::RALPH_PROGRESS_KEY,
                value,
            )
            .await
    }

    /// Write the Ralph feature-list checkpoint to the durable project-id
    /// [`RunStore`](crate::storage::RunStore) (issue #142, decision 3). Each entry
    /// is `{ "name", "passes" }`; any `passes: false` keeps the run incomplete.
    pub async fn write_ralph_feature_list(
        &self,
        features: &[(String, bool)],
    ) -> Result<(), crate::storage::StorageError> {
        let entries: Vec<serde_json::Value> = features
            .iter()
            .map(|(name, passes)| serde_json::json!({ "name": name, "passes": passes }))
            .collect();
        let value = serde_json::Value::Array(entries);
        self.config
            .storage
            .run()
            .put(
                &self.config.project_id.namespace(),
                crate::storage::RALPH_FEATURE_LIST_KEY,
                value,
            )
            .await
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
    //       * keep   → status Kept, commit the accepted neighbor (advancing
    //                  `current_sha`/`current_best`, and `best_sha`/`best_score`
    //                  if it beats the best), reset the stagnation counter to 0.
    //       * no keep → status Discarded; if `revert_on_no_improvement`, run
    //                  `git reset --hard <current_sha>` via the sandbox (the last
    //                  ACCEPTED state, NOT the original HEAD — hill-climbing-loop-
    //                  correctness §4.3); increment the stagnation counter.
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
    //   1. REVERT/COMMIT (revised by hill-climbing-loop-correctness): the harness
    //      now COMMITS each accepted candidate through the sandbox (`git add -A &&
    //      git commit`, D1) so `commit_hash` carries the real accepted SHA, and
    //      `revert_on_no_improvement: true` reverts via `git reset --hard
    //      <current_sha>` — the last accepted state, not the original HEAD. A
    //      wired `VcsProvider` owns commit/head/reset; with no provider AND no
    //      usable git (e.g. a fake sandbox), the SHAs are `None` and the loop
    //      degrades to the pre-fix behaviour (empty `commit_hash`, reset to HEAD).
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

    /// Reset the working tree for a no-improvement iteration
    /// (`hill-climbing-loop-correctness` §4.3). The load-bearing fix: revert
    /// targets the last ACCEPTED candidate (`target_sha`), NOT the original
    /// `HEAD`. Routing precedence (SC-14 — the harness NEVER spawns git
    /// directly):
    ///   - a wired [`VcsProvider`] owns the reset: `reset_to(sha)` for a known
    ///     SHA, else `revert()` (its revert-to-last-known-good);
    ///   - otherwise git-through-sandbox: `git reset --hard <sha>` for a known
    ///     SHA, else `git reset --hard HEAD` — byte-identical to the pre-fix
    ///     fallback when the workspace has no usable VCS (`target_sha == None`).
    ///
    /// Best-effort: an error / non-zero exit is swallowed and the loop continues.
    async fn hill_climbing_reset_to(&self, target_sha: Option<&str>) {
        let known = target_sha.filter(|s| !s.is_empty());
        if let Some(vcs) = &self.config.vcs_provider {
            match known {
                Some(sha) => {
                    let _ = vcs.reset_to(sha).await;
                }
                // No durable SHA (no-op/degenerate VCS) ⇒ fall back to the
                // provider's revert-to-last-known-good, preserving pre-fix
                // behaviour for providers that only implement `revert`.
                None => {
                    let _ = vcs.revert().await;
                }
            }
            return;
        }
        let target = known.unwrap_or("HEAD").to_string();
        let args = ["reset".to_string(), "--hard".to_string(), target];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &args, None, None)
            .await;
    }

    /// Resolve the workspace's current commit SHA
    /// (`hill-climbing-loop-correctness` §4.4). A wired [`VcsProvider`] supplies
    /// it (`head()`); otherwise `git rev-parse HEAD` THROUGH the sandbox. Returns
    /// `None` when there is no usable VCS — a blocked spawn, a non-zero exit, or
    /// empty stdout (the case for a fake/no-op sandbox). A `None` here is what
    /// makes the loop degrade gracefully to its pre-fix behaviour rather than
    /// recording a bogus SHA.
    async fn hill_climbing_head_sha(&self) -> Option<String> {
        if let Some(vcs) = &self.config.vcs_provider {
            return vcs
                .head()
                .await
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
        }
        let args = ["rev-parse".to_string(), "HEAD".to_string()];
        let out = self
            .config
            .sandbox
            .execute_command("git", &args, None, None)
            .await
            .ok()?;
        if out.exit_code != 0 {
            return None;
        }
        let sha = out.stdout.trim().to_string();
        (!sha.is_empty()).then_some(sha)
    }

    /// Commit the current working tree as an accepted candidate and return its
    /// SHA (`hill-climbing-loop-correctness` §4.4, Decision D1: commit through
    /// the sandbox in the default path — no `VcsProvider` required). A wired
    /// provider's `commit_checkpoint` wins; otherwise `git add -A && git commit
    /// -m <label>` THROUGH the sandbox, then [`hill_climbing_head_sha`]. Returns
    /// `None` when no SHA could be obtained (no usable VCS) — the caller then
    /// records an empty `commit_hash` and keeps the pre-fix score-only behaviour
    /// (there is nothing durable to revert to, which is inherent without a VCS).
    ///
    /// [`hill_climbing_head_sha`]: Self::hill_climbing_head_sha
    async fn hill_climbing_commit_accepted(&self, label: &str) -> Option<String> {
        if let Some(vcs) = &self.config.vcs_provider {
            return vcs
                .commit_checkpoint(label)
                .await
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
        }
        // Stage everything EXCEPT the harness's own `.spore/` diagnostics: it
        // must stay UNTRACKED so a later `git reset --hard` (revert / winner
        // apply) leaves the live results ledger on disk (`git reset --hard` never
        // touches untracked files). Committing `.spore/` would let a revert
        // clobber the in-progress TSV with a stale committed copy.
        let add = [
            "add".to_string(),
            "-A".to_string(),
            "--".to_string(),
            HILL_CLIMB_COMMIT_EXCLUDE.to_string(),
        ];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &add, None, None)
            .await;
        let commit = ["commit".to_string(), "-m".to_string(), label.to_string()];
        let _ = self
            .config
            .sandbox
            .execute_command("git", &commit, None, None)
            .await;
        self.hill_climbing_head_sha().await
    }

    /// The commit message for an accepted HillClimbing candidate
    /// (`hill-climbing-loop-correctness` §4.4). Kept as a single helper so the
    /// exact string is byte-identical across the four reference implementations
    /// (per `consumer-friction-port-patterns`). Iteration 0 is the baseline.
    fn hill_commit_label(iteration: u32) -> String {
        if iteration == 0 {
            "hill-climb baseline".to_string()
        } else {
            format!("hill-climb iter {iteration}")
        }
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
                HillClimbingDirection::Minimize => "minimize",
                HillClimbingDirection::Maximize => "maximize",
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
        // #142: the task_list is DURABLE — key it by the STABLE project namespace
        // (`project_id.namespace()`), NOT the per-window `session_id` the Ralph
        // wrapper regenerates each context window. Namespace-reuse keeps the
        // RunStore trait keyed by `&SessionId` while the value keyed is the
        // stable project id. The incoming `session_id` remains the EPHEMERAL key
        // and is intentionally unused for this durable write.
        let _ = session_id;
        let durable_ns = self.config.project_id.namespace();
        if let Ok(value) = serde_json::to_value(task_list) {
            // Durable write through the storage seam (RunStore).
            let _ = self
                .config
                .storage
                .run()
                .put(&durable_ns, crate::tasklist::TASK_LIST_EXTRAS_KEY, value)
                .await;
        }
    }

    /// The planning directive seeded before the constrained plan turn (R1): the
    /// "respond with a single JSON plan object" instruction wrapped around the
    /// task. Seeded by the recursive [`PlanExecuteConfig::run`] plan dispatch
    /// before `self.plan.run(cx)` (#124).
    fn plan_directive(instruction: &str) -> String {
        format!(
            "Produce a step-by-step plan for the following task. Respond with a \
             single JSON object: {{\"tasks\": [<ordered step strings>], \
             \"rationale\": <string>}}.\n\nTask:\n{instruction}"
        )
    }

    /// Capture + persist a [`PlanArtifact`] from the plan turn's output text
    /// (#124): R3 (parse), R11 (fire `OnPlanCreated`, mutable), R4 (persist to the
    /// RunStore under `PLAN_EXECUTE_EXTRAS_KEY`). The model turn that produced
    /// `plan_output` ran elsewhere — the recursive `self.plan` child via
    /// `self.plan.run(cx)` — so this carries no agent call. Returns the captured
    /// artifact + accounting or a terminal failure to propagate.
    ///
    /// # SC-28 — a free-text / markdown plan is NOT a hard failure
    /// The strict canonical grammar (+ prose repair) runs first, so a JSON plan
    /// captures exactly as before — `tasks`/`rationale` come straight from the
    /// object. When BOTH fail the planner emitted prose rather than the JSON
    /// object; rather than aborting the whole run we capture it as a free-text
    /// artifact: the verbatim prose is preserved in `rationale`, and the runnable
    /// `tasks` are sourced from the durable `task_list` tool store — the ONE
    /// authoring path (#126 decision C) that the execute phase already prefers
    /// over the artifact, so JSON was never the only source of executable steps.
    /// Mirroring it here keeps the `OnPlanCreated` payload's `tasks` populated for
    /// panel consumers (looper `plan_tracker`, cordyceps `plan_announcer`). A
    /// prose plan that authored no `task_list` yields empty `tasks` — the execute
    /// phase then halts with `EmptyPlan`, exactly as a JSON `{"tasks": []}` would.
    /// The pure [`capture_plan_artifact`](crate::plan::capture_plan_artifact)
    /// grammar stays strict and byte-identical across languages; only this harness
    /// driver is tolerant.
    async fn capture_and_persist_plan(
        &self,
        session_id: &SessionId,
        plan_output: &str,
        usage: AggregateUsage,
        turns: u32,
    ) -> Result<PlanPhaseOutcome, RunResult> {
        // R3: capture the artifact from the response text. Uses the prose-repair
        // fallback: a planner that wraps its plan JSON in prose still captures
        // (the strict canonical grammar runs first; repair only rescues a failure).
        let mut artifact = match crate::plan::capture_plan_artifact_with_repair(plan_output) {
            Ok(a) => a,
            // SC-28: not JSON → capture as free-text. `tasks` from the task_list
            // tool store (load_task_list reads the durable, project-scoped list
            // the plan leaf authored via the tool); prose preserved verbatim.
            Err(_strict_err) => {
                let tasks = self
                    .load_task_list(session_id)
                    .await
                    .map(|list| {
                        list.tasks
                            .iter()
                            .map(|t| t.description.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                crate::plan::PlanArtifact {
                    tasks,
                    rationale: plan_output.to_string(),
                }
            }
        };

        // R11: fire OnPlanCreated synchronously; the hook may rewrite `artifact`
        // in place. The stored artifact reflects any mutation.
        if let Some(chain) = self.config.hooks.as_ref() {
            let mut ctx = crate::hooks::HookContext::OnPlanCreated {
                session_id,
                plan: &mut artifact,
            };
            // Errors are non-fatal: an observability/handler error must not lose
            // a successfully-captured plan. The (possibly mutated) artifact is
            // still stored.
            let _ = chain.fire(&mut ctx).await;
        }

        // R4: persist the produced artifact to the RunStore seam under
        // PLAN_EXECUTE_EXTRAS_KEY (#76 — the durable single source of truth;
        // no longer mirrored into SessionState.extras). #142: the plan artifact
        // is DURABLE — key it by the STABLE project namespace, NOT the per-window
        // `session_id` (so a Ralph window reset re-reads the prior window's
        // plan). The put result is swallowed (matching the execute-phase
        // persist): a successfully captured plan must not be lost to a storage
        // hiccup.
        match serde_json::to_value(&artifact) {
            Ok(value) => {
                let _ = self
                    .config
                    .storage
                    .run()
                    .put(
                        &self.config.project_id.namespace(),
                        crate::plan::PLAN_EXECUTE_EXTRAS_KEY,
                        value,
                    )
                    .await;
            }
            Err(e) => {
                return Err(RunResult::Failure {
                    reason: HaltReason::PlanPhaseFailed {
                        error: crate::plan::PlanPhaseError::UnparseablePlan {
                            message: format!("failed to serialize plan artifact: {e}"),
                        },
                    },
                    session_id: session_id.clone(),
                    usage,
                    turns,
                    session_state: SessionState::default(),
                });
            }
        }

        Ok(PlanPhaseOutcome {
            artifact,
            usage,
            turns,
        })
    }

    /// A.6 deep-resume reconcile (#124, Q2): mark every task already `Completed`
    /// on the DURABLE RunStore checkpoint as `Completed` in the freshly-parsed
    /// `task_list` so it is NOT re-run. Tasks are matched positionally by `id`
    /// (the task list is regenerated deterministically from the same artifact).
    /// Extracted from the legacy `run_execute_phase` so the recursive
    /// [`PlanExecuteConfig::run`] owns the per-task loop while the durable-state
    /// reconcile stays here.
    async fn reconcile_deep_resume(
        &self,
        session_id: &SessionId,
        task_list: &mut crate::tasklist::TaskList,
    ) {
        use crate::tasklist::TaskStatus;
        // #142: the durable checkpoint is keyed by the STABLE project namespace,
        // NOT the per-window `session_id` — so deep-resume reads the SAME list a
        // prior Ralph window persisted. The ephemeral `session_id` is unused for
        // this durable read.
        let _ = session_id;
        if let Ok(Some(value)) = self
            .config
            .storage
            .run()
            .get(
                &self.config.project_id.namespace(),
                crate::tasklist::TASK_LIST_EXTRAS_KEY,
            )
            .await
        {
            if let Ok(saved) = serde_json::from_value::<crate::tasklist::TaskList>(value) {
                for t in &mut task_list.tasks {
                    if saved
                        .tasks
                        .iter()
                        .any(|s| s.id == t.id && s.status == TaskStatus::Completed)
                    {
                        t.status = TaskStatus::Completed;
                    }
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
        // #124: the compaction summary turn runs on the SAME resolved worker
        // agent driving the ReAct window (no `config.agent`).
        agent: &Arc<dyn Agent>,
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
            let result = agent.turn(turn.context.clone()).await;
            let summary = match result {
                TurnResult::FinalResponse {
                    content, usage: u, ..
                } => {
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
                    &turn.dropped_messages,
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
                &turn.dropped_messages,
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
        dropped: &[Message],
        tokens_before: u32,
        session_id: &SessionId,
        task_id: &TaskId,
        span_seq: &mut u64,
    ) {
        self.config
            .context_manager
            .apply_compaction(session_state, summary, dropped);

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

/// [`StrategyExecutor`] impl (#124): the harness-side primitives the recursive
/// per-variant [`RunStrategy::run`] bodies delegate to. Each primitive wraps an
/// existing, tested orchestration method so behavior stays at parity — the only
/// structural change is that the per-variant bodies now own their loops and the
/// central dispatch `match` is gone.
impl StrategyExecutor for StandardHarness {
    fn react_window<'a>(
        &'a self,
        task: Task,
        max_iterations: u32,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        agent: Arc<dyn Agent>,
        toolset: ToolsetRef,
        output_schema: Option<serde_json::Value>,
        output_schema_max_retries: u32,
        system_prompt: Option<String>,
        execute_step: bool,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            self.run_react_inner(
                task,
                max_iterations,
                session_state,
                budget_used,
                on_stream,
                agent,
                toolset,
                output_schema,
                output_schema_max_retries,
                system_prompt,
                execute_step,
            )
            .await
        })
    }

    fn resolve_agent_ref(
        &self,
        r: &AgentRef,
        session_id: &SessionId,
    ) -> Result<Arc<dyn Agent>, Box<RunResult>> {
        self.config
            .registry
            .resolve_agent(r)
            .cloned()
            .ok_or_else(|| {
                Box::new(RunResult::Failure {
                    reason: HaltReason::ConfigurationError {
                        error: HarnessError::UnresolvedHandle {
                            kind: "agent".to_string(),
                            key: r.0.clone(),
                        },
                    },
                    session_id: session_id.clone(),
                    usage: AggregateUsage::default(),
                    turns: 0,
                    session_state: SessionState::default(),
                })
            })
    }

    fn evaluate_phase<'a>(
        &'a self,
        task: &'a Task,
        eval_agent: Arc<dyn Agent>,
        eval_toolset: ToolsetRef,
        carried: &'a mut BudgetSnapshot,
        total_usage: &'a mut AggregateUsage,
    ) -> BoxFut<'a, RunResult> {
        Box::pin(async move {
            self.run_evaluate_phase(task, eval_agent, eval_toolset, carried, total_usage)
                .await
        })
    }

    fn append_user_message<'a>(
        &'a self,
        session_state: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            self.config
                .context_manager
                .append_user_message(session_state, text)
                .await;
        })
    }

    fn workspace_root(&self) -> std::path::PathBuf {
        self.config.sandbox.workspace_root().to_path_buf()
    }

    fn seed_user_message<'a>(
        &'a self,
        session_state: &'a mut SessionState,
        text: &'a str,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            self.config
                .context_manager
                .append_user_message(session_state, text)
                .await;
        })
    }

    fn capture_plan_artifact<'a>(
        &'a self,
        session_id: &'a SessionId,
        plan_output: &'a str,
        usage: AggregateUsage,
        turns: u32,
    ) -> BoxFut<'a, Result<PlanPhaseOutcome, RunResult>> {
        Box::pin(async move {
            self.capture_and_persist_plan(session_id, plan_output, usage, turns)
                .await
        })
    }

    fn plan_directive(&self, instruction: &str) -> String {
        StandardHarness::plan_directive(instruction)
    }

    fn plan_empty_max_retries(&self) -> u32 {
        self.config.plan_empty_max_retries
    }

    fn run_plan_subtree<'a>(
        &'a self,
        plan: &'a LoopStrategy,
        plan_task: Task,
        plan_session: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        forced_output_schema: Option<serde_json::Value>,
    ) -> BoxFut<'a, Option<RunResult>> {
        Box::pin(async move {
            // #124 Q1: the planner concept is DROPPED — the plan child's leaf
            // `ReactConfig.agent` is authoritative and resolved by the recursing
            // leaf itself. `plan_task.loop_strategy == plan` (the caller stamps it),
            // so the WHOLE plan loop runs by driving that task.
            //
            // Drive it through `drive_strategy_with_resume_seed` — the SAME single
            // budget-resolution site the top level uses — instead of a bare
            // `plan.run(cx)` that read only `terminal_override`. A plan phase that
            // exhausts its turn budget propagates a `BudgetExhausted` StrategyOutcome
            // (rule 6: the leaf never self-resolves), which the bare drive DISCARDED,
            // returning `None` → the opaque `PlanPhaseFailed{"plan sub-strategy
            // produced no terminal"}`. Routing through the resolver makes the plan
            // phase auto-continue (Continue / AutoContinue) and then surface a
            // `WaitingForHuman{BudgetExhausted}` under `SurfaceToHuman` — exactly
            // like the execute phase — so the host can grant more budget and the
            // run continues (the plan-phase resume seed, #138 AC3, re-attaches the
            // stalled planner conversation). A plan that COMPLETES still returns its
            // `Success` terminal verbatim (byte-identical to before). `plan` is
            // unused now (the task carries the same strategy); kept for the trait
            // signature / call-site symmetry.
            let _ = plan;
            let result = self
                .drive_strategy_with_resume_seed(
                    plan_task,
                    plan_session,
                    budget_used,
                    on_stream,
                    None,  // resume_continues
                    None,  // resume_seed (the plan child plans fresh each attempt)
                    false, // continue_task_list
                    forced_output_schema,
                )
                .await;
            Some(result)
        })
    }

    fn reconcile_completed_tasks<'a>(
        &'a self,
        session_id: &'a SessionId,
        task_list: &'a mut crate::tasklist::TaskList,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            self.reconcile_deep_resume(session_id, task_list).await;
        })
    }

    fn fire_task_advance<'a>(
        &'a self,
        session_id: &'a SessionId,
        step_task: &'a mut Task,
        task_index: usize,
        total_tasks: usize,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            if let Some(chain) = self.config.hooks.as_ref() {
                let mut ctx = crate::hooks::HookContext::OnTaskAdvance {
                    session_id,
                    task: step_task,
                    task_index,
                    total_tasks,
                };
                let _ = chain.fire(&mut ctx).await;
            }
        })
    }

    fn persist_task_list<'a>(
        &'a self,
        session_id: &'a SessionId,
        task_list: &'a crate::tasklist::TaskList,
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            // Delegate to the existing private method (same RunStore seam).
            StandardHarness::persist_task_list(self, session_id, task_list).await
        })
    }

    fn load_task_list<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> BoxFut<'a, Option<crate::tasklist::TaskList>> {
        // #142: the task_list is DURABLE — read it from the STABLE project
        // namespace, NOT the per-window `session_id` (so the execute phase of a
        // fresh Ralph window sees the prior window's list). The ephemeral
        // `session_id` is unused for this durable read.
        let _ = session_id;
        Box::pin(async move {
            // #126 (decision C): the one authoring path — read the persisted
            // TaskList (with real blockers) from the RunStore under
            // TASK_LIST_EXTRAS_KEY. A storage miss / deserialize failure yields
            // None (the executor then falls back to the plan artifact).
            match self
                .config
                .storage
                .run()
                .get(
                    &self.config.project_id.namespace(),
                    crate::tasklist::TASK_LIST_EXTRAS_KEY,
                )
                .await
            {
                Ok(Some(value)) => serde_json::from_value::<crate::tasklist::TaskList>(value).ok(),
                _ => None,
            }
        })
    }

    fn clear_task_list<'a>(&'a self, session_id: &'a SessionId) -> BoxFut<'a, ()> {
        // #142: the durable task_list is keyed by the STABLE project namespace,
        // NOT the ephemeral per-run `session_id` — so the delete must target the
        // same project-scoped key as `persist_task_list`/`load_task_list`.
        let _ = session_id;
        Box::pin(async move {
            let _ = self
                .config
                .storage
                .run()
                .delete(
                    &self.config.project_id.namespace(),
                    crate::tasklist::TASK_LIST_EXTRAS_KEY,
                )
                .await;
        })
    }

    fn take_observed_writes(&self) -> Vec<String> {
        match self.observed_writes.lock() {
            Ok(mut acc) => std::mem::take(&mut *acc),
            Err(_) => Vec::new(),
        }
    }

    fn clear_observed_writes(&self) {
        if let Ok(mut acc) = self.observed_writes.lock() {
            acc.clear();
        }
    }

    fn ralph_seed_session<'a>(&'a self, instruction: &'a str) -> BoxFut<'a, SessionState> {
        Box::pin(async move {
            let mut session_state = SessionState::default();
            self.config
                .context_manager
                .append_user_message(&mut session_state, instruction)
                .await;
            // #142: the reload context now comes from the durable project-id
            // RunStore checkpoint, not the `.spore/` filesystem.
            if let Some(reload) =
                Self::ralph_reload_context(self.config.storage.run(), &self.config.project_id).await
            {
                self.config
                    .context_manager
                    .append_user_message(&mut session_state, &reload)
                    .await;
            }
            // R3 (issue #58 v2): inject git history when a `VcsProvider` is wired.
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
            session_state
        })
    }

    fn ralph_completion_status<'a>(&'a self) -> BoxFut<'a, Option<String>> {
        Box::pin(async move {
            // #142: read the checkpoint from the durable project-id RunStore.
            Self::ralph_completion_status(self.config.storage.run(), &self.config.project_id).await
        })
    }

    fn ralph_max_resets(&self) -> u32 {
        self.config.max_resets
    }

    fn escalation_mode(&self) -> EscalationMode {
        // No longer `Copy` (SC-5's `AutoContinue` holds an `Arc` callback); clone
        // out — the callback is refcounted, so this is cheap.
        self.config.escalation_mode.clone()
    }

    fn enforce_output_schemas(&self) -> bool {
        self.config.enforce_output_schemas
    }

    fn output_schema_max_retries(&self) -> u32 {
        self.config.output_schema_max_retries
    }

    fn resolve_metric_evaluator(
        &self,
        key: &str,
        session_id: &SessionId,
    ) -> Result<Arc<dyn crate::metric::MetricEvaluator>, Box<RunResult>> {
        self.config
            .registry
            .resolve_metric_evaluator(key)
            .cloned()
            .ok_or_else(|| {
                Box::new(RunResult::Failure {
                    reason: HaltReason::HillClimbingMisconfigured {
                        reason: format!(
                            "HillClimbing requires a metric evaluator registered under key {key:?}"
                        ),
                    },
                    session_id: session_id.clone(),
                    usage: AggregateUsage::default(),
                    turns: 0,
                    session_state: SessionState::default(),
                })
            })
    }

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
    ) -> BoxFut<'a, Result<(f64, Option<String>), RunResult>> {
        Box::pin(async move {
            use crate::metric::{IterationStatus, MetricResult, ResultsEntry};
            use crate::termination::SessionStateSnapshot;
            let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
            let snapshot = SessionStateSnapshot::new(
                session_id.clone(),
                task_id.clone(),
                SessionState::default(),
                workspace_root.clone(),
            );
            let baseline = evaluator
                .evaluate(self.config.sandbox.as_ref(), &snapshot)
                .await;
            match baseline {
                Ok(MetricResult {
                    value, duration, ..
                }) => {
                    // D4: commit the starting tree as the baseline candidate so
                    // `current_sha`/reverts are always well-defined (handles a
                    // dirty start). `None` when the workspace has no usable VCS,
                    // in which case the loop degrades to its pre-fix behaviour.
                    let start_sha = self
                        .hill_climbing_commit_accepted(&Self::hill_commit_label(0))
                        .await;
                    rows.push(ResultsEntry {
                        iteration: 0,
                        commit_hash: start_sha.clone(),
                        metric_value: value,
                        direction,
                        status: IterationStatus::Kept,
                        duration,
                        description: evaluator.description(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        session_id,
                        task_id,
                        span_seq,
                        0,
                        Some(value),
                        None,
                        IterationStatus::Kept,
                        false,
                    );
                    Ok((value, start_sha))
                }
                Err(err) => {
                    let status = IterationStatus::from_error(&err);
                    rows.push(ResultsEntry {
                        iteration: 0,
                        // Baseline eval failed ⇒ no baseline candidate to commit.
                        commit_hash: None,
                        metric_value: f64::NAN,
                        direction,
                        status,
                        duration: Duration::ZERO,
                        description: evaluator.description(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        session_id, task_id, span_seq, 0, None, None, status, false,
                    );
                    self.write_hill_climbing_tsv(&workspace_root, task_id, rows)
                        .await;
                    Err(RunResult::Failure {
                        reason: HaltReason::HillClimbingMisconfigured {
                            reason: format!("baseline evaluation failed: {err}"),
                        },
                        session_id: session_id.clone(),
                        usage: total_usage,
                        turns,
                        session_state: SessionState::default(),
                    })
                }
            }
        })
    }

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
    ) -> BoxFut<'a, bool> {
        Box::pin(async move {
            use crate::metric::{should_keep, IterationStatus, MetricResult, ResultsEntry};
            use crate::termination::SessionStateSnapshot;
            let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
            let snapshot = SessionStateSnapshot::new(
                session_id.clone(),
                task_id.clone(),
                SessionState::default(),
                workspace_root,
            );
            let eval = evaluator
                .evaluate(self.config.sandbox.as_ref(), &snapshot)
                .await;
            match eval {
                Ok(MetricResult {
                    value, duration, ..
                }) => {
                    let kept =
                        should_keep(value, cand.current_best, direction, min_improvement_delta);
                    let delta = match direction {
                        HillClimbingDirection::Minimize => cand.current_best - value,
                        HillClimbingDirection::Maximize => value - cand.current_best,
                    };
                    if kept {
                        // Commit the accepted neighbor → durable pointer (§4.2).
                        // `None` ⇒ no usable VCS: the keep still advances the
                        // score high-water mark (pre-fix behaviour), but there is
                        // no SHA to revert a later discard to.
                        let nbr_sha = self
                            .hill_climbing_commit_accepted(&Self::hill_commit_label(iteration))
                            .await;
                        cand.current_sha = nbr_sha.clone();
                        cand.current_best = value;
                        // Update the best-ever pointer when this strictly beats it
                        // (D2: tracked separately even though equal under greedy).
                        let beats_best = match direction {
                            HillClimbingDirection::Minimize => value < cand.best_score,
                            HillClimbingDirection::Maximize => value > cand.best_score,
                        };
                        if beats_best {
                            cand.best_sha = nbr_sha.clone();
                            cand.best_score = value;
                        }
                        rows.push(ResultsEntry {
                            iteration,
                            commit_hash: nbr_sha,
                            metric_value: value,
                            direction,
                            status: IterationStatus::Kept,
                            duration,
                            description: evaluator.description(),
                            metadata: Default::default(),
                        });
                        self.emit_hill_climbing_span(
                            session_id,
                            task_id,
                            span_seq,
                            iteration,
                            Some(value),
                            Some(delta),
                            IterationStatus::Kept,
                            false,
                        );
                        false
                    } else {
                        // Revert to the last ACCEPTED state (§4.3), NOT HEAD.
                        let reverted = if revert_on_no_improvement {
                            self.hill_climbing_reset_to(cand.current_sha.as_deref())
                                .await;
                            true
                        } else {
                            false
                        };
                        rows.push(ResultsEntry {
                            iteration,
                            // The discard row records the surviving accepted SHA
                            // (== HEAD after the reset; == HEAD anyway when revert
                            // is off, since uncommitted neighbor edits don't move
                            // HEAD).
                            commit_hash: cand.current_sha.clone(),
                            metric_value: value,
                            direction,
                            status: IterationStatus::Discarded,
                            duration,
                            description: evaluator.description(),
                            metadata: Default::default(),
                        });
                        self.emit_hill_climbing_span(
                            session_id,
                            task_id,
                            span_seq,
                            iteration,
                            Some(value),
                            Some(delta),
                            IterationStatus::Discarded,
                            reverted,
                        );
                        true
                    }
                }
                Err(err) => {
                    let status = IterationStatus::from_error(&err);
                    let reverted = if revert_on_no_improvement {
                        self.hill_climbing_reset_to(cand.current_sha.as_deref())
                            .await;
                        true
                    } else {
                        false
                    };
                    rows.push(ResultsEntry {
                        iteration,
                        commit_hash: cand.current_sha.clone(),
                        metric_value: f64::NAN,
                        direction,
                        status,
                        duration: Duration::ZERO,
                        description: evaluator.description(),
                        metadata: Default::default(),
                    });
                    self.emit_hill_climbing_span(
                        session_id, task_id, span_seq, iteration, None, None, status, reverted,
                    );
                    true
                }
            }
        })
    }

    fn hill_apply_winner<'a>(&'a self, best_sha: Option<&'a str>) -> BoxFut<'a, ()> {
        Box::pin(async move {
            // Winner application (§4.6): reset to the best accepted candidate
            // before a terminal return. No-op when there is no committed best.
            if let Some(sha) = best_sha.filter(|s| !s.is_empty()) {
                self.hill_climbing_reset_to(Some(sha)).await;
            }
        })
    }

    fn hill_write_tsv<'a>(
        &'a self,
        task_id: &'a TaskId,
        rows: &'a [crate::metric::ResultsEntry],
    ) -> BoxFut<'a, ()> {
        Box::pin(async move {
            let workspace_root = self.config.sandbox.workspace_root().to_path_buf();
            self.write_hill_climbing_tsv(&workspace_root, task_id, rows)
                .await;
        })
    }

    fn finalize<'a>(&'a self, result: &'a RunResult) -> BoxFut<'a, ()> {
        Box::pin(async move {
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
                RunResult::Escalate { session_id, .. } => {
                    self.finalize_observability(session_id, SessionOutcome::Escalated)
                        .await;
                }
                // Non-terminal pauses are never finalized.
                RunResult::WaitingForHuman { .. } | RunResult::Consult { .. } => {}
            }
        })
    }
}

impl StandardHarness {
    /// Consult resume seam (issue #114). Mirrors the `resume_inner`
    /// clarification branch: the [`ConsultResponse`] text is injected as the
    /// tool RESULT for the head pending (consult) call — NOT appended as a
    /// free-standing user message (R10) — then any remaining pending calls are
    /// dispatched and the ReAct loop resumes.
    async fn resume_consult_inner(
        &self,
        state: PausedState,
        response: ConsultResponse,
        on_stream: Option<StreamSink>,
    ) -> RunResult {
        let PausedState {
            session_id,
            task_id: _,
            turn_number: _,
            mut session_state,
            pending_tool_calls,
            approved_results: _,
            human_request: _,
            task,
            budget_used,
            child_state: _,
            // #140: the pausing leaf's toolset handle, carried through the pause.
            toolset,
        } = state;

        let (text, answered) = match &response {
            ConsultResponse::Answer { text } => (text.clone(), true),
            ConsultResponse::BudgetExhausted { message } => (message.clone(), false),
        };

        // Observability: lightweight consult-resume event.
        if let Some(obs) = self.config.observability.as_ref() {
            // Recover the consult `kind` from the head pending call's args, if
            // present, else fall back to a generic label.
            let kind = pending_tool_calls
                .first()
                .and_then(|c| c.input.get("kind"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let base = SpanBase::new_root(
                SpanId::new(format!("{}-consult-resume", session_id.as_str())),
                session_id.clone(),
                task.id.clone(),
                SpanKind::ContextAssembly,
                Timestamp::now(),
            );
            obs.emit_context(ContextSpan {
                base,
                operation: ContextOperation::ConsultResumed {
                    consult_kind: kind,
                    answered,
                },
                tokens_before: 0,
                tokens_after: 0,
                utilization_before: 0.0,
                utilization_after: 0.0,
            });
        }

        // #140: resume routes the preserved consulting call (and any remaining
        // pending calls) through the pausing leaf's own toolset handle, restoring
        // its scoped per-node catalogue. An empty handle (the default) still falls
        // back to the global catalogue, so pre-#140 blobs behave unchanged.
        let tool_registry = self.effective_tool_registry(&session_id, &toolset);

        // Inject the consult answer as the RESULT of the head pending (consult)
        // call, then dispatch the remaining pending calls in the same batch.
        let mut pending = pending_tool_calls.into_iter();
        if let Some(consult_call) = pending.next() {
            let tr = ToolResult {
                call_id: consult_call.id.clone(),
                output: ToolOutput::Success {
                    content: text,
                    truncated: false,
                },
            };
            self.config
                .context_manager
                .append_tool_result(&mut session_state, &tr)
                .await;
        }
        for call in pending {
            let output =
                self.normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
            let tr = ToolResult {
                call_id: call.id,
                output,
            };
            self.config
                .context_manager
                .append_tool_result(&mut session_state, &tr)
                .await;
        }

        // #131: a consult that surfaced from inside a composed tree carries the
        // FULL strategy in `task.loop_strategy` (each combinator's `finish`
        // rewrote the pause's task on the way up). Re-DRIVE that strategy rather
        // than resuming only the worker leaf: the PlanExecute walk resumes its
        // in-progress task from the injected worker session (`consult_resume`
        // seed), so the worker finishes mid-loop, its SelfVerifying evaluator runs,
        // the task is marked Completed, and the remaining ready-set is walked. A
        // BARE worker leaf (depth-1, e.g. a SubagentTool-mediated consult) has no
        // surrounding walk, so it keeps the original leaf-only resume (back-compat).
        if !matches!(task.loop_strategy, LoopStrategy::ReAct(_)) {
            return self
                .drive_strategy_with_resume_seed(
                    task,
                    // Top-level session starts fresh; the worker conversation is
                    // threaded into the in-progress task via the consult seed.
                    SessionState::default(),
                    budget_used,
                    on_stream,
                    None,
                    Some(session_state),
                    // resume_seed carries the continuation signal.
                    false,
                    None,
                )
                .await;
        }

        let max_iterations = match &task.loop_strategy {
            LoopStrategy::ReAct(c) => c.max_iterations(),
            _ => u32::MAX,
        };
        let agent = match self.resolve_worker_agent(&task.loop_strategy) {
            Ok(a) => a,
            Err(error) => {
                return RunResult::Failure {
                    reason: HaltReason::ConfigurationError { error },
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: budget_used.turns,
                    session_state,
                };
            }
        };
        self.run_react(
            task,
            max_iterations,
            session_state,
            budget_used,
            on_stream,
            agent,
            // legacy wrapper: empty handle ⇒ global-catalogue fallback.
            ToolsetRef(String::new()),
        )
        .await
    }

    /// Issue #102 auto-persist seam: write the terminal run state to the
    /// [`SessionStore`](crate::storage::SessionStore) when
    /// [`auto_persist_sessions`](HarnessConfig::auto_persist_sessions) is enabled.
    ///
    /// One write per `run()`/`resume()`, at the same terminal seam as the
    /// observability flush. For `Success`/`Failure` a [`PausedState`] is
    /// synthesized carrying the final `session_state` with empty pending fields
    /// (D4); for `WaitingForHuman`/`Escalate` the carried `PausedState` is
    /// persisted (D6 — the cross-process pause case). Storage errors are
    /// swallowed-and-logged (D8): a put failure must never lose the run nor
    /// surface as a [`HaltReason`].
    ///
    /// When disabled (the default), this returns immediately WITHOUT touching the
    /// store — the off-by-default zero-I/O contract (#102).
    async fn auto_persist_terminal(&self, result: &RunResult) {
        if !self.config.auto_persist_sessions {
            return;
        }
        // Build the `(SessionId, PausedState)` to persist for this terminal.
        let (session_id, state) = match result {
            RunResult::Success {
                session_id,
                session_state,
                turns,
                ..
            }
            | RunResult::Failure {
                session_id,
                session_state,
                turns,
                ..
            } => {
                // Synthesize a completed-run PausedState: empty pending fields,
                // no human request, no child — it carries only the final history
                // so a later `get_session(..).session_state` resumes losslessly.
                let state = PausedState {
                    session_id: session_id.clone(),
                    task_id: TaskId::new(session_id.as_str()),
                    turn_number: *turns,
                    session_state: session_state.clone(),
                    pending_tool_calls: vec![],
                    approved_results: vec![],
                    human_request: None,
                    task: Task::new(
                        String::new(),
                        session_id.clone(),
                        LoopStrategy::ReAct(ReactConfig::per_loop(0)),
                    ),
                    budget_used: BudgetSnapshot::default(),
                    child_state: None,
                    // #140: a synthesized completed-run state has empty pending
                    // fields; the handle is behaviorally irrelevant here.
                    toolset: ToolsetRef::default(),
                };
                (session_id.clone(), state)
            }
            // Persist the carried pause state directly (D6).
            RunResult::WaitingForHuman { state, .. } => {
                (state.session_id.clone(), (**state).clone())
            }
            RunResult::Escalate {
                state, session_id, ..
            } => (session_id.clone(), (**state).clone()),
            // Consult (issue #114) is non-terminal — persist the carried pause
            // state directly (same as WaitingForHuman) so a cross-process host
            // can later `resume_consult` it.
            RunResult::Consult {
                state, session_id, ..
            } => (session_id.clone(), (**state).clone()),
        };
        // Swallow-and-log on Err (D8): a storage hiccup must not lose the run.
        let _ = self
            .config
            .storage
            .session()
            .put_session(&session_id, &state)
            .await;
    }

    async fn run_inner(&self, options: HarnessRunOptions) -> RunResult {
        let HarnessRunOptions {
            task,
            on_stream,
            session_state,
            continue_task_list,
        } = options;

        // #124 startup validation: every serializable handle in the task's
        // strategy tree must resolve against the configured ExecutionRegistry,
        // BEFORE the first turn. The legacy single-collaborator fields are gone —
        // resolution is now the SINGLE path, so validation ALWAYS runs (the
        // `!is_empty()` gate is dropped). An unresolved handle is a startup error.
        if let Err(error) = self.config.registry.validate(&task) {
            return RunResult::Failure {
                reason: HaltReason::ConfigurationError { error },
                session_id: task.session_id.clone(),
                usage: AggregateUsage::default(),
                turns: 0,
                session_state: SessionState::default(),
            };
        }

        // Issue #102 auto-load: when enabled AND no explicit session_state was
        // provided AND the strategy seeds incoming state (ReAct / SelfVerifying —
        // Ralph/HillClimbing discard it by design, D7), load the prior session for
        // this `session_id` from the SessionStore so a caller can resume by id.
        // Explicit `with_session_state` always wins (D5). Errors are
        // swallowed-and-logged: a load failure starts fresh (D8).
        let session_state = match session_state {
            Some(explicit) => Some(explicit),
            None if self.config.auto_persist_sessions
                && matches!(
                    task.loop_strategy,
                    LoopStrategy::ReAct(_) | LoopStrategy::SelfVerifying(_)
                ) =>
            {
                match self
                    .config
                    .storage
                    .session()
                    .get_session(&task.session_id)
                    .await
                {
                    Ok(Some(prior)) => Some(prior.session_state),
                    Ok(None) => None,
                    Err(_e) => {
                        // Swallow-and-log: start fresh on a load failure (D8).
                        // (No logging facade is wired into spore-core; the error
                        // is intentionally dropped rather than surfaced as a
                        // HaltReason.)
                        None
                    }
                }
            }
            None => None,
        };

        let mut session_state = session_state.unwrap_or_default();
        let budget_used = BudgetSnapshot::default();

        // #124: the central dispatch `match` is GONE — the only `match` left is
        // the enum→config delegation inside `LoopStrategy::run`. The harness entry
        // collapses to `task.loop_strategy.run(cx).await` via the recursive
        // executor. Strategies that BUILD ON incoming state (ReAct / PlanExecute /
        // SelfVerifying) get the instruction seeded here on the FRESH run (the
        // compaction adapter ignores `task`, so the harness owns prompt delivery);
        // Ralph / HillClimbing re-seed a fresh window internally (D7), so their
        // incoming state is discarded by the config body.
        if matches!(
            task.loop_strategy,
            LoopStrategy::ReAct(_) | LoopStrategy::PlanExecute(_) | LoopStrategy::SelfVerifying(_)
        ) {
            self.config
                .context_manager
                .append_user_message(&mut session_state, &task.instruction)
                .await;
        }
        self.drive_strategy_with_resume_seed(
            task,
            session_state,
            budget_used,
            on_stream,
            None,
            None,
            // Model (A): a fresh `run()` plans fresh unless the caller explicitly
            // opts in to continuing a prior run's durable task list.
            continue_task_list,
            None,
        )
        .await
    }

    /// The recursive-executor entry (#124): build the shared [`ExecutionContext`],
    /// seed the per-run scratch (task / session / budget), drive
    /// `task.loop_strategy.run(cx)`, and translate the outcome back into a
    /// terminal [`RunResult`] (Q5). A non-terminal pause / escalate stashed in
    /// `scratch.terminal_override` propagates VERBATIM (it never collapses into a
    /// `StrategyOutcome`).
    /// Drive a task's strategy through the recursive executor, with an optional
    /// cross-process Continue checkpoint seed
    /// (#129, AC2): `resume_continues = Some((phase, continues_used))` seeds the
    /// FIRST matching budget scope's `continues_used` (via [`BudgetContext::resumed`])
    /// so a `Continue` that spanned a process pause resumes with the correct
    /// continue count instead of a zeroed one. `None` is the fresh-run path.
    ///
    /// `resume_seed = Some(session)` (#131 consult + #138 budget) seeds the FIRST
    /// PlanExecute walk from `session` (the stalled worker conversation). For a
    /// consult resume the consult answer is already injected; for a budget resume
    /// (#138) it is the worker's full post-exhaustion session. The walk re-attaches
    /// it to the single `InProgress` task (execute-phase exhaustion) or, when the
    /// durable task_list has no `InProgress` task, to the PLAN session (#138 AC3
    /// plan-phase exhaustion). `None` on every fresh / non-resume path.
    // The resume/continuation surface is genuinely multi-axis (session, budget,
    // stream, continue-checkpoint, resume-seed, continue-task-list); grouping it
    // into a struct would not improve clarity at the handful of call sites.
    #[allow(clippy::too_many_arguments)]
    async fn drive_strategy_with_resume_seed(
        &self,
        task: Task,
        session_state: SessionState,
        budget_used: BudgetSnapshot,
        on_stream: Option<StreamSink>,
        resume_continues: Option<(String, u32)>,
        resume_seed: Option<SessionState>,
        // Model (A): an explicit "continue the durable task list" signal for a
        // SEEDLESS PlanExecute continuation (Ralph window re-entry / `Skip`
        // resolution). The seed-carrying resume paths pass `false` — a
        // `resume_seed` already implies continuation; a fresh run passes `false`.
        continue_task_list: bool,
        // #139: a one-shot FORCED output schema for the FIRST round's leaf (the
        // plan-empty repair re-ask, threaded by `run_plan_subtree`). Consumed by
        // the first round only — mirrors `resume_seed`/`resume_continues`. `None`
        // on every other drive path (byte-identical to before).
        forced_output_schema: Option<serde_json::Value>,
    ) -> RunResult {
        // #129: the BARE-LEAF resolution site is `drive_strategy` (a bare leaf
        // never self-resolves inside its own body — rule 6 — it PROPAGATES a typed
        // `BudgetExhausted` here, the single recovery site for a top-level leaf).
        // When the leaf's CONFIGURED `behavior` resolves to `Continue`, grant the
        // leaf one more round IN-PROCESS (bump `continues_used`, refresh the
        // step cap) and re-drive WITHOUT any serialization (AC3) — looping until
        // the behavior resolves to `Fail`/`Escalate` or the strategy completes.
        // `behavior_for_resolution` carries the resolution chain's mutated state
        // across in-process continues so `max_continues` is honored.
        let session_id = task.session_id.clone();
        let mut task = task;
        let mut session_state = session_state;
        let mut budget_used = budget_used;
        // #stream: read-only now — cloned into `cx.stream` each round, never moved.
        let on_stream = on_stream;
        // The Continue resolution state threaded across in-process rounds: the
        // (possibly fallen-through) behavior + how many continues have been spent.
        let mut behavior_for_resolution: Option<(BudgetExhaustedBehavior, u32)> = None;
        // #129 (AC2): the cross-process checkpoint seed is consumed by the FIRST
        // round only (the resumed node's scope); later in-process rounds carry
        // their continue count via `behavior_for_resolution`.
        let mut resume_continues = resume_continues;
        // #131/#138: the resume seed is consumed by the FIRST round's
        // PlanExecute walk only; later in-process rounds run normally.
        let mut resume_seed = resume_seed;
        // Model (A): the seedless continue signal is likewise consumed by the
        // FIRST round (an in-process `Continue` re-drive is a bare-leaf ReAct and
        // never reads it, so this is belt-and-suspenders against re-asserting it).
        let mut continue_task_list = continue_task_list;
        // #139: the forced output schema is consumed by the FIRST round's leaf
        // only (the plan-empty repair re-ask); later in-process continue rounds
        // run without it.
        let mut forced_output_schema = forced_output_schema;
        // SC-5: auto-grants spent at the BARE-LEAF Escalate site under
        // `EscalationMode::AutoContinue`. The scope is rebuilt each round here
        // (unlike the in-loop combinator scopes, which carry their own
        // `auto_grants_used`), so the counter lives across rounds in this local.
        let mut auto_grants_used: u32 = 0;

        loop {
            let mut cx = ExecutionContext::new(&self.config.registry);
            cx.executor = Some(self);
            // #stream: CLONE (not take) the Arc sink so the binding survives every
            // in-process re-drive round (a `Continue` rebuilds `cx` at the top of
            // this loop and re-installs the same sink). Nested nodes wrap it.
            cx.stream = on_stream.clone();
            cx.scratch.run_session = std::mem::take(&mut session_state);
            cx.scratch.run_budget = budget_used.clone();
            cx.scratch.task = Some(task.clone());
            cx.scratch.resume_continues = resume_continues.take();
            cx.scratch.resume_seed = resume_seed.take();
            cx.scratch.continue_task_list = std::mem::take(&mut continue_task_list);
            // #139: one-shot forced schema for the first round's leaf (the
            // plan-empty repair re-ask); `take()` so continue rounds run without it.
            cx.scratch.forced_output_schema = forced_output_schema.take();

            let outcome = task.loop_strategy.run(&mut cx).await;

            // A pause / escalate propagates verbatim (preserves the HITL / consult /
            // escalation contract through the recursive executor).
            if let Some(terminal) = cx.scratch.terminal_override.take() {
                return terminal;
            }
            match outcome {
                StrategyOutcome::Complete(output) => {
                    return RunResult::Success {
                        output,
                        session_id,
                        usage: cx.usage.clone(),
                        turns: cx.scratch.run_budget.turns,
                        session_state: std::mem::take(&mut cx.scratch.run_session),
                    }
                }
                StrategyOutcome::Failed(error) => {
                    return RunResult::Failure {
                        reason: HaltReason::ConfigurationError { error },
                        session_id,
                        usage: cx.usage.clone(),
                        turns: cx.scratch.run_budget.turns,
                        session_state: std::mem::take(&mut cx.scratch.run_session),
                    }
                }
                StrategyOutcome::BudgetExhausted {
                    policy,
                    behavior,
                    steps_taken,
                    continues_used,
                    phase,
                    partial_output,
                    cause,
                } => {
                    // #129: a BARE LEAF exhaustion propagates here carrying its
                    // CONFIGURED `behavior` (Q1 — the bare-leaf resolution site
                    // honors it; the leaf body never did). Resolve it ONCE: a
                    // `Continue` with continues left re-drives in-process; a spent
                    // `Continue` falls through to `Fail`/`Escalate`. Carry the
                    // resolution chain's state across rounds so `max_continues` is
                    // respected. (A combinator under `SurfaceToHuman` never reaches
                    // this arm — it sets `terminal_override`, returned above.)
                    let turns = if steps_taken > 0 {
                        steps_taken
                    } else {
                        cx.scratch.run_budget.turns
                    };
                    // #137: the terminal `HaltReason` for a Fail / Escalate→Fail
                    // resolution depends on the CAUSE — a genuine budget
                    // exhaustion reports `BudgetExceeded`; an error-loop break
                    // reports `ToolErrorLoop`. (A granted `Continue` re-drives
                    // the window, whose loop-local error counter starts fresh —
                    // parallel to how `consume_continue` resets `steps_taken`.)
                    let terminal_reason = match &cause {
                        ExhaustionCause::Budget => HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        ExhaustionCause::ToolErrorLoop {
                            tool,
                            consecutive_errors,
                        } => HaltReason::ToolErrorLoop {
                            tool: tool.clone(),
                            consecutive_errors: *consecutive_errors,
                        },
                    };
                    // Reconstruct the resolution scope: the FIRST round uses the
                    // leaf's propagated behavior + continues_used; later in-process
                    // rounds reuse the threaded (possibly fallen-through) state.
                    let (resolve_behavior, resolve_continues) = behavior_for_resolution
                        .take()
                        .unwrap_or((behavior, continues_used));
                    let mut scope = BudgetContext::resumed(
                        policy.clone(),
                        resolve_behavior,
                        phase.clone(),
                        resolve_continues,
                    );
                    let resolution = scope.resolve_exhausted();
                    match resolution {
                        ExhaustedResolution::Continue => {
                            // In-process continue (AC3: NO serialization). Refresh
                            // the leaf's step cap and re-enter the loop carrying the
                            // post-run session so the conversation context survives
                            // (Continue PRESERVES context — AC4). The granted cap is
                            // `steps_taken + policy.value` so the leaf gets a fresh
                            // window after the checkpoint.
                            let granted =
                                steps_taken.saturating_add(policy.allowance_value().unwrap_or(1));
                            grant_task_budget(&mut task, granted);
                            session_state = std::mem::take(&mut cx.scratch.run_session);
                            budget_used = cx.scratch.run_budget.clone();
                            // #stream: no reclaim needed — `on_stream` (Arc) is
                            // cloned, never moved, into `cx.stream` each round.
                            // Thread the resolution chain's post-continue state so a
                            // subsequent exhaustion sees the bumped `continues_used`.
                            behavior_for_resolution =
                                Some((scope.behavior.clone(), scope.continues_used));
                            continue;
                        }
                        ExhaustedResolution::Escalate => {
                            // SC-5: AutoContinue auto-grants `steps_per_grant` more
                            // steps and re-drives IN-PROCESS (mirroring the
                            // `Continue` arm above), up to `max_grants` times,
                            // firing `on_grant` per grant. This is the bare-leaf /
                            // top-level keep-working-but-capped site — where
                            // consumers otherwise hand-roll a drive loop. Once the
                            // grants are spent it falls through to the
                            // surface/abort handling below (the `Autonomous`
                            // terminal).
                            if let EscalationMode::AutoContinue {
                                max_grants,
                                steps_per_grant,
                                on_grant,
                            } = &self.config.escalation_mode
                            {
                                if auto_grants_used < *max_grants {
                                    auto_grants_used += 1;
                                    if let Some(cb) = on_grant {
                                        cb(AutoGrantInfo {
                                            grant_number: auto_grants_used,
                                            steps_granted: *steps_per_grant,
                                            phase: phase.clone(),
                                        });
                                    }
                                    let granted = steps_taken.saturating_add(*steps_per_grant);
                                    grant_task_budget(&mut task, granted);
                                    session_state = std::mem::take(&mut cx.scratch.run_session);
                                    budget_used = cx.scratch.run_budget.clone();
                                    continue;
                                }
                            }
                            if matches!(self.config.escalation_mode, EscalationMode::SurfaceToHuman)
                            {
                                let err = BudgetExhausted {
                                    policy,
                                    behavior: scope.behavior.clone(),
                                    steps_taken,
                                    continues_used: scope.continues_used,
                                    phase,
                                };
                                let actions = leaf_escalation_actions(&err);
                                // #138 AC2-a: carry the FULL stalled leaf session
                                // (it propagated into scratch with its conversation)
                                // and the leaf's own toolset handle (#140/AC4-a).
                                let worker_session = std::mem::take(&mut cx.scratch.run_session);
                                let worker_toolset = worker_toolset_of(&task.loop_strategy);
                                return promote_budget_exhausted_to_human(
                                    &err,
                                    partial_output,
                                    actions,
                                    session_id,
                                    task,
                                    cx.scratch.run_budget.clone(),
                                    turns,
                                    worker_session,
                                    worker_toolset,
                                );
                            }
                            return RunResult::Failure {
                                // #137: ToolErrorLoop vs BudgetExceeded per cause.
                                reason: terminal_reason,
                                session_id,
                                usage: cx.usage.clone(),
                                turns,
                                session_state: SessionState {
                                    messages: partial_output
                                        .map(|t| {
                                            vec![Message {
                                                role: crate::model::Role::Assistant,
                                                content: crate::model::Content::Text { text: t },
                                            }]
                                        })
                                        .unwrap_or_default(),
                                    ..Default::default()
                                },
                            };
                        }
                        ExhaustedResolution::Fail => {
                            // Fail contract: the partial is DISCARDED.
                            return RunResult::Failure {
                                // #137: ToolErrorLoop vs BudgetExceeded per cause.
                                reason: terminal_reason,
                                session_id,
                                usage: cx.usage.clone(),
                                turns,
                                session_state: SessionState::default(),
                            };
                        }
                    }
                }
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
            // #140: the pausing leaf's toolset handle, carried through the pause.
            toolset,
        } = state;

        // Budget-escalation resume (#130/#129): if this pause came from
        // `HumanRequest::BudgetExhausted`, map the operator's `EscalationAction`
        // BEFORE the generic `match response` (mirrors the Clarification branch).
        // The node's `BudgetContext` is reconstructed from the request fields:
        // `steps_taken` raises the granted cap, and `continues_used` (#129, AC2)
        // is the SOLE `BudgetContext` field that rides a process pause — it is
        // seeded back into the rebuilt scope so a `Continue` spanning the pause
        // resumes with the correct continue count (it cannot exceed
        // `max_continues`). Q3: `continues_used` rides the REQUEST payload, NOT a
        // new serialized `BudgetContext`/`PausedState` field.
        if let Some(HumanRequest::BudgetExhausted {
            steps_taken,
            continues_used,
            phase,
            ..
        }) = &hr
        {
            let steps_taken = *steps_taken;
            let resume_continues = (phase.clone(), *continues_used);
            match &response {
                // Grant `steps` ADDITIONAL allowance and re-enter the loop from the
                // restored checkpoint. The strategy tree is rebuilt with the node's
                // budget policy raised to `steps_taken + steps` so the restored
                // scope has room for `steps` more steps, and the resumed scope's
                // `continues_used` is seeded from the request (#129, AC2).
                HumanResponse::Escalate {
                    action: EscalationAction::ContinueWithBudget { steps },
                } => {
                    let granted = steps_taken.saturating_add(*steps);
                    let mut resumed_task = task.clone();
                    grant_task_budget(&mut resumed_task, granted);
                    // #138 AC2-b: for a COMPOSED task (PlanExecute etc.) thread the
                    // carried worker session through the phase-agnostic resume seed
                    // and start the TOP-LEVEL session EMPTY — exactly mirroring
                    // `resume_consult_inner`. The PlanExecute walk re-attaches the
                    // seed to the single `InProgress` task (execute-phase
                    // exhaustion) via the InProgress→Pending→complete machinery, or
                    // to the PLAN session (AC3 plan-phase exhaustion) — so the
                    // stalled worker continues mid-loop instead of re-planning. A
                    // BARE leaf has no surrounding walk, so it resumes from
                    // `session_state` directly (the seed has nowhere to attach).
                    let (top_session, resume_seed) =
                        if matches!(resumed_task.loop_strategy, LoopStrategy::ReAct(_)) {
                            (session_state, None)
                        } else {
                            (SessionState::default(), Some(session_state))
                        };
                    return self
                        .drive_strategy_with_resume_seed(
                            resumed_task,
                            top_session,
                            budget_used,
                            on_stream,
                            Some(resume_continues),
                            resume_seed,
                            // resume_seed (when present) carries the continuation
                            // signal; a bare ReAct leaf needs none.
                            false,
                            None,
                        )
                        .await;
                }
                // Skip: the node is marked skipped and the outer loop advances. For
                // a combinator (PlanExecute) the per-task partial already recorded
                // the blocked task and persisted the list; re-entering the loop from
                // the checkpoint advances to the remaining ready tasks. For a leaf
                // there is no sibling, so a skip resolves to a clean (empty)
                // Success carrying whatever partial history was captured.
                HumanResponse::Escalate {
                    action: EscalationAction::Skip,
                } => {
                    if matches!(task.loop_strategy, LoopStrategy::PlanExecute(_)) {
                        // Model (A): a Skip carries NO resume_seed (the task is
                        // abandoned, not continued), so it must explicitly SIGNAL
                        // continuation — otherwise PlanExecute would now treat the
                        // re-entry as a fresh run, clear the durable list, and
                        // re-plan instead of walking past the skipped task to the
                        // remaining ready ones.
                        return self
                            .drive_strategy_with_resume_seed(
                                task,
                                session_state,
                                budget_used,
                                on_stream,
                                None,
                                None,
                                true,
                                None,
                            )
                            .await;
                    }
                    return RunResult::Success {
                        output: String::new(),
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: turn_number,
                        session_state,
                    };
                }
                // Fail: abort the node and propagate `BudgetExceeded`; the partial
                // is discarded (the `Fail` resolution contract).
                HumanResponse::Escalate {
                    action: EscalationAction::Fail,
                } => {
                    return RunResult::Failure {
                        reason: HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: turn_number,
                        session_state: SessionState::default(),
                    };
                }
                // A non-`Escalate` response to a budget pause is out of contract;
                // treat it conservatively as a `Fail` (abort + propagate).
                _ => {
                    return RunResult::Failure {
                        reason: HaltReason::BudgetExceeded {
                            limit_type: BudgetLimitType::Turns,
                        },
                        session_id,
                        usage: AggregateUsage::default(),
                        turns: turn_number,
                        session_state: SessionState::default(),
                    };
                }
            }
        }

        // Resolve the effective tool registry for this resumed session — bridges
        // catalogue tools the same way the turn loop does, so pending tool calls
        // dispatched during resume thread the run's storage + sandbox. #140:
        // resume now routes through the pausing leaf's own toolset handle,
        // restoring its scoped per-node catalogue. An empty handle (the default)
        // still falls back to the global catalogue, so pre-#140 blobs and root
        // pauses behave unchanged. NOTE: the budget-escalation branch above
        // returned early via `drive_strategy_with_resume_seed`, which re-resolves
        // per-leaf toolsets during the re-drive — so this line is only reached by
        // the Clarification / direct-resume paths whose pending calls need the
        // carried handle.
        let tool_registry = self.effective_tool_registry(&session_id, &toolset);

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
                    let output = self
                        .normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
                    let tr = ToolResult {
                        call_id: call.id,
                        output,
                    };
                    self.config
                        .context_manager
                        .append_tool_result(&mut session_state, &tr)
                        .await;
                }
                // SC-BUG-1: a clarification that surfaced from inside a composed
                // tree carries the FULL strategy in `task.loop_strategy` (each
                // combinator's `finish` rewrote the pause's task on the way up).
                // Re-DRIVE that strategy from the answered worker session — exactly
                // as the consult resume does — so the SelfVerifying evaluate phase /
                // PlanExecute walk runs instead of only the bare worker leaf. A
                // BARE ReAct leaf has no surrounding frame, so it keeps the
                // leaf-only resume below (back-compat).
                if !matches!(task.loop_strategy, LoopStrategy::ReAct(_)) {
                    return self
                        .drive_strategy_with_resume_seed(
                            task,
                            // Top-level session starts fresh; the answered worker
                            // conversation threads in as the resume seed.
                            SessionState::default(),
                            budget_used,
                            on_stream,
                            None,
                            Some(session_state),
                            // resume_seed carries the continuation signal.
                            false,
                            None,
                        )
                        .await;
                }
                let max_iterations = match &task.loop_strategy {
                    LoopStrategy::ReAct(c) => c.max_iterations(),
                    _ => u32::MAX,
                };
                let agent = match self.resolve_worker_agent(&task.loop_strategy) {
                    Ok(a) => a,
                    Err(error) => {
                        return RunResult::Failure {
                            reason: HaltReason::ConfigurationError { error },
                            session_id,
                            usage: AggregateUsage::default(),
                            turns: budget_used.turns,
                            session_state,
                        };
                    }
                };
                return self
                    .run_react(
                        task,
                        max_iterations,
                        session_state,
                        budget_used,
                        on_stream,
                        agent,
                        // legacy wrapper: empty handle ⇒ global-catalogue fallback.
                        ToolsetRef(String::new()),
                    )
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
                    session_state,
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
                // Mirror the turn-loop's stream events (ToolCall before dispatch,
                // ToolResult after) so an approved tool's execution and output
                // render in the UI sink. Without these emits the resume path runs
                // silently, leaving every human-gated tool (bash, writes, gated
                // reads) with an approval line but no visible result.
                for call in pending_tool_calls {
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolCall {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            args: call.input.clone(),
                            node: None,
                        },
                    );
                    let output = self
                        .normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolResult {
                            call_id: call.id.clone(),
                            is_error: matches!(output, ToolOutput::Error { .. }),
                            content: match &output {
                                ToolOutput::Success { content, .. } => content.clone(),
                                ToolOutput::Error { message, .. } => message.clone(),
                                _ => String::new(),
                            },
                            revised: false,
                            node: None,
                        },
                    );
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
                // Same stream-event mirroring as `Allow` (see above) so the
                // modified-and-approved calls render their output in the UI sink.
                for call in calls {
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolCall {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            args: call.input.clone(),
                            node: None,
                        },
                    );
                    let output = self
                        .normalize_sandbox_violation(tool_registry.dispatch(call.clone()).await);
                    Self::emit(
                        &on_stream,
                        StreamEvent::ToolResult {
                            call_id: call.id.clone(),
                            is_error: matches!(output, ToolOutput::Error { .. }),
                            content: match &output {
                                ToolOutput::Success { content, .. } => content.clone(),
                                ToolOutput::Error { message, .. } => message.clone(),
                                _ => String::new(),
                            },
                            revised: false,
                            node: None,
                        },
                    );
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
            // An `Escalate` response delivered to a non-budget pause is out of
            // contract (#130): the budget-escalation resume is handled by the
            // dedicated `HumanRequest::BudgetExhausted` branch ABOVE, which
            // returns early. Reaching here means the operator sent an
            // `EscalationAction` for a `ToolApproval`/`Review`/`None` pause —
            // halt cleanly rather than mis-resuming the loop.
            HumanResponse::Escalate { .. } => {
                return RunResult::Failure {
                    reason: HaltReason::HumanHalted,
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: turn_number,
                    session_state,
                };
            }
        }

        // SC-BUG-1: an Allow / Deny / Answer / Reject resume that surfaced from
        // inside a composed tree carries the FULL strategy in `task.loop_strategy`
        // (each combinator's `finish` rewrote the pause's task on the way up). The
        // human response has already been applied to `session_state` above (the
        // approved calls dispatched, or the deny/answer message appended). Re-DRIVE
        // the whole strategy from that mutated worker session — mirroring the
        // consult resume — so the SelfVerifying evaluate phase (the looper's
        // eval-frame reviewer) / PlanExecute walk re-runs instead of degrading to a
        // plain executor. A BARE ReAct leaf keeps the leaf-only resume below.
        if !matches!(task.loop_strategy, LoopStrategy::ReAct(_)) {
            return self
                .drive_strategy_with_resume_seed(
                    task,
                    SessionState::default(),
                    budget_used,
                    on_stream,
                    None,
                    Some(session_state),
                    // resume_seed carries the continuation signal.
                    false,
                    None,
                )
                .await;
        }

        // Resume the ReAct loop from where we left off.
        let max_iterations = match &task.loop_strategy {
            LoopStrategy::ReAct(c) => c.max_iterations(),
            _ => u32::MAX,
        };
        let agent = match self.resolve_worker_agent(&task.loop_strategy) {
            Ok(a) => a,
            Err(error) => {
                return RunResult::Failure {
                    reason: HaltReason::ConfigurationError { error },
                    session_id,
                    usage: AggregateUsage::default(),
                    turns: budget_used.turns,
                    session_state,
                };
            }
        };
        self.run_react(
            task,
            max_iterations,
            session_state,
            budget_used,
            on_stream,
            agent,
            // legacy wrapper: empty handle ⇒ global-catalogue fallback.
            ToolsetRef(String::new()),
        )
        .await
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
