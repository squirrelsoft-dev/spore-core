//! Budget tracking and exhaustion scaffolding extracted from the harness
//! god-module (Phase 6.1): the budget limit/usage/policy value types plus the
//! per-scope `BudgetContext`, `BudgetStack`, and `SpanStack` machinery. Pure move.

use super::*;
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
// BudgetPolicy + BudgetExhaustedBehavior (issue #117)
// ============================================================================

/// Composable-execution budget vocabulary (PRD Part B). These are pure,
/// serializable value types — no executor wiring. Later slices thread them
/// through the strategy tree. They layer *on top of* [`BudgetLimits`] (the
/// global turns/tokens/wall/cost backstop), which is unchanged.
///
/// ## Types & variants
///
/// [`BudgetPolicy`] — a per-scope step allowance. A **step is one model turn**
/// (matches [`BudgetSnapshot::turns`]). `PerGoal` is intentionally excluded in
/// v1.
///   - `Unlimited` — no per-scope cap.
///   - `TotalSteps { value }` — cap across the whole run.
///   - `PerLoop { value }` — cap per loop iteration.
///   - `PerAttempt { value }` — cap per attempt.
///
/// [`BudgetExhaustedBehavior`] — what to do when a policy's allowance is spent.
/// No node silently defaults to `Continue` (there is deliberately **no**
/// `Default` impl).
///   - `Continue { max_continues, on_exhausted }` — grant up to `max_continues`
///     extra rounds, then fall through to the boxed `on_exhausted` behavior.
///     `max_continues == 0` means immediate fall-through.
///   - `Escalate` — hand off to a parent/escalation path.
///   - `Fail` — terminate with failure.
///
/// ## Serialized forms (internally tagged on `kind`, snake_case)
///   - `{"kind":"unlimited"}`
///   - `{"kind":"total_steps","value":N}`
///   - `{"kind":"per_loop","value":N}`
///   - `{"kind":"per_attempt","value":N}`
///   - `{"kind":"escalate"}`
///   - `{"kind":"fail"}`
///   - `{"kind":"continue","max_continues":N,"on_exhausted":{...nested...}}`
///
/// ## Rules enforced
///   - Data-carrying variants use named struct fields named `value` (serde
///     internally-tagged enums cannot serialize tuple variants).
///   - Integer width is `u32` everywhere (a step == a turn).
///   - `max_continues` is a required `u32` (no default, no `Option`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BudgetPolicy {
    Unlimited,
    TotalSteps { value: u32 },
    PerLoop { value: u32 },
    PerAttempt { value: u32 },
}

impl BudgetPolicy {
    /// The per-scope step allowance carried by this policy (`None` for
    /// `Unlimited`). Shared by [`BudgetContext::allowance`] and the leaf
    /// cap-binding check in [`ReactConfig::run`] (#125).
    pub fn allowance_value(&self) -> Option<u32> {
        match self {
            BudgetPolicy::Unlimited => None,
            BudgetPolicy::TotalSteps { value }
            | BudgetPolicy::PerLoop { value }
            | BudgetPolicy::PerAttempt { value } => Some(*value),
        }
    }
}

/// See [`BudgetPolicy`] for the full type/variant/wire-format/rules doc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BudgetExhaustedBehavior {
    Continue {
        max_continues: u32,
        #[serde(rename = "on_exhausted")]
        on_exhausted: Box<BudgetExhaustedBehavior>,
    },
    Escalate,
    Fail,
}

/// The default [`BudgetExhaustedBehavior`] for a config node's serialized
/// `behavior` field (#129): `Escalate`. Two roles:
///   - `#[serde(default = "default_budget_behavior")]` so a strategy tree
///     serialized BEFORE #129 (no `behavior` key) still deserializes to the
///     historical placeholder, preserving backward-compat reads.
///   - the value `ReactConfig::per_loop` / migration shims stamp so a bare leaf
///     keeps its pre-#129 propagate-to-parent contract by default.
///
/// The field is NOT `skip_serializing_if`: it ALWAYS serializes (uniform wire
/// shape across all five config structs, Q1), so the cross-language fixtures
/// carry an explicit `"behavior":{"kind":"escalate"}` on every node.
pub(crate) fn default_budget_behavior() -> BudgetExhaustedBehavior {
    BudgetExhaustedBehavior::Escalate
}
/// Why a [`StrategyOutcome::BudgetExhausted`] was raised (issue #137). The
/// budget-exhaustion resolution site routes BOTH causes through the node's
/// [`BudgetExhaustedBehavior`], but stamps a DIFFERENT terminal
/// [`HaltReason`] on the Fail / Escalate→Fail terminals so a caller can tell a
/// genuine budget exhaustion from an error-grinding circuit-break. Runtime-only
/// (NOT serialized).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ExhaustionCause {
    /// The scope's step allowance ran out (the pre-#137 behavior). Resolves to
    /// [`HaltReason::BudgetExceeded`] on a Fail / Escalate→Fail terminal.
    #[default]
    Budget,
    /// The ReAct consecutive-recoverable-tool-error breaker hard-stopped at
    /// `2 * error_loop_threshold` identical-argument errors for `tool` (#137).
    /// Resolves to [`HaltReason::ToolErrorLoop`] on a Fail / Escalate→Fail
    /// terminal — NEVER [`HaltReason::BudgetExceeded`].
    ToolErrorLoop {
        tool: String,
        consecutive_errors: u32,
    },
}

/// The current run of consecutive recoverable tool errors for ONE tool name
/// (issue #137). Tracked per tool name in a loop-local
/// `HashMap<String, ErrorRun>` inside [`StandardHarness::run_react_inner`]:
///
/// - On a recoverable [`ToolOutput::Error`] for tool `T` with args `A`: if the
///   existing run for `T` has `args` STRUCTURALLY equal to `A`, `count` is
///   incremented; otherwise the run is REPLACED with a fresh `{ args: A,
///   count: 1 }` (covers both the first error and the args-changed case).
/// - On ANY success for tool `T`: the entry is REMOVED (AC1 — success resets the
///   run regardless of args).
/// - At `count == N` (`error_loop_threshold`): inject ONE corrective message
///   (AC2). `injected` is set so the message is NOT re-injected between `N` and
///   `2 * N` for the same run.
/// - At `count == 2 * N`: stop looping (AC3).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ErrorRun {
    /// The tool-call arguments shared by every error in this run.
    pub(crate) args: serde_json::Value,
    /// How many consecutive identical-argument recoverable errors so far.
    pub(crate) count: u32,
    /// Whether the `N`-threshold corrective message has already been injected
    /// for THIS run (so we inject exactly once between `N` and `2 * N`).
    pub(crate) injected: bool,
}

/// The error [`BudgetContext::charge`] returns when a debit would exceed the
/// scope's step allowance. Captures the budget state at the moment of
/// exhaustion. Runtime-only (NOT serialized).
///
/// Promotes to [`StrategyOutcome::BudgetExhausted`] (which adds
/// `partial_output`) at the strategy boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetExhausted {
    pub policy: BudgetPolicy,
    pub behavior: BudgetExhaustedBehavior,
    pub steps_taken: u32,
    pub continues_used: u32,
    pub phase: String,
}

/// One budget scope in the strategy tree. Each recursion node gets its OWN
/// `BudgetContext`; siblings do NOT share. Runtime-only (NOT serialized).
///
/// The per-scope step allowance is the policy's own `value`:
/// `TotalSteps`/`PerLoop`/`PerAttempt` all expose `value` as the cap for this
/// scope; `Unlimited` is uncapped (`remaining()` → `None`).
///
/// `continues_used` is an in-memory field ONLY in this slice; its checkpoint
/// persistence is deferred to the enforcement slice (see section header,
/// resolution #3).
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetContext {
    pub policy: BudgetPolicy,
    pub behavior: BudgetExhaustedBehavior,
    pub steps_taken: u32,
    pub continues_used: u32,
    pub phase: String,
    /// In-process auto-grants spent at this scope's escalation point under
    /// [`EscalationMode::AutoContinue`] (SC-5). Bounded by the mode's
    /// `max_grants`. Runtime-only, like `continues_used`; never serialized.
    pub auto_grants_used: u32,
}

impl BudgetContext {
    /// Construct a fresh scope (zeroed counters) for `policy`/`behavior`/`phase`.
    pub fn new(
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        phase: impl Into<String>,
    ) -> Self {
        Self {
            policy,
            behavior,
            steps_taken: 0,
            continues_used: 0,
            phase: phase.into(),
            auto_grants_used: 0,
        }
    }

    /// Reconstruct a RESUMED scope (#129) whose `continues_used` is seeded from a
    /// cross-process checkpoint — the sole field of [`BudgetContext`] that must
    /// survive a process pause. `steps_taken` starts at 0 (the resumed run
    /// re-enters the loop with a fresh per-round step budget; the checkpoint only
    /// carries how many continues were ALREADY spent so a `Continue` spanning the
    /// pause cannot exceed `max_continues`). Runtime-only — `continues_used` is
    /// read off the `HumanRequest::BudgetExhausted` payload (Q3: NOT a new
    /// serialized `BudgetContext`/`PausedState` field).
    pub fn resumed(
        policy: BudgetPolicy,
        behavior: BudgetExhaustedBehavior,
        phase: impl Into<String>,
        continues_used: u32,
    ) -> Self {
        Self {
            policy,
            behavior,
            steps_taken: 0,
            continues_used,
            phase: phase.into(),
            auto_grants_used: 0,
        }
    }

    /// The per-scope step allowance (`None` for `Unlimited`).
    fn allowance(&self) -> Option<u32> {
        self.policy.allowance_value()
    }

    /// Debit `turns` steps against the scope allowance (pure arithmetic — see
    /// section header, resolution #2). On success increments `steps_taken` and
    /// returns `Ok(())`. If the debit would exceed the allowance, returns
    /// `Err(BudgetExhausted{..})` from current state WITHOUT mutating. Does NOT
    /// walk the behavior chain or consume continues. `Unlimited` never exhausts.
    pub fn charge(&mut self, turns: u32) -> Result<(), BudgetExhausted> {
        if let Some(allowance) = self.allowance() {
            if self.steps_taken.saturating_add(turns) > allowance {
                return Err(BudgetExhausted {
                    policy: self.policy.clone(),
                    behavior: self.behavior.clone(),
                    steps_taken: self.steps_taken,
                    continues_used: self.continues_used,
                    phase: self.phase.clone(),
                });
            }
        }
        self.steps_taken = self.steps_taken.saturating_add(turns);
        Ok(())
    }

    /// Steps left in this scope (`allowance - steps_taken`, saturating).
    /// `None` for `Unlimited` (no cap).
    pub fn remaining(&self) -> Option<u32> {
        self.allowance().map(|a| a.saturating_sub(self.steps_taken))
    }

    /// Continues left before fall-through. For a `Continue { max_continues }`
    /// behavior this is `max_continues - continues_used` (saturating); for
    /// `Escalate`/`Fail` there are no continues, so `0`.
    pub fn continues_remaining(&self) -> u32 {
        match self.behavior {
            BudgetExhaustedBehavior::Continue { max_continues, .. } => {
                max_continues.saturating_sub(self.continues_used)
            }
            BudgetExhaustedBehavior::Escalate | BudgetExhaustedBehavior::Fail => 0,
        }
    }

    /// Grant one in-process continue (#125): bump `continues_used` and RESET
    /// `steps_taken` to 0 so the scope's step allowance refreshes for the next
    /// round. This is a purely in-memory reset — the session / messages are
    /// untouched (the loop keeps the same conversation; only the per-scope step
    /// counter rewinds). The `continues_used` persistence across a serialized
    /// checkpoint is DEFERRED to #129.
    pub fn consume_continue(&mut self) {
        self.continues_used = self.continues_used.saturating_add(1);
        self.steps_taken = 0;
    }

    /// Grant one in-process AUTO-continue at this scope's escalation point
    /// (SC-5, [`EscalationMode::AutoContinue`]): bump `auto_grants_used` and raise
    /// the scope's cap to `steps_taken + steps_per_grant` so the loop gets exactly
    /// `steps_per_grant` more steps after the exhaustion point (mirrors a human
    /// `ContinueWithBudget { steps }` grant, applied automatically). `Unlimited`
    /// never reaches here. Unlike [`consume_continue`](Self::consume_continue) this
    /// does NOT rewind `steps_taken` — the cap moves up instead, so the grant is a
    /// strict, additive `steps_per_grant`.
    pub fn grant_auto_continue(&mut self, steps_per_grant: u32) {
        self.auto_grants_used = self.auto_grants_used.saturating_add(1);
        let granted = self.steps_taken.saturating_add(steps_per_grant);
        grant_budget_policy(&mut self.policy, granted);
    }

    /// Auto-grants still available before falling through to the autonomous
    /// terminal, given the mode's `max_grants` (SC-5). `0` once spent.
    pub fn auto_grants_remaining(&self, max_grants: u32) -> u32 {
        max_grants.saturating_sub(self.auto_grants_used)
    }

    /// Resolve this scope's [`BudgetExhaustedBehavior`] at the moment of
    /// exhaustion (#125), walking the on-continues-exhausted fall-through chain:
    ///
    ///   - `Fail`     → [`ExhaustedResolution::Fail`].
    ///   - `Escalate` → [`ExhaustedResolution::Escalate`].
    ///   - `Continue { max_continues, on_exhausted }` →
    ///       - if [`continues_remaining`](Self::continues_remaining) `> 0`:
    ///         [`consume_continue`](Self::consume_continue) (reset counter, bump
    ///         `continues_used`) and return [`ExhaustedResolution::Continue`];
    ///       - otherwise the continues are spent: ADOPT the boxed `on_exhausted`
    ///         behavior as this scope's behavior and recurse into it (the
    ///         fall-through), so a `Continue { on_exhausted: Escalate }` whose
    ///         continues are spent resolves to `Escalate`.
    ///
    /// Mutates `self`: on a granted continue the counter resets; on fall-through
    /// `self.behavior` is replaced by the boxed inner behavior so subsequent
    /// resolutions see the post-fall-through behavior.
    pub fn resolve_exhausted(&mut self) -> ExhaustedResolution {
        match self.behavior.clone() {
            BudgetExhaustedBehavior::Fail => ExhaustedResolution::Fail,
            BudgetExhaustedBehavior::Escalate => ExhaustedResolution::Escalate,
            BudgetExhaustedBehavior::Continue { on_exhausted, .. } => {
                if self.continues_remaining() > 0 {
                    self.consume_continue();
                    ExhaustedResolution::Continue
                } else {
                    // Continues spent — fall through to the boxed behavior.
                    self.behavior = *on_exhausted;
                    self.resolve_exhausted()
                }
            }
        }
    }
}

/// The runtime-only resolution of a [`BudgetExhaustedBehavior`] chain at the
/// moment of exhaustion (#125). NOT serialized — purely a control-flow signal
/// returned by [`BudgetContext::resolve_exhausted`].
///
///   - `Continue` — the scope was granted an in-process continue (counter reset,
///     `continues_used` bumped); the caller loops again.
///   - `Fail`     — terminate; `partial_output = None` (discarded by contract).
///   - `Escalate` — hand off to the parent; `partial_output = Some(..)` carries
///     the node-concrete partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExhaustedResolution {
    Continue,
    Fail,
    Escalate,
}

/// Runtime push/pop stack of [`BudgetContext`] scopes — one node per recursion
/// frame, pushed on descent and popped on ascent. Runtime-only (NOT
/// serialized). Siblings get DISTINCT contexts and do not share state.
#[derive(Debug, Default)]
pub struct BudgetStack {
    pub stack: Vec<BudgetContext>,
}

impl BudgetStack {
    /// Push a new scope onto the stack.
    pub fn push(&mut self, cx: BudgetContext) {
        self.stack.push(cx);
    }

    /// Pop the current scope, returning it (if any).
    pub fn pop(&mut self) -> Option<BudgetContext> {
        self.stack.pop()
    }

    /// The current (innermost) scope, mutably.
    pub fn current_mut(&mut self) -> Option<&mut BudgetContext> {
        self.stack.last_mut()
    }

    /// The current stack depth (recursion frames active).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

/// Runtime push/pop stack of [`observability::SpanId`](crate::observability::SpanId)
/// (resolution #4). Runtime-only (NOT serialized).
#[derive(Debug, Default)]
pub struct SpanStack {
    pub stack: Vec<SpanId>,
}

impl SpanStack {
    /// Push a span id onto the stack.
    pub fn push(&mut self, id: SpanId) {
        self.stack.push(id);
    }

    /// Pop the current span id, returning it (if any).
    pub fn pop(&mut self) -> Option<SpanId> {
        self.stack.pop()
    }

    /// The current stack depth.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}
