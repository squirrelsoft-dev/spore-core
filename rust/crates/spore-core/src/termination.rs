//! Issue #13 — `TerminationPolicy`: evaluate after each turn whether to
//! continue, halt with success, halt with failure, or halt because a budget
//! limit was breached.
//!
//! See `docs/harness-engineering-concepts.md` § "TerminationPolicy" for the
//! authoritative rules. This module ships:
//!   - The full [`TerminationDecision`] / [`TerminationFailureReason`] /
//!     [`BudgetValue`] surface from the spec.
//!   - The [`CompletionCheck`] trait and standard checks
//!     ([`NullCompletionCheck`], [`FixedCompletionCheck`]).
//!   - [`StandardTerminationPolicy`] — the reference policy that runs budget
//!     first, then sensor halts, then the injected `CompletionCheck`.
//!
//! ## Rules enforced
//!   - The model's `agent_claims_done` is *one input*, not the decision.
//!   - Budget limits are unconditional hard stops — evaluated before anything
//!     else and regardless of `agent_claims_done`.
//!   - `HaltFailure` carries a typed [`TerminationFailureReason`]; it cannot
//!     be a free string.
//!   - The `CompletionCheck` is injected at construction time — the policy
//!     itself is domain-agnostic.
//!   - If `!agent_claims_done`, always [`TerminationDecision::Continue`]
//!     (after budget check).
//!   - When `agent_claims_done`, any sensor result with
//!     [`SensorOutcome::Halt`] becomes
//!     [`TerminationFailureReason::UnrecoverableSensorHalt`].
//!   - `CompletionCheck::check` returning `Some(reason)` ⇒
//!     [`TerminationDecision::Continue`] (the harness re-injects the reason).
//!   - `CompletionCheck::check` returning `None` ⇒
//!     [`TerminationDecision::HaltSuccess`] using the agent's last response
//!     as summary (empty string if absent).
//!   - `HumanHalted` is reserved for the harness; the policy never produces
//!     it. (Captured by [`TerminationFailureReason::HumanHalted`] for
//!     completeness of the public type.)
//!
//! ## Implementor notes
//!   - `check_budget` is exposed as a free function on the trait so callers
//!     can poll it cheaply every turn before assembling the rest of
//!     `TerminationInput`.
//!   - `BudgetValue` carries the *measured* value at the moment of halt —
//!     not the limit — so observers can compute overshoot.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::agent::AgentError;
use crate::harness::{
    BoxFut, BudgetLimitType, BudgetLimits, BudgetSnapshot, SessionId, SessionState, TaskId,
};
use crate::middleware::HookPoint;
use crate::sensor::{SensorId, SensorOutcome, SensorResult};

// ============================================================================
// BudgetValue
// ============================================================================

/// A measured budget quantity, carried on [`TerminationDecision::HaltBudgetExceeded`]
/// so observers can compute overshoot against the configured limit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BudgetValue {
    Turns {
        value: u32,
    },
    Tokens {
        value: u64,
    },
    Duration {
        #[serde(with = "duration_secs")]
        value: Duration,
    },
    Usd {
        value: f64,
    },
}

impl BudgetValue {
    pub fn turns(v: u32) -> Self {
        Self::Turns { value: v }
    }
    pub fn tokens(v: u64) -> Self {
        Self::Tokens { value: v }
    }
    pub fn duration(v: Duration) -> Self {
        Self::Duration { value: v }
    }
    pub fn usd(v: f64) -> Self {
        Self::Usd { value: v }
    }
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(v: &Duration, s: S) -> Result<S::Ok, S::Error> {
        v.as_secs().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs(u64::deserialize(d)?))
    }
}

// ============================================================================
// SessionStateSnapshot
// ============================================================================

/// Read-only snapshot of session state handed to [`CompletionCheck::check`].
///
/// Wraps [`SessionState`] so the policy can identify the source session and
/// task — completion checks frequently key into per-session scratchpads
/// (e.g. `feature_list.json` under `.spore/<session>/`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionStateSnapshot {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub state: SessionState,
}

impl SessionStateSnapshot {
    pub fn new(session_id: SessionId, task_id: TaskId, state: SessionState) -> Self {
        Self {
            session_id,
            task_id,
            state,
        }
    }
}

// ============================================================================
// TerminationFailureReason
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TerminationFailureReason {
    CompletionCheckFailed {
        detail: String,
    },
    MaxRetriesExhausted {
        tool: String,
        attempts: u32,
    },
    UnrecoverableSensorHalt {
        sensor_id: SensorId,
        detail: String,
    },
    MiddlewareHalt {
        hook: HookPoint,
        reason: String,
    },
    AgentError {
        error: AgentError,
    },
    PolicyViolation {
        detail: String,
    },
    /// Set by the harness directly when `HumanResponse::Halt` is received;
    /// the [`TerminationPolicy`] never produces this variant.
    HumanHalted,
}

// ============================================================================
// TerminationDecision
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerminationDecision {
    Continue,
    HaltSuccess {
        summary: String,
    },
    HaltFailure {
        reason: TerminationFailureReason,
    },
    HaltBudgetExceeded {
        limit_type: BudgetLimitType,
        used: BudgetValue,
        limit: BudgetValue,
    },
}

// ============================================================================
// TerminationInput
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminationInput {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub turn_number: u32,
    pub agent_claims_done: bool,
    #[serde(default)]
    pub agent_response: Option<String>,
    pub budget_used: BudgetSnapshot,
    pub budget_limits: BudgetLimits,
    #[serde(default)]
    pub sensor_results: Vec<SensorResult>,
    pub session_state: SessionStateSnapshot,
}

// ============================================================================
// CompletionCheck
// ============================================================================

/// Pluggable domain-specific completion check. Injected at construction —
/// the policy is otherwise domain-agnostic.
///
/// Returns `None` if complete, `Some(reason)` if not yet done. The harness
/// injects `reason` into the next turn's context when `Some` is returned.
pub trait CompletionCheck: Send + Sync {
    fn check<'a>(&'a self, state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>>;
    fn description(&self) -> String;
}

/// Always-complete check. Causes the policy to halt with success the moment
/// the agent claims done.
pub struct NullCompletionCheck;

impl CompletionCheck for NullCompletionCheck {
    fn check<'a>(&'a self, _state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>> {
        Box::pin(async { None })
    }
    fn description(&self) -> String {
        "null (always complete)".into()
    }
}

/// Test/fixture completion check that returns a configured outcome.
pub struct FixedCompletionCheck {
    pub outcome: Option<String>,
    pub label: String,
}

impl FixedCompletionCheck {
    pub fn complete() -> Self {
        Self {
            outcome: None,
            label: "fixed:complete".into(),
        }
    }
    pub fn incomplete(reason: impl Into<String>) -> Self {
        Self {
            outcome: Some(reason.into()),
            label: "fixed:incomplete".into(),
        }
    }
}

impl CompletionCheck for FixedCompletionCheck {
    fn check<'a>(&'a self, _state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>> {
        let v = self.outcome.clone();
        Box::pin(async move { v })
    }
    fn description(&self) -> String {
        self.label.clone()
    }
}

// ============================================================================
// TerminationPolicy trait
// ============================================================================

pub trait TerminationPolicy: Send + Sync {
    fn evaluate<'a>(&'a self, input: &'a TerminationInput) -> BoxFut<'a, TerminationDecision>;

    /// Cheap budget poll — does not require sensor results or a completion
    /// check. Returns `Some(HaltBudgetExceeded)` if any limit is breached.
    fn check_budget(
        &self,
        snapshot: &BudgetSnapshot,
        limits: &BudgetLimits,
    ) -> Option<TerminationDecision> {
        check_budget_default(snapshot, limits)
    }
}

/// Default budget check used by [`StandardTerminationPolicy`] and exposed
/// for direct use by the harness loop.
pub fn check_budget_default(
    snapshot: &BudgetSnapshot,
    limits: &BudgetLimits,
) -> Option<TerminationDecision> {
    if let Some(max) = limits.max_turns {
        if snapshot.turns >= max {
            return Some(TerminationDecision::HaltBudgetExceeded {
                limit_type: BudgetLimitType::Turns,
                used: BudgetValue::turns(snapshot.turns),
                limit: BudgetValue::turns(max),
            });
        }
    }
    if let Some(max) = limits.max_input_tokens {
        if snapshot.input_tokens >= max as u64 {
            return Some(TerminationDecision::HaltBudgetExceeded {
                limit_type: BudgetLimitType::InputTokens,
                used: BudgetValue::tokens(snapshot.input_tokens),
                limit: BudgetValue::tokens(max as u64),
            });
        }
    }
    if let Some(max) = limits.max_output_tokens {
        if snapshot.output_tokens >= max as u64 {
            return Some(TerminationDecision::HaltBudgetExceeded {
                limit_type: BudgetLimitType::OutputTokens,
                used: BudgetValue::tokens(snapshot.output_tokens),
                limit: BudgetValue::tokens(max as u64),
            });
        }
    }
    if let Some(max) = limits.max_wall_time {
        let used = snapshot.wall_time.unwrap_or_default();
        if used >= max {
            return Some(TerminationDecision::HaltBudgetExceeded {
                limit_type: BudgetLimitType::WallTime,
                used: BudgetValue::duration(used),
                limit: BudgetValue::duration(max),
            });
        }
    }
    if let Some(max) = limits.max_cost_usd {
        if snapshot.cost_usd >= max {
            return Some(TerminationDecision::HaltBudgetExceeded {
                limit_type: BudgetLimitType::CostUsd,
                used: BudgetValue::usd(snapshot.cost_usd),
                limit: BudgetValue::usd(max),
            });
        }
    }
    None
}

// ============================================================================
// StandardTerminationPolicy
// ============================================================================

/// Reference [`TerminationPolicy`]. Runs:
///   1. Budget check (unconditional)
///   2. Continue if `!agent_claims_done`
///   3. UnrecoverableSensorHalt if any sensor returned [`SensorOutcome::Halt`]
///   4. The injected `CompletionCheck`
pub struct StandardTerminationPolicy {
    check: Arc<dyn CompletionCheck>,
}

impl StandardTerminationPolicy {
    pub fn new(check: Arc<dyn CompletionCheck>) -> Self {
        Self { check }
    }
    pub fn with_null_check() -> Self {
        Self::new(Arc::new(NullCompletionCheck))
    }
    pub fn completion_check(&self) -> &Arc<dyn CompletionCheck> {
        &self.check
    }
}

impl TerminationPolicy for StandardTerminationPolicy {
    fn evaluate<'a>(&'a self, input: &'a TerminationInput) -> BoxFut<'a, TerminationDecision> {
        Box::pin(async move {
            if let Some(halt) = self.check_budget(&input.budget_used, &input.budget_limits) {
                return halt;
            }
            if !input.agent_claims_done {
                return TerminationDecision::Continue;
            }
            if let Some(r) = input
                .sensor_results
                .iter()
                .find(|r| r.outcome == SensorOutcome::Halt)
            {
                return TerminationDecision::HaltFailure {
                    reason: TerminationFailureReason::UnrecoverableSensorHalt {
                        sensor_id: r.sensor_id.clone(),
                        detail: r.detail.clone(),
                    },
                };
            }
            match self.check.check(&input.session_state).await {
                None => TerminationDecision::HaltSuccess {
                    summary: input.agent_response.clone().unwrap_or_default(),
                },
                Some(_) => TerminationDecision::Continue,
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Timestamp;
    use crate::sensor::SensorId;

    fn snapshot() -> SessionStateSnapshot {
        SessionStateSnapshot::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            SessionState::default(),
        )
    }

    fn input_at(turn: u32, done: bool) -> TerminationInput {
        TerminationInput {
            session_id: SessionId::new("s1"),
            task_id: TaskId::new("t1"),
            turn_number: turn,
            agent_claims_done: done,
            agent_response: Some("ok".into()),
            budget_used: BudgetSnapshot::default(),
            budget_limits: BudgetLimits::default(),
            sensor_results: vec![],
            session_state: snapshot(),
        }
    }

    fn sensor_result(id: &str, outcome: SensorOutcome) -> SensorResult {
        SensorResult {
            sensor_id: SensorId::new(id),
            outcome,
            observation: None,
            detail: format!("{outcome:?}"),
            fired_at: Timestamp::new("2026-05-17T00:00:00Z"),
        }
    }

    // ── Rule: budget is always checked first ────────────────────────────────

    #[tokio::test]
    async fn budget_hard_stop_when_done() {
        let p = StandardTerminationPolicy::with_null_check();
        let mut input = input_at(1, true);
        input.budget_used.turns = 5;
        input.budget_limits.max_turns = Some(5);
        match p.evaluate(&input).await {
            TerminationDecision::HaltBudgetExceeded { limit_type, .. } => {
                assert_eq!(limit_type, BudgetLimitType::Turns);
            }
            other => panic!("expected HaltBudgetExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn budget_hard_stop_when_not_done() {
        // Budget is checked before agent_claims_done.
        let p = StandardTerminationPolicy::with_null_check();
        let mut input = input_at(1, false);
        input.budget_used.turns = 5;
        input.budget_limits.max_turns = Some(5);
        assert!(matches!(
            p.evaluate(&input).await,
            TerminationDecision::HaltBudgetExceeded { .. }
        ));
    }

    #[tokio::test]
    async fn budget_check_covers_every_limit_type() {
        let cases: Vec<(BudgetSnapshot, BudgetLimits, BudgetLimitType)> = vec![
            (
                BudgetSnapshot {
                    turns: 3,
                    ..Default::default()
                },
                BudgetLimits {
                    max_turns: Some(3),
                    ..Default::default()
                },
                BudgetLimitType::Turns,
            ),
            (
                BudgetSnapshot {
                    input_tokens: 10,
                    ..Default::default()
                },
                BudgetLimits {
                    max_input_tokens: Some(10),
                    ..Default::default()
                },
                BudgetLimitType::InputTokens,
            ),
            (
                BudgetSnapshot {
                    output_tokens: 10,
                    ..Default::default()
                },
                BudgetLimits {
                    max_output_tokens: Some(10),
                    ..Default::default()
                },
                BudgetLimitType::OutputTokens,
            ),
            (
                BudgetSnapshot {
                    wall_time: Some(Duration::from_secs(10)),
                    ..Default::default()
                },
                BudgetLimits {
                    max_wall_time: Some(Duration::from_secs(10)),
                    ..Default::default()
                },
                BudgetLimitType::WallTime,
            ),
            (
                BudgetSnapshot {
                    cost_usd: 1.0,
                    ..Default::default()
                },
                BudgetLimits {
                    max_cost_usd: Some(1.0),
                    ..Default::default()
                },
                BudgetLimitType::CostUsd,
            ),
        ];
        for (s, l, want) in cases {
            let got = check_budget_default(&s, &l);
            match got {
                Some(TerminationDecision::HaltBudgetExceeded { limit_type, .. }) => {
                    assert_eq!(limit_type, want);
                }
                other => panic!("expected HaltBudgetExceeded({want:?}), got {other:?}"),
            }
        }
        assert!(
            check_budget_default(&BudgetSnapshot::default(), &BudgetLimits::default()).is_none()
        );
    }

    // ── Rule: not-done always continues (after budget) ──────────────────────

    #[tokio::test]
    async fn not_done_continues() {
        let p = StandardTerminationPolicy::with_null_check();
        let input = input_at(1, false);
        assert!(matches!(
            p.evaluate(&input).await,
            TerminationDecision::Continue
        ));
    }

    // ── Rule: sensor halt becomes UnrecoverableSensorHalt ───────────────────

    #[tokio::test]
    async fn sensor_halt_overrides_completion_success() {
        let p = StandardTerminationPolicy::with_null_check();
        let mut input = input_at(1, true);
        input
            .sensor_results
            .push(sensor_result("guardrail", SensorOutcome::Halt));
        match p.evaluate(&input).await {
            TerminationDecision::HaltFailure {
                reason: TerminationFailureReason::UnrecoverableSensorHalt { sensor_id, .. },
            } => assert_eq!(sensor_id.as_str(), "guardrail"),
            other => panic!("expected UnrecoverableSensorHalt, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sensor_warn_does_not_halt() {
        let p = StandardTerminationPolicy::with_null_check();
        let mut input = input_at(1, true);
        input
            .sensor_results
            .push(sensor_result("guardrail", SensorOutcome::Warn));
        assert!(matches!(
            p.evaluate(&input).await,
            TerminationDecision::HaltSuccess { .. }
        ));
    }

    // ── Rule: completion check returning Some(reason) ⇒ Continue ────────────

    #[tokio::test]
    async fn incomplete_check_continues_with_agent_claimed_done() {
        let p = StandardTerminationPolicy::new(Arc::new(FixedCompletionCheck::incomplete(
            "feature B not implemented",
        )));
        let input = input_at(1, true);
        assert!(matches!(
            p.evaluate(&input).await,
            TerminationDecision::Continue
        ));
    }

    // ── Rule: completion check returning None ⇒ HaltSuccess(summary) ────────

    #[tokio::test]
    async fn complete_check_halts_success_with_summary() {
        let p = StandardTerminationPolicy::with_null_check();
        let mut input = input_at(1, true);
        input.agent_response = Some("all green".into());
        match p.evaluate(&input).await {
            TerminationDecision::HaltSuccess { summary } => assert_eq!(summary, "all green"),
            other => panic!("expected HaltSuccess, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn halt_success_summary_empty_when_no_response() {
        let p = StandardTerminationPolicy::with_null_check();
        let mut input = input_at(1, true);
        input.agent_response = None;
        match p.evaluate(&input).await {
            TerminationDecision::HaltSuccess { summary } => assert_eq!(summary, ""),
            other => panic!("expected HaltSuccess, got {other:?}"),
        }
    }

    // ── Rule: HaltFailure carries typed reason ──────────────────────────────

    #[tokio::test]
    async fn halt_failure_reason_is_typed() {
        // Round-trip every variant through serde to prove the wire format
        // matches the cross-language fixture schema.
        let cases = vec![
            TerminationFailureReason::CompletionCheckFailed {
                detail: "nope".into(),
            },
            TerminationFailureReason::MaxRetriesExhausted {
                tool: "bash".into(),
                attempts: 3,
            },
            TerminationFailureReason::UnrecoverableSensorHalt {
                sensor_id: SensorId::new("g"),
                detail: "tripped".into(),
            },
            TerminationFailureReason::MiddlewareHalt {
                hook: HookPoint::BeforeTurn,
                reason: "veto".into(),
            },
            TerminationFailureReason::AgentError {
                error: AgentError::EmptyResponse,
            },
            TerminationFailureReason::PolicyViolation {
                detail: "policy".into(),
            },
            TerminationFailureReason::HumanHalted,
        ];
        for c in cases {
            let json = serde_json::to_string(&c).unwrap();
            let back: TerminationFailureReason = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back);
        }
    }

    // ── Send/Sync ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn policy_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let p: Arc<dyn TerminationPolicy> = Arc::new(StandardTerminationPolicy::with_null_check());
        assert_send_sync(&p);
    }

    // ── Fixture replay ──────────────────────────────────────────────────────

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        agent_claims_done: bool,
        #[serde(default)]
        agent_response: Option<String>,
        budget_used: BudgetSnapshot,
        budget_limits: BudgetLimits,
        #[serde(default)]
        sensor_results: Vec<SensorResult>,
        completion_check: FixtureCheck,
        expected: serde_json::Value,
    }

    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum FixtureCheck {
        Complete,
        Incomplete { reason: String },
    }

    #[derive(Deserialize)]
    struct FixtureSuite {
        cases: Vec<FixtureCase>,
    }

    #[tokio::test]
    async fn fixture_replay_basic() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/termination_policy/basic.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: FixtureSuite = serde_json::from_str(&raw).unwrap();
        for case in suite.cases {
            let check: Arc<dyn CompletionCheck> = match &case.completion_check {
                FixtureCheck::Complete => Arc::new(FixedCompletionCheck::complete()),
                FixtureCheck::Incomplete { reason } => {
                    Arc::new(FixedCompletionCheck::incomplete(reason.clone()))
                }
            };
            let policy = StandardTerminationPolicy::new(check);
            let input = TerminationInput {
                session_id: SessionId::new("fixture"),
                task_id: TaskId::new("fixture-task"),
                turn_number: 1,
                agent_claims_done: case.agent_claims_done,
                agent_response: case.agent_response.clone(),
                budget_used: case.budget_used.clone(),
                budget_limits: case.budget_limits.clone(),
                sensor_results: case.sensor_results.clone(),
                session_state: snapshot(),
            };
            let got = policy.evaluate(&input).await;
            let got_json = serde_json::to_value(&got).unwrap();
            assert_eq!(
                got_json, case.expected,
                "fixture case `{}` produced unexpected decision",
                case.name
            );
        }
    }
}
