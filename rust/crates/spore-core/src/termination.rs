//! Issue #13 — `TerminationPolicy`: evaluate after each turn whether to
//! continue, halt with success, halt with failure, or halt because a budget
//! limit was breached.
//!
// `CompletionCheck` is `#[deprecated]` in favour of the `Stop` lifecycle hook
// (issue #69), but the standard checks and `StandardTerminationPolicy` in this
// module still implement/consume it for backward compatibility. Silence the
// self-referential deprecation warnings module-wide; external callers still see
// the deprecation on the public trait.
#![allow(deprecated)]
//!
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
///
/// `workspace_root` is populated by the harness from
/// `SandboxProvider::workspace_root()` so checks like [`FeatureListCheck`] can
/// resolve workspace-relative paths without being given a sandbox handle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionStateSnapshot {
    pub session_id: SessionId,
    pub task_id: TaskId,
    pub state: SessionState,
    #[serde(default)]
    pub workspace_root: std::path::PathBuf,
}

impl SessionStateSnapshot {
    pub fn new(
        session_id: SessionId,
        task_id: TaskId,
        state: SessionState,
        workspace_root: std::path::PathBuf,
    ) -> Self {
        Self {
            session_id,
            task_id,
            state,
            workspace_root,
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
///
/// # Deprecated
///
/// Superseded by the `Stop` lifecycle hook (issue #69). A `Stop` hook that
/// returns [`crate::hooks::HookDecision::Block`] replaces the
/// `CompletionCheck`-returns-`Some(reason)` path: the reason is injected into
/// the next turn via the same mechanism, and the per-run `max_stop_blocks` cap
/// prevents runaway loops. This trait remains for backward compatibility.
#[deprecated(
    since = "0.1.0",
    note = "use the `Stop` lifecycle hook (issue #69) — register a Hook that returns HookDecision::Block"
)]
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

/// Spec alias from issue #43. Returns `None` immediately — the task is
/// considered done the moment the agent claims done. Use for single-turn
/// tasks where the model's self-assessment is sufficient.
pub type AlwaysComplete = NullCompletionCheck;

// ============================================================================
// FeatureListCheck (issue #43)
// ============================================================================

/// Reads `.spore/feature_list.json` under the snapshot's `workspace_root`
/// (issue #58, B2). Returns `Some` with a list of incomplete feature names if
/// any entry has `passes: false`. Returns `None` when all entries pass.
///
/// File schema: a JSON array of `{ "name": string, "passes": bool }`. Missing
/// or unreadable file → `Some(".spore/feature_list.json missing")` (treated as
/// incomplete so the agent learns to create it).
pub struct FeatureListCheck {
    pub path: std::path::PathBuf,
}

impl FeatureListCheck {
    /// Default location: `<workspace_root>/.spore/feature_list.json` (issue
    /// #58, B2 — the canonical `.spore/`-prefixed path shared with the Ralph
    /// loop strategy; one source of truth).
    pub fn new() -> Self {
        Self {
            path: std::path::PathBuf::from(".spore/feature_list.json"),
        }
    }

    pub fn with_path(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl Default for FeatureListCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct FeatureEntry {
    name: String,
    passes: bool,
}

impl CompletionCheck for FeatureListCheck {
    fn check<'a>(&'a self, state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>> {
        Box::pin(async move {
            let full = if self.path.is_absolute() {
                self.path.clone()
            } else {
                state.workspace_root.join(&self.path)
            };
            let raw = match std::fs::read_to_string(&full) {
                Ok(s) => s,
                Err(_) => {
                    return Some(format!("{} missing", self.path.display()));
                }
            };
            let entries: Vec<FeatureEntry> = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    return Some(format!("{} invalid JSON: {e}", self.path.display()));
                }
            };
            let incomplete: Vec<String> = entries
                .into_iter()
                .filter(|e| !e.passes)
                .map(|e| e.name)
                .collect();
            if incomplete.is_empty() {
                None
            } else {
                Some(format!("incomplete features: {}", incomplete.join(", ")))
            }
        })
    }

    fn description(&self) -> String {
        format!("feature list at {}", self.path.display())
    }
}

// ============================================================================
// TestSuiteCheck (issue #43)
// ============================================================================

/// Runs an external test command via the injected [`SandboxProvider`]. Returns
/// `None` if exit code is 0, otherwise `Some(failure_summary)` containing the
/// trailing portion of stderr/stdout so the next turn knows what failed.
///
/// `command` is parsed shell-style: first whitespace-separated token is the
/// program, remainder become args. For more complex invocations, callers
/// should build a wrapper script and invoke it instead.
pub struct TestSuiteCheck {
    pub command: String,
    pub working_dir: std::path::PathBuf,
    pub timeout: Duration,
    pub sandbox: Arc<dyn crate::harness::SandboxProvider>,
}

impl TestSuiteCheck {
    pub fn new(
        command: impl Into<String>,
        working_dir: impl Into<std::path::PathBuf>,
        timeout: Duration,
        sandbox: Arc<dyn crate::harness::SandboxProvider>,
    ) -> Self {
        Self {
            command: command.into(),
            working_dir: working_dir.into(),
            timeout,
            sandbox,
        }
    }
}

impl CompletionCheck for TestSuiteCheck {
    fn check<'a>(&'a self, _state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>> {
        Box::pin(async move {
            let mut parts = self.command.split_whitespace();
            let program = match parts.next() {
                Some(p) => p.to_string(),
                None => return Some("empty test command".into()),
            };
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();
            let result = self
                .sandbox
                .execute_command(
                    &program,
                    &args,
                    Some(self.working_dir.as_path()),
                    Some(self.timeout),
                )
                .await;
            match result {
                Ok(out) if out.exit_code == 0 && !out.timed_out => None,
                Ok(out) => {
                    let tail = tail_lines(&out.stderr, 20);
                    let tail = if tail.trim().is_empty() {
                        tail_lines(&out.stdout, 20)
                    } else {
                        tail
                    };
                    Some(format!(
                        "test suite failed (exit {}, timed_out={}):\n{}",
                        out.exit_code, out.timed_out, tail
                    ))
                }
                Err(v) => Some(format!("sandbox refused test command: {v:?}")),
            }
        })
    }

    fn description(&self) -> String {
        format!(
            "test suite: `{}` in {}",
            self.command,
            self.working_dir.display()
        )
    }
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

// ============================================================================
// QuestionAnsweredCheck (issue #43)
// ============================================================================

/// LLM-as-judge: asks a judge model whether the agent's final response
/// actually answered the original question.
///
/// Spec note: the issue lists `judge_model: ModelConfig`, but `ModelConfig`
/// today is a 2-field placeholder with no client surface. Until a
/// `ModelConfig → Arc<dyn ModelInterface>` factory exists, this struct is
/// generic over a concrete `M: ModelInterface` (the same pattern used by
/// `LlmJudgeEvaluator` in `metric.rs`). `ModelInterface` is not yet
/// dyn-compatible because of RPITIT (issue #45); revisit once #45 lands.
pub struct QuestionAnsweredCheck<M: crate::model::ModelInterface + 'static> {
    pub judge: Arc<M>,
    pub original_question: String,
    pub rubric: Option<String>,
}

impl<M: crate::model::ModelInterface + 'static> QuestionAnsweredCheck<M> {
    pub fn new(judge: Arc<M>, original_question: impl Into<String>) -> Self {
        Self {
            judge,
            original_question: original_question.into(),
            rubric: None,
        }
    }

    pub fn with_rubric(mut self, rubric: impl Into<String>) -> Self {
        self.rubric = Some(rubric.into());
        self
    }
}

impl<M: crate::model::ModelInterface + 'static> CompletionCheck for QuestionAnsweredCheck<M> {
    fn check<'a>(&'a self, state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>> {
        use crate::model::{Content, ContentBlock, Message, ModelParams, ModelRequest, Role};
        Box::pin(async move {
            let agent_response = last_assistant_text(&state.state.messages)
                .unwrap_or_else(|| "<no agent response>".to_string());
            let rubric_clause = self
                .rubric
                .as_deref()
                .map(|r| format!("\n\nRubric:\n{r}"))
                .unwrap_or_default();
            let user_text = format!(
                "Question:\n{}\n\nAgent's final response:\n{}\n\nDid the agent's response \
                 answer the question? Reply with the first line `ANSWERED: YES` or \
                 `ANSWERED: NO`, then a brief reason on the next line.{rubric_clause}",
                self.original_question, agent_response
            );
            let req = ModelRequest {
                messages: vec![
                    Message {
                        role: Role::System,
                        content: Content::Text {
                            text: "You are an evaluation judge. Reply with `ANSWERED: YES` or \
                                   `ANSWERED: NO` on the first line, no other prefix."
                                .into(),
                        },
                    },
                    Message {
                        role: Role::User,
                        content: Content::Text { text: user_text },
                    },
                ],
                tools: vec![],
                params: ModelParams::default(),
                stream: false,
            };
            let resp = match self.judge.call(req).await {
                Ok(r) => r,
                Err(e) => return Some(format!("judge model error: {e}")),
            };
            let verdict = resp
                .content
                .iter()
                .find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            let first = verdict
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_uppercase();
            if first.starts_with("ANSWERED: YES") {
                None
            } else {
                Some(format!("judge says not answered: {verdict}"))
            }
        })
    }

    fn description(&self) -> String {
        format!(
            "LLM-judge: did the response answer `{}`",
            self.original_question
        )
    }
}

fn last_assistant_text(messages: &[crate::model::Message]) -> Option<String> {
    use crate::model::{Content, Role};
    messages
        .iter()
        .rev()
        .find_map(|m| match (&m.role, &m.content) {
            (Role::Assistant, Content::Text { text }) => Some(text.clone()),
            _ => None,
        })
}

// ============================================================================
// SqlResultCheck (issue #43)
// ============================================================================

/// Validates the most recent SQL tool result in the session. Scans
/// `state.messages` in reverse for the last `Content::ToolResult` whose
/// matching `Content::ToolCall` has `name == sql_tool_name`, then parses the
/// result content as `{ "columns": [string], "rows": [[any]] }`.
///
/// Returns `None` when the result satisfies all configured constraints.
/// Returns `Some(reason)` if no SQL result was found, parsing failed, or a
/// constraint was violated.
pub struct SqlResultCheck {
    pub sql_tool_name: String,
    pub expected_columns: Option<Vec<String>>,
    pub min_rows: Option<usize>,
}

impl SqlResultCheck {
    /// Default tool name is `execute_sql`.
    pub fn new() -> Self {
        Self {
            sql_tool_name: "execute_sql".into(),
            expected_columns: None,
            min_rows: None,
        }
    }

    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.sql_tool_name = name.into();
        self
    }

    pub fn with_expected_columns(mut self, cols: Vec<String>) -> Self {
        self.expected_columns = Some(cols);
        self
    }

    pub fn with_min_rows(mut self, n: usize) -> Self {
        self.min_rows = Some(n);
        self
    }
}

impl Default for SqlResultCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Deserialize)]
struct SqlResultPayload {
    #[serde(default)]
    columns: Vec<String>,
    #[serde(default)]
    rows: Vec<serde_json::Value>,
}

impl CompletionCheck for SqlResultCheck {
    fn check<'a>(&'a self, state: &'a SessionStateSnapshot) -> BoxFut<'a, Option<String>> {
        use crate::model::Content;
        Box::pin(async move {
            // Build id -> tool_name map from ToolCalls so we can match
            // ToolResults back to their originating tool.
            let mut id_to_name: std::collections::HashMap<&str, &str> =
                std::collections::HashMap::new();
            for m in &state.state.messages {
                if let Content::ToolCall(call) = &m.content {
                    id_to_name.insert(call.id.as_str(), call.name.as_str());
                }
            }
            // Find the most recent ToolResult belonging to sql_tool_name.
            let result_content = state
                .state
                .messages
                .iter()
                .rev()
                .find_map(|m| match &m.content {
                    Content::ToolResult(r) => match id_to_name.get(r.tool_use_id.as_str()) {
                        Some(&n) if n == self.sql_tool_name => Some(r.content.clone()),
                        _ => None,
                    },
                    _ => None,
                });
            let raw = match result_content {
                Some(c) => c,
                None => {
                    return Some(format!(
                        "no `{}` tool result found in session",
                        self.sql_tool_name
                    ));
                }
            };
            let payload: SqlResultPayload = match serde_json::from_str(&raw) {
                Ok(p) => p,
                Err(e) => return Some(format!("sql result is not JSON: {e}")),
            };
            if let Some(expected) = &self.expected_columns {
                if &payload.columns != expected {
                    return Some(format!(
                        "sql columns mismatch: expected {expected:?}, got {:?}",
                        payload.columns
                    ));
                }
            }
            let min = self.min_rows.unwrap_or(1);
            if payload.rows.len() < min {
                return Some(format!(
                    "sql result has {} rows, expected at least {min}",
                    payload.rows.len()
                ));
            }
            None
        })
    }

    fn description(&self) -> String {
        format!("sql result check on tool `{}`", self.sql_tool_name)
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
            std::path::PathBuf::new(),
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

    // ========================================================================
    // FeatureListCheck
    // ========================================================================

    fn snapshot_in(dir: &std::path::Path) -> SessionStateSnapshot {
        SessionStateSnapshot::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            SessionState::default(),
            dir.to_path_buf(),
        )
    }

    #[tokio::test]
    async fn feature_list_all_pass_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".spore")).unwrap();
        std::fs::write(
            dir.path().join(".spore/feature_list.json"),
            r#"[{"name":"a","passes":true},{"name":"b","passes":true}]"#,
        )
        .unwrap();
        let snap = snapshot_in(dir.path());
        assert_eq!(FeatureListCheck::new().check(&snap).await, None);
    }

    #[tokio::test]
    async fn feature_list_some_fail_returns_reason() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".spore")).unwrap();
        std::fs::write(
            dir.path().join(".spore/feature_list.json"),
            r#"[{"name":"a","passes":true},{"name":"b","passes":false},{"name":"c","passes":false}]"#,
        )
        .unwrap();
        let snap = snapshot_in(dir.path());
        let r = FeatureListCheck::new().check(&snap).await.unwrap();
        assert!(
            r.contains("b") && r.contains("c") && !r.contains("a, "),
            "got: {r}"
        );
    }

    #[tokio::test]
    async fn feature_list_missing_file_returns_reason() {
        let dir = tempfile::tempdir().unwrap();
        let snap = snapshot_in(dir.path());
        let r = FeatureListCheck::new().check(&snap).await.unwrap();
        assert!(r.contains("missing"), "got: {r}");
    }

    #[tokio::test]
    async fn feature_list_custom_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("custom.json"),
            r#"[{"name":"x","passes":true}]"#,
        )
        .unwrap();
        let snap = snapshot_in(dir.path());
        assert_eq!(
            FeatureListCheck::with_path("custom.json")
                .check(&snap)
                .await,
            None
        );
    }

    // ========================================================================
    // TestSuiteCheck
    // ========================================================================

    struct StubSandbox {
        out: crate::harness::CommandOutput,
        root: std::path::PathBuf,
    }

    impl crate::harness::SandboxProvider for StubSandbox {
        fn validate<'a>(
            &'a self,
            _call: &'a crate::model::ToolCall,
        ) -> BoxFut<'a, Result<(), crate::harness::SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
        fn workspace_root(&self) -> &std::path::Path {
            &self.root
        }
        fn execute_command<'a>(
            &'a self,
            _command: &'a str,
            _args: &'a [String],
            _working_dir: Option<&'a std::path::Path>,
            _timeout: Option<Duration>,
        ) -> BoxFut<'a, Result<crate::harness::CommandOutput, crate::harness::SandboxViolation>>
        {
            let out = self.out.clone();
            Box::pin(async move { Ok(out) })
        }
    }

    fn stub_sandbox(exit: i32, stderr: &str) -> Arc<dyn crate::harness::SandboxProvider> {
        Arc::new(StubSandbox {
            out: crate::harness::CommandOutput {
                stdout: String::new(),
                stderr: stderr.to_string(),
                exit_code: exit,
                timed_out: false,
                truncated: false,
            },
            root: std::path::PathBuf::from("/"),
        })
    }

    #[tokio::test]
    async fn test_suite_pass_returns_none() {
        let check = TestSuiteCheck::new(
            "cargo test",
            std::path::PathBuf::from("."),
            Duration::from_secs(30),
            stub_sandbox(0, ""),
        );
        assert_eq!(check.check(&snapshot()).await, None);
    }

    #[tokio::test]
    async fn test_suite_fail_includes_stderr_tail() {
        let check = TestSuiteCheck::new(
            "cargo test",
            std::path::PathBuf::from("."),
            Duration::from_secs(30),
            stub_sandbox(1, "test foo ... FAILED\nassertion failed"),
        );
        let r = check.check(&snapshot()).await.unwrap();
        assert!(r.contains("FAILED"), "got: {r}");
    }

    #[tokio::test]
    async fn test_suite_empty_command_fails_cleanly() {
        let check = TestSuiteCheck::new(
            "",
            std::path::PathBuf::from("."),
            Duration::from_secs(30),
            stub_sandbox(0, ""),
        );
        assert!(check.check(&snapshot()).await.is_some());
    }

    // ========================================================================
    // QuestionAnsweredCheck
    // ========================================================================

    struct StubJudge {
        verdict: String,
    }

    impl crate::model::ModelInterface for StubJudge {
        fn call<'a>(
            &'a self,
            _req: crate::model::ModelRequest,
        ) -> BoxFut<'a, Result<crate::model::ModelResponse, crate::model::ModelError>> {
            Box::pin(async move {
            Ok(crate::model::ModelResponse {
                content: vec![crate::model::ContentBlock::Text {
                    text: self.verdict.clone(),
                }],
                stop_reason: crate::model::StopReason::EndTurn,
                usage: crate::model::TokenUsage::default(),
            })
            })
        }
        fn call_streaming<'a>(
            &'a self,
            _req: crate::model::ModelRequest,
        ) -> BoxFut<'a, Result<crate::model::ModelStream, crate::model::ModelError>> {
            Box::pin(async move {
            unreachable!("StubJudge::call_streaming is not used by QuestionAnsweredCheck")
            })
        }
        fn count_tokens<'a>(
            &'a self,
            _req: &'a crate::model::ModelRequest,
        ) -> BoxFut<'a, Result<u32, crate::model::ModelError>> {
            Box::pin(async move { Ok(0) })
        }
        fn provider(&self) -> crate::model::ProviderInfo {
            crate::model::ProviderInfo {
                name: "stub".into(),
                model_id: "stub".into(),
                context_window: 4096,
            }
        }
    }

    fn snap_with_assistant(text: &str) -> SessionStateSnapshot {
        let mut state = SessionState::default();
        state.messages.push(crate::model::Message {
            role: crate::model::Role::Assistant,
            content: crate::model::Content::Text {
                text: text.to_string(),
            },
        });
        SessionStateSnapshot::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            state,
            std::path::PathBuf::new(),
        )
    }

    #[tokio::test]
    async fn question_answered_yes_returns_none() {
        let judge = Arc::new(StubJudge {
            verdict: "ANSWERED: YES\nLooks good.".into(),
        });
        let c = QuestionAnsweredCheck::new(judge, "What is 2+2?");
        let snap = snap_with_assistant("It is 4.");
        assert_eq!(c.check(&snap).await, None);
    }

    #[tokio::test]
    async fn question_answered_no_returns_reason() {
        let judge = Arc::new(StubJudge {
            verdict: "ANSWERED: NO\nMissed the point.".into(),
        });
        let c = QuestionAnsweredCheck::new(judge, "What is 2+2?");
        let snap = snap_with_assistant("I don't know.");
        let r = c.check(&snap).await.unwrap();
        assert!(r.contains("not answered"), "got: {r}");
    }

    #[tokio::test]
    async fn question_answered_uses_rubric() {
        let judge = Arc::new(StubJudge {
            verdict: "ANSWERED: YES".into(),
        });
        let c = QuestionAnsweredCheck::new(judge, "q").with_rubric("Be strict about citations.");
        assert!(c.description().contains("q"));
        assert_eq!(c.check(&snap_with_assistant("a")).await, None);
    }

    // ========================================================================
    // SqlResultCheck
    // ========================================================================

    fn snap_with_sql_result(tool_name: &str, body: &str) -> SessionStateSnapshot {
        use crate::model::{Content, Message, Role, ToolCall as MTC, ToolResult as MTR};
        let mut state = SessionState::default();
        state.messages.push(Message {
            role: Role::Assistant,
            content: Content::ToolCall(MTC {
                id: "call-1".into(),
                name: tool_name.into(),
                input: serde_json::json!({"q":"select 1"}),
            }),
        });
        state.messages.push(Message {
            role: Role::Tool,
            content: Content::ToolResult(MTR {
                tool_use_id: "call-1".into(),
                content: body.into(),
                is_error: false,
            }),
        });
        SessionStateSnapshot::new(
            SessionId::new("s1"),
            TaskId::new("t1"),
            state,
            std::path::PathBuf::new(),
        )
    }

    #[tokio::test]
    async fn sql_result_check_default_passes_when_rows_present() {
        let snap = snap_with_sql_result(
            "execute_sql",
            r#"{"columns":["id","name"],"rows":[[1,"a"],[2,"b"]]}"#,
        );
        assert_eq!(SqlResultCheck::new().check(&snap).await, None);
    }

    #[tokio::test]
    async fn sql_result_check_empty_rows_fails() {
        let snap = snap_with_sql_result("execute_sql", r#"{"columns":["id"],"rows":[]}"#);
        let r = SqlResultCheck::new().check(&snap).await.unwrap();
        assert!(r.contains("0 rows"), "got: {r}");
    }

    #[tokio::test]
    async fn sql_result_check_column_mismatch_fails() {
        let snap = snap_with_sql_result("execute_sql", r#"{"columns":["id"],"rows":[[1]]}"#);
        let r = SqlResultCheck::new()
            .with_expected_columns(vec!["id".into(), "name".into()])
            .check(&snap)
            .await
            .unwrap();
        assert!(r.contains("columns mismatch"), "got: {r}");
    }

    #[tokio::test]
    async fn sql_result_check_min_rows_enforced() {
        let snap = snap_with_sql_result("execute_sql", r#"{"columns":["id"],"rows":[[1]]}"#);
        let r = SqlResultCheck::new()
            .with_min_rows(5)
            .check(&snap)
            .await
            .unwrap();
        assert!(r.contains("at least 5"), "got: {r}");
    }

    #[tokio::test]
    async fn sql_result_check_custom_tool_name() {
        let snap = snap_with_sql_result("run_query", r#"{"columns":[],"rows":[[1]]}"#);
        let c = SqlResultCheck::new().with_tool_name("run_query");
        assert_eq!(c.check(&snap).await, None);
    }

    #[tokio::test]
    async fn sql_result_check_no_matching_tool_fails() {
        let snap = snap_with_sql_result("other_tool", r#"{"columns":[],"rows":[[1]]}"#);
        let r = SqlResultCheck::new().check(&snap).await.unwrap();
        assert!(r.contains("no `execute_sql`"), "got: {r}");
    }

    // ========================================================================
    // SqlResultCheck — cross-language fixture replay
    // ========================================================================

    #[derive(Deserialize)]
    struct SqlFixtureCase {
        name: String,
        sql_tool_name: String,
        #[serde(default)]
        expected_columns: Option<Vec<String>>,
        #[serde(default)]
        min_rows: Option<usize>,
        messages: Vec<crate::model::Message>,
        expected: SqlFixtureExpected,
    }

    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum SqlFixtureExpected {
        Complete,
        Incomplete { contains: String },
    }

    #[derive(Deserialize)]
    struct SqlFixtureSuite {
        cases: Vec<SqlFixtureCase>,
    }

    #[tokio::test]
    async fn fixture_replay_sql_result_check() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/completion_check/sql_result.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: SqlFixtureSuite = serde_json::from_str(&raw).unwrap();
        for case in suite.cases {
            let state = SessionState {
                messages: case.messages,
                ..Default::default()
            };
            let snap = SessionStateSnapshot::new(
                SessionId::new("fix"),
                TaskId::new("fix"),
                state,
                std::path::PathBuf::new(),
            );
            let mut check = SqlResultCheck::new().with_tool_name(case.sql_tool_name);
            if let Some(cols) = case.expected_columns {
                check = check.with_expected_columns(cols);
            }
            if let Some(n) = case.min_rows {
                check = check.with_min_rows(n);
            }
            let got = check.check(&snap).await;
            match (got, case.expected) {
                (None, SqlFixtureExpected::Complete) => {}
                (Some(reason), SqlFixtureExpected::Incomplete { contains }) => {
                    assert!(
                        reason.contains(&contains),
                        "case `{}`: expected reason to contain `{contains}`, got `{reason}`",
                        case.name
                    );
                }
                (got, expected) => panic!(
                    "case `{}`: mismatch — got {got:?}, expected {}",
                    case.name,
                    match expected {
                        SqlFixtureExpected::Complete => "Complete".to_string(),
                        SqlFixtureExpected::Incomplete { contains } =>
                            format!("Incomplete(contains={contains})"),
                    }
                ),
            }
        }
    }

    // ========================================================================
    // AlwaysComplete alias
    // ========================================================================

    #[tokio::test]
    async fn always_complete_is_null_check() {
        // Alias sanity — same observable behavior.
        let a: AlwaysComplete = NullCompletionCheck;
        assert_eq!(a.check(&snapshot()).await, None);
    }
}
