//! Issue #70 — Plan phase / plan artifact (PlanExecute, phase 1 of 2).
//!
//! This module owns the *capture* half of the PlanExecute plan phase: turning a
//! planner model's `FinalResponse` text into a structured [`PlanArtifact`]. The
//! *phase driver* itself (`run_plan_phase`) lives on
//! [`StandardHarness`](crate::harness::StandardHarness) because it needs the
//! harness's turn machinery; this module supplies the deterministic, total
//! text→artifact step and the phase error type.
//!
//! # Public surface
//! - [`PlanArtifact`] — re-exported from [`crate::hooks`]; the existing,
//!   serializable contract (`{ tasks: Vec<String>, rationale: String }`) that is
//!   the payload of the `OnPlanCreated` hook. This issue REUSES it rather than
//!   defining a competing type. It is the contract consumed by #72 / #59.
//! - [`PlanPhaseError`] — `#[non_exhaustive]` error type for the plan phase.
//! - [`capture_plan_artifact`] — the model-text → [`PlanArtifact`] capture
//!   function. Deterministic and total: never panics; malformed input yields
//!   [`PlanPhaseError::UnparseablePlan`].
//!
//! The phase method on the harness:
//! - `StandardHarness::run_plan_phase` — seeds a planning directive, runs EXACTLY
//!   ONE constrained turn, captures the response via [`capture_plan_artifact`],
//!   fires `OnPlanCreated` (allowing in-place mutation), stores the resulting
//!   artifact into `SessionState.extras["plan_execute"]`, emits the turn span,
//!   and counts the turn against the shared budget.
//!
//! # Rules enforced
//! - R1  The plan phase runs exactly once (one planner turn).
//! - R2  One-shot: a tool call in the plan turn is a planning failure
//!   ([`PlanPhaseError::PlanningTurnFailed`]) — never a dispatch loop.
//! - R3  The artifact is captured from the response text.
//! - R4  The artifact is stored in `extras["plan_execute"]` as serialized JSON.
//! - R5  When a planner agent is configured it runs the plan turn.
//! - R6  Otherwise the plan turn runs on the default agent.
//! - R7  The plan turn counts against the shared budget.
//! - R8  Exactly one `TurnSpan` is recorded for the plan turn.
//! - R9  [`capture_plan_artifact`] is deterministic and total.
//! - R10 Budget exhausted before the plan turn → budget-exceeded `Failure`,
//!   no artifact stored.
//! - R11 `OnPlanCreated` can rewrite the plan before storage.
//!
//! # Resolved spec decisions (issue #70 — all four FINAL)
//! - **Q1 (model routing):** `HarnessConfig.planner_agent: Option<Arc<dyn Agent>>`
//!   plus a `HarnessBuilder::planner_agent` setter. When the strategy is
//!   `PlanExecute` and `planner_agent` is `Some`, the plan turn runs on it;
//!   otherwise it runs on the default `config.agent`. `plan_model` stays as
//!   DESCRIPTIVE metadata only — there is no `ModelConfig`→agent factory.
//! - **Q2 (HITL):** The plan phase ALWAYS runs to completion. It fires
//!   `OnPlanCreated` synchronously (the hook may rewrite the artifact via
//!   `&mut PlanArtifact`); the stored artifact reflects any mutation. No pause,
//!   no `WaitingForHuman` path — HITL is deferred.
//! - **Q3 (capture grammar):** JSON-in-response. Trim ASCII whitespace; strip a
//!   single leading ```` ``` ````/```` ```json ```` fence line and a single
//!   trailing ```` ``` ```` fence if present; parse a JSON object with `tasks`
//!   (required array of strings, kept verbatim, may be empty) and `rationale`
//!   (optional string, default `""`). Any failure →
//!   [`PlanPhaseError::UnparseablePlan`].
//! - **Q4 (terminal RunResult):** After producing, firing `OnPlanCreated`, and
//!   storing the artifact, the plan phase hands off to the execute phase (issue
//!   #59), which parses the artifact into a `TaskList` and loops over the steps.
//!   (Historically — before #59 — the `PlanExecute` arm halted here with a
//!   `HaltReason::ExecutePhaseNotImplemented` marker; that variant has been
//!   removed now that the execute phase exists.)

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use crate::hooks::PlanArtifact;

/// Key under which the produced [`PlanArtifact`] is stored in
/// `SessionState.extras` (serialized JSON). Stable across all four languages.
pub const PLAN_EXECUTE_EXTRAS_KEY: &str = "plan_execute";

/// Errors raised by the plan phase. `#[non_exhaustive]` per crate conventions.
#[derive(Debug, Clone, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PlanPhaseError {
    /// The planner's response text could not be parsed into a [`PlanArtifact`]
    /// under the Q3 grammar (not valid JSON, not a JSON object, or `tasks`
    /// absent / not an array / containing a non-string element).
    #[error("unparseable plan: {message}")]
    UnparseablePlan { message: String },

    /// The plan turn errored or did not produce a `FinalResponse` (e.g. the
    /// planner requested a tool call — R2 — or the agent returned an error).
    #[error("planning turn failed: {message}")]
    PlanningTurnFailed { message: String },
}

/// Capture a [`PlanArtifact`] from a planner's `FinalResponse` text.
///
/// This is the canonical Q3 grammar — it MUST be byte-identical across all four
/// languages, so it is kept simple and total:
///
/// 1. Trim leading/trailing ASCII whitespace.
/// 2. If the trimmed text begins with a triple-backtick fence, strip a single
///    leading fence line (the opening ```` ``` ```` plus any language tag up to
///    and including the first newline) and a single trailing ```` ``` ````
///    fence, then trim again.
/// 3. Parse the result as a JSON object with `tasks` (required array of JSON
///    strings, kept verbatim; an empty array is allowed) and `rationale`
///    (optional string, default `""`).
///
/// Any deviation → [`PlanPhaseError::UnparseablePlan`]. Never panics.
pub fn capture_plan_artifact(final_text: &str) -> Result<PlanArtifact, PlanPhaseError> {
    let trimmed = final_text.trim_matches(is_ascii_ws);
    let body = strip_code_fence(trimmed);

    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| PlanPhaseError::UnparseablePlan {
            message: format!("invalid JSON: {e}"),
        })?;

    let obj = value
        .as_object()
        .ok_or_else(|| PlanPhaseError::UnparseablePlan {
            message: "top-level JSON value is not an object".to_string(),
        })?;

    let tasks_value = obj
        .get("tasks")
        .ok_or_else(|| PlanPhaseError::UnparseablePlan {
            message: "missing required field `tasks`".to_string(),
        })?;

    let tasks_array = tasks_value
        .as_array()
        .ok_or_else(|| PlanPhaseError::UnparseablePlan {
            message: "field `tasks` is not an array".to_string(),
        })?;

    let mut tasks = Vec::with_capacity(tasks_array.len());
    for (i, element) in tasks_array.iter().enumerate() {
        match element.as_str() {
            // Verbatim — do NOT trim or filter.
            Some(s) => tasks.push(s.to_string()),
            None => {
                return Err(PlanPhaseError::UnparseablePlan {
                    message: format!("element {i} of `tasks` is not a string"),
                });
            }
        }
    }

    // `rationale` is optional; default "". If present it must be a string.
    let rationale = match obj.get("rationale") {
        None => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(_) => {
            return Err(PlanPhaseError::UnparseablePlan {
                message: "field `rationale` is not a string".to_string(),
            });
        }
    };

    Ok(PlanArtifact { tasks, rationale })
}

/// ASCII-whitespace predicate. Matches `' '`, `'\t'`, `'\n'`, `'\r'`, and the
/// form-feed / vertical-tab the JSON-adjacent grammar treats as whitespace —
/// kept to the ASCII set so trimming is byte-identical cross-language.
fn is_ascii_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{000B}' | '\u{000C}')
}

/// Strip a single leading ```` ``` ````/```` ```json ```` fence line and a
/// single trailing ```` ``` ```` fence, if the (already-trimmed) input opens
/// with a triple-backtick fence. Returns the inner body, re-trimmed. If the
/// input does not open with a fence it is returned unchanged.
fn strip_code_fence(trimmed: &str) -> &str {
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };

    // Drop the rest of the opening fence line (the optional language tag) up to
    // and including the first newline. A fence with no newline at all has no
    // body to parse; let JSON parsing reject it downstream.
    let body_start = match after_open.find('\n') {
        Some(nl) => &after_open[nl + 1..],
        None => after_open,
    };

    // Strip a single trailing closing fence if present, then re-trim.
    let body = match body_start.trim_end_matches(is_ascii_ws).strip_suffix("```") {
        Some(without_close) => without_close,
        None => body_start,
    };

    body.trim_matches(is_ascii_ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    // R3 / R9: a known JSON object captures to exact tasks + rationale.
    #[test]
    fn captures_plain_json_object() {
        let text = r#"{"tasks":["a","b","c"],"rationale":"because"}"#;
        let artifact = capture_plan_artifact(text).unwrap();
        assert_eq!(artifact.tasks, vec!["a", "b", "c"]);
        assert_eq!(artifact.rationale, "because");
    }

    // Q3: surrounding ASCII whitespace is trimmed.
    #[test]
    fn trims_surrounding_whitespace() {
        let text = "\n\t  {\"tasks\":[\"x\"]}  \r\n";
        let artifact = capture_plan_artifact(text).unwrap();
        assert_eq!(artifact.tasks, vec!["x"]);
        assert_eq!(artifact.rationale, "");
    }

    // Q3 fence-strip: ```json … ``` is stripped before parsing.
    #[test]
    fn strips_json_fence() {
        let text = "```json\n{\"tasks\":[\"step 1\",\"step 2\"],\"rationale\":\"r\"}\n```";
        let artifact = capture_plan_artifact(text).unwrap();
        assert_eq!(artifact.tasks, vec!["step 1", "step 2"]);
        assert_eq!(artifact.rationale, "r");
    }

    // Q3 fence-strip: a bare ``` fence (no language tag) is also stripped.
    #[test]
    fn strips_bare_fence() {
        let text = "```\n{\"tasks\":[\"only\"]}\n```";
        let artifact = capture_plan_artifact(text).unwrap();
        assert_eq!(artifact.tasks, vec!["only"]);
    }

    // Q3 fence-strip: uppercase ```JSON tag is stripped (language tag agnostic).
    #[test]
    fn strips_uppercase_json_fence() {
        let text = "```JSON\n{\"tasks\":[\"u\"]}\n```";
        let artifact = capture_plan_artifact(text).unwrap();
        assert_eq!(artifact.tasks, vec!["u"]);
    }

    // Q3: rationale is optional and defaults to "".
    #[test]
    fn rationale_defaults_to_empty() {
        let artifact = capture_plan_artifact(r#"{"tasks":["a"]}"#).unwrap();
        assert_eq!(artifact.rationale, "");
    }

    // Q3: an empty tasks array is ALLOWED (degenerate-plan handling is #72).
    #[test]
    fn empty_tasks_array_is_allowed() {
        let artifact = capture_plan_artifact(r#"{"tasks":[]}"#).unwrap();
        assert!(artifact.tasks.is_empty());
    }

    // Q3: task strings are kept verbatim — no trimming or filtering.
    #[test]
    fn tasks_kept_verbatim() {
        let artifact = capture_plan_artifact(r#"{"tasks":["  spaced  ",""]}"#).unwrap();
        assert_eq!(artifact.tasks, vec!["  spaced  ", ""]);
    }

    // R9: malformed inputs return UnparseablePlan, never panic.
    #[test]
    fn invalid_json_is_unparseable() {
        let err = capture_plan_artifact("not json at all").unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    #[test]
    fn non_object_top_level_is_unparseable() {
        let err = capture_plan_artifact("[1,2,3]").unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    #[test]
    fn missing_tasks_is_unparseable() {
        let err = capture_plan_artifact(r#"{"rationale":"x"}"#).unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    #[test]
    fn tasks_not_array_is_unparseable() {
        let err = capture_plan_artifact(r#"{"tasks":"a"}"#).unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    #[test]
    fn non_string_task_element_is_unparseable() {
        let err = capture_plan_artifact(r#"{"tasks":["a",2]}"#).unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    #[test]
    fn non_string_rationale_is_unparseable() {
        let err = capture_plan_artifact(r#"{"tasks":["a"],"rationale":5}"#).unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    #[test]
    fn empty_input_is_unparseable() {
        let err = capture_plan_artifact("   \n  ").unwrap_err();
        assert!(matches!(err, PlanPhaseError::UnparseablePlan { .. }));
    }

    // R9: deterministic — identical input yields identical artifact.
    #[test]
    fn capture_is_deterministic() {
        let text = "```json\n{\"tasks\":[\"a\",\"b\"],\"rationale\":\"r\"}\n```";
        let a1 = capture_plan_artifact(text).unwrap();
        let a2 = capture_plan_artifact(text).unwrap();
        assert_eq!(a1, a2);
    }
}
