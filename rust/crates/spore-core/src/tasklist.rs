//! Issue #71 — Persisted task-list tool (PlanExecute, drives the execute loop).
//!
//! Decomposed out of #59. The accepted plan (#70) is parsed into a persisted
//! task list (#72), and the execute phase (#59) loops over the tasks until the
//! list is complete. This module owns the **task-list primitive** that holds
//! and mutates that list, plus its single mutating tool. It is consumed by #72
//! (which populates the list) and #59 (whose execute loop drains it).
//!
//! # Types
//! - [`TaskStatus`] — `Pending | InProgress | Completed | Blocked`. Serializes
//!   to the snake_case strings `"pending"`, `"in_progress"`, `"completed"`,
//!   `"blocked"`. Byte-identical across all four languages.
//! - [`Task`] — `{ id: u32, description: String, status: TaskStatus }`. A flat
//!   record: NO hierarchy/subtasks, NO timestamps (byte-identity constraint),
//!   NO order field (order is positional in the [`TaskList::tasks`] `Vec`).
//! - [`TaskList`] — `{ tasks: Vec<Task>, next_id: u32 }`. The persisted
//!   collection. `TaskList::default()` is empty with `next_id == 1`.
//! - [`TaskListError`] — `#[non_exhaustive]` domain error
//!   ([`TaskNotFound`](TaskListError::TaskNotFound),
//!   [`InvalidTransition`](TaskListError::InvalidTransition)). These map to a
//!   recoverable [`ToolOutput::Error`] at the tool boundary; the tool NEVER
//!   panics.
//!
//! # The tool — `task_list`
//! One tool ([`crate::tools::tasklist::TaskListTool`]) with an `action`
//! discriminator. Actions:
//! - `add_task { description }` — append a new `Pending` task (assigned the next
//!   sequential id) to the END of the list; increment `next_id`.
//! - `update_task { id, status?, description? }` — find the task by id; if
//!   `status` is given validate the transition then apply it; if `description`
//!   is given set it. With neither field present it is a no-op success.
//! - `complete_task { id }` — find the task by id, validate the transition to
//!   `Completed`, then mark it `Completed`.
//! - `list_tasks {}` — return the current list WITHOUT mutating it.
//!
//! Every action returns the JSON-serialized current [`TaskList`] as the
//! tool's success content.
//!
//! # ID scheme
//! Sequential `u32`, 1-based, assigned monotonically from
//! [`TaskList::next_id`]. Ids are never reused — `next_id` only ever grows,
//! and it is preserved across reload so a freshly-loaded list keeps minting
//! fresh ids.
//!
//! # Rules enforced
//! - R1  Ids are assigned 1, 2, 3, … monotonically from `next_id`; never reused.
//! - R2  `add_task` APPENDS to the end of `tasks` (positional order is stable).
//! - R3  `list_tasks` never mutates the list.
//! - R4  Unknown id on update/complete → [`TaskListError::TaskNotFound`]
//!   (recoverable at the tool boundary).
//! - R5  Status transitions follow the matrix in [`validate_transition`]
//!   (DECISION 1, "permissive-except-terminal-Completed"). A rejected
//!   transition → [`TaskListError::InvalidTransition`].
//! - R6  `Completed` is terminal: ANY transition OUT of `Completed` is rejected
//!   (the idempotent `Completed → Completed` self-transition is allowed).
//! - R7  Self-transitions `X → X` are idempotent and always allowed.
//! - R8  Persistence is through the storage seam (#75): the standalone
//!   [`TaskListTool`](crate::tools::TaskListTool) persists via the
//!   [`RunStore`](crate::storage::RunStore) on the `ToolContext`, keyed by
//!   `SessionId` under [`TASK_LIST_EXTRAS_KEY`]. The retired interim sandbox
//!   path (`.spore/task_list.json`) is GONE; with the library's default no-op
//!   storage a standalone tool call persists nothing across processes (an
//!   accepted behavior change — no migration shim). #59's execute loop shares
//!   the same `RunStore` key.
//!
//! There are no open `// SPEC QUESTION:` markers: both design forks
//! (transition matrix, state seam) were resolved before implementation.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::plan::PlanArtifact;

/// Key under which the [`TaskList`] is persisted in the
/// [`RunStore`](crate::storage::RunStore), keyed by `SessionId`. Both the
/// harness-side #59 execute loop and the standalone
/// [`TaskListTool`](crate::tools::TaskListTool) (#75) share this single key, so
/// a standalone tool call and a PlanExecute run on the same session
/// intentionally share one blob. Stable across all four languages. The JSON
/// shape is the canonical serde form of [`TaskList`]
/// (`{"tasks":[...],"next_id":N}`).
pub const TASK_LIST_EXTRAS_KEY: &str = "task_list";

// ============================================================================
// Types
// ============================================================================

/// Lifecycle status of a [`Task`]. Serializes to snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

/// A single task: flat, no hierarchy, no timestamps, no order field (order is
/// positional in [`TaskList::tasks`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: u32,
    pub description: String,
    pub status: TaskStatus,
}

/// The persisted collection of tasks plus the monotonic id counter.
///
/// Serializes as `{"tasks":[...],"next_id":N}`. `next_id` is `#[serde(default)]`
/// so an older/handwritten blob without it still deserializes (defaulting to
/// `0`), but [`TaskList::default`] and every freshly-minted list start at `1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskList {
    pub tasks: Vec<Task>,
    #[serde(default)]
    pub next_id: u32,
}

impl Default for TaskList {
    fn default() -> Self {
        Self {
            tasks: Vec::new(),
            next_id: 1,
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Errors raised by task-list mutations. `#[non_exhaustive]` per crate
/// conventions. Both variants are recoverable at the tool boundary.
#[derive(Debug, Clone, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TaskListError {
    /// No task with the given id exists in the list.
    #[error("task not found: {id}")]
    TaskNotFound { id: u32 },

    /// The requested status transition is not permitted by the matrix in
    /// [`validate_transition`] (notably any transition out of `Completed`).
    #[error("invalid transition for task {id}: {from:?} -> {to:?}")]
    InvalidTransition {
        id: u32,
        from: TaskStatus,
        to: TaskStatus,
    },
}

// ============================================================================
// Transition matrix (DECISION 1)
// ============================================================================

/// Validate a status transition under DECISION 1
/// ("permissive-except-terminal-Completed").
///
/// Allowed:
/// - any self-transition `X → X` (idempotent),
/// - `Pending → InProgress | Completed | Blocked`,
/// - `InProgress → Completed | Blocked`,
/// - `Blocked → InProgress | Completed`.
///
/// Rejected: ANY transition OUT of `Completed` (it is terminal) — except the
/// idempotent `Completed → Completed`.
///
/// The `id` is carried only to populate
/// [`TaskListError::InvalidTransition`]; it is not otherwise inspected.
pub fn validate_transition(id: u32, from: TaskStatus, to: TaskStatus) -> Result<(), TaskListError> {
    use TaskStatus::*;

    // Idempotent self-transition always allowed (incl. Completed -> Completed).
    if from == to {
        return Ok(());
    }

    let allowed = matches!(
        (from, to),
        (Pending, InProgress)
            | (Pending, Completed)
            | (Pending, Blocked)
            | (InProgress, Completed)
            | (InProgress, Blocked)
            | (Blocked, InProgress)
            | (Blocked, Completed)
    );

    if allowed {
        Ok(())
    } else {
        Err(TaskListError::InvalidTransition { id, from, to })
    }
}

// ============================================================================
// TaskList mutation helpers (the seam #72 / #59 will call)
// ============================================================================

impl TaskList {
    /// Append a new `Pending` task with the next sequential id and return that
    /// id. Increments [`TaskList::next_id`]. R1, R2.
    pub fn add(&mut self, description: String) -> u32 {
        let id = self.next_id;
        self.tasks.push(Task {
            id,
            description,
            status: TaskStatus::Pending,
        });
        self.next_id += 1;
        id
    }

    /// Update a task's status and/or description.
    ///
    /// - Unknown id → [`TaskListError::TaskNotFound`].
    /// - `status` present → validated via [`validate_transition`] then applied.
    /// - `description` present → set verbatim.
    /// - Both absent → no-op success.
    ///
    /// Status is validated BEFORE any field is written, so a rejected
    /// transition leaves the task untouched.
    pub fn update(
        &mut self,
        id: u32,
        status: Option<TaskStatus>,
        description: Option<String>,
    ) -> Result<(), TaskListError> {
        let task = self
            .tasks
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or(TaskListError::TaskNotFound { id })?;

        if let Some(to) = status {
            validate_transition(id, task.status, to)?;
            task.status = to;
        }
        if let Some(desc) = description {
            task.description = desc;
        }
        Ok(())
    }

    /// Mark a task `Completed`, validating the transition first.
    ///
    /// - Unknown id → [`TaskListError::TaskNotFound`].
    /// - Already `Completed` → idempotent success.
    pub fn complete(&mut self, id: u32) -> Result<(), TaskListError> {
        let task = self
            .tasks
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or(TaskListError::TaskNotFound { id })?;
        validate_transition(id, task.status, TaskStatus::Completed)?;
        task.status = TaskStatus::Completed;
        Ok(())
    }
}

// ============================================================================
// Plan → TaskList parser (issue #72; the bridge between #70 and #59)
// ============================================================================

/// Parse an accepted [`PlanArtifact`] (#70) into a fresh, ready-to-persist
/// [`TaskList`] (#71). This is the bridge between the plan phase and the
/// execute loop: once a plan is produced and accepted, its steps become the
/// task list that #59's execute loop drains.
///
/// # Types bridged
/// - Input: [`PlanArtifact`] `{ tasks: Vec<String>, rationale: String }`.
/// - Output: [`TaskList`] `{ tasks: Vec<Task>, next_id: u32 }`.
///
/// # Rules enforced
/// - One [`Task`] per plan step, in plan order (positional, via [`TaskList::add`]).
/// - Every produced task is [`TaskStatus::Pending`].
/// - Step descriptions are copied VERBATIM — no trim, no normalize, no filter
///   (matches #70's `tasks_kept_verbatim`: even `"  spaced  "` and `""` are kept).
/// - Ids are assigned `1..=n` sequentially via the [`TaskList::next_id`] scheme;
///   `next_id` ends at `n + 1`.
/// - An empty plan (`tasks: []`) yields [`TaskList::default`] —
///   `{ tasks: [], next_id: 1 }`. That is a valid EMPTY list, not an error and
///   not "immediate completion"; the execute loop (#59) decides loop semantics.
/// - `rationale` is DROPPED — neither [`Task`] nor [`TaskList`] carries it.
///
/// # Determinism
/// Pure and total: `&PlanArtifact -> TaskList`, no async, no I/O, no `Result`,
/// never panics. The same artifact always yields the same task list, so the
/// mapping is byte-identical across all four languages.
///
/// # Re-parsing / wiring
/// Always builds a fresh [`TaskList::default`]; it never merges into an
/// existing list (replanning is out of scope — single parse per accepted plan).
/// The accepted plan is mirrored into the [`RunStore`](crate::storage::RunStore)
/// under [`TASK_LIST_EXTRAS_KEY`] by #59's execute loop; the standalone
/// [`TaskListTool`](crate::tools::TaskListTool) shares that same key (#75).
pub fn plan_artifact_to_task_list(artifact: &PlanArtifact) -> TaskList {
    let mut list = TaskList::default(); // next_id == 1
    for step in &artifact.tasks {
        list.add(step.clone()); // verbatim; appends Pending; bumps next_id
    }
    list
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn list_with(statuses: &[TaskStatus]) -> TaskList {
        let mut l = TaskList::default();
        for _ in statuses {
            l.add("t".into());
        }
        for (i, s) in statuses.iter().enumerate() {
            l.tasks[i].status = *s;
        }
        l
    }

    // R1: ids are assigned 1, 2, 3, … sequentially.
    #[test]
    fn ids_are_sequential_from_one() {
        let mut l = TaskList::default();
        assert_eq!(l.add("a".into()), 1);
        assert_eq!(l.add("b".into()), 2);
        assert_eq!(l.add("c".into()), 3);
        assert_eq!(l.next_id, 4);
        assert_eq!(l.tasks.iter().map(|t| t.id).collect::<Vec<_>>(), [1, 2, 3]);
    }

    // R2: add appends to the end, preserving positional order.
    #[test]
    fn add_appends_in_order() {
        let mut l = TaskList::default();
        l.add("first".into());
        l.add("second".into());
        l.add("third".into());
        let descs: Vec<&str> = l.tasks.iter().map(|t| t.description.as_str()).collect();
        assert_eq!(descs, ["first", "second", "third"]);
        // New tasks start Pending.
        assert!(l.tasks.iter().all(|t| t.status == TaskStatus::Pending));
    }

    // R3: list (no helper — reading the Vec) does not mutate; we assert that
    // serializing for a list_tasks action leaves state untouched.
    #[test]
    fn serialize_does_not_mutate() {
        let l = list_with(&[TaskStatus::Pending, TaskStatus::InProgress]);
        let before = l.clone();
        let _ = serde_json::to_string(&l).unwrap();
        assert_eq!(l, before);
    }

    // update: status applied on a valid transition.
    #[test]
    fn update_status_valid() {
        let mut l = list_with(&[TaskStatus::Pending]);
        l.update(1, Some(TaskStatus::InProgress), None).unwrap();
        assert_eq!(l.tasks[0].status, TaskStatus::InProgress);
    }

    // update: description set independently of status.
    #[test]
    fn update_description_only() {
        let mut l = list_with(&[TaskStatus::Pending]);
        l.update(1, None, Some("rewritten".into())).unwrap();
        assert_eq!(l.tasks[0].description, "rewritten");
        assert_eq!(l.tasks[0].status, TaskStatus::Pending);
    }

    // update: both fields at once.
    #[test]
    fn update_status_and_description() {
        let mut l = list_with(&[TaskStatus::Pending]);
        l.update(1, Some(TaskStatus::Blocked), Some("blocked on x".into()))
            .unwrap();
        assert_eq!(l.tasks[0].status, TaskStatus::Blocked);
        assert_eq!(l.tasks[0].description, "blocked on x");
    }

    // update: neither field → no-op success.
    #[test]
    fn update_no_fields_is_noop_success() {
        let mut l = list_with(&[TaskStatus::InProgress]);
        let before = l.clone();
        l.update(1, None, None).unwrap();
        assert_eq!(l, before);
    }

    // complete: marks the task Completed.
    #[test]
    fn complete_marks_completed() {
        let mut l = list_with(&[TaskStatus::InProgress]);
        l.complete(1).unwrap();
        assert_eq!(l.tasks[0].status, TaskStatus::Completed);
    }

    // R4: unknown id on update/complete → TaskNotFound.
    #[test]
    fn unknown_id_is_task_not_found() {
        let mut l = list_with(&[TaskStatus::Pending]);
        assert_eq!(
            l.update(99, Some(TaskStatus::Completed), None).unwrap_err(),
            TaskListError::TaskNotFound { id: 99 }
        );
        assert_eq!(
            l.complete(99).unwrap_err(),
            TaskListError::TaskNotFound { id: 99 }
        );
    }

    // R5/R6: a rejected transition leaves the task untouched.
    #[test]
    fn rejected_transition_does_not_mutate() {
        let mut l = list_with(&[TaskStatus::Completed]);
        let before = l.clone();
        let err = l.update(1, Some(TaskStatus::InProgress), None).unwrap_err();
        assert!(matches!(err, TaskListError::InvalidTransition { .. }));
        assert_eq!(l, before);
    }

    // DECISION 1: every ALLOWED transition.
    #[test]
    fn allowed_transitions() {
        use TaskStatus::*;
        let allowed = [
            (Pending, InProgress),
            (Pending, Completed),
            (Pending, Blocked),
            (InProgress, Completed),
            (InProgress, Blocked),
            (Blocked, InProgress),
            (Blocked, Completed),
            // Idempotent self-transitions.
            (Pending, Pending),
            (InProgress, InProgress),
            (Completed, Completed),
            (Blocked, Blocked),
        ];
        for (from, to) in allowed {
            assert!(
                validate_transition(1, from, to).is_ok(),
                "expected {from:?} -> {to:?} to be allowed"
            );
        }
    }

    // DECISION 1 / R6: every transition OUT of Completed (except self) rejected.
    #[test]
    fn out_of_completed_rejected() {
        use TaskStatus::*;
        for to in [Pending, InProgress, Blocked] {
            let err = validate_transition(7, Completed, to).unwrap_err();
            assert_eq!(
                err,
                TaskListError::InvalidTransition {
                    id: 7,
                    from: Completed,
                    to
                }
            );
        }
    }

    // The remaining rejected (non-Completed-origin) transitions.
    #[test]
    fn other_rejected_transitions() {
        use TaskStatus::*;
        // InProgress -> Pending and Blocked -> Pending are NOT in the matrix.
        assert!(validate_transition(1, InProgress, Pending).is_err());
        assert!(validate_transition(1, Blocked, Pending).is_err());
    }

    #[test]
    fn pending_to_completed_allowed() {
        let mut l = list_with(&[TaskStatus::Pending]);
        l.complete(1).unwrap();
        assert_eq!(l.tasks[0].status, TaskStatus::Completed);
    }

    #[test]
    fn blocked_to_in_progress_and_completed_allowed() {
        let mut l = list_with(&[TaskStatus::Blocked, TaskStatus::Blocked]);
        l.update(1, Some(TaskStatus::InProgress), None).unwrap();
        l.complete(2).unwrap();
        assert_eq!(l.tasks[0].status, TaskStatus::InProgress);
        assert_eq!(l.tasks[1].status, TaskStatus::Completed);
    }

    // R7: idempotent self-transition is a success and a no-op on state.
    #[test]
    fn idempotent_self_transition() {
        let mut l = list_with(&[TaskStatus::Completed]);
        l.update(1, Some(TaskStatus::Completed), None).unwrap();
        l.complete(1).unwrap(); // Completed -> Completed via complete().
        assert_eq!(l.tasks[0].status, TaskStatus::Completed);
    }

    // Reload preserves next_id (ids never reused after a round-trip).
    #[test]
    fn reload_preserves_next_id() {
        let mut l = TaskList::default();
        l.add("a".into());
        l.add("b".into());
        let json = serde_json::to_string(&l).unwrap();
        let mut reloaded: TaskList = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded.next_id, 3);
        assert_eq!(reloaded.add("c".into()), 3); // continues from 3, not 1.
    }

    // Serde round-trip is byte-identical (re-serializing the parsed form).
    #[test]
    fn serde_round_trip_byte_identical() {
        let mut l = TaskList::default();
        l.add("alpha".into());
        l.add("beta".into());
        l.update(2, Some(TaskStatus::InProgress), None).unwrap();
        let json1 = serde_json::to_string(&l).unwrap();
        let parsed: TaskList = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2);
        assert_eq!(l, parsed);
    }

    // Status snake_case spellings are exact.
    #[test]
    fn status_snake_case_spellings() {
        assert_eq!(
            serde_json::to_string(&TaskStatus::Pending).unwrap(),
            "\"pending\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::InProgress).unwrap(),
            "\"in_progress\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Completed).unwrap(),
            "\"completed\""
        );
        assert_eq!(
            serde_json::to_string(&TaskStatus::Blocked).unwrap(),
            "\"blocked\""
        );
        // And back.
        let back: TaskStatus = serde_json::from_str("\"in_progress\"").unwrap();
        assert_eq!(back, TaskStatus::InProgress);
    }

    // Canonical empty-list serialization.
    #[test]
    fn default_serializes_canonically() {
        let l = TaskList::default();
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"tasks":[],"next_id":1}"#
        );
    }

    // Canonical populated-list serialization (exact spelling).
    #[test]
    fn populated_serializes_canonically() {
        let mut l = TaskList::default();
        l.add("write tests".into());
        l.update(1, Some(TaskStatus::InProgress), None).unwrap();
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"tasks":[{"id":1,"description":"write tests","status":"in_progress"}],"next_id":2}"#
        );
    }

    // ========================================================================
    // plan_artifact_to_task_list (#72)
    // ========================================================================

    fn artifact(tasks: &[&str], rationale: &str) -> PlanArtifact {
        PlanArtifact {
            tasks: tasks.iter().map(|s| s.to_string()).collect(),
            rationale: rationale.to_string(),
        }
    }

    // One task per plan step, plan order preserved, all Pending.
    #[test]
    fn plan_one_task_per_step_in_order_all_pending() {
        let list = plan_artifact_to_task_list(&artifact(&["first", "second", "third"], ""));
        let descs: Vec<&str> = list.tasks.iter().map(|t| t.description.as_str()).collect();
        assert_eq!(descs, ["first", "second", "third"]);
        assert!(list.tasks.iter().all(|t| t.status == TaskStatus::Pending));
    }

    // Sequential ids [1,2,3] and next_id == 4.
    #[test]
    fn plan_assigns_sequential_ids() {
        let list = plan_artifact_to_task_list(&artifact(&["a", "b", "c"], ""));
        assert_eq!(
            list.tasks.iter().map(|t| t.id).collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(list.next_id, 4);
    }

    // Deterministic: same artifact parsed twice → equal lists.
    #[test]
    fn plan_is_deterministic() {
        let a = artifact(&["x", "y"], "why");
        assert_eq!(
            plan_artifact_to_task_list(&a),
            plan_artifact_to_task_list(&a)
        );
    }

    // Descriptions copied VERBATIM — whitespace and empty strings preserved.
    #[test]
    fn plan_keeps_descriptions_verbatim() {
        let list = plan_artifact_to_task_list(&artifact(&["  spaced  ", ""], ""));
        assert_eq!(list.tasks[0].description, "  spaced  ");
        assert_eq!(list.tasks[1].description, "");
        assert_eq!(list.tasks[0].id, 1);
        assert_eq!(list.tasks[1].id, 2);
    }

    // Empty plan → the canonical empty list {tasks:[], next_id:1}.
    #[test]
    fn plan_empty_yields_default_list() {
        let list = plan_artifact_to_task_list(&artifact(&[], ""));
        assert_eq!(list, TaskList::default());
        assert!(list.tasks.is_empty());
        assert_eq!(list.next_id, 1);
    }

    // rationale is DROPPED — it appears nowhere in the resulting TaskList.
    #[test]
    fn plan_drops_rationale() {
        let list = plan_artifact_to_task_list(&artifact(&["do thing"], "SECRET_RATIONALE_TOKEN"));
        let json = serde_json::to_string(&list).unwrap();
        assert!(!json.contains("SECRET_RATIONALE_TOKEN"));
        assert!(!json.contains("rationale"));
    }

    // Serde round-trip of the parsed result is byte-identical / canonical.
    #[test]
    fn plan_result_serde_round_trip_byte_identical() {
        let list = plan_artifact_to_task_list(&artifact(&["alpha", "beta"], "r"));
        let json1 = serde_json::to_string(&list).unwrap();
        assert_eq!(
            json1,
            r#"{"tasks":[{"id":1,"description":"alpha","status":"pending"},{"id":2,"description":"beta","status":"pending"}],"next_id":3}"#
        );
        let parsed: TaskList = serde_json::from_str(&json1).unwrap();
        let json2 = serde_json::to_string(&parsed).unwrap();
        assert_eq!(json1, json2);
        assert_eq!(list, parsed);
    }

    // ------------------------------------------------------------------------
    // Fixture replay (#72)
    // ------------------------------------------------------------------------

    fn plan_fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/plan_to_tasklist")
            .join(name)
    }

    #[derive(serde::Deserialize)]
    struct PlanCase {
        name: String,
        input: PlanArtifact,
        expected: TaskList,
    }

    // Load cases.json, run plan_artifact_to_task_list on each input, and assert
    // the serialized TaskList equals `expected` byte-for-byte (canonical compact
    // form, field order tasks then next_id).
    #[test]
    fn fixture_replay_plan_to_tasklist() {
        let data = std::fs::read_to_string(plan_fixture_path("cases.json")).unwrap();
        let cases: Vec<PlanCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty(), "expected >=1 case");

        for case in cases {
            let got = plan_artifact_to_task_list(&case.input);
            // Structural equality with the fixture's expected list.
            assert_eq!(got, case.expected, "case {}: structural", case.name);
            // Byte-for-byte canonical serialization equality.
            let got_json = serde_json::to_string(&got).unwrap();
            let want_json = serde_json::to_string(&case.expected).unwrap();
            assert_eq!(got_json, want_json, "case {}: canonical bytes", case.name);
        }
    }
}
