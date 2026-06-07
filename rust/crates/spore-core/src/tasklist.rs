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
//! - [`Task`] — `{ id: u32, description: String, status: TaskStatus,
//!   blockers: Vec<u32> }`. A flat record: NO hierarchy/subtasks, NO timestamps
//!   (byte-identity constraint), NO order field (order is positional in the
//!   [`TaskList::tasks`] `Vec`). `blockers` (#118) are ids of other tasks that
//!   must be `Completed` before this task runs; it is the LAST wire field and is
//!   `#[serde(default)]` so pre-#118 blobs still load. Empty blockers ALWAYS
//!   serialize as `[]` (no `skip_serializing_if`), mirroring `next_id`.
//! - [`TaskList`] — `{ tasks: Vec<Task>, next_id: u32 }`. The persisted
//!   collection. `TaskList::default()` is empty with `next_id == 1`.
//! - [`TaskListError`] — `#[non_exhaustive]` domain error
//!   ([`TaskNotFound`](TaskListError::TaskNotFound),
//!   [`InvalidTransition`](TaskListError::InvalidTransition),
//!   [`InvalidBlockers`](TaskListError::InvalidBlockers)). These map to a
//!   recoverable [`ToolOutput::Error`] at the tool boundary; the tool NEVER
//!   panics.
//! - [`BlockerRejection`] — why a blocker set was rejected by
//!   [`TaskList::add`]: [`SelfBlock`](BlockerRejection::SelfBlock),
//!   [`UnknownId`](BlockerRejection::UnknownId),
//!   [`Cycle`](BlockerRejection::Cycle).
//!
//! # Trait / method surface
//! - [`TaskList::add`] `(description, blockers) -> Result<u32, TaskListError>` —
//!   fallible (#118): validates blockers BEFORE mutating; reject leaves the list
//!   untouched.
//! - [`TaskList::update`], [`TaskList::complete`] — unchanged.
//! - [`validate_transition`], [`would_create_cycle`], [`plan_artifact_to_task_list`].
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
//! - R9  (#118) `add_task` blockers are validated BEFORE any mutation; a reject
//!   leaves the list untouched (mirrors `update`). Validation:
//!   self-block (a blocker == the about-to-be-assigned id) →
//!   [`InvalidBlockers`](TaskListError::InvalidBlockers) `{ SelfBlock }`;
//!   unknown id (a blocker matching no existing task) → `{ UnknownId }`;
//!   cycle (the new edges would close a directed cycle in the blockers graph,
//!   checked by [`would_create_cycle`]) → `{ Cycle }`. Empty blockers never
//!   reject.
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
    /// Waiting on a blocker. (#118) This means BOTH "waiting on a blocker that
    /// has not yet completed" AND "a blocker failed terminally" — the status is
    /// the same in either case; the distinction (if any) lives in the
    /// scheduler, not the schema.
    Blocked,
}

/// A single task: flat, no hierarchy, no timestamps, no order field (order is
/// positional in [`TaskList::tasks`]).
///
/// `blockers` (#118) are the ids of other tasks that must be
/// [`Completed`](TaskStatus::Completed) before this task runs. The canonical
/// wire field order is `id, description, status, blockers` (blockers LAST).
/// `blockers` is `#[serde(default)]` so a pre-#118 blob without the key still
/// deserializes (to an empty `Vec`); empty blockers ALWAYS serialize as `[]`
/// (no `skip_serializing_if`), the same treatment as [`TaskList::next_id`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub id: u32,
    pub description: String,
    pub status: TaskStatus,
    #[serde(default)]
    pub blockers: Vec<u32>,
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
// Step ledger (#126 — Tier-2 global running ledger)
// ============================================================================

/// The maximum number of [`StepLedgerEntry`] rows the Tier-2 running ledger
/// retains (#126, RESOLVED spec decision B). Past this bound the ledger drops
/// the OLDEST entries (a deterministic, NO-model, byte-identical policy — never
/// summarize). This constant MUST be identical across all four language
/// implementations.
pub const STEP_LEDGER_MAX_ENTRIES: usize = 20;

/// A single compact row in the Tier-2 global running ledger (#126). Appended on
/// task completion and injected into EVERY subsequent execute step so each step
/// sees a compact record of what has already happened across the whole DAG.
///
/// `files_touched` is **harness-observed from write/edit tool calls** during the
/// task's execution — NOT a model-self-reported field. A task whose execute step
/// only *narrated* touching a file (no actual write/edit tool call) records an
/// EMPTY `files_touched` (#126 AC2).
///
/// Wire field order is `task_id, summary, files_touched`. Byte-identical across
/// all four languages (the same serde shape as [`Task`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepLedgerEntry {
    pub task_id: u32,
    pub summary: String,
    pub files_touched: Vec<String>,
}

/// The static marker row inserted (once) when the Tier-2 ledger drops its oldest
/// entries past [`STEP_LEDGER_MAX_ENTRIES`] (#126, decision B). It is a fixed
/// string — NOT a summary of the elided rows — so the elision note is
/// byte-identical across languages. Rendered as a leading ledger line by
/// [`render_step_ledger`].
pub const STEP_LEDGER_ELISION_MARKER: &str = "[older ledger entries elided]";

/// Append `entry` to a bounded Tier-2 running ledger, keeping at most
/// [`STEP_LEDGER_MAX_ENTRIES`] rows via **drop-oldest** (#126, decision B).
///
/// Pure and deterministic: NO model call, NO summarization. When the push would
/// exceed the bound the oldest entries are removed from the front so exactly the
/// `STEP_LEDGER_MAX_ENTRIES` most-recent entries remain (completion order
/// preserved). Returns `true` iff at least one entry was dropped on this call
/// (so the caller can render the static [`STEP_LEDGER_ELISION_MARKER`]).
pub fn push_step_ledger(ledger: &mut Vec<StepLedgerEntry>, entry: StepLedgerEntry) -> bool {
    ledger.push(entry);
    let mut dropped = false;
    while ledger.len() > STEP_LEDGER_MAX_ENTRIES {
        ledger.remove(0);
        dropped = true;
    }
    dropped
}

/// Render the Tier-2 ledger as a compact, deterministic text block for seeding
/// into an execute step (#126). One line per entry: `#<id> <summary> [files:
/// a, b]` (the `[files: …]` suffix omitted when `files_touched` is empty). When
/// `elided` is true a single leading [`STEP_LEDGER_ELISION_MARKER`] line is
/// emitted. Returns `None` for an empty ledger (nothing to inject).
pub fn render_step_ledger(ledger: &[StepLedgerEntry], elided: bool) -> Option<String> {
    if ledger.is_empty() {
        return None;
    }
    let mut lines: Vec<String> = Vec::new();
    if elided {
        lines.push(STEP_LEDGER_ELISION_MARKER.to_string());
    }
    for e in ledger {
        if e.files_touched.is_empty() {
            lines.push(format!("#{} {}", e.task_id, e.summary));
        } else {
            lines.push(format!(
                "#{} {} [files: {}]",
                e.task_id,
                e.summary,
                e.files_touched.join(", ")
            ));
        }
    }
    Some(format!("Progress ledger so far:\n{}", lines.join("\n")))
}

// ============================================================================
// Errors
// ============================================================================

/// Why an `add_task` blockers set was rejected (#118). Internally tagged on
/// `reason` (snake_case), matching the tagging discipline of [`TaskListError`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum BlockerRejection {
    /// A blocker referenced the id about to be assigned to this very task.
    SelfBlock,
    /// A blocker referenced an id that matches no existing task.
    UnknownId { blocker: u32 },
    /// The new blocker edges would close a directed cycle in the graph.
    Cycle,
}

impl std::fmt::Display for BlockerRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockerRejection::SelfBlock => write!(f, "a task cannot block itself"),
            BlockerRejection::UnknownId { blocker } => {
                write!(f, "unknown blocker id: {blocker}")
            }
            BlockerRejection::Cycle => write!(f, "blocker edges would create a cycle"),
        }
    }
}

/// Errors raised by task-list mutations. `#[non_exhaustive]` per crate
/// conventions. Every variant is recoverable at the tool boundary.
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

    /// The blockers supplied to `add_task` for the task about to be assigned
    /// `id` were rejected (self-block, unknown id, or cycle). The list is left
    /// untouched. (#118)
    #[error("invalid blockers for task {id}: {reason}")]
    InvalidBlockers { id: u32, reason: BlockerRejection },
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
    ///
    /// Fallible since #118: `blockers` are validated BEFORE any mutation, so a
    /// rejected blocker set leaves the list completely untouched (mirroring how
    /// [`TaskList::update`] validates before writing). R9. Validation order:
    /// 1. self-block — a blocker equal to the id about to be assigned
    ///    (`next_id`) → [`BlockerRejection::SelfBlock`].
    /// 2. unknown id — a blocker matching no existing task id →
    ///    [`BlockerRejection::UnknownId`].
    /// 3. cycle — the new edges would close a directed cycle, checked by
    ///    [`would_create_cycle`] → [`BlockerRejection::Cycle`].
    ///
    /// Empty `blockers` always pass (and serialize as `[]`).
    pub fn add(&mut self, description: String, blockers: Vec<u32>) -> Result<u32, TaskListError> {
        let id = self.next_id;

        for &blocker in &blockers {
            if blocker == id {
                return Err(TaskListError::InvalidBlockers {
                    id,
                    reason: BlockerRejection::SelfBlock,
                });
            }
            if !self.tasks.iter().any(|t| t.id == blocker) {
                return Err(TaskListError::InvalidBlockers {
                    id,
                    reason: BlockerRejection::UnknownId { blocker },
                });
            }
        }

        if self.would_create_cycle(id, &blockers) {
            return Err(TaskListError::InvalidBlockers {
                id,
                reason: BlockerRejection::Cycle,
            });
        }

        self.tasks.push(Task {
            id,
            description,
            status: TaskStatus::Pending,
            blockers,
        });
        self.next_id += 1;
        Ok(id)
    }

    /// Would adding a node `new_id` whose outgoing blocker edges are
    /// `new_blockers` close a directed cycle in the blockers graph?
    ///
    /// The graph is `task -> blocker` (a task points at each id it is blocked
    /// by). A cycle exists if, starting from any of the new edges' targets, a
    /// directed path leads back to `new_id`. Since a single append-only `add`
    /// only references EARLIER real ids, this can never actually fire today; the
    /// helper exists as a spec acceptance criterion (#118) and is unit-tested
    /// directly against a hand-built cyclic graph.
    fn would_create_cycle(&self, new_id: u32, new_blockers: &[u32]) -> bool {
        use std::collections::HashSet;

        let mut stack: Vec<u32> = new_blockers.to_vec();
        let mut visited: HashSet<u32> = HashSet::new();

        while let Some(node) = stack.pop() {
            if node == new_id {
                return true;
            }
            if !visited.insert(node) {
                continue;
            }
            if let Some(task) = self.tasks.iter().find(|t| t.id == node) {
                stack.extend(task.blockers.iter().copied());
            }
        }
        false
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

    // ========================================================================
    // DAG scheduling helpers (#126) — pure, deterministic, no I/O.
    // ========================================================================

    /// The next READY task to run under the ready-set scheduler (#126): the
    /// LOWEST-`id` [`Pending`](TaskStatus::Pending) task ALL of whose `blockers`
    /// are [`Completed`](TaskStatus::Completed). Returns its `id`, or `None` when
    /// no Pending task is currently runnable (every Pending task is waiting on an
    /// un-completed blocker, or none remain).
    ///
    /// The id tiebreak is deterministic and language-agnostic: among ready
    /// tasks, the smallest id always wins, regardless of positional order in
    /// [`tasks`](TaskList::tasks).
    pub fn next_ready(&self) -> Option<u32> {
        self.tasks
            .iter()
            .filter(|t| {
                t.status == TaskStatus::Pending && t.blockers.iter().all(|b| self.is_completed(*b))
            })
            .map(|t| t.id)
            .min()
    }

    /// Whether the task with `id` exists and is [`Completed`](TaskStatus::Completed).
    fn is_completed(&self, id: u32) -> bool {
        self.tasks
            .iter()
            .any(|t| t.id == id && t.status == TaskStatus::Completed)
    }

    /// The TRANSITIVE-blocker closure of `id` (#126, Tier-1 scoped context): the
    /// set of all tasks `id` (transitively) depends on — its direct `blockers`,
    /// their blockers, and so on. Excludes `id` itself. Returned SORTED ascending
    /// for deterministic, byte-identical seeding across languages.
    ///
    /// Independent branches (tasks NOT reachable from `id` via the
    /// `task -> blocker` edges) never appear, which is what keeps a task's
    /// Tier-1 seed free of unrelated branches (#126 AC1 branch isolation).
    pub fn transitive_blockers(&self, id: u32) -> Vec<u32> {
        use std::collections::HashSet;
        let mut seen: HashSet<u32> = HashSet::new();
        let mut stack: Vec<u32> = self
            .tasks
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.blockers.clone())
            .unwrap_or_default();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            if let Some(t) = self.tasks.iter().find(|t| t.id == node) {
                stack.extend(t.blockers.iter().copied());
            }
        }
        let mut out: Vec<u32> = seen.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// The TRANSITIVE-dependent closure of `id` (#126, failure cascade): every
    /// task that depends on `id` directly or transitively (i.e. has a directed
    /// `task -> blocker` path reaching `id`). Excludes `id` itself. Returned
    /// SORTED ascending. When `id` fails terminally, exactly these tasks are
    /// marked [`Blocked`](TaskStatus::Blocked); tasks NOT in this set are
    /// unaffected and keep scheduling (#126 AC3).
    pub fn transitive_dependents(&self, id: u32) -> Vec<u32> {
        use std::collections::HashSet;
        // Reverse reachability: repeatedly add any task whose blockers intersect
        // the growing dependent set (seeded with `id`).
        let mut dependents: HashSet<u32> = HashSet::new();
        let mut frontier: Vec<u32> = vec![id];
        while let Some(node) = frontier.pop() {
            for t in &self.tasks {
                if t.id == id {
                    continue;
                }
                if t.blockers.contains(&node) && dependents.insert(t.id) {
                    frontier.push(t.id);
                }
            }
        }
        let mut out: Vec<u32> = dependents.into_iter().collect();
        out.sort_unstable();
        out
    }

    /// Whole-graph cycle detection over the `task -> blocker` edges (#126,
    /// defense-in-depth at execute entry). Generalizes the single-edge
    /// [`would_create_cycle`](TaskList::would_create_cycle) check used by
    /// [`add`](TaskList::add): it inspects the ENTIRE persisted graph (which the
    /// `task_list` tool authoring path could in principle leave cyclic via
    /// out-of-band edits) rather than one prospective `add`.
    ///
    /// Returns `true` iff any directed cycle exists. Pure; O(V+E) via DFS
    /// three-coloring. Blockers referencing unknown ids are simply ignored (no
    /// edge), matching `next_ready`'s "an unknown blocker is never completed"
    /// — but an unknown blocker also cannot form a cycle.
    pub fn has_cycle(&self) -> bool {
        use std::collections::{HashMap, HashSet};
        // 0 = unvisited, 1 = on-stack (gray), 2 = done (black).
        let mut color: HashMap<u32, u8> = HashMap::new();
        // Iterative DFS so a pathological graph never blows the stack.
        for start in self.tasks.iter().map(|t| t.id) {
            if color.get(&start).copied().unwrap_or(0) != 0 {
                continue;
            }
            // Each stack frame: (node, whether we've already pushed its children).
            let mut stack: Vec<(u32, bool)> = vec![(start, false)];
            let mut on_path: HashSet<u32> = HashSet::new();
            while let Some((node, expanded)) = stack.pop() {
                if expanded {
                    color.insert(node, 2);
                    on_path.remove(&node);
                    continue;
                }
                if color.get(&node).copied().unwrap_or(0) == 2 {
                    continue;
                }
                color.insert(node, 1);
                on_path.insert(node);
                stack.push((node, true));
                if let Some(t) = self.tasks.iter().find(|t| t.id == node) {
                    for &b in &t.blockers {
                        // An edge to a non-existent task is no edge at all.
                        if !self.tasks.iter().any(|x| x.id == b) {
                            continue;
                        }
                        if on_path.contains(&b) {
                            return true; // back-edge → cycle.
                        }
                        if color.get(&b).copied().unwrap_or(0) != 2 {
                            stack.push((b, false));
                        }
                    }
                }
            }
        }
        false
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
///
/// # Deprecated (#126, decision C)
/// This flat bridge can only ever produce a LINEAR chain (it always sets
/// `blockers: vec![]`), so it cannot express the blocker DAG the #126 ready-set
/// executor walks. The `task_list` tool path
/// ([`TaskListTool`](crate::tools::TaskListTool), which calls
/// [`TaskList::add`] with real blockers) is now the ONE authoring path; the
/// PlanExecute executor reads its [`TaskList`] from the persisted `task_list`
/// store rather than from this bridge. The function is RETAINED (its replay
/// tests stay green) but should not be used to seed a DAG run.
#[deprecated(
    since = "0.1.0",
    note = "linear-only bridge; author the task list via the `task_list` tool (TaskList::add with blockers). See #126."
)]
pub fn plan_artifact_to_task_list(artifact: &PlanArtifact) -> TaskList {
    let mut list = TaskList::default(); // next_id == 1
    for step in &artifact.tasks {
        // verbatim; appends Pending; bumps next_id. `blockers: vec![]` can never
        // reject, so `add` is always `Ok` here and the parser stays total (no
        // `Result`, never panics in practice). (#118)
        let added = list.add(step.clone(), Vec::new());
        debug_assert!(added.is_ok(), "empty blockers must never reject");
    }
    list
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(deprecated)] // exercises the retained-for-compat `plan_artifact_to_task_list` bridge (#126 C).
mod tests {
    use super::*;

    fn list_with(statuses: &[TaskStatus]) -> TaskList {
        let mut l = TaskList::default();
        for _ in statuses {
            l.add("t".into(), Vec::new()).unwrap();
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
        assert_eq!(l.add("a".into(), Vec::new()).unwrap(), 1);
        assert_eq!(l.add("b".into(), Vec::new()).unwrap(), 2);
        assert_eq!(l.add("c".into(), Vec::new()).unwrap(), 3);
        assert_eq!(l.next_id, 4);
        assert_eq!(l.tasks.iter().map(|t| t.id).collect::<Vec<_>>(), [1, 2, 3]);
    }

    // R2: add appends to the end, preserving positional order.
    #[test]
    fn add_appends_in_order() {
        let mut l = TaskList::default();
        l.add("first".into(), Vec::new()).unwrap();
        l.add("second".into(), Vec::new()).unwrap();
        l.add("third".into(), Vec::new()).unwrap();
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
        l.add("a".into(), Vec::new()).unwrap();
        l.add("b".into(), Vec::new()).unwrap();
        let json = serde_json::to_string(&l).unwrap();
        let mut reloaded: TaskList = serde_json::from_str(&json).unwrap();
        assert_eq!(reloaded.next_id, 3);
        assert_eq!(reloaded.add("c".into(), Vec::new()).unwrap(), 3); // continues from 3, not 1.
    }

    // Serde round-trip is byte-identical (re-serializing the parsed form).
    #[test]
    fn serde_round_trip_byte_identical() {
        let mut l = TaskList::default();
        l.add("alpha".into(), Vec::new()).unwrap();
        l.add("beta".into(), Vec::new()).unwrap();
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
        l.add("write tests".into(), Vec::new()).unwrap();
        l.update(1, Some(TaskStatus::InProgress), None).unwrap();
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"tasks":[{"id":1,"description":"write tests","status":"in_progress","blockers":[]}],"next_id":2}"#
        );
    }

    // ========================================================================
    // blockers (#118)
    // ========================================================================

    // Happy path: blockers referencing earlier real ids are accepted and stored.
    #[test]
    fn add_with_valid_blockers_ok() {
        let mut l = TaskList::default();
        assert_eq!(l.add("a".into(), Vec::new()).unwrap(), 1);
        assert_eq!(l.add("b".into(), Vec::new()).unwrap(), 2);
        let id = l.add("c".into(), vec![1, 2]).unwrap();
        assert_eq!(id, 3);
        assert_eq!(l.tasks[2].blockers, vec![1, 2]);
        assert_eq!(l.next_id, 4);
    }

    // Empty blockers never reject and store as an empty Vec.
    #[test]
    fn add_with_empty_blockers_ok() {
        let mut l = TaskList::default();
        l.add("a".into(), Vec::new()).unwrap();
        assert!(l.tasks[0].blockers.is_empty());
    }

    // Self-block: a blocker equal to the about-to-be-assigned id is rejected.
    #[test]
    fn self_block_rejected() {
        let mut l = TaskList::default();
        // next_id is 1, blocker 1 == self.
        let err = l.add("a".into(), vec![1]).unwrap_err();
        assert_eq!(
            err,
            TaskListError::InvalidBlockers {
                id: 1,
                reason: BlockerRejection::SelfBlock
            }
        );
    }

    // Unknown id: a blocker matching no existing task is rejected.
    #[test]
    fn unknown_blocker_id_rejected() {
        let mut l = TaskList::default();
        l.add("a".into(), Vec::new()).unwrap(); // id 1
        let err = l.add("b".into(), vec![99]).unwrap_err();
        assert_eq!(
            err,
            TaskListError::InvalidBlockers {
                id: 2,
                reason: BlockerRejection::UnknownId { blocker: 99 }
            }
        );
    }

    // A rejected add leaves the list completely untouched (R9, mirrors update).
    #[test]
    fn rejected_blockers_do_not_mutate() {
        let mut l = TaskList::default();
        l.add("a".into(), Vec::new()).unwrap();
        let before = l.clone();
        let _ = l.add("b".into(), vec![99]).unwrap_err();
        assert_eq!(l, before);
        // next_id did NOT advance.
        assert_eq!(l.next_id, 2);
    }

    // Self-block takes precedence over unknown-id when both are present
    // (self-block is checked first per the documented order).
    #[test]
    fn self_block_checked_before_unknown() {
        let mut l = TaskList::default();
        // next_id 1: vec contains self (1) and an unknown (99); self wins.
        let err = l.add("a".into(), vec![1, 99]).unwrap_err();
        assert_eq!(
            err,
            TaskListError::InvalidBlockers {
                id: 1,
                reason: BlockerRejection::SelfBlock
            }
        );
    }

    // would_create_cycle: tested directly against a hand-built cyclic graph,
    // since an append-only add can never close a cycle on its own.
    #[test]
    fn would_create_cycle_detects_back_edge() {
        // Build edges (task -> blocker): 3 -> 2, 2 -> 1. So from node 3 there is
        // a directed path 3 -> 2 -> 1 reaching node 1.
        let mut l = TaskList::default();
        l.add("a".into(), Vec::new()).unwrap(); // 1
        l.add("b".into(), Vec::new()).unwrap(); // 2
        l.add("c".into(), Vec::new()).unwrap(); // 3
        l.tasks[2].blockers = vec![2]; // 3 -> 2
        l.tasks[1].blockers = vec![1]; // 2 -> 1
                                       // Re-adding node 1 with a blocker on 3 closes 1 -> 3 -> 2 -> 1.
        assert!(l.would_create_cycle(1, &[3]));
        // Node 4 with blocker 3 has no path back to 4, so no cycle.
        assert!(!l.would_create_cycle(4, &[3]));
    }

    // would_create_cycle: a direct self-edge is a cycle.
    #[test]
    fn would_create_cycle_self_edge() {
        let l = TaskList::default();
        assert!(l.would_create_cycle(5, &[5]));
    }

    // would_create_cycle: empty new edges are never a cycle.
    #[test]
    fn would_create_cycle_empty_is_false() {
        let l = TaskList::default();
        assert!(!l.would_create_cycle(1, &[]));
    }

    // The cycle path of `add` rejects when the helper reports a cycle. We build
    // a graph where re-adding an existing id with a back-edge would cycle. Since
    // a normal add uses a fresh next_id, we exercise the cycle branch via a
    // crafted state: task 2 already blocks on a future id equal to next_id.
    #[test]
    fn add_rejects_cycle() {
        let mut l = TaskList::default();
        l.add("a".into(), Vec::new()).unwrap(); // id 1
        l.add("b".into(), Vec::new()).unwrap(); // id 2
                                                // Make task 1 depend on id 3 (the next id about to be assigned).
        l.tasks[0].blockers = vec![3];
        // Now add task 3 blocked by 1: path 3 -> 1 -> 3 is a cycle.
        let err = l.add("c".into(), vec![1]).unwrap_err();
        assert_eq!(
            err,
            TaskListError::InvalidBlockers {
                id: 3,
                reason: BlockerRejection::Cycle
            }
        );
    }

    // Non-empty blockers serialize as the LAST field, byte-exact.
    #[test]
    fn blockers_serialize_last_and_exact() {
        let mut l = TaskList::default();
        l.add("a".into(), Vec::new()).unwrap();
        l.add("b".into(), vec![1]).unwrap();
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"tasks":[{"id":1,"description":"a","status":"pending","blockers":[]},{"id":2,"description":"b","status":"pending","blockers":[1]}],"next_id":3}"#
        );
    }

    // Backward-compat: a pre-#118 blob WITHOUT a blockers key still loads, with
    // blockers defaulting to an empty Vec.
    #[test]
    fn deserializes_pre_118_blob_without_blockers() {
        let json = r#"{"tasks":[{"id":1,"description":"old","status":"pending"}],"next_id":2}"#;
        let l: TaskList = serde_json::from_str(json).unwrap();
        assert_eq!(l.tasks.len(), 1);
        assert!(l.tasks[0].blockers.is_empty());
        // Re-serializing now emits the canonical form WITH blockers:[].
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"tasks":[{"id":1,"description":"old","status":"pending","blockers":[]}],"next_id":2}"#
        );
    }

    // BlockerRejection serde tag is snake_case on `reason`.
    #[test]
    fn blocker_rejection_serde_tags() {
        assert_eq!(
            serde_json::to_string(&BlockerRejection::SelfBlock).unwrap(),
            r#"{"reason":"self_block"}"#
        );
        assert_eq!(
            serde_json::to_string(&BlockerRejection::UnknownId { blocker: 7 }).unwrap(),
            r#"{"reason":"unknown_id","blocker":7}"#
        );
        assert_eq!(
            serde_json::to_string(&BlockerRejection::Cycle).unwrap(),
            r#"{"reason":"cycle"}"#
        );
    }

    // TaskListError::InvalidBlockers wire tag is snake_case `invalid_blockers`.
    #[test]
    fn invalid_blockers_error_wire_tag() {
        let e = TaskListError::InvalidBlockers {
            id: 3,
            reason: BlockerRejection::SelfBlock,
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains(r#""kind":"invalid_blockers""#), "{json}");
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
            r#"{"tasks":[{"id":1,"description":"alpha","status":"pending","blockers":[]},{"id":2,"description":"beta","status":"pending","blockers":[]}],"next_id":3}"#
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

    // ========================================================================
    // #126 DAG helpers + step ledger
    // ========================================================================

    fn dag() -> TaskList {
        // 1 (root) ; 2 -> 1 ; 3 -> 1 ; 4 -> 2,3 (diamond) ; 5 (independent).
        let mut l = TaskList::default();
        l.add("a".into(), vec![]).unwrap(); // 1
        l.add("b".into(), vec![1]).unwrap(); // 2
        l.add("c".into(), vec![1]).unwrap(); // 3
        l.add("d".into(), vec![2, 3]).unwrap(); // 4
        l.add("e".into(), vec![]).unwrap(); // 5
        l
    }

    // next_ready: lowest-id Pending task with all blockers Completed.
    #[test]
    fn next_ready_respects_blockers_and_id_tiebreak() {
        let mut l = dag();
        // Initially tasks 1 and 5 are ready (no blockers); lowest id wins → 1.
        assert_eq!(l.next_ready(), Some(1));
        l.complete(1).unwrap();
        // Now 2, 3, 5 are ready (4 still waits on 2 & 3). Lowest is 2.
        assert_eq!(l.next_ready(), Some(2));
        l.complete(2).unwrap();
        // 3 and 5 ready; 4 still needs 3. Lowest is 3.
        assert_eq!(l.next_ready(), Some(3));
        l.complete(3).unwrap();
        // Now 4 and 5 ready; lowest is 4.
        assert_eq!(l.next_ready(), Some(4));
        l.complete(4).unwrap();
        assert_eq!(l.next_ready(), Some(5));
        l.complete(5).unwrap();
        assert_eq!(l.next_ready(), None);
    }

    // A Pending task whose blocker is NOT completed is never ready.
    #[test]
    fn next_ready_skips_unsatisfied_blockers() {
        let l = dag();
        // Only 1 and 5 are runnable; 2/3/4 wait. next_ready is the min of {1,5}.
        assert_eq!(l.next_ready(), Some(1));
    }

    // transitive_blockers: the full upstream closure, sorted, excluding self.
    #[test]
    fn transitive_blockers_closure() {
        let l = dag();
        assert_eq!(l.transitive_blockers(1), Vec::<u32>::new());
        assert_eq!(l.transitive_blockers(2), vec![1]);
        assert_eq!(l.transitive_blockers(4), vec![1, 2, 3]);
        // Independent task has no blockers.
        assert_eq!(l.transitive_blockers(5), Vec::<u32>::new());
    }

    // transitive_dependents: the full downstream closure, sorted, excluding self.
    #[test]
    fn transitive_dependents_closure() {
        let l = dag();
        // Everything downstream of 1: 2, 3, 4.
        assert_eq!(l.transitive_dependents(1), vec![2, 3, 4]);
        assert_eq!(l.transitive_dependents(2), vec![4]);
        assert_eq!(l.transitive_dependents(3), vec![4]);
        assert_eq!(l.transitive_dependents(4), Vec::<u32>::new());
        // Independent task has no dependents.
        assert_eq!(l.transitive_dependents(5), Vec::<u32>::new());
    }

    // has_cycle: acyclic graphs return false; a hand-built cycle returns true.
    #[test]
    fn has_cycle_detects_cycles() {
        let l = dag();
        assert!(!l.has_cycle(), "diamond DAG is acyclic");

        // Build a cycle out of band: 1 -> 2 -> 1.
        let mut c = TaskList::default();
        c.add("a".into(), vec![]).unwrap(); // 1
        c.add("b".into(), vec![1]).unwrap(); // 2 -> 1
        c.tasks[0].blockers = vec![2]; // 1 -> 2 closes the cycle
        assert!(c.has_cycle());

        // A self-edge is a cycle.
        let mut s = TaskList::default();
        s.add("x".into(), vec![]).unwrap();
        s.tasks[0].blockers = vec![1];
        assert!(s.has_cycle());
    }

    // has_cycle: an edge to a non-existent task is ignored (no false positive).
    #[test]
    fn has_cycle_ignores_unknown_blockers() {
        let mut l = TaskList::default();
        l.add("a".into(), vec![]).unwrap();
        l.tasks[0].blockers = vec![99]; // dangling edge, not a cycle
        assert!(!l.has_cycle());
    }

    // push_step_ledger: drop-oldest past N, returns whether anything was dropped.
    #[test]
    fn step_ledger_drop_oldest_past_n() {
        let mut ledger: Vec<StepLedgerEntry> = Vec::new();
        // Push exactly N entries: no drop.
        for i in 0..STEP_LEDGER_MAX_ENTRIES as u32 {
            let dropped = push_step_ledger(
                &mut ledger,
                StepLedgerEntry {
                    task_id: i + 1,
                    summary: format!("s{i}"),
                    files_touched: vec![],
                },
            );
            assert!(!dropped, "no drop before exceeding N");
        }
        assert_eq!(ledger.len(), STEP_LEDGER_MAX_ENTRIES);
        // One more: drop-oldest fires.
        let dropped = push_step_ledger(
            &mut ledger,
            StepLedgerEntry {
                task_id: 999,
                summary: "newest".into(),
                files_touched: vec![],
            },
        );
        assert!(dropped, "drop fires at N+1");
        assert_eq!(ledger.len(), STEP_LEDGER_MAX_ENTRIES, "stays bounded at N");
        // The OLDEST (task_id 1) was dropped; the newest is retained.
        assert!(ledger.iter().all(|e| e.task_id != 1));
        assert_eq!(ledger.last().unwrap().task_id, 999);
    }

    // render_step_ledger: deterministic text; files suffix only when non-empty;
    // elision marker leads when requested; None for empty.
    #[test]
    fn render_step_ledger_shape() {
        assert_eq!(render_step_ledger(&[], false), None);
        let entries = vec![
            StepLedgerEntry {
                task_id: 1,
                summary: "scaffolded".into(),
                files_touched: vec![],
            },
            StepLedgerEntry {
                task_id: 2,
                summary: "wrote tests".into(),
                files_touched: vec!["a.rs".into(), "b.rs".into()],
            },
        ];
        let rendered = render_step_ledger(&entries, false).unwrap();
        assert!(rendered.contains("#1 scaffolded"));
        assert!(
            !rendered.contains("#1 scaffolded [files"),
            "no empty files suffix"
        );
        assert!(rendered.contains("#2 wrote tests [files: a.rs, b.rs]"));
        assert!(!rendered.contains(STEP_LEDGER_ELISION_MARKER));
        // With elision the static marker leads.
        let elided = render_step_ledger(&entries, true).unwrap();
        assert!(elided.contains(STEP_LEDGER_ELISION_MARKER));
    }

    // StepLedgerEntry serde field order is task_id, summary, files_touched.
    #[test]
    fn step_ledger_entry_serde_shape() {
        let e = StepLedgerEntry {
            task_id: 7,
            summary: "did".into(),
            files_touched: vec!["x".into()],
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"task_id":7,"summary":"did","files_touched":["x"]}"#
        );
    }
}
