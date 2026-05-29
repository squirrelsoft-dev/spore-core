// Persisted task-list primitive — PlanExecute, drives the execute loop (#71).
//
// Decomposed out of #59. The accepted plan (#70) is parsed into a persisted
// task list (#72), and the execute phase (#59) loops over the tasks until the
// list is complete. This file owns the task-list primitive that holds and
// mutates that list plus its disk-persistence helpers. The single mutating
// tool over it lives in tools/tasklist.go. It is consumed by #72 (which
// populates the list) and #59 (whose execute loop drains it).
//
// Types (serializable, byte-identical across all four languages):
//   - TaskStatus — "pending" | "in_progress" | "completed" | "blocked".
//   - TaskListItem — {id uint32, description string, status TaskStatus}. A flat
//     record: NO hierarchy/subtasks, NO timestamps (byte-identity constraint),
//     NO order field (order is positional in TaskList.Tasks). JSON field order
//     is id, description, status. (Named TaskListItem rather than Task because
//     the package already has a Task type — the harness run input.)
//   - TaskList — {tasks []TaskListItem, next_id uint32}, serialized as
//     {"tasks":[...],"next_id":N}. DefaultTaskList() is empty with next_id == 1.
//     next_id is monotonic and never reused; it is preserved across reload.
//   - TaskListError — recoverable domain error (TaskNotFound / InvalidTransition).
//     Maps to a recoverable ToolOutput at the tool boundary; never a panic.
//
// ID scheme: sequential uint32, 1-based, minted monotonically from
// TaskList.NextID. Ids are never reused — NextID only grows, and it survives a
// reload so a freshly-loaded list keeps minting fresh ids.
//
// Rules enforced:
//   - R1  Ids are assigned 1, 2, 3, … monotonically from NextID; never reused.
//   - R2  Add APPENDS to the end of Tasks (positional order is stable).
//   - R3  list_tasks never mutates the list (no mutating helper is invoked).
//   - R4  Unknown id on Update/Complete → TaskNotFound (recoverable).
//   - R5  Status transitions follow ValidateTransition (DECISION 1,
//     "permissive-except-terminal-Completed"). A rejected transition →
//     InvalidTransition.
//   - R6  Completed is terminal: ANY transition OUT of Completed is rejected
//     (the idempotent Completed → Completed self-transition is allowed).
//   - R7  Self-transitions X → X are idempotent and always allowed.
//   - R8  Persistence is through the storage seam (#75): the standalone tool
//     (tools/tasklist.go) persists via the RunStore on the *ToolContext, keyed
//     by SessionID under TaskListExtrasKey. The retired interim sandbox path
//     (.spore/task_list.json) is GONE; with the library's default no-op storage
//     a standalone tool call persists nothing across processes (an accepted
//     behavior change — no migration shim). #76's PlanExecute execute loop
//     shares the same RunStore key.
//
// Both design forks (transition matrix, state seam) were resolved before
// implementation; there are no open spec questions here.

package sporecore

import (
	"encoding/json"
	"fmt"
)

// TaskListExtrasKey is the key under which the TaskList is persisted in the
// RunStore, keyed by SessionID. Both the harness-side #76 execute loop and the
// standalone TaskListTool (tools/tasklist.go, #75) share this single key, so a
// standalone tool call and a PlanExecute run on the same session intentionally
// share one blob. Stable across all four languages. The JSON shape is the
// canonical TaskList serialization ({"tasks":[...],"next_id":N}).
const TaskListExtrasKey = "task_list"

// ============================================================================
// Types
// ============================================================================

// TaskStatus is the lifecycle status of a Task. Serializes to snake_case.
type TaskStatus string

const (
	TaskStatusPending    TaskStatus = "pending"
	TaskStatusInProgress TaskStatus = "in_progress"
	TaskStatusCompleted  TaskStatus = "completed"
	TaskStatusBlocked    TaskStatus = "blocked"
)

// TaskListItem is a single task: flat, no hierarchy, no timestamps, no order
// field (order is positional in TaskList.Tasks). JSON field order is id,
// description, status — kept byte-identical to the other languages. Named
// TaskListItem to avoid colliding with the package's Task (harness run input).
type TaskListItem struct {
	ID          uint32     `json:"id"`
	Description string     `json:"description"`
	Status      TaskStatus `json:"status"`
}

// TaskList is the persisted collection of tasks plus the monotonic id counter.
//
// Serializes as {"tasks":[...],"next_id":N}. The Tasks slice is rendered as an
// empty array (never null) for the empty list so the default form matches the
// pinned {"tasks":[],"next_id":1}.
type TaskList struct {
	Tasks  []TaskListItem `json:"tasks"`
	NextID uint32         `json:"next_id"`
}

// DefaultTaskList returns the canonical empty list: no tasks, next_id == 1.
func DefaultTaskList() TaskList {
	return TaskList{Tasks: []TaskListItem{}, NextID: 1}
}

// MarshalJSON renders Tasks as [] rather than null when empty, so the
// serialization is byte-identical to the other languages' {"tasks":[],...}.
func (l TaskList) MarshalJSON() ([]byte, error) {
	type alias TaskList // avoid recursion
	a := alias(l)
	if a.Tasks == nil {
		a.Tasks = []TaskListItem{}
	}
	return json.Marshal(a)
}

// ============================================================================
// Errors
// ============================================================================

// TaskListErrorKind discriminates TaskListError variants. Tag values are
// snake_case to match the Rust enum on the wire.
type TaskListErrorKind string

const (
	// TaskListErrTaskNotFound: no task with the given id exists in the list.
	TaskListErrTaskNotFound TaskListErrorKind = "task_not_found"
	// TaskListErrInvalidTransition: the requested status transition is not
	// permitted by ValidateTransition (notably any transition out of completed).
	TaskListErrInvalidTransition TaskListErrorKind = "invalid_transition"
)

// TaskListError is the recoverable domain error raised by task-list mutations.
// Both variants map to a recoverable ToolOutput at the tool boundary.
type TaskListError struct {
	Kind TaskListErrorKind `json:"kind"`
	ID   uint32            `json:"id"`
	// From / To are populated for InvalidTransition only.
	From TaskStatus `json:"from,omitempty"`
	To   TaskStatus `json:"to,omitempty"`
}

// Error implements the error interface.
func (e *TaskListError) Error() string {
	switch e.Kind {
	case TaskListErrTaskNotFound:
		return fmt.Sprintf("task not found: %d", e.ID)
	case TaskListErrInvalidTransition:
		return fmt.Sprintf("invalid transition for task %d: %s -> %s", e.ID, e.From, e.To)
	default:
		return fmt.Sprintf("task list error (%s): task %d", e.Kind, e.ID)
	}
}

func newTaskNotFound(id uint32) *TaskListError {
	return &TaskListError{Kind: TaskListErrTaskNotFound, ID: id}
}

func newInvalidTransition(id uint32, from, to TaskStatus) *TaskListError {
	return &TaskListError{Kind: TaskListErrInvalidTransition, ID: id, From: from, To: to}
}

// ============================================================================
// Transition matrix (DECISION 1)
// ============================================================================

// ValidateTransition validates a status transition under DECISION 1
// ("permissive-except-terminal-Completed").
//
// Allowed:
//   - any self-transition X → X (idempotent, incl. completed → completed),
//   - pending → in_progress | completed | blocked,
//   - in_progress → completed | blocked,
//   - blocked → in_progress | completed.
//
// Rejected: ANY transition OUT of completed (it is terminal) — except the
// idempotent completed → completed. id is carried only to populate the
// resulting InvalidTransition error.
func ValidateTransition(id uint32, from, to TaskStatus) error {
	// Idempotent self-transition is always allowed (incl. completed -> completed).
	if from == to {
		return nil
	}
	allowed := false
	switch from {
	case TaskStatusPending:
		allowed = to == TaskStatusInProgress || to == TaskStatusCompleted || to == TaskStatusBlocked
	case TaskStatusInProgress:
		allowed = to == TaskStatusCompleted || to == TaskStatusBlocked
	case TaskStatusBlocked:
		allowed = to == TaskStatusInProgress || to == TaskStatusCompleted
	case TaskStatusCompleted:
		// Terminal: only the self-transition handled above is allowed.
		allowed = false
	}
	if allowed {
		return nil
	}
	return newInvalidTransition(id, from, to)
}

// ============================================================================
// TaskList mutation helpers (the seam #72 / #59 will call)
// ============================================================================

// Add appends a new pending task with the next sequential id and returns that
// id. Increments NextID. R1, R2.
func (l *TaskList) Add(description string) uint32 {
	id := l.NextID
	l.Tasks = append(l.Tasks, TaskListItem{
		ID:          id,
		Description: description,
		Status:      TaskStatusPending,
	})
	l.NextID++
	return id
}

// Update updates a task's status and/or description.
//
//   - Unknown id → TaskNotFound.
//   - status non-nil → validated via ValidateTransition then applied.
//   - description non-nil → set verbatim.
//   - Both nil → no-op success.
//
// Status is validated BEFORE any field is written, so a rejected transition
// leaves the task untouched.
func (l *TaskList) Update(id uint32, status *TaskStatus, description *string) error {
	task := l.find(id)
	if task == nil {
		return newTaskNotFound(id)
	}
	if status != nil {
		if err := ValidateTransition(id, task.Status, *status); err != nil {
			return err
		}
		task.Status = *status
	}
	if description != nil {
		task.Description = *description
	}
	return nil
}

// Complete marks a task completed, validating the transition first.
//
//   - Unknown id → TaskNotFound.
//   - Already completed → idempotent success.
func (l *TaskList) Complete(id uint32) error {
	task := l.find(id)
	if task == nil {
		return newTaskNotFound(id)
	}
	if err := ValidateTransition(id, task.Status, TaskStatusCompleted); err != nil {
		return err
	}
	task.Status = TaskStatusCompleted
	return nil
}

// find returns a pointer to the task with the given id, or nil if absent.
func (l *TaskList) find(id uint32) *TaskListItem {
	for i := range l.Tasks {
		if l.Tasks[i].ID == id {
			return &l.Tasks[i]
		}
	}
	return nil
}

// ============================================================================
// Plan → TaskList parser (issue #72; the bridge between #70 and #59)
// ============================================================================

// PlanArtifactToTaskList parses an accepted PlanArtifact (#70) into a fresh,
// ready-to-persist TaskList (#71). This is the bridge between the plan phase
// and the execute loop: once a plan is produced and accepted, its steps become
// the task list that #59's execute loop drains.
//
// Rules enforced:
//   - One TaskListItem per plan step, in plan order (positional, via Add).
//   - Every produced task is pending.
//   - Step descriptions are copied VERBATIM — no trim, no normalize, no filter
//     (even "  spaced  " and "" are kept exactly).
//   - Ids are assigned 1..=n sequentially via the NextID scheme; NextID ends at
//     n + 1.
//   - An empty plan (Tasks == nil or len 0) yields DefaultTaskList —
//     {"tasks":[],"next_id":1}. That is a valid EMPTY list, not an error and
//     not "immediate completion"; the execute loop (#59) decides loop semantics.
//   - Rationale is DROPPED — neither TaskListItem nor TaskList carries it.
//
// Pure and total: it never errors, never panics, performs no I/O, and always
// builds a fresh list (replanning is out of scope — single parse per accepted
// plan), so the same artifact always yields the same task list, byte-identical
// across all four languages. Wiring this into the plan-acceptance seam is
// DEFERRED to #59; #72 ships only this pure function.
func PlanArtifactToTaskList(artifact PlanArtifact) TaskList {
	list := DefaultTaskList() // NextID == 1, Tasks == []
	for _, step := range artifact.Tasks {
		list.Add(step) // verbatim; appends pending; bumps NextID
	}
	return list
}
