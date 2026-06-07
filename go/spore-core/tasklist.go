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
//   - TaskListItem — {id uint32, description string, status TaskStatus,
//     blockers []uint32}. A flat record: NO hierarchy/subtasks, NO timestamps
//     (byte-identity constraint), NO order field (order is positional in
//     TaskList.Tasks). JSON field order is id, description, status, blockers
//     (blockers LAST). (Named TaskListItem rather than Task because the package
//     already has a Task type — the harness run input.) blockers (#118) are ids
//     of other tasks that must be completed before this task runs; an empty
//     blockers set ALWAYS serializes as [] (never null, never omitted), the
//     same treatment as TaskList.Tasks. A pre-#118 blob without the key still
//     deserializes (blockers defaulting to empty).
//   - TaskList — {tasks []TaskListItem, next_id uint32}, serialized as
//     {"tasks":[...],"next_id":N}. DefaultTaskList() is empty with next_id == 1.
//     next_id is monotonic and never reused; it is preserved across reload.
//   - TaskListError — recoverable domain error (TaskNotFound / InvalidTransition
//     / InvalidBlockers). Maps to a recoverable ToolOutput at the tool boundary;
//     never a panic.
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
//   - R9  (#118) Add blockers are validated BEFORE any mutation; a reject leaves
//     the list untouched (mirrors Update). Validation: self-block (a blocker ==
//     the about-to-be-assigned id) → InvalidBlockers{SelfBlock}; unknown id (a
//     blocker matching no existing task) → InvalidBlockers{UnknownID, blocker};
//     cycle (the new edges would close a directed cycle in the blockers graph,
//     checked by wouldCreateCycle) → InvalidBlockers{Cycle}. Empty blockers
//     never reject.
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
	"sort"
	"strings"
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
	// TaskStatusBlocked (#118) means BOTH "waiting on a blocker that has not yet
	// completed" AND "a blocker failed terminally" — the status is the same in
	// either case; the distinction (if any) lives in the scheduler, not the schema.
	TaskStatusBlocked TaskStatus = "blocked"
)

// TaskListItem is a single task: flat, no hierarchy, no timestamps, no order
// field (order is positional in TaskList.Tasks). JSON field order is id,
// description, status, blockers — kept byte-identical to the other languages.
// Named TaskListItem to avoid colliding with the package's Task (harness run
// input).
//
// Blockers (#118) are the ids of other tasks that must be completed before this
// task runs; it is the LAST wire field. A pre-#118 blob without the key still
// deserializes (Blockers defaulting to a nil slice); empty Blockers ALWAYS
// serialize as [] (see MarshalJSON), the same treatment as TaskList.Tasks.
type TaskListItem struct {
	ID          uint32     `json:"id"`
	Description string     `json:"description"`
	Status      TaskStatus `json:"status"`
	Blockers    []uint32   `json:"blockers"`
}

// MarshalJSON renders Blockers as [] rather than null when empty, so the
// serialization is byte-identical to the other languages.
func (i TaskListItem) MarshalJSON() ([]byte, error) {
	type alias TaskListItem // avoid recursion
	a := alias(i)
	if a.Blockers == nil {
		a.Blockers = []uint32{}
	}
	return json.Marshal(a)
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
// Step ledger (#126 — Tier-2 global running ledger)
// ============================================================================

// StepLedgerMaxEntries is the maximum number of StepLedgerEntry rows the Tier-2
// running ledger retains (#126, RESOLVED spec decision B). Past this bound the
// ledger drops the OLDEST entries (a deterministic, NO-model, byte-identical
// policy — never summarize). This constant MUST be identical across all four
// language implementations.
const StepLedgerMaxEntries = 20

// StepLedgerElisionMarker is the static marker row inserted (once) when the
// Tier-2 ledger drops its oldest entries past StepLedgerMaxEntries (#126,
// decision B). It is a fixed string — NOT a summary of the elided rows — so the
// elision note is byte-identical across languages. Rendered as a leading ledger
// line by RenderStepLedger.
const StepLedgerElisionMarker = "[older ledger entries elided]"

// StepLedgerEntry is a single compact row in the Tier-2 global running ledger
// (#126). Appended on task completion and injected into EVERY subsequent execute
// step so each step sees a compact record of what has already happened across
// the whole DAG.
//
// FilesTouched is HARNESS-OBSERVED from write/edit tool calls during the task's
// execution — NOT a model-self-reported field. A task whose execute step only
// NARRATED touching a file (no actual write/edit tool call) records an EMPTY
// FilesTouched (#126 AC2).
//
// Wire field order is task_id, summary, files_touched. Byte-identical across all
// four languages (the same serde shape as TaskListItem).
type StepLedgerEntry struct {
	TaskID       uint32   `json:"task_id"`
	Summary      string   `json:"summary"`
	FilesTouched []string `json:"files_touched"`
}

// MarshalJSON renders FilesTouched as [] rather than null when empty, so the
// serialization is byte-identical to the other languages.
func (e StepLedgerEntry) MarshalJSON() ([]byte, error) {
	type alias StepLedgerEntry // avoid recursion
	a := alias(e)
	if a.FilesTouched == nil {
		a.FilesTouched = []string{}
	}
	return json.Marshal(a)
}

// PushStepLedger appends entry to a bounded Tier-2 running ledger, keeping at
// most StepLedgerMaxEntries rows via drop-oldest (#126, decision B).
//
// Pure and deterministic: NO model call, NO summarization. When the push would
// exceed the bound the oldest entries are removed from the front so exactly the
// StepLedgerMaxEntries most-recent entries remain (completion order preserved).
// Returns the updated ledger and true iff at least one entry was dropped on this
// call (so the caller can render the static StepLedgerElisionMarker).
func PushStepLedger(ledger []StepLedgerEntry, entry StepLedgerEntry) ([]StepLedgerEntry, bool) {
	ledger = append(ledger, entry)
	dropped := false
	for len(ledger) > StepLedgerMaxEntries {
		ledger = ledger[1:]
		dropped = true
	}
	return ledger, dropped
}

// RenderStepLedger renders the Tier-2 ledger as a compact, deterministic text
// block for seeding into an execute step (#126). One line per entry:
// "#<id> <summary> [files: a, b]" (the "[files: …]" suffix omitted when
// FilesTouched is empty). When elided is true a single leading
// StepLedgerElisionMarker line is emitted. Returns ("", false) for an empty
// ledger (nothing to inject).
func RenderStepLedger(ledger []StepLedgerEntry, elided bool) (string, bool) {
	if len(ledger) == 0 {
		return "", false
	}
	var lines []string
	if elided {
		lines = append(lines, StepLedgerElisionMarker)
	}
	for _, e := range ledger {
		if len(e.FilesTouched) == 0 {
			lines = append(lines, fmt.Sprintf("#%d %s", e.TaskID, e.Summary))
		} else {
			lines = append(lines, fmt.Sprintf("#%d %s [files: %s]", e.TaskID, e.Summary, strings.Join(e.FilesTouched, ", ")))
		}
	}
	return "Progress ledger so far:\n" + strings.Join(lines, "\n"), true
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
	// TaskListErrInvalidBlockers (#118): the blockers supplied to Add for the task
	// about to be assigned ID were rejected (self-block, unknown id, or cycle).
	TaskListErrInvalidBlockers TaskListErrorKind = "invalid_blockers"
)

// BlockerRejectionReason discriminates why an Add blockers set was rejected
// (#118). Tag values are snake_case to match the Rust enum on the wire.
type BlockerRejectionReason string

const (
	// BlockerRejectionSelfBlock: a blocker referenced the id about to be assigned
	// to this very task.
	BlockerRejectionSelfBlock BlockerRejectionReason = "self_block"
	// BlockerRejectionUnknownID: a blocker referenced an id matching no existing
	// task; the offending id is carried in BlockerRejection.Blocker.
	BlockerRejectionUnknownID BlockerRejectionReason = "unknown_id"
	// BlockerRejectionCycle: the new blocker edges would close a directed cycle.
	BlockerRejectionCycle BlockerRejectionReason = "cycle"
)

// BlockerRejection explains why a blocker set was rejected by TaskList.Add
// (#118). Internally tagged on `reason` (snake_case), matching the Rust enum:
// SelfBlock and Cycle carry only the reason; UnknownID also carries the
// offending Blocker id.
type BlockerRejection struct {
	Reason BlockerRejectionReason `json:"reason"`
	// Blocker is the offending id, populated for the unknown_id reason only.
	Blocker uint32 `json:"blocker,omitempty"`
}

// String renders the rejection reason for inclusion in the error message.
func (r BlockerRejection) String() string {
	switch r.Reason {
	case BlockerRejectionSelfBlock:
		return "a task cannot block itself"
	case BlockerRejectionUnknownID:
		return fmt.Sprintf("unknown blocker id: %d", r.Blocker)
	case BlockerRejectionCycle:
		return "blocker edges would create a cycle"
	default:
		return string(r.Reason)
	}
}

// TaskListError is the recoverable domain error raised by task-list mutations.
// Both variants map to a recoverable ToolOutput at the tool boundary.
type TaskListError struct {
	Kind TaskListErrorKind `json:"kind"`
	ID   uint32            `json:"id"`
	// From / To are populated for InvalidTransition only.
	From TaskStatus `json:"from,omitempty"`
	To   TaskStatus `json:"to,omitempty"`
	// Reason is populated for InvalidBlockers only (#118).
	Reason *BlockerRejection `json:"reason,omitempty"`
}

// Error implements the error interface.
func (e *TaskListError) Error() string {
	switch e.Kind {
	case TaskListErrTaskNotFound:
		return fmt.Sprintf("task not found: %d", e.ID)
	case TaskListErrInvalidTransition:
		return fmt.Sprintf("invalid transition for task %d: %s -> %s", e.ID, e.From, e.To)
	case TaskListErrInvalidBlockers:
		return fmt.Sprintf("invalid blockers for task %d: %s", e.ID, e.Reason)
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

func newInvalidBlockers(id uint32, reason BlockerRejection) *TaskListError {
	return &TaskListError{Kind: TaskListErrInvalidBlockers, ID: id, Reason: &reason}
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
//
// Fallible since #118: blockers are validated BEFORE any mutation, so a rejected
// blocker set leaves the list completely untouched (mirroring how Update
// validates before writing). R9. Validation order:
//  1. self-block — a blocker equal to the id about to be assigned (NextID) →
//     BlockerRejectionSelfBlock.
//  2. unknown id — a blocker matching no existing task id → BlockerRejectionUnknownID.
//  3. cycle — the new edges would close a directed cycle, checked by
//     wouldCreateCycle → BlockerRejectionCycle.
//
// Empty blockers always pass (and serialize as []).
func (l *TaskList) Add(description string, blockers []uint32) (uint32, error) {
	id := l.NextID

	for _, blocker := range blockers {
		if blocker == id {
			return 0, newInvalidBlockers(id, BlockerRejection{Reason: BlockerRejectionSelfBlock})
		}
		if l.find(blocker) == nil {
			return 0, newInvalidBlockers(id, BlockerRejection{
				Reason:  BlockerRejectionUnknownID,
				Blocker: blocker,
			})
		}
	}

	if l.wouldCreateCycle(id, blockers) {
		return 0, newInvalidBlockers(id, BlockerRejection{Reason: BlockerRejectionCycle})
	}

	l.Tasks = append(l.Tasks, TaskListItem{
		ID:          id,
		Description: description,
		Status:      TaskStatusPending,
		Blockers:    blockers,
	})
	l.NextID++
	return id, nil
}

// wouldCreateCycle reports whether adding a node newID whose outgoing blocker
// edges are newBlockers would close a directed cycle in the blockers graph
// (#118).
//
// The graph is task -> blocker (a task points at each id it is blocked by). A
// cycle exists if, starting from any of the new edges' targets, a directed path
// leads back to newID. Since a single append-only Add only references EARLIER
// real ids, this can never actually fire today; the helper exists as a spec
// acceptance criterion (#118) and is unit-tested directly against a hand-built
// cyclic graph.
func (l *TaskList) wouldCreateCycle(newID uint32, newBlockers []uint32) bool {
	stack := append([]uint32(nil), newBlockers...)
	visited := make(map[uint32]struct{})

	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if node == newID {
			return true
		}
		if _, seen := visited[node]; seen {
			continue
		}
		visited[node] = struct{}{}
		if task := l.find(node); task != nil {
			stack = append(stack, task.Blockers...)
		}
	}
	return false
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
// DAG scheduling helpers (#126) — pure, deterministic, no I/O.
// ============================================================================

// NextReady returns the next READY task to run under the ready-set scheduler
// (#126): the LOWEST-id pending task ALL of whose blockers are completed.
// Returns (id, true), or (0, false) when no pending task is currently runnable
// (every pending task is waiting on an un-completed blocker, or none remain).
//
// The id tiebreak is deterministic and language-agnostic: among ready tasks,
// the smallest id always wins, regardless of positional order in Tasks.
func (l *TaskList) NextReady() (uint32, bool) {
	var (
		best  uint32
		found bool
	)
	for i := range l.Tasks {
		t := &l.Tasks[i]
		if t.Status != TaskStatusPending {
			continue
		}
		ready := true
		for _, b := range t.Blockers {
			if !l.isCompleted(b) {
				ready = false
				break
			}
		}
		if !ready {
			continue
		}
		if !found || t.ID < best {
			best = t.ID
			found = true
		}
	}
	return best, found
}

// isCompleted reports whether the task with id exists and is completed.
func (l *TaskList) isCompleted(id uint32) bool {
	t := l.find(id)
	return t != nil && t.Status == TaskStatusCompleted
}

// TransitiveBlockers returns the TRANSITIVE-blocker closure of id (#126, Tier-1
// scoped context): the set of all tasks id (transitively) depends on — its
// direct blockers, their blockers, and so on. Excludes id itself. Returned
// SORTED ascending for deterministic, byte-identical seeding across languages.
//
// Independent branches (tasks NOT reachable from id via the task -> blocker
// edges) never appear, which keeps a task's Tier-1 seed free of unrelated
// branches (#126 AC1 branch isolation).
func (l *TaskList) TransitiveBlockers(id uint32) []uint32 {
	seen := make(map[uint32]struct{})
	var stack []uint32
	if t := l.find(id); t != nil {
		stack = append(stack, t.Blockers...)
	}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if _, ok := seen[node]; ok {
			continue
		}
		seen[node] = struct{}{}
		if t := l.find(node); t != nil {
			stack = append(stack, t.Blockers...)
		}
	}
	out := make([]uint32, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// TransitiveDependents returns the TRANSITIVE-dependent closure of id (#126,
// failure cascade): every task that depends on id directly or transitively
// (i.e. has a directed task -> blocker path reaching id). Excludes id itself.
// Returned SORTED ascending. When id fails terminally, exactly these tasks are
// marked Blocked; tasks NOT in this set are unaffected and keep scheduling
// (#126 AC3).
func (l *TaskList) TransitiveDependents(id uint32) []uint32 {
	dependents := make(map[uint32]struct{})
	frontier := []uint32{id}
	for len(frontier) > 0 {
		node := frontier[len(frontier)-1]
		frontier = frontier[:len(frontier)-1]
		for i := range l.Tasks {
			t := &l.Tasks[i]
			if t.ID == id {
				continue
			}
			for _, b := range t.Blockers {
				if b != node {
					continue
				}
				if _, ok := dependents[t.ID]; !ok {
					dependents[t.ID] = struct{}{}
					frontier = append(frontier, t.ID)
				}
				break
			}
		}
	}
	out := make([]uint32, 0, len(dependents))
	for k := range dependents {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// HasCycle reports whether the whole-graph blocker edges (task -> blocker)
// contain any directed cycle (#126, defense-in-depth at execute entry). It
// generalizes the single-edge wouldCreateCycle check used by Add: it inspects
// the ENTIRE persisted graph (which the task_list tool authoring path could in
// principle leave cyclic via out-of-band edits) rather than one prospective Add.
//
// Pure; O(V+E) via iterative DFS three-coloring so a pathological graph never
// blows the stack. Blockers referencing unknown ids are simply ignored (no
// edge) — an unknown blocker can never form a cycle.
func (l *TaskList) HasCycle() bool {
	const (
		colorGray  = 1 // on the current DFS path
		colorBlack = 2 // fully explored
	)
	color := make(map[uint32]uint8)
	for i := range l.Tasks {
		start := l.Tasks[i].ID
		if color[start] != 0 {
			continue
		}
		type frame struct {
			node     uint32
			expanded bool
		}
		stack := []frame{{start, false}}
		onPath := make(map[uint32]struct{})
		for len(stack) > 0 {
			fr := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if fr.expanded {
				color[fr.node] = colorBlack
				delete(onPath, fr.node)
				continue
			}
			if color[fr.node] == colorBlack {
				continue
			}
			color[fr.node] = colorGray
			onPath[fr.node] = struct{}{}
			stack = append(stack, frame{fr.node, true})
			if t := l.find(fr.node); t != nil {
				for _, b := range t.Blockers {
					if l.find(b) == nil {
						continue // edge to a non-existent task is no edge at all
					}
					if _, gray := onPath[b]; gray {
						return true // back-edge → cycle
					}
					if color[b] != colorBlack {
						stack = append(stack, frame{b, false})
					}
				}
			}
		}
	}
	return false
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
//
// Deprecated (#126, decision C): this flat bridge can only ever produce a LINEAR
// chain (it always sets empty blockers), so it cannot express the blocker DAG
// the #126 ready-set executor walks. The task_list tool path (TaskList.Add with
// real blockers) is now the ONE authoring path; the PlanExecute executor reads
// its TaskList from the persisted task_list store rather than from this bridge.
// The function is RETAINED (its replay tests stay green) but should not be used
// to seed a DAG run. Mirrors Rust's #[deprecated] on plan_artifact_to_task_list.
func PlanArtifactToTaskList(artifact PlanArtifact) TaskList {
	list := DefaultTaskList() // NextID == 1, Tasks == []
	for _, step := range artifact.Tasks {
		// verbatim; appends pending; bumps NextID. Empty blockers can never
		// reject, so Add is always nil-error here and the parser stays total
		// (never errors, never panics in practice). (#118)
		_, _ = list.Add(step, nil)
	}
	return list
}
