// Unit tests for the task-list primitive (#71): types, transition matrix, and
// mutation helpers. Persistence moved off the sandbox onto the RunStore (#75);
// the standalone tool's persistence is covered in tools/tasklist_test.go.

package sporecore

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// listWith builds a list of len(statuses) tasks (descriptions "t") and sets
// each task's status positionally.
func listWith(statuses ...TaskStatus) TaskList {
	l := DefaultTaskList()
	for range statuses {
		if _, err := l.Add("t", nil); err != nil {
			panic(err)
		}
	}
	for i, s := range statuses {
		l.Tasks[i].Status = s
	}
	return l
}

// R1: ids are assigned 1, 2, 3, … sequentially.
func TestIDsAreSequentialFromOne(t *testing.T) {
	l := DefaultTaskList()
	if got, err := l.Add("a", nil); err != nil || got != 1 {
		t.Fatalf("first id = %d, err = %v, want 1", got, err)
	}
	if got, err := l.Add("b", nil); err != nil || got != 2 {
		t.Fatalf("second id = %d, err = %v, want 2", got, err)
	}
	if got, err := l.Add("c", nil); err != nil || got != 3 {
		t.Fatalf("third id = %d, err = %v, want 3", got, err)
	}
	if l.NextID != 4 {
		t.Fatalf("next_id = %d, want 4", l.NextID)
	}
	for i, want := range []uint32{1, 2, 3} {
		if l.Tasks[i].ID != want {
			t.Fatalf("task[%d].id = %d, want %d", i, l.Tasks[i].ID, want)
		}
	}
}

// R2: Add appends to the end, preserving positional order, new tasks pending.
func TestAddAppendsInOrder(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "first")
	mustAdd(t, &l, "second")
	mustAdd(t, &l, "third")
	want := []string{"first", "second", "third"}
	for i, w := range want {
		if l.Tasks[i].Description != w {
			t.Fatalf("task[%d].description = %q, want %q", i, l.Tasks[i].Description, w)
		}
		if l.Tasks[i].Status != TaskStatusPending {
			t.Fatalf("task[%d].status = %q, want pending", i, l.Tasks[i].Status)
		}
	}
}

// R3: serializing (the work list_tasks does) does not mutate state.
func TestSerializeDoesNotMutate(t *testing.T) {
	l := listWith(TaskStatusPending, TaskStatusInProgress)
	before, _ := json.Marshal(l)
	_, _ = json.Marshal(l)
	after, _ := json.Marshal(l)
	if string(before) != string(after) {
		t.Fatalf("serialize mutated state: %s != %s", before, after)
	}
}

func TestUpdateStatusValid(t *testing.T) {
	l := listWith(TaskStatusPending)
	s := TaskStatusInProgress
	if err := l.Update(1, &s, nil); err != nil {
		t.Fatal(err)
	}
	if l.Tasks[0].Status != TaskStatusInProgress {
		t.Fatalf("status = %q", l.Tasks[0].Status)
	}
}

func TestUpdateDescriptionOnly(t *testing.T) {
	l := listWith(TaskStatusPending)
	d := "rewritten"
	if err := l.Update(1, nil, &d); err != nil {
		t.Fatal(err)
	}
	if l.Tasks[0].Description != "rewritten" {
		t.Fatalf("description = %q", l.Tasks[0].Description)
	}
	if l.Tasks[0].Status != TaskStatusPending {
		t.Fatalf("status changed to %q", l.Tasks[0].Status)
	}
}

func TestUpdateStatusAndDescription(t *testing.T) {
	l := listWith(TaskStatusPending)
	s := TaskStatusBlocked
	d := "blocked on x"
	if err := l.Update(1, &s, &d); err != nil {
		t.Fatal(err)
	}
	if l.Tasks[0].Status != TaskStatusBlocked || l.Tasks[0].Description != "blocked on x" {
		t.Fatalf("task = %+v", l.Tasks[0])
	}
}

func TestUpdateNoFieldsIsNoopSuccess(t *testing.T) {
	l := listWith(TaskStatusInProgress)
	before, _ := json.Marshal(l)
	if err := l.Update(1, nil, nil); err != nil {
		t.Fatal(err)
	}
	after, _ := json.Marshal(l)
	if string(before) != string(after) {
		t.Fatalf("no-op update mutated state")
	}
}

func TestCompleteMarksCompleted(t *testing.T) {
	l := listWith(TaskStatusInProgress)
	if err := l.Complete(1); err != nil {
		t.Fatal(err)
	}
	if l.Tasks[0].Status != TaskStatusCompleted {
		t.Fatalf("status = %q", l.Tasks[0].Status)
	}
}

// R4: unknown id on update/complete → TaskNotFound.
func TestUnknownIDIsTaskNotFound(t *testing.T) {
	l := listWith(TaskStatusPending)
	s := TaskStatusCompleted
	err := l.Update(99, &s, nil)
	var te *TaskListError
	if !asTaskListError(err, &te) || te.Kind != TaskListErrTaskNotFound || te.ID != 99 {
		t.Fatalf("update: got %v", err)
	}
	err = l.Complete(99)
	if !asTaskListError(err, &te) || te.Kind != TaskListErrTaskNotFound || te.ID != 99 {
		t.Fatalf("complete: got %v", err)
	}
}

// R5/R6: a rejected transition leaves the task untouched.
func TestRejectedTransitionDoesNotMutate(t *testing.T) {
	l := listWith(TaskStatusCompleted)
	before, _ := json.Marshal(l)
	s := TaskStatusInProgress
	err := l.Update(1, &s, nil)
	var te *TaskListError
	if !asTaskListError(err, &te) || te.Kind != TaskListErrInvalidTransition {
		t.Fatalf("expected invalid transition, got %v", err)
	}
	after, _ := json.Marshal(l)
	if string(before) != string(after) {
		t.Fatalf("rejected transition mutated state")
	}
}

// DECISION 1: every allowed transition (incl. idempotent self-transitions).
func TestAllowedTransitions(t *testing.T) {
	allowed := [][2]TaskStatus{
		{TaskStatusPending, TaskStatusInProgress},
		{TaskStatusPending, TaskStatusCompleted},
		{TaskStatusPending, TaskStatusBlocked},
		{TaskStatusInProgress, TaskStatusCompleted},
		{TaskStatusInProgress, TaskStatusBlocked},
		{TaskStatusBlocked, TaskStatusInProgress},
		{TaskStatusBlocked, TaskStatusCompleted},
		{TaskStatusPending, TaskStatusPending},
		{TaskStatusInProgress, TaskStatusInProgress},
		{TaskStatusCompleted, TaskStatusCompleted},
		{TaskStatusBlocked, TaskStatusBlocked},
	}
	for _, tr := range allowed {
		if err := ValidateTransition(1, tr[0], tr[1]); err != nil {
			t.Fatalf("expected %s -> %s allowed, got %v", tr[0], tr[1], err)
		}
	}
}

// DECISION 1 / R6: every transition OUT of completed (except self) rejected.
func TestOutOfCompletedRejected(t *testing.T) {
	for _, to := range []TaskStatus{TaskStatusPending, TaskStatusInProgress, TaskStatusBlocked} {
		err := ValidateTransition(7, TaskStatusCompleted, to)
		var te *TaskListError
		if !asTaskListError(err, &te) || te.Kind != TaskListErrInvalidTransition ||
			te.ID != 7 || te.From != TaskStatusCompleted || te.To != to {
			t.Fatalf("completed -> %s: got %v", to, err)
		}
	}
}

// The remaining rejected (non-completed-origin) transitions.
func TestOtherRejectedTransitions(t *testing.T) {
	if ValidateTransition(1, TaskStatusInProgress, TaskStatusPending) == nil {
		t.Fatal("in_progress -> pending should be rejected")
	}
	if ValidateTransition(1, TaskStatusBlocked, TaskStatusPending) == nil {
		t.Fatal("blocked -> pending should be rejected")
	}
}

func TestPendingToCompletedAllowed(t *testing.T) {
	l := listWith(TaskStatusPending)
	if err := l.Complete(1); err != nil {
		t.Fatal(err)
	}
	if l.Tasks[0].Status != TaskStatusCompleted {
		t.Fatalf("status = %q", l.Tasks[0].Status)
	}
}

func TestBlockedToInProgressAndCompletedAllowed(t *testing.T) {
	l := listWith(TaskStatusBlocked, TaskStatusBlocked)
	s := TaskStatusInProgress
	if err := l.Update(1, &s, nil); err != nil {
		t.Fatal(err)
	}
	if err := l.Complete(2); err != nil {
		t.Fatal(err)
	}
	if l.Tasks[0].Status != TaskStatusInProgress || l.Tasks[1].Status != TaskStatusCompleted {
		t.Fatalf("tasks = %+v", l.Tasks)
	}
}

// R7: idempotent self-transition is a success and a no-op on state.
func TestIdempotentSelfTransition(t *testing.T) {
	l := listWith(TaskStatusCompleted)
	s := TaskStatusCompleted
	if err := l.Update(1, &s, nil); err != nil {
		t.Fatal(err)
	}
	if err := l.Complete(1); err != nil { // completed -> completed via Complete
		t.Fatal(err)
	}
	if l.Tasks[0].Status != TaskStatusCompleted {
		t.Fatalf("status = %q", l.Tasks[0].Status)
	}
}

// Reload preserves next_id (ids never reused after a round-trip).
func TestReloadPreservesNextID(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "a")
	mustAdd(t, &l, "b")
	encoded, _ := json.Marshal(l)
	var reloaded TaskList
	if err := json.Unmarshal(encoded, &reloaded); err != nil {
		t.Fatal(err)
	}
	if reloaded.NextID != 3 {
		t.Fatalf("reloaded next_id = %d, want 3", reloaded.NextID)
	}
	if got, err := reloaded.Add("c", nil); err != nil || got != 3 {
		t.Fatalf("continued id = %d, err = %v, want 3", got, err)
	}
}

// Serde round-trip is byte-identical (re-serializing the parsed form).
func TestSerdeRoundTripByteIdentical(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "alpha")
	mustAdd(t, &l, "beta")
	s := TaskStatusInProgress
	if err := l.Update(2, &s, nil); err != nil {
		t.Fatal(err)
	}
	json1, _ := json.Marshal(l)
	var parsed TaskList
	if err := json.Unmarshal(json1, &parsed); err != nil {
		t.Fatal(err)
	}
	json2, _ := json.Marshal(parsed)
	if string(json1) != string(json2) {
		t.Fatalf("round-trip not byte-identical: %s != %s", json1, json2)
	}
}

// Status snake_case spellings are exact, both directions.
func TestStatusSnakeCaseSpellings(t *testing.T) {
	cases := map[TaskStatus]string{
		TaskStatusPending:    `"pending"`,
		TaskStatusInProgress: `"in_progress"`,
		TaskStatusCompleted:  `"completed"`,
		TaskStatusBlocked:    `"blocked"`,
	}
	for status, want := range cases {
		got, _ := json.Marshal(status)
		if string(got) != want {
			t.Fatalf("marshal %v = %s, want %s", status, got, want)
		}
	}
	var back TaskStatus
	if err := json.Unmarshal([]byte(`"in_progress"`), &back); err != nil {
		t.Fatal(err)
	}
	if back != TaskStatusInProgress {
		t.Fatalf("unmarshal = %q", back)
	}
}

// Canonical empty-list serialization.
func TestDefaultSerializesCanonically(t *testing.T) {
	got, _ := json.Marshal(DefaultTaskList())
	if string(got) != `{"tasks":[],"next_id":1}` {
		t.Fatalf("got %s", got)
	}
}

// Canonical populated-list serialization (exact spelling).
func TestPopulatedSerializesCanonically(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "write tests")
	s := TaskStatusInProgress
	if err := l.Update(1, &s, nil); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(l)
	want := `{"tasks":[{"id":1,"description":"write tests","status":"in_progress","blockers":[]}],"next_id":2}`
	if string(got) != want {
		t.Fatalf("got %s", got)
	}
}

// ============================================================================
// PlanArtifactToTaskList (#72)
// ============================================================================

// planArtifact is a small constructor for test inputs.
func planArtifact(tasks []string, rationale string) PlanArtifact {
	return PlanArtifact{Tasks: tasks, Rationale: rationale}
}

// One task per plan step, plan order preserved, all pending.
func TestPlanOneTaskPerStepInOrderAllPending(t *testing.T) {
	list := PlanArtifactToTaskList(planArtifact([]string{"first", "second", "third"}, ""))
	want := []string{"first", "second", "third"}
	if len(list.Tasks) != len(want) {
		t.Fatalf("len(tasks) = %d, want %d", len(list.Tasks), len(want))
	}
	for i, w := range want {
		if list.Tasks[i].Description != w {
			t.Fatalf("task[%d].description = %q, want %q", i, list.Tasks[i].Description, w)
		}
		if list.Tasks[i].Status != TaskStatusPending {
			t.Fatalf("task[%d].status = %q, want pending", i, list.Tasks[i].Status)
		}
	}
}

// Sequential ids [1,2,3] and next_id == 4.
func TestPlanAssignsSequentialIDs(t *testing.T) {
	list := PlanArtifactToTaskList(planArtifact([]string{"a", "b", "c"}, ""))
	for i, want := range []uint32{1, 2, 3} {
		if list.Tasks[i].ID != want {
			t.Fatalf("task[%d].id = %d, want %d", i, list.Tasks[i].ID, want)
		}
	}
	if list.NextID != 4 {
		t.Fatalf("next_id = %d, want 4", list.NextID)
	}
}

// Deterministic: same artifact parsed twice → byte-identical lists.
func TestPlanIsDeterministic(t *testing.T) {
	a := planArtifact([]string{"x", "y"}, "why")
	j1, _ := json.Marshal(PlanArtifactToTaskList(a))
	j2, _ := json.Marshal(PlanArtifactToTaskList(a))
	if string(j1) != string(j2) {
		t.Fatalf("nondeterministic: %s != %s", j1, j2)
	}
}

// Descriptions copied VERBATIM — whitespace and empty strings preserved.
func TestPlanKeepsDescriptionsVerbatim(t *testing.T) {
	list := PlanArtifactToTaskList(planArtifact([]string{"  spaced  ", ""}, ""))
	if list.Tasks[0].Description != "  spaced  " {
		t.Fatalf("task[0].description = %q, want %q", list.Tasks[0].Description, "  spaced  ")
	}
	if list.Tasks[1].Description != "" {
		t.Fatalf("task[1].description = %q, want empty", list.Tasks[1].Description)
	}
	if list.Tasks[0].ID != 1 || list.Tasks[1].ID != 2 {
		t.Fatalf("ids = %d,%d want 1,2", list.Tasks[0].ID, list.Tasks[1].ID)
	}
}

// Empty plan → the canonical empty list, byte-exact {"tasks":[],"next_id":1}.
func TestPlanEmptyYieldsDefaultList(t *testing.T) {
	for _, tasks := range [][]string{nil, {}} {
		list := PlanArtifactToTaskList(planArtifact(tasks, ""))
		if len(list.Tasks) != 0 || list.NextID != 1 {
			t.Fatalf("empty plan list = %+v, want empty default", list)
		}
		got, _ := json.Marshal(list)
		if string(got) != `{"tasks":[],"next_id":1}` {
			t.Fatalf("got %s, want canonical empty", got)
		}
	}
}

// Rationale is DROPPED — it appears nowhere in the resulting TaskList JSON.
func TestPlanDropsRationale(t *testing.T) {
	list := PlanArtifactToTaskList(planArtifact([]string{"do thing"}, "SECRET_RATIONALE_TOKEN"))
	got, _ := json.Marshal(list)
	if s := string(got); contains(s, "SECRET_RATIONALE_TOKEN") || contains(s, "rationale") {
		t.Fatalf("rationale leaked into %s", s)
	}
}

// Serde round-trip of the parsed result is byte-identical / canonical.
func TestPlanResultSerdeRoundTripByteIdentical(t *testing.T) {
	list := PlanArtifactToTaskList(planArtifact([]string{"alpha", "beta"}, "r"))
	json1, _ := json.Marshal(list)
	want := `{"tasks":[{"id":1,"description":"alpha","status":"pending","blockers":[]},{"id":2,"description":"beta","status":"pending","blockers":[]}],"next_id":3}`
	if string(json1) != want {
		t.Fatalf("canonical = %s, want %s", json1, want)
	}
	var parsed TaskList
	if err := json.Unmarshal(json1, &parsed); err != nil {
		t.Fatal(err)
	}
	json2, _ := json.Marshal(parsed)
	if string(json1) != string(json2) {
		t.Fatalf("round-trip not byte-identical: %s != %s", json1, json2)
	}
}

// ----------------------------------------------------------------------------
// Fixture replay (#72) — fixtures/plan_to_tasklist/cases.json is GROUND TRUTH.
// For each case: unmarshal input into a PlanArtifact, run the function, and
// assert the marshalled TaskList equals expected byte-for-byte (canonical
// compact form, field order tasks then next_id). Never edit the fixture.
// ----------------------------------------------------------------------------
func TestFixtureReplayPlanToTaskList(t *testing.T) {
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	path := filepath.Join(dir, "..", "..", "fixtures", "plan_to_tasklist", "cases.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	var cases []struct {
		Name     string          `json:"name"`
		Input    PlanArtifact    `json:"input"`
		Expected json.RawMessage `json:"expected"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("expected >= 1 case")
	}

	for _, c := range cases {
		got := PlanArtifactToTaskList(c.Input)
		gotJSON, err := json.Marshal(got)
		if err != nil {
			t.Fatalf("case %s: marshal: %v", c.Name, err)
		}
		// Re-marshal expected through TaskList so both sides use the canonical
		// compact form (field order tasks then next_id, [] over null).
		var want TaskList
		if err := json.Unmarshal(c.Expected, &want); err != nil {
			t.Fatalf("case %s: parse expected: %v", c.Name, err)
		}
		wantJSON, _ := json.Marshal(want)
		if string(gotJSON) != string(wantJSON) {
			t.Fatalf("case %s: got %s, want %s", c.Name, gotJSON, wantJSON)
		}
	}
}

// asTaskListError is a small errors.As shim that keeps the call sites terse.
func asTaskListError(err error, target **TaskListError) bool {
	te, ok := err.(*TaskListError)
	if ok {
		*target = te
	}
	return ok
}

// mustAdd adds a task with no blockers and fails the test on error.
func mustAdd(t *testing.T, l *TaskList, description string) uint32 {
	t.Helper()
	id, err := l.Add(description, nil)
	if err != nil {
		t.Fatalf("Add(%q): %v", description, err)
	}
	return id
}

// ============================================================================
// blockers (#118)
// ============================================================================

// Happy path: blockers referencing earlier real ids are accepted and stored.
func TestAddWithValidBlockersOK(t *testing.T) {
	l := DefaultTaskList()
	if id := mustAdd(t, &l, "a"); id != 1 {
		t.Fatalf("id = %d, want 1", id)
	}
	if id := mustAdd(t, &l, "b"); id != 2 {
		t.Fatalf("id = %d, want 2", id)
	}
	id, err := l.Add("c", []uint32{1, 2})
	if err != nil || id != 3 {
		t.Fatalf("Add c: id = %d, err = %v, want 3", id, err)
	}
	if got := l.Tasks[2].Blockers; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("blockers = %v, want [1 2]", got)
	}
	if l.NextID != 4 {
		t.Fatalf("next_id = %d, want 4", l.NextID)
	}
}

// Empty blockers never reject and store as an empty slice.
func TestAddWithEmptyBlockersOK(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "a")
	if len(l.Tasks[0].Blockers) != 0 {
		t.Fatalf("blockers = %v, want empty", l.Tasks[0].Blockers)
	}
}

// Self-block: a blocker equal to the about-to-be-assigned id is rejected.
func TestSelfBlockRejected(t *testing.T) {
	l := DefaultTaskList()
	// NextID is 1, blocker 1 == self.
	_, err := l.Add("a", []uint32{1})
	var te *TaskListError
	if !asTaskListError(err, &te) || te.Kind != TaskListErrInvalidBlockers ||
		te.ID != 1 || te.Reason == nil || te.Reason.Reason != BlockerRejectionSelfBlock {
		t.Fatalf("got %v", err)
	}
}

// Unknown id: a blocker matching no existing task is rejected, carrying the id.
func TestUnknownBlockerIDRejected(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "a") // id 1
	_, err := l.Add("b", []uint32{99})
	var te *TaskListError
	if !asTaskListError(err, &te) || te.Kind != TaskListErrInvalidBlockers ||
		te.ID != 2 || te.Reason == nil || te.Reason.Reason != BlockerRejectionUnknownID ||
		te.Reason.Blocker != 99 {
		t.Fatalf("got %v", err)
	}
}

// A rejected add leaves the list completely untouched (R9, mirrors Update).
func TestRejectedBlockersDoNotMutate(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "a")
	before, _ := json.Marshal(l)
	if _, err := l.Add("b", []uint32{99}); err == nil {
		t.Fatal("expected rejection")
	}
	after, _ := json.Marshal(l)
	if string(before) != string(after) {
		t.Fatalf("rejected add mutated state: %s != %s", before, after)
	}
	if l.NextID != 2 {
		t.Fatalf("next_id advanced to %d, want 2", l.NextID)
	}
}

// Self-block takes precedence over unknown-id when both are present (self-block
// is checked first per the documented order).
func TestSelfBlockCheckedBeforeUnknown(t *testing.T) {
	l := DefaultTaskList()
	// NextID 1: slice contains self (1) and an unknown (99); self wins.
	_, err := l.Add("a", []uint32{1, 99})
	var te *TaskListError
	if !asTaskListError(err, &te) || te.Reason == nil ||
		te.Reason.Reason != BlockerRejectionSelfBlock {
		t.Fatalf("got %v", err)
	}
}

// wouldCreateCycle: tested directly against a hand-built cyclic graph, since an
// append-only add can never close a cycle on its own.
func TestWouldCreateCycleDetectsBackEdge(t *testing.T) {
	// Build edges (task -> blocker): 3 -> 2, 2 -> 1. So from node 3 there is a
	// directed path 3 -> 2 -> 1 reaching node 1.
	l := DefaultTaskList()
	mustAdd(t, &l, "a") // 1
	mustAdd(t, &l, "b") // 2
	mustAdd(t, &l, "c") // 3
	l.Tasks[2].Blockers = []uint32{2}
	l.Tasks[1].Blockers = []uint32{1}
	// Re-adding node 1 with a blocker on 3 closes 1 -> 3 -> 2 -> 1.
	if !l.wouldCreateCycle(1, []uint32{3}) {
		t.Fatal("expected cycle for node 1 -> 3")
	}
	// Node 4 with blocker 3 has no path back to 4, so no cycle.
	if l.wouldCreateCycle(4, []uint32{3}) {
		t.Fatal("unexpected cycle for node 4 -> 3")
	}
}

// wouldCreateCycle: a direct self-edge is a cycle.
func TestWouldCreateCycleSelfEdge(t *testing.T) {
	l := DefaultTaskList()
	if !l.wouldCreateCycle(5, []uint32{5}) {
		t.Fatal("self-edge should be a cycle")
	}
}

// wouldCreateCycle: empty new edges are never a cycle.
func TestWouldCreateCycleEmptyIsFalse(t *testing.T) {
	l := DefaultTaskList()
	if l.wouldCreateCycle(1, nil) {
		t.Fatal("empty edges should not be a cycle")
	}
}

// The cycle branch of Add rejects when the helper reports a cycle. We craft a
// state where re-adding an id with a back-edge would cycle: task 1 already
// blocks on id 3 (the next id about to be assigned), so adding task 3 blocked by
// 1 closes 3 -> 1 -> 3.
func TestAddRejectsCycle(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "a") // id 1
	mustAdd(t, &l, "b") // id 2
	l.Tasks[0].Blockers = []uint32{3}
	_, err := l.Add("c", []uint32{1})
	var te *TaskListError
	if !asTaskListError(err, &te) || te.Kind != TaskListErrInvalidBlockers ||
		te.ID != 3 || te.Reason == nil || te.Reason.Reason != BlockerRejectionCycle {
		t.Fatalf("got %v", err)
	}
}

// Non-empty blockers serialize as the LAST field, byte-exact.
func TestBlockersSerializeLastAndExact(t *testing.T) {
	l := DefaultTaskList()
	mustAdd(t, &l, "a")
	if _, err := l.Add("b", []uint32{1}); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(l)
	want := `{"tasks":[{"id":1,"description":"a","status":"pending","blockers":[]},{"id":2,"description":"b","status":"pending","blockers":[1]}],"next_id":3}`
	if string(got) != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

// Backward-compat: a pre-#118 blob WITHOUT a blockers key still loads, with
// blockers defaulting to empty; re-serializing emits the canonical form WITH
// blockers:[].
func TestDeserializesPre118BlobWithoutBlockers(t *testing.T) {
	in := `{"tasks":[{"id":1,"description":"old","status":"pending"}],"next_id":2}`
	var l TaskList
	if err := json.Unmarshal([]byte(in), &l); err != nil {
		t.Fatal(err)
	}
	if len(l.Tasks) != 1 || len(l.Tasks[0].Blockers) != 0 {
		t.Fatalf("blockers should default empty: %+v", l.Tasks)
	}
	got, _ := json.Marshal(l)
	want := `{"tasks":[{"id":1,"description":"old","status":"pending","blockers":[]}],"next_id":2}`
	if string(got) != want {
		t.Fatalf("reserialized %s, want %s", got, want)
	}
}

// BlockerRejection serde tags are snake_case on `reason`, with `blocker` carried
// only for unknown_id.
func TestBlockerRejectionSerdeTags(t *testing.T) {
	cases := []struct {
		r    BlockerRejection
		want string
	}{
		{BlockerRejection{Reason: BlockerRejectionSelfBlock}, `{"reason":"self_block"}`},
		{BlockerRejection{Reason: BlockerRejectionUnknownID, Blocker: 7}, `{"reason":"unknown_id","blocker":7}`},
		{BlockerRejection{Reason: BlockerRejectionCycle}, `{"reason":"cycle"}`},
	}
	for _, c := range cases {
		got, _ := json.Marshal(c.r)
		if string(got) != c.want {
			t.Fatalf("marshal %+v = %s, want %s", c.r, got, c.want)
		}
	}
}

// TaskListError InvalidBlockers wire tag is snake_case `invalid_blockers`.
func TestInvalidBlockersErrorWireTag(t *testing.T) {
	e := newInvalidBlockers(3, BlockerRejection{Reason: BlockerRejectionSelfBlock})
	got, _ := json.Marshal(e)
	if !contains(string(got), `"kind":"invalid_blockers"`) {
		t.Fatalf("wire tag missing: %s", got)
	}
}

// ============================================================================
// #126 DAG helpers + step ledger
// ============================================================================

// dagList builds the diamond DAG used across the #126 helper tests:
// 1 (root); 2 -> 1; 3 -> 1; 4 -> 2,3 (diamond); 5 (independent).
func dagList(t *testing.T) TaskList {
	t.Helper()
	l := DefaultTaskList()
	if _, err := l.Add("a", nil); err != nil { // 1
		t.Fatal(err)
	}
	if _, err := l.Add("b", []uint32{1}); err != nil { // 2 -> 1
		t.Fatal(err)
	}
	if _, err := l.Add("c", []uint32{1}); err != nil { // 3 -> 1
		t.Fatal(err)
	}
	if _, err := l.Add("d", []uint32{2, 3}); err != nil { // 4 -> 2,3
		t.Fatal(err)
	}
	if _, err := l.Add("e", nil); err != nil { // 5 independent
		t.Fatal(err)
	}
	return l
}

// NextReady picks the lowest-id pending task with all blockers completed.
func TestNextReadyRespectsBlockersAndIDTiebreak(t *testing.T) {
	l := dagList(t)
	// Initially 1 and 5 are ready (no blockers); lowest id wins → 1.
	if id, ok := l.NextReady(); !ok || id != 1 {
		t.Fatalf("next ready = (%d,%v), want (1,true)", id, ok)
	}
	if err := l.Complete(1); err != nil {
		t.Fatal(err)
	}
	// Now 2, 3, 5 ready (4 still waits on 2 & 3). Lowest is 2.
	if id, _ := l.NextReady(); id != 2 {
		t.Fatalf("next ready = %d, want 2", id)
	}
	_ = l.Complete(2)
	if id, _ := l.NextReady(); id != 3 {
		t.Fatalf("next ready = %d, want 3", id)
	}
	_ = l.Complete(3)
	if id, _ := l.NextReady(); id != 4 {
		t.Fatalf("next ready = %d, want 4", id)
	}
	_ = l.Complete(4)
	if id, _ := l.NextReady(); id != 5 {
		t.Fatalf("next ready = %d, want 5", id)
	}
	_ = l.Complete(5)
	if _, ok := l.NextReady(); ok {
		t.Fatalf("next ready should be exhausted")
	}
}

// A pending task whose blocker is NOT completed is never ready.
func TestNextReadySkipsUnsatisfiedBlockers(t *testing.T) {
	l := dagList(t)
	if id, _ := l.NextReady(); id != 1 {
		t.Fatalf("only 1 and 5 runnable; min is 1, got %d", id)
	}
}

// TransitiveBlockers: the full upstream closure, sorted, excluding self.
func TestTransitiveBlockersClosure(t *testing.T) {
	l := dagList(t)
	if got := l.TransitiveBlockers(1); len(got) != 0 {
		t.Fatalf("blockers(1) = %v, want []", got)
	}
	if got := l.TransitiveBlockers(2); !equalU32s(got, []uint32{1}) {
		t.Fatalf("blockers(2) = %v, want [1]", got)
	}
	if got := l.TransitiveBlockers(4); !equalU32s(got, []uint32{1, 2, 3}) {
		t.Fatalf("blockers(4) = %v, want [1 2 3]", got)
	}
	if got := l.TransitiveBlockers(5); len(got) != 0 {
		t.Fatalf("blockers(5) = %v, want []", got)
	}
}

// TransitiveDependents: the full downstream closure, sorted, excluding self.
func TestTransitiveDependentsClosure(t *testing.T) {
	l := dagList(t)
	if got := l.TransitiveDependents(1); !equalU32s(got, []uint32{2, 3, 4}) {
		t.Fatalf("dependents(1) = %v, want [2 3 4]", got)
	}
	if got := l.TransitiveDependents(2); !equalU32s(got, []uint32{4}) {
		t.Fatalf("dependents(2) = %v, want [4]", got)
	}
	if got := l.TransitiveDependents(3); !equalU32s(got, []uint32{4}) {
		t.Fatalf("dependents(3) = %v, want [4]", got)
	}
	if got := l.TransitiveDependents(4); len(got) != 0 {
		t.Fatalf("dependents(4) = %v, want []", got)
	}
	if got := l.TransitiveDependents(5); len(got) != 0 {
		t.Fatalf("dependents(5) = %v, want []", got)
	}
}

// HasCycle: acyclic graphs return false; hand-built cycles return true.
func TestHasCycleDetectsCycles(t *testing.T) {
	l := dagList(t)
	if l.HasCycle() {
		t.Fatal("diamond DAG is acyclic")
	}

	// 1 -> 2 -> 1.
	c := DefaultTaskList()
	_, _ = c.Add("a", nil)            // 1
	_, _ = c.Add("b", []uint32{1})    // 2 -> 1
	c.Tasks[0].Blockers = []uint32{2} // 1 -> 2 closes the cycle
	if !c.HasCycle() {
		t.Fatal("expected cycle 1->2->1")
	}

	// Self-edge is a cycle.
	s := DefaultTaskList()
	_, _ = s.Add("x", nil)
	s.Tasks[0].Blockers = []uint32{1}
	if !s.HasCycle() {
		t.Fatal("self-edge is a cycle")
	}
}

// HasCycle: an edge to a non-existent task is ignored (no false positive).
func TestHasCycleIgnoresUnknownBlockers(t *testing.T) {
	l := DefaultTaskList()
	_, _ = l.Add("a", nil)
	l.Tasks[0].Blockers = []uint32{99} // dangling edge
	if l.HasCycle() {
		t.Fatal("a dangling edge is not a cycle")
	}
}

// PushStepLedger: drop-oldest past N, returns whether anything was dropped.
func TestStepLedgerDropOldestPastN(t *testing.T) {
	var ledger []StepLedgerEntry
	for i := 0; i < StepLedgerMaxEntries; i++ {
		var dropped bool
		ledger, dropped = PushStepLedger(ledger, StepLedgerEntry{TaskID: uint32(i + 1), Summary: "s"})
		if dropped {
			t.Fatalf("no drop before exceeding N (i=%d)", i)
		}
	}
	if len(ledger) != StepLedgerMaxEntries {
		t.Fatalf("len = %d, want %d", len(ledger), StepLedgerMaxEntries)
	}
	var dropped bool
	ledger, dropped = PushStepLedger(ledger, StepLedgerEntry{TaskID: 999, Summary: "newest"})
	if !dropped {
		t.Fatal("drop fires at N+1")
	}
	if len(ledger) != StepLedgerMaxEntries {
		t.Fatalf("stays bounded at N, got %d", len(ledger))
	}
	for _, e := range ledger {
		if e.TaskID == 1 {
			t.Fatal("the oldest (task 1) must have been dropped")
		}
	}
	if ledger[len(ledger)-1].TaskID != 999 {
		t.Fatal("newest must be retained")
	}
}

// RenderStepLedger: deterministic text; files suffix only when non-empty;
// elision marker leads when requested; ("",false) for empty.
func TestRenderStepLedgerShape(t *testing.T) {
	if _, ok := RenderStepLedger(nil, false); ok {
		t.Fatal("empty ledger renders nothing")
	}
	entries := []StepLedgerEntry{
		{TaskID: 1, Summary: "scaffolded"},
		{TaskID: 2, Summary: "wrote tests", FilesTouched: []string{"a.go", "b.go"}},
	}
	rendered, ok := RenderStepLedger(entries, false)
	if !ok {
		t.Fatal("expected a rendered block")
	}
	if !contains(rendered, "#1 scaffolded") {
		t.Fatalf("missing #1 line: %s", rendered)
	}
	if contains(rendered, "#1 scaffolded [files") {
		t.Fatalf("empty files must have no suffix: %s", rendered)
	}
	if !contains(rendered, "#2 wrote tests [files: a.go, b.go]") {
		t.Fatalf("missing #2 files suffix: %s", rendered)
	}
	if contains(rendered, StepLedgerElisionMarker) {
		t.Fatalf("no elision marker when not elided: %s", rendered)
	}
	elided, _ := RenderStepLedger(entries, true)
	if !contains(elided, StepLedgerElisionMarker) {
		t.Fatalf("elision marker should lead: %s", elided)
	}
}

// StepLedgerEntry serde field order is task_id, summary, files_touched; empty
// files_touched serializes as [] (never null).
func TestStepLedgerEntrySerdeShape(t *testing.T) {
	e := StepLedgerEntry{TaskID: 7, Summary: "did", FilesTouched: []string{"x"}}
	got, _ := json.Marshal(e)
	if string(got) != `{"task_id":7,"summary":"did","files_touched":["x"]}` {
		t.Fatalf("marshal = %s", got)
	}
	empty := StepLedgerEntry{TaskID: 1, Summary: "s"}
	got2, _ := json.Marshal(empty)
	if string(got2) != `{"task_id":1,"summary":"s","files_touched":[]}` {
		t.Fatalf("empty files marshal = %s", got2)
	}
}
