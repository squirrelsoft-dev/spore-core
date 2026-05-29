// Unit tests for the task-list primitive (#71): types, transition matrix,
// mutation helpers, and disk persistence.

package sporecore

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
)

// tempRootSandbox roots resolved paths inside a tempdir so the read-modify-write
// hits a real, isolated file. It embeds DefaultSandbox for the methods the
// task-list path never exercises.
type tempRootSandbox struct {
	AllowAllSandbox
	root string
}

func (s tempRootSandbox) ResolvePath(_ context.Context, path string, _ Operation) (string, *SandboxViolation) {
	return filepath.Join(s.root, path), nil
}

func newTempRootSandbox(t *testing.T) tempRootSandbox {
	t.Helper()
	return tempRootSandbox{root: t.TempDir()}
}

// listWith builds a list of len(statuses) tasks (descriptions "t") and sets
// each task's status positionally.
func listWith(statuses ...TaskStatus) TaskList {
	l := DefaultTaskList()
	for range statuses {
		l.Add("t")
	}
	for i, s := range statuses {
		l.Tasks[i].Status = s
	}
	return l
}

// R1: ids are assigned 1, 2, 3, … sequentially.
func TestIDsAreSequentialFromOne(t *testing.T) {
	l := DefaultTaskList()
	if got := l.Add("a"); got != 1 {
		t.Fatalf("first id = %d, want 1", got)
	}
	if got := l.Add("b"); got != 2 {
		t.Fatalf("second id = %d, want 2", got)
	}
	if got := l.Add("c"); got != 3 {
		t.Fatalf("third id = %d, want 3", got)
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
	l.Add("first")
	l.Add("second")
	l.Add("third")
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
	l.Add("a")
	l.Add("b")
	encoded, _ := json.Marshal(l)
	var reloaded TaskList
	if err := json.Unmarshal(encoded, &reloaded); err != nil {
		t.Fatal(err)
	}
	if reloaded.NextID != 3 {
		t.Fatalf("reloaded next_id = %d, want 3", reloaded.NextID)
	}
	if got := reloaded.Add("c"); got != 3 {
		t.Fatalf("continued id = %d, want 3", got)
	}
}

// Serde round-trip is byte-identical (re-serializing the parsed form).
func TestSerdeRoundTripByteIdentical(t *testing.T) {
	l := DefaultTaskList()
	l.Add("alpha")
	l.Add("beta")
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
	l.Add("write tests")
	s := TaskStatusInProgress
	if err := l.Update(1, &s, nil); err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(l)
	want := `{"tasks":[{"id":1,"description":"write tests","status":"in_progress"}],"next_id":2}`
	if string(got) != want {
		t.Fatalf("got %s", got)
	}
}

// LoadTaskList on an absent file yields the default; store-then-reload is
// identical.
func TestPersistThenReloadIdentical(t *testing.T) {
	sb := newTempRootSandbox(t)
	ctx := context.Background()

	// Absent file → default.
	loaded, v, err := LoadTaskList(ctx, sb)
	if v != nil || err != nil {
		t.Fatalf("load absent: v=%v err=%v", v, err)
	}
	if loaded.NextID != 1 || len(loaded.Tasks) != 0 {
		t.Fatalf("absent load not default: %+v", loaded)
	}

	loaded.Add("one")
	loaded.Add("two")
	if v, err := StoreTaskList(ctx, loaded, sb); v != nil || err != nil {
		t.Fatalf("store: v=%v err=%v", v, err)
	}

	reloaded, v, err := LoadTaskList(ctx, sb)
	if v != nil || err != nil {
		t.Fatalf("reload: v=%v err=%v", v, err)
	}
	a, _ := json.Marshal(loaded)
	b, _ := json.Marshal(reloaded)
	if string(a) != string(b) {
		t.Fatalf("persist/reload differ: %s != %s", a, b)
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
