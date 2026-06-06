// Tool-boundary tests for the TaskList tool (#71, storage seam #75), plus
// fixture replay against the shared /fixtures/tasklist ground-truth files.
//
// The tool now persists via the *ToolContext's RunStore (keyed by SessionID
// under TaskListExtrasKey), NOT the retired .spore/task_list.json sandbox path.
// These tests drive over an in-memory RunStore; the sandbox is irrelevant to
// persistence and is proven so by persists_to_run_store_not_sandbox below.

package tools

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

const testSession = "test-session"

// inMemCtx builds a ToolContext over a fresh in-memory run store and the default
// test session id. Returns the context and the underlying store so a test can
// read the persisted blob straight off the store.
func inMemCtx() (*sporecore.ToolContext, *storage.InMemoryStorageProvider) {
	store := storage.NewInMemoryStorageProvider()
	return sporecore.NewToolContext(testSession, store, nil), store
}

// loadFromStore reads the persisted blob off a run store as a TaskList. found is
// false when nothing has been persisted.
func loadFromStore(t *testing.T, store *storage.InMemoryStorageProvider, session string) (sporecore.TaskList, bool) {
	t.Helper()
	value, found, err := store.Get(context.Background(), sporecore.SessionID(session), sporecore.TaskListExtrasKey)
	if err != nil {
		t.Fatalf("store.Get: %v", err)
	}
	if !found {
		return sporecore.TaskList{}, false
	}
	var l sporecore.TaskList
	if err := json.Unmarshal(value, &l); err != nil {
		t.Fatalf("unmarshal persisted blob %q: %v", value, err)
	}
	return l, true
}

// denyPathSandbox approves Validate but denies every ResolvePath, proving the
// tool persists to the RunStore and never touches the filesystem sandbox.
type denyPathSandbox struct{ sporecore.AllowAllSandbox }

func (denyPathSandbox) ResolvePath(_ context.Context, path string, _ sporecore.Operation) (string, *sporecore.SandboxViolation) {
	return "", &sporecore.SandboxViolation{Kind: sporecore.SandboxPathEscape, Path: path}
}

// failingRunStore always errors on Get/Put, to prove storage errors map to a
// recoverable tool error.
type failingRunStore struct{}

func (failingRunStore) Get(context.Context, sporecore.SessionID, string) (json.RawMessage, bool, error) {
	return nil, false, errors.New("boom")
}
func (failingRunStore) Put(context.Context, sporecore.SessionID, string, json.RawMessage) error {
	return errors.New("boom")
}

// corruptRunStore returns a present-but-malformed blob for any key, to prove a
// parse failure is recoverable.
type corruptRunStore struct{}

func (corruptRunStore) Get(context.Context, sporecore.SessionID, string) (json.RawMessage, bool, error) {
	return json.RawMessage(`{"not":"a task list","tasks":42}`), true, nil
}
func (corruptRunStore) Put(context.Context, sporecore.SessionID, string, json.RawMessage) error {
	return nil
}

func tlCall(input any) sporecore.ToolCall {
	b, _ := json.Marshal(input)
	return sporecore.ToolCall{ID: "c1", Name: TaskListToolName, Input: b}
}

func parseList(t *testing.T, out sporecore.ToolOutput) sporecore.TaskList {
	t.Helper()
	if out.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected Success, got %+v", out)
	}
	var l sporecore.TaskList
	if err := json.Unmarshal([]byte(out.Content), &l); err != nil {
		t.Fatalf("unmarshal content %q: %v", out.Content, err)
	}
	return l
}

func TestAddThenListPersistsAndAssignsIDs(t *testing.T) {
	tc, store := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()

	r1 := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "a"}), sb, tc)
	l1 := parseList(t, r1)
	if len(l1.Tasks) != 1 || l1.Tasks[0].ID != 1 || l1.NextID != 2 {
		t.Fatalf("after add a: %+v", l1)
	}

	r2 := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "b"}), sb, tc)
	l2 := parseList(t, r2)
	if len(l2.Tasks) != 2 || l2.Tasks[0].ID != 1 || l2.Tasks[1].ID != 2 {
		t.Fatalf("after add b: %+v", l2)
	}

	// The blob actually exists in the run store under the shared key.
	persisted, found := loadFromStore(t, store, testSession)
	if !found {
		t.Fatal("task_list blob not persisted to run store")
	}
	pa, _ := json.Marshal(persisted)
	la, _ := json.Marshal(l2)
	if string(pa) != string(la) {
		t.Fatalf("persisted %s != tool %s", pa, la)
	}

	// list_tasks returns the same list and does not mutate.
	r3 := tool.Execute(ctx, tlCall(map[string]any{"action": "list_tasks"}), sb, tc)
	l3 := parseList(t, r3)
	a, _ := json.Marshal(l2)
	b, _ := json.Marshal(l3)
	if string(a) != string(b) {
		t.Fatalf("list_tasks mutated: %s != %s", a, b)
	}
}

// Storage seam: persists to the RunStore, NOT the sandbox. Even with a sandbox
// that denies every path, add_task succeeds and persists.
func TestPersistsToRunStoreNotSandbox(t *testing.T) {
	tc, store := inMemCtx()
	tool := NewTaskListTool()
	r := tool.Execute(context.Background(),
		tlCall(map[string]any{"action": "add_task", "description": "via run store"}),
		denyPathSandbox{}, tc)
	list := parseList(t, r)
	if len(list.Tasks) != 1 {
		t.Fatalf("expected 1 task, got %+v", list)
	}
	persisted, found := loadFromStore(t, store, testSession)
	if !found {
		t.Fatal("persisted despite sandbox path denial: blob missing")
	}
	pa, _ := json.Marshal(persisted)
	la, _ := json.Marshal(list)
	if string(pa) != string(la) {
		t.Fatalf("persisted %s != tool %s", pa, la)
	}
}

// Keyed by SessionID: two sessions over the SAME run store keep separate lists.
func TestListsAreKeyedBySessionID(t *testing.T) {
	store := storage.NewInMemoryStorageProvider()
	tcA := sporecore.NewToolContext("session-a", store, nil)
	tcB := sporecore.NewToolContext("session-b", store, nil)
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()

	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "a1"}), sb, tcA)
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "b1"}), sb, tcB)
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "b2"}), sb, tcB)

	a, foundA := loadFromStore(t, store, "session-a")
	b, foundB := loadFromStore(t, store, "session-b")
	if !foundA || !foundB {
		t.Fatalf("both sessions should persist: a=%v b=%v", foundA, foundB)
	}
	if len(a.Tasks) != 1 || a.Tasks[0].Description != "a1" {
		t.Fatalf("session-a: %+v", a)
	}
	if len(b.Tasks) != 2 || b.Tasks[0].Description != "b1" || b.Tasks[1].Description != "b2" {
		t.Fatalf("session-b: %+v", b)
	}
}

// Persist then reload with a FRESH tool over the SAME ctx yields the identical
// list.
func TestPersistThenReloadYieldsIdenticalList(t *testing.T) {
	tc, _ := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	ctx := context.Background()

	tool1 := NewTaskListTool()
	tool1.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "one"}), sb, tc)
	r := tool1.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "two"}), sb, tc)
	fromTool := parseList(t, r)

	// Fresh tool instance, same ctx: list_tasks reads back identical state.
	tool2 := NewTaskListTool()
	reloaded := tool2.Execute(ctx, tlCall(map[string]any{"action": "list_tasks"}), sb, tc)
	a, _ := json.Marshal(fromTool)
	b, _ := json.Marshal(parseList(t, reloaded))
	if string(a) != string(b) {
		t.Fatalf("reload differs: %s != %s", a, b)
	}
}

func TestUpdateStatusAndComplete(t *testing.T) {
	tc, _ := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "x"}), sb, tc)

	r := tool.Execute(ctx, tlCall(map[string]any{"action": "update_task", "id": 1, "status": "in_progress"}), sb, tc)
	if parseList(t, r).Tasks[0].Status != sporecore.TaskStatusInProgress {
		t.Fatalf("update: %+v", r)
	}

	r = tool.Execute(ctx, tlCall(map[string]any{"action": "complete_task", "id": 1}), sb, tc)
	if parseList(t, r).Tasks[0].Status != sporecore.TaskStatusCompleted {
		t.Fatalf("complete: %+v", r)
	}
}

func TestUpdateDescriptionViaTool(t *testing.T) {
	tc, _ := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "x"}), sb, tc)

	r := tool.Execute(ctx, tlCall(map[string]any{"action": "update_task", "id": 1, "description": "y"}), sb, tc)
	l := parseList(t, r)
	if l.Tasks[0].Description != "y" || l.Tasks[0].Status != sporecore.TaskStatusPending {
		t.Fatalf("update description: %+v", l.Tasks[0])
	}
}

func TestUnknownIDIsRecoverableError(t *testing.T) {
	tc, _ := inMemCtx()
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "complete_task", "id": 42}), sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "not found") {
		t.Fatalf("message %q does not mention not found", r.Message)
	}
}

func TestInvalidTransitionOutOfCompletedIsRecoverable(t *testing.T) {
	tc, _ := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "x"}), sb, tc)
	tool.Execute(ctx, tlCall(map[string]any{"action": "complete_task", "id": 1}), sb, tc)
	r := tool.Execute(ctx, tlCall(map[string]any{"action": "update_task", "id": 1, "status": "pending"}), sb, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "invalid transition") {
		t.Fatalf("message %q does not mention invalid transition", r.Message)
	}
}

func TestBadParamsIsRecoverableError(t *testing.T) {
	tc, _ := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()
	cases := []map[string]any{
		{"action": "nope"},          // unknown action
		{"action": "add_task"},      // missing description
		{"action": "update_task"},   // missing id
		{"action": "complete_task"}, // missing id
	}
	for _, c := range cases {
		r := tool.Execute(ctx, tlCall(c), sb, tc)
		if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
			t.Fatalf("%v: expected recoverable error, got %+v", c, r)
		}
	}
	// Empty input is also recoverable.
	r := tool.Execute(ctx, sporecore.ToolCall{ID: "c1", Name: TaskListToolName, Input: json.RawMessage(`{}`)}, sb, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("empty input: expected recoverable error, got %+v", r)
	}
}

// Storage failure (Get/Put) → recoverable error.
func TestStorageFailureIsRecoverableError(t *testing.T) {
	tc := sporecore.NewToolContext(testSession, failingRunStore{}, nil)
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "add_task", "description": "x"}),
		sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// Malformed persisted blob → recoverable parse error.
func TestCorruptBlobIsRecoverableError(t *testing.T) {
	tc := sporecore.NewToolContext(testSession, corruptRunStore{}, nil)
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "list_tasks"}),
		sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// list_tasks does not write: a fresh ctx with a never-written store stays empty
// after a list_tasks call.
func TestListTasksDoesNotWrite(t *testing.T) {
	tc, store := inMemCtx()
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "list_tasks"}),
		sporecore.AllowAllSandbox{}, tc)
	// Returns the empty default list.
	want, _ := json.Marshal(sporecore.DefaultTaskList())
	if r.Content != string(want) {
		t.Fatalf("list_tasks content %q != default %q", r.Content, want)
	}
	// Nothing was persisted (list_tasks must not write).
	if _, found := loadFromStore(t, store, testSession); found {
		t.Fatal("list_tasks must not write to the run store")
	}
}

// No-op default: a ToolContext with no RunStore persists nothing across
// dispatches. add_task succeeds (no error) but the next tool sees an empty list.
func TestNoOpStoragePersistsNothing(t *testing.T) {
	tc := sporecore.NewToolContext(testSession, nil, nil)
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()

	r := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "ephemeral"}), sb, tc)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("add_task should succeed under no-op storage, got %+v", r)
	}
	// A subsequent list_tasks sees an empty list — the write was discarded.
	r2 := tool.Execute(ctx, tlCall(map[string]any{"action": "list_tasks"}), sb, tc)
	want, _ := json.Marshal(sporecore.DefaultTaskList())
	if r2.Content != string(want) {
		t.Fatalf("no-op storage should not persist; got %q", r2.Content)
	}
}

// ============================================================================
// blockers (#118) — tool boundary
// ============================================================================

// add_task passes blockers through to the list and stores them.
func TestAddTaskPassesBlockersThrough(t *testing.T) {
	tc, _ := inMemCtx()
	sb := sporecore.AllowAllSandbox{}
	tool := NewTaskListTool()
	ctx := context.Background()

	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "a"}), sb, tc)
	r := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "b", "blockers": []uint32{1}}), sb, tc)
	l := parseList(t, r)
	if len(l.Tasks) != 2 {
		t.Fatalf("expected 2 tasks, got %+v", l)
	}
	if got := l.Tasks[1].Blockers; len(got) != 1 || got[0] != 1 {
		t.Fatalf("blockers = %v, want [1]", got)
	}
}

// Omitting blockers defaults to empty (backward-compatible call).
func TestAddTaskWithoutBlockersDefaultsEmpty(t *testing.T) {
	tc, _ := inMemCtx()
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "add_task", "description": "a"}),
		sporecore.AllowAllSandbox{}, tc)
	l := parseList(t, r)
	if len(l.Tasks[0].Blockers) != 0 {
		t.Fatalf("blockers = %v, want empty", l.Tasks[0].Blockers)
	}
}

// A self-blocking add maps to a recoverable tool error mentioning blockers.
func TestSelfBlockIsRecoverableError(t *testing.T) {
	tc, _ := inMemCtx()
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "add_task", "description": "a", "blockers": []uint32{1}}),
		sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "invalid blockers") {
		t.Fatalf("message %q does not mention invalid blockers", r.Message)
	}
}

// An unknown blocker id maps to a recoverable tool error.
func TestUnknownBlockerIsRecoverableError(t *testing.T) {
	tc, _ := inMemCtx()
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "add_task", "description": "a", "blockers": []uint32{99}}),
		sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "invalid blockers") {
		t.Fatalf("message %q does not mention invalid blockers", r.Message)
	}
}

// The advertised schema lists blockers as an integer array, in sorted property
// order (after action, before description).
func TestSchemaAdvertisesBlockers(t *testing.T) {
	s := NewTaskListTool().Schema()
	var parsed struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(s.Parameters, &parsed); err != nil {
		t.Fatal(err)
	}
	blockers, ok := parsed.Properties["blockers"]
	if !ok {
		t.Fatal("schema missing blockers property")
	}
	var want, got map[string]any
	_ = json.Unmarshal([]byte(`{"type":"array","items":{"type":"integer"}}`), &want)
	_ = json.Unmarshal(blockers, &got)
	wb, _ := json.Marshal(want)
	gb, _ := json.Marshal(got)
	if string(wb) != string(gb) {
		t.Fatalf("blockers schema = %s, want %s", gb, wb)
	}
}

func TestSchemaIsNotReadOnly(t *testing.T) {
	s := NewTaskListTool().Schema()
	if s.Annotations.ReadOnly {
		t.Fatal("task_list must NOT be read_only (it mutates shared state)")
	}
	if s.Annotations.Destructive || s.Annotations.OpenWorld {
		t.Fatalf("task_list should not be destructive/open_world: %+v", s.Annotations)
	}
	if !json.Valid(s.Parameters) {
		t.Fatal("schema parameters are not valid JSON")
	}
}

// ============================================================================
// Fixture replay (shared ground truth — /fixtures/tasklist)
// ============================================================================

func taskListFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, here, _, _ := runtime.Caller(0)
	// here = .../go/spore-core/tools/tasklist_test.go
	// up: tools → spore-core → go → repo root → fixtures/tasklist
	return filepath.Join(filepath.Dir(here), "..", "..", "..", "fixtures", "tasklist", name)
}

type opStep struct {
	Action   json.RawMessage `json:"action"`
	Expected opExpected      `json:"expected"`
}

type opExpected struct {
	OK    bool                `json:"ok"`
	List  *sporecore.TaskList `json:"list"`
	Error string              `json:"error"`
}

type opScenario struct {
	Name  string   `json:"name"`
	Steps []opStep `json:"steps"`
}

// Replay each operations scenario step-by-step against a read-modify-write over
// a fresh in-memory RunStore, asserting the resulting list (or error kind) per
// step. Must replay byte-identically to the retired sandbox path.
func TestFixtureReplayOperations(t *testing.T) {
	data, err := os.ReadFile(taskListFixturePath(t, "operations.json"))
	if err != nil {
		t.Fatal(err)
	}
	var scenarios []opScenario
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatal(err)
	}
	if len(scenarios) == 0 {
		t.Fatal("expected >=1 scenario")
	}
	tool := NewTaskListTool()
	sb := sporecore.AllowAllSandbox{}
	ctx := context.Background()

	for _, sc := range scenarios {
		// Fresh isolated run store per scenario.
		tc, _ := inMemCtx()
		for i, step := range sc.Steps {
			out := tool.Execute(ctx, sporecore.ToolCall{ID: "c1", Name: TaskListToolName, Input: step.Action}, sb, tc)
			if step.Expected.OK {
				if out.Kind != sporecore.ToolOutputSuccess {
					t.Fatalf("%s step %d: expected Success, got %+v", sc.Name, i, out)
				}
				if step.Expected.List == nil {
					t.Fatalf("%s step %d: ok step missing `list`", sc.Name, i)
				}
				want, _ := json.Marshal(*step.Expected.List)
				if out.Content != string(want) {
					t.Fatalf("%s step %d: list mismatch\n got: %s\nwant: %s", sc.Name, i, out.Content, want)
				}
			} else {
				if out.Kind != sporecore.ToolOutputError {
					t.Fatalf("%s step %d: expected Error, got %+v", sc.Name, i, out)
				}
				if !out.Recoverable {
					t.Fatalf("%s step %d: errors must be recoverable", sc.Name, i)
				}
				var kind string
				switch {
				case strings.Contains(out.Message, "not found"):
					kind = "task_not_found"
				case strings.Contains(out.Message, "invalid transition"):
					kind = "invalid_transition"
				case strings.Contains(out.Message, "invalid blockers"):
					kind = "invalid_blockers"
				default:
					kind = "other"
				}
				if kind != step.Expected.Error {
					t.Fatalf("%s step %d: error kind %q (%s), want %q", sc.Name, i, kind, out.Message, step.Expected.Error)
				}
			}
		}
	}
}

type transitionCase struct {
	From     sporecore.TaskStatus `json:"from"`
	To       sporecore.TaskStatus `json:"to"`
	Expected string               `json:"expected"`
}

// Replay the full transition matrix against ValidateTransition.
func TestFixtureReplayTransitions(t *testing.T) {
	data, err := os.ReadFile(taskListFixturePath(t, "transitions.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []transitionCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("expected >=1 case")
	}
	for _, c := range cases {
		got := "ok"
		if sporecore.ValidateTransition(1, c.From, c.To) != nil {
			got = "invalid_transition"
		}
		if got != c.Expected {
			t.Fatalf("%s -> %s: got %q, want %q", c.From, c.To, got, c.Expected)
		}
	}
}

type serCase struct {
	Name string             `json:"name"`
	List sporecore.TaskList `json:"list"`
	JSON string             `json:"json"`
}

// Replay canonical serialization blobs: marshal(list) must equal the pinned
// JSON, and unmarshal(json) must equal the list (byte-identity).
func TestFixtureReplaySerialization(t *testing.T) {
	data, err := os.ReadFile(taskListFixturePath(t, "serialization.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []serCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("expected >=1 case")
	}
	for _, c := range cases {
		serialized, _ := json.Marshal(c.List)
		if string(serialized) != c.JSON {
			t.Fatalf("serialize %s:\n got: %s\nwant: %s", c.Name, serialized, c.JSON)
		}
		var parsed sporecore.TaskList
		if err := json.Unmarshal([]byte(c.JSON), &parsed); err != nil {
			t.Fatalf("parse %s: %v", c.Name, err)
		}
		reSerialized, _ := json.Marshal(parsed)
		if string(reSerialized) != c.JSON {
			t.Fatalf("parse %s not byte-identical:\n got: %s\nwant: %s", c.Name, reSerialized, c.JSON)
		}
	}
}

type deserCase struct {
	Name         string             `json:"name"`
	JSON         string             `json:"json"`
	Expected     sporecore.TaskList `json:"expected"`
	Reserialized string             `json:"reserialized"`
}

// #118 backward-compat: a pre-#118 blob WITHOUT a blockers key deserializes
// (blockers default to empty), and re-serializing emits the canonical form WITH
// blockers:[]. Replayed byte-identically across all four languages.
func TestFixtureReplayDeserializeBackwardCompat(t *testing.T) {
	data, err := os.ReadFile(taskListFixturePath(t, "deserialize.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []deserCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("expected >=1 case")
	}
	for _, c := range cases {
		var parsed sporecore.TaskList
		if err := json.Unmarshal([]byte(c.JSON), &parsed); err != nil {
			t.Fatalf("parse %s: %v", c.Name, err)
		}
		// Structural equality with the fixture's expected list.
		gotJSON, _ := json.Marshal(parsed)
		wantJSON, _ := json.Marshal(c.Expected)
		if string(gotJSON) != string(wantJSON) {
			t.Fatalf("parse %s: got %s, want %s", c.Name, gotJSON, wantJSON)
		}
		// All blockers defaulted to empty.
		for _, task := range parsed.Tasks {
			if len(task.Blockers) != 0 {
				t.Fatalf("parse %s: blockers should default empty, got %v", c.Name, task.Blockers)
			}
		}
		// Re-serializing emits the canonical form WITH blockers:[].
		if string(gotJSON) != c.Reserialized {
			t.Fatalf("reserialize %s: got %s, want %s", c.Name, gotJSON, c.Reserialized)
		}
	}
}
