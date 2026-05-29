// Tool-boundary tests for the TaskList tool (#71), plus fixture replay against
// the shared /fixtures/tasklist ground-truth files.

package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// tlSandbox roots resolved paths inside a tempdir so the read-modify-write hits
// a real, isolated file. Embeds DefaultSandbox for the unused methods.
type tlSandbox struct {
	sporecore.AllowAllSandbox
	root string
}

func (s tlSandbox) ResolvePath(_ context.Context, path string, _ sporecore.Operation) (string, *sporecore.SandboxViolation) {
	return filepath.Join(s.root, path), nil
}

func newTLSandbox(t *testing.T) tlSandbox {
	t.Helper()
	return tlSandbox{root: t.TempDir()}
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
	sb := newTLSandbox(t)
	tool := NewTaskListTool()
	ctx := context.Background()

	r1 := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "a"}), sb)
	l1 := parseList(t, r1)
	if len(l1.Tasks) != 1 || l1.Tasks[0].ID != 1 || l1.NextID != 2 {
		t.Fatalf("after add a: %+v", l1)
	}

	r2 := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "b"}), sb)
	l2 := parseList(t, r2)
	if len(l2.Tasks) != 2 || l2.Tasks[0].ID != 1 || l2.Tasks[1].ID != 2 {
		t.Fatalf("after add b: %+v", l2)
	}

	// The file actually exists on disk.
	if _, err := os.Stat(filepath.Join(sb.root, sporecore.TaskListPath)); err != nil {
		t.Fatalf("task list file missing: %v", err)
	}

	// list_tasks returns the same list and does not mutate.
	r3 := tool.Execute(ctx, tlCall(map[string]any{"action": "list_tasks"}), sb)
	l3 := parseList(t, r3)
	a, _ := json.Marshal(l2)
	b, _ := json.Marshal(l3)
	if string(a) != string(b) {
		t.Fatalf("list_tasks mutated: %s != %s", a, b)
	}
}

func TestUpdateStatusAndComplete(t *testing.T) {
	sb := newTLSandbox(t)
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "x"}), sb)

	r := tool.Execute(ctx, tlCall(map[string]any{"action": "update_task", "id": 1, "status": "in_progress"}), sb)
	if parseList(t, r).Tasks[0].Status != sporecore.TaskStatusInProgress {
		t.Fatalf("update: %+v", r)
	}

	r = tool.Execute(ctx, tlCall(map[string]any{"action": "complete_task", "id": 1}), sb)
	if parseList(t, r).Tasks[0].Status != sporecore.TaskStatusCompleted {
		t.Fatalf("complete: %+v", r)
	}
}

func TestUpdateDescriptionViaTool(t *testing.T) {
	sb := newTLSandbox(t)
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "x"}), sb)

	r := tool.Execute(ctx, tlCall(map[string]any{"action": "update_task", "id": 1, "description": "y"}), sb)
	l := parseList(t, r)
	if l.Tasks[0].Description != "y" || l.Tasks[0].Status != sporecore.TaskStatusPending {
		t.Fatalf("update description: %+v", l.Tasks[0])
	}
}

func TestUnknownIDIsRecoverableError(t *testing.T) {
	sb := newTLSandbox(t)
	r := NewTaskListTool().Execute(context.Background(),
		tlCall(map[string]any{"action": "complete_task", "id": 42}), sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "not found") {
		t.Fatalf("message %q does not mention not found", r.Message)
	}
}

func TestInvalidTransitionOutOfCompletedIsRecoverable(t *testing.T) {
	sb := newTLSandbox(t)
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "x"}), sb)
	tool.Execute(ctx, tlCall(map[string]any{"action": "complete_task", "id": 1}), sb)
	r := tool.Execute(ctx, tlCall(map[string]any{"action": "update_task", "id": 1, "status": "pending"}), sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if !strings.Contains(r.Message, "invalid transition") {
		t.Fatalf("message %q does not mention invalid transition", r.Message)
	}
}

func TestBadParamsIsRecoverableError(t *testing.T) {
	sb := newTLSandbox(t)
	tool := NewTaskListTool()
	ctx := context.Background()
	cases := []map[string]any{
		{"action": "nope"},          // unknown action
		{"action": "add_task"},      // missing description
		{"action": "update_task"},   // missing id
		{"action": "complete_task"}, // missing id
	}
	for _, c := range cases {
		r := tool.Execute(ctx, tlCall(c), sb)
		if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
			t.Fatalf("%v: expected recoverable error, got %+v", c, r)
		}
	}
	// Empty input is also recoverable.
	r := tool.Execute(ctx, sporecore.ToolCall{ID: "c1", Name: TaskListToolName, Input: json.RawMessage(`{}`)}, sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("empty input: expected recoverable error, got %+v", r)
	}
}

func TestSchemaIsNotReadOnly(t *testing.T) {
	s := NewTaskListTool().Schema()
	if s.Annotations.ReadOnly {
		t.Fatal("task_list must NOT be read_only (it mutates shared on-disk state)")
	}
	if s.Annotations.Destructive || s.Annotations.OpenWorld {
		t.Fatalf("task_list should not be destructive/open_world: %+v", s.Annotations)
	}
	if !json.Valid(s.Parameters) {
		t.Fatal("schema parameters are not valid JSON")
	}
}

func TestPersistThenReloadYieldsIdenticalList(t *testing.T) {
	sb := newTLSandbox(t)
	tool := NewTaskListTool()
	ctx := context.Background()
	tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "one"}), sb)
	r := tool.Execute(ctx, tlCall(map[string]any{"action": "add_task", "description": "two"}), sb)
	fromTool := parseList(t, r)

	reloaded, v, err := sporecore.LoadTaskList(ctx, sb)
	if v != nil || err != nil {
		t.Fatalf("reload: v=%v err=%v", v, err)
	}
	a, _ := json.Marshal(fromTool)
	b, _ := json.Marshal(reloaded)
	if string(a) != string(b) {
		t.Fatalf("tool vs disk differ: %s != %s", a, b)
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

// Replay each operations scenario step-by-step against a real on-disk
// read-modify-write, asserting the resulting list (or error kind) per step.
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
	ctx := context.Background()

	for _, sc := range scenarios {
		// Fresh isolated workspace per scenario.
		sb := newTLSandbox(t)
		for i, step := range sc.Steps {
			out := tool.Execute(ctx, sporecore.ToolCall{ID: "c1", Name: TaskListToolName, Input: step.Action}, sb)
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
