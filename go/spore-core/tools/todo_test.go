package tools

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestTodoWritePersistsUnderTodoKey(t *testing.T) {
	tc, store := inMemCtx()
	r := NewTodoWriteTool().Execute(context.Background(),
		call("todo_write", "c1", map[string]any{"todos": []map[string]any{
			{"content": "a", "status": "pending"},
			{"content": "b", "status": "in_progress"},
		}}), sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	var got []TodoItem
	if err := json.Unmarshal([]byte(r.Content), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[1].Status != TodoStatusInProgress {
		t.Fatalf("got %+v", got)
	}
	// Persisted under the "todo" key.
	blob, found, err := store.Get(context.Background(), sporecore.SessionID(testSession), TodoStoreKey)
	if err != nil || !found {
		t.Fatalf("todo blob not present: found=%v err=%v", found, err)
	}
	var persisted []TodoItem
	if err := json.Unmarshal(blob, &persisted); err != nil {
		t.Fatal(err)
	}
	if len(persisted) != 2 {
		t.Fatalf("persisted %+v", persisted)
	}
}

func TestTodoWriteReplacesWholesale(t *testing.T) {
	tc, _ := inMemCtx()
	tool := NewTodoWriteTool()
	tool.Execute(context.Background(),
		call("todo_write", "c1", map[string]any{"todos": []map[string]any{
			{"content": "old1", "status": "pending"},
			{"content": "old2", "status": "pending"},
		}}), sporecore.AllowAllSandbox{}, tc)
	r := tool.Execute(context.Background(),
		call("todo_write", "c2", map[string]any{"todos": []map[string]any{
			{"content": "new", "status": "completed"},
		}}), sporecore.AllowAllSandbox{}, tc)
	var got []TodoItem
	if err := json.Unmarshal([]byte(r.Content), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Content != "new" {
		t.Fatalf("got %+v", got)
	}
}

func TestTodoWriteBadParamsRecoverable(t *testing.T) {
	tc, _ := inMemCtx()
	r := NewTodoWriteTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "todo_write", Input: json.RawMessage(`{"todos": "not-an-array"}`)},
		sporecore.AllowAllSandbox{}, tc)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

func TestTodoWriteSchemaNotReadOnly(t *testing.T) {
	s := NewTodoWriteTool().Schema()
	if s.Annotations.ReadOnly || s.Annotations.Destructive {
		t.Fatalf("todo_write must be neither read_only nor destructive: %+v", s.Annotations)
	}
}

// Fixture replay: todo_write.json.
func TestTodoWriteFixtureReplay(t *testing.T) {
	type step struct {
		Input             json.RawMessage `json:"input"`
		ExpectedPersisted []TodoItem      `json:"expected_persisted"`
	}
	type todoCase struct {
		Name  string `json:"name"`
		Steps []step `json:"steps"`
	}
	data := readFixture(t, "todo_write.json")
	var cases []todoCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			tc, store := inMemCtx()
			tool := NewTodoWriteTool()
			for i, s := range c.Steps {
				r := tool.Execute(context.Background(),
					sporecore.ToolCall{ID: "c", Name: "todo_write", Input: s.Input}, sporecore.AllowAllSandbox{}, tc)
				if r.Kind != sporecore.ToolOutputSuccess {
					t.Fatalf("step %d: expected success, got %+v", i, r)
				}
				blob, found, err := store.Get(context.Background(), sporecore.SessionID(testSession), TodoStoreKey)
				if err != nil || !found {
					t.Fatalf("step %d: persisted blob missing", i)
				}
				var persisted []TodoItem
				if err := json.Unmarshal(blob, &persisted); err != nil {
					t.Fatal(err)
				}
				if len(persisted) != len(s.ExpectedPersisted) {
					t.Fatalf("step %d: persisted len got %d want %d", i, len(persisted), len(s.ExpectedPersisted))
				}
				for j := range persisted {
					if persisted[j] != s.ExpectedPersisted[j] {
						t.Fatalf("step %d item %d: got %+v want %+v", i, j, persisted[j], s.ExpectedPersisted[j])
					}
				}
			}
		})
	}
}
