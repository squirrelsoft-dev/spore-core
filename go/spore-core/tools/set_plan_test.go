package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// TestCreateDirCreatesNestedDirectories mirrors the Rust create_dir test:
// mkdir -p semantics — missing parents are created.
func TestCreateDirCreatesNestedDirectories(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "a", "b", "c")
	r := NewCreateDirTool().Execute(context.Background(),
		call("create_dir", "c1", map[string]any{"path": "a/b/c"}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	info, err := os.Stat(target)
	if err != nil {
		t.Fatalf("directory not created: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("not a directory")
	}
}

// TestCreateDirPathEscapeIsTypedViolation — a path outside the workspace root
// surfaces the typed SandboxViolation, not a fake success.
func TestCreateDirPathEscapeIsTypedViolation(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	r := NewCreateDirTool().Execute(context.Background(),
		call("create_dir", "c1", map[string]any{"path": "../outside"}), sb, nil)
	if r.Kind != sporecore.ToolOutputSandboxViolation ||
		r.Violation == nil || r.Violation.Kind != sporecore.SandboxPathEscape {
		t.Fatalf("expected typed PathEscape SandboxViolation, got %+v", r)
	}
}

// TestSetPlanPersistsLinearTaskList mirrors the Rust set_plan behavior: an
// ordered, non-empty task list is persisted under TaskListExtrasKey and the
// success content reports the step count.
func TestSetPlanPersistsLinearTaskList(t *testing.T) {
	store := sporecore.NewInMemoryToolRunStore()
	ctx := sporecore.NewToolContext("sess", store, nil)
	r := NewSetPlanTool().Execute(context.Background(),
		call("set_plan", "c1", map[string]any{
			"tasks":     []any{"step one", "step two", "step three"},
			"rationale": "because",
		}), sporecore.AllowAllSandbox{}, ctx)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	if r.Content != "Recorded a 3-step plan." {
		t.Fatalf("unexpected content %q", r.Content)
	}
	// The durable blob round-trips through TaskList load.
	raw, found, err := store.Get(context.Background(), "sess", sporecore.TaskListExtrasKey)
	if err != nil || !found {
		t.Fatalf("durable blob missing: found=%v err=%v", found, err)
	}
	var list sporecore.TaskList
	if err := json.Unmarshal(raw, &list); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(list.Tasks) != 3 {
		t.Fatalf("expected 3 tasks, got %d", len(list.Tasks))
	}
	for i, task := range list.Tasks {
		if task.Description != []string{"step one", "step two", "step three"}[i] {
			t.Fatalf("task %d description = %q", i, task.Description)
		}
		if task.Status != sporecore.TaskStatusPending {
			t.Fatalf("task %d status = %v, want pending", i, task.Status)
		}
	}
	if list.NextID != 4 { // started at 1, added 3
		t.Fatalf("next_id = %d, want 4", list.NextID)
	}
}

// TestSetPlanRejectsEmptyTasks — an empty tasks array is a recoverable error.
func TestSetPlanRejectsEmptyTasks(t *testing.T) {
	ctx := sporecore.NewToolContext("sess", sporecore.NewInMemoryToolRunStore(), nil)
	r := NewSetPlanTool().Execute(context.Background(),
		call("set_plan", "c1", map[string]any{"tasks": []any{}}), sporecore.AllowAllSandbox{}, ctx)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// TestSetPlanRejectsBlankTask — a blank step string is a recoverable error.
func TestSetPlanRejectsBlankTask(t *testing.T) {
	ctx := sporecore.NewToolContext("sess", sporecore.NewInMemoryToolRunStore(), nil)
	r := NewSetPlanTool().Execute(context.Background(),
		call("set_plan", "c1", map[string]any{"tasks": []any{"ok", "   "}}), sporecore.AllowAllSandbox{}, ctx)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// TestSetPlanSchemaRequiresTasksArray pins the schema shape (cross-language).
func TestSetPlanSchemaRequiresTasksArray(t *testing.T) {
	s := NewSetPlanTool().Schema()
	if s.Name != "set_plan" {
		t.Fatalf("name = %q", s.Name)
	}
	var params map[string]any
	if err := json.Unmarshal(s.Parameters, &params); err != nil {
		t.Fatal(err)
	}
	required, _ := params["required"].([]any)
	hasTasks := false
	for _, r := range required {
		if r == "tasks" {
			hasTasks = true
		}
	}
	if !hasTasks {
		t.Fatalf("tasks must be required: %v", required)
	}
	props, _ := params["properties"].(map[string]any)
	tasksProp, _ := props["tasks"].(map[string]any)
	if tasksProp["type"] != "array" {
		t.Fatalf("tasks type = %v, want array", tasksProp["type"])
	}
}

// TestCodingSetIncludesNewTools is the cross-language name-set drift guard for
// the §A1/§A2/§A3 additions: CodingSet must include create_dir, move_file,
// delete_file, and set_plan (matching the Rust coding_set exactly).
func TestCodingSetIncludesNewTools(t *testing.T) {
	n := names((StandardTools{}).CodingSet())
	for _, want := range []string{"create_dir", "move_file", "delete_file", "set_plan"} {
		if !n[want] {
			t.Fatalf("coding_set missing %q", want)
		}
	}
}

// TestFullSetIncludesNewTools confirms FullSet inherits the new tools.
func TestFullSetIncludesNewTools(t *testing.T) {
	n := names((StandardTools{}).FullSet())
	for _, want := range []string{"create_dir", "move_file", "delete_file", "set_plan"} {
		if !n[want] {
			t.Fatalf("full_set missing %q", want)
		}
	}
}

// TestReadonlySetExcludesNewMutatingTools confirms the new mutating tools are
// NOT in the read-only set.
func TestReadonlySetExcludesNewMutatingTools(t *testing.T) {
	n := names((StandardTools{}).ReadonlySet())
	for _, forbidden := range []string{"create_dir", "move_file", "delete_file", "set_plan"} {
		if n[forbidden] {
			t.Fatalf("readonly_set leaked %q", forbidden)
		}
	}
}
