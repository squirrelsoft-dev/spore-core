package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func call(name, id string, input any) sporecore.ToolCall {
	b, _ := json.Marshal(input)
	return sporecore.ToolCall{ID: id, Name: name, Input: b}
}

func TestWriteThenReadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	sb := sporecore.AllowAllSandbox{}
	ctx := context.Background()

	w := NewWriteFileTool().Execute(ctx, call("write_file", "c1", map[string]any{"path": p, "content": "hello"}), sb, nil)
	if w.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("write: %+v", w)
	}
	r := NewReadFileTool().Execute(ctx, call("read_file", "c2", map[string]any{"path": p}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "hello" {
		t.Fatalf("read: %+v", r)
	}
}

func TestAppendModeConcatenates(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	sb := sporecore.AllowAllSandbox{}
	ctx := context.Background()
	w := NewWriteFileTool()
	w.Execute(ctx, call("write_file", "c1", map[string]any{"path": p, "content": "a"}), sb, nil)
	w.Execute(ctx, call("write_file", "c2", map[string]any{"path": p, "content": "b", "append": true}), sb, nil)
	got, _ := os.ReadFile(p)
	if string(got) != "ab" {
		t.Fatalf("got %q", got)
	}
}

func TestListDirSorted(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"z", "a", "m"} {
		_ = os.WriteFile(filepath.Join(dir, n), nil, 0o644)
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewListDirTool().Execute(context.Background(), call("list_dir", "c1", map[string]any{"path": dir}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	// Lines must be sorted.
	lines := splitLines(r.Content)
	for i := 1; i < len(lines); i++ {
		if lines[i] < lines[i-1] {
			t.Fatalf("not sorted: %v", lines)
		}
	}
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d: %v", len(lines), lines)
	}
}

// Regression for #93: every entry list_dir returns must round-trip straight
// back into read_file under the *real* WorkspaceScopedSandbox, which treats
// all input paths as root-relative. Absolute paths (the old behavior) would
// be rejected as a non-recoverable path-escape violation.
func TestListDirEntriesRoundtripThroughWorkspaceSandbox(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("alpha"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "b.txt"), []byte("beta"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "sub", "c.txt"), []byte("gamma"), 0o644); err != nil {
		t.Fatal(err)
	}
	sb, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// Recursive so we exercise both top-level files and a nested file.
	r := NewListDirTool().Execute(ctx,
		call("list_dir", "c1", map[string]any{"path": ".", "recursive": true}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("list_dir failed: %+v", r)
	}
	entries := splitLines(r.Content)

	var sawTop, sawNested bool
	for _, e := range entries {
		if e == "a.txt" {
			sawTop = true
		}
		if e == "sub/c.txt" {
			sawNested = true
		}
		if e == "" || e == "." {
			t.Fatalf("must not emit the listed dir itself, got %v", entries)
		}
	}
	if !sawTop {
		t.Fatalf("expected bare root-relative name a.txt, got %v", entries)
	}
	if !sawNested {
		t.Fatalf("expected nested entry sub/c.txt, got %v", entries)
	}

	// The actual bug check: feed each entry straight into read_file.
	for _, entry := range entries {
		rr := NewReadFileTool().Execute(ctx,
			call("read_file", "c2", map[string]any{"path": entry}), sb, nil)
		// A directory entry (e.g. `sub`) reads as a recoverable error but must
		// NOT be a non-recoverable sandbox violation — that's the regression.
		if rr.Kind == sporecore.ToolOutputError && !rr.Recoverable {
			t.Fatalf("entry %q did not round-trip: %+v", entry, rr)
		}
	}
}

func TestDeleteMissingIsRecoverable(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewDeleteFileTool().Execute(context.Background(),
		call("delete_file", "c1", map[string]any{"path": "/no/such/path/here"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// Regression for #63: a read_file of a not-yet-created file *inside* the
// workspace must surface a recoverable not-found, not a (non-recoverable)
// sandbox path-escape violation.
func TestReadMissingInWorkspaceFileIsRecoverableNotFound(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	r := NewReadFileTool().Execute(context.Background(),
		call("read_file", "c1", map[string]any{"path": "output.txt"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable not-found error, got %+v", r)
	}
}

// Regression for #63: a read_file of a path resolving *outside* the workspace
// root must still be a (non-recoverable) sandbox path-escape violation.
func TestReadOutsideRootIsPathEscape(t *testing.T) {
	dir := t.TempDir()
	root, _ := filepath.EvalSymlinks(dir)
	sb, err := sporecore.NewWorkspaceScopedSandbox(sporecore.WorkspaceConfig{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	r := NewReadFileTool().Execute(context.Background(),
		call("read_file", "c1", map[string]any{"path": "../nonexistent_passwd"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || r.Recoverable {
		t.Fatalf("expected non-recoverable path-escape error, got %+v", r)
	}
}

func TestMoveFileRenames(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "s")
	dst := filepath.Join(dir, "d")
	_ = os.WriteFile(src, []byte("hi"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewMoveFileTool().Execute(context.Background(),
		call("move_file", "c1", map[string]any{"src": src, "dst": dst}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	if _, err := os.Stat(src); err == nil {
		t.Fatalf("src still exists")
	}
	if _, err := os.Stat(dst); err != nil {
		t.Fatalf("dst missing: %v", err)
	}
}

func TestInvalidParamsReturnsRecoverableError(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewReadFileTool().Execute(context.Background(),
		sporecore.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, sb, nil)
	// Missing required field is decoded as zero-value path; the actual
	// failure surfaces from os.ReadFile. We accept either recoverable error
	// (Rust serde-style "missing field") or the os error path.
	if r.Kind != sporecore.ToolOutputError {
		t.Fatalf("expected error, got %+v", r)
	}
	if !r.Recoverable {
		t.Fatalf("expected recoverable")
	}
}
