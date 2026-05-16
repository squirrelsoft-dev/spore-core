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

	w := NewWriteFileTool().Execute(ctx, call("write_file", "c1", map[string]any{"path": p, "content": "hello"}), sb)
	if w.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("write: %+v", w)
	}
	r := NewReadFileTool().Execute(ctx, call("read_file", "c2", map[string]any{"path": p}), sb)
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
	w.Execute(ctx, call("write_file", "c1", map[string]any{"path": p, "content": "a"}), sb)
	w.Execute(ctx, call("write_file", "c2", map[string]any{"path": p, "content": "b", "append": true}), sb)
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
	r := NewListDirTool().Execute(context.Background(), call("list_dir", "c1", map[string]any{"path": dir}), sb)
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

func TestDeleteMissingIsRecoverable(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewDeleteFileTool().Execute(context.Background(),
		call("delete_file", "c1", map[string]any{"path": "/no/such/path/here"}), sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

func TestMoveFileRenames(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "s")
	dst := filepath.Join(dir, "d")
	_ = os.WriteFile(src, []byte("hi"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewMoveFileTool().Execute(context.Background(),
		call("move_file", "c1", map[string]any{"src": src, "dst": dst}), sb)
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
		sporecore.ToolCall{ID: "c1", Name: "read_file", Input: json.RawMessage(`{}`)}, sb)
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
