package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestEditReplacesUniqueOccurrence(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	if err := os.WriteFile(p, []byte("hello world\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewEditFileTool().Execute(context.Background(),
		call("edit_file", "c1", map[string]any{"path": p, "old_string": "world", "new_string": "there"}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	got, _ := os.ReadFile(p)
	if string(got) != "hello there\n" {
		t.Fatalf("got %q", got)
	}
}

func TestEditNotFoundRecoverable(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	_ = os.WriteFile(p, []byte("hello\n"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewEditFileTool().Execute(context.Background(),
		call("edit_file", "c1", map[string]any{"path": p, "old_string": "absent", "new_string": "x"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

func TestEditNotUniqueRecoverable(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	_ = os.WriteFile(p, []byte("x x x\n"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewEditFileTool().Execute(context.Background(),
		call("edit_file", "c1", map[string]any{"path": p, "old_string": "x", "new_string": "y"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

func TestEditMissingFileRecoverable(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewEditFileTool().Execute(context.Background(),
		call("edit_file", "c1", map[string]any{"path": "/no/such/file", "old_string": "a", "new_string": "b"}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

func TestEditSchemaIsDestructive(t *testing.T) {
	s := NewEditFileTool().Schema()
	if !s.Annotations.Destructive {
		t.Fatalf("expected destructive")
	}
	if s.Annotations.ReadOnly {
		t.Fatalf("must not be read_only")
	}
}

// Fixture replay: edit_file_cases.json.
func TestEditFileFixtureReplay(t *testing.T) {
	type expected struct {
		Kind         string `json:"kind"`
		Recoverable  bool   `json:"recoverable"`
		Reason       string `json:"reason"`
		FinalContent string `json:"final_content"`
	}
	type editCase struct {
		Name           string   `json:"name"`
		InitialContent string   `json:"initial_content"`
		OldString      string   `json:"old_string"`
		NewString      string   `json:"new_string"`
		Expected       expected `json:"expected"`
	}
	data := readFixture(t, "edit_file_cases.json")
	var cases []editCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	sb := sporecore.AllowAllSandbox{}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			dir := t.TempDir()
			p := filepath.Join(dir, "f.txt")
			if err := os.WriteFile(p, []byte(c.InitialContent), 0o644); err != nil {
				t.Fatal(err)
			}
			r := NewEditFileTool().Execute(context.Background(),
				call("edit_file", "c1", map[string]any{"path": p, "old_string": c.OldString, "new_string": c.NewString}), sb, nil)
			switch c.Expected.Kind {
			case "success":
				if r.Kind != sporecore.ToolOutputSuccess {
					t.Fatalf("expected success, got %+v", r)
				}
				got, _ := os.ReadFile(p)
				if string(got) != c.Expected.FinalContent {
					t.Fatalf("final content: got %q want %q", got, c.Expected.FinalContent)
				}
			case "error":
				if r.Kind != sporecore.ToolOutputError {
					t.Fatalf("expected error, got %+v", r)
				}
				if r.Recoverable != c.Expected.Recoverable {
					t.Fatalf("recoverable: got %v want %v", r.Recoverable, c.Expected.Recoverable)
				}
			}
		})
	}
}

// readFixture loads a shared fixture from /fixtures/tools.
func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	_, here, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(here), "..", "..", "..", "fixtures", "tools", name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return data
}
