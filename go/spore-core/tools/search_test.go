package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestGrepFindsMatches(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	_ = os.WriteFile(p, []byte("alpha\nbeta\nalpha2"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewGrepFilesTool().Execute(context.Background(),
		call("grep_files", "c1", map[string]any{"pattern": "^alpha", "path": dir}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	if !strings.Contains(r.Content, "alpha") || !strings.Contains(r.Content, "alpha2") {
		t.Fatalf("expected matches in %q", r.Content)
	}
}

func TestGrepInvalidRegex(t *testing.T) {
	dir := t.TempDir()
	sb := sporecore.AllowAllSandbox{}
	r := NewGrepFilesTool().Execute(context.Background(),
		call("grep_files", "c1", map[string]any{"pattern": "(unclosed", "path": dir}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}

func TestFindFilesGlob(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"a.go", "b.go", "c.txt"} {
		_ = os.WriteFile(filepath.Join(dir, n), nil, 0o644)
	}
	sb := sporecore.AllowAllSandbox{}
	r := NewFindFilesTool().Execute(context.Background(),
		call("find_files", "c1", map[string]any{"glob": "*.go", "path": dir}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	lines := splitLines(r.Content)
	if len(lines) != 2 {
		t.Fatalf("expected 2 .go files, got %d: %v", len(lines), lines)
	}
}
