package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func grepOut(t *testing.T, dir, mode string) string {
	t.Helper()
	sb := sporecore.AllowAllSandbox{}
	r := NewGrepTool().Execute(context.Background(),
		call("grep", "c1", map[string]any{
			"pattern": "alpha", "path": dir, "recursive": true, "output_mode": mode,
		}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("grep %s: %+v", mode, r)
	}
	return r.Content
}

func TestGrepOutputModeContent(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha\nbeta\nalpha2"), 0o644)
	out := grepOut(t, dir, "content")
	if len(splitLines(out)) != 2 {
		t.Fatalf("expected 2 lines, got %q", out)
	}
	if !strings.Contains(out, ":1:alpha") || !strings.Contains(out, ":3:alpha2") {
		t.Fatalf("content mode: %q", out)
	}
}

func TestGrepOutputModeFilesWithMatches(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha\nalpha"), 0o644)
	_ = os.WriteFile(filepath.Join(dir, "b.txt"), []byte("nope"), 0o644)
	out := grepOut(t, dir, "files_with_matches")
	if len(splitLines(out)) != 1 {
		t.Fatalf("expected 1 line, got %q", out)
	}
	if !strings.HasSuffix(out, "a.txt") {
		t.Fatalf("files_with_matches: %q", out)
	}
}

func TestGrepOutputModeCount(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha\nalpha\nx"), 0o644)
	out := grepOut(t, dir, "count")
	if len(splitLines(out)) != 1 {
		t.Fatalf("expected 1 line, got %q", out)
	}
	if !strings.HasSuffix(out, ":2") {
		t.Fatalf("count: %q", out)
	}
}

func TestGrepDefaultsToContent(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "a.txt"), []byte("alpha"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewGrepTool().Execute(context.Background(),
		call("grep", "c1", map[string]any{"pattern": "alpha", "path": dir}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess || !strings.Contains(r.Content, ":1:alpha") {
		t.Fatalf("default content mode: %+v", r)
	}
}

func TestGrepInvalidRegexRecoverable(t *testing.T) {
	dir := t.TempDir()
	sb := sporecore.AllowAllSandbox{}
	r := NewGrepTool().Execute(context.Background(),
		call("grep", "c1", map[string]any{"pattern": "(unclosed", "path": dir}), sb, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

// Fixture replay: grep_output_modes.json.
func TestGrepOutputModesFixtureReplay(t *testing.T) {
	type grepCase struct {
		Name             string            `json:"name"`
		Files            map[string]string `json:"files"`
		Pattern          string            `json:"pattern"`
		OutputMode       string            `json:"output_mode"`
		ExpectedLines    int               `json:"expected_lines"`
		ExpectedContains []string          `json:"expected_contains"`
	}
	data := readFixture(t, "grep_output_modes.json")
	var cases []grepCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	sb := sporecore.AllowAllSandbox{}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			dir := t.TempDir()
			for name, content := range c.Files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			r := NewGrepTool().Execute(context.Background(),
				call("grep", "c1", map[string]any{
					"pattern": c.Pattern, "path": dir, "recursive": true, "output_mode": c.OutputMode,
				}), sb, nil)
			if r.Kind != sporecore.ToolOutputSuccess {
				t.Fatalf("expected success, got %+v", r)
			}
			if got := len(splitLines(r.Content)); got != c.ExpectedLines {
				t.Fatalf("lines: got %d want %d (%q)", got, c.ExpectedLines, r.Content)
			}
			for _, want := range c.ExpectedContains {
				if !strings.Contains(r.Content, want) {
					t.Fatalf("expected %q in %q", want, r.Content)
				}
			}
		})
	}
}
