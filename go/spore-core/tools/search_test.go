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

// ============================================================================
// GrepTool context_lines unit tests
// ============================================================================

// grepContextOut is a helper that runs GrepTool.Execute and returns the content.
func grepContextOut(t *testing.T, filePath string, params map[string]any) string {
	t.Helper()
	sb := sporecore.AllowAllSandbox{}
	params["path"] = filePath
	r := NewGrepTool().Execute(context.Background(),
		call("grep", "c1", params), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	return r.Content
}

func TestGrepContextLinesZeroUnchanged(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	_ = os.WriteFile(p, []byte("alpha\nbeta\ngamma\n"), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "beta", "context_lines": 0})
	want := p + ":2:beta"
	if out != want {
		t.Fatalf("context_lines=0: got %q, want %q", out, want)
	}
}

func TestGrepContextLinesSingleMatch(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	_ = os.WriteFile(p, []byte("one\ntwo\nthree\nfour\nfive\n"), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "three", "context_lines": 1})
	want := p + ":2-two\n" + p + ":3:three\n" + p + ":4-four"
	if out != want {
		t.Fatalf("single match: got %q, want %q", out, want)
	}
}

func TestGrepContextLinesOverlappingWindowsMerged(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	_ = os.WriteFile(p, []byte("a\nb\nc\nd\ne\n"), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "b|d", "context_lines": 2})
	want := p + ":1-a\n" + p + ":2:b\n" + p + ":3-c\n" + p + ":4:d\n" + p + ":5-e"
	if out != want {
		t.Fatalf("overlapping: got %q, want %q", out, want)
	}
}

func TestGrepContextLinesNonOverlappingGroupsSeparated(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	content := "match1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nmatch10\nline11\nline12\n"
	_ = os.WriteFile(p, []byte(content), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "match", "context_lines": 1})
	want := p + ":1:match1\n" + p + ":2-line2\n--\n" + p + ":9-line9\n" + p + ":10:match10\n" + p + ":11-line11"
	if out != want {
		t.Fatalf("non-overlapping: got %q, want %q", out, want)
	}
}

func TestGrepContextLinesClampedAtStart(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	_ = os.WriteFile(p, []byte("match\nline2\nline3\nline4\nline5\n"), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "match", "context_lines": 3})
	want := p + ":1:match\n" + p + ":2-line2\n" + p + ":3-line3\n" + p + ":4-line4"
	if out != want {
		t.Fatalf("clamp start: got %q, want %q", out, want)
	}
}

func TestGrepContextLinesClampedAtEnd(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	_ = os.WriteFile(p, []byte("line1\nline2\nline3\nline4\nmatch\n"), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "match", "context_lines": 3})
	want := p + ":2-line2\n" + p + ":3-line3\n" + p + ":4-line4\n" + p + ":5:match"
	if out != want {
		t.Fatalf("clamp end: got %q, want %q", out, want)
	}
}

func TestGrepContextLineMatchAlsoMatch(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.txt")
	_ = os.WriteFile(p, []byte("alpha\nbeta\ngamma\n"), 0o644)
	out := grepContextOut(t, p, map[string]any{"pattern": "alpha|beta", "context_lines": 1})
	want := p + ":1:alpha\n" + p + ":2:beta\n" + p + ":3-gamma"
	if out != want {
		t.Fatalf("context line is match: got %q, want %q", out, want)
	}
}

// ============================================================================
// Fixture replay
// ============================================================================

type grepFixtureCase struct {
	Name           string          `json:"name"`
	InitialContent string          `json:"initial_content"`
	Params         json.RawMessage `json:"params"`
	Expected       string          `json:"expected"`
}

func TestGrepContextLinesFixtureReplay(t *testing.T) {
	fixturePath := "../../../fixtures/tools/grep_context_lines.json"
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []grepFixtureCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.Name, func(t *testing.T) {
			// Write the initial content to a temp file.
			dir := t.TempDir()
			p := filepath.Join(dir, "fixture.txt")
			_ = os.WriteFile(p, []byte(tc.InitialContent), 0o644)

			// Decode params, substituting <FIXTURE_PATH> with the actual path.
			raw := strings.ReplaceAll(string(tc.Params), "<FIXTURE_PATH>", p)
			var params map[string]any
			if err := json.Unmarshal([]byte(raw), &params); err != nil {
				t.Fatalf("parse params: %v", err)
			}

			sb := sporecore.AllowAllSandbox{}
			r := NewGrepTool().Execute(context.Background(),
				call("grep", "c1", params), sb, nil)
			if r.Kind != sporecore.ToolOutputSuccess {
				t.Fatalf("expected success, got %+v", r)
			}

			expected := strings.ReplaceAll(tc.Expected, "<FIXTURE_PATH>", p)
			if r.Content != expected {
				t.Fatalf("case %q:\ngot:  %q\nwant: %q", tc.Name, r.Content, expected)
			}
		})
	}
}
