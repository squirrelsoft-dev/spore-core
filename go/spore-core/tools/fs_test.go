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

// ---- #132: read_file range scan + line numbers ----

// parseReadFileParams is a test helper that deserializes JSON into ReadFileParams.
func parseReadFileParams(t *testing.T, raw string) *ReadFileParams {
	t.Helper()
	var p ReadFileParams
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return &p
}

func TestReadRangeDefaultsByteIdentical(t *testing.T) {
	body := "line1\nline2\nline3\n"
	params := parseReadFileParams(t, `{"path":"f"}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	if got != body {
		t.Fatalf("got %q, want %q", got, body)
	}
}

func TestReadRangeOffsetHeaderRunsToEOF(t *testing.T) {
	body := "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
	params := parseReadFileParams(t, `{"path":"f","offset":3}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 3–10 of 10]\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestReadRangeLengthTrimsAtEOFSilently(t *testing.T) {
	body := "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
	// offset 8 + length 5 would reach line 12, but only 10 lines exist.
	params := parseReadFileParams(t, `{"path":"f","offset":8,"length":5}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 8–10 of 10]\nline8\nline9\nline10\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestReadRangeLineNumbersPadToTotalWidth(t *testing.T) {
	body := "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
	// total = 10 → width 2 → single-digit numbers are right-padded.
	params := parseReadFileParams(t, `{"path":"f","offset":2,"length":3,"line_numbers":true}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 2–4 of 10]\n 2 | line2\n 3 | line3\n 4 | line4\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestReadRangeLineNumbersNoPadWhenSingleDigitTotal(t *testing.T) {
	body := "alpha\nbeta\ngamma\n"
	params := parseReadFileParams(t, `{"path":"f","line_numbers":true}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 1–3 of 3]\n1 | alpha\n2 | beta\n3 | gamma\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestReadRangeLengthZeroAlwaysMeansNoLimit(t *testing.T) {
	body := "line1\nline2\nline3\nline4\nline5\n"
	// length: 0 with offset > 1 is never an error — it reads to EOF.
	params := parseReadFileParams(t, `{"path":"f","offset":3,"length":0}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 3–5 of 5]\nline3\nline4\nline5\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestReadRangeOffsetZeroIsError(t *testing.T) {
	body := "alpha\nbeta\n"
	params := parseReadFileParams(t, `{"path":"f","offset":0}`)
	_, errMsg := applyReadRange(body, params)
	if errMsg == "" {
		t.Fatal("expected an error for offset=0")
	}
	if !strings.Contains(errMsg, "offset") {
		t.Fatalf("error message should mention 'offset', got %q", errMsg)
	}
}

func TestReadRangeOffsetPastEOFIsError(t *testing.T) {
	body := "alpha\nbeta\ngamma\n"
	params := parseReadFileParams(t, `{"path":"f","offset":11}`)
	_, errMsg := applyReadRange(body, params)
	if errMsg != "offset 11 exceeds file length 3" {
		t.Fatalf("got %q", errMsg)
	}
}

func TestReadRangeEmptyFileAnyParamsNoHeader(t *testing.T) {
	params := parseReadFileParams(t, `{"path":"f","offset":1,"length":5,"line_numbers":true}`)
	got, errMsg := applyReadRange("", params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	if got != "" {
		t.Fatalf("got %q, want empty string", got)
	}
}

func TestReadRangeFinalLineWithoutNewlinePreserved(t *testing.T) {
	// Last line lacks a trailing '\n'; SplitAfter keeps it verbatim.
	body := "a\nb\nc"
	params := parseReadFileParams(t, `{"path":"f","offset":2}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 2–3 of 3]\nb\nc"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestReadFileLengthOnly(t *testing.T) {
	body := "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
	params := parseReadFileParams(t, `{"path":"f","length":3}`)
	got, errMsg := applyReadRange(body, params)
	if errMsg != "" {
		t.Fatalf("unexpected error: %s", errMsg)
	}
	want := "[lines 1–3 of 10]\nline1\nline2\nline3\n"
	if got != want {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

// TestReadFileWithOffsetEndToEnd tests the full Execute path with offset.
func TestReadFileWithOffsetEndToEnd(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	_ = os.WriteFile(p, []byte("l1\nl2\nl3\n"), 0o644)
	sb := sporecore.AllowAllSandbox{}
	r := NewReadFileTool().Execute(context.Background(),
		call("read_file", "c1", map[string]any{"path": p, "offset": 2}), sb, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("%+v", r)
	}
	want := "[lines 2–3 of 3]\nl2\nl3\n"
	if r.Content != want {
		t.Fatalf("got %q\nwant %q", r.Content, want)
	}
}

// TestReadFileRangeFixtureReplay loads the shared fixture and verifies the Go
// implementation produces the expected outcome for every case.
func TestReadFileRangeFixtureReplay(t *testing.T) {
	_, here, _, _ := runtime.Caller(0)
	fixturePath := filepath.Join(filepath.Dir(here), "..", "..", "..", "fixtures", "tools", "read_file_range.json")
	data, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Skipf("fixture not found: %v", err)
	}

	var scenarios []struct {
		Name           string          `json:"name"`
		InitialContent string          `json:"initial_content"`
		Params         json.RawMessage `json:"params"`
		Expected       struct {
			Kind           string `json:"kind"`
			Content        string `json:"content"`
			Recoverable    *bool  `json:"recoverable"`
			MessageContains string `json:"message_contains"`
		} `json:"expected"`
	}
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	if len(scenarios) == 0 {
		t.Fatal("expected >= 1 scenario")
	}

	sb := sporecore.AllowAllSandbox{}
	ctx := context.Background()

	for _, sc := range scenarios {
		sc := sc
		t.Run(sc.Name, func(t *testing.T) {
			// Write initial_content to a temp file and substitute <FIXTURE_PATH>.
			dir := t.TempDir()
			fixFile := filepath.Join(dir, "fixture.txt")
			if err := os.WriteFile(fixFile, []byte(sc.InitialContent), 0o644); err != nil {
				t.Fatalf("write fixture file: %v", err)
			}
			// Replace the <FIXTURE_PATH> placeholder in the raw params JSON.
			rawParams := strings.ReplaceAll(string(sc.Params), "<FIXTURE_PATH>", fixFile)

			callInput := sporecore.ToolCall{
				ID:    "fx",
				Name:  ReadFileToolName,
				Input: json.RawMessage(rawParams),
			}
			r := NewReadFileTool().Execute(ctx, callInput, sb, nil)

			switch sc.Expected.Kind {
			case "success":
				if r.Kind != sporecore.ToolOutputSuccess {
					t.Fatalf("expected success, got kind=%v message=%q", r.Kind, r.Message)
				}
				if r.Content != sc.Expected.Content {
					t.Fatalf("content mismatch:\ngot  %q\nwant %q", r.Content, sc.Expected.Content)
				}
			case "error":
				if r.Kind != sporecore.ToolOutputError {
					t.Fatalf("expected error, got kind=%v content=%q", r.Kind, r.Content)
				}
				if sc.Expected.Recoverable != nil && r.Recoverable != *sc.Expected.Recoverable {
					t.Fatalf("recoverable=%v, expected %v", r.Recoverable, *sc.Expected.Recoverable)
				}
				if sc.Expected.MessageContains != "" && !strings.Contains(r.Message, sc.Expected.MessageContains) {
					t.Fatalf("message %q does not contain %q", r.Message, sc.Expected.MessageContains)
				}
			default:
				t.Fatalf("unknown expected.kind %q", sc.Expected.Kind)
			}
		})
	}
}
