package termination

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// AlwaysComplete (alias for NullCompletionCheck)
// ============================================================================

func TestAlwaysCompleteIsNullCheck(t *testing.T) {
	var a AlwaysComplete
	reason, complete, err := a.Check(context.Background(), nil)
	if err != nil || !complete || reason != "" {
		t.Fatalf("AlwaysComplete should be complete with no reason; got (%q, %v, %v)", reason, complete, err)
	}
}

// ============================================================================
// FeatureListCheck
// ============================================================================

func snapshotIn(dir string) SessionStateSnapshot {
	return NewSessionStateSnapshotWithRoot(
		sporecore.SessionID("s1"),
		sporecore.TaskID("t1"),
		sporecore.SessionState{},
		dir,
	)
}

func writeFile(t *testing.T, dir, name, body string) {
	t.Helper()
	full := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestFeatureListCheckAllPassReturnsComplete(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, ".spore/feature_list.json", `[{"name":"a","passes":true},{"name":"b","passes":true}]`)
	snap := snapshotIn(dir)
	reason, complete, err := NewFeatureListCheck().Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v)", reason, complete)
	}
}

func TestFeatureListCheckSomeFailReturnsReason(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, ".spore/feature_list.json",
		`[{"name":"a","passes":true},{"name":"b","passes":false},{"name":"c","passes":false}]`)
	snap := snapshotIn(dir)
	reason, complete, err := NewFeatureListCheck().Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete {
		t.Fatalf("expected incomplete")
	}
	if !strings.Contains(reason, "b") || !strings.Contains(reason, "c") {
		t.Fatalf("reason should list incomplete features; got %q", reason)
	}
	if strings.Contains(reason, "a, ") {
		t.Fatalf("reason should not mention passing feature 'a'; got %q", reason)
	}
}

func TestFeatureListCheckMissingFileReturnsReason(t *testing.T) {
	dir := t.TempDir()
	snap := snapshotIn(dir)
	reason, complete, err := NewFeatureListCheck().Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "missing") {
		t.Fatalf("expected missing-file reason; got (%q, %v)", reason, complete)
	}
}

func TestFeatureListCheckInvalidJSONReturnsReason(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, ".spore/feature_list.json", `{not json`)
	snap := snapshotIn(dir)
	reason, complete, err := NewFeatureListCheck().Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "invalid JSON") {
		t.Fatalf("expected invalid JSON reason; got (%q, %v)", reason, complete)
	}
}

func TestFeatureListCheckCustomPath(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "custom.json", `[{"name":"x","passes":true}]`)
	snap := snapshotIn(dir)
	reason, complete, err := NewFeatureListCheckWithPath("custom.json").Check(context.Background(), &snap)
	if err != nil || !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v, %v)", reason, complete, err)
	}
}

// ============================================================================
// TestSuiteCheck
// ============================================================================

// stubSandbox implements sporecore.SandboxProvider for tests. It only
// implements ExecuteCommand non-trivially; the other methods return zero
// values / nil.
type stubSandbox struct {
	out  sporecore.CommandOutput
	root string
}

func (s *stubSandbox) Validate(_ context.Context, _ sporecore.ToolCall) *sporecore.SandboxViolation {
	return nil
}

func (s *stubSandbox) ExecuteCommand(_ context.Context, _ string, _ []string, _ string, _ time.Duration) (sporecore.CommandOutput, *sporecore.SandboxViolation) {
	return s.out, nil
}

func (s *stubSandbox) HandleLargeOutput(_ context.Context, content string, _ string, _ uint32, _ uint32) sporecore.TruncatedOutput {
	return sporecore.TruncatedOutput{Content: content, Truncated: false, OriginalSize: uint64(len(content))}
}

func (s *stubSandbox) ResolvePath(_ context.Context, path string, _ sporecore.Operation) (string, *sporecore.SandboxViolation) {
	return path, nil
}

func (s *stubSandbox) IsolationMode() sporecore.IsolationMode { return sporecore.IsolationNone{} }

func (s *stubSandbox) WorkspaceRoot() string { return s.root }

func newStubSandbox(exit int, stderr string) *stubSandbox {
	return &stubSandbox{
		out: sporecore.CommandOutput{Stderr: stderr, ExitCode: exit},
	}
}

func emptySnap() SessionStateSnapshot {
	return NewSessionStateSnapshot(
		sporecore.SessionID("s1"),
		sporecore.TaskID("t1"),
		sporecore.SessionState{},
	)
}

func TestTestSuiteCheckPassReturnsComplete(t *testing.T) {
	check := NewTestSuiteCheck("cargo test", ".", 30*time.Second, newStubSandbox(0, ""))
	snap := emptySnap()
	reason, complete, err := check.Check(context.Background(), &snap)
	if err != nil || !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v, %v)", reason, complete, err)
	}
}

func TestTestSuiteCheckFailIncludesStderrTail(t *testing.T) {
	check := NewTestSuiteCheck(
		"cargo test", ".", 30*time.Second,
		newStubSandbox(1, "test foo ... FAILED\nassertion failed"),
	)
	snap := emptySnap()
	reason, complete, err := check.Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete {
		t.Fatalf("expected incomplete")
	}
	if !strings.Contains(reason, "FAILED") {
		t.Fatalf("reason should contain stderr tail; got %q", reason)
	}
}

func TestTestSuiteCheckEmptyCommandFails(t *testing.T) {
	check := NewTestSuiteCheck("", ".", 30*time.Second, newStubSandbox(0, ""))
	snap := emptySnap()
	reason, complete, err := check.Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || reason == "" {
		t.Fatalf("expected incomplete with reason; got (%q, %v)", reason, complete)
	}
}

// ============================================================================
// QuestionAnsweredCheck
// ============================================================================

type stubJudge struct {
	verdict string
}

func (s *stubJudge) Call(_ context.Context, _ sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	return sporecore.ModelResponse{
		Content: []sporecore.ContentBlock{
			{Type: sporecore.ContentBlockTypeText, Text: s.verdict},
		},
		StopReason: sporecore.StopEndTurn,
	}, nil
}

func (s *stubJudge) CallStreaming(_ context.Context, _ sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	ch := make(chan sporecore.StreamEventOrErr)
	close(ch)
	return ch, nil
}

func (s *stubJudge) CountTokens(_ context.Context, _ sporecore.ModelRequest) (uint32, error) {
	return 0, nil
}

func (s *stubJudge) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{Name: "stub", ModelID: "stub", ContextWindow: 4096}
}

func snapWithAssistant(text string) SessionStateSnapshot {
	state := sporecore.SessionState{
		Messages: []sporecore.Message{
			{Role: sporecore.RoleAssistant, Content: sporecore.NewTextContent(text)},
		},
	}
	return NewSessionStateSnapshot(sporecore.SessionID("s1"), sporecore.TaskID("t1"), state)
}

func TestQuestionAnsweredYesReturnsComplete(t *testing.T) {
	c := NewQuestionAnsweredCheck(&stubJudge{verdict: "ANSWERED: YES\nLooks good."}, "What is 2+2?")
	snap := snapWithAssistant("It is 4.")
	reason, complete, err := c.Check(context.Background(), &snap)
	if err != nil || !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v, %v)", reason, complete, err)
	}
}

func TestQuestionAnsweredNoReturnsReason(t *testing.T) {
	c := NewQuestionAnsweredCheck(&stubJudge{verdict: "ANSWERED: NO\nMissed the point."}, "What is 2+2?")
	snap := snapWithAssistant("I don't know.")
	reason, complete, err := c.Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "not answered") {
		t.Fatalf("expected not-answered reason; got (%q, %v)", reason, complete)
	}
}

func TestQuestionAnsweredUsesRubric(t *testing.T) {
	c := NewQuestionAnsweredCheck(&stubJudge{verdict: "ANSWERED: YES"}, "q").WithRubric("Be strict about citations.")
	if !strings.Contains(c.Description(), "q") {
		t.Fatalf("description should mention question; got %q", c.Description())
	}
	snap := snapWithAssistant("a")
	reason, complete, err := c.Check(context.Background(), &snap)
	if err != nil || !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v, %v)", reason, complete, err)
	}
}

// ============================================================================
// SqlResultCheck
// ============================================================================

func snapWithSQLResult(toolName, body string) SessionStateSnapshot {
	state := sporecore.SessionState{
		Messages: []sporecore.Message{
			{
				Role: sporecore.RoleAssistant,
				Content: sporecore.NewToolCallContent(sporecore.ToolCall{
					ID:    "call-1",
					Name:  toolName,
					Input: json.RawMessage(`{"q":"select 1"}`),
				}),
			},
			{
				Role: sporecore.RoleTool,
				Content: sporecore.NewToolResultContent(sporecore.ToolResult{
					ToolUseID: "call-1",
					Content:   body,
				}),
			},
		},
	}
	return NewSessionStateSnapshot(sporecore.SessionID("s1"), sporecore.TaskID("t1"), state)
}

func TestSqlResultCheckDefaultPassesWhenRowsPresent(t *testing.T) {
	snap := snapWithSQLResult("execute_sql", `{"columns":["id","name"],"rows":[[1,"a"],[2,"b"]]}`)
	reason, complete, err := NewSqlResultCheck().Check(context.Background(), &snap)
	if err != nil || !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v, %v)", reason, complete, err)
	}
}

func TestSqlResultCheckEmptyRowsFails(t *testing.T) {
	snap := snapWithSQLResult("execute_sql", `{"columns":["id"],"rows":[]}`)
	reason, complete, err := NewSqlResultCheck().Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "0 rows") {
		t.Fatalf("expected 0-rows reason; got (%q, %v)", reason, complete)
	}
}

func TestSqlResultCheckColumnMismatchFails(t *testing.T) {
	snap := snapWithSQLResult("execute_sql", `{"columns":["id"],"rows":[[1]]}`)
	check := NewSqlResultCheck().WithExpectedColumns([]string{"id", "name"})
	reason, complete, err := check.Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "columns mismatch") {
		t.Fatalf("expected columns-mismatch reason; got (%q, %v)", reason, complete)
	}
}

func TestSqlResultCheckMinRowsEnforced(t *testing.T) {
	snap := snapWithSQLResult("execute_sql", `{"columns":["id"],"rows":[[1]]}`)
	check := NewSqlResultCheck().WithMinRows(5)
	reason, complete, err := check.Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "at least 5") {
		t.Fatalf("expected min-rows reason; got (%q, %v)", reason, complete)
	}
}

func TestSqlResultCheckCustomToolName(t *testing.T) {
	snap := snapWithSQLResult("run_query", `{"columns":["id"],"rows":[[1]]}`)
	c := NewSqlResultCheck().WithToolName("run_query")
	reason, complete, err := c.Check(context.Background(), &snap)
	if err != nil || !complete || reason != "" {
		t.Fatalf("expected complete; got (%q, %v, %v)", reason, complete, err)
	}
}

func TestSqlResultCheckNoMatchingToolFails(t *testing.T) {
	snap := snapWithSQLResult("other_tool", `{"columns":["id"],"rows":[[1]]}`)
	reason, complete, err := NewSqlResultCheck().Check(context.Background(), &snap)
	if err != nil {
		t.Fatal(err)
	}
	if complete || !strings.Contains(reason, "no `execute_sql`") {
		t.Fatalf("expected no-matching-tool reason; got (%q, %v)", reason, complete)
	}
}
