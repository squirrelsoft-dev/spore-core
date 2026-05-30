// completion_check.go — issue #43 standard CompletionCheck implementations.
//
// Mirrors the Rust reference at rust/crates/spore-core/src/termination.rs:
//   - AlwaysComplete         (spec-named alias for NullCompletionCheck)
//   - FeatureListCheck       (reads <workspace_root>/<path>)
//   - TestSuiteCheck         (runs a command via an injected SandboxProvider)
//   - QuestionAnsweredCheck  (LLM-as-judge over the last assistant text)
//   - SqlResultCheck         (scans messages for the last SQL tool result)
//
// CompletionCheck wiring into the harness Ralph loop is deferred to a
// follow-up issue (see the Rust commit for context).
package termination

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// AlwaysComplete — spec alias for NullCompletionCheck.
// ============================================================================

// AlwaysComplete is the spec-named alias from issue #43. It always reports
// complete — use it when the model's self-assessment is sufficient.
type AlwaysComplete = NullCompletionCheck

// ============================================================================
// FeatureListCheck
// ============================================================================

// FeatureListCheck reads <workspace_root>/<Path> and reports incomplete
// features. The file is a JSON array of {"name": string, "passes": bool}.
//
// Returns ("", true, nil) when every entry has passes=true. Returns a
// reason and (..., false, nil) when entries are incomplete, when the file
// is missing, or when the file is unreadable JSON. The check never returns
// a non-nil error: read/parse failures are surfaced as incomplete reasons
// so the agent can learn to create / repair the file.
type FeatureListCheck struct {
	// Path is the workspace-relative path to feature_list.json. Defaults
	// to ".spore/feature_list.json" (issue #58, B2 — agrees with the Ralph
	// loop strategy's canonical path). Absolute paths are honoured verbatim.
	Path string
}

// NewFeatureListCheck returns a FeatureListCheck targeting the default
// ".spore/feature_list.json" path (issue #58, B2).
func NewFeatureListCheck() *FeatureListCheck {
	return &FeatureListCheck{Path: ".spore/feature_list.json"}
}

// NewFeatureListCheckWithPath returns a FeatureListCheck targeting a
// custom path.
func NewFeatureListCheckWithPath(path string) *FeatureListCheck {
	return &FeatureListCheck{Path: path}
}

type featureEntry struct {
	Name   string `json:"name"`
	Passes bool   `json:"passes"`
}

// Check implements CompletionCheck.
func (f *FeatureListCheck) Check(_ context.Context, state *SessionStateSnapshot) (string, bool, error) {
	full := f.Path
	if !filepath.IsAbs(full) {
		full = filepath.Join(state.WorkspaceRoot, f.Path)
	}
	raw, err := os.ReadFile(full)
	if err != nil {
		return fmt.Sprintf("%s missing", f.Path), false, nil
	}
	var entries []featureEntry
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Sprintf("%s invalid JSON: %v", f.Path, err), false, nil
	}
	var incomplete []string
	for _, e := range entries {
		if !e.Passes {
			incomplete = append(incomplete, e.Name)
		}
	}
	if len(incomplete) == 0 {
		return "", true, nil
	}
	return fmt.Sprintf("incomplete features: %s", strings.Join(incomplete, ", ")), false, nil
}

// Description implements CompletionCheck.
func (f *FeatureListCheck) Description() string {
	return fmt.Sprintf("feature list at %s", f.Path)
}

// Compile-time interface check.
var _ CompletionCheck = (*FeatureListCheck)(nil)

// ============================================================================
// TestSuiteCheck
// ============================================================================

// TestSuiteCheck runs an external test command via the injected
// SandboxProvider. Returns complete=true when exit code is 0 and the
// command did not time out, otherwise an incomplete reason containing the
// tail of stderr (falling back to stdout). The first whitespace-separated
// token of Command is the program; remaining tokens become args. Callers
// needing shell quoting / pipes should invoke a wrapper script.
type TestSuiteCheck struct {
	Command    string
	WorkingDir string
	Timeout    time.Duration
	Sandbox    sporecore.SandboxProvider
}

// NewTestSuiteCheck constructs a TestSuiteCheck.
func NewTestSuiteCheck(command, workingDir string, timeout time.Duration, sandbox sporecore.SandboxProvider) *TestSuiteCheck {
	return &TestSuiteCheck{Command: command, WorkingDir: workingDir, Timeout: timeout, Sandbox: sandbox}
}

// Check implements CompletionCheck.
func (t *TestSuiteCheck) Check(ctx context.Context, _ *SessionStateSnapshot) (string, bool, error) {
	parts := strings.Fields(t.Command)
	if len(parts) == 0 {
		return "empty test command", false, nil
	}
	program := parts[0]
	args := parts[1:]
	out, viol := t.Sandbox.ExecuteCommand(ctx, program, args, t.WorkingDir, t.Timeout)
	if viol != nil {
		return fmt.Sprintf("sandbox refused test command: %v", viol), false, nil
	}
	if out.ExitCode == 0 && !out.TimedOut {
		return "", true, nil
	}
	tail := tailLines(out.Stderr, 20)
	if strings.TrimSpace(tail) == "" {
		tail = tailLines(out.Stdout, 20)
	}
	return fmt.Sprintf(
		"test suite failed (exit %d, timed_out=%t):\n%s",
		out.ExitCode, out.TimedOut, tail,
	), false, nil
}

// Description implements CompletionCheck.
func (t *TestSuiteCheck) Description() string {
	return fmt.Sprintf("test suite: `%s` in %s", t.Command, t.WorkingDir)
}

func tailLines(s string, n int) string {
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	start := len(lines) - n
	if start < 0 {
		start = 0
	}
	return strings.Join(lines[start:], "\n")
}

// Compile-time interface check.
var _ CompletionCheck = (*TestSuiteCheck)(nil)

// ============================================================================
// QuestionAnsweredCheck
// ============================================================================

// QuestionAnsweredCheck asks an injected judge model whether the agent's
// final response actually answered the original question. The Go
// implementation takes a ModelInterface directly (the spec lists
// `judge_model: ModelConfig`, but ModelConfig is a placeholder today —
// matches the Rust generic-over-ModelInterface decision).
type QuestionAnsweredCheck struct {
	Judge            sporecore.ModelInterface
	OriginalQuestion string
	// Rubric is an optional set of evaluation criteria appended to the
	// judge prompt. Empty string disables the clause.
	Rubric string
}

// NewQuestionAnsweredCheck constructs a QuestionAnsweredCheck with no rubric.
func NewQuestionAnsweredCheck(judge sporecore.ModelInterface, originalQuestion string) *QuestionAnsweredCheck {
	return &QuestionAnsweredCheck{Judge: judge, OriginalQuestion: originalQuestion}
}

// WithRubric returns the check with the given rubric set.
func (q *QuestionAnsweredCheck) WithRubric(rubric string) *QuestionAnsweredCheck {
	q.Rubric = rubric
	return q
}

// Check implements CompletionCheck.
func (q *QuestionAnsweredCheck) Check(ctx context.Context, state *SessionStateSnapshot) (string, bool, error) {
	agentResponse := lastAssistantText(state.State.Messages)
	if agentResponse == "" {
		agentResponse = "<no agent response>"
	}
	rubricClause := ""
	if q.Rubric != "" {
		rubricClause = "\n\nRubric:\n" + q.Rubric
	}
	userText := fmt.Sprintf(
		"Question:\n%s\n\nAgent's final response:\n%s\n\nDid the agent's response "+
			"answer the question? Reply with the first line `ANSWERED: YES` or "+
			"`ANSWERED: NO`, then a brief reason on the next line.%s",
		q.OriginalQuestion, agentResponse, rubricClause,
	)
	req := sporecore.ModelRequest{
		Messages: []sporecore.Message{
			{
				Role: sporecore.RoleSystem,
				Content: sporecore.NewTextContent(
					"You are an evaluation judge. Reply with `ANSWERED: YES` or " +
						"`ANSWERED: NO` on the first line, no other prefix.",
				),
			},
			{
				Role:    sporecore.RoleUser,
				Content: sporecore.NewTextContent(userText),
			},
		},
		Tools:  []sporecore.ToolSchema{},
		Params: sporecore.ModelParams{},
		Stream: false,
	}
	resp, err := q.Judge.Call(ctx, req)
	if err != nil {
		return fmt.Sprintf("judge model error: %v", err), false, nil
	}
	var verdict string
	for _, b := range resp.Content {
		if b.Type == sporecore.ContentBlockTypeText {
			verdict = b.Text
			break
		}
	}
	firstLine := verdict
	if idx := strings.IndexByte(verdict, '\n'); idx >= 0 {
		firstLine = verdict[:idx]
	}
	first := strings.ToUpper(strings.TrimSpace(firstLine))
	if strings.HasPrefix(first, "ANSWERED: YES") {
		return "", true, nil
	}
	return fmt.Sprintf("judge says not answered: %s", verdict), false, nil
}

// Description implements CompletionCheck.
func (q *QuestionAnsweredCheck) Description() string {
	return fmt.Sprintf("LLM-judge: did the response answer `%s`", q.OriginalQuestion)
}

func lastAssistantText(messages []sporecore.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role == sporecore.RoleAssistant && m.Content.Type == sporecore.ContentTypeText {
			return m.Content.Text
		}
	}
	return ""
}

// Compile-time interface check.
var _ CompletionCheck = (*QuestionAnsweredCheck)(nil)

// ============================================================================
// SqlResultCheck
// ============================================================================

// SqlResultCheck validates the most recent SQL tool result in the session.
// It scans state.State.Messages in reverse for the last tool_result
// Content whose originating tool_call Content matches SQLToolName, then
// parses the result content as {"columns":[string], "rows":[[any]]}.
//
// Returns complete=true when constraints are satisfied; otherwise an
// incomplete reason describing the violation. The default minimum row
// count is 1 (so a non-empty result set passes when no expectations are
// configured).
type SqlResultCheck struct {
	// SQLToolName is the name of the tool whose results we validate.
	// Defaults to "execute_sql".
	SQLToolName string
	// ExpectedColumns, when non-nil, must match the result columns
	// element-for-element.
	ExpectedColumns []string
	// MinRows, when non-nil, requires the result set to contain at least
	// that many rows. Nil means default 1.
	MinRows *int
}

// NewSqlResultCheck returns a SqlResultCheck with the default tool name
// "execute_sql".
func NewSqlResultCheck() *SqlResultCheck {
	return &SqlResultCheck{SQLToolName: "execute_sql"}
}

// WithToolName sets a custom SQL tool name.
func (s *SqlResultCheck) WithToolName(name string) *SqlResultCheck {
	s.SQLToolName = name
	return s
}

// WithExpectedColumns sets the expected column list.
func (s *SqlResultCheck) WithExpectedColumns(cols []string) *SqlResultCheck {
	s.ExpectedColumns = cols
	return s
}

// WithMinRows sets the minimum required row count.
func (s *SqlResultCheck) WithMinRows(n int) *SqlResultCheck {
	s.MinRows = &n
	return s
}

type sqlResultPayload struct {
	Columns []string          `json:"columns"`
	Rows    []json.RawMessage `json:"rows"`
}

// Check implements CompletionCheck.
func (s *SqlResultCheck) Check(_ context.Context, state *SessionStateSnapshot) (string, bool, error) {
	// Build id -> tool_name map from ToolCalls.
	idToName := map[string]string{}
	for _, m := range state.State.Messages {
		if m.Content.Type == sporecore.ContentTypeToolCall && m.Content.ToolCall != nil {
			idToName[m.Content.ToolCall.ID] = m.Content.ToolCall.Name
		}
	}
	// Scan in reverse for the most recent matching tool_result.
	var raw string
	found := false
	for i := len(state.State.Messages) - 1; i >= 0; i-- {
		m := state.State.Messages[i]
		if m.Content.Type != sporecore.ContentTypeToolResult || m.Content.ToolResult == nil {
			continue
		}
		if name, ok := idToName[m.Content.ToolResult.ToolUseID]; ok && name == s.SQLToolName {
			raw = m.Content.ToolResult.Content
			found = true
			break
		}
	}
	if !found {
		return fmt.Sprintf("no `%s` tool result found in session", s.SQLToolName), false, nil
	}
	var payload sqlResultPayload
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return fmt.Sprintf("sql result is not JSON: %v", err), false, nil
	}
	if s.ExpectedColumns != nil {
		if !stringSlicesEqual(payload.Columns, s.ExpectedColumns) {
			return fmt.Sprintf(
				"sql columns mismatch: expected %v, got %v",
				s.ExpectedColumns, payload.Columns,
			), false, nil
		}
	}
	min := 1
	if s.MinRows != nil {
		min = *s.MinRows
	}
	if len(payload.Rows) < min {
		return fmt.Sprintf(
			"sql result has %d rows, expected at least %d",
			len(payload.Rows), min,
		), false, nil
	}
	return "", true, nil
}

// Description implements CompletionCheck.
func (s *SqlResultCheck) Description() string {
	return fmt.Sprintf("sql result check on tool `%s`", s.SQLToolName)
}

func stringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Compile-time interface check.
var _ CompletionCheck = (*SqlResultCheck)(nil)
