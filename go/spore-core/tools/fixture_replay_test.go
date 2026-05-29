// Fixture-replay tests for issue #5.
//
// Loads shared fixtures from /fixtures/tools and asserts the Go
// implementation produces the same outcome categories the Rust, TypeScript,
// and Python implementations do.

package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// fixturesDir locates /fixtures/tools relative to this test file.
func fixturesDir(t *testing.T) string {
	t.Helper()
	_, here, _, _ := runtime.Caller(0)
	// here = .../spore-core/go/spore-core/tools/fixture_replay_test.go
	// go up: tools → spore-core → go → repo root → fixtures/tools
	root := filepath.Join(filepath.Dir(here), "..", "..", "..", "fixtures", "tools")
	return root
}

// paramParseOK mirrors the Rust helper of the same name: returns true if
// parameter validation passes, false if InvalidParameters would be raised.
func paramParseOK(tool string, input json.RawMessage) bool {
	tryParse := func(v any) bool {
		if len(input) == 0 {
			return false
		}
		return json.Unmarshal(input, v) == nil
	}
	requireFields := func(v map[string]any, fields ...string) bool {
		var probe map[string]any
		if err := json.Unmarshal(input, &probe); err != nil {
			return false
		}
		for _, f := range fields {
			if _, ok := probe[f]; !ok {
				return false
			}
		}
		return true
	}
	_ = tryParse
	switch tool {
	case "read_file":
		return requireFields(nil, "path")
	case "write_file":
		return requireFields(nil, "path", "content")
	case "list_dir":
		return requireFields(nil, "path")
	case "delete_file":
		return requireFields(nil, "path")
	case "move_file":
		return requireFields(nil, "src", "dst")
	case "grep_files":
		if !requireFields(nil, "pattern", "path") {
			return false
		}
		var p GrepFilesParams
		if json.Unmarshal(input, &p) != nil {
			return false
		}
		_, err := regexp.Compile(p.Pattern)
		return err == nil
	case "find_files":
		return requireFields(nil, "glob", "path")
	case "git_status":
		return true
	case "git_log":
		var p GitLogParams
		return json.Unmarshal(input, &p) == nil
	case "git_diff":
		var p GitDiffParams
		return json.Unmarshal(input, &p) == nil
	case "git_commit":
		return requireFields(nil, "message")
	case "git_reset":
		return requireFields(nil, "target", "mode")
	case "http_get":
		return requireFields(nil, "url")
	case "http_post":
		return requireFields(nil, "url", "body")
	default:
		return true
	}
}

func TestFixtureReplayParamValidation(t *testing.T) {
	path := filepath.Join(fixturesDir(t), "param_validation.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not found: %v", err)
	}
	var scenarios []struct {
		Tool     string          `json:"tool"`
		Input    json.RawMessage `json:"input"`
		Expected string          `json:"expected"`
	}
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(scenarios) == 0 {
		t.Fatal("expected >= 1 scenario")
	}
	for _, sc := range scenarios {
		actual := "invalid_parameters"
		if paramParseOK(sc.Tool, sc.Input) {
			actual = "ok"
		}
		if actual != sc.Expected {
			t.Errorf("scenario tool=%s got %s expected %s", sc.Tool, actual, sc.Expected)
		}
	}
}

func TestFixtureReplayOutputTruncation(t *testing.T) {
	path := filepath.Join(fixturesDir(t), "output_truncation.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not found: %v", err)
	}
	var scenarios []struct {
		ContentLength    int    `json:"content_length"`
		HeadTokens       uint32 `json:"head_tokens"`
		TailTokens       uint32 `json:"tail_tokens"`
		ExpectsTruncated bool   `json:"expects_truncated"`
	}
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	sb := sporecore.DefaultSandbox{}
	for _, sc := range scenarios {
		content := strings.Repeat("x", sc.ContentLength)
		out := sb.HandleLargeOutput(context.Background(), content, "fx", sc.HeadTokens, sc.TailTokens)
		actuallyTruncated := out.Truncated
		if actuallyTruncated != sc.ExpectsTruncated {
			t.Errorf("content_length=%d head=%d tail=%d: truncated=%v, expected=%v",
				sc.ContentLength, sc.HeadTokens, sc.TailTokens, actuallyTruncated, sc.ExpectsTruncated)
		}
	}
}

func TestFixtureReplaySubagentScenarios(t *testing.T) {
	path := filepath.Join(fixturesDir(t), "subagent_scenarios.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture not found: %v", err)
	}
	var scenarios []struct {
		Name           string `json:"name"`
		ChildRunResult struct {
			Kind   string `json:"kind"`
			Output string `json:"output"`
		} `json:"child_run_result"`
		ParentCallID string `json:"parent_call_id"`
		Expected     struct {
			Kind             string `json:"kind"`
			Content          string `json:"content"`
			Recoverable      *bool  `json:"recoverable"`
			ParentToolCallID string `json:"parent_tool_call_id"`
		} `json:"expected"`
	}
	if err := json.Unmarshal(data, &scenarios); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, sc := range scenarios {
		var runResult sporecore.RunResult
		switch sc.ChildRunResult.Kind {
		case "success":
			runResult = sporecore.RunResult{Kind: sporecore.RunSuccess, Output: sc.ChildRunResult.Output}
		case "failure":
			runResult = sporecore.RunResult{Kind: sporecore.RunFailure, Reason: sporecore.HaltReason{Kind: sporecore.HaltHumanHalted}}
		case "waiting_for_human":
			paused := &sporecore.PausedState{
				SessionID:    "s",
				Task:         sporecore.NewTask("x", "s", sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 1}),
				HumanRequest: sporecore.HumanRequest{Kind: sporecore.HumanReqClarification, Question: "?"},
			}
			req := &sporecore.HumanRequest{Kind: sporecore.HumanReqClarification, Question: "?"}
			runResult = sporecore.RunResult{Kind: sporecore.RunWaitingForHuman, State: paused, Request: req}
		default:
			t.Fatalf("unknown child_run_result.kind %q", sc.ChildRunResult.Kind)
		}
		h := &scriptedHarness{results: []sporecore.RunResult{runResult}}
		child := sporecore.NewStandardToolRegistry()
		sub, err := NewSubagentTool("subagent", "child", json.RawMessage(`{"type":"object"}`),
			0, Isolated{}, h, child)
		if err != nil {
			t.Fatalf("[%s] new: %v", sc.Name, err)
		}
		callIn := sporecore.ToolCall{ID: sc.ParentCallID, Name: "subagent", Input: json.RawMessage(`{"instruction":"x"}`)}
		out := sub.Execute(context.Background(), callIn, sporecore.AllowAllSandbox{}, nil)

		switch sc.Expected.Kind {
		case "success":
			if out.Kind != sporecore.ToolOutputSuccess {
				t.Errorf("[%s] expected success, got %+v", sc.Name, out)
			}
			if sc.Expected.Content != "" && out.Content != sc.Expected.Content {
				t.Errorf("[%s] content mismatch: %q vs %q", sc.Name, out.Content, sc.Expected.Content)
			}
		case "error":
			if out.Kind != sporecore.ToolOutputError {
				t.Errorf("[%s] expected error, got %+v", sc.Name, out)
			}
			if sc.Expected.Recoverable != nil && out.Recoverable != *sc.Expected.Recoverable {
				t.Errorf("[%s] recoverable=%v, expected %v", sc.Name, out.Recoverable, *sc.Expected.Recoverable)
			}
		case "waiting_for_human":
			if out.Kind != sporecore.ToolOutputWaitingForHuman {
				t.Errorf("[%s] expected waiting_for_human, got %+v", sc.Name, out)
				continue
			}
			if out.ChildState == nil || out.ChildState.ParentToolCallID != sc.Expected.ParentToolCallID {
				t.Errorf("[%s] parent_tool_call_id mismatch: %+v vs %s",
					sc.Name, out.ChildState, sc.Expected.ParentToolCallID)
			}
		default:
			t.Fatalf("[%s] unknown expected.kind %q", sc.Name, sc.Expected.Kind)
		}
	}
}
