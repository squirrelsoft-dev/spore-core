package sporecore

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func harnessFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "model_responses", "harness", "react_loop.jsonl")
}

// Mirrors rust/crates/spore-core/tests/harness_fixture_replay.rs.
// Drives a StandardHarness backed by a ReplayModel-fed ModelAgent against
// the shared JSONL fixture, asserting the same outcome as Rust:
//
//	Success { output: "127.0.0.1 localhost", turns: 2, input_tokens: 30, output_tokens: 14 }
//	tool registry dispatched exactly once.
func TestHarnessReActLoopDispatchesToolThenCompletes(t *testing.T) {
	raw, err := os.ReadFile(harnessFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	agent := NewModelAgent(AgentID("fixture-agent"), replay)

	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "127.0.0.1 localhost"})

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      reg,
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	h := NewStandardHarness(cfg)

	task := NewTask(
		"read /etc/hosts then summarize",
		SessionID("fixture-session"),
		ReActStrategy(5),
	)

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "127.0.0.1 localhost" {
		t.Fatalf("output = %q", r.Output)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2", r.Turns)
	}
	if r.Usage.InputTokens != 30 {
		t.Fatalf("input tokens = %d, want 30", r.Usage.InputTokens)
	}
	if r.Usage.OutputTokens != 14 {
		t.Fatalf("output tokens = %d, want 14", r.Usage.OutputTokens)
	}
	if reg.CallCount.Load() != 1 {
		t.Fatalf("tool registry call count = %d, want 1", reg.CallCount.Load())
	}
}
