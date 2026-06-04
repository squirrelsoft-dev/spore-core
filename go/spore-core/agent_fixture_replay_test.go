package sporecore

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func agentFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	// this = .../go/spore-core/agent_fixture_replay_test.go
	return filepath.Join(dir, "..", "..", "fixtures", "model_responses", "agent", "turn_classification.jsonl")
}

// Mirrors rust/crates/spore-core/tests/agent_fixture_replay.rs.
// Drives a ModelAgent backed by ReplayModel against the shared JSONL
// fixture and asserts every classification matches Rust.
func TestAgentClassifiesRecordedTurnsConsistently(t *testing.T) {
	raw, err := os.ReadFile(agentFixturePath(t))
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
	ctx := context.Background()

	// 1. Plain text response → FinalResponse("hello")
	r1 := agent.Turn(ctx, Context{})
	if r1.Kind != TurnFinalResponse {
		t.Fatalf("turn 1 kind = %q", r1.Kind)
	}
	if r1.Content != "hello" {
		t.Fatalf("turn 1 content = %q", r1.Content)
	}
	if r1.Usage == nil || r1.Usage.InputTokens != 5 || r1.Usage.OutputTokens != 1 {
		t.Fatalf("turn 1 usage = %+v", r1.Usage)
	}

	// 2. Single tool call → ToolCallRequested with one call.
	r2 := agent.Turn(ctx, Context{})
	if r2.Kind != TurnToolCallRequested {
		t.Fatalf("turn 2 kind = %q", r2.Kind)
	}
	if len(r2.Calls) != 1 || r2.Calls[0].Name != "read_file" || r2.Calls[0].ID != "toolu_a" {
		t.Fatalf("turn 2 calls = %+v", r2.Calls)
	}
	if r2.Usage == nil || r2.Usage.InputTokens != 20 {
		t.Fatalf("turn 2 usage = %+v", r2.Usage)
	}

	// 3. Parallel tool calls → ToolCallRequested with two calls.
	r3 := agent.Turn(ctx, Context{})
	if r3.Kind != TurnToolCallRequested {
		t.Fatalf("turn 3 kind = %q", r3.Kind)
	}
	if len(r3.Calls) != 2 || r3.Calls[0].ID != "toolu_b1" || r3.Calls[1].ID != "toolu_b2" {
		t.Fatalf("turn 3 calls = %+v", r3.Calls)
	}

	// 4. Empty content blocks with a truncated stop (MaxTokens) →
	//    AgentError EmptyResponse. (A clean EndTurn empty is instead a
	//    completion; only abnormal/truncated empties remain errors.)
	r4 := agent.Turn(ctx, Context{})
	if r4.Kind != TurnError || r4.Err == nil || r4.Err.Kind != AgentErrEmptyResponse {
		t.Fatalf("turn 4 = %+v / err=%+v", r4, r4.Err)
	}
	if r4.Usage == nil || r4.Usage.InputTokens != 3 || r4.Usage.OutputTokens != 0 {
		t.Fatalf("turn 4 usage = %+v", r4.Usage)
	}
}
