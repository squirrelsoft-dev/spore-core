package observability

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// Issue #12 — the harness emits real spans through an injected
// ObservabilityProvider (via the HarnessObserver adapter) and flushes a
// terminal session summary. Hermetic: SPORE_OTLP_ENDPOINT is left unset so
// the outbox writes JSONL only (Loki, not Tempo). Mirrors the Rust test
// harness_emits_spans_through_outbox_jsonl.
func TestHarnessEmitsSpansThroughOutboxJSONL(t *testing.T) {
	os.Unsetenv("SPORE_OTLP_ENDPOINT")
	tmp := t.TempDir()

	a := sporecore.NewMockAgent("test")
	a.Push(sporecore.NewToolCallRequested(
		[]sporecore.ToolCall{{ID: "c1", Name: "do_thing", Input: json.RawMessage(`{"k":"v"}`)}},
		sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1},
	))
	a.Push(sporecore.NewFinalResponse("done", sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1}))

	reg := sporecore.NewScriptedToolRegistry()
	reg.Push(sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: "tool-ok"})

	harness := NewHarnessBuilder(
		a,
		reg,
		sporecore.AllowAllSandbox{},
		sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{},
	).WithObservabilityOutbox(tmp).Build()

	task := sporecore.NewTask("do something", sporecore.SessionID("s1"),
		sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 5})

	res := harness.Run(context.Background(), sporecore.NewHarnessRunOptions(task))
	if res.Kind != sporecore.RunSuccess {
		t.Fatalf("expected RunSuccess, got %v", res.Kind)
	}
	if res.Turns != 2 {
		t.Fatalf("expected 2 turns, got %d", res.Turns)
	}

	path := filepath.Join(tmp, "sessions", "s1", "trace.jsonl")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("trace.jsonl not written: %v", err)
	}

	var lines []map[string]any
	for _, raw := range splitLines(string(body)) {
		if raw == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			t.Fatalf("bad jsonl line %q: %v", raw, err)
		}
		lines = append(lines, m)
	}
	if len(lines) == 0 {
		t.Fatal("no trace lines written")
	}

	var kinds []string
	traceIDs := map[string]struct{}{}
	var toolLine map[string]any
	for _, l := range lines {
		k, _ := l["kind"].(string)
		kinds = append(kinds, k)
		if tid, ok := l["trace_id"].(string); ok {
			traceIDs[tid] = struct{}{}
		}
		if k == "tool_call" {
			toolLine = l
		}
	}

	if !contains(kinds, "turn") {
		t.Fatalf("expected a turn span, got %v", kinds)
	}
	if !contains(kinds, "tool_call") {
		t.Fatalf("expected a tool_call span, got %v", kinds)
	}
	// The terminal summary line is written last.
	if kinds[len(kinds)-1] != "session" {
		t.Fatalf("expected last kind to be session, got %v", kinds)
	}

	summary := lines[len(lines)-1]
	attrs, _ := summary["attributes"].(map[string]any)
	if attrs["outcome"] != "success" {
		t.Fatalf("expected outcome success, got %v", attrs["outcome"])
	}
	if tt, ok := attrs["total_turns"].(float64); !ok || tt != 2 {
		t.Fatalf("expected total_turns 2, got %v", attrs["total_turns"])
	}

	if toolLine == nil {
		t.Fatal("no tool_call line found")
	}
	tattrs, _ := toolLine["attributes"].(map[string]any)
	if tattrs["tool_name"] != "do_thing" {
		t.Fatalf("expected tool_name do_thing, got %v", tattrs["tool_name"])
	}
	if tattrs["call_id"] != "c1" {
		t.Fatalf("expected call_id c1, got %v", tattrs["call_id"])
	}

	// All spans share one trace id.
	if len(traceIDs) != 1 {
		t.Fatalf("expected 1 trace id, got %d", len(traceIDs))
	}

	if _, err := os.Stat(filepath.Join(tmp, "sessions", "s1", ".flushed")); err != nil {
		t.Fatalf(".flushed marker missing: %v", err)
	}
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
