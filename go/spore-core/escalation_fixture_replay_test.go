package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Tool Escalation Protocol fixture replays (issue #80). These consume the
// SHARED, cross-language fixtures created by the Rust reference agent and must
// produce the same outcomes as rust/crates/spore-core/tests/
// escalation_loop_fixture_replay.rs and escalation_signals_fixture_replay.rs.
// Never edit the fixtures to make a failing implementation pass.

func fixtureRoot(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures")
}

// Loop-replay: a one-turn model trace requests a single tool call; the scripted
// registry returns ToolOutput.Escalate{Abort}. The harness must return
// RunResult.Escalate (not Success/Failure), skip the history append, and carry
// the Abort signal. Mirrors escalation_loop_fixture_replay.rs.
func TestEscalationLoopReturnsEscalateAndSkipsHistoryAppend(t *testing.T) {
	path := filepath.Join(fixtureRoot(t), "model_responses", "harness", "escalation_loop.jsonl")
	raw, err := os.ReadFile(path)
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
	reg.Push(ToolOutput{
		Kind:   ToolOutputEscalate,
		Signal: &HarnessSignal{Kind: SignalAbort, Reason: "blocked on missing credentials"},
	})

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      reg,
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	h := NewStandardHarness(cfg)

	task := NewTask(
		"investigate then decide whether to abort",
		SessionID("escalation-loop-session"),
		ReActStrategy(5),
	)

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunEscalate {
		t.Fatalf("expected Escalate, got %q (%+v)", r.Kind, r)
	}
	if r.Signal == nil || r.Signal.Kind != SignalAbort || r.Signal.Reason != "blocked on missing credentials" {
		t.Fatalf("carried signal = %+v", r.Signal)
	}
	if r.SessionID != "escalation-loop-session" {
		t.Fatalf("session id = %q", r.SessionID)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1 (one turn consumed before the escalating dispatch)", r.Turns)
	}
	if r.State == nil {
		t.Fatal("state must be preserved")
	}
	// The escalation is never appended as a tool-result turn.
	var toolResults int
	for _, m := range r.State.SessionState.Messages {
		if m.Role == RoleTool {
			toolResults++
		}
	}
	if toolResults != 0 {
		t.Fatalf("escalation must not append a tool result; got %d", toolResults)
	}
	// Single-call batch: no remaining pending calls.
	if len(r.State.PendingToolCalls) != 0 {
		t.Fatalf("pending tool calls = %d, want 0", len(r.State.PendingToolCalls))
	}
	// The dispatch ran exactly once.
	if reg.CallCount.Load() != 1 {
		t.Fatalf("tool dispatch count = %d, want 1", reg.CallCount.Load())
	}
}

// escalationSignalsFixture is the shape of escalation_signals.json.
type escalationSignalsFixture struct {
	RunResultCases  []json.RawMessage `json:"run_result_cases"`
	ToolOutputCases []json.RawMessage `json:"tool_output_cases"`
}

// Serde round-trip: each fixture case deserializes into the Go type and
// re-serializes to a STRUCTURALLY IDENTICAL JSON value, locking the
// byte-identical wire guarantee across the four languages. Mirrors
// escalation_signals_fixture_replay.rs.
func TestEscalationSignalsFixtureRoundTrip(t *testing.T) {
	path := filepath.Join(fixtureRoot(t), "harness", "escalation_signals.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx escalationSignalsFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}

	// tool_output_cases: each is a ToolOutput.Escalate wrapping one of the four
	// HarnessSignal variants.
	if len(fx.ToolOutputCases) != 4 {
		t.Fatalf("tool_output_cases = %d, want 4 (all HarnessSignal variants)", len(fx.ToolOutputCases))
	}
	seen := map[HarnessSignalKind]bool{}
	for _, c := range fx.ToolOutputCases {
		var out ToolOutput
		if err := json.Unmarshal(c, &out); err != nil {
			t.Fatalf("deserialize ToolOutput.Escalate: %v\ncase: %s", err, c)
		}
		if out.Kind != ToolOutputEscalate || out.Signal == nil {
			t.Fatalf("expected ToolOutput.Escalate with signal, got %+v", out)
		}
		seen[out.Signal.Kind] = true
		reser, err := json.Marshal(out)
		if err != nil {
			t.Fatalf("re-serialize: %v", err)
		}
		if !jsonEqual(t, reser, c) {
			t.Fatalf("structural round-trip mismatch:\n got: %s\nwant: %s", reser, c)
		}
	}
	for _, k := range []HarnessSignalKind{SignalEnterPlanMode, SignalExitPlanMode, SignalSwitchMode, SignalAbort} {
		if !seen[k] {
			t.Fatalf("HarnessSignal variant %q not covered by fixture", k)
		}
	}

	// run_result_cases: RunResult.Escalate carrying a full PausedState, for at
	// least ExitPlanMode and Abort.
	if len(fx.RunResultCases) < 2 {
		t.Fatalf("run_result_cases = %d, want >= 2", len(fx.RunResultCases))
	}
	signalKinds := map[HarnessSignalKind]bool{}
	for _, c := range fx.RunResultCases {
		// Escalation-derived state carries no human request.
		var probe struct {
			Kind  RunResultKind `json:"kind"`
			State struct {
				HumanRequest json.RawMessage `json:"human_request"`
			} `json:"state"`
		}
		_ = json.Unmarshal(c, &probe)
		if probe.Kind != RunEscalate {
			t.Fatalf("outer RunResult tag = %q, want escalate", probe.Kind)
		}
		if string(probe.State.HumanRequest) != "null" {
			t.Fatalf("escalation PausedState.human_request must be null, got %q", probe.State.HumanRequest)
		}

		var rr RunResult
		if err := json.Unmarshal(c, &rr); err != nil {
			t.Fatalf("deserialize RunResult.Escalate: %v\ncase: %s", err, c)
		}
		if rr.Signal == nil || rr.State == nil {
			t.Fatalf("RunResult.Escalate missing signal/state: %+v", rr)
		}
		signalKinds[rr.Signal.Kind] = true
		// The five fields are present and consistent with the state.
		if rr.State.HumanRequest != nil {
			t.Fatalf("decoded escalation state must have nil human request")
		}
		if rr.Turns != rr.State.BudgetUsed.Turns {
			t.Fatalf("turns %d != state.budget_used.turns %d", rr.Turns, rr.State.BudgetUsed.Turns)
		}
		if uint64(rr.Usage.InputTokens) != rr.State.BudgetUsed.InputTokens {
			t.Fatalf("usage.input_tokens %d != state.budget_used.input_tokens %d", rr.Usage.InputTokens, rr.State.BudgetUsed.InputTokens)
		}
		reser, err := json.Marshal(rr)
		if err != nil {
			t.Fatalf("re-serialize: %v", err)
		}
		if !jsonEqual(t, reser, c) {
			t.Fatalf("structural round-trip mismatch:\n got: %s\nwant: %s", reser, c)
		}
	}
	if !signalKinds[SignalExitPlanMode] || !signalKinds[SignalAbort] {
		t.Fatalf("run_result_cases must cover exit_plan_mode and abort; got %+v", signalKinds)
	}
}
