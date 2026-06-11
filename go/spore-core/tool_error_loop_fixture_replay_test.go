package sporecore

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// Loop-replay integration test for the consecutive-recoverable-tool-error
// breaker (issue #137).
//
// Loads fixtures/model_responses/harness/tool_error_loop.jsonl — a recorded
// trace in which the model repeatedly emits the SAME malformed add_task tool
// call (the gemma task_list/add_task-without-description scenario) — and drives a
// StandardHarness with LoopStrategy ReAct. The scripted tool registry returns an
// identical recoverable ToolOutput.Error for every dispatch. With
// ErrorLoopThreshold = 3 (N) and a generous turn budget (50), the ONLY thing that
// can stop the run is the breaker, which hard-stops at 2N (6 identical errors)
// and resolves the leaf's Fail behavior into RunResult.Failure{ToolErrorLoop}
// WITHOUT burning the remaining budget.
//
// Must produce the same outcome in all four language implementations — never edit
// the fixture to make a failing implementation pass (see fixtures/README.md).
func TestToolErrorLoopBreakerHardStopsAtTwoN(t *testing.T) {
	path := filepath.Join(fixtureRoot(t), "model_responses", "harness", "tool_error_loop.jsonl")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "ollama", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	agent := NewModelAgent(AgentID("fixture-agent"), replay)

	// Every dispatch of the malformed add_task call returns the same recoverable
	// error, regardless of args.
	reg := NewScriptedToolRegistry()
	reg.AlwaysRecoverableError("missing required parameter `description`")

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      reg,
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		// N == 3 → inject at 3 identical errors, hard-stop at 6.
		ErrorLoopThreshold: 3,
		// Autonomous so the leaf's Fail behavior produces a terminal Failure (not a
		// HITL pause) at the 2N hard stop.
		EscalationMode: AutonomousEscalation(),
	}
	h := NewStandardHarness(cfg)

	// A bare ReAct leaf with a generous budget (50) and Fail behavior — so the
	// ONLY thing that can stop the run early is the error-loop breaker.
	task := NewTask(
		"add a task to the task list",
		SessionID("tool-error-loop-session"),
		LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
			Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: 50},
			Behavior: BudgetExhaustedBehavior{Kind: BehaviorFail},
			Agent:    AgentRef(""),
			Toolset:  ToolsetRef(""),
		}},
	)

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltToolErrorLoop {
		t.Fatalf("expected Failure{ToolErrorLoop}, got kind=%q reason=%+v", r.Kind, r.Reason)
	}
	if r.Reason.Tool != "add_task" {
		t.Fatalf("tool = %q, want add_task", r.Reason.Tool)
	}
	if r.Reason.ConsecutiveErrors != 6 {
		t.Fatalf("consecutive_errors = %d, want 6 (hard stop at 2N)", r.Reason.ConsecutiveErrors)
	}
	if r.SessionID != "tool-error-loop-session" {
		t.Fatalf("session id = %q", r.SessionID)
	}
	// The breaker stopped EARLY: exactly 2N turns, far below the budget.
	if r.Turns != 6 {
		t.Fatalf("turns = %d, want 6 (exactly 2N turns before the hard stop)", r.Turns)
	}
	if r.Turns >= 50 {
		t.Fatalf("budget NOT fully burned expected, got turns = %d", r.Turns)
	}
	// The breaker stops AT the 2N dispatch — it does not append/continue past it —
	// so the registry saw exactly 2N == 6 calls (the 7th fixture line is unused
	// headroom: the breaker, not fixture exhaustion, ended the run).
	if reg.CallCount.Load() != 6 {
		t.Fatalf("tool dispatched %d times, want 6 (2N), then the breaker stopped", reg.CallCount.Load())
	}
}
