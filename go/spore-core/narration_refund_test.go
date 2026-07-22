package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// TestNarrationRefundLetsFinalTurnRun exercises the ReAct window-gate
// narration refund (Rust react.rs lines 334-340, 377, 589-601): a
// send_message-ONLY turn is pure narration (#81) and refunds one turn of the
// gate cap, so the gate is widened by the refund count. With maxIterations=2
// and two narration turns then a final response, the gate before turn 3 is
// 2 >= saturating(2, 2)=4 (false), so turn 3 (the final) RUNS.
//
// The outer leaf-scope budget is a SEPARATE accounting scope that charges raw
// turns (Rust react.rs line 2213: `cx.charge_current(window_turns)`), so the
// top-level Run may still surface BudgetExceeded — that is NOT what this test
// asserts. The discriminating observable is that the model was called 3 times
// (the final turn was not gated out) and the final_response stream event was
// produced.
func TestNarrationRefundLetsFinalTurnRun(t *testing.T) {
	a := NewMockAgent("t")
	msgCall := []ToolCall{{ID: "c1", Name: sendMessageToolName, Input: json.RawMessage(`{"content":"narrating"}`)}}
	a.Push(NewToolCallRequested(msgCall, turnUsage())) // turn 1: narration
	a.Push(NewToolCallRequested(msgCall, turnUsage())) // turn 2: narration
	a.Push(NewFinalResponse("done", turnUsage()))      // turn 3: final

	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "narrating"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "narrating"})

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	var sawFinal bool
	opts := NewHarnessRunOptions(reactTask(2))
	opts.OnStream = func(ev HarnessStreamEvent) {
		if ev.Kind == "final_response" {
			sawFinal = true
		}
	}
	h.Run(context.Background(), opts)

	// The model was called exactly 3 times: the two narration turns plus the
	// final response turn. Without the refund the gate would fire before turn
	// 3 (2 >= 2) and the model would be called only twice.
	if got := a.CallCount.Load(); got != 3 {
		t.Fatalf("model call count = %d, want 3 (final turn must run via refund)", got)
	}
	// The final_response stream event was produced — the final turn ran and
	// emitted its output into the stream.
	if !sawFinal {
		t.Fatalf("final_response stream event was not produced (final turn gated out)")
	}
}

// TestNarrationRefundBoundedByCap confirms the refund is bounded by the cap so
// a pure-narration loop still terminates (~2x budget): with maxIterations=2,
// four narration turns then a final response still halts at the cap (refund
// capped at 2, so the gate before turn 5 fires: 4 >= saturating(2,2)=4).
func TestNarrationRefundBoundedByCap(t *testing.T) {
	a := NewMockAgent("t")
	msgCall := []ToolCall{{ID: "c1", Name: sendMessageToolName, Input: json.RawMessage(`{"content":"narrating"}`)}}
	for i := 0; i < 4; i++ {
		a.Push(NewToolCallRequested(msgCall, turnUsage()))
	}
	a.Push(NewFinalResponse("done", turnUsage()))

	reg := NewScriptedToolRegistry()
	for i := 0; i < 4; i++ {
		reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "narrating"})
	}

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunFailure {
		t.Fatalf("kind=%q, want failure (narration loop must terminate at ~2x budget)", r.Kind)
	}
	if r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("reason=%q, want budget_exceeded", r.Reason.Kind)
	}
}

// TestNonNarrationTurnNotRefunded confirms a turn with a non-send_message tool
// is NOT refunded (it counts against the productive budget): with
// maxIterations=2, two real tool calls then a final response halts at the cap
// (no refund, gate before turn 3 fires: 2 >= 2).
func TestNonNarrationTurnNotRefunded(t *testing.T) {
	a := NewMockAgent("t")
	realCall := []ToolCall{{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)}}
	a.Push(NewToolCallRequested(realCall, turnUsage()))
	a.Push(NewToolCallRequested(realCall, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))

	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	// ErrorLoopThreshold is nil => default N=3, so the repeated-success breaker
	// (2N=6) does not fire at 2 identical calls.
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("kind=%q reason=%+v, want BudgetExceeded (non-narration turns not refunded)", r.Kind, r.Reason)
	}
}
