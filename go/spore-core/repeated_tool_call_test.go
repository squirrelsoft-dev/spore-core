package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// TestRepeatedIdenticalSuccessfulCallsHardStop confirms the no-progress twin of
// the error breaker: a model that calls the SAME tool with identical args and
// SUCCEEDS 2*N times in a row hard-stops with HaltRepeatedToolCall. Mirrors the
// Rust repeated_identical_successful_calls_hard_stop (default N=3 -> stop at 6).
func TestRepeatedIdenticalSuccessfulCallsHardStop(t *testing.T) {
	// Use N=1 so 2N=2: two identical successful calls stop the run.
	n := uint32(1)
	a := NewMockAgent("t")
	sameCall := []ToolCall{{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)}}
	// Turn 1: one identical call (count=1, =N -> nudge). Turn 2: another
	// identical call (count=2, =2N -> hard stop).
	a.Push(NewToolCallRequested(sameCall, turnUsage()))
	a.Push(NewToolCallRequested(sameCall, turnUsage()))

	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"}) // turn 1
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"}) // turn 2

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	cfg.ErrorLoopThreshold = &n
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunFailure {
		t.Fatalf("kind=%q, want failure", r.Kind)
	}
	if r.Reason.Kind != HaltRepeatedToolCall {
		t.Fatalf("reason=%q, want repeated_tool_call", r.Reason.Kind)
	}
	if r.Reason.Tool != "x" {
		t.Fatalf("tool=%q, want x", r.Reason.Tool)
	}
	if r.Reason.ConsecutiveErrors != 2 {
		t.Fatalf("count=%d, want 2", r.Reason.ConsecutiveErrors)
	}
}

// TestRepeatedCallNudgeAtN confirms that at N identical successes a corrective
// nudge is injected (the run continues), and a DIFFERENT call resets the run.
func TestRepeatedCallNudgeAtNThenDifferentResets(t *testing.T) {
	n := uint32(2) // 2N=4
	a := NewMockAgent("t")
	sameCall := []ToolCall{{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)}}
	diffCall := []ToolCall{{ID: "c2", Name: "x", Input: json.RawMessage(`{"q":"other"}`)}}
	// Turn 1: same (count=1). Turn 2: same (count=2 =N -> nudge). Turn 3:
	// DIFFERENT args (resets count=1). Turn 4: final response.
	a.Push(NewToolCallRequested(sameCall, turnUsage()))
	a.Push(NewToolCallRequested(sameCall, turnUsage()))
	a.Push(NewToolCallRequested(diffCall, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))

	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok-other"})

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	cfg.ErrorLoopThreshold = &n
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v, want success (different call reset the run)", r.Kind, r.Reason)
	}
}

// TestRepeatedCallExemptSendMessage confirms send_message is exempt from the
// repeated-success breaker (benign narration a model may legitimately repeat).
func TestRepeatedCallExemptSendMessage(t *testing.T) {
	n := uint32(1) // 2N=2
	a := NewMockAgent("t")
	// Two identical send_message calls (both succeed) should NOT hard-stop.
	msgCall := []ToolCall{{ID: "c1", Name: sendMessageToolName, Input: json.RawMessage(`{"content":"hi"}`)}}
	a.Push(NewToolCallRequested(msgCall, turnUsage()))
	a.Push(NewToolCallRequested(msgCall, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))

	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "hi"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "hi"})

	cfg := standardCfg(a)
	cfg.ToolRegistry = reg
	cfg.ErrorLoopThreshold = &n
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(20)))
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v, want success (send_message is exempt)", r.Kind, r.Reason)
	}
}
