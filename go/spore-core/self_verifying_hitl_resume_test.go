package sporecore

// SC-BUG-1 (#156): a HITL approval / deny / clarification resume raised INSIDE a
// SelfVerifying frame must RE-ENTER the SelfVerifying frame on resume, not run
// only the bare build (ReAct) leaf. The original run pauses on the BUILD phase's
// first tool call (a caller approval middleware → SurfaceToHuman, or a tool that
// returns AwaitingClarification). Before the fix, resumeInner ran runReAct on the
// paused ReAct leaf and returned its Success directly — the evaluate phase +
// verifier never ran, so the looper's eval-frame reviewer was silently skipped
// (SC-30 inert for the looper). After the fix the pause carries the full composed
// task (finish rewrites it on the way up, exactly as it already does for Consult)
// and the resume re-drives the whole SelfVerifying strategy from the approved
// worker session, so the verifier runs and a verdict is reached.
//
// These live in package sporecore (alongside the SelfVerifying scaffolding —
// svVerifier / selfVerifyTask / selfVerifyCfg) and use the in-package
// ScriptedMiddleware double, which supports SurfaceToHuman at BeforeTool. No
// import of the middleware package is needed, so there is no import cycle.

import (
	"context"
	"encoding/json"
	"testing"
)

// hitlWorker scripts a SelfVerifying worker for the HITL-resume tests:
//   - build turn 1: a tool call (gated → pause)
//   - build turn 2: a FinalResponse once the approved/answered call dispatches
//   - eval turn:    a tool-FREE FinalResponse so it never re-trips the gate
//
// The eval phase reuses the inner worker's resolved agent (Q1c), so the same
// MockAgent queue is drained across the build and evaluate phases.
func hitlWorker(toolName string) *MockAgent {
	a := NewMockAgent("a")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: toolName, Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("built", turnUsage()))
	a.Push(NewFinalResponse("reviewed: PASS", turnUsage()))
	return a
}

// approvalMiddleware surfaces the build phase's first tool call to the human.
func approvalMiddleware(toolName string, risk RiskLevel) *ScriptedMiddleware {
	mw := NewScriptedMiddleware()
	req := HumanRequest{
		Kind:      HumanReqToolApproval,
		Calls:     []ToolCall{{ID: "c1", Name: toolName, Input: json.RawMessage(`{}`)}},
		RiskLevel: risk,
	}
	mw.Push(HookBeforeTool, MiddlewareDecision{Kind: MiddlewareSurfaceToHuman, Request: &req})
	return mw
}

// TestSelfVerifyingHITLResumeReentersEvalFrame: an APPROVED tool-approval resume
// must re-enter the SelfVerifying frame so the eval-phase verifier runs.
func TestSelfVerifyingHITLResumeReentersEvalFrame(t *testing.T) {
	worker := hitlWorker("write_file")
	v := newSVVerifier(3, "pass")
	cfg := selfVerifyCfg(worker, v)
	reg := NewScriptedToolRegistry()
	cfg.ToolRegistry = reg
	cfg.Middleware = approvalMiddleware("write_file", RiskMedium)
	h := NewStandardHarness(cfg)

	// 1) The run pauses on the build-phase tool call.
	paused := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if paused.Kind != RunWaitingForHuman || paused.State == nil {
		t.Fatalf("expected WaitingForHuman on the build tool call, got %+v", paused)
	}
	state := *paused.State
	// Part 1: the unwound pause must carry the COMPOSED task so the resume can
	// re-enter the frame — not the bare ReAct build leaf.
	if state.Task.LoopStrategy.Kind != StrategySelfVerifying {
		t.Fatalf("the unwound pause must carry the SelfVerifying task, got %q",
			state.Task.LoopStrategy.Kind)
	}

	// 2) Approve and resume.
	resumed := h.Resume(context.Background(), state, HumanResponse{Kind: HumanRespAllow}, nil)
	if resumed.Kind != RunSuccess {
		t.Fatalf("resume must run to a verdict, got %+v", resumed)
	}

	// Load-bearing: the eval-phase verifier ran AFTER the resume — i.e. the
	// SelfVerifying frame was re-entered. Before the fix the resume returned the
	// bare leaf's Success and this count stays 0.
	if got := len(v.seenInputs()); got < 1 {
		t.Fatalf("SelfVerifying frame must be re-entered on HITL resume so the "+
			"eval-phase verifier runs (got %d verifier calls)", got)
	}
	// And the approved build tool actually dispatched on resume.
	if got := reg.CallCount.Load(); got < 1 {
		t.Fatalf("the approved build-phase tool must dispatch on resume (got %d)", got)
	}
}

// TestSelfVerifyingHITLDenyReentersEvalFrame: a DENIED tool-approval resume must
// also re-enter the SelfVerifying frame. Deny appends a recoverable error tool
// result for the gated call and then re-drives the strategy (the same final-match
// tail as Allow). The build continues from the denial and the evaluate phase runs.
func TestSelfVerifyingHITLDenyReentersEvalFrame(t *testing.T) {
	worker := hitlWorker("write_file")
	v := newSVVerifier(3, "pass")
	cfg := selfVerifyCfg(worker, v)
	cfg.Middleware = approvalMiddleware("write_file", RiskHigh)
	h := NewStandardHarness(cfg)

	paused := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if paused.Kind != RunWaitingForHuman || paused.State == nil {
		t.Fatalf("expected WaitingForHuman, got %+v", paused)
	}
	state := *paused.State
	if state.Task.LoopStrategy.Kind != StrategySelfVerifying {
		t.Fatalf("the unwound pause must carry the SelfVerifying task, got %q",
			state.Task.LoopStrategy.Kind)
	}

	resumed := h.Resume(context.Background(), state,
		HumanResponse{Kind: HumanRespDeny, Reason: "not allowed"}, nil)
	if resumed.Kind != RunSuccess {
		t.Fatalf("deny resume must run to a verdict, got %+v", resumed)
	}
	if got := len(v.seenInputs()); got < 1 {
		t.Fatalf("SelfVerifying frame must be re-entered on a DENY resume so the "+
			"eval-phase verifier runs (got %d verifier calls)", got)
	}
}

// TestSelfVerifyingHITLClarificationReentersEvalFrame: a clarification resume (the
// SEPARATE resumeInner clarification tail) must also re-enter the SelfVerifying
// frame. The build worker's tool returns AwaitingClarification; the human's Answer
// is injected as that call's tool result and the strategy is re-driven, so the
// evaluate phase runs. NO approval middleware — the tool itself pauses the build.
func TestSelfVerifyingHITLClarificationReentersEvalFrame(t *testing.T) {
	worker := hitlWorker("ask")
	v := newSVVerifier(3, "pass")
	cfg := selfVerifyCfg(worker, v)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputAwaitingClarification, Question: "which file?"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	paused := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if paused.Kind != RunWaitingForHuman || paused.State == nil {
		t.Fatalf("expected WaitingForHuman, got %+v", paused)
	}
	if paused.Request == nil || paused.Request.Kind != HumanReqClarification {
		t.Fatalf("expected a clarification pause, got %+v", paused.Request)
	}
	state := *paused.State
	if state.Task.LoopStrategy.Kind != StrategySelfVerifying {
		t.Fatalf("the unwound pause must carry the SelfVerifying task, got %q",
			state.Task.LoopStrategy.Kind)
	}

	resumed := h.Resume(context.Background(), state,
		HumanResponse{Kind: HumanRespAnswer, Text: "the config file"}, nil)
	if resumed.Kind != RunSuccess {
		t.Fatalf("clarification resume must run to a verdict, got %+v", resumed)
	}
	if got := len(v.seenInputs()); got < 1 {
		t.Fatalf("SelfVerifying frame must be re-entered on a clarification resume "+
			"so the eval-phase verifier runs (got %d verifier calls)", got)
	}
}
