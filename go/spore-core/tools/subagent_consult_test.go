package tools

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// Mid-loop consult mediation — seam A1 (issue #114). Orchestrator-side rules:
// R2 (mediate, not bubble), R3 (route by kind, no parent model, parent sees
// Success), R4 (per-kind budget), R5a (SoftFail → BudgetExhausted resume),
// R5b (EscalateToHuman → WaitingForHuman), R6 (no-kind / no-handlers →
// Escalate), R7 (depth-1: the handler is the orchestrator's direct child).
// ============================================================================

// recordingHandler records each instruction it is run with and returns a fixed
// answer — used to assert depth-1 routing (R3/R7).
type recordingHandler struct {
	answer string
	mu     sync.Mutex
	seen   []string
}

func (h *recordingHandler) Run(_ context.Context, opts sporecore.HarnessRunOptions) sporecore.RunResult {
	h.mu.Lock()
	h.seen = append(h.seen, opts.Task.Instruction)
	h.mu.Unlock()
	return sporecore.RunResult{Kind: sporecore.RunSuccess, Output: h.answer}
}
func (h *recordingHandler) Resume(context.Context, sporecore.PausedState, sporecore.HumanResponse, sporecore.StreamSink) sporecore.RunResult {
	return sporecore.RunResult{Kind: sporecore.RunFailure, Reason: sporecore.HaltReason{Kind: sporecore.HaltHumanHalted}}
}
func (h *recordingHandler) ResumeConsult(context.Context, sporecore.PausedState, sporecore.ConsultResponse, sporecore.StreamSink) sporecore.RunResult {
	return sporecore.RunResult{Kind: sporecore.RunFailure, Reason: sporecore.HaltReason{Kind: sporecore.HaltHumanHalted}}
}
func (h *recordingHandler) seenCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.seen)
}

func consultPaused() *sporecore.PausedState {
	return &sporecore.PausedState{
		SessionID:  "worker",
		TaskID:     "t",
		TurnNumber: 1,
		PendingToolCalls: []sporecore.ToolCall{
			{ID: "consult-call", Name: "ask_advice", Input: json.RawMessage(`{"kind":"advice"}`)},
		},
		Task: sporecore.NewTask("audit", "worker", sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: 4}),
	}
}

func consultRequest(kind string) *sporecore.ConsultRequest {
	return &sporecore.ConsultRequest{Kind: kind, Situation: "drowning", Attempts: 2, Question: "what now?"}
}

func consultResult(kind string) sporecore.RunResult {
	return sporecore.RunResult{
		Kind:           sporecore.RunConsult,
		ConsultRequest: consultRequest(kind),
		State:          consultPaused(),
		SessionID:      "worker",
		Turns:          1,
	}
}

func consultHandlers(kind, answer string, budget uint32, overflow sporecore.ConsultOverflowPolicyKind, h *recordingHandler) map[string]sporecore.ConsultHandlerEntry {
	return map[string]sporecore.ConsultHandlerEntry{
		kind: {
			Handler:  h,
			Budget:   budget,
			Overflow: sporecore.ConsultOverflowPolicy{Kind: overflow},
		},
	}
}

// R2/R3: child Consult is MEDIATED here (not bubbled). With a registered
// handler, the handler runs (no parent model), the worker is resumed, and the
// parent ultimately sees Success.
func TestConsultIsMediatedAndResumedToSuccess(t *testing.T) {
	handler := &recordingHandler{answer: "try plan B"}
	// First Run() => Consult; ResumeConsult (default scripted) => Success.
	h := &scriptedHarness{results: []sporecore.RunResult{consultResult("advice")}}
	s := newSubagent(t, h, Isolated{}).
		WithConsultHandlers(consultHandlers("advice", "try plan B", 3, sporecore.ConsultOverflowSoftFail, handler))

	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "child done after consult" {
		t.Fatalf("expected mediated Success, got %+v", r)
	}
	// R3/R7: the handler ran exactly once, on the rendered consult request.
	if handler.seenCount() != 1 {
		t.Fatalf("handler ran %d times, want 1", handler.seenCount())
	}
	if !strings.Contains(handler.seen[0], "advice") || !strings.Contains(handler.seen[0], "what now?") {
		t.Fatalf("handler instruction missing context: %q", handler.seen[0])
	}
	// R3: worker resumed with the handler's answer.
	if len(h.resumeLog) != 1 || h.resumeLog[0].Kind != sporecore.ConsultRespAnswer || h.resumeLog[0].Text != "try plan B" {
		t.Fatalf("expected resume with Answer 'try plan B', got %+v", h.resumeLog)
	}
}

// R4 + R5a: handler runs up to budget times; the (budget+1)th consult overflows.
// With SoftFail, the worker is resumed with BudgetExhausted and finishes.
func TestBudgetOverflowSoftFailResumesWithBudgetExhausted(t *testing.T) {
	handler := &recordingHandler{answer: "advice answer"}
	// budget = 1: Run() => Consult; ResumeConsult => Consult again (over budget);
	// ResumeConsult => Success.
	h := &scriptedHarness{results: []sporecore.RunResult{
		consultResult("advice"),
		consultResult("advice"),
	}}
	s := newSubagent(t, h, Isolated{}).
		WithConsultHandlers(consultHandlers("advice", "advice answer", 1, sporecore.ConsultOverflowSoftFail, handler))

	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	// R4: handler ran exactly once (budget = 1).
	if handler.seenCount() != 1 {
		t.Fatalf("handler ran %d times, want 1", handler.seenCount())
	}
	// R5a: first resume = Answer, second resume = BudgetExhausted.
	if len(h.resumeLog) != 2 {
		t.Fatalf("expected 2 resumes, got %d", len(h.resumeLog))
	}
	if h.resumeLog[0].Kind != sporecore.ConsultRespAnswer {
		t.Fatalf("first resume should be Answer, got %q", h.resumeLog[0].Kind)
	}
	if h.resumeLog[1].Kind != sporecore.ConsultRespBudgetExhausted {
		t.Fatalf("second resume should be BudgetExhausted, got %q", h.resumeLog[1].Kind)
	}
}

// R5b: budget overflow with EscalateToHuman → ToolOutput.WaitingForHuman.
func TestBudgetOverflowEscalateToHuman(t *testing.T) {
	handler := &recordingHandler{answer: "x"}
	// budget = 0: the FIRST consult is already over budget → escalate.
	h := &scriptedHarness{results: []sporecore.RunResult{consultResult("advice")}}
	s := newSubagent(t, h, Isolated{}).
		WithConsultHandlers(consultHandlers("advice", "x", 0, sporecore.ConsultOverflowEscalateToHuman, handler))

	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputWaitingForHuman {
		t.Fatalf("expected WaitingForHuman, got %+v", r)
	}
	if r.ChildState == nil || r.ChildState.ParentToolCallID != "parent-call-1" {
		t.Fatalf("bad child state: %+v", r.ChildState)
	}
	if r.Request == nil || r.Request.Kind != sporecore.HumanReqReview {
		t.Fatalf("expected Review human request, got %+v", r.Request)
	}
	// Handler never ran (over budget from the start).
	if handler.seenCount() != 0 {
		t.Fatalf("handler should not have run, ran %d", handler.seenCount())
	}
}

// R6: a consult with NO matching handler (map present, wrong kind) →
// ToolOutput.Escalate (loud, not silent).
func TestConsultNoMatchingKindEscalates(t *testing.T) {
	handler := &recordingHandler{answer: "x"}
	h := &scriptedHarness{results: []sporecore.RunResult{consultResult("research")}}
	s := newSubagent(t, h, Isolated{}).
		WithConsultHandlers(consultHandlers("advice", "x", 3, sporecore.ConsultOverflowSoftFail, handler))

	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputEscalate {
		t.Fatalf("expected Escalate, got %+v", r)
	}
	if r.Signal == nil || r.Signal.Kind != sporecore.SignalAbort || !strings.Contains(r.Signal.Reason, "research") {
		t.Fatalf("expected Abort mentioning research, got %+v", r.Signal)
	}
}

// R6 (degradation, R9): with NO handlers installed at all, a child consult is
// treated as the no-matching-kind case → Escalate. Existing callers that never
// set ConsultHandlers are therefore unaffected — their subagents simply do not
// mediate.
func TestConsultWithNoHandlersEscalates(t *testing.T) {
	h := &scriptedHarness{results: []sporecore.RunResult{consultResult("advice")}}
	s := newSubagent(t, h, Isolated{})
	r := s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputEscalate {
		t.Fatalf("expected Escalate, got %+v", r)
	}
}

// R7 (depth-1): the handler harness is run as the ORCHESTRATOR's direct child —
// it is never installed under the worker's registry, and SubagentTool itself
// still reports IsSubagentTool so the depth-1 construction guard remains intact.
func TestConsultHandlerRunsAsOrchestratorChild(t *testing.T) {
	handler := &recordingHandler{answer: "ans"}
	h := &scriptedHarness{results: []sporecore.RunResult{consultResult("advice")}}
	s := newSubagent(t, h, Isolated{}).
		WithConsultHandlers(consultHandlers("advice", "ans", 3, sporecore.ConsultOverflowSoftFail, handler))
	_ = s.Execute(context.Background(), subagentCall(map[string]any{"instruction": "x"}), sporecore.AllowAllSandbox{}, nil)
	// The handler ran directly (its Run was invoked), proving it is the
	// orchestrator's direct child rather than nested under the worker.
	if handler.seenCount() != 1 {
		t.Fatalf("handler should run once as orchestrator child, ran %d", handler.seenCount())
	}
	if !s.IsSubagentTool() {
		t.Fatal("SubagentTool must still report IsSubagentTool for the depth-1 guard")
	}
}
