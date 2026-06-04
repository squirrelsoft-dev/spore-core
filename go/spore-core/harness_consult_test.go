package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// ============================================================================
// Mid-loop consult primitive (issue #114) — harness-side rules.
// R1 + R10 (worker pause), R6 (empty-map standalone surfaces unchanged), and
// the ResumeConsult seam live here; the SubagentTool mediation rules
// (R2–R5b, R6 no-kind, R7) live in tools/subagent_consult_test.go. R8 (fixture
// replay) lives in consult_fixture_replay_test.go.
// ============================================================================

func consultReq() ConsultRequest {
	return ConsultRequest{
		Kind:      "advice",
		Situation: "stuck on auth",
		Attempts:  3,
		Question:  "what next?",
	}
}

// recordingConsultObserver records the consult span events the loop emits so a
// test can assert on them. All other methods are no-ops.
type recordingConsultObserver struct {
	spawned []string
	resumed []struct {
		kind     string
		answered bool
	}
}

func (o *recordingConsultObserver) EmitTurn(string, SessionID, TaskID, uint32, string, uint64, TokenUsage, float64, StopReason, uint32, string, string, []ToolCall, []Message) {
}
func (o *recordingConsultObserver) EmitToolCall(string, string, SessionID, TaskID, string, string, string, uint64, uint64, uint64, bool, bool, json.RawMessage, string) {
}
func (o *recordingConsultObserver) SetSessionOutcome(SessionID, TerminalOutcome, string) {}
func (o *recordingConsultObserver) FlushSession(context.Context, SessionID)              {}
func (o *recordingConsultObserver) CostFor(TokenUsage) float64                           { return 0 }
func (o *recordingConsultObserver) EmitCompaction(string, SessionID, TaskID, string, uint32, uint32, uint32, uint32) {
}
func (o *recordingConsultObserver) EmitCompactionVerificationFailed(string, SessionID, TaskID, string, []string, bool) {
}
func (o *recordingConsultObserver) EmitHillClimbingIteration(string, SessionID, TaskID, string, uint32, float64, bool, float64, bool, string, bool) {
}
func (o *recordingConsultObserver) EmitConsultSpawned(_ string, _ SessionID, _ TaskID, _ string, kind string) {
	o.spawned = append(o.spawned, kind)
}
func (o *recordingConsultObserver) EmitConsultResumed(_ string, _ SessionID, _ TaskID, _ string, kind string, answered bool) {
	o.resumed = append(o.resumed, struct {
		kind     string
		answered bool
	}{kind, answered})
}

var _ HarnessObserver = (*recordingConsultObserver)(nil)

// R1 + R10: a worker-side tool returning ToolOutput.Consult pauses the loop and
// returns RunResult.Consult. The consult call is the HEAD of PendingToolCalls,
// HumanRequest is nil, there is no ChildState, and the consult is NOT appended
// to message history (R10).
func TestWorkerConsultPausesAndReturnsRunConsult(t *testing.T) {
	a := NewMockAgent("t")
	// A single turn with TWO tool calls: c0 consults, c1 trails (preserved).
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c0", Name: "ask_advice", Input: json.RawMessage(`{"kind":"advice"}`)},
		{ID: "c1", Name: "noop", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(NewToolOutputConsult(consultReq()))
	cfg.ToolRegistry = reg
	obs := &recordingConsultObserver{}
	cfg.Observability = obs
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunConsult {
		t.Fatalf("expected RunConsult, got %q (%+v)", r.Kind, r)
	}
	if r.ConsultRequest == nil || r.ConsultRequest.Kind != "advice" || r.ConsultRequest.Question != "what next?" {
		t.Fatalf("bad request: %+v", r.ConsultRequest)
	}
	if r.State == nil {
		t.Fatal("nil state")
	}
	// R1: human_request nil, no child_state.
	if r.State.HumanRequest != nil {
		t.Fatalf("HumanRequest should be nil, got %+v", r.State.HumanRequest)
	}
	if r.State.ChildState != nil {
		t.Fatalf("ChildState should be nil, got %+v", r.State.ChildState)
	}
	// R1: consulting call (c0) is head; c1 trails.
	if len(r.State.PendingToolCalls) != 2 {
		t.Fatalf("expected 2 pending, got %d", len(r.State.PendingToolCalls))
	}
	if r.State.PendingToolCalls[0].ID != "c0" || r.State.PendingToolCalls[1].ID != "c1" {
		t.Fatalf("bad pending order: %+v", r.State.PendingToolCalls)
	}
	// R10: the consult produced NO tool-result turn in history.
	toolTurns := 0
	for _, m := range r.State.SessionState.Messages {
		if m.Role == RoleTool {
			toolTurns++
		}
	}
	if toolTurns != 0 {
		t.Fatalf("consult must not append a tool result yet; got %d tool turns", toolTurns)
	}
	// Exactly one dispatch (the consulting call); c1 was preserved, not run.
	if reg.CallCount.Load() != 1 {
		t.Fatalf("call count = %d, want 1", reg.CallCount.Load())
	}
	// Observability: a consult-spawn event for kind "advice".
	if len(obs.spawned) != 1 || obs.spawned[0] != "advice" {
		t.Fatalf("expected one consult-spawn for advice, got %+v", obs.spawned)
	}
}

// R6 (graceful degradation): an empty ConsultHandlers map means a standalone
// worker simply surfaces RunResult.Consult to its caller unchanged (existing
// callers unaffected). The harness itself never mediates — mediation is the
// SubagentTool's job.
func TestEmptyConsultHandlersSurfacesConsultUnchanged(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c0", Name: "ask_advice", Input: json.RawMessage(`{"kind":"advice"}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	if len(cfg.ConsultHandlers) != 0 {
		t.Fatalf("expected empty handlers by default")
	}
	reg := NewScriptedToolRegistry()
	reg.Push(NewToolOutputConsult(consultReq()))
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunConsult {
		t.Fatalf("expected RunConsult surfaced unchanged, got %q", r.Kind)
	}
}

// R10 + resume seam: ResumeConsult with an Answer injects it as the TOOL RESULT
// for the head pending (consult) call, then continues to Success. Also asserts
// the consult-resumed observability event (answered=true).
func TestResumeConsultInjectsAnswerAsToolResult(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("consult-resumed-done", turnUsage()))
	cfg := standardCfg(a)
	obs := &recordingConsultObserver{}
	cfg.Observability = obs
	h := NewStandardHarness(cfg)

	state := PausedState{
		SessionID:  SessionID("s"),
		TaskID:     TaskID("t"),
		TurnNumber: 1,
		PendingToolCalls: []ToolCall{
			{ID: "consult", Name: "ask_advice", Input: json.RawMessage(`{"kind":"advice"}`)},
		},
		Task: reactTask(5),
	}
	r := h.ResumeConsult(context.Background(), state, NewConsultAnswer("the answer"), nil)
	if r.Kind != RunSuccess || r.Output != "consult-resumed-done" {
		t.Fatalf("expected Success consult-resumed-done, got %q out=%q", r.Kind, r.Output)
	}
	if len(obs.resumed) != 1 || obs.resumed[0].kind != "advice" || !obs.resumed[0].answered {
		t.Fatalf("expected one consult-resume (advice, answered=true), got %+v", obs.resumed)
	}
	// The answer must be the head consult call's tool RESULT in history.
	foundAnswer := false
	for _, m := range r.SessionState.Messages {
		if m.Role == RoleTool && m.Content.Text == "the answer" {
			foundAnswer = true
		}
	}
	if !foundAnswer {
		t.Fatalf("answer was not injected as a tool result: %+v", r.SessionState.Messages)
	}
}

// Resume seam: a BudgetExhausted response injects its message as the tool result
// and records answered=false.
func TestResumeConsultBudgetExhaustedAnsweredFalse(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("finished-without-help", turnUsage()))
	cfg := standardCfg(a)
	obs := &recordingConsultObserver{}
	cfg.Observability = obs
	h := NewStandardHarness(cfg)

	state := PausedState{
		SessionID:  SessionID("s"),
		TaskID:     TaskID("t"),
		TurnNumber: 1,
		PendingToolCalls: []ToolCall{
			{ID: "consult", Name: "ask_advice", Input: json.RawMessage(`{"kind":"research"}`)},
		},
		Task: reactTask(5),
	}
	r := h.ResumeConsult(context.Background(), state, NewConsultBudgetExhausted("no more help"), nil)
	if r.Kind != RunSuccess || r.Output != "finished-without-help" {
		t.Fatalf("got %q out=%q", r.Kind, r.Output)
	}
	if len(obs.resumed) != 1 || obs.resumed[0].kind != "research" || obs.resumed[0].answered {
		t.Fatalf("expected one consult-resume (research, answered=false), got %+v", obs.resumed)
	}
}
