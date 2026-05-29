package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// Tool Escalation Protocol (issue #80). Rules R1–R9 mirror the Rust reference
// tests in rust/crates/spore-core/src/harness.rs. The Go ToolOutput.Escalate
// variant terminates the run cleanly and surfaces RunResult.Escalate; the
// harness is a pure intermediary that never acts on the signal.

func abortSignal() HarnessSignal {
	return HarnessSignal{Kind: SignalAbort, Reason: "agent requested stop"}
}

// escalatingAgentCfg builds a config whose agent requests one tool call and
// whose scripted registry returns ToolOutput.Escalate for it.
func escalatingAgentCfg(t *testing.T, sig HarnessSignal) HarnessConfig {
	t.Helper()
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "x", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputEscalate, Signal: &sig})
	cfg.ToolRegistry = reg
	return cfg
}

// R1 + R8: a dispatched Escalate{Abort} terminates the run and returns
// RunResult.Escalate, NOT RunResult.Failure.
func TestEscalateAbortTerminatesWithEscalateNotFailure(t *testing.T) {
	sig := abortSignal()
	h := NewStandardHarness(escalatingAgentCfg(t, sig))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind == RunFailure {
		t.Fatalf("R8: Abort must NOT surface as Failure: %+v", r.Reason)
	}
	if r.Kind != RunEscalate {
		t.Fatalf("R1: expected Escalate, got %q", r.Kind)
	}
	if r.Signal == nil || *r.Signal != sig {
		t.Fatalf("carried signal = %+v, want %+v", r.Signal, sig)
	}
}

// R2: the escalation is NOT appended to message history. With one escalating
// call the only appended message is the assistant tool-call turn — there is no
// Tool-role result message.
func TestEscalateIsNotAppendedToHistory(t *testing.T) {
	h := NewStandardHarness(escalatingAgentCfg(t, abortSignal()))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunEscalate || r.State == nil {
		t.Fatalf("expected Escalate with state, got %+v", r)
	}
	// The escalation is a control signal, not a conversation turn: the harness
	// must NOT append a Tool-role result for the escalating call. (The default
	// NoopContextManager appends a Tool message on every AppendToolResult, so a
	// non-zero count here would prove the escalation was wrongly recorded.)
	var toolResults int
	for _, m := range r.State.SessionState.Messages {
		if m.Role == RoleTool {
			toolResults++
		}
	}
	if toolResults != 0 {
		t.Fatalf("escalation must not append a tool result; got %d", toolResults)
	}
}

// escalationObserver captures the terminal outcome the loop finalizes with and
// whether FlushSession ran. A no-op HarnessObserver otherwise.
type escalationObserver struct {
	outcomeSet bool
	outcome    TerminalOutcome
	flushed    bool
}

func (o *escalationObserver) EmitTurn(string, SessionID, TaskID, uint32, string, uint64, TokenUsage, float64, StopReason, uint32, string, string, []ToolCall, []Message) {
}
func (o *escalationObserver) EmitToolCall(string, string, SessionID, TaskID, string, string, string, uint64, uint64, uint64, bool, bool, json.RawMessage, string) {
}
func (o *escalationObserver) SetSessionOutcome(_ SessionID, outcome TerminalOutcome, _ string) {
	o.outcomeSet = true
	o.outcome = outcome
}
func (o *escalationObserver) FlushSession(context.Context, SessionID) { o.flushed = true }
func (o *escalationObserver) CostFor(TokenUsage) float64              { return 0 }
func (o *escalationObserver) EmitCompaction(string, SessionID, TaskID, string, uint32, uint32, uint32, uint32) {
}
func (o *escalationObserver) EmitCompactionVerificationFailed(string, SessionID, TaskID, string, []string, bool) {
}

var _ HarnessObserver = (*escalationObserver)(nil)

// R3: observability is finalized with the Escalated terminal outcome.
func TestEscalateFinalizesObservabilityAsEscalated(t *testing.T) {
	cfg := escalatingAgentCfg(t, abortSignal())
	obs := &escalationObserver{}
	cfg.Observability = obs
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunEscalate {
		t.Fatalf("expected Escalate, got %q", r.Kind)
	}
	if !obs.outcomeSet || obs.outcome != TerminalEscalated {
		t.Fatalf("R3: outcome = (set=%v, %v), want Escalated", obs.outcomeSet, obs.outcome)
	}
	if !obs.flushed {
		t.Fatalf("R3: escalation must flush the finalized session")
	}
}

// R3 (contrast): WaitingForHuman does NOT finalize observability — it is not a
// terminal outcome, so no SetSessionOutcome / FlushSession is recorded.
func TestWaitingForHumanDoesNotFinalizeObservability(t *testing.T) {
	a := NewMockAgent("t")
	calls := []ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}
	a.Push(NewToolCallRequested(calls, turnUsage()))
	cfg := standardCfg(a)
	obs := &escalationObserver{}
	cfg.Observability = obs
	mw := NewScriptedMiddleware()
	req := HumanRequest{Kind: HumanReqToolApproval, Calls: calls, RiskLevel: RiskMedium}
	mw.Push(HookBeforeTool, MiddlewareDecision{Kind: MiddlewareSurfaceToHuman, Request: &req})
	cfg.Middleware = mw
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected WaitingForHuman, got %q", r.Kind)
	}
	if obs.outcomeSet || obs.flushed {
		t.Fatalf("WaitingForHuman must NOT finalize observability (set=%v, flushed=%v)", obs.outcomeSet, obs.flushed)
	}
}

// R4: all five RunResult.Escalate fields are populated (signal, state,
// session_id, usage, turns).
func TestEscalatePopulatesAllFiveFields(t *testing.T) {
	sig := abortSignal()
	h := NewStandardHarness(escalatingAgentCfg(t, sig))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunEscalate {
		t.Fatalf("expected Escalate, got %q", r.Kind)
	}
	if r.Signal == nil {
		t.Fatal("signal must be populated")
	}
	if r.State == nil {
		t.Fatal("state must be populated")
	}
	if r.SessionID == "" {
		t.Fatal("session_id must be populated")
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1 (one turn consumed before the escalating dispatch)", r.Turns)
	}
	// Turns and usage are consistent with the preserved state's budget.
	if r.Turns != r.State.BudgetUsed.Turns {
		t.Fatalf("turns %d != state.budget_used.turns %d", r.Turns, r.State.BudgetUsed.Turns)
	}
}

// R5: the preserved state is resumable — HumanRequest is nil (an escalation has
// no human request) and the session state round-trips.
func TestEscalateStateIsResumable(t *testing.T) {
	h := NewStandardHarness(escalatingAgentCfg(t, abortSignal()))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunEscalate || r.State == nil {
		t.Fatalf("expected Escalate with state, got %+v", r)
	}
	if r.State.HumanRequest != nil {
		t.Fatalf("escalation state must carry no human request, got %+v", r.State.HumanRequest)
	}
	if r.State.ChildState != nil {
		t.Fatalf("escalation state must carry no child state, got %+v", r.State.ChildState)
	}
}

// R6: the signal is discarded on resume — it is NOT stored in PausedState, so a
// resume of the preserved state never re-acts on it. Resuming the escalation
// state (with a Continue response) drives the original session to completion.
func TestEscalateSignalDiscardedOnResume(t *testing.T) {
	// Agent: one tool call (which escalates), then on resume a final response.
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c", Name: "x", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("resumed-done", turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	sig := abortSignal()
	reg.Push(ToolOutput{Kind: ToolOutputEscalate, Signal: &sig})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunEscalate || r.State == nil {
		t.Fatalf("expected Escalate, got %+v", r)
	}
	// The PausedState carries no signal field at all — it is structurally
	// impossible for resume to re-act on it. Resume continues the session.
	resumed := h.Resume(context.Background(), *r.State, HumanResponse{Kind: HumanRespAllow}, nil)
	if resumed.Kind != RunSuccess {
		t.Fatalf("resume after escalation must continue the original session, got %+v", resumed)
	}
	if resumed.Output != "resumed-done" {
		t.Fatalf("resumed output = %q, want resumed-done", resumed.Output)
	}
}

// R7: the four HarnessSignal variants and the Escalate wrappers round-trip
// through JSON byte-identically (nested "kind"-tagged unions).
func TestEscalateWireRoundTrip(t *testing.T) {
	signals := []HarnessSignal{
		{Kind: SignalEnterPlanMode, Context: "ctx"},
		{Kind: SignalExitPlanMode, Plan: &PlanArtifact{Tasks: []string{"a"}, Rationale: "r"}},
		{Kind: SignalSwitchMode, Mode: ModePlan},
		{Kind: SignalAbort, Reason: "stop"},
	}
	for _, sig := range signals {
		// HarnessSignal itself.
		data, err := json.Marshal(sig)
		if err != nil {
			t.Fatalf("marshal signal %q: %v", sig.Kind, err)
		}
		var back HarnessSignal
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("unmarshal signal %q: %v", sig.Kind, err)
		}
		reser, _ := json.Marshal(back)
		if string(reser) != string(data) {
			t.Fatalf("signal %q round-trip: %s != %s", sig.Kind, reser, data)
		}

		// ToolOutput.Escalate wrapping the signal (nested tagged union).
		out := ToolOutput{Kind: ToolOutputEscalate, Signal: &sig}
		od, err := json.Marshal(out)
		if err != nil {
			t.Fatalf("marshal ToolOutput.Escalate: %v", err)
		}
		var ob ToolOutput
		if err := json.Unmarshal(od, &ob); err != nil {
			t.Fatalf("unmarshal ToolOutput.Escalate: %v", err)
		}
		if ob.Kind != ToolOutputEscalate || ob.Signal == nil {
			t.Fatalf("ToolOutput.Escalate round-trip lost the signal: %+v", ob)
		}
		oreser, _ := json.Marshal(ob)
		if string(oreser) != string(od) {
			t.Fatalf("ToolOutput.Escalate round-trip: %s != %s", oreser, od)
		}
	}
}

// R7 (RunResult): RunResult.Escalate round-trips, carrying the signal + state +
// session_id + usage + turns, with the escalation state's human_request as null.
func TestRunResultEscalateWireRoundTrip(t *testing.T) {
	sig := HarnessSignal{Kind: SignalAbort, Reason: "stop"}
	rr := RunResult{
		Kind:      RunEscalate,
		Signal:    &sig,
		SessionID: "s",
		Usage:     AggregateUsage{InputTokens: 3, OutputTokens: 4},
		Turns:     2,
		State: &PausedState{
			SessionID: "s", TaskID: "t", TurnNumber: 2,
			Task: reactTask(5), BudgetUsed: BudgetSnapshot{Turns: 2},
		},
	}
	data, err := json.Marshal(rr)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// human_request must be present-and-null (matches Rust Option + serde default).
	var asMap map[string]json.RawMessage
	_ = json.Unmarshal(data, &asMap)
	stateRaw := asMap["state"]
	var stateMap map[string]json.RawMessage
	_ = json.Unmarshal(stateRaw, &stateMap)
	if hr, ok := stateMap["human_request"]; !ok || string(hr) != "null" {
		t.Fatalf("escalation PausedState.human_request must be null, got %q (present=%v)", hr, ok)
	}
	var back RunResult
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Kind != RunEscalate || back.Signal == nil || *back.Signal != sig {
		t.Fatalf("RunResult.Escalate round-trip lost the signal: %+v", back)
	}
	if back.Turns != 2 || back.State == nil || back.State.HumanRequest != nil {
		t.Fatalf("RunResult.Escalate round-trip lost fields: %+v", back)
	}
	reser, _ := json.Marshal(back)
	if string(reser) != string(data) {
		t.Fatalf("RunResult.Escalate round-trip: %s != %s", reser, data)
	}
}

// R9: the remaining tool calls in the batch are preserved into pending — the
// escalating call is consumed; the calls after it are carried forward.
func TestEscalatePreservesRemainingCallsInPending(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
		{ID: "c2", Name: "y", Input: json.RawMessage(`{}`)},
		{ID: "c3", Name: "z", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	// First call escalates; c2 and c3 must be preserved unprocessed.
	sig := abortSignal()
	reg.Push(ToolOutput{Kind: ToolOutputEscalate, Signal: &sig})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunEscalate || r.State == nil {
		t.Fatalf("expected Escalate, got %+v", r)
	}
	pending := r.State.PendingToolCalls
	if len(pending) != 2 {
		t.Fatalf("remaining pending = %d, want 2", len(pending))
	}
	if pending[0].ID != "c2" || pending[1].ID != "c3" {
		t.Fatalf("preserved pending order wrong: %+v", pending)
	}
	// Only the escalating call was dispatched.
	if reg.CallCount.Load() != 1 {
		t.Fatalf("dispatch count = %d, want 1 (remaining calls not dispatched)", reg.CallCount.Load())
	}
}
