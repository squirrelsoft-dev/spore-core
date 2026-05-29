package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// Issue #81 (Q4b): a tool returning AwaitingClarification pauses the loop with a
// PausedState whose HumanRequest is a Clarification (NO child_state), and the
// clarifying call is preserved as the HEAD of PendingToolCalls.
func TestAwaitingClarificationPausesWithClarification(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "ask1", Name: "ask_user_question", Input: json.RawMessage(`{"question":"which?","options":["a","b"]}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	opts := []string{"a", "b"}
	reg.Push(ToolOutput{Kind: ToolOutputAwaitingClarification, Question: "which?", Options: &opts})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))

	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected WaitingForHuman, got %q", r.Kind)
	}
	if r.Request == nil || r.Request.Kind != HumanReqClarification || r.Request.Question != "which?" {
		t.Fatalf("request: %+v", r.Request)
	}
	if r.Request.Options == nil || len(*r.Request.Options) != 2 {
		t.Fatalf("request options: %+v", r.Request.Options)
	}
	if r.State == nil {
		t.Fatalf("expected paused state")
	}
	if r.State.ChildState != nil {
		t.Fatalf("clarification pause must have NO child_state")
	}
	if len(r.State.PendingToolCalls) != 1 || r.State.PendingToolCalls[0].ID != "ask1" {
		t.Fatalf("clarifying call must be head of pending: %+v", r.State.PendingToolCalls)
	}
}

// Issue #81 (Q4b): resuming a Clarification pause injects the human's answer as
// the tool RESULT for the clarifying call, then continues the loop.
func TestResumeClarificationInjectsAnswerAsToolResult(t *testing.T) {
	a := NewMockAgent("t")
	// After resume the loop runs the agent again; it produces a final response.
	a.Push(NewFinalResponse("done-after-answer", turnUsage()))
	cfg := standardCfg(a)
	cm := &recordingContextManager{}
	cfg.ContextManager = cm
	h := NewStandardHarness(cfg)

	req := HumanRequest{Kind: HumanReqClarification, Question: "which?"}
	state := PausedState{
		SessionID:        SessionID("s1"),
		TaskID:           reactTask(5).ID,
		TurnNumber:       1,
		SessionState:     SessionState{},
		PendingToolCalls: []ToolCall{{ID: "ask1", Name: "ask_user_question", Input: json.RawMessage(`{"question":"which?"}`)}},
		HumanRequest:     &req,
		Task:             reactTask(5),
	}
	r := h.Resume(context.Background(), state, HumanResponse{Kind: HumanRespAnswer, Text: "option-a"}, nil)
	if r.Kind != RunSuccess || r.Output != "done-after-answer" {
		t.Fatalf("expected success after answer, got %+v", r)
	}
	// The answer must have been injected as the tool RESULT for the clarifying
	// call ("ask1") — a Tool-role row keyed by that call id — NOT appended as a
	// free-standing user message.
	var injectedAsToolResult bool
	for i, role := range cm.roles {
		if role == RoleTool && cm.callIDs[i] == "ask1" {
			injectedAsToolResult = true
		}
		if role == RoleUser && cm.texts[i] == "option-a" {
			t.Fatalf("answer must NOT be appended as a free-standing user message")
		}
	}
	if !injectedAsToolResult {
		t.Fatalf("answer not injected as a tool result for the clarifying call: roles=%+v", cm.roles)
	}
}

// Issue #81: the send_message tool surfaces a HarnessStreamEvent of kind
// user_message and records a minimal success result so the loop continues.
func TestSendMessageEmitsUserMessageEvent(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "m1", Name: "send_message", Input: json.RawMessage(`{"content":"hello human"}`)},
	}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "hello human"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	var events []HarnessStreamEvent
	sink := func(e HarnessStreamEvent) { events = append(events, e) }
	opts := NewHarnessRunOptions(reactTask(5))
	opts.OnStream = sink
	r := h.Run(context.Background(), opts)
	if r.Kind != RunSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	var found bool
	for _, e := range events {
		if e.Kind == HarnessStreamUserMessage {
			found = true
			if e.Content != "hello human" {
				t.Fatalf("user_message content: %q", e.Content)
			}
		}
	}
	if !found {
		t.Fatalf("expected a user_message stream event")
	}
}

// recordingContextManager is defined in harness_test.go (same package); reused
// here to assert the clarification-resume injection path.
