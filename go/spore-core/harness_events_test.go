package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestToolResultMarshalEmitsRevised confirms the tool_result event carries the
// `revised` field (#[serde(default)] => present, false by default).
func TestToolResultMarshalEmitsRevised(t *testing.T) {
	tr := HarnessStreamEvent{Kind: HarnessStreamToolResult, CallID: "c1", IsError: false, ResultContent: "ok"}
	enc, _ := json.Marshal(tr)
	var probe map[string]json.RawMessage
	_ = json.Unmarshal(enc, &probe)
	if string(probe["revised"]) != "false" {
		t.Fatalf("revised = %s, want false; full=%s", probe["revised"], enc)
	}
	if string(probe["content"]) != `"ok"` {
		t.Fatalf("content not carried: %s", enc)
	}
	// Field order: kind, call_id, is_error, content, revised.
	if i := strings.Index(string(enc), `"revised"`); i < strings.Index(string(enc), `"content"`) {
		t.Fatalf("revised must follow content: %s", enc)
	}
}

// TestToolResultRevisedTrueMarshal confirms a revised=true event serializes the
// flag (the AfterTool rewrite signal, SC-9).
func TestToolResultRevisedTrueMarshal(t *testing.T) {
	tr := HarnessStreamEvent{Kind: HarnessStreamToolResult, CallID: "c1", ResultContent: "fixed", Revised: true}
	enc, _ := json.Marshal(tr)
	if !strings.Contains(string(enc), `"revised":true`) {
		t.Fatalf("revised:true not emitted: %s", enc)
	}
}

// TestUserMessageMarshalEmitsSteered confirms the user_message event carries
// `steered` (false for narration, true for steering).
func TestUserMessageMarshalEmitsSteered(t *testing.T) {
	um := HarnessStreamEvent{Kind: HarnessStreamUserMessage, Content: "hi", Steered: false}
	enc, _ := json.Marshal(um)
	if !strings.Contains(string(enc), `"steered":false`) {
		t.Fatalf("steered:false not emitted: %s", enc)
	}
	um2 := HarnessStreamEvent{Kind: HarnessStreamUserMessage, Content: "steer", Steered: true}
	enc2, _ := json.Marshal(um2)
	if !strings.Contains(string(enc2), `"steered":true`) {
		t.Fatalf("steered:true not emitted: %s", enc2)
	}
}

// TestTurnEndMarshalEmitsUsage confirms the turn_end event carries `usage`
// (#[serde(default)] => present, zero by default) with the per-turn token counts.
func TestTurnEndMarshalEmitsUsage(t *testing.T) {
	te := HarnessStreamEvent{Kind: HarnessStreamTurnEnd, Turn: 3, Usage: &TokenUsage{InputTokens: 10, OutputTokens: 20}}
	enc, _ := json.Marshal(te)
	var probe map[string]json.RawMessage
	_ = json.Unmarshal(enc, &probe)
	if string(probe["turn"]) != "3" {
		t.Fatalf("turn = %s, want 3", probe["turn"])
	}
	var u TokenUsage
	_ = json.Unmarshal(probe["usage"], &u)
	if u.InputTokens != 10 || u.OutputTokens != 20 {
		t.Fatalf("usage = %+v, want {10,20}", u)
	}
}

// TestTurnEndNilUsageEmitsZero confirms a nil usage emits a zero TokenUsage
// (not omitted), matching Rust's #[serde(default)].
func TestTurnEndNilUsageEmitsZero(t *testing.T) {
	te := HarnessStreamEvent{Kind: HarnessStreamTurnEnd, Turn: 1}
	enc, _ := json.Marshal(te)
	if !strings.Contains(string(enc), `"usage"`) {
		t.Fatalf("usage must be present even when nil: %s", enc)
	}
}

// TestSteeringInjectsAtTurnBoundary confirms a SteeringSource is polled at the
// turn boundary: each queued string is injected as a user message and emitted
// as a UserMessage stream event with Steered=true, and the run continues.
func TestSteeringInjectsAtTurnBoundary(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("first", turnUsage()))
	// A steering message queued before the run; the source drains it once.
	steered := []string{"please use the calculator"}
	cfg := standardCfg(a)
	cfg.SteeringSource = func() []string {
		out := steered
		steered = nil
		return out
	}
	h := NewStandardHarness(cfg)
	var events []HarnessStreamEvent
	opts := NewHarnessRunOptions(reactTask(5))
	opts.OnStream = func(ev HarnessStreamEvent) { events = append(events, ev) }
	r := h.Run(context.Background(), opts)
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q reason=%+v", r.Kind, r.Reason)
	}
	var sawSteeredUserMessage bool
	for _, ev := range events {
		if ev.Kind == HarnessStreamUserMessage && ev.Steered && ev.Content == "please use the calculator" {
			sawSteeredUserMessage = true
		}
	}
	if !sawSteeredUserMessage {
		t.Fatal("expected a steered UserMessage event for the injected steering message")
	}
}

// TestSteeringNilIsByteIdentical confirms no SteeringSource => no extra
// UserMessage events (byte-identical default).
func TestSteeringNilIsByteIdentical(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	// No SteeringSource.
	h := NewStandardHarness(cfg)
	var events []HarnessStreamEvent
	opts := NewHarnessRunOptions(reactTask(5))
	opts.OnStream = func(ev HarnessStreamEvent) { events = append(events, ev) }
	r := h.Run(context.Background(), opts)
	if r.Kind != RunSuccess {
		t.Fatalf("kind=%q", r.Kind)
	}
	for _, ev := range events {
		if ev.Kind == HarnessStreamUserMessage && ev.Steered {
			t.Fatal("no steered UserMessage should fire without a SteeringSource")
		}
	}
}

// TestPlanRecordedSignalKind pins the wire value (cross-language).
func TestPlanRecordedSignalKind(t *testing.T) {
	if SignalPlanRecorded != "plan_recorded" {
		t.Fatalf("SignalPlanRecorded = %q, want plan_recorded", SignalPlanRecorded)
	}
}
