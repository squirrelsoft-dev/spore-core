package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// replayAgentFor builds a ModelAgent backed by a ReplayModel that synthesizes a
// delta stream from a single recorded response (issue #103, Q6).
func replayAgentFor(resp ModelResponse) *ModelAgent {
	exchanges := []RecordedExchange{{
		Request:  ModelRequest{Stream: true},
		Response: resp,
		Provider: "anthropic",
	}}
	replay := NewReplayModel(exchanges, ProviderInfo{
		Name: "anthropic", ModelID: "replay", ContextWindow: 200_000,
	})
	return NewModelAgent(AgentID("stream-agent"), replay)
}

func collectSink(events *[]StreamEvent) AgentStreamSink {
	return func(ev StreamEvent) { *events = append(*events, ev) }
}

// Rule: TurnStreaming forwards raw model StreamEvents in order, concatenates
// text deltas into the FinalResponse content, and surfaces reasoning both as
// ThinkingDelta events AND in TurnResult.Reasoning (Q4).
func TestTurnStreamingForwardsTextAndThinkingDeltas(t *testing.T) {
	agent := replayAgentFor(ModelResponse{
		Content: []ContentBlock{
			NewThinkingBlock("reasoning here"),
			NewTextBlock("final text"),
		},
		Usage:      usage(3, 4),
		StopReason: StopEndTurn,
	})
	var seen []StreamEvent
	result := agent.TurnStreaming(context.Background(), userCtx("hi"), collectSink(&seen))

	if result.Kind != TurnFinalResponse {
		t.Fatalf("expected FinalResponse, got %v", result.Kind)
	}
	if result.Content != "final text" {
		t.Fatalf("content = %q", result.Content)
	}
	if result.Reasoning == nil || *result.Reasoning != "reasoning here" {
		t.Fatalf("reasoning = %v, want %q", result.Reasoning, "reasoning here")
	}

	if len(seen) == 0 || seen[0].Type != StreamMessageStart {
		t.Fatalf("first event must be message_start, got %+v", seen)
	}
	if seen[len(seen)-1].Type != StreamMessageStop {
		t.Fatalf("last event must be message_stop, got %v", seen[len(seen)-1].Type)
	}
	foundThinking, foundText := false, false
	for _, ev := range seen {
		if ev.Type == StreamThinkingDelta && ev.Delta == "reasoning here" {
			foundThinking = true
		}
		if ev.Type == StreamContentBlockDelta && ev.Delta == "final text" {
			foundText = true
		}
	}
	if !foundThinking {
		t.Fatal("expected a thinking_delta carrying the reasoning")
	}
	if !foundText {
		t.Fatal("expected a content_block_delta carrying the text")
	}
}

// Rule: tool-use streaming yields ToolUseDelta with the full args JSON, and
// TurnStreaming reassembles the ToolCall with accumulated args. Known
// limitation (#103): the reassembled tool name is empty and the id is the
// synthesized per-index id.
func TestTurnStreamingReassemblesToolCall(t *testing.T) {
	agent := replayAgentFor(ModelResponse{
		Content: []ContentBlock{
			NewThinkingBlock("let me think"),
			NewToolUseBlock(ToolCall{
				ID:    "toolu_1",
				Name:  "lookup",
				Input: json.RawMessage(`{"q":"rust"}`),
			}),
		},
		Usage:      usage(7, 11),
		StopReason: StopToolUse,
	})
	var seen []StreamEvent
	result := agent.TurnStreaming(context.Background(), userCtx("hi"), collectSink(&seen))

	if result.Kind != TurnToolCallRequested {
		t.Fatalf("expected ToolCallRequested, got %v", result.Kind)
	}
	if len(result.Calls) != 1 {
		t.Fatalf("calls = %d, want 1", len(result.Calls))
	}
	// Args round-trip faithfully.
	var got map[string]string
	if err := json.Unmarshal(result.Calls[0].Input, &got); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if got["q"] != "rust" {
		t.Fatalf("args = %v, want q=rust", got)
	}
	// KNOWN LIMITATION: name is empty under streamed turns, id is synthesized.
	if result.Calls[0].Name != "" {
		t.Fatalf("streamed coarse tool name must be empty, got %q", result.Calls[0].Name)
	}
	if result.Calls[0].ID != "call_1" {
		t.Fatalf("streamed id = %q, want call_1 (per-index synthesized)", result.Calls[0].ID)
	}
	if result.Reasoning == nil || *result.Reasoning != "let me think" {
		t.Fatalf("reasoning = %v", result.Reasoning)
	}
	foundArgs := false
	for _, ev := range seen {
		if ev.Type == StreamToolUseDelta && ev.PartialJSON != "" {
			foundArgs = true
		}
	}
	if !foundArgs {
		t.Fatal("expected a tool_use_delta with partial json")
	}
}

// Back-compat: an agent that does NOT implement StreamingAgent delegates to
// Turn via TurnStreamingOrDelegate, ignoring the sink (no events).
func TestDefaultTurnStreamingIgnoresSink(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(textRespEndTurn("done"))
	// Wrap ModelAgent in a non-streaming Agent to exercise the delegate path.
	base := makeAgent(m)
	nonStreaming := nonStreamingAgent{base}

	count := 0
	sink := func(StreamEvent) { count++ }
	result := TurnStreamingOrDelegate(context.Background(), nonStreaming, userCtx("hi"), sink)
	if result.Kind != TurnFinalResponse || result.Content != "done" {
		t.Fatalf("unexpected result %+v", result)
	}
	if count != 0 {
		t.Fatalf("default delegate path must not emit events, got %d", count)
	}
}

// nonStreamingAgent wraps an Agent and exposes ONLY the Agent interface, so a
// StreamingAgent type assertion fails and TurnStreamingOrDelegate falls back to
// Turn.
type nonStreamingAgent struct{ inner Agent }

func (a nonStreamingAgent) Turn(ctx context.Context, c Context) TurnResult {
	return a.inner.Turn(ctx, c)
}
func (a nonStreamingAgent) ID() AgentID { return a.inner.ID() }

// Turn and TurnStreaming classify identically for the same response.
func TestTurnAndTurnStreamingClassifyIdentically(t *testing.T) {
	resp := ModelResponse{
		Content:    []ContentBlock{NewTextBlock("same")},
		Usage:      usage(2, 2),
		StopReason: StopEndTurn,
	}
	blocking := replayAgentFor(resp)
	rBlock := blocking.Turn(context.Background(), userCtx("x"))

	streaming := replayAgentFor(resp)
	rStream := streaming.TurnStreaming(context.Background(), userCtx("x"), func(StreamEvent) {})

	bb, _ := json.Marshal(rBlock)
	sb, _ := json.Marshal(rStream)
	if string(bb) != string(sb) {
		t.Fatalf("classification diverged:\n blocking=%s\nstreaming=%s", bb, sb)
	}
}

// Ordered text deltas across multiple text blocks concatenate in index order.
func TestStreamAccumulatorOrderedConcatenation(t *testing.T) {
	agent := replayAgentFor(ModelResponse{
		Content: []ContentBlock{
			NewTextBlock("foo"),
			NewTextBlock("bar"),
		},
		Usage:      usage(1, 1),
		StopReason: StopEndTurn,
	})
	result := agent.TurnStreaming(context.Background(), userCtx("x"), func(StreamEvent) {})
	if result.Content != "foobar" {
		t.Fatalf("content = %q, want foobar", result.Content)
	}
}

// Serde back-compat: a pre-#103 TurnResult JSON (no reasoning field) still
// deserializes, with Reasoning defaulting to nil; and the existing variants
// round-trip including a populated reasoning.
func TestTurnResultReasoningSerde(t *testing.T) {
	// Pre-#103 JSON without reasoning.
	pre := `{"kind":"final_response","content":"hi","usage":{"input_tokens":1,"output_tokens":1}}`
	var back TurnResult
	if err := json.Unmarshal([]byte(pre), &back); err != nil {
		t.Fatalf("unmarshal pre-#103: %v", err)
	}
	if back.Kind != TurnFinalResponse || back.Reasoning != nil {
		t.Fatalf("pre-#103 must deserialize with nil reasoning, got %+v", back)
	}

	// Round-trip with populated reasoning.
	reason := "because"
	r := NewFinalResponseWithReasoning("hi", usage(1, 1), &reason)
	enc, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var rt TurnResult
	if err := json.Unmarshal(enc, &rt); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if rt.Reasoning == nil || *rt.Reasoning != "because" {
		t.Fatalf("reasoning did not round-trip: %s", enc)
	}

	// nil reasoning must omit the field on the wire.
	noReason := NewFinalResponse("hi", usage(1, 1))
	enc2, _ := json.Marshal(noReason)
	var probe map[string]json.RawMessage
	_ = json.Unmarshal(enc2, &probe)
	if _, present := probe["reasoning"]; present {
		t.Fatalf("nil reasoning must be omitted, got %s", enc2)
	}
}
