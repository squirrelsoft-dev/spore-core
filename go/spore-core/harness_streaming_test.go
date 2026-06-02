package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// ── mapModelStreamEvent unit tests (issue #103) ─────────────────────────────

// Q3: model message_start / message_stop are dropped at the harness boundary.
func TestMapDropsMessageStartStop(t *testing.T) {
	state := newTurnStreamState()
	if got := mapModelStreamEvent(StreamEvent{Type: StreamMessageStart}, state); len(got) != 0 {
		t.Fatalf("message_start must be dropped, got %+v", got)
	}
	if got := mapModelStreamEvent(StreamEvent{Type: StreamMessageStop}, state); len(got) != 0 {
		t.Fatalf("message_stop must be dropped, got %+v", got)
	}
}

// Q2 + text deltas: the first text delta for an index opens a text block; a
// content_block_stop closes it with the matching index.
func TestMapTextDeltaBracketing(t *testing.T) {
	state := newTurnStreamState()
	out := mapModelStreamEvent(StreamEvent{Type: StreamContentBlockDelta, Index: 1, Delta: "hi"}, state)
	if len(out) != 2 {
		t.Fatalf("first text delta must emit block_start + text_delta, got %+v", out)
	}
	if out[0].Kind != HarnessStreamBlockStart || out[0].Index != 1 || out[0].Block != BlockText {
		t.Fatalf("block_start wrong: %+v", out[0])
	}
	if out[1].Kind != HarnessStreamTextDelta || out[1].Content != "hi" {
		t.Fatalf("text_delta wrong: %+v", out[1])
	}
	// Second delta for the same index does NOT re-open the block.
	out = mapModelStreamEvent(StreamEvent{Type: StreamContentBlockDelta, Index: 1, Delta: "!"}, state)
	if len(out) != 1 || out[0].Kind != HarnessStreamTextDelta {
		t.Fatalf("second delta must emit only text_delta, got %+v", out)
	}
	// Stop closes with the same index.
	out = mapModelStreamEvent(StreamEvent{Type: StreamContentBlockStop, Index: 1}, state)
	if len(out) != 1 || out[0].Kind != HarnessStreamBlockStop || out[0].Index != 1 {
		t.Fatalf("block_stop wrong: %+v", out)
	}
}

// Reasoning deltas open a reasoning block and map to reasoning_delta.
func TestMapReasoningDelta(t *testing.T) {
	state := newTurnStreamState()
	out := mapModelStreamEvent(StreamEvent{Type: StreamThinkingDelta, Index: 0, Delta: "think"}, state)
	if len(out) != 2 || out[0].Block != BlockReasoning || out[1].Kind != HarnessStreamReasoningDelta {
		t.Fatalf("reasoning mapping wrong: %+v", out)
	}
	if out[1].Content != "think" {
		t.Fatalf("reasoning content = %q", out[1].Content)
	}
}

// Tool lifecycle (normal path): tool_use_start carries the real id + name onto
// tool_call_start; subsequent tool_use_delta fragments correlate by that id.
func TestMapToolLifecycle(t *testing.T) {
	state := newTurnStreamState()
	out := mapModelStreamEvent(StreamEvent{Type: StreamToolUseStart, Index: 2, ID: "toolu_x", Name: "lookup"}, state)
	if len(out) != 2 {
		t.Fatalf("tool_use_start must emit block_start + tool_call_start, got %+v", out)
	}
	if out[0].Kind != HarnessStreamBlockStart || out[0].Block != BlockToolUse {
		t.Fatalf("block_start wrong: %+v", out[0])
	}
	if out[1].Kind != HarnessStreamToolCallStart || out[1].CallID != "toolu_x" || out[1].Name != "lookup" {
		t.Fatalf("tool_call_start must carry real id + name: %+v", out[1])
	}
	out = mapModelStreamEvent(StreamEvent{Type: StreamToolUseDelta, Index: 2, PartialJSON: `{"q":`}, state)
	if len(out) != 1 || out[0].Kind != HarnessStreamToolArgsDelta || out[0].CallID != "toolu_x" || out[0].PartialJSON != `{"q":` {
		t.Fatalf("tool_args_delta wrong: %+v", out)
	}
	// Subsequent fragment correlates by the same call_id without re-opening.
	out = mapModelStreamEvent(StreamEvent{Type: StreamToolUseDelta, Index: 2, PartialJSON: `"rust"}`}, state)
	if len(out) != 1 || out[0].Kind != HarnessStreamToolArgsDelta || out[0].CallID != "toolu_x" {
		t.Fatalf("second tool fragment wrong: %+v", out)
	}
}

// Tool lifecycle (fallback path): if a stream omits tool_use_start and opens the
// block on a tool_use_delta, the harness synthesizes call_{index} with an empty
// name so args still surface.
func TestMapToolLifecycleFallbackWithoutStart(t *testing.T) {
	state := newTurnStreamState()
	out := mapModelStreamEvent(StreamEvent{Type: StreamToolUseDelta, Index: 2, PartialJSON: `{"q":`}, state)
	if len(out) != 3 {
		t.Fatalf("first tool delta must emit block_start + tool_call_start + tool_args_delta, got %+v", out)
	}
	if out[1].Kind != HarnessStreamToolCallStart || out[1].CallID != "call_2" || out[1].Name != "" {
		t.Fatalf("fallback tool_call_start wrong (name empty, id call_2): %+v", out[1])
	}
	if out[2].Kind != HarnessStreamToolArgsDelta || out[2].CallID != "call_2" || out[2].PartialJSON != `{"q":` {
		t.Fatalf("tool_args_delta wrong: %+v", out[2])
	}
}

// Q5: the coarse tool_call event carries Args and tool_result carries Content
// over the wire, and pre-#103 (missing) values default to null / "".
func TestCoarseEventsCarryArgsAndContent(t *testing.T) {
	tc := HarnessStreamEvent{
		Kind: HarnessStreamToolCall, CallID: "c1", Name: "lookup",
		Args: json.RawMessage(`{"q":"rust"}`),
	}
	enc, err := json.Marshal(tc)
	if err != nil {
		t.Fatalf("marshal tool_call: %v", err)
	}
	var probe map[string]json.RawMessage
	_ = json.Unmarshal(enc, &probe)
	if string(probe["args"]) != `{"q":"rust"}` {
		t.Fatalf("args not carried: %s", enc)
	}

	// Missing args defaults to JSON null.
	enc2, _ := json.Marshal(HarnessStreamEvent{Kind: HarnessStreamToolCall, CallID: "c1", Name: "x"})
	_ = json.Unmarshal(enc2, &probe)
	if string(probe["args"]) != "null" {
		t.Fatalf("missing args must default to null, got %s", enc2)
	}

	tr := HarnessStreamEvent{
		Kind: HarnessStreamToolResult, CallID: "c1", IsError: false, ResultContent: "ok",
	}
	enc3, _ := json.Marshal(tr)
	_ = json.Unmarshal(enc3, &probe)
	if string(probe["content"]) != `"ok"` {
		t.Fatalf("tool_result content not carried: %s", enc3)
	}
}

// ── Fixture-replay ordering test (issue #103) ───────────────────────────────

func streamingTurnFixturePath(t *testing.T, parts ...string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	base := []string{dir, "..", ".."}
	return filepath.Join(append(base, parts...)...)
}

// Replays fixtures/model_responses/harness/streaming_turn.jsonl through the
// ReplayModel + harness streaming turn and asserts the emitted DELTA/frame
// StreamEvents match fixtures/harness/streaming_events.json exactly (in order).
// Coarse tool_call / tool_result events are emitted later by the loop and are
// not part of the golden, so the test filters to the delta/frame subset.
func TestStreamingTurnFixtureOrdering(t *testing.T) {
	jsonlPath := streamingTurnFixturePath(t, "fixtures", "model_responses", "harness", "streaming_turn.jsonl")
	raw, err := os.ReadFile(jsonlPath)
	if err != nil {
		t.Fatalf("read jsonl fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	agent := NewModelAgent(AgentID("stream-fixture"), replay)

	// Drive a single streaming turn directly, mapping deltas exactly as the
	// harness loop does (runStreamingTurn).
	state := newTurnStreamState()
	var emitted []HarnessStreamEvent
	sink := func(ev StreamEvent) {
		emitted = append(emitted, mapModelStreamEvent(ev, state)...)
	}
	c := Context{Messages: []Message{{Role: RoleUser, Content: NewTextContent("look up rust and explain")}}}
	result := agent.TurnStreaming(context.Background(), c, sink)
	if result.Kind != TurnToolCallRequested {
		t.Fatalf("expected ToolCallRequested, got %v", result.Kind)
	}

	// Load the golden ordering.
	goldenPath := streamingTurnFixturePath(t, "fixtures", "harness", "streaming_events.json")
	goldRaw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var golden struct {
		Events []json.RawMessage `json:"events"`
	}
	if err := json.Unmarshal(goldRaw, &golden); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	if len(emitted) != len(golden.Events) {
		t.Fatalf("event count = %d, want %d\nemitted=%s", len(emitted), len(golden.Events), dumpEvents(emitted))
	}
	for i, ev := range emitted {
		got, err := json.Marshal(ev)
		if err != nil {
			t.Fatalf("marshal emitted[%d]: %v", i, err)
		}
		if !jsonEqual(t, json.RawMessage(got), golden.Events[i]) {
			t.Fatalf("event[%d] mismatch:\n got=%s\nwant=%s", i, got, golden.Events[i])
		}
	}
}

func dumpEvents(evs []HarnessStreamEvent) string {
	b, _ := json.Marshal(evs)
	return string(b)
}
