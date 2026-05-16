package sporecore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func fixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	// this = .../go/spore-core/model_replay_test.go
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "model_responses", "model_interface", "basic_text.jsonl")
}

func replayProvider() ProviderInfo {
	return ProviderInfo{Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000}
}

// Mirrors rust/crates/spore-core/tests/model_fixture_replay.rs.
func TestBasicTextFixtureReplaysInOrder(t *testing.T) {
	raw, err := os.ReadFile(fixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), replayProvider())
	if err != nil {
		t.Fatalf("ParseReplayJSONL: %v", err)
	}
	if got := replay.Remaining(); got != 3 {
		t.Fatalf("remaining = %d, want 3", got)
	}

	ctx := context.Background()

	r1, err := replay.Call(ctx, emptyRequest())
	if err != nil {
		t.Fatalf("r1: %v", err)
	}
	if r1.StopReason != StopEndTurn {
		t.Fatalf("r1 stop_reason = %q, want %q", r1.StopReason, StopEndTurn)
	}
	if r1.Usage.InputTokens != 8 || r1.Usage.OutputTokens != 11 {
		t.Fatalf("r1 usage = %+v", r1.Usage)
	}
	if len(r1.Content) != 1 || r1.Content[0].Type != ContentBlockTypeText ||
		r1.Content[0].Text != "Hello! How can I help you today?" {
		t.Fatalf("r1 content = %+v", r1.Content)
	}

	r2, err := replay.Call(ctx, emptyRequest())
	if err != nil {
		t.Fatalf("r2: %v", err)
	}
	if r2.Usage.InputTokens != 10 || r2.Usage.OutputTokens != 1 {
		t.Fatalf("r2 usage = %+v", r2.Usage)
	}

	r3, err := replay.Call(ctx, emptyRequest())
	if err != nil {
		t.Fatalf("r3: %v", err)
	}
	if r3.StopReason != StopToolUse {
		t.Fatalf("r3 stop_reason = %q, want %q", r3.StopReason, StopToolUse)
	}
	if len(r3.Content) != 1 || r3.Content[0].Type != ContentBlockTypeToolUse {
		t.Fatalf("r3 content = %+v", r3.Content)
	}
	tc := r3.Content[0].ToolCall
	if tc == nil || tc.Name != "echo" {
		t.Fatalf("r3 tool call = %+v", tc)
	}
	var input map[string]string
	if err := json.Unmarshal(tc.Input, &input); err != nil {
		t.Fatalf("r3 input unmarshal: %v", err)
	}
	if input["text"] != "hi" {
		t.Fatalf("r3 input[text] = %q, want hi", input["text"])
	}
}

func TestReplayExhaustionReturnsTypedError(t *testing.T) {
	replay := NewReplayModel(nil, replayProvider())
	_, err := replay.Call(context.Background(), emptyRequest())
	if err == nil {
		t.Fatal("expected error on exhausted replay")
	}
	var me *ModelError
	if !errors.As(err, &me) {
		t.Fatalf("expected *ModelError, got %T: %v", err, err)
	}
	if me.Kind != ModelErrProviderError {
		t.Fatalf("kind = %q, want %q", me.Kind, ModelErrProviderError)
	}
	if me.Code != 0 || me.Message == "" {
		t.Fatalf("provider error fields = %+v", me)
	}
}

func TestReplayStreamingSynthesisesEvents(t *testing.T) {
	raw, err := os.ReadFile(fixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), replayProvider())
	if err != nil {
		t.Fatal(err)
	}
	ch, err := replay.CallStreaming(context.Background(), emptyRequest())
	if err != nil {
		t.Fatal(err)
	}
	var (
		sawStart, sawStop, sawDelta, sawBlockStop bool
		finalUsage                                *TokenUsage
		finalStop                                 StopReason
	)
	for e := range ch {
		if e.Err != nil {
			t.Fatalf("stream err: %v", e.Err)
		}
		switch e.Event.Type {
		case StreamMessageStart:
			sawStart = true
		case StreamContentBlockDelta:
			sawDelta = true
			if e.Event.Delta != "Hello! How can I help you today?" {
				t.Fatalf("delta = %q", e.Event.Delta)
			}
		case StreamContentBlockStop:
			sawBlockStop = true
		case StreamMessageStop:
			sawStop = true
			finalUsage = e.Event.Usage
			finalStop = e.Event.StopReason
		}
	}
	if !sawStart || !sawStop || !sawDelta || !sawBlockStop {
		t.Fatalf("missing events: start=%v stop=%v delta=%v blockStop=%v",
			sawStart, sawStop, sawDelta, sawBlockStop)
	}
	if finalUsage == nil || finalUsage.InputTokens != 8 || finalUsage.OutputTokens != 11 {
		t.Fatalf("final usage = %+v", finalUsage)
	}
	if finalStop != StopEndTurn {
		t.Fatalf("final stop = %q", finalStop)
	}
}

func TestReplayCountTokensDeterministic(t *testing.T) {
	replay := NewReplayModel(nil, replayProvider())
	req := ModelRequest{
		Messages: []Message{{Role: RoleUser, Content: NewTextContent(strRepeat("a", 40))}},
	}
	n, err := replay.CountTokens(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if n != 10 {
		t.Fatalf("count = %d, want 10", n)
	}
}

func strRepeat(s string, n int) string {
	out := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		out = append(out, s...)
	}
	return string(out)
}
