package ollama

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// showWithThinking returns a /api/show handler that advertises the "thinking"
// capability.
func showWithThinking(w http.ResponseWriter, _ *http.Request) {
	_, _ = io.WriteString(w, `{"capabilities":["tools","thinking"]}`)
}

// showWithoutThinking returns a /api/show handler with NO "thinking" capability.
func showWithoutThinking(w http.ResponseWriter, _ *http.Request) {
	_, _ = io.WriteString(w, `{"capabilities":["tools"]}`)
}

// TestBuildRequestThinkFromReasoningEffort confirms reasoning_effort maps to a
// bare level string on the wire.
func TestBuildRequestThinkFromReasoningEffort(t *testing.T) {
	r := req(userMsg("hi"))
	eff := sporecore.ReasoningEffortHigh
	r.Params.ReasoningEffort = &eff
	body := buildRequest("qwen3", "", nil, r, false)
	if body.Think == nil {
		t.Fatal("think was nil")
	}
	out, _ := json.Marshal(body.Think)
	if string(out) != `"high"` {
		t.Fatalf("think = %s, want \"high\"", out)
	}
}

// TestBuildRequestThinkFromReasoningBudget confirms a positive reasoning_budget
// maps to a bare boolean true.
func TestBuildRequestThinkFromReasoningBudget(t *testing.T) {
	r := req(userMsg("hi"))
	b := uint32(2048)
	r.Params.ReasoningBudget = &b
	body := buildRequest("qwen3", "", nil, r, false)
	if body.Think == nil {
		t.Fatal("think was nil")
	}
	out, _ := json.Marshal(body.Think)
	if string(out) != `true` {
		t.Fatalf("think = %s, want true", out)
	}
}

// TestBuildRequestThinkOmittedByDefault confirms think is absent when neither
// reasoning_effort nor a positive reasoning_budget is set (byte-identical).
func TestBuildRequestThinkOmittedByDefault(t *testing.T) {
	body := buildRequest("llama3.2", "", nil, req(userMsg("hi")), false)
	if body.Think != nil {
		t.Fatalf("think should be nil by default: %v", body.Think)
	}
	out, _ := json.Marshal(body)
	if strings.Contains(string(out), "think") {
		t.Fatalf("think key leaked: %s", out)
	}
}

// TestGateThinkingDropsThinkForUnsupportedModel confirms gateThinking drops the
// think flag (and returns true) when the model lacks the "thinking" capability.
func TestGateThinkingDropsThinkForUnsupportedModel(t *testing.T) {
	c := New("llama3.2")
	c.checkMeta = modelMeta{capabilities: []string{"tools"}} // no "thinking"
	c.metaReady.Store(true)
	body := buildRequest("llama3.2", "", nil, req(userMsg("hi")), false)
	eff := sporecore.ReasoningEffortMedium
	body.Think = &ollamaThink{level: "medium"}
	_ = eff
	dropped := c.gateThinking(&body)
	if !dropped {
		t.Fatal("expected gateThinking to drop the think flag")
	}
	if body.Think != nil {
		t.Fatalf("think should be nil after gating: %v", body.Think)
	}
}

// TestGateThinkingKeepsThinkForSupportedModel confirms gateThinking keeps the
// think flag when the model advertises "thinking".
func TestGateThinkingKeepsThinkForSupportedModel(t *testing.T) {
	c := New("qwen3")
	c.checkMeta = modelMeta{capabilities: []string{"tools", "thinking"}}
	c.metaReady.Store(true)
	body := buildRequest("qwen3", "", nil, req(userMsg("hi")), false)
	eff := sporecore.ReasoningEffortHigh
	body.Think = &ollamaThink{level: "high"}
	_ = eff
	if c.gateThinking(&body) {
		t.Fatal("gateThinking should NOT drop think for a thinking-capable model")
	}
	if body.Think == nil {
		t.Fatal("think should be retained")
	}
}

// TestParseResponseThinkingBlockPrecedesText confirms a unary response with a
// thinking channel surfaces a Thinking block ahead of the Text block.
func TestParseResponseThinkingBlockPrecedesText(t *testing.T) {
	wr := wireResponse{
		Message: wireResponseMessage{
			Role:     "assistant",
			Content:  "the answer",
			Thinking: "let me reason about this",
		},
		Done: true,
	}
	resp := parseResponse(wr, false)
	if len(resp.Content) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(resp.Content))
	}
	if resp.Content[0].Type != sporecore.ContentBlockTypeThinking || resp.Content[0].Text != "let me reason about this" {
		t.Fatalf("block 0 = %+v, want Thinking", resp.Content[0])
	}
	if resp.Content[1].Type != sporecore.ContentBlockTypeText || resp.Content[1].Text != "the answer" {
		t.Fatalf("block 1 = %+v, want Text", resp.Content[1])
	}
}

// TestStreamingThinkingThenTextEmitsBlockStop confirms a streaming response
// with a thinking prefix emits ThinkingDelta on index 0, a ContentBlockStop on
// index 0, then text on index 1.
func TestStreamingThinkingThenTextEmitsBlockStop(t *testing.T) {
	chunks := []string{
		`{"message":{"thinking":"reasoning part 1"},"done":false}`,
		`{"message":{"thinking":"reasoning part 2"},"done":false}`,
		`{"message":{"content":"the answer"},"done":false}`,
		`{"done":true,"done_reason":"stop","eval_count":10,"prompt_eval_count":5}`,
	}
	srv := &splitHandler{model: "qwen3", show: showWithThinking, chat: func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/x-ndjson")
		for _, c := range chunks {
			_, _ = io.WriteString(w, c+"\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}}
	server := httptest.NewServer(srv)
	defer server.Close()
	c := WithBaseURL("qwen3", server.URL)
	eff := sporecore.ReasoningEffortHigh
	r := req(userMsg("hi"))
	r.Params.ReasoningEffort = &eff
	ch, err := c.CallStreaming(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	var events []sporecore.StreamEvent
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("stream err: %v", ev.Err)
		}
		events = append(events, ev.Event)
	}
	// Expect: MessageStart, ThinkingDelta(0), ThinkingDelta(0),
	// ContentBlockStop(0), ContentBlockDelta(1), MessageStop.
	var sawThinkingDelta, sawBlockStop0, sawTextDelta1 bool
	for _, e := range events {
		switch e.Type {
		case sporecore.StreamThinkingDelta:
			if e.Index != 0 {
				t.Errorf("ThinkingDelta index = %d, want 0", e.Index)
			}
			sawThinkingDelta = true
		case sporecore.StreamContentBlockStop:
			if e.Index == 0 {
				sawBlockStop0 = true
			}
		case sporecore.StreamContentBlockDelta:
			if e.Index == 1 && e.Delta == "the answer" {
				sawTextDelta1 = true
			}
		}
	}
	if !sawThinkingDelta {
		t.Error("expected a ThinkingDelta event on index 0")
	}
	if !sawBlockStop0 {
		t.Error("expected a ContentBlockStop on index 0 (closing the thinking block)")
	}
	if !sawTextDelta1 {
		t.Error("expected text on index 1 after the thinking block closed")
	}
}

// TestStreamingTextOnlyStaysOnIndexZero confirms that when there is NO thinking
// prefix, text stays on index 0 (byte-identical to pre-thinking behavior).
func TestStreamingTextOnlyStaysOnIndexZero(t *testing.T) {
	chunks := []string{
		`{"message":{"content":"hello"},"done":false}`,
		`{"done":true,"done_reason":"stop","eval_count":5,"prompt_eval_count":2}`,
	}
	srv := &splitHandler{model: "llama3.2", show: showWithoutThinking, chat: func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/x-ndjson")
		for _, c := range chunks {
			_, _ = io.WriteString(w, c+"\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}}
	server := httptest.NewServer(srv)
	defer server.Close()
	c := WithBaseURL("llama3.2", server.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var textIndex uint32 = 999
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("stream err: %v", ev.Err)
		}
		if ev.Event.Type == sporecore.StreamContentBlockDelta {
			textIndex = ev.Event.Index
		}
	}
	if textIndex != 0 {
		t.Errorf("text index = %d, want 0 (no thinking prefix)", textIndex)
	}
}
