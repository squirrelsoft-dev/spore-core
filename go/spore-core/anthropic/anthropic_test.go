package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func userMsg(text string) sporecore.Message {
	return sporecore.Message{Role: sporecore.RoleUser, Content: sporecore.NewTextContent(text)}
}

func sysMsg(text string) sporecore.Message {
	return sporecore.Message{Role: sporecore.RoleSystem, Content: sporecore.NewTextContent(text)}
}

func req(msgs ...sporecore.Message) sporecore.ModelRequest {
	return sporecore.ModelRequest{Messages: msgs}
}

// ---------------------------------------------------------------------------
// buildRequest
// ---------------------------------------------------------------------------

func TestBuildRequestExtractsSystemMessage(t *testing.T) {
	body := buildRequest("claude-sonnet-4-6", req(sysMsg("be helpful"), userMsg("hi")), false)
	if body.System != "be helpful" {
		t.Fatalf("system = %q", body.System)
	}
	if len(body.Messages) != 1 || body.Messages[0].Role != "user" {
		t.Fatalf("messages: %+v", body.Messages)
	}
}

func TestBuildRequestJoinsMultipleSystemMessages(t *testing.T) {
	body := buildRequest("claude-sonnet-4-6", req(sysMsg("first"), sysMsg("second"), userMsg("hi")), false)
	if body.System != "first\n\nsecond" {
		t.Fatalf("system = %q", body.System)
	}
}

func TestBuildRequestDefaultsMaxTokens(t *testing.T) {
	body := buildRequest("claude-sonnet-4-6", req(userMsg("hi")), false)
	if body.MaxTokens != 4096 {
		t.Fatalf("max_tokens = %d", body.MaxTokens)
	}
}

func TestBuildRequestRespectsMaxTokens(t *testing.T) {
	r := req(userMsg("hi"))
	mt := uint32(256)
	r.Params.MaxTokens = &mt
	body := buildRequest("claude-sonnet-4-6", r, false)
	if body.MaxTokens != 256 {
		t.Fatalf("max_tokens = %d", body.MaxTokens)
	}
}

func TestBuildRequestMapsToolCallMessage(t *testing.T) {
	r := req(sporecore.Message{
		Role: sporecore.RoleAssistant,
		Content: sporecore.NewToolCallContent(sporecore.ToolCall{
			ID:    "call-1",
			Name:  "fetch",
			Input: json.RawMessage(`{"url":"x"}`),
		}),
	})
	body := buildRequest("claude-sonnet-4-6", r, false)
	out, _ := json.Marshal(body)
	s := string(out)
	if !strings.Contains(s, `"type":"tool_use"`) || !strings.Contains(s, `"id":"call-1"`) {
		t.Fatalf("wire: %s", s)
	}
}

func TestBuildRequestMapsToolResultToUserRole(t *testing.T) {
	r := req(sporecore.Message{
		Role: sporecore.RoleTool,
		Content: sporecore.NewToolResultContent(sporecore.ToolResult{
			ToolUseID: "call-1",
			Content:   "ok",
		}),
	})
	body := buildRequest("claude-sonnet-4-6", r, false)
	if body.Messages[0].Role != "user" {
		t.Fatalf("role = %s", body.Messages[0].Role)
	}
	out, _ := json.Marshal(body.Messages[0].Content)
	if !strings.Contains(string(out), `"type":"tool_result"`) {
		t.Fatalf("wire: %s", out)
	}
}

// ---------------------------------------------------------------------------
// parseResponse / parseStopReason
// ---------------------------------------------------------------------------

func TestParseResponseExtractsTextAndUsage(t *testing.T) {
	var body wireResponse
	raw := `{"content":[{"type":"text","text":"hi there"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":2}}`
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if len(r.Content) != 1 || r.Content[0].Text != "hi there" {
		t.Fatalf("content: %+v", r.Content)
	}
	if r.Usage.InputTokens != 4 || r.Usage.OutputTokens != 2 {
		t.Fatalf("usage: %+v", r.Usage)
	}
	if r.StopReason != sporecore.StopEndTurn {
		t.Fatalf("stop: %s", r.StopReason)
	}
}

func TestParseResponseExtractsToolUse(t *testing.T) {
	raw := `{"content":[{"type":"tool_use","id":"c1","name":"search","input":{"q":"go"}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.Content[0].Type != sporecore.ContentBlockTypeToolUse || r.Content[0].ToolCall == nil {
		t.Fatalf("content: %+v", r.Content)
	}
	if r.Content[0].ToolCall.ID != "c1" || r.Content[0].ToolCall.Name != "search" {
		t.Fatalf("tool: %+v", r.Content[0].ToolCall)
	}
	if r.StopReason != sporecore.StopToolUse {
		t.Fatalf("stop: %s", r.StopReason)
	}
}

func TestParseResponseExtractsThinkingBlock(t *testing.T) {
	raw := `{"content":[{"type":"thinking","thinking":"reasoning"},{"type":"text","text":"answer"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.Content[0].Type != sporecore.ContentBlockTypeThinking {
		t.Fatalf("first block: %+v", r.Content[0])
	}
	if r.Content[1].Type != sporecore.ContentBlockTypeText {
		t.Fatalf("second block: %+v", r.Content[1])
	}
}

func TestParseResponseExtractsCacheUsage(t *testing.T) {
	raw := `{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":100,"output_tokens":2,"cache_read_input_tokens":50,"cache_creation_input_tokens":30}}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.Usage.CacheReadTokens == nil || *r.Usage.CacheReadTokens != 50 {
		t.Fatalf("cache_read = %+v", r.Usage.CacheReadTokens)
	}
	if r.Usage.CacheWriteTokens == nil || *r.Usage.CacheWriteTokens != 30 {
		t.Fatalf("cache_write = %+v", r.Usage.CacheWriteTokens)
	}
}

func TestStopReasonMapping(t *testing.T) {
	cases := map[string]sporecore.StopReason{
		"end_turn":      sporecore.StopEndTurn,
		"tool_use":      sporecore.StopToolUse,
		"max_tokens":    sporecore.StopMaxTokens,
		"stop_sequence": sporecore.StopStopSequence,
		"???":           sporecore.StopEndTurn,
	}
	for in, want := range cases {
		s := in
		got := parseStopReason(&s)
		if got != want {
			t.Fatalf("%q -> %s (want %s)", in, got, want)
		}
	}
	if parseStopReason(nil) != sporecore.StopEndTurn {
		t.Fatalf("nil -> not EndTurn")
	}
}

// ---------------------------------------------------------------------------
// Backoff / context window / provider info
// ---------------------------------------------------------------------------

func TestBackoffGrowsThenCaps(t *testing.T) {
	d0 := backoffDelay(0)
	d3 := backoffDelay(3)
	dmax := backoffDelay(20)
	if d3 <= d0 {
		t.Fatalf("d3=%s d0=%s", d3, d0)
	}
	if dmax > 30*time.Second {
		t.Fatalf("dmax=%s", dmax)
	}
	if d0 != 500*time.Millisecond {
		t.Fatalf("d0=%s", d0)
	}
}

func TestContextWindowKnownAndUnknown(t *testing.T) {
	if ContextWindow("claude-sonnet-4-6") != 200_000 {
		t.Fatalf("sonnet")
	}
	if ContextWindow("claude-opus-4-7") != 200_000 {
		t.Fatalf("opus-4-7")
	}
	if ContextWindow("claude-imaginary-9") != 200_000 {
		t.Fatalf("imaginary")
	}
	if ContextWindow("gpt-4o") != 0 {
		t.Fatalf("foreign")
	}
}

func TestProviderInfoUsesModelID(t *testing.T) {
	c := New("test-key", "claude-sonnet-4-6")
	p := c.Provider()
	if p.Name != "anthropic" || p.ModelID != "claude-sonnet-4-6" || p.ContextWindow != 200_000 {
		t.Fatalf("provider: %+v", p)
	}
}

func TestStringRedactsAPIKey(t *testing.T) {
	c := New("super-secret-key-xyz", "claude-sonnet-4-6")
	s := c.String()
	if strings.Contains(s, "super-secret-key-xyz") {
		t.Fatalf("api key leaked: %s", s)
	}
	if !strings.Contains(s, "<redacted>") {
		t.Fatalf("missing redaction marker: %s", s)
	}
}

func TestFromEnvErrorsWhenUnset(t *testing.T) {
	const name = "__SPORE_TEST_ANTHROPIC_KEY_UNSET__"
	_ = os.Unsetenv(name)
	_, err := FromEnv(name, "claude-sonnet-4-6")
	if err == nil {
		t.Fatalf("expected error")
	}
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// retry-after parsing
// ---------------------------------------------------------------------------

func TestParseRetryAfterSeconds(t *testing.T) {
	d := parseRetryAfter("7")
	if d != 7*time.Second {
		t.Fatalf("d=%s", d)
	}
}

func TestParseRetryAfterEmpty(t *testing.T) {
	if parseRetryAfter("") != 0 {
		t.Fatalf("nonzero")
	}
}

// ---------------------------------------------------------------------------
// End-to-end via httptest
// ---------------------------------------------------------------------------

func TestCallAgainstMockReturnsResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Fatalf("api key header missing: %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != AnthropicVersion {
			t.Fatalf("version header: %s", r.Header.Get("anthropic-version"))
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg","type":"message","role":"assistant","content":[{"type":"text","text":"hello there"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`)
	}))
	defer srv.Close()
	c := New("test-key", "claude-sonnet-4-6").WithBaseURL(srv.URL)
	r, err := c.Call(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	if r.Content[0].Text != "hello there" {
		t.Fatalf("content: %+v", r.Content)
	}
	if r.Usage.InputTokens != 5 {
		t.Fatalf("usage: %+v", r.Usage)
	}
}

func TestCallMaps429ToRateLimited(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(429)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL).WithMaxRetries(0)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrRateLimited {
		t.Fatalf("expected RateLimited, got %v", err)
	}
	if merr.RetryAfter == nil || *merr.RetryAfter != 7*time.Second {
		t.Fatalf("retry_after: %v", merr.RetryAfter)
	}
}

func TestCallMaps529ToRateLimitedNoRetryAfter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(529)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL).WithMaxRetries(0)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrRateLimited {
		t.Fatalf("expected RateLimited, got %v", err)
	}
	if merr.RetryAfter != nil {
		t.Fatalf("retry_after should be nil: %v", merr.RetryAfter)
	}
}

func TestCallMaps400ToProviderErrorWithAnthropicMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens must be > 0"}}`)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL).WithMaxRetries(0)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 400 || !strings.Contains(merr.Message, "max_tokens") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestCallRetries429ThenSucceeds(t *testing.T) {
	var hits int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt64(&hits, 1)
		if n == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(429)
			return
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"content":[{"type":"text","text":"after retry"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL)
	r, err := c.Call(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	if r.Content[0].Text != "after retry" {
		t.Fatalf("content: %+v", r.Content)
	}
	if atomic.LoadInt64(&hits) != 2 {
		t.Fatalf("hits = %d", hits)
	}
}

func TestCountTokensUsesRealEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages/count_tokens" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"input_tokens":42}`)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL)
	n, err := c.CountTokens(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	if n != 42 {
		t.Fatalf("n = %d", n)
	}
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

func TestStreamingEmitsTextDeltaThenStop(t *testing.T) {
	sse := strings.Join([]string{
		"event: message_start",
		`data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}`,
		"",
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		"",
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"text_delta","text":" world"}}`,
		"",
		"event: content_block_stop",
		`data: {"index":0}`,
		"",
		"event: message_delta",
		`data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var kinds []sporecore.StreamEventType
	var last sporecore.StreamEvent
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("stream err: %v", ev.Err)
		}
		kinds = append(kinds, ev.Event.Type)
		last = ev.Event
	}
	expect := []sporecore.StreamEventType{
		sporecore.StreamMessageStart,
		sporecore.StreamContentBlockDelta,
		sporecore.StreamContentBlockDelta,
		sporecore.StreamContentBlockStop,
		sporecore.StreamMessageStop,
	}
	if fmt.Sprintf("%v", kinds) != fmt.Sprintf("%v", expect) {
		t.Fatalf("kinds = %v", kinds)
	}
	if last.Type != sporecore.StreamMessageStop || last.Usage == nil {
		t.Fatalf("last = %+v", last)
	}
	if last.Usage.InputTokens != 3 || last.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", last.Usage)
	}
	if last.StopReason != sporecore.StopEndTurn {
		t.Fatalf("stop = %s", last.StopReason)
	}
}

func TestStreamingEmitsToolUseAndThinkingDeltas(t *testing.T) {
	sse := strings.Join([]string{
		"event: content_block_delta",
		`data: {"index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}`,
		"",
		"event: content_block_start",
		`data: {"index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"lookup"}}`,
		"",
		"event: content_block_delta",
		`data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\""}}`,
		"",
		"event: message_stop",
		`data: {"type":"message_stop"}`,
		"",
	}, "\n")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()
	c := New("k", "claude-sonnet-4-6").WithBaseURL(srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var thinking, toolstart, tooluse, stop bool
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("err: %v", ev.Err)
		}
		switch ev.Event.Type {
		case sporecore.StreamThinkingDelta:
			thinking = true
			if ev.Event.Delta != "hmm" {
				t.Fatalf("thinking delta: %q", ev.Event.Delta)
			}
		case sporecore.StreamToolUseStart:
			toolstart = true
			if ev.Event.ID != "toolu_a" || ev.Event.Name != "lookup" {
				t.Fatalf("tool_use_start id=%q name=%q", ev.Event.ID, ev.Event.Name)
			}
		case sporecore.StreamToolUseDelta:
			tooluse = true
			if ev.Event.PartialJSON != `{"a"` {
				t.Fatalf("partial_json: %q", ev.Event.PartialJSON)
			}
		case sporecore.StreamMessageStop:
			stop = true
		}
	}
	if !thinking || !toolstart || !tooluse || !stop {
		t.Fatalf("got thinking=%v toolstart=%v tooluse=%v stop=%v", thinking, toolstart, tooluse, stop)
	}
}

// ---------------------------------------------------------------------------
// Live API integration tests (gated)
// ---------------------------------------------------------------------------

func skipUnlessLive(t *testing.T) string {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	return key
}

func TestLiveCallReturnsResponse(t *testing.T) {
	_ = skipUnlessLive(t)
	c, err := FromEnv("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
	if err != nil {
		t.Fatal(err)
	}
	r, err := c.Call(context.Background(), req(userMsg("Reply with the word 'pong'.")))
	if err != nil {
		t.Fatal(err)
	}
	if r.Usage.InputTokens == 0 || r.Usage.OutputTokens == 0 {
		t.Fatalf("usage: %+v", r.Usage)
	}
}

func TestLiveCountTokensIsNonzero(t *testing.T) {
	_ = skipUnlessLive(t)
	c, err := FromEnv("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
	if err != nil {
		t.Fatal(err)
	}
	n, err := c.CountTokens(context.Background(), req(userMsg("count my tokens please")))
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatalf("n = 0")
	}
}

func TestLiveStreamingEmitsEvents(t *testing.T) {
	_ = skipUnlessLive(t)
	c, err := FromEnv("ANTHROPIC_API_KEY", "claude-sonnet-4-6")
	if err != nil {
		t.Fatal(err)
	}
	ch, err := c.CallStreaming(context.Background(), req(userMsg("Reply with the word 'pong'.")))
	if err != nil {
		t.Fatal(err)
	}
	var sawStop bool
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("stream err: %v", ev.Err)
		}
		if ev.Event.Type == sporecore.StreamMessageStop {
			sawStop = true
		}
	}
	if !sawStop {
		t.Fatalf("no MessageStop")
	}
}
