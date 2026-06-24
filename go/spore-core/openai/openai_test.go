package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
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

func TestBuildRequestKeepsSystemInMessages(t *testing.T) {
	body := buildRequest("gpt-4o", req(sysMsg("be helpful"), userMsg("hi")), false)
	if len(body.Messages) != 2 {
		t.Fatalf("len = %d", len(body.Messages))
	}
	if body.Messages[0].Role != "system" || body.Messages[1].Role != "user" {
		t.Fatalf("roles: %+v", body.Messages)
	}
}

func TestBuildRequestSetsMaxTokensForChatModels(t *testing.T) {
	r := req(userMsg("hi"))
	mt := uint32(256)
	r.Params.MaxTokens = &mt
	body := buildRequest("gpt-4o", r, false)
	if body.MaxTokens == nil || *body.MaxTokens != 256 {
		t.Fatalf("max_tokens: %v", body.MaxTokens)
	}
	if body.MaxCompletionTokens != nil {
		t.Fatalf("max_completion_tokens: %v", body.MaxCompletionTokens)
	}
}

func TestBuildRequestOSeriesUsesMaxCompletionTokensAndNoTemperature(t *testing.T) {
	r := req(userMsg("hi"))
	mt := uint32(512)
	temp := float32(0.7)
	r.Params.MaxTokens = &mt
	r.Params.Temperature = &temp
	body := buildRequest("o3", r, false)
	if body.MaxTokens != nil {
		t.Fatalf("max_tokens should be nil for o3: %v", body.MaxTokens)
	}
	if body.MaxCompletionTokens == nil || *body.MaxCompletionTokens != 512 {
		t.Fatalf("max_completion_tokens: %v", body.MaxCompletionTokens)
	}
	if body.Temperature != nil {
		t.Fatalf("temperature should be nil for o3: %v", body.Temperature)
	}
}

func TestIsReasoningModel(t *testing.T) {
	cases := map[string]bool{
		"o4-mini": true,
		"o3":      true,
		"o1-pro":  true,
		"gpt-4o":  false,
		"gpt-4.1": false,
	}
	for id, want := range cases {
		if IsReasoningModel(id) != want {
			t.Fatalf("IsReasoningModel(%q) = %v, want %v", id, !want, want)
		}
	}
}

func TestBuildRequestMapsToolCallToAssistantToolCalls(t *testing.T) {
	r := req(sporecore.Message{
		Role: sporecore.RoleAssistant,
		Content: sporecore.NewToolCallContent(sporecore.ToolCall{
			ID:    "call-1",
			Name:  "fetch",
			Input: json.RawMessage(`{"url":"x"}`),
		}),
	})
	body := buildRequest("gpt-4o", r, false)
	if body.Messages[0].Role != "assistant" {
		t.Fatalf("role: %s", body.Messages[0].Role)
	}
	if len(body.Messages[0].ToolCalls) != 1 {
		t.Fatalf("tool_calls: %+v", body.Messages[0].ToolCalls)
	}
	tc := body.Messages[0].ToolCalls[0]
	if tc.ID != "call-1" || tc.Function.Name != "fetch" {
		t.Fatalf("tc: %+v", tc)
	}
	// arguments must be a JSON-encoded string, not a nested object
	out, _ := json.Marshal(body.Messages[0])
	s := string(out)
	if !strings.Contains(s, `"arguments":"{`) {
		t.Fatalf("arguments not stringified: %s", s)
	}
}

func TestBuildRequestMapsToolResultToToolRoleMessage(t *testing.T) {
	r := req(sporecore.Message{
		Role: sporecore.RoleTool,
		Content: sporecore.NewToolResultContent(sporecore.ToolResult{
			ToolUseID: "call-1",
			Content:   "ok",
		}),
	})
	body := buildRequest("gpt-4o", r, false)
	if body.Messages[0].Role != "tool" {
		t.Fatalf("role: %s", body.Messages[0].Role)
	}
	if body.Messages[0].ToolCallID != "call-1" {
		t.Fatalf("tool_call_id: %s", body.Messages[0].ToolCallID)
	}
	if body.Messages[0].Content == nil || *body.Messages[0].Content != "ok" {
		t.Fatalf("content: %v", body.Messages[0].Content)
	}
}

func TestBuildRequestStreamingSetsIncludeUsage(t *testing.T) {
	body := buildRequest("gpt-4o", req(userMsg("hi")), true)
	if !body.Stream {
		t.Fatalf("stream: false")
	}
	if body.StreamOptions == nil || !body.StreamOptions.IncludeUsage {
		t.Fatalf("stream_options: %+v", body.StreamOptions)
	}
}

// ---------------------------------------------------------------------------
// parseResponse / parseStopReason
// ---------------------------------------------------------------------------

func TestParseResponseExtractsTextAndUsage(t *testing.T) {
	raw := `{"choices":[{"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}`
	var body wireResponse
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

func TestParseResponseExtractsToolCalls(t *testing.T) {
	raw := `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"search","arguments":"{\"q\":\"go\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.StopReason != sporecore.StopToolUse {
		t.Fatalf("stop: %s", r.StopReason)
	}
	if r.Content[0].Type != sporecore.ContentBlockTypeToolUse || r.Content[0].ToolCall == nil {
		t.Fatalf("block: %+v", r.Content[0])
	}
	tc := r.Content[0].ToolCall
	if tc.ID != "c1" || tc.Name != "search" {
		t.Fatalf("tc: %+v", tc)
	}
	var parsed map[string]any
	if err := json.Unmarshal(tc.Input, &parsed); err != nil {
		t.Fatalf("input not JSON: %s", tc.Input)
	}
	if parsed["q"] != "go" {
		t.Fatalf("input: %v", parsed)
	}
}

func TestParseResponseExtractsReasoningAsThinking(t *testing.T) {
	raw := `{"choices":[{"message":{"role":"assistant","reasoning":"let me think","content":"the answer is 4"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.Content[0].Type != sporecore.ContentBlockTypeThinking {
		t.Fatalf("first: %+v", r.Content[0])
	}
	if r.Content[1].Type != sporecore.ContentBlockTypeText {
		t.Fatalf("second: %+v", r.Content[1])
	}
}

func TestParseResponseExtractsCacheReadOnly(t *testing.T) {
	raw := `{"choices":[{"message":{"role":"assistant","content":"x"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":50}}}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.Usage.CacheReadTokens == nil || *r.Usage.CacheReadTokens != 50 {
		t.Fatalf("cache_read: %v", r.Usage.CacheReadTokens)
	}
	if r.Usage.CacheWriteTokens != nil {
		t.Fatalf("cache_write should be nil: %v", r.Usage.CacheWriteTokens)
	}
}

func TestStopReasonMapping(t *testing.T) {
	cases := map[string]sporecore.StopReason{
		"stop":          sporecore.StopEndTurn,
		"tool_calls":    sporecore.StopToolUse,
		"function_call": sporecore.StopToolUse,
		"length":        sporecore.StopMaxTokens,
		"???":           sporecore.StopEndTurn,
	}
	for in, want := range cases {
		s := in
		if got := parseStopReason(&s); got != want {
			t.Fatalf("%q -> %s (want %s)", in, got, want)
		}
	}
	if parseStopReason(nil) != sporecore.StopEndTurn {
		t.Fatalf("nil -> not EndTurn")
	}
}

// ---------------------------------------------------------------------------
// Context window / provider info / from_env / String
// ---------------------------------------------------------------------------

func TestContextWindowKnownAndUnknown(t *testing.T) {
	cases := map[string]uint32{
		"gpt-4o":      128_000,
		"gpt-4o-mini": 128_000,
		"gpt-4.1":     1_000_000,
		"o3":          200_000,
		"o4-mini":     200_000,
		"o1-pro":      128_000,
		"claude-x":    0,
	}
	for id, want := range cases {
		if got := ContextWindow(id); got != want {
			t.Fatalf("ContextWindow(%q) = %d, want %d", id, got, want)
		}
	}
}

func TestProviderInfoUsesModelID(t *testing.T) {
	c := New("test-key", "gpt-4o")
	p := c.Provider()
	if p.Name != "openai" || p.ModelID != "gpt-4o" || p.ContextWindow != 128_000 {
		t.Fatalf("provider: %+v", p)
	}
}

func TestStringRedactsAPIKey(t *testing.T) {
	c := New("super-secret-key-xyz", "gpt-4o")
	s := c.String()
	if strings.Contains(s, "super-secret-key-xyz") {
		t.Fatalf("api key leaked: %s", s)
	}
	if !strings.Contains(s, "<redacted>") {
		t.Fatalf("missing redaction marker: %s", s)
	}
}

func TestFromEnvErrorsWhenUnset(t *testing.T) {
	const name = "__SPORE_TEST_OPENAI_KEY_UNSET__"
	_ = os.Unsetenv(name)
	_, err := FromEnv(name, "gpt-4o")
	if err == nil {
		t.Fatalf("expected error")
	}
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
}

// ---------------------------------------------------------------------------
// End-to-end via httptest
// ---------------------------------------------------------------------------

func TestCallAgainstMockReturnsResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("authorization header: %s", r.Header.Get("Authorization"))
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"hello there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`)
	}))
	defer srv.Close()
	c := New("test-key", "gpt-4o").WithBaseURL(srv.URL)
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
	c := New("k", "gpt-4o").WithBaseURL(srv.URL).WithMaxRetries(0)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrRateLimited {
		t.Fatalf("expected RateLimited, got %v", err)
	}
	if merr.RetryAfter == nil || *merr.RetryAfter != 7*time.Second {
		t.Fatalf("retry_after: %v", merr.RetryAfter)
	}
}

func TestCallMaps400ToProviderErrorWithOpenAIMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","message":"max_tokens must be > 0"}}`)
	}))
	defer srv.Close()
	c := New("k", "gpt-4o").WithBaseURL(srv.URL).WithMaxRetries(0)
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
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"after retry"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
	}))
	defer srv.Close()
	c := New("k", "gpt-4o").WithBaseURL(srv.URL)
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

func TestCountTokensUsesBytesOver4Heuristic(t *testing.T) {
	c := New("k", "gpt-4o")
	n, err := c.CountTokens(context.Background(), req(userMsg(strings.Repeat("a", 40))))
	if err != nil {
		t.Fatal(err)
	}
	if n != 10 {
		t.Fatalf("n = %d, want 10", n)
	}
}

// ---------------------------------------------------------------------------
// SSE streaming
// ---------------------------------------------------------------------------

func TestStreamingEmitsTextDeltaThenStop(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":5}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()
	c := New("k", "gpt-4o").WithBaseURL(srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var (
		texts []string
		last  sporecore.StreamEvent
		start bool
	)
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("err: %v", ev.Err)
		}
		switch ev.Event.Type {
		case sporecore.StreamMessageStart:
			start = true
		case sporecore.StreamContentBlockDelta:
			texts = append(texts, ev.Event.Delta)
		}
		last = ev.Event
	}
	if !start {
		t.Fatalf("no MessageStart")
	}
	if strings.Join(texts, "") != "hello world" {
		t.Fatalf("texts: %v", texts)
	}
	if last.Type != sporecore.StreamMessageStop {
		t.Fatalf("last: %+v", last)
	}
	if last.Usage == nil || last.Usage.InputTokens != 3 || last.Usage.OutputTokens != 5 {
		t.Fatalf("usage: %+v", last.Usage)
	}
	if last.StopReason != sporecore.StopEndTurn {
		t.Fatalf("stop: %s", last.StopReason)
	}
}

func TestStreamingAccumulatesToolCallDeltas(t *testing.T) {
	// Three partial chunks for the same tool call (index=0): the first carries
	// id+name; subsequent chunks carry incremental arguments fragments.
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"fetch","arguments":"{\"u"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"rl\":\"x\""}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()
	c := New("k", "gpt-4o").WithBaseURL(srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var fragments []string
	var finalStop sporecore.StopReason
	var startID, startName string
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("err: %v", ev.Err)
		}
		switch ev.Event.Type {
		case sporecore.StreamToolUseStart:
			startID = ev.Event.ID
			startName = ev.Event.Name
		case sporecore.StreamToolUseDelta:
			fragments = append(fragments, ev.Event.PartialJSON)
		case sporecore.StreamMessageStop:
			finalStop = ev.Event.StopReason
		}
	}
	if startID != "call-1" || startName != "fetch" {
		t.Fatalf("tool_use_start id=%q name=%q, want call-1/fetch", startID, startName)
	}
	joined := strings.Join(fragments, "")
	var parsed map[string]any
	if err := json.Unmarshal([]byte(joined), &parsed); err != nil {
		t.Fatalf("joined not JSON: %q (err=%v)", joined, err)
	}
	if parsed["url"] != "x" {
		t.Fatalf("parsed: %v", parsed)
	}
	if finalStop != sporecore.StopToolUse {
		t.Fatalf("stop: %s", finalStop)
	}
}

// ---------------------------------------------------------------------------
// Fixture replay round-trip
// ---------------------------------------------------------------------------

func TestFixtureReplayRoundTrip(t *testing.T) {
	_, this, _, _ := runtime.Caller(0)
	dir := filepath.Dir(this)
	path := filepath.Join(dir, "..", "..", "..", "fixtures", "model_responses", "model_interface", "openai_basic_text.jsonl")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := sporecore.ParseReplayJSONL(string(raw), sporecore.ProviderInfo{
		Name: "openai", ModelID: "gpt-4o-mini", ContextWindow: 128_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	r, err := replay.Call(context.Background(), req(userMsg("hello")))
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if len(r.Content) == 0 || r.Content[0].Text != "Hello! How can I help you today?" {
		t.Fatalf("content: %+v", r.Content)
	}
	if r.Usage.InputTokens != 8 || r.Usage.OutputTokens != 11 {
		t.Fatalf("usage: %+v", r.Usage)
	}
	if r.StopReason != sporecore.StopEndTurn {
		t.Fatalf("stop: %s", r.StopReason)
	}
}

// SC-3: a connection dropped mid-stream surfaces as the typed, retryable
// StreamInterrupted variant — a consumer drives its retry off Retryable(), not
// a substring match on the error text. A raw TCP server promises a 200-byte
// body (Content-Length) but sends a few bytes then closes the socket, so the
// client's body stream errors mid-read.
func TestStreamingInterruptionIsTypedAndRetryable(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()

	done := make(chan struct{})
	go func() {
		defer close(done)
		conn, aerr := ln.Accept()
		if aerr != nil {
			return
		}
		// Drain the request headers so the client's write completes.
		buf := make([]byte, 2048)
		_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		_, _ = conn.Read(buf)
		// 200 OK so CallStreaming returns Ok (headers arrived), then promise 200
		// body bytes but deliver only a partial SSE line and drop the socket —
		// EOF before Content-Length errors the body stream mid-read.
		_, _ = io.WriteString(conn,
			"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 200\r\n\r\ndata: partial")
		_ = conn.Close() // closes mid-body
	}()

	c := New("k", "gpt-4o").WithBaseURL("http://" + ln.Addr().String())
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatalf("headers (200) should arrive before the body is truncated: %v", err)
	}

	var streamErr error
	for ev := range ch {
		if ev.Err != nil {
			streamErr = ev.Err
			break
		}
	}
	if streamErr == nil {
		t.Fatal("the truncated body must error the stream")
	}
	var me *sporecore.ModelError
	if !errors.As(streamErr, &me) || me.Kind != sporecore.ModelErrStreamInterrupted {
		t.Fatalf("expected StreamInterrupted, got %v", streamErr)
	}
	if !me.Retryable() {
		t.Fatal("a mid-stream interruption is retryable")
	}
	<-done
}
