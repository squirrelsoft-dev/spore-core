package ollama

import (
	"context"
	"encoding/json"
	"errors"
	"io"
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

func req(msgs ...sporecore.Message) sporecore.ModelRequest {
	return sporecore.ModelRequest{Messages: msgs}
}

// composite test server: dispatches /api/tags to tags, /api/show to show
// (defaulting to 404 so discovery degrades gracefully), others to chat.
type splitHandler struct {
	tagsCount int64
	showCount int64
	chat      http.HandlerFunc
	show      http.HandlerFunc // optional; nil → 404
	model     string
}

func (s *splitHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/api/tags":
		atomic.AddInt64(&s.tagsCount, 1)
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"models":[{"name":"`+s.model+`:latest"}]}`)
		return
	case "/api/show":
		atomic.AddInt64(&s.showCount, 1)
		if s.show == nil {
			http.NotFound(w, r)
			return
		}
		s.show(w, r)
		return
	}
	s.chat(w, r)
}

// ---------------------------------------------------------------------------
// Constructors / defaults
// ---------------------------------------------------------------------------

func TestNewUsesLocalhostDefaults(t *testing.T) {
	c := New("llama3.2")
	if c.baseURL != "http://localhost:11434" {
		t.Fatalf("base_url: %s", c.baseURL)
	}
	if c.modelID != "llama3.2" {
		t.Fatalf("model_id: %s", c.modelID)
	}
	if c.timeout != 300*time.Second {
		t.Fatalf("timeout: %s", c.timeout)
	}
	if c.keepAlive != "5m" {
		t.Fatalf("keep_alive: %s", c.keepAlive)
	}
}

func TestWithBaseURLOverrides(t *testing.T) {
	c := WithBaseURL("mistral", "http://remote:9999")
	if c.baseURL != "http://remote:9999" || c.modelID != "mistral" {
		t.Fatalf("c = %+v", c)
	}
}

func TestDefaultsMatchSpec(t *testing.T) {
	if DefaultBaseURL != "http://localhost:11434" {
		t.Fatalf("DefaultBaseURL: %s", DefaultBaseURL)
	}
	if DefaultTimeout != 300*time.Second {
		t.Fatalf("DefaultTimeout: %s", DefaultTimeout)
	}
	if DefaultKeepAlive != "5m" {
		t.Fatalf("DefaultKeepAlive: %s", DefaultKeepAlive)
	}
}

// ---------------------------------------------------------------------------
// buildRequest
// ---------------------------------------------------------------------------

func TestBuildRequestSerializesOptionsAndKeepAlive(t *testing.T) {
	r := req(userMsg("hi"))
	mt := uint32(256)
	temp := float32(0.7)
	top := float32(0.9)
	r.Params.MaxTokens = &mt
	r.Params.Temperature = &temp
	r.Params.TopP = &top
	r.Params.StopSequences = []string{"END"}
	body := buildRequest("llama3.2", "10m", r, false)
	out, _ := json.Marshal(body)
	s := string(out)
	for _, want := range []string{
		`"keep_alive":"10m"`,
		`"num_predict":256`,
		`"temperature":0.7`,
		`"top_p":0.9`,
		`"stop":["END"]`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in: %s", want, s)
		}
	}
	if strings.Contains(s, `"stream":true`) {
		t.Fatalf("expected stream=false: %s", s)
	}
}

func TestBuildRequestSerializesTools(t *testing.T) {
	r := req(userMsg("hi"))
	r.Tools = []sporecore.ToolSchema{{
		Name:        "search",
		Description: "search the web",
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}}
	body := buildRequest("llama3.2", "", r, false)
	out, _ := json.Marshal(body)
	s := string(out)
	for _, want := range []string{`"tools":[`, `"name":"search"`, `"type":"function"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %s in: %s", want, s)
		}
	}
}

func TestBuildRequestToolCallUsesObjectArguments(t *testing.T) {
	r := req(sporecore.Message{
		Role: sporecore.RoleAssistant,
		Content: sporecore.NewToolCallContent(sporecore.ToolCall{
			ID:    "call-0",
			Name:  "fetch",
			Input: json.RawMessage(`{"url":"x"}`),
		}),
	})
	body := buildRequest("llama3.2", "", r, false)
	out, _ := json.Marshal(body.Messages[0])
	s := string(out)
	if !strings.Contains(s, `"arguments":{"url":"x"}`) {
		t.Fatalf("arguments not an object: %s", s)
	}
	// must NOT be a JSON-encoded string (OpenAI shape)
	if strings.Contains(s, `"arguments":"{`) {
		t.Fatalf("arguments unexpectedly stringified: %s", s)
	}
}

func TestBuildRequestToolResultMapsToToolRole(t *testing.T) {
	r := req(sporecore.Message{
		Role: sporecore.RoleTool,
		Content: sporecore.NewToolResultContent(sporecore.ToolResult{
			ToolUseID: "call-0",
			Content:   "ok",
		}),
	})
	body := buildRequest("llama3.2", "", r, false)
	m := body.Messages[0]
	if m.Role != "tool" || m.Content != "ok" || m.ToolCallID != "call-0" {
		t.Fatalf("msg: %+v", m)
	}
}

func TestBuildRequestStreamingFlag(t *testing.T) {
	body := buildRequest("llama3.2", "", req(userMsg("hi")), true)
	if !body.Stream {
		t.Fatalf("stream: false")
	}
}

func TestBuildRequestOptionsOmittedWhenEmpty(t *testing.T) {
	body := buildRequest("llama3.2", "", req(userMsg("hi")), false)
	out, _ := json.Marshal(body)
	if strings.Contains(string(out), `"options"`) {
		t.Fatalf("options should be omitted: %s", out)
	}
}

func TestThinkingBlockOmittedInRequest(t *testing.T) {
	// Thinking blocks are response-side only; a normal request must produce
	// no "thinking" key in the wire payload.
	body := buildRequest("llama3.2", "", req(userMsg("hi")), false)
	out, _ := json.Marshal(body)
	if strings.Contains(string(out), "thinking") {
		t.Fatalf("thinking key leaked: %s", out)
	}
}

// ---------------------------------------------------------------------------
// parseStopReason
// ---------------------------------------------------------------------------

func TestStopReasonMapping(t *testing.T) {
	cases := map[string]sporecore.StopReason{
		"stop":       sporecore.StopEndTurn,
		"tool_calls": sporecore.StopToolUse,
		"length":     sporecore.StopMaxTokens,
		"":           sporecore.StopEndTurn,
		"???":        sporecore.StopEndTurn,
	}
	for in, want := range cases {
		if got := parseStopReason(in); got != want {
			t.Fatalf("%q -> %s (want %s)", in, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// parseResponse
// ---------------------------------------------------------------------------

func TestParseResponseExtractsUsageAndText(t *testing.T) {
	raw := `{"message":{"role":"assistant","content":"hi"},"done":true,"done_reason":"stop","prompt_eval_count":7,"eval_count":2}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.Usage.InputTokens != 7 || r.Usage.OutputTokens != 2 {
		t.Fatalf("usage: %+v", r.Usage)
	}
	if r.StopReason != sporecore.StopEndTurn {
		t.Fatalf("stop: %s", r.StopReason)
	}
	if len(r.Content) != 1 || r.Content[0].Text != "hi" {
		t.Fatalf("content: %+v", r.Content)
	}
}

func TestParseResponseCacheFieldsAlwaysNil(t *testing.T) {
	raw := `{"message":{"role":"assistant","content":"x"},"done":true,"prompt_eval_count":1,"eval_count":1}`
	var body wireResponse
	_ = json.Unmarshal([]byte(raw), &body)
	r := parseResponse(body)
	if r.Usage.CacheReadTokens != nil || r.Usage.CacheWriteTokens != nil {
		t.Fatalf("cache fields not nil: %+v", r.Usage)
	}
}

func TestParseResponseToolCallSynthesizesID(t *testing.T) {
	raw := `{"message":{"role":"assistant","tool_calls":[{"function":{"name":"fetch","arguments":{"url":"x"}}},{"function":{"name":"search","arguments":{"q":"y"}}}]},"done":true,"done_reason":"tool_calls","prompt_eval_count":1,"eval_count":1}`
	var body wireResponse
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatal(err)
	}
	r := parseResponse(body)
	if r.StopReason != sporecore.StopToolUse {
		t.Fatalf("stop: %s", r.StopReason)
	}
	if len(r.Content) != 2 {
		t.Fatalf("content len = %d", len(r.Content))
	}
	tc0 := r.Content[0].ToolCall
	tc1 := r.Content[1].ToolCall
	if tc0 == nil || tc0.ID != "call-0" || tc0.Name != "fetch" {
		t.Fatalf("tc0: %+v", tc0)
	}
	if tc1 == nil || tc1.ID != "call-1" {
		t.Fatalf("tc1: %+v", tc1)
	}
	var parsed map[string]any
	if err := json.Unmarshal(tc0.Input, &parsed); err != nil {
		t.Fatalf("input not JSON: %s", tc0.Input)
	}
	if parsed["url"] != "x" {
		t.Fatalf("input: %v", parsed)
	}
}

// ---------------------------------------------------------------------------
// Context window / provider info / String
// ---------------------------------------------------------------------------

func TestContextWindowKnownAndUnknown(t *testing.T) {
	cases := map[string]uint32{
		"llama3.2":         128_000,
		"llama3.2:3b":      128_000,
		"qwen2.5-coder":    128_000,
		"qwen2.5-coder-7b": 128_000,
		"mistral":          32_000,
		"mistral-large":    32_000,
		"gemma":            8_192,
		"gemma2":           8_192,
		"unknown-model":    0,
	}
	for id, want := range cases {
		if got := ContextWindow(id); got != want {
			t.Fatalf("ContextWindow(%q) = %d, want %d", id, got, want)
		}
	}
}

func TestProviderInfoUsesStaticTable(t *testing.T) {
	c := New("llama3.2")
	p := c.Provider()
	if p.Name != "ollama" || p.ModelID != "llama3.2" || p.ContextWindow != 128_000 {
		t.Fatalf("provider: %+v", p)
	}
}

func TestStringIncludesModelAndBaseURL(t *testing.T) {
	c := New("llama3.2")
	s := c.String()
	if !strings.Contains(s, "llama3.2") || !strings.Contains(s, "http://localhost:11434") {
		t.Fatalf("string: %s", s)
	}
}

// ---------------------------------------------------------------------------
// nameMatches
// ---------------------------------------------------------------------------

func TestNameMatchesHandlesLatestTag(t *testing.T) {
	cases := []struct {
		tag, requested string
		want           bool
	}{
		{"llama3.2:latest", "llama3.2", true},
		{"llama3.2", "llama3.2", true},
		{"llama3.2:3b", "llama3.2", true},
		{"llama3.1", "llama3.2", false},
		{"mistral:latest", "llama3.2", false},
	}
	for _, c := range cases {
		if got := nameMatches(c.tag, c.requested); got != c.want {
			t.Fatalf("nameMatches(%q,%q)=%v want %v", c.tag, c.requested, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Call against mock server
// ---------------------------------------------------------------------------

func newSplitServer(t *testing.T, model string, chat http.HandlerFunc) (*httptest.Server, *splitHandler) {
	h := &splitHandler{chat: chat, model: model}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv, h
}

func newSplitServerWithShow(t *testing.T, model string, chat, show http.HandlerFunc) (*httptest.Server, *splitHandler) {
	h := &splitHandler{chat: chat, show: show, model: model}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv, h
}

func TestCallAgainstMockReturnsResponse(t *testing.T) {
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Fatalf("path: %s", r.URL.Path)
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"hello there"},"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}`)
	})
	c := WithBaseURL("llama3.2", srv.URL)
	r, err := c.Call(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Content) != 1 || r.Content[0].Text != "hello there" {
		t.Fatalf("content: %+v", r.Content)
	}
	if r.Usage.InputTokens != 5 || r.Usage.OutputTokens != 2 {
		t.Fatalf("usage: %+v", r.Usage)
	}
}

func TestConnectionRefusedHelpfulMessage(t *testing.T) {
	c := WithBaseURL("llama3.2", "http://127.0.0.1:1").SetTimeout(2 * time.Second)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	if err == nil {
		t.Fatal("expected error")
	}
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 0 || !strings.Contains(merr.Message, "Ollama not running") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestConnectionErrorDoesNotRetry(t *testing.T) {
	c := WithBaseURL("llama3.2", "http://127.0.0.1:1").SetTimeout(5 * time.Second)
	start := time.Now()
	_, _ = c.Call(context.Background(), req(userMsg("hi")))
	if d := time.Since(start); d > 500*time.Millisecond {
		t.Fatalf("expected fail-fast (<500ms); took %v", d)
	}
}

func TestModelNotFoundSuggestsPull(t *testing.T) {
	// /api/tags returns a different model
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"models":[{"name":"mistral:latest"}]}`)
	}))
	defer srv.Close()
	c := WithBaseURL("llama3.2", srv.URL)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 404 || !strings.Contains(merr.Message, "ollama pull llama3.2") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestChat404MapsToPullSuggestion(t *testing.T) {
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
		_, _ = io.WriteString(w, `{"error":"model 'llama3.2' not found"}`)
	})
	c := WithBaseURL("llama3.2", srv.URL)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 404 || !strings.Contains(merr.Message, "ollama pull llama3.2") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestCallTimeoutMapsToTimeout(t *testing.T) {
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
	})
	c := WithBaseURL("llama3.2", srv.URL).SetTimeout(200 * time.Millisecond)
	_, err := c.Call(context.Background(), req(userMsg("hi")))
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrTimeout {
		t.Fatalf("expected Timeout, got %v", err)
	}
}

func TestModelCheckCachedAfterFirstCall(t *testing.T) {
	srv, h := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`)
	})
	c := WithBaseURL("llama3.2", srv.URL)
	if _, err := c.Call(context.Background(), req(userMsg("a"))); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Call(context.Background(), req(userMsg("b"))); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt64(&h.tagsCount); got != 1 {
		t.Fatalf("/api/tags hit %d times, want 1", got)
	}
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

func TestStreamingEmitsTextDeltaThenStop(t *testing.T) {
	ndjson := strings.Join([]string{
		`{"message":{"role":"assistant","content":"hello"},"done":false}`,
		`{"message":{"role":"assistant","content":" world"},"done":false}`,
		`{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}`,
		``,
	}, "\n")
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/x-ndjson")
		_, _ = io.WriteString(w, ndjson)
	})
	c := WithBaseURL("llama3.2", srv.URL)
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

func TestStreamingParsesNDJSONLines(t *testing.T) {
	ndjson := strings.Join([]string{
		`{"message":{"role":"assistant","content":"ab"},"done":false}`,
		`{"message":{"role":"assistant","content":"cd"},"done":false}`,
		`{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`,
		``,
	}, "\n")
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, ndjson)
	})
	c := WithBaseURL("llama3.2", srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var deltas []string
	for ev := range ch {
		if ev.Event.Type == sporecore.StreamContentBlockDelta {
			deltas = append(deltas, ev.Event.Delta)
		}
	}
	if strings.Join(deltas, ",") != "ab,cd" {
		t.Fatalf("deltas: %v", deltas)
	}
}

func TestStreamingDoneCarriesUsage(t *testing.T) {
	ndjson := `{"message":{"role":"assistant","content":"x"},"done":true,"done_reason":"stop","prompt_eval_count":42,"eval_count":7}` + "\n"
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, ndjson)
	})
	c := WithBaseURL("llama3.2", srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var u *sporecore.TokenUsage
	for ev := range ch {
		if ev.Event.Type == sporecore.StreamMessageStop {
			u = ev.Event.Usage
		}
	}
	if u == nil || u.InputTokens != 42 || u.OutputTokens != 7 {
		t.Fatalf("usage: %+v", u)
	}
}

func TestStreamingAccumulatesToolCalls(t *testing.T) {
	ndjson := strings.Join([]string{
		`{"message":{"role":"assistant","tool_calls":[{"function":{"name":"fetch","arguments":{"url":"x"}}}]},"done":false}`,
		`{"message":{"role":"assistant","content":""},"done":true,"done_reason":"tool_calls","prompt_eval_count":1,"eval_count":1}`,
		``,
	}, "\n")
	srv, _ := newSplitServer(t, "llama3.2", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, ndjson)
	})
	c := WithBaseURL("llama3.2", srv.URL)
	ch, err := c.CallStreaming(context.Background(), req(userMsg("hi")))
	if err != nil {
		t.Fatal(err)
	}
	var toolJSONs []string
	var finalStop sporecore.StopReason
	for ev := range ch {
		switch ev.Event.Type {
		case sporecore.StreamToolUseDelta:
			toolJSONs = append(toolJSONs, ev.Event.PartialJSON)
		case sporecore.StreamMessageStop:
			finalStop = ev.Event.StopReason
		}
	}
	if len(toolJSONs) != 1 {
		t.Fatalf("tool jsons: %v", toolJSONs)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(toolJSONs[0]), &parsed); err != nil {
		t.Fatalf("not JSON: %q", toolJSONs[0])
	}
	if parsed["url"] != "x" {
		t.Fatalf("args: %v", parsed)
	}
	if finalStop != sporecore.StopToolUse {
		t.Fatalf("stop: %s", finalStop)
	}
}

// ---------------------------------------------------------------------------
// count_tokens
// ---------------------------------------------------------------------------

func TestCountTokensUsesEmbedEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Fatalf("path: %s", r.URL.Path)
		}
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, `{"prompt_eval_count":123}`)
	}))
	defer srv.Close()
	c := WithBaseURL("llama3.2", srv.URL)
	n, err := c.CountTokens(context.Background(), req(userMsg("hello world")))
	if err != nil {
		t.Fatal(err)
	}
	if n != 123 {
		t.Fatalf("n=%d", n)
	}
}

func TestCountTokensFallsBackToHeuristic(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	defer srv.Close()
	c := WithBaseURL("llama3.2", srv.URL)
	// 40 'a' chars + 1 newline = 41 bytes -> 41/4 = 10
	n, err := c.CountTokens(context.Background(), req(userMsg(strings.Repeat("a", 40))))
	if err != nil {
		t.Fatal(err)
	}
	if n != 10 {
		t.Fatalf("n=%d, want 10", n)
	}
}

// ---------------------------------------------------------------------------
// Fixture replay round-trip
// ---------------------------------------------------------------------------

func TestFixtureReplayRoundTrip(t *testing.T) {
	_, this, _, _ := runtime.Caller(0)
	dir := filepath.Dir(this)
	path := filepath.Join(dir, "..", "..", "..", "fixtures", "model_responses", "model_interface", "ollama_basic_text.jsonl")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := sporecore.ParseReplayJSONL(string(raw), sporecore.ProviderInfo{
		Name: "ollama", ModelID: "llama3.2", ContextWindow: 128_000,
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

// ---------------------------------------------------------------------------
// /api/show discovery + tool-capability guard
// ---------------------------------------------------------------------------

func toolReq() sporecore.ModelRequest {
	r := req(userMsg("use a tool"))
	r.Tools = []sporecore.ToolSchema{{
		Name:        "search",
		Description: "search the web",
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}}
	return r
}

func chatOK(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("content-type", "application/json")
	_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`)
}

func showJSON(body string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "application/json")
		_, _ = io.WriteString(w, body)
	}
}

func TestProviderReflectsDiscoveredContextWindow(t *testing.T) {
	srv, _ := newSplitServerWithShow(t, "llama3.2", chatOK,
		showJSON(`{"model_info":{"llama.context_length":16384},"capabilities":["tools"]}`))
	c := WithBaseURL("llama3.2", srv.URL)
	// Before the probe runs, Provider falls back to the static table.
	if got := c.Provider().ContextWindow; got != 128_000 {
		t.Fatalf("pre-probe context window = %d, want 128000", got)
	}
	if _, err := c.Call(context.Background(), req(userMsg("hi"))); err != nil {
		t.Fatal(err)
	}
	// After the probe, Provider reflects the discovered value.
	if got := c.Provider().ContextWindow; got != 16_384 {
		t.Fatalf("post-probe context window = %d, want 16384", got)
	}
}

func TestProviderFallsBackWhenShow404s(t *testing.T) {
	// show handler nil → splitHandler returns 404 for /api/show.
	srv, _ := newSplitServer(t, "llama3.2", chatOK)
	c := WithBaseURL("llama3.2", srv.URL)
	if _, err := c.Call(context.Background(), req(userMsg("hi"))); err != nil {
		t.Fatal(err)
	}
	if got := c.Provider().ContextWindow; got != 128_000 {
		t.Fatalf("context window = %d, want 128000 (static fallback)", got)
	}
}

func TestProviderFallsBackWhenContextLengthMissing(t *testing.T) {
	srv, _ := newSplitServerWithShow(t, "llama3.2", chatOK,
		showJSON(`{"model_info":{"general.architecture":"llama"},"capabilities":["tools"]}`))
	c := WithBaseURL("llama3.2", srv.URL)
	if _, err := c.Call(context.Background(), req(userMsg("hi"))); err != nil {
		t.Fatal(err)
	}
	if got := c.Provider().ContextWindow; got != 128_000 {
		t.Fatalf("context window = %d, want 128000 (static fallback)", got)
	}
}

func TestToolRequestRejectedWhenCapabilityAbsent(t *testing.T) {
	// capabilities lacks "tools"; chat handler fails the test if hit.
	chat := func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("/api/chat must not be hit, path=%s", r.URL.Path)
	}
	srv, _ := newSplitServerWithShow(t, "gemma", chat,
		showJSON(`{"model_info":{"gemma.context_length":8192},"capabilities":["completion"]}`))
	c := WithBaseURL("gemma", srv.URL)
	_, err := c.Call(context.Background(), toolReq())
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 0 || !strings.Contains(merr.Message, "does not support tool calling") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestToolRequestProceedsWhenCapabilityPresent(t *testing.T) {
	srv, _ := newSplitServerWithShow(t, "llama3.2", chatOK,
		showJSON(`{"model_info":{"llama.context_length":128000},"capabilities":["completion","tools"]}`))
	c := WithBaseURL("llama3.2", srv.URL)
	r, err := c.Call(context.Background(), toolReq())
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Content) != 1 || r.Content[0].Text != "ok" {
		t.Fatalf("content: %+v", r.Content)
	}
}

func TestToolRequestRejectedWhenCapabilitiesEmpty(t *testing.T) {
	// /api/show returns an empty capabilities array. With the static whitelist
	// removed, empty capabilities fail closed — even for a model id (llama3.2)
	// that the old prefix table would have allowed. The chat handler fails the
	// test if hit, proving the guard rejected before any call.
	chat := func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("/api/chat must not be hit, path=%s", r.URL.Path)
	}
	srv, _ := newSplitServerWithShow(t, "llama3.2", chat,
		showJSON(`{"model_info":{"llama.context_length":128000},"capabilities":[]}`))
	c := WithBaseURL("llama3.2", srv.URL)
	_, err := c.Call(context.Background(), toolReq())
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 0 || !strings.Contains(merr.Message, "does not support tool calling") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestToolRequestRejectedWhenShow404s(t *testing.T) {
	// /api/show 404s ⟹ empty modelMeta ⟹ NOT tool-capable (fail closed).
	// nil show handler ⟹ splitHandler returns 404 for /api/show. The chat
	// handler fails the test if hit.
	chat := func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("/api/chat must not be hit, path=%s", r.URL.Path)
	}
	srv, _ := newSplitServer(t, "llama3.2", chat)
	c := WithBaseURL("llama3.2", srv.URL)
	_, err := c.Call(context.Background(), toolReq())
	var merr *sporecore.ModelError
	if !errors.As(err, &merr) || merr.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if merr.Code != 0 || !strings.Contains(merr.Message, "does not support tool calling") {
		t.Fatalf("err: %+v", merr)
	}
}

func TestShowFetchedAtMostOnce(t *testing.T) {
	srv, h := newSplitServerWithShow(t, "llama3.2", chatOK,
		showJSON(`{"model_info":{"llama.context_length":32000},"capabilities":["tools"]}`))
	c := WithBaseURL("llama3.2", srv.URL)
	if _, err := c.Call(context.Background(), req(userMsg("a"))); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Call(context.Background(), req(userMsg("b"))); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt64(&h.showCount); got != 1 {
		t.Fatalf("/api/show hit %d times, want 1", got)
	}
}

// ---------------------------------------------------------------------------
// Live integration tests (gated on OLLAMA_LIVE=1)
// ---------------------------------------------------------------------------

func skipUnlessLive(t *testing.T) {
	t.Helper()
	if os.Getenv("OLLAMA_LIVE") != "1" {
		t.Skip("OLLAMA_LIVE not set; requires local ollama with llama3.2 pulled")
	}
}

func TestLiveCallReturnsResponse(t *testing.T) {
	skipUnlessLive(t)
	c := New("llama3.2")
	r, err := c.Call(context.Background(), req(userMsg("Reply with the word 'pong'.")))
	if err != nil {
		t.Fatal(err)
	}
	if r.Usage.InputTokens == 0 || r.Usage.OutputTokens == 0 {
		t.Fatalf("usage: %+v", r.Usage)
	}
}

func TestLiveStreamingEmitsEvents(t *testing.T) {
	skipUnlessLive(t)
	c := New("llama3.2")
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

func TestLiveCountTokensNonzero(t *testing.T) {
	skipUnlessLive(t)
	c := New("llama3.2")
	n, err := c.CountTokens(context.Background(), req(userMsg("count my tokens please")))
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatalf("n=0")
	}
}
