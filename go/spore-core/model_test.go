package sporecore

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"
)

func testProvider() ProviderInfo {
	return ProviderInfo{Name: "test", ModelID: "test-1", ContextWindow: 1000}
}

func emptyRequest() ModelRequest {
	return ModelRequest{
		Messages: []Message{},
		Tools:    []ToolSchema{},
		Params:   ModelParams{},
		Stream:   false,
	}
}

func textResponse(text string, in, out uint32) ModelResponse {
	return ModelResponse{
		Content:    []ContentBlock{NewTextBlock(text)},
		Usage:      TokenUsage{InputTokens: in, OutputTokens: out},
		StopReason: StopEndTurn,
	}
}

// Rule: Call returns the queued response.
func TestCallReturnsQueuedResponse(t *testing.T) {
	m := NewMockModel(testProvider())
	m.PushResponse(textResponse("hi", 3, 1))
	r, err := m.Call(context.Background(), emptyRequest())
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(r.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(r.Content))
	}
	if r.StopReason != StopEndTurn {
		t.Fatalf("stop reason = %q, want %q", r.StopReason, StopEndTurn)
	}
}

// Rule: token counts reported on every call, not optional.
func TestTokenUsageReportedOnEveryCall(t *testing.T) {
	m := NewMockModel(testProvider())
	m.PushResponse(textResponse("a", 5, 7))
	m.PushResponse(textResponse("b", 11, 13))
	ctx := context.Background()
	r1, err := m.Call(ctx, emptyRequest())
	if err != nil {
		t.Fatal(err)
	}
	r2, err := m.Call(ctx, emptyRequest())
	if err != nil {
		t.Fatal(err)
	}
	if r1.Usage.InputTokens != 5 || r1.Usage.OutputTokens != 7 {
		t.Fatalf("r1 usage = %+v", r1.Usage)
	}
	if r2.Usage.InputTokens != 11 || r2.Usage.OutputTokens != 13 {
		t.Fatalf("r2 usage = %+v", r2.Usage)
	}
	if got := m.CallCount(); got != 2 {
		t.Fatalf("call count = %d, want 2", got)
	}
}

// Rule: ContextLimitExceeded raised before API call when detected.
func TestContextLimitEnforcedPreCall(t *testing.T) {
	err := EnforceContextLimit(1500, 1000)
	if err == nil {
		t.Fatal("expected error")
	}
	var me *ModelError
	if !errors.As(err, &me) || me.Kind != ModelErrContextLimitExceeded {
		t.Fatalf("expected ContextLimitExceeded, got %v", err)
	}
	if me.Limit != 1000 || me.Actual != 1500 {
		t.Fatalf("limit/actual = %d/%d", me.Limit, me.Actual)
	}
	if !errors.Is(err, ErrContextLimitExceeded) {
		t.Fatalf("errors.Is(err, ErrContextLimitExceeded) = false")
	}
}

func TestContextLimitPassesWhenUnder(t *testing.T) {
	if err := EnforceContextLimit(999, 1000); err != nil {
		t.Fatal(err)
	}
	if err := EnforceContextLimit(1000, 1000); err != nil {
		t.Fatal(err)
	}
}

// Rule: BudgetExceeded is a harness-side check against ModelParams.MaxTokens.
func TestBudgetEnforcedAgainstMaxTokens(t *testing.T) {
	budget := uint32(100)
	err := EnforceBudget(101, &budget)
	if err == nil {
		t.Fatal("expected error")
	}
	var me *ModelError
	if !errors.As(err, &me) || me.Kind != ModelErrBudgetExceeded {
		t.Fatalf("expected BudgetExceeded, got %v", err)
	}
	if me.Budget != 100 || me.Used != 101 {
		t.Fatalf("budget/used = %d/%d", me.Budget, me.Used)
	}
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("errors.Is(err, ErrBudgetExceeded) = false")
	}
}

func TestBudgetPassesWhenUnderOrUnset(t *testing.T) {
	b := uint32(100)
	if err := EnforceBudget(99, &b); err != nil {
		t.Fatal(err)
	}
	if err := EnforceBudget(100, &b); err != nil {
		t.Fatal(err)
	}
	if err := EnforceBudget(1_000_000, nil); err != nil {
		t.Fatal(err)
	}
}

// Every error variant constructible & non-empty Error().
func TestEveryErrorVariantIsConstructible(t *testing.T) {
	d := 5 * time.Second
	variants := []*ModelError{
		NewProviderError(500, "boom"),
		NewRateLimited(&d),
		NewRateLimited(nil),
		NewContextLimitExceeded(1, 2),
		NewBudgetExceeded(1, 2),
		NewTimeout(),
	}
	for i, v := range variants {
		if v.Error() == "" {
			t.Fatalf("variant %d had empty Error()", i)
		}
	}
}

// Rule: provider identity is reported.
func TestProviderIdentityReported(t *testing.T) {
	m := NewMockModel(testProvider())
	p := m.Provider()
	if p.Name != "test" || p.ModelID != "test-1" || p.ContextWindow != 1000 {
		t.Fatalf("provider = %+v", p)
	}
}

// Rule: streaming yields MessageStop with usage.
func TestStreamingYieldsMessageStopWithUsage(t *testing.T) {
	m := NewMockModel(testProvider())
	m.PushResponse(textResponse("hello", 4, 2))
	ch, err := m.CallStreaming(context.Background(), emptyRequest())
	if err != nil {
		t.Fatal(err)
	}
	sawStart := false
	var finalUsage *TokenUsage
	for e := range ch {
		if e.Err != nil {
			t.Fatalf("stream err: %v", e.Err)
		}
		switch e.Event.Type {
		case StreamMessageStart:
			sawStart = true
		case StreamMessageStop:
			finalUsage = e.Event.Usage
		}
	}
	if !sawStart {
		t.Fatal("never saw MessageStart")
	}
	if finalUsage == nil {
		t.Fatal("MessageStop missing usage")
	}
	if finalUsage.InputTokens != 4 || finalUsage.OutputTokens != 2 {
		t.Fatalf("final usage = %+v", finalUsage)
	}
}

// Rule: provider errors surface as typed harness errors.
func TestProviderErrorsSurfaceTyped(t *testing.T) {
	m := NewMockModel(testProvider())
	m.PushError(NewProviderError(503, "unavailable"))
	_, err := m.Call(context.Background(), emptyRequest())
	var me *ModelError
	if !errors.As(err, &me) || me.Kind != ModelErrProviderError || me.Code != 503 {
		t.Fatalf("expected ProviderError(503), got %v", err)
	}
}

func TestRateLimitSurfaceWithRetryAfter(t *testing.T) {
	m := NewMockModel(testProvider())
	d := 2 * time.Second
	m.PushError(NewRateLimited(&d))
	_, err := m.Call(context.Background(), emptyRequest())
	var me *ModelError
	if !errors.As(err, &me) || me.Kind != ModelErrRateLimited {
		t.Fatalf("expected RateLimited, got %v", err)
	}
	if me.RetryAfter == nil || *me.RetryAfter != 2*time.Second {
		t.Fatalf("retry_after = %v", me.RetryAfter)
	}
}

func TestTimeoutSurface(t *testing.T) {
	m := NewMockModel(testProvider())
	m.PushError(NewTimeout())
	_, err := m.Call(context.Background(), emptyRequest())
	var me *ModelError
	if !errors.As(err, &me) || me.Kind != ModelErrTimeout {
		t.Fatalf("expected Timeout, got %v", err)
	}
}

// JSON round-trip for ModelRequest.
func TestModelRequestRoundtripsJSON(t *testing.T) {
	temp := float32(0.7)
	maxT := uint32(1024)
	req := ModelRequest{
		Messages: []Message{{
			Role:    RoleUser,
			Content: NewTextContent("hi"),
		}},
		Tools: []ToolSchema{{
			Name:        "echo",
			Description: "echoes input",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
		Params: ModelParams{
			Temperature: &temp,
			MaxTokens:   &maxT,
		},
		Stream: false,
	}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var back ModelRequest
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v\njson=%s", err, data)
	}
	if back.Messages[0].Role != RoleUser {
		t.Fatalf("role = %q", back.Messages[0].Role)
	}
	if back.Messages[0].Content.Type != ContentTypeText || back.Messages[0].Content.Text != "hi" {
		t.Fatalf("message content = %+v", back.Messages[0].Content)
	}
	if back.Tools[0].Name != "echo" {
		t.Fatalf("tool name = %q", back.Tools[0].Name)
	}
	if back.Params.Temperature == nil || *back.Params.Temperature != 0.7 {
		t.Fatalf("temperature = %v", back.Params.Temperature)
	}
	if back.Params.MaxTokens == nil || *back.Params.MaxTokens != 1024 {
		t.Fatalf("max_tokens = %v", back.Params.MaxTokens)
	}
}

// JSON round-trip for ModelResponse including tool_use ContentBlock.
func TestModelResponseRoundtripsJSON(t *testing.T) {
	cacheRead := uint32(1)
	cacheWrite := uint32(2)
	resp := ModelResponse{
		Content: []ContentBlock{
			NewTextBlock("ok"),
			NewToolUseBlock(ToolCall{
				ID:    "1",
				Name:  "x",
				Input: json.RawMessage(`{"a":1}`),
			}),
		},
		Usage: TokenUsage{
			InputTokens:      3,
			OutputTokens:     4,
			CacheReadTokens:  &cacheRead,
			CacheWriteTokens: &cacheWrite,
		},
		StopReason: StopToolUse,
	}
	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatal(err)
	}
	var back ModelResponse
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v\njson=%s", err, data)
	}
	if len(back.Content) != 2 {
		t.Fatalf("content len = %d", len(back.Content))
	}
	if back.Content[0].Type != ContentBlockTypeText || back.Content[0].Text != "ok" {
		t.Fatalf("block 0 = %+v", back.Content[0])
	}
	if back.Content[1].Type != ContentBlockTypeToolUse {
		t.Fatalf("block 1 type = %q", back.Content[1].Type)
	}
	if back.Content[1].ToolCall == nil ||
		back.Content[1].ToolCall.ID != "1" ||
		back.Content[1].ToolCall.Name != "x" {
		t.Fatalf("block 1 tool_call = %+v", back.Content[1].ToolCall)
	}
	var input map[string]int
	if err := json.Unmarshal(back.Content[1].ToolCall.Input, &input); err != nil {
		t.Fatalf("input unmarshal: %v", err)
	}
	if input["a"] != 1 {
		t.Fatalf("input = %+v", input)
	}
	if back.StopReason != StopToolUse {
		t.Fatalf("stop_reason = %q", back.StopReason)
	}
	if back.Usage.CacheReadTokens == nil || *back.Usage.CacheReadTokens != 1 {
		t.Fatalf("cache_read = %v", back.Usage.CacheReadTokens)
	}
}

// ModelError JSON round-trip with tag layout matching Rust.
func TestModelErrorRoundtripsJSON(t *testing.T) {
	d := 7 * time.Second
	cases := []*ModelError{
		NewProviderError(500, "boom"),
		NewRateLimited(&d),
		NewRateLimited(nil),
		NewContextLimitExceeded(1000, 1500),
		NewBudgetExceeded(100, 200),
		NewTimeout(),
	}
	for _, e := range cases {
		data, err := json.Marshal(e)
		if err != nil {
			t.Fatalf("marshal %v: %v", e.Kind, err)
		}
		var back ModelError
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("unmarshal %s: %v\njson=%s", e.Kind, err, data)
		}
		if back.Kind != e.Kind {
			t.Fatalf("kind: got %q want %q", back.Kind, e.Kind)
		}
		if back.Error() != e.Error() {
			t.Fatalf("Error() mismatch: got %q want %q (json=%s)", back.Error(), e.Error(), data)
		}
	}
}

// SC-3: Retryable() classifies transient vs deterministic model errors.
// Transport drops, mid-stream interruptions, timeouts, and rate-limits are
// transient; provider/context/budget errors are deterministic.
func TestModelErrorRetryableClassification(t *testing.T) {
	d := 5 * time.Second
	retryable := []*ModelError{
		NewTransport("boom"),
		NewStreamInterrupted("eof"),
		NewTimeout(),
		NewRateLimited(&d),
		NewRateLimited(nil),
	}
	for _, e := range retryable {
		if !e.Retryable() {
			t.Fatalf("%s should be retryable", e.Kind)
		}
	}
	deterministic := []*ModelError{
		NewProviderError(0, "x"),
		NewProviderError(500, "x"),
		NewContextLimitExceeded(1, 2),
		NewBudgetExceeded(1, 2),
	}
	for _, e := range deterministic {
		if e.Retryable() {
			t.Fatalf("%s should NOT be retryable", e.Kind)
		}
	}
}

// SC-3: the two new variants carry the exact {"kind","message"} tag shape the
// other languages mirror byte-for-byte, and round-trip back to the same kind.
func TestTypedTransportErrorsSerdeTagShape(t *testing.T) {
	cases := []struct {
		err  *ModelError
		want string
	}{
		{NewTransport("m"), `{"kind":"Transport","message":"m"}`},
		{NewStreamInterrupted("m"), `{"kind":"StreamInterrupted","message":"m"}`},
	}
	for _, tc := range cases {
		data, err := json.Marshal(tc.err)
		if err != nil {
			t.Fatalf("marshal %s: %v", tc.err.Kind, err)
		}
		if string(data) != tc.want {
			t.Fatalf("%s bytes: got %s want %s", tc.err.Kind, data, tc.want)
		}
		var back ModelError
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("unmarshal %s: %v", tc.err.Kind, err)
		}
		if back.Kind != tc.err.Kind || back.Message != tc.err.Message {
			t.Fatalf("round-trip %s: got kind=%q msg=%q", tc.err.Kind, back.Kind, back.Message)
		}
		if !back.Retryable() {
			t.Fatalf("%s should be retryable after round-trip", back.Kind)
		}
	}
}

// JSON round-trip for every Content variant.
func TestContentVariantsRoundtripJSON(t *testing.T) {
	cases := []Content{
		NewTextContent("hello"),
		NewToolCallContent(ToolCall{ID: "t1", Name: "echo", Input: json.RawMessage(`{"x":1}`)}),
		NewToolResultContent(ToolResult{ToolUseID: "t1", Content: "done", IsError: false}),
		NewImageContent("image/png", "base64data"),
	}
	for _, c := range cases {
		data, err := json.Marshal(c)
		if err != nil {
			t.Fatalf("marshal %s: %v", c.Type, err)
		}
		var back Content
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("unmarshal %s: %v\njson=%s", c.Type, err, data)
		}
		if back.Type != c.Type {
			t.Fatalf("type mismatch %q vs %q", back.Type, c.Type)
		}
		// Compare key fields per variant.
		switch c.Type {
		case ContentTypeText:
			if back.Text != c.Text {
				t.Fatalf("text mismatch")
			}
		case ContentTypeToolCall:
			if back.ToolCall == nil || back.ToolCall.Name != c.ToolCall.Name {
				t.Fatalf("tool_call mismatch")
			}
		case ContentTypeToolResult:
			if back.ToolResult == nil || back.ToolResult.ToolUseID != c.ToolResult.ToolUseID {
				t.Fatalf("tool_result mismatch")
			}
		case ContentTypeImage:
			if back.MediaType != c.MediaType || back.Data != c.Data {
				t.Fatalf("image mismatch")
			}
		}
	}
}

// Sanity: ModelRequest with default params serialises stop_sequences as [].
func TestModelRequestEmptyParamsSerialisesSliceAsEmpty(t *testing.T) {
	req := ModelRequest{}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	// We expect messages:[], tools:[], stop_sequences:[]
	want := `"messages":[]`
	if !contains(string(data), want) {
		t.Fatalf("expected %s in %s", want, data)
	}
	if !contains(string(data), `"tools":[]`) {
		t.Fatalf("expected tools:[] in %s", data)
	}
	if !contains(string(data), `"stop_sequences":[]`) {
		t.Fatalf("expected stop_sequences:[] in %s", data)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// Spot-check Marshalled shape of a tool_use ContentBlock matches the fixture
// layout (flat fields, not nested under "tool_use").
func TestToolUseBlockMarshalsFlat(t *testing.T) {
	b := NewToolUseBlock(ToolCall{ID: "toolu_01", Name: "echo", Input: json.RawMessage(`{"text":"hi"}`)})
	data, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"type":"tool_use"`, `"id":"toolu_01"`, `"name":"echo"`, `"input":{"text":"hi"}`} {
		if !contains(string(data), want) {
			t.Fatalf("missing %s in %s", want, data)
		}
	}
}

// Sentinel sanity: Unwrap chain works for both Layer 1 kinds.
func TestSentinelUnwrap(t *testing.T) {
	err := NewContextLimitExceeded(1, 2)
	if !errors.Is(err, ErrContextLimitExceeded) {
		t.Fatal("ContextLimitExceeded did not unwrap to sentinel")
	}
	err2 := NewBudgetExceeded(1, 2)
	if !errors.Is(err2, ErrBudgetExceeded) {
		t.Fatal("BudgetExceeded did not unwrap to sentinel")
	}
	// Non-Layer-1 should not unwrap to sentinels.
	if errors.Is(NewTimeout(), ErrContextLimitExceeded) {
		t.Fatal("Timeout unexpectedly unwrapped to ErrContextLimitExceeded")
	}
}

// Ensure structural equality after round-trip of a Message containing
// a tool_call Content.
func TestMessageWithToolCallRoundtrip(t *testing.T) {
	msg := Message{
		Role: RoleAssistant,
		Content: NewToolCallContent(ToolCall{
			ID:    "abc",
			Name:  "n",
			Input: json.RawMessage(`{"k":"v"}`),
		}),
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var back Message
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.Role != RoleAssistant {
		t.Fatalf("role = %q", back.Role)
	}
	if back.Content.Type != ContentTypeToolCall ||
		back.Content.ToolCall == nil ||
		back.Content.ToolCall.ID != "abc" {
		t.Fatalf("content = %+v", back.Content)
	}
	// Input bytes should decode to identical map.
	var a, b map[string]string
	_ = json.Unmarshal(back.Content.ToolCall.Input, &a)
	_ = json.Unmarshal(msg.Content.ToolCall.Input, &b)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("input mismatch: %v vs %v", a, b)
	}
}
