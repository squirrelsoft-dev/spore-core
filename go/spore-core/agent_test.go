package sporecore

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func agentProvider() ProviderInfo {
	return ProviderInfo{Name: "test", ModelID: "test-1", ContextWindow: 1000}
}

func userCtx(text string) Context {
	return Context{
		Messages: []Message{{Role: RoleUser, Content: NewTextContent(text)}},
		Tools:    []ToolSchema{},
		Params:   ModelParams{},
	}
}

func usage(in, out uint32) TokenUsage {
	return TokenUsage{InputTokens: in, OutputTokens: out}
}

func textRespEndTurn(text string) ModelResponse {
	return ModelResponse{
		Content:    []ContentBlock{NewTextBlock(text)},
		Usage:      usage(3, 4),
		StopReason: StopEndTurn,
	}
}

func toolResp(calls []ToolCall) ModelResponse {
	blocks := make([]ContentBlock, 0, len(calls))
	for _, c := range calls {
		blocks = append(blocks, NewToolUseBlock(c))
	}
	return ModelResponse{
		Content:    blocks,
		Usage:      usage(5, 6),
		StopReason: StopToolUse,
	}
}

func makeAgent(m *MockModel) *ModelAgent {
	return NewModelAgent(AgentID("coding-agent"), m)
}

// Rule: one turn = one model call.
func TestTurnMakesExactlyOneModelCall(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(textRespEndTurn("ok"))
	agent := makeAgent(m)
	_ = agent.Turn(context.Background(), userCtx("hi"))
	if m.CallCount() != 1 {
		t.Fatalf("call count = %d, want 1", m.CallCount())
	}
}

// Rule: FinalResponse classification on stop_reason=EndTurn with text.
func TestFinalResponseOnEndTurnWithText(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(textRespEndTurn("hello world"))
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("hi"))
	if r.Kind != TurnFinalResponse {
		t.Fatalf("kind = %q, want %q", r.Kind, TurnFinalResponse)
	}
	if r.Content != "hello world" {
		t.Fatalf("content = %q", r.Content)
	}
	if r.Usage == nil || r.Usage.InputTokens != 3 || r.Usage.OutputTokens != 4 {
		t.Fatalf("usage = %+v", r.Usage)
	}
}

// Rule: ToolCallRequested classification on stop_reason=ToolUse.
func TestToolCallRequestedOnToolUseStop(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(toolResp([]ToolCall{
		{ID: "call_1", Name: "read_file", Input: json.RawMessage(`{"path":"/x"}`)},
	}))
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("read /x"))
	if r.Kind != TurnToolCallRequested {
		t.Fatalf("kind = %q", r.Kind)
	}
	if len(r.Calls) != 1 || r.Calls[0].ID != "call_1" || r.Calls[0].Name != "read_file" {
		t.Fatalf("calls = %+v", r.Calls)
	}
	if r.Usage == nil || r.Usage.InputTokens != 5 {
		t.Fatalf("usage = %+v", r.Usage)
	}
}

// Rule: ToolCallRequested may carry multiple parallel tool calls.
func TestToolCallRequestedCarriesMultipleCalls(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(toolResp([]ToolCall{
		{ID: "a", Name: "read_file", Input: json.RawMessage(`{"path":"/a"}`)},
		{ID: "b", Name: "read_file", Input: json.RawMessage(`{"path":"/b"}`)},
	}))
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("both"))
	if r.Kind != TurnToolCallRequested || len(r.Calls) != 2 {
		t.Fatalf("got %+v", r)
	}
	if r.Calls[0].ID != "a" || r.Calls[1].ID != "b" {
		t.Fatalf("calls = %+v", r.Calls)
	}
}

// Rule: a clean EndTurn with no text and no tool calls is the model's
// voluntary completion signal → a (possibly empty) terminal FinalResponse,
// NOT an EmptyResponse error.
func TestCleanEndTurnWithNoContentIsEmptyFinalResponse(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{},
		Usage:      usage(1, 0),
		StopReason: StopEndTurn,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnFinalResponse {
		t.Fatalf("expected empty FinalResponse, got %+v / err=%+v", r, r.Err)
	}
	if r.Content != "" {
		t.Fatalf("content = %q, want empty", r.Content)
	}
	if r.Reasoning != nil {
		t.Fatalf("reasoning = %+v, want nil", r.Reasoning)
	}
	if r.Usage.InputTokens != 1 {
		t.Fatalf("usage = %+v", r.Usage)
	}
}

// Rule: a truncated/abnormal empty (MaxTokens with no content) remains a
// genuine EmptyResponse error — only clean EndTurn is reclassified.
func TestMaxTokensWithNoContentIsEmptyResponse(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{},
		Usage:      usage(1, 0),
		StopReason: StopMaxTokens,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnError || r.Err == nil || r.Err.Kind != AgentErrEmptyResponse {
		t.Fatalf("got %+v / err=%+v", r, r.Err)
	}
	if r.Usage == nil || r.Usage.InputTokens != 1 {
		t.Fatalf("usage = %+v", r.Usage)
	}
}

// Rule: Thinking-only content is treated as empty. Under MaxTokens (a
// truncated stop) this remains an EmptyResponse error; thinking is not a
// terminal response.
func TestThinkingBlocksDoNotSatisfyFinalResponse(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{NewThinkingBlock("musing")},
		Usage:      usage(1, 2),
		StopReason: StopMaxTokens,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnError || r.Err.Kind != AgentErrEmptyResponse {
		t.Fatalf("got %+v", r)
	}
}

// Rule: ModelError surfaces wrapped, usage=nil.
func TestModelErrorSurfacesWrapped(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushError(NewTimeout())
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("hi"))
	if r.Kind != TurnError {
		t.Fatalf("kind = %q", r.Kind)
	}
	if r.Err == nil || r.Err.Kind != AgentErrModelError || r.Err.ModelError == nil {
		t.Fatalf("err = %+v", r.Err)
	}
	if r.Err.ModelError.Kind != ModelErrTimeout {
		t.Fatalf("inner kind = %q", r.Err.ModelError.Kind)
	}
	if r.Usage != nil {
		t.Fatalf("usage should be nil for model error, got %+v", r.Usage)
	}
	// errors.As should unwrap through AgentError to *ModelError.
	var me *ModelError
	if !errors.As(r.Err, &me) {
		t.Fatalf("errors.As to *ModelError failed")
	}
}

// Rule: stop_reason=ToolUse without ToolUse blocks → MalformedToolCall.
func TestMalformedWhenToolUseStopButNoToolBlocks(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{NewTextBlock("hmm")},
		Usage:      usage(2, 2),
		StopReason: StopToolUse,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnError || r.Err == nil || r.Err.Kind != AgentErrMalformedToolCall {
		t.Fatalf("got %+v", r)
	}
	if r.Usage == nil {
		t.Fatalf("expected usage on malformed tool call")
	}
}

// Tool calls present with non-ToolUse stop_reason are still dispatched.
func TestToolCallsDispatchedEvenWhenStopIsEndTurn(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content: []ContentBlock{
			NewToolUseBlock(ToolCall{ID: "x", Name: "noop", Input: json.RawMessage(`{}`)}),
		},
		Usage:      usage(1, 1),
		StopReason: StopEndTurn,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnToolCallRequested || len(r.Calls) != 1 {
		t.Fatalf("got %+v", r)
	}
}

// Agent identity is reported.
func TestAgentIDReported(t *testing.T) {
	m := NewMockModel(agentProvider())
	agent := NewModelAgent(AgentID("initializer"), m)
	if agent.ID() != AgentID("initializer") {
		t.Fatalf("id = %q", agent.ID())
	}
}

// MaxTokens stop classifies as FinalResponse with text.
func TestMaxTokensStopClassifiesAsFinalResponse(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{NewTextBlock("truncated")},
		Usage:      usage(2, 5),
		StopReason: StopMaxTokens,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnFinalResponse || r.Content != "truncated" {
		t.Fatalf("got %+v", r)
	}
}

func TestStopSequenceClassifiesAsFinalResponse(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{NewTextBlock("done.")},
		Usage:      usage(2, 1),
		StopReason: StopStopSequence,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnFinalResponse {
		t.Fatalf("got %+v", r)
	}
}

// Multiple text blocks are concatenated.
func TestMultipleTextBlocksAreConcatenated(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushResponse(ModelResponse{
		Content:    []ContentBlock{NewTextBlock("foo"), NewTextBlock("bar")},
		Usage:      usage(1, 1),
		StopReason: StopEndTurn,
	})
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("?"))
	if r.Kind != TurnFinalResponse || r.Content != "foobar" {
		t.Fatalf("got %+v", r)
	}
}

// JSON round-trip for cross-language fixture portability.
func TestTurnResultRoundtripJSON(t *testing.T) {
	cases := []TurnResult{
		NewToolCallRequested(
			[]ToolCall{{ID: "1", Name: "x", Input: json.RawMessage(`{"a":1}`)}},
			usage(2, 3),
		),
		NewFinalResponse("hi", usage(1, 1)),
		NewTurnError(NewEmptyResponseError(), nil),
		NewTurnError(NewMalformedToolCallError("foo", "bad"), &TokenUsage{InputTokens: 2}),
		NewTurnError(NewModelAgentError(NewTimeout()), nil),
	}
	for i, r := range cases {
		data, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("case %d: marshal: %v", i, err)
		}
		var back TurnResult
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("case %d: unmarshal %s: %v", i, data, err)
		}
		if back.Kind != r.Kind {
			t.Fatalf("case %d: kind mismatch %q vs %q", i, back.Kind, r.Kind)
		}
		// Re-marshal and compare bytes to detect representation drift.
		again, err := json.Marshal(back)
		if err != nil {
			t.Fatalf("case %d: re-marshal: %v", i, err)
		}
		if string(again) != string(data) {
			t.Fatalf("case %d: round-trip drift\n a=%s\n b=%s", i, data, again)
		}
	}
}

// JSON tag values match Rust serde_snake_case.
func TestTurnResultJSONTagsSnakeCase(t *testing.T) {
	r := NewFinalResponse("ok", usage(1, 1))
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var probe map[string]interface{}
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatal(err)
	}
	if probe["kind"] != "final_response" {
		t.Fatalf("kind tag = %v", probe["kind"])
	}
}

// AgentError variant tags match Rust.
func TestAgentErrorJSONShape(t *testing.T) {
	e := NewMalformedToolCallError("x", "y")
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"kind":"MalformedToolCall","tool_name":"x","reason":"y"}`
	if string(data) != want {
		t.Fatalf("got %s", data)
	}

	// EmptyResponse
	e2 := NewEmptyResponseError()
	data2, _ := json.Marshal(e2)
	if string(data2) != `{"kind":"EmptyResponse"}` {
		t.Fatalf("empty: %s", data2)
	}

	// ModelError (transparent — inner kind takes over)
	e3 := NewModelAgentError(NewTimeout())
	data3, _ := json.Marshal(e3)
	if string(data3) != `{"kind":"Timeout"}` {
		t.Fatalf("model: %s", data3)
	}
}

// Context.IntoRequest preserves messages/tools/params and sets stream=false.
func TestContextIntoRequest(t *testing.T) {
	c := userCtx("hello")
	req := c.IntoRequest()
	if len(req.Messages) != 1 || req.Stream {
		t.Fatalf("req = %+v", req)
	}
}

// AgentError implements the error interface and is non-empty for every kind.
func TestAgentErrorString(t *testing.T) {
	variants := []*AgentError{
		NewModelAgentError(NewTimeout()),
		NewEmptyResponseError(),
		NewMalformedToolCallError("x", "y"),
	}
	for _, v := range variants {
		if v.Error() == "" {
			t.Fatalf("empty Error() for kind %q", v.Kind)
		}
	}
}

// Non-ModelError Go errors from the model are wrapped into ProviderError.
func TestNonModelErrorWrappedAsProviderError(t *testing.T) {
	m := NewMockModel(agentProvider())
	m.PushError(errors.New("boom"))
	agent := makeAgent(m)
	r := agent.Turn(context.Background(), userCtx("hi"))
	if r.Kind != TurnError || r.Err.Kind != AgentErrModelError {
		t.Fatalf("got %+v", r)
	}
	if r.Err.ModelError == nil || r.Err.ModelError.Kind != ModelErrProviderError {
		t.Fatalf("inner = %+v", r.Err.ModelError)
	}
	if r.Err.ModelError.Message != "boom" {
		t.Fatalf("msg = %q", r.Err.ModelError.Message)
	}
}
