package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
)

// Adaptive prompt-based tool-calling fallback (#111). These unit tests mirror
// the inline tests in the Rust reference (prompt_tool_call.rs and
// tool_call_repair.rs's prose-detection tests).

func ptcProvider() ProviderInfo {
	return ProviderInfo{Name: "test", ModelID: "test-1", ContextWindow: 4096}
}

func ptcToolSchema() ToolSchema {
	return ToolSchema{
		Name:        "calculator",
		Description: "evaluate math",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
	}
}

func ptcReqWithTools(system string, hasSystem bool) ModelRequest {
	var messages []Message
	if hasSystem {
		messages = append(messages, Message{Role: RoleSystem, Content: NewTextContent(system)})
	}
	messages = append(messages, Message{Role: RoleUser, Content: NewTextContent("what is 2+2?")})
	return ModelRequest{Messages: messages, Tools: []ToolSchema{ptcToolSchema()}}
}

func ptcUsage() TokenUsage {
	return TokenUsage{InputTokens: 1, OutputTokens: 1}
}

func ptcProse(text string, stop StopReason) ModelResponse {
	return ModelResponse{
		Content:    []ContentBlock{NewTextBlock(text)},
		Usage:      ptcUsage(),
		StopReason: stop,
	}
}

func sysText(t *testing.T, req ModelRequest) string {
	t.Helper()
	if req.Messages[0].Role != RoleSystem || req.Messages[0].Content.Type != ContentTypeText {
		t.Fatalf("messages[0] is not a system text message: %+v", req.Messages[0])
	}
	return req.Messages[0].Content.Text
}

// --- injection --------------------------------------------------------------

func TestInjectionAppendsToExistingSystemPrompt(t *testing.T) {
	req := ptcReqWithTools("You are a helpful assistant.", true)
	InjectToolPrompt(&req)
	sys := sysText(t, req)
	if !strings.HasPrefix(sys, "You are a helpful assistant.") {
		t.Fatalf("system prompt did not retain original prefix: %q", sys)
	}
	if !strings.Contains(sys, "<available_tools>") {
		t.Fatalf("missing <available_tools> block")
	}
	if !strings.Contains(sys, "<name>calculator</name>") {
		t.Fatalf("missing tool name")
	}
	if req.Messages[1].Role != RoleUser {
		t.Fatalf("user message not preserved after system message")
	}
}

func TestInjectionInsertsSystemPromptWhenAbsent(t *testing.T) {
	req := ptcReqWithTools("", false)
	InjectToolPrompt(&req)
	if req.Messages[0].Role != RoleSystem {
		t.Fatalf("expected inserted system message at index 0")
	}
	if !strings.Contains(sysText(t, req), "<available_tools>") {
		t.Fatalf("inserted system message missing block")
	}
}

func TestInjectionIsIdempotent(t *testing.T) {
	req := ptcReqWithTools("base", true)
	InjectToolPrompt(&req)
	once := sysText(t, req)
	InjectToolPrompt(&req)
	twice := sysText(t, req)
	if once != twice {
		t.Fatalf("second injection must be a no-op:\nonce=%q\ntwice=%q", once, twice)
	}
}

func TestInjectionNoopWithoutTools(t *testing.T) {
	req := ptcReqWithTools("base", true)
	req.Tools = nil
	before := len(req.Messages)
	beforeText := req.Messages[0].Content.Text
	InjectToolPrompt(&req)
	if len(req.Messages) != before || req.Messages[0].Content.Text != beforeText {
		t.Fatalf("injection must be a no-op when no tools are advertised")
	}
}

// --- parsing ----------------------------------------------------------------

func toolUseBlocks(content []ContentBlock) []*ToolCall {
	var out []*ToolCall
	for _, b := range content {
		if b.Type == ContentBlockTypeToolUse {
			out = append(out, b.ToolCall)
		}
	}
	return out
}

func TestParsesSingleToolCallMarker(t *testing.T) {
	resp := ptcProse(`<tool_call><name>calculator</name><input>{"expression": "2+2"}</input></tool_call>`, StopEndTurn)
	out := ParseProseResponse(resp)
	if out.StopReason != StopToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", out.StopReason)
	}
	tus := toolUseBlocks(out.Content)
	if len(tus) != 1 {
		t.Fatalf("expected one tool-use block, got %d", len(tus))
	}
	if tus[0].Name != "calculator" {
		t.Fatalf("name = %q", tus[0].Name)
	}
	if tus[0].ID != "ptc_call_0" {
		t.Fatalf("id = %q, want ptc_call_0", tus[0].ID)
	}
	var got map[string]any
	_ = json.Unmarshal(tus[0].Input, &got)
	if got["expression"] != "2+2" {
		t.Fatalf("input = %s", tus[0].Input)
	}
}

func TestParsesMultipleToolCallMarkers(t *testing.T) {
	text := `<tool_call><name>a</name><input>{"x":1}</input></tool_call>` + "\n" +
		"some chatter\n" +
		`<tool_call><name>b</name><input>{"y":2}</input></tool_call>`
	out := ParseProseResponse(ptcProse(text, StopEndTurn))
	tus := toolUseBlocks(out.Content)
	if len(tus) != 2 || tus[0].Name != "a" || tus[1].Name != "b" {
		t.Fatalf("names = %v, want [a b]", tus)
	}
	if tus[0].ID != "ptc_call_0" || tus[1].ID != "ptc_call_1" {
		t.Fatalf("ids = %q,%q want ptc_call_0,ptc_call_1", tus[0].ID, tus[1].ID)
	}
	if out.StopReason != StopToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", out.StopReason)
	}
}

func TestMalformedInputJSONFallsThroughAsProse(t *testing.T) {
	resp := ptcProse(`<tool_call><name>calculator</name><input>{not valid json}</input></tool_call>`, StopEndTurn)
	out := ParseProseResponse(resp)
	if out.StopReason != StopEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn (unchanged)", out.StopReason)
	}
	if out.Content[0].Type != ContentBlockTypeText {
		t.Fatalf("content[0] = %q, want text", out.Content[0].Type)
	}
}

func TestPlainProseReturnedAsIs(t *testing.T) {
	resp := ptcProse("The answer is 4.", StopEndTurn)
	out := ParseProseResponse(resp)
	if out.StopReason != StopEndTurn || len(out.Content) != 1 || out.Content[0].Text != "The answer is 4." {
		t.Fatalf("plain prose was modified: %+v", out)
	}
}

func TestNativeToolUseLeftUntouched(t *testing.T) {
	resp := ModelResponse{
		Content: []ContentBlock{NewToolUseBlock(ToolCall{
			ID: "native", Name: "calculator", Input: json.RawMessage(`{"expression":"1"}`),
		})},
		Usage:      ptcUsage(),
		StopReason: StopToolUse,
	}
	out := ParseProseResponse(resp)
	tus := toolUseBlocks(out.Content)
	if len(tus) != 1 || tus[0].ID != "native" {
		t.Fatalf("native tool-use block was rewritten: %+v", out.Content)
	}
}

func TestThinkingBlocksPreservedAlongsideSynthesizedCalls(t *testing.T) {
	resp := ModelResponse{
		Content: []ContentBlock{
			NewThinkingBlock("reasoning"),
			NewTextBlock(`<tool_call><name>t</name><input>{}</input></tool_call>`),
		},
		Usage:      ptcUsage(),
		StopReason: StopEndTurn,
	}
	out := ParseProseResponse(resp)
	if len(out.Content) != 2 {
		t.Fatalf("expected thinking + tool-use, got %d blocks", len(out.Content))
	}
	if out.Content[0].Type != ContentBlockTypeThinking {
		t.Fatalf("content[0] = %q, want thinking", out.Content[0].Type)
	}
	if out.Content[1].Type != ContentBlockTypeToolUse {
		t.Fatalf("content[1] = %q, want tool_use", out.Content[1].Type)
	}
}

// --- always-on wrapper ------------------------------------------------------

func TestAlwaysOnWrapperInjectsAndParses(t *testing.T) {
	m := NewMockModel(ptcProvider())
	m.PushResponse(ptcProse(`<tool_call><name>calculator</name><input>{"expression":"2+2"}</input></tool_call>`, StopEndTurn))
	wrapper := NewPromptBasedToolCallModelInterface(m)
	resp, err := wrapper.Call(context.Background(), ptcReqWithTools("base", true))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StopReason != StopToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", resp.StopReason)
	}
	if resp.Content[0].Type != ContentBlockTypeToolUse {
		t.Fatalf("content[0] = %q, want tool_use", resp.Content[0].Type)
	}
}

// --- adaptive wrapper -------------------------------------------------------

func TestAdaptiveWrapperDelegatesNativelyWhenFlagUnset(t *testing.T) {
	m := NewMockModel(ptcProvider())
	// A prose response containing a marker — but the flag is OFF, so the wrapper
	// must NOT parse it; it delegates verbatim.
	m.PushResponse(ptcProse(`<tool_call><name>x</name><input>{}</input></tool_call>`, StopEndTurn))
	flag := &atomic.Bool{}
	wrapper := NewAdaptiveToolCallModelInterface(m, flag)
	resp, err := wrapper.Call(context.Background(), ptcReqWithTools("base", true))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StopReason != StopEndTurn {
		t.Fatalf("stop_reason = %q, want end_turn (untouched)", resp.StopReason)
	}
	if resp.Content[0].Type != ContentBlockTypeText {
		t.Fatalf("content[0] = %q, want text (untouched)", resp.Content[0].Type)
	}
}

func TestAdaptiveWrapperParsesWhenFlagSet(t *testing.T) {
	m := NewMockModel(ptcProvider())
	m.PushResponse(ptcProse(`<tool_call><name>x</name><input>{"k":1}</input></tool_call>`, StopEndTurn))
	flag := &atomic.Bool{}
	flag.Store(true)
	wrapper := NewAdaptiveToolCallModelInterface(m, flag)
	resp, err := wrapper.Call(context.Background(), ptcReqWithTools("base", true))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StopReason != StopToolUse {
		t.Fatalf("stop_reason = %q, want tool_use", resp.StopReason)
	}
	tus := toolUseBlocks(resp.Content)
	if len(tus) != 1 || tus[0].Name != "x" {
		t.Fatalf("expected tool use named x, got %+v", resp.Content)
	}
	var got map[string]any
	_ = json.Unmarshal(tus[0].Input, &got)
	if got["k"].(float64) != 1 {
		t.Fatalf("input = %s", tus[0].Input)
	}
}

func TestAdaptiveWrapperProviderDelegates(t *testing.T) {
	m := NewMockModel(ptcProvider())
	flag := &atomic.Bool{}
	wrapper := NewAdaptiveToolCallModelInterface(m, flag)
	if wrapper.Provider().ModelID != "test-1" {
		t.Fatalf("provider model id = %q, want test-1", wrapper.Provider().ModelID)
	}
}

// --- prose detection --------------------------------------------------------

func TestProseDetectedOnActionIntentWithTools(t *testing.T) {
	if _, ok := DetectProseResponse("Sure, I'll use the calculator tool to add these.", true); !ok {
		t.Fatalf("expected prose detection on action intent")
	}
}

func TestProseDetectedCaseInsensitive(t *testing.T) {
	if _, ok := DetectProseResponse("LET ME CALL the search tool now.", true); !ok {
		t.Fatalf("expected case-insensitive prose detection")
	}
}

func TestProseNotDetectedWithoutToolsAdvertised(t *testing.T) {
	if _, ok := DetectProseResponse("I'll use the calculator.", false); ok {
		t.Fatalf("must not detect prose when no tools advertised")
	}
}

func TestProseNotDetectedForPlainFinalAnswer(t *testing.T) {
	if _, ok := DetectProseResponse("The answer is 42.", true); ok {
		t.Fatalf("plain final answer must not trip the heuristic")
	}
}

func TestProseNotDetectedForEmptyText(t *testing.T) {
	if _, ok := DetectProseResponse("   ", true); ok {
		t.Fatalf("empty text must not trip the heuristic")
	}
}
