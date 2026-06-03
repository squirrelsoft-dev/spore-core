// Prompt-based tool calling — an adaptive fallback for models that do not
// reliably emit native tool calls.
//
// Some models (small local ones especially) respond in prose even when a tool
// call is the right action. Rather than maintaining a list of known-bad models
// or asking callers to wrap them manually, the harness discovers this at
// runtime: native tool calling is tried first, and when a turn comes back as
// prose while tools were advertised (see DetectProseResponse), the harness
// flips a session-scoped flag that activates PromptBasedToolCallModelInterface
// for the rest of the run.
//
// Two wrappers:
//
//   - PromptBasedToolCallModelInterface — an *always-on* transparent wrapper.
//     It injects a tool-definition block into the system prompt and parses
//     <tool_call> markers out of the model's text response into native
//     ToolCalls. Construct it directly for advanced use.
//   - AdaptiveToolCallModelInterface — a *flag-gated* wrapper installed
//     automatically by the conversational preset. While its shared flag is
//     unset it delegates natively (byte-for-byte); once the harness sets the
//     flag it behaves exactly like the always-on wrapper.
//
// Both share the free functions InjectToolPrompt and ParseProseResponse so
// injection and parsing can never diverge between them. Injection is
// idempotent — double-wrapping (e.g. an Adaptive around a PromptBased) never
// appends the block twice.
//
// Streaming: CallStreaming buffers the full inner stream, parses it for
// markers, then re-emits the reconstructed response as a stream. Streaming and
// marker parsing do not compose cleanly; buffering is the accepted trade-off.
//
// Cross-language note: the wire constants (the injected prompt block, the
// <tool_call> marker grammar, the action-intent phrases, and the NUDGE) are
// mirrored byte-for-byte against the Rust reference
// (rust/crates/spore-core/src/prompt_tool_call.rs and tool_call_repair.rs).

package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"
)

// toolsBlockOpen is the sentinel that marks an already-injected tool-prompt
// block. Used to make InjectToolPrompt idempotent.
const toolsBlockOpen = "<available_tools>"

// PromptToolCallNudge is the user-facing nudge appended after a prose response
// is detected, asking the model to use the tool-call format. Byte-identical to
// the Rust reference.
const PromptToolCallNudge = "You described an action but did not call a tool. Use the provided tool-call format to actually invoke the tool."

// toolCallMarkerRe matches a <tool_call> marker in model text. Non-greedy, with
// the `s` flag so `.` spans newlines (RE2 equivalent of Rust's `(?s)`).
// Compiled once at package load.
var toolCallMarkerRe = regexp.MustCompile(`(?s)<tool_call>\s*<name>(.*?)</name>\s*<input>(.*?)</input>\s*</tool_call>`)

// ============================================================================
// System-prompt injection
// ============================================================================

// buildToolPrompt renders the tool-definition + response-format block appended
// to the system prompt when prompt-based tool calling is active. The schema is
// rendered as compact JSON. Byte-identical to Rust's build_tool_prompt.
func buildToolPrompt(tools []ToolSchema) string {
	var s strings.Builder
	s.WriteString("You have access to the following tools. Use them when they would help complete the task.\n\n")
	s.WriteString("<available_tools>\n")
	for _, tool := range tools {
		s.WriteString("<tool>\n")
		s.WriteString(fmt.Sprintf("  <name>%s</name>\n", tool.Name))
		s.WriteString(fmt.Sprintf("  <description>%s</description>\n", tool.Description))
		schemaJSON := compactSchemaJSON(tool.InputSchema)
		s.WriteString(fmt.Sprintf("  <input_schema>%s</input_schema>\n", schemaJSON))
		s.WriteString("</tool>\n")
	}
	s.WriteString("</available_tools>\n\n")
	s.WriteString("When you want to use a tool, respond with ONLY the following format and nothing else:\n")
	s.WriteString("<tool_call>\n  <name>tool_name_here</name>\n  <input>{\"key\": \"value\"}</input>\n</tool_call>\n\n")
	s.WriteString("When you have a final answer that does not require a tool, respond normally in prose.")
	return s.String()
}

// compactSchemaJSON re-marshals a tool's input schema to compact JSON so the
// rendered block is stable regardless of how the schema was stored. Mirrors
// Rust's serde_json::to_string(&tool.input_schema), falling back to "{}".
func compactSchemaJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return "{}"
	}
	out, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(out)
}

// InjectToolPrompt appends the tool-definition block to a request's system
// prompt, in place.
//
//   - No-op when the request advertises no tools (nothing to describe).
//   - Idempotent: if a leading System text message already contains the
//     toolsBlockOpen sentinel, nothing is appended (so wrapping a wrapper does
//     not double-inject).
//   - Appends to an existing leading System text message when present,
//     otherwise inserts a new one at the front — never clobbering the caller's
//     system prompt.
func InjectToolPrompt(request *ModelRequest) {
	if len(request.Tools) == 0 {
		return
	}
	block := buildToolPrompt(request.Tools)

	if len(request.Messages) > 0 {
		first := &request.Messages[0]
		if first.Role == RoleSystem && first.Content.Type == ContentTypeText {
			if strings.Contains(first.Content.Text, toolsBlockOpen) {
				return // already injected — idempotent.
			}
			first.Content.Text += "\n\n" + block
			return
		}
	}

	request.Messages = append([]Message{{
		Role:    RoleSystem,
		Content: NewTextContent(block),
	}}, request.Messages...)
}

// ============================================================================
// Response parsing
// ============================================================================

// parsedToolCall is a (name, input) pair extracted from a marker.
type parsedToolCall struct {
	name  string
	input json.RawMessage
}

// extractToolCalls pulls (name, input) pairs from <tool_call> markers in model
// text. Markers whose <input> is not valid JSON are skipped (the caller decides
// what to do when nothing parses). Supports multiple markers in one response.
func extractToolCalls(text string) []parsedToolCall {
	matches := toolCallMarkerRe.FindAllStringSubmatch(text, -1)
	out := make([]parsedToolCall, 0, len(matches))
	for _, m := range matches {
		name := strings.TrimSpace(m[1])
		rawInput := strings.TrimSpace(m[2])
		if name == "" {
			continue
		}
		var probe any
		if err := json.Unmarshal([]byte(rawInput), &probe); err != nil {
			// Malformed input JSON: skip this marker. If no marker parses, the
			// whole response falls through as prose (graceful degradation).
			continue
		}
		out = append(out, parsedToolCall{name: name, input: json.RawMessage(rawInput)})
	}
	return out
}

// ParseProseResponse rewrites a model response so <tool_call> markers in its
// text become native ToolUse content blocks.
//
//   - If the response already carries native tool-use blocks, it is returned
//     unchanged (native tool calling succeeded — never second-guess it).
//   - Otherwise text blocks are scanned for markers. When at least one parses,
//     the response's content becomes any Thinking blocks followed by the
//     synthesized tool-use blocks, and StopReason becomes StopToolUse.
//   - When no marker parses, the response is returned unchanged (prose as-is).
func ParseProseResponse(response ModelResponse) ModelResponse {
	for _, b := range response.Content {
		if b.Type == ContentBlockTypeToolUse {
			return response
		}
	}

	var sb strings.Builder
	for _, b := range response.Content {
		if b.Type == ContentBlockTypeText {
			sb.WriteString(b.Text)
		}
	}

	parsed := extractToolCalls(sb.String())
	if len(parsed) == 0 {
		return response // no tool markers — genuine prose response.
	}

	// Preserve reasoning, replace text with synthesized tool-use blocks.
	content := make([]ContentBlock, 0, len(response.Content)+len(parsed))
	for _, b := range response.Content {
		if b.Type == ContentBlockTypeThinking {
			content = append(content, b)
		}
	}
	for i, pc := range parsed {
		content = append(content, NewToolUseBlock(ToolCall{
			ID:    fmt.Sprintf("ptc_call_%d", i),
			Name:  pc.name,
			Input: pc.input,
		}))
	}

	return ModelResponse{
		Content:    content,
		Usage:      response.Usage,
		StopReason: StopToolUse,
	}
}

// ============================================================================
// Prose detection
// ============================================================================

// promptToolCallActionPhrases are curated action-intent phrases. A lower-cased
// substring match against any of these (with tools advertised) classifies a
// response as prose where a tool call was expected. Byte-identical to the Rust
// reference's ACTION_PHRASES.
var promptToolCallActionPhrases = []string{
	"i'll use",
	"i will use",
	"i'll call",
	"i will call",
	"i'll run",
	"i will run",
	"let me use",
	"let me call",
	"let me run",
	"i need to use",
	"i need to call",
	"i should use",
	"i should call",
	"i'll invoke",
	"i will invoke",
	"using the",
	"i can use the",
	"i'm going to use",
	"i am going to use",
}

// DetectProseResponse is a conservative heuristic: did the model respond in
// prose when a tool call was the expected next step?
//
// Returns (trimmedText, true) only when BOTH:
//  1. tools were advertised this turn (toolsAdvertised), and
//  2. the response text contains an explicit action-intent phrase suggesting
//     the model meant to act (e.g. "I'll use the … tool", "let me call …").
//
// The bias is deliberately toward false negatives: a missed prose response
// costs one extra turn, but a false positive activates prompt-based mode for a
// model that was simply giving a final answer. A bare final answer with no
// action-intent language is therefore NOT classified as a prose response.
func DetectProseResponse(text string, toolsAdvertised bool) (string, bool) {
	if !toolsAdvertised {
		return "", false
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return "", false
	}
	lower := strings.ToLower(trimmed)
	for _, p := range promptToolCallActionPhrases {
		if strings.Contains(lower, p) {
			return trimmed, true
		}
	}
	return "", false
}

// ============================================================================
// Stream buffering helpers
// ============================================================================

// bufBlock is one accumulating block in a buffered stream.
type bufBlock struct {
	kind ContentBlockType
	text string
	id   string
	name string
	json string
}

// responseBuffer reassembles a ModelResponse from a buffered stream of
// StreamEvents. Mirrors the agent's accumulator, kept local so this module owns
// its own buffering for the parse-then-re-emit path.
type responseBuffer struct {
	order  []uint32
	blocks map[uint32]*bufBlock
	usage  TokenUsage
	stop   StopReason
	hasTop bool
}

func newResponseBuffer() *responseBuffer {
	return &responseBuffer{blocks: make(map[uint32]*bufBlock)}
}

func (rb *responseBuffer) entry(index uint32, kind ContentBlockType) *bufBlock {
	if b, ok := rb.blocks[index]; ok {
		return b
	}
	b := &bufBlock{kind: kind}
	rb.blocks[index] = b
	rb.order = append(rb.order, index)
	return b
}

func (rb *responseBuffer) fold(ev StreamEvent) {
	switch ev.Type {
	case StreamMessageStart:
	case StreamContentBlockDelta:
		rb.entry(ev.Index, ContentBlockTypeText).text += ev.Delta
	case StreamThinkingDelta:
		rb.entry(ev.Index, ContentBlockTypeThinking).text += ev.Delta
	case StreamToolUseStart:
		b := rb.entry(ev.Index, ContentBlockTypeToolUse)
		b.kind = ContentBlockTypeToolUse
		b.id = ev.ID
		b.name = ev.Name
	case StreamToolUseDelta:
		b := rb.entry(ev.Index, ContentBlockTypeToolUse)
		b.kind = ContentBlockTypeToolUse
		b.json += ev.PartialJSON
	case StreamContentBlockStop:
	case StreamMessageStop:
		if ev.Usage != nil {
			rb.usage = *ev.Usage
		}
		rb.stop = ev.StopReason
		rb.hasTop = true
	}
}

func (rb *responseBuffer) intoResponse() ModelResponse {
	content := make([]ContentBlock, 0, len(rb.order))
	for _, idx := range rb.order {
		b := rb.blocks[idx]
		switch b.kind {
		case ContentBlockTypeText:
			content = append(content, NewTextBlock(b.text))
		case ContentBlockTypeThinking:
			content = append(content, NewThinkingBlock(b.text))
		case ContentBlockTypeToolUse:
			input := json.RawMessage(b.json)
			if len(input) == 0 {
				input = json.RawMessage("null")
			}
			id := b.id
			if id == "" {
				id = fmt.Sprintf("call_%d", idx)
			}
			content = append(content, NewToolUseBlock(ToolCall{ID: id, Name: b.name, Input: input}))
		}
	}
	stop := rb.stop
	if !rb.hasTop {
		stop = StopEndTurn
	}
	return ModelResponse{Content: content, Usage: rb.usage, StopReason: stop}
}

// responseToStream re-emits a ModelResponse as a channel of StreamEvents the
// harness accumulator understands. The inverse of responseBuffer.
func responseToStream(response ModelResponse) <-chan StreamEventOrErr {
	ch := make(chan StreamEventOrErr, len(response.Content)*3+2)
	ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamMessageStart}}
	for i, block := range response.Content {
		idx := uint32(i)
		switch block.Type {
		case ContentBlockTypeText:
			ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamContentBlockDelta, Index: idx, Delta: block.Text}}
		case ContentBlockTypeThinking:
			ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamThinkingDelta, Index: idx, Delta: block.Text}}
		case ContentBlockTypeToolUse:
			var id, name string
			partial := "{}"
			if block.ToolCall != nil {
				id = block.ToolCall.ID
				name = block.ToolCall.Name
				if len(block.ToolCall.Input) > 0 {
					partial = string(block.ToolCall.Input)
				}
			}
			ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamToolUseStart, Index: idx, ID: id, Name: name}}
			ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamToolUseDelta, Index: idx, PartialJSON: partial}}
		}
		ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamContentBlockStop, Index: idx}}
	}
	usage := response.Usage
	ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamMessageStop, Usage: &usage, StopReason: response.StopReason}}
	close(ch)
	return ch
}

// streamingPromptCall is the shared streaming path: inject, buffer the inner
// stream, parse, re-emit.
func streamingPromptCall(ctx context.Context, inner ModelInterface, request ModelRequest) (<-chan StreamEventOrErr, error) {
	InjectToolPrompt(&request)
	stream, err := inner.CallStreaming(ctx, request)
	if err != nil {
		return nil, err
	}
	buf := newResponseBuffer()
	for item := range stream {
		if item.Err != nil {
			return nil, item.Err
		}
		buf.fold(item.Event)
	}
	parsed := ParseProseResponse(buf.intoResponse())
	return responseToStream(parsed), nil
}

// ============================================================================
// PromptBasedToolCallModelInterface — always-on wrapper
// ============================================================================

// PromptBasedToolCallModelInterface is a transparent, *always-on* prompt-based
// tool-calling wrapper around any ModelInterface.
//
// Every call injects the tool-definition block into the system prompt and
// parses <tool_call> markers from the response into native ToolCalls.
// CountTokens and Provider delegate to the inner model unchanged.
type PromptBasedToolCallModelInterface struct {
	inner ModelInterface
}

// NewPromptBasedToolCallModelInterface wraps a model so it always uses
// prompt-based tool calling.
func NewPromptBasedToolCallModelInterface(inner ModelInterface) *PromptBasedToolCallModelInterface {
	return &PromptBasedToolCallModelInterface{inner: inner}
}

// Inner returns the wrapped model.
func (m *PromptBasedToolCallModelInterface) Inner() ModelInterface { return m.inner }

// Call injects the tool prompt, calls the inner model, and parses the response.
func (m *PromptBasedToolCallModelInterface) Call(ctx context.Context, request ModelRequest) (ModelResponse, error) {
	InjectToolPrompt(&request)
	resp, err := m.inner.Call(ctx, request)
	if err != nil {
		return ModelResponse{}, err
	}
	return ParseProseResponse(resp), nil
}

// CallStreaming buffers the inner stream, parses it, then re-emits.
func (m *PromptBasedToolCallModelInterface) CallStreaming(ctx context.Context, request ModelRequest) (<-chan StreamEventOrErr, error) {
	return streamingPromptCall(ctx, m.inner, request)
}

// CountTokens delegates to the inner model.
func (m *PromptBasedToolCallModelInterface) CountTokens(ctx context.Context, request ModelRequest) (uint32, error) {
	return m.inner.CountTokens(ctx, request)
}

// Provider delegates to the inner model.
func (m *PromptBasedToolCallModelInterface) Provider() ProviderInfo { return m.inner.Provider() }

// ============================================================================
// AdaptiveToolCallModelInterface — flag-gated wrapper
// ============================================================================

// AdaptiveToolCallModelInterface is a flag-gated prompt-based wrapper. While
// flag is false it delegates to the inner model byte-for-byte (native tool
// calling). Once the harness sets flag — on detecting a prose response where a
// tool call was expected — it behaves exactly like
// PromptBasedToolCallModelInterface for the rest of the run.
//
// Installed automatically by the conversational preset; the harness holds the
// SAME *atomic.Bool and flips it from the run loop. Not normally constructed
// directly.
type AdaptiveToolCallModelInterface struct {
	inner ModelInterface
	flag  *atomic.Bool
}

// NewAdaptiveToolCallModelInterface wraps inner, sharing flag with the harness
// loop.
func NewAdaptiveToolCallModelInterface(inner ModelInterface, flag *atomic.Bool) *AdaptiveToolCallModelInterface {
	return &AdaptiveToolCallModelInterface{inner: inner, flag: flag}
}

// IsActive reports whether prompt-based mode has been activated for the run.
func (m *AdaptiveToolCallModelInterface) IsActive() bool { return m.flag.Load() }

// Call delegates natively while the flag is unset; otherwise injects + parses.
func (m *AdaptiveToolCallModelInterface) Call(ctx context.Context, request ModelRequest) (ModelResponse, error) {
	if !m.flag.Load() {
		return m.inner.Call(ctx, request)
	}
	InjectToolPrompt(&request)
	resp, err := m.inner.Call(ctx, request)
	if err != nil {
		return ModelResponse{}, err
	}
	return ParseProseResponse(resp), nil
}

// CallStreaming delegates natively while the flag is unset; otherwise buffers,
// parses, and re-emits.
func (m *AdaptiveToolCallModelInterface) CallStreaming(ctx context.Context, request ModelRequest) (<-chan StreamEventOrErr, error) {
	if !m.flag.Load() {
		return m.inner.CallStreaming(ctx, request)
	}
	return streamingPromptCall(ctx, m.inner, request)
}

// CountTokens delegates to the inner model.
func (m *AdaptiveToolCallModelInterface) CountTokens(ctx context.Context, request ModelRequest) (uint32, error) {
	return m.inner.CountTokens(ctx, request)
}

// Provider delegates to the inner model.
func (m *AdaptiveToolCallModelInterface) Provider() ProviderInfo { return m.inner.Provider() }

// Compile-time interface checks.
var (
	_ ModelInterface = (*PromptBasedToolCallModelInterface)(nil)
	_ ModelInterface = (*AdaptiveToolCallModelInterface)(nil)
)
