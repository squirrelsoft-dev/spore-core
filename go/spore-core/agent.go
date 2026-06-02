// Agent — executes a single turn against a ModelInterface.
//
// Implements issue #2. The agent is one turn: it accepts a fully assembled
// Context (produced upstream by the ContextManager, issue #7) and returns a
// TurnResult classifying what the model wants to do next.
//
// What this component does:
//   - Translate Context → ModelRequest
//   - Invoke ModelInterface.Call
//   - Classify the response as ToolCallRequested, FinalResponse, or Error
//
// What this component does NOT do:
//   - Assemble context (the ContextManager's job — issue #7)
//   - Execute tool calls (the harness loop dispatches via ToolRegistry — #3, #4)
//   - Validate tool call parameters against tool schemas (ToolRegistry — #4)
//   - Decide termination (TerminationPolicy — #13)
//   - Retry on transient errors (lives in the ModelInterface implementation)
//
// Rules enforced here:
//
//  1. One call to Agent.Turn performs exactly one model call.
//  2. ToolCallRequested may carry multiple tool calls (parallel tool use).
//  3. A response with neither text nor tool calls is reported as
//     AgentError EmptyResponse — never silently swallowed.
//  4. Classification uses the model's stop_reason:
//     - StopToolUse with tool_use blocks → ToolCallRequested
//     - StopToolUse without tool_use blocks → MalformedToolCall
//     - StopEndTurn|MaxTokens|StopSequence with tool_use blocks →
//     still ToolCallRequested (do not silently drop tool calls)
//     - StopEndTurn|MaxTokens|StopSequence with text → FinalResponse
//     (concatenated text blocks; Thinking is accumulated into Reasoning, not
//     discarded — issue #103, Q4)
//     - StopEndTurn|MaxTokens|StopSequence with neither → EmptyResponse
//     (thinking-only output is still empty: thinking is not a terminal response)
//  5. ModelError is surfaced wrapped in AgentError ModelError, with no
//     partial usage information (usage is nil).
//
// Delta-level streaming (issue #103):
//
//   - AgentStreamSink is an owned callback of RAW model-layer StreamEvents
//     (Q1): the agent emits raw model events and does NOT depend on the harness
//     HarnessStreamEvent type. The harness owns the model→harness mapping.
//   - StreamingAgent is the optional interface a model-backed agent implements
//     to support streaming; TurnStreamingOrDelegate dispatches to it or falls
//     back to plain Turn (ignoring the sink). Go interfaces cannot carry default
//     methods, so this optional-interface seam (matching AssistantMessageAppender
//     / TokenBudgetReader in this package) provides the default-delegates-to-Turn
//     behaviour without forcing every existing Agent impl to add a method.
//   - TurnResult.Reasoning carries accumulated thinking text (Q4). Thinking is
//     streamed as raw ThinkingDelta events AND returned here; it is NOT added as
//     a content variant nor preserved in SessionState (deferred to issue #104).
//   - Turn and TurnStreaming share classifyResponse so classification can never
//     diverge between the blocking and streaming paths.
//
// Cross-language note: TurnResult, AgentError, and Context shapes are
// mirrored byte-for-byte in the Rust / TypeScript / Python packages. The
// Rust implementation is the spec reference.

package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// ============================================================================
// Identity
// ============================================================================

// AgentID is a caller-assigned agent configuration name, used to correlate
// turns in traces when multiple agents run in the same session.
type AgentID string

// ============================================================================
// Context — the assembled input handed to the agent
// ============================================================================

// Context is the fully assembled per-turn input produced by the
// ContextManager (issue #7). The agent never modifies it.
type Context struct {
	Messages []Message    `json:"messages"`
	Tools    []ToolSchema `json:"tools"`
	Params   ModelParams  `json:"params"`
}

// MarshalJSON ensures slice fields serialise as [] rather than null.
func (c Context) MarshalJSON() ([]byte, error) {
	type alias Context
	a := alias(c)
	if a.Messages == nil {
		a.Messages = []Message{}
	}
	if a.Tools == nil {
		a.Tools = []ToolSchema{}
	}
	return json.Marshal(a)
}

// IntoRequest builds the ModelRequest corresponding to this Context.
func (c Context) IntoRequest() ModelRequest {
	return c.intoRequestWithStream(false)
}

// IntoRequestStreaming builds the ModelRequest with the stream flag set. The
// streaming turn path (issue #103) uses this so the model layer knows to drive
// CallStreaming.
func (c Context) IntoRequestStreaming() ModelRequest {
	return c.intoRequestWithStream(true)
}

func (c Context) intoRequestWithStream(stream bool) ModelRequest {
	return ModelRequest{
		Messages: c.Messages,
		Tools:    c.Tools,
		Params:   c.Params,
		Stream:   stream,
	}
}

// ============================================================================
// AgentError
// ============================================================================

// AgentErrorKind discriminates AgentError variants. Tag values match the
// Rust serde tag for cross-language wire compatibility.
type AgentErrorKind string

const (
	AgentErrModelError        AgentErrorKind = "ModelError"
	AgentErrEmptyResponse     AgentErrorKind = "EmptyResponse"
	AgentErrMalformedToolCall AgentErrorKind = "MalformedToolCall"
)

// AgentError is the typed error variant carried by TurnResult Error.
//
// One discriminated struct (rather than separate types) keeps the JSON
// shape adjacent to the Rust enum (#[serde(tag = "kind")]) and matches the
// model.go ModelError pattern in this package.
type AgentError struct {
	Kind AgentErrorKind `json:"kind"`
	// ModelError variant payload (flattened — matches Rust #[from] transparent)
	ModelError *ModelError `json:"-"`
	// MalformedToolCall variant fields
	ToolName string `json:"tool_name,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// Error implements the error interface.
func (e *AgentError) Error() string {
	switch e.Kind {
	case AgentErrModelError:
		if e.ModelError != nil {
			return e.ModelError.Error()
		}
		return "model error"
	case AgentErrEmptyResponse:
		return "model returned neither text nor tool calls"
	case AgentErrMalformedToolCall:
		return fmt.Sprintf("malformed tool call from model (tool=%s): %s", e.ToolName, e.Reason)
	default:
		return fmt.Sprintf("agent error: %s", e.Kind)
	}
}

// Unwrap returns the wrapped ModelError when applicable.
func (e *AgentError) Unwrap() error {
	if e.Kind == AgentErrModelError && e.ModelError != nil {
		return e.ModelError
	}
	return nil
}

// MarshalJSON serialises AgentError to match Rust's serde tag layout.
// ModelError is transparent: its fields are flattened alongside "kind".
func (e AgentError) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case AgentErrModelError:
		// Transparent: emit the ModelError as-is (it already carries its
		// own "kind"). To match Rust's #[serde(transparent)] for that
		// variant under tag="kind", the inner kind takes over.
		if e.ModelError == nil {
			return nil, fmt.Errorf("AgentError ModelError: payload missing")
		}
		return json.Marshal(e.ModelError)
	case AgentErrEmptyResponse:
		return json.Marshal(struct {
			Kind AgentErrorKind `json:"kind"`
		}{e.Kind})
	case AgentErrMalformedToolCall:
		return json.Marshal(struct {
			Kind     AgentErrorKind `json:"kind"`
			ToolName string         `json:"tool_name"`
			Reason   string         `json:"reason"`
		}{e.Kind, e.ToolName, e.Reason})
	default:
		return nil, fmt.Errorf("AgentError: unknown kind %q", e.Kind)
	}
}

// UnmarshalJSON decodes AgentError. The kind tag is inspected first; if it
// matches a ModelError kind, the payload decodes as a ModelError under
// the ModelError variant.
func (e *AgentError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind     string `json:"kind"`
		ToolName string `json:"tool_name"`
		Reason   string `json:"reason"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	switch AgentErrorKind(probe.Kind) {
	case AgentErrEmptyResponse:
		e.Kind = AgentErrEmptyResponse
		return nil
	case AgentErrMalformedToolCall:
		e.Kind = AgentErrMalformedToolCall
		e.ToolName = probe.ToolName
		e.Reason = probe.Reason
		return nil
	}
	// Otherwise decode as ModelError (the transparent ModelError variant).
	me := &ModelError{}
	if err := me.UnmarshalJSON(data); err != nil {
		return err
	}
	e.Kind = AgentErrModelError
	e.ModelError = me
	return nil
}

// NewModelAgentError wraps a *ModelError into an AgentError.
func NewModelAgentError(me *ModelError) *AgentError {
	return &AgentError{Kind: AgentErrModelError, ModelError: me}
}

// NewEmptyResponseError constructs the EmptyResponse variant.
func NewEmptyResponseError() *AgentError {
	return &AgentError{Kind: AgentErrEmptyResponse}
}

// NewMalformedToolCallError constructs the MalformedToolCall variant.
func NewMalformedToolCallError(toolName, reason string) *AgentError {
	return &AgentError{Kind: AgentErrMalformedToolCall, ToolName: toolName, Reason: reason}
}

// ============================================================================
// ContextError (routing type — issue #32)
// ============================================================================
//
// The rich ContextError that ContextManager methods return lives in the
// contextmgr subpackage (which imports this package). HaltReason — defined
// here — cannot reference that type without an import cycle, so this package
// carries a routing-level ContextError that HaltReason::ContextError wraps.
// This mirrors the established Go divergence pattern (consumer-side seam +
// bridge) already documented in PROJECT_STATE for MetricEvaluator/Verifier.
// The variant exists so the routing TYPE exists; the live StandardHarness loop
// does not yet trigger it (its placeholder ContextManager.Assemble is
// infallible pending the #7 migration), mirroring Block 1.

// ContextErrorKind discriminates ContextError variants. Tag values match the
// Rust serde tag (rename_all = "snake_case") for cross-language wire
// compatibility.
type ContextErrorKind string

const (
	ContextErrTokenCountFailed  ContextErrorKind = "token_count_failed"
	ContextErrCompactionFailed  ContextErrorKind = "compaction_failed"
	ContextErrAssemblyFailed    ContextErrorKind = "assembly_failed"
	ContextErrCacheHashMismatch ContextErrorKind = "cache_hash_mismatch"
)

// ContextError is the typed error surfaced by a ContextManager and routed by
// HaltReason::ContextError. CacheHashMismatch carries the offending cache
// block (as the same snake_case string the CacheBlock enum serialises to),
// the expected/actual hashes, and the turn the mismatch was detected on.
type ContextError struct {
	Kind       ContextErrorKind `json:"kind"`
	Reason     string           `json:"reason,omitempty"`
	Block      string           `json:"block,omitempty"`
	Expected   uint64           `json:"expected,omitempty"`
	Actual     uint64           `json:"actual,omitempty"`
	TurnNumber uint32           `json:"turn_number,omitempty"`
}

// Error implements the error interface.
func (e *ContextError) Error() string {
	switch e.Kind {
	case ContextErrTokenCountFailed:
		return "token count failed"
	case ContextErrCompactionFailed:
		return fmt.Sprintf("compaction failed: %s", e.Reason)
	case ContextErrAssemblyFailed:
		return fmt.Sprintf("assembly failed: %s", e.Reason)
	case ContextErrCacheHashMismatch:
		return fmt.Sprintf("cache hash mismatch on block %s at turn %d: expected %d, got %d", e.Block, e.TurnNumber, e.Expected, e.Actual)
	default:
		return fmt.Sprintf("context error: %s", e.Kind)
	}
}

// ============================================================================
// TurnResult
// ============================================================================

// TurnResultKind discriminates TurnResult variants. JSON tag values match
// the Rust serde tag (rename_all = "snake_case").
type TurnResultKind string

const (
	TurnToolCallRequested TurnResultKind = "tool_call_requested"
	TurnFinalResponse     TurnResultKind = "final_response"
	TurnError             TurnResultKind = "error"
)

// TurnResult is the outcome of a single Agent turn.
//
// Discriminated by Kind. Pointer fields hold the variant payload that is
// meaningful for each Kind.
type TurnResult struct {
	Kind TurnResultKind `json:"kind"`
	// ToolCallRequested variant
	Calls []ToolCall `json:"-"`
	// FinalResponse variant
	Content string `json:"-"`
	// Error variant
	Err *AgentError `json:"-"`
	// Usage:
	//   ToolCallRequested, FinalResponse: required (carries TokenUsage)
	//   Error: optional (nil when no partial usage was reported, e.g. on
	//          ModelError)
	Usage *TokenUsage `json:"-"`
	// Reasoning is the accumulated thinking/reasoning text produced in this
	// turn, if any (issue #103, Q4). Meaningful for the ToolCallRequested and
	// FinalResponse variants. It is nil when the turn produced no reasoning;
	// serialised with omitempty so pre-#103 fixtures round-trip (they simply
	// omit the field, which deserialises back to nil). Reasoning is NOT
	// preserved in SessionState — that is deferred to issue #104.
	Reasoning *string `json:"-"`
}

// MarshalJSON serialises TurnResult as a flat tagged object matching Rust.
func (r TurnResult) MarshalJSON() ([]byte, error) {
	switch r.Kind {
	case TurnToolCallRequested:
		if r.Usage == nil {
			return nil, fmt.Errorf("TurnResult tool_call_requested: usage missing")
		}
		calls := r.Calls
		if calls == nil {
			calls = []ToolCall{}
		}
		return json.Marshal(struct {
			Kind      TurnResultKind `json:"kind"`
			Calls     []ToolCall     `json:"calls"`
			Usage     TokenUsage     `json:"usage"`
			Reasoning *string        `json:"reasoning,omitempty"`
		}{r.Kind, calls, *r.Usage, r.Reasoning})
	case TurnFinalResponse:
		if r.Usage == nil {
			return nil, fmt.Errorf("TurnResult final_response: usage missing")
		}
		return json.Marshal(struct {
			Kind      TurnResultKind `json:"kind"`
			Content   string         `json:"content"`
			Usage     TokenUsage     `json:"usage"`
			Reasoning *string        `json:"reasoning,omitempty"`
		}{r.Kind, r.Content, *r.Usage, r.Reasoning})
	case TurnError:
		if r.Err == nil {
			return nil, fmt.Errorf("TurnResult error: error payload missing")
		}
		return json.Marshal(struct {
			Kind  TurnResultKind `json:"kind"`
			Error *AgentError    `json:"error"`
			Usage *TokenUsage    `json:"usage"`
		}{r.Kind, r.Err, r.Usage})
	default:
		return nil, fmt.Errorf("TurnResult: unknown kind %q", r.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (r *TurnResult) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind TurnResultKind `json:"kind"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	r.Kind = probe.Kind
	switch probe.Kind {
	case TurnToolCallRequested:
		var v struct {
			Calls     []ToolCall `json:"calls"`
			Usage     TokenUsage `json:"usage"`
			Reasoning *string    `json:"reasoning"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		r.Calls = v.Calls
		usage := v.Usage
		r.Usage = &usage
		r.Reasoning = v.Reasoning
	case TurnFinalResponse:
		var v struct {
			Content   string     `json:"content"`
			Usage     TokenUsage `json:"usage"`
			Reasoning *string    `json:"reasoning"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		r.Content = v.Content
		usage := v.Usage
		r.Usage = &usage
		r.Reasoning = v.Reasoning
	case TurnError:
		var v struct {
			Error *AgentError `json:"error"`
			Usage *TokenUsage `json:"usage"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		r.Err = v.Error
		r.Usage = v.Usage
	default:
		return fmt.Errorf("TurnResult: unknown kind %q", probe.Kind)
	}
	return nil
}

// NewToolCallRequested builds a ToolCallRequested TurnResult.
func NewToolCallRequested(calls []ToolCall, usage TokenUsage) TurnResult {
	return TurnResult{Kind: TurnToolCallRequested, Calls: calls, Usage: &usage}
}

// NewToolCallRequestedWithReasoning builds a ToolCallRequested TurnResult that
// also carries accumulated reasoning text (issue #103, Q4). A nil reasoning is
// treated identically to NewToolCallRequested.
func NewToolCallRequestedWithReasoning(calls []ToolCall, usage TokenUsage, reasoning *string) TurnResult {
	return TurnResult{Kind: TurnToolCallRequested, Calls: calls, Usage: &usage, Reasoning: reasoning}
}

// NewFinalResponse builds a FinalResponse TurnResult.
func NewFinalResponse(content string, usage TokenUsage) TurnResult {
	return TurnResult{Kind: TurnFinalResponse, Content: content, Usage: &usage}
}

// NewFinalResponseWithReasoning builds a FinalResponse TurnResult that also
// carries accumulated reasoning text (issue #103, Q4). A nil reasoning is
// treated identically to NewFinalResponse.
func NewFinalResponseWithReasoning(content string, usage TokenUsage, reasoning *string) TurnResult {
	return TurnResult{Kind: TurnFinalResponse, Content: content, Usage: &usage, Reasoning: reasoning}
}

// NewTurnError builds an Error TurnResult with optional usage.
func NewTurnError(err *AgentError, usage *TokenUsage) TurnResult {
	return TurnResult{Kind: TurnError, Err: err, Usage: usage}
}

// ============================================================================
// The Agent interface
// ============================================================================

// Agent executes a single turn given a fully assembled Context.
type Agent interface {
	// Turn executes exactly one model call and classifies its response.
	// Errors from the model are surfaced inside the returned TurnResult
	// (never as a Go error return); Turn always returns a TurnResult.
	Turn(ctx context.Context, c Context) TurnResult

	// ID returns the agent's configuration identifier (for tracing).
	ID() AgentID
}

// AgentStreamSink receives RAW model-layer StreamEvents as a streaming agent
// drains a streaming model call (issue #103, Q1).
//
// The agent boundary deals only in model-layer StreamEvent values; it does NOT
// depend on the harness-level HarnessStreamEvent type. The harness wraps the
// caller's StreamSink in an adapter that maps each model StreamEvent into zero
// or more HarnessStreamEvents (see Harness mapping). May be called zero times
// (e.g. an agent with no deltas) and is never called after the turn returns.
type AgentStreamSink func(StreamEvent)

// StreamingAgent is the optional interface a model-backed Agent implements to
// support delta-level streaming (issue #103).
//
// Go interfaces cannot carry default methods, so rather than widen the Agent
// interface (which would force every existing impl to add a delegating method),
// streaming is an OPTIONAL capability discovered via a type assertion — the
// same interface-evolution pattern this package already uses for
// AssistantMessageAppender and TokenBudgetReader. The harness calls
// TurnStreamingOrDelegate, which uses TurnStreaming when the agent implements
// this interface and otherwise falls back to plain Turn (ignoring the sink).
// The default behaviour for any agent that does NOT implement StreamingAgent is
// therefore exactly Turn with the sink ignored.
type StreamingAgent interface {
	Agent

	// TurnStreaming executes one turn while forwarding each raw model
	// StreamEvent to sink as it arrives, then classifies the reassembled
	// response identically to Turn (sharing classifyResponse). It still returns
	// a complete TurnResult.
	TurnStreaming(ctx context.Context, c Context, sink AgentStreamSink) TurnResult
}

// TurnStreamingOrDelegate runs a streaming turn when agent supports it,
// otherwise delegates to agent.Turn (ignoring the sink). This is the single
// call site the harness uses so the default (non-streaming) behaviour is
// guaranteed for every Agent impl that does not opt into StreamingAgent.
func TurnStreamingOrDelegate(ctx context.Context, agent Agent, c Context, sink AgentStreamSink) TurnResult {
	if sa, ok := agent.(StreamingAgent); ok {
		return sa.TurnStreaming(ctx, c, sink)
	}
	return agent.Turn(ctx, c)
}

// classifyResponse classifies an accumulated ModelResponse into a TurnResult.
//
// Single source of truth shared by ModelAgent.Turn and
// ModelAgent.TurnStreaming (issue #103): both the blocking and streaming paths
// buffer a complete ModelResponse and then run this identical logic, so
// classification can never diverge between them. Thinking blocks are
// accumulated into the Reasoning field (Q4) instead of being discarded.
func classifyResponse(resp ModelResponse) TurnResult {
	usage := resp.Usage

	var toolCalls []ToolCall
	var textParts []string
	var reasoningParts []string
	for _, block := range resp.Content {
		switch block.Type {
		case ContentBlockTypeToolUse:
			if block.ToolCall != nil {
				toolCalls = append(toolCalls, *block.ToolCall)
			}
		case ContentBlockTypeText:
			textParts = append(textParts, block.Text)
		case ContentBlockTypeThinking:
			// Q4: accumulate thinking text instead of discarding it.
			reasoningParts = append(reasoningParts, block.Text)
		}
	}
	var reasoning *string
	if len(reasoningParts) > 0 {
		r := strings.Join(reasoningParts, "")
		reasoning = &r
	}

	switch resp.StopReason {
	case StopToolUse:
		if len(toolCalls) == 0 {
			u := usage
			return NewTurnError(
				NewMalformedToolCallError("", "stop_reason=tool_use but no tool_use blocks present"),
				&u,
			)
		}
		return NewToolCallRequestedWithReasoning(toolCalls, usage, reasoning)
	case StopEndTurn, StopMaxTokens, StopStopSequence:
		// Tool-use blocks present despite a non-tool stop_reason are still
		// dispatched — silently dropping a tool call is worse than the
		// surprising classification.
		if len(toolCalls) > 0 {
			return NewToolCallRequestedWithReasoning(toolCalls, usage, reasoning)
		}
		if len(textParts) == 0 {
			u := usage
			return NewTurnError(NewEmptyResponseError(), &u)
		}
		return NewFinalResponseWithReasoning(strings.Join(textParts, ""), usage, reasoning)
	default:
		// Unknown stop_reason: treat as malformed to be safe.
		u := usage
		return NewTurnError(
			NewMalformedToolCallError("", fmt.Sprintf("unknown stop_reason %q", resp.StopReason)),
			&u,
		)
	}
}

// ============================================================================
// ModelAgent — the standard implementation
// ============================================================================

// ModelAgent is the standard Agent implementation: forwards Context to a
// ModelInterface and classifies the response per the rules in the file
// header.
type ModelAgent struct {
	id    AgentID
	model ModelInterface
}

// NewModelAgent constructs a ModelAgent.
func NewModelAgent(id AgentID, model ModelInterface) *ModelAgent {
	return &ModelAgent{id: id, model: model}
}

// ID returns the agent identifier.
func (a *ModelAgent) ID() AgentID { return a.id }

// wrapModelError normalises an error returned by the model layer into a
// *ModelError so the AgentError always carries a *ModelError payload
// (cross-language parity).
func wrapModelError(err error) *ModelError {
	if me, ok := err.(*ModelError); ok {
		return me
	}
	return NewProviderError(0, err.Error())
}

// Turn executes a single model call and classifies its response.
func (a *ModelAgent) Turn(ctx context.Context, c Context) TurnResult {
	request := c.IntoRequest()
	resp, err := a.model.Call(ctx, request)
	if err != nil {
		// Surface model errors wrapped; no partial usage.
		return NewTurnError(NewModelAgentError(wrapModelError(err)), nil)
	}
	return classifyResponse(resp)
}

// TurnStreaming executes a streaming turn (issue #103). It builds a streaming
// request, drains the model stream forwarding each raw StreamEvent to sink,
// reassembles a complete ModelResponse, then runs the EXACT SAME
// classifyResponse logic as Turn.
//
// Reassembly rules (mirrored from the Rust reference):
//   - content_block_delta text deltas concatenate per block index into a Text
//     block.
//   - thinking_delta deltas accumulate into a Thinking block (Q4 — surfaced via
//     Reasoning, not discarded).
//   - tool_use_delta fragments concatenate and parse into a ToolUse block's
//     input.
//   - message_stop carries the final usage + stop_reason.
//
// Block ordering is preserved by first-seen index order.
//
// KNOWN LIMITATION (issue #103): model-layer tool-argument deltas do NOT carry
// the tool name or id (providers drop them; provider SSE work is out of scope).
// The accumulator therefore synthesizes a stable per-index call id "call_{index}"
// (matching the harness correlation key) and an EMPTY tool name. Under streamed
// turns the coarse ToolCall name is empty while the args round-trip faithfully.
// Recovering the name would require a new field on the model StreamEvent.
func (a *ModelAgent) TurnStreaming(ctx context.Context, c Context, sink AgentStreamSink) TurnResult {
	request := c.IntoRequestStreaming()
	ch, err := a.model.CallStreaming(ctx, request)
	if err != nil {
		return NewTurnError(NewModelAgentError(wrapModelError(err)), nil)
	}

	acc := newStreamAccumulator()
	for item := range ch {
		if item.Err != nil {
			return NewTurnError(NewModelAgentError(wrapModelError(item.Err)), nil)
		}
		// Forward the RAW model event to the sink first (Q1), then fold it into
		// the in-progress response.
		if sink != nil {
			sink(item.Event)
		}
		acc.fold(item.Event)
	}
	return classifyResponse(acc.intoResponse())
}

// partialBlockKind tags a streamAccumulator partial block.
type partialBlockKind int

const (
	partialText partialBlockKind = iota
	partialThinking
	partialToolJSON
)

// partialBlock is an in-progress content block keyed by stream index.
type partialBlock struct {
	index uint32
	kind  partialBlockKind
	buf   strings.Builder
}

// streamAccumulator reassembles streamed StreamEvents into a ModelResponse
// (issue #103). It tracks partial blocks in first-seen index order so the
// reconstructed content preserves emission order regardless of interleaving.
type streamAccumulator struct {
	blocks     []*partialBlock
	byIndex    map[uint32]*partialBlock
	usage      TokenUsage
	stopReason StopReason
	hasStop    bool
}

func newStreamAccumulator() *streamAccumulator {
	return &streamAccumulator{byIndex: make(map[uint32]*partialBlock)}
}

func (s *streamAccumulator) entry(index uint32, kind partialBlockKind) *partialBlock {
	if b, ok := s.byIndex[index]; ok {
		return b
	}
	b := &partialBlock{index: index, kind: kind}
	s.byIndex[index] = b
	s.blocks = append(s.blocks, b)
	return b
}

func (s *streamAccumulator) fold(ev StreamEvent) {
	switch ev.Type {
	case StreamMessageStart:
		// Dropped at reassembly; carries no content.
	case StreamContentBlockDelta:
		b := s.entry(ev.Index, partialText)
		if b.kind == partialText {
			b.buf.WriteString(ev.Delta)
		}
	case StreamThinkingDelta:
		b := s.entry(ev.Index, partialThinking)
		if b.kind == partialThinking {
			b.buf.WriteString(ev.Delta)
		}
	case StreamToolUseDelta:
		b := s.entry(ev.Index, partialToolJSON)
		if b.kind == partialToolJSON {
			b.buf.WriteString(ev.PartialJSON)
		}
	case StreamContentBlockStop:
		// No content; block boundary only.
	case StreamMessageStop:
		if ev.Usage != nil {
			s.usage = *ev.Usage
		}
		s.stopReason = ev.StopReason
		s.hasStop = true
	}
}

func (s *streamAccumulator) intoResponse() ModelResponse {
	content := make([]ContentBlock, 0, len(s.blocks))
	for _, b := range s.blocks {
		switch b.kind {
		case partialText:
			content = append(content, NewTextBlock(b.buf.String()))
		case partialThinking:
			content = append(content, NewThinkingBlock(b.buf.String()))
		case partialToolJSON:
			// The streamed model events do not carry the tool id / name (only
			// the partial JSON args). Synthesize a stable per-index id; the
			// harness correlates by this index-derived id consistently.
			raw := b.buf.String()
			input := json.RawMessage(raw)
			if !json.Valid(input) {
				input = json.RawMessage("null")
			}
			content = append(content, NewToolUseBlock(ToolCall{
				ID:    fmt.Sprintf("call_%d", b.index),
				Name:  "",
				Input: input,
			}))
		}
	}
	stop := s.stopReason
	if !s.hasStop {
		// Default to EndTurn if the stream ended without message_stop.
		stop = StopEndTurn
	}
	return ModelResponse{
		Content:    content,
		Usage:      s.usage,
		StopReason: stop,
	}
}

// Compile-time interface checks.
var (
	_ Agent          = (*ModelAgent)(nil)
	_ StreamingAgent = (*ModelAgent)(nil)
)
