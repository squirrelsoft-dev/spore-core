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
//     (concatenated text blocks; Thinking discarded — observability only)
//     - StopEndTurn|MaxTokens|StopSequence with neither → EmptyResponse
//  5. ModelError is surfaced wrapped in AgentError ModelError, with no
//     partial usage information (usage is nil).
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
	return ModelRequest{
		Messages: c.Messages,
		Tools:    c.Tools,
		Params:   c.Params,
		Stream:   false,
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
			Kind  TurnResultKind `json:"kind"`
			Calls []ToolCall     `json:"calls"`
			Usage TokenUsage     `json:"usage"`
		}{r.Kind, calls, *r.Usage})
	case TurnFinalResponse:
		if r.Usage == nil {
			return nil, fmt.Errorf("TurnResult final_response: usage missing")
		}
		return json.Marshal(struct {
			Kind    TurnResultKind `json:"kind"`
			Content string         `json:"content"`
			Usage   TokenUsage     `json:"usage"`
		}{r.Kind, r.Content, *r.Usage})
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
			Calls []ToolCall `json:"calls"`
			Usage TokenUsage `json:"usage"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		r.Calls = v.Calls
		usage := v.Usage
		r.Usage = &usage
	case TurnFinalResponse:
		var v struct {
			Content string     `json:"content"`
			Usage   TokenUsage `json:"usage"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		r.Content = v.Content
		usage := v.Usage
		r.Usage = &usage
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

// NewFinalResponse builds a FinalResponse TurnResult.
func NewFinalResponse(content string, usage TokenUsage) TurnResult {
	return TurnResult{Kind: TurnFinalResponse, Content: content, Usage: &usage}
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

// Turn executes a single model call and classifies its response.
func (a *ModelAgent) Turn(ctx context.Context, c Context) TurnResult {
	request := c.IntoRequest()
	resp, err := a.model.Call(ctx, request)
	if err != nil {
		// Surface model errors wrapped; no partial usage.
		var me *ModelError
		if asMe, ok := err.(*ModelError); ok {
			me = asMe
		} else {
			// Wrap arbitrary errors into a ProviderError so the AgentError
			// always carries a *ModelError payload (cross-lang parity).
			me = NewProviderError(0, err.Error())
		}
		return NewTurnError(NewModelAgentError(me), nil)
	}

	usage := resp.Usage

	// Extract tool-use blocks and text blocks regardless of stop_reason;
	// the stop_reason determines classification, but presence of tool_use
	// blocks always wins over silently dropping them.
	var toolCalls []ToolCall
	var textParts []string
	for _, block := range resp.Content {
		switch block.Type {
		case ContentBlockTypeToolUse:
			if block.ToolCall != nil {
				toolCalls = append(toolCalls, *block.ToolCall)
			}
		case ContentBlockTypeText:
			textParts = append(textParts, block.Text)
		case ContentBlockTypeThinking:
			// observability only — discarded
		}
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
		return NewToolCallRequested(toolCalls, usage)
	case StopEndTurn, StopMaxTokens, StopStopSequence:
		// Tool-use blocks present despite a non-tool stop_reason are
		// still dispatched — silently dropping a tool call is worse than
		// the surprising classification.
		if len(toolCalls) > 0 {
			return NewToolCallRequested(toolCalls, usage)
		}
		if len(textParts) == 0 {
			u := usage
			return NewTurnError(NewEmptyResponseError(), &u)
		}
		return NewFinalResponse(strings.Join(textParts, ""), usage)
	default:
		// Unknown stop_reason: treat as empty/malformed to be safe.
		u := usage
		return NewTurnError(
			NewMalformedToolCallError("", fmt.Sprintf("unknown stop_reason %q", resp.StopReason)),
			&u,
		)
	}
}

// Compile-time interface check.
var _ Agent = (*ModelAgent)(nil)
