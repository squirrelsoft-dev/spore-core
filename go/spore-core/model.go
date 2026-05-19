// ModelInterface — boundary between the harness and the underlying LLM.
//
// Implements issue #1. The harness only ever talks to a model through this
// interface; provider-specific concerns (Anthropic, OpenAI, Ollama, replay)
// live behind concrete implementations.
//
// Rules enforced here:
//
//  1. TokenUsage is reported on every successful call (Call and the final
//     MessageStop event of CallStreaming). It is not optional.
//  2. ContextLimitExceeded is reported by the implementation BEFORE the
//     provider is contacted whenever CountTokens exceeds the provider's
//     context window. The helper EnforceContextLimit does the check.
//  3. BudgetExceeded is a harness-side check against
//     ModelRequest.Params.MaxTokens, surfaced as a typed error so the
//     harness loop can halt with a useful reason.
//  4. Provider-specific retry / backoff for transient errors (429, 529,
//     timeouts) lives in the implementation, not in the interface.
//
// Cross-language note: the type names here are mirrored byte-for-byte in
// the Rust / TypeScript / Python packages. See fixtures/README.md.

package sporecore

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ============================================================================
// Roles, content, messages
// ============================================================================

// Role is the speaker of a Message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ContentType discriminates Content variants.
type ContentType string

const (
	ContentTypeText       ContentType = "text"
	ContentTypeToolCall   ContentType = "tool_call"
	ContentTypeToolResult ContentType = "tool_result"
	ContentTypeImage      ContentType = "image"
)

// Content is a tagged-union message payload. The Type field discriminates
// which payload fields are meaningful. JSON shape mirrors the Rust enum with
// `#[serde(tag = "type")]`: variant fields are flattened alongside "type".
type Content struct {
	Type ContentType `json:"type"`
	// text variant
	Text string `json:"text,omitempty"`
	// tool_call variant
	ToolCall *ToolCall `json:"-"`
	// tool_result variant
	ToolResult *ToolResult `json:"-"`
	// image variant
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
}

// ToolCall is a model-emitted request to invoke a tool.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// ToolResult is the outcome of a tool invocation returned to the model.
type ToolResult struct {
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
	IsError   bool   `json:"is_error,omitempty"`
}

// MarshalJSON serialises Content as a flat object keyed by "type".
func (c Content) MarshalJSON() ([]byte, error) {
	switch c.Type {
	case ContentTypeText:
		return json.Marshal(struct {
			Type ContentType `json:"type"`
			Text string      `json:"text"`
		}{c.Type, c.Text})
	case ContentTypeToolCall:
		if c.ToolCall == nil {
			return nil, fmt.Errorf("Content tool_call: payload missing")
		}
		return json.Marshal(struct {
			Type  ContentType     `json:"type"`
			ID    string          `json:"id"`
			Name  string          `json:"name"`
			Input json.RawMessage `json:"input"`
		}{c.Type, c.ToolCall.ID, c.ToolCall.Name, c.ToolCall.Input})
	case ContentTypeToolResult:
		if c.ToolResult == nil {
			return nil, fmt.Errorf("Content tool_result: payload missing")
		}
		return json.Marshal(struct {
			Type      ContentType `json:"type"`
			ToolUseID string      `json:"tool_use_id"`
			Content   string      `json:"content"`
			IsError   bool        `json:"is_error,omitempty"`
		}{c.Type, c.ToolResult.ToolUseID, c.ToolResult.Content, c.ToolResult.IsError})
	case ContentTypeImage:
		return json.Marshal(struct {
			Type      ContentType `json:"type"`
			MediaType string      `json:"media_type"`
			Data      string      `json:"data"`
		}{c.Type, c.MediaType, c.Data})
	default:
		return nil, fmt.Errorf("Content: unknown type %q", c.Type)
	}
}

// UnmarshalJSON decodes the flat tagged-union form into Content.
func (c *Content) UnmarshalJSON(data []byte) error {
	var probe struct {
		Type ContentType `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	c.Type = probe.Type
	switch probe.Type {
	case ContentTypeText:
		var v struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		c.Text = v.Text
	case ContentTypeToolCall:
		tc := &ToolCall{}
		if err := json.Unmarshal(data, tc); err != nil {
			return err
		}
		c.ToolCall = tc
	case ContentTypeToolResult:
		tr := &ToolResult{}
		if err := json.Unmarshal(data, tr); err != nil {
			return err
		}
		c.ToolResult = tr
	case ContentTypeImage:
		var v struct {
			MediaType string `json:"media_type"`
			Data      string `json:"data"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		c.MediaType = v.MediaType
		c.Data = v.Data
	default:
		return fmt.Errorf("Content: unknown type %q", probe.Type)
	}
	return nil
}

// NewTextContent constructs a text Content.
func NewTextContent(text string) Content {
	return Content{Type: ContentTypeText, Text: text}
}

// NewToolCallContent constructs a tool_call Content.
func NewToolCallContent(tc ToolCall) Content {
	return Content{Type: ContentTypeToolCall, ToolCall: &tc}
}

// NewToolResultContent constructs a tool_result Content.
func NewToolResultContent(tr ToolResult) Content {
	return Content{Type: ContentTypeToolResult, ToolResult: &tr}
}

// NewImageContent constructs an image Content.
func NewImageContent(mediaType, data string) Content {
	return Content{Type: ContentTypeImage, MediaType: mediaType, Data: data}
}

// Message is one entry in the model conversation.
type Message struct {
	Role    Role    `json:"role"`
	Content Content `json:"content"`
}

// ============================================================================
// Tool schema (subset — the canonical type lives with ToolRegistry, #4)
// ============================================================================

// ToolSchema is the model-visible declaration of an available tool.
type ToolSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// ============================================================================
// Request / params / response
// ============================================================================

// ModelParams carries optional generation knobs. Pointer fields serialise as
// null when unset, matching the Rust Option<> semantics.
type ModelParams struct {
	Temperature     *float32 `json:"temperature"`
	MaxTokens       *uint32  `json:"max_tokens"`
	ReasoningBudget *uint32  `json:"reasoning_budget"`
	TopP            *float32 `json:"top_p"`
	StopSequences   []string `json:"stop_sequences"`
}

// MarshalJSON ensures StopSequences serialises as [] rather than null.
func (p ModelParams) MarshalJSON() ([]byte, error) {
	type alias ModelParams
	a := alias(p)
	if a.StopSequences == nil {
		a.StopSequences = []string{}
	}
	return json.Marshal(a)
}

// ModelRequest is the harness-built input to a model call.
type ModelRequest struct {
	Messages []Message    `json:"messages"`
	Tools    []ToolSchema `json:"tools"`
	Params   ModelParams  `json:"params"`
	Stream   bool         `json:"stream"`
}

// MarshalJSON ensures slice fields serialise as [] rather than null.
func (r ModelRequest) MarshalJSON() ([]byte, error) {
	type alias ModelRequest
	a := alias(r)
	if a.Messages == nil {
		a.Messages = []Message{}
	}
	if a.Tools == nil {
		a.Tools = []ToolSchema{}
	}
	return json.Marshal(a)
}

// StopReason explains why generation stopped.
type StopReason string

const (
	StopToolUse      StopReason = "tool_use"
	StopEndTurn      StopReason = "end_turn"
	StopMaxTokens    StopReason = "max_tokens"
	StopStopSequence StopReason = "stop_sequence"
)

// ContentBlockType discriminates ContentBlock variants.
type ContentBlockType string

const (
	ContentBlockTypeText     ContentBlockType = "text"
	ContentBlockTypeThinking ContentBlockType = "thinking"
	ContentBlockTypeToolUse  ContentBlockType = "tool_use"
)

// ContentBlock is one slice of a ModelResponse body. JSON shape mirrors the
// Rust enum: tagged on "type", with tool_use flattening a ToolCall.
type ContentBlock struct {
	Type ContentBlockType `json:"type"`
	// text and thinking variants
	Text string `json:"text,omitempty"`
	// tool_use variant
	ToolCall *ToolCall `json:"-"`
}

// MarshalJSON serialises ContentBlock as a flat tagged object.
func (b ContentBlock) MarshalJSON() ([]byte, error) {
	switch b.Type {
	case ContentBlockTypeText, ContentBlockTypeThinking:
		return json.Marshal(struct {
			Type ContentBlockType `json:"type"`
			Text string           `json:"text"`
		}{b.Type, b.Text})
	case ContentBlockTypeToolUse:
		if b.ToolCall == nil {
			return nil, fmt.Errorf("ContentBlock tool_use: payload missing")
		}
		return json.Marshal(struct {
			Type  ContentBlockType `json:"type"`
			ID    string           `json:"id"`
			Name  string           `json:"name"`
			Input json.RawMessage  `json:"input"`
		}{b.Type, b.ToolCall.ID, b.ToolCall.Name, b.ToolCall.Input})
	default:
		return nil, fmt.Errorf("ContentBlock: unknown type %q", b.Type)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (b *ContentBlock) UnmarshalJSON(data []byte) error {
	var probe struct {
		Type ContentBlockType `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	b.Type = probe.Type
	switch probe.Type {
	case ContentBlockTypeText, ContentBlockTypeThinking:
		var v struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(data, &v); err != nil {
			return err
		}
		b.Text = v.Text
	case ContentBlockTypeToolUse:
		tc := &ToolCall{}
		if err := json.Unmarshal(data, tc); err != nil {
			return err
		}
		b.ToolCall = tc
	default:
		return fmt.Errorf("ContentBlock: unknown type %q", probe.Type)
	}
	return nil
}

// NewTextBlock constructs a text ContentBlock.
func NewTextBlock(text string) ContentBlock {
	return ContentBlock{Type: ContentBlockTypeText, Text: text}
}

// NewThinkingBlock constructs a thinking ContentBlock.
func NewThinkingBlock(text string) ContentBlock {
	return ContentBlock{Type: ContentBlockTypeThinking, Text: text}
}

// NewToolUseBlock constructs a tool_use ContentBlock.
func NewToolUseBlock(tc ToolCall) ContentBlock {
	return ContentBlock{Type: ContentBlockTypeToolUse, ToolCall: &tc}
}

// TokenUsage is reported on every successful model call.
type TokenUsage struct {
	InputTokens      uint32  `json:"input_tokens"`
	OutputTokens     uint32  `json:"output_tokens"`
	CacheReadTokens  *uint32 `json:"cache_read_tokens"`
	CacheWriteTokens *uint32 `json:"cache_write_tokens"`
}

// ModelResponse is the result of a successful model call.
type ModelResponse struct {
	Content    []ContentBlock `json:"content"`
	Usage      TokenUsage     `json:"usage"`
	StopReason StopReason     `json:"stop_reason"`
}

// MarshalJSON ensures Content serialises as [] rather than null.
func (r ModelResponse) MarshalJSON() ([]byte, error) {
	type alias ModelResponse
	a := alias(r)
	if a.Content == nil {
		a.Content = []ContentBlock{}
	}
	return json.Marshal(a)
}

// ============================================================================
// Streaming
// ============================================================================

// StreamEventType discriminates StreamEvent variants.
type StreamEventType string

const (
	StreamMessageStart      StreamEventType = "message_start"
	StreamContentBlockDelta StreamEventType = "content_block_delta"
	StreamThinkingDelta     StreamEventType = "thinking_delta"
	StreamToolUseDelta      StreamEventType = "tool_use_delta"
	StreamContentBlockStop  StreamEventType = "content_block_stop"
	StreamMessageStop       StreamEventType = "message_stop"
)

// StreamEvent is a single event in a streaming model response.
type StreamEvent struct {
	Type StreamEventType `json:"type"`
	// content_block_delta, thinking_delta, tool_use_delta, content_block_stop
	Index uint32 `json:"index,omitempty"`
	// content_block_delta, thinking_delta
	Delta string `json:"delta,omitempty"`
	// tool_use_delta
	PartialJSON string `json:"partial_json,omitempty"`
	// message_stop
	Usage      *TokenUsage `json:"usage,omitempty"`
	StopReason StopReason  `json:"stop_reason,omitempty"`
}

// StreamEventOrErr lets a channel carry an event or a terminal error.
type StreamEventOrErr struct {
	Event StreamEvent
	Err   error
}

// ============================================================================
// Provider identity
// ============================================================================

// ProviderInfo identifies the model behind an implementation.
type ProviderInfo struct {
	Name          string `json:"name"`
	ModelID       string `json:"model_id"`
	ContextWindow uint32 `json:"context_window"`
}

// ============================================================================
// Errors
// ============================================================================

// Sentinel errors for Layer 1 always-halt conditions. ModelError values
// returned by helpers below Unwrap to these so callers can match with
// errors.Is.
var (
	ErrContextLimitExceeded = errors.New("context limit exceeded")
	ErrBudgetExceeded       = errors.New("budget exceeded")
)

// ModelErrorKind discriminates ModelError variants. Tag values match the
// Rust serde tag for cross-language wire compatibility.
type ModelErrorKind string

const (
	ModelErrProviderError        ModelErrorKind = "ProviderError"
	ModelErrRateLimited          ModelErrorKind = "RateLimited"
	ModelErrContextLimitExceeded ModelErrorKind = "ContextLimitExceeded"
	ModelErrBudgetExceeded       ModelErrorKind = "BudgetExceeded"
	ModelErrTimeout              ModelErrorKind = "Timeout"
)

// ModelError is the typed harness error returned by ModelInterface methods.
//
// One struct per variant would also be idiomatic; a single discriminated
// struct keeps the JSON shape adjacent to the Rust enum (#[serde(tag = "kind")])
// and avoids an interface allocation per error.
type ModelError struct {
	Kind       ModelErrorKind `json:"kind"`
	Code       uint16         `json:"code,omitempty"`
	Message    string         `json:"message,omitempty"`
	RetryAfter *time.Duration `json:"-"`
	Limit      uint32         `json:"limit,omitempty"`
	Actual     uint32         `json:"actual,omitempty"`
	Budget     uint32         `json:"budget,omitempty"`
	Used       uint32         `json:"used,omitempty"`
}

// Error implements error.
func (e *ModelError) Error() string {
	switch e.Kind {
	case ModelErrProviderError:
		return fmt.Sprintf("provider error %d: %s", e.Code, e.Message)
	case ModelErrRateLimited:
		if e.RetryAfter != nil {
			return fmt.Sprintf("rate limited (retry_after=%s)", *e.RetryAfter)
		}
		return "rate limited (retry_after=none)"
	case ModelErrContextLimitExceeded:
		return fmt.Sprintf("context limit exceeded: %d tokens > limit %d", e.Actual, e.Limit)
	case ModelErrBudgetExceeded:
		return fmt.Sprintf("budget exceeded: %d > budget %d", e.Used, e.Budget)
	case ModelErrTimeout:
		return "model call timed out"
	default:
		return fmt.Sprintf("model error: %s", e.Kind)
	}
}

// Unwrap returns the matching Layer 1 sentinel for always-halt kinds, so
// callers can use errors.Is(err, ErrContextLimitExceeded) etc.
func (e *ModelError) Unwrap() error {
	switch e.Kind {
	case ModelErrContextLimitExceeded:
		return ErrContextLimitExceeded
	case ModelErrBudgetExceeded:
		return ErrBudgetExceeded
	default:
		return nil
	}
}

// MarshalJSON serialises ModelError mirroring the Rust serde tag layout.
// retry_after is encoded as seconds (or null) to match Rust's duration_secs_opt.
func (e ModelError) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case ModelErrProviderError:
		return json.Marshal(struct {
			Kind    ModelErrorKind `json:"kind"`
			Code    uint16         `json:"code"`
			Message string         `json:"message"`
		}{e.Kind, e.Code, e.Message})
	case ModelErrRateLimited:
		var retry *uint64
		if e.RetryAfter != nil {
			s := uint64(e.RetryAfter.Seconds())
			retry = &s
		}
		return json.Marshal(struct {
			Kind       ModelErrorKind `json:"kind"`
			RetryAfter *uint64        `json:"retry_after"`
		}{e.Kind, retry})
	case ModelErrContextLimitExceeded:
		return json.Marshal(struct {
			Kind   ModelErrorKind `json:"kind"`
			Limit  uint32         `json:"limit"`
			Actual uint32         `json:"actual"`
		}{e.Kind, e.Limit, e.Actual})
	case ModelErrBudgetExceeded:
		return json.Marshal(struct {
			Kind   ModelErrorKind `json:"kind"`
			Budget uint32         `json:"budget"`
			Used   uint32         `json:"used"`
		}{e.Kind, e.Budget, e.Used})
	case ModelErrTimeout:
		return json.Marshal(struct {
			Kind ModelErrorKind `json:"kind"`
		}{e.Kind})
	default:
		return nil, fmt.Errorf("ModelError: unknown kind %q", e.Kind)
	}
}

// UnmarshalJSON decodes ModelError from the Rust-compatible tagged shape.
func (e *ModelError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind       ModelErrorKind `json:"kind"`
		Code       uint16         `json:"code"`
		Message    string         `json:"message"`
		RetryAfter *uint64        `json:"retry_after"`
		Limit      uint32         `json:"limit"`
		Actual     uint32         `json:"actual"`
		Budget     uint32         `json:"budget"`
		Used       uint32         `json:"used"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Kind = probe.Kind
	e.Code = probe.Code
	e.Message = probe.Message
	e.Limit = probe.Limit
	e.Actual = probe.Actual
	e.Budget = probe.Budget
	e.Used = probe.Used
	if probe.RetryAfter != nil {
		d := time.Duration(*probe.RetryAfter) * time.Second
		e.RetryAfter = &d
	}
	return nil
}

// Constructor helpers.

// NewProviderError builds a typed ProviderError.
func NewProviderError(code uint16, message string) *ModelError {
	return &ModelError{Kind: ModelErrProviderError, Code: code, Message: message}
}

// NewRateLimited builds a typed RateLimited error; retryAfter may be nil.
func NewRateLimited(retryAfter *time.Duration) *ModelError {
	return &ModelError{Kind: ModelErrRateLimited, RetryAfter: retryAfter}
}

// NewContextLimitExceeded builds a typed ContextLimitExceeded error.
func NewContextLimitExceeded(limit, actual uint32) *ModelError {
	return &ModelError{Kind: ModelErrContextLimitExceeded, Limit: limit, Actual: actual}
}

// NewBudgetExceeded builds a typed BudgetExceeded error.
func NewBudgetExceeded(budget, used uint32) *ModelError {
	return &ModelError{Kind: ModelErrBudgetExceeded, Budget: budget, Used: used}
}

// NewTimeout builds a typed Timeout error.
func NewTimeout() *ModelError {
	return &ModelError{Kind: ModelErrTimeout}
}

// ============================================================================
// The interface
// ============================================================================

// ModelInterface is the boundary between the harness and the underlying LLM.
//
// Implementations must observe the rules documented at the top of this file.
type ModelInterface interface {
	// Call performs one blocking model call. TokenUsage must be populated
	// in the returned ModelResponse on success.
	Call(ctx context.Context, req ModelRequest) (ModelResponse, error)

	// CallStreaming performs a streaming model call. Events are delivered
	// over the returned channel; the final MessageStop event carries the
	// accumulated TokenUsage. A terminal error is delivered as a
	// StreamEventOrErr with Err set, after which the channel is closed.
	CallStreaming(ctx context.Context, req ModelRequest) (<-chan StreamEventOrErr, error)

	// CountTokens estimates token usage for the supplied request so that
	// the harness can detect ContextLimitExceeded before contacting the
	// provider.
	CountTokens(ctx context.Context, req ModelRequest) (uint32, error)

	// Provider reports the underlying model identity.
	Provider() ProviderInfo
}

// EnforceContextLimit is the shared pre-call validation. Implementations
// should call this before contacting the provider.
func EnforceContextLimit(actual, contextWindow uint32) error {
	if actual > contextWindow {
		return NewContextLimitExceeded(contextWindow, actual)
	}
	return nil
}

// EnforceBudget is the shared post-call budget check. budget==nil means no
// budget enforcement (matches Rust Option<u32>).
func EnforceBudget(used uint32, budget *uint32) error {
	if budget != nil && used > *budget {
		return NewBudgetExceeded(*budget, used)
	}
	return nil
}

// ============================================================================
// Fixture-replay implementation
// ============================================================================

// RecordedExchange is one (request, response) pair serialised in the shared
// fixtures/model_responses/ JSONL files. See fixtures/README.md.
//
// RequestHash (issue #37) is populated by RecordingModelInterface (#38) to
// enable content-addressed replay. Fixtures recorded before #37 do not
// include it; absence triggers positional fallback in ReplayModel.
type RecordedExchange struct {
	RequestHash string        `json:"request_hash,omitempty"`
	Request     ModelRequest  `json:"request"`
	Response    ModelResponse `json:"response"`
	Provider    string        `json:"provider"`
	ModelID     string        `json:"model_id,omitempty"`
	RecordedAt  string        `json:"recorded_at,omitempty"`
	DurationMs  *uint64       `json:"duration_ms,omitempty"`
}

// ============================================================================
// Cross-language request hashing (#37, #38)
// ============================================================================

// RequestHash returns the stable content hash of a ModelRequest. Produced by
// canonicalizing the request to JSON (object keys sorted lexicographically,
// no insignificant whitespace) and SHA-256 hashing the resulting bytes, then
// hex-encoding the first 8 bytes (16 hex chars).
//
// All four language implementations of RecordingModelInterface and
// ReplayModelInterface must produce the same hash for the same request. The
// cross-language consistency fixture lives at
// fixtures/model_hashing/cases.json.
func RequestHash(req ModelRequest) string {
	raw, err := json.Marshal(req)
	if err != nil {
		// ModelRequest is always serializable; this path is unreachable in
		// practice. Hash the error string as a defensive fallback so that
		// callers still get a stable, non-empty value.
		sum := sha256.Sum256([]byte(err.Error()))
		return hex.EncodeToString(sum[:8])
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		sum := sha256.Sum256(raw)
		return hex.EncodeToString(sum[:8])
	}
	canon := canonicalizeJSON(tree)
	sum := sha256.Sum256([]byte(canon))
	return hex.EncodeToString(sum[:8])
}

// canonicalizeJSON walks a generic JSON tree (as produced by encoding/json
// with UseNumber) and returns a stable canonical string representation.
func canonicalizeJSON(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case bool:
		if x {
			return "true"
		}
		return "false"
	case json.Number:
		return string(x)
	case string:
		b, _ := json.Marshal(x)
		return string(b)
	case []any:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = canonicalizeJSON(e)
		}
		return "[" + strings.Join(parts, ",") + "]"
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, len(keys))
		for i, k := range keys {
			kb, _ := json.Marshal(k)
			parts[i] = string(kb) + ":" + canonicalizeJSON(x[k])
		}
		return "{" + strings.Join(parts, ",") + "}"
	default:
		// Fallback: marshal and inline. Should not happen with UseNumber.
		b, _ := json.Marshal(x)
		return string(b)
	}
}

// ReplayMode chooses how a ReplayModel matches incoming requests to recorded
// responses (issue #37).
type ReplayMode int

const (
	// ReplayModePositional is the pre-#37 behavior: the n-th Call returns
	// the n-th recorded response. Fragile against loop-order changes but
	// compatible with old fixtures.
	ReplayModePositional ReplayMode = iota
	// ReplayModeHashMatched is the new behavior: each Call hashes its
	// request and looks up the matching recorded entry. Order-independent.
	ReplayModeHashMatched
)

// String returns a human-readable name.
func (m ReplayMode) String() string {
	switch m {
	case ReplayModeHashMatched:
		return "HashMatched"
	case ReplayModePositional:
		return "Positional"
	default:
		return fmt.Sprintf("ReplayMode(%d)", int(m))
	}
}

// ReplayModel returns recorded responses. Defaults to ReplayModeHashMatched
// when every entry has a RequestHash and the slice is non-empty; otherwise
// falls back to ReplayModePositional so pre-#37 fixtures continue to work.
type ReplayModel struct {
	exchanges []RecordedExchange
	provider  ProviderInfo
	mode      ReplayMode
	mu        sync.Mutex
	cursor    int
}

// NewReplayModel constructs a ReplayModel with the auto-detected mode.
func NewReplayModel(exchanges []RecordedExchange, provider ProviderInfo) *ReplayModel {
	mode := ReplayModePositional
	if len(exchanges) > 0 {
		all := true
		for _, e := range exchanges {
			if e.RequestHash == "" {
				all = false
				break
			}
		}
		if all {
			mode = ReplayModeHashMatched
		}
	}
	return NewReplayModelWithMode(exchanges, provider, mode)
}

// NewReplayModelWithMode constructs a ReplayModel with an explicit mode.
// Useful when forcing positional replay against a hash-tagged fixture.
func NewReplayModelWithMode(exchanges []RecordedExchange, provider ProviderInfo, mode ReplayMode) *ReplayModel {
	return &ReplayModel{exchanges: exchanges, provider: provider, mode: mode}
}

// Mode reports the active ReplayMode.
func (r *ReplayModel) Mode() ReplayMode {
	return r.mode
}

// ParseReplayJSONL parses a JSONL string of RecordedExchange records and
// auto-detects the replay mode.
func ParseReplayJSONL(text string, provider ProviderInfo) (*ReplayModel, error) {
	var exchanges []RecordedExchange
	for i, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var ex RecordedExchange
		if err := json.Unmarshal([]byte(line), &ex); err != nil {
			return nil, fmt.Errorf("ParseReplayJSONL line %d: %w", i+1, err)
		}
		exchanges = append(exchanges, ex)
	}
	return NewReplayModel(exchanges, provider), nil
}

// Remaining reports how many recorded exchanges are left. In hash-matched
// mode this reports the total number of exchanges, since matching is not
// cursor-based.
func (r *ReplayModel) Remaining() int {
	if r.mode == ReplayModeHashMatched {
		return len(r.exchanges)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cursor >= len(r.exchanges) {
		return 0
	}
	return len(r.exchanges) - r.cursor
}

// Call returns the recorded response matching the request, dispatching on
// the active ReplayMode.
func (r *ReplayModel) Call(_ context.Context, req ModelRequest) (ModelResponse, error) {
	switch r.mode {
	case ReplayModeHashMatched:
		want := RequestHash(req)
		for _, ex := range r.exchanges {
			if ex.RequestHash == want {
				return ex.Response, nil
			}
		}
		return ModelResponse{}, NewProviderError(0, fmt.Sprintf("no matching fixture for request_hash=%s", want))
	case ReplayModePositional:
		fallthrough
	default:
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.cursor >= len(r.exchanges) {
			return ModelResponse{}, NewProviderError(0, "replay exhausted: no more recorded exchanges")
		}
		resp := r.exchanges[r.cursor].Response
		r.cursor++
		return resp, nil
	}
}

// CallStreaming synthesises a stream from the next recorded response.
func (r *ReplayModel) CallStreaming(ctx context.Context, req ModelRequest) (<-chan StreamEventOrErr, error) {
	resp, err := r.Call(ctx, req)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEventOrErr, 4)
	go func() {
		defer close(ch)
		send := func(ev StreamEvent) bool {
			select {
			case <-ctx.Done():
				ch <- StreamEventOrErr{Err: ctx.Err()}
				return false
			case ch <- StreamEventOrErr{Event: ev}:
				return true
			}
		}
		if !send(StreamEvent{Type: StreamMessageStart}) {
			return
		}
		for i, block := range resp.Content {
			idx := uint32(i)
			switch block.Type {
			case ContentBlockTypeText:
				if !send(StreamEvent{Type: StreamContentBlockDelta, Index: idx, Delta: block.Text}) {
					return
				}
			case ContentBlockTypeThinking:
				if !send(StreamEvent{Type: StreamThinkingDelta, Index: idx, Delta: block.Text}) {
					return
				}
			case ContentBlockTypeToolUse:
				var partial string
				if block.ToolCall != nil && len(block.ToolCall.Input) > 0 {
					partial = string(block.ToolCall.Input)
				} else {
					partial = "{}"
				}
				if !send(StreamEvent{Type: StreamToolUseDelta, Index: idx, PartialJSON: partial}) {
					return
				}
			}
			if !send(StreamEvent{Type: StreamContentBlockStop, Index: idx}) {
				return
			}
		}
		usage := resp.Usage
		send(StreamEvent{Type: StreamMessageStop, Usage: &usage, StopReason: resp.StopReason})
	}()
	return ch, nil
}

// CountTokens prefers the recorded input_tokens from the matching fixture
// (when the replay is hash-matched and a fixture is found), otherwise falls
// back to a deterministic ~4 chars/token estimate. Real providers override
// this entirely.
func (r *ReplayModel) CountTokens(_ context.Context, req ModelRequest) (uint32, error) {
	if r.mode == ReplayModeHashMatched {
		want := RequestHash(req)
		for _, ex := range r.exchanges {
			if ex.RequestHash == want {
				return ex.Response.Usage.InputTokens, nil
			}
		}
	}
	n := 0
	for _, m := range req.Messages {
		switch m.Content.Type {
		case ContentTypeText:
			n += len(m.Content.Text)
		case ContentTypeToolCall:
			if m.Content.ToolCall != nil {
				n += len(m.Content.ToolCall.Name) + len(m.Content.ToolCall.Input)
			}
		case ContentTypeToolResult:
			if m.Content.ToolResult != nil {
				n += len(m.Content.ToolResult.Content)
			}
		case ContentTypeImage:
			// images contribute 0 in the rough estimate
		}
	}
	return uint32(n / 4), nil
}

// Provider reports the configured ProviderInfo.
func (r *ReplayModel) Provider() ProviderInfo {
	return r.provider
}

// Compile-time interface check.
var _ ModelInterface = (*ReplayModel)(nil)

// ============================================================================
// Mock implementation (test helper, kept in-package so tests can use it)
// ============================================================================

// MockResponse is one queued outcome for MockModel.
type MockResponse struct {
	Response ModelResponse
	Err      error
}

// MockModel is a programmable model for unit tests. Each queued response is
// yielded on successive Call invocations.
type MockModel struct {
	mu          sync.Mutex
	responses   []MockResponse
	tokenCounts []uint32
	provider    ProviderInfo
	calls       atomic.Int64
}

// NewMockModel constructs a MockModel.
func NewMockModel(provider ProviderInfo) *MockModel {
	return &MockModel{provider: provider}
}

// PushResponse queues a response for the next Call.
func (m *MockModel) PushResponse(r ModelResponse) *MockModel {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses = append(m.responses, MockResponse{Response: r})
	return m
}

// PushError queues an error for the next Call.
func (m *MockModel) PushError(err error) *MockModel {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses = append(m.responses, MockResponse{Err: err})
	return m
}

// PushTokenCount queues a value for the next CountTokens call.
func (m *MockModel) PushTokenCount(n uint32) *MockModel {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenCounts = append(m.tokenCounts, n)
	return m
}

// CallCount returns the number of Call invocations observed.
func (m *MockModel) CallCount() int64 {
	return m.calls.Load()
}

// Call returns the next queued response.
func (m *MockModel) Call(_ context.Context, _ ModelRequest) (ModelResponse, error) {
	m.calls.Add(1)
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.responses) == 0 {
		return ModelResponse{}, NewProviderError(0, "mock: no response queued")
	}
	r := m.responses[0]
	m.responses = m.responses[1:]
	return r.Response, r.Err
}

// CallStreaming synthesises a minimal stream from the next queued response.
func (m *MockModel) CallStreaming(ctx context.Context, req ModelRequest) (<-chan StreamEventOrErr, error) {
	resp, err := m.Call(ctx, req)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamEventOrErr, 2)
	go func() {
		defer close(ch)
		ch <- StreamEventOrErr{Event: StreamEvent{Type: StreamMessageStart}}
		usage := resp.Usage
		ch <- StreamEventOrErr{Event: StreamEvent{
			Type:       StreamMessageStop,
			Usage:      &usage,
			StopReason: resp.StopReason,
		}}
	}()
	return ch, nil
}

// CountTokens returns the next queued token count, or 0 when empty.
func (m *MockModel) CountTokens(_ context.Context, _ ModelRequest) (uint32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.tokenCounts) == 0 {
		return 0, nil
	}
	n := m.tokenCounts[0]
	m.tokenCounts = m.tokenCounts[1:]
	return n, nil
}

// Provider reports the configured ProviderInfo.
func (m *MockModel) Provider() ProviderInfo {
	return m.provider
}

// Compile-time interface check.
var _ ModelInterface = (*MockModel)(nil)

// ============================================================================
// RecordingModelInterface (issue #38)
// ============================================================================

// RecordingMode controls how RecordingModel writes recorded exchanges.
type RecordingMode int

const (
	// RecordingModeRecord appends every (request, response) pair to the
	// output file.
	RecordingModeRecord RecordingMode = iota
	// RecordingModeRecordIfNew appends only if the output file does not
	// yet exist. Useful when running tests for the first time against a
	// real provider, then never re-recording.
	RecordingModeRecordIfNew
	// RecordingModePassthrough calls the inner ModelInterface but writes
	// nothing. Used to disable recording without changing call sites.
	RecordingModePassthrough
)

// String returns a human-readable name.
func (m RecordingMode) String() string {
	switch m {
	case RecordingModeRecord:
		return "Record"
	case RecordingModeRecordIfNew:
		return "RecordIfNew"
	case RecordingModePassthrough:
		return "Passthrough"
	default:
		return fmt.Sprintf("RecordingMode(%d)", int(m))
	}
}

// RecordingModel is a transparent wrapper around a real ModelInterface that
// appends each (request, response) pair to a JSONL fixture file as a
// RecordedExchange with a stable RequestHash.
type RecordingModel struct {
	inner      ModelInterface
	outputPath string
	mode       RecordingMode
	mu         sync.Mutex
}

// NewRecordingModel constructs a RecordingModel wrapping inner.
func NewRecordingModel(inner ModelInterface, outputPath string, mode RecordingMode) *RecordingModel {
	return &RecordingModel{inner: inner, outputPath: outputPath, mode: mode}
}

// OutputPath reports the configured JSONL output path.
func (r *RecordingModel) OutputPath() string {
	return r.outputPath
}

// Mode reports the configured RecordingMode.
func (r *RecordingModel) Mode() RecordingMode {
	return r.mode
}

// Call delegates to the inner ModelInterface and records the resulting
// exchange (subject to the configured RecordingMode).
func (r *RecordingModel) Call(ctx context.Context, req ModelRequest) (ModelResponse, error) {
	start := time.Now()
	resp, err := r.inner.Call(ctx, req)
	if err != nil {
		return ModelResponse{}, err
	}
	durMs := uint64(time.Since(start).Milliseconds())
	if writeErr := r.record(req, resp, durMs); writeErr != nil {
		return ModelResponse{}, NewProviderError(0, fmt.Sprintf("recorder write failed: %v", writeErr))
	}
	return resp, nil
}

// CallStreaming passes through to the inner ModelInterface. Streaming
// recording is not implemented: the spec only requires the blocking Call
// pair to be recorded.
func (r *RecordingModel) CallStreaming(ctx context.Context, req ModelRequest) (<-chan StreamEventOrErr, error) {
	return r.inner.CallStreaming(ctx, req)
}

// CountTokens delegates to the inner ModelInterface.
func (r *RecordingModel) CountTokens(ctx context.Context, req ModelRequest) (uint32, error) {
	return r.inner.CountTokens(ctx, req)
}

// Provider reports the inner ModelInterface's provider.
func (r *RecordingModel) Provider() ProviderInfo {
	return r.inner.Provider()
}

func (r *RecordingModel) record(req ModelRequest, resp ModelResponse, durationMs uint64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var shouldWrite bool
	switch r.mode {
	case RecordingModeRecord:
		shouldWrite = true
	case RecordingModeRecordIfNew:
		_, statErr := os.Stat(r.outputPath)
		shouldWrite = os.IsNotExist(statErr)
	case RecordingModePassthrough:
		shouldWrite = false
	}
	if !shouldWrite {
		return nil
	}
	if parent := filepath.Dir(r.outputPath); parent != "" && parent != "." {
		if err := os.MkdirAll(parent, 0o755); err != nil {
			return err
		}
	}
	provider := r.inner.Provider()
	dur := durationMs
	entry := RecordedExchange{
		RequestHash: RequestHash(req),
		Request:     req,
		Response:    resp,
		Provider:    provider.Name,
		ModelID:     provider.ModelID,
		DurationMs:  &dur,
	}
	line, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(r.outputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(line); err != nil {
		return err
	}
	if _, err := f.Write([]byte("\n")); err != nil {
		return err
	}
	return nil
}

// Compile-time interface check.
var _ ModelInterface = (*RecordingModel)(nil)
