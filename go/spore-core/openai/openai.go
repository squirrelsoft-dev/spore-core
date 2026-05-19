// Package openai implements ModelInterface against the OpenAI Chat
// Completions API (https://api.openai.com/v1/chat/completions).
//
// Issue #40. Translates the spore-core ModelRequest / ModelResponse types
// to and from OpenAI's wire format, parses the OpenAI SSE event stream for
// CallStreaming, and maps HTTP errors to typed ModelError values with
// retry / backoff for transient failures.
//
// Provider-specific shape
//
//   - System messages stay in the messages array (Anthropic extracts them
//     into a top-level field — OpenAI does not).
//   - Assistant tool calls travel in a tool_calls array on the assistant
//     message; function.arguments is a JSON-encoded STRING.
//   - Tool results travel as standalone messages with role: "tool" and a
//     tool_call_id linking back to the call.
//   - Reasoning models (o1, o3, o4*) do not accept temperature and replace
//     max_tokens with max_completion_tokens. Detection is by model-id
//     prefix.
//   - Streaming SSE chunks contain delta.content (text), delta.tool_calls
//     (partial tool calls indexed and accumulated across chunks), and end
//     with a literal `data: [DONE]` line. The final usage block only
//     appears when the request set stream_options: {include_usage: true}.
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ---------------------------------------------------------------------------
// Public client
// ---------------------------------------------------------------------------

const (
	// DefaultBaseURL is the production OpenAI endpoint.
	DefaultBaseURL = "https://api.openai.com/v1"

	// DefaultTimeout is the per-request timeout.
	DefaultTimeout = 120 * time.Second

	// DefaultMaxRetries is the retry count for transient 408/425/429/500/
	// 502/503/504 responses and timeouts.
	DefaultMaxRetries = 3
)

// ModelInterface is the OpenAI Chat Completions API client. It implements
// sporecore.ModelInterface.
//
// The API key is never returned by String / GoString and is omitted from
// any debug-style output.
type ModelInterface struct {
	apiKey     string
	modelID    string
	baseURL    string
	timeout    time.Duration
	maxRetries uint32
	httpClient *http.Client
}

// New constructs a ModelInterface with default base URL, timeout, retries,
// and an http.Client.
func New(apiKey, modelID string) *ModelInterface {
	return &ModelInterface{
		apiKey:     apiKey,
		modelID:    modelID,
		baseURL:    DefaultBaseURL,
		timeout:    DefaultTimeout,
		maxRetries: DefaultMaxRetries,
		httpClient: &http.Client{},
	}
}

// FromEnv reads the API key from the environment variable envVar.
// Returns a ProviderError when the variable is unset or empty.
func FromEnv(envVar, modelID string) (*ModelInterface, error) {
	key, ok := os.LookupEnv(envVar)
	if !ok {
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("env var %q not set", envVar))
	}
	if strings.TrimSpace(key) == "" {
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("env var %q is empty", envVar))
	}
	return New(key, modelID), nil
}

// WithBaseURL overrides the base URL (use for Azure OpenAI, proxies, or
// mocking).
func (c *ModelInterface) WithBaseURL(u string) *ModelInterface {
	c.baseURL = u
	return c
}

// WithTimeout overrides the per-request timeout.
func (c *ModelInterface) WithTimeout(d time.Duration) *ModelInterface {
	c.timeout = d
	return c
}

// WithMaxRetries overrides the retry count for transient responses.
func (c *ModelInterface) WithMaxRetries(n uint32) *ModelInterface {
	c.maxRetries = n
	return c
}

// WithHTTPClient overrides the http.Client used to issue requests.
func (c *ModelInterface) WithHTTPClient(h *http.Client) *ModelInterface {
	c.httpClient = h
	return c
}

// String redacts the API key.
func (c *ModelInterface) String() string {
	return fmt.Sprintf(
		"OpenAIModelInterface{api_key=<redacted>, model_id=%q, base_url=%q, timeout=%s, max_retries=%d}",
		c.modelID, c.baseURL, c.timeout, c.maxRetries,
	)
}

// GoString redacts the API key.
func (c *ModelInterface) GoString() string { return c.String() }

// ContextWindow returns the known context window for an OpenAI model id.
// Unknown models return 0 so callers can detect them.
func ContextWindow(modelID string) uint32 {
	switch {
	case strings.HasPrefix(modelID, "gpt-4o"):
		return 128_000
	case strings.HasPrefix(modelID, "gpt-4.1"):
		return 1_000_000
	case strings.HasPrefix(modelID, "o3"), strings.HasPrefix(modelID, "o4"):
		return 200_000
	case strings.HasPrefix(modelID, "o1"):
		return 128_000
	default:
		return 0
	}
}

// IsReasoningModel reports true for o-series reasoning models, which have
// different parameter constraints (no temperature, max_completion_tokens
// instead of max_tokens).
func IsReasoningModel(modelID string) bool {
	return strings.HasPrefix(modelID, "o1") ||
		strings.HasPrefix(modelID, "o3") ||
		strings.HasPrefix(modelID, "o4")
}

// Provider reports the underlying model identity.
func (c *ModelInterface) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{
		Name:          "openai",
		ModelID:       c.modelID,
		ContextWindow: ContextWindow(c.modelID),
	}
}

var _ sporecore.ModelInterface = (*ModelInterface)(nil)

// ---------------------------------------------------------------------------
// Wire-format types (OpenAI Chat Completions API)
// ---------------------------------------------------------------------------

type wireRequest struct {
	Model               string         `json:"model"`
	Messages            []wireMessage  `json:"messages"`
	MaxTokens           *uint32        `json:"max_tokens,omitempty"`
	MaxCompletionTokens *uint32        `json:"max_completion_tokens,omitempty"`
	Temperature         *float32       `json:"temperature,omitempty"`
	TopP                *float32       `json:"top_p,omitempty"`
	Stop                []string       `json:"stop,omitempty"`
	Tools               []wireTool     `json:"tools,omitempty"`
	Stream              bool           `json:"stream,omitempty"`
	StreamOptions       *streamOptions `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    *string        `json:"content,omitempty"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // always "function"
	Function wireFunctionCall `json:"function"`
}

type wireFunctionCall struct {
	Name string `json:"name"`
	// Arguments is a JSON-encoded string per OpenAI's wire format.
	Arguments string `json:"arguments"`
}

type wireTool struct {
	Type     string           `json:"type"` // always "function"
	Function wireToolFunction `json:"function"`
}

type wireToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type wireResponse struct {
	Choices []wireChoice `json:"choices"`
	Usage   wireUsage    `json:"usage"`
}

type wireChoice struct {
	Message      wireResponseMessage `json:"message"`
	FinishReason *string             `json:"finish_reason,omitempty"`
}

type wireResponseMessage struct {
	Content   *string                `json:"content,omitempty"`
	Reasoning *string                `json:"reasoning,omitempty"`
	ToolCalls []wireResponseToolCall `json:"tool_calls,omitempty"`
}

type wireResponseToolCall struct {
	ID       string                   `json:"id"`
	Function wireResponseFunctionCall `json:"function"`
}

type wireResponseFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type wireUsage struct {
	PromptTokens        uint32                  `json:"prompt_tokens"`
	CompletionTokens    uint32                  `json:"completion_tokens"`
	PromptTokensDetails *wirePromptTokenDetails `json:"prompt_tokens_details,omitempty"`
}

type wirePromptTokenDetails struct {
	CachedTokens *uint32 `json:"cached_tokens,omitempty"`
}

type wireErrorBody struct {
	Error *wireErrorInner `json:"error,omitempty"`
}

type wireErrorInner struct {
	Message string `json:"message,omitempty"`
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

func buildRequest(modelID string, req sporecore.ModelRequest, stream bool) wireRequest {
	messages := make([]wireMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		messages = append(messages, messageToWire(m))
	}

	tools := make([]wireTool, 0, len(req.Tools))
	for _, t := range req.Tools {
		params := t.InputSchema
		if len(params) == 0 {
			params = json.RawMessage("{}")
		}
		tools = append(tools, wireTool{
			Type: "function",
			Function: wireToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}

	reasoning := IsReasoningModel(modelID)
	var maxTokens, maxCompletion *uint32
	if reasoning {
		maxCompletion = req.Params.MaxTokens
	} else {
		maxTokens = req.Params.MaxTokens
	}
	temperature := req.Params.Temperature
	if reasoning {
		temperature = nil
	}

	w := wireRequest{
		Model:               modelID,
		Messages:            messages,
		MaxTokens:           maxTokens,
		MaxCompletionTokens: maxCompletion,
		Temperature:         temperature,
		TopP:                req.Params.TopP,
		Stop:                req.Params.StopSequences,
		Tools:               tools,
		Stream:              stream,
	}
	if stream {
		w.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	return w
}

func messageToWire(m sporecore.Message) wireMessage {
	role := "user"
	switch m.Role {
	case sporecore.RoleSystem:
		role = "system"
	case sporecore.RoleAssistant:
		role = "assistant"
	case sporecore.RoleTool:
		role = "tool"
	}
	switch m.Content.Type {
	case sporecore.ContentTypeText:
		t := m.Content.Text
		return wireMessage{Role: role, Content: &t}
	case sporecore.ContentTypeToolCall:
		if m.Content.ToolCall != nil {
			args := "{}"
			if len(m.Content.ToolCall.Input) > 0 {
				args = string(m.Content.ToolCall.Input)
			}
			return wireMessage{
				Role: "assistant",
				ToolCalls: []wireToolCall{{
					ID:   m.Content.ToolCall.ID,
					Type: "function",
					Function: wireFunctionCall{
						Name:      m.Content.ToolCall.Name,
						Arguments: args,
					},
				}},
			}
		}
		return wireMessage{Role: "assistant"}
	case sporecore.ContentTypeToolResult:
		if m.Content.ToolResult != nil {
			c := m.Content.ToolResult.Content
			return wireMessage{
				Role:       "tool",
				Content:    &c,
				ToolCallID: m.Content.ToolResult.ToolUseID,
			}
		}
		return wireMessage{Role: "tool"}
	case sporecore.ContentTypeImage:
		// OpenAI's chat-completions image input uses a content-parts array
		// (`{type: "image_url", image_url: {url: "data:..."}}`). The
		// harness does not currently emit image content into requests, so
		// we serialize a textual placeholder rather than introduce a
		// heterogeneous shape.
		txt := fmt.Sprintf("[image: %s]", m.Content.MediaType)
		return wireMessage{Role: role, Content: &txt}
	}
	t := m.Content.Text
	return wireMessage{Role: role, Content: &t}
}

func parseResponse(body wireResponse) sporecore.ModelResponse {
	var choice wireChoice
	if len(body.Choices) > 0 {
		choice = body.Choices[0]
	}
	blocks := make([]sporecore.ContentBlock, 0, 2+len(choice.Message.ToolCalls))
	if choice.Message.Reasoning != nil && *choice.Message.Reasoning != "" {
		blocks = append(blocks, sporecore.NewThinkingBlock(*choice.Message.Reasoning))
	}
	if choice.Message.Content != nil && *choice.Message.Content != "" {
		blocks = append(blocks, sporecore.NewTextBlock(*choice.Message.Content))
	}
	for _, tc := range choice.Message.ToolCalls {
		var input json.RawMessage
		switch {
		case tc.Function.Arguments == "":
			input = json.RawMessage("{}")
		default:
			// Validate it's parseable JSON; if not, encode as a JSON string.
			var probe any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &probe); err == nil {
				input = json.RawMessage(tc.Function.Arguments)
			} else {
				if b, mErr := json.Marshal(tc.Function.Arguments); mErr == nil {
					input = b
				} else {
					input = json.RawMessage("{}")
				}
			}
		}
		blocks = append(blocks, sporecore.NewToolUseBlock(sporecore.ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		}))
	}

	usage := sporecore.TokenUsage{
		InputTokens:  body.Usage.PromptTokens,
		OutputTokens: body.Usage.CompletionTokens,
	}
	if body.Usage.PromptTokensDetails != nil && body.Usage.PromptTokensDetails.CachedTokens != nil {
		v := *body.Usage.PromptTokensDetails.CachedTokens
		usage.CacheReadTokens = &v
	}

	return sporecore.ModelResponse{
		Content:    blocks,
		Usage:      usage,
		StopReason: parseStopReason(choice.FinishReason),
	}
}

func parseStopReason(s *string) sporecore.StopReason {
	if s == nil {
		return sporecore.StopEndTurn
	}
	switch *s {
	case "tool_calls", "function_call":
		return sporecore.StopToolUse
	case "length":
		return sporecore.StopMaxTokens
	case "stop":
		return sporecore.StopEndTurn
	default:
		return sporecore.StopEndTurn
	}
}

// ---------------------------------------------------------------------------
// HTTP plumbing with retry
// ---------------------------------------------------------------------------

// backoffDelay returns the exponential-with-cap delay for the given retry
// attempt. 0 → 500ms, 1 → 1s, 2 → 2s, 3 → 4s, ... capped at 30s.
func backoffDelay(attempt uint32) time.Duration {
	shift := attempt
	if shift > 6 {
		shift = 6
	}
	ms := uint64(500) * (uint64(1) << shift)
	if ms > 30_000 {
		ms = 30_000
	}
	return time.Duration(ms) * time.Millisecond
}

func isRetryableStatus(code int) bool {
	switch code {
	case 408, 425, 429, 500, 502, 503, 504:
		return true
	}
	return false
}

// sendWithRetry issues the request returned by build() and retries transient
// failures up to maxRetries.
func (c *ModelInterface) sendWithRetry(
	ctx context.Context,
	build func(ctx context.Context) (*http.Request, error),
) (*http.Response, error) {
	var attempt uint32
	for {
		reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
		req, err := build(reqCtx)
		if err != nil {
			cancel()
			return nil, sporecore.NewProviderError(0, fmt.Sprintf("HTTP request build failed: %v", err))
		}
		resp, err := c.httpClient.Do(req)
		if err != nil {
			cancel()
			if isTimeout(err, reqCtx) {
				if attempt < c.maxRetries {
					if sleepErr := sleepCtx(ctx, backoffDelay(attempt)); sleepErr != nil {
						return nil, sleepErr
					}
					attempt++
					continue
				}
				return nil, sporecore.NewTimeout()
			}
			return nil, sporecore.NewProviderError(0, fmt.Sprintf("HTTP transport error: %v", err))
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, cancel: cancel}
			return resp, nil
		}
		retryable := isRetryableStatus(resp.StatusCode) && attempt < c.maxRetries
		if retryable {
			delay := parseRetryAfter(resp.Header.Get("Retry-After"))
			if delay == 0 {
				delay = backoffDelay(attempt)
			}
			_ = resp.Body.Close()
			cancel()
			if sleepErr := sleepCtx(ctx, delay); sleepErr != nil {
				return nil, sleepErr
			}
			attempt++
			continue
		}
		modelErr := mapStatusError(resp)
		_ = resp.Body.Close()
		cancel()
		return nil, modelErr
	}
}

// cancelOnCloseBody calls cancel() when the underlying body is closed.
type cancelOnCloseBody struct {
	io.ReadCloser
	cancel context.CancelFunc
	once   bool
}

func (b *cancelOnCloseBody) Close() error {
	err := b.ReadCloser.Close()
	if !b.once && b.cancel != nil {
		b.cancel()
		b.once = true
	}
	return err
}

func isTimeout(err error, reqCtx context.Context) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	if reqCtx != nil && errors.Is(reqCtx.Err(), context.DeadlineExceeded) {
		return true
	}
	var ne interface{ Timeout() bool }
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	return false
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func parseRetryAfter(h string) time.Duration {
	if h == "" {
		return 0
	}
	if n, err := strconv.ParseUint(h, 10, 32); err == nil {
		return time.Duration(n) * time.Second
	}
	if t, err := http.ParseTime(h); err == nil {
		d := time.Until(t)
		if d < 0 {
			return 0
		}
		return d
	}
	return 0
}

func mapStatusError(resp *http.Response) error {
	code := resp.StatusCode
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	retryAfterDur := parseRetryAfter(resp.Header.Get("Retry-After"))
	var retryAfterPtr *time.Duration
	if retryAfterDur > 0 {
		d := retryAfterDur
		retryAfterPtr = &d
	}
	message := ""
	var eb wireErrorBody
	if jerr := json.Unmarshal(body, &eb); jerr == nil && eb.Error != nil && eb.Error.Message != "" {
		message = eb.Error.Message
	} else {
		message = string(body)
		if len(message) > 500 {
			message = message[:500]
		}
	}
	switch code {
	case 429:
		return sporecore.NewRateLimited(retryAfterPtr)
	case 408, 504:
		return sporecore.NewTimeout()
	default:
		if code < 0 || code > math.MaxUint16 {
			code = 500
		}
		return sporecore.NewProviderError(uint16(code), message)
	}
}

// ---------------------------------------------------------------------------
// ModelInterface impl
// ---------------------------------------------------------------------------

// Call performs one blocking Chat Completions call.
func (c *ModelInterface) Call(ctx context.Context, req sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	body := buildRequest(c.modelID, req, false)
	payload, err := json.Marshal(body)
	if err != nil {
		return sporecore.ModelResponse{}, sporecore.NewProviderError(0, fmt.Sprintf("request encode failed: %v", err))
	}
	url := c.baseURL + "/chat/completions"
	resp, err := c.sendWithRetry(ctx, func(ctx context.Context) (*http.Request, error) {
		r, e := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if e != nil {
			return nil, e
		}
		c.setStandardHeaders(r)
		return r, nil
	})
	if err != nil {
		return sporecore.ModelResponse{}, err
	}
	defer resp.Body.Close()
	var wr wireResponse
	if err := json.NewDecoder(resp.Body).Decode(&wr); err != nil {
		return sporecore.ModelResponse{}, sporecore.NewProviderError(0, fmt.Sprintf("response decode failed: %v", err))
	}
	return parseResponse(wr), nil
}

// CountTokens estimates token usage using a bytes/4 heuristic. OpenAI does
// not expose a dedicated token-counting endpoint; this matches
// ReplayModelInterface and is sufficient for compaction decisions. Exact
// counts come back via response.usage.
func (c *ModelInterface) CountTokens(_ context.Context, req sporecore.ModelRequest) (uint32, error) {
	n := 0
	for _, m := range req.Messages {
		switch m.Content.Type {
		case sporecore.ContentTypeText:
			n += len(m.Content.Text)
		case sporecore.ContentTypeToolCall:
			if m.Content.ToolCall != nil {
				n += len(m.Content.ToolCall.Name) + len(m.Content.ToolCall.Input)
			}
		case sporecore.ContentTypeToolResult:
			if m.Content.ToolResult != nil {
				n += len(m.Content.ToolResult.Content)
			}
		case sporecore.ContentTypeImage:
			// images contribute 0 in the rough estimate
		}
	}
	return uint32(n / 4), nil
}

// CallStreaming opens an SSE stream and emits StreamEvents.
func (c *ModelInterface) CallStreaming(ctx context.Context, req sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	body := buildRequest(c.modelID, req, true)
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("request encode failed: %v", err))
	}
	url := c.baseURL + "/chat/completions"
	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		cancel()
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("HTTP request build failed: %v", err))
	}
	c.setStandardHeaders(httpReq)
	httpReq.Header.Set("Accept", "text/event-stream")
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		cancel()
		if isTimeout(err, reqCtx) {
			return nil, sporecore.NewTimeout()
		}
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("HTTP transport error: %v", err))
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		mapped := mapStatusError(resp)
		_ = resp.Body.Close()
		cancel()
		return nil, mapped
	}
	ch := make(chan sporecore.StreamEventOrErr, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		defer cancel()
		parseSSEStream(ctx, resp.Body, ch)
	}()
	return ch, nil
}

func (c *ModelInterface) setStandardHeaders(r *http.Request) {
	r.Header.Set("Authorization", "Bearer "+c.apiKey)
	r.Header.Set("content-type", "application/json")
}

// ---------------------------------------------------------------------------
// SSE parsing — OpenAI Chat Completions
// ---------------------------------------------------------------------------

// parseSSEStream reads OpenAI SSE chunks. Each `data:` line carries a JSON
// object shaped like:
//
//	{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}
//
// Tool calls arrive as partial entries in delta.tool_calls, indexed; the id
// and function.name arrive on the first chunk for a given index, and
// subsequent chunks carry incremental function.arguments fragment strings.
// The stream ends with `data: [DONE]`. When stream_options.include_usage was
// set, the final non-[DONE] chunk also carries usage.
func parseSSEStream(ctx context.Context, r io.Reader, ch chan<- sporecore.StreamEventOrErr) {
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 0, 1<<16)
	scanner.Buffer(buf, 1<<20)

	var (
		usage              sporecore.TokenUsage
		stopReason         = sporecore.StopEndTurn
		started            bool
		toolIndicesSeen    = map[uint32]bool{}
		contentIndex       uint32
		contentIndexActive bool
	)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		data = strings.TrimLeft(data, " ")
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			u := usage
			sendEvent(ctx, ch, sporecore.StreamEvent{
				Type: sporecore.StreamMessageStop, Usage: &u, StopReason: stopReason,
			})
			return
		}
		var value map[string]any
		if err := json.Unmarshal([]byte(data), &value); err != nil {
			continue
		}
		if !started {
			started = true
			if !sendEvent(ctx, ch, sporecore.StreamEvent{Type: sporecore.StreamMessageStart}) {
				return
			}
		}
		if u, ok := value["usage"].(map[string]any); ok {
			if pt, ok := u["prompt_tokens"].(float64); ok {
				usage.InputTokens = uint32(pt)
			}
			if ct, ok := u["completion_tokens"].(float64); ok {
				usage.OutputTokens = uint32(ct)
			}
			if d, ok := u["prompt_tokens_details"].(map[string]any); ok {
				if c, ok := d["cached_tokens"].(float64); ok {
					v := uint32(c)
					usage.CacheReadTokens = &v
				}
			}
		}
		choices, _ := value["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]any)
		if fr, ok := choice["finish_reason"].(string); ok {
			s := fr
			stopReason = parseStopReason(&s)
		}
		delta, _ := choice["delta"].(map[string]any)
		if delta == nil {
			continue
		}
		if text, ok := delta["content"].(string); ok && text != "" {
			contentIndexActive = true
			if !sendEvent(ctx, ch, sporecore.StreamEvent{
				Type: sporecore.StreamContentBlockDelta, Index: contentIndex, Delta: text,
			}) {
				return
			}
		}
		if reasoning, ok := delta["reasoning"].(string); ok && reasoning != "" {
			if !sendEvent(ctx, ch, sporecore.StreamEvent{
				Type: sporecore.StreamThinkingDelta, Index: contentIndex, Delta: reasoning,
			}) {
				return
			}
		}
		if tcs, ok := delta["tool_calls"].([]any); ok {
			for _, raw := range tcs {
				tc, _ := raw.(map[string]any)
				if tc == nil {
					continue
				}
				var i uint32
				if idx, ok := tc["index"].(float64); ok {
					i = uint32(idx)
				}
				eventIndex := i + 1
				if !toolIndicesSeen[eventIndex] {
					toolIndicesSeen[eventIndex] = true
					if contentIndexActive {
						if !sendEvent(ctx, ch, sporecore.StreamEvent{
							Type: sporecore.StreamContentBlockStop, Index: contentIndex,
						}) {
							return
						}
						contentIndexActive = false
						contentIndex = eventIndex
					}
				}
				var argDelta string
				if fn, ok := tc["function"].(map[string]any); ok {
					if a, ok := fn["arguments"].(string); ok {
						argDelta = a
					}
				}
				if argDelta != "" {
					if !sendEvent(ctx, ch, sporecore.StreamEvent{
						Type: sporecore.StreamToolUseDelta, Index: eventIndex, PartialJSON: argDelta,
					}) {
						return
					}
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		select {
		case ch <- sporecore.StreamEventOrErr{Err: sporecore.NewProviderError(0, fmt.Sprintf("stream read error: %v", err))}:
		case <-ctx.Done():
		}
		return
	}
	// Stream ended without explicit [DONE] marker — still emit terminator.
	u := usage
	sendEvent(ctx, ch, sporecore.StreamEvent{
		Type: sporecore.StreamMessageStop, Usage: &u, StopReason: stopReason,
	})
}

func sendEvent(ctx context.Context, ch chan<- sporecore.StreamEventOrErr, ev sporecore.StreamEvent) bool {
	select {
	case <-ctx.Done():
		return false
	case ch <- sporecore.StreamEventOrErr{Event: ev}:
		return true
	}
}
