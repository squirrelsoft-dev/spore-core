// Package anthropic implements ModelInterface against the Anthropic
// Messages API (https://api.anthropic.com/v1/messages).
//
// Issue #39. Translates the spore-core ModelRequest / ModelResponse types
// to and from Anthropic's wire format, parses the SSE event stream for
// CallStreaming, hits /v1/messages/count_tokens for accurate token counts,
// and maps HTTP errors to typed ModelError values with retry / backoff for
// transient failures.
package anthropic

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
	// DefaultBaseURL is the production Anthropic Messages endpoint.
	DefaultBaseURL = "https://api.anthropic.com"

	// DefaultTimeout is the per-request timeout.
	DefaultTimeout = 120 * time.Second

	// DefaultMaxRetries is the retry count for transient 408/425/429/500/
	// 502/503/504/529 responses and timeouts.
	DefaultMaxRetries = 3

	// AnthropicVersion is the API version header value pinned on every
	// request.
	AnthropicVersion = "2023-06-01"

	defaultMaxTokens uint32 = 4096
)

// ModelInterface is the Anthropic Messages API client. It implements
// sporecore.ModelInterface.
//
// The API key is never returned by Format or String and is omitted from any
// debug-style output.
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

// WithBaseURL overrides the base URL (use for proxying or mocking).
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
		"AnthropicModelInterface{api_key=<redacted>, model_id=%q, base_url=%q, timeout=%s, max_retries=%d}",
		c.modelID, c.baseURL, c.timeout, c.maxRetries,
	)
}

// GoString redacts the API key.
func (c *ModelInterface) GoString() string { return c.String() }

// ContextWindow returns the known context window for a model id. Any
// "claude-*" id falls back to 200_000; unknown providers return 0 so callers
// can detect them.
func ContextWindow(modelID string) uint32 {
	switch modelID {
	case "claude-sonnet-4-5", "claude-sonnet-4-6",
		"claude-opus-4-5", "claude-opus-4-6", "claude-opus-4-7",
		"claude-haiku-4-5", "claude-haiku-4-5-20251001":
		return 200_000
	}
	if strings.HasPrefix(modelID, "claude-") {
		return 200_000
	}
	return 0
}

// Provider reports the underlying model identity.
func (c *ModelInterface) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{
		Name:          "anthropic",
		ModelID:       c.modelID,
		ContextWindow: ContextWindow(c.modelID),
	}
}

var _ sporecore.ModelInterface = (*ModelInterface)(nil)

// ---------------------------------------------------------------------------
// Wire-format types
// ---------------------------------------------------------------------------

type wireRequest struct {
	Model         string        `json:"model"`
	MaxTokens     uint32        `json:"max_tokens"`
	Messages      []wireMessage `json:"messages"`
	System        string        `json:"system,omitempty"`
	Temperature   *float32      `json:"temperature,omitempty"`
	TopP          *float32      `json:"top_p,omitempty"`
	StopSequences []string      `json:"stop_sequences,omitempty"`
	Tools         []wireTool    `json:"tools,omitempty"`
	Stream        bool          `json:"stream,omitempty"`
}

type wireMessage struct {
	Role    string        `json:"role"`
	Content []wireContent `json:"content"`
}

type wireContent struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Source    *wireImageSrc   `json:"source,omitempty"`
}

type wireImageSrc struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type wireTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type wireResponse struct {
	Content    []wireResponseBlock `json:"content"`
	StopReason *string             `json:"stop_reason"`
	Usage      wireUsage           `json:"usage"`
}

type wireResponseBlock struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	Thinking string          `json:"thinking,omitempty"`
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name,omitempty"`
	Input    json.RawMessage `json:"input,omitempty"`
}

type wireUsage struct {
	InputTokens              uint32  `json:"input_tokens"`
	OutputTokens             uint32  `json:"output_tokens"`
	CacheReadInputTokens     *uint32 `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens *uint32 `json:"cache_creation_input_tokens,omitempty"`
}

type wireCountTokensResponse struct {
	InputTokens uint32 `json:"input_tokens"`
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

// buildRequest translates a sporecore.ModelRequest into the Anthropic wire
// body. System-role messages are joined into the top-level "system" field
// with "\n\n"; tool-role messages become user-role messages whose content
// blocks are tool_result entries.
func buildRequest(modelID string, req sporecore.ModelRequest, stream bool) wireRequest {
	var systemParts []string
	messages := make([]wireMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		if m.Role == sporecore.RoleSystem {
			if m.Content.Type == sporecore.ContentTypeText {
				systemParts = append(systemParts, m.Content.Text)
			}
			continue
		}
		role := "user"
		switch m.Role {
		case sporecore.RoleAssistant:
			role = "assistant"
		case sporecore.RoleUser, sporecore.RoleTool:
			// Tool-role results travel inside a user-role message.
			role = "user"
		}
		messages = append(messages, wireMessage{
			Role:    role,
			Content: []wireContent{contentToWire(m.Content)},
		})
	}
	system := ""
	if len(systemParts) > 0 {
		system = strings.Join(systemParts, "\n\n")
	}
	tools := make([]wireTool, 0, len(req.Tools))
	for _, t := range req.Tools {
		tools = append(tools, wireTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	maxTokens := defaultMaxTokens
	if req.Params.MaxTokens != nil {
		maxTokens = *req.Params.MaxTokens
	}
	return wireRequest{
		Model:         modelID,
		MaxTokens:     maxTokens,
		Messages:      messages,
		System:        system,
		Temperature:   req.Params.Temperature,
		TopP:          req.Params.TopP,
		StopSequences: req.Params.StopSequences,
		Tools:         tools,
		Stream:        stream,
	}
}

func contentToWire(c sporecore.Content) wireContent {
	switch c.Type {
	case sporecore.ContentTypeText:
		return wireContent{Type: "text", Text: c.Text}
	case sporecore.ContentTypeToolCall:
		if c.ToolCall != nil {
			return wireContent{
				Type:  "tool_use",
				ID:    c.ToolCall.ID,
				Name:  c.ToolCall.Name,
				Input: c.ToolCall.Input,
			}
		}
		return wireContent{Type: "tool_use"}
	case sporecore.ContentTypeToolResult:
		if c.ToolResult != nil {
			return wireContent{
				Type:      "tool_result",
				ToolUseID: c.ToolResult.ToolUseID,
				Content:   c.ToolResult.Content,
				IsError:   c.ToolResult.IsError,
			}
		}
		return wireContent{Type: "tool_result"}
	case sporecore.ContentTypeImage:
		return wireContent{
			Type: "image",
			Source: &wireImageSrc{
				Type:      "base64",
				MediaType: c.MediaType,
				Data:      c.Data,
			},
		}
	}
	return wireContent{Type: "text", Text: c.Text}
}

func parseResponse(body wireResponse) sporecore.ModelResponse {
	blocks := make([]sporecore.ContentBlock, 0, len(body.Content))
	for _, b := range body.Content {
		switch b.Type {
		case "text":
			blocks = append(blocks, sporecore.NewTextBlock(b.Text))
		case "thinking":
			blocks = append(blocks, sporecore.NewThinkingBlock(b.Thinking))
		case "tool_use":
			blocks = append(blocks, sporecore.NewToolUseBlock(sporecore.ToolCall{
				ID:    b.ID,
				Name:  b.Name,
				Input: b.Input,
			}))
		}
	}
	usage := sporecore.TokenUsage{
		InputTokens:      body.Usage.InputTokens,
		OutputTokens:     body.Usage.OutputTokens,
		CacheReadTokens:  body.Usage.CacheReadInputTokens,
		CacheWriteTokens: body.Usage.CacheCreationInputTokens,
	}
	var stop *string
	stop = body.StopReason
	return sporecore.ModelResponse{
		Content:    blocks,
		Usage:      usage,
		StopReason: parseStopReason(stop),
	}
}

func parseStopReason(s *string) sporecore.StopReason {
	if s == nil {
		return sporecore.StopEndTurn
	}
	switch *s {
	case "tool_use":
		return sporecore.StopToolUse
	case "max_tokens":
		return sporecore.StopMaxTokens
	case "stop_sequence":
		return sporecore.StopStopSequence
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
	case 408, 425, 429, 500, 502, 503, 504, 529:
		return true
	}
	return false
}

// sendWithRetry issues the request returned by build() and retries transient
// failures up to maxRetries. Each attempt gets the per-request timeout via
// context.WithTimeout.
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
			return nil, sporecore.NewTransport(fmt.Sprintf("HTTP transport error: %v", err))
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			// IMPORTANT: cancel must not run yet — the body is still being
			// read by the caller. Spawn a goroutine that cancels when the
			// body closes by wrapping in a closer-aware reader.
			resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, cancel: cancel}
			return resp, nil
		}
		// Non-2xx: maybe retry.
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
	// Try to extract Anthropic's error.message.
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
	case 529:
		return sporecore.NewRateLimited(nil)
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

// Call performs one blocking Messages call.
func (c *ModelInterface) Call(ctx context.Context, req sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	body := buildRequest(c.modelID, req, false)
	payload, err := json.Marshal(body)
	if err != nil {
		return sporecore.ModelResponse{}, sporecore.NewProviderError(0, fmt.Sprintf("request encode failed: %v", err))
	}
	url := c.baseURL + "/v1/messages"
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

// CountTokens hits /v1/messages/count_tokens for an accurate count.
func (c *ModelInterface) CountTokens(ctx context.Context, req sporecore.ModelRequest) (uint32, error) {
	body := buildRequest(c.modelID, req, false)
	payload, err := json.Marshal(body)
	if err != nil {
		return 0, sporecore.NewProviderError(0, fmt.Sprintf("count_tokens encode failed: %v", err))
	}
	url := c.baseURL + "/v1/messages/count_tokens"
	resp, err := c.sendWithRetry(ctx, func(ctx context.Context) (*http.Request, error) {
		r, e := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if e != nil {
			return nil, e
		}
		c.setStandardHeaders(r)
		return r, nil
	})
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	var ct wireCountTokensResponse
	if err := json.NewDecoder(resp.Body).Decode(&ct); err != nil {
		return 0, sporecore.NewProviderError(0, fmt.Sprintf("count_tokens decode failed: %v", err))
	}
	return ct.InputTokens, nil
}

// CallStreaming opens an SSE stream and emits StreamEvents.
func (c *ModelInterface) CallStreaming(ctx context.Context, req sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	body := buildRequest(c.modelID, req, true)
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("request encode failed: %v", err))
	}
	url := c.baseURL + "/v1/messages"
	// Streaming responses are not retried — open once.
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
		return nil, sporecore.NewTransport(fmt.Sprintf("HTTP transport error: %v", err))
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
	r.Header.Set("x-api-key", c.apiKey)
	r.Header.Set("anthropic-version", AnthropicVersion)
	r.Header.Set("content-type", "application/json")
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

// parseSSEStream reads SSE events from r and writes StreamEvents to ch. The
// caller is responsible for closing ch when this returns.
func parseSSEStream(ctx context.Context, r io.Reader, ch chan<- sporecore.StreamEventOrErr) {
	scanner := bufio.NewScanner(r)
	// Allow generously large event payloads (tool input deltas, etc.).
	buf := make([]byte, 0, 1<<16)
	scanner.Buffer(buf, 1<<20)

	var (
		eventName  string
		dataLines  []string
		usage      sporecore.TokenUsage
		stopReason = sporecore.StopEndTurn
	)

	flush := func() bool {
		defer func() {
			eventName = ""
			dataLines = dataLines[:0]
		}()
		if eventName == "" {
			return true
		}
		data := strings.Join(dataLines, "\n")
		if data == "" || data == "{}" {
			return true
		}
		var v map[string]any
		if err := json.Unmarshal([]byte(data), &v); err != nil {
			return true
		}
		switch eventName {
		case "message_start":
			if msg, ok := v["message"].(map[string]any); ok {
				if u, ok := msg["usage"].(map[string]any); ok {
					if it, ok := u["input_tokens"].(float64); ok {
						usage.InputTokens = uint32(it)
					}
				}
			}
			return sendEvent(ctx, ch, sporecore.StreamEvent{Type: sporecore.StreamMessageStart})
		case "content_block_start":
			// A tool_use block opens here with its id + name; emit tool_use_start
			// so the accumulator captures them before the input_json_delta arg
			// fragments arrive.
			idx := readUint32(v["index"])
			block, _ := v["content_block"].(map[string]any)
			if kind, _ := block["type"].(string); kind == "tool_use" {
				id, _ := block["id"].(string)
				name, _ := block["name"].(string)
				return sendEvent(ctx, ch, sporecore.StreamEvent{
					Type: sporecore.StreamToolUseStart, Index: idx, ID: id, Name: name,
				})
			}
			return true
		case "content_block_delta":
			idx := readUint32(v["index"])
			delta, _ := v["delta"].(map[string]any)
			kind, _ := delta["type"].(string)
			switch kind {
			case "text_delta":
				text, _ := delta["text"].(string)
				return sendEvent(ctx, ch, sporecore.StreamEvent{
					Type: sporecore.StreamContentBlockDelta, Index: idx, Delta: text,
				})
			case "thinking_delta":
				text, _ := delta["thinking"].(string)
				return sendEvent(ctx, ch, sporecore.StreamEvent{
					Type: sporecore.StreamThinkingDelta, Index: idx, Delta: text,
				})
			case "input_json_delta":
				partial, _ := delta["partial_json"].(string)
				return sendEvent(ctx, ch, sporecore.StreamEvent{
					Type: sporecore.StreamToolUseDelta, Index: idx, PartialJSON: partial,
				})
			}
			return true
		case "content_block_stop":
			idx := readUint32(v["index"])
			return sendEvent(ctx, ch, sporecore.StreamEvent{Type: sporecore.StreamContentBlockStop, Index: idx})
		case "message_delta":
			if d, ok := v["delta"].(map[string]any); ok {
				if s, ok := d["stop_reason"].(string); ok {
					stopReason = parseStopReason(&s)
				}
			}
			if u, ok := v["usage"].(map[string]any); ok {
				if ot, ok := u["output_tokens"].(float64); ok {
					usage.OutputTokens = uint32(ot)
				}
			}
			return true
		case "message_stop":
			u := usage
			return sendEvent(ctx, ch, sporecore.StreamEvent{
				Type: sporecore.StreamMessageStop, Usage: &u, StopReason: stopReason,
			})
		}
		return true
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if !flush() {
				return
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			// Comment / keepalive.
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	// Process any trailing event without a closing blank line.
	flush()
	if err := scanner.Err(); err != nil {
		select {
		case ch <- sporecore.StreamEventOrErr{Err: sporecore.NewStreamInterrupted(fmt.Sprintf("stream chunk error: %v", err))}:
		case <-ctx.Done():
		}
	}
}

func readUint32(v any) uint32 {
	switch x := v.(type) {
	case float64:
		return uint32(x)
	case int:
		return uint32(x)
	case json.Number:
		if n, err := x.Int64(); err == nil {
			return uint32(n)
		}
	}
	return 0
}

func sendEvent(ctx context.Context, ch chan<- sporecore.StreamEventOrErr, ev sporecore.StreamEvent) bool {
	select {
	case <-ctx.Done():
		return false
	case ch <- sporecore.StreamEventOrErr{Event: ev}:
		return true
	}
}
