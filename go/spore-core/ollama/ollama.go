// Package ollama implements ModelInterface against a local Ollama server's
// /api/chat, /api/tags, and /api/embed endpoints.
//
// Issue #41. Translates the spore-core ModelRequest / ModelResponse types to
// and from the Ollama wire format, parses Ollama's NDJSON stream (one JSON
// object per line — not SSE) for CallStreaming, and maps HTTP/transport
// errors to typed ModelError variants.
//
// Unlike the Anthropic and OpenAI clients there is NO retry: spec says fail
// fast on connection errors with a helpful message ("Ollama not running",
// "Run: ollama pull <model>").
//
// Provider-specific shape
//
//   - No API key; default base URL is http://localhost:11434.
//   - Sampling parameters (num_predict, temperature, top_p, stop) are nested
//     under `options` rather than being top-level keys.
//   - keep_alive (default "5m") controls how long Ollama keeps the model
//     loaded after the call returns.
//   - Tool-call arguments are a JSON OBJECT in the wire format, not a
//     JSON-encoded string like OpenAI.
//   - Ollama does not return tool-call ids; we synthesize "call-{i}" per
//     index so downstream ToolResult.tool_use_id round-trips work.
//   - Thinking blocks are silently omitted from outgoing requests — Ollama
//     has no structured reasoning shape.
//   - Cache fields on TokenUsage are always nil (Ollama has no prefix
//     caching).
//   - Model availability is verified lazily against /api/tags on first call
//     and cached for the lifetime of the instance.
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ---------------------------------------------------------------------------
// Public client
// ---------------------------------------------------------------------------

const (
	// DefaultBaseURL is the local Ollama endpoint.
	DefaultBaseURL = "http://localhost:11434"

	// DefaultTimeout is the per-request timeout. Local models are slower
	// than hosted providers, so 300s is generous.
	DefaultTimeout = 300 * time.Second

	// DefaultKeepAlive tells Ollama to keep the model loaded for 5 minutes
	// after the call returns.
	DefaultKeepAlive = "5m"
)

// ModelInterface is the Ollama HTTP client. It implements
// sporecore.ModelInterface.
type ModelInterface struct {
	modelID    string
	baseURL    string
	timeout    time.Duration
	keepAlive  string
	httpClient *http.Client

	// Lazy /api/tags availability + /api/show discovery probe. The first
	// Call/CallStreaming triggers the check; the result is cached for the
	// instance lifetime. checkMeta holds the /api/show-discovered metadata
	// (best-effort; empty when discovery failed but availability succeeded).
	checkOnce sync.Once
	checkErr  error
	checkMeta modelMeta
	metaReady atomic.Bool
}

// modelMeta is the /api/show-discovered metadata for the model. Populated once,
// alongside the /api/tags availability check. All fields are best-effort —
// /api/show failures leave them unset rather than failing the call.
type modelMeta struct {
	// contextLength is the discovered context window (*.context_length in
	// model_info). nil when /api/show was unavailable or omitted it.
	contextLength *uint32
	// capabilities is the top-level capabilities array (may contain "tools").
	capabilities []string
}

func (m modelMeta) supportsTools() bool {
	for _, c := range m.capabilities {
		if c == "tools" {
			return true
		}
	}
	return false
}

// New constructs a ModelInterface with localhost defaults.
func New(modelID string) *ModelInterface {
	return &ModelInterface{
		modelID:    modelID,
		baseURL:    DefaultBaseURL,
		timeout:    DefaultTimeout,
		keepAlive:  DefaultKeepAlive,
		httpClient: &http.Client{},
	}
}

// WithBaseURL constructs a ModelInterface pointing at a non-default Ollama
// host (remote instance, custom port, mock server).
func WithBaseURL(modelID, baseURL string) *ModelInterface {
	c := New(modelID)
	c.baseURL = baseURL
	return c
}

// SetTimeout overrides the per-request timeout.
func (c *ModelInterface) SetTimeout(d time.Duration) *ModelInterface {
	c.timeout = d
	return c
}

// SetKeepAlive overrides how long Ollama keeps the model resident after a
// call. Pass empty string to omit the field entirely.
func (c *ModelInterface) SetKeepAlive(s string) *ModelInterface {
	c.keepAlive = s
	return c
}

// SetHTTPClient overrides the http.Client.
func (c *ModelInterface) SetHTTPClient(h *http.Client) *ModelInterface {
	c.httpClient = h
	return c
}

// String returns a debug representation.
func (c *ModelInterface) String() string {
	return fmt.Sprintf(
		"OllamaModelInterface{model_id=%q, base_url=%q, timeout=%s, keep_alive=%q}",
		c.modelID, c.baseURL, c.timeout, c.keepAlive,
	)
}

// GoString matches String.
func (c *ModelInterface) GoString() string { return c.String() }

// ContextWindow returns the known context window for an Ollama model id from
// the static prefix table. Unknown models return 0 so callers can detect them.
// Provider prefers the /api/show-discovered value (when the probe has run and
// produced one), falling back to this table.
func ContextWindow(modelID string) uint32 {
	switch {
	case strings.HasPrefix(modelID, "llama3.2"):
		return 128_000
	case strings.HasPrefix(modelID, "qwen2.5-coder"):
		return 128_000
	case strings.HasPrefix(modelID, "mistral"):
		return 32_000
	case strings.HasPrefix(modelID, "gemma"):
		return 8_192
	default:
		return 0
	}
}

// Provider reports the underlying model identity. The context window prefers
// the /api/show-discovered value when the availability probe has already run
// and produced one; otherwise it falls back to the static ContextWindow table.
func (c *ModelInterface) Provider() sporecore.ProviderInfo {
	cw := ContextWindow(c.modelID)
	if c.metaReady.Load() && c.checkMeta.contextLength != nil {
		cw = *c.checkMeta.contextLength
	}
	return sporecore.ProviderInfo{
		Name:          "ollama",
		ModelID:       c.modelID,
		ContextWindow: cw,
	}
}

var _ sporecore.ModelInterface = (*ModelInterface)(nil)

// ---------------------------------------------------------------------------
// Wire-format types (Ollama Chat API)
// ---------------------------------------------------------------------------

type wireRequest struct {
	Model     string        `json:"model"`
	Messages  []wireMessage `json:"messages"`
	Stream    bool          `json:"stream"`
	KeepAlive string        `json:"keep_alive,omitempty"`
	Options   *wireOptions  `json:"options,omitempty"`
	Tools     []wireTool    `json:"tools,omitempty"`
	// Format is the constrained-decoding JSON schema (Ollama's `format`
	// parameter). Set only in structured-tool-calls mode
	// (ModelParams.StructuredToolCalls); when present, native `tools` are
	// dropped and the model is forced to emit a single schema-constrained JSON
	// object instead of routing tool calls through Llama's `<|python_tag|>`
	// channel (which Ollama does not parse, causing the call to leak into
	// message.content).
	Format json.RawMessage `json:"format,omitempty"`
}

type wireOptions struct {
	NumPredict  *uint32  `json:"num_predict,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

func (o *wireOptions) isEmpty() bool {
	return o.NumPredict == nil && o.Temperature == nil && o.TopP == nil && len(o.Stop) == 0
}

type wireMessage struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCalls  []wireToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type wireToolCall struct {
	Function wireFunctionCall `json:"function"`
}

type wireFunctionCall struct {
	Name string `json:"name"`
	// Arguments is a JSON object (not a JSON-encoded string).
	Arguments json.RawMessage `json:"arguments"`
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
	Message         wireResponseMessage `json:"message"`
	Done            bool                `json:"done"`
	DoneReason      string              `json:"done_reason"`
	PromptEvalCount *uint32             `json:"prompt_eval_count,omitempty"`
	EvalCount       *uint32             `json:"eval_count,omitempty"`
}

type wireResponseMessage struct {
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	ToolCalls []wireResponseToolCall `json:"tool_calls"`
}

type wireResponseToolCall struct {
	Function wireResponseFunctionCall `json:"function"`
}

type wireResponseFunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type wireTagsResponse struct {
	Models []wireTagModel `json:"models"`
}

type wireTagModel struct {
	Name string `json:"name"`
}

type wireShowRequest struct {
	Model string `json:"model"`
}

type wireShowResponse struct {
	// ModelInfo holds architecture-specific keys; we look for *.context_length.
	ModelInfo map[string]json.RawMessage `json:"model_info"`
	// Capabilities is the top-level capabilities array (may contain "tools").
	Capabilities []string `json:"capabilities"`
}

type wireEmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type wireEmbedResponse struct {
	PromptEvalCount *uint32 `json:"prompt_eval_count,omitempty"`
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

func buildRequest(modelID, keepAlive string, req sporecore.ModelRequest, stream bool) wireRequest {
	// Structured-tool-calls mode (opt-in): constrained decoding via `format`.
	// We send NO native tools — describing them in a system message instead —
	// and force the model to emit a single schema-constrained JSON object. The
	// `tool` enum is the critical constraint: it eliminates the `<|python_tag|>`
	// leak that small local models (llama3.2) otherwise produce.
	structured := req.Params.StructuredToolCalls && len(req.Tools) > 0

	messages := make([]wireMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		messages = append(messages, messageToWire(m))
	}

	var tools []wireTool
	var format json.RawMessage
	if structured {
		// Inject a system message describing the tools. Merge into an existing
		// leading system message when present; otherwise prepend a new one.
		preamble := structuredToolsPreamble(req.Tools)
		if len(messages) > 0 && messages[0].Role == "system" {
			messages[0].Content = messages[0].Content + "\n\n" + preamble
		} else {
			messages = append([]wireMessage{{Role: "system", Content: preamble}}, messages...)
		}
		format = structuredFormatSchema(req.Tools)
	} else {
		tools = make([]wireTool, 0, len(req.Tools))
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
		// Issue #139: the harness sets req.Params.OutputSchema for the terminal turn
		// of an output-schema-enforced ReAct leaf. Route it into the same `format`
		// constrained-decoding channel the structured-tool-calls path uses, so the
		// model is forced onto the schema — even with NO tools. (When structured tool
		// calls ARE active, that schema wins via the `if structured` arm above, since
		// the leaf is still requesting tools, not emitting its terminal.) Absent (the
		// default) leaves format nil, byte-identical to pre-#139.
		if len(req.Params.OutputSchema) > 0 {
			format = req.Params.OutputSchema
		}
	}

	opts := &wireOptions{
		NumPredict:  req.Params.MaxTokens,
		Temperature: req.Params.Temperature,
		TopP:        req.Params.TopP,
		Stop:        req.Params.StopSequences,
	}
	var optsOut *wireOptions
	if !opts.isEmpty() {
		optsOut = opts
	}

	return wireRequest{
		Model:     modelID,
		Messages:  messages,
		Stream:    stream,
		KeepAlive: keepAlive,
		Options:   optsOut,
		Tools:     tools,
		Format:    format,
	}
}

// structuredFormatSchema builds the constrained-decoding JSON schema for
// structured-tool-calls mode. The `tool` enum lists every tool name plus
// "final"; this enum is the critical constraint that keeps tool calls on the
// constrained-JSON channel and away from Llama's `<|python_tag|>`.
func structuredFormatSchema(tools []sporecore.ToolSchema) json.RawMessage {
	names := make([]string, 0, len(tools)+1)
	for _, t := range tools {
		names = append(names, t.Name)
	}
	names = append(names, "final")
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tool":      map[string]any{"type": "string", "enum": names},
			"arguments": map[string]any{"type": "object"},
			"content":   map[string]any{"type": "string"},
		},
		"required": []string{"tool"},
	}
	encoded, err := json.Marshal(schema)
	if err != nil {
		return nil
	}
	return encoded
}

// structuredToolsPreamble builds the system-message preamble for
// structured-tool-calls mode. It describes each tool's name, description, and
// parameter property names/types (read from t.InputSchema), then gives the
// model explicit single-JSON-object output instructions. This — together with
// the `format` schema's `tool` enum — keeps tool calls on the constrained-JSON
// channel and away from `<|python_tag|>`.
func structuredToolsPreamble(tools []sporecore.ToolSchema) string {
	var b strings.Builder
	b.WriteString("You have access to the following tools:\n")
	for _, t := range tools {
		b.WriteString(fmt.Sprintf("\n- %s: %s", t.Name, t.Description))
		params := schemaParamSummary(t.InputSchema)
		if params != "" {
			b.WriteString("\n  parameters: ")
			b.WriteString(params)
		}
	}
	b.WriteString("\n\nRespond with a SINGLE JSON object and nothing else. " +
		"To call a tool, set \"tool\" to the tool name and \"arguments\" to its " +
		"inputs. When the task is fully done, set \"tool\" to \"final\" and put " +
		"your reply in \"content\".")
	return b.String()
}

// schemaParamSummary extracts a "name (type), ..." summary from a tool input
// schema's `properties` object. Returns "" when there are no properties.
func schemaParamSummary(inputSchema json.RawMessage) string {
	if len(inputSchema) == 0 {
		return ""
	}
	var parsed struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(inputSchema, &parsed); err != nil {
		return ""
	}
	if len(parsed.Properties) == 0 {
		return ""
	}
	// Sort property names for deterministic output (Go map order is random).
	names := make([]string, 0, len(parsed.Properties))
	for name := range parsed.Properties {
		names = append(names, name)
	}
	sort.Strings(names)
	parts := make([]string, 0, len(names))
	for _, name := range names {
		ty := "any"
		var ps struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(parsed.Properties[name], &ps); err == nil && ps.Type != "" {
			ty = ps.Type
		}
		parts = append(parts, fmt.Sprintf("%s (%s)", name, ty))
	}
	return strings.Join(parts, ", ")
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
		return wireMessage{Role: role, Content: m.Content.Text}
	case sporecore.ContentTypeToolCall:
		if m.Content.ToolCall != nil {
			args := m.Content.ToolCall.Input
			if len(args) == 0 {
				args = json.RawMessage("{}")
			}
			return wireMessage{
				Role: "assistant",
				ToolCalls: []wireToolCall{{
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
			return wireMessage{
				Role:       "tool",
				Content:    m.Content.ToolResult.Content,
				ToolCallID: m.Content.ToolResult.ToolUseID,
			}
		}
		return wireMessage{Role: "tool"}
	case sporecore.ContentTypeImage:
		// Ollama vision support varies by model; the harness does not
		// currently emit image content. Emit a textual placeholder so
		// the request remains well-formed.
		return wireMessage{Role: role, Content: fmt.Sprintf("[image: %s]", m.Content.MediaType)}
	}
	return wireMessage{Role: role, Content: m.Content.Text}
}

func parseResponse(body wireResponse, structured bool) sporecore.ModelResponse {
	usage := sporecore.TokenUsage{}
	if body.PromptEvalCount != nil {
		usage.InputTokens = *body.PromptEvalCount
	}
	if body.EvalCount != nil {
		usage.OutputTokens = *body.EvalCount
	}

	if structured {
		// In structured mode the assistant content is a single constrained JSON
		// object — never a native tool_calls array. Parsing it back into a
		// tool-use block (rather than treating the raw JSON as answer text) is
		// precisely what avoids the `<|python_tag|>` leak: the tool call never
		// touches the native channel.
		blocks, stopReason := parseStructuredContent(body.Message.Content, 0)
		return sporecore.ModelResponse{
			Content:    blocks,
			Usage:      usage,
			StopReason: stopReason,
		}
	}

	blocks := make([]sporecore.ContentBlock, 0, 1+len(body.Message.ToolCalls))
	if body.Message.Content != "" {
		blocks = append(blocks, sporecore.NewTextBlock(body.Message.Content))
	}
	for i, tc := range body.Message.ToolCalls {
		input := tc.Function.Arguments
		if len(input) == 0 || string(input) == "null" {
			input = json.RawMessage("{}")
		}
		blocks = append(blocks, sporecore.NewToolUseBlock(sporecore.ToolCall{
			ID:    fmt.Sprintf("call-%d", i),
			Name:  tc.Function.Name,
			Input: input,
		}))
	}

	return sporecore.ModelResponse{
		Content:    blocks,
		Usage:      usage,
		StopReason: parseStopReason(body.DoneReason),
	}
}

// parseStructuredContent parses the constrained-decoding JSON object produced
// in structured-tool-calls mode into (content blocks, stop reason). index is
// used to synthesize the tool-call id, reusing this file's "call-{i}"
// convention.
//
// Defensive: if raw is missing/empty/not valid JSON/lacks a `tool` field, fall
// back to a single text block with the raw content and StopEndTurn — weak
// models occasionally violate even constrained decoding, and we never panic on
// their output.
func parseStructuredContent(raw string, index int) ([]sporecore.ContentBlock, sporecore.StopReason) {
	fallback := func() ([]sporecore.ContentBlock, sporecore.StopReason) {
		return []sporecore.ContentBlock{sporecore.NewTextBlock(raw)}, sporecore.StopEndTurn
	}
	// Capable/cloud models often ignore the constrained-decoding grammar and
	// wrap the JSON tool call in a markdown code fence. Reuse the plan parser's
	// fence stripping so a fenced `{"tool":...}` still dispatches instead of
	// being mis-read as a final text answer.
	trimmed := sporecore.StripCodeFence(strings.TrimSpace(raw))
	var value map[string]json.RawMessage
	if err := json.Unmarshal([]byte(trimmed), &value); err != nil {
		return fallback()
	}
	var tool string
	rawTool, ok := value["tool"]
	if !ok {
		return fallback()
	}
	if err := json.Unmarshal(rawTool, &tool); err != nil || tool == "" {
		return fallback()
	}
	if tool == "final" {
		text := ""
		if rawContent, ok := value["content"]; ok {
			_ = json.Unmarshal(rawContent, &text)
		}
		return []sporecore.ContentBlock{sporecore.NewTextBlock(text)}, sporecore.StopEndTurn
	}
	input := json.RawMessage("{}")
	if rawArgs, ok := value["arguments"]; ok {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(rawArgs, &obj); err == nil && obj != nil {
			input = rawArgs
		}
	}
	return []sporecore.ContentBlock{sporecore.NewToolUseBlock(sporecore.ToolCall{
		ID:    fmt.Sprintf("call-%d", index),
		Name:  tool,
		Input: input,
	})}, sporecore.StopToolUse
}

func parseStopReason(s string) sporecore.StopReason {
	switch s {
	case "tool_calls":
		return sporecore.StopToolUse
	case "length":
		return sporecore.StopMaxTokens
	case "stop":
		return sporecore.StopEndTurn
	default:
		return sporecore.StopEndTurn
	}
}

// nameMatches compares an Ollama tag (e.g. "llama3.2:latest") to a requested
// model id. Matches the full tag or its bare-name prefix.
func nameMatches(tag, requested string) bool {
	if tag == requested {
		return true
	}
	if i := strings.IndexByte(tag, ':'); i >= 0 {
		return tag[:i] == requested
	}
	return false
}

// ---------------------------------------------------------------------------
// HTTP plumbing (no retry — fail fast per spec)
// ---------------------------------------------------------------------------

// transportError classifies a transport-layer error into a typed ModelError.
// Connection refused → helpful "Ollama not running" message. Timeout →
// ModelErrTimeout. Anything else → generic ProviderError.
func (c *ModelInterface) transportError(reqCtx context.Context, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return sporecore.NewTimeout()
	}
	if reqCtx != nil && errors.Is(reqCtx.Err(), context.DeadlineExceeded) {
		return sporecore.NewTimeout()
	}
	var ne interface{ Timeout() bool }
	if errors.As(err, &ne) && ne.Timeout() {
		return sporecore.NewTimeout()
	}
	if isConnectionError(err) {
		return sporecore.NewProviderError(0, fmt.Sprintf("Ollama not running at %s", c.baseURL))
	}
	return sporecore.NewProviderError(0, fmt.Sprintf("HTTP transport error: %v", err))
}

// isConnectionError reports whether the error looks like an inability to
// reach the server (connection refused, no route, DNS, etc.) — anything
// that maps to the "Ollama not running" hint.
func isConnectionError(err error) bool {
	if err == nil {
		return false
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		if urlErr.Err != nil {
			err = urlErr.Err
		}
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	msg := err.Error()
	for _, frag := range []string{
		"connection refused",
		"no such host",
		"no route to host",
		"connect: cannot",
		"network is unreachable",
		"actively refused",
	} {
		if strings.Contains(msg, frag) {
			return true
		}
	}
	return false
}

// mapStatusError converts a non-2xx response into a typed ModelError. 404 on
// either /api/chat or /api/tags surfaces as the helpful "ollama pull" hint.
func (c *ModelInterface) mapStatusError(resp *http.Response) error {
	code := resp.StatusCode
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	switch code {
	case 404:
		return sporecore.NewProviderError(404, fmt.Sprintf(
			"Model %s not found. Run: ollama pull %s", c.modelID, c.modelID,
		))
	case 408, 504:
		return sporecore.NewTimeout()
	}
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = http.StatusText(code)
	}
	if len(msg) > 500 {
		msg = msg[:500]
	}
	if code < 0 || code > math.MaxUint16 {
		code = 500
	}
	return sporecore.NewProviderError(uint16(code), msg)
}

// ensureModelAvailable performs the one-time /api/tags probe. The result is
// cached via sync.Once for the instance lifetime — every subsequent call
// returns the same checkErr without hitting the server.
func (c *ModelInterface) ensureModelAvailable(ctx context.Context) error {
	c.checkOnce.Do(func() {
		c.checkErr = c.checkModel(ctx)
		if c.checkErr == nil {
			// Best-effort discovery — never fails the call.
			c.checkMeta = c.discoverMeta(ctx)
		}
		c.metaReady.Store(true)
	})
	return c.checkErr
}

func (c *ModelInterface) checkModel(ctx context.Context) error {
	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	r, err := http.NewRequestWithContext(reqCtx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return sporecore.NewProviderError(0, fmt.Sprintf("HTTP request build failed: %v", err))
	}
	resp, err := c.httpClient.Do(r)
	if err != nil {
		return c.transportError(reqCtx, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return c.mapStatusError(resp)
	}
	var tags wireTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return sporecore.NewProviderError(0, fmt.Sprintf("tags decode failed: %v", err))
	}
	for _, m := range tags.Models {
		if nameMatches(m.Name, c.modelID) {
			return nil
		}
	}
	return sporecore.NewProviderError(404, fmt.Sprintf(
		"Model %s not found. Run: ollama pull %s", c.modelID, c.modelID,
	))
}

// discoverMeta performs a best-effort POST /api/show. It returns an empty
// modelMeta on any failure (404, transport error, decode error, missing
// fields) so that discovery being unavailable never errors the whole call.
func (c *ModelInterface) discoverMeta(ctx context.Context) modelMeta {
	payload, err := json.Marshal(wireShowRequest{Model: c.modelID})
	if err != nil {
		return modelMeta{}
	}
	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	r, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL+"/api/show", bytes.NewReader(payload))
	if err != nil {
		return modelMeta{}
	}
	r.Header.Set("content-type", "application/json")
	resp, err := c.httpClient.Do(r)
	if err != nil {
		return modelMeta{}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return modelMeta{}
	}
	var show wireShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return modelMeta{}
	}
	meta := modelMeta{capabilities: show.Capabilities}
	for k, v := range show.ModelInfo {
		if !strings.HasSuffix(k, ".context_length") {
			continue
		}
		var n uint64
		if err := json.Unmarshal(v, &n); err == nil {
			cl := uint32(n)
			meta.contextLength = &cl
		}
		break
	}
	return meta
}

// guardToolSupport rejects tool-bearing requests when the model does not
// support tools. The /api/show capabilities array is the sole authority for
// tool support: the model is tool-capable iff capabilities contains "tools".
// Empty or unavailable /api/show metadata ⟹ NOT tool-capable (fail closed).
func (c *ModelInterface) guardToolSupport(req sporecore.ModelRequest) error {
	if len(req.Tools) == 0 {
		return nil
	}
	supported := c.checkMeta.supportsTools()
	if !supported {
		return sporecore.NewProviderError(0, fmt.Sprintf(
			"Model %s does not support tool calling", c.modelID,
		))
	}
	return nil
}

// ---------------------------------------------------------------------------
// ModelInterface impl
// ---------------------------------------------------------------------------

// Call performs one blocking /api/chat call.
func (c *ModelInterface) Call(ctx context.Context, req sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	if err := c.ensureModelAvailable(ctx); err != nil {
		return sporecore.ModelResponse{}, err
	}
	if err := c.guardToolSupport(req); err != nil {
		return sporecore.ModelResponse{}, err
	}
	body := buildRequest(c.modelID, c.keepAlive, req, false)
	payload, err := json.Marshal(body)
	if err != nil {
		return sporecore.ModelResponse{}, sporecore.NewProviderError(0, fmt.Sprintf("request encode failed: %v", err))
	}
	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(payload))
	if err != nil {
		return sporecore.ModelResponse{}, sporecore.NewProviderError(0, fmt.Sprintf("HTTP request build failed: %v", err))
	}
	httpReq.Header.Set("content-type", "application/json")
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return sporecore.ModelResponse{}, c.transportError(reqCtx, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return sporecore.ModelResponse{}, c.mapStatusError(resp)
	}
	var wr wireResponse
	if err := json.NewDecoder(resp.Body).Decode(&wr); err != nil {
		return sporecore.ModelResponse{}, sporecore.NewProviderError(0, fmt.Sprintf("response decode failed: %v", err))
	}
	structured := req.Params.StructuredToolCalls && len(req.Tools) > 0
	return parseResponse(wr, structured), nil
}

// CountTokens estimates token usage. Ollama has no dedicated count endpoint
// — try /api/embed for prompt_eval_count, then fall back to bytes/4.
func (c *ModelInterface) CountTokens(ctx context.Context, req sporecore.ModelRequest) (uint32, error) {
	text := concatRequestText(req)
	if n, ok := c.tryEmbedCount(ctx, text); ok {
		return n, nil
	}
	return uint32(len(text) / 4), nil
}

func (c *ModelInterface) tryEmbedCount(ctx context.Context, text string) (uint32, bool) {
	payload, err := json.Marshal(wireEmbedRequest{Model: c.modelID, Input: text})
	if err != nil {
		return 0, false
	}
	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	r, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL+"/api/embed", bytes.NewReader(payload))
	if err != nil {
		return 0, false
	}
	r.Header.Set("content-type", "application/json")
	resp, err := c.httpClient.Do(r)
	if err != nil {
		return 0, false
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0, false
	}
	var er wireEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&er); err != nil {
		return 0, false
	}
	if er.PromptEvalCount == nil {
		return 0, false
	}
	return *er.PromptEvalCount, true
}

func concatRequestText(req sporecore.ModelRequest) string {
	var b strings.Builder
	for _, m := range req.Messages {
		switch m.Content.Type {
		case sporecore.ContentTypeText:
			b.WriteString(m.Content.Text)
		case sporecore.ContentTypeToolCall:
			if m.Content.ToolCall != nil {
				b.WriteString(m.Content.ToolCall.Name)
				b.WriteByte(' ')
				b.Write(m.Content.ToolCall.Input)
			}
		case sporecore.ContentTypeToolResult:
			if m.Content.ToolResult != nil {
				b.WriteString(m.Content.ToolResult.Content)
			}
		case sporecore.ContentTypeImage:
			// 0 contribution
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// CallStreaming opens an NDJSON stream and emits StreamEvents.
func (c *ModelInterface) CallStreaming(ctx context.Context, req sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	if err := c.ensureModelAvailable(ctx); err != nil {
		return nil, err
	}
	if err := c.guardToolSupport(req); err != nil {
		return nil, err
	}
	body := buildRequest(c.modelID, c.keepAlive, req, true)
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("request encode failed: %v", err))
	}
	reqCtx, cancel := context.WithTimeout(ctx, c.timeout)
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, c.baseURL+"/api/chat", bytes.NewReader(payload))
	if err != nil {
		cancel()
		return nil, sporecore.NewProviderError(0, fmt.Sprintf("HTTP request build failed: %v", err))
	}
	httpReq.Header.Set("content-type", "application/json")
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		cancel()
		return nil, c.transportError(reqCtx, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		mapped := c.mapStatusError(resp)
		_ = resp.Body.Close()
		cancel()
		return nil, mapped
	}
	structured := req.Params.StructuredToolCalls && len(req.Tools) > 0
	ch := make(chan sporecore.StreamEventOrErr, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		defer cancel()
		parseNDJSONStream(ctx, resp.Body, ch, structured)
	}()
	return ch, nil
}

// ---------------------------------------------------------------------------
// NDJSON parsing — Ollama chat streaming
// ---------------------------------------------------------------------------

// parseNDJSONStream reads newline-delimited JSON. Each line carries an
// incremental message.content delta; tool_calls arrive as full argument
// objects per chunk; the terminator line carries done=true plus
// prompt_eval_count and eval_count.
func parseNDJSONStream(ctx context.Context, r io.Reader, ch chan<- sporecore.StreamEventOrErr, structured bool) {
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
		// Structured mode: the constrained JSON object arrives as message.content
		// text deltas spread across chunks. We must NOT surface that raw JSON to
		// the user — instead buffer it for the whole response and parse it at the
		// `done` chunk exactly like parseResponse, emitting reconstructable tool /
		// text events. This keeps the tool call off the native channel and is what
		// prevents the `<|python_tag|>` leak in streaming mode too.
		structuredContent strings.Builder
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var value map[string]any
		if err := json.Unmarshal([]byte(line), &value); err != nil {
			continue
		}
		if !started {
			started = true
			if !sendEvent(ctx, ch, sporecore.StreamEvent{Type: sporecore.StreamMessageStart}) {
				return
			}
		}
		if structured {
			// Buffer content deltas; defer all emission to the `done` chunk.
			if message, ok := value["message"].(map[string]any); ok {
				if text, ok := message["content"].(string); ok {
					structuredContent.WriteString(text)
				}
			}
			if done, _ := value["done"].(bool); done {
				blocks, sr := parseStructuredContent(structuredContent.String(), 0)
				for _, block := range blocks {
					switch block.Type {
					case sporecore.ContentBlockTypeToolUse:
						if block.ToolCall == nil {
							continue
						}
						partial := "{}"
						if len(block.ToolCall.Input) > 0 {
							partial = string(block.ToolCall.Input)
						}
						if !sendEvent(ctx, ch, sporecore.StreamEvent{
							Type:  sporecore.StreamToolUseStart,
							Index: 1,
							ID:    block.ToolCall.ID,
							Name:  block.ToolCall.Name,
						}) {
							return
						}
						if !sendEvent(ctx, ch, sporecore.StreamEvent{
							Type:        sporecore.StreamToolUseDelta,
							Index:       1,
							PartialJSON: partial,
						}) {
							return
						}
					case sporecore.ContentBlockTypeText:
						if !sendEvent(ctx, ch, sporecore.StreamEvent{
							Type:  sporecore.StreamContentBlockDelta,
							Index: 0,
							Delta: block.Text,
						}) {
							return
						}
					}
				}
				if pec, ok := value["prompt_eval_count"].(float64); ok {
					usage.InputTokens = uint32(pec)
				}
				if ec, ok := value["eval_count"].(float64); ok {
					usage.OutputTokens = uint32(ec)
				}
				u := usage
				sendEvent(ctx, ch, sporecore.StreamEvent{
					Type:       sporecore.StreamMessageStop,
					Usage:      &u,
					StopReason: sr,
				})
				return
			}
			continue
		}
		message, _ := value["message"].(map[string]any)
		if message != nil {
			if text, ok := message["content"].(string); ok && text != "" {
				contentIndexActive = true
				if !sendEvent(ctx, ch, sporecore.StreamEvent{
					Type:  sporecore.StreamContentBlockDelta,
					Index: contentIndex,
					Delta: text,
				}) {
					return
				}
			}
			if tcs, ok := message["tool_calls"].([]any); ok {
				for i, raw := range tcs {
					tc, _ := raw.(map[string]any)
					if tc == nil {
						continue
					}
					// Ollama identifies a distinct tool call by `function.index`,
					// which is stable across chunks. A response with multiple
					// calls streams them in SEPARATE chunks, each a one-element
					// `tool_calls` array — so the array position `i` is 0 for
					// every call and must NOT be used as the index, or every call
					// collapses onto the same block and their argument JSON
					// fragments concatenate into garbage. Fall back to `i` only
					// when `function.index` is absent.
					modelIndex := uint32(i)
					if fn, ok := tc["function"].(map[string]any); ok {
						if idx, ok := fn["index"].(float64); ok {
							modelIndex = uint32(idx)
						}
					}
					eventIndex := modelIndex + 1
					if !toolIndicesSeen[eventIndex] {
						toolIndicesSeen[eventIndex] = true
						if contentIndexActive {
							if !sendEvent(ctx, ch, sporecore.StreamEvent{
								Type:  sporecore.StreamContentBlockStop,
								Index: contentIndex,
							}) {
								return
							}
							contentIndexActive = false
							contentIndex = eventIndex
						}
						// Ollama delivers the full call (id + name + complete args)
						// on the chunk — emit a tool_use_start carrying the name and
						// id so the accumulator can reconstruct the call faithfully.
						// A missing id is synthesized stably.
						var name string
						if fn, ok := tc["function"].(map[string]any); ok {
							if n, ok := fn["name"].(string); ok {
								name = n
							}
						}
						id, _ := tc["id"].(string)
						if id == "" {
							id = fmt.Sprintf("call_%d", eventIndex)
						}
						if !sendEvent(ctx, ch, sporecore.StreamEvent{
							Type:  sporecore.StreamToolUseStart,
							Index: eventIndex,
							ID:    id,
							Name:  name,
						}) {
							return
						}
					}
					if fn, ok := tc["function"].(map[string]any); ok {
						if args, ok := fn["arguments"]; ok {
							encoded, err := json.Marshal(args)
							if err != nil {
								encoded = []byte("{}")
							}
							if !sendEvent(ctx, ch, sporecore.StreamEvent{
								Type:        sporecore.StreamToolUseDelta,
								Index:       eventIndex,
								PartialJSON: string(encoded),
							}) {
								return
							}
						}
					}
				}
			}
		}
		if done, _ := value["done"].(bool); done {
			if pec, ok := value["prompt_eval_count"].(float64); ok {
				usage.InputTokens = uint32(pec)
			}
			if ec, ok := value["eval_count"].(float64); ok {
				usage.OutputTokens = uint32(ec)
			}
			if dr, ok := value["done_reason"].(string); ok {
				stopReason = parseStopReason(dr)
			}
			u := usage
			sendEvent(ctx, ch, sporecore.StreamEvent{
				Type:       sporecore.StreamMessageStop,
				Usage:      &u,
				StopReason: stopReason,
			})
			return
		}
	}
	if err := scanner.Err(); err != nil {
		select {
		case ch <- sporecore.StreamEventOrErr{Err: sporecore.NewProviderError(0, fmt.Sprintf("stream read error: %v", err))}:
		case <-ctx.Done():
		}
		return
	}
	// Defensive terminator if the stream closed without an explicit done line.
	u := usage
	sendEvent(ctx, ch, sporecore.StreamEvent{
		Type:       sporecore.StreamMessageStop,
		Usage:      &u,
		StopReason: stopReason,
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
