// Web tools (#81, net-new Tier-1 tools): web_fetch, web_search.
//
// Both use the shared net/http client (the same sharedClient as http.go). They
// are open_world (external effects) and read_only.
//
//   - WebFetchTool (web_fetch)   — GET a URL, return the body text.
//   - WebSearchTool (web_search) — POST the query to a configurable search
//     endpoint and return the structured JSON response verbatim. There is no
//     live web-search backend in spore-core; the endpoint is injected at
//     construction so tests drive it against a mock HTTP server (NEVER the live
//     network). The default endpoint is empty, which yields a recoverable error
//     until a real backend is configured.

package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// WebFetch
// ============================================================================

// WebFetchToolName is the registered tool name.
const WebFetchToolName = "web_fetch"

// WebFetchTool GETs a URL and returns the response body.
type WebFetchTool struct{}

// NewWebFetchTool constructs a WebFetchTool.
func NewWebFetchTool() *WebFetchTool { return &WebFetchTool{} }

func (*WebFetchTool) Name() string                { return WebFetchToolName }
func (*WebFetchTool) IsSubagentTool() bool        { return false }
func (*WebFetchTool) MayProduceLargeOutput() bool { return true }

func (*WebFetchTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        WebFetchToolName,
		Description: "Fetch the contents of a URL",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"url": {"type": "string"}},
			"required": ["url"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, OpenWorld: true},
	}
}

func (t *WebFetchTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params WebFetchParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, params.URL, nil)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web fetch failed: %s", err), true).ToToolOutput()
	}
	resp, err := sharedClient().Do(req)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web fetch failed: %s", err), true).ToToolOutput()
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web fetch body read failed: %s", err), true).ToToolOutput()
	}
	return finishWithPossibleTruncation(ctx, string(body), call.ID, sandbox)
}

// ============================================================================
// WebSearch
// ============================================================================

// WebSearchToolName is the registered tool name.
const WebSearchToolName = "web_search"

// WebSearchTool POSTs a query to a configurable search backend endpoint and
// returns the response verbatim. The endpoint is injected so tests run against
// a mock HTTP server. With no endpoint configured (the default), every call is
// a recoverable error.
type WebSearchTool struct {
	endpoint string
}

// NewWebSearchTool constructs a WebSearchTool with no backend configured (calls
// error until one is set via NewWebSearchToolWithEndpoint).
func NewWebSearchTool() *WebSearchTool { return &WebSearchTool{} }

// NewWebSearchToolWithEndpoint constructs a WebSearchTool that POSTs the query
// to endpoint as JSON {"query": ...}; the response body is returned verbatim.
func NewWebSearchToolWithEndpoint(endpoint string) *WebSearchTool {
	return &WebSearchTool{endpoint: endpoint}
}

func (*WebSearchTool) Name() string                { return WebSearchToolName }
func (*WebSearchTool) IsSubagentTool() bool        { return false }
func (*WebSearchTool) MayProduceLargeOutput() bool { return true }

func (*WebSearchTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        WebSearchToolName,
		Description: "Search the web and return structured results",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"query": {"type": "string"}},
			"required": ["query"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, OpenWorld: true},
	}
}

func (t *WebSearchTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params WebSearchParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	if t.endpoint == "" {
		return ExecutionFailed("web_search backend not configured", true).ToToolOutput()
	}
	payload, err := json.Marshal(map[string]string{"query": params.Query})
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web search failed: %s", err), true).ToToolOutput()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, bytes.NewReader(payload))
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web search failed: %s", err), true).ToToolOutput()
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := sharedClient().Do(req)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web search failed: %s", err), true).ToToolOutput()
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web search body read failed: %s", err), true).ToToolOutput()
	}
	return finishWithPossibleTruncation(ctx, string(body), call.ID, sandbox)
}

var (
	_ sporecore.Tool = (*WebFetchTool)(nil)
	_ sporecore.Tool = (*WebSearchTool)(nil)
)
