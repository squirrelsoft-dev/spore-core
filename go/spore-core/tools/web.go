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
	"net/url"
	"os"
	"strings"

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
			"properties": {
				"url": {"type": "string"},
				"start_byte": {
					"type": "integer",
					"description": "Byte offset into the response body to start reading from. Default 0 (no offset, output identical to a plain fetch). Use to page through responses larger than the 64 KB truncation window.",
					"default": 0
				}
			},
			"required": ["url"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, OpenWorld: true},
	}
}

// applyWebFetchRange applies start_byte slicing to a fetched response body.
//
//   - start_byte == 0: return body unchanged (no header).
//   - 0 < start_byte < len(body): prepend "[starting at byte N of total]\n" and
//     return the slice from start_byte.
//   - start_byte >= len(body) (non-empty): recoverable error.
//   - Empty body + start_byte > 0: recoverable error.
func applyWebFetchRange(body string, startByte uint64) (string, *sporecore.ToolOutput) {
	if startByte == 0 {
		return body, nil
	}
	bodyBytes := []byte(body)
	total := uint64(len(bodyBytes))
	if startByte >= total {
		out := sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     fmt.Sprintf("start_byte %d exceeds response length %d", startByte, total),
			Recoverable: true,
		}
		return "", &out
	}
	slice := string(bodyBytes[startByte:])
	return fmt.Sprintf("[starting at byte %d of %d]\n%s", startByte, total, slice), nil
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
	sliced, rangeErr := applyWebFetchRange(string(body), params.StartByte)
	if rangeErr != nil {
		return *rangeErr
	}
	return finishWithPossibleTruncation(ctx, sliced, call.ID, sandbox)
}

// ============================================================================
// WebSearch
// ============================================================================

// WebSearchToolName is the registered tool name.
const WebSearchToolName = "web_search"

// SearchMethod is the HTTP method used to dispatch a web search. It is an enum
// (NOT a bool) so the wire shape is explicit and extensible.
type SearchMethod string

const (
	// SearchMethodGet URL-encodes the query (and any body-auth params) into the
	// query string.
	SearchMethodGet SearchMethod = "get"
	// SearchMethodPost sends a JSON body keyed by QueryParam (and any body-auth
	// params). This is the default.
	SearchMethodPost SearchMethod = "post"
)

// defaultQueryParam is the default field/param name the query is keyed under.
const defaultQueryParam = "query"

// AuthHeader maps an HTTP header name to the env var holding its value. The env
// var NAME is stored in config; the value is resolved at construction time.
type AuthHeader struct {
	HeaderName string `json:"header_name"`
	EnvVar     string `json:"env_var"`
}

// BodyAuthParam maps a request field name to the env var holding its value. The
// env var NAME is stored in config; the value is resolved at construction time.
// For POST it is injected into the JSON body alongside the query; for GET it is
// appended to the URL query string.
type BodyAuthParam struct {
	FieldName string `json:"field_name"`
	EnvVar    string `json:"env_var"`
}

// WebSearchConfig is the construction-time configuration for
// NewWebSearchToolFromConfig. Env-var NAMES (not values) are stored here; values
// are resolved when the tool is constructed. Resolved secrets never live in this
// serializable struct.
type WebSearchConfig struct {
	// Endpoint is the search backend URL.
	Endpoint string `json:"endpoint"`
	// Method is GET or POST (default POST).
	Method SearchMethod `json:"method,omitempty"`
	// AuthHeaders are (header_name, env_var) pairs — each env value is attached
	// as an HTTP header on every request, for both GET and POST.
	AuthHeaders []AuthHeader `json:"auth_headers,omitempty"`
	// QueryParam is the field/param name the query is keyed under (default
	// "query"; Brave uses "q").
	QueryParam string `json:"query_param,omitempty"`
	// BodyAuthParams are (field_name, env_var) pairs — each env value is injected
	// into the JSON body (POST) or query string (GET) as a secret param.
	BodyAuthParams []BodyAuthParam `json:"body_auth_params,omitempty"`
}

// NewWebSearchConfig returns a config with POST + "query" + no auth — equivalent
// to the frozen NewWebSearchToolWithEndpoint behavior.
func NewWebSearchConfig(endpoint string) WebSearchConfig {
	return WebSearchConfig{Endpoint: endpoint, Method: SearchMethodPost, QueryParam: defaultQueryParam}
}

// WebSearchConfigErrorKind distinguishes the construction-time auth errors.
type WebSearchConfigErrorKind string

const (
	// EnvVarNotSet means the referenced env var is absent from the environment.
	EnvVarNotSet WebSearchConfigErrorKind = "env_var_not_set"
	// EnvVarEmpty means the referenced env var is present but blank.
	EnvVarEmpty WebSearchConfigErrorKind = "env_var_empty"
)

// WebSearchConfigError is returned by NewWebSearchToolFromConfig when a
// referenced env var is unset or empty. Mirrors the FromEnv precedent: a request
// is never sent with a missing/empty secret.
type WebSearchConfigError struct {
	Kind   WebSearchConfigErrorKind
	EnvVar string
}

func (e *WebSearchConfigError) Error() string {
	switch e.Kind {
	case EnvVarEmpty:
		return fmt.Sprintf("env var %q is empty", e.EnvVar)
	default:
		return fmt.Sprintf("env var %q not set", e.EnvVar)
	}
}

// resolvedHeader / resolvedParam hold a name paired with its resolved secret
// value. These live only in the unexported tool struct, never in serializable
// config.
type resolvedHeader struct {
	name  string
	value string
}
type resolvedParam struct {
	field string
	value string
}

// resolvedBackend is the unexported, resolved backend config. Resolved secrets
// live here only. Its String method redacts secret values so they never leak.
type resolvedBackend struct {
	endpoint     string
	method       SearchMethod
	queryParam   string
	authHeaders  []resolvedHeader
	bodyAuthVals []resolvedParam
}

// String redacts resolved secret values.
func (b *resolvedBackend) String() string {
	headers := make([]string, len(b.authHeaders))
	for i, h := range b.authHeaders {
		headers[i] = fmt.Sprintf("%s=<redacted>", h.name)
	}
	params := make([]string, len(b.bodyAuthVals))
	for i, p := range b.bodyAuthVals {
		params[i] = fmt.Sprintf("%s=<redacted>", p.field)
	}
	return fmt.Sprintf("resolvedBackend{endpoint:%s method:%s queryParam:%s authHeaders:[%s] bodyAuthParams:[%s]}",
		b.endpoint, b.method, b.queryParam, strings.Join(headers, " "), strings.Join(params, " "))
}

// resolveEnv reads an env var by NAME at construction time. Unset or empty
// (after trimming) yields a *WebSearchConfigError.
func resolveEnv(envVar string) (string, error) {
	v, ok := os.LookupEnv(envVar)
	if !ok {
		return "", &WebSearchConfigError{Kind: EnvVarNotSet, EnvVar: envVar}
	}
	if strings.TrimSpace(v) == "" {
		return "", &WebSearchConfigError{Kind: EnvVarEmpty, EnvVar: envVar}
	}
	return v, nil
}

// WebSearchTool POSTs (or GETs) a query to a configurable search backend
// endpoint and returns the response verbatim. The endpoint is injected so tests
// run against a mock HTTP server. With no backend configured (the default),
// every call is a recoverable error.
type WebSearchTool struct {
	backend *resolvedBackend
}

// NewWebSearchTool constructs a WebSearchTool with no backend configured (calls
// error until one is set via NewWebSearchToolWithEndpoint).
func NewWebSearchTool() *WebSearchTool { return &WebSearchTool{} }

// NewWebSearchToolWithEndpoint constructs a WebSearchTool that POSTs the query
// to endpoint as JSON {"query": ...}; the response body is returned verbatim.
//
// FROZEN behavior — kept compatible with the original tool.
func NewWebSearchToolWithEndpoint(endpoint string) *WebSearchTool {
	return &WebSearchTool{backend: &resolvedBackend{
		endpoint:   endpoint,
		method:     SearchMethodPost,
		queryParam: defaultQueryParam,
	}}
}

// NewWebSearchToolFromConfig constructs a WebSearchTool from a WebSearchConfig,
// resolving every referenced env var at construction time. Returns a
// *WebSearchConfigError if any auth env var is unset or empty — no request is
// ever attempted in that case.
func NewWebSearchToolFromConfig(config WebSearchConfig) (*WebSearchTool, error) {
	method := config.Method
	if method == "" {
		method = SearchMethodPost
	}
	queryParam := config.QueryParam
	if queryParam == "" {
		queryParam = defaultQueryParam
	}
	authHeaders := make([]resolvedHeader, 0, len(config.AuthHeaders))
	for _, h := range config.AuthHeaders {
		v, err := resolveEnv(h.EnvVar)
		if err != nil {
			return nil, err
		}
		authHeaders = append(authHeaders, resolvedHeader{name: h.HeaderName, value: v})
	}
	bodyAuthVals := make([]resolvedParam, 0, len(config.BodyAuthParams))
	for _, p := range config.BodyAuthParams {
		v, err := resolveEnv(p.EnvVar)
		if err != nil {
			return nil, err
		}
		bodyAuthVals = append(bodyAuthVals, resolvedParam{field: p.FieldName, value: v})
	}
	return &WebSearchTool{backend: &resolvedBackend{
		endpoint:     config.Endpoint,
		method:       method,
		queryParam:   queryParam,
		authHeaders:  authHeaders,
		bodyAuthVals: bodyAuthVals,
	}}, nil
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
	b := t.backend
	if b == nil {
		return ExecutionFailed("web_search backend not configured", true).ToToolOutput()
	}
	var req *http.Request
	var err error
	switch b.method {
	case SearchMethodGet:
		// Query + body-auth params are URL-encoded into the query string.
		q := url.Values{}
		q.Set(b.queryParam, params.Query)
		for _, p := range b.bodyAuthVals {
			q.Set(p.field, p.value)
		}
		u := b.endpoint
		if strings.Contains(u, "?") {
			u += "&" + q.Encode()
		} else {
			u += "?" + q.Encode()
		}
		req, err = http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	default:
		// Query + body-auth params go into the JSON body (Tavily shape:
		// {"api_key": ..., "query": ...}).
		body := map[string]string{b.queryParam: params.Query}
		for _, p := range b.bodyAuthVals {
			body[p.field] = p.value
		}
		var payload []byte
		payload, err = json.Marshal(body)
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("web search failed: %s", err), true).ToToolOutput()
		}
		req, err = http.NewRequestWithContext(ctx, http.MethodPost, b.endpoint, bytes.NewReader(payload))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
		}
	}
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("web search failed: %s", err), true).ToToolOutput()
	}
	for _, h := range b.authHeaders {
		req.Header.Set(h.name, h.value)
	}
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
