// HTTP tools: HttpGet, HttpPost.
//
// Uses the net/http standard library. A shared http.Client is reused across
// calls so connection pooling works.

package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

var (
	httpClientOnce sync.Once
	httpClient     *http.Client
)

func sharedClient() *http.Client {
	httpClientOnce.Do(func() {
		httpClient = &http.Client{}
	})
	return httpClient
}

func applyHeaders(req *http.Request, headers map[string]string) {
	for k, v := range headers {
		req.Header.Set(k, v)
	}
}

func hasContentType(headers map[string]string) bool {
	for k := range headers {
		if strings.EqualFold(k, "content-type") {
			return true
		}
	}
	return false
}

// ============================================================================
// HttpGet
// ============================================================================

type HttpGetTool struct{}

func NewHttpGetTool() *HttpGetTool { return &HttpGetTool{} }

const HttpGetToolName = "http_get"

func (*HttpGetTool) Name() string                { return HttpGetToolName }
func (*HttpGetTool) IsSubagentTool() bool        { return false }
func (*HttpGetTool) MayProduceLargeOutput() bool { return true }

func (*HttpGetTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        HttpGetToolName,
		Description: "Perform an HTTP GET",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"url": {"type": "string"},
				"headers": {"type": "object"}
			},
			"required": ["url"]
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true, OpenWorld: true},
	}
}

func (t *HttpGetTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params HttpGetParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, params.URL, nil)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("http get failed: %s", err), true).ToToolOutput()
	}
	applyHeaders(req, params.Headers)
	resp, err := sharedClient().Do(req)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("http get failed: %s", err), true).ToToolOutput()
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("http body read failed: %s", err), true).ToToolOutput()
	}
	return finishWithPossibleTruncation(ctx, string(body), call.ID, sandbox)
}

// ============================================================================
// HttpPost
// ============================================================================

type HttpPostTool struct{}

func NewHttpPostTool() *HttpPostTool { return &HttpPostTool{} }

const HttpPostToolName = "http_post"

func (*HttpPostTool) Name() string                { return HttpPostToolName }
func (*HttpPostTool) IsSubagentTool() bool        { return false }
func (*HttpPostTool) MayProduceLargeOutput() bool { return true }

func (*HttpPostTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        HttpPostToolName,
		Description: "Perform an HTTP POST with a JSON body",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"url": {"type": "string"},
				"body": {},
				"headers": {"type": "object"}
			},
			"required": ["url", "body"]
		}`),
		Annotations: sporecore.ToolAnnotations{OpenWorld: true},
	}
}

func (t *HttpPostTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params HttpPostParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	body := []byte(params.Body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, params.URL, bytes.NewReader(body))
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("http post failed: %s", err), true).ToToolOutput()
	}
	if !hasContentType(params.Headers) {
		req.Header.Set("Content-Type", "application/json")
	}
	applyHeaders(req, params.Headers)
	resp, err := sharedClient().Do(req)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("http post failed: %s", err), true).ToToolOutput()
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("http body read failed: %s", err), true).ToToolOutput()
	}
	return finishWithPossibleTruncation(ctx, string(respBody), call.ID, sandbox)
}

var (
	_ sporecore.Tool = (*HttpGetTool)(nil)
	_ sporecore.Tool = (*HttpPostTool)(nil)
)
