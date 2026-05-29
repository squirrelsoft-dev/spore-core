package tools

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestWebFetchReturnsBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("page-body"))
	}))
	defer srv.Close()
	r := NewWebFetchTool().Execute(context.Background(),
		call("web_fetch", "c1", map[string]any{"url": srv.URL + "/page"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "page-body" {
		t.Fatalf("expected page-body, got %+v", r)
	}
}

func TestWebFetchBadURLRecoverable(t *testing.T) {
	r := NewWebFetchTool().Execute(context.Background(),
		call("web_fetch", "c1", map[string]any{"url": "not-a-url://////"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}

func TestWebSearchReturnsStructuredResultsFromMock(t *testing.T) {
	results := `{"results":[{"title":"t","url":"u"}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		_, _ = w.Write([]byte(results))
	}))
	defer srv.Close()
	r := NewWebSearchToolWithEndpoint(srv.URL+"/search").Execute(context.Background(),
		call("web_search", "c1", map[string]any{"query": "rust"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != results {
		t.Fatalf("expected mock results, got %+v", r)
	}
}

func TestWebSearchWithoutBackendRecoverable(t *testing.T) {
	r := NewWebSearchTool().Execute(context.Background(),
		call("web_search", "c1", map[string]any{"query": "x"}), sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
}
