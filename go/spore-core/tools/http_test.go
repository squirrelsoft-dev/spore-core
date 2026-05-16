package tools

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestHttpGetReturnsBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/hello" {
			w.Write([]byte("world"))
		}
	}))
	defer srv.Close()

	sb := sporecore.AllowAllSandbox{}
	r := NewHttpGetTool().Execute(context.Background(),
		call("http_get", "c1", map[string]any{"url": srv.URL + "/hello"}), sb)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "world" {
		t.Fatalf("%+v", r)
	}
}

func TestHttpPostSendsJSONBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	defer srv.Close()
	sb := sporecore.AllowAllSandbox{}
	r := NewHttpPostTool().Execute(context.Background(),
		call("http_post", "c1", map[string]any{"url": srv.URL + "/echo", "body": map[string]any{"x": 1}}), sb)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "ok" {
		t.Fatalf("%+v", r)
	}
}

func TestHttpGetInvalidURL(t *testing.T) {
	sb := sporecore.AllowAllSandbox{}
	r := NewHttpGetTool().Execute(context.Background(),
		call("http_get", "c1", map[string]any{"url": "not-a-url://////"}), sb)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("%+v", r)
	}
}
