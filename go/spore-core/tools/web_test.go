package tools

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
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

// ── #108: GET / auth headers / in-body auth / env resolution ─────────────────

func runSearch(t *testing.T, tool *WebSearchTool, query string) sporecore.ToolOutput {
	t.Helper()
	return tool.Execute(context.Background(),
		call("web_search", "c1", map[string]any{"query": query}), sporecore.AllowAllSandbox{}, nil)
}

func TestWebSearchGetURLEncodesQueryIntoQueryString(t *testing.T) {
	// Brave-style: GET with the query under `q`, special chars encoded.
	var gotQ string
	var gotMethod string
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotQ = r.URL.Query().Get("q")
		gotBody, _ = io.ReadAll(r.Body)
		_, _ = w.Write([]byte("get-results"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:   srv.URL + "/search",
		Method:     SearchMethodGet,
		QueryParam: "q",
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "rust & go")
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "get-results" {
		t.Fatalf("expected get-results, got %+v", r)
	}
	if gotMethod != http.MethodGet {
		t.Fatalf("expected GET, got %s", gotMethod)
	}
	if gotQ != "rust & go" {
		t.Fatalf("expected decoded query %q, got %q", "rust & go", gotQ)
	}
	if len(gotBody) != 0 {
		t.Fatalf("expected no body on GET, got %q", gotBody)
	}
}

func TestWebSearchAuthHeaderAttachedOnGet(t *testing.T) {
	t.Setenv("__SPORE_TEST_BRAVE_KEY_GET__", "brave-secret")
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Subscription-Token")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:    srv.URL + "/search",
		Method:      SearchMethodGet,
		QueryParam:  "q",
		AuthHeaders: []AuthHeader{{HeaderName: "X-Subscription-Token", EnvVar: "__SPORE_TEST_BRAVE_KEY_GET__"}},
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "x")
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	if gotHeader != "brave-secret" {
		t.Fatalf("expected header brave-secret, got %q", gotHeader)
	}
}

func TestWebSearchAuthHeaderAttachedOnPost(t *testing.T) {
	t.Setenv("__SPORE_TEST_BRAVE_KEY_POST__", "brave-secret")
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Subscription-Token")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:    srv.URL + "/search",
		Method:      SearchMethodPost,
		AuthHeaders: []AuthHeader{{HeaderName: "X-Subscription-Token", EnvVar: "__SPORE_TEST_BRAVE_KEY_POST__"}},
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "x")
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	if gotHeader != "brave-secret" {
		t.Fatalf("expected header brave-secret, got %q", gotHeader)
	}
}

func TestWebSearchMultipleAuthHeadersAllAttached(t *testing.T) {
	t.Setenv("__SPORE_TEST_MULTI_A__", "aaa")
	t.Setenv("__SPORE_TEST_MULTI_B__", "bbb")
	var gotA, gotB string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotA = r.Header.Get("X-A")
		gotB = r.Header.Get("X-B")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint: srv.URL + "/search",
		Method:   SearchMethodPost,
		AuthHeaders: []AuthHeader{
			{HeaderName: "X-A", EnvVar: "__SPORE_TEST_MULTI_A__"},
			{HeaderName: "X-B", EnvVar: "__SPORE_TEST_MULTI_B__"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "x")
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	if gotA != "aaa" || gotB != "bbb" {
		t.Fatalf("expected both headers, got X-A=%q X-B=%q", gotA, gotB)
	}
}

func TestWebSearchInBodyAuthParamTavilyShape(t *testing.T) {
	t.Setenv("__SPORE_TEST_TAVILY_KEY__", "tav-secret")
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte("tavily-results"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:       srv.URL + "/search",
		Method:         SearchMethodPost,
		QueryParam:     "query",
		BodyAuthParams: []BodyAuthParam{{FieldName: "api_key", EnvVar: "__SPORE_TEST_TAVILY_KEY__"}},
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "rust")
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "tavily-results" {
		t.Fatalf("expected tavily-results, got %+v", r)
	}
	if gotBody["api_key"] != "tav-secret" || gotBody["query"] != "rust" {
		t.Fatalf("expected tavily body shape, got %+v", gotBody)
	}
}

func TestWebSearchMissingEnvVarIsConstructionErrorNoRequest(t *testing.T) {
	const envName = "__SPORE_TEST_WEB_MISSING__"
	os.Unsetenv(envName)
	// Unroutable endpoint: if a request were attempted it would fail loudly,
	// but construction must error out first.
	_, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:    "http://127.0.0.1:1/never",
		Method:      SearchMethodPost,
		AuthHeaders: []AuthHeader{{HeaderName: "X-Key", EnvVar: envName}},
	})
	var cfgErr *WebSearchConfigError
	if !errors.As(err, &cfgErr) || cfgErr.Kind != EnvVarNotSet || cfgErr.EnvVar != envName {
		t.Fatalf("expected EnvVarNotSet for %s, got %v", envName, err)
	}
}

func TestWebSearchEmptyEnvVarIsConstructionError(t *testing.T) {
	const envName = "__SPORE_TEST_WEB_EMPTY__"
	t.Setenv(envName, "   ")
	_, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:       "http://127.0.0.1:1/never",
		Method:         SearchMethodPost,
		BodyAuthParams: []BodyAuthParam{{FieldName: "api_key", EnvVar: envName}},
	})
	var cfgErr *WebSearchConfigError
	if !errors.As(err, &cfgErr) || cfgErr.Kind != EnvVarEmpty || cfgErr.EnvVar != envName {
		t.Fatalf("expected EnvVarEmpty for %s, got %v", envName, err)
	}
}

func TestWebSearchNoAuthPostCarriesOnlyContentType(t *testing.T) {
	var gotCT string
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotCT = r.Header.Get("Content-Type")
		gotAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:   srv.URL + "/search",
		Method:     SearchMethodPost,
		QueryParam: "query",
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "x")
	if r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
	if gotCT != "application/json" {
		t.Fatalf("expected Content-Type application/json, got %q", gotCT)
	}
	if gotAuth != "" {
		t.Fatalf("expected no auth header, got %q", gotAuth)
	}
}

func TestWebSearchPostDefaultQueryShapeUnchangedViaConfig(t *testing.T) {
	var gotBody map[string]any
	var gotMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte("res"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(NewWebSearchConfig(srv.URL + "/search"))
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "rust")
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "res" {
		t.Fatalf("expected res, got %+v", r)
	}
	if gotMethod != http.MethodPost {
		t.Fatalf("expected POST, got %s", gotMethod)
	}
	if len(gotBody) != 1 || gotBody["query"] != "rust" {
		t.Fatalf("expected {\"query\":\"rust\"}, got %+v", gotBody)
	}
}

func TestWebSearchGetReturnsBodyVerbatim(t *testing.T) {
	raw := `{"web":{"results":[{"title":"t"}]}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(raw))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:   srv.URL + "/search",
		Method:     SearchMethodGet,
		QueryParam: "q",
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "t")
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != raw {
		t.Fatalf("expected verbatim body, got %+v", r)
	}
}

func TestWebSearchGetPreservesExistingQueryString(t *testing.T) {
	// SearXNG-style: the endpoint already carries a query string
	// (?format=json). The GET path must PRESERVE it and append the query under
	// the configured param — the received request must have BOTH format=json
	// AND q=<query>.
	var gotFormat string
	var gotQ string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotFormat = r.URL.Query().Get("format")
		gotQ = r.URL.Query().Get("q")
		_, _ = w.Write([]byte("searxng-results"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{
		Endpoint:   srv.URL + "/search?format=json",
		Method:     SearchMethodGet,
		QueryParam: "q",
	})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	r := runSearch(t, tool, "rust wasm")
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "searxng-results" {
		t.Fatalf("expected searxng-results, got %+v", r)
	}
	if gotFormat != "json" {
		t.Fatalf("expected preserved format=json, got %q", gotFormat)
	}
	if gotQ != "rust wasm" {
		t.Fatalf("expected appended q=%q, got %q", "rust wasm", gotQ)
	}
}

func TestSearchMethodDefaultIsPost(t *testing.T) {
	// An unset Method in config resolves to POST.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST default, got %s", r.Method)
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()
	tool, err := NewWebSearchToolFromConfig(WebSearchConfig{Endpoint: srv.URL + "/search"})
	if err != nil {
		t.Fatalf("unexpected construction error: %v", err)
	}
	if r := runSearch(t, tool, "x"); r.Kind != sporecore.ToolOutputSuccess {
		t.Fatalf("expected success, got %+v", r)
	}
}

// ── SSRF guard: validateFetchURL (#145) ──────────────────────────────────────

func TestValidateFetchURLDenyPrivateRejects(t *testing.T) {
	policy := UrlPolicyDenyPrivate()
	denied := []string{
		"http://169.254.169.254/latest/meta-data/",
		"http://localhost/",
		"file:///etc/passwd",
		"http://127.0.0.1/",
		"http://10.1.2.3/",
		"http://192.168.0.1/",
		"http://172.16.5.5/",
		"http://[::1]/",
		"ftp://example.com/x",
	}
	for _, u := range denied {
		out := validateFetchURL(u, policy)
		if out == nil {
			t.Fatalf("expected %q to be denied", u)
		}
		if out.Kind != sporecore.ToolOutputError || !out.Recoverable {
			t.Fatalf("expected recoverable error for %q, got %+v", u, out)
		}
	}
}

func TestValidateFetchURLDenyPrivateAllowsPublic(t *testing.T) {
	policy := UrlPolicyDenyPrivate()
	for _, u := range []string{
		"https://example.com/",
		"http://93.184.216.34/",
	} {
		if out := validateFetchURL(u, policy); out != nil {
			t.Fatalf("expected %q to be allowed, got %+v", u, out)
		}
	}
}

func TestValidateFetchURLPermissiveAllowsEverything(t *testing.T) {
	// Both the explicit permissive policy and the zero value allow everything
	// (proves no churn for existing wiring/tests/examples).
	policies := []UrlPolicy{UrlPolicyPermissive(), {}}
	urls := []string{
		"http://169.254.169.254/latest/meta-data/",
		"http://localhost:8080/",
		"file:///etc/passwd",
		"https://example.com/",
		"http://127.0.0.1/",
		"http://[::1]/",
	}
	for _, policy := range policies {
		for _, u := range urls {
			if out := validateFetchURL(u, policy); out != nil {
				t.Fatalf("permissive policy should allow %q, got %+v", u, out)
			}
		}
	}
}

func TestValidateFetchURLBlocksIPv4MappedMetadata(t *testing.T) {
	// ::ffff:169.254.169.254 must be unmapped and blocked.
	policy := UrlPolicyDenyPrivate()
	if out := validateFetchURL("http://[::ffff:169.254.169.254]/", policy); out == nil {
		t.Fatal("expected IPv4-mapped metadata address to be denied")
	}
}

func TestWebFetchWithDenyPrivateBlocksMetadataEndpoint(t *testing.T) {
	// httptest server with a hit flag: validation must run before any request,
	// so the server is never hit.
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		_, _ = w.Write([]byte("should-not-reach"))
	}))
	defer srv.Close()

	tool := NewWebFetchTool().WithURLPolicy(UrlPolicyDenyPrivate())
	r := tool.Execute(context.Background(),
		call("web_fetch", "c1", map[string]any{"url": "http://169.254.169.254/"}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	if hit {
		t.Fatal("validation should run before any outbound request; server was hit")
	}
}

// ── applyWebFetchRange unit tests (#135) ─────────────────────────────────────

func TestApplyWebFetchRangeStartZeroNoHeader(t *testing.T) {
	got, errOut := applyWebFetchRange("hello world", 0)
	if errOut != nil {
		t.Fatalf("expected success, got error: %+v", errOut)
	}
	if got != "hello world" {
		t.Fatalf("expected %q, got %q", "hello world", got)
	}
}

func TestApplyWebFetchRangeStartZeroEmptyBodyNoHeader(t *testing.T) {
	got, errOut := applyWebFetchRange("", 0)
	if errOut != nil {
		t.Fatalf("expected success, got error: %+v", errOut)
	}
	if got != "" {
		t.Fatalf("expected empty string, got %q", got)
	}
}

func TestApplyWebFetchRangeStartMidPrependsHeader(t *testing.T) {
	got, errOut := applyWebFetchRange("hello world", 6)
	if errOut != nil {
		t.Fatalf("expected success, got error: %+v", errOut)
	}
	want := "[starting at byte 6 of 11]\nworld"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestApplyWebFetchRangeStartAtLastByte(t *testing.T) {
	got, errOut := applyWebFetchRange("hello", 4)
	if errOut != nil {
		t.Fatalf("expected success, got error: %+v", errOut)
	}
	want := "[starting at byte 4 of 5]\no"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestApplyWebFetchRangeStartPastEndIsError(t *testing.T) {
	_, errOut := applyWebFetchRange("hello", 10)
	if errOut == nil {
		t.Fatal("expected error, got success")
	}
	if errOut.Kind != sporecore.ToolOutputError || !errOut.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", errOut)
	}
	want := "start_byte 10 exceeds response length 5"
	if errOut.Message != want {
		t.Fatalf("expected message %q, got %q", want, errOut.Message)
	}
}

func TestApplyWebFetchRangeStartAtBodyLenIsError(t *testing.T) {
	_, errOut := applyWebFetchRange("hello", 5)
	if errOut == nil {
		t.Fatal("expected error, got success")
	}
	if errOut.Kind != sporecore.ToolOutputError || !errOut.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", errOut)
	}
	want := "start_byte 5 exceeds response length 5"
	if errOut.Message != want {
		t.Fatalf("expected message %q, got %q", want, errOut.Message)
	}
}

func TestApplyWebFetchRangeEmptyBodyNonzeroStartIsError(t *testing.T) {
	_, errOut := applyWebFetchRange("", 1)
	if errOut == nil {
		t.Fatal("expected error, got success")
	}
	if errOut.Kind != sporecore.ToolOutputError || !errOut.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", errOut)
	}
	want := "start_byte 1 exceeds response length 0"
	if errOut.Message != want {
		t.Fatalf("expected message %q, got %q", want, errOut.Message)
	}
}

// ── fixture replay: web_fetch_range.json (#135) ───────────────────────────────

type webFetchRangeCase struct {
	Name          string  `json:"name"`
	Body          string  `json:"body"`
	StartByte     uint64  `json:"start_byte"`
	Expected      *string `json:"expected,omitempty"`
	ExpectedError *string `json:"expected_error,omitempty"`
}

func TestWebFetchRangeFixtureReplay(t *testing.T) {
	// Walk up from this file's location to find the repo root fixtures dir.
	_, thisFile, _, _ := runtime.Caller(0)
	// thisFile: .../go/spore-core/tools/web_test.go → go up 3 dirs
	fixturesPath := filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "fixtures", "tools", "web_fetch_range.json")
	data, err := os.ReadFile(fixturesPath)
	if err != nil {
		t.Skipf("fixture not found at %s: %v", fixturesPath, err)
	}
	var cases []webFetchRangeCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatalf("failed to parse fixture: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("expected at least one fixture case")
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			got, errOut := applyWebFetchRange(c.Body, c.StartByte)
			if errOut != nil {
				if c.ExpectedError == nil {
					t.Fatalf("unexpected error: %s", errOut.Message)
				}
				if errOut.Message != *c.ExpectedError {
					t.Fatalf("expected error %q, got %q", *c.ExpectedError, errOut.Message)
				}
				return
			}
			if c.Expected == nil {
				t.Fatalf("expected success but no 'expected' field in fixture case %q", c.Name)
			}
			if got != *c.Expected {
				t.Fatalf("expected %q, got %q", *c.Expected, got)
			}
		})
	}
}

// ── integration: web_fetch with start_byte via mock server (#135) ─────────────

func TestWebFetchStartByteZeroNoHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("hello world"))
	}))
	defer srv.Close()
	r := NewWebFetchTool().Execute(context.Background(),
		call("web_fetch", "c1", map[string]any{"url": srv.URL + "/page", "start_byte": 0}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != "hello world" {
		t.Fatalf("expected 'hello world', got %+v", r)
	}
}

func TestWebFetchStartByteMidSlicesWithHeader(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("hello world"))
	}))
	defer srv.Close()
	r := NewWebFetchTool().Execute(context.Background(),
		call("web_fetch", "c1", map[string]any{"url": srv.URL + "/page", "start_byte": 6}),
		sporecore.AllowAllSandbox{}, nil)
	want := "[starting at byte 6 of 11]\nworld"
	if r.Kind != sporecore.ToolOutputSuccess || r.Content != want {
		t.Fatalf("expected %q, got %+v", want, r)
	}
}

func TestWebFetchStartBytePastEndRecoverable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("hello"))
	}))
	defer srv.Close()
	r := NewWebFetchTool().Execute(context.Background(),
		call("web_fetch", "c1", map[string]any{"url": srv.URL + "/page", "start_byte": 99}),
		sporecore.AllowAllSandbox{}, nil)
	if r.Kind != sporecore.ToolOutputError || !r.Recoverable {
		t.Fatalf("expected recoverable error, got %+v", r)
	}
	want := "start_byte 99 exceeds response length 5"
	if r.Message != want {
		t.Fatalf("expected message %q, got %q", want, r.Message)
	}
}
