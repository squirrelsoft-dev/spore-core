package embedding

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func TestCosineBasics(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	c := []float32{0, 1, 0}
	if sim := CosineSimilarity(a, b); sim != 1.0 {
		t.Errorf("a·b = %v, want 1", sim)
	}
	if sim := CosineSimilarity(a, c); sim != 0.0 {
		t.Errorf("a·c = %v, want 0", sim)
	}
	// Degenerate inputs are 0.0, never NaN.
	if sim := CosineSimilarity(a, []float32{0, 0, 0}); sim != 0.0 {
		t.Errorf("zero-magnitude = %v, want 0", sim)
	}
	if sim := CosineSimilarity(a, []float32{1, 2}); sim != 0.0 {
		t.Errorf("length-mismatch = %v, want 0", sim)
	}
}

func TestOllamaEmbedRoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("path = %q, want /api/embed", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}},
		})
	}))
	defer server.Close()
	m := NewOllamaEmbeddingModel("nomic-embed-text").WithBaseURL(server.URL)
	vectors, err := m.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 2 || !eqVec(vectors[0], []float32{0.1, 0.2, 0.3}) || !eqVec(vectors[1], []float32{0.4, 0.5, 0.6}) {
		t.Fatalf("vectors = %v", vectors)
	}
}

func TestOllamaEmptyInputSkipsCall(t *testing.T) {
	// No server mounted: an empty batch must not contact the server.
	m := NewOllamaEmbeddingModel("nomic-embed-text").WithBaseURL("http://127.0.0.1:1")
	vectors, err := m.Embed(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(vectors) != 0 {
		t.Fatalf("expected empty, got %v", vectors)
	}
}

func TestOpenAIEmbedOrdersByIndexAndAuthenticates(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("authorization")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.9, 0.9}, "index": 1},
				{"embedding": []float32{0.1, 0.1}, "index": 0},
			},
		})
	}))
	defer server.Close()
	m := NewOpenAIEmbeddingModel("test-key", "text-embedding-3-small").WithBaseURL(server.URL)
	vectors, err := m.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if gotAuth != "Bearer test-key" {
		t.Errorf("auth = %q, want Bearer test-key", gotAuth)
	}
	// Reordered by index: input "a" (index 0) -> [0.1, 0.1].
	if len(vectors) != 2 || !eqVec(vectors[0], []float32{0.1, 0.1}) || !eqVec(vectors[1], []float32{0.9, 0.9}) {
		t.Fatalf("vectors = %v (expected reordered by index)", vectors)
	}
}

func TestOpenAIMapsErrorEnvelope(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad model"}}`))
	}))
	defer server.Close()
	m := NewOpenAIEmbeddingModel("k", "text-embedding-3-small").
		WithBaseURL(server.URL).WithMaxRetries(0)
	_, err := m.Embed(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected an error")
	}
	pe, ok := err.(*sporecore.ModelError)
	if !ok || pe.Kind != sporecore.ModelErrProviderError {
		t.Fatalf("expected ProviderError, got %v", err)
	}
	if pe.Code != 400 || pe.Message != "bad model" {
		t.Fatalf("ProviderError = {code:%d msg:%q}, want {400, bad model}", pe.Code, pe.Message)
	}
}

type embeddingRoundTripFunc func(*http.Request) (*http.Response, error)

func (f embeddingRoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestSendWithRetryTimeoutBackoffHonorsCancellation(t *testing.T) {
	client := &http.Client{Transport: embeddingRoundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, context.DeadlineExceeded
	})}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	_, err := sendWithRetry(ctx, client, func() (*http.Request, error) {
		return http.NewRequest(http.MethodPost, "http://example.test", nil)
	}, 1, 0, func(int) bool { return false }, func(resp *http.Response) error { return nil })
	if err == nil {
		t.Fatal("expected cancellation/timeout error")
	}
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("cancelled retry slept through backoff: %s", elapsed)
	}
}

func TestSendWithRetrySuccessfulResponseCancelsOnBodyClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"data":[]}`)
	}))
	defer server.Close()
	resp, err := sendWithRetry(context.Background(), server.Client(), func() (*http.Request, error) {
		return http.NewRequest(http.MethodGet, server.URL, nil)
	}, 0, time.Second, func(int) bool { return false }, func(resp *http.Response) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	body, ok := resp.Body.(*cancelOnCloseBody)
	if !ok {
		t.Fatalf("response body type=%T, want cancelOnCloseBody", resp.Body)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if body.cancel != nil {
		t.Fatal("body close did not clear per-attempt cancel")
	}
}

func TestOpenAIRetriesOn429ThenSucceeds(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{1.0, 2.0}, "index": 0},
			},
		})
	}))
	defer server.Close()
	m := NewOpenAIEmbeddingModel("k", "text-embedding-3-small").
		WithBaseURL(server.URL).WithMaxRetries(2)
	vectors, err := m.Embed(context.Background(), []string{"a"})
	if err != nil {
		t.Fatal(err)
	}
	if calls < 2 {
		t.Fatalf("expected a retry, calls=%d", calls)
	}
	if !eqVec(vectors[0], []float32{1.0, 2.0}) {
		t.Fatalf("vectors = %v", vectors)
	}
}

func TestOpenAIFromEnvUnsetRejected(t *testing.T) {
	t.Setenv("SPORE_TEST_UNSET_KEY", "")
	// Unset the var entirely.
	_, err := FromEnv("SPORE_TEST_UNSET_KEY_DOES_NOT_EXIST", "text-embedding-3-small")
	if err == nil {
		t.Fatal("expected an error for an unset env var")
	}
	pe, ok := err.(*sporecore.ModelError)
	if !ok {
		t.Fatalf("expected *ModelError, got %T", err)
	}
	if pe.Message != "env var `SPORE_TEST_UNSET_KEY_DOES_NOT_EXIST` not set" {
		t.Fatalf("message = %q", pe.Message)
	}
}

func TestOpenAIFromEnvEmptyRejected(t *testing.T) {
	t.Setenv("SPORE_TEST_EMPTY_KEY", "   ")
	_, err := FromEnv("SPORE_TEST_EMPTY_KEY", "text-embedding-3-small")
	if err == nil {
		t.Fatal("expected an error for an empty env var")
	}
	pe, ok := err.(*sporecore.ModelError)
	if !ok {
		t.Fatalf("expected *ModelError, got %T", err)
	}
	if pe.Message != "env var `SPORE_TEST_EMPTY_KEY` is empty" {
		t.Fatalf("message = %q", pe.Message)
	}
}

func TestOpenAIDefaultDimensions(t *testing.T) {
	if d := NewOpenAIEmbeddingModel("k", "text-embedding-3-large").Dimensions(); d != 3072 {
		t.Errorf("3-large dims = %d, want 3072", d)
	}
	if d := NewOpenAIEmbeddingModel("k", "text-embedding-3-small").Dimensions(); d != 1536 {
		t.Errorf("3-small dims = %d, want 1536", d)
	}
	if d := NewOpenAIEmbeddingModel("k", "text-embedding-ada-002").Dimensions(); d != 1536 {
		t.Errorf("ada-002 dims = %d, want 1536", d)
	}
	if d := NewOpenAIEmbeddingModel("k", "unknown-model").Dimensions(); d != 0 {
		t.Errorf("unknown dims = %d, want 0", d)
	}
	m := NewOpenAIEmbeddingModel("k", "text-embedding-3-small")
	if !m.supportsDimensionsParam() {
		t.Error("text-embedding-3-small should support the dimensions param")
	}
	m2 := NewOpenAIEmbeddingModel("k", "text-embedding-ada-002")
	if m2.supportsDimensionsParam() {
		t.Error("ada-002 should NOT support the dimensions param")
	}
}

func TestEmbedOneSharedHelper(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float32{{0.7, 0.8}},
		})
	}))
	defer server.Close()
	m := NewOllamaEmbeddingModel("nomic-embed-text").WithBaseURL(server.URL)
	v, err := m.EmbedOne(context.Background(), "probe")
	if err != nil {
		t.Fatal(err)
	}
	if !eqVec(v, []float32{0.7, 0.8}) {
		t.Fatalf("EmbedOne = %v", v)
	}
}

// TestEmbedOneNoVectorsErrors confirms the shared helper returns the typed
// "no vectors" ProviderError when a provider returns an empty batch.
func TestEmbedOneNoVectorsErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"embeddings":[]}`)
	}))
	defer server.Close()
	m := NewOllamaEmbeddingModel("nomic-embed-text").WithBaseURL(server.URL)
	_, err := m.EmbedOne(context.Background(), "probe")
	if err == nil {
		t.Fatal("expected an error")
	}
	pe, ok := err.(*sporecore.ModelError)
	if !ok || pe.Message != "embedding provider returned no vectors" {
		t.Fatalf("err = %v", err)
	}
}

func eqVec(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
