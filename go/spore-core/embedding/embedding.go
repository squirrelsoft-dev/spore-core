// Package embedding provides the EmbeddingModel boundary between the RAG
// layer and an embedding provider, plus two in-core providers (Ollama, OpenAI)
// that reuse the standard library's net/http client (no new third-party dep).
//
// Mirrors the Rust reference (embedding.rs + provider_http.rs). The trait is
// BATCH + single:
//   - Embed(ctx, []string) — one vector per input IN INPUT ORDER; an empty
//     inputs slice returns an empty result WITHOUT contacting the provider.
//   - EmbedOne(ctx, string) — convenience single; Go interfaces cannot supply
//     a default method body, so both providers implement it through the shared
//     embedOne helper (calls Embed, pops the single vector, errors if none).
package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// EmbeddingInfo identifies the model behind an implementation.
type EmbeddingInfo struct {
	Provider   string `json:"provider"`
	ModelID    string `json:"model_id"`
	Dimensions int    `json:"dimensions"`
}

// EmbeddingModel turns text into embedding vectors. Object-safe by construction
// (Go interfaces are dynamically dispatched). Implementations must be safe for
// concurrent use.
type EmbeddingModel interface {
	// Embed embeds a batch of inputs, returning one vector per input IN INPUT
	// ORDER. An empty inputs slice returns an empty slice without contacting the
	// provider.
	Embed(ctx context.Context, inputs []string) ([][]float32, error)

	// EmbedOne embeds a single input. Go interfaces cannot supply a default
	// method body, so every implementation routes this through the shared
	// embedOne helper (which calls Embed and pops the single vector).
	EmbedOne(ctx context.Context, input string) ([]float32, error)

	// Dimensions returns the vector length this model produces (0 = unknown).
	Dimensions() int

	// Info reports the provider identity for tracing/routing.
	Info() EmbeddingInfo
}

// CosineSimilarity returns the cosine similarity of two equal-length vectors in
// [-1, 1] (higher = closer). Returns 0.0 when lengths differ or either vector
// has zero magnitude — a degenerate input is "no similarity" rather than NaN.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (sqrt32(na) * sqrt32(nb))
}

func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// embedOne is the shared helper that gives every EmbeddingModel its EmbedOne
// body (Go interfaces cannot default it). It calls Embed with a one-element
// batch, pops the single vector, and returns a ProviderError if the provider
// returned no vectors.
func embedOne(ctx context.Context, m EmbeddingModel, input string) ([]float32, error) {
	vectors, err := m.Embed(ctx, []string{input})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, sporecore.NewProviderError(0, "embedding provider returned no vectors")
	}
	return vectors[0], nil
}

// ============================================================================
// Shared HTTP retry plumbing (mirrors provider_http.rs)
// ============================================================================

// isRetryableStatus is the base set of retryable HTTP status codes shared by the
// hosted providers. (Anthropic additionally retries 529; it composes this with
// `|| code == 529`.)
func isRetryableStatus(code int) bool {
	switch code {
	case 408, 425, 429, 500, 502, 503, 504:
		return true
	}
	return false
}

// backoffDelay is the exponential backoff with a 30s ceiling: 0.5s, 1s, 2s, …
func backoffDelay(attempt uint32) time.Duration {
	shift := attempt
	if shift > 6 {
		shift = 6
	}
	ms := uint64(500) << shift
	if ms > 30_000 {
		ms = 30_000
	}
	return time.Duration(ms) * time.Millisecond
}

// retryAfter parses a `retry-after` header (whole seconds) into a Duration.
func retryAfter(header http.Header) *time.Duration {
	v := header.Get("Retry-After")
	if v == "" {
		return nil
	}
	secs, err := strconv.ParseUint(strings.TrimSpace(v), 10, 64)
	if err != nil {
		return nil
	}
	d := time.Duration(secs) * time.Second
	return &d
}

// errorParts extracts the status code, retry-after, and human message from a
// non-success response. The message is pulled from the typed
// {"error":{"message":…}} body, falling back to the first 500 chars of the raw
// body.
func errorParts(resp *http.Response) (code int, ra *time.Duration, message string) {
	code = resp.StatusCode
	ra = retryAfter(resp.Header)
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var typed struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &typed) == nil && typed.Error != nil && typed.Error.Message != "" {
		message = typed.Error.Message
	} else {
		if len(body) > 500 {
			message = string(body[:500])
		} else {
			message = string(body)
		}
	}
	return
}

type cancelOnCloseBody struct {
	io.ReadCloser
	cancel context.CancelFunc
}

func (b *cancelOnCloseBody) Close() error {
	err := b.ReadCloser.Close()
	if b.cancel != nil {
		b.cancel()
		b.cancel = nil
	}
	return err
}

func sleepWithContext(ctx context.Context, delay time.Duration) bool {
	if delay <= 0 {
		return true
	}
	t := time.NewTimer(delay)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// sendWithRetry is the retrying HTTP sender shared by the hosted providers.
// requestFn builds a fresh request each attempt; retryable decides per-status;
// mapError converts a terminal non-retryable response into a ModelError.
func sendWithRetry(
	ctx context.Context,
	client *http.Client,
	requestFn func() (*http.Request, error),
	maxRetries uint32,
	timeout time.Duration,
	retryable func(int) bool,
	mapError func(resp *http.Response) error,
) (*http.Response, error) {
	var attempt uint32
	for {
		callCtx := ctx
		var cancel context.CancelFunc
		if timeout > 0 {
			callCtx, cancel = context.WithTimeout(ctx, timeout)
		}
		req, err := requestFn()
		if err != nil {
			if cancel != nil {
				cancel()
			}
			return nil, sporecore.NewProviderError(0, "request encode failed: "+err.Error())
		}
		req = req.WithContext(callCtx)
		resp, err := client.Do(req)
		if err != nil {
			if cancel != nil {
				cancel()
			}
			// Transport error: retry on timeout up to maxRetries, else Transport.
			if isTimeoutErr(err) && attempt < maxRetries {
				if !sleepWithContext(ctx, backoffDelay(attempt)) {
					return nil, sporecore.NewTimeout()
				}
				attempt++
				continue
			}
			if isTimeoutErr(err) {
				return nil, sporecore.NewTimeout()
			}
			return nil, sporecore.NewTransport("HTTP transport error: " + err.Error())
		}
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			if cancel != nil {
				resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, cancel: cancel}
			}
			return resp, nil
		}
		code, ra, msg := errorParts(resp)
		_ = resp.Body.Close()
		if cancel != nil {
			cancel()
		}
		if retryable(code) && attempt < maxRetries {
			delay := backoffDelay(attempt)
			if ra != nil {
				delay = *ra
			}
			if !sleepWithContext(ctx, delay) {
				return nil, ctx.Err()
			}
			attempt++
			continue
		}
		// Terminal non-retryable.
		return nil, mapErrorFromParts(code, ra, msg)
	}
}

func mapErrorFromParts(code int, _ *time.Duration, message string) error {
	return sporecore.NewProviderError(uint16(code), message)
}

func isTimeoutErr(err error) bool {
	if err == nil {
		return false
	}
	if ue, ok := err.(*url.Error); ok {
		return ue.Timeout()
	}
	return strings.Contains(err.Error(), "context deadline exceeded") || strings.Contains(err.Error(), "Client.Timeout exceeded")
}

// ============================================================================
// OllamaEmbeddingModel
// ============================================================================

// OllamaEmbeddingModel is an EmbeddingModel over a local Ollama server's
// /api/embed endpoint. No API key. Fails fast (no retry). Ollama does not report
// vector length except in the embedding itself, so Dimensions starts at 0 —
// pin it with WithDimensions or discover it with ProbeDimensions.
type OllamaEmbeddingModel struct {
	modelID    string
	baseURL    string
	dimensions int
	timeout    time.Duration
	httpClient *http.Client
}

// OllamaEmbeddingDefaultBaseURL is the default Ollama base URL.
const OllamaEmbeddingDefaultBaseURL = "http://localhost:11434"

// OllamaEmbeddingDefaultTimeout is the default request timeout.
const OllamaEmbeddingDefaultTimeout = 120 * time.Second

// NewOllamaEmbeddingModel constructs an OllamaEmbeddingModel.
func NewOllamaEmbeddingModel(modelID string) *OllamaEmbeddingModel {
	return &OllamaEmbeddingModel{
		modelID:    modelID,
		baseURL:    OllamaEmbeddingDefaultBaseURL,
		dimensions: 0,
		timeout:    OllamaEmbeddingDefaultTimeout,
		httpClient: &http.Client{},
	}
}

// WithBaseURL sets the Ollama base URL.
func (m *OllamaEmbeddingModel) WithBaseURL(baseURL string) *OllamaEmbeddingModel {
	m.baseURL = baseURL
	return m
}

// WithDimensions pins the vector length.
func (m *OllamaEmbeddingModel) WithDimensions(dimensions int) *OllamaEmbeddingModel {
	m.dimensions = dimensions
	return m
}

// WithTimeout sets the request timeout.
func (m *OllamaEmbeddingModel) WithTimeout(timeout time.Duration) *OllamaEmbeddingModel {
	m.timeout = timeout
	return m
}

// WithHTTPClient sets the HTTP client.
func (m *OllamaEmbeddingModel) WithHTTPClient(client *http.Client) *OllamaEmbeddingModel {
	m.httpClient = client
	return m
}

// ProbeDimensions discovers the model's vector length by embedding a probe
// string. Returns the length; does NOT mutate m — pin it with WithDimensions.
func (m *OllamaEmbeddingModel) ProbeDimensions(ctx context.Context) (int, error) {
	vector, err := m.EmbedOne(ctx, "dimension probe")
	if err != nil {
		return 0, err
	}
	return len(vector), nil
}

func (m *OllamaEmbeddingModel) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return [][]float32{}, nil
	}
	body, err := json.Marshal(struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}{m.modelID, inputs})
	if err != nil {
		return nil, sporecore.NewProviderError(0, "request encode failed: "+err.Error())
	}
	callCtx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(callCtx, http.MethodPost, m.baseURL+"/api/embed", strings.NewReader(string(body)))
	if err != nil {
		return nil, sporecore.NewProviderError(0, "request encode failed: "+err.Error())
	}
	req.Header.Set("content-type", "application/json")
	resp, err := m.httpClient.Do(req)
	if err != nil {
		if isTimeoutErr(err) {
			return nil, sporecore.NewTimeout()
		}
		if isConnectErr(err) {
			return nil, sporecore.NewProviderError(0, fmt.Sprintf("Ollama not running at %s", m.baseURL))
		}
		return nil, sporecore.NewTransport("HTTP transport error: " + err.Error())
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		msg := string(raw)
		if len(msg) > 500 {
			msg = msg[:500]
		}
		return nil, sporecore.NewProviderError(uint16(resp.StatusCode), msg)
	}
	var parsed struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, sporecore.NewProviderError(0, "decode embed response failed: "+err.Error())
	}
	return parsed.Embeddings, nil
}

func (m *OllamaEmbeddingModel) EmbedOne(ctx context.Context, input string) ([]float32, error) {
	return embedOne(ctx, m, input)
}

func (m *OllamaEmbeddingModel) Dimensions() int { return m.dimensions }

func (m *OllamaEmbeddingModel) Info() EmbeddingInfo {
	return EmbeddingInfo{Provider: "ollama", ModelID: m.modelID, Dimensions: m.dimensions}
}

// isConnectErr reports whether err is a connection-refused-style failure (used
// to surface the "Ollama not running" hint, matching the Rust reference).
func isConnectErr(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "connection refused") ||
		strings.Contains(s, "connect: connection refused") ||
		strings.Contains(s, "no such host") ||
		strings.Contains(s, "dial tcp") && strings.Contains(s, "connect:")
}

// ============================================================================
// OpenAIEmbeddingModel
// ============================================================================

// OpenAIEmbeddingModel is an EmbeddingModel over ${base_url}/embeddings (OpenAI
// + OpenAI-compatible servers). It reuses the shared retry loop and the
// {"error":{"message"}} envelope, exactly like the chat OpenAIModelInterface.
// Dimensions defaults from the model id for the known text-embedding-* families;
// WithDimensions overrides it AND, for the v3 families that accept the
// truncation parameter, asks the API for that length.
type OpenAIEmbeddingModel struct {
	apiKey     string
	modelID    string
	baseURL    string
	dimensions int
	timeout    time.Duration
	maxRetries uint32
	httpClient *http.Client
}

// OpenAIEmbeddingDefaultBaseURL is the default OpenAI base URL.
const OpenAIEmbeddingDefaultBaseURL = "https://api.openai.com/v1"

// OpenAIEmbeddingDefaultTimeout is the default request timeout.
const OpenAIEmbeddingDefaultTimeout = 120 * time.Second

// OpenAIEmbeddingDefaultMaxRetries is the default retry count.
const OpenAIEmbeddingDefaultMaxRetries uint32 = 3

// NewOpenAIEmbeddingModel constructs an OpenAIEmbeddingModel.
func NewOpenAIEmbeddingModel(apiKey, modelID string) *OpenAIEmbeddingModel {
	return &OpenAIEmbeddingModel{
		apiKey:     apiKey,
		modelID:    modelID,
		baseURL:    OpenAIEmbeddingDefaultBaseURL,
		dimensions: openAIDefaultDimensions(modelID),
		timeout:    OpenAIEmbeddingDefaultTimeout,
		maxRetries: OpenAIEmbeddingDefaultMaxRetries,
		httpClient: &http.Client{},
	}
}

// FromEnv constructs an OpenAIEmbeddingModel from an API key read from envVar.
// Rejects an unset or empty/whitespace value.
func FromEnv(envVar, modelID string) (*OpenAIEmbeddingModel, error) {
	key, err := keyFromEnv(envVar)
	if err != nil {
		return nil, err
	}
	return NewOpenAIEmbeddingModel(key, modelID), nil
}

// keyFromEnv reads an API key from envVar, rejecting an unset or empty value.
func keyFromEnv(envVar string) (string, error) {
	key, ok := os.LookupEnv(envVar)
	if !ok {
		return "", sporecore.NewProviderError(0, fmt.Sprintf("env var `%s` not set", envVar))
	}
	if strings.TrimSpace(key) == "" {
		return "", sporecore.NewProviderError(0, fmt.Sprintf("env var `%s` is empty", envVar))
	}
	return key, nil
}

// WithBaseURL sets the base URL.
func (m *OpenAIEmbeddingModel) WithBaseURL(baseURL string) *OpenAIEmbeddingModel {
	m.baseURL = baseURL
	return m
}

// WithDimensions pins the vector length. For the text-embedding-3-* families
// this is also sent as the request `dimensions` parameter (the API truncates to
// it); older models that reject the parameter ignore this on the wire.
func (m *OpenAIEmbeddingModel) WithDimensions(dimensions int) *OpenAIEmbeddingModel {
	m.dimensions = dimensions
	return m
}

// WithTimeout sets the request timeout.
func (m *OpenAIEmbeddingModel) WithTimeout(timeout time.Duration) *OpenAIEmbeddingModel {
	m.timeout = timeout
	return m
}

// WithMaxRetries sets the max retry count.
func (m *OpenAIEmbeddingModel) WithMaxRetries(n uint32) *OpenAIEmbeddingModel {
	m.maxRetries = n
	return m
}

// WithHTTPClient sets the HTTP client.
func (m *OpenAIEmbeddingModel) WithHTTPClient(client *http.Client) *OpenAIEmbeddingModel {
	m.httpClient = client
	return m
}

// supportsDimensionsParam reports whether the model accepts the request
// `dimensions` truncation parameter.
func (m *OpenAIEmbeddingModel) supportsDimensionsParam() bool {
	return strings.HasPrefix(m.modelID, "text-embedding-3")
}

// openAIDefaultDimensions returns the vector length for the known
// text-embedding-* families; 0 otherwise.
func openAIDefaultDimensions(modelID string) int {
	switch modelID {
	case "text-embedding-3-large":
		return 3072
	case "text-embedding-3-small", "text-embedding-ada-002":
		return 1536
	default:
		return 0
	}
}

type openAIEmbedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions *int     `json:"dimensions,omitempty"`
}

type openAIEmbedDatum struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type openAIEmbedResponse struct {
	Data []openAIEmbedDatum `json:"data"`
}

func (m *OpenAIEmbeddingModel) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if len(inputs) == 0 {
		return [][]float32{}, nil
	}
	var dimParam *int
	if m.dimensions > 0 && m.supportsDimensionsParam() {
		d := m.dimensions
		dimParam = &d
	}
	reqBody := openAIEmbedRequest{Model: m.modelID, Input: inputs, Dimensions: dimParam}
	resp, err := sendWithRetry(
		ctx, m.httpClient,
		func() (*http.Request, error) {
			body, err := json.Marshal(reqBody)
			if err != nil {
				return nil, err
			}
			req, err := http.NewRequest(http.MethodPost, m.baseURL+"/embeddings", strings.NewReader(string(body)))
			if err != nil {
				return nil, err
			}
			req.Header.Set("content-type", "application/json")
			req.Header.Set("authorization", "Bearer "+m.apiKey)
			return req, nil
		},
		m.maxRetries, m.timeout, isRetryableStatus,
		func(r *http.Response) error {
			code, _, msg := errorParts(r)
			_ = r.Body.Close()
			return sporecore.NewProviderError(uint16(code), msg)
		},
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var parsed openAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, sporecore.NewProviderError(0, "decode embeddings response failed: "+err.Error())
	}
	// The API returns data in request order, but `index` is authoritative —
	// sort by it so a non-conforming compatible server can't scramble the
	// input↔vector correspondence.
	sortByIndex(parsed.Data)
	out := make([][]float32, len(parsed.Data))
	for i, d := range parsed.Data {
		out[i] = d.Embedding
	}
	return out, nil
}

func (m *OpenAIEmbeddingModel) EmbedOne(ctx context.Context, input string) ([]float32, error) {
	return embedOne(ctx, m, input)
}

func (m *OpenAIEmbeddingModel) Dimensions() int { return m.dimensions }

func (m *OpenAIEmbeddingModel) Info() EmbeddingInfo {
	return EmbeddingInfo{Provider: "openai", ModelID: m.modelID, Dimensions: m.dimensions}
}

// sortByIndex sorts embedding data by its `index` field (stable in input order
// for ties).
func sortByIndex(data []openAIEmbedDatum) {
	// Simple stable sort by Index.
	for i := 1; i < len(data); i++ {
		for j := i; j > 0 && data[j-1].Index > data[j].Index; j-- {
			data[j], data[j-1] = data[j-1], data[j]
		}
	}
}
