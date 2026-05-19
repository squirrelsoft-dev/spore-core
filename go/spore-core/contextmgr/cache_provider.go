// Issue #25 — CacheProvider: provider-specific cache annotation and stats.
//
// Cache control is provider-specific at the API level: Anthropic uses
// explicit `cache_control` markers, OpenAI caches automatically above a token
// threshold, Ollama has no caching. CacheProvider is the abstraction that
// keeps these concerns out of provider-agnostic ContextManager.
//
// Flow (see `docs/harness-engineering-concepts.md` §"Cache Architecture"):
//
//	ContextManager.Assemble():
//	    ... build and render segments ...
//	    if cacheProvider.SupportsCaching() {
//	        cacheProvider.Annotate(context)
//	    }
//	    return context
//
//	// After each model response:
//	stats, ok := cacheProvider.ParseCacheStats(response)
//	contextManager.RecordCacheResult(context, stats)
//	observability.EmitCacheStats(sessionID, stats)
//
// Cross-language consistency: see fixtures/cache_provider/parse_cache_stats.json.

package contextmgr

import (
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// Spec-defined types
// ============================================================================

// CacheAnnotationResult is the result of annotating a context with
// provider-specific cache markers.
type CacheAnnotationResult struct {
	MarkersInserted          uint32 `json:"markers_inserted"`
	EstimatedCacheableTokens uint32 `json:"estimated_cacheable_tokens"`
}

// CacheStats is cache token usage parsed from a single model response.
//
// ParseCacheStats returns (CacheStats{}, false) when the response had no
// cache metadata at all (caching wasn't attempted). A CacheStats with
// all-zero fields and ok=true means caching was attempted and missed — the
// distinction matters for observability.
type CacheStats struct {
	CacheReadTokens   uint32  `json:"cache_read_tokens"`
	CacheWriteTokens  uint32  `json:"cache_write_tokens"`
	CacheReadCostUSD  float64 `json:"cache_read_cost_usd"`
	CacheWriteCostUSD float64 `json:"cache_write_cost_usd"`
}

// ============================================================================
// Interface
// ============================================================================

// CacheProvider annotates a fully assembled Context with provider-specific
// cache markers and parses cache usage from model responses.
type CacheProvider interface {
	// SupportsCaching reports whether this provider supports prefix caching.
	SupportsCaching() bool

	// Annotate annotates a fully assembled context with provider-specific
	// cache markers. No-op when SupportsCaching is false.
	Annotate(ctx *Context) CacheAnnotationResult

	// ParseCacheStats parses cache usage from a model response. The second
	// return is false when the response has no cache metadata at all.
	ParseCacheStats(resp *sporecore.ModelResponse) (CacheStats, bool)

	// ProviderName is the provider identity used for observability and
	// auto-detection.
	ProviderName() string
}

// ============================================================================
// NullCacheProvider
// ============================================================================

// NullCacheProvider is the testing default. All operations are no-ops;
// SupportsCaching is false.
//
// Always use NullCacheProvider in unit tests so cache logic never interferes
// with assertions.
type NullCacheProvider struct{}

// SupportsCaching reports false.
func (NullCacheProvider) SupportsCaching() bool { return false }

// Annotate is a no-op returning the zero CacheAnnotationResult.
func (NullCacheProvider) Annotate(_ *Context) CacheAnnotationResult {
	return CacheAnnotationResult{}
}

// ParseCacheStats always returns (CacheStats{}, false).
func (NullCacheProvider) ParseCacheStats(_ *sporecore.ModelResponse) (CacheStats, bool) {
	return CacheStats{}, false
}

// ProviderName returns "null".
func (NullCacheProvider) ProviderName() string { return "null" }

var _ CacheProvider = NullCacheProvider{}

// ============================================================================
// AnthropicCacheProvider
// ============================================================================

// AnthropicCacheProvider implements Anthropic prefix caching.
//
// Inserts logical `cache_control: ephemeral` breakpoints after each stable
// block boundary (Block 1: Static, Block 2: PerSession, plus history and
// optional tool-schema anchors). Reads cache_read_tokens and
// cache_write_tokens from response usage.
type AnthropicCacheProvider struct {
	// MaxCacheAnchors is the maximum number of breakpoints per request.
	// Anthropic supports up to 4.
	MaxCacheAnchors uint32
	// CacheReadUSDPerMillion is the USD price per 1M cache-read tokens.
	// Default matches Sonnet 4.x published pricing (0.30 USD / 1M).
	CacheReadUSDPerMillion float64
	// CacheWriteUSDPerMillion is the USD price per 1M cache-write tokens
	// (5-minute TTL). Default matches Sonnet 4.x (3.75 USD / 1M).
	CacheWriteUSDPerMillion float64
}

// NewAnthropicCacheProvider returns an AnthropicCacheProvider with the
// default MaxCacheAnchors of 4 and Sonnet 4.x cache pricing. Callers can
// override per model with WithModelPricing.
func NewAnthropicCacheProvider() AnthropicCacheProvider {
	return AnthropicCacheProvider{
		MaxCacheAnchors:         4,
		CacheReadUSDPerMillion:  0.30,
		CacheWriteUSDPerMillion: 3.75,
	}
}

// WithModelPricing overrides cache pricing for a specific model id. Pricing
// data lives in the implementation so callers don't have to import a table —
// pass the model id and we look it up. Unknown ids return Sonnet pricing.
//
// Anthropic cache pricing (USD per 1M tokens, 5-minute TTL):
//   - opus-4.x:   1.50 read / 18.75 write
//   - sonnet-4.x: 0.30 read /  3.75 write
//   - haiku-4.x:  0.08 read /  1.00 write
func (p AnthropicCacheProvider) WithModelPricing(modelID string) AnthropicCacheProvider {
	read, write := anthropicCachePricing(modelID)
	p.CacheReadUSDPerMillion = read
	p.CacheWriteUSDPerMillion = write
	return p
}

func anthropicCachePricing(modelID string) (float64, float64) {
	switch {
	case strings.Contains(modelID, "opus"):
		return 1.50, 18.75
	case strings.Contains(modelID, "haiku"):
		return 0.08, 1.00
	default:
		return 0.30, 3.75
	}
}

// SupportsCaching reports true.
func (AnthropicCacheProvider) SupportsCaching() bool { return true }

// Annotate inserts up to MaxCacheAnchors breakpoints. Anchors are derived
// from existing system-prompt breakpoints (Block-1 / Block-2 boundaries)
// plus an optional history anchor when there are prior messages.
func (p AnthropicCacheProvider) Annotate(ctx *Context) CacheAnnotationResult {
	anchors := uint32(len(ctx.SystemPrompt.Breakpoints))

	// History anchor: if there are messages, the last message in the history
	// is a logical cache anchor. Recorded as a synthetic BreakpointInfo so
	// downstream code can detect annotation without mutating Messages.
	if len(ctx.Messages) > 0 && anchors < p.MaxCacheAnchors {
		ctx.SystemPrompt.Breakpoints = append(ctx.SystemPrompt.Breakpoints, BreakpointInfo{
			AfterSegment: "__history_tail__",
			TokenOffset:  ctx.TokenCount,
		})
		anchors++
	}

	markers := anchors
	if markers > p.MaxCacheAnchors {
		markers = p.MaxCacheAnchors
	}
	var estimated uint32
	if markers > 0 {
		estimated = ctx.TokenCount
	}
	return CacheAnnotationResult{
		MarkersInserted:          markers,
		EstimatedCacheableTokens: estimated,
	}
}

// ParseCacheStats reads cache_read_tokens and cache_write_tokens from the
// response and computes per-token USD cost from the provider's configured
// pricing. Returns (_, false) only when both token fields are absent.
func (p AnthropicCacheProvider) ParseCacheStats(resp *sporecore.ModelResponse) (CacheStats, bool) {
	read := resp.Usage.CacheReadTokens
	write := resp.Usage.CacheWriteTokens
	if read == nil && write == nil {
		return CacheStats{}, false
	}
	stats := CacheStats{}
	if read != nil {
		stats.CacheReadTokens = *read
	}
	if write != nil {
		stats.CacheWriteTokens = *write
	}
	stats.CacheReadCostUSD = float64(stats.CacheReadTokens) / 1_000_000.0 * p.CacheReadUSDPerMillion
	stats.CacheWriteCostUSD = float64(stats.CacheWriteTokens) / 1_000_000.0 * p.CacheWriteUSDPerMillion
	return stats, true
}

// ProviderName returns "anthropic".
func (AnthropicCacheProvider) ProviderName() string { return "anthropic" }

var _ CacheProvider = AnthropicCacheProvider{}

// ============================================================================
// OpenAICacheProvider
// ============================================================================

// OpenAICacheProvider implements OpenAI prefix caching.
//
// OpenAI caches automatically on prompts above MinCacheableTokens (1024 by
// default) — no explicit markers required. Annotate is a no-op (returning
// zero markers, but reporting cacheable token count when above threshold).
// ParseCacheStats reads cache_read_tokens; OpenAI does not return a cache
// write count, so writes remain zero.
type OpenAICacheProvider struct {
	// MinCacheableTokens is the lower bound on context size for OpenAI to
	// cache. Below this value the response will not include cache metadata.
	MinCacheableTokens uint32
	// CacheReadUSDPerMillion is the USD price per 1M cache-read tokens.
	// OpenAI's prompt caching gives a discount on cached input tokens; we
	// charge `cache_read_tokens` at the reduced rate. Default matches gpt-4o
	// pricing (1.25 USD / 1M cached).
	CacheReadUSDPerMillion float64
}

// NewOpenAICacheProvider returns an OpenAICacheProvider with the default
// MinCacheableTokens of 1024 and gpt-4o cache-read pricing. Callers can
// override per model with WithModelPricing.
func NewOpenAICacheProvider() OpenAICacheProvider {
	return OpenAICacheProvider{
		MinCacheableTokens:     1024,
		CacheReadUSDPerMillion: 1.25,
	}
}

// WithModelPricing overrides cache-read pricing for a specific OpenAI model
// id. OpenAI cache-read pricing (USD per 1M tokens):
//   - gpt-4o-mini:  0.075
//   - gpt-4o:       1.25
//   - o4-mini:      0.275
//   - o3:           2.50
//   - o1:           7.50
//
// Unknown ids default to gpt-4o pricing.
func (p OpenAICacheProvider) WithModelPricing(modelID string) OpenAICacheProvider {
	p.CacheReadUSDPerMillion = openaiCacheReadPricing(modelID)
	return p
}

func openaiCacheReadPricing(modelID string) float64 {
	switch {
	case strings.HasPrefix(modelID, "gpt-4o-mini"):
		return 0.075
	case strings.HasPrefix(modelID, "gpt-4o"):
		return 1.25
	case strings.HasPrefix(modelID, "o4-mini"):
		return 0.275
	case strings.HasPrefix(modelID, "o3"):
		return 2.50
	case strings.HasPrefix(modelID, "o1"):
		return 7.50
	default:
		return 1.25
	}
}

// SupportsCaching reports true.
func (OpenAICacheProvider) SupportsCaching() bool { return true }

// Annotate is a no-op (markers_inserted is always 0). When the context
// token count meets MinCacheableTokens, estimated_cacheable_tokens is the
// full context size; below the threshold it is 0.
func (p OpenAICacheProvider) Annotate(ctx *Context) CacheAnnotationResult {
	var cacheable uint32
	if ctx.TokenCount >= p.MinCacheableTokens {
		cacheable = ctx.TokenCount
	}
	return CacheAnnotationResult{
		MarkersInserted:          0,
		EstimatedCacheableTokens: cacheable,
	}
}

// ParseCacheStats returns (_, false) when cache_read_tokens is absent.
// When present, cache_write_tokens is forced to zero — OpenAI does not
// expose a write count — and cache_read_cost_usd is computed from the
// provider's configured per-million pricing.
func (p OpenAICacheProvider) ParseCacheStats(resp *sporecore.ModelResponse) (CacheStats, bool) {
	if resp.Usage.CacheReadTokens == nil {
		return CacheStats{}, false
	}
	read := *resp.Usage.CacheReadTokens
	return CacheStats{
		CacheReadTokens:   read,
		CacheWriteTokens:  0,
		CacheReadCostUSD:  float64(read) / 1_000_000.0 * p.CacheReadUSDPerMillion,
		CacheWriteCostUSD: 0,
	}, true
}

// ProviderName returns "openai".
func (OpenAICacheProvider) ProviderName() string { return "openai" }

var _ CacheProvider = OpenAICacheProvider{}

// ============================================================================
// OllamaCacheProvider
// ============================================================================

// OllamaCacheProvider has no caching support. Every method is a no-op.
type OllamaCacheProvider struct{}

// SupportsCaching reports false.
func (OllamaCacheProvider) SupportsCaching() bool { return false }

// Annotate is a no-op returning the zero CacheAnnotationResult.
func (OllamaCacheProvider) Annotate(_ *Context) CacheAnnotationResult {
	return CacheAnnotationResult{}
}

// ParseCacheStats always returns (CacheStats{}, false).
func (OllamaCacheProvider) ParseCacheStats(_ *sporecore.ModelResponse) (CacheStats, bool) {
	return CacheStats{}, false
}

// ProviderName returns "ollama".
func (OllamaCacheProvider) ProviderName() string { return "ollama" }

var _ CacheProvider = OllamaCacheProvider{}

// ============================================================================
// Auto-detection
// ============================================================================

// AutoDetectCacheProvider maps a ModelInterface.Provider().Name to the
// appropriate CacheProvider. The second return is false when the provider
// is unknown — the caller (typically HarnessBuilder) should emit a
// CacheProviderNotDetected warning and fall back to NullCacheProvider.
//
// Matching is case-insensitive.
func AutoDetectCacheProvider(name string) (CacheProvider, bool) {
	switch strings.ToLower(name) {
	case "anthropic":
		return NewAnthropicCacheProvider(), true
	case "openai":
		return NewOpenAICacheProvider(), true
	case "ollama":
		return OllamaCacheProvider{}, true
	default:
		return nil, false
	}
}
