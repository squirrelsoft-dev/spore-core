package contextmgr

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func newCtx(tokens uint32, breakpoints []BreakpointInfo, msgs int) *Context {
	messages := make([]sporecore.Message, 0, msgs)
	for i := 0; i < msgs; i++ {
		messages = append(messages, sporecore.Message{
			Role:    sporecore.RoleUser,
			Content: sporecore.NewTextContent("m"),
		})
	}
	return &Context{
		SystemPrompt: RenderedSystemPrompt{
			Content:     "system",
			Breakpoints: breakpoints,
		},
		Messages:    messages,
		ToolSchemas: nil,
		TokenCount:  tokens,
		WindowLimit: 200_000,
	}
}

func responseWith(read, write *uint32) *sporecore.ModelResponse {
	return &sporecore.ModelResponse{
		Content: []sporecore.ContentBlock{sporecore.NewTextBlock("hi")},
		Usage: sporecore.TokenUsage{
			InputTokens:      0,
			OutputTokens:     0,
			CacheReadTokens:  read,
			CacheWriteTokens: write,
		},
		StopReason: sporecore.StopEndTurn,
	}
}

func u32p(v uint32) *uint32 { return &v }

// ---------------------------------------------------------------------------
// NullCacheProvider — Rule: every operation is a no-op
// ---------------------------------------------------------------------------

func TestNullProviderDoesNothing(t *testing.T) {
	p := NullCacheProvider{}
	if p.SupportsCaching() {
		t.Error("SupportsCaching should be false")
	}
	if p.ProviderName() != "null" {
		t.Errorf("ProviderName = %q, want %q", p.ProviderName(), "null")
	}
	c := newCtx(100, nil, 0)
	if got := p.Annotate(c); got != (CacheAnnotationResult{}) {
		t.Errorf("Annotate = %+v, want zero", got)
	}
	if _, ok := p.ParseCacheStats(responseWith(u32p(5), nil)); ok {
		t.Error("ParseCacheStats should return ok=false")
	}
}

// ---------------------------------------------------------------------------
// AnthropicCacheProvider
// ---------------------------------------------------------------------------

// Rule: identity fields and defaults
func TestAnthropicIdentity(t *testing.T) {
	p := NewAnthropicCacheProvider()
	if !p.SupportsCaching() {
		t.Error("SupportsCaching should be true")
	}
	if p.ProviderName() != "anthropic" {
		t.Errorf("ProviderName = %q", p.ProviderName())
	}
	if p.MaxCacheAnchors != 4 {
		t.Errorf("MaxCacheAnchors = %d, want 4", p.MaxCacheAnchors)
	}
}

// Rule: Annotate caps inserted markers at MaxCacheAnchors
func TestAnthropicAnnotateCapsAtMaxAnchors(t *testing.T) {
	p := AnthropicCacheProvider{MaxCacheAnchors: 2}
	bps := []BreakpointInfo{
		{AfterSegment: "block_1_static", TokenOffset: 10},
		{AfterSegment: "block_2_per_session", TokenOffset: 20},
	}
	c := newCtx(50, bps, 3)
	r := p.Annotate(c)
	if r.MarkersInserted != 2 {
		t.Errorf("MarkersInserted = %d, want 2", r.MarkersInserted)
	}
	if r.EstimatedCacheableTokens == 0 {
		t.Error("EstimatedCacheableTokens should be > 0")
	}
	if len(c.SystemPrompt.Breakpoints) != 2 {
		t.Errorf("Breakpoints len = %d, want 2", len(c.SystemPrompt.Breakpoints))
	}
}

// Rule: Annotate adds a history anchor when room and history are present
func TestAnthropicAnnotateAddsHistoryAnchor(t *testing.T) {
	p := NewAnthropicCacheProvider()
	bps := []BreakpointInfo{{AfterSegment: "block_1_static", TokenOffset: 10}}
	c := newCtx(75, bps, 4)
	r := p.Annotate(c)
	if r.MarkersInserted != 2 {
		t.Errorf("MarkersInserted = %d, want 2", r.MarkersInserted)
	}
	if len(c.SystemPrompt.Breakpoints) != 2 {
		t.Fatalf("Breakpoints len = %d, want 2", len(c.SystemPrompt.Breakpoints))
	}
	if c.SystemPrompt.Breakpoints[1].AfterSegment != "__history_tail__" {
		t.Errorf("synthetic anchor segment = %q", c.SystemPrompt.Breakpoints[1].AfterSegment)
	}
}

// Rule: zero markers when there are no existing breakpoints and no history
func TestAnthropicAnnotateZeroWhenEmpty(t *testing.T) {
	p := NewAnthropicCacheProvider()
	c := newCtx(50, nil, 0)
	r := p.Annotate(c)
	if r.MarkersInserted != 0 {
		t.Errorf("MarkersInserted = %d, want 0", r.MarkersInserted)
	}
	if r.EstimatedCacheableTokens != 0 {
		t.Errorf("EstimatedCacheableTokens = %d, want 0", r.EstimatedCacheableTokens)
	}
}

// Rule: ParseCacheStats returns ok=false without metadata
func TestAnthropicParseReturnsFalseWithoutMetadata(t *testing.T) {
	p := NewAnthropicCacheProvider()
	if _, ok := p.ParseCacheStats(responseWith(nil, nil)); ok {
		t.Error("expected ok=false without metadata")
	}
}

// Rule: ParseCacheStats reads both read and write token counts
func TestAnthropicParseReadsTokens(t *testing.T) {
	p := NewAnthropicCacheProvider()
	s, ok := p.ParseCacheStats(responseWith(u32p(900), u32p(120)))
	if !ok {
		t.Fatal("expected ok=true")
	}
	if s.CacheReadTokens != 900 {
		t.Errorf("read = %d, want 900", s.CacheReadTokens)
	}
	if s.CacheWriteTokens != 120 {
		t.Errorf("write = %d, want 120", s.CacheWriteTokens)
	}
}

// Rule: ParseCacheStats treats one-sided metadata as ok=true
func TestAnthropicParseOneSidedIsSome(t *testing.T) {
	p := NewAnthropicCacheProvider()
	s, ok := p.ParseCacheStats(responseWith(u32p(0), nil))
	if !ok {
		t.Fatal("expected ok=true for one-sided metadata")
	}
	if s.CacheReadTokens != 0 || s.CacheWriteTokens != 0 {
		t.Errorf("got %+v, want zeros", s)
	}
}

// ---------------------------------------------------------------------------
// OpenAICacheProvider
// ---------------------------------------------------------------------------

// Rule: identity
func TestOpenAIIdentity(t *testing.T) {
	p := NewOpenAICacheProvider()
	if !p.SupportsCaching() {
		t.Error("SupportsCaching should be true")
	}
	if p.ProviderName() != "openai" {
		t.Errorf("ProviderName = %q", p.ProviderName())
	}
	if p.MinCacheableTokens != 1024 {
		t.Errorf("MinCacheableTokens = %d, want 1024", p.MinCacheableTokens)
	}
}

// Rule: Annotate is a no-op; cacheable tokens reported only above threshold
func TestOpenAIAnnotateThreshold(t *testing.T) {
	p := NewOpenAICacheProvider()

	below := newCtx(1023, nil, 0)
	r := p.Annotate(below)
	if r.MarkersInserted != 0 {
		t.Errorf("below: MarkersInserted = %d, want 0", r.MarkersInserted)
	}
	if r.EstimatedCacheableTokens != 0 {
		t.Errorf("below: EstimatedCacheableTokens = %d, want 0", r.EstimatedCacheableTokens)
	}

	above := newCtx(2048, nil, 0)
	r = p.Annotate(above)
	if r.MarkersInserted != 0 {
		t.Errorf("above: MarkersInserted = %d, want 0", r.MarkersInserted)
	}
	if r.EstimatedCacheableTokens != 2048 {
		t.Errorf("above: EstimatedCacheableTokens = %d, want 2048", r.EstimatedCacheableTokens)
	}
}

// Rule: ParseCacheStats reads cache_read_tokens only; write forced to zero
func TestOpenAIParseReadsOnlyReads(t *testing.T) {
	p := NewOpenAICacheProvider()
	s, ok := p.ParseCacheStats(responseWith(u32p(512), u32p(99)))
	if !ok {
		t.Fatal("expected ok=true")
	}
	if s.CacheReadTokens != 512 {
		t.Errorf("read = %d, want 512", s.CacheReadTokens)
	}
	if s.CacheWriteTokens != 0 {
		t.Errorf("write = %d, want 0 (forced)", s.CacheWriteTokens)
	}
	if _, ok := p.ParseCacheStats(responseWith(nil, nil)); ok {
		t.Error("expected ok=false without read metadata")
	}
}

// Rule: ParseCacheStats ok=false when only write is present (no read)
func TestOpenAIParseFalseWhenOnlyWrite(t *testing.T) {
	p := NewOpenAICacheProvider()
	if _, ok := p.ParseCacheStats(responseWith(nil, u32p(500))); ok {
		t.Error("expected ok=false when only write is present")
	}
}

// ---------------------------------------------------------------------------
// OllamaCacheProvider
// ---------------------------------------------------------------------------

// Rule: SupportsCaching is false; all ops are no-ops
func TestOllamaNoOp(t *testing.T) {
	p := OllamaCacheProvider{}
	if p.SupportsCaching() {
		t.Error("SupportsCaching should be false")
	}
	if p.ProviderName() != "ollama" {
		t.Errorf("ProviderName = %q", p.ProviderName())
	}
	c := newCtx(99, nil, 0)
	if got := p.Annotate(c); got != (CacheAnnotationResult{}) {
		t.Errorf("Annotate = %+v, want zero", got)
	}
	if _, ok := p.ParseCacheStats(responseWith(u32p(5), u32p(5))); ok {
		t.Error("ParseCacheStats should return ok=false")
	}
}

// ---------------------------------------------------------------------------
// AutoDetectCacheProvider
// ---------------------------------------------------------------------------

// Rule: maps provider names case-insensitively; unknowns return ok=false
func TestAutoDetectMapsKnownProviders(t *testing.T) {
	cases := []struct {
		input string
		name  string
	}{
		{"anthropic", "anthropic"},
		{"OpenAI", "openai"},
		{"OLLAMA", "ollama"},
	}
	for _, c := range cases {
		p, ok := AutoDetectCacheProvider(c.input)
		if !ok {
			t.Errorf("AutoDetectCacheProvider(%q) ok=false", c.input)
			continue
		}
		if p.ProviderName() != c.name {
			t.Errorf("AutoDetectCacheProvider(%q).ProviderName = %q, want %q",
				c.input, p.ProviderName(), c.name)
		}
	}
	if _, ok := AutoDetectCacheProvider("mystery"); ok {
		t.Error("AutoDetectCacheProvider(\"mystery\") should be ok=false")
	}
}

// ---------------------------------------------------------------------------
// AnthropicCacheProvider.WithModelPricing — Rule: per-model cache cost (#39)
// ---------------------------------------------------------------------------

func TestAnthropicParseComputesCostDefaultSonnet(t *testing.T) {
	p := NewAnthropicCacheProvider()
	stats, ok := p.ParseCacheStats(responseWith(u32p(1_000_000), u32p(1_000_000)))
	if !ok {
		t.Fatal("expected ok")
	}
	// Sonnet pricing: 0.30 read / 3.75 write per 1M.
	if diff := stats.CacheReadCostUSD - 0.30; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("read cost = %v", stats.CacheReadCostUSD)
	}
	if diff := stats.CacheWriteCostUSD - 3.75; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("write cost = %v", stats.CacheWriteCostUSD)
	}
}

func TestAnthropicParseWithOpusPricing(t *testing.T) {
	p := NewAnthropicCacheProvider().WithModelPricing("claude-opus-4-7")
	stats, ok := p.ParseCacheStats(responseWith(u32p(1_000_000), u32p(1_000_000)))
	if !ok {
		t.Fatal("expected ok")
	}
	// Opus: 1.50 read / 18.75 write.
	if diff := stats.CacheReadCostUSD - 1.50; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("read cost = %v", stats.CacheReadCostUSD)
	}
	if diff := stats.CacheWriteCostUSD - 18.75; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("write cost = %v", stats.CacheWriteCostUSD)
	}
}

func TestAnthropicParseWithHaikuPricing(t *testing.T) {
	p := NewAnthropicCacheProvider().WithModelPricing("claude-haiku-4-5")
	stats, ok := p.ParseCacheStats(responseWith(u32p(1_000_000), u32p(1_000_000)))
	if !ok {
		t.Fatal("expected ok")
	}
	// Haiku: 0.08 read / 1.00 write.
	if diff := stats.CacheReadCostUSD - 0.08; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("read cost = %v", stats.CacheReadCostUSD)
	}
	if diff := stats.CacheWriteCostUSD - 1.00; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("write cost = %v", stats.CacheWriteCostUSD)
	}
}

func TestAnthropicParseWithUnknownModelFallsBackToSonnet(t *testing.T) {
	p := NewAnthropicCacheProvider().WithModelPricing("mystery-3")
	stats, _ := p.ParseCacheStats(responseWith(u32p(1_000_000), u32p(1_000_000)))
	if diff := stats.CacheReadCostUSD - 0.30; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("read cost = %v", stats.CacheReadCostUSD)
	}
}
