package sporecore

import (
	"context"
	"testing"
)

type fakeChunkProvider struct{ chunks []harnessPromptChunk }

func (p fakeChunkProvider) Load(context.Context) ([]harnessPromptChunk, error) { return p.chunks, nil }

type badChunkProvider struct{}

func (badChunkProvider) Load(context.Context) ([]harnessPromptChunk, int) { return nil, 0 }

func TestHarnessDefaultChunkProviderPresenceOnly(t *testing.T) {
	h := NewStandardHarness(standardCfg(NewMockAgent("t")))
	if h.config.ChunkProvider == nil {
		t.Fatal("ChunkProvider default is nil")
	}
	chunks, ok := loadChunkProvider(context.Background(), h.config.ChunkProvider)
	if !ok {
		t.Fatal("default provider did not load")
	}
	if len(chunks) != 0 {
		t.Fatalf("default chunks=%d, want 0", len(chunks))
	}
	if got, ok := h.roleEvaluatorChunk(context.Background()); ok || got != "" {
		t.Fatalf("default role evaluator chunk=(%q,%v), want absent", got, ok)
	}
}

func TestRoleEvaluatorChunkLoadedFromProvider(t *testing.T) {
	provider := fakeChunkProvider{chunks: []harnessPromptChunk{
		{ID: "other", Content: "ignored"},
		{ID: "role-evaluator", Content: "CUSTOM REVIEWER"},
	}}
	cfg := standardCfg(NewMockAgent("t")).WithChunkProvider(provider)
	h := NewStandardHarness(cfg)
	got, ok := h.roleEvaluatorChunk(context.Background())
	if !ok || got != "CUSTOM REVIEWER" {
		t.Fatalf("role evaluator chunk=(%q,%v), want provider content", got, ok)
	}
}

func TestLoadChunkProviderRejectsMalformedErrorReturn(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("loadChunkProvider panicked on malformed provider: %v", r)
		}
	}()
	if chunks, ok := loadChunkProvider(context.Background(), badChunkProvider{}); ok || chunks != nil {
		t.Fatalf("malformed provider loaded chunks=%v ok=%v, want rejected", chunks, ok)
	}
}
