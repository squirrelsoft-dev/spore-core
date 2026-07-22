package sporecore

import (
	"context"
	"testing"
)

type fakeChunkProvider struct{ chunks []harnessPromptChunk }

func (p fakeChunkProvider) Load(context.Context) ([]harnessPromptChunk, error) { return p.chunks, nil }

func TestHarnessDefaultChunkProviderEmpty(t *testing.T) {
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
	if got := h.roleEvaluatorChunk(context.Background()); got != RoleEvaluatorChunk {
		t.Fatalf("default role evaluator chunk = %q, want fallback constant", got)
	}
}

func TestRoleEvaluatorChunkLoadedFromProvider(t *testing.T) {
	provider := fakeChunkProvider{chunks: []harnessPromptChunk{
		{ID: "other", Content: "ignored"},
		{ID: "role-evaluator", Content: "CUSTOM REVIEWER"},
	}}
	cfg := standardCfg(NewMockAgent("t")).WithChunkProvider(provider)
	h := NewStandardHarness(cfg)
	if got := h.roleEvaluatorChunk(context.Background()); got != "CUSTOM REVIEWER" {
		t.Fatalf("role evaluator chunk = %q, want provider content", got)
	}
}
