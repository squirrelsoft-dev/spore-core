package anthropic

import "testing"

// ============================================================================
// SC-6 — Anthropic context-window override (#155)
//
// Mirrors rust/crates/spore-core/src/anthropic.rs (f1c0beb). Additive — no
// fixture re-baseline.
// ============================================================================

// SC-6: an unrecognized claude-* id normally reports the family default
// (200_000); the override pins it, so the harness's compaction budget sizes
// correctly. A bare foreign (non-claude) id still reports 0.
func TestWithContextWindowOverridesReportedWindow(t *testing.T) {
	bare := New("test-key", "claude-imaginary-9")
	if bare.Provider().ContextWindow != 200_000 {
		t.Fatalf("claude-* family default = %d, want 200000", bare.Provider().ContextWindow)
	}
	pinned := New("test-key", "some-proxy-model").WithContextWindow(500_000)
	if pinned.Provider().ContextWindow != 500_000 {
		t.Fatalf("override window = %d, want 500000", pinned.Provider().ContextWindow)
	}
	// A bare foreign id still reports 0 (callers can detect "unknown").
	if got := New("k", "some-proxy-model").Provider().ContextWindow; got != 0 {
		t.Fatalf("bare foreign id window = %d, want 0", got)
	}
}
