package ollama

import (
	"encoding/json"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/contextmgr"
)

// ============================================================================
// SC-4 / SC-6 — Ollama context-window setter (#155)
//
// Mirrors rust/crates/spore-core/src/ollama.rs (f1c0beb) Phase 2 tests. Same
// behavior, same precedence. Additive — no fixture re-baseline.
// ============================================================================

// SC-4: ONE call fans out to the model-loading knob (num_ctx) AND the window
// REPORTED to the harness's compaction budget (Provider()).
func TestWithContextWindowSetsBothNumCtxAndReportedWindow(t *testing.T) {
	c := New("gemma3:4b").WithContextWindow(256_000)
	if c.numCtx == nil || *c.numCtx != 256_000 {
		t.Fatalf("num_ctx: what Ollama loads = %v, want 256000", c.numCtx)
	}
	if c.Provider().ContextWindow != 256_000 {
		t.Fatalf("reported window: what compaction sizes to = %d, want 256000", c.Provider().ContextWindow)
	}
}

// SC-6: gemma* statically reports 8_192; the explicit override wins. No
// override → the static table still governs.
func TestContextWindowOverrideBeatsStaticTable(t *testing.T) {
	if ContextWindow("gemma3:4b") != 8_192 {
		t.Fatalf("static table for gemma3:4b = %d, want 8192", ContextWindow("gemma3:4b"))
	}
	c := New("gemma3:4b").WithContextWindow(128_000)
	if c.Provider().ContextWindow != 128_000 {
		t.Fatalf("override window = %d, want 128000", c.Provider().ContextWindow)
	}
	bare := New("gemma3:4b")
	if bare.Provider().ContextWindow != 8_192 {
		t.Fatalf("bare static window = %d, want 8192", bare.Provider().ContextWindow)
	}
}

// SC-4: with_context_window sets num_ctx as the FIRST options key on the wire
// (matching the Rust wire byte order) and serializes the requested window.
func TestWithContextWindowSerializesNumCtxFirst(t *testing.T) {
	c := New("llama3.2").WithContextWindow(256_000)
	body := buildRequest("llama3.2", "", c.numCtx, req(userMsg("hi")), false)
	out, _ := json.Marshal(body)
	s := string(out)
	if !strings.Contains(s, `"options":{"num_ctx":256000`) {
		t.Fatalf("num_ctx must be the first options key: %s", s)
	}
}

// SC-4 acceptance: the reported window flows through the context manager's #141
// resolve chain into the seeded WindowLimit — the compaction budget — with no
// separate CompactionConfig setting.
func TestWithContextWindowFansOutToCompactionBudget(t *testing.T) {
	model := New("gemma3:4b").WithContextWindow(256_000)
	mgr := contextmgr.NewStandardContextManager(
		model,
		contextmgr.NullCacheProvider{},
		contextmgr.DefaultCompactionConfig(), // ContextLength nil → falls to provider window
	)
	if got := mgr.ResolveContextLength(); got != 256_000 {
		t.Fatalf("ResolveContextLength = %d, want 256000", got)
	}
	state := mgr.SeedSession(
		sporecore.SessionID("s"),
		sporecore.TaskID("t"),
		"do the thing",
	)
	if state.WindowLimit != 256_000 {
		t.Fatalf("seeded window_limit = %d, want 256000 (200K conversation won't compact prematurely)", state.WindowLimit)
	}
}
