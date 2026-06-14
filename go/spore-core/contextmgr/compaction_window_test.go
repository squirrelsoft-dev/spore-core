package contextmgr

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ── Test double: a model whose Provider().ContextWindow is parameterizable so
// the resolver fallback chain (issue #141) can be exercised at any window,
// including 0 (model metadata unavailable). ─────────────────────────────────

type windowModel struct {
	contextWindow uint32
}

func (windowModel) Call(_ context.Context, _ sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	return sporecore.ModelResponse{}, nil
}
func (windowModel) CallStreaming(_ context.Context, _ sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	return nil, nil
}
func (windowModel) CountTokens(_ context.Context, _ sporecore.ModelRequest) (uint32, error) {
	return 0, nil
}
func (w windowModel) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{Name: "win", ModelID: "win", ContextWindow: w.contextWindow}
}

func u32(v uint32) *uint32 { return &v }

func mgrWith(configContextLength *uint32, modelWindow uint32) *StandardContextManager {
	cfg := DefaultCompactionConfig()
	cfg.ContextLength = configContextLength
	return NewStandardContextManager(windowModel{contextWindow: modelWindow}, NullCacheProvider{}, cfg)
}

// ── Rule: ResolveContextLength fallback chain (issue #141) ───────────────────

func TestResolveContextLength(t *testing.T) {
	cases := []struct {
		name                string
		configContextLength *uint32
		modelWindow         uint32
		want                uint32
	}{
		{"config_wins_over_model", u32(8000), 128000, 8000},
		{"model_fallback_when_config_nil", nil, 128000, 128000},
		{"default_when_both_absent", nil, 0, DefaultContextLength},
		{"explicit_zero_config_falls_through_to_model", u32(0), 128000, 128000},
		{"explicit_zero_config_and_no_model_uses_default", u32(0), 0, DefaultContextLength},
		{"no_clamp_config_larger_than_model", u32(500000), 128000, 500000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mgr := mgrWith(tc.configContextLength, tc.modelWindow)
			if got := mgr.ResolveContextLength(); got != tc.want {
				t.Fatalf("ResolveContextLength() = %d, want %d", got, tc.want)
			}
		})
	}
}

// DefaultContextLength is the conservative 8K fallback, NOT the old 200K.
func TestDefaultContextLengthIs8000(t *testing.T) {
	if DefaultContextLength != 8000 {
		t.Fatalf("DefaultContextLength = %d, want 8000", DefaultContextLength)
	}
}

// NewSessionState now defaults WindowLimit to DefaultContextLength (8K), not
// the dangerous old 200K — this encoded the new behavior (issue #141).
func TestNewSessionStateWindowLimitDefault(t *testing.T) {
	s := NewSessionState("s1", "t1", "do the thing")
	if s.WindowLimit != DefaultContextLength {
		t.Fatalf("NewSessionState WindowLimit = %d, want %d", s.WindowLimit, DefaultContextLength)
	}
}

// ── Rule: trigger math at a small window respects the configured value ───────

func TestShouldCompactSmallWindow(t *testing.T) {
	cases := []struct {
		name        string
		windowLimit uint32
		tokensUsed  uint32
		threshold   float32
		want        bool
	}{
		{"small_window_overrun_triggers", 8000, 6400, 0.8, true},
		{"small_window_just_under", 8000, 6399, 0.8, false},
		{"zero_window_never_compacts", 0, 9999, 0.8, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultCompactionConfig()
			cfg.Threshold = tc.threshold
			mgr := NewStandardContextManager(&fakeModel{tokens: 0}, NullCacheProvider{}, cfg)
			st := NewSessionState("s1", "t1", "task")
			st.WindowLimit = tc.windowLimit
			st.TokenBudgetUsed = tc.tokensUsed
			if got := mgr.ShouldCompact(&st); got != tc.want {
				t.Fatalf("ShouldCompact() = %v, want %v", got, tc.want)
			}
		})
	}
}

// ── Rule: SeedSession sets WindowLimit == ResolveContextLength() ─────────────

func TestSeedSessionSetsResolvedWindow(t *testing.T) {
	// config wins: explicit small window over a large model window.
	mgr := mgrWith(u32(8000), 128000)
	st := mgr.SeedSession("s1", "t1", "do the thing")
	if st.WindowLimit != mgr.ResolveContextLength() {
		t.Fatalf("SeedSession WindowLimit = %d, want ResolveContextLength()=%d", st.WindowLimit, mgr.ResolveContextLength())
	}
	if st.WindowLimit != 8000 {
		t.Fatalf("SeedSession WindowLimit = %d, want 8000", st.WindowLimit)
	}

	// nil config falls back to the model window.
	mgr2 := mgrWith(nil, 128000)
	st2 := mgr2.SeedSession("s1", "t1", "do the thing")
	if st2.WindowLimit != 128000 {
		t.Fatalf("SeedSession (nil config) WindowLimit = %d, want 128000", st2.WindowLimit)
	}
}

// ── Rule: DefaultCompactionConfig omits context_length from marshaled JSON ───

func TestDefaultCompactionConfigOmitsContextLength(t *testing.T) {
	cfg := DefaultCompactionConfig()
	if cfg.ContextLength != nil {
		t.Fatalf("DefaultCompactionConfig().ContextLength = %v, want nil", cfg.ContextLength)
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatal(err)
	}
	if _, present := probe["context_length"]; present {
		t.Fatalf("marshaled DefaultCompactionConfig must omit context_length; got %s", string(data))
	}
}

// ── Fixture-replay against the shared cross-language fixture ─────────────────
//
// Iterate trigger_cases and resolver_cases exactly as the Rust replay test
// does. For resolver_cases, JSON null ⇒ nil pointer; a non-null value ⇒ pointer
// to that value; the stub model's ContextWindow == model_context_window. The
// fixture is byte-identical across Rust / TypeScript / Python / Go — do not
// edit it to make this test pass; fix the implementation instead.

type cwResolverCase struct {
	Name                string  `json:"name"`
	ConfigContextLength *uint32 `json:"config_context_length"`
	ModelContextWindow  uint32  `json:"model_context_window"`
	ExpectedResolved    uint32  `json:"expected_resolved"`
}

type cwTriggerCase struct {
	Name                  string  `json:"name"`
	WindowLimit           uint32  `json:"window_limit"`
	TokenBudgetUsed       uint32  `json:"token_budget_used"`
	Threshold             float32 `json:"threshold"`
	ExpectedShouldCompact bool    `json:"expected_should_compact"`
}

type cwFixtureFile struct {
	ResolverCases []cwResolverCase `json:"resolver_cases"`
	TriggerCases  []cwTriggerCase  `json:"trigger_cases"`
}

func TestCompactionWindowFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/contextmgr → ../../../fixtures/compaction_window/cases.json
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "compaction_window", "cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var file cwFixtureFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(file.ResolverCases) == 0 || len(file.TriggerCases) == 0 {
		t.Fatal("expected at least one resolver_case and one trigger_case")
	}

	for _, c := range file.ResolverCases {
		t.Run("resolver/"+c.Name, func(t *testing.T) {
			// JSON null ⇒ nil pointer; non-null ⇒ pointer to the value.
			mgr := mgrWith(c.ConfigContextLength, c.ModelContextWindow)
			if got := mgr.ResolveContextLength(); got != c.ExpectedResolved {
				t.Fatalf("ResolveContextLength() = %d, want %d", got, c.ExpectedResolved)
			}
		})
	}

	for _, c := range file.TriggerCases {
		t.Run("trigger/"+c.Name, func(t *testing.T) {
			cfg := DefaultCompactionConfig()
			cfg.Threshold = c.Threshold
			mgr := NewStandardContextManager(&fakeModel{tokens: 0}, NullCacheProvider{}, cfg)
			st := NewSessionState("s1", "t1", "task")
			st.WindowLimit = c.WindowLimit
			st.TokenBudgetUsed = c.TokenBudgetUsed
			if got := mgr.ShouldCompact(&st); got != c.ExpectedShouldCompact {
				t.Fatalf("ShouldCompact() = %v, want %v", got, c.ExpectedShouldCompact)
			}
		})
	}
}
