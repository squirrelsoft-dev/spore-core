package observability

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
)

// fixtureCase mirrors the shared JSON shape in
// fixtures/observability/session_metrics_basic.json so the same fixture
// deserialises across all language implementations (Rust reference at
// `rust/crates/spore-core/src/observability.rs::FixtureCase`).
type fixtureCase struct {
	SessionID string        `json:"session_id"`
	Turns     []fixtureTurn `json:"turns"`
	Outcome   string        `json:"outcome"`
	Expected  fixtureExpect `json:"expected"`
}

type fixtureTurn struct {
	SpanID string `json:"span_id"`
	Turn   uint32 `json:"turn"`
	Input  uint32 `json:"input"`
	Output uint32 `json:"output"`
}

type fixtureExpect struct {
	TotalTurns        uint32 `json:"total_turns"`
	TotalInputTokens  uint32 `json:"total_input_tokens"`
	TotalOutputTokens uint32 `json:"total_output_tokens"`
}

// TestObservabilityFixtureReplay loads the shared session_metrics_basic.json
// fixture, replays each turn through the in-memory provider, and asserts the
// aggregated SessionMetrics match the fixture's expected values.
func TestObservabilityFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/observability → ../../../fixtures/observability/...
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "observability", "session_metrics_basic.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fc fixtureCase
	if err := json.Unmarshal(raw, &fc); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	obs := NewInMemoryObservabilityProvider()
	for _, ft := range fc.Turns {
		obs.EmitTurn(turnSpan(fc.SessionID, ft.SpanID, ft.Turn, ft.Input, ft.Output))
	}
	var outcome guideregistry.SessionOutcome
	switch fc.Outcome {
	case "success":
		outcome = guideregistry.NewOutcomeSuccess()
	case "partial":
		outcome = guideregistry.NewOutcomePartial()
	default:
		outcome = guideregistry.NewOutcomeFailure(fc.Outcome)
	}
	obs.SetSessionOutcome(sid(fc.SessionID), outcome)

	m, err := obs.GetSessionMetrics(context.Background(), sid(fc.SessionID))
	if err != nil || m == nil {
		t.Fatalf("metrics: err=%v m=%v", err, m)
	}
	if m.TotalTurns != fc.Expected.TotalTurns {
		t.Fatalf("total_turns=%d want %d", m.TotalTurns, fc.Expected.TotalTurns)
	}
	if m.TotalInputTokens != fc.Expected.TotalInputTokens {
		t.Fatalf("total_input_tokens=%d want %d", m.TotalInputTokens, fc.Expected.TotalInputTokens)
	}
	if m.TotalOutputTokens != fc.Expected.TotalOutputTokens {
		t.Fatalf("total_output_tokens=%d want %d", m.TotalOutputTokens, fc.Expected.TotalOutputTokens)
	}
}
