package middleware

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// fixtureCase mirrors the shared JSON shape in
// fixtures/middleware/*.json so the same fixture deserialises across all
// language implementations (Rust reference at
// `rust/crates/spore-core/src/middleware.rs::FixtureCase`).
type fixtureCase struct {
	Required []string `json:"required"`
	Response string   `json:"response"`
	Expected string   `json:"expected"`
}

// TestMiddlewareChecklistFixtureReplay loads the shared checklist_basic.json
// fixture, registers the PreCompletionChecklistMiddleware with the required
// substrings, and asserts the decision the chain returns matches the
// fixture's `expected` field. "continue" and "force_another_turn" are the
// two outcomes the fixture is allowed to encode.
func TestMiddlewareChecklistFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/middleware → ../../../fixtures/middleware/...
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "middleware", "checklist_basic.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fc fixtureCase
	if err := json.Unmarshal(raw, &fc); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	chain := NewStandardMiddlewareChain()
	if err := chain.Register(NewPreCompletionChecklistMiddleware(fc.Required)); err != nil {
		t.Fatalf("register: %v", err)
	}
	state := sporecore.SessionState{}
	d, err := chain.FireBeforeCompletion(context.Background(), fc.Response, 1, &state)
	if err != nil {
		t.Fatalf("fire: %v", err)
	}
	switch fc.Expected {
	case "continue":
		if d.Kind != DecisionContinue {
			t.Fatalf("expected Continue, got %v (inject=%q)", d.Kind, d.Inject)
		}
	case "force_another_turn":
		if d.Kind != DecisionForceAnotherTurn {
			t.Fatalf("expected ForceAnotherTurn, got %v", d.Kind)
		}
	default:
		t.Fatalf("fixture expected field %q is not one of continue/force_another_turn", fc.Expected)
	}
}
