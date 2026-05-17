package termination

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/sensor"
)

// fixtureCheck mirrors the `completion_check` shape in the shared
// fixtures/termination_policy/*.json files: a tagged union of
// {"kind":"complete"} or {"kind":"incomplete","reason":"..."}.
type fixtureCheck struct {
	Kind   string `json:"kind"`
	Reason string `json:"reason"`
}

// fixtureCase mirrors the shared JSON shape. The `expected` field is a
// serialised TerminationDecision and is asserted byte-equivalent after
// round-tripping the policy's output back through JSON.
type fixtureCase struct {
	Name            string                   `json:"name"`
	AgentClaimsDone bool                     `json:"agent_claims_done"`
	AgentResponse   *string                  `json:"agent_response"`
	BudgetUsed      sporecore.BudgetSnapshot `json:"budget_used"`
	BudgetLimits    sporecore.BudgetLimits   `json:"budget_limits"`
	SensorResults   []sensor.SensorResult    `json:"sensor_results"`
	CompletionCheck fixtureCheck             `json:"completion_check"`
	Expected        json.RawMessage          `json:"expected"`
}

type fixtureSuite struct {
	Cases []fixtureCase `json:"cases"`
}

// TestTerminationPolicyFixtureReplay loads the shared
// fixtures/termination_policy/basic.json suite and asserts that the Go
// StandardTerminationPolicy produces a TerminationDecision whose JSON
// serialisation matches the fixture's `expected` field (parsed as a
// generic map, so the comparison is structural rather than positional).
func TestTerminationPolicyFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/termination → ../../../fixtures/termination_policy/...
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "termination_policy", "basic.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var suite fixtureSuite
	if err := json.Unmarshal(raw, &suite); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}
	if len(suite.Cases) == 0 {
		t.Fatalf("fixture has no cases")
	}
	for _, c := range suite.Cases {
		t.Run(c.Name, func(t *testing.T) {
			var check CompletionCheck
			switch c.CompletionCheck.Kind {
			case "complete":
				check = NewFixedComplete()
			case "incomplete":
				check = NewFixedIncomplete(c.CompletionCheck.Reason)
			default:
				t.Fatalf("unknown completion_check kind %q", c.CompletionCheck.Kind)
			}
			policy := NewStandardTerminationPolicy(check)
			in := TerminationInput{
				SessionID:       sporecore.SessionID("fixture"),
				TaskID:          sporecore.TaskID("fixture-task"),
				TurnNumber:      1,
				AgentClaimsDone: c.AgentClaimsDone,
				AgentResponse:   c.AgentResponse,
				BudgetUsed:      c.BudgetUsed,
				BudgetLimits:    c.BudgetLimits,
				SensorResults:   c.SensorResults,
				SessionState:    snapshot(),
			}
			got, err := policy.Evaluate(context.Background(), &in)
			if err != nil {
				t.Fatalf("evaluate: %v", err)
			}
			gotJSON, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal decision: %v", err)
			}
			var gotAny, wantAny any
			if err := json.Unmarshal(gotJSON, &gotAny); err != nil {
				t.Fatalf("re-unmarshal got: %v", err)
			}
			if err := json.Unmarshal(c.Expected, &wantAny); err != nil {
				t.Fatalf("unmarshal expected: %v", err)
			}
			if !jsonEqual(gotAny, wantAny) {
				t.Fatalf("case %s: decision mismatch\n got: %s\nwant: %s", c.Name, gotJSON, string(c.Expected))
			}
		})
	}
}

// jsonEqual compares two values decoded via encoding/json. Map iteration
// order does not matter; we rely on reflect-style structural equality.
func jsonEqual(a, b any) bool {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for k, va := range av {
			vb, ok := bv[k]
			if !ok {
				return false
			}
			if !jsonEqual(va, vb) {
				return false
			}
		}
		return true
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if !jsonEqual(av[i], bv[i]) {
				return false
			}
		}
		return true
	default:
		return a == b
	}
}
