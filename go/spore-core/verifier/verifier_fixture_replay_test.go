package verifier

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// Cross-language consistency fixture for EvaluatorResponseVerifier (issue #44).
//
// Spec lists `evaluator_pass.jsonl` and `evaluator_fail.jsonl`; we follow the
// `fixtures/completion_check/sql_result.json` precedent of a single JSON file
// with a `cases` array.

type fixtureExpected struct {
	Kind     string `json:"kind"`
	Contains string `json:"contains"`
}

type fixtureCase struct {
	Name        string              `json:"name"`
	PassPattern string              `json:"pass_pattern"`
	FailPattern string              `json:"fail_pattern"`
	BuildResult sporecore.RunResult `json:"build_result"`
	EvalResult  sporecore.RunResult `json:"eval_result"`
	Expected    fixtureExpected     `json:"expected"`
}

type fixtureSuite struct {
	Cases []fixtureCase `json:"cases"`
}

// TestEvaluatorResponseVerifierFixtureReplay replays
// fixtures/verifier/evaluator_response.json case-for-case.
func TestEvaluatorResponseVerifierFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/verifier → ../../../fixtures/verifier/evaluator_response.json
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "verifier", "evaluator_response.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var suite fixtureSuite
	if err := json.Unmarshal(raw, &suite); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(suite.Cases) == 0 {
		t.Fatalf("fixture has no cases")
	}
	for _, c := range suite.Cases {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			v, err := NewEvaluatorResponseVerifier(c.PassPattern, c.FailPattern, 3)
			if err != nil {
				t.Fatalf("case %q: regex compile: %v", c.Name, err)
			}
			in := &VerifierInput{
				BuildResult: c.BuildResult,
				EvalResult:  c.EvalResult,
				Workspace:   "/fixture",
				Iteration:   0,
			}
			got := v.Verify(context.Background(), in)
			switch c.Expected.Kind {
			case "passed":
				if got.Kind != VerdictPassed {
					t.Fatalf("case %q: expected Passed, got %+v", c.Name, got)
				}
			case "failed":
				if got.Kind != VerdictFailed {
					t.Fatalf("case %q: expected Failed, got %+v", c.Name, got)
				}
				if !strings.Contains(got.Reason, c.Expected.Contains) {
					t.Fatalf("case %q: expected reason to contain %q, got %q",
						c.Name, c.Expected.Contains, got.Reason)
				}
			default:
				t.Fatalf("case %q: unknown expected kind %q", c.Name, c.Expected.Kind)
			}
		})
	}
}
