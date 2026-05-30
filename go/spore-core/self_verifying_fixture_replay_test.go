package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Mirrors the Rust/TS/Python SelfVerifying fixture replay (issue #61). Each case
// scripts a verdict sequence ("pass" => Passed, any other string => Failed
// reason) replayed across the build<->evaluate round-trips, the verifier's
// max_iterations cap, and the expected terminal RunResult. The build/evaluate
// agents always claim done, so the verdict sequence fully determines the
// outcome. `misconfigured` cases run with NO verifier configured.
type selfVerifyFixture struct {
	Description string                  `json:"description"`
	Cases       []selfVerifyFixtureCase `json:"cases"`
}

type selfVerifyFixtureCase struct {
	Name          string   `json:"name"`
	Verdicts      []string `json:"verdicts"`
	MaxIterations uint32   `json:"max_iterations"`
	Expected      struct {
		Kind       string `json:"kind"`
		Iterations uint32 `json:"iterations"`
	} `json:"expected"`
}

func selfVerifyFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "harness", "self_verifying.json")
}

func TestSelfVerifyingFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(selfVerifyFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx selfVerifyFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}

	for _, c := range fx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			build := newRecordingAgent("build", "built")
			eval := newRecordingAgent("eval", "evaluated")
			cfg := standardCfg(build)
			cfg.EvaluatorAgent = eval
			// `misconfigured` cases run with NO verifier configured.
			if c.Expected.Kind != "misconfigured" {
				cfg.Verifier = newSVVerifier(c.MaxIterations, c.Verdicts...)
			}
			h := NewStandardHarness(cfg)
			r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))

			switch c.Expected.Kind {
			case "success":
				if r.Kind != RunSuccess {
					t.Fatalf("expected Success, got %+v", r)
				}
			case "exhausted":
				if r.Kind != RunFailure || r.Reason.Kind != HaltSelfVerifyExhausted {
					t.Fatalf("expected SelfVerifyExhausted, got %+v", r)
				}
				if r.Reason.Iterations != c.Expected.Iterations {
					t.Fatalf("expected %d iterations, got %d", c.Expected.Iterations, r.Reason.Iterations)
				}
			case "misconfigured":
				if r.Kind != RunFailure || r.Reason.Kind != HaltSelfVerifyMisconfigured {
					t.Fatalf("expected SelfVerifyMisconfigured, got %+v", r)
				}
			default:
				t.Fatalf("unknown expected kind %q", c.Expected.Kind)
			}
		})
	}
}
