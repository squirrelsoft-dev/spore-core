package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Mirrors the Rust/TS/Python Ralph fixture replay (issue #58). Each case scripts
// the per-window progress-file body the agent writes (windows: one {complete,
// remaining} entry per agent turn), the outer-loop reset cap (max_resets, B3),
// and the expected terminal RunResult. The Stop hook reads .spore/progress.json
// (B1); on incomplete it intercepts the exit and re-prompts within the window,
// and the outer loop resets the context window (fresh SessionState, reload
// .spore/ from disk — B4) until completion passes or max_resets windows are
// exhausted.
//
//   - success: iterations == the number of agent turns run (one per scripted
//     window) before the first complete file => Success.
//   - completion_unmet: iterations == the number of context-window resets
//     (== max_resets) the loop spent before halting with RalphCompletionUnmet.
type ralphFixture struct {
	Description string             `json:"description"`
	Cases       []ralphFixtureCase `json:"cases"`
}

type ralphFixtureCase struct {
	Name      string             `json:"name"`
	Windows   []ralphFixtureBody `json:"windows"`
	MaxResets uint32             `json:"max_resets"`
	Expected  struct {
		Kind       string `json:"kind"`
		Iterations uint32 `json:"iterations"`
	} `json:"expected"`
}

type ralphFixtureBody struct {
	Complete  bool     `json:"complete"`
	Remaining []string `json:"remaining"`
}

func ralphFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "harness", "ralph.json")
}

func TestRalphFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(ralphFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx ralphFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}

	for _, c := range fx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			dir := t.TempDir()
			// Seed an initial incomplete progress file so window 1 reloads state.
			writeRalphProgress(dir, ralphWindow{complete: false, remaining: []string{"task A"}})

			windows := make([]ralphWindow, len(c.Windows))
			for i, w := range c.Windows {
				windows[i] = ralphWindow{complete: w.Complete, remaining: w.Remaining}
			}
			a := newRalphAgent(dir, windows...)
			cfg := ralphCfg(a, dir)
			cfg.MaxResets = c.MaxResets
			h := NewStandardHarness(cfg)
			r := h.Run(context.Background(), NewHarnessRunOptions(ralphTask()))

			switch c.Expected.Kind {
			case "success":
				if r.Kind != RunSuccess {
					t.Fatalf("expected Success, got %+v", r)
				}
				if uint32(a.callCount()) != c.Expected.Iterations {
					t.Fatalf("expected %d agent turns, got %d", c.Expected.Iterations, a.callCount())
				}
			case "completion_unmet":
				if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
					t.Fatalf("expected RalphCompletionUnmet, got %+v", r)
				}
				if r.Reason.Iterations != c.Expected.Iterations {
					t.Fatalf("expected %d iterations, got %d", c.Expected.Iterations, r.Reason.Iterations)
				}
			default:
				t.Fatalf("unknown expected kind %q", c.Expected.Kind)
			}
		})
	}
}
