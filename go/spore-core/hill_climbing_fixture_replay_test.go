package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Mirrors the Rust/TS/Python HillClimbing fixture replay (issue #60). Each
// scenario scripts a metric_sequence (element 0 is the iteration-0 BASELINE,
// measured with NO agent turn; status kept; element index i>=1 is the metric
// AFTER iteration i's agent turn; a null element is a crash that counts as a
// non-improvement with an EMPTY metric_value in the TSV). The payload direction
// is authoritative for the keep/revert decision via ShouldKeep. All scripted
// MetricResults carry a ZERO duration, so every duration_secs renders as
// 0.000000 — letting expected_tsv be asserted byte-identically across languages.

type hcSeqFixture struct {
	Scenarios []hcScenario `json:"scenarios"`
}

type hcScenario struct {
	Name           string     `json:"name"`
	MetricSequence []*float64 `json:"metric_sequence"`
	MaxTurns       *uint32    `json:"max_turns"`
	Payload        struct {
		Direction             OptimizationDirection `json:"direction"`
		MaxStagnation         *uint32               `json:"max_stagnation"`
		RevertOnNoImprovement bool                  `json:"revert_on_no_improvement"`
		MinImprovementDelta   *float64              `json:"min_improvement_delta"`
	} `json:"payload"`
	Expected struct {
		HaltReason     string   `json:"halt_reason"`
		KeptIterations int      `json:"kept_iterations"`
		RevertCount    int      `json:"revert_count"`
		BestMetric     *float64 `json:"best_metric"`
	} `json:"expected"`
	ExpectedTSV *string `json:"expected_tsv"`
}

func hcSeqFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "metric_evaluator", "hill_climbing_sequences.json")
}

func TestHillClimbingFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(hcSeqFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fix hcSeqFixture
	if err := json.Unmarshal(raw, &fix); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	if len(fix.Scenarios) != 7 {
		t.Fatalf("expected 7 scenarios, got %d", len(fix.Scenarios))
	}

	for _, sc := range fix.Scenarios {
		t.Run(sc.Name, func(t *testing.T) {
			// Build the scripted evaluator from the metric_sequence: a null entry
			// is a crash (counts as non-improvement, EMPTY metric).
			eval := &scriptedMetricEvaluator{}
			for _, v := range sc.MetricSequence {
				if v == nil {
					eval.results = append(eval.results, nil)
					eval.errors = append(eval.errors, &HillClimbMetricError{Status: HillClimbCrashed, Message: "scripted crash"})
				} else {
					eval.results = append(eval.results, res(*v))
					eval.errors = append(eval.errors, nil)
				}
			}

			cfg, sb := hcConfig(t, eval)
			task := hcTask(
				sc.Payload.Direction,
				sc.Payload.MaxStagnation,
				sc.Payload.RevertOnNoImprovement,
				sc.Payload.MinImprovementDelta,
				sc.MaxTurns,
			)
			h := NewStandardHarness(cfg)
			r := h.Run(context.Background(), NewHarnessRunOptions(task))

			// halt_reason.
			if r.Kind != RunFailure {
				t.Fatalf("expected failure halt, got %+v", r)
			}
			switch sc.Expected.HaltReason {
			case "stagnation":
				if r.Reason.Kind != HaltStagnationLimitReached {
					t.Fatalf("halt = %q, want stagnation_limit_reached", r.Reason.Kind)
				}
			case "budget_turns":
				if r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
					t.Fatalf("halt = %+v, want budget_exceeded/turns", r.Reason)
				}
			default:
				t.Fatalf("unhandled expected halt_reason %q", sc.Expected.HaltReason)
			}

			// best_metric (only checked when the scenario pins it).
			if sc.Expected.BestMetric != nil && r.Reason.Kind == HaltStagnationLimitReached {
				if r.Reason.BestMetric != *sc.Expected.BestMetric {
					t.Fatalf("best_metric = %v, want %v", r.Reason.BestMetric, *sc.Expected.BestMetric)
				}
			}

			// kept_iterations: count "kept" rows in the produced TSV.
			tsv := readTSV(t, sb, task.ID)
			kept := 0
			for _, line := range splitTSVRows(tsv) {
				if tsvField(line, 4) == "kept" {
					kept++
				}
			}
			if kept != sc.Expected.KeptIterations {
				t.Fatalf("kept_iterations = %d, want %d (tsv:\n%s)", kept, sc.Expected.KeptIterations, tsv)
			}

			// revert_count: count `git reset --hard HEAD` calls.
			reverts := 0
			for _, c := range sb.commands {
				if len(c) == 4 && c[0] == "git" && c[1] == "reset" && c[2] == "--hard" && c[3] == "HEAD" {
					reverts++
				}
			}
			if reverts != sc.Expected.RevertCount {
				t.Fatalf("revert_count = %d, want %d", reverts, sc.Expected.RevertCount)
			}

			// expected_tsv: byte-identical where the scenario embeds it.
			if sc.ExpectedTSV != nil && tsv != *sc.ExpectedTSV {
				t.Fatalf("TSV mismatch:\n got %q\nwant %q", tsv, *sc.ExpectedTSV)
			}
		})
	}
}

// splitTSVRows returns the data rows (header excluded), dropping the trailing
// empty element from the final newline.
func splitTSVRows(tsv string) []string {
	var rows []string
	start := 0
	first := true
	for i := 0; i < len(tsv); i++ {
		if tsv[i] == '\n' {
			line := tsv[start:i]
			start = i + 1
			if first {
				first = false // skip header
				continue
			}
			rows = append(rows, line)
		}
	}
	return rows
}

// tsvField returns the n-th (0-based) tab-separated field of a row.
func tsvField(row string, n int) string {
	field := 0
	start := 0
	for i := 0; i < len(row); i++ {
		if row[i] == '\t' {
			if field == n {
				return row[start:i]
			}
			field++
			start = i + 1
		}
	}
	if field == n {
		return row[start:]
	}
	return ""
}
