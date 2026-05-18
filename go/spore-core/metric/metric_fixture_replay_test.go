package metric

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// should_keep.json
// ============================================================================

type shouldKeepCase struct {
	Name        string                          `json:"name"`
	NewValue    float64                         `json:"new_value"`
	CurrentBest float64                         `json:"current_best"`
	Direction   sporecore.OptimizationDirection `json:"direction"`
	MinDelta    *float64                        `json:"min_delta"`
	Expected    bool                            `json:"expected"`
}

type shouldKeepFixture struct {
	Cases []shouldKeepCase `json:"cases"`
}

func fixturePath(t *testing.T, name string) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/metric → ../../../fixtures/metric_evaluator/<name>
	return filepath.Join(wd, "..", "..", "..", "fixtures", "metric_evaluator", name)
}

func TestShouldKeepFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(fixturePath(t, "should_keep.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx shouldKeepFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}
	for _, c := range fx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			got := ShouldKeep(c.NewValue, c.CurrentBest, c.Direction, c.MinDelta)
			if got != c.Expected {
				t.Fatalf("case %s: got %v want %v", c.Name, got, c.Expected)
			}
		})
	}
}

// ============================================================================
// parse_metric.json
// ============================================================================

type parseExpected struct {
	Kind  string  `json:"kind"`
	Value float64 `json:"value"`
}

type parseCase struct {
	Name     string        `json:"name"`
	Output   string        `json:"output"`
	Pattern  string        `json:"pattern"`
	Expected parseExpected `json:"expected"`
}

type parseFixture struct {
	Cases []parseCase `json:"cases"`
}

func TestParseMetricFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(fixturePath(t, "parse_metric.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx parseFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("fixture has no cases")
	}
	for _, c := range fx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			got, merr := ParseMetric(c.Output, c.Pattern)
			switch c.Expected.Kind {
			case "value":
				if merr != nil {
					t.Fatalf("case %s: unexpected error %v", c.Name, merr)
				}
				if math.Abs(got-c.Expected.Value) > 1e-9 {
					t.Fatalf("case %s: got %v want %v", c.Name, got, c.Expected.Value)
				}
			case "parse_failed":
				if merr == nil || merr.Kind != MetricErrParseFailed {
					t.Fatalf("case %s: expected parse_failed, got %v", c.Name, merr)
				}
			default:
				t.Fatalf("case %s: unknown expected kind %q", c.Name, c.Expected.Kind)
			}
		})
	}
}
