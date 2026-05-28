package report

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/metricmap"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/stats"
)

func mkStats(mean, sd float64, n uint32) stats.MetricStats {
	return stats.MetricStats{Mean: mean, Stddev: sd, P50: mean, P95: mean, N: n}
}

func TestClassifyRespectsDirection(t *testing.T) {
	if ClassifyDirection(0.2, sporecore.OptimizationMaximize, 1e-9) != DirectionBetter {
		t.Fatal("maximize positive should be better")
	}
	if ClassifyDirection(0.2, sporecore.OptimizationMinimize, 1e-9) != DirectionWorse {
		t.Fatal("minimize positive should be worse")
	}
	if ClassifyDirection(0.0, sporecore.OptimizationMaximize, 1e-9) != DirectionNoChange {
		t.Fatal("zero should be no change")
	}
}

func TestAdoptWhenPrimaryImprovesSignificantly(t *testing.T) {
	cmp := MetricComparison{
		MetricName: metricmap.TaskSuccessRate().Name(),
		Baseline:   mkStats(0.5, 0.1, 5),
		Candidate:  mkStats(0.9, 0.1, 5),
		Delta:      0.4,
		PValue:     0.01,
		Direction:  DirectionBetter,
	}
	rec := DeriveRecommendation("cand", []MetricComparison{cmp}, metricmap.TaskSuccessRate())
	if rec.Kind != RecAdopt {
		t.Fatalf("got %+v", rec)
	}
}

func TestRejectWhenPrimaryRegressesSignificantly(t *testing.T) {
	cmp := MetricComparison{
		MetricName: metricmap.TaskSuccessRate().Name(),
		Baseline:   mkStats(0.9, 0.05, 5),
		Candidate:  mkStats(0.4, 0.05, 5),
		Delta:      -0.5,
		PValue:     0.001,
		Direction:  DirectionWorse,
	}
	rec := DeriveRecommendation("cand", []MetricComparison{cmp}, metricmap.TaskSuccessRate())
	if rec.Kind != RecReject {
		t.Fatalf("got %+v", rec)
	}
}

func TestNeedsMoreRunsWhenInconclusive(t *testing.T) {
	cmp := MetricComparison{
		MetricName: metricmap.TaskSuccessRate().Name(),
		Baseline:   mkStats(0.5, 0.3, 3),
		Candidate:  mkStats(0.6, 0.3, 3),
		Delta:      0.1,
		PValue:     0.4,
		Direction:  DirectionBetter,
	}
	rec := DeriveRecommendation("cand", []MetricComparison{cmp}, metricmap.TaskSuccessRate())
	if rec.Kind != RecNeedsMoreRuns || rec.CurrentN != 3 || rec.RecommendedN <= 3 {
		t.Fatalf("got %+v", rec)
	}
}

func TestAmbiguousWhenMixed(t *testing.T) {
	primary := MetricComparison{
		MetricName: metricmap.TaskSuccessRate().Name(),
		Baseline:   mkStats(0.5, 0.3, 3),
		Candidate:  mkStats(0.55, 0.3, 3),
		Delta:      0.05,
		PValue:     0.5,
		Direction:  DirectionBetter,
	}
	other := MetricComparison{
		MetricName: metricmap.MeanCost().Name(),
		Baseline:   mkStats(0.1, 0.01, 3),
		Candidate:  mkStats(0.2, 0.01, 3),
		Delta:      0.1,
		PValue:     0.5,
		Direction:  DirectionWorse,
	}
	rec := DeriveRecommendation("cand", []MetricComparison{primary, other}, metricmap.TaskSuccessRate())
	if rec.Kind != RecAmbiguous {
		t.Fatalf("got %+v", rec)
	}
}
