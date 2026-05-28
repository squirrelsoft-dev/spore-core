// Package report defines the comparison + recommendation types and their
// derivation (Rules 19-25).
package report

import (
	"fmt"
	"math"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/metricmap"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/stats"
)

// ComparisonDirection is whether the candidate is better/worse/unchanged on a
// metric, relative to the metric's optimization direction (Rule 22).
type ComparisonDirection string

const (
	// DirectionBetter — candidate improves the metric.
	DirectionBetter ComparisonDirection = "better"
	// DirectionWorse — candidate regresses the metric.
	DirectionWorse ComparisonDirection = "worse"
	// DirectionNoChange — within epsilon of no change.
	DirectionNoChange ComparisonDirection = "no_change"
)

// MetricComparison is one metric's baseline-vs-candidate comparison
// (Rules 19-22).
type MetricComparison struct {
	MetricName string            `json:"metric_name"`
	Baseline   stats.MetricStats `json:"baseline"`
	Candidate  stats.MetricStats `json:"candidate"`
	// Delta is candidate.mean - baseline.mean.
	Delta float64 `json:"delta"`
	// PValue is Welch's t-test two-sided p-value (Rule 20).
	PValue float64 `json:"p_value"`
	// CI is the bootstrap CI for non-deterministic metrics (Rule 21).
	CI        *stats.ConfidenceInterval `json:"ci,omitempty"`
	Direction ComparisonDirection       `json:"direction"`
}

// RecommendationKind discriminates Recommendation variants.
type RecommendationKind string

const (
	// RecAdopt — adopt the candidate; primary improved with p < 0.05.
	RecAdopt RecommendationKind = "adopt"
	// RecReject — candidate is clearly worse on the primary metric.
	RecReject RecommendationKind = "reject"
	// RecNeedsMoreRuns — not enough runs to call it (p >= 0.05).
	RecNeedsMoreRuns RecommendationKind = "needs_more_runs"
	// RecAmbiguous — mixed signals across metrics.
	RecAmbiguous RecommendationKind = "ambiguous"
)

// Recommendation is the runner's recommendation (Rules 23-24).
type Recommendation struct {
	Kind RecommendationKind `json:"kind"`
	// adopt
	ConfigID   string  `json:"-"`
	Confidence float64 `json:"-"`
	// reject
	Reason string `json:"-"`
	// needs_more_runs
	CurrentN     uint32 `json:"-"`
	RecommendedN uint32 `json:"-"`
	// ambiguous
	Tradeoffs []string `json:"-"`
}

// MarshalJSON serialises as a flat tagged object keyed by "kind".
func (r Recommendation) MarshalJSON() ([]byte, error) {
	switch r.Kind {
	case RecAdopt:
		return jsonMarshal(struct {
			Kind       RecommendationKind `json:"kind"`
			ConfigID   string             `json:"config_id"`
			Confidence float64            `json:"confidence"`
		}{r.Kind, r.ConfigID, r.Confidence})
	case RecReject:
		return jsonMarshal(struct {
			Kind   RecommendationKind `json:"kind"`
			Reason string             `json:"reason"`
		}{r.Kind, r.Reason})
	case RecNeedsMoreRuns:
		return jsonMarshal(struct {
			Kind         RecommendationKind `json:"kind"`
			CurrentN     uint32             `json:"current_n"`
			RecommendedN uint32             `json:"recommended_n"`
		}{r.Kind, r.CurrentN, r.RecommendedN})
	case RecAmbiguous:
		tr := r.Tradeoffs
		if tr == nil {
			tr = []string{}
		}
		return jsonMarshal(struct {
			Kind      RecommendationKind `json:"kind"`
			Tradeoffs []string           `json:"tradeoffs"`
		}{r.Kind, tr})
	default:
		return nil, fmt.Errorf("Recommendation: unknown kind %q", r.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (r *Recommendation) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind         RecommendationKind `json:"kind"`
		ConfigID     string             `json:"config_id"`
		Confidence   float64            `json:"confidence"`
		Reason       string             `json:"reason"`
		CurrentN     uint32             `json:"current_n"`
		RecommendedN uint32             `json:"recommended_n"`
		Tradeoffs    []string           `json:"tradeoffs"`
	}
	if err := jsonUnmarshal(data, &probe); err != nil {
		return err
	}
	r.Kind = probe.Kind
	r.ConfigID = probe.ConfigID
	r.Confidence = probe.Confidence
	r.Reason = probe.Reason
	r.CurrentN = probe.CurrentN
	r.RecommendedN = probe.RecommendedN
	r.Tradeoffs = probe.Tradeoffs
	return nil
}

// ComparisonReport is the full comparison report for one candidate config
// (Rule 25 includes TraceLinks).
type ComparisonReport struct {
	BaselineConfigID  string             `json:"baseline_config_id"`
	CandidateConfigID string             `json:"candidate_config_id"`
	Metrics           []MetricComparison `json:"metrics"`
	Recommendation    Recommendation     `json:"recommendation"`
	TraceLinks        []string           `json:"trace_links"`
}

// ClassifyDirection classifies the direction of a delta relative to the
// metric's optimization direction (Rule 22). A delta within eps of zero is
// NoChange.
func ClassifyDirection(delta float64, direction sporecore.OptimizationDirection, eps float64) ComparisonDirection {
	if math.Abs(delta) <= eps {
		return DirectionNoChange
	}
	switch direction {
	case sporecore.OptimizationMaximize:
		if delta > 0.0 {
			return DirectionBetter
		}
		return DirectionWorse
	default: // Minimize
		if delta < 0.0 {
			return DirectionBetter
		}
		return DirectionWorse
	}
}

// SignificanceAlpha is the significance threshold for "improved"/"regressed"
// (Rule 23).
const SignificanceAlpha = 0.05

// DeriveRecommendation derives a Recommendation from the per-metric comparisons
// and the primary metric (Rules 23-24).
//
//   - Primary improves with p < 0.05            -> Adopt.
//   - Primary clearly worse (Worse, p < 0.05)   -> Reject.
//   - Primary inconclusive (p >= 0.05)          -> NeedsMoreRuns.
//   - Otherwise mixed across metrics            -> Ambiguous.
func DeriveRecommendation(candidateConfigID string, comparisons []MetricComparison, primary metricmap.EvalMetric) Recommendation {
	primaryName := primary.Name()
	var primaryCmp *MetricComparison
	for i := range comparisons {
		if comparisons[i].MetricName == primaryName {
			primaryCmp = &comparisons[i]
			break
		}
	}
	if primaryCmp == nil {
		return Recommendation{
			Kind:      RecAmbiguous,
			Tradeoffs: []string{fmt.Sprintf("primary metric %s not measured", primaryName)},
		}
	}

	significant := primaryCmp.PValue < SignificanceAlpha

	if significant && primaryCmp.Direction == DirectionBetter {
		return Recommendation{Kind: RecAdopt, ConfigID: candidateConfigID, Confidence: 1.0 - primaryCmp.PValue}
	}
	if significant && primaryCmp.Direction == DirectionWorse {
		return Recommendation{
			Kind:   RecReject,
			Reason: fmt.Sprintf("primary metric %s regressed (delta=%.4f, p=%.4f)", primaryName, primaryCmp.Delta, primaryCmp.PValue),
		}
	}

	// Inconclusive on the primary metric. If other metrics disagree in
	// direction, flag the tradeoffs; otherwise ask for more runs.
	var mixed []string
	anyBetter, anyWorse := false, false
	for i := range comparisons {
		c := &comparisons[i]
		if c.Direction != DirectionNoChange {
			mixed = append(mixed, fmt.Sprintf("%s: %s (p=%.3f)", c.MetricName, c.Direction, c.PValue))
		}
		if c.Direction == DirectionBetter {
			anyBetter = true
		}
		if c.Direction == DirectionWorse {
			anyWorse = true
		}
	}
	if anyBetter && anyWorse {
		if mixed == nil {
			mixed = []string{}
		}
		return Recommendation{Kind: RecAmbiguous, Tradeoffs: mixed}
	}
	return Recommendation{
		Kind:         RecNeedsMoreRuns,
		CurrentN:     primaryCmp.Candidate.N,
		RecommendedN: RecommendedN(primaryCmp),
	}
}

// RecommendedN is the power-based estimate of the runs needed to detect the
// observed effect at alpha=0.05, power ~= 0.8 (Rule 24): n ~= 16 * sigma^2 /
// delta^2. Pooled variance from baseline and candidate; clamped to a floor
// above the current n.
func RecommendedN(cmp *MetricComparison) uint32 {
	pooledVar := (cmp.Baseline.Stddev*cmp.Baseline.Stddev + cmp.Candidate.Stddev*cmp.Candidate.Stddev) / 2.0
	delta := math.Abs(cmp.Delta)
	current := cmp.Candidate.N
	if current < 1 {
		current = 1
	}
	if delta <= 2.220446049250313e-16 || pooledVar <= 0.0 {
		twice := current * 2
		if twice < current+1 {
			twice = current + 1
		}
		return twice
	}
	est := uint32(math.Ceil(16.0 * pooledVar / (delta * delta)))
	if est < current+1 {
		est = current + 1
	}
	return est
}
