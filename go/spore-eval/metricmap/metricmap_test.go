package metricmap

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
)

func TestNameAndDirection(t *testing.T) {
	cases := []struct {
		m    EvalMetric
		name string
		dir  sporecore.OptimizationDirection
	}{
		{TaskSuccessRate(), "task_success_rate", sporecore.OptimizationMaximize},
		{MeanTurns(), "mean_turns_to_completion", sporecore.OptimizationMinimize},
		{MeanCost(), "mean_cost_usd", sporecore.OptimizationMinimize},
		{MeanWallTime(), "mean_wall_time", sporecore.OptimizationMinimize},
		{VerificationScore(), "verification_score", sporecore.OptimizationMaximize},
		{CacheHitRate("blk"), "cache_hit_rate[blk]", sporecore.OptimizationMaximize},
		{SensorFireRate("s1"), "sensor_fire_rate[s1]", sporecore.OptimizationMinimize},
		{MiddlewareInterventionRate("h"), "middleware_intervention_rate[h]", sporecore.OptimizationMinimize},
	}
	for _, c := range cases {
		if c.m.Name() != c.name {
			t.Errorf("name(%v)=%q want %q", c.m.Kind, c.m.Name(), c.name)
		}
		if c.m.Direction() != c.dir {
			t.Errorf("dir(%v)=%v want %v", c.m.Kind, c.m.Direction(), c.dir)
		}
	}
}

func TestSampleSimpleAggregates(t *testing.T) {
	sm := &observability.SessionMetrics{TotalTurns: 5, TotalCostUSD: 0.25, TotalDurationMs: 1200}
	in := RunSampleInputs{VerifierPassed: true, VerifierScore: 0.8}
	if got := SampleFor(TaskSuccessRate(), sm, nil, in); got != 1.0 {
		t.Errorf("success=%v", got)
	}
	if got := SampleFor(VerificationScore(), sm, nil, in); got != 0.8 {
		t.Errorf("score=%v", got)
	}
	if got := SampleFor(MeanTurns(), sm, nil, in); got != 5.0 {
		t.Errorf("turns=%v", got)
	}
	if got := SampleFor(MeanCost(), sm, nil, in); got != 0.25 {
		t.Errorf("cost=%v", got)
	}
	if got := SampleFor(MeanWallTime(), sm, nil, in); got != 1200.0 {
		t.Errorf("wall=%v", got)
	}
}

// Resolution 1: span-sourced metrics read concrete span fields via type switch.
func TestSampleFromTraceSpans(t *testing.T) {
	u32 := func(v uint32) *uint32 { return &v }
	spans := []observability.Span{
		observability.TurnSpan{InputTokens: 20, CacheReadTokens: u32(80)},
		observability.TurnSpan{InputTokens: 0, CacheReadTokens: u32(0)},
		observability.SensorSpan{SensorID: observability.SensorID("lint"), Fired: true},
		observability.SensorSpan{SensorID: observability.SensorID("lint"), Fired: false},
		observability.SensorSpan{SensorID: observability.SensorID("other"), Fired: true},
		observability.MiddlewareSpan{Hook: observability.HookPoint("before_turn"), Decision: middleware.MiddlewareDecision{Kind: middleware.DecisionContinue}},
		observability.MiddlewareSpan{Hook: observability.HookPoint("before_turn"), Decision: middleware.MiddlewareDecision{Kind: middleware.DecisionHalt}},
	}
	in := RunSampleInputs{}
	// cache: 80 / (20+80) = 0.8
	if got := SampleFor(CacheHitRate("any"), nil, spans, in); got != 0.8 {
		t.Errorf("cache=%v want 0.8", got)
	}
	// sensor lint: 1 fired / 2 total = 0.5
	if got := SampleFor(SensorFireRate("lint"), nil, spans, in); got != 0.5 {
		t.Errorf("sensor=%v want 0.5", got)
	}
	// middleware before_turn: 1 intervention / 2 total = 0.5
	if got := SampleFor(MiddlewareInterventionRate("before_turn"), nil, spans, in); got != 0.5 {
		t.Errorf("mw=%v want 0.5", got)
	}
	// unknown sensor -> 0
	if got := SampleFor(SensorFireRate("nope"), nil, spans, in); got != 0.0 {
		t.Errorf("unknown sensor=%v", got)
	}
}

// Regression guard (#68): a ContinueWithModification decision MUST count as an
// intervention (predicate is "Kind != DecisionContinue"), keeping Go aligned
// with Rust/TypeScript/Python. Rust had diverged here.
func TestMiddlewareInterventionRateContinueWithModification(t *testing.T) {
	cwm := []observability.Span{
		observability.MiddlewareSpan{Hook: observability.HookPoint("before_turn"), Decision: middleware.MiddlewareDecision{Kind: middleware.DecisionContinueWithModification}},
	}
	if got := SampleFor(MiddlewareInterventionRate("before_turn"), nil, cwm, RunSampleInputs{}); got != 1.0 {
		t.Errorf("continue_with_modification rate=%v want 1.0 (must count as intervention)", got)
	}

	cont := []observability.Span{
		observability.MiddlewareSpan{Hook: observability.HookPoint("before_turn"), Decision: middleware.MiddlewareDecision{Kind: middleware.DecisionContinue}},
	}
	if got := SampleFor(MiddlewareInterventionRate("before_turn"), nil, cont, RunSampleInputs{}); got != 0.0 {
		t.Errorf("continue rate=%v want 0.0 (must NOT count as intervention)", got)
	}
}

func TestFromTrace(t *testing.T) {
	if !CacheHitRate("x").FromTrace() || !SensorFireRate("x").FromTrace() || !MiddlewareInterventionRate("x").FromTrace() {
		t.Error("filtered metrics should be from-trace")
	}
	if TaskSuccessRate().FromTrace() || MeanCost().FromTrace() {
		t.Error("aggregate metrics should not be from-trace")
	}
}
