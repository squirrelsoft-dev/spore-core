// Package metricmap defines the EvalMetric type + extraction of metric samples
// from observability (Rule 16, Rule 17, Resolution 1).
//
// Simple aggregates (turns, cost, wall-time, success) come from SessionMetrics
// via GetSessionMetrics. Filtered metrics (CacheHitRate{block},
// SensorFireRate{sensor_id}, MiddlewareInterventionRate{hook}) are computed
// from GetTrace spans (Resolution 1).
//
// Per Resolution 1 (Go note): the three span-sourced metrics are read via a
// clean type switch on the concrete span payload types (TurnSpan,
// SensorSpan, MiddlewareSpan) — NOT via the Debug-string-parsing hack the Rust
// reference uses as a workaround. The ObservabilityProvider public surface is
// not modified.
package metricmap

import (
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
)

// EvalMetricKind discriminates EvalMetric variants.
type EvalMetricKind string

const (
	// MetricTaskSuccessRate — fraction of runs whose verifier passed.
	MetricTaskSuccessRate EvalMetricKind = "task_success_rate"
	// MetricMeanTurnsToCompletion — mean total turns.
	MetricMeanTurnsToCompletion EvalMetricKind = "mean_turns_to_completion"
	// MetricMeanCostUsd — mean total cost in USD.
	MetricMeanCostUsd EvalMetricKind = "mean_cost_usd"
	// MetricMeanWallTime — mean wall time (ms).
	MetricMeanWallTime EvalMetricKind = "mean_wall_time"
	// MetricCacheHitRate — cache-read hit rate for a named cache block.
	MetricCacheHitRate EvalMetricKind = "cache_hit_rate"
	// MetricSensorFireRate — fire rate of a named sensor.
	MetricSensorFireRate EvalMetricKind = "sensor_fire_rate"
	// MetricMiddlewareInterventionRate — intervention rate of a named hook.
	MetricMiddlewareInterventionRate EvalMetricKind = "middleware_intervention_rate"
	// MetricVerificationScore — mean verification score across runs.
	MetricVerificationScore EvalMetricKind = "verification_score"
)

// EvalMetric is a metric the EvalHarness aggregates and compares (Rule 17).
// Block / SensorID / Hook carry the parameter for the filtered variants.
type EvalMetric struct {
	Kind EvalMetricKind `json:"kind"`
	// cache_hit_rate
	Block string `json:"-"`
	// sensor_fire_rate
	SensorID string `json:"-"`
	// middleware_intervention_rate
	Hook string `json:"-"`
}

// Constructors for the simple variants.

// TaskSuccessRate is the success-rate metric.
func TaskSuccessRate() EvalMetric { return EvalMetric{Kind: MetricTaskSuccessRate} }

// VerificationScore is the mean verification-score metric.
func VerificationScore() EvalMetric { return EvalMetric{Kind: MetricVerificationScore} }

// MeanTurns is the mean-turns metric.
func MeanTurns() EvalMetric { return EvalMetric{Kind: MetricMeanTurnsToCompletion} }

// MeanCost is the mean-cost metric.
func MeanCost() EvalMetric { return EvalMetric{Kind: MetricMeanCostUsd} }

// MeanWallTime is the mean-wall-time metric.
func MeanWallTime() EvalMetric { return EvalMetric{Kind: MetricMeanWallTime} }

// CacheHitRate is the cache-hit-rate metric for a named block.
func CacheHitRate(block string) EvalMetric {
	return EvalMetric{Kind: MetricCacheHitRate, Block: block}
}

// SensorFireRate is the sensor-fire-rate metric for a named sensor.
func SensorFireRate(sensorID string) EvalMetric {
	return EvalMetric{Kind: MetricSensorFireRate, SensorID: sensorID}
}

// MiddlewareInterventionRate is the intervention-rate metric for a named hook.
func MiddlewareInterventionRate(hook string) EvalMetric {
	return EvalMetric{Kind: MetricMiddlewareInterventionRate, Hook: hook}
}

// MarshalJSON serialises as a flat tagged object keyed by "kind".
func (m EvalMetric) MarshalJSON() ([]byte, error) {
	switch m.Kind {
	case MetricCacheHitRate:
		return jsonMarshal(struct {
			Kind  EvalMetricKind `json:"kind"`
			Block string         `json:"block"`
		}{m.Kind, m.Block})
	case MetricSensorFireRate:
		return jsonMarshal(struct {
			Kind     EvalMetricKind `json:"kind"`
			SensorID string         `json:"sensor_id"`
		}{m.Kind, m.SensorID})
	case MetricMiddlewareInterventionRate:
		return jsonMarshal(struct {
			Kind EvalMetricKind `json:"kind"`
			Hook string         `json:"hook"`
		}{m.Kind, m.Hook})
	default:
		return jsonMarshal(struct {
			Kind EvalMetricKind `json:"kind"`
		}{m.Kind})
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (m *EvalMetric) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind     EvalMetricKind `json:"kind"`
		Block    string         `json:"block"`
		SensorID string         `json:"sensor_id"`
		Hook     string         `json:"hook"`
	}
	if err := jsonUnmarshal(data, &probe); err != nil {
		return err
	}
	m.Kind = probe.Kind
	m.Block = probe.Block
	m.SensorID = probe.SensorID
	m.Hook = probe.Hook
	if m.Kind == "" {
		return fmt.Errorf("EvalMetric: missing kind")
	}
	return nil
}

// Direction returns the optimization direction: higher-is-better metrics
// maximize; cost, turns, wall-time, and intervention/sensor rates minimize.
func (m EvalMetric) Direction() sporecore.OptimizationDirection {
	switch m.Kind {
	case MetricTaskSuccessRate, MetricCacheHitRate, MetricVerificationScore:
		return sporecore.OptimizationMaximize
	default:
		return sporecore.OptimizationMinimize
	}
}

// Name returns a stable display/serialization name.
func (m EvalMetric) Name() string {
	switch m.Kind {
	case MetricTaskSuccessRate:
		return "task_success_rate"
	case MetricMeanTurnsToCompletion:
		return "mean_turns_to_completion"
	case MetricMeanCostUsd:
		return "mean_cost_usd"
	case MetricMeanWallTime:
		return "mean_wall_time"
	case MetricCacheHitRate:
		return fmt.Sprintf("cache_hit_rate[%s]", m.Block)
	case MetricSensorFireRate:
		return fmt.Sprintf("sensor_fire_rate[%s]", m.SensorID)
	case MetricMiddlewareInterventionRate:
		return fmt.Sprintf("middleware_intervention_rate[%s]", m.Hook)
	case MetricVerificationScore:
		return "verification_score"
	default:
		return string(m.Kind)
	}
}

// FromTrace reports whether this metric is sourced from filtered trace spans
// (Resolution 1) rather than from the SessionMetrics aggregate.
func (m EvalMetric) FromTrace() bool {
	switch m.Kind {
	case MetricCacheHitRate, MetricSensorFireRate, MetricMiddlewareInterventionRate:
		return true
	default:
		return false
	}
}

// RunSampleInputs are the per-run inputs not in observability: the verifier
// outcome.
type RunSampleInputs struct {
	VerifierPassed bool
	VerifierScore  float64
}

// SampleFor computes the sample value of metric for a single run, given that
// run's SessionMetrics (simple aggregates), the run's trace spans (filtered
// metrics, Resolution 1), and the verifier outcome.
func SampleFor(metric EvalMetric, session *observability.SessionMetrics, spans []observability.Span, inputs RunSampleInputs) float64 {
	switch metric.Kind {
	case MetricTaskSuccessRate:
		if inputs.VerifierPassed {
			return 1.0
		}
		return 0.0
	case MetricVerificationScore:
		return inputs.VerifierScore
	case MetricMeanTurnsToCompletion:
		return float64(session.TotalTurns)
	case MetricMeanCostUsd:
		return session.TotalCostUSD
	case MetricMeanWallTime:
		return float64(session.TotalDurationMs)
	case MetricCacheHitRate:
		return cacheHitRate(spans)
	case MetricSensorFireRate:
		return sensorFireRate(spans, metric.SensorID)
	case MetricMiddlewareInterventionRate:
		return middlewareInterventionRate(spans, metric.Hook)
	default:
		return 0.0
	}
}

// cacheHitRate is cache_read_tokens / (input + cache_read) summed over turn
// spans (Resolution 1, TurnSpan.CacheReadTokens). 0.0 with no tokens.
func cacheHitRate(spans []observability.Span) float64 {
	var cacheRead, input uint64
	for _, s := range spans {
		if ts, ok := s.(observability.TurnSpan); ok {
			if ts.CacheReadTokens != nil {
				cacheRead += uint64(*ts.CacheReadTokens)
			}
			input += uint64(ts.InputTokens)
		}
	}
	denom := input + cacheRead
	if denom == 0 {
		return 0.0
	}
	return float64(cacheRead) / float64(denom)
}

// sensorFireRate is fired evaluations / total evaluations of the named sensor
// (Resolution 1, SensorSpan.SensorID). 0.0 if never evaluated.
func sensorFireRate(spans []observability.Span, sensorID string) float64 {
	var fired, total uint32
	for _, s := range spans {
		if ss, ok := s.(observability.SensorSpan); ok {
			if string(ss.SensorID) == sensorID {
				total++
				if ss.Fired {
					fired++
				}
			}
		}
	}
	if total == 0 {
		return 0.0
	}
	return float64(fired) / float64(total)
}

// middlewareInterventionRate is non-Continue decisions / total firings at the
// named hook (Resolution 1, MiddlewareSpan.Hook).
func middlewareInterventionRate(spans []observability.Span, hook string) float64 {
	var interventions, total uint32
	for _, s := range spans {
		if ms, ok := s.(observability.MiddlewareSpan); ok {
			if string(ms.Hook) == hook {
				total++
				if ms.Decision.Kind != middleware.DecisionContinue {
					interventions++
				}
			}
		}
	}
	if total == 0 {
		return 0.0
	}
	return float64(interventions) / float64(total)
}
