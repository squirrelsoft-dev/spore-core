/**
 * {@link EvalMetric} + extraction of metric samples from observability
 * (Rule 16, Rule 17, Resolution 1).
 *
 * Simple aggregates (turns, cost, wall-time, success) come from
 * {@link SessionMetrics} via `getSessionMetrics`. Filtered metrics
 * (`cache_hit_rate{block}`, `sensor_fire_rate{sensor_id}`,
 * `middleware_intervention_rate{hook}`) are computed from `getTrace` spans
 * (Resolution 1). Issue #12's public surface is NOT modified.
 *
 * Resolution 1 (TS clean-accessor form): unlike the Rust reference — which
 * reads these three trace-sourced metrics out of the span `Debug` string
 * because its `Span` type is not downcastable — TS spans are structurally
 * typed, so we read the typed fields directly (`TurnSpan.cache_read_tokens`,
 * `SensorSpan.sensor_id`, `MiddlewareSpan.hook`). No Debug-string parsing.
 */

import type { observability } from "@spore/core";
import type { OptimizationDirection } from "@spore/core";

type Span = observability.Span;
type SessionMetrics = observability.SessionMetrics;
type TurnSpan = observability.TurnSpan;
type SensorSpan = observability.SensorSpan;
type MiddlewareSpan = observability.MiddlewareSpan;

// The `Span` union shares a nested discriminant (`base.kind`), which TS does
// not auto-narrow on. These guards narrow to the structurally-typed payload so
// Resolution 1's trace-sourced metrics read the typed fields directly.
function isTurnSpan(s: Span): s is TurnSpan {
  return s.base.kind === "turn";
}
function isSensorSpan(s: Span): s is SensorSpan {
  return s.base.kind === "sensor_evaluation";
}
function isMiddlewareSpan(s: Span): s is MiddlewareSpan {
  return s.base.kind === "middleware_hook";
}

/** A metric the EvalHarness aggregates and compares (Rule 17). */
export type EvalMetric =
  /** Fraction of runs whose verifier `passed`. */
  | { kind: "task_success_rate" }
  | { kind: "mean_turns_to_completion" }
  | { kind: "mean_cost_usd" }
  | { kind: "mean_wall_time" }
  /** Cache-read hit rate for a named cache block (from turn spans). */
  | { kind: "cache_hit_rate"; block: string }
  /** How often a named sensor fired (from sensor spans). */
  | { kind: "sensor_fire_rate"; sensor_id: string }
  /** How often a named middleware hook intervened (from middleware spans). */
  | { kind: "middleware_intervention_rate"; hook: string }
  /** Mean verification score across runs. */
  | { kind: "verification_score" };

/**
 * The optimization direction (Rule 22): higher-is-better metrics maximize;
 * cost, turns, wall-time, sensor-fire, and intervention rate minimize.
 */
export function metricDirection(metric: EvalMetric): OptimizationDirection {
  switch (metric.kind) {
    case "task_success_rate":
    case "cache_hit_rate":
    case "verification_score":
      return "maximize";
    case "mean_turns_to_completion":
    case "mean_cost_usd":
    case "mean_wall_time":
    case "sensor_fire_rate":
    case "middleware_intervention_rate":
      return "minimize";
    default: {
      const _exhaustive: never = metric;
      return _exhaustive;
    }
  }
}

/** A stable display/serialization name. */
export function metricName(metric: EvalMetric): string {
  switch (metric.kind) {
    case "task_success_rate":
      return "task_success_rate";
    case "mean_turns_to_completion":
      return "mean_turns_to_completion";
    case "mean_cost_usd":
      return "mean_cost_usd";
    case "mean_wall_time":
      return "mean_wall_time";
    case "cache_hit_rate":
      return `cache_hit_rate[${metric.block}]`;
    case "sensor_fire_rate":
      return `sensor_fire_rate[${metric.sensor_id}]`;
    case "middleware_intervention_rate":
      return `middleware_intervention_rate[${metric.hook}]`;
    case "verification_score":
      return "verification_score";
    default: {
      const _exhaustive: never = metric;
      return _exhaustive;
    }
  }
}

/** Whether this metric is sourced from filtered trace spans (Resolution 1)
 *  rather than from the simple {@link SessionMetrics} aggregate. */
export function metricFromTrace(metric: EvalMetric): boolean {
  return (
    metric.kind === "cache_hit_rate" ||
    metric.kind === "sensor_fire_rate" ||
    metric.kind === "middleware_intervention_rate"
  );
}

/** Per-run inputs needed to derive a metric sample that are not in
 *  observability: the verifier outcome. */
export interface RunSampleInputs {
  verifierPassed: boolean;
  verifierScore: number;
}

/**
 * Compute the sample value of `metric` for a single run, given that run's
 * {@link SessionMetrics} (simple aggregates), the run's trace `spans` (filtered
 * metrics, Resolution 1), and the verifier outcome.
 */
export function sampleFor(
  metric: EvalMetric,
  session: SessionMetrics,
  spans: readonly Span[],
  inputs: RunSampleInputs,
): number {
  switch (metric.kind) {
    case "task_success_rate":
      return inputs.verifierPassed ? 1.0 : 0.0;
    case "verification_score":
      return inputs.verifierScore;
    case "mean_turns_to_completion":
      return session.total_turns;
    case "mean_cost_usd":
      return session.total_cost_usd;
    case "mean_wall_time":
      return session.total_duration_ms;
    case "cache_hit_rate":
      return cacheHitRate(spans);
    case "sensor_fire_rate":
      return sensorFireRate(spans, metric.sensor_id);
    case "middleware_intervention_rate":
      return middlewareInterventionRate(spans, metric.hook);
    default: {
      const _exhaustive: never = metric;
      return _exhaustive;
    }
  }
}

/** Cache-read hit rate from turn spans: cache_read / (input + cache_read)
 *  (Resolution 1, `TurnSpan.cache_read_tokens`). 0 with no tokens. */
function cacheHitRate(spans: readonly Span[]): number {
  let cacheRead = 0;
  let input = 0;
  for (const s of spans) {
    if (isTurnSpan(s)) {
      cacheRead += s.cache_read_tokens ?? 0;
      input += s.input_tokens;
    }
  }
  const denom = input + cacheRead;
  return denom === 0 ? 0 : cacheRead / denom;
}

/** Fire rate of a named sensor: fired evaluations / total evaluations of that
 *  sensor (Resolution 1, `SensorSpan.sensor_id`). 0 if never evaluated. */
function sensorFireRate(spans: readonly Span[], sensorId: string): number {
  let fired = 0;
  let total = 0;
  for (const s of spans) {
    if (isSensorSpan(s) && s.sensor_id.asString() === sensorId) {
      total += 1;
      if (s.fired) fired += 1;
    }
  }
  return total === 0 ? 0 : fired / total;
}

/** Intervention rate of a named middleware hook: non-continue decisions /
 *  total firings at that hook (Resolution 1, `MiddlewareSpan.hook`). */
function middlewareInterventionRate(
  spans: readonly Span[],
  hook: string,
): number {
  let interventions = 0;
  let total = 0;
  const target = hook.toLowerCase();
  for (const s of spans) {
    if (isMiddlewareSpan(s) && s.hook.toLowerCase() === target) {
      total += 1;
      if (s.decision.kind !== "continue") interventions += 1;
    }
  }
  return total === 0 ? 0 : interventions / total;
}
