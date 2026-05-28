/**
 * Comparison + recommendation types and derivation (Rules 19-25).
 */

import type { OptimizationDirection } from "@spore/core";

import { metricName, type EvalMetric } from "./metric-map.js";
import type { ConfidenceInterval, MetricStats } from "./stats.js";

/**
 * Whether the candidate is better/worse/unchanged on a metric, relative to the
 * metric's optimization direction (Rule 22).
 */
export type ComparisonDirection = "better" | "worse" | "no_change";

/** One metric's baseline-vs-candidate comparison (Rules 19-22). */
export interface MetricComparison {
  metricName: string;
  baseline: MetricStats;
  candidate: MetricStats;
  /** candidate.mean - baseline.mean. */
  delta: number;
  /** Welch's t-test two-sided p-value (Rule 20). */
  pValue: number;
  /** Bootstrap CI for non-deterministic metrics (Rule 21). */
  ci?: ConfidenceInterval;
  direction: ComparisonDirection;
}

/** The runner's recommendation (Rules 23-24). */
export type Recommendation =
  /** Adopt the candidate: primary metric improved with p < 0.05. */
  | { kind: "adopt"; configId: string; confidence: number }
  /** Reject: the candidate is clearly worse on the primary metric. */
  | { kind: "reject"; reason: string }
  /** Not enough runs to call it (p ≥ 0.05) — power-estimated `recommendedN`. */
  | { kind: "needs_more_runs"; currentN: number; recommendedN: number }
  /** Mixed signals across metrics. */
  | { kind: "ambiguous"; tradeoffs: string[] };

/**
 * The full comparison report for one candidate config (Rule 25 includes
 * `traceLinks`).
 */
export interface ComparisonReport {
  baselineConfigId: string;
  candidateConfigId: string;
  metrics: MetricComparison[];
  recommendation: Recommendation;
  traceLinks: string[];
}

/**
 * Classify the direction of a delta relative to the metric's optimization
 * direction (Rule 22). A delta within `eps` of zero is `no_change`.
 */
export function classifyDirection(
  delta: number,
  direction: OptimizationDirection,
  eps: number,
): ComparisonDirection {
  if (Math.abs(delta) <= eps) return "no_change";
  if (direction === "maximize") {
    return delta > 0 ? "better" : "worse";
  }
  return delta < 0 ? "better" : "worse";
}

/** The significance threshold for "improved" / "regressed" (Rule 23). */
export const SIGNIFICANCE_ALPHA = 0.05;

/**
 * Derive a {@link Recommendation} from the per-metric comparisons and the
 * primary metric (Rules 23-24).
 *
 * - Primary improves with p < 0.05            → `adopt`.
 * - Primary clearly worse (worse, p < 0.05)   → `reject`.
 * - Primary inconclusive (p ≥ 0.05)           → `needs_more_runs`.
 * - Otherwise mixed across metrics            → `ambiguous`.
 */
export function deriveRecommendation(
  candidateConfigId: string,
  comparisons: readonly MetricComparison[],
  primary: EvalMetric,
): Recommendation {
  const primaryName = metricName(primary);
  const primaryCmp = comparisons.find((c) => c.metricName === primaryName);
  if (!primaryCmp) {
    return {
      kind: "ambiguous",
      tradeoffs: [`primary metric ${primaryName} not measured`],
    };
  }

  const significant = primaryCmp.pValue < SIGNIFICANCE_ALPHA;

  if (significant && primaryCmp.direction === "better") {
    return {
      kind: "adopt",
      configId: candidateConfigId,
      confidence: 1 - primaryCmp.pValue,
    };
  }
  if (significant && primaryCmp.direction === "worse") {
    return {
      kind: "reject",
      reason: `primary metric ${primaryName} regressed (delta=${primaryCmp.delta.toFixed(4)}, p=${primaryCmp.pValue.toFixed(4)})`,
    };
  }

  // Inconclusive on the primary metric. If other metrics disagree in direction,
  // flag the tradeoffs; otherwise ask for more runs.
  const mixed = comparisons
    .filter((c) => c.direction !== "no_change")
    .map((c) => `${c.metricName}: ${c.direction} (p=${c.pValue.toFixed(3)})`);
  const anyBetter = comparisons.some((c) => c.direction === "better");
  const anyWorse = comparisons.some((c) => c.direction === "worse");
  if (anyBetter && anyWorse) {
    return { kind: "ambiguous", tradeoffs: mixed };
  }
  return {
    kind: "needs_more_runs",
    currentN: primaryCmp.candidate.n,
    recommendedN: recommendedN(primaryCmp),
  };
}

/**
 * Power-based estimate of the runs needed to detect the observed effect at
 * α=0.05, power≈0.8 (Rule 24): n ≈ 16 σ² / δ². Pooled variance from baseline
 * and candidate; clamped to a sane floor above the current n.
 */
export function recommendedN(cmp: MetricComparison): number {
  const pooledVar = (cmp.baseline.stddev ** 2 + cmp.candidate.stddev ** 2) / 2;
  const delta = Math.abs(cmp.delta);
  const current = Math.max(cmp.candidate.n, 1);
  if (delta <= Number.EPSILON || pooledVar <= 0) {
    // No detectable effect / no variance: doubling is the conservative ask.
    return Math.max(current * 2, current + 1);
  }
  // 16 * sigma^2 / delta^2 is the standard two-sample rule-of-thumb.
  const est = Math.ceil((16 * pooledVar) / (delta * delta));
  return Math.max(est, current + 1);
}
