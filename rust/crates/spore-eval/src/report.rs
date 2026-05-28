//! Comparison + recommendation types and derivation (Rules 19-25).

use serde::{Deserialize, Serialize};
use spore_core::harness::OptimizationDirection;

use crate::metric_map::EvalMetric;
use crate::stats::{ConfidenceInterval, MetricStats};

/// Whether the candidate is better/worse/unchanged on a metric, relative to the
/// metric's optimization direction (Rule 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonDirection {
    Better,
    Worse,
    NoChange,
}

/// One metric's baseline-vs-candidate comparison (Rules 19-22).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric_name: String,
    pub baseline: MetricStats,
    pub candidate: MetricStats,
    /// candidate.mean - baseline.mean.
    pub delta: f64,
    /// Welch's t-test two-sided p-value (Rule 20).
    pub p_value: f64,
    /// Bootstrap CI for non-deterministic metrics (Rule 21).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci: Option<ConfidenceInterval>,
    pub direction: ComparisonDirection,
}

/// The runner's recommendation (Rules 23-24).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Recommendation {
    /// Adopt the candidate: primary metric improved with p < 0.05.
    Adopt { config_id: String, confidence: f64 },
    /// Reject: the candidate is clearly worse on the primary metric.
    Reject { reason: String },
    /// Not enough runs to call it (p ≥ 0.05) — power-estimated `recommended_n`.
    NeedsMoreRuns { current_n: u32, recommended_n: u32 },
    /// Mixed signals across metrics.
    Ambiguous { tradeoffs: Vec<String> },
}

/// The full comparison report for one candidate config (Rule 25 includes
/// `trace_links`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub baseline_config_id: String,
    pub candidate_config_id: String,
    pub metrics: Vec<MetricComparison>,
    pub recommendation: Recommendation,
    pub trace_links: Vec<String>,
}

/// Classify the direction of a delta relative to the metric's optimization
/// direction (Rule 22). A delta within `eps` of zero is `NoChange`.
pub fn classify_direction(
    delta: f64,
    direction: OptimizationDirection,
    eps: f64,
) -> ComparisonDirection {
    if delta.abs() <= eps {
        return ComparisonDirection::NoChange;
    }
    match direction {
        OptimizationDirection::Maximize => {
            if delta > 0.0 {
                ComparisonDirection::Better
            } else {
                ComparisonDirection::Worse
            }
        }
        OptimizationDirection::Minimize => {
            if delta < 0.0 {
                ComparisonDirection::Better
            } else {
                ComparisonDirection::Worse
            }
        }
    }
}

/// The significance threshold for "improved" / "regressed" (Rule 23).
pub const SIGNIFICANCE_ALPHA: f64 = 0.05;

/// Derive a [`Recommendation`] from the per-metric comparisons and the primary
/// metric (Rules 23-24).
///
/// - Primary improves with p < 0.05            → `Adopt`.
/// - Primary clearly worse (Worse, p < 0.05)   → `Reject`.
/// - Primary inconclusive (p ≥ 0.05)           → `NeedsMoreRuns`.
/// - Otherwise mixed across metrics            → `Ambiguous`.
pub fn derive_recommendation(
    candidate_config_id: &str,
    comparisons: &[MetricComparison],
    primary: &EvalMetric,
) -> Recommendation {
    let primary_name = primary.name();
    let primary_cmp = comparisons.iter().find(|c| c.metric_name == primary_name);
    let Some(primary_cmp) = primary_cmp else {
        return Recommendation::Ambiguous {
            tradeoffs: vec![format!("primary metric {primary_name} not measured")],
        };
    };

    let significant = primary_cmp.p_value < SIGNIFICANCE_ALPHA;

    match (significant, primary_cmp.direction) {
        (true, ComparisonDirection::Better) => Recommendation::Adopt {
            config_id: candidate_config_id.to_string(),
            confidence: 1.0 - primary_cmp.p_value,
        },
        (true, ComparisonDirection::Worse) => Recommendation::Reject {
            reason: format!(
                "primary metric {primary_name} regressed (delta={:.4}, p={:.4})",
                primary_cmp.delta, primary_cmp.p_value
            ),
        },
        _ => {
            // Inconclusive on the primary metric. If other metrics disagree in
            // direction, flag the tradeoffs; otherwise ask for more runs.
            let mixed: Vec<String> = comparisons
                .iter()
                .filter(|c| c.direction != ComparisonDirection::NoChange)
                .map(|c| format!("{}: {:?} (p={:.3})", c.metric_name, c.direction, c.p_value))
                .collect();
            let any_better = comparisons
                .iter()
                .any(|c| c.direction == ComparisonDirection::Better);
            let any_worse = comparisons
                .iter()
                .any(|c| c.direction == ComparisonDirection::Worse);
            if any_better && any_worse {
                Recommendation::Ambiguous { tradeoffs: mixed }
            } else {
                Recommendation::NeedsMoreRuns {
                    current_n: primary_cmp.candidate.n,
                    recommended_n: recommended_n(primary_cmp),
                }
            }
        }
    }
}

/// Power-based estimate of the runs needed to detect the observed effect at
/// α=0.05, power≈0.8 (Rule 24): n ≈ 16 σ² / δ². Pooled variance from baseline
/// and candidate; clamped to a sane floor above the current n.
pub fn recommended_n(cmp: &MetricComparison) -> u32 {
    let pooled_var = (cmp.baseline.stddev.powi(2) + cmp.candidate.stddev.powi(2)) / 2.0;
    let delta = cmp.delta.abs();
    let current = cmp.candidate.n.max(1);
    if delta <= f64::EPSILON || pooled_var <= 0.0 {
        // No detectable effect / no variance: doubling is the conservative ask.
        return (current * 2).max(current + 1);
    }
    // 16 * sigma^2 / delta^2 is the standard two-sample rule-of-thumb.
    let est = (16.0 * pooled_var / (delta * delta)).ceil() as u32;
    est.max(current + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(mean: f64, sd: f64, n: u32) -> MetricStats {
        MetricStats {
            mean,
            stddev: sd,
            p50: mean,
            p95: mean,
            n,
        }
    }

    #[test]
    fn classify_respects_direction() {
        // Maximize: positive delta is better.
        assert_eq!(
            classify_direction(0.2, OptimizationDirection::Maximize, 1e-9),
            ComparisonDirection::Better
        );
        // Minimize: positive delta is worse.
        assert_eq!(
            classify_direction(0.2, OptimizationDirection::Minimize, 1e-9),
            ComparisonDirection::Worse
        );
        assert_eq!(
            classify_direction(0.0, OptimizationDirection::Maximize, 1e-9),
            ComparisonDirection::NoChange
        );
    }

    #[test]
    fn adopt_when_primary_improves_significantly() {
        let cmp = MetricComparison {
            metric_name: EvalMetric::TaskSuccessRate.name(),
            baseline: stats(0.5, 0.1, 5),
            candidate: stats(0.9, 0.1, 5),
            delta: 0.4,
            p_value: 0.01,
            ci: None,
            direction: ComparisonDirection::Better,
        };
        let rec = derive_recommendation("cand", &[cmp], &EvalMetric::TaskSuccessRate);
        assert!(matches!(rec, Recommendation::Adopt { .. }));
    }

    #[test]
    fn reject_when_primary_regresses_significantly() {
        let cmp = MetricComparison {
            metric_name: EvalMetric::TaskSuccessRate.name(),
            baseline: stats(0.9, 0.05, 5),
            candidate: stats(0.4, 0.05, 5),
            delta: -0.5,
            p_value: 0.001,
            ci: None,
            direction: ComparisonDirection::Worse,
        };
        let rec = derive_recommendation("cand", &[cmp], &EvalMetric::TaskSuccessRate);
        assert!(matches!(rec, Recommendation::Reject { .. }));
    }

    #[test]
    fn needs_more_runs_when_inconclusive() {
        let cmp = MetricComparison {
            metric_name: EvalMetric::TaskSuccessRate.name(),
            baseline: stats(0.5, 0.3, 3),
            candidate: stats(0.6, 0.3, 3),
            delta: 0.1,
            p_value: 0.4,
            ci: None,
            direction: ComparisonDirection::Better,
        };
        let rec = derive_recommendation("cand", &[cmp], &EvalMetric::TaskSuccessRate);
        match rec {
            Recommendation::NeedsMoreRuns {
                current_n,
                recommended_n,
            } => {
                assert_eq!(current_n, 3);
                assert!(recommended_n > 3);
            }
            other => panic!("expected NeedsMoreRuns, got {other:?}"),
        }
    }
}
