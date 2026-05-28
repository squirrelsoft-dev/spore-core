//! [`EvalMetric`] enum + extraction of metric samples from observability
//! (Rule 16, Rule 17, Resolution 1).
//!
//! Simple aggregates (turns, cost, wall-time, success) come from
//! [`SessionMetrics`] via `get_session_metrics`. Filtered metrics
//! (`CacheHitRate{block}`, `SensorFireRate{sensor_id}`,
//! `MiddlewareInterventionRate{hook}`) are computed from `get_trace` spans
//! (Resolution 1). Issue #12's public surface is NOT modified.

use serde::{Deserialize, Serialize};
use spore_core::harness::OptimizationDirection;
use spore_core::observability::{SessionMetrics, Span, SpanKind};

/// A metric the EvalHarness aggregates and compares (Rule 17).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvalMetric {
    /// Fraction of runs whose verifier `passed`.
    TaskSuccessRate,
    MeanTurnsToCompletion,
    MeanCostUsd,
    MeanWallTime,
    /// Cache-read hit rate for a named cache block (from turn spans).
    CacheHitRate {
        block: String,
    },
    /// How often a named sensor fired (from sensor spans).
    SensorFireRate {
        sensor_id: String,
    },
    /// How often a named middleware hook intervened (from middleware spans).
    MiddlewareInterventionRate {
        hook: String,
    },
    /// Mean verification score across runs.
    VerificationScore,
}

impl EvalMetric {
    /// The optimization direction: higher-is-better metrics maximize; cost,
    /// turns, wall-time, and intervention rate minimize.
    pub fn direction(&self) -> OptimizationDirection {
        match self {
            EvalMetric::TaskSuccessRate
            | EvalMetric::CacheHitRate { .. }
            | EvalMetric::VerificationScore => OptimizationDirection::Maximize,
            EvalMetric::MeanTurnsToCompletion
            | EvalMetric::MeanCostUsd
            | EvalMetric::MeanWallTime
            | EvalMetric::SensorFireRate { .. }
            | EvalMetric::MiddlewareInterventionRate { .. } => OptimizationDirection::Minimize,
        }
    }

    /// A stable display/serialization name.
    pub fn name(&self) -> String {
        match self {
            EvalMetric::TaskSuccessRate => "task_success_rate".into(),
            EvalMetric::MeanTurnsToCompletion => "mean_turns_to_completion".into(),
            EvalMetric::MeanCostUsd => "mean_cost_usd".into(),
            EvalMetric::MeanWallTime => "mean_wall_time".into(),
            EvalMetric::CacheHitRate { block } => format!("cache_hit_rate[{block}]"),
            EvalMetric::SensorFireRate { sensor_id } => format!("sensor_fire_rate[{sensor_id}]"),
            EvalMetric::MiddlewareInterventionRate { hook } => {
                format!("middleware_intervention_rate[{hook}]")
            }
            EvalMetric::VerificationScore => "verification_score".into(),
        }
    }

    /// Whether this metric is sourced from filtered trace spans (Resolution 1)
    /// rather than from the simple [`SessionMetrics`] aggregate.
    pub fn from_trace(&self) -> bool {
        matches!(
            self,
            EvalMetric::CacheHitRate { .. }
                | EvalMetric::SensorFireRate { .. }
                | EvalMetric::MiddlewareInterventionRate { .. }
        )
    }
}

/// Per-run inputs needed to derive a metric sample that are not in
/// observability: the verifier outcome.
pub struct RunSampleInputs {
    pub verifier_passed: bool,
    pub verifier_score: f64,
}

/// Compute the sample value of `metric` for a single run, given that run's
/// `SessionMetrics` (simple aggregates), the run's trace `spans` (filtered
/// metrics, Resolution 1), and the verifier outcome.
pub fn sample_for(
    metric: &EvalMetric,
    session: &SessionMetrics,
    spans: &[Box<dyn Span>],
    inputs: &RunSampleInputs,
) -> f64 {
    match metric {
        EvalMetric::TaskSuccessRate => {
            if inputs.verifier_passed {
                1.0
            } else {
                0.0
            }
        }
        EvalMetric::VerificationScore => inputs.verifier_score,
        EvalMetric::MeanTurnsToCompletion => session.total_turns as f64,
        EvalMetric::MeanCostUsd => session.total_cost_usd,
        EvalMetric::MeanWallTime => session.total_duration_ms as f64,
        EvalMetric::CacheHitRate { .. } => cache_hit_rate(spans),
        EvalMetric::SensorFireRate { sensor_id } => sensor_fire_rate(spans, sensor_id),
        EvalMetric::MiddlewareInterventionRate { hook } => {
            middleware_intervention_rate(spans, hook)
        }
    }
}

// `get_trace()` returns `Box<dyn Span>`. Issue #12's `Span` trait exposes only
// `base()` and `kind()` and is deliberately neither `Any`-downcastable nor
// `Serialize` (Resolution 1 forbids modifying #12's surface). The concrete
// payload fields are therefore read out of each span's stable `Debug`
// representation, keyed by `kind()`. This is the only get_trace-compatible path
// that does not touch #12.

/// Cache-read hit rate from turn spans: cache_read_tokens / (input + cache_read)
/// (Resolution 1, `TurnSpan.cache_read_tokens`). 0.0 with no tokens.
fn cache_hit_rate(spans: &[Box<dyn Span>]) -> f64 {
    let mut cache_read = 0u64;
    let mut input = 0u64;
    for s in spans {
        if s.kind() == SpanKind::Turn {
            let dbg = format!("{s:?}");
            cache_read += parse_opt_u64_field(&dbg, "cache_read_tokens").unwrap_or(0);
            input += parse_u64_field(&dbg, "input_tokens").unwrap_or(0);
        }
    }
    let denom = input + cache_read;
    if denom == 0 {
        0.0
    } else {
        cache_read as f64 / denom as f64
    }
}

/// Fire rate of a named sensor: fired evaluations / total evaluations of that
/// sensor (Resolution 1, `SensorSpan.sensor_id`). 0.0 if never evaluated.
fn sensor_fire_rate(spans: &[Box<dyn Span>], sensor_id: &str) -> f64 {
    let mut fired = 0u32;
    let mut total = 0u32;
    for s in spans {
        if s.kind() == SpanKind::SensorEvaluation {
            let dbg = format!("{s:?}");
            if parse_string_field(&dbg, "sensor_id").as_deref() == Some(sensor_id) {
                total += 1;
                if parse_bool_field(&dbg, "fired") == Some(true) {
                    fired += 1;
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        fired as f64 / total as f64
    }
}

/// Intervention rate of a named middleware hook: non-Continue decisions /
/// total firings at that hook (Resolution 1, `MiddlewareSpan.hook`).
fn middleware_intervention_rate(spans: &[Box<dyn Span>], hook: &str) -> f64 {
    let mut interventions = 0u32;
    let mut total = 0u32;
    for s in spans {
        if s.kind() == SpanKind::MiddlewareHook {
            let dbg = format!("{s:?}");
            // `hook: BeforeTurn` etc. — match case-insensitively.
            if let Some(hook_name) = parse_ident_field(&dbg, "hook") {
                if hook_name.to_lowercase() == hook.to_lowercase() {
                    total += 1;
                    // A non-`Continue` decision is an intervention.
                    if !dbg.contains("decision: Continue") {
                        interventions += 1;
                    }
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        interventions as f64 / total as f64
    }
}

// ---- Debug-representation field scanners (cross-span-kind reuse) ----

fn field_slice<'a>(dbg: &'a str, field: &str) -> Option<&'a str> {
    let needle = format!("{field}: ");
    let idx = dbg.find(&needle)? + needle.len();
    Some(&dbg[idx..])
}

fn parse_u64_field(dbg: &str, field: &str) -> Option<u64> {
    let s = field_slice(dbg, field)?;
    let num: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

fn parse_opt_u64_field(dbg: &str, field: &str) -> Option<u64> {
    let s = field_slice(dbg, field)?;
    if s.starts_with("None") {
        return Some(0);
    }
    // `Some(123)`
    let inner = s.strip_prefix("Some(")?;
    let num: String = inner.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

fn parse_bool_field(dbg: &str, field: &str) -> Option<bool> {
    let s = field_slice(dbg, field)?;
    if s.starts_with("true") {
        Some(true)
    } else if s.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// Parse a string field that renders as a tuple-struct newtype, e.g.
/// `sensor_id: SensorId("lint")` → `lint`, or a bare quoted string.
fn parse_string_field(dbg: &str, field: &str) -> Option<String> {
    let s = field_slice(dbg, field)?;
    let q = s.find('"')? + 1;
    let rest = &s[q..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse a bare identifier field, e.g. `hook: BeforeTurn` → `BeforeTurn`.
fn parse_ident_field(dbg: &str, field: &str) -> Option<String> {
    let s = field_slice(dbg, field)?;
    let ident: String = s
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if ident.is_empty() {
        None
    } else {
        Some(ident)
    }
}

#[cfg(test)]
mod field_tests {
    use super::*;

    #[test]
    fn scanners_parse_debug_fields() {
        let dbg = "TurnSpan { input_tokens: 100, cache_read_tokens: Some(80), \
                   fired: true, sensor_id: SensorId(\"lint\"), hook: BeforeTurn }";
        assert_eq!(parse_u64_field(dbg, "input_tokens"), Some(100));
        assert_eq!(parse_opt_u64_field(dbg, "cache_read_tokens"), Some(80));
        assert_eq!(parse_bool_field(dbg, "fired"), Some(true));
        assert_eq!(
            parse_string_field(dbg, "sensor_id").as_deref(),
            Some("lint")
        );
        assert_eq!(
            parse_ident_field(dbg, "hook").as_deref(),
            Some("BeforeTurn")
        );
        let none = "X { cache_read_tokens: None }";
        assert_eq!(parse_opt_u64_field(none, "cache_read_tokens"), Some(0));
    }
}
