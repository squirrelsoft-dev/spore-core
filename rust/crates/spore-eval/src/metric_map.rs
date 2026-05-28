//! [`EvalMetric`] enum + extraction of metric samples from observability
//! (Rule 16, Rule 17, Resolution 1).
//!
//! Simple aggregates (turns, cost, wall-time, success) come from
//! [`SessionMetrics`] via `get_session_metrics`. Filtered metrics
//! (`CacheHitRate{block}`, `SensorFireRate{sensor_id}`,
//! `MiddlewareInterventionRate{hook}`) are computed from `get_trace` spans
//! (Resolution 1) via the typed `Span::as_turn`/`as_sensor`/`as_middleware`
//! accessors (issue #68) — the same clean typed read used by the
//! TypeScript, Python, and Go EvalHarnesses.

use serde::{Deserialize, Serialize};
use spore_core::harness::OptimizationDirection;
use spore_core::middleware::MiddlewareDecision;
use spore_core::observability::{SessionMetrics, Span};

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

/// Cache-read hit rate from turn spans: cache_read_tokens / (input + cache_read)
/// (Resolution 1, `TurnSpan.cache_read_tokens`). 0.0 with no tokens.
fn cache_hit_rate(spans: &[Box<dyn Span>]) -> f64 {
    let mut cache_read = 0u64;
    let mut input = 0u64;
    for s in spans {
        if let Some(t) = s.as_turn() {
            cache_read += u64::from(t.cache_read_tokens.unwrap_or(0));
            input += u64::from(t.input_tokens);
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
        if let Some(ss) = s.as_sensor() {
            if ss.sensor_id.as_str() == sensor_id {
                total += 1;
                if ss.fired {
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
///
/// Hook names are matched case-insensitively against the `HookPoint` variant
/// name (e.g. target `"beforeturn"` matches `HookPoint::BeforeTurn`). A decision
/// counts as an intervention when it is neither `Continue` nor
/// `ContinueWithModification` — i.e. `ForceAnotherTurn`, `Halt`, or
/// `SurfaceToHuman`. This preserves the prior `!debug.contains("decision: Continue")`
/// semantics byte-for-byte (the `ContinueWithModification` Debug form shares the
/// `Continue` prefix, so the old scanner classified it as non-intervening).
fn middleware_intervention_rate(spans: &[Box<dyn Span>], hook: &str) -> f64 {
    let mut interventions = 0u32;
    let mut total = 0u32;
    let target = hook.to_lowercase();
    for s in spans {
        if let Some(ms) = s.as_middleware() {
            if format!("{:?}", ms.hook).to_lowercase() == target {
                total += 1;
                if !matches!(
                    ms.decision,
                    MiddlewareDecision::Continue | MiddlewareDecision::ContinueWithModification
                ) {
                    interventions += 1;
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

#[cfg(test)]
mod typed_access_tests {
    use super::*;
    use spore_core::harness::HumanRequest;
    use spore_core::harness::{SessionId, TaskId};
    use spore_core::memory::Timestamp;
    use spore_core::middleware::{HookPoint, MiddlewareDecision};
    use spore_core::model::StopReason;
    use spore_core::observability::{
        MiddlewareSpan, SensorSpan, Span, SpanBase, SpanId, SpanKind, SpanStatus, TurnSpan,
    };
    use spore_core::sensor::{SensorId, SensorKind, SensorOutcome, SensorTrigger};

    fn base(kind: SpanKind) -> SpanBase {
        SpanBase {
            span_id: SpanId::new("sp"),
            parent_span_id: None,
            session_id: SessionId::new("s1"),
            task_id: TaskId::new("t1"),
            kind,
            started_at: Timestamp::new("2026-05-16T00:00:00Z"),
            ended_at: Timestamp::new("2026-05-16T00:00:00Z"),
            duration_ms: 0,
            status: SpanStatus::Ok,
        }
    }

    fn turn(input: u32, cache_read: Option<u32>) -> TurnSpan {
        TurnSpan {
            base: base(SpanKind::Turn),
            turn_number: 1,
            input_tokens: input,
            output_tokens: 0,
            cache_read_tokens: cache_read,
            cache_write_tokens: None,
            cost_usd: 0.0,
            stop_reason: StopReason::EndTurn,
            tool_calls_requested: 0,
            output_text: None,
            tool_calls: None,
            input_messages: None,
        }
    }

    fn sensor(id: &str, fired: bool) -> SensorSpan {
        SensorSpan {
            base: base(SpanKind::SensorEvaluation),
            sensor_id: SensorId::new(id),
            sensor_kind: SensorKind::Computational,
            trigger: SensorTrigger::PostTurn,
            outcome: if fired {
                SensorOutcome::Warn
            } else {
                SensorOutcome::Pass
            },
            fired,
        }
    }

    fn middleware(hook: HookPoint, decision: MiddlewareDecision) -> MiddlewareSpan {
        MiddlewareSpan {
            base: base(SpanKind::MiddlewareHook),
            hook,
            decision,
        }
    }

    #[test]
    fn typed_accessors_return_some_only_for_matching_kind() {
        let t: Box<dyn Span> = Box::new(turn(100, Some(80)));
        assert!(t.as_turn().is_some());
        assert!(t.as_sensor().is_none());
        assert!(t.as_middleware().is_none());

        let s: Box<dyn Span> = Box::new(sensor("lint", true));
        assert!(s.as_sensor().is_some());
        assert!(s.as_turn().is_none());
        assert!(s.as_middleware().is_none());

        let m: Box<dyn Span> = Box::new(middleware(
            HookPoint::BeforeTurn,
            MiddlewareDecision::Continue,
        ));
        assert!(m.as_middleware().is_some());
        assert!(m.as_turn().is_none());
        assert!(m.as_sensor().is_none());
    }

    #[test]
    fn cache_hit_rate_typed() {
        // cache_read = 80 + 0; input = 100 + 200 = 300; rate = 80 / 380.
        let spans: Vec<Box<dyn Span>> = vec![
            Box::new(turn(100, Some(80))),
            Box::new(turn(200, None)),
            Box::new(sensor("lint", true)), // ignored
        ];
        assert!((cache_hit_rate(&spans) - 80.0 / 380.0).abs() < 1e-12);
        // Empty / no turn spans → 0.0.
        assert_eq!(cache_hit_rate(&[]), 0.0);
        let only_sensor: Vec<Box<dyn Span>> = vec![Box::new(sensor("lint", true))];
        assert_eq!(cache_hit_rate(&only_sensor), 0.0);
    }

    #[test]
    fn sensor_fire_rate_typed() {
        let spans: Vec<Box<dyn Span>> = vec![
            Box::new(sensor("lint", true)),
            Box::new(sensor("lint", false)),
            Box::new(sensor("other", true)), // non-matching id
            Box::new(turn(1, None)),         // ignored
        ];
        // 1 fired of 2 matching "lint".
        assert!((sensor_fire_rate(&spans, "lint") - 0.5).abs() < 1e-12);
        // No match → 0.0.
        assert_eq!(sensor_fire_rate(&spans, "missing"), 0.0);
        assert_eq!(sensor_fire_rate(&[], "lint"), 0.0);
    }

    #[test]
    fn middleware_intervention_rate_typed() {
        let spans: Vec<Box<dyn Span>> = vec![
            Box::new(middleware(
                HookPoint::BeforeTurn,
                MiddlewareDecision::Continue,
            )),
            Box::new(middleware(
                HookPoint::BeforeTurn,
                MiddlewareDecision::ContinueWithModification,
            )),
            Box::new(middleware(
                HookPoint::BeforeTurn,
                MiddlewareDecision::Halt {
                    reason: "stop".into(),
                },
            )),
            // non-matching hook
            Box::new(middleware(
                HookPoint::AfterTool,
                MiddlewareDecision::Continue,
            )),
        ];
        // 3 matching "beforeturn" (case-insensitive); only Halt intervenes → 1/3.
        assert!((middleware_intervention_rate(&spans, "beforeturn") - 1.0 / 3.0).abs() < 1e-12);
        // ContinueWithModification is NOT an intervention (prior behavior).
        let cwm: Vec<Box<dyn Span>> = vec![Box::new(middleware(
            HookPoint::BeforeTool,
            MiddlewareDecision::ContinueWithModification,
        ))];
        assert_eq!(middleware_intervention_rate(&cwm, "BeforeTool"), 0.0);
        // SurfaceToHuman counts as an intervention.
        let sth: Vec<Box<dyn Span>> = vec![Box::new(middleware(
            HookPoint::BeforeTool,
            MiddlewareDecision::SurfaceToHuman {
                request: HumanRequest::Clarification {
                    question: "why".into(),
                },
            },
        ))];
        assert_eq!(middleware_intervention_rate(&sth, "beforetool"), 1.0);
        // No match → 0.0.
        assert_eq!(middleware_intervention_rate(&spans, "aftersession"), 0.0);
        assert_eq!(middleware_intervention_rate(&[], "beforeturn"), 0.0);
    }
}
