""":data:`EvalMetric` tagged union + extraction of metric samples from
observability (Rule 16, Rule 17, Resolution 1).

Mirrors ``rust/crates/spore-eval/src/metric_map.rs`` in shape and rules, but
NOT in implementation: the simple aggregates (turns, cost, wall-time, success)
come from :class:`SessionMetrics` via ``get_session_metrics``, and the filtered
metrics (``CacheHitRate{block}``, ``SensorFireRate{sensor_id}``,
``MiddlewareInterventionRate{hook}``) are computed from ``get_trace`` spans by
reading the **typed span attributes directly** (Resolution 1 NOTE) — Python
does not replicate the Rust Debug-string-parsing workaround.

Issue #12's public surface is NOT modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from spore_core.harness import OptimizationDirection
from spore_core.middleware import MiddlewareContinue
from spore_core.observability import (
    MiddlewareSpan,
    SensorSpan,
    SessionMetrics,
    Span,
    TurnSpan,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# EvalMetric — tagged union on ``kind`` (Rule 17)
# ============================================================================


class EvalMetricTaskSuccessRate(_Model):
    kind: Literal["task_success_rate"] = "task_success_rate"


class EvalMetricMeanTurnsToCompletion(_Model):
    kind: Literal["mean_turns_to_completion"] = "mean_turns_to_completion"


class EvalMetricMeanCostUsd(_Model):
    kind: Literal["mean_cost_usd"] = "mean_cost_usd"


class EvalMetricMeanWallTime(_Model):
    kind: Literal["mean_wall_time"] = "mean_wall_time"


class EvalMetricCacheHitRate(_Model):
    kind: Literal["cache_hit_rate"] = "cache_hit_rate"
    block: str


class EvalMetricSensorFireRate(_Model):
    kind: Literal["sensor_fire_rate"] = "sensor_fire_rate"
    sensor_id: str


class EvalMetricMiddlewareInterventionRate(_Model):
    kind: Literal["middleware_intervention_rate"] = "middleware_intervention_rate"
    hook: str


class EvalMetricVerificationScore(_Model):
    kind: Literal["verification_score"] = "verification_score"


EvalMetric = Annotated[
    EvalMetricTaskSuccessRate
    | EvalMetricMeanTurnsToCompletion
    | EvalMetricMeanCostUsd
    | EvalMetricMeanWallTime
    | EvalMetricCacheHitRate
    | EvalMetricSensorFireRate
    | EvalMetricMiddlewareInterventionRate
    | EvalMetricVerificationScore,
    Field(discriminator="kind"),
]

_MAXIMIZE: set[str] = {"task_success_rate", "cache_hit_rate", "verification_score"}


def metric_direction(metric: EvalMetric) -> OptimizationDirection:
    """The optimization direction: higher-is-better metrics maximize; cost,
    turns, wall-time, sensor-fire and intervention rates minimize."""
    return "maximize" if metric.kind in _MAXIMIZE else "minimize"


def metric_name(metric: EvalMetric) -> str:
    """A stable display/serialization name."""
    if isinstance(metric, EvalMetricCacheHitRate):
        return f"cache_hit_rate[{metric.block}]"
    if isinstance(metric, EvalMetricSensorFireRate):
        return f"sensor_fire_rate[{metric.sensor_id}]"
    if isinstance(metric, EvalMetricMiddlewareInterventionRate):
        return f"middleware_intervention_rate[{metric.hook}]"
    return metric.kind


def metric_from_trace(metric: EvalMetric) -> bool:
    """Whether this metric is sourced from filtered trace spans (Resolution 1)
    rather than from the simple :class:`SessionMetrics` aggregate."""
    return isinstance(
        metric,
        (
            EvalMetricCacheHitRate,
            EvalMetricSensorFireRate,
            EvalMetricMiddlewareInterventionRate,
        ),
    )


# ============================================================================
# RunSampleInputs + sample_for
# ============================================================================


@dataclass
class RunSampleInputs:
    """Per-run inputs needed to derive a metric sample that are not in
    observability: the verifier outcome."""

    verifier_passed: bool
    verifier_score: float


def sample_for(
    metric: EvalMetric,
    session: SessionMetrics,
    spans: list[Span],
    inputs: RunSampleInputs,
) -> float:
    """Compute the sample value of ``metric`` for a single run, given that
    run's :class:`SessionMetrics` (simple aggregates), the run's trace
    ``spans`` (filtered metrics, Resolution 1), and the verifier outcome."""
    if isinstance(metric, EvalMetricTaskSuccessRate):
        return 1.0 if inputs.verifier_passed else 0.0
    if isinstance(metric, EvalMetricVerificationScore):
        return inputs.verifier_score
    if isinstance(metric, EvalMetricMeanTurnsToCompletion):
        return float(session.total_turns)
    if isinstance(metric, EvalMetricMeanCostUsd):
        return session.total_cost_usd
    if isinstance(metric, EvalMetricMeanWallTime):
        return float(session.total_duration_ms)
    if isinstance(metric, EvalMetricCacheHitRate):
        return _cache_hit_rate(spans)
    if isinstance(metric, EvalMetricSensorFireRate):
        return _sensor_fire_rate(spans, metric.sensor_id)
    if isinstance(metric, EvalMetricMiddlewareInterventionRate):
        return _middleware_intervention_rate(spans, metric.hook)
    raise AssertionError(f"unhandled metric {metric!r}")  # pragma: no cover


# ---- Resolution 1: read typed span attributes directly (no Debug parsing) ---


def _cache_hit_rate(spans: list[Span]) -> float:
    """Cache-read hit rate from turn spans:
    ``cache_read / (input + cache_read)``. 0.0 with no tokens."""
    cache_read = 0
    input_tokens = 0
    for s in spans:
        if isinstance(s, TurnSpan):
            cache_read += s.cache_read_tokens or 0
            input_tokens += s.input_tokens
    denom = input_tokens + cache_read
    return cache_read / denom if denom else 0.0


def _sensor_fire_rate(spans: list[Span], sensor_id: str) -> float:
    """Fire rate of a named sensor: fired / total evaluations of that sensor.
    0.0 if never evaluated."""
    fired = 0
    total = 0
    for s in spans:
        if isinstance(s, SensorSpan) and str(s.sensor_id) == sensor_id:
            total += 1
            if s.fired:
                fired += 1
    return fired / total if total else 0.0


def _middleware_intervention_rate(spans: list[Span], hook: str) -> float:
    """Intervention rate of a named middleware hook: non-Continue decisions /
    total firings at that hook."""
    interventions = 0
    total = 0
    for s in spans:
        if isinstance(s, MiddlewareSpan) and _hook_name(s.hook).lower() == hook.lower():
            total += 1
            if not isinstance(s.decision, MiddlewareContinue):
                interventions += 1
    return interventions / total if total else 0.0


def _hook_name(hook: object) -> str:
    """The hook's stable string name (``HookPoint`` is a ``str`` enum)."""
    value = getattr(hook, "value", None)
    return value if isinstance(value, str) else str(hook)
