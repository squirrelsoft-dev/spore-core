"""Comparison + recommendation types and derivation (Rules 19-25).

Mirrors ``rust/crates/spore-eval/src/report.rs``.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from spore_core.harness import OptimizationDirection

from .metric_map import EvalMetric, metric_name
from .stats import ConfidenceInterval, MetricStats


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# ComparisonDirection
# ============================================================================

ComparisonDirection = Literal["better", "worse", "no_change"]


# ============================================================================
# MetricComparison
# ============================================================================


class MetricComparison(_Model):
    """One metric's baseline-vs-candidate comparison (Rules 19-22)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, arbitrary_types_allowed=True)

    metric_name: str
    baseline: MetricStats
    candidate: MetricStats
    delta: float
    p_value: float
    ci: ConfidenceInterval | None = None
    direction: ComparisonDirection


# ============================================================================
# Recommendation — tagged union on ``kind`` (Rules 23-24)
# ============================================================================


class RecommendationAdopt(_Model):
    kind: Literal["adopt"] = "adopt"
    config_id: str
    confidence: float


class RecommendationReject(_Model):
    kind: Literal["reject"] = "reject"
    reason: str


class RecommendationNeedsMoreRuns(_Model):
    kind: Literal["needs_more_runs"] = "needs_more_runs"
    current_n: int
    recommended_n: int


class RecommendationAmbiguous(_Model):
    kind: Literal["ambiguous"] = "ambiguous"
    tradeoffs: list[str] = Field(default_factory=list)


Recommendation = Annotated[
    RecommendationAdopt
    | RecommendationReject
    | RecommendationNeedsMoreRuns
    | RecommendationAmbiguous,
    Field(discriminator="kind"),
]


# ============================================================================
# ComparisonReport
# ============================================================================


class ComparisonReport(_Model):
    """The full comparison report for one candidate config (Rule 25 includes
    ``trace_links``)."""

    baseline_config_id: str
    candidate_config_id: str
    model_config = ConfigDict(extra="forbid", populate_by_name=True, arbitrary_types_allowed=True)

    metrics: list[MetricComparison] = Field(default_factory=list)
    recommendation: Recommendation
    trace_links: list[str] = Field(default_factory=list)


# ============================================================================
# Derivation
# ============================================================================


def classify_direction(
    delta: float, direction: OptimizationDirection, eps: float = 1e-9
) -> ComparisonDirection:
    """Classify the direction of a delta relative to the metric's optimization
    direction (Rule 22). A delta within ``eps`` of zero is ``no_change``."""
    if abs(delta) <= eps:
        return "no_change"
    if direction == "maximize":
        return "better" if delta > 0.0 else "worse"
    return "better" if delta < 0.0 else "worse"


SIGNIFICANCE_ALPHA = 0.05
"""The significance threshold for "improved" / "regressed" (Rule 23)."""


def derive_recommendation(
    candidate_config_id: str,
    comparisons: list[MetricComparison],
    primary: EvalMetric,
) -> Recommendation:
    """Derive a :data:`Recommendation` from the per-metric comparisons and the
    primary metric (Rules 23-24).

    * Primary improves with p < 0.05            → ``adopt``.
    * Primary clearly worse (worse, p < 0.05)   → ``reject``.
    * Primary inconclusive (p >= 0.05)          → ``needs_more_runs``.
    * Otherwise mixed across metrics            → ``ambiguous``."""
    primary_name = metric_name(primary)
    primary_cmp = next((c for c in comparisons if c.metric_name == primary_name), None)
    if primary_cmp is None:
        return RecommendationAmbiguous(tradeoffs=[f"primary metric {primary_name} not measured"])

    significant = primary_cmp.p_value < SIGNIFICANCE_ALPHA

    if significant and primary_cmp.direction == "better":
        return RecommendationAdopt(
            config_id=candidate_config_id,
            confidence=1.0 - primary_cmp.p_value,
        )
    if significant and primary_cmp.direction == "worse":
        return RecommendationReject(
            reason=(
                f"primary metric {primary_name} regressed "
                f"(delta={primary_cmp.delta:.4f}, p={primary_cmp.p_value:.4f})"
            )
        )

    # Inconclusive on the primary metric.
    mixed = [
        f"{c.metric_name}: {c.direction} (p={c.p_value:.3f})"
        for c in comparisons
        if c.direction != "no_change"
    ]
    any_better = any(c.direction == "better" for c in comparisons)
    any_worse = any(c.direction == "worse" for c in comparisons)
    if any_better and any_worse:
        return RecommendationAmbiguous(tradeoffs=mixed)
    return RecommendationNeedsMoreRuns(
        current_n=primary_cmp.candidate.n,
        recommended_n=recommended_n(primary_cmp),
    )


def recommended_n(cmp: MetricComparison) -> int:
    """Power-based estimate of the runs needed to detect the observed effect at
    alpha=0.05, power~=0.8 (Rule 24): ``n ~= 16 sigma^2 / delta^2``. Pooled
    variance from baseline and candidate; clamped above the current n."""
    pooled_var = (cmp.baseline.stddev**2 + cmp.candidate.stddev**2) / 2.0
    delta = abs(cmp.delta)
    current = max(cmp.candidate.n, 1)
    if delta <= 2.220446049250313e-16 or pooled_var <= 0.0:
        return max(current * 2, current + 1)
    est = math.ceil(16.0 * pooled_var / (delta * delta))
    return max(est, current + 1)
