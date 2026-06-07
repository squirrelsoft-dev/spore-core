""":class:`EvalHarness` — the runner (Rules 14-25) — plus its fluent builder
and the deferred :class:`TraceAnalyzer` interface (Rule 30).

Mirrors ``rust/crates/spore-eval/src/harness.rs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import anyio
from spore_core.harness import (
    AggregateUsage,
    BudgetLimits,
    HaltReasonBudgetExceeded,
    HarnessConfig,
    HarnessRunOptions,
    ReactConfig,
    RunResult,
    RunResultFailure,
    RunResultWaitingForHuman,
    SessionId,
    StandardHarness,
    Task,
    TaskId,
)
from spore_core.observability import ObservabilityProvider, SessionMetrics, Span

from .metric_map import (
    EvalMetric,
    EvalMetricTaskSuccessRate,
    EvalMetricVerificationScore,
    RunSampleInputs,
    metric_direction,
    metric_name,
    sample_for,
)
from .report import (
    ComparisonReport,
    MetricComparison,
    Recommendation,
    classify_direction,
    derive_recommendation,
)
from .stats import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    MetricStats,
    bootstrap_ci,
    welch_t_test,
)
from .task import EvalError, EvalTask, MissingMetricsError, TaskSuite
from .verifier import build_verifier
from .worktree import Workspace

_BASELINE_CONFIG_ID = "baseline"


# ============================================================================
# TraceAnalyzer (Rule 30 — interface only, no built-in impl ships)
# ============================================================================


@dataclass
class HarnessConfigDiff:
    """A proposed change to a :class:`HarnessConfig` produced by a
    :class:`TraceAnalyzer`. Marker stub: the optimization loop is deferred
    (Rule 30)."""

    description: str = ""


@runtime_checkable
class TraceAnalyzer(Protocol):
    """Analyzes failure traces and proposes candidate config diffs (Rule 30).
    Interface only — no built-in implementation ships in the MVP."""

    async def analyze(self, traces: list[Span]) -> list[HarnessConfigDiff]: ...


# ============================================================================
# Per-config sample accumulation
# ============================================================================


@dataclass
class _ConfigSamples:
    per_metric: list[list[float]]
    trace_links: list[str] = field(default_factory=list)
    waiting_for_human: int = 0

    @classmethod
    def new(cls, metrics: list[EvalMetric]) -> _ConfigSamples:
        return cls(per_metric=[[] for _ in metrics])


# ============================================================================
# EvalHarness
# ============================================================================


class EvalHarness:
    """The evaluation harness: runs a task suite against a baseline and
    candidate configs, aggregates metrics, and compares them."""

    def __init__(
        self,
        *,
        task_suite: TaskSuite,
        baseline_config: HarnessConfig,
        candidate_configs: list[tuple[str, HarnessConfig]],
        n_runs_per_config: int,
        metrics: list[EvalMetric],
        observability: ObservabilityProvider,
        bootstrap_iterations: int,
        primary_metric: EvalMetric,
    ) -> None:
        self.task_suite = task_suite
        self.baseline_config = baseline_config
        self.candidate_configs = candidate_configs
        self.n_runs_per_config = n_runs_per_config
        self.metrics = metrics
        self.observability = observability
        self.bootstrap_iterations = bootstrap_iterations
        self.primary_metric = primary_metric

    async def run(self) -> list[ComparisonReport]:
        """Run the full comparison (Rules 14-25). Produces one
        :class:`ComparisonReport` per candidate config."""
        if not self.metrics:
            raise MissingMetricsError("no metrics configured for comparison")

        baseline = await self._run_config(self.baseline_config, _BASELINE_CONFIG_ID)
        reports: list[ComparisonReport] = []
        for config_id, config in self.candidate_configs:
            candidate = await self._run_config(config, config_id)
            reports.append(self._compare(baseline, candidate, config_id))
        return reports

    async def _run_config(self, config: HarnessConfig, config_id: str) -> _ConfigSamples:
        samples = _ConfigSamples.new(self.metrics)
        for _category, task in self.task_suite.all_tasks():
            for run_idx in range(self.n_runs_per_config):
                await self._run_one(config, config_id, task, run_idx, samples)
        return samples

    async def _run_one(
        self,
        config: HarnessConfig,
        config_id: str,
        task: EvalTask,
        run_idx: int,
        samples: _ConfigSamples,
    ) -> None:
        # Rule 2/3: fresh workspace restored from the snapshot, torn down after.
        async with await Workspace.restore(task.workspace_snapshot) as workspace:
            unique = f"{config_id}-{task.id}-{run_idx}"
            session_id = SessionId(unique)
            harness = StandardHarness(config)
            max_turns = task.expected_turns[1] if task.expected_turns else 20
            core_task = Task(
                id=TaskId(unique),
                instruction=task.instruction,
                session_id=session_id,
                budget=BudgetLimits(
                    max_turns=max_turns,
                    max_wall_time=max(task.timeout, 1),
                ),
                loop_strategy=ReactConfig.per_loop(max_turns),
            )

            # Rule 4: timeout bounds a single run and yields a failed run
            # rather than raising — guard the await.
            run_result = await self._run_with_timeout(harness, core_task, session_id, task.timeout)

            # Rule 16: read metrics from observability (do not recompute).
            session_metrics = await self.observability.get_session_metrics(session_id)
            if session_metrics is None:
                session_metrics = _empty_session_metrics(session_id, TaskId(unique))
            trace = await self.observability.get_trace(session_id)

            # Run the verifier (Rules 7-13).
            verifier = build_verifier(task.verifier_spec)
            verification = await verifier.verify(task, run_result, workspace.path)

            # Rule 18: WaitingForHuman counts as neither success nor failure; it
            # is reported separately and excluded from success-rate / score.
            waiting = isinstance(run_result, RunResultWaitingForHuman)
            if waiting:
                samples.waiting_for_human += 1

            inputs = RunSampleInputs(
                verifier_passed=verification.passed,
                verifier_score=verification.score,
            )
            for metric, vec in zip(self.metrics, samples.per_metric, strict=True):
                if waiting and isinstance(
                    metric, (EvalMetricTaskSuccessRate, EvalMetricVerificationScore)
                ):
                    continue
                vec.append(sample_for(metric, session_metrics, trace, inputs))

            # Rule 25: collect trace links for failed or non-passing runs.
            if not verification.passed or isinstance(run_result, RunResultFailure):
                samples.trace_links.append(str(session_id))

    async def _run_with_timeout(
        self,
        harness: StandardHarness,
        core_task: Task,
        session_id: SessionId,
        timeout_secs: int,
    ) -> RunResult:
        budget_failure = RunResultFailure(
            reason=HaltReasonBudgetExceeded(limit_type="wall_time"),
            session_id=session_id,
            usage=AggregateUsage(),
            turns=0,
        )
        deadline = max(timeout_secs, 0.001)
        try:
            with anyio.fail_after(deadline):
                return await harness.run(HarnessRunOptions(core_task))
        except TimeoutError:
            return budget_failure
        except EvalError:
            raise
        except Exception:  # noqa: BLE001 — Rule 4: a crashed run is a failed run
            return budget_failure

    def _compare(
        self,
        baseline: _ConfigSamples,
        candidate: _ConfigSamples,
        config_id: str,
    ) -> ComparisonReport:
        comparisons: list[MetricComparison] = []
        for i, metric in enumerate(self.metrics):
            base = baseline.per_metric[i]
            cand = candidate.per_metric[i]
            base_stats = MetricStats.from_samples(base)  # Rule 19
            cand_stats = MetricStats.from_samples(cand)
            delta = cand_stats.mean - base_stats.mean
            welch = welch_t_test(base, cand)  # Rule 20
            direction = classify_direction(delta, metric_direction(metric))  # Rule 22

            # Rule 21: bootstrap CI for metrics from non-deterministic verifiers.
            ci = (
                bootstrap_ci(cand, self.bootstrap_iterations, 0.95, DEFAULT_BOOTSTRAP_SEED)
                if self._metric_is_non_deterministic(metric)
                else None
            )

            comparisons.append(
                MetricComparison(
                    metric_name=metric_name(metric),
                    baseline=base_stats,
                    candidate=cand_stats,
                    delta=delta,
                    p_value=welch.p_value,
                    ci=ci,
                    direction=direction,
                )
            )

        recommendation: Recommendation = derive_recommendation(
            config_id, comparisons, self.primary_metric
        )
        trace_links = list(candidate.trace_links) + list(baseline.trace_links)
        return ComparisonReport(
            baseline_config_id=_BASELINE_CONFIG_ID,
            candidate_config_id=config_id,
            metrics=comparisons,
            recommendation=recommendation,
            trace_links=trace_links,
        )

    def _metric_is_non_deterministic(self, metric: EvalMetric) -> bool:
        """Whether a metric should carry a bootstrap CI (Rule 21): metrics
        derived from non-deterministic verifiers."""
        if not isinstance(metric, (EvalMetricTaskSuccessRate, EvalMetricVerificationScore)):
            return False
        return any(
            not build_verifier(t.verifier_spec).is_deterministic()
            for _cat, t in self.task_suite.all_tasks()
        )


def _empty_session_metrics(session_id: SessionId, task_id: TaskId) -> SessionMetrics:
    from spore_core.guide_registry import SessionOutcomePartial

    return SessionMetrics(
        session_id=session_id,
        task_id=task_id,
        total_turns=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        total_duration_ms=0,
        tool_calls=0,
        sensor_fires=0,
        sensor_halts=0,
        compactions=0,
        outcome=SessionOutcomePartial(),
    )


# ============================================================================
# EvalHarnessBuilder
# ============================================================================


class EvalHarnessBuilder:
    """Fluent assembler for an :class:`EvalHarness`, mirroring
    ``HarnessBuilder``."""

    def __init__(
        self,
        task_suite: TaskSuite,
        baseline_config: HarnessConfig,
        observability: ObservabilityProvider,
    ) -> None:
        self._task_suite = task_suite
        self._baseline_config = baseline_config
        self._observability = observability
        self._candidate_configs: list[tuple[str, HarnessConfig]] = []
        self._n_runs_per_config = 3
        self._metrics: list[EvalMetric] = [EvalMetricTaskSuccessRate()]
        self._bootstrap_iterations = DEFAULT_BOOTSTRAP_ITERATIONS
        self._primary_metric: EvalMetric = EvalMetricTaskSuccessRate()

    def candidate(self, config_id: str, config: HarnessConfig) -> EvalHarnessBuilder:
        self._candidate_configs.append((config_id, config))
        return self

    def n_runs_per_config(self, n: int) -> EvalHarnessBuilder:
        self._n_runs_per_config = n
        return self

    def metrics(self, metrics: list[EvalMetric]) -> EvalHarnessBuilder:
        self._metrics = metrics
        return self

    def bootstrap_iterations(self, n: int) -> EvalHarnessBuilder:
        self._bootstrap_iterations = n
        return self

    def primary_metric(self, metric: EvalMetric) -> EvalHarnessBuilder:
        self._primary_metric = metric
        return self

    def build(self) -> EvalHarness:
        return EvalHarness(
            task_suite=self._task_suite,
            baseline_config=self._baseline_config,
            candidate_configs=self._candidate_configs,
            n_runs_per_config=self._n_runs_per_config,
            metrics=self._metrics,
            observability=self._observability,
            bootstrap_iterations=self._bootstrap_iterations,
            primary_metric=self._primary_metric,
        )
