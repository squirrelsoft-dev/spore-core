"""spore_eval: the EvalHarness — outer ring of the improvement flywheel.

Runs regression / challenge / canary task suites against the harness
implemented in ``spore_core`` (issues #1–#13). See the Improvement
Flywheel section of ``docs/harness-engineering-concepts.md``.

The EvalHarness eval infrastructure (issue #26) lives in
:mod:`spore_eval.eval`; the #57 end-to-end scenario harness lives in
:mod:`spore_eval.e2e_agent` / :mod:`spore_eval.scenarios`. They coexist.
"""

from __future__ import annotations

from .eval import (
    ComparisonReport,
    EvalHarness,
    EvalHarnessBuilder,
    EvalMetric,
    EvalMetricTaskSuccessRate,
    MetricStats,
    Recommendation,
    TaskSuite,
    VerificationResult,
    bootstrap_ci,
    load_suite_path,
    load_suite_str,
    promote_challenge_task,
    suite_to_json,
    welch_t_test,
)

__all__ = [
    "ComparisonReport",
    "EvalHarness",
    "EvalHarnessBuilder",
    "EvalMetric",
    "EvalMetricTaskSuccessRate",
    "MetricStats",
    "Recommendation",
    "TaskSuite",
    "VerificationResult",
    "bootstrap_ci",
    "load_suite_path",
    "load_suite_str",
    "promote_challenge_task",
    "suite_to_json",
    "welch_t_test",
]
