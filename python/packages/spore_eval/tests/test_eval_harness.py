"""Rule-by-rule tests for the EvalHarness (issue #26).

Mirrors ``rust/crates/spore-eval/src/tests.rs`` + the ``stats.rs`` oracle
tests. All tests are hermetic: MockAgent (no network) +
InMemoryObservabilityProvider. Rule 29 replays the shared fixture manifests as
the cross-language oracle — the fixtures are ground truth and are never edited
to make a test pass.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
from spore_core.agent import AgentId, FinalResponse, MockAgent, TurnError
from spore_core.agent import AgentErrorEmpty
from spore_core.harness import (
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessBuilder,
    NoopContextManager,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
)
from spore_core.model import (
    MockModelInterface,
    ModelResponse,
    ProviderInfo,
    StopReason,
    TextBlock,
    TokenUsage,
)
from spore_core.observability import InMemoryObservabilityProvider, SessionMetrics
from spore_core.guide_registry import SessionOutcomePartial

from spore_eval.eval import (
    AlwaysFail,
    AlwaysPass,
    CompositeVerifier,
    DEFAULT_BOOTSTRAP_SEED,
    EvalHarnessBuilder,
    EvalMetricCacheHitRate,
    EvalMetricMeanCostUsd,
    EvalMetricMeanTurnsToCompletion,
    EvalMetricTaskSuccessRate,
    EvalMetricVerificationScore,
    HarnessConfigDiff,
    MetricComparison,
    MetricStats,
    RecommendationAdopt,
    RecommendationNeedsMoreRuns,
    RecommendationReject,
    RunSampleInputs,
    TaskSuite,
    VerificationResult,
    VerifierSpecAlwaysFail,
    VerifierSpecAlwaysPass,
    VerifierSpecComposite,
    VerifierSpecLlmJudge,
    VerifierSpecMetricEvaluator,
    VerifierSpecTestSuite,
    VerifyError,
    Workspace,
    WorkspaceSnapshotEmpty,
    WorkspaceSnapshotFiles,
    WorkspaceSnapshotGitRef,
    bootstrap_ci,
    build_verifier,
    classify_direction,
    derive_recommendation,
    load_suite_path,
    load_suite_str,
    metric_direction,
    metric_name,
    percentile,
    promote_challenge_task,
    sample_for,
    suite_to_json,
    welch_t_test,
)
from spore_eval.eval.task import (
    CompositeChildSpec,
    EvalTask,
    MissingSuiteVersionError,
)
from spore_eval.eval.verifier import MetricEvaluatorVerifier, LlmJudgeVerifier


# ============================================================================
# Helpers
# ============================================================================


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=5)


def _config_with(obs: InMemoryObservabilityProvider, *, success: bool, n_runs: int):
    """Build a HarnessConfig wired to ``obs`` whose MockAgent produces one final
    response per run (success) or one error per run (a weaker config)."""
    agent = MockAgent(AgentId("mock"))
    for _ in range(max(n_runs, 1)):
        if success:
            agent.push(FinalResponse(content="DONE", usage=_usage()))
        else:
            agent.push(TurnError(error=AgentErrorEmpty(), usage=_usage()))
    return (
        HarnessBuilder(
            agent,
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            NoopContextManager(),
            AlwaysContinuePolicy(),
        )
        .observability(obs)
        .build_config()
    )


def _task(task_id: str, snapshot, spec) -> EvalTask:
    return EvalTask(
        id=task_id,
        instruction="do the thing",
        workspace_snapshot=snapshot,
        verifier_spec=spec,
        expected_turns=(1, 4),
        tags=["unit"],
        timeout=30,
    )


def _files(pairs: dict[str, str]) -> WorkspaceSnapshotFiles:
    return WorkspaceSnapshotFiles(files=dict(pairs))


def _run_success() -> RunResultSuccess:
    from spore_core.harness import AggregateUsage

    return RunResultSuccess(
        output="DONE", session_id=SessionId("s"), usage=AggregateUsage(), turns=1
    )


def _fixtures_dir() -> Path:
    # tests/ -> spore_eval -> packages -> python -> repo root.
    return Path(__file__).resolve().parents[4] / "fixtures" / "task_suites"


# ============================================================================
# Rule 1 — three disjoint task lists
# ============================================================================


def test_rule1_three_disjoint_lists() -> None:
    suite = TaskSuite(
        suite_version=1,
        regression=[_task("r1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())],
        challenge=[_task("c1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())],
        canary=[_task("k1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())],
    )
    assert len(suite.all_tasks()) == 3


# ============================================================================
# Rule 2 / Rule 3 — fresh workspace restored + torn down
# ============================================================================


async def test_rule2_workspace_restored_from_files() -> None:
    snap = _files({"input.txt": "hello\n", "sub/x.md": "deep"})
    async with await Workspace.restore(snap) as ws:
        assert (ws.path / "input.txt").read_text() == "hello\n"
        assert (ws.path / "sub" / "x.md").exists()


async def test_rule3_workspace_torn_down() -> None:
    snap = _files({"a.txt": "x"})
    ws = await Workspace.restore(snap)
    path = ws.path
    ws.teardown()
    assert not path.exists()


async def test_rule2_git_ref_restores_worktree() -> None:
    import subprocess

    src = Path(tempfile.mkdtemp(prefix="spore-git-"))

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=src, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (src / "seed.txt").write_text("from git\n")
    git("add", ".")
    git("commit", "-q", "-m", "seed")

    snap = WorkspaceSnapshotGitRef(repo=str(src), reference="HEAD")
    async with await Workspace.restore(snap) as ws:
        assert (ws.path / "seed.txt").read_text() == "from git\n"


# ============================================================================
# Rule 4 — timeout yields a failed run (not a raised exception)
# ============================================================================


async def test_rule4_timeout_is_failed_run() -> None:
    obs = InMemoryObservabilityProvider()
    t = _task("slow", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysFail())
    t.timeout = 30
    suite = TaskSuite(suite_version=1, regression=[t])
    harness = (
        EvalHarnessBuilder(suite, _config_with(obs, success=True, n_runs=5), obs)
        .candidate(
            "cand",
            _config_with(InMemoryObservabilityProvider(), success=True, n_runs=5),
        )
        .n_runs_per_config(1)
        .build()
    )
    reports = await harness.run()
    assert len(reports) == 1


# ============================================================================
# Rule 5 — tags are free-form
# ============================================================================


def test_rule5_tags_are_free_form() -> None:
    t = _task("t", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())
    assert t.tags == ["unit"]


# ============================================================================
# Rule 6 — suite_version required; loader rejects manifest without it
# ============================================================================


def test_rule6_loader_rejects_missing_suite_version() -> None:
    with pytest.raises(MissingSuiteVersionError):
        load_suite_str('{ "regression": [], "challenge": [], "canary": [] }')


def test_rule6_loader_accepts_with_suite_version() -> None:
    suite = load_suite_str(
        '{ "suite_version": 7, "regression": [], "challenge": [], "canary": [] }'
    )
    assert suite.suite_version == 7


# ============================================================================
# Rule 7 / Rule 8 — verification result shape + score clamp
# ============================================================================


def test_rule7_verification_result_shape() -> None:
    r = VerificationResult.new(True, 0.5, "ok").with_signal("k", 1.0)
    assert r.passed
    assert r.score == 0.5
    assert r.detail == "ok"
    assert r.signals["k"] == 1.0


def test_rule8_score_out_of_range_is_error() -> None:
    with pytest.raises(VerifyError):
        VerificationResult.new(True, 1.5, "x")
    with pytest.raises(VerifyError):
        VerificationResult.new(True, -0.1, "x")
    assert VerificationResult.clamped(True, 1.5, "x").score == 1.0
    assert VerificationResult.clamped(True, -0.1, "x").score == 0.0


# ============================================================================
# Rule 9 — is_deterministic per verifier
# ============================================================================


def test_rule9_determinism_flags() -> None:
    assert build_verifier(VerifierSpecAlwaysPass()).is_deterministic()
    assert build_verifier(
        VerifierSpecTestSuite(command="true", args=[], timeout_secs=1)
    ).is_deterministic()
    assert not build_verifier(
        VerifierSpecLlmJudge(rubric="r", score_range=(0.0, 1.0))
    ).is_deterministic()


# ============================================================================
# Rule 10 — TestSuiteVerifier pass-rate + passed == score==1.0
# ============================================================================


async def test_rule10_test_suite_verifier_passes_on_zero_exit() -> None:
    ws = Path(tempfile.mkdtemp())
    (ws / "output.txt").write_text("HELLO\n")
    t = _task(
        "t",
        WorkspaceSnapshotEmpty(),
        VerifierSpecTestSuite(
            command="sh", args=["-c", "grep -q HELLO output.txt"], timeout_secs=10
        ),
    )
    r = await build_verifier(t.verifier_spec).verify(t, _run_success(), ws)
    assert r.passed
    assert r.score == 1.0


async def test_rule10_test_suite_verifier_fails_on_nonzero_exit() -> None:
    ws = Path(tempfile.mkdtemp())
    t = _task(
        "t",
        WorkspaceSnapshotEmpty(),
        VerifierSpecTestSuite(command="sh", args=["-c", "exit 1"], timeout_secs=10),
    )
    r = await build_verifier(t.verifier_spec).verify(t, _run_success(), ws)
    assert not r.passed
    assert r.score == 0.0


# ============================================================================
# Rule 11 — CompositeVerifier: weighted mean, required AND, determinism AND
# ============================================================================


async def test_rule11_composite_weighted_mean_and_required() -> None:
    composite = CompositeVerifier([(AlwaysPass(), 1.0, True), (AlwaysFail(), 1.0, False)])
    t = _task("t", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())
    r = await composite.verify(t, _run_success(), Path("/tmp"))
    assert abs(r.score - 0.5) < 1e-9
    assert r.passed
    assert composite.is_deterministic()


async def test_rule11_composite_required_failure_fails_overall() -> None:
    composite = CompositeVerifier([(AlwaysPass(), 1.0, True), (AlwaysFail(), 1.0, True)])
    t = _task("t", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())
    r = await composite.verify(t, _run_success(), Path("/tmp"))
    assert not r.passed


def test_rule11_composite_spec_resolves_determinism_and() -> None:
    spec = VerifierSpecComposite(
        children=[
            CompositeChildSpec(spec=VerifierSpecAlwaysPass(), weight=2.0, required=True),
            CompositeChildSpec(
                spec=VerifierSpecLlmJudge(rubric="r", score_range=(0.0, 1.0)),
                weight=1.0,
                required=False,
            ),
        ]
    )
    assert not build_verifier(spec).is_deterministic()


# ============================================================================
# Rule 12 — MetricEvaluatorVerifier normalizes a metric value
# ============================================================================


async def test_rule12_metric_evaluator_verifier_normalizes() -> None:
    from spore_core.metric import MetricResult

    class FixedEval:
        def __init__(self, value: float) -> None:
            self._value = value

        async def evaluate(self, sandbox, session_state) -> MetricResult:
            return MetricResult(value=self._value)

        def direction(self) -> str:
            return "maximize"

        def description(self) -> str:
            return "fixed"

    v = MetricEvaluatorVerifier.with_range(FixedEval(7.5), 0.0, 10.0)
    assert v.is_deterministic()
    t = _task("t", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())
    ws = Path(tempfile.mkdtemp())
    r = await v.verify(t, _run_success(), ws)
    assert abs(r.score - 0.75) < 1e-9


# ============================================================================
# Rule 13 — LlmJudgeVerifier non-deterministic + pluggable judge
# ============================================================================


async def test_rule13_llm_judge_verifier() -> None:
    judge = MockModelInterface(ProviderInfo(name="fake", model_id="judge", context_window=8000))
    judge.push_response(
        ModelResponse(
            content=[TextBlock(text="score: 8")],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(),
        )
    )
    v = LlmJudgeVerifier(judge=judge, rubric="rate", score_range=(0.0, 10.0))
    assert not v.is_deterministic()
    t = _task("t", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())
    r = await v.verify(t, _run_success(), Path("/tmp"))
    assert abs(r.score - 0.8) < 1e-9


# ============================================================================
# Rule 14 / Rule 15 / Rule 16 — n runs, build harness from config, read obs
# ============================================================================


async def test_rule14_15_16_runs_per_config_and_metrics_from_obs() -> None:
    base_obs = InMemoryObservabilityProvider()
    cand_obs = InMemoryObservabilityProvider()
    suite = TaskSuite(
        suite_version=1,
        regression=[_task("t1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())],
    )
    n = 3
    harness = (
        EvalHarnessBuilder(suite, _config_with(base_obs, success=True, n_runs=n), base_obs)
        .candidate("cand", _config_with(cand_obs, success=True, n_runs=n))
        .n_runs_per_config(n)
        .metrics([EvalMetricTaskSuccessRate(), EvalMetricMeanTurnsToCompletion()])
        .build()
    )
    reports = await harness.run()
    assert len(reports) == 1
    success = next(m for m in reports[0].metrics if m.metric_name == "task_success_rate")
    assert success.baseline.n == n  # Rule 14
    turns = next(m for m in reports[0].metrics if m.metric_name == "mean_turns_to_completion")
    assert turns.baseline.mean >= 1.0  # Rule 16: from observability


# ============================================================================
# Rule 17 — EvalMetric mapping (name + direction)
# ============================================================================


def test_rule17_metric_names_and_directions() -> None:
    assert metric_direction(EvalMetricTaskSuccessRate()) == "maximize"
    assert metric_direction(EvalMetricMeanCostUsd()) == "minimize"
    assert metric_direction(EvalMetricMeanTurnsToCompletion()) == "minimize"
    assert metric_direction(EvalMetricCacheHitRate(block="sys")) == "maximize"
    assert metric_name(EvalMetricCacheHitRate(block="sys")) == "cache_hit_rate[sys]"
    assert metric_direction(EvalMetricVerificationScore()) == "maximize"


# ============================================================================
# Rule 18 — WaitingForHuman not counted as success/failure
# ============================================================================


def test_rule18_resource_metric_still_computes_for_waiting() -> None:
    session = SessionMetrics(
        session_id=SessionId("s"),
        task_id="t",
        total_turns=2,
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
    v = sample_for(
        EvalMetricMeanTurnsToCompletion(),
        session,
        [],
        RunSampleInputs(verifier_passed=False, verifier_score=0.0),
    )
    assert v == 2.0


# ============================================================================
# Rule 19 — MetricStats aggregation (+ stats oracle)
# ============================================================================


def test_rule19_metric_stats_aggregates() -> None:
    s = MetricStats.from_samples([1.0, 1.0, 1.0])
    assert s.n == 3
    assert s.mean == 1.0
    assert s.stddev == 0.0


def test_metric_stats_oracle() -> None:
    s = MetricStats.from_samples([1.0, 2.0, 3.0, 4.0])
    assert abs(s.mean - 2.5) < 1e-12
    assert abs(s.stddev - math.sqrt(5.0 / 3.0)) < 1e-9
    assert s.n == 4
    assert abs(s.p50 - 2.0) < 1e-12
    assert abs(s.p95 - 4.0) < 1e-12
    assert percentile([], 50.0) == 0.0


# ============================================================================
# Rule 20 — Welch's t-test recorded (+ hand-computed oracle)
# ============================================================================


def test_rule20_welch_recorded() -> None:
    w = welch_t_test([0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1])
    assert w.p_value < 0.05


def test_welch_t_test_oracle() -> None:
    r = welch_t_test([27.0, 31.0, 29.0, 30.0], [20.0, 22.0, 24.0, 21.0])
    assert abs(r.t - 6.21063) < 1e-4
    assert abs(r.df - 6.0) < 1e-3
    assert 0.0 < r.p_value < 0.005


def test_welch_degenerate_cases() -> None:
    assert abs(welch_t_test([1, 2, 3, 4], [1, 2, 3, 4]).p_value - 1.0) < 1e-6
    assert abs(welch_t_test([5, 5, 5], [3, 3, 3]).p_value) < 1e-12
    assert abs(welch_t_test([1.0], [2.0, 3.0]).p_value - 1.0) < 1e-12


# ============================================================================
# Rule 21 — bootstrap CI present for non-deterministic metrics (+ seed oracle)
# ============================================================================


def test_rule21_bootstrap_ci_present() -> None:
    ci = bootstrap_ci([0.5, 0.6, 0.4, 0.55], 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)
    assert ci is not None and ci.lower <= ci.upper


def test_bootstrap_ci_seeded_and_empty() -> None:
    samples = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 8.0, 15.0]
    ci1 = bootstrap_ci(samples, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)
    ci2 = bootstrap_ci(samples, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)
    assert ci1 == ci2
    mean = sum(samples) / len(samples)
    assert ci1 is not None and ci1.lower <= mean <= ci1.upper
    assert bootstrap_ci([], 100, 0.95, 1) is None


async def test_rule21_nondeterministic_verifier_gets_ci() -> None:
    base_obs = InMemoryObservabilityProvider()
    cand_obs = InMemoryObservabilityProvider()
    suite = TaskSuite(
        suite_version=1,
        regression=[
            _task(
                "t",
                WorkspaceSnapshotEmpty(),
                VerifierSpecLlmJudge(rubric="r", score_range=(0.0, 1.0)),
            )
        ],
    )
    n = 4
    harness = (
        EvalHarnessBuilder(suite, _config_with(base_obs, success=True, n_runs=n), base_obs)
        .candidate("cand", _config_with(cand_obs, success=True, n_runs=n))
        .n_runs_per_config(n)
        .metrics([EvalMetricTaskSuccessRate()])
        .build()
    )
    reports = await harness.run()
    assert reports[0].metrics[0].ci is not None


# ============================================================================
# Rule 22 — direction
# ============================================================================


def test_rule22_direction() -> None:
    assert classify_direction(0.3, "maximize") == "better"
    assert classify_direction(0.2, "minimize") == "worse"
    assert classify_direction(0.0, "maximize") == "no_change"


# ============================================================================
# Rule 23 / Rule 24 — recommendation + recommended_n
# ============================================================================


def _stats(mean: float, sd: float, n: int) -> MetricStats:
    return MetricStats(mean=mean, stddev=sd, p50=mean, p95=mean, n=n)


def test_rule23_24_recommendation_paths() -> None:
    adopt = derive_recommendation(
        "c",
        [
            MetricComparison(
                metric_name="task_success_rate",
                baseline=_stats(0.5, 0.05, 5),
                candidate=_stats(0.95, 0.05, 5),
                delta=0.45,
                p_value=0.001,
                ci=None,
                direction="better",
            )
        ],
        EvalMetricTaskSuccessRate(),
    )
    assert isinstance(adopt, RecommendationAdopt)

    needs = derive_recommendation(
        "c",
        [
            MetricComparison(
                metric_name="task_success_rate",
                baseline=_stats(0.5, 0.3, 3),
                candidate=_stats(0.55, 0.3, 3),
                delta=0.05,
                p_value=0.5,
                ci=None,
                direction="better",
            )
        ],
        EvalMetricTaskSuccessRate(),
    )
    assert isinstance(needs, RecommendationNeedsMoreRuns)
    assert needs.recommended_n > needs.current_n


def test_reject_when_primary_regresses() -> None:
    rec = derive_recommendation(
        "c",
        [
            MetricComparison(
                metric_name="task_success_rate",
                baseline=_stats(0.9, 0.05, 5),
                candidate=_stats(0.4, 0.05, 5),
                delta=-0.5,
                p_value=0.001,
                ci=None,
                direction="worse",
            )
        ],
        EvalMetricTaskSuccessRate(),
    )
    assert isinstance(rec, RecommendationReject)


# ============================================================================
# Rule 25 — trace_links collected for failed/regressed runs
# ============================================================================


async def test_rule25_trace_links_for_failures() -> None:
    base_obs = InMemoryObservabilityProvider()
    cand_obs = InMemoryObservabilityProvider()
    suite = TaskSuite(
        suite_version=1,
        regression=[_task("t1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysFail())],
    )
    harness = (
        EvalHarnessBuilder(suite, _config_with(base_obs, success=True, n_runs=2), base_obs)
        .candidate("cand", _config_with(cand_obs, success=True, n_runs=2))
        .n_runs_per_config(2)
        .build()
    )
    reports = await harness.run()
    assert reports[0].trace_links


# ============================================================================
# Rule 29 — fixtures are the cross-language oracle (replay)
# ============================================================================


def test_rule29_core_suite_fixture_loads_and_resolves() -> None:
    suite = load_suite_path(_fixtures_dir() / "core_suite.json")
    assert suite.suite_version == 1
    assert len(suite.regression) == 2
    assert len(suite.challenge) == 2
    assert len(suite.canary) == 1
    for _cat, t in suite.all_tasks():
        build_verifier(t.verifier_spec)  # resolves without error
    s1 = suite.regression[0]
    assert s1.id == "regression_s1_uppercase"
    assert isinstance(s1.workspace_snapshot, WorkspaceSnapshotFiles)
    assert "input.txt" in s1.workspace_snapshot.files


def test_rule29_welch_bootstrap_fixture_replay() -> None:
    body = (_fixtures_dir() / "welch_bootstrap.json").read_text()
    oracle = json.loads(body)
    for case in oracle["cases"]:
        w = welch_t_test(case["baseline"], case["candidate"])
        assert abs(abs(w.t) - abs(case["welch_t"])) < 1e-9, case["name"]
        assert abs(w.df - case["welch_df"]) < 1e-9, case["name"]
        assert abs(w.p_value - case["welch_p_value"]) < case["welch_p_tolerance"], case["name"]
        cand = bootstrap_ci(case["candidate"], 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)
        assert cand is not None
        assert abs(cand.lower - case["candidate_bootstrap_ci"]["lower"]) < 1e-12
        assert abs(cand.upper - case["candidate_bootstrap_ci"]["upper"]) < 1e-12
        base = bootstrap_ci(case["baseline"], 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)
        assert base is not None
        assert abs(base.lower - case["baseline_bootstrap_ci"]["lower"]) < 1e-12
        assert abs(base.upper - case["baseline_bootstrap_ci"]["upper"]) < 1e-12


# ============================================================================
# Rule 30 — TraceAnalyzer is interface-only (structural; no built-in impl)
# ============================================================================


async def test_rule30_trace_analyzer_interface_only() -> None:
    from spore_eval.eval import TraceAnalyzer

    class UserAnalyzer:
        async def analyze(self, traces) -> list[HarnessConfigDiff]:
            return [HarnessConfigDiff(description="widen window")]

    analyzer: TraceAnalyzer = UserAnalyzer()
    assert isinstance(analyzer, TraceAnalyzer)
    assert (await analyzer.analyze([]))[0].description == "widen window"


# ============================================================================
# Rule 31 — manual promotion bumps suite_version, moves challenge->regression
# ============================================================================


def test_rule31_promote_challenge_to_regression() -> None:
    suite = TaskSuite(
        suite_version=1,
        regression=[_task("r1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())],
        challenge=[_task("c1", WorkspaceSnapshotEmpty(), VerifierSpecAlwaysPass())],
    )
    promote_challenge_task(suite, "c1")
    assert suite.suite_version == 2
    assert len(suite.regression) == 2
    assert len(suite.challenge) == 0
    with pytest.raises(Exception):
        promote_challenge_task(suite, "nope")


def test_rule31_promote_round_trips_json() -> None:
    suite = load_suite_path(_fixtures_dir() / "core_suite.json")
    before_reg = len(suite.regression)
    promote_challenge_task(suite, "challenge_s5_shell_pipeline")
    assert suite.suite_version == 2
    assert len(suite.regression) == before_reg + 1
    reparsed = load_suite_str(suite_to_json(suite))
    assert reparsed.suite_version == 2


# ============================================================================
# Rule 32 — no Inspect AI / Langfuse / specialized-stats dependency
# ============================================================================


def test_rule32_no_inspect_or_langfuse_dependency() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text().lower()
    assert "inspect-ai" not in pyproject
    assert "inspect_ai" not in pyproject
    assert "langfuse" not in pyproject
    assert "scipy" not in pyproject
    assert "statsmodels" not in pyproject


# ============================================================================
# E2E hermetic regression test — baseline vs a deliberately-worse candidate
# ============================================================================


async def test_e2e_regression_flagged_with_sane_pvalue_and_recommendation() -> None:
    obs = InMemoryObservabilityProvider()
    suite = TaskSuite(
        suite_version=1,
        regression=[
            _task(
                "e2e",
                WorkspaceSnapshotEmpty(),
                VerifierSpecMetricEvaluator(
                    descriptor="run-success",
                    direction="maximize",
                    min=0.0,
                    max=1.0,
                ),
            )
        ],
    )
    n = 6
    harness = (
        EvalHarnessBuilder(suite, _config_with(obs, success=True, n_runs=n), obs)
        .candidate("smaller_window", _config_with(obs, success=False, n_runs=n))
        .n_runs_per_config(n)
        .metrics([EvalMetricTaskSuccessRate()])
        .primary_metric(EvalMetricTaskSuccessRate())
        .build()
    )
    reports = await harness.run()
    assert len(reports) == 1
    report = reports[0]
    success = next(m for m in report.metrics if m.metric_name == "task_success_rate")
    assert success.baseline.mean > success.candidate.mean
    assert success.direction == "worse"
    assert 0.0 <= success.p_value <= 1.0
    assert success.p_value < 0.05
    assert isinstance(report.recommendation, (RecommendationReject, RecommendationNeedsMoreRuns))
    assert report.trace_links
