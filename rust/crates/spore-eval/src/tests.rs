//! Rule-by-rule tests for the EvalHarness (Rules 1-29), the interface-only
//! compile test (Rule 30), the promote test (Rule 31), the no-Inspect/Langfuse
//! dependency assertion (Rule 32), the E2E hermetic regression test, and the
//! fixture-replay tests.
//!
//! All tests are hermetic: MockAgent (no network) + InMemoryObservabilityProvider.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use spore_core::agent::mock::MockAgent;
use spore_core::agent::{AgentId, TurnResult};
use spore_core::harness::testing::{AllowAllSandbox, AlwaysContinuePolicy, NoopContextManager};
use spore_core::harness::{HarnessBuilder, HarnessConfig, RunResult, SessionId, TaskId};
use spore_core::model::TokenUsage;
use spore_core::observability::InMemoryObservabilityProvider;

use crate::harness::{EvalHarnessBuilder, HarnessConfigDiff, TraceAnalyzer};
use crate::metric_map::EvalMetric;
use crate::report::{ComparisonDirection, Recommendation};
use crate::stats::{bootstrap_ci, welch_t_test, DEFAULT_BOOTSTRAP_SEED};
use crate::task::{
    CompositeChildSpec, EvalTask, TaskSuite, VerificationResult, VerifierSpec, WorkspaceSnapshot,
};
use crate::verifier::{build_verifier, AlwaysFail, AlwaysPass, CompositeVerifier, TaskVerifier};
use crate::worktree::Workspace;

// ============================================================================
// Helpers
// ============================================================================

fn usage() -> TokenUsage {
    TokenUsage {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: None,
        cache_write_tokens: None,
    }
}

/// Build a HarnessConfig wired to a shared observability provider, with a
/// MockAgent that produces `n_runs_needed` final responses (one per run).
/// `success` controls whether each run succeeds or errors out (a weaker config).
fn config_with(
    obs: Arc<InMemoryObservabilityProvider>,
    success: bool,
    n_runs_needed: u32,
) -> HarnessConfig {
    let agent = Arc::new(MockAgent::new(AgentId::new("mock")));
    for _ in 0..(n_runs_needed.max(1)) {
        if success {
            agent.push(TurnResult::FinalResponse {
                content: "DONE".into(),
                usage: usage(),
            });
        } else {
            agent.push(TurnResult::Error {
                error: spore_core::agent::AgentError::EmptyResponse,
                usage: Some(usage()),
            });
        }
    }
    HarnessBuilder::new(
        agent,
        Arc::new(spore_core::harness::testing::ScriptedToolRegistry::new()),
        Arc::new(AllowAllSandbox),
        Arc::new(NoopContextManager),
        Arc::new(AlwaysContinuePolicy),
    )
    .observability(obs)
    .build_config()
}

fn task(id: &str, snapshot: WorkspaceSnapshot, spec: VerifierSpec) -> EvalTask {
    let mut t = EvalTask {
        id: TaskId::new(id),
        instruction: "do the thing".into(),
        workspace_snapshot: snapshot,
        verifier: None,
        verifier_spec: spec,
        expected_turns: Some((1, 4)),
        expected_cost_usd: None,
        tags: vec!["unit".into()],
        timeout: Duration::from_secs(30),
        model_fixture: None,
    };
    t.resolve_verifier();
    t
}

fn files(pairs: &[(&str, &str)]) -> WorkspaceSnapshot {
    let mut m = BTreeMap::new();
    for (k, v) in pairs {
        m.insert((*k).to_string(), (*v).to_string());
    }
    WorkspaceSnapshot::Files { files: m }
}

fn run_result_success() -> RunResult {
    RunResult::Success {
        output: "DONE".into(),
        session_id: SessionId::new("s"),
        usage: Default::default(),
        turns: 1,
    }
}

// ============================================================================
// Rule 1 — three disjoint task lists
// ============================================================================

#[test]
fn rule1_three_disjoint_lists() {
    let suite = TaskSuite {
        suite_version: 1,
        regression: vec![task(
            "r1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysPass,
        )],
        challenge: vec![task(
            "c1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysPass,
        )],
        canary: vec![task(
            "k1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysPass,
        )],
    };
    assert_eq!(suite.all_tasks().len(), 3);
}

// ============================================================================
// Rule 2 / Rule 3 — fresh workspace restored + torn down
// ============================================================================

#[tokio::test]
async fn rule2_workspace_restored_from_files() {
    let snap = files(&[("input.txt", "hello\n"), ("sub/x.md", "deep")]);
    let ws = Workspace::restore(&snap).await.unwrap();
    let body = tokio::fs::read_to_string(ws.path().join("input.txt"))
        .await
        .unwrap();
    assert_eq!(body, "hello\n");
    assert!(ws.path().join("sub/x.md").exists());
}

#[tokio::test]
async fn rule3_workspace_torn_down_on_drop() {
    let snap = files(&[("a.txt", "x")]);
    let path = {
        let ws = Workspace::restore(&snap).await.unwrap();
        ws.path().to_path_buf()
    };
    assert!(!path.exists(), "workspace dir should be removed on drop");
}

#[tokio::test]
async fn rule2_git_ref_restores_worktree() {
    let src = tempfile::tempdir().unwrap();
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(src.path())
            .output()
            .unwrap()
    };
    git(&["init", "-q"]);
    git(&["config", "user.email", "t@example.com"]);
    git(&["config", "user.name", "t"]);
    std::fs::write(src.path().join("seed.txt"), "from git\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-q", "-m", "seed"]);

    let snap = WorkspaceSnapshot::GitRef {
        repo: src.path().to_string_lossy().into_owned(),
        reference: "HEAD".into(),
    };
    let ws = Workspace::restore(&snap).await.unwrap();
    let body = tokio::fs::read_to_string(ws.path().join("seed.txt"))
        .await
        .unwrap();
    assert_eq!(body, "from git\n");
}

// ============================================================================
// Rule 4 — timeout yields a failed run (not a panic)
// ============================================================================

#[tokio::test]
async fn rule4_timeout_is_failed_run_not_panic() {
    let obs = Arc::new(InMemoryObservabilityProvider::new());
    let mut t = task("slow", WorkspaceSnapshot::Empty, VerifierSpec::AlwaysFail);
    t.timeout = Duration::from_millis(1);
    let suite = TaskSuite {
        suite_version: 1,
        regression: vec![t],
        challenge: vec![],
        canary: vec![],
    };
    let harness = EvalHarnessBuilder::new(suite, config_with(obs.clone(), true, 5), obs)
        .candidate(
            "cand",
            config_with(Arc::new(InMemoryObservabilityProvider::new()), true, 5),
        )
        .n_runs_per_config(1)
        .build();
    let reports = harness.run().await.unwrap();
    assert_eq!(reports.len(), 1);
}

// ============================================================================
// Rule 5 — tags are free-form
// ============================================================================

#[test]
fn rule5_tags_are_free_form() {
    let t = task("t", WorkspaceSnapshot::Empty, VerifierSpec::AlwaysPass);
    assert_eq!(t.tags, vec!["unit".to_string()]);
}

// ============================================================================
// Rule 6 — suite_version required; loader rejects manifest without it
// ============================================================================

#[test]
fn rule6_loader_rejects_missing_suite_version() {
    let json = r#"{ "regression": [], "challenge": [], "canary": [] }"#;
    let err = crate::manifest::load_suite_str(json).unwrap_err();
    assert!(matches!(err, crate::task::EvalError::MissingSuiteVersion));
}

#[test]
fn rule6_loader_accepts_with_suite_version() {
    let json = r#"{ "suite_version": 7, "regression": [], "challenge": [], "canary": [] }"#;
    let suite = crate::manifest::load_suite_str(json).unwrap();
    assert_eq!(suite.suite_version, 7);
}

// ============================================================================
// Rule 7 / Rule 8 — verification result shape + score clamp
// ============================================================================

#[test]
fn rule7_verification_result_shape() {
    let r = VerificationResult::new(true, 0.5, "ok")
        .unwrap()
        .with_signal("k", 1.0);
    assert!(r.passed);
    assert_eq!(r.score, 0.5);
    assert_eq!(r.detail, "ok");
    assert_eq!(r.signals.get("k"), Some(&1.0));
}

#[test]
fn rule8_score_out_of_range_is_error() {
    assert!(VerificationResult::new(true, 1.5, "x").is_err());
    assert!(VerificationResult::new(true, -0.1, "x").is_err());
    assert_eq!(VerificationResult::clamped(true, 1.5, "x").score, 1.0);
    assert_eq!(VerificationResult::clamped(true, -0.1, "x").score, 0.0);
}

// ============================================================================
// Rule 9 — is_deterministic per verifier
// ============================================================================

#[test]
fn rule9_determinism_flags() {
    assert!(build_verifier(&VerifierSpec::AlwaysPass).is_deterministic());
    assert!(build_verifier(&VerifierSpec::TestSuite {
        command: "true".into(),
        args: vec![],
        timeout_secs: Some(1),
    })
    .is_deterministic());
    assert!(!build_verifier(&VerifierSpec::LlmJudge {
        rubric: "r".into(),
        score_range: (0.0, 1.0),
    })
    .is_deterministic());
}

// ============================================================================
// Rule 10 — TestSuiteVerifier pass-rate + passed == score==1.0
// ============================================================================

#[tokio::test]
async fn rule10_test_suite_verifier_passes_on_zero_exit() {
    let ws = tempfile::tempdir().unwrap();
    tokio::fs::write(ws.path().join("output.txt"), "HELLO\n")
        .await
        .unwrap();
    let t = task(
        "t",
        WorkspaceSnapshot::Empty,
        VerifierSpec::TestSuite {
            command: "sh".into(),
            args: vec!["-c".into(), "grep -q HELLO output.txt".into()],
            timeout_secs: Some(10),
        },
    );
    let r = t
        .verifier()
        .verify(&t, &run_result_success(), ws.path())
        .await
        .unwrap();
    assert!(r.passed);
    assert_eq!(r.score, 1.0);
}

#[tokio::test]
async fn rule10_test_suite_verifier_fails_on_nonzero_exit() {
    let ws = tempfile::tempdir().unwrap();
    let t = task(
        "t",
        WorkspaceSnapshot::Empty,
        VerifierSpec::TestSuite {
            command: "sh".into(),
            args: vec!["-c".into(), "exit 1".into()],
            timeout_secs: Some(10),
        },
    );
    let r = t
        .verifier()
        .verify(&t, &run_result_success(), ws.path())
        .await
        .unwrap();
    assert!(!r.passed);
    assert_eq!(r.score, 0.0);
}

// ============================================================================
// Rule 11 — CompositeVerifier: weighted mean, required AND, determinism AND
// ============================================================================

#[tokio::test]
async fn rule11_composite_weighted_mean_and_required() {
    let composite = CompositeVerifier::new(vec![
        (Arc::new(AlwaysPass) as Arc<dyn TaskVerifier>, 1.0, true),
        (Arc::new(AlwaysFail) as Arc<dyn TaskVerifier>, 1.0, false),
    ]);
    let t = task("t", WorkspaceSnapshot::Empty, VerifierSpec::AlwaysPass);
    let r = composite
        .verify(&t, &run_result_success(), Path::new("/tmp"))
        .await
        .unwrap();
    assert!((r.score - 0.5).abs() < 1e-9);
    assert!(r.passed);
    assert!(composite.is_deterministic());
}

#[tokio::test]
async fn rule11_composite_required_failure_fails_overall() {
    let composite = CompositeVerifier::new(vec![
        (Arc::new(AlwaysPass) as Arc<dyn TaskVerifier>, 1.0, true),
        (Arc::new(AlwaysFail) as Arc<dyn TaskVerifier>, 1.0, true),
    ]);
    let t = task("t", WorkspaceSnapshot::Empty, VerifierSpec::AlwaysPass);
    let r = composite
        .verify(&t, &run_result_success(), Path::new("/tmp"))
        .await
        .unwrap();
    assert!(!r.passed);
}

#[test]
fn rule11_composite_spec_resolves_determinism_and() {
    let spec = VerifierSpec::Composite {
        children: vec![
            CompositeChildSpec {
                spec: VerifierSpec::AlwaysPass,
                weight: 2.0,
                required: true,
            },
            CompositeChildSpec {
                spec: VerifierSpec::LlmJudge {
                    rubric: "r".into(),
                    score_range: (0.0, 1.0),
                },
                weight: 1.0,
                required: false,
            },
        ],
    };
    assert!(!build_verifier(&spec).is_deterministic());
}

// ============================================================================
// Rule 12 — MetricEvaluatorVerifier normalizes a metric value
// ============================================================================

#[tokio::test]
async fn rule12_metric_evaluator_verifier_normalizes() {
    use spore_core::harness::{BoxFut, OptimizationDirection, SandboxProvider};
    use spore_core::metric::{MetricError, MetricEvaluator, MetricResult};
    use spore_core::termination::SessionStateSnapshot;

    struct FixedEval(f64);
    impl MetricEvaluator for FixedEval {
        fn evaluate<'a>(
            &'a self,
            _sandbox: &'a dyn SandboxProvider,
            _state: &'a SessionStateSnapshot,
        ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
            let v = self.0;
            Box::pin(async move { Ok(MetricResult::new(v)) })
        }
        fn direction(&self) -> OptimizationDirection {
            OptimizationDirection::Maximize
        }
        fn description(&self) -> String {
            "fixed".into()
        }
    }

    let v =
        crate::verifier::MetricEvaluatorVerifier::with_range(Arc::new(FixedEval(7.5)), 0.0, 10.0);
    assert!(v.is_deterministic());
    let t = task("t", WorkspaceSnapshot::Empty, VerifierSpec::AlwaysPass);
    let ws = tempfile::tempdir().unwrap();
    let r = v
        .verify(&t, &run_result_success(), ws.path())
        .await
        .unwrap();
    assert!((r.score - 0.75).abs() < 1e-9);
}

// ============================================================================
// Rule 13 — LlmJudgeVerifier non-deterministic + pluggable judge
// ============================================================================

#[tokio::test]
async fn rule13_llm_judge_verifier() {
    use spore_core::model::mock::MockModelInterface;
    use spore_core::model::{
        ContentBlock, ModelResponse, ProviderInfo, StopReason, TokenUsage as TU,
    };

    let judge = Arc::new(MockModelInterface::new(ProviderInfo {
        name: "fake".into(),
        model_id: "judge".into(),
        context_window: 8000,
    }));
    judge.push_response(Ok(ModelResponse {
        content: vec![ContentBlock::Text {
            text: "score: 8".into(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: TU::default(),
    }));
    let v = crate::verifier::LlmJudgeVerifier {
        judge,
        rubric: "rate".into(),
        score_range: (0.0, 10.0),
        params: Default::default(),
    };
    assert!(!v.is_deterministic());
    let t = task("t", WorkspaceSnapshot::Empty, VerifierSpec::AlwaysPass);
    let r = v
        .verify(&t, &run_result_success(), Path::new("/tmp"))
        .await
        .unwrap();
    assert!((r.score - 0.8).abs() < 1e-9);
}

// ============================================================================
// Rule 14 / Rule 15 / Rule 16 — n runs, build harness from config, read obs
// ============================================================================

#[tokio::test]
async fn rule14_15_16_runs_per_config_and_metrics_from_obs() {
    let base_obs = Arc::new(InMemoryObservabilityProvider::new());
    let cand_obs = Arc::new(InMemoryObservabilityProvider::new());
    let suite = TaskSuite {
        suite_version: 1,
        regression: vec![task(
            "t1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysPass,
        )],
        challenge: vec![],
        canary: vec![],
    };
    let n = 3;
    let harness = EvalHarnessBuilder::new(
        suite,
        config_with(base_obs.clone(), true, n),
        base_obs.clone(),
    )
    .candidate("cand", config_with(cand_obs.clone(), true, n))
    .n_runs_per_config(n)
    .metrics(vec![
        EvalMetric::TaskSuccessRate,
        EvalMetric::MeanTurnsToCompletion,
    ])
    .build();

    let reports = harness.run().await.unwrap();
    assert_eq!(reports.len(), 1);
    let success = reports[0]
        .metrics
        .iter()
        .find(|m| m.metric_name == EvalMetric::TaskSuccessRate.name())
        .unwrap();
    assert_eq!(success.baseline.n, n); // Rule 14
    let turns = reports[0]
        .metrics
        .iter()
        .find(|m| m.metric_name == EvalMetric::MeanTurnsToCompletion.name())
        .unwrap();
    assert!(turns.baseline.mean >= 1.0); // Rule 16: from turn spans
}

// ============================================================================
// Rule 17 — EvalMetric mapping (name + direction)
// ============================================================================

#[test]
fn rule17_metric_names_and_directions() {
    use spore_core::harness::OptimizationDirection::*;
    assert_eq!(EvalMetric::TaskSuccessRate.direction(), Maximize);
    assert_eq!(EvalMetric::MeanCostUsd.direction(), Minimize);
    assert_eq!(EvalMetric::MeanTurnsToCompletion.direction(), Minimize);
    assert_eq!(
        EvalMetric::CacheHitRate {
            block: "sys".into()
        }
        .direction(),
        Maximize
    );
    assert_eq!(
        EvalMetric::CacheHitRate {
            block: "sys".into()
        }
        .name(),
        "cache_hit_rate[sys]"
    );
    assert_eq!(EvalMetric::VerificationScore.direction(), Maximize);
}

// ============================================================================
// Rule 18 — WaitingForHuman not counted as success/failure
// ============================================================================

#[test]
fn rule18_resource_metric_still_computes_for_waiting() {
    use crate::metric_map::{sample_for, RunSampleInputs};
    let session = spore_core::observability::SessionMetrics {
        session_id: SessionId::new("s"),
        task_id: TaskId::new("t"),
        total_turns: 2,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_cost_usd: 0.0,
        total_duration_ms: 0,
        tool_calls: 0,
        sensor_fires: 0,
        sensor_halts: 0,
        compactions: 0,
        outcome: spore_core::guide_registry::SessionOutcome::Partial,
        guides_used: vec![],
        patch_count: 0,
        patch_rate: 0.0,
        patches_by_tool: Default::default(),
        compaction_verification_failures: 0,
    };
    let v = sample_for(
        &EvalMetric::MeanTurnsToCompletion,
        &session,
        &[],
        &RunSampleInputs {
            verifier_passed: false,
            verifier_score: 0.0,
        },
    );
    assert_eq!(v, 2.0);
}

// ============================================================================
// Rule 19 — MetricStats aggregation (sanity; full oracle in stats.rs)
// ============================================================================

#[test]
fn rule19_metric_stats_aggregates() {
    let s = crate::stats::MetricStats::from_samples(&[1.0, 1.0, 1.0]);
    assert_eq!(s.n, 3);
    assert_eq!(s.mean, 1.0);
    assert_eq!(s.stddev, 0.0);
}

// ============================================================================
// Rule 20 — Welch's t-test recorded
// ============================================================================

#[test]
fn rule20_welch_recorded() {
    let w = welch_t_test(&[0.9, 0.9, 0.9, 0.9], &[0.1, 0.1, 0.1, 0.1]);
    assert!(w.p_value < 0.05);
}

// ============================================================================
// Rule 21 — bootstrap CI present for non-deterministic metrics
// ============================================================================

#[test]
fn rule21_bootstrap_ci_present() {
    let ci = bootstrap_ci(&[0.5, 0.6, 0.4, 0.55], 1000, 0.95, DEFAULT_BOOTSTRAP_SEED).unwrap();
    assert!(ci.lower <= ci.upper);
}

#[tokio::test]
async fn rule21_nondeterministic_verifier_gets_ci() {
    // A suite whose verifier is non-deterministic (LlmJudge stub) → the
    // TaskSuccessRate comparison carries a bootstrap CI.
    let base_obs = Arc::new(InMemoryObservabilityProvider::new());
    let cand_obs = Arc::new(InMemoryObservabilityProvider::new());
    let suite = TaskSuite {
        suite_version: 1,
        regression: vec![task(
            "t",
            WorkspaceSnapshot::Empty,
            VerifierSpec::LlmJudge {
                rubric: "r".into(),
                score_range: (0.0, 1.0),
            },
        )],
        challenge: vec![],
        canary: vec![],
    };
    let n = 4;
    let harness = EvalHarnessBuilder::new(
        suite,
        config_with(base_obs.clone(), true, n),
        base_obs.clone(),
    )
    .candidate("cand", config_with(cand_obs.clone(), true, n))
    .n_runs_per_config(n)
    .metrics(vec![EvalMetric::TaskSuccessRate])
    .build();
    let reports = harness.run().await.unwrap();
    let cmp = &reports[0].metrics[0];
    assert!(
        cmp.ci.is_some(),
        "non-deterministic metric should carry a CI"
    );
}

// ============================================================================
// Rule 22 — direction
// ============================================================================

#[test]
fn rule22_direction() {
    use crate::report::classify_direction;
    use spore_core::harness::OptimizationDirection;
    assert_eq!(
        classify_direction(0.3, OptimizationDirection::Maximize, 1e-9),
        ComparisonDirection::Better
    );
}

// ============================================================================
// Rule 23 / Rule 24 — recommendation + recommended_n
// ============================================================================

#[test]
fn rule23_24_recommendation_paths() {
    use crate::report::{derive_recommendation, MetricComparison};
    use crate::stats::MetricStats;
    let stats = |m, n| MetricStats {
        mean: m,
        stddev: 0.05,
        p50: m,
        p95: m,
        n,
    };
    let adopt = derive_recommendation(
        "c",
        &[MetricComparison {
            metric_name: EvalMetric::TaskSuccessRate.name(),
            baseline: stats(0.5, 5),
            candidate: stats(0.95, 5),
            delta: 0.45,
            p_value: 0.001,
            ci: None,
            direction: ComparisonDirection::Better,
        }],
        &EvalMetric::TaskSuccessRate,
    );
    assert!(matches!(adopt, Recommendation::Adopt { .. }));

    let needs = derive_recommendation(
        "c",
        &[MetricComparison {
            metric_name: EvalMetric::TaskSuccessRate.name(),
            baseline: stats(0.5, 3),
            candidate: stats(0.55, 3),
            delta: 0.05,
            p_value: 0.5,
            ci: None,
            direction: ComparisonDirection::Better,
        }],
        &EvalMetric::TaskSuccessRate,
    );
    match needs {
        Recommendation::NeedsMoreRuns {
            current_n,
            recommended_n,
        } => assert!(recommended_n > current_n),
        other => panic!("expected NeedsMoreRuns, got {other:?}"),
    }
}

// ============================================================================
// Rule 25 — trace_links collected for failed/regressed runs
// ============================================================================

#[tokio::test]
async fn rule25_trace_links_for_failures() {
    let base_obs = Arc::new(InMemoryObservabilityProvider::new());
    let cand_obs = Arc::new(InMemoryObservabilityProvider::new());
    let suite = TaskSuite {
        suite_version: 1,
        regression: vec![task(
            "t1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysFail,
        )],
        challenge: vec![],
        canary: vec![],
    };
    let harness = EvalHarnessBuilder::new(suite, config_with(base_obs.clone(), true, 2), base_obs)
        .candidate("cand", config_with(cand_obs.clone(), true, 2))
        .n_runs_per_config(2)
        .build();
    let reports = harness.run().await.unwrap();
    assert!(!reports[0].trace_links.is_empty());
}

// ============================================================================
// Rule 26-28 — native stats (oracle in stats.rs); structural assertion here
// ============================================================================

#[test]
fn rule26_28_native_stats_resolve_from_crate() {
    let _ = welch_t_test(&[1.0, 2.0], &[3.0, 4.0]);
    let _ = bootstrap_ci(&[1.0, 2.0], 10, 0.95, 1);
}

// ============================================================================
// Rule 29 — fixtures are the cross-language oracle (replay)
// ============================================================================

fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures/task_suites")
}

#[test]
fn rule29_core_suite_fixture_loads_and_resolves() {
    let path = fixtures_dir().join("core_suite.json");
    let suite = crate::manifest::load_suite_path(&path).unwrap();
    assert_eq!(suite.suite_version, 1);
    assert_eq!(suite.regression.len(), 2);
    assert_eq!(suite.challenge.len(), 2);
    assert_eq!(suite.canary.len(), 1);
    for (_cat, t) in suite.all_tasks() {
        let _ = t.verifier();
    }
    let s1 = &suite.regression[0];
    assert_eq!(s1.id.as_str(), "regression_s1_uppercase");
    match &s1.workspace_snapshot {
        WorkspaceSnapshot::Files { files } => assert!(files.contains_key("input.txt")),
        other => panic!("expected Files, got {other:?}"),
    }
}

#[derive(serde::Deserialize)]
struct StatsOracle {
    cases: Vec<StatsCase>,
}
#[derive(serde::Deserialize)]
struct StatsCase {
    name: String,
    baseline: Vec<f64>,
    candidate: Vec<f64>,
    welch_t: f64,
    welch_df: f64,
    welch_p_value: f64,
    welch_p_tolerance: f64,
    candidate_bootstrap_ci: CiBounds,
    baseline_bootstrap_ci: CiBounds,
}
#[derive(serde::Deserialize)]
struct CiBounds {
    lower: f64,
    upper: f64,
}

#[test]
fn rule29_welch_bootstrap_fixture_replay() {
    let path = fixtures_dir().join("welch_bootstrap.json");
    let body = std::fs::read_to_string(&path).unwrap();
    let oracle: StatsOracle = serde_json::from_str(&body).unwrap();
    for case in oracle.cases {
        let w = welch_t_test(&case.baseline, &case.candidate);
        assert!(
            (w.t.abs() - case.welch_t.abs()).abs() < 1e-9,
            "case {} t: {} vs {}",
            case.name,
            w.t,
            case.welch_t
        );
        assert!((w.df - case.welch_df).abs() < 1e-9, "case {} df", case.name);
        assert!(
            (w.p_value - case.welch_p_value).abs() < case.welch_p_tolerance,
            "case {} p: {} vs {}",
            case.name,
            w.p_value,
            case.welch_p_value
        );
        let cand = bootstrap_ci(&case.candidate, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED).unwrap();
        assert!((cand.lower - case.candidate_bootstrap_ci.lower).abs() < 1e-12);
        assert!((cand.upper - case.candidate_bootstrap_ci.upper).abs() < 1e-12);
        let base = bootstrap_ci(&case.baseline, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED).unwrap();
        assert!((base.lower - case.baseline_bootstrap_ci.lower).abs() < 1e-12);
        assert!((base.upper - case.baseline_bootstrap_ci.upper).abs() < 1e-12);
    }
}

// ============================================================================
// Rule 30 — TraceAnalyzer is interface-only (compile test, no built-in impl)
// ============================================================================

#[test]
fn rule30_trace_analyzer_interface_only() {
    struct UserAnalyzer;
    impl TraceAnalyzer for UserAnalyzer {
        fn analyze<'a>(
            &'a self,
            _traces: Vec<Box<dyn spore_core::observability::Span>>,
        ) -> spore_core::harness::BoxFut<'a, Vec<HarnessConfigDiff>> {
            Box::pin(async { vec![HarnessConfigDiff::default()] })
        }
    }
    let _obj: Box<dyn TraceAnalyzer> = Box::new(UserAnalyzer);
}

// ============================================================================
// Rule 31 — manual promotion bumps suite_version, moves challenge->regression
// ============================================================================

#[test]
fn rule31_promote_challenge_to_regression() {
    let mut suite = TaskSuite {
        suite_version: 1,
        regression: vec![task(
            "r1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysPass,
        )],
        challenge: vec![task(
            "c1",
            WorkspaceSnapshot::Empty,
            VerifierSpec::AlwaysPass,
        )],
        canary: vec![],
    };
    crate::manifest::promote_challenge_task(&mut suite, "c1").unwrap();
    assert_eq!(suite.suite_version, 2);
    assert_eq!(suite.regression.len(), 2);
    assert_eq!(suite.challenge.len(), 0);
    assert!(crate::manifest::promote_challenge_task(&mut suite, "nope").is_err());
}

#[test]
fn rule31_promote_round_trips_json() {
    let path = fixtures_dir().join("core_suite.json");
    let mut suite = crate::manifest::load_suite_path(&path).unwrap();
    let before_reg = suite.regression.len();
    crate::manifest::promote_challenge_task(&mut suite, "challenge_s5_shell_pipeline").unwrap();
    assert_eq!(suite.suite_version, 2);
    assert_eq!(suite.regression.len(), before_reg + 1);
    let json = crate::manifest::suite_to_json(&suite).unwrap();
    let reparsed = crate::manifest::load_suite_str(&json).unwrap();
    assert_eq!(reparsed.suite_version, 2);
}

// ============================================================================
// Rule 32 — no Inspect AI / Langfuse (or specialized stats) dependency
// ============================================================================

#[test]
fn rule32_no_inspect_or_langfuse_dependency() {
    let manifest = include_str!("../Cargo.toml").to_lowercase();
    assert!(!manifest.contains("inspect"), "no inspect-ai dependency");
    assert!(!manifest.contains("langfuse"), "no langfuse dependency");
    assert!(
        !manifest.contains("statrs"),
        "no statrs dependency (Rule 26)"
    );
}

// ============================================================================
// E2E hermetic regression test — baseline vs a deliberately-worse candidate
// ============================================================================

#[tokio::test]
async fn e2e_regression_flagged_with_sane_pvalue_and_recommendation() {
    // Shared provider: both configs and the EvalHarness read the SAME provider
    // so the runner sees each config's metrics. Baseline succeeds; candidate is
    // deliberately worse (its agent errors → Failure → run-success verifier 0).
    let obs = Arc::new(InMemoryObservabilityProvider::new());
    let suite = TaskSuite {
        suite_version: 1,
        regression: vec![task(
            "e2e",
            WorkspaceSnapshot::Empty,
            // MetricEvaluator manifest placeholder scores from run success.
            VerifierSpec::MetricEvaluator {
                descriptor: "run-success".into(),
                direction: crate::task::MetricDirection::Maximize,
                min: Some(0.0),
                max: Some(1.0),
                threshold: None,
            },
        )],
        challenge: vec![],
        canary: vec![],
    };

    let n = 6;
    let baseline = config_with(obs.clone(), true, n);
    let candidate = config_with(obs.clone(), false, n);

    let harness = EvalHarnessBuilder::new(suite, baseline, obs.clone())
        .candidate("smaller_window", candidate)
        .n_runs_per_config(n)
        .metrics(vec![EvalMetric::TaskSuccessRate])
        .primary_metric(EvalMetric::TaskSuccessRate)
        .build();

    let reports = harness.run().await.unwrap();
    assert_eq!(reports.len(), 1);
    let report = &reports[0];

    let success = report
        .metrics
        .iter()
        .find(|m| m.metric_name == EvalMetric::TaskSuccessRate.name())
        .unwrap();
    assert!(success.baseline.mean > success.candidate.mean);
    assert_eq!(success.direction, ComparisonDirection::Worse);
    assert!(success.p_value >= 0.0 && success.p_value <= 1.0);
    assert!(success.p_value < 0.05, "p={}", success.p_value);
    assert!(
        matches!(
            report.recommendation,
            Recommendation::Reject { .. } | Recommendation::NeedsMoreRuns { .. }
        ),
        "got {:?}",
        report.recommendation
    );
    assert!(!report.trace_links.is_empty());
}
