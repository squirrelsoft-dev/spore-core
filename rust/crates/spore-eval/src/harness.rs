//! [`EvalHarness`] — the runner (Rules 14-25) — plus its fluent builder and the
//! deferred [`TraceAnalyzer`] interface (Rule 30).

use std::sync::Arc;

use spore_core::harness::{
    BoxFut, Harness, HarnessConfig, HarnessRunOptions, RunResult, SessionId, StandardHarness, Task,
    TaskId,
};
use spore_core::harness::{BudgetLimits, LoopStrategy, ReactConfig};
use spore_core::observability::{ObservabilityProvider, SessionMetrics, Span};

use crate::metric_map::{sample_for, EvalMetric, RunSampleInputs};
use crate::report::{
    classify_direction, derive_recommendation, ComparisonReport, MetricComparison,
};
use crate::stats::{
    bootstrap_ci, welch_t_test, MetricStats, DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_BOOTSTRAP_SEED,
};
use crate::task::{ConfigId, EvalError, EvalTask, TaskSuite};
use crate::worktree::Workspace;

// ============================================================================
// TraceAnalyzer (Rule 30 — interface only, no built-in impl ships)
// ============================================================================

/// A proposed change to a [`HarnessConfig`] produced by a [`TraceAnalyzer`].
/// Marker stub: the optimization loop (propose → run → compare → open PR) is
/// deferred (Rule 30).
#[derive(Debug, Clone, Default)]
pub struct HarnessConfigDiff {
    /// Free-form human-readable description of the proposed change.
    pub description: String,
}

/// Analyzes failure traces and proposes candidate config diffs (Rule 30).
/// Interface only — no built-in implementation ships in the MVP.
pub trait TraceAnalyzer: Send + Sync {
    fn analyze<'a>(&'a self, traces: Vec<Box<dyn Span>>) -> BoxFut<'a, Vec<HarnessConfigDiff>>;
}

// ============================================================================
// EvalHarness
// ============================================================================

/// The evaluation harness: runs a task suite against a baseline and candidate
/// configs, aggregates metrics, and compares them.
pub struct EvalHarness {
    pub task_suite: TaskSuite,
    pub baseline_config: HarnessConfig,
    pub candidate_configs: Vec<(ConfigId, HarnessConfig)>,
    pub n_runs_per_config: u32,
    pub metrics: Vec<EvalMetric>,
    pub observability: Arc<dyn ObservabilityProvider>,
    pub bootstrap_iterations: u32,
    pub primary_metric: EvalMetric,
}

const BASELINE_CONFIG_ID: &str = "baseline";

impl EvalHarness {
    /// Run the full comparison (Rules 14-25). Produces one [`ComparisonReport`]
    /// per candidate config.
    pub async fn run(&self) -> Result<Vec<ComparisonReport>, EvalError> {
        if self.metrics.is_empty() {
            return Err(EvalError::MissingMetrics(
                "no metrics configured for comparison".into(),
            ));
        }

        // Collect per-metric samples for the baseline once.
        let baseline = self
            .run_config(&self.baseline_config, BASELINE_CONFIG_ID)
            .await?;

        let mut reports = Vec::new();
        for (config_id, config) in &self.candidate_configs {
            let candidate = self.run_config(config, config_id.as_str()).await?;
            let report = self.compare(&baseline, &candidate, config_id);
            reports.push(report);
        }
        Ok(reports)
    }

    /// Run every task `n_runs_per_config` times for one config, collecting
    /// per-metric samples and trace links for interesting runs.
    async fn run_config(
        &self,
        config: &HarnessConfig,
        config_id: &str,
    ) -> Result<ConfigSamples, EvalError> {
        let mut samples = ConfigSamples::new(&self.metrics, config_id);
        for (_category, task) in self.task_suite.all_tasks() {
            for run_idx in 0..self.n_runs_per_config {
                self.run_one(config, config_id, task, run_idx, &mut samples)
                    .await?;
            }
        }
        Ok(samples)
    }

    /// Execute a single (config, task) run (Rules 2-3, 14-18, 25).
    async fn run_one(
        &self,
        config: &HarnessConfig,
        config_id: &str,
        task: &EvalTask,
        run_idx: u32,
        samples: &mut ConfigSamples,
    ) -> Result<(), EvalError> {
        // Rule 2: fresh workspace restored from the snapshot.
        let workspace = Workspace::restore(&task.workspace_snapshot).await?;

        // Build a fresh harness from the config (Rule 15) and a unique session.
        let session_id = SessionId::new(format!("{config_id}-{}-{run_idx}", task.id.as_str()));
        let harness = StandardHarness::new(config.clone());
        let core_task = Task {
            id: TaskId::new(format!("{config_id}-{}-{run_idx}", task.id.as_str())),
            instruction: task.instruction.clone(),
            session_id: session_id.clone(),
            budget: BudgetLimits {
                max_turns: Some(task.expected_turns.map(|(_, hi)| hi).unwrap_or(20)),
                max_wall_time: Some(task.timeout),
                ..Default::default()
            },
            loop_strategy: LoopStrategy::ReAct(ReactConfig::per_loop(
                task.expected_turns.map(|(_, hi)| hi).unwrap_or(20),
            )),
        };

        // Rule 15: run the harness. Rule 4: timeout bounds a single run and
        // yields a failed run rather than a panic — guard the await.
        let run_result = match tokio::time::timeout(
            task.timeout.max(std::time::Duration::from_millis(1)),
            harness.run(HarnessRunOptions::new(core_task)),
        )
        .await
        {
            Ok(r) => r,
            Err(_) => RunResult::Failure {
                reason: spore_core::harness::HaltReason::BudgetExceeded {
                    limit_type: spore_core::harness::BudgetLimitType::WallTime,
                },
                session_id: session_id.clone(),
                usage: Default::default(),
                turns: 0,
                session_state: Default::default(),
            },
        };

        // Rule 16: read metrics from observability (do not recompute).
        let session_metrics = self
            .observability
            .get_session_metrics(&session_id)
            .await
            .unwrap_or_else(|| {
                empty_session_metrics(&session_id, &core_task_id(config_id, task, run_idx))
            });
        let trace = self.observability.get_trace(&session_id).await;

        // Run the verifier (Rules 7-13).
        let verifier = task.verifier();
        let verification = verifier.verify(task, &run_result, workspace.path()).await?;

        // Rule 18: WaitingForHuman counts as neither success nor failure; it is
        // reported separately and excluded from success-rate / score samples.
        let waiting = matches!(run_result, RunResult::WaitingForHuman { .. });
        if waiting {
            samples.waiting_for_human += 1;
        }

        let inputs = RunSampleInputs {
            verifier_passed: verification.passed,
            verifier_score: verification.score,
        };

        for (metric, vec) in self.metrics.iter().zip(samples.per_metric.iter_mut()) {
            // Skip success-rate / verification-score samples for WaitingForHuman
            // runs (Rule 18); resource metrics still count.
            if waiting
                && matches!(
                    metric,
                    EvalMetric::TaskSuccessRate | EvalMetric::VerificationScore
                )
            {
                continue;
            }
            vec.push(sample_for(metric, &session_metrics, &trace, &inputs));
        }

        // Rule 25: collect trace links for failed or non-passing runs.
        if !verification.passed || matches!(run_result, RunResult::Failure { .. }) {
            samples.trace_links.push(session_id.0.clone());
        }

        // Rule 3: workspace torn down here regardless of outcome (Drop).
        drop(workspace);
        Ok(())
    }

    /// Compare baseline vs candidate samples (Rules 19-25).
    fn compare(
        &self,
        baseline: &ConfigSamples,
        candidate: &ConfigSamples,
        config_id: &ConfigId,
    ) -> ComparisonReport {
        let mut comparisons = Vec::new();
        for (i, metric) in self.metrics.iter().enumerate() {
            let base = &baseline.per_metric[i];
            let cand = &candidate.per_metric[i];
            let base_stats = MetricStats::from_samples(base); // Rule 19
            let cand_stats = MetricStats::from_samples(cand);
            let delta = cand_stats.mean - base_stats.mean;
            let welch = welch_t_test(base, cand); // Rule 20
            let direction = classify_direction(delta, metric.direction(), 1e-9); // Rule 22

            // Rule 21: bootstrap CI for metrics from non-deterministic verifiers.
            // The verification-score / success-rate metrics carry the verifier's
            // determinism; we approximate "non-deterministic" as those metrics
            // whose task verifier is non-deterministic. Compute a CI on the
            // candidate's delta-bearing samples for those metrics.
            let ci = if self.metric_is_non_deterministic(metric) {
                bootstrap_ci(
                    cand,
                    self.bootstrap_iterations,
                    0.95,
                    DEFAULT_BOOTSTRAP_SEED,
                )
            } else {
                None
            };

            comparisons.push(MetricComparison {
                metric_name: metric.name(),
                baseline: base_stats,
                candidate: cand_stats,
                delta,
                p_value: welch.p_value,
                ci,
                direction,
            });
        }

        // Rules 23-24.
        let recommendation =
            derive_recommendation(config_id.as_str(), &comparisons, &self.primary_metric);

        // Rule 25.
        let mut trace_links = candidate.trace_links.clone();
        trace_links.extend(baseline.trace_links.iter().cloned());

        ComparisonReport {
            baseline_config_id: BASELINE_CONFIG_ID.to_string(),
            candidate_config_id: config_id.0.clone(),
            metrics: comparisons,
            recommendation,
            trace_links,
        }
    }

    /// Whether a metric should carry a bootstrap CI (Rule 21): metrics derived
    /// from non-deterministic verifiers (any task whose verifier reports
    /// `is_deterministic() == false`).
    fn metric_is_non_deterministic(&self, metric: &EvalMetric) -> bool {
        let verifier_dependent = matches!(
            metric,
            EvalMetric::TaskSuccessRate | EvalMetric::VerificationScore
        );
        if !verifier_dependent {
            return false;
        }
        self.task_suite
            .all_tasks()
            .iter()
            .any(|(_, t)| !t.verifier().is_deterministic())
    }
}

fn core_task_id(config_id: &str, task: &EvalTask, run_idx: u32) -> TaskId {
    TaskId::new(format!("{config_id}-{}-{run_idx}", task.id.as_str()))
}

fn empty_session_metrics(session_id: &SessionId, task_id: &TaskId) -> SessionMetrics {
    SessionMetrics {
        session_id: session_id.clone(),
        task_id: task_id.clone(),
        total_turns: 0,
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_cost_usd: 0.0,
        total_duration_ms: 0,
        tool_calls: 0,
        sensor_fires: 0,
        sensor_halts: 0,
        compactions: 0,
        outcome: spore_core::guide_registry::SessionOutcome::Partial,
        guides_used: Vec::new(),
        patch_count: 0,
        patch_rate: 0.0,
        patches_by_tool: Default::default(),
        compaction_verification_failures: 0,
    }
}

/// Per-config metric samples + the metadata the comparison needs.
struct ConfigSamples {
    per_metric: Vec<Vec<f64>>,
    trace_links: Vec<String>,
    waiting_for_human: u32,
}

impl ConfigSamples {
    fn new(metrics: &[EvalMetric], _config_id: &str) -> Self {
        Self {
            per_metric: metrics.iter().map(|_| Vec::new()).collect(),
            trace_links: Vec::new(),
            waiting_for_human: 0,
        }
    }
}

// ============================================================================
// EvalHarnessBuilder
// ============================================================================

/// Fluent assembler for an [`EvalHarness`], mirroring `HarnessBuilder`.
pub struct EvalHarnessBuilder {
    task_suite: TaskSuite,
    baseline_config: HarnessConfig,
    candidate_configs: Vec<(ConfigId, HarnessConfig)>,
    n_runs_per_config: u32,
    metrics: Vec<EvalMetric>,
    observability: Arc<dyn ObservabilityProvider>,
    bootstrap_iterations: u32,
    primary_metric: EvalMetric,
}

impl EvalHarnessBuilder {
    /// Start from the required pieces: a suite, the baseline config, and the
    /// observability provider the runner reads metrics from.
    pub fn new(
        task_suite: TaskSuite,
        baseline_config: HarnessConfig,
        observability: Arc<dyn ObservabilityProvider>,
    ) -> Self {
        Self {
            task_suite,
            baseline_config,
            candidate_configs: Vec::new(),
            n_runs_per_config: 3,
            metrics: vec![EvalMetric::TaskSuccessRate],
            observability,
            bootstrap_iterations: DEFAULT_BOOTSTRAP_ITERATIONS,
            primary_metric: EvalMetric::TaskSuccessRate,
        }
    }

    pub fn candidate(mut self, id: impl Into<String>, config: HarnessConfig) -> Self {
        self.candidate_configs.push((ConfigId::new(id), config));
        self
    }

    pub fn n_runs_per_config(mut self, n: u32) -> Self {
        self.n_runs_per_config = n;
        self
    }

    pub fn metrics(mut self, metrics: Vec<EvalMetric>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn bootstrap_iterations(mut self, n: u32) -> Self {
        self.bootstrap_iterations = n;
        self
    }

    pub fn primary_metric(mut self, metric: EvalMetric) -> Self {
        self.primary_metric = metric;
        self
    }

    pub fn build(self) -> EvalHarness {
        EvalHarness {
            task_suite: self.task_suite,
            baseline_config: self.baseline_config,
            candidate_configs: self.candidate_configs,
            n_runs_per_config: self.n_runs_per_config,
            metrics: self.metrics,
            observability: self.observability,
            bootstrap_iterations: self.bootstrap_iterations,
            primary_metric: self.primary_metric,
        }
    }
}
