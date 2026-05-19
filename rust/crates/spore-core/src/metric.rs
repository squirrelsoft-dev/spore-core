//! Issue #23 — `MetricEvaluator`: pluggable scoring for the `HillClimbing`
//! loop strategy.
//!
//! See `docs/harness-engineering-concepts.md` § "Loop Strategies / HillClimbing"
//! and the issue spec for authoritative rules. This module ships:
//!   - [`MetricError`] / [`MetricResult`] — the error and success surfaces.
//!   - The [`MetricEvaluator`] trait — used by the harness to score each
//!     iteration of a `HillClimbing` run.
//!   - Standard evaluators: [`CommandMetricEvaluator`],
//!     [`TestPassRateEvaluator`], [`LatencyEvaluator`], [`LlmJudgeEvaluator`].
//!   - [`ResultsEntry`] / [`IterationStatus`] — the row format the harness
//!     writes to `.spore/results/{task_id}.tsv`.
//!   - [`should_keep`] — the keep/revert decision the harness applies after
//!     each iteration.
//!
//! ## Rules enforced
//!   - `evaluate` receives the `SandboxProvider`. All subprocess execution
//!     goes through it; evaluators never touch `std::process` directly.
//!   - `CommandMetricEvaluator` writes captured stdout+stderr to
//!     `log_output_to` *before* parsing the metric, so a partial run is still
//!     diagnosable.
//!   - A regex that does not match the captured output is
//!     [`MetricError::ParseFailed`], not a crash.
//!   - Non-zero exit / panic from the subprocess maps to
//!     [`MetricError::Crashed`]; an exceeded timeout maps to
//!     [`MetricError::Timeout`]. Both are valid iteration outcomes — the
//!     harness logs them and asks the agent to try a different approach.
//!   - `should_keep` strictly compares against `current_best`: a delta of
//!     exactly `min_delta` (or `0.0` when unset) does NOT count as
//!     improvement. Equal scores are discarded.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::{BoxFut, OptimizationDirection, SandboxProvider};
use crate::model::{
    Content, ContentBlock, Message, ModelInterface, ModelParams, ModelRequest, Role,
};
use crate::termination::SessionStateSnapshot;

// ============================================================================
// MetricError
// ============================================================================

#[derive(Debug, Clone, Error, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MetricError {
    #[error("execution failed: {reason}")]
    ExecutionFailed { reason: String },
    #[error("evaluator timed out after {after:?}")]
    Timeout {
        #[serde(with = "duration_secs")]
        after: Duration,
    },
    #[error("could not parse metric from output (pattern: {pattern})")]
    ParseFailed { output: String, pattern: String },
    #[error("experiment crashed: {log}")]
    Crashed { log: String },
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(v: &Duration, s: S) -> Result<S::Ok, S::Error> {
        v.as_secs_f64().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs_f64(f64::deserialize(d)?))
    }
}

// ============================================================================
// MetricResult
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricResult {
    pub value: f64,
    #[serde(default)]
    pub raw_output: String,
    #[serde(with = "duration_secs")]
    pub duration: Duration,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl MetricResult {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            raw_output: String::new(),
            duration: Duration::ZERO,
            metadata: BTreeMap::new(),
        }
    }
}

// ============================================================================
// MetricEvaluator trait
// ============================================================================

/// Pluggable scoring strategy for the `HillClimbing` loop. The harness calls
/// [`MetricEvaluator::evaluate`] after the agent completes each iteration and
/// feeds the result into [`should_keep`].
pub trait MetricEvaluator: Send + Sync {
    fn evaluate<'a>(
        &'a self,
        sandbox: &'a dyn SandboxProvider,
        session_state: &'a SessionStateSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>>;

    fn direction(&self) -> OptimizationDirection;

    fn description(&self) -> String;
}

// ============================================================================
// should_keep
// ============================================================================

/// The keep-or-revert decision the harness applies after every iteration.
///
/// Returns `true` only when `new_value` strictly beats `current_best` by more
/// than `min_delta` (default `0.0`). Equal scores are discarded — a flat run
/// is not progress.
pub fn should_keep(
    new_value: f64,
    current_best: f64,
    direction: OptimizationDirection,
    min_delta: Option<f64>,
) -> bool {
    let delta = match direction {
        OptimizationDirection::Minimize => current_best - new_value,
        OptimizationDirection::Maximize => new_value - current_best,
    };
    delta > min_delta.unwrap_or(0.0)
}

// ============================================================================
// ResultsEntry / IterationStatus
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IterationStatus {
    Kept,
    Discarded,
    Crashed,
    Timeout,
}

impl IterationStatus {
    /// Map an evaluator outcome to the iteration status the harness records.
    /// Successful evaluations are then routed through [`should_keep`] to
    /// resolve [`IterationStatus::Kept`] vs [`IterationStatus::Discarded`].
    pub fn from_error(err: &MetricError) -> Self {
        match err {
            MetricError::Timeout { .. } => IterationStatus::Timeout,
            _ => IterationStatus::Crashed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultsEntry {
    pub iteration: u32,
    #[serde(default)]
    pub commit_hash: Option<String>,
    pub metric_value: f64,
    pub direction: OptimizationDirection,
    pub status: IterationStatus,
    #[serde(with = "duration_secs")]
    pub duration: Duration,
    pub description: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

// ============================================================================
// Internal helpers
// ============================================================================

fn parse_metric(output: &str, pattern: &str) -> Result<f64, MetricError> {
    let re = Regex::new(pattern).map_err(|e| MetricError::ExecutionFailed {
        reason: format!("invalid regex {pattern:?}: {e}"),
    })?;
    let captures = re
        .captures(output)
        .ok_or_else(|| MetricError::ParseFailed {
            output: output.to_string(),
            pattern: pattern.to_string(),
        })?;
    let group = captures
        .get(1)
        .ok_or_else(|| MetricError::ParseFailed {
            output: output.to_string(),
            pattern: pattern.to_string(),
        })?
        .as_str();
    group
        .trim()
        .parse::<f64>()
        .map_err(|_| MetricError::ParseFailed {
            output: output.to_string(),
            pattern: pattern.to_string(),
        })
}

async fn write_log(sandbox: &dyn SandboxProvider, path: &PathBuf, body: &str) {
    let _ = tokio::fs::write(sandbox.workspace_root().join(path), body).await;
}

// ============================================================================
// CommandMetricEvaluator
// ============================================================================

/// Runs a shell command through the sandbox, parses a numeric metric out of
/// its combined stdout+stderr via a single-capture-group regex.
///
/// Models the autoresearch pattern (`uv run train.py` ⇒ `val_bpb`).
#[derive(Debug, Clone)]
pub struct CommandMetricEvaluator {
    pub command: String,
    pub args: Vec<String>,
    pub metric_pattern: String,
    pub timeout: Duration,
    pub log_output_to: PathBuf,
    pub working_dir: Option<PathBuf>,
    pub direction: OptimizationDirection,
    pub description: String,
}

impl MetricEvaluator for CommandMetricEvaluator {
    fn evaluate<'a>(
        &'a self,
        sandbox: &'a dyn SandboxProvider,
        _session_state: &'a SessionStateSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
        Box::pin(async move {
            let start = Instant::now();
            let out = sandbox
                .execute_command(
                    &self.command,
                    &self.args,
                    self.working_dir.as_deref(),
                    Some(self.timeout),
                )
                .await
                .map_err(|v| MetricError::ExecutionFailed {
                    reason: format!("sandbox rejected command: {v:?}"),
                })?;

            let combined = format!("{}{}", out.stdout, out.stderr);
            // Always redirect to log_output_to BEFORE parsing so a parse
            // failure is still diagnosable.
            write_log(sandbox, &self.log_output_to, &combined).await;

            if out.timed_out {
                return Err(MetricError::Timeout {
                    after: self.timeout,
                });
            }
            if out.exit_code != 0 {
                return Err(MetricError::Crashed { log: combined });
            }

            let value = parse_metric(&combined, &self.metric_pattern)?;
            let mut metadata = BTreeMap::new();
            metadata.insert("command".into(), self.command.clone());
            metadata.insert("exit_code".into(), out.exit_code.to_string());
            Ok(MetricResult {
                value,
                raw_output: combined,
                duration: start.elapsed(),
                metadata,
            })
        })
    }

    fn direction(&self) -> OptimizationDirection {
        self.direction
    }

    fn description(&self) -> String {
        self.description.clone()
    }
}

// ============================================================================
// TestPassRateEvaluator
// ============================================================================

/// Runs a test suite, extracts pass / total counts via two regexes, reports
/// the fraction of passing tests in `[0.0, 1.0]`. Direction is fixed to
/// `Maximize`.
#[derive(Debug, Clone)]
pub struct TestPassRateEvaluator {
    pub command: String,
    pub args: Vec<String>,
    pub timeout: Duration,
    pub pass_pattern: String,
    pub total_pattern: String,
    pub working_dir: Option<PathBuf>,
}

impl MetricEvaluator for TestPassRateEvaluator {
    fn evaluate<'a>(
        &'a self,
        sandbox: &'a dyn SandboxProvider,
        _session_state: &'a SessionStateSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
        Box::pin(async move {
            let start = Instant::now();
            let out = sandbox
                .execute_command(
                    &self.command,
                    &self.args,
                    self.working_dir.as_deref(),
                    Some(self.timeout),
                )
                .await
                .map_err(|v| MetricError::ExecutionFailed {
                    reason: format!("sandbox rejected command: {v:?}"),
                })?;
            let combined = format!("{}{}", out.stdout, out.stderr);

            if out.timed_out {
                return Err(MetricError::Timeout {
                    after: self.timeout,
                });
            }
            // A failing test run is a normal outcome here — we still want the
            // pass-rate. Only treat the run as crashed if we cannot parse it.

            let pass = parse_metric(&combined, &self.pass_pattern)?;
            let total = parse_metric(&combined, &self.total_pattern)?;
            if total <= 0.0 {
                return Err(MetricError::ParseFailed {
                    output: combined,
                    pattern: self.total_pattern.clone(),
                });
            }
            let value = pass / total;
            let mut metadata = BTreeMap::new();
            metadata.insert("pass".into(), pass.to_string());
            metadata.insert("total".into(), total.to_string());
            Ok(MetricResult {
                value,
                raw_output: combined,
                duration: start.elapsed(),
                metadata,
            })
        })
    }

    fn direction(&self) -> OptimizationDirection {
        OptimizationDirection::Maximize
    }

    fn description(&self) -> String {
        format!("test pass rate ({})", self.command)
    }
}

// ============================================================================
// LatencyEvaluator
// ============================================================================

/// Measures wall-clock latency of `command`, averaged over `measured_runs`
/// trials after `warmup_runs` warm-ups. Direction is fixed to `Minimize`.
#[derive(Debug, Clone)]
pub struct LatencyEvaluator {
    pub command: String,
    pub args: Vec<String>,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub timeout: Duration,
    pub working_dir: Option<PathBuf>,
}

impl MetricEvaluator for LatencyEvaluator {
    fn evaluate<'a>(
        &'a self,
        sandbox: &'a dyn SandboxProvider,
        _session_state: &'a SessionStateSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
        Box::pin(async move {
            if self.measured_runs == 0 {
                return Err(MetricError::ExecutionFailed {
                    reason: "measured_runs must be > 0".into(),
                });
            }
            let start = Instant::now();

            for _ in 0..self.warmup_runs {
                let _ = sandbox
                    .execute_command(
                        &self.command,
                        &self.args,
                        self.working_dir.as_deref(),
                        Some(self.timeout),
                    )
                    .await
                    .map_err(|v| MetricError::ExecutionFailed {
                        reason: format!("sandbox rejected command: {v:?}"),
                    })?;
            }

            let mut total = Duration::ZERO;
            let mut last_output = String::new();
            for _ in 0..self.measured_runs {
                let trial_start = Instant::now();
                let out = sandbox
                    .execute_command(
                        &self.command,
                        &self.args,
                        self.working_dir.as_deref(),
                        Some(self.timeout),
                    )
                    .await
                    .map_err(|v| MetricError::ExecutionFailed {
                        reason: format!("sandbox rejected command: {v:?}"),
                    })?;
                if out.timed_out {
                    return Err(MetricError::Timeout {
                        after: self.timeout,
                    });
                }
                if out.exit_code != 0 {
                    return Err(MetricError::Crashed {
                        log: format!("{}{}", out.stdout, out.stderr),
                    });
                }
                total += trial_start.elapsed();
                last_output = format!("{}{}", out.stdout, out.stderr);
            }

            let avg_secs = total.as_secs_f64() / self.measured_runs as f64;
            let mut metadata = BTreeMap::new();
            metadata.insert("warmup_runs".into(), self.warmup_runs.to_string());
            metadata.insert("measured_runs".into(), self.measured_runs.to_string());
            Ok(MetricResult {
                value: avg_secs,
                raw_output: last_output,
                duration: start.elapsed(),
                metadata,
            })
        })
    }

    fn direction(&self) -> OptimizationDirection {
        OptimizationDirection::Minimize
    }

    fn description(&self) -> String {
        format!("latency ({})", self.command)
    }
}

// ============================================================================
// LlmJudgeEvaluator
// ============================================================================

/// Uses an LLM-as-judge to score `sample_input` against `rubric`. The judge
/// is expected to emit a `score: <number>` line; that number is normalized
/// into `[0.0, 1.0]` using `score_range`. Direction is fixed to `Maximize`.
///
/// The trait shape from the spec only carries a [`JudgeModelConfig`]; the
/// concrete [`ModelInterface`] used to dispatch the judge call is supplied
/// at construction time. That keeps the evaluator independent of model
/// routing while still letting `JudgeModelConfig` flow through the results
/// log for observability.
#[derive(Debug, Clone)]
pub struct JudgeModelConfig {
    pub provider: String,
    pub model_id: String,
    pub params: ModelParams,
}

pub struct LlmJudgeEvaluator<M: ModelInterface> {
    pub judge_model: JudgeModelConfig,
    pub rubric: String,
    pub score_range: (f64, f64),
    pub sample_input: String,
    pub client: Arc<M>,
}

impl<M: ModelInterface> LlmJudgeEvaluator<M> {
    fn parse_score(&self, text: &str) -> Result<f64, MetricError> {
        // `score:\s*<number>` — first match wins.
        let raw = parse_metric(text, r"(?i)score\s*:\s*([-+]?\d+(?:\.\d+)?)")?;
        let (lo, hi) = self.score_range;
        if hi <= lo {
            return Err(MetricError::ExecutionFailed {
                reason: format!("invalid score_range: ({lo}, {hi})"),
            });
        }
        let clamped = raw.clamp(lo, hi);
        Ok((clamped - lo) / (hi - lo))
    }
}

impl<M: ModelInterface + 'static> MetricEvaluator for LlmJudgeEvaluator<M> {
    fn evaluate<'a>(
        &'a self,
        _sandbox: &'a dyn SandboxProvider,
        _session_state: &'a SessionStateSnapshot,
    ) -> BoxFut<'a, Result<MetricResult, MetricError>> {
        Box::pin(async move {
            let start = Instant::now();
            let prompt = format!(
                "{}\n\nInput to evaluate:\n{}\n\nReply with a single line `score: <number>` where the number is within {:?}.",
                self.rubric, self.sample_input, self.score_range
            );
            let request = ModelRequest {
                messages: vec![Message {
                    role: Role::User,
                    content: Content::Text { text: prompt },
                }],
                tools: Vec::new(),
                params: self.judge_model.params.clone(),
                stream: false,
            };
            let response =
                self.client
                    .call(request)
                    .await
                    .map_err(|e| MetricError::ExecutionFailed {
                        reason: format!("judge model call failed: {e}"),
                    })?;

            let text = response
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            let value = self.parse_score(&text)?;
            let mut metadata = BTreeMap::new();
            metadata.insert("judge_model".into(), self.judge_model.model_id.clone());
            metadata.insert("judge_provider".into(), self.judge_model.provider.clone());
            Ok(MetricResult {
                value,
                raw_output: text,
                duration: start.elapsed(),
                metadata,
            })
        })
    }

    fn direction(&self) -> OptimizationDirection {
        OptimizationDirection::Maximize
    }

    fn description(&self) -> String {
        format!(
            "llm judge ({}/{})",
            self.judge_model.provider, self.judge_model.model_id
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{SessionId, SessionState, TaskId};
    use crate::model::{
        ContentBlock as MCB, ModelError, ModelResponse, ProviderInfo, StopReason, TokenUsage,
    };
    use std::path::Path;

    fn snapshot() -> SessionStateSnapshot {
        SessionStateSnapshot::new(
            SessionId::new("sess"),
            TaskId::new("task"),
            SessionState::default(),
            std::path::PathBuf::new(),
        )
    }

    // ---------- should_keep ----------

    #[test]
    fn should_keep_minimize_lower_is_better() {
        assert!(should_keep(1.0, 2.0, OptimizationDirection::Minimize, None));
        assert!(!should_keep(
            2.0,
            1.0,
            OptimizationDirection::Minimize,
            None
        ));
    }

    #[test]
    fn should_keep_maximize_higher_is_better() {
        assert!(should_keep(2.0, 1.0, OptimizationDirection::Maximize, None));
        assert!(!should_keep(
            1.0,
            2.0,
            OptimizationDirection::Maximize,
            None
        ));
    }

    #[test]
    fn should_keep_equal_is_discarded() {
        assert!(!should_keep(
            1.0,
            1.0,
            OptimizationDirection::Minimize,
            None
        ));
        assert!(!should_keep(
            1.0,
            1.0,
            OptimizationDirection::Maximize,
            None
        ));
    }

    #[test]
    fn should_keep_respects_min_delta() {
        // improvement of 0.5 with min_delta 0.5 must be discarded
        assert!(!should_keep(
            1.5,
            2.0,
            OptimizationDirection::Minimize,
            Some(0.5)
        ));
        assert!(should_keep(
            1.49,
            2.0,
            OptimizationDirection::Minimize,
            Some(0.5)
        ));
    }

    // ---------- parse_metric ----------

    #[test]
    fn parse_metric_extracts_capture_group() {
        let v = parse_metric("val_bpb:  3.125\nother", r"val_bpb:\s+([\d.]+)").unwrap();
        assert!((v - 3.125).abs() < 1e-9);
    }

    #[test]
    fn parse_metric_no_match_is_parse_failed() {
        let err = parse_metric("no metric here", r"val_bpb:\s+([\d.]+)").unwrap_err();
        assert!(matches!(err, MetricError::ParseFailed { .. }));
    }

    #[test]
    fn parse_metric_unparseable_capture_is_parse_failed() {
        let err = parse_metric("val_bpb: oops", r"val_bpb:\s+(\S+)").unwrap_err();
        assert!(matches!(err, MetricError::ParseFailed { .. }));
    }

    #[test]
    fn parse_metric_invalid_regex_is_execution_failed() {
        let err = parse_metric("x", "(unbalanced").unwrap_err();
        assert!(matches!(err, MetricError::ExecutionFailed { .. }));
    }

    // ---------- IterationStatus ----------

    #[test]
    fn iteration_status_from_error_maps_timeout() {
        let s = IterationStatus::from_error(&MetricError::Timeout {
            after: Duration::from_secs(1),
        });
        assert_eq!(s, IterationStatus::Timeout);
    }

    #[test]
    fn iteration_status_from_error_maps_others_to_crashed() {
        for err in [
            MetricError::Crashed { log: "x".into() },
            MetricError::ExecutionFailed { reason: "x".into() },
            MetricError::ParseFailed {
                output: "".into(),
                pattern: "".into(),
            },
        ] {
            assert_eq!(IterationStatus::from_error(&err), IterationStatus::Crashed);
        }
    }

    // ---------- evaluator integration with a fake sandbox ----------

    struct FakeSandbox {
        stdout: String,
        stderr: String,
        exit_code: i32,
        timed_out: bool,
        root: tempfile::TempDir,
    }

    impl FakeSandbox {
        fn new(stdout: &str) -> Self {
            Self {
                stdout: stdout.into(),
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
                root: tempfile::tempdir().unwrap(),
            }
        }
    }

    impl SandboxProvider for FakeSandbox {
        fn validate<'a>(
            &'a self,
            _call: &'a crate::model::ToolCall,
        ) -> BoxFut<'a, Result<(), crate::harness::SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }

        fn execute_command<'a>(
            &'a self,
            _command: &'a str,
            _args: &'a [String],
            _working_dir: Option<&'a Path>,
            _timeout: Option<Duration>,
        ) -> BoxFut<'a, Result<crate::harness::CommandOutput, crate::harness::SandboxViolation>>
        {
            let out = crate::harness::CommandOutput {
                stdout: self.stdout.clone(),
                stderr: self.stderr.clone(),
                exit_code: self.exit_code,
                timed_out: self.timed_out,
                truncated: false,
            };
            Box::pin(async move { Ok(out) })
        }

        fn workspace_root(&self) -> &Path {
            self.root.path()
        }
    }

    #[tokio::test]
    async fn command_evaluator_happy_path_writes_log_and_parses() {
        let sb = FakeSandbox::new("val_bpb: 1.234\n");
        let eval = CommandMetricEvaluator {
            command: "uv".into(),
            args: vec!["run".into(), "train.py".into()],
            metric_pattern: r"val_bpb:\s+([\d.]+)".into(),
            timeout: Duration::from_secs(60),
            log_output_to: PathBuf::from("run.log"),
            working_dir: None,
            direction: OptimizationDirection::Minimize,
            description: "autoresearch val_bpb".into(),
        };
        let r = eval.evaluate(&sb, &snapshot()).await.unwrap();
        assert!((r.value - 1.234).abs() < 1e-9);
        let log = tokio::fs::read_to_string(sb.root.path().join("run.log"))
            .await
            .unwrap();
        assert!(log.contains("val_bpb"));
        assert_eq!(eval.direction(), OptimizationDirection::Minimize);
    }

    #[tokio::test]
    async fn command_evaluator_timeout_maps_to_timeout_error() {
        let mut sb = FakeSandbox::new("");
        sb.timed_out = true;
        let eval = CommandMetricEvaluator {
            command: "x".into(),
            args: vec![],
            metric_pattern: r"v:(\d+)".into(),
            timeout: Duration::from_millis(1),
            log_output_to: PathBuf::from("run.log"),
            working_dir: None,
            direction: OptimizationDirection::Minimize,
            description: "x".into(),
        };
        let err = eval.evaluate(&sb, &snapshot()).await.unwrap_err();
        assert!(matches!(err, MetricError::Timeout { .. }));
    }

    #[tokio::test]
    async fn command_evaluator_nonzero_exit_is_crashed() {
        let mut sb = FakeSandbox::new("boom");
        sb.exit_code = 1;
        let eval = CommandMetricEvaluator {
            command: "x".into(),
            args: vec![],
            metric_pattern: r"v:(\d+)".into(),
            timeout: Duration::from_secs(1),
            log_output_to: PathBuf::from("run.log"),
            working_dir: None,
            direction: OptimizationDirection::Minimize,
            description: "x".into(),
        };
        let err = eval.evaluate(&sb, &snapshot()).await.unwrap_err();
        assert!(matches!(err, MetricError::Crashed { .. }));
    }

    #[tokio::test]
    async fn command_evaluator_parse_failed_when_regex_doesnt_match() {
        let sb = FakeSandbox::new("no metric");
        let eval = CommandMetricEvaluator {
            command: "x".into(),
            args: vec![],
            metric_pattern: r"v:(\d+)".into(),
            timeout: Duration::from_secs(1),
            log_output_to: PathBuf::from("run.log"),
            working_dir: None,
            direction: OptimizationDirection::Minimize,
            description: "x".into(),
        };
        let err = eval.evaluate(&sb, &snapshot()).await.unwrap_err();
        assert!(matches!(err, MetricError::ParseFailed { .. }));
    }

    #[tokio::test]
    async fn test_pass_rate_evaluator_returns_fraction() {
        let sb = FakeSandbox::new("passed 17 of 20");
        let eval = TestPassRateEvaluator {
            command: "pytest".into(),
            args: vec![],
            timeout: Duration::from_secs(60),
            pass_pattern: r"passed (\d+)".into(),
            total_pattern: r"of (\d+)".into(),
            working_dir: None,
        };
        let r = eval.evaluate(&sb, &snapshot()).await.unwrap();
        assert!((r.value - 0.85).abs() < 1e-9);
        assert_eq!(eval.direction(), OptimizationDirection::Maximize);
    }

    #[tokio::test]
    async fn latency_evaluator_averages_runs() {
        let sb = FakeSandbox::new("ok");
        let eval = LatencyEvaluator {
            command: "echo".into(),
            args: vec!["ok".into()],
            warmup_runs: 1,
            measured_runs: 2,
            timeout: Duration::from_secs(5),
            working_dir: None,
        };
        let r = eval.evaluate(&sb, &snapshot()).await.unwrap();
        assert!(r.value >= 0.0);
        assert_eq!(eval.direction(), OptimizationDirection::Minimize);
        assert_eq!(r.metadata.get("measured_runs").unwrap(), "2");
    }

    #[tokio::test]
    async fn latency_evaluator_zero_measured_runs_rejects() {
        let sb = FakeSandbox::new("");
        let eval = LatencyEvaluator {
            command: "x".into(),
            args: vec![],
            warmup_runs: 0,
            measured_runs: 0,
            timeout: Duration::from_secs(1),
            working_dir: None,
        };
        let err = eval.evaluate(&sb, &snapshot()).await.unwrap_err();
        assert!(matches!(err, MetricError::ExecutionFailed { .. }));
    }

    // ---------- LlmJudgeEvaluator ----------

    struct FakeModel {
        text: String,
    }

    impl ModelInterface for FakeModel {
        async fn call(&self, _req: ModelRequest) -> Result<ModelResponse, ModelError> {
            Ok(ModelResponse {
                content: vec![MCB::Text {
                    text: self.text.clone(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: TokenUsage::default(),
            })
        }

        async fn call_streaming(
            &self,
            _req: ModelRequest,
        ) -> Result<crate::model::ModelStream, ModelError> {
            unimplemented!()
        }

        async fn count_tokens(&self, _req: &ModelRequest) -> Result<u32, ModelError> {
            Ok(0)
        }

        fn provider(&self) -> ProviderInfo {
            ProviderInfo {
                name: "fake".into(),
                model_id: "fake".into(),
                context_window: 8000,
            }
        }
    }

    #[tokio::test]
    async fn llm_judge_normalizes_score_into_unit_range() {
        let sb = FakeSandbox::new("");
        let eval = LlmJudgeEvaluator {
            judge_model: JudgeModelConfig {
                provider: "fake".into(),
                model_id: "judge-1".into(),
                params: ModelParams::default(),
            },
            rubric: "rate this".into(),
            score_range: (0.0, 10.0),
            sample_input: "the answer".into(),
            client: Arc::new(FakeModel {
                text: "score: 7.5".into(),
            }),
        };
        let r = eval.evaluate(&sb, &snapshot()).await.unwrap();
        assert!((r.value - 0.75).abs() < 1e-9);
        assert_eq!(eval.direction(), OptimizationDirection::Maximize);
    }

    #[tokio::test]
    async fn llm_judge_clamps_score_outside_range() {
        let sb = FakeSandbox::new("");
        let eval = LlmJudgeEvaluator {
            judge_model: JudgeModelConfig {
                provider: "fake".into(),
                model_id: "judge-1".into(),
                params: ModelParams::default(),
            },
            rubric: "rate this".into(),
            score_range: (0.0, 10.0),
            sample_input: "x".into(),
            client: Arc::new(FakeModel {
                text: "score: 42".into(),
            }),
        };
        let r = eval.evaluate(&sb, &snapshot()).await.unwrap();
        assert!((r.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn llm_judge_parse_failed_when_no_score() {
        let sb = FakeSandbox::new("");
        let eval = LlmJudgeEvaluator {
            judge_model: JudgeModelConfig {
                provider: "fake".into(),
                model_id: "judge-1".into(),
                params: ModelParams::default(),
            },
            rubric: "rate this".into(),
            score_range: (0.0, 10.0),
            sample_input: "x".into(),
            client: Arc::new(FakeModel {
                text: "no score in here".into(),
            }),
        };
        let err = eval.evaluate(&sb, &snapshot()).await.unwrap_err();
        assert!(matches!(err, MetricError::ParseFailed { .. }));
    }

    // ---------- Fixture replay ----------

    #[derive(Deserialize)]
    struct ShouldKeepFixture {
        cases: Vec<ShouldKeepCase>,
    }
    #[derive(Deserialize)]
    struct ShouldKeepCase {
        name: String,
        new_value: f64,
        current_best: f64,
        direction: OptimizationDirection,
        min_delta: Option<f64>,
        expected: bool,
    }

    #[test]
    fn should_keep_fixture_replay() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/metric_evaluator/should_keep.json");
        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let fx: ShouldKeepFixture = serde_json::from_str(&body).unwrap();
        for c in fx.cases {
            let got = should_keep(c.new_value, c.current_best, c.direction, c.min_delta);
            assert_eq!(got, c.expected, "case {} mismatched", c.name);
        }
    }

    #[derive(Deserialize)]
    struct ParseFixture {
        cases: Vec<ParseCase>,
    }
    #[derive(Deserialize)]
    struct ParseCase {
        name: String,
        output: String,
        pattern: String,
        expected: ParseExpected,
    }
    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum ParseExpected {
        Value { value: f64 },
        ParseFailed,
    }

    #[test]
    fn parse_metric_fixture_replay() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/metric_evaluator/parse_metric.json");
        let body = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let fx: ParseFixture = serde_json::from_str(&body).unwrap();
        for c in fx.cases {
            let got = parse_metric(&c.output, &c.pattern);
            match (got, c.expected) {
                (Ok(v), ParseExpected::Value { value }) => {
                    assert!((v - value).abs() < 1e-9, "case {} value mismatch", c.name);
                }
                (Err(MetricError::ParseFailed { .. }), ParseExpected::ParseFailed) => {}
                (got, expected) => panic!(
                    "case {} mismatched: got {got:?}, expected {expected:?}",
                    c.name
                ),
            }
        }
    }

    impl std::fmt::Debug for ParseExpected {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ParseExpected::Value { value } => write!(f, "Value({value})"),
                ParseExpected::ParseFailed => write!(f, "ParseFailed"),
            }
        }
    }
}
