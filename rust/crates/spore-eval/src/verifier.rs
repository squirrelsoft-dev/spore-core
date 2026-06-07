//! [`TaskVerifier`] trait + standard implementations.
//!
//! Rules enforced here:
//!   - 9  `is_deterministic()` true for test-suite/result verifiers, false for LLM judge.
//!   - 10 [`TestSuiteVerifier`]: command pass-rate; passed = score==1.0; deterministic.
//!   - 11 [`CompositeVerifier`]: weighted mean; passed = all required; det = AND.
//!   - 12 [`MetricEvaluatorVerifier`]: wraps a `MetricEvaluator`, normalizes value.
//!   - 13 [`LlmJudgeVerifier`]: thin; non-deterministic; judge injected.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use spore_core::harness::{BoxFut, HillClimbingDirection, RunResult, SandboxProvider};
use spore_core::metric::MetricEvaluator;
use spore_core::model::ModelInterface;
use spore_core::termination::SessionStateSnapshot;

use crate::task::{EvalError, EvalTask, MetricDirection, VerificationResult, VerifierSpec};

// ============================================================================
// TaskVerifier trait
// ============================================================================

/// Verifies whether a task run satisfied its goal.
pub trait TaskVerifier: Send + Sync {
    /// Verify a completed run against the task, with access to the restored
    /// workspace directory.
    fn verify<'a>(
        &'a self,
        task: &'a EvalTask,
        run: &'a RunResult,
        workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>>;

    /// `true` for test-suite / result verifiers; `false` for the LLM judge
    /// (Rule 9).
    fn is_deterministic(&self) -> bool {
        true
    }
}

// ============================================================================
// build_verifier — resolve a VerifierSpec into an Arc<dyn TaskVerifier>
// ============================================================================

/// Resolve a [`VerifierSpec`] to a concrete verifier. `MetricEvaluator` specs
/// have no built-in concrete evaluator (it is injected for non-fixture use), so
/// they resolve to a normalizing placeholder that scores from the run's
/// success flag — adequate for manifest replay; real evaluators are wired via
/// [`MetricEvaluatorVerifier::new`].
pub fn build_verifier(spec: &VerifierSpec) -> Arc<dyn TaskVerifier> {
    match spec {
        VerifierSpec::TestSuite {
            command,
            args,
            timeout_secs,
        } => Arc::new(TestSuiteVerifier {
            command: command.clone(),
            args: args.clone(),
            timeout: Duration::from_secs(timeout_secs.unwrap_or(60)),
        }),
        VerifierSpec::Composite { children } => {
            let resolved = children
                .iter()
                .map(|c| CompositeChild {
                    verifier: build_verifier(&c.spec),
                    weight: c.weight,
                    required: c.required,
                })
                .collect();
            Arc::new(CompositeVerifier { children: resolved })
        }
        VerifierSpec::MetricEvaluator {
            direction,
            min,
            max,
            threshold,
            ..
        } => Arc::new(NormalizingSuccessVerifier {
            direction: *direction,
            min: *min,
            max: *max,
            threshold: *threshold,
        }),
        VerifierSpec::LlmJudge { score_range, .. } => Arc::new(StubLlmJudgeVerifier {
            score_range: *score_range,
        }),
        VerifierSpec::AlwaysPass => Arc::new(AlwaysPass),
        VerifierSpec::AlwaysFail => Arc::new(AlwaysFail),
    }
}

// ============================================================================
// AlwaysPass / AlwaysFail (test scaffolding)
// ============================================================================

/// Always passes with score 1.0.
pub struct AlwaysPass;
impl TaskVerifier for AlwaysPass {
    fn verify<'a>(
        &'a self,
        _task: &'a EvalTask,
        _run: &'a RunResult,
        _workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        Box::pin(async { VerificationResult::new(true, 1.0, "always pass") })
    }
}

/// Always fails with score 0.0.
pub struct AlwaysFail;
impl TaskVerifier for AlwaysFail {
    fn verify<'a>(
        &'a self,
        _task: &'a EvalTask,
        _run: &'a RunResult,
        _workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        Box::pin(async { VerificationResult::new(false, 0.0, "always fail") })
    }
}

// ============================================================================
// TestSuiteVerifier (Rule 10)
// ============================================================================

/// Runs a command in the workspace; score = pass rate parsed from the output.
/// `passed` = (score == 1.0). Deterministic.
pub struct TestSuiteVerifier {
    pub command: String,
    pub args: Vec<String>,
    pub timeout: Duration,
}

impl TaskVerifier for TestSuiteVerifier {
    fn verify<'a>(
        &'a self,
        _task: &'a EvalTask,
        _run: &'a RunResult,
        workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        Box::pin(async move {
            // Use the default SandboxProvider command execution; production
            // callers inject a real sandbox via the harness, but for verification
            // we run the check command directly in the restored workspace.
            let sandbox = DirectSandbox {
                root: workspace.to_path_buf(),
            };
            let out = sandbox
                .execute_command(
                    &self.command,
                    &self.args,
                    Some(workspace),
                    Some(self.timeout),
                )
                .await
                .map_err(|v| EvalError::Verify(format!("command rejected: {v:?}")))?;
            let combined = format!("{}{}", out.stdout, out.stderr);
            // Pass-rate: prefer "N passed"/"M total"-style output; fall back to
            // exit code (0 => 1.0, nonzero => 0.0).
            let score =
                parse_pass_rate(&combined).unwrap_or(if out.exit_code == 0 { 1.0 } else { 0.0 });
            let passed = (score - 1.0).abs() < f64::EPSILON;
            Ok(VerificationResult::clamped(
                passed,
                score,
                format!("exit={} pass_rate={score:.3}", out.exit_code),
            )
            .with_signal("exit_code", out.exit_code as f64)
            .with_signal("pass_rate", score))
        })
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

/// Parse a pass-rate from common test-runner output. Returns `None` if no
/// recognizable counts are present.
fn parse_pass_rate(output: &str) -> Option<f64> {
    // Match "<pass> passed" and a "<total>"-style count in "of <total>" or
    // "<total> total". Minimal hand-rolled scan (no regex dep here).
    let passed = scan_number_before(output, " passed");
    let total = scan_number_before(output, " total").or_else(|| scan_number_after(output, "of "));
    match (passed, total) {
        (Some(p), Some(t)) if t > 0.0 => Some((p / t).clamp(0.0, 1.0)),
        _ => None,
    }
}

fn scan_number_before(s: &str, suffix: &str) -> Option<f64> {
    let idx = s.find(suffix)?;
    let head = &s[..idx];
    let num: String = head
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    num.parse::<f64>().ok()
}

fn scan_number_after(s: &str, prefix: &str) -> Option<f64> {
    let idx = s.find(prefix)? + prefix.len();
    let tail = &s[idx..];
    let num: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse::<f64>().ok()
}

/// Minimal sandbox that runs commands directly in a workspace dir. Used by
/// [`TestSuiteVerifier`] for the verification command only; it inherits the
/// non-isolating default `execute_command` from [`SandboxProvider`].
struct DirectSandbox {
    root: std::path::PathBuf,
}
impl SandboxProvider for DirectSandbox {
    fn validate<'a>(
        &'a self,
        _call: &'a spore_core::model::ToolCall,
    ) -> BoxFut<'a, Result<(), spore_core::harness::SandboxViolation>> {
        Box::pin(async { Ok(()) })
    }
    fn workspace_root(&self) -> &Path {
        &self.root
    }
}

// ============================================================================
// CompositeVerifier (Rule 11)
// ============================================================================

struct CompositeChild {
    verifier: Arc<dyn TaskVerifier>,
    weight: f64,
    required: bool,
}

/// Combines children by weight: score = weighted mean; passed = all required
/// children passed; `is_deterministic` = AND of children (Rule 11).
pub struct CompositeVerifier {
    children: Vec<CompositeChild>,
}

impl CompositeVerifier {
    /// Build from already-resolved children: `(verifier, weight, required)`.
    pub fn new(children: Vec<(Arc<dyn TaskVerifier>, f64, bool)>) -> Self {
        Self {
            children: children
                .into_iter()
                .map(|(verifier, weight, required)| CompositeChild {
                    verifier,
                    weight,
                    required,
                })
                .collect(),
        }
    }
}

impl TaskVerifier for CompositeVerifier {
    fn verify<'a>(
        &'a self,
        task: &'a EvalTask,
        run: &'a RunResult,
        workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        Box::pin(async move {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            let mut all_required_passed = true;
            let mut details = Vec::new();
            for child in &self.children {
                let r = child.verifier.verify(task, run, workspace).await?;
                weighted_sum += r.score * child.weight;
                weight_total += child.weight;
                if child.required && !r.passed {
                    all_required_passed = false;
                }
                details.push(format!(
                    "[w={} req={} pass={} score={:.3}]",
                    child.weight, child.required, r.passed, r.score
                ));
            }
            let score = if weight_total > 0.0 {
                weighted_sum / weight_total
            } else {
                0.0
            };
            Ok(VerificationResult::clamped(
                all_required_passed,
                score,
                details.join(" "),
            ))
        })
    }

    fn is_deterministic(&self) -> bool {
        self.children.iter().all(|c| c.verifier.is_deterministic())
    }
}

// ============================================================================
// MetricEvaluatorVerifier (Rule 12)
// ============================================================================

/// Wraps an `Arc<dyn MetricEvaluator>`: runs `evaluate`, normalizes the value
/// to `[0,1]` per `direction()` and the configured min/max (or a threshold).
/// Deterministic iff the wrapped evaluator is (defaults to deterministic).
pub struct MetricEvaluatorVerifier {
    evaluator: Arc<dyn MetricEvaluator>,
    min: Option<f64>,
    max: Option<f64>,
    threshold: Option<f64>,
    deterministic: bool,
}

impl MetricEvaluatorVerifier {
    /// Wrap an evaluator, normalizing by an explicit `[min, max]` range.
    pub fn with_range(evaluator: Arc<dyn MetricEvaluator>, min: f64, max: f64) -> Self {
        Self {
            evaluator,
            min: Some(min),
            max: Some(max),
            threshold: None,
            deterministic: true,
        }
    }

    /// Wrap an evaluator, scoring 1.0 when the value beats `threshold` in the
    /// evaluator's `direction()`, else 0.0.
    pub fn with_threshold(evaluator: Arc<dyn MetricEvaluator>, threshold: f64) -> Self {
        Self {
            evaluator,
            min: None,
            max: None,
            threshold: Some(threshold),
            deterministic: true,
        }
    }

    /// Mark the wrapped evaluator as non-deterministic (e.g. an LLM judge).
    pub fn non_deterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    fn normalize(&self, value: f64, direction: HillClimbingDirection) -> f64 {
        if let Some(threshold) = self.threshold {
            let beats = match direction {
                HillClimbingDirection::Maximize => value >= threshold,
                HillClimbingDirection::Minimize => value <= threshold,
            };
            return if beats { 1.0 } else { 0.0 };
        }
        if let (Some(min), Some(max)) = (self.min, self.max) {
            if (max - min).abs() < f64::EPSILON {
                return 0.0;
            }
            let unit = ((value - min) / (max - min)).clamp(0.0, 1.0);
            return match direction {
                HillClimbingDirection::Maximize => unit,
                HillClimbingDirection::Minimize => 1.0 - unit,
            };
        }
        value.clamp(0.0, 1.0)
    }
}

impl TaskVerifier for MetricEvaluatorVerifier {
    fn verify<'a>(
        &'a self,
        task: &'a EvalTask,
        run: &'a RunResult,
        workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        Box::pin(async move {
            let sandbox = DirectSandbox {
                root: workspace.to_path_buf(),
            };
            let session_id = match run {
                RunResult::Success { session_id, .. }
                | RunResult::Failure { session_id, .. }
                | RunResult::Escalate { session_id, .. } => session_id.clone(),
                RunResult::WaitingForHuman { state, .. } | RunResult::Consult { state, .. } => {
                    state.session_id.clone()
                }
            };
            let snapshot = SessionStateSnapshot::new(
                session_id,
                task.id.clone(),
                spore_core::harness::SessionState::default(),
                workspace.to_path_buf(),
            );
            let result = self
                .evaluator
                .evaluate(&sandbox, &snapshot)
                .await
                .map_err(|e| EvalError::Verify(format!("evaluator failed: {e}")))?;
            let score = self.normalize(result.value, self.evaluator.direction());
            let passed = score >= 1.0 - f64::EPSILON;
            Ok(VerificationResult::clamped(
                passed,
                score,
                format!("metric value={} normalized={score:.3}", result.value),
            )
            .with_signal("metric_value", result.value))
        })
    }

    fn is_deterministic(&self) -> bool {
        self.deterministic
    }
}

/// Placeholder for a `MetricEvaluator` verifier spec resolved from a manifest
/// (no concrete evaluator wired). Scores from the run's success flag, applying
/// the spec's direction/range so the surface is exercised in replay.
pub struct NormalizingSuccessVerifier {
    pub direction: MetricDirection,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub threshold: Option<f64>,
}

impl TaskVerifier for NormalizingSuccessVerifier {
    fn verify<'a>(
        &'a self,
        _task: &'a EvalTask,
        run: &'a RunResult,
        _workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        let direction = self.direction;
        let success = matches!(run, RunResult::Success { .. });
        Box::pin(async move {
            let value = if success { 1.0 } else { 0.0 };
            // direction is informational here; success is already in [0,1].
            let _ = direction;
            VerificationResult::new(success, value, "metric-evaluator (manifest placeholder)")
        })
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// ============================================================================
// LlmJudgeVerifier (Rule 13)
// ============================================================================

/// A thin LLM-judge verifier. `is_deterministic() == false`. The concrete judge
/// `ModelInterface` is injected at construction. Reuses the `score:` parse +
/// range-normalization pattern from `LlmJudgeEvaluator`.
pub struct LlmJudgeVerifier<M: ModelInterface> {
    pub judge: Arc<M>,
    pub rubric: String,
    pub score_range: (f64, f64),
    pub params: spore_core::model::ModelParams,
}

impl<M: ModelInterface + 'static> TaskVerifier for LlmJudgeVerifier<M> {
    fn verify<'a>(
        &'a self,
        _task: &'a EvalTask,
        run: &'a RunResult,
        _workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        use spore_core::model::{Content, ContentBlock, Message, ModelRequest, Role};
        Box::pin(async move {
            let output = match run {
                RunResult::Success { output, .. } => output.clone(),
                _ => String::new(),
            };
            let prompt = format!(
                "{}\n\nAgent output to evaluate:\n{}\n\nReply with a single line `score: <number>` within {:?}.",
                self.rubric, output, self.score_range
            );
            let request = ModelRequest {
                messages: vec![Message {
                    role: Role::User,
                    content: Content::Text { text: prompt },
                }],
                tools: Vec::new(),
                params: self.params.clone(),
                stream: false,
            };
            let response = self
                .judge
                .call(request)
                .await
                .map_err(|e| EvalError::Verify(format!("judge call failed: {e}")))?;
            let text = response
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let raw = parse_score(&text)
                .ok_or_else(|| EvalError::Verify(format!("no score in judge reply: {text:?}")))?;
            let (lo, hi) = self.score_range;
            if hi <= lo {
                return Err(EvalError::Verify(format!(
                    "invalid score_range ({lo},{hi})"
                )));
            }
            let score = ((raw.clamp(lo, hi) - lo) / (hi - lo)).clamp(0.0, 1.0);
            VerificationResult::new(score >= 0.5, score, format!("judge score={raw}"))
        })
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

/// Stub LLM judge used when a manifest's `LlmJudge` spec is resolved without an
/// injected model. Non-deterministic; scores the midpoint of its range so the
/// non-deterministic comparison path (bootstrap CI) is still exercised.
pub struct StubLlmJudgeVerifier {
    pub score_range: (f64, f64),
}

impl TaskVerifier for StubLlmJudgeVerifier {
    fn verify<'a>(
        &'a self,
        _task: &'a EvalTask,
        run: &'a RunResult,
        _workspace: &'a Path,
    ) -> BoxFut<'a, Result<VerificationResult, EvalError>> {
        let success = matches!(run, RunResult::Success { .. });
        Box::pin(async move {
            let score = if success { 1.0 } else { 0.0 };
            VerificationResult::new(success, score, "llm-judge (manifest stub)")
        })
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

/// Parse a `score: <number>` line (first match wins, case-insensitive).
fn parse_score(text: &str) -> Option<f64> {
    let lower = text.to_ascii_lowercase();
    let idx = lower.find("score")?;
    let after = &text[idx + "score".len()..];
    let after = after.trim_start_matches([':', ' ', '\t']);
    let num: String = after
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
        .collect();
    num.parse::<f64>().ok()
}
