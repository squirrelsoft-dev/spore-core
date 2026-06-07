//! Task-suite types: [`TaskSuite`], [`EvalTask`], [`WorkspaceSnapshot`],
//! [`VerifierSpec`], [`VerificationResult`], [`ConfigId`], and the
//! [`EvalError`] enum.
//!
//! Rules enforced here: 1 (three disjoint lists), 5 (tags free-form), 6
//! (`suite_version` required), 7/8 (verification result shape + score clamp).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use spore_core::harness::TaskId;
use thiserror::Error;

use crate::verifier::TaskVerifier;

// ============================================================================
// ConfigId
// ============================================================================

/// Identifies a candidate harness configuration in a comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConfigId(pub String);

impl ConfigId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ============================================================================
// EvalError (Rule: thiserror, #[non_exhaustive])
// ============================================================================

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EvalError {
    /// A manifest was loaded without the required `suite_version` (Rule 6).
    #[error("manifest is missing required field `suite_version`")]
    MissingSuiteVersion,
    /// A manifest failed to parse.
    #[error("manifest parse error: {0}")]
    ManifestParse(String),
    /// A verifier failed in a way that is not a normal "task failed" outcome
    /// (e.g. an out-of-range score, Rule 8).
    #[error("verification error: {0}")]
    Verify(String),
    /// Restoring or tearing down a workspace/worktree failed (Rules 2-3).
    #[error("worktree error: {0}")]
    Worktree(String),
    /// An `EvalHarness` was built or run without the metrics it needs.
    #[error("missing metrics: {0}")]
    MissingMetrics(String),
    /// An I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

// ============================================================================
// VerificationResult (Rule 7)
// ============================================================================

/// The outcome of a [`TaskVerifier`] (Rule 7): a pass/fail flag, a `score`
/// clamped to `[0.0, 1.0]`, a human-readable `detail`, and granular `signals`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passed: bool,
    pub score: f64,
    pub detail: String,
    #[serde(default)]
    pub signals: BTreeMap<String, f64>,
}

impl VerificationResult {
    /// Build a result, clamping `score` to `[0.0, 1.0]`. An out-of-range score
    /// is an error per Rule 8, surfaced as [`EvalError::Verify`].
    pub fn new(passed: bool, score: f64, detail: impl Into<String>) -> Result<Self, EvalError> {
        if !(0.0..=1.0).contains(&score) {
            return Err(EvalError::Verify(format!(
                "score {score} out of range [0.0, 1.0]"
            )));
        }
        Ok(Self {
            passed,
            score,
            detail: detail.into(),
            signals: BTreeMap::new(),
        })
    }

    /// Build a result, clamping any out-of-range `score` into `[0.0, 1.0]`
    /// instead of erroring. Use for evaluator-derived scores that are
    /// guaranteed-finite but may drift slightly outside the unit interval.
    pub fn clamped(passed: bool, score: f64, detail: impl Into<String>) -> Self {
        Self {
            passed,
            score: score.clamp(0.0, 1.0),
            detail: detail.into(),
            signals: BTreeMap::new(),
        }
    }

    pub fn with_signal(mut self, key: impl Into<String>, value: f64) -> Self {
        self.signals.insert(key.into(), value);
        self
    }
}

// ============================================================================
// WorkspaceSnapshot (Resolution 2)
// ============================================================================

/// How a task's workspace is restored before a run (Rule 2).
///
/// [`WorkspaceSnapshot::Files`] is the canonical hermetic form the shipped
/// fixtures use — no real git repo is needed for cross-language replay.
/// [`WorkspaceSnapshot::GitRef`] is supported for real snapshots (init a
/// throwaway repo + `git worktree add`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkspaceSnapshot {
    /// Canonical hermetic form: a map of relative path → file contents.
    Files { files: BTreeMap<String, String> },
    /// A real git snapshot: a repo URL/path and a ref to check out.
    GitRef { repo: String, reference: String },
    /// An empty workspace.
    Empty,
}

// ============================================================================
// VerifierSpec (serializable; resolved to Arc<dyn TaskVerifier>)
// ============================================================================

/// A serializable description of a verifier. Resolved to an
/// `Arc<dyn TaskVerifier>` by [`crate::verifier::build_verifier`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VerifierSpec {
    /// Run a command in the workspace; score = pass rate (Rule 10).
    TestSuite {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        timeout_secs: Option<u64>,
    },
    /// Combine children by weight; `required` children must all pass (Rule 11).
    Composite { children: Vec<CompositeChildSpec> },
    /// Adapt a metric evaluator, normalizing its value to a score (Rule 12).
    MetricEvaluator {
        /// Free-form descriptor of the wrapped evaluator (informational; the
        /// concrete evaluator is injected at build time for non-fixture use).
        descriptor: String,
        direction: MetricDirection,
        #[serde(default)]
        min: Option<f64>,
        #[serde(default)]
        max: Option<f64>,
        #[serde(default)]
        threshold: Option<f64>,
    },
    /// An LLM-judge verifier; non-deterministic (Rule 13).
    LlmJudge {
        rubric: String,
        score_range: (f64, f64),
    },
    /// Test scaffolding: always passes with score 1.0.
    AlwaysPass,
    /// Test scaffolding: always fails with score 0.0.
    AlwaysFail,
}

/// One child of a [`VerifierSpec::Composite`] with its weight and required-ness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeChildSpec {
    pub spec: VerifierSpec,
    pub weight: f64,
    #[serde(default)]
    pub required: bool,
}

/// Optimization direction for a metric-evaluator verifier. Mirrors
/// [`spore_core::harness::HillClimbingDirection`] but is serializable here as a
/// self-contained spec field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    Minimize,
    Maximize,
}

// ============================================================================
// TaskCategory + EvalTask + TaskSuite
// ============================================================================

/// Which of the three disjoint task lists a task belongs to (Rule 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskCategory {
    Regression,
    Challenge,
    Canary,
}

/// One evaluation task.
#[derive(Clone, Serialize, Deserialize)]
pub struct EvalTask {
    pub id: TaskId,
    pub instruction: String,
    pub workspace_snapshot: WorkspaceSnapshot,
    /// The resolved verifier. Skipped in serde; rebuilt from `verifier_spec`.
    #[serde(skip)]
    pub verifier: Option<Arc<dyn TaskVerifier>>,
    pub verifier_spec: VerifierSpec,
    #[serde(default)]
    pub expected_turns: Option<(u32, u32)>,
    #[serde(default)]
    pub expected_cost_usd: Option<f64>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Per-run timeout (Rule 4), serialized as whole seconds.
    #[serde(with = "duration_secs", default = "default_timeout")]
    pub timeout: Duration,
    /// Optional model-response fixture for live/recorded replay (kept per
    /// fixtures/README.md reconciliation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_fixture: Option<String>,
}

impl std::fmt::Debug for EvalTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvalTask")
            .field("id", &self.id)
            .field("instruction", &self.instruction)
            .field("workspace_snapshot", &self.workspace_snapshot)
            .field("verifier_spec", &self.verifier_spec)
            .field("expected_turns", &self.expected_turns)
            .field("expected_cost_usd", &self.expected_cost_usd)
            .field("tags", &self.tags)
            .field("timeout", &self.timeout)
            .field("model_fixture", &self.model_fixture)
            .finish()
    }
}

impl EvalTask {
    /// Resolve `verifier` from `verifier_spec` in place (Rule: build verifier
    /// from spec). Idempotent.
    pub fn resolve_verifier(&mut self) {
        self.verifier = Some(crate::verifier::build_verifier(&self.verifier_spec));
    }

    /// The resolved verifier, building it on demand if not yet resolved.
    pub fn verifier(&self) -> Arc<dyn TaskVerifier> {
        match &self.verifier {
            Some(v) => v.clone(),
            None => crate::verifier::build_verifier(&self.verifier_spec),
        }
    }
}

fn default_timeout() -> Duration {
    Duration::from_secs(300)
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;
    pub fn serialize<S: Serializer>(v: &Duration, s: S) -> Result<S::Ok, S::Error> {
        v.as_secs().serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs(u64::deserialize(d)?))
    }
}

/// A versioned task suite holding three disjoint task lists (Rule 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSuite {
    /// Required (Rule 6) — the loader rejects a manifest without it.
    pub suite_version: u32,
    #[serde(default)]
    pub regression: Vec<EvalTask>,
    #[serde(default)]
    pub challenge: Vec<EvalTask>,
    #[serde(default)]
    pub canary: Vec<EvalTask>,
}

impl TaskSuite {
    /// Resolve every task's verifier from its spec.
    pub fn resolve_verifiers(&mut self) {
        for t in self
            .regression
            .iter_mut()
            .chain(self.challenge.iter_mut())
            .chain(self.canary.iter_mut())
        {
            t.resolve_verifier();
        }
    }

    /// All tasks across the three categories, tagged with their category.
    pub fn all_tasks(&self) -> Vec<(TaskCategory, &EvalTask)> {
        let mut out = Vec::new();
        out.extend(
            self.regression
                .iter()
                .map(|t| (TaskCategory::Regression, t)),
        );
        out.extend(self.challenge.iter().map(|t| (TaskCategory::Challenge, t)));
        out.extend(self.canary.iter().map(|t| (TaskCategory::Canary, t)));
        out
    }
}
