//! spore-eval: the EvalHarness — the outer ring of the improvement flywheel
//! (issue #26). Runs regression / challenge / canary task suites against the
//! `spore-core` harness, compares a baseline config against candidate configs,
//! and recommends whether to adopt. This is the Rust reference implementation;
//! TypeScript, Python, and Go derive from it.
//!
//! # Types
//! - [`ConfigId`], [`TaskSuite`], [`TaskCategory`], [`EvalTask`],
//!   [`WorkspaceSnapshot`], [`VerifierSpec`], [`CompositeChildSpec`],
//!   [`MetricDirection`], [`VerificationResult`], [`EvalError`].
//! - [`EvalMetric`] (`direction()`, `name()`).
//! - [`MetricStats`], [`ConfidenceInterval`], [`SplitMix64`].
//! - [`ComparisonDirection`], [`MetricComparison`], [`Recommendation`],
//!   [`ComparisonReport`].
//! - [`EvalHarness`], [`EvalHarnessBuilder`], [`HarnessConfigDiff`].
//!
//! # Traits + key methods
//! - [`TaskVerifier`]: `verify(task, run, workspace) -> VerificationResult`;
//!   `is_deterministic() -> bool` (Rule 9).
//! - [`TraceAnalyzer`]: `analyze(traces) -> Vec<HarnessConfigDiff>`
//!   (INTERFACE ONLY — Rule 30; no built-in impl ships).
//! - [`EvalHarness::run`] `-> Result<Vec<ComparisonReport>, EvalError>`.
//!
//! # Standard verifier impls
//! - [`TestSuiteVerifier`] (Rule 10), [`CompositeVerifier`] (Rule 11),
//!   [`MetricEvaluatorVerifier`] (Rule 12), [`LlmJudgeVerifier`] (Rule 13),
//!   [`AlwaysPass`] / [`AlwaysFail`] (test scaffolding).
//!
//! # Statistics (native, no external stat crate — Rules 26-28)
//! - [`welch_t_test`] (Welch–Satterthwaite df, two-sided p via the regularized
//!   incomplete beta / Lentz continued fraction).
//! - [`bootstrap_ci`] (percentile CI, seeded SplitMix64, default 1000 iters).
//!
//! # Rules enforced
//! 1 (three disjoint lists), 2-3 (fresh/torn-down workspace), 4 (timeout =>
//! failed run), 5 (free-form tags), 6 (`suite_version` required), 7-8
//! (verification result + score clamp), 9-13 (verifier impls + determinism),
//! 14 (n runs per (config,task)), 15 (build harness from config), 16 (read
//! metrics from observability — Resolution 1), 17 (metric mapping), 18
//! (WaitingForHuman reported separately), 19 (MetricStats aggregation), 20
//! (Welch's t-test), 21 (bootstrap CI for non-deterministic metrics), 22
//! (direction), 23-24 (recommendation + power-estimated `recommended_n`), 25
//! (trace links for failed/regressed runs), 26-28 (native stats + oracle
//! tests), 29 (fixtures are the cross-language oracle), 30 (TraceAnalyzer
//! interface-only), 31 (manual promotion CLI subcommand), 32 (no Inspect AI /
//! Langfuse dependency — adapter seam only).
//!
//! There are no `// SPEC QUESTION:` markers: both open questions (metric source
//! and `workspace_snapshot`) are resolved in the issue #26 spec comment.

pub mod harness;
pub mod manifest;
pub mod metric_map;
pub mod report;
pub mod stats;
pub mod task;
pub mod verifier;
pub mod worktree;

pub use harness::{EvalHarness, EvalHarnessBuilder, HarnessConfigDiff, TraceAnalyzer};
pub use manifest::{load_suite_path, load_suite_str, promote_challenge_task, suite_to_json};
pub use metric_map::EvalMetric;
pub use report::{
    classify_direction, derive_recommendation, ComparisonDirection, ComparisonReport,
    MetricComparison, Recommendation,
};
pub use stats::{
    bootstrap_ci, welch_t_test, ConfidenceInterval, MetricStats, SplitMix64, WelchResult,
};
pub use task::{
    CompositeChildSpec, ConfigId, EvalError, EvalTask, MetricDirection, TaskCategory, TaskSuite,
    VerificationResult, VerifierSpec, WorkspaceSnapshot,
};
pub use verifier::{
    build_verifier, AlwaysFail, AlwaysPass, CompositeVerifier, LlmJudgeVerifier,
    MetricEvaluatorVerifier, TaskVerifier, TestSuiteVerifier,
};
pub use worktree::Workspace;

#[cfg(test)]
mod tests;
