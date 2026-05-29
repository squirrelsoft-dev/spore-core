//! Issue #44 — `Verifier`: the oracle for the `SelfVerifying` loop strategy.
//!
//! The `Verifier` sits between an evaluator harness's `RunResult` and the
//! build loop's halt decision. It translates `(build_result, eval_result)`
//! into a `VerifierVerdict` — either `Passed` (halt with success) or
//! `Failed { reason }` (re-enter the build loop with `reason` injected into
//! the next turn's context).
//!
//! ## Ambiguity resolutions (see issue #44 comment thread)
//!
//! 1. `EvaluatorResponseVerifier` when neither `pass_pattern` nor
//!    `fail_pattern` matches → `Failed` with a descriptive reason including
//!    a truncated copy of the output. Default-FAIL is **not** configurable.
//! 2. Any non-`Success` `RunResult` in `build_result` or `eval_result` →
//!    `Failed`. `WaitingForHuman` is treated as a misconfiguration signal
//!    and surfaced in the reason.
//! 3. `CompositeVerifier` concatenates all child failure reasons (joined by
//!    `"\n"`), capped at 2000 characters total. Children that pass are not
//!    mentioned.
//! 4. `LoopStrategy::SelfVerifying` wiring is **deferred** — bundled with
//!    the Ralph wiring and #45. The strategy continues to return
//!    `StrategyNotYetImplemented` in the harness loop.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::harness::{HaltReason, RunResult, SandboxProvider};

// ============================================================================
// VerifierVerdict
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VerifierVerdict {
    Passed,
    Failed { reason: String },
}

impl VerifierVerdict {
    pub fn failed(reason: impl Into<String>) -> Self {
        Self::Failed {
            reason: reason.into(),
        }
    }
}

// ============================================================================
// VerifierInput
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierInput {
    pub build_result: RunResult,
    pub eval_result: RunResult,
    pub workspace: std::path::PathBuf,
    /// Which build-evaluate cycle this is (0-indexed).
    pub iteration: u32,
}

// ============================================================================
// Verifier trait
// ============================================================================

/// Async hand-rolled futures (`BoxFut`) match the rest of the crate's
/// trait-object pattern; not using `trait_variant::make` here because
/// `Arc<dyn Verifier>` must be dyn-compatible (the harness `SelfVerifying`
/// strategy will hold one).
pub trait Verifier: Send + Sync {
    fn verify<'a>(
        &'a self,
        input: &'a VerifierInput,
    ) -> crate::harness::BoxFut<'a, VerifierVerdict>;

    /// Maximum number of build-evaluate cycles before the harness halts the
    /// loop regardless of verdict. Prevents infinite loops when the evaluator
    /// always finds problems. Spec default: 3.
    fn max_iterations(&self) -> u32 {
        3
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Common reduction of a `RunResult` to either its `Success` output or a
/// descriptive failure reason.
enum ResultView<'a> {
    Output(&'a str),
    Failed(String),
}

fn view<'a>(label: &str, r: &'a RunResult) -> ResultView<'a> {
    match r {
        RunResult::Success { output, .. } => ResultView::Output(output.as_str()),
        RunResult::Failure { reason, .. } => {
            ResultView::Failed(format!("{label} run halted: {}", describe_halt(reason)))
        }
        RunResult::WaitingForHuman { .. } => ResultView::Failed(format!(
            "{label} run is WaitingForHuman — verifier received a paused harness; \
             this is a misconfiguration signal (the {label} should run to completion \
             before being verified)",
            label = label
        )),
        // An escalation (issue #80) is a clean terminal stop, but the verifier
        // expects a completed run — surface it as a misconfiguration like the
        // paused case.
        RunResult::Escalate { signal, .. } => ResultView::Failed(format!(
            "{label} run escalated ({signal:?}) — verifier received an escalated \
             harness; the caller must handle the signal before verification"
        )),
    }
}

fn describe_halt(reason: &HaltReason) -> String {
    // HaltReason is `#[non_exhaustive]` — fall back to Debug for unknown
    // variants. The exact serialized shape is not load-bearing for verifier
    // output; the caller treats this as opaque diagnostic text.
    format!("{reason:?}")
}

fn truncate_for_reason(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out = s[..max].to_string();
        out.push_str("... [truncated]");
        out
    }
}

// ============================================================================
// EvaluatorResponseVerifier
// ============================================================================

/// Pattern-matches the evaluator harness's final text response. The simplest
/// verifier — trusts whatever the evaluator wrote.
///
/// Rules:
///   - If `build_result` is not `Success` → `Failed` with the halt reason.
///   - If `eval_result` is not `Success` → `Failed` with the halt reason.
///   - If `pass_pattern` matches the eval output → `Passed`.
///   - If `fail_pattern` matches the eval output → `Failed` with the matched
///     line(s) as the reason.
///   - Neither matches → `Failed` with a descriptive default reason
///     (Default-FAIL contract; not configurable).
pub struct EvaluatorResponseVerifier {
    pub pass_pattern: regex::Regex,
    pub fail_pattern: regex::Regex,
    pub max_iterations: u32,
}

impl EvaluatorResponseVerifier {
    pub fn new(
        pass_pattern: &str,
        fail_pattern: &str,
        max_iterations: u32,
    ) -> Result<Self, regex::Error> {
        Ok(Self {
            pass_pattern: regex::Regex::new(pass_pattern)?,
            fail_pattern: regex::Regex::new(fail_pattern)?,
            max_iterations,
        })
    }
}

impl Verifier for EvaluatorResponseVerifier {
    fn verify<'a>(
        &'a self,
        input: &'a VerifierInput,
    ) -> crate::harness::BoxFut<'a, VerifierVerdict> {
        Box::pin(async move {
            if let ResultView::Failed(r) = view("build", &input.build_result) {
                return VerifierVerdict::failed(r);
            }
            let output = match view("evaluator", &input.eval_result) {
                ResultView::Output(s) => s,
                ResultView::Failed(r) => return VerifierVerdict::failed(r),
            };
            if let Some(m) = self.pass_pattern.find(output) {
                let _ = m;
                return VerifierVerdict::Passed;
            }
            if let Some(m) = self.fail_pattern.find(output) {
                return VerifierVerdict::failed(format!(
                    "evaluator reported failure: {}",
                    truncate_for_reason(m.as_str(), 500)
                ));
            }
            VerifierVerdict::failed(format!(
                "evaluator output matched neither pass_pattern (`{}`) nor \
                 fail_pattern (`{}`). Output was:\n{}",
                self.pass_pattern.as_str(),
                self.fail_pattern.as_str(),
                truncate_for_reason(output, 1000),
            ))
        })
    }

    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }
}

// ============================================================================
// TestSuiteVerifier
// ============================================================================

/// Runs a test command via the injected `SandboxProvider` and uses the exit
/// code as the verdict. Ignores the evaluator's text output — ground truth is
/// the tests.
///
/// Rules:
///   - If `build_result` is not `Success` → `Failed` with the halt reason.
///   - Run `command` in `working_dir` via `sandbox.execute_command`.
///   - Exit 0, not timed out → `Passed`.
///   - Anything else → `Failed` with a stderr/stdout tail.
pub struct TestSuiteVerifier {
    pub command: String,
    pub working_dir: std::path::PathBuf,
    pub timeout: Duration,
    pub sandbox: Arc<dyn SandboxProvider>,
    pub max_iterations: u32,
}

impl TestSuiteVerifier {
    pub fn new(
        command: impl Into<String>,
        working_dir: impl Into<std::path::PathBuf>,
        timeout: Duration,
        sandbox: Arc<dyn SandboxProvider>,
        max_iterations: u32,
    ) -> Self {
        Self {
            command: command.into(),
            working_dir: working_dir.into(),
            timeout,
            sandbox,
            max_iterations,
        }
    }
}

impl Verifier for TestSuiteVerifier {
    fn verify<'a>(
        &'a self,
        input: &'a VerifierInput,
    ) -> crate::harness::BoxFut<'a, VerifierVerdict> {
        Box::pin(async move {
            if let ResultView::Failed(r) = view("build", &input.build_result) {
                return VerifierVerdict::failed(r);
            }
            let mut parts = self.command.split_whitespace();
            let program = match parts.next() {
                Some(p) => p.to_string(),
                None => return VerifierVerdict::failed("empty test command"),
            };
            let args: Vec<String> = parts.map(|s| s.to_string()).collect();
            let r = self
                .sandbox
                .execute_command(
                    &program,
                    &args,
                    Some(self.working_dir.as_path()),
                    Some(self.timeout),
                )
                .await;
            match r {
                Ok(out) if out.exit_code == 0 && !out.timed_out => VerifierVerdict::Passed,
                Ok(out) => {
                    let tail = tail_lines(&out.stderr, 20);
                    let tail = if tail.trim().is_empty() {
                        tail_lines(&out.stdout, 20)
                    } else {
                        tail
                    };
                    VerifierVerdict::failed(format!(
                        "test suite failed (exit {}, timed_out={}):\n{}",
                        out.exit_code, out.timed_out, tail
                    ))
                }
                Err(v) => VerifierVerdict::failed(format!("sandbox refused test command: {v:?}")),
            }
        })
    }

    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

// ============================================================================
// CompositeVerifier
// ============================================================================

const COMPOSITE_REASON_CAP: usize = 2000;

/// Passes only when **all** child verifiers pass. On failure, concatenates
/// every child's failure reason (joined by `"\n"`), capped at 2000 characters
/// total. Children that pass are not mentioned in the failure reason.
pub struct CompositeVerifier {
    pub verifiers: Vec<Arc<dyn Verifier>>,
    pub max_iterations: u32,
}

impl CompositeVerifier {
    pub fn new(verifiers: Vec<Arc<dyn Verifier>>, max_iterations: u32) -> Self {
        Self {
            verifiers,
            max_iterations,
        }
    }
}

impl Verifier for CompositeVerifier {
    fn verify<'a>(
        &'a self,
        input: &'a VerifierInput,
    ) -> crate::harness::BoxFut<'a, VerifierVerdict> {
        Box::pin(async move {
            let mut failures: Vec<String> = Vec::new();
            for (i, v) in self.verifiers.iter().enumerate() {
                if let VerifierVerdict::Failed { reason } = v.verify(input).await {
                    failures.push(format!("[verifier {i}] {reason}"));
                }
            }
            if failures.is_empty() {
                return VerifierVerdict::Passed;
            }
            let joined = failures.join("\n");
            VerifierVerdict::failed(truncate_for_reason(&joined, COMPOSITE_REASON_CAP))
        })
    }

    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{
        AggregateUsage, BoxFut, CommandOutput, HaltReason, RunResult, SandboxViolation, SessionId,
    };
    use crate::model::ToolCall;

    fn success(output: &str) -> RunResult {
        RunResult::Success {
            output: output.into(),
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 1,
        }
    }

    fn failure() -> RunResult {
        RunResult::Failure {
            reason: HaltReason::StrategyNotYetImplemented {
                strategy: "x".into(),
            },
            session_id: SessionId::new("s"),
            usage: AggregateUsage::default(),
            turns: 0,
        }
    }

    fn input_with(build: RunResult, eval: RunResult) -> VerifierInput {
        VerifierInput {
            build_result: build,
            eval_result: eval,
            workspace: std::path::PathBuf::from("/tmp"),
            iteration: 0,
        }
    }

    // ── EvaluatorResponseVerifier ────────────────────────────────────────────

    fn make_resp_verifier() -> EvaluatorResponseVerifier {
        EvaluatorResponseVerifier::new(r"(?i)\bPASS\b", r"(?i)\bFAIL: .+", 3).unwrap()
    }

    #[tokio::test]
    async fn response_verifier_pass_pattern_matches() {
        let v = make_resp_verifier();
        let i = input_with(success("ok"), success("all checks PASS, ready to ship"));
        assert_eq!(v.verify(&i).await, VerifierVerdict::Passed);
    }

    #[tokio::test]
    async fn response_verifier_fail_pattern_matches_with_reason() {
        let v = make_resp_verifier();
        let i = input_with(
            success("ok"),
            success("FAIL: missing edge case in handler.rs"),
        );
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => {
                assert!(reason.contains("missing edge case"), "got: {reason}")
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn response_verifier_neither_pattern_default_fails() {
        let v = make_resp_verifier();
        let i = input_with(success("ok"), success("indeterminate output"));
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => {
                assert!(reason.contains("matched neither"), "got: {reason}");
                assert!(reason.contains("indeterminate output"), "got: {reason}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn response_verifier_build_failure_propagates() {
        let v = make_resp_verifier();
        let i = input_with(failure(), success("PASS"));
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => assert!(reason.starts_with("build run halted")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn response_verifier_eval_failure_propagates() {
        let v = make_resp_verifier();
        let i = input_with(success("ok"), failure());
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => {
                assert!(reason.starts_with("evaluator run halted"), "got: {reason}")
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn response_verifier_default_max_iterations_overrideable() {
        let v = make_resp_verifier();
        assert_eq!(v.max_iterations(), 3);
        let v2 = EvaluatorResponseVerifier::new("a", "b", 10).unwrap();
        assert_eq!(v2.max_iterations(), 10);
    }

    // ── TestSuiteVerifier ────────────────────────────────────────────────────

    struct StubSandbox {
        out: CommandOutput,
        root: std::path::PathBuf,
    }

    impl SandboxProvider for StubSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async { Ok(()) })
        }
        fn workspace_root(&self) -> &std::path::Path {
            &self.root
        }
        fn execute_command<'a>(
            &'a self,
            _command: &'a str,
            _args: &'a [String],
            _working_dir: Option<&'a std::path::Path>,
            _timeout: Option<Duration>,
        ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
            let out = self.out.clone();
            Box::pin(async move { Ok(out) })
        }
    }

    fn stub_sandbox(exit: i32, stderr: &str) -> Arc<dyn SandboxProvider> {
        Arc::new(StubSandbox {
            out: CommandOutput {
                stdout: String::new(),
                stderr: stderr.to_string(),
                exit_code: exit,
                timed_out: false,
                truncated: false,
            },
            root: std::path::PathBuf::from("/"),
        })
    }

    #[tokio::test]
    async fn test_suite_verifier_pass() {
        let v = TestSuiteVerifier::new(
            "cargo test",
            "/work",
            Duration::from_secs(60),
            stub_sandbox(0, ""),
            3,
        );
        let i = input_with(success("ok"), success(""));
        assert_eq!(v.verify(&i).await, VerifierVerdict::Passed);
    }

    #[tokio::test]
    async fn test_suite_verifier_fail_includes_stderr() {
        let v = TestSuiteVerifier::new(
            "cargo test",
            "/work",
            Duration::from_secs(60),
            stub_sandbox(1, "test foo ... FAILED"),
            3,
        );
        let i = input_with(success("ok"), success(""));
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => assert!(reason.contains("FAILED"), "{reason}"),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_suite_verifier_build_failure_short_circuits() {
        let v = TestSuiteVerifier::new(
            "cargo test",
            "/work",
            Duration::from_secs(60),
            stub_sandbox(0, ""),
            3,
        );
        let i = input_with(failure(), success(""));
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => assert!(reason.starts_with("build run halted")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_suite_verifier_empty_command_fails() {
        let v =
            TestSuiteVerifier::new("", "/work", Duration::from_secs(60), stub_sandbox(0, ""), 3);
        let i = input_with(success("ok"), success(""));
        match v.verify(&i).await {
            VerifierVerdict::Failed { reason } => assert!(reason.contains("empty test command")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    // ── CompositeVerifier ────────────────────────────────────────────────────

    struct FixedVerifier(VerifierVerdict);

    impl Verifier for FixedVerifier {
        fn verify<'a>(&'a self, _i: &'a VerifierInput) -> BoxFut<'a, VerifierVerdict> {
            let v = self.0.clone();
            Box::pin(async move { v })
        }
    }

    fn pass_v() -> Arc<dyn Verifier> {
        Arc::new(FixedVerifier(VerifierVerdict::Passed))
    }

    fn fail_v(reason: &str) -> Arc<dyn Verifier> {
        Arc::new(FixedVerifier(VerifierVerdict::failed(reason)))
    }

    #[tokio::test]
    async fn composite_all_pass_returns_passed() {
        let c = CompositeVerifier::new(vec![pass_v(), pass_v(), pass_v()], 3);
        let i = input_with(success("ok"), success("ok"));
        assert_eq!(c.verify(&i).await, VerifierVerdict::Passed);
    }

    #[tokio::test]
    async fn composite_one_fail_returns_that_reason() {
        let c = CompositeVerifier::new(vec![pass_v(), fail_v("oops"), pass_v()], 3);
        let i = input_with(success("ok"), success("ok"));
        match c.verify(&i).await {
            VerifierVerdict::Failed { reason } => {
                assert!(reason.contains("oops"));
                assert!(reason.contains("[verifier 1]"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn composite_many_fails_concatenated() {
        let c = CompositeVerifier::new(
            vec![fail_v("first"), pass_v(), fail_v("second"), fail_v("third")],
            3,
        );
        let i = input_with(success("ok"), success("ok"));
        match c.verify(&i).await {
            VerifierVerdict::Failed { reason } => {
                assert!(reason.contains("first"));
                assert!(reason.contains("second"));
                assert!(reason.contains("third"));
                assert!(
                    !reason.contains("[verifier 1]"),
                    "pass-verifier should not appear"
                );
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn composite_truncates_at_2000_chars() {
        let long = "x".repeat(5000);
        let c = CompositeVerifier::new(vec![fail_v(&long)], 3);
        let i = input_with(success("ok"), success("ok"));
        match c.verify(&i).await {
            VerifierVerdict::Failed { reason } => {
                assert!(reason.len() <= 2000 + "... [truncated]".len());
                assert!(reason.ends_with("... [truncated]"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn verifier_is_dyn_compatible() {
        // Compile-time guarantee that the trait is dyn-compatible.
        fn _assert_dyn(_: &dyn Verifier) {}
        let v: Arc<dyn Verifier> =
            Arc::new(EvaluatorResponseVerifier::new("PASS", "FAIL", 3).unwrap());
        _assert_dyn(v.as_ref());
    }

    // ── Cross-language fixture replay ────────────────────────────────────────
    //
    // The spec lists `fixtures/verifier/evaluator_pass.jsonl` and
    // `evaluator_fail.jsonl`. Per the issue comment thread, we deviate from
    // that naming and ship a single JSON file with a `cases` array — same
    // pattern as `fixtures/completion_check/sql_result.json`.

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        pass_pattern: String,
        fail_pattern: String,
        build_result: RunResult,
        eval_result: RunResult,
        expected: FixtureExpected,
    }

    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum FixtureExpected {
        Passed,
        Failed { contains: String },
    }

    #[derive(Deserialize)]
    struct FixtureSuite {
        cases: Vec<FixtureCase>,
    }

    #[tokio::test]
    async fn fixture_replay_evaluator_response_verifier() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/verifier/evaluator_response.json");
        let raw = std::fs::read_to_string(&path).expect("fixture present");
        let suite: FixtureSuite = serde_json::from_str(&raw).unwrap();
        for case in suite.cases {
            let v = EvaluatorResponseVerifier::new(&case.pass_pattern, &case.fail_pattern, 3)
                .expect("valid regexes");
            let input = VerifierInput {
                build_result: case.build_result,
                eval_result: case.eval_result,
                workspace: std::path::PathBuf::from("/fixture"),
                iteration: 0,
            };
            let got = v.verify(&input).await;
            match (got, case.expected) {
                (VerifierVerdict::Passed, FixtureExpected::Passed) => {}
                (VerifierVerdict::Failed { reason }, FixtureExpected::Failed { contains }) => {
                    assert!(
                        reason.contains(&contains),
                        "case `{}`: expected reason to contain `{contains}`, got `{reason}`",
                        case.name
                    );
                }
                (got, expected) => panic!(
                    "case `{}`: mismatch — got {got:?}, expected {}",
                    case.name,
                    match expected {
                        FixtureExpected::Passed => "Passed".to_string(),
                        FixtureExpected::Failed { contains } =>
                            format!("Failed(contains={contains})"),
                    }
                ),
            }
        }
    }
}
