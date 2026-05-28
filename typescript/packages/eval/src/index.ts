/**
 * @spore/eval — the EvalHarness (issue #26): the outer ring of the improvement
 * flywheel. Runs regression / challenge / canary task suites against the
 * `@spore/core` harness, compares a baseline config against candidate configs,
 * and recommends whether to adopt.
 *
 * Derived from the Rust reference (`rust/crates/spore-eval`). Same fixtures,
 * same outcomes (Rule 29). Native statistics — no external stats library
 * (Rules 26-28). The trace-sourced metrics (cache/sensor/middleware) read
 * structurally-typed span fields directly (Resolution 1, TS clean-accessor
 * form). `TraceAnalyzer` is interface-only (Rule 30); auto-promotion is
 * deferred (Rule 31); no Inspect AI / Langfuse dependency (Rule 32).
 */

export {
  EvalError,
  ConfigId,
  allTasks,
  newVerificationResult,
  clampedVerificationResult,
  DEFAULT_TASK_TIMEOUT_SECS,
  type EvalErrorKind,
  type VerificationResult,
  type WorkspaceSnapshot,
  type MetricDirection,
  type VerifierSpec,
  type CompositeChildSpec,
  type TaskCategory,
  type EvalTask,
  type TaskSuite,
} from "./task.js";

export {
  buildVerifier,
  AlwaysPass,
  AlwaysFail,
  TestSuiteVerifier,
  CompositeVerifier,
  MetricEvaluatorVerifier,
  NormalizingSuccessVerifier,
  LlmJudgeVerifier,
  StubLlmJudgeVerifier,
  type TaskVerifier,
} from "./verifier.js";

export {
  metricDirection,
  metricName,
  metricFromTrace,
  sampleFor,
  type EvalMetric,
  type RunSampleInputs,
} from "./metric-map.js";

export {
  metricStatsFromSamples,
  percentile,
  welchTTest,
  bootstrapCi,
  SplitMix64,
  DEFAULT_BOOTSTRAP_ITERATIONS,
  DEFAULT_BOOTSTRAP_SEED,
  type MetricStats,
  type WelchResult,
  type ConfidenceInterval,
} from "./stats.js";

export {
  classifyDirection,
  deriveRecommendation,
  recommendedN,
  SIGNIFICANCE_ALPHA,
  type ComparisonDirection,
  type MetricComparison,
  type Recommendation,
  type ComparisonReport,
} from "./report.js";

export {
  loadSuiteStr,
  loadSuitePath,
  resolveVerifiers,
  taskVerifier,
  suiteToJson,
  promoteChallengeTask,
} from "./manifest.js";

export { Workspace } from "./worktree.js";

export {
  EvalHarness,
  EvalHarnessBuilder,
  type EvalHarnessOptions,
  type TraceAnalyzer,
  type HarnessConfigDiff,
} from "./harness.js";
