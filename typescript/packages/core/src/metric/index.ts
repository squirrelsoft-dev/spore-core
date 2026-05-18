/**
 * Public re-exports for the canonical {@link MetricEvaluator} (spore-core
 * issue #23). Re-exported under the `metric` namespace at the package root
 * for symmetry with `termination`, `observability`, `sensor`, etc.
 */

export * from "./types.js";
export {
  CommandMetricEvaluator,
  TestPassRateEvaluator,
  LatencyEvaluator,
  LlmJudgeEvaluator,
} from "./standard.js";
export type {
  CommandMetricEvaluatorConfig,
  TestPassRateEvaluatorConfig,
  LatencyEvaluatorConfig,
  LlmJudgeEvaluatorConfig,
  JudgeModelConfig,
} from "./standard.js";
