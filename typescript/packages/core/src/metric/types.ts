/**
 * MetricEvaluator — canonical types (spore-core issue #23).
 *
 * Mirrors `rust/crates/spore-core/src/metric.rs` byte-for-byte on the wire:
 * tagged unions use a `kind` discriminator in `snake_case`; struct fields
 * are `snake_case`. Same fixture, same outcome — see `/fixtures/README.md`.
 *
 * The `HillClimbing` loop strategy uses a {@link MetricEvaluator} to score
 * each iteration. The harness then feeds the result through {@link shouldKeep}
 * to decide whether to advance (keep the change) or revert (discard it).
 *
 * ## Rules enforced
 *   - `evaluate` receives the {@link SandboxProvider}. All subprocess
 *     execution goes through it; evaluators never spawn directly.
 *   - {@link CommandMetricEvaluator} writes captured stdout+stderr to
 *     `log_output_to` *before* parsing the metric, so a partial run is still
 *     diagnosable.
 *   - A regex that does not match the captured output is
 *     {@link MetricError} `parse_failed`, not a crash.
 *   - Non-zero exit / panic maps to `crashed`; an exceeded timeout maps to
 *     `timeout`. Both are valid iteration outcomes — the harness logs them
 *     and asks the agent to try a different approach.
 *   - {@link shouldKeep} strictly compares against `current_best`: a delta of
 *     exactly `min_delta` (or `0.0` when unset) does NOT count as
 *     improvement. Equal scores are discarded.
 */

import type { OptimizationDirection } from "../harness/types.js";
import type { SandboxProvider } from "../harness/types.js";
import type { SessionStateSnapshot } from "../termination/types.js";

// ============================================================================
// MetricError
// ============================================================================

/**
 * Discriminated error returned by {@link MetricEvaluator.evaluate}. Wire shape
 * mirrors the Rust `#[serde(tag = "kind", rename_all = "snake_case")]` enum:
 *
 *   - `execution_failed { reason }`  — evaluator harness-level failure (bad
 *      regex, sandbox rejected the command, invalid config).
 *   - `timeout { after }`            — evaluator exceeded its configured
 *      `timeout`. `after` is whole seconds (mirrors Rust `Duration` wire
 *      form via `as_secs_f64`).
 *   - `parse_failed { output, pattern }` — regex matched zero captures, or
 *      the captured substring was not a finite number.
 *   - `crashed { log }`              — experiment ran but failed (non-zero
 *      exit, panic, OOM); the combined stdout+stderr is carried in `log`.
 */
export type MetricError =
  | { kind: "execution_failed"; reason: string }
  | { kind: "timeout"; after: number }
  | { kind: "parse_failed"; output: string; pattern: string }
  | { kind: "crashed"; log: string };

/**
 * A {@link MetricError}-as-thrown wrapper. Evaluators return errors as
 * tagged values (in a `MetricOutcome`), but a few call sites need to throw —
 * keep the wire shape intact via the `error` field.
 */
export class MetricErrorException extends Error {
  override readonly name = "MetricErrorException";
  constructor(readonly error: MetricError) {
    super(metricErrorMessage(error));
  }
}

export function metricErrorMessage(err: MetricError): string {
  switch (err.kind) {
    case "execution_failed":
      return `execution failed: ${err.reason}`;
    case "timeout":
      return `evaluator timed out after ${err.after}s`;
    case "parse_failed":
      return `could not parse metric from output (pattern: ${err.pattern})`;
    case "crashed":
      return `experiment crashed: ${err.log}`;
  }
}

// ============================================================================
// MetricResult / MetricOutcome
// ============================================================================

/**
 * Successful evaluator output. `duration` is whole seconds with fractional
 * precision (mirrors Rust `Duration::as_secs_f64`).
 */
export interface MetricResult {
  value: number;
  raw_output: string;
  duration: number;
  metadata: Record<string, string>;
}

export function newMetricResult(value: number): MetricResult {
  return { value, raw_output: "", duration: 0, metadata: {} };
}

/** Tagged success-or-error outcome from {@link MetricEvaluator.evaluate}. */
export type MetricOutcome =
  | { kind: "ok"; result: MetricResult }
  | { kind: "err"; error: MetricError };

// ============================================================================
// MetricEvaluator interface
// ============================================================================

/**
 * Pluggable scoring strategy for the `HillClimbing` loop. The harness calls
 * {@link MetricEvaluator.evaluate} after the agent completes each iteration
 * and feeds the result into {@link shouldKeep}.
 */
export interface MetricEvaluator {
  evaluate(
    sandbox: SandboxProvider,
    sessionState: SessionStateSnapshot,
    signal?: AbortSignal,
  ): Promise<MetricOutcome>;

  direction(): OptimizationDirection;

  description(): string;
}

// ============================================================================
// shouldKeep
// ============================================================================

/**
 * The keep-or-revert decision the harness applies after every iteration.
 *
 * Returns `true` only when `newValue` strictly beats `currentBest` by more
 * than `minDelta` (default `0.0`). Equal scores are discarded — a flat run
 * is not progress.
 */
export function shouldKeep(
  newValue: number,
  currentBest: number,
  direction: OptimizationDirection,
  minDelta: number | null | undefined,
): boolean {
  const delta = direction === "minimize" ? currentBest - newValue : newValue - currentBest;
  return delta > (minDelta ?? 0.0);
}

// ============================================================================
// ResultsEntry / IterationStatus
// ============================================================================

export type IterationStatus = "kept" | "discarded" | "crashed" | "timeout";

/**
 * Map an evaluator error outcome to the iteration status the harness
 * records. Successful evaluations are then routed through {@link shouldKeep}
 * to resolve `kept` vs `discarded`.
 */
export function iterationStatusFromError(err: MetricError): IterationStatus {
  return err.kind === "timeout" ? "timeout" : "crashed";
}

export interface ResultsEntry {
  iteration: number;
  commit_hash: string | null;
  metric_value: number;
  direction: OptimizationDirection;
  status: IterationStatus;
  duration: number;
  description: string;
  metadata: Record<string, string>;
}

// ============================================================================
// Internal helper — exported for fixture replay
// ============================================================================

/**
 * Extract a single-capture-group numeric metric from `output` using
 * `pattern`. Invalid regex ⇒ `execution_failed`. Missing match or
 * unparseable capture ⇒ `parse_failed`. First match wins.
 *
 * Exported so the cross-language fixture replay can hit this function
 * directly the way the Rust suite does.
 */
export function parseMetric(output: string, pattern: string): number {
  let re: RegExp;
  try {
    re = new RegExp(pattern);
  } catch (e) {
    throw new MetricErrorException({
      kind: "execution_failed",
      reason: `invalid regex ${JSON.stringify(pattern)}: ${(e as Error).message}`,
    });
  }
  const m = re.exec(output);
  if (m == null || m[1] == null) {
    throw new MetricErrorException({
      kind: "parse_failed",
      output,
      pattern,
    });
  }
  const captured = m[1].trim();
  // Strict float parse: Number() returns NaN for "" and accepts "1.0extra"
  // would be wrong, so use a regex check first.
  if (!/^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$/.test(captured)) {
    throw new MetricErrorException({
      kind: "parse_failed",
      output,
      pattern,
    });
  }
  const value = Number(captured);
  if (!Number.isFinite(value)) {
    throw new MetricErrorException({
      kind: "parse_failed",
      output,
      pattern,
    });
  }
  return value;
}
