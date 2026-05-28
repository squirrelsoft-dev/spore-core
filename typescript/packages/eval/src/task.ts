/**
 * Task-suite types: {@link TaskSuite}, {@link EvalTask}, {@link WorkspaceSnapshot},
 * {@link VerifierSpec}, {@link VerificationResult}, {@link ConfigId}, and the
 * {@link EvalError} hierarchy.
 *
 * Rules enforced here: 1 (three disjoint lists), 5 (tags free-form), 6
 * (`suite_version` required), 7/8 (verification result shape + score clamp).
 *
 * Wire shape mirrors the Rust reference byte-for-byte where it is serialized:
 * tagged unions use a `kind` discriminator in `snake_case`; the manifest JSON
 * keys are `snake_case` (e.g. `workspace_snapshot`, `verifier_spec`,
 * `suite_version`). The cross-language fixture oracle (Rule 29) depends on it.
 */

import type { TaskVerifier } from "./verifier.js";

// ============================================================================
// ConfigId
// ============================================================================

/** Identifies a candidate harness configuration in a comparison. */
export class ConfigId {
  constructor(readonly value: string) {}
  static of(value: string): ConfigId {
    return new ConfigId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
}

// ============================================================================
// EvalError (discriminated `kind`, per CONVENTIONS.md)
// ============================================================================

/** The discriminant tags for {@link EvalError}. */
export type EvalErrorKind =
  | "missing_suite_version"
  | "manifest_parse"
  | "verify"
  | "worktree"
  | "missing_metrics"
  | "io";

/**
 * Domain error for the eval harness. Extends `Error` with a `name` set to the
 * class name and a discriminant `kind` for exhaustive `switch`es (per
 * `CONVENTIONS.md`). Never thrown as a plain `Error` across package boundaries.
 */
export class EvalError extends Error {
  override readonly name = "EvalError";
  constructor(
    readonly kind: EvalErrorKind,
    message: string,
  ) {
    super(message);
  }

  /** A manifest was loaded without the required `suite_version` (Rule 6). */
  static missingSuiteVersion(): EvalError {
    return new EvalError(
      "missing_suite_version",
      "manifest is missing required field `suite_version`",
    );
  }
  /** A manifest failed to parse. */
  static manifestParse(detail: string): EvalError {
    return new EvalError("manifest_parse", `manifest parse error: ${detail}`);
  }
  /** A verifier failed in a non-"task failed" way (e.g. out-of-range score, Rule 8). */
  static verify(detail: string): EvalError {
    return new EvalError("verify", `verification error: ${detail}`);
  }
  /** Restoring or tearing down a workspace/worktree failed (Rules 2-3). */
  static worktree(detail: string): EvalError {
    return new EvalError("worktree", `worktree error: ${detail}`);
  }
  /** An `EvalHarness` was built or run without the metrics it needs. */
  static missingMetrics(detail: string): EvalError {
    return new EvalError("missing_metrics", `missing metrics: ${detail}`);
  }
  /** An I/O error. */
  static io(detail: string): EvalError {
    return new EvalError("io", detail);
  }
}

// ============================================================================
// VerificationResult (Rule 7)
// ============================================================================

/**
 * The outcome of a {@link TaskVerifier} (Rule 7): a pass/fail flag, a `score`
 * clamped to `[0, 1]`, a human-readable `detail`, and granular `signals`.
 */
export interface VerificationResult {
  passed: boolean;
  score: number;
  detail: string;
  signals: Record<string, number>;
}

/**
 * Build a result, rejecting an out-of-range `score` with an {@link EvalError}
 * (Rule 8). Throws `EvalError(kind: "verify")` when `score` ∉ `[0, 1]`.
 */
export function newVerificationResult(
  passed: boolean,
  score: number,
  detail: string,
  signals: Record<string, number> = {},
): VerificationResult {
  if (!(score >= 0 && score <= 1)) {
    throw EvalError.verify(`score ${score} out of range [0.0, 1.0]`);
  }
  return { passed, score, detail, signals };
}

/**
 * Build a result, clamping any out-of-range `score` into `[0, 1]` instead of
 * erroring. Use for evaluator-derived scores that are guaranteed-finite but may
 * drift slightly outside the unit interval.
 */
export function clampedVerificationResult(
  passed: boolean,
  score: number,
  detail: string,
  signals: Record<string, number> = {},
): VerificationResult {
  return { passed, score: Math.min(Math.max(score, 0), 1), detail, signals };
}

// ============================================================================
// WorkspaceSnapshot (Resolution 2)
// ============================================================================

/**
 * How a task's workspace is restored before a run (Rule 2).
 *
 * `files` (kind `"files"`) is the canonical hermetic form the shipped fixtures
 * use — no real git repo is needed for cross-language replay. `git_ref` is
 * supported for real snapshots (init a throwaway repo + `git worktree add`).
 */
export type WorkspaceSnapshot =
  | { kind: "files"; files: Record<string, string> }
  | { kind: "git_ref"; repo: string; reference: string }
  | { kind: "empty" };

// ============================================================================
// MetricDirection
// ============================================================================

/**
 * Optimization direction for a metric-evaluator verifier. Mirrors
 * `OptimizationDirection` from `@spore/core` but is serialized here as a
 * self-contained spec field.
 */
export type MetricDirection = "minimize" | "maximize";

// ============================================================================
// VerifierSpec (serializable; resolved to a TaskVerifier)
// ============================================================================

/** One child of a {@link VerifierSpec} composite, with weight and required-ness. */
export interface CompositeChildSpec {
  spec: VerifierSpec;
  weight: number;
  required?: boolean;
}

/**
 * A serializable description of a verifier. Resolved to a {@link TaskVerifier}
 * by {@link import("./verifier.js").buildVerifier}.
 */
export type VerifierSpec =
  | {
      /** Run a command in the workspace; score = pass rate (Rule 10). */
      kind: "test_suite";
      command: string;
      args?: string[];
      timeout_secs?: number | null;
    }
  | {
      /** Combine children by weight; `required` children must all pass (Rule 11). */
      kind: "composite";
      children: CompositeChildSpec[];
    }
  | {
      /** Adapt a metric evaluator, normalizing its value to a score (Rule 12). */
      kind: "metric_evaluator";
      descriptor: string;
      direction: MetricDirection;
      min?: number | null;
      max?: number | null;
      threshold?: number | null;
    }
  | {
      /** An LLM-judge verifier; non-deterministic (Rule 13). */
      kind: "llm_judge";
      rubric: string;
      score_range: [number, number];
    }
  /** Test scaffolding: always passes with score 1.0. */
  | { kind: "always_pass" }
  /** Test scaffolding: always fails with score 0.0. */
  | { kind: "always_fail" };

// ============================================================================
// TaskCategory + EvalTask + TaskSuite
// ============================================================================

/** Which of the three disjoint task lists a task belongs to (Rule 1). */
export type TaskCategory = "regression" | "challenge" | "canary";

/** The default per-run timeout when a manifest omits one, in seconds. */
export const DEFAULT_TASK_TIMEOUT_SECS = 300;

/**
 * One evaluation task. `timeout` is the per-run timeout in **seconds** (Rule 4),
 * matching the manifest's whole-second wire form.
 */
export interface EvalTask {
  id: string;
  instruction: string;
  workspace_snapshot: WorkspaceSnapshot;
  verifier_spec: VerifierSpec;
  expected_turns?: [number, number] | null;
  expected_cost_usd?: number | null;
  tags: string[];
  /** Per-run timeout (Rule 4), in whole seconds. */
  timeout: number;
  /** Optional model-response fixture for live/recorded replay. */
  model_fixture?: string | null;
  /** The resolved verifier; rebuilt from `verifier_spec` on demand. */
  verifier?: TaskVerifier;
}

/** A versioned task suite holding three disjoint task lists (Rule 1). */
export interface TaskSuite {
  /** Required (Rule 6) — the loader rejects a manifest without it. */
  suite_version: number;
  regression: EvalTask[];
  challenge: EvalTask[];
  canary: EvalTask[];
}

/** All tasks across the three categories, tagged with their category. */
export function allTasks(suite: TaskSuite): [TaskCategory, EvalTask][] {
  return [
    ...suite.regression.map((t): [TaskCategory, EvalTask] => ["regression", t]),
    ...suite.challenge.map((t): [TaskCategory, EvalTask] => ["challenge", t]),
    ...suite.canary.map((t): [TaskCategory, EvalTask] => ["canary", t]),
  ];
}
