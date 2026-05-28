/**
 * {@link TaskVerifier} interface + standard implementations.
 *
 * Rules enforced here:
 *   - 9  `isDeterministic()` true for test-suite/result verifiers, false for LLM judge.
 *   - 10 {@link TestSuiteVerifier}: command pass-rate; passed = score==1.0; deterministic.
 *   - 11 {@link CompositeVerifier}: weighted mean; passed = all required; det = AND.
 *   - 12 {@link MetricEvaluatorVerifier}: wraps a `MetricEvaluator`, normalizes value.
 *   - 13 {@link LlmJudgeVerifier}: thin; non-deterministic; judge injected.
 */

import { spawn } from "node:child_process";

import {
  emptySessionState,
  termination,
  TaskId,
  SessionId,
  type ModelInterface,
  type ModelParams,
  type ModelRequest,
  type RunResult,
  type SandboxProvider,
} from "@spore/core";
import type { metric } from "@spore/core";
import type { OptimizationDirection } from "@spore/core";

import {
  EvalError,
  clampedVerificationResult,
  newVerificationResult,
  type EvalTask,
  type MetricDirection,
  type VerificationResult,
  type VerifierSpec,
} from "./task.js";

type MetricEvaluator = metric.MetricEvaluator;

// ============================================================================
// TaskVerifier interface
// ============================================================================

/** Verifies whether a task run satisfied its goal. */
export interface TaskVerifier {
  /**
   * Verify a completed run against the task, with access to the restored
   * workspace directory.
   */
  verify(
    task: EvalTask,
    run: RunResult,
    workspace: string,
  ): Promise<VerificationResult>;

  /**
   * `true` for test-suite / result verifiers; `false` for the LLM judge
   * (Rule 9).
   */
  isDeterministic(): boolean;
}

// ============================================================================
// buildVerifier — resolve a VerifierSpec into a TaskVerifier
// ============================================================================

/**
 * Resolve a {@link VerifierSpec} to a concrete verifier. `metric_evaluator`
 * specs have no built-in concrete evaluator (it is injected for non-fixture
 * use), so they resolve to a normalizing placeholder that scores from the run's
 * success flag — adequate for manifest replay; real evaluators are wired via
 * {@link MetricEvaluatorVerifier}.
 */
export function buildVerifier(spec: VerifierSpec): TaskVerifier {
  switch (spec.kind) {
    case "test_suite":
      return new TestSuiteVerifier(
        spec.command,
        spec.args ?? [],
        (spec.timeout_secs ?? 60) * 1000,
      );
    case "composite":
      return new CompositeVerifier(
        spec.children.map((c) => ({
          verifier: buildVerifier(c.spec),
          weight: c.weight,
          required: c.required ?? false,
        })),
      );
    case "metric_evaluator":
      return new NormalizingSuccessVerifier(
        spec.direction,
        spec.min ?? null,
        spec.max ?? null,
        spec.threshold ?? null,
      );
    case "llm_judge":
      return new StubLlmJudgeVerifier(spec.score_range);
    case "always_pass":
      return new AlwaysPass();
    case "always_fail":
      return new AlwaysFail();
    default: {
      const _exhaustive: never = spec;
      return _exhaustive;
    }
  }
}

// ============================================================================
// AlwaysPass / AlwaysFail (test scaffolding)
// ============================================================================

/** Always passes with score 1.0. */
export class AlwaysPass implements TaskVerifier {
  async verify(): Promise<VerificationResult> {
    return newVerificationResult(true, 1.0, "always pass");
  }
  isDeterministic(): boolean {
    return true;
  }
}

/** Always fails with score 0.0. */
export class AlwaysFail implements TaskVerifier {
  async verify(): Promise<VerificationResult> {
    return newVerificationResult(false, 0.0, "always fail");
  }
  isDeterministic(): boolean {
    return true;
  }
}

// ============================================================================
// TestSuiteVerifier (Rule 10)
// ============================================================================

/**
 * Runs a command in the workspace; score = pass rate parsed from the output.
 * `passed` = (score == 1.0). Deterministic.
 */
export class TestSuiteVerifier implements TaskVerifier {
  constructor(
    readonly command: string,
    readonly args: readonly string[],
    readonly timeoutMs: number,
  ) {}

  async verify(
    _task: EvalTask,
    _run: RunResult,
    workspace: string,
  ): Promise<VerificationResult> {
    const out = await runCommand(
      this.command,
      this.args,
      workspace,
      this.timeoutMs,
    );
    const combined = `${out.stdout}${out.stderr}`;
    // Pass-rate: prefer "N passed"/"M total"-style output; fall back to exit
    // code (0 => 1.0, nonzero => 0.0).
    const parsed = parsePassRate(combined);
    const score = parsed ?? (out.exitCode === 0 ? 1.0 : 0.0);
    const passed = Math.abs(score - 1.0) < Number.EPSILON;
    return clampedVerificationResult(
      passed,
      score,
      `exit=${out.exitCode} pass_rate=${score.toFixed(3)}`,
      {
        exit_code: out.exitCode,
        pass_rate: score,
      },
    );
  }

  isDeterministic(): boolean {
    return true;
  }
}

interface CommandResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/** Run a command (no shell) in `cwd`, bounded by `timeoutMs`. A timeout or a
 *  spawn failure resolves to a non-zero exit code rather than throwing. */
function runCommand(
  command: string,
  args: readonly string[],
  cwd: string,
  timeoutMs: number,
): Promise<CommandResult> {
  return new Promise((resolve) => {
    const child = spawn(command, [...args], { cwd });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timer = setTimeout(
      () => {
        if (!settled) child.kill("SIGKILL");
      },
      Math.max(1, timeoutMs),
    );
    child.stdout?.on("data", (d) => (stdout += d.toString()));
    child.stderr?.on("data", (d) => (stderr += d.toString()));
    child.on("error", (e) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({
        stdout,
        stderr: `${stderr}${(e as Error).message}`,
        exitCode: 127,
      });
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ stdout, stderr, exitCode: code ?? 1 });
    });
  });
}

/** Parse a pass-rate from common test-runner output. `undefined` if no
 *  recognizable counts are present. */
function parsePassRate(output: string): number | undefined {
  const passed = scanNumberBefore(output, " passed");
  const total =
    scanNumberBefore(output, " total") ?? scanNumberAfter(output, "of ");
  if (passed != null && total != null && total > 0) {
    return Math.min(Math.max(passed / total, 0), 1);
  }
  return undefined;
}

function scanNumberBefore(s: string, suffix: string): number | undefined {
  const idx = s.indexOf(suffix);
  if (idx < 0) return undefined;
  const head = s.slice(0, idx);
  const m = head.match(/(\d+)$/);
  return m ? Number(m[1]) : undefined;
}

function scanNumberAfter(s: string, prefix: string): number | undefined {
  const idx = s.indexOf(prefix);
  if (idx < 0) return undefined;
  const tail = s.slice(idx + prefix.length);
  const m = tail.match(/^(\d+)/);
  return m ? Number(m[1]) : undefined;
}

// ============================================================================
// CompositeVerifier (Rule 11)
// ============================================================================

interface CompositeChild {
  verifier: TaskVerifier;
  weight: number;
  required: boolean;
}

/**
 * Combines children by weight: score = weighted mean; passed = all required
 * children passed; `isDeterministic` = AND of children (Rule 11).
 */
export class CompositeVerifier implements TaskVerifier {
  constructor(private readonly children: CompositeChild[]) {}

  async verify(
    task: EvalTask,
    run: RunResult,
    workspace: string,
  ): Promise<VerificationResult> {
    let weightedSum = 0;
    let weightTotal = 0;
    let allRequiredPassed = true;
    const details: string[] = [];
    for (const child of this.children) {
      const r = await child.verifier.verify(task, run, workspace);
      weightedSum += r.score * child.weight;
      weightTotal += child.weight;
      if (child.required && !r.passed) allRequiredPassed = false;
      details.push(
        `[w=${child.weight} req=${child.required} pass=${r.passed} score=${r.score.toFixed(3)}]`,
      );
    }
    const score = weightTotal > 0 ? weightedSum / weightTotal : 0;
    return clampedVerificationResult(
      allRequiredPassed,
      score,
      details.join(" "),
    );
  }

  isDeterministic(): boolean {
    return this.children.every((c) => c.verifier.isDeterministic());
  }
}

// ============================================================================
// MetricEvaluatorVerifier (Rule 12)
// ============================================================================

/**
 * Wraps a {@link MetricEvaluator}: runs `evaluate`, normalizes the value to
 * `[0,1]` per `direction()` and the configured min/max (or a threshold).
 * Deterministic iff the wrapped evaluator is (defaults to deterministic).
 */
export class MetricEvaluatorVerifier implements TaskVerifier {
  private constructor(
    private readonly evaluator: MetricEvaluator,
    private readonly min: number | null,
    private readonly max: number | null,
    private readonly threshold: number | null,
    private deterministic: boolean,
  ) {}

  /** Wrap an evaluator, normalizing by an explicit `[min, max]` range. */
  static withRange(
    evaluator: MetricEvaluator,
    min: number,
    max: number,
  ): MetricEvaluatorVerifier {
    return new MetricEvaluatorVerifier(evaluator, min, max, null, true);
  }

  /** Wrap an evaluator, scoring 1.0 when the value beats `threshold` in the
   *  evaluator's `direction()`, else 0.0. */
  static withThreshold(
    evaluator: MetricEvaluator,
    threshold: number,
  ): MetricEvaluatorVerifier {
    return new MetricEvaluatorVerifier(evaluator, null, null, threshold, true);
  }

  /** Mark the wrapped evaluator as non-deterministic (e.g. an LLM judge). */
  nonDeterministic(): this {
    this.deterministic = false;
    return this;
  }

  private normalize(value: number, direction: OptimizationDirection): number {
    if (this.threshold != null) {
      const beats =
        direction === "maximize"
          ? value >= this.threshold
          : value <= this.threshold;
      return beats ? 1.0 : 0.0;
    }
    if (this.min != null && this.max != null) {
      if (Math.abs(this.max - this.min) < Number.EPSILON) return 0.0;
      const unit = Math.min(
        Math.max((value - this.min) / (this.max - this.min), 0),
        1,
      );
      return direction === "maximize" ? unit : 1 - unit;
    }
    return Math.min(Math.max(value, 0), 1);
  }

  async verify(
    task: EvalTask,
    run: RunResult,
    workspace: string,
  ): Promise<VerificationResult> {
    const sessionId = sessionIdOf(run);
    const snapshot = termination.newSessionStateSnapshot(
      sessionId,
      TaskId.of(task.id),
      emptySessionState(),
      workspace,
    );
    const outcome = await this.evaluator.evaluate(
      directSandbox(workspace),
      snapshot,
    );
    if (outcome.kind === "err") {
      throw EvalError.verify(
        `evaluator failed: ${JSON.stringify(outcome.error)}`,
      );
    }
    const value = outcome.result.value;
    const score = this.normalize(value, this.evaluator.direction());
    const passed = score >= 1.0 - Number.EPSILON;
    return clampedVerificationResult(
      passed,
      score,
      `metric value=${value} normalized=${score.toFixed(3)}`,
      {
        metric_value: value,
      },
    );
  }

  isDeterministic(): boolean {
    return this.deterministic;
  }
}

/**
 * Placeholder for a `metric_evaluator` verifier spec resolved from a manifest
 * (no concrete evaluator wired). Scores from the run's success flag, applying
 * the spec's direction so the surface is exercised in replay.
 */
export class NormalizingSuccessVerifier implements TaskVerifier {
  constructor(
    readonly direction: MetricDirection,
    readonly min: number | null,
    readonly max: number | null,
    readonly threshold: number | null,
  ) {}

  async verify(_task: EvalTask, run: RunResult): Promise<VerificationResult> {
    const success = run.kind === "success";
    // direction is informational here; success is already in [0,1].
    return newVerificationResult(
      success,
      success ? 1.0 : 0.0,
      "metric-evaluator (manifest placeholder)",
    );
  }

  isDeterministic(): boolean {
    return true;
  }
}

// ============================================================================
// LlmJudgeVerifier (Rule 13)
// ============================================================================

/**
 * A thin LLM-judge verifier. `isDeterministic() === false`. The concrete judge
 * {@link ModelInterface} is injected at construction.
 */
export class LlmJudgeVerifier implements TaskVerifier {
  constructor(
    private readonly judge: ModelInterface,
    private readonly rubric: string,
    private readonly scoreRange: [number, number],
    private readonly params: ModelParams = { stop_sequences: [] },
  ) {}

  async verify(_task: EvalTask, run: RunResult): Promise<VerificationResult> {
    const output = run.kind === "success" ? run.output : "";
    const [lo, hi] = this.scoreRange;
    const prompt = `${this.rubric}\n\nAgent output to evaluate:\n${output}\n\nReply with a single line \`score: <number>\` within [${lo}, ${hi}].`;
    const request: ModelRequest = {
      messages: [{ role: "user", content: { type: "text", text: prompt } }],
      tools: [],
      params: this.params,
      stream: false,
    };
    let response;
    try {
      response = await this.judge.call(request);
    } catch (e) {
      throw EvalError.verify(`judge call failed: ${(e as Error).message}`);
    }
    const text = response.content
      .map((b) => (b.type === "text" || b.type === "thinking" ? b.text : ""))
      .join("\n");
    const raw = parseScore(text);
    if (raw == null) {
      throw EvalError.verify(
        `no score in judge reply: ${JSON.stringify(text)}`,
      );
    }
    if (hi <= lo) {
      throw EvalError.verify(`invalid score_range (${lo},${hi})`);
    }
    const clampedRaw = Math.min(Math.max(raw, lo), hi);
    const score = Math.min(Math.max((clampedRaw - lo) / (hi - lo), 0), 1);
    return newVerificationResult(score >= 0.5, score, `judge score=${raw}`);
  }

  isDeterministic(): boolean {
    return false;
  }
}

/**
 * Stub LLM judge used when a manifest's `llm_judge` spec is resolved without an
 * injected model. Non-deterministic; scores from the run's success flag so the
 * non-deterministic comparison path (bootstrap CI) is still exercised.
 */
export class StubLlmJudgeVerifier implements TaskVerifier {
  constructor(readonly scoreRange: [number, number]) {}

  async verify(_task: EvalTask, run: RunResult): Promise<VerificationResult> {
    const success = run.kind === "success";
    return newVerificationResult(
      success,
      success ? 1.0 : 0.0,
      "llm-judge (manifest stub)",
    );
  }

  isDeterministic(): boolean {
    return false;
  }
}

/** Parse a `score: <number>` line (first match wins, case-insensitive). */
function parseScore(text: string): number | undefined {
  const idx = text.toLowerCase().indexOf("score");
  if (idx < 0) return undefined;
  const after = text.slice(idx + "score".length).replace(/^[:\s\t]+/, "");
  const m = after.match(/^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?/);
  if (!m) return undefined;
  const v = Number(m[0]);
  return Number.isFinite(v) ? v : undefined;
}

// ============================================================================
// Helpers
// ============================================================================

function sessionIdOf(run: RunResult): SessionId {
  switch (run.kind) {
    case "success":
    case "failure":
      return run.session_id;
    case "waiting_for_human":
      return run.state.session_id;
    default: {
      const _exhaustive: never = run;
      return _exhaustive;
    }
  }
}

/**
 * Minimal {@link import("@spore/core").SandboxProvider} that runs commands
 * directly in a workspace dir, used by {@link MetricEvaluatorVerifier} to give
 * the wrapped evaluator a sandbox handle. It permits everything and exposes the
 * workspace root; the evaluator's own command execution falls through to the
 * Node default `executeCommand` rooted at the workspace.
 */
function directSandbox(root: string): SandboxProvider {
  return {
    async validate() {
      return null;
    },
    workspaceRoot() {
      return root;
    },
  };
}
