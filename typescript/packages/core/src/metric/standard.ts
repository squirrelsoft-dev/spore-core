/**
 * Standard {@link MetricEvaluator} implementations (spore-core issue #23):
 *
 *   - {@link CommandMetricEvaluator}    — autoresearch-style command+regex.
 *   - {@link TestPassRateEvaluator}     — pass/total via two regexes.
 *   - {@link LatencyEvaluator}          — averaged wall-clock latency.
 *   - {@link LlmJudgeEvaluator}         — LLM-as-judge, normalized to [0,1].
 *
 * Mirrors `rust/crates/spore-core/src/metric.rs`.
 */

import { promises as fs } from "node:fs";
import { isAbsolute, join } from "node:path";

import type {
  CommandOutput,
  OptimizationDirection,
  SandboxProvider,
  SandboxViolation,
} from "../harness/types.js";
import type { ModelInterface, ModelParams, ModelRequest } from "../model/index.js";
import type { SessionStateSnapshot } from "../termination/types.js";

import {
  MetricErrorException,
  parseMetric,
  type MetricError,
  type MetricEvaluator,
  type MetricOutcome,
} from "./types.js";

// ============================================================================
// Internal helpers
// ============================================================================

function nowSeconds(): number {
  // Monotonic-ish; sufficient for duration measurement.
  return performance.now() / 1000;
}

async function execOrViolation(
  sandbox: SandboxProvider,
  command: string,
  args: readonly string[],
  cwd: string | null | undefined,
  timeoutSeconds: number,
  signal?: AbortSignal,
): Promise<CommandOutput | MetricError> {
  if (sandbox.executeCommand == null) {
    return {
      kind: "execution_failed",
      reason: "sandbox does not implement executeCommand",
    };
  }
  let res: CommandOutput | SandboxViolation;
  try {
    res = await sandbox.executeCommand(
      command,
      args,
      cwd ?? null,
      Math.round(timeoutSeconds * 1000),
      signal,
    );
  } catch (e) {
    return {
      kind: "execution_failed",
      reason: `sandbox rejected command: ${(e as Error).message ?? String(e)}`,
    };
  }
  if ("kind" in res) {
    // SandboxViolation — every variant carries a `kind`. CommandOutput does
    // not, so this disambiguates the union safely.
    return {
      kind: "execution_failed",
      reason: `sandbox rejected command: ${JSON.stringify(res)}`,
    };
  }
  return res;
}

async function writeLog(sandbox: SandboxProvider, path: string, body: string): Promise<void> {
  // Mirrors Rust's `let _ = tokio::fs::write(sandbox.workspace_root().join(path), body)`.
  // Best-effort: failures are swallowed so a write-log problem never
  // masks the real evaluator outcome.
  const root = sandbox.workspaceRoot?.();
  const target = isAbsolute(path) ? path : root != null ? join(root, path) : null;
  if (target == null) return;
  try {
    await fs.writeFile(target, body);
  } catch {
    // ignore — exactly like the Rust impl
  }
}

function combined(out: CommandOutput): string {
  return `${out.stdout}${out.stderr}`;
}

// ============================================================================
// CommandMetricEvaluator
// ============================================================================

/**
 * Runs a shell command through the sandbox, parses a numeric metric out of
 * its combined stdout+stderr via a single-capture-group regex.
 *
 * Models the autoresearch pattern (`uv run train.py` ⇒ `val_bpb`).
 *
 * The log file at `log_output_to` is written **before** parsing the metric,
 * so a regex-mismatched run is still diagnosable from disk.
 */
export interface CommandMetricEvaluatorConfig {
  command: string;
  args: readonly string[];
  /** Regex with exactly one capture group. */
  metric_pattern: string;
  /** Whole seconds. Treated as crash if exceeded by the sandbox. */
  timeout: number;
  /** Path (relative to workspace root, or absolute) to write combined output. */
  log_output_to: string;
  working_dir?: string | null;
  direction: OptimizationDirection;
  description: string;
}

export class CommandMetricEvaluator implements MetricEvaluator {
  constructor(readonly config: CommandMetricEvaluatorConfig) {}

  async evaluate(
    sandbox: SandboxProvider,
    _sessionState: SessionStateSnapshot,
    signal?: AbortSignal,
  ): Promise<MetricOutcome> {
    const start = nowSeconds();
    const c = this.config;
    const out = await execOrViolation(sandbox, c.command, c.args, c.working_dir, c.timeout, signal);
    if (!("stdout" in out)) {
      return { kind: "err", error: out };
    }
    const body = combined(out);
    // Always log BEFORE parsing so a parse-fail iteration is diagnosable.
    await writeLog(sandbox, c.log_output_to, body);

    if (out.timed_out) {
      return { kind: "err", error: { kind: "timeout", after: c.timeout } };
    }
    if (out.exit_code !== 0) {
      return { kind: "err", error: { kind: "crashed", log: body } };
    }
    try {
      const value = parseMetric(body, c.metric_pattern);
      return {
        kind: "ok",
        result: {
          value,
          raw_output: body,
          duration: nowSeconds() - start,
          metadata: {
            command: c.command,
            exit_code: out.exit_code.toString(),
          },
        },
      };
    } catch (e) {
      if (e instanceof MetricErrorException) {
        return { kind: "err", error: e.error };
      }
      throw e;
    }
  }

  direction(): OptimizationDirection {
    return this.config.direction;
  }

  description(): string {
    return this.config.description;
  }
}

// ============================================================================
// TestPassRateEvaluator
// ============================================================================

/**
 * Runs a test suite, extracts pass / total counts via two regexes, reports
 * the fraction of passing tests in `[0.0, 1.0]`. Direction is fixed to
 * `maximize`.
 */
export interface TestPassRateEvaluatorConfig {
  command: string;
  args: readonly string[];
  timeout: number;
  pass_pattern: string;
  total_pattern: string;
  working_dir?: string | null;
}

export class TestPassRateEvaluator implements MetricEvaluator {
  constructor(readonly config: TestPassRateEvaluatorConfig) {}

  async evaluate(
    sandbox: SandboxProvider,
    _sessionState: SessionStateSnapshot,
    signal?: AbortSignal,
  ): Promise<MetricOutcome> {
    const start = nowSeconds();
    const c = this.config;
    const out = await execOrViolation(sandbox, c.command, c.args, c.working_dir, c.timeout, signal);
    if (!("stdout" in out)) {
      return { kind: "err", error: out };
    }
    const body = combined(out);
    if (out.timed_out) {
      return { kind: "err", error: { kind: "timeout", after: c.timeout } };
    }
    // A failing test run is a normal outcome — we still want the pass-rate.
    try {
      const pass = parseMetric(body, c.pass_pattern);
      const total = parseMetric(body, c.total_pattern);
      if (total <= 0) {
        return {
          kind: "err",
          error: {
            kind: "parse_failed",
            output: body,
            pattern: c.total_pattern,
          },
        };
      }
      return {
        kind: "ok",
        result: {
          value: pass / total,
          raw_output: body,
          duration: nowSeconds() - start,
          metadata: {
            pass: pass.toString(),
            total: total.toString(),
          },
        },
      };
    } catch (e) {
      if (e instanceof MetricErrorException) {
        return { kind: "err", error: e.error };
      }
      throw e;
    }
  }

  direction(): OptimizationDirection {
    return "maximize";
  }

  description(): string {
    return `test pass rate (${this.config.command})`;
  }
}

// ============================================================================
// LatencyEvaluator
// ============================================================================

/**
 * Measures wall-clock latency of `command`, averaged over `measured_runs`
 * trials after `warmup_runs` warm-ups. Direction is fixed to `minimize`.
 */
export interface LatencyEvaluatorConfig {
  command: string;
  args: readonly string[];
  warmup_runs: number;
  measured_runs: number;
  timeout: number;
  working_dir?: string | null;
}

export class LatencyEvaluator implements MetricEvaluator {
  constructor(readonly config: LatencyEvaluatorConfig) {}

  async evaluate(
    sandbox: SandboxProvider,
    _sessionState: SessionStateSnapshot,
    signal?: AbortSignal,
  ): Promise<MetricOutcome> {
    const c = this.config;
    if (c.measured_runs === 0) {
      return {
        kind: "err",
        error: { kind: "execution_failed", reason: "measured_runs must be > 0" },
      };
    }
    const start = nowSeconds();

    for (let i = 0; i < c.warmup_runs; i += 1) {
      const out = await execOrViolation(
        sandbox,
        c.command,
        c.args,
        c.working_dir,
        c.timeout,
        signal,
      );
      if (!("stdout" in out)) {
        return { kind: "err", error: out };
      }
    }

    let totalSeconds = 0;
    let lastOutput = "";
    for (let i = 0; i < c.measured_runs; i += 1) {
      const trialStart = nowSeconds();
      const out = await execOrViolation(
        sandbox,
        c.command,
        c.args,
        c.working_dir,
        c.timeout,
        signal,
      );
      if (!("stdout" in out)) {
        return { kind: "err", error: out };
      }
      if (out.timed_out) {
        return { kind: "err", error: { kind: "timeout", after: c.timeout } };
      }
      if (out.exit_code !== 0) {
        return {
          kind: "err",
          error: { kind: "crashed", log: combined(out) },
        };
      }
      totalSeconds += nowSeconds() - trialStart;
      lastOutput = combined(out);
    }

    const avgSeconds = totalSeconds / c.measured_runs;
    return {
      kind: "ok",
      result: {
        value: avgSeconds,
        raw_output: lastOutput,
        duration: nowSeconds() - start,
        metadata: {
          warmup_runs: c.warmup_runs.toString(),
          measured_runs: c.measured_runs.toString(),
        },
      },
    };
  }

  direction(): OptimizationDirection {
    return "minimize";
  }

  description(): string {
    return `latency (${this.config.command})`;
  }
}

// ============================================================================
// LlmJudgeEvaluator
// ============================================================================

/**
 * Identity of the judge model — flows through the results log for
 * observability. The concrete {@link ModelInterface} that dispatches the
 * judge call is supplied at construction time.
 */
export interface JudgeModelConfig {
  provider: string;
  model_id: string;
  params: ModelParams;
}

export interface LlmJudgeEvaluatorConfig {
  judge_model: JudgeModelConfig;
  rubric: string;
  /** Tuple `[lo, hi]`. The judge's score is clamped to this range, then
   *  normalized into `[0, 1]`. */
  score_range: [number, number];
  sample_input: string;
  client: ModelInterface;
}

/**
 * Uses an LLM-as-judge to score `sample_input` against `rubric`. The judge
 * is expected to emit a line `score: <number>` (case-insensitive); that
 * number is normalized into `[0.0, 1.0]` using `score_range`. Direction is
 * fixed to `maximize`.
 */
export class LlmJudgeEvaluator implements MetricEvaluator {
  constructor(readonly config: LlmJudgeEvaluatorConfig) {}

  private parseScore(text: string): number {
    // Case-insensitive `score: <number>`, first match wins. Matches the
    // Rust `(?i)score\s*:\s*([-+]?\d+(?:\.\d+)?)` regex.
    const re = /score\s*:\s*([-+]?\d+(?:\.\d+)?)/i;
    const m = re.exec(text);
    if (m == null || m[1] == null) {
      throw new MetricErrorException({
        kind: "parse_failed",
        output: text,
        pattern: "score:\\s*<number>",
      });
    }
    const raw = Number(m[1]);
    if (!Number.isFinite(raw)) {
      throw new MetricErrorException({
        kind: "parse_failed",
        output: text,
        pattern: "score:\\s*<number>",
      });
    }
    const [lo, hi] = this.config.score_range;
    if (hi <= lo) {
      throw new MetricErrorException({
        kind: "execution_failed",
        reason: `invalid score_range: (${lo}, ${hi})`,
      });
    }
    const clamped = Math.min(Math.max(raw, lo), hi);
    return (clamped - lo) / (hi - lo);
  }

  async evaluate(
    _sandbox: SandboxProvider,
    _sessionState: SessionStateSnapshot,
    signal?: AbortSignal,
  ): Promise<MetricOutcome> {
    const start = nowSeconds();
    const c = this.config;
    const prompt =
      `${c.rubric}\n\nInput to evaluate:\n${c.sample_input}\n\n` +
      `Reply with a single line \`score: <number>\` where the number is within (${c.score_range[0]}, ${c.score_range[1]}).`;
    const request: ModelRequest = {
      messages: [
        {
          role: "user",
          content: { type: "text", text: prompt },
        },
      ],
      tools: [],
      params: c.judge_model.params,
      stream: false,
    };
    let response;
    try {
      response = await c.client.call(request, signal);
    } catch (e) {
      return {
        kind: "err",
        error: {
          kind: "execution_failed",
          reason: `judge model call failed: ${(e as Error).message ?? String(e)}`,
        },
      };
    }
    const text = response.content
      .filter((b): b is { type: "text"; text: string } => b.type === "text")
      .map((b) => b.text)
      .join("\n");

    try {
      const value = this.parseScore(text);
      return {
        kind: "ok",
        result: {
          value,
          raw_output: text,
          duration: nowSeconds() - start,
          metadata: {
            judge_model: c.judge_model.model_id,
            judge_provider: c.judge_model.provider,
          },
        },
      };
    } catch (e) {
      if (e instanceof MetricErrorException) {
        return { kind: "err", error: e.error };
      }
      throw e;
    }
  }

  direction(): OptimizationDirection {
    return "maximize";
  }

  description(): string {
    return `llm judge (${this.config.judge_model.provider}/${this.config.judge_model.model_id})`;
  }
}
