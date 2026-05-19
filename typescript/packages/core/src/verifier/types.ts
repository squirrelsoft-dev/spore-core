/**
 * Verifier — canonical types (spore-core issue #44).
 *
 * Mirrors `rust/crates/spore-core/src/verifier.rs`. The `Verifier` is the
 * oracle for the `SelfVerifying` loop strategy: it sits between an
 * evaluator harness's {@link RunResult} and the build loop's halt decision,
 * translating `(build_result, eval_result)` into a {@link VerifierVerdict}
 * — either `passed` (halt with success) or `failed` (re-enter the build
 * loop with `reason` injected into the next turn's context).
 *
 * ## Ambiguity resolutions (see issue #44 comment thread)
 *
 * 1. {@link EvaluatorResponseVerifier} when neither `pass_pattern` nor
 *    `fail_pattern` matches → `failed` with a descriptive reason including
 *    a truncated copy of the output. Default-FAIL is **not** configurable.
 * 2. Any non-`success` {@link RunResult} in `build_result` or `eval_result`
 *    → `failed`. `waiting_for_human` is treated as a misconfiguration
 *    signal and surfaced in the reason.
 * 3. {@link CompositeVerifier} concatenates all child failure reasons
 *    (joined by `"\n"`), capped at 2000 characters total. Children that
 *    pass are not mentioned.
 * 4. `LoopStrategy.self_verifying` wiring is **deferred** — bundled with
 *    the Ralph wiring and issue #45. The strategy continues to return
 *    `strategy_not_yet_implemented` in the harness loop.
 */

import { z } from "zod";

import type { CommandOutput, HaltReason, RunResult, SandboxProvider } from "../harness/types.js";

// ============================================================================
// VerifierVerdict
// ============================================================================

export const VerifierVerdictSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("passed") }),
  z.object({ kind: z.literal("failed"), reason: z.string() }),
]);

export type VerifierVerdict = z.infer<typeof VerifierVerdictSchema>;

/** Convenience constructor for the `failed` variant. */
export function failed(reason: string): VerifierVerdict {
  return { kind: "failed", reason };
}

/** Convenience constructor for the `passed` variant. */
export function passed(): VerifierVerdict {
  return { kind: "passed" };
}

// ============================================================================
// VerifierInput
// ============================================================================

export interface VerifierInput {
  build_result: RunResult;
  eval_result: RunResult;
  /** Workspace path (caller-supplied; verifier does not interpret). */
  workspace: string;
  /** Which build-evaluate cycle this is (0-indexed). */
  iteration: number;
}

// ============================================================================
// Verifier interface
// ============================================================================

export const DEFAULT_MAX_ITERATIONS = 3;

/**
 * Verdict oracle for the `SelfVerifying` loop strategy.
 *
 * Implementations should be `Send + Sync`-equivalent — they are held by
 * value (or by shared reference) inside the harness loop.
 */
export interface Verifier {
  verify(input: VerifierInput, signal?: AbortSignal): Promise<VerifierVerdict>;

  /**
   * Maximum number of build-evaluate cycles before the harness halts the
   * loop regardless of verdict. Prevents infinite loops when the evaluator
   * always finds problems. Spec default: 3.
   */
  maxIterations(): number;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Common reduction of a {@link RunResult} to either its success output or a
 * descriptive failure reason.
 */
type ResultView = { kind: "output"; output: string } | { kind: "failed"; reason: string };

function view(label: string, r: RunResult): ResultView {
  switch (r.kind) {
    case "success":
      return { kind: "output", output: r.output };
    case "failure":
      return {
        kind: "failed",
        reason: `${label} run halted: ${describeHalt(r.reason)}`,
      };
    case "waiting_for_human":
      return {
        kind: "failed",
        reason:
          `${label} run is WaitingForHuman — verifier received a paused harness; ` +
          `this is a misconfiguration signal (the ${label} should run to completion ` +
          `before being verified)`,
      };
  }
}

function describeHalt(reason: HaltReason): string {
  // HaltReason is `#[non_exhaustive]` in Rust — the verifier treats this as
  // opaque diagnostic text. We serialize via JSON so the shape stays stable
  // across the cross-language fixture.
  try {
    return JSON.stringify(reason);
  } catch {
    return String(reason);
  }
}

const TRUNCATION_MARKER = "... [truncated]";

function truncateForReason(s: string, max: number): string {
  if (s.length <= max) {
    return s;
  }
  return s.slice(0, max) + TRUNCATION_MARKER;
}

function tailLines(s: string, n: number): string {
  const lines = s.split("\n");
  const start = Math.max(0, lines.length - n);
  return lines.slice(start).join("\n");
}

// ============================================================================
// EvaluatorResponseVerifier
// ============================================================================

export interface EvaluatorResponseVerifierOptions {
  /** Pattern that, when matched, yields a `passed` verdict. */
  pass_pattern: string;
  /** Pattern that, when matched, yields a `failed` verdict. */
  fail_pattern: string;
  /** Maximum build-evaluate cycles. Defaults to {@link DEFAULT_MAX_ITERATIONS}. */
  max_iterations?: number;
}

/**
 * Pattern-matches the evaluator harness's final text response. The simplest
 * verifier — trusts whatever the evaluator wrote.
 *
 * Rules:
 *   - If `build_result` is not `success` → `failed` with the halt reason.
 *   - If `eval_result` is not `success` → `failed` with the halt reason.
 *   - If `pass_pattern` matches the eval output → `passed`.
 *   - If `fail_pattern` matches the eval output → `failed` with the matched
 *     fragment as the reason.
 *   - Neither matches → `failed` with a descriptive default reason
 *     (Default-FAIL contract; not configurable).
 */
export class EvaluatorResponseVerifier implements Verifier {
  readonly pass_pattern: RegExp;
  readonly fail_pattern: RegExp;
  readonly max_iterations: number;

  constructor(options: EvaluatorResponseVerifierOptions) {
    this.pass_pattern = compileFixturePattern(options.pass_pattern);
    this.fail_pattern = compileFixturePattern(options.fail_pattern);
    this.max_iterations = options.max_iterations ?? DEFAULT_MAX_ITERATIONS;
  }

  async verify(input: VerifierInput, _signal?: AbortSignal): Promise<VerifierVerdict> {
    void _signal;
    const buildView = view("build", input.build_result);
    if (buildView.kind === "failed") {
      return failed(buildView.reason);
    }
    const evalView = view("evaluator", input.eval_result);
    if (evalView.kind === "failed") {
      return failed(evalView.reason);
    }
    const output = evalView.output;
    const passMatch = this.pass_pattern.exec(output);
    if (passMatch !== null) {
      return passed();
    }
    const failMatch = this.fail_pattern.exec(output);
    if (failMatch !== null) {
      return failed(`evaluator reported failure: ${truncateForReason(failMatch[0], 500)}`);
    }
    return failed(
      `evaluator output matched neither pass_pattern (\`${this.pass_pattern.source}\`) nor ` +
        `fail_pattern (\`${this.fail_pattern.source}\`). Output was:\n` +
        truncateForReason(output, 1000),
    );
  }

  maxIterations(): number {
    return this.max_iterations;
  }
}

/**
 * Compile a regex string from the cross-language fixture format
 * (Rust `regex` crate syntax) into a JavaScript `RegExp`.
 *
 * Supports the leading `(?i)` inline flag — Rust accepts it inline,
 * JavaScript does not, so we strip it and set the `i` flag.
 */
function compileFixturePattern(source: string): RegExp {
  let flags = "";
  let body = source;
  // Repeatedly strip leading inline flag groups: (?i), (?im), etc.
  while (true) {
    const m = body.match(/^\(\?([imsux]+)\)/);
    if (m === null) break;
    for (const ch of m[1]) {
      if (ch === "i" && !flags.includes("i")) flags += "i";
      if (ch === "m" && !flags.includes("m")) flags += "m";
      if (ch === "s" && !flags.includes("s")) flags += "s";
      // 'u' / 'x' have no direct JS equivalent that's safe to set
      // unconditionally; ignore.
    }
    body = body.slice(m[0].length);
  }
  return new RegExp(body, flags);
}

// ============================================================================
// TestSuiteVerifier
// ============================================================================

export interface TestSuiteVerifierOptions {
  /** Whitespace-separated command (e.g. `"pnpm test"`). */
  command: string;
  /** Working directory for the test command. */
  working_dir: string;
  /** Timeout in milliseconds. */
  timeout_ms: number;
  /** Sandbox to run the command through. */
  sandbox: SandboxProvider;
  /** Maximum build-evaluate cycles. Defaults to {@link DEFAULT_MAX_ITERATIONS}. */
  max_iterations?: number;
}

/**
 * Runs a test command via the injected {@link SandboxProvider} and uses
 * the exit code as the verdict. Ignores the evaluator's text output —
 * ground truth is the tests.
 *
 * Rules:
 *   - If `build_result` is not `success` → `failed` with the halt reason.
 *   - Run `command` in `working_dir` via `sandbox.executeCommand`.
 *   - Exit 0, not timed out → `passed`.
 *   - Anything else → `failed` with a stderr/stdout tail.
 */
export class TestSuiteVerifier implements Verifier {
  readonly command: string;
  readonly working_dir: string;
  readonly timeout_ms: number;
  readonly sandbox: SandboxProvider;
  readonly max_iterations: number;

  constructor(options: TestSuiteVerifierOptions) {
    this.command = options.command;
    this.working_dir = options.working_dir;
    this.timeout_ms = options.timeout_ms;
    this.sandbox = options.sandbox;
    this.max_iterations = options.max_iterations ?? DEFAULT_MAX_ITERATIONS;
  }

  async verify(input: VerifierInput, signal?: AbortSignal): Promise<VerifierVerdict> {
    const buildView = view("build", input.build_result);
    if (buildView.kind === "failed") {
      return failed(buildView.reason);
    }
    const parts = this.command.split(/\s+/).filter((p) => p.length > 0);
    if (parts.length === 0) {
      return failed("empty test command");
    }
    const [program, ...args] = parts;

    if (!this.sandbox.executeCommand) {
      return failed("sandbox does not support executeCommand");
    }
    const result = await this.sandbox.executeCommand(
      program,
      args,
      this.working_dir,
      this.timeout_ms,
      signal,
    );

    // SandboxViolation vs CommandOutput: violations don't have `exit_code`.
    if (!isCommandOutput(result)) {
      return failed(`sandbox refused test command: ${JSON.stringify(result)}`);
    }
    if (result.exit_code === 0 && !result.timed_out) {
      return passed();
    }
    let tail = tailLines(result.stderr, 20);
    if (tail.trim().length === 0) {
      tail = tailLines(result.stdout, 20);
    }
    return failed(
      `test suite failed (exit ${result.exit_code}, timed_out=${result.timed_out}):\n${tail}`,
    );
  }

  maxIterations(): number {
    return this.max_iterations;
  }
}

function isCommandOutput(v: unknown): v is CommandOutput {
  return (
    typeof v === "object" &&
    v !== null &&
    "exit_code" in v &&
    "stdout" in v &&
    "stderr" in v &&
    "timed_out" in v
  );
}

// ============================================================================
// CompositeVerifier
// ============================================================================

export const COMPOSITE_REASON_CAP = 2000;

export interface CompositeVerifierOptions {
  verifiers: Verifier[];
  max_iterations?: number;
}

/**
 * Passes only when **all** child verifiers pass. On failure, concatenates
 * every child's failure reason (joined by `"\n"`), capped at 2000
 * characters total. Children that pass are not mentioned in the failure
 * reason.
 */
export class CompositeVerifier implements Verifier {
  readonly verifiers: Verifier[];
  readonly max_iterations: number;

  constructor(options: CompositeVerifierOptions) {
    this.verifiers = options.verifiers;
    this.max_iterations = options.max_iterations ?? DEFAULT_MAX_ITERATIONS;
  }

  async verify(input: VerifierInput, signal?: AbortSignal): Promise<VerifierVerdict> {
    const failures: string[] = [];
    for (let i = 0; i < this.verifiers.length; i++) {
      const v = this.verifiers[i];
      const verdict = await v.verify(input, signal);
      if (verdict.kind === "failed") {
        failures.push(`[verifier ${i}] ${verdict.reason}`);
      }
    }
    if (failures.length === 0) {
      return passed();
    }
    return failed(truncateForReason(failures.join("\n"), COMPOSITE_REASON_CAP));
  }

  maxIterations(): number {
    return this.max_iterations;
  }
}
