/**
 * Unit tests for {@link Verifier} (spore-core issue #44).
 *
 * Mirrors `rust/crates/spore-core/src/verifier.rs#tests` — same rules,
 * same verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";

import { SessionId, verifier } from "../src/index.js";
import type {
  CommandOutput,
  HaltReason,
  RunResult,
  SandboxProvider,
  SandboxViolation,
  ToolCall,
} from "../src/index.js";

const {
  CompositeVerifier,
  EvaluatorResponseVerifier,
  TestSuiteVerifier,
  COMPOSITE_REASON_CAP,
  DEFAULT_MAX_ITERATIONS,
  failed,
  passed,
} = verifier;

type Verifier = verifier.Verifier;
type VerifierInput = verifier.VerifierInput;
type VerifierVerdict = verifier.VerifierVerdict;

// ── helpers ──────────────────────────────────────────────────────────────────

function success(output: string): RunResult {
  return {
    kind: "success",
    output,
    session_id: SessionId.of("s"),
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      cost_usd: 0,
    },
    turns: 1,
  };
}

function failure(
  reason: HaltReason = { kind: "strategy_not_yet_implemented", strategy: "x" },
): RunResult {
  return {
    kind: "failure",
    reason,
    session_id: SessionId.of("s"),
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: 0,
      cache_write_tokens: 0,
      cost_usd: 0,
    },
    turns: 0,
  };
}

function inputWith(build: RunResult, evalR: RunResult): VerifierInput {
  return {
    build_result: build,
    eval_result: evalR,
    workspace: "/tmp",
    iteration: 0,
  };
}

function makeRespVerifier(): verifier.EvaluatorResponseVerifier {
  return new EvaluatorResponseVerifier({
    pass_pattern: "(?i)\\bPASS\\b",
    fail_pattern: "(?i)\\bFAIL: .+",
  });
}

// ── EvaluatorResponseVerifier ────────────────────────────────────────────────

describe("EvaluatorResponseVerifier", () => {
  it("pass pattern matches → Passed", async () => {
    const v = makeRespVerifier();
    const i = inputWith(success("ok"), success("all checks PASS, ready to ship"));
    expect(await v.verify(i)).toEqual(passed());
  });

  it("fail pattern matches → Failed with reason", async () => {
    const v = makeRespVerifier();
    const i = inputWith(success("ok"), success("FAIL: missing edge case in handler.rs"));
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason).toContain("missing edge case");
    }
  });

  it("neither pattern → default Failed mentioning output", async () => {
    const v = makeRespVerifier();
    const i = inputWith(success("ok"), success("indeterminate output"));
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason).toContain("matched neither");
      expect(verdict.reason).toContain("indeterminate output");
    }
  });

  it("build failure propagates", async () => {
    const v = makeRespVerifier();
    const i = inputWith(failure(), success("PASS"));
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason.startsWith("build run halted")).toBe(true);
    }
  });

  it("eval failure propagates", async () => {
    const v = makeRespVerifier();
    const i = inputWith(success("ok"), failure());
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason.startsWith("evaluator run halted")).toBe(true);
    }
  });

  it("max_iterations defaults to 3 and is overridable", () => {
    const v = makeRespVerifier();
    expect(v.maxIterations()).toBe(DEFAULT_MAX_ITERATIONS);
    expect(v.maxIterations()).toBe(3);
    const v2 = new EvaluatorResponseVerifier({
      pass_pattern: "a",
      fail_pattern: "b",
      max_iterations: 10,
    });
    expect(v2.maxIterations()).toBe(10);
  });
});

// ── TestSuiteVerifier ────────────────────────────────────────────────────────

class StubSandbox implements SandboxProvider {
  constructor(private readonly out: CommandOutput) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    void _call;
    return null;
  }
  async executeCommand(): Promise<CommandOutput | SandboxViolation> {
    return this.out;
  }
}

function stubSandbox(exit: number, stderr: string): SandboxProvider {
  return new StubSandbox({
    stdout: "",
    stderr,
    exit_code: exit,
    timed_out: false,
    truncated: false,
  });
}

describe("TestSuiteVerifier", () => {
  it("exit 0 → Passed", async () => {
    const v = new TestSuiteVerifier({
      command: "pnpm test",
      working_dir: "/work",
      timeout_ms: 60_000,
      sandbox: stubSandbox(0, ""),
    });
    const i = inputWith(success("ok"), success(""));
    expect(await v.verify(i)).toEqual(passed());
  });

  it("non-zero exit → Failed includes stderr tail", async () => {
    const v = new TestSuiteVerifier({
      command: "pnpm test",
      working_dir: "/work",
      timeout_ms: 60_000,
      sandbox: stubSandbox(1, "test foo ... FAILED"),
    });
    const i = inputWith(success("ok"), success(""));
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason).toContain("FAILED");
    }
  });

  it("build failure short-circuits", async () => {
    const v = new TestSuiteVerifier({
      command: "pnpm test",
      working_dir: "/work",
      timeout_ms: 60_000,
      sandbox: stubSandbox(0, ""),
    });
    const i = inputWith(failure(), success(""));
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason.startsWith("build run halted")).toBe(true);
    }
  });

  it("empty command → Failed", async () => {
    const v = new TestSuiteVerifier({
      command: "",
      working_dir: "/work",
      timeout_ms: 60_000,
      sandbox: stubSandbox(0, ""),
    });
    const i = inputWith(success("ok"), success(""));
    const verdict = await v.verify(i);
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason).toContain("empty test command");
    }
  });
});

// ── CompositeVerifier ────────────────────────────────────────────────────────

class FixedVerifier implements Verifier {
  constructor(private readonly verdict: VerifierVerdict) {}
  async verify(): Promise<VerifierVerdict> {
    return this.verdict;
  }
  maxIterations(): number {
    return DEFAULT_MAX_ITERATIONS;
  }
}

const passV = (): Verifier => new FixedVerifier(passed());
const failV = (reason: string): Verifier => new FixedVerifier(failed(reason));

describe("CompositeVerifier", () => {
  it("all pass → Passed", async () => {
    const c = new CompositeVerifier({ verifiers: [passV(), passV(), passV()] });
    expect(await c.verify(inputWith(success("ok"), success("ok")))).toEqual(passed());
  });

  it("one fail → Failed with that reason and tagged index", async () => {
    const c = new CompositeVerifier({ verifiers: [passV(), failV("oops"), passV()] });
    const verdict = await c.verify(inputWith(success("ok"), success("ok")));
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason).toContain("oops");
      expect(verdict.reason).toContain("[verifier 1]");
    }
  });

  it("many fails concatenated, passing children not mentioned", async () => {
    const c = new CompositeVerifier({
      verifiers: [failV("first"), passV(), failV("second"), failV("third")],
    });
    const verdict = await c.verify(inputWith(success("ok"), success("ok")));
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason).toContain("first");
      expect(verdict.reason).toContain("second");
      expect(verdict.reason).toContain("third");
      expect(verdict.reason).not.toContain("[verifier 1]");
    }
  });

  it("truncates at 2000 chars", async () => {
    const long = "x".repeat(5000);
    const c = new CompositeVerifier({ verifiers: [failV(long)] });
    const verdict = await c.verify(inputWith(success("ok"), success("ok")));
    expect(verdict.kind).toBe("failed");
    if (verdict.kind === "failed") {
      expect(verdict.reason.length).toBeLessThanOrEqual(
        COMPOSITE_REASON_CAP + "... [truncated]".length,
      );
      expect(verdict.reason.endsWith("... [truncated]")).toBe(true);
    }
  });

  it("max_iterations honored", () => {
    const c = new CompositeVerifier({ verifiers: [passV()], max_iterations: 7 });
    expect(c.maxIterations()).toBe(7);
  });
});
