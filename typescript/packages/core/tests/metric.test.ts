/**
 * Unit tests for {@link MetricEvaluator} (spore-core issue #23).
 *
 * Covers every rule in the spec, every error variant, the keep/revert
 * decision, and integration of each standard evaluator with a fake sandbox.
 */

import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  SessionId,
  TaskId,
  emptySessionState,
  type CommandOutput,
  type SandboxProvider,
  type SandboxViolation,
} from "../src/harness/types.js";
import { metric } from "../src/index.js";
import { newSessionStateSnapshot, type SessionStateSnapshot } from "../src/termination/types.js";
import type {
  ModelInterface,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StreamEvent,
} from "../src/model/index.js";
import type { ToolCall } from "../src/model/schemas.js";

const {
  CommandMetricEvaluator,
  LatencyEvaluator,
  LlmJudgeEvaluator,
  TestPassRateEvaluator,
  iterationStatusFromError,
  metricErrorMessage,
  newMetricResult,
  parseMetric,
  shouldKeep,
} = metric;

type MetricError = metric.MetricError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function snapshot(): SessionStateSnapshot {
  return newSessionStateSnapshot(new SessionId("sess"), new TaskId("task"), emptySessionState());
}

class FakeSandbox implements SandboxProvider {
  stdout = "";
  stderr = "";
  exit_code = 0;
  timed_out = false;
  root: string;
  calls = 0;

  constructor(root: string, stdout = "") {
    this.root = root;
    this.stdout = stdout;
  }

  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }

  async executeCommand(
    _command: string,
    _args: readonly string[],
    _cwd?: string | null,
    _timeoutMs?: number | null,
    _signal?: AbortSignal,
  ): Promise<CommandOutput | SandboxViolation> {
    this.calls += 1;
    return {
      stdout: this.stdout,
      stderr: this.stderr,
      exit_code: this.exit_code,
      timed_out: this.timed_out,
      truncated: false,
    };
  }

  workspaceRoot(): string {
    return this.root;
  }
}

let tmpRoot: string;
beforeEach(async () => {
  tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), "metric-test-"));
});
afterEach(async () => {
  await fs.rm(tmpRoot, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// shouldKeep
// ---------------------------------------------------------------------------

describe("shouldKeep", () => {
  it("minimize: lower is better", () => {
    expect(shouldKeep(1.0, 2.0, "minimize", null)).toBe(true);
    expect(shouldKeep(2.0, 1.0, "minimize", null)).toBe(false);
  });
  it("maximize: higher is better", () => {
    expect(shouldKeep(2.0, 1.0, "maximize", null)).toBe(true);
    expect(shouldKeep(1.0, 2.0, "maximize", null)).toBe(false);
  });
  it("equal is discarded for both directions", () => {
    expect(shouldKeep(1.0, 1.0, "minimize", null)).toBe(false);
    expect(shouldKeep(1.0, 1.0, "maximize", null)).toBe(false);
  });
  it("respects min_delta strictly (equal-to-delta discarded)", () => {
    expect(shouldKeep(1.5, 2.0, "minimize", 0.5)).toBe(false);
    expect(shouldKeep(1.49, 2.0, "minimize", 0.5)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// parseMetric
// ---------------------------------------------------------------------------

describe("parseMetric", () => {
  it("extracts capture group as float", () => {
    expect(parseMetric("val_bpb:  3.125\nother", "val_bpb:\\s+([\\d.]+)")).toBeCloseTo(3.125, 9);
  });
  it("no match throws ParseFailed", () => {
    expect(() => parseMetric("no metric here", "val_bpb:\\s+([\\d.]+)")).toThrow();
    try {
      parseMetric("no metric here", "val_bpb:\\s+([\\d.]+)");
    } catch (e) {
      expect((e as metric.MetricErrorException).error.kind).toBe("parse_failed");
    }
  });
  it("unparseable capture is ParseFailed", () => {
    try {
      parseMetric("val_bpb: oops", "val_bpb:\\s+(\\S+)");
      throw new Error("should have thrown");
    } catch (e) {
      expect((e as metric.MetricErrorException).error.kind).toBe("parse_failed");
    }
  });
  it("invalid regex is ExecutionFailed", () => {
    try {
      parseMetric("x", "(unbalanced");
      throw new Error("should have thrown");
    } catch (e) {
      expect((e as metric.MetricErrorException).error.kind).toBe("execution_failed");
    }
  });
});

// ---------------------------------------------------------------------------
// IterationStatus / MetricError messages
// ---------------------------------------------------------------------------

describe("iterationStatusFromError", () => {
  it("maps timeout to timeout", () => {
    expect(iterationStatusFromError({ kind: "timeout", after: 1 })).toBe("timeout");
  });
  it("maps others to crashed", () => {
    const errs: MetricError[] = [
      { kind: "crashed", log: "x" },
      { kind: "execution_failed", reason: "x" },
      { kind: "parse_failed", output: "", pattern: "" },
    ];
    for (const e of errs) {
      expect(iterationStatusFromError(e)).toBe("crashed");
    }
  });
});

describe("metricErrorMessage", () => {
  it("renders every variant", () => {
    expect(metricErrorMessage({ kind: "execution_failed", reason: "x" })).toContain("x");
    expect(metricErrorMessage({ kind: "timeout", after: 5 })).toContain("5");
    expect(metricErrorMessage({ kind: "parse_failed", output: "", pattern: "p" })).toContain("p");
    expect(metricErrorMessage({ kind: "crashed", log: "boom" })).toContain("boom");
  });
});

describe("newMetricResult", () => {
  it("constructs with sane defaults", () => {
    const r = newMetricResult(0.5);
    expect(r.value).toBe(0.5);
    expect(r.raw_output).toBe("");
    expect(r.duration).toBe(0);
    expect(r.metadata).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// CommandMetricEvaluator
// ---------------------------------------------------------------------------

describe("CommandMetricEvaluator", () => {
  it("happy path: writes log file before parsing and returns value", async () => {
    const sb = new FakeSandbox(tmpRoot, "val_bpb: 1.234\n");
    const ev = new CommandMetricEvaluator({
      command: "uv",
      args: ["run", "train.py"],
      metric_pattern: "val_bpb:\\s+([\\d.]+)",
      timeout: 60,
      log_output_to: "run.log",
      direction: "minimize",
      description: "autoresearch val_bpb",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") {
      expect(r.result.value).toBeCloseTo(1.234, 9);
      expect(r.result.metadata.command).toBe("uv");
    }
    const body = await fs.readFile(path.join(tmpRoot, "run.log"), "utf-8");
    expect(body).toContain("val_bpb");
    expect(ev.direction()).toBe("minimize");
    expect(ev.description()).toBe("autoresearch val_bpb");
  });

  it("timeout maps to Timeout error", async () => {
    const sb = new FakeSandbox(tmpRoot, "");
    sb.timed_out = true;
    const ev = new CommandMetricEvaluator({
      command: "x",
      args: [],
      metric_pattern: "v:(\\d+)",
      timeout: 0.001,
      log_output_to: "run.log",
      direction: "minimize",
      description: "x",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("timeout");
  });

  it("non-zero exit is Crashed", async () => {
    const sb = new FakeSandbox(tmpRoot, "boom");
    sb.exit_code = 1;
    const ev = new CommandMetricEvaluator({
      command: "x",
      args: [],
      metric_pattern: "v:(\\d+)",
      timeout: 1,
      log_output_to: "run.log",
      direction: "minimize",
      description: "x",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("crashed");
  });

  it("regex mismatch is ParseFailed (but log is still written)", async () => {
    const sb = new FakeSandbox(tmpRoot, "no metric");
    const ev = new CommandMetricEvaluator({
      command: "x",
      args: [],
      metric_pattern: "v:(\\d+)",
      timeout: 1,
      log_output_to: "run.log",
      direction: "minimize",
      description: "x",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("parse_failed");
    const body = await fs.readFile(path.join(tmpRoot, "run.log"), "utf-8");
    expect(body).toBe("no metric");
  });

  it("invalid regex is ExecutionFailed", async () => {
    const sb = new FakeSandbox(tmpRoot, "anything");
    const ev = new CommandMetricEvaluator({
      command: "x",
      args: [],
      metric_pattern: "(unbalanced",
      timeout: 1,
      log_output_to: "run.log",
      direction: "minimize",
      description: "x",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("execution_failed");
  });
});

// ---------------------------------------------------------------------------
// TestPassRateEvaluator
// ---------------------------------------------------------------------------

describe("TestPassRateEvaluator", () => {
  it("returns pass / total fraction", async () => {
    const sb = new FakeSandbox(tmpRoot, "passed 17 of 20");
    const ev = new TestPassRateEvaluator({
      command: "pytest",
      args: [],
      timeout: 60,
      pass_pattern: "passed (\\d+)",
      total_pattern: "of (\\d+)",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") {
      expect(r.result.value).toBeCloseTo(0.85, 9);
      expect(r.result.metadata.pass).toBe("17");
      expect(r.result.metadata.total).toBe("20");
    }
    expect(ev.direction()).toBe("maximize");
    expect(ev.description()).toBe("test pass rate (pytest)");
  });

  it("zero total is ParseFailed", async () => {
    const sb = new FakeSandbox(tmpRoot, "passed 0 of 0");
    const ev = new TestPassRateEvaluator({
      command: "pytest",
      args: [],
      timeout: 60,
      pass_pattern: "passed (\\d+)",
      total_pattern: "of (\\d+)",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("parse_failed");
  });

  it("timeout maps to Timeout", async () => {
    const sb = new FakeSandbox(tmpRoot, "");
    sb.timed_out = true;
    const ev = new TestPassRateEvaluator({
      command: "pytest",
      args: [],
      timeout: 1,
      pass_pattern: "p (\\d+)",
      total_pattern: "t (\\d+)",
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("timeout");
  });
});

// ---------------------------------------------------------------------------
// LatencyEvaluator
// ---------------------------------------------------------------------------

describe("LatencyEvaluator", () => {
  it("averages measured_runs after warmup", async () => {
    const sb = new FakeSandbox(tmpRoot, "ok");
    const ev = new LatencyEvaluator({
      command: "echo",
      args: ["ok"],
      warmup_runs: 1,
      measured_runs: 2,
      timeout: 5,
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") {
      expect(r.result.value).toBeGreaterThanOrEqual(0);
      expect(r.result.metadata.measured_runs).toBe("2");
      expect(r.result.metadata.warmup_runs).toBe("1");
    }
    expect(sb.calls).toBe(3);
    expect(ev.direction()).toBe("minimize");
    expect(ev.description()).toBe("latency (echo)");
  });

  it("zero measured_runs is rejected as ExecutionFailed", async () => {
    const sb = new FakeSandbox(tmpRoot, "");
    const ev = new LatencyEvaluator({
      command: "x",
      args: [],
      warmup_runs: 0,
      measured_runs: 0,
      timeout: 1,
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("execution_failed");
  });

  it("non-zero exit during measured run is Crashed", async () => {
    const sb = new FakeSandbox(tmpRoot, "fail");
    sb.exit_code = 1;
    const ev = new LatencyEvaluator({
      command: "x",
      args: [],
      warmup_runs: 0,
      measured_runs: 1,
      timeout: 1,
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("crashed");
  });

  it("timed-out measured run is Timeout", async () => {
    const sb = new FakeSandbox(tmpRoot, "");
    sb.timed_out = true;
    const ev = new LatencyEvaluator({
      command: "x",
      args: [],
      warmup_runs: 0,
      measured_runs: 1,
      timeout: 1,
    });
    const r = await ev.evaluate(sb, snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("timeout");
  });
});

// ---------------------------------------------------------------------------
// LlmJudgeEvaluator
// ---------------------------------------------------------------------------

class FakeModel implements ModelInterface {
  constructor(
    private readonly text: string,
    private readonly throws = false,
  ) {}

  async call(_req: ModelRequest): Promise<ModelResponse> {
    if (this.throws) throw new Error("network sad");
    return {
      content: [{ type: "text", text: this.text }],
      usage: { input_tokens: 0, output_tokens: 0 },
      stop_reason: "end_turn",
    };
  }

  // eslint-disable-next-line @typescript-eslint/require-await
  async *callStreaming(_req: ModelRequest): AsyncIterable<StreamEvent> {
    throw new Error("unimplemented");
  }

  async countTokens(_req: ModelRequest): Promise<number> {
    return 0;
  }

  provider(): ProviderInfo {
    return { name: "fake", model_id: "fake", context_window: 8000 };
  }
}

describe("LlmJudgeEvaluator", () => {
  function build(text: string, throws = false): metric.LlmJudgeEvaluator {
    return new LlmJudgeEvaluator({
      judge_model: {
        provider: "fake",
        model_id: "judge-1",
        params: { stop_sequences: [] },
      },
      rubric: "rate this",
      score_range: [0, 10],
      sample_input: "x",
      client: new FakeModel(text, throws),
    });
  }

  it("normalizes score into [0, 1]", async () => {
    const ev = build("score: 7.5");
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") {
      expect(r.result.value).toBeCloseTo(0.75, 9);
      expect(r.result.metadata.judge_model).toBe("judge-1");
      expect(r.result.metadata.judge_provider).toBe("fake");
    }
    expect(ev.direction()).toBe("maximize");
    expect(ev.description()).toBe("llm judge (fake/judge-1)");
  });

  it("clamps scores above range to 1.0", async () => {
    const ev = build("score: 42");
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") expect(r.result.value).toBeCloseTo(1.0, 9);
  });

  it("clamps scores below range to 0.0", async () => {
    const ev = build("score: -5");
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") expect(r.result.value).toBeCloseTo(0.0, 9);
  });

  it("is case-insensitive for the 'score:' literal", async () => {
    const ev = build("Final SCORE: 5.0");
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("ok");
    if (r.kind === "ok") expect(r.result.value).toBeCloseTo(0.5, 9);
  });

  it("missing score is ParseFailed", async () => {
    const ev = build("no score in here");
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("parse_failed");
  });

  it("model call failure is ExecutionFailed", async () => {
    const ev = build("score: 1", true);
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("execution_failed");
  });

  it("invalid score_range is ExecutionFailed", async () => {
    const ev = new LlmJudgeEvaluator({
      judge_model: {
        provider: "fake",
        model_id: "judge-1",
        params: { stop_sequences: [] },
      },
      rubric: "rate",
      score_range: [5, 5],
      sample_input: "x",
      client: new FakeModel("score: 5"),
    });
    const r = await ev.evaluate(new FakeSandbox(tmpRoot), snapshot());
    expect(r.kind).toBe("err");
    if (r.kind === "err") expect(r.error.kind).toBe("execution_failed");
  });
});

// ---------------------------------------------------------------------------
// MetricError wire shape
// ---------------------------------------------------------------------------

describe("MetricError wire format", () => {
  it("every variant carries snake_case kind", () => {
    const variants: MetricError[] = [
      { kind: "execution_failed", reason: "x" },
      { kind: "timeout", after: 1 },
      { kind: "parse_failed", output: "x", pattern: "p" },
      { kind: "crashed", log: "boom" },
    ];
    for (const v of variants) {
      const json = JSON.parse(JSON.stringify(v)) as Record<string, unknown>;
      expect(json.kind).toBe(v.kind);
    }
  });
});
