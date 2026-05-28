/**
 * Rule-by-rule tests for the EvalHarness (Rules 1-29), the interface-only
 * compile test (Rule 30), the promote test (Rule 31), the no-Inspect/Langfuse
 * dependency assertion (Rule 32), the E2E hermetic regression test, and the
 * fixture-replay tests.
 *
 * All tests are hermetic: MockAgent (no network) + InMemoryObservabilityProvider.
 */

import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyResponse,
  HarnessBuilder,
  MockAgent,
  MockModelInterface,
  SessionId,
  harnessTesting,
  observability,
  type HarnessConfig,
  type RunResult,
  type Span,
  type TokenUsage,
} from "@spore/core";

const { InMemoryObservabilityProvider } = observability;

import {
  AlwaysFail,
  AlwaysPass,
  CompositeVerifier,
  EvalHarnessBuilder,
  LlmJudgeVerifier,
  MetricEvaluatorVerifier,
  Workspace,
  allTasks,
  bootstrapCi,
  buildVerifier,
  classifyDirection,
  clampedVerificationResult,
  deriveRecommendation,
  loadSuitePath,
  loadSuiteStr,
  metricDirection,
  metricName,
  newVerificationResult,
  promoteChallengeTask,
  sampleFor,
  suiteToJson,
  taskVerifier,
  welchTTest,
  DEFAULT_BOOTSTRAP_SEED,
  type ComparisonReport,
  type EvalTask,
  type HarnessConfigDiff,
  type MetricStats,
  type TaskSuite,
  type TraceAnalyzer,
  type VerifierSpec,
  type WorkspaceSnapshot,
} from "../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "../../../../fixtures/task_suites");

// ============================================================================
// Helpers
// ============================================================================

function usage(): TokenUsage {
  return { input_tokens: 10, output_tokens: 5 };
}

/** A HarnessConfig wired to a shared observability provider, with a MockAgent
 *  that produces `nRunsNeeded` final responses (one per run). `success`
 *  controls whether each run succeeds or errors out (a weaker config). */
function configWith(
  obs: InMemoryObservabilityProvider,
  success: boolean,
  nRunsNeeded: number,
): HarnessConfig {
  const agent = new MockAgent(AgentId.of("mock"));
  for (let i = 0; i < Math.max(1, nRunsNeeded); i++) {
    if (success) {
      agent.push({ kind: "final_response", content: "DONE", usage: usage() });
    } else {
      agent.push({ kind: "error", error: new EmptyResponse(), usage: usage() });
    }
  }
  return new HarnessBuilder(
    agent,
    new harnessTesting.ScriptedToolRegistry(),
    new harnessTesting.AllowAllSandbox(),
    new harnessTesting.NoopContextManager(),
    new harnessTesting.AlwaysContinuePolicy(),
  )
    .observability(obs)
    .buildConfig();
}

function task(
  id: string,
  snapshot: WorkspaceSnapshot,
  spec: VerifierSpec,
): EvalTask {
  const t: EvalTask = {
    id,
    instruction: "do the thing",
    workspace_snapshot: snapshot,
    verifier_spec: spec,
    expected_turns: [1, 4],
    expected_cost_usd: null,
    tags: ["unit"],
    timeout: 30,
    model_fixture: null,
  };
  t.verifier = buildVerifier(spec);
  return t;
}

function files(pairs: Record<string, string>): WorkspaceSnapshot {
  return { kind: "files", files: pairs };
}

function runResultSuccess(): RunResult {
  return {
    kind: "success",
    output: "DONE",
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

// ============================================================================
// Rule 1 — three disjoint task lists
// ============================================================================

describe("Rule 1 — three disjoint task lists", () => {
  it("exposes all tasks across categories", () => {
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [task("r1", { kind: "empty" }, { kind: "always_pass" })],
      challenge: [task("c1", { kind: "empty" }, { kind: "always_pass" })],
      canary: [task("k1", { kind: "empty" }, { kind: "always_pass" })],
    };
    expect(allTasks(suite).length).toBe(3);
  });
});

// ============================================================================
// Rule 2 / Rule 3 — fresh workspace restored + torn down
// ============================================================================

describe("Rules 2-3 — workspace restore/teardown", () => {
  it("restores files into a fresh dir", async () => {
    const ws = await Workspace.restore(
      files({ "input.txt": "hello\n", "sub/x.md": "deep" }),
    );
    expect(await readFile(join(ws.path, "input.txt"), "utf8")).toBe("hello\n");
    expect(await readFile(join(ws.path, "sub/x.md"), "utf8")).toBe("deep");
    await ws.teardown();
  });

  it("teardown removes the directory tree", async () => {
    const ws = await Workspace.restore(files({ "a.txt": "x" }));
    const p = ws.path;
    await ws.teardown();
    await expect(readFile(join(p, "a.txt"), "utf8")).rejects.toThrow();
  });
});

// ============================================================================
// Rule 4 — timeout yields a failed run (not a throw)
// ============================================================================

describe("Rule 4 — timeout is a failed run", () => {
  it("produces a report rather than throwing", async () => {
    const obs = new InMemoryObservabilityProvider();
    const t = task("slow", { kind: "empty" }, { kind: "always_fail" });
    t.timeout = 0; // clamps to 1ms inside the runner
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [t],
      challenge: [],
      canary: [],
    };
    const harness = new EvalHarnessBuilder(suite, configWith(obs, true, 5), obs)
      .candidate(
        "cand",
        configWith(new InMemoryObservabilityProvider(), true, 5),
      )
      .nRunsPerConfig(1)
      .build();
    const reports = await harness.run();
    expect(reports.length).toBe(1);
  });
});

// ============================================================================
// Rule 5 — tags are free-form
// ============================================================================

describe("Rule 5 — tags free-form", () => {
  it("carries arbitrary tags", () => {
    const t = task("t", { kind: "empty" }, { kind: "always_pass" });
    expect(t.tags).toEqual(["unit"]);
  });
});

// ============================================================================
// Rule 6 — suite_version required
// ============================================================================

describe("Rule 6 — suite_version required", () => {
  it("rejects a manifest without suite_version", () => {
    try {
      loadSuiteStr(`{ "regression": [], "challenge": [], "canary": [] }`);
      throw new Error("should have thrown");
    } catch (e) {
      expect((e as { kind?: string }).kind).toBe("missing_suite_version");
    }
  });

  it("accepts a manifest with suite_version", () => {
    const suite = loadSuiteStr(
      `{ "suite_version": 7, "regression": [], "challenge": [], "canary": [] }`,
    );
    expect(suite.suite_version).toBe(7);
  });
});

// ============================================================================
// Rules 7-8 — verification result shape + score clamp
// ============================================================================

describe("Rules 7-8 — verification result + clamp", () => {
  it("builds a result with signals", () => {
    const r = newVerificationResult(true, 0.5, "ok", { k: 1.0 });
    expect(r.passed).toBe(true);
    expect(r.score).toBe(0.5);
    expect(r.detail).toBe("ok");
    expect(r.signals.k).toBe(1.0);
  });

  it("rejects an out-of-range score and clamps via clamped()", () => {
    expect(() => newVerificationResult(true, 1.5, "x")).toThrow();
    expect(() => newVerificationResult(true, -0.1, "x")).toThrow();
    expect(clampedVerificationResult(true, 1.5, "x").score).toBe(1.0);
    expect(clampedVerificationResult(true, -0.1, "x").score).toBe(0.0);
  });
});

// ============================================================================
// Rule 9 — determinism flags
// ============================================================================

describe("Rule 9 — is_deterministic per verifier", () => {
  it("flags judge as non-deterministic", () => {
    expect(buildVerifier({ kind: "always_pass" }).isDeterministic()).toBe(true);
    expect(
      buildVerifier({
        kind: "test_suite",
        command: "true",
        args: [],
        timeout_secs: 1,
      }).isDeterministic(),
    ).toBe(true);
    expect(
      buildVerifier({
        kind: "llm_judge",
        rubric: "r",
        score_range: [0, 1],
      }).isDeterministic(),
    ).toBe(false);
  });
});

// ============================================================================
// Rule 10 — TestSuiteVerifier pass-rate
// ============================================================================

describe("Rule 10 — TestSuiteVerifier", () => {
  it("passes on zero exit", async () => {
    const dir = await mkdtemp(join(tmpdir(), "spore-eval-test-"));
    await writeFile(join(dir, "output.txt"), "HELLO\n");
    const t = task(
      "t",
      { kind: "empty" },
      {
        kind: "test_suite",
        command: "sh",
        args: ["-c", "grep -q HELLO output.txt"],
        timeout_secs: 10,
      },
    );
    const r = await taskVerifier(t).verify(t, runResultSuccess(), dir);
    expect(r.passed).toBe(true);
    expect(r.score).toBe(1.0);
    await rm(dir, { recursive: true, force: true });
  });

  it("fails on nonzero exit", async () => {
    const dir = await mkdtemp(join(tmpdir(), "spore-eval-test-"));
    const t = task(
      "t",
      { kind: "empty" },
      {
        kind: "test_suite",
        command: "sh",
        args: ["-c", "exit 1"],
        timeout_secs: 10,
      },
    );
    const r = await taskVerifier(t).verify(t, runResultSuccess(), dir);
    expect(r.passed).toBe(false);
    expect(r.score).toBe(0.0);
    await rm(dir, { recursive: true, force: true });
  });
});

// ============================================================================
// Rule 11 — CompositeVerifier
// ============================================================================

describe("Rule 11 — CompositeVerifier", () => {
  it("computes weighted mean, required AND, determinism AND", async () => {
    const composite = new CompositeVerifier([
      { verifier: new AlwaysPass(), weight: 1.0, required: true },
      { verifier: new AlwaysFail(), weight: 1.0, required: false },
    ]);
    const t = task("t", { kind: "empty" }, { kind: "always_pass" });
    const r = await composite.verify(t, runResultSuccess(), "/tmp");
    expect(Math.abs(r.score - 0.5)).toBeLessThan(1e-9);
    expect(r.passed).toBe(true);
    expect(composite.isDeterministic()).toBe(true);
  });

  it("a required failure fails overall", async () => {
    const composite = new CompositeVerifier([
      { verifier: new AlwaysPass(), weight: 1.0, required: true },
      { verifier: new AlwaysFail(), weight: 1.0, required: true },
    ]);
    const t = task("t", { kind: "empty" }, { kind: "always_pass" });
    const r = await composite.verify(t, runResultSuccess(), "/tmp");
    expect(r.passed).toBe(false);
  });

  it("resolves determinism AND from a spec", () => {
    const spec: VerifierSpec = {
      kind: "composite",
      children: [
        { spec: { kind: "always_pass" }, weight: 2.0, required: true },
        {
          spec: { kind: "llm_judge", rubric: "r", score_range: [0, 1] },
          weight: 1.0,
          required: false,
        },
      ],
    };
    expect(buildVerifier(spec).isDeterministic()).toBe(false);
  });
});

// ============================================================================
// Rule 12 — MetricEvaluatorVerifier normalizes
// ============================================================================

describe("Rule 12 — MetricEvaluatorVerifier", () => {
  it("normalizes a metric value to a score", async () => {
    const evaluator = {
      async evaluate() {
        return {
          kind: "ok" as const,
          result: { value: 7.5, raw_output: "", duration: 0, metadata: {} },
        };
      },
      direction() {
        return "maximize" as const;
      },
      description() {
        return "fixed";
      },
    };
    const v = MetricEvaluatorVerifier.withRange(evaluator, 0.0, 10.0);
    expect(v.isDeterministic()).toBe(true);
    const t = task("t", { kind: "empty" }, { kind: "always_pass" });
    const dir = await mkdtemp(join(tmpdir(), "spore-eval-test-"));
    const r = await v.verify(t, runResultSuccess(), dir);
    expect(Math.abs(r.score - 0.75)).toBeLessThan(1e-9);
    await rm(dir, { recursive: true, force: true });
  });
});

// ============================================================================
// Rule 13 — LlmJudgeVerifier non-deterministic + pluggable judge
// ============================================================================

describe("Rule 13 — LlmJudgeVerifier", () => {
  it("parses the judge score and normalizes by range", async () => {
    const judge = new MockModelInterface({
      name: "fake",
      model_id: "judge",
      context_window: 8000,
    });
    judge.pushResponse({
      content: [{ type: "text", text: "score: 8" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 0, output_tokens: 0 },
    });
    const v = new LlmJudgeVerifier(judge, "rate", [0, 10]);
    expect(v.isDeterministic()).toBe(false);
    const t = task("t", { kind: "empty" }, { kind: "always_pass" });
    const r = await v.verify(t, runResultSuccess(), "/tmp");
    expect(Math.abs(r.score - 0.8)).toBeLessThan(1e-9);
  });
});

// ============================================================================
// Rules 14-16 — n runs, build harness from config, read obs metrics
// ============================================================================

describe("Rules 14-16 — runs per config + metrics from obs", () => {
  it("runs n times and reads turns from observability", async () => {
    const baseObs = new InMemoryObservabilityProvider();
    const candObs = new InMemoryObservabilityProvider();
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [task("t1", { kind: "empty" }, { kind: "always_pass" })],
      challenge: [],
      canary: [],
    };
    const n = 3;
    const harness = new EvalHarnessBuilder(
      suite,
      configWith(baseObs, true, n),
      baseObs,
    )
      .candidate("cand", configWith(candObs, true, n))
      .nRunsPerConfig(n)
      .metrics([
        { kind: "task_success_rate" },
        { kind: "mean_turns_to_completion" },
      ])
      .build();

    const reports = await harness.run();
    expect(reports.length).toBe(1);
    const success = reports[0]!.metrics.find(
      (m) => m.metricName === "task_success_rate",
    )!;
    expect(success.baseline.n).toBe(n); // Rule 14
    const turns = reports[0]!.metrics.find(
      (m) => m.metricName === "mean_turns_to_completion",
    )!;
    expect(turns.baseline.mean).toBeGreaterThanOrEqual(1.0); // Rule 16
  });
});

// ============================================================================
// Rule 17 — EvalMetric mapping (name + direction)
// ============================================================================

describe("Rule 17 — metric names + directions", () => {
  it("maps names and directions", () => {
    expect(metricDirection({ kind: "task_success_rate" })).toBe("maximize");
    expect(metricDirection({ kind: "mean_cost_usd" })).toBe("minimize");
    expect(metricDirection({ kind: "mean_turns_to_completion" })).toBe(
      "minimize",
    );
    expect(metricDirection({ kind: "cache_hit_rate", block: "sys" })).toBe(
      "maximize",
    );
    expect(metricName({ kind: "cache_hit_rate", block: "sys" })).toBe(
      "cache_hit_rate[sys]",
    );
    expect(metricDirection({ kind: "verification_score" })).toBe("maximize");
  });
});

// ============================================================================
// Rule 18 — WaitingForHuman: resource metric still computes
// ============================================================================

describe("Rule 18 — resource metric computes for waiting", () => {
  it("returns the turn count regardless of verifier outcome", () => {
    const session = {
      session_id: SessionId.of("s"),
      task_id: {
        asString: () => "t",
        value: "t",
        toString: () => "t",
        equals: () => false,
        toJSON: () => "t",
      } as never,
      total_turns: 2,
      total_input_tokens: 0,
      total_output_tokens: 0,
      total_cost_usd: 0,
      total_duration_ms: 0,
      tool_calls: 0,
      sensor_fires: 0,
      sensor_halts: 0,
      compactions: 0,
      outcome: { kind: "partial" as const },
      guides_used: [],
      patch_count: 0,
      patch_rate: 0,
      patches_by_tool: {},
      compaction_verification_failures: 0,
    };
    const v = sampleFor({ kind: "mean_turns_to_completion" }, session, [], {
      verifierPassed: false,
      verifierScore: 0,
    });
    expect(v).toBe(2.0);
  });
});

// ============================================================================
// Rules 19-22 — stats aggregation, Welch, direction
// ============================================================================

describe("Rules 19-22 — comparison primitives", () => {
  it("aggregates and classifies direction", () => {
    const w = welchTTest([0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]);
    expect(w.pValue).toBeLessThan(0.05);
    expect(classifyDirection(0.3, "maximize", 1e-9)).toBe("better");
  });
});

// ============================================================================
// Rule 21 — bootstrap CI for non-deterministic verifiers
// ============================================================================

describe("Rule 21 — bootstrap CI", () => {
  it("brackets the resample means", () => {
    const ci = bootstrapCi(
      [0.5, 0.6, 0.4, 0.55],
      1000,
      0.95,
      DEFAULT_BOOTSTRAP_SEED,
    )!;
    expect(ci.lower).toBeLessThanOrEqual(ci.upper);
  });

  it("a non-deterministic verifier yields a CI on the success-rate comparison", async () => {
    const baseObs = new InMemoryObservabilityProvider();
    const candObs = new InMemoryObservabilityProvider();
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [
        task(
          "t",
          { kind: "empty" },
          { kind: "llm_judge", rubric: "r", score_range: [0, 1] },
        ),
      ],
      challenge: [],
      canary: [],
    };
    const n = 4;
    const harness = new EvalHarnessBuilder(
      suite,
      configWith(baseObs, true, n),
      baseObs,
    )
      .candidate("cand", configWith(candObs, true, n))
      .nRunsPerConfig(n)
      .metrics([{ kind: "task_success_rate" }])
      .build();
    const reports = await harness.run();
    expect(reports[0]!.metrics[0]!.ci).toBeDefined();
  });
});

// ============================================================================
// Rules 23-24 — recommendation + recommended_n
// ============================================================================

describe("Rules 23-24 — recommendation paths", () => {
  const stats = (mean: number, n: number): MetricStats => ({
    mean,
    stddev: 0.05,
    p50: mean,
    p95: mean,
    n,
  });

  it("adopts when the primary improves significantly", () => {
    const rec = deriveRecommendation(
      "c",
      [
        {
          metricName: "task_success_rate",
          baseline: stats(0.5, 5),
          candidate: stats(0.95, 5),
          delta: 0.45,
          pValue: 0.001,
          direction: "better",
        },
      ],
      { kind: "task_success_rate" },
    );
    expect(rec.kind).toBe("adopt");
  });

  it("rejects when the primary regresses significantly", () => {
    const rec = deriveRecommendation(
      "c",
      [
        {
          metricName: "task_success_rate",
          baseline: stats(0.9, 5),
          candidate: stats(0.4, 5),
          delta: -0.5,
          pValue: 0.001,
          direction: "worse",
        },
      ],
      { kind: "task_success_rate" },
    );
    expect(rec.kind).toBe("reject");
  });

  it("needs more runs when inconclusive", () => {
    const rec = deriveRecommendation(
      "c",
      [
        {
          metricName: "task_success_rate",
          baseline: stats(0.5, 3),
          candidate: stats(0.55, 3),
          delta: 0.05,
          pValue: 0.5,
          direction: "better",
        },
      ],
      { kind: "task_success_rate" },
    );
    expect(rec.kind).toBe("needs_more_runs");
    if (rec.kind === "needs_more_runs") {
      expect(rec.currentN).toBe(3);
      expect(rec.recommendedN).toBeGreaterThan(3);
    }
  });
});

// ============================================================================
// Rule 25 — trace_links collected for failures
// ============================================================================

describe("Rule 25 — trace links", () => {
  it("collects links for failing runs", async () => {
    const baseObs = new InMemoryObservabilityProvider();
    const candObs = new InMemoryObservabilityProvider();
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [task("t1", { kind: "empty" }, { kind: "always_fail" })],
      challenge: [],
      canary: [],
    };
    const harness = new EvalHarnessBuilder(
      suite,
      configWith(baseObs, true, 2),
      baseObs,
    )
      .candidate("cand", configWith(candObs, true, 2))
      .nRunsPerConfig(2)
      .build();
    const reports = await harness.run();
    expect(reports[0]!.traceLinks.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Rule 29 — fixtures are the cross-language oracle (replay)
// ============================================================================

describe("Rule 29 — core_suite.json fixture", () => {
  it("loads and resolves verifiers", async () => {
    const suite = await loadSuitePath(join(FIXTURES, "core_suite.json"));
    expect(suite.suite_version).toBe(1);
    expect(suite.regression.length).toBe(2);
    expect(suite.challenge.length).toBe(2);
    expect(suite.canary.length).toBe(1);
    for (const [, t] of allTasks(suite)) {
      expect(taskVerifier(t)).toBeDefined();
    }
    const s1 = suite.regression[0]!;
    expect(s1.id).toBe("regression_s1_uppercase");
    expect(s1.workspace_snapshot.kind).toBe("files");
    if (s1.workspace_snapshot.kind === "files") {
      expect("input.txt" in s1.workspace_snapshot.files).toBe(true);
    }
  });
});

// ============================================================================
// Rule 30 — TraceAnalyzer is interface-only
// ============================================================================

describe("Rule 30 — TraceAnalyzer interface only", () => {
  it("a user can implement it; no built-in ships", async () => {
    class UserAnalyzer implements TraceAnalyzer {
      async analyze(_traces: Span[]): Promise<HarnessConfigDiff[]> {
        return [{ description: "proposed" }];
      }
    }
    const a: TraceAnalyzer = new UserAnalyzer();
    expect((await a.analyze([]))[0]!.description).toBe("proposed");
  });
});

// ============================================================================
// Rule 31 — manual promotion
// ============================================================================

describe("Rule 31 — manual promotion", () => {
  it("moves challenge->regression and bumps suite_version", () => {
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [task("r1", { kind: "empty" }, { kind: "always_pass" })],
      challenge: [task("c1", { kind: "empty" }, { kind: "always_pass" })],
      canary: [],
    };
    promoteChallengeTask(suite, "c1");
    expect(suite.suite_version).toBe(2);
    expect(suite.regression.length).toBe(2);
    expect(suite.challenge.length).toBe(0);
    expect(() => promoteChallengeTask(suite, "nope")).toThrow();
  });

  it("round-trips JSON through the fixture", async () => {
    const suite = await loadSuitePath(join(FIXTURES, "core_suite.json"));
    const beforeReg = suite.regression.length;
    promoteChallengeTask(suite, "challenge_s5_shell_pipeline");
    expect(suite.suite_version).toBe(2);
    expect(suite.regression.length).toBe(beforeReg + 1);
    const json = suiteToJson(suite);
    const reparsed = loadSuiteStr(json);
    expect(reparsed.suite_version).toBe(2);
  });
});

// ============================================================================
// Rule 32 — no Inspect AI / Langfuse / stats-library dependency
// ============================================================================

describe("Rule 32 — no banned dependencies", () => {
  it("package.json names no Inspect AI / Langfuse / stats library", async () => {
    const pkg = (
      await readFile(join(HERE, "../package.json"), "utf8")
    ).toLowerCase();
    expect(pkg).not.toContain("inspect-ai");
    expect(pkg).not.toContain("langfuse");
    expect(pkg).not.toContain("simple-statistics");
    expect(pkg).not.toContain("jstat");
  });
});

// ============================================================================
// E2E hermetic regression test — baseline vs a deliberately-worse candidate
// ============================================================================

describe("E2E — regression flagged with sane p-value + recommendation", () => {
  it("flags the worse candidate", async () => {
    // Shared provider: both configs and the EvalHarness read the SAME provider.
    const obs = new InMemoryObservabilityProvider();
    const suite: TaskSuite = {
      suite_version: 1,
      regression: [
        task(
          "e2e",
          { kind: "empty" },
          {
            kind: "metric_evaluator",
            descriptor: "run-success",
            direction: "maximize",
            min: 0.0,
            max: 1.0,
            threshold: null,
          },
        ),
      ],
      challenge: [],
      canary: [],
    };

    const n = 6;
    const harness = new EvalHarnessBuilder(suite, configWith(obs, true, n), obs)
      .candidate("smaller_window", configWith(obs, false, n))
      .nRunsPerConfig(n)
      .metrics([{ kind: "task_success_rate" }])
      .primaryMetric({ kind: "task_success_rate" })
      .build();

    const reports: ComparisonReport[] = await harness.run();
    expect(reports.length).toBe(1);
    const report = reports[0]!;
    const success = report.metrics.find(
      (m) => m.metricName === "task_success_rate",
    )!;
    expect(success.baseline.mean).toBeGreaterThan(success.candidate.mean);
    expect(success.direction).toBe("worse");
    expect(success.pValue).toBeGreaterThanOrEqual(0);
    expect(success.pValue).toBeLessThanOrEqual(1);
    expect(success.pValue).toBeLessThan(0.05);
    expect(["reject", "needs_more_runs"]).toContain(report.recommendation.kind);
    expect(report.traceLinks.length).toBeGreaterThan(0);
  });
});
