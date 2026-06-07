/**
 * Unit tests for the HillClimbing loop strategy (issue #60).
 *
 * Mirrors the inline `run_hill_climbing` tests in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same outcomes, same
 * byte-identical TSV.
 *
 * Rules covered:
 *   - baseline-first (iteration 0, NO agent turn, status kept, sets current_best)
 *   - keep-on-improve / discard-on-regress
 *   - strict min_delta boundary (exactly min_delta is NOT progress)
 *   - revert on / off (git reset --hard HEAD through the sandbox seam)
 *   - stagnation halt / stagnation reset on improvement
 *   - crash / timeout counts as a non-improvement (empty metric_value)
 *   - misconfiguration (no evaluator) → hill_climbing_misconfigured (no throw)
 *   - baseline-error → hill_climbing_misconfigured (D7), NOT a stagnation increment
 *   - budget gate (turn cap)
 *   - exact-TSV byte content
 *   - direction is the payload direction, recorded in the TSV
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type CommandOutput,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type OptimizationDirection,
  type SandboxProvider,
  type SandboxViolation,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import type { MetricError, MetricEvaluator, MetricOutcome } from "../src/metric/types.js";
import type { SessionStateSnapshot } from "../src/termination/types.js";
import { observability as obsNs } from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const { InMemoryObservabilityProvider } = obsNs;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

/** An agent that always claims done — each iteration's "propose a change". */
class AlwaysDoneAgent implements Agent {
  ran = 0;
  constructor(private readonly agentId: AgentId) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(_ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.ran += 1;
    return fr("done");
  }
}

/**
 * A {@link MetricEvaluator} test double returning a pre-programmed sequence of
 * metric values. Element 0 is the baseline; a `null` element is a crash. All
 * scripted results carry a ZERO duration so the TSV is byte-deterministic.
 */
class ScriptedMetricEvaluator implements MetricEvaluator {
  calls = 0;
  constructor(
    private readonly sequence: (number | null)[],
    private readonly dir: OptimizationDirection,
    private readonly desc = "scripted metric",
  ) {}
  async evaluate(
    _sandbox: SandboxProvider,
    _state: SessionStateSnapshot,
    _signal?: AbortSignal,
  ): Promise<MetricOutcome> {
    const i = this.calls;
    this.calls += 1;
    const v = this.sequence[i];
    if (v == null) {
      const error: MetricError = { kind: "crashed", log: "scripted crash" };
      return { kind: "err", error };
    }
    return { kind: "ok", result: { value: v, raw_output: "", duration: 0, metadata: {} } };
  }
  direction(): OptimizationDirection {
    return this.dir;
  }
  description(): string {
    return this.desc;
  }
}

/** A {@link MetricEvaluator} that always errors — for the baseline-error path. */
class AlwaysErrorEvaluator implements MetricEvaluator {
  calls = 0;
  constructor(private readonly dir: OptimizationDirection = "maximize") {}
  async evaluate(): Promise<MetricOutcome> {
    this.calls += 1;
    return { kind: "err", error: { kind: "crashed", log: "boom" } };
  }
  direction(): OptimizationDirection {
    return this.dir;
  }
  description(): string {
    return "scripted metric";
  }
}

/**
 * Sandbox that records every {@link executeCommand} call so the revert
 * (`git reset --hard HEAD`) can be counted exactly.
 */
class RecordingSandbox implements SandboxProvider {
  readonly commands: { command: string; args: readonly string[] }[] = [];
  constructor(private readonly root: string = "") {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async executeCommand(
    command: string,
    args: readonly string[],
    _cwd?: string | null,
    _timeoutMs?: number | null,
    _signal?: AbortSignal,
  ): Promise<CommandOutput | SandboxViolation> {
    this.commands.push({ command, args });
    return { stdout: "", stderr: "", exit_code: 0, timed_out: false, truncated: false };
  }
  workspaceRoot(): string {
    return this.root;
  }
  revertCount(): number {
    return this.commands.filter(
      (c) => c.command === "git" && c.args.join(" ") === "reset --hard HEAD",
    ).length;
  }
}

function hcStrategy(opts: {
  direction: OptimizationDirection;
  max_stagnation?: number | null;
  revert_on_no_improvement?: boolean;
  min_improvement_delta?: number | null;
}): LoopStrategy {
  return {
    kind: "hill_climbing",
    inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
    direction: opts.direction,
    // `max_stagnation`/`min_improvement_delta` are required numbers (#119). The
    // old `null` ("unbounded" / "no delta") inputs map to behavior-preserving
    // concretes: MAX_SAFE_INTEGER never trips the stagnation halt, and `0`
    // matches `shouldKeep`'s null-as-0.0 default.
    max_stagnation: opts.max_stagnation ?? Number.MAX_SAFE_INTEGER,
    revert_on_no_improvement: opts.revert_on_no_improvement ?? false,
    min_improvement_delta: opts.min_improvement_delta ?? 0,
    evaluator: "",
  };
}

const SID = SessionId.of("hc-session");

function config(agent: Agent, overrides: Partial<HarnessConfig> = {}): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    ...overrides,
  };
}

function task(strategy: LoopStrategy, maxTurns = 100) {
  return newTask("optimize the thing", SID, strategy, { max_turns: maxTurns });
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("HillClimbing loop strategy (issue #60)", () => {
  it("D6: no evaluator → hill_climbing_misconfigured (typed halt, not a throw); agent never runs", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const h = new StandardHarness(config(agent)); // no metricEvaluator
    const r = await h.run({ task: task(hcStrategy({ direction: "maximize" })) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("hill_climbing_misconfigured");
      if (r.reason.kind === "hill_climbing_misconfigured") {
        expect(r.reason.reason).toContain("metricEvaluator");
      }
    }
    expect(agent.ran).toBe(0);
  });

  it("D7: baseline evaluation errors → hill_climbing_misconfigured, NOT a stagnation increment", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new AlwaysErrorEvaluator("maximize");
    const h = new StandardHarness(
      config(agent, { metricEvaluator: evaluator, sandbox: new RecordingSandbox() }),
    );
    const r = await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 1 })),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("hill_climbing_misconfigured");
    }
    // Only the baseline call happened — no agent turn, no second evaluation.
    expect(evaluator.calls).toBe(1);
    expect(agent.ran).toBe(0);
  });

  it("D5: baseline is iteration 0 with NO agent turn and status kept; keep-on-improve then stagnation halt", async () => {
    // maximize: baseline 1.0 (kept), iter1 2.0 (kept, improve), iter2 1.5
    // (discarded, regress vs 2.0). max_stagnation 1 halts; best stays 2.0.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([1.0, 2.0, 1.5], "maximize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 1 })),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("stagnation_limit_reached");
      if (r.reason.kind === "stagnation_limit_reached") {
        expect(r.reason.best_metric).toBe(2.0);
        expect(r.reason.iterations).toBe(1);
      }
    }
    // 2 agent turns (iterations 1 and 2); baseline ran none.
    expect(agent.ran).toBe(2);
  });

  it("stagnation halt after three consecutive non-improvements; best preserved", async () => {
    // maximize: baseline 5.0 (kept), then three regresses to 4.0. max_stagnation 3.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([5.0, 4.0, 4.0, 4.0], "maximize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 3 })),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "stagnation_limit_reached") {
      expect(r.reason.best_metric).toBe(5.0);
      expect(r.reason.iterations).toBe(3);
    }
    expect(agent.ran).toBe(3);
  });

  it("stagnation resets on improvement → run ends on the turn budget, not the stagnation cap", async () => {
    // maximize: baseline 1.0 (kept), 0.5 (discard), 0.5 (discard, stagnation=2),
    // 2.0 (kept, IMPROVE → reset to 0), 1.0 (discard, stagnation=1). max_turns 4,
    // max_stagnation 3 → ends on the turn budget.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([1.0, 0.5, 0.5, 2.0, 1.0], "maximize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 3 }), 4),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("budget_exceeded");
      if (r.reason.kind === "budget_exceeded") {
        expect(r.reason.limit_type).toBe("turns");
      }
    }
  });

  it("crash counts as a non-improvement (empty metric_value, status crashed); stagnation halts", async () => {
    // maximize: baseline 1.0 (kept), iter1 crash (null) → non-improvement.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([1.0, null], "maximize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 1 })),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "stagnation_limit_reached") {
      expect(r.reason.best_metric).toBe(1.0);
    }
  });

  it("revert ON: a regress (minimize) issues exactly one git reset --hard HEAD", async () => {
    // minimize: baseline 2.0 (kept), iter1 3.0 (worse → discard). revert true.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([2.0, 3.0], "minimize");
    const sandbox = new RecordingSandbox();
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator, sandbox }));
    await h.run({
      task: task(
        hcStrategy({ direction: "minimize", max_stagnation: 1, revert_on_no_improvement: true }),
      ),
    });
    expect(sandbox.revertCount()).toBe(1);
  });

  it("revert OFF: a regress (minimize) issues NO git reset", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([2.0, 3.0], "minimize");
    const sandbox = new RecordingSandbox();
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator, sandbox }));
    await h.run({
      task: task(
        hcStrategy({ direction: "minimize", max_stagnation: 1, revert_on_no_improvement: false }),
      ),
    });
    expect(sandbox.revertCount()).toBe(0);
  });

  it("strict min_delta boundary: an improvement of EXACTLY min_delta is NOT progress → discarded", async () => {
    // minimize: baseline 2.0 (kept), iter1 1.5 with min_improvement_delta 0.5.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([2.0, 1.5], "minimize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({
      task: task(
        hcStrategy({ direction: "minimize", max_stagnation: 1, min_improvement_delta: 0.5 }),
      ),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "stagnation_limit_reached") {
      expect(r.reason.best_metric).toBe(2.0); // 1.5 was discarded
    }
  });

  it("budget gate: max_turns 0 halts before any iteration with budget_exceeded(turns)", async () => {
    // Baseline (no agent turn) still measured; iteration 1 is gated out by the
    // turn cap of 0 → clean budget halt.
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([1.0, 2.0], "maximize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({ task: task(hcStrategy({ direction: "maximize" }), 0) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("budget_exceeded");
    }
    // Only the baseline was measured; no agent turn ran.
    expect(agent.ran).toBe(0);
    expect(evaluator.calls).toBe(1);
  });

  it("exact TSV byte content (D2/D3): header + one row per iteration, 6-decimal floats", () => {
    const tsv = StandardHarness.renderHillClimbingTsv([
      {
        iteration: 0,
        commit_hash: null,
        metric_value: 1.0,
        direction: "maximize",
        status: "kept",
        duration: 0,
        description: "scripted metric",
        metadata: {},
      },
      {
        iteration: 1,
        commit_hash: null,
        metric_value: 2.0,
        direction: "maximize",
        status: "kept",
        duration: 0,
        description: "scripted metric",
        metadata: {},
      },
      {
        iteration: 2,
        commit_hash: null,
        metric_value: 1.5,
        direction: "maximize",
        status: "discarded",
        duration: 0,
        description: "scripted metric",
        metadata: {},
      },
    ]);
    expect(tsv).toBe(
      "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
        "0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
        "1\t\t2.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
        "2\t\t1.500000\tmaximize\tdiscarded\t0.000000\tscripted metric\n",
    );
  });

  it("crashed/timeout rows render an EMPTY metric_value (D3)", () => {
    const tsv = StandardHarness.renderHillClimbingTsv([
      {
        iteration: 0,
        commit_hash: null,
        metric_value: 1.0,
        direction: "maximize",
        status: "kept",
        duration: 0,
        description: "scripted metric",
        metadata: {},
      },
      {
        iteration: 1,
        commit_hash: null,
        metric_value: NaN,
        direction: "maximize",
        status: "crashed",
        duration: 0,
        description: "scripted metric",
        metadata: {},
      },
    ]);
    expect(tsv).toBe(
      "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
        "0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
        "1\t\t\tmaximize\tcrashed\t0.000000\tscripted metric\n",
    );
  });

  it("emits a per-iteration observability span with metric value + delta", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([1.0, 2.0, 1.5], "maximize");
    const obs = new InMemoryObservabilityProvider();
    const h = new StandardHarness(
      config(agent, { metricEvaluator: evaluator, observability: obs }),
    );
    await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 1 })),
    });
    const warns = obs.warnSpans(SID).filter((w) => w.event.warn === "hill_climbing_iteration");
    expect(warns.length).toBe(3); // baseline + 2 iterations
    const baseline = warns[0]!.event;
    if (baseline.warn === "hill_climbing_iteration") {
      expect(baseline.iteration).toBe(0);
      expect(baseline.metric_value).toBe(1.0);
      expect(baseline.delta).toBeNull();
      expect(baseline.status).toBe("kept");
    }
    const improve = warns[1]!.event;
    if (improve.warn === "hill_climbing_iteration") {
      expect(improve.status).toBe("kept");
      expect(improve.delta).toBe(1.0); // 2.0 - 1.0 maximize
    }
    const regress = warns[2]!.event;
    if (regress.warn === "hill_climbing_iteration") {
      expect(regress.status).toBe("discarded");
    }
  });

  it("updated StrategyNotYetImplemented: hill_climbing no longer returns strategy_not_yet_implemented", async () => {
    const agent = new AlwaysDoneAgent(AgentId.of("a"));
    const evaluator = new ScriptedMetricEvaluator([1.0, 0.5], "maximize");
    const h = new StandardHarness(config(agent, { metricEvaluator: evaluator }));
    const r = await h.run({
      task: task(hcStrategy({ direction: "maximize", max_stagnation: 1 })),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).not.toBe("strategy_not_yet_implemented");
    }
  });
});
