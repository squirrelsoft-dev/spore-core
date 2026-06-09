/**
 * Composable Execution Part A (spore-core issue #119).
 *
 * Recursive {@link LoopStrategy} config newtypes + {@link StrategyRef} +
 * {@link RunStrategy} seam. These tests assert:
 *   - every variant round-trips through serialize/parse,
 *   - the canonical cordyceps tree round-trips and serializes byte-identically,
 *   - the `react` tag (NOT `re_act`) and flat config layout,
 *   - StrategyRef BuiltIn/Custom round-trip + adjacent tagging,
 *   - handle types (AgentRef/ToolsetRef/SchemaRef) round-trip as bare strings,
 *   - the stub `run` dispatch returns a `complete` placeholder (no throw),
 *   - PausedState/ChildPausedState round-trip with a cordyceps task.loop_strategy,
 *   - byte-identity replay against the shared `fixtures/strategy/` ground truth.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  ExecutionRegistry,
  SessionId,
  asRunStrategy,
  loopStrategyFromJson,
  loopStrategyMaxSteps,
  loopStrategyToJson,
  newExecutionContext,
  newTask,
  runStrategy,
  strategyRefFromJson,
  strategyRefMaxSteps,
  strategyRefToJson,
  type BudgetExhaustedBehavior,
  type BudgetPolicy,
  type LoopStrategy,
  type StrategyRef,
} from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/strategy");

function fixture(name: string): string {
  return readFileSync(resolve(fixtureDir, name), "utf-8");
}

/**
 * Minify a pretty-printed JSON string WITHOUT reordering keys (mirrors the Rust
 * `minify_preserving_order` helper). The fixtures are pretty-printed in
 * declaration order; the minified form is the byte-identity target.
 */
function minifyPreservingOrder(s: string): string {
  let out = "";
  let inString = false;
  let escaped = false;
  for (const ch of s) {
    if (inString) {
      out += ch;
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
    } else if (ch === '"') {
      inString = true;
      out += ch;
    } else if (!/\s/.test(ch)) {
      out += ch;
    }
  }
  return out;
}

/** The canonical cordyceps tree `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`. */
// #129: every node carries an explicit `behavior: escalate` (the parse injects
// the default, so the in-memory tree must include it for deep-equality).
const ESCALATE: BudgetExhaustedBehavior = { kind: "escalate" };

function cordycepsTree(): LoopStrategy {
  return {
    kind: "ralph",
    inner: {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 4 },
        behavior: ESCALATE,
        agent: "planner",
        toolset: "plan-tools",
        output: "plan-schema",
      },
      execute: {
        kind: "self_verifying",
        inner: {
          kind: "react",
          budget: { kind: "per_loop", value: 12 },
          behavior: ESCALATE,
          agent: "executor",
          toolset: "exec-tools",
          output: "worker-schema",
        },
        evaluator: "exec-evaluator",
        behavior: ESCALATE,
      },
      behavior: ESCALATE,
    },
    agent: "ralph-agent",
    behavior: ESCALATE,
  };
}

describe("LoopStrategy", () => {
  const variants: LoopStrategy[] = [
    {
      kind: "react",
      budget: { kind: "per_loop", value: 8 },
      behavior: ESCALATE,
      agent: "a",
      toolset: "t",
      output: "out",
    },
    {
      kind: "react",
      budget: { kind: "per_loop", value: 3 },
      behavior: ESCALATE,
      agent: "",
      toolset: "",
    },
    {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      execute: {
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      behavior: ESCALATE,
    },
    {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      execute: {
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      plan_model: { provider: "anthropic", model_id: "claude" },
      behavior: ESCALATE,
    },
    {
      kind: "self_verifying",
      inner: {
        kind: "react",
        budget: { kind: "per_loop", value: 2 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      evaluator: "ev",
      behavior: ESCALATE,
    },
    {
      kind: "ralph",
      inner: {
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      agent: "r",
      behavior: ESCALATE,
    },
    {
      kind: "hill_climbing",
      inner: {
        kind: "react",
        budget: { kind: "per_loop", value: 5 },
        behavior: ESCALATE,
        agent: "",
        toolset: "",
      },
      direction: "maximize",
      max_stagnation: 3,
      revert_on_no_improvement: true,
      min_improvement_delta: 0.25,
      evaluator: "metric",
      behavior: ESCALATE,
    },
  ];

  it.each(variants)("round-trips %j", (strategy) => {
    const json = JSON.stringify(loopStrategyToJson(strategy));
    const back = loopStrategyFromJson(JSON.parse(json));
    expect(back).toEqual(strategy);
  });

  it("uses the `react` tag (NOT `re_act`) and flattens config next to it", () => {
    const json = JSON.stringify(
      loopStrategyToJson({
        kind: "react",
        budget: { kind: "per_loop", value: 8 },
        agent: "",
        toolset: "",
      }),
    );
    expect(json).toContain('"kind":"react"');
    expect(json).not.toContain("re_act");
    // #129: `behavior` is always serialized, immediately after `budget`.
    expect(json).toBe(
      '{"kind":"react","budget":{"kind":"per_loop","value":8},' +
        '"behavior":{"kind":"escalate"},"agent":"","toolset":""}',
    );
  });

  it("omits `output` when absent (never null)", () => {
    const json = JSON.stringify(
      loopStrategyToJson({
        kind: "react",
        budget: { kind: "per_loop", value: 1 },
        agent: "",
        toolset: "",
      }),
    );
    expect(json).not.toContain("output");
  });

  it("omits `plan_model` when absent (never null)", () => {
    const json = JSON.stringify(
      loopStrategyToJson({
        kind: "plan_execute",
        plan: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
        execute: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
      }),
    );
    expect(json).not.toContain("plan_model");
  });

  it("round-trips handle types as bare strings", () => {
    const json = JSON.stringify(loopStrategyToJson(cordycepsTree()));
    expect(json).toContain('"agent":"planner"');
    expect(json).toContain('"toolset":"plan-tools"');
    expect(json).toContain('"evaluator":"exec-evaluator"');
    expect(json).toContain('"agent":"ralph-agent"');
  });

  it("round-trips the cordyceps tree", () => {
    const tree = cordycepsTree();
    const json = JSON.stringify(loopStrategyToJson(tree));
    expect(loopStrategyFromJson(JSON.parse(json))).toEqual(tree);
  });

  it("serializes the cordyceps tree to the exact compact form", () => {
    expect(JSON.stringify(loopStrategyToJson(cordycepsTree()))).toBe(
      '{"kind":"ralph","inner":{"kind":"plan_execute","plan":{"kind":"react",' +
        '"budget":{"kind":"per_loop","value":4},"behavior":{"kind":"escalate"},' +
        '"agent":"planner","toolset":"plan-tools","output":"plan-schema"},' +
        '"execute":{"kind":"self_verifying","inner":{"kind":"react","budget":' +
        '{"kind":"per_loop","value":12},"behavior":{"kind":"escalate"},' +
        '"agent":"executor","toolset":"exec-tools","output":"worker-schema"},' +
        '"evaluator":"exec-evaluator","behavior":{"kind":"escalate"}},' +
        '"behavior":{"kind":"escalate"}},"agent":"ralph-agent",' +
        '"behavior":{"kind":"escalate"}}',
    );
  });

  it("rejects an unknown kind", () => {
    expect(() => loopStrategyFromJson({ kind: "re_act", max_iterations: 8 })).toThrow();
  });
});

describe("StrategyRef", () => {
  const refs: StrategyRef[] = [
    { kind: "built_in", value: cordycepsTree() },
    { kind: "custom", value: "my-harness::DoubleVerify" },
  ];

  it.each(refs)("round-trips %j", (ref) => {
    const json = JSON.stringify(strategyRefToJson(ref));
    expect(strategyRefFromJson(JSON.parse(json))).toEqual(ref);
  });

  it("adjacently tags custom on kind/value", () => {
    expect(
      JSON.stringify(strategyRefToJson({ kind: "custom", value: "my-harness::DoubleVerify" })),
    );
    expect(
      JSON.stringify(strategyRefToJson({ kind: "custom", value: "my-harness::DoubleVerify" })),
    ).toBe('{"kind":"custom","value":"my-harness::DoubleVerify"}');
  });

  it("adjacently tags built_in with the nested LoopStrategy under value", () => {
    const json = JSON.stringify(strategyRefToJson({ kind: "built_in", value: cordycepsTree() }));
    expect(json.startsWith('{"kind":"built_in","value":{"kind":"ralph"')).toBe(true);
  });
});

describe("RunStrategy dispatch without a wired executor (#124)", () => {
  const strategies: LoopStrategy[] = [
    { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
    {
      kind: "plan_execute",
      plan: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
      execute: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
    },
    {
      kind: "self_verifying",
      inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
      evaluator: "e",
    },
    {
      kind: "ralph",
      inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
      agent: "r",
    },
    {
      kind: "hill_climbing",
      inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
      direction: "minimize",
      max_stagnation: 1,
      revert_on_no_improvement: false,
      min_improvement_delta: 0,
      evaluator: "m",
    },
  ];

  // #124: the per-variant bodies are real now. Without a wired StrategyExecutor
  // (the scaffold-only context) every body returns a TYPED `failed` outcome —
  // never a throw (CONVENTIONS), never the old `complete` stub.
  it.each(strategies)("returns a typed failed (no throw) for %j", async (strategy) => {
    const cx = newExecutionContext(ExecutionRegistry.empty());
    // scratch.task must be set before running a strategy (the harness entry does
    // this; here we set it directly so the body reaches the executor check).
    cx.scratch.task = newTask("t", SessionId.generate(), strategy);
    const outcome = await runStrategy(strategy, cx);
    expect(outcome.kind).toBe("failed");
  });

  it("asRunStrategy delegates to the single dispatch", async () => {
    const cx = newExecutionContext(ExecutionRegistry.empty());
    cx.scratch.task = newTask("t", SessionId.generate(), cordycepsTree());
    const rs = asRunStrategy(cordycepsTree());
    const outcome = await rs.run(cx);
    expect(outcome.kind).toBe("failed");
  });
});

describe("PausedState / ChildPausedState round-trip with cordyceps strategy", () => {
  it("paused_state fixture parses, deep-equals, and the loop_strategy is byte-identical", () => {
    const raw = fixture("paused_state.json");
    const parsed = JSON.parse(raw) as { task: { loop_strategy: unknown } };
    const strategy = loopStrategyFromJson(parsed.task.loop_strategy);
    expect(strategy).toEqual(cordycepsTree());
    // The loop_strategy subtree (float-free) re-serializes byte-identically.
    const minified = minifyPreservingOrder(JSON.stringify(parsed.task.loop_strategy, null, 2));
    expect(JSON.stringify(loopStrategyToJson(strategy))).toBe(minified);
  });

  it("child_paused_state fixture parses, deep-equals, and the loop_strategy is byte-identical", () => {
    const raw = fixture("child_paused_state.json");
    const parsed = JSON.parse(raw) as { task: { loop_strategy: unknown } };
    const strategy = loopStrategyFromJson(parsed.task.loop_strategy);
    expect(strategy).toEqual(cordycepsTree());
    const minified = minifyPreservingOrder(JSON.stringify(parsed.task.loop_strategy, null, 2));
    expect(JSON.stringify(loopStrategyToJson(strategy))).toBe(minified);
  });
});

describe("fixtures/strategy replay — byte identity", () => {
  it("cordyceps_tree.json round-trips byte-identically", () => {
    const raw = fixture("cordyceps_tree.json");
    const parsed = loopStrategyFromJson(JSON.parse(raw));
    expect(parsed).toEqual(cordycepsTree());
    expect(JSON.stringify(loopStrategyToJson(parsed))).toBe(minifyPreservingOrder(raw));
  });

  it("strategy_ref.json round-trips byte-identically (suite order: built_in, custom)", () => {
    const raw = fixture("strategy_ref.json");
    const parsed = JSON.parse(raw) as { built_in: unknown; custom: unknown };
    const builtIn = strategyRefFromJson(parsed.built_in);
    const custom = strategyRefFromJson(parsed.custom);
    expect(builtIn).toEqual({ kind: "built_in", value: cordycepsTree() });
    expect(custom).toEqual({ kind: "custom", value: "my-harness::DoubleVerify" });
    // Whole-suite byte identity, field order built_in then custom.
    const suiteJson = JSON.stringify({
      built_in: strategyRefToJson(builtIn),
      custom: strategyRefToJson(custom),
    });
    expect(suiteJson).toBe(minifyPreservingOrder(raw));
  });
});

describe("maxSteps — advisory worst-case turn bound (#122)", () => {
  function react(budget: BudgetPolicy): LoopStrategy {
    return { kind: "react", budget, agent: "", toolset: "" };
  }

  function selfVerifying(inner: LoopStrategy): LoopStrategy {
    return { kind: "self_verifying", inner, evaluator: "ev" };
  }

  function planExecute(plan: LoopStrategy, execute: LoopStrategy): LoopStrategy {
    return { kind: "plan_execute", plan, execute };
  }

  function ralph(inner: LoopStrategy): LoopStrategy {
    return { kind: "ralph", inner, agent: "r" };
  }

  function hillClimbing(inner: LoopStrategy, maxStagnation: number): LoopStrategy {
    return {
      kind: "hill_climbing",
      inner,
      direction: "maximize",
      max_stagnation: maxStagnation,
      revert_on_no_improvement: true,
      min_improvement_delta: 0,
      evaluator: "metric",
    };
  }

  it("react leaf returns its budget allowance (undefined when unlimited)", () => {
    expect(loopStrategyMaxSteps(react({ kind: "per_loop", value: 4 }))).toBe(4);
    expect(loopStrategyMaxSteps(react({ kind: "total_steps", value: 7 }))).toBe(7);
    expect(loopStrategyMaxSteps(react({ kind: "per_attempt", value: 5 }))).toBe(5);
    expect(loopStrategyMaxSteps(react({ kind: "unlimited" }))).toBeUndefined();
  });

  it("self_verifying adds exactly one evaluator turn", () => {
    expect(loopStrategyMaxSteps(selfVerifying(react({ kind: "per_loop", value: 12 })))).toBe(13);
  });

  it("plan_execute is the per-task sum of plan + execute", () => {
    const s = planExecute(
      react({ kind: "per_loop", value: 4 }),
      react({ kind: "per_loop", value: 6 }),
    );
    expect(loopStrategyMaxSteps(s)).toBe(10);
  });

  it("hill_climbing is inner × (max_stagnation + 1)", () => {
    // inner=5, max_stagnation=2 ⇒ 5 × (2 + 1) = 15.
    expect(loopStrategyMaxSteps(hillClimbing(react({ kind: "per_loop", value: 5 }), 2))).toBe(15);
    // The Number.MAX_SAFE_INTEGER sentinel ⇒ unbounded windows ⇒ undefined.
    expect(
      loopStrategyMaxSteps(
        hillClimbing(react({ kind: "per_loop", value: 5 }), Number.MAX_SAFE_INTEGER),
      ),
    ).toBeUndefined();
  });

  it("ralph is the per-window bound (just its inner)", () => {
    expect(loopStrategyMaxSteps(ralph(react({ kind: "per_loop", value: 9 })))).toBe(9);
    // Canonical cordyceps subtree wrapped in Ralph ⇒ per-window 17.
    const s = ralph(
      planExecute(
        react({ kind: "per_loop", value: 4 }),
        selfVerifying(react({ kind: "per_loop", value: 12 })),
      ),
    );
    expect(loopStrategyMaxSteps(s)).toBe(17);
  });

  it("canonical cordyceps subtree PlanExecute[ReAct{4}, SelfVerifying[ReAct{12}]] = 17", () => {
    const subtree = planExecute(
      react({ kind: "per_loop", value: 4 }),
      selfVerifying(react({ kind: "per_loop", value: 12 })),
    );
    expect(loopStrategyMaxSteps(subtree)).toBe(17);
  });

  it("the cordyceps fixture's per-window bound is 17", () => {
    expect(loopStrategyMaxSteps(cordycepsTree())).toBe(17);
    const parsed = loopStrategyFromJson(JSON.parse(fixture("cordyceps_tree.json")));
    expect(loopStrategyMaxSteps(parsed)).toBe(17);
  });

  it("any unlimited node anywhere collapses the whole figure to undefined", () => {
    // Plan leaf unlimited ⇒ undefined.
    expect(
      loopStrategyMaxSteps(
        planExecute(
          react({ kind: "unlimited" }),
          selfVerifying(react({ kind: "per_loop", value: 12 })),
        ),
      ),
    ).toBeUndefined();
    // Execute's inner ReAct unlimited ⇒ undefined.
    expect(
      loopStrategyMaxSteps(
        planExecute(
          react({ kind: "per_loop", value: 4 }),
          selfVerifying(react({ kind: "unlimited" })),
        ),
      ),
    ).toBeUndefined();
    // HillClimbing-wrapped unlimited inner ⇒ undefined (propagates to root).
    expect(
      loopStrategyMaxSteps(ralph(hillClimbing(react({ kind: "unlimited" }), 2))),
    ).toBeUndefined();
  });

  it("StrategyRef: custom is undefined, built_in delegates", () => {
    expect(strategyRefMaxSteps({ kind: "custom", value: "x" })).toBeUndefined();
    expect(
      strategyRefMaxSteps({ kind: "built_in", value: react({ kind: "per_loop", value: 4 }) }),
    ).toBe(4);
  });

  it("an overflowing bound (exceeds u32 range) yields undefined", () => {
    // inner ≈ u32::MAX/2, max_stagnation=3 ⇒ (u32::MAX/2) × 4 overflows ⇒ undefined.
    const s = hillClimbing(react({ kind: "per_loop", value: Math.floor(0xffffffff / 2) }), 3);
    expect(loopStrategyMaxSteps(s)).toBeUndefined();
  });
});
