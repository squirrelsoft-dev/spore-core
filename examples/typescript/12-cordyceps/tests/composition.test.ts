/**
 * Example-crate tests (NO model): the tree is data, max_steps is computable, the
 * registry validates the real task, and the registered evaluator is Default-FAIL.
 * Mirrors the Rust example's `mod tests` in `main.rs`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  LoopStrategySchema,
  SessionId,
  emptyAggregateUsage,
  loopStrategyMaxSteps,
  type RunResult,
} from "@spore/core";

import {
  EXEC_EVALUATOR_KEY,
  buildRegistry,
  buildTask,
  execEvaluator,
} from "../src/main.js";

const here = dirname(fileURLToPath(import.meta.url));
const TREE_PATH = resolve(
  here,
  "..",
  "..",
  "..",
  "..",
  "fixtures",
  "strategy",
  "cordyceps_tree.json",
);

const MODEL = "gemma4:e4b";
const BASE = "http://localhost:11434";

function readTreeRaw(): unknown {
  return JSON.parse(readFileSync(TREE_PATH, "utf8"));
}

describe("cordyceps example composition", () => {
  // AC: the tree is DATA. Deserialize the canonical fixture, re-serialize, and
  // assert the value round-trips; then assert the expected keys / budgets are
  // present.
  it("tree is data and round-trips through serde", () => {
    const raw = readTreeRaw();
    const tree = LoopStrategySchema.parse(raw);

    if (tree.kind !== "ralph") throw new Error("root must be Ralph");
    expect(tree.agent).toBe("ralph-agent");
    expect(tree.behavior).toEqual({ kind: "escalate" });
    if (tree.inner.kind !== "plan_execute")
      throw new Error("Ralph inner must be PlanExecute");
    const pe = tree.inner;
    if (pe.plan.kind !== "react") throw new Error("plan must be ReAct");
    expect(pe.plan.agent).toBe("planner");
    expect(pe.plan.toolset).toBe("plan-tools");
    expect(pe.plan.output).toBe("plan-schema");
    expect(pe.plan.budget).toEqual({ kind: "per_loop", value: 4 });
    if (pe.execute.kind !== "self_verifying")
      throw new Error("execute must be SelfVerifying");
    const sv = pe.execute;
    expect(sv.evaluator).toBe(EXEC_EVALUATOR_KEY);
    if (sv.inner.kind !== "react") throw new Error("worker must be ReAct");
    expect(sv.inner.agent).toBe("executor");
    expect(sv.inner.toolset).toBe("exec-tools");
    expect(sv.inner.output).toBe("worker-schema");
    expect(sv.inner.budget).toEqual({ kind: "per_loop", value: 12 });
  });

  // AC5: the fully-bounded tree's per-window worst case is 17; one `unlimited`
  // anywhere collapses it to undefined.
  it("max_steps is 17", () => {
    const tree = LoopStrategySchema.parse(readTreeRaw());
    expect(loopStrategyMaxSteps(tree)).toBe(17);

    if (tree.kind !== "ralph") throw new Error("unreachable");
    const pe = tree.inner;
    if (pe.kind !== "plan_execute") throw new Error("unreachable");
    const sv = pe.execute;
    if (sv.kind !== "self_verifying") throw new Error("unreachable");
    const worker = sv.inner;
    if (worker.kind !== "react") throw new Error("unreachable");
    worker.budget = { kind: "unlimited" };
    expect(loopStrategyMaxSteps(tree)).toBeUndefined();
  });

  // AC: handles resolve from the ExecutionRegistry at run entry. Build the real
  // registry + task and assert `validate()` does not throw.
  it("registry validates the real task", () => {
    const registry = buildRegistry(MODEL, BASE);
    const task = buildTask("audit the repo", SessionId.generate());
    expect(() => registry.validate(task)).not.toThrow();
  });

  // The Default-FAIL evaluator: PASS clears, indeterminate output fails.
  it("exec evaluator is Default-FAIL", async () => {
    const v = execEvaluator();
    expect(v.maxIterations()).toBe(1);
    const success = (out: string): RunResult => ({
      kind: "success",
      output: out,
      session_id: SessionId.of("s"),
      usage: emptyAggregateUsage(),
      turns: 1,
      session_state: { messages: [], extras: {} },
    });
    const input = (evalText: string) => ({
      build_result: success("audited"),
      eval_result: success(evalText),
      workspace: "/tmp",
      iteration: 0,
    });
    expect((await v.verify(input("verdict: PASS"))).kind).toBe("passed");
    expect((await v.verify(input("hmm, unclear"))).kind).toBe("failed");
  });
});
