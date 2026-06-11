/**
 * Example-crate test (NO model): the orchestrator's composed `PlanExecute(plan:
 * ReAct, execute: ReAct)` strategy validates against the example's
 * {@link ExecutionRegistry} — the structured `plan` slot's bare ReAct declares
 * `output: "plan-schema"` (resolved in the registry), and the empty agent/toolset
 * handles are default-filled exactly as the assembled harness would. Mirrors the
 * Rust example's `registry_validates` test.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ModelAgent,
  OllamaModelInterface,
  SessionId,
  newTask,
} from "@spore/core";

import { buildRegistry, planExecuteStrategy } from "../src/main.js";

const MODEL = "gemma4:e4b";
const BASE = "http://localhost:11434";

describe("multi-agent orchestrator composition", () => {
  // The orchestrator strategy is `PlanExecute(plan: ReAct{plan-schema}, execute: ReAct)`.
  it("strategy shape carries the structured plan-slot output schema", () => {
    const s = planExecuteStrategy();
    if (s.kind !== "plan_execute") throw new Error("root must be PlanExecute");
    if (s.plan.kind !== "react") throw new Error("plan must be ReAct");
    expect(s.plan.output).toBe("plan-schema");
    expect(s.execute.kind).toBe("react");
  });

  // AC: handles resolve from the registry at run entry. Default-fill the empty-key
  // agent + toolset (as `HarnessBuilder.build` does), then assert `validate()`
  // does not throw against the real task.
  it("registry validates the real task", () => {
    const model = OllamaModelInterface.withBaseUrl(MODEL, BASE);
    const registry = buildRegistry()
      .toBuilder()
      .fillDefaultAgent(new ModelAgent(AgentId.of("default"), model))
      .fillDefaultToolset(new EmptyToolRegistry())
      .build();
    const task = newTask(
      "research, write, and save a report",
      SessionId.generate(),
      planExecuteStrategy(),
    );
    expect(() => registry.validate(task)).not.toThrow();
  });
});
