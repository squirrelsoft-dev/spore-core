/**
 * Example-crate test (NO model): the composed `HillClimbing(inner: ReAct,
 * evaluator)` strategy validates against the example's {@link ExecutionRegistry}
 * — the structured `propose` slot's bare ReAct declares `output:
 * "propose-schema"` (resolved in the registry), and the empty-handle `evaluator`
 * resolves to the default-filled metric evaluator. The leaves use EMPTY
 * agent/toolset handles that the harness default-fills at `build`; here we mirror
 * that fill (empty-key agent + toolset + metric evaluator) so the standalone
 * registry validates exactly as the assembled harness would. Mirrors the Rust
 * example's `registry_validates` test.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ModelAgent,
  OllamaModelInterface,
  SessionId,
  metric,
  newTask,
  termination,
  type HillClimbingDirection,
  type SandboxProvider,
} from "@spore/core";

import { buildRegistry, hillClimbingStrategy } from "../src/main.js";

const MODEL = "gemma4:e4b";
const BASE = "http://localhost:11434";

/** A trivial constant-score metric evaluator — enough to fill the empty-key slot. */
class StubEvaluator implements metric.MetricEvaluator {
  async evaluate(
    _sandbox: SandboxProvider,
    _sessionState: termination.SessionStateSnapshot,
    _signal?: AbortSignal,
  ): Promise<metric.MetricOutcome> {
    return { kind: "ok", result: metric.newMetricResult(0) };
  }
  direction(): HillClimbingDirection {
    return "maximize";
  }
  description(): string {
    return "stub";
  }
}

describe("hill-climbing example composition", () => {
  // The composed strategy is `HillClimbing(inner: ReAct{propose-schema}, "")`.
  it("strategy shape carries the structured propose-slot output schema", () => {
    const s = hillClimbingStrategy(8);
    if (s.kind !== "hill_climbing")
      throw new Error("root must be HillClimbing");
    if (s.inner.kind !== "react") throw new Error("propose must be ReAct");
    expect(s.inner.output).toBe("propose-schema");
    expect(s.inner.budget).toEqual({ kind: "per_loop", value: 8 });
    expect(s.direction).toBe("maximize");
    expect(s.max_stagnation).toBe(2);
    expect(s.min_improvement_delta).toBe(0);
    expect(s.revert_on_no_improvement).toBe(true);
    expect(s.evaluator).toBe("");
  });

  // AC: handles resolve from the registry at run entry. Default-fill the empty-key
  // agent + toolset + metric evaluator (as `HarnessBuilder.build` does), then
  // assert `validate()` does not throw against the real task.
  it("registry validates the real task", () => {
    const model = OllamaModelInterface.withBaseUrl(MODEL, BASE);
    const registry = buildRegistry()
      .toBuilder()
      .fillDefaultAgent(new ModelAgent(AgentId.of("default"), model))
      .fillDefaultToolset(new EmptyToolRegistry())
      .fillDefaultMetricEvaluator(new StubEvaluator())
      .build();
    const task = newTask(
      "refine the README",
      SessionId.generate(),
      hillClimbingStrategy(8),
    );
    expect(() => registry.validate(task)).not.toThrow();
  });
});
