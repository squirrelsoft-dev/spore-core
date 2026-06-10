/**
 * Example-crate test (NO model): the composed `SelfVerifying(inner: ReAct,
 * evaluator)` strategy validates against the example's {@link ExecutionRegistry}
 * — the structured `worker` slot's bare ReAct declares `output: "worker-schema"`
 * (resolved in the registry), and the empty-handle `evaluator` resolves to the
 * default-filled verifier. The leaves use EMPTY agent/toolset handles that the
 * harness default-fills at `build`; here we mirror that fill (empty-key agent +
 * toolset + verifier) so the standalone registry validates exactly as the
 * assembled harness would. Mirrors the Rust example's `registry_validates` test.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ModelAgent,
  OllamaModelInterface,
  SessionId,
  newTask,
  verifier,
} from "@spore/core";

import { buildRegistry, selfVerifyingStrategy } from "../src/main.js";

const MODEL = "gemma4:e4b";
const BASE = "http://localhost:11434";

describe("self-verifying example composition", () => {
  // The composed strategy is `SelfVerifying(inner: ReAct{worker-schema}, "")`.
  it("strategy shape carries the structured worker-slot output schema", () => {
    const s = selfVerifyingStrategy(3);
    if (s.kind !== "self_verifying")
      throw new Error("root must be SelfVerifying");
    if (s.inner.kind !== "react") throw new Error("worker must be ReAct");
    expect(s.inner.output).toBe("worker-schema");
    expect(s.inner.budget).toEqual({ kind: "per_loop", value: 3 });
    // Empty evaluator handle ⇒ default-filled verifier.
    expect(s.evaluator).toBe("");
  });

  // AC: handles resolve from the registry at run entry. Default-fill the empty-key
  // agent + toolset + verifier (as `HarnessBuilder.build` does), then assert
  // `validate()` does not throw against the real task.
  it("registry validates the real task", () => {
    const model = OllamaModelInterface.withBaseUrl(MODEL, BASE);
    const v = new verifier.EvaluatorResponseVerifier({
      pass_pattern: "(?im)^\\s*PASS\\s*$",
      fail_pattern: "(?im)FAIL:\\s*.+",
      max_iterations: 3,
    });
    const registry = buildRegistry()
      .toBuilder()
      .fillDefaultAgent(new ModelAgent(AgentId.of("default"), model))
      .fillDefaultToolset(new EmptyToolRegistry())
      .fillDefaultVerifier(v)
      .build();
    const task = newTask(
      "draft parseIntList",
      SessionId.generate(),
      selfVerifyingStrategy(3),
    );
    expect(() => registry.validate(task)).not.toThrow();
  });
});
