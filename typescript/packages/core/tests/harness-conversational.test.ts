/**
 * Unit tests for the few-lines conversational path (parity with Rust's
 * `HarnessBuilder::conversational` + `Task::simple`).
 *
 * Mirrors the intent of `rust/crates/spore-core/src/harness.rs` —
 * `conversational(model)` wires a ModelAgent / EmptyToolRegistry / NullSandbox /
 * StandardContextManager / CompleteOnFinalResponse and succeeds on the model's
 * first final response; `simpleTask` defaults to a fresh session + ReAct(8).
 */

import { describe, expect, it } from "vitest";

import {
  CompleteOnFinalResponse,
  EmptyToolRegistry,
  HarnessBuilder,
  NullSandbox,
  emptyBudgetSnapshot,
  emptySessionState,
  runResultSessionState,
  simpleTask,
} from "../src/index.js";
import { MockModelInterface } from "../src/model/mock.js";
import type { ModelResponse } from "../src/model/schemas.js";

function finalResponse(text: string): ModelResponse {
  return {
    content: [{ type: "text", text }],
    usage: { input_tokens: 5, output_tokens: 7 },
    stop_reason: "end_turn",
  };
}

function mockModel(...texts: string[]): MockModelInterface {
  const model = new MockModelInterface({ name: "mock", model_id: "mock-model" });
  for (const t of texts) model.pushResponse(finalResponse(t));
  return model;
}

describe("HarnessBuilder.conversational", () => {
  it("builds a single-turn harness that succeeds on the first final response", async () => {
    const model = mockModel("Hello there, friend!");
    const harness = HarnessBuilder.conversational(model).build();

    const result = await harness.run({ task: simpleTask("Say hi.") });

    expect(result.kind).toBe("success");
    if (result.kind !== "success") throw new Error("expected success");
    expect(result.output).toBe("Hello there, friend!");
    expect(result.turns).toBe(1);
    // The model was called exactly once — respond-and-stop termination.
    expect(model.callCount).toBe(1);
    // Lossless post-run history (issue #102): user line + assistant reply.
    const state = runResultSessionState(result);
    expect(state.messages.length).toBeGreaterThanOrEqual(2);
  });

  it("wires the documented defaults (no tools, allow-all null sandbox, respond-and-stop)", async () => {
    // EmptyToolRegistry advertises no schemas, so the model is offered no tools.
    expect(new EmptyToolRegistry().schemas()).toEqual([]);
    expect(new EmptyToolRegistry().isAlwaysHalt("anything")).toBe(false);

    // NullSandbox allows everything (the boundary is never exercised).
    const sandbox = new NullSandbox();
    await expect(sandbox.validate({ id: "x", name: "noop", input: {} })).resolves.toBeNull();
    expect(sandbox.isolationMode()).toEqual({ kind: "workspace_scoped" });

    // CompleteOnFinalResponse always continues (accept the first final response).
    const policy = new CompleteOnFinalResponse();
    await expect(policy.evaluate(emptySessionState(), emptyBudgetSnapshot())).resolves.toEqual({
      kind: "continue",
    });
  });
});

describe("simpleTask", () => {
  it("defaults to a fresh session id and a ReAct(8) loop", () => {
    const a = simpleTask("Do a thing.");
    const b = simpleTask("Do a thing.");

    expect(a.instruction).toBe("Do a thing.");
    expect(a.loop_strategy).toEqual({
      kind: "react",
      budget: { kind: "per_loop", value: 8 },
      agent: "",
      toolset: "",
    });
    // A fresh session id each call.
    expect(a.session_id.asString()).not.toBe(b.session_id.asString());
    // Default (empty) budget.
    expect(a.budget).toEqual({});
  });
});
