/**
 * End-to-end coverage for adaptive prompt-based tool-calling escalation (#111) —
 * the `AdaptiveToolCallModelInterface` + harness-loop seam.
 *
 * Parity with `rust/crates/spore-core/tests/prompt_tool_call_escalation.rs`.
 * Drives a full `conversational` harness with a scripted `MockModelInterface` to
 * prove the escalation path: native-first, then automatic switch to prompt-based
 * mode after a prose response, then `<tool_call>` markers parsed into a real
 * dispatch — with no model lists and no manual wrapping.
 */

import { describe, expect, it } from "vitest";
import { z } from "zod";

import { HarnessBuilder, simpleTask, toolRegistry } from "../src/index.js";
import { toolOutput } from "../src/harness/types.js";
import { MockModelInterface } from "../src/model/mock.js";
import type { ModelResponse, ProviderInfo } from "../src/model/schemas.js";

const { defineTool } = toolRegistry;

function provider(): ProviderInfo {
  return { name: "mock", model_id: "mock-1", context_window: 8192 };
}

function usage(): ModelResponse["usage"] {
  return { input_tokens: 1, output_tokens: 1 };
}

function text(t: string): ModelResponse {
  return { content: [{ type: "text", text: t }], usage: usage(), stop_reason: "end_turn" };
}

/** A pure tool that records how many times it was dispatched. */
function calculatorTool(hits: { count: number }) {
  return defineTool({
    name: "calculator",
    description: "Evaluate a math expression",
    input: z.object({ expression: z.string() }),
    execute: async () => {
      hits.count += 1;
      return toolOutput.success("4");
    },
  });
}

describe("adaptive prompt-based tool-calling escalation (#111)", () => {
  it("escalates a prose response to a prompt-based tool call and dispatches once", async () => {
    // Turn 1: prose with action-intent → escalate. Turn 2 (now prompt mode):
    // a <tool_call> marker → parsed + dispatched. Turn 3: final answer.
    const hits = { count: 0 };
    const model = new MockModelInterface(provider());
    model.pushResponse(text("Sure — I'll use the calculator tool to compute 2+2."));
    model.pushResponse(
      text('<tool_call><name>calculator</name><input>{"expression":"2+2"}</input></tool_call>'),
    );
    model.pushResponse(text("The answer is 4"));

    const harness = HarnessBuilder.conversational(model).tool(calculatorTool(hits)).build();
    const result = await harness.run({ task: simpleTask("What is 2+2?") });

    expect(result.kind).toBe("success");
    if (result.kind !== "success") throw new Error("expected success");
    // Reaching turn 3's answer proves: turn 1 did NOT terminate the run
    // (escalation fired), and turn 2's marker text was parsed into a real tool
    // call rather than being treated as a prose final answer.
    expect(result.output).toBe("The answer is 4");
    expect(result.turns).toBe(3);
    expect(hits.count).toBe(1);
  });

  it("does not escalate a plain final answer", async () => {
    // Tools advertised, but a plain final answer with no action-intent language.
    // The conservative heuristic must NOT escalate — completes on turn 1.
    const hits = { count: 0 };
    const model = new MockModelInterface(provider());
    model.pushResponse(text("The answer is 4."));

    const harness = HarnessBuilder.conversational(model).tool(calculatorTool(hits)).build();
    const result = await harness.run({ task: simpleTask("What is 2+2?") });

    expect(result.kind).toBe("success");
    if (result.kind !== "success") throw new Error("expected success");
    expect(result.output).toBe("The answer is 4.");
    expect(result.turns).toBe(1);
    expect(hits.count).toBe(0);
  });

  it("leaves the native tool-call path unaffected", async () => {
    // A model that emits a native tool-use block (tools advertised) dispatches
    // normally through the adaptive wrapper while the flag is unset — no prompt
    // injection, no marker parsing involved.
    const hits = { count: 0 };
    const model = new MockModelInterface(provider());
    model.pushResponse({
      content: [{ type: "tool_use", id: "c1", name: "calculator", input: { expression: "2+2" } }],
      usage: usage(),
      stop_reason: "tool_use",
    });
    model.pushResponse(text("The answer is 4"));

    const harness = HarnessBuilder.conversational(model).tool(calculatorTool(hits)).build();
    const result = await harness.run({ task: simpleTask("What is 2+2?") });

    expect(result.kind).toBe("success");
    if (result.kind !== "success") throw new Error("expected success");
    expect(result.output).toBe("The answer is 4");
    expect(result.turns).toBe(2);
    expect(hits.count).toBe(1);
  });
});
