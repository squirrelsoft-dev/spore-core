/**
 * Fixture-replay integration test for Agent (issue #2).
 *
 * Loads `fixtures/model_responses/agent/turn_classification.jsonl` from the
 * repo root, drives a `ModelAgent` backed by `ReplayModelInterface`, and
 * asserts the same classifications as the Rust integration test.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  AgentId,
  emptyContext,
  ModelAgent,
  ReplayModelInterface,
  type ProviderInfo,
} from "../src/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
// tests/ -> packages/core/ -> packages/ -> typescript/ -> repo root.
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/model_responses/agent/turn_classification.jsonl");

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

describe("Agent fixture replay — turn_classification.jsonl", () => {
  it("classifies recorded turns consistently with Rust", async () => {
    const jsonl = readFileSync(fixturePath, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const agent = new ModelAgent(AgentId.of("fixture-agent"), replay);

    // 1. Plain text → FinalResponse("hello")
    const r1 = await agent.turn(emptyContext());
    expect(r1.kind).toBe("final_response");
    if (r1.kind === "final_response") {
      expect(r1.content).toBe("hello");
      expect(r1.usage.input_tokens).toBe(5);
      expect(r1.usage.output_tokens).toBe(1);
    }

    // 2. Single tool call → ToolCallRequested(1)
    const r2 = await agent.turn(emptyContext());
    expect(r2.kind).toBe("tool_call_requested");
    if (r2.kind === "tool_call_requested") {
      expect(r2.calls).toHaveLength(1);
      expect(r2.calls[0]!.name).toBe("read_file");
      expect(r2.calls[0]!.id).toBe("toolu_a");
      expect(r2.usage.input_tokens).toBe(20);
    }

    // 3. Parallel tool calls → ToolCallRequested(2)
    const r3 = await agent.turn(emptyContext());
    expect(r3.kind).toBe("tool_call_requested");
    if (r3.kind === "tool_call_requested") {
      expect(r3.calls).toHaveLength(2);
      expect(r3.calls[0]!.id).toBe("toolu_b1");
      expect(r3.calls[1]!.id).toBe("toolu_b2");
    }

    // 4. Empty content + end_turn → EmptyResponse(usage_some)
    const r4 = await agent.turn(emptyContext());
    expect(r4.kind).toBe("error");
    if (r4.kind === "error") {
      expect(r4.error.kind).toBe("empty_response");
      expect(r4.usage?.input_tokens).toBe(3);
      expect(r4.usage?.output_tokens).toBe(0);
    }
  });
});
