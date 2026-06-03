/**
 * Fixture-replay integration test for Harness (issue #3).
 *
 * Loads `fixtures/model_responses/harness/react_loop.jsonl` and drives a
 * `StandardHarness` with `LoopStrategy::ReAct`, asserting the loop:
 *   1. Dispatches the recorded tool call
 *   2. Loops to the next agent turn
 *   3. Returns `Success` with the recorded final response
 *
 * Must produce the same outcome as the Rust integration test.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  ModelAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  newTask,
  type HarnessConfig,
  type ProviderInfo,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/model_responses/harness/react_loop.jsonl");

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

describe("Harness fixture replay — react_loop.jsonl", () => {
  it("dispatches recorded tool call then completes consistently with Rust", async () => {
    const jsonl = readFileSync(fixturePath, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const agent = new ModelAgent(AgentId.of("fixture-agent"), replay);

    const toolRegistry = new ScriptedToolRegistry().push({
      kind: "success",
      content: "127.0.0.1 localhost",
    });

    const config: HarnessConfig = {
      agent,
      toolRegistry,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    };
    const harness = new StandardHarness(config);

    const task = newTask("read /etc/hosts then summarize", SessionId.of("fixture-session"), {
      kind: "re_act",
      max_iterations: 5,
    });

    const result = await harness.run({ task });
    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      expect(result.output).toBe("127.0.0.1 localhost");
      expect(result.turns).toBe(2);
      expect(result.usage.input_tokens).toBe(30); // 12 + 18
      expect(result.usage.output_tokens).toBe(14); // 8 + 6
    }
    expect(toolRegistry.callCount).toBe(1);
  });
});
