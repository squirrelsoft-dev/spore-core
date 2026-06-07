/**
 * Fixture-replay integration test for the PlanExecute loop (issue #59).
 *
 * Loads `fixtures/model_responses/harness/plan_execute_loop.jsonl` — a full
 * two-phase trace: one plan turn producing K=2 tasks, followed by K execute
 * completions — and drives a `StandardHarness` with `LoopStrategy.plan_execute`,
 * asserting that:
 *   1. The plan turn is captured and parsed into a 2-task list.
 *   2. The execute phase drains BOTH tasks sequentially (one turn each).
 *   3. The run SUCCEEDS, `output` is the LAST step's final_response (Q2), and
 *      `turns === 3` (one plan turn + one per task — shared budget, Q1).
 *
 * Must produce the same outcome as the Rust integration test
 * (`rust/crates/spore-core/tests/plan_execute_loop_fixture_replay.rs`) — never
 * edit the fixture to make a failing implementation pass.
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
  registryWith,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/model_responses/harness/plan_execute_loop.jsonl");

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

function config(): HarnessConfig {
  const jsonl = readFileSync(fixturePath, "utf-8");
  // Positional replay: the plan turn (line 1) then the two execute steps.
  const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
  const agent = new ModelAgent(AgentId.of("plan-execute"), replay);
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
  };
}

describe("PlanExecute loop fixture replay — plan_execute_loop.jsonl", () => {
  it("drives the full two-phase trace to Success identically to Rust", async () => {
    const harness = new StandardHarness(config());
    const task = newTask("build a CLI", SessionId.of("plan-execute-fixture"), {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
        output: "",
      },
      execute: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
      },
    });
    const result = await harness.run({ task });
    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      // Q2: the success handle is the LAST completed step's final text.
      expect(result.output).toBe("wrote the integration tests");
      // Q1: one plan turn + one turn per task (2) under the shared budget.
      expect(result.turns).toBe(3);
    }
  });
});
