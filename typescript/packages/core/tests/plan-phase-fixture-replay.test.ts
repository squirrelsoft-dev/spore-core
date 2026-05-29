/**
 * Fixture-replay integration test for the PlanExecute plan phase (issue #70).
 *
 * Loads `fixtures/model_responses/harness/plan_phase_basic.jsonl` and drives a
 * `StandardHarness` with `LoopStrategy.plan_execute`, asserting that:
 *   1. The plan turn's recorded `final_response` is captured into the exact
 *      `PlanArtifact` (tasks + rationale), stored under `extras["plan_execute"]`.
 *   2. The fenced ```json variant is captured identically (fence-strip rule).
 *   3. The plan turn is consumed and parsed into a non-empty task list, so the
 *      run proceeds into the execute phase (issue #59). This fixture provides
 *      ONLY the single plan turn, so the first execute step's ReAct sub-loop
 *      exhausts the replay and the run aborts with `step_failed` (task_index 0)
 *      — proving the harness consumed the planner response and entered execute.
 *
 * Must produce the same outcome as the Rust integration test
 * (`rust/crates/spore-core/tests/plan_phase_fixture_replay.rs`) — never edit the
 * fixture to make a failing implementation pass.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  ModelAgent,
  PLAN_EXECUTE_EXTRAS_KEY,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  capturePlanArtifact,
  emptySessionState,
  newTask,
  type HarnessConfig,
  type ProviderInfo,
  type RecordedExchange,
} from "../src/index.js";
import { RecordedExchangeSchema } from "../src/model/schemas.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/model_responses/harness/plan_phase_basic.jsonl");

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

function fixtureExchanges(): RecordedExchange[] {
  const jsonl = readFileSync(fixturePath, "utf-8");
  return jsonl
    .split(/\r?\n/)
    .filter((l) => l.trim() !== "")
    .map((l) => RecordedExchangeSchema.parse(JSON.parse(l)));
}

/** Extract the single text block from a recorded response. */
function responseText(exchange: RecordedExchange): string {
  const block = exchange.response.content.find((b) => b.type === "text");
  if (block == null || block.type !== "text") {
    throw new Error("recorded response has no text block");
  }
  return block.text;
}

function configFor(exchange: RecordedExchange): HarnessConfig {
  // Positional single-exchange replay: each plan run consumes exactly one turn.
  const replay = new ReplayModelInterface([exchange], provider);
  const agent = new ModelAgent(AgentId.of("planner"), replay);
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
  };
}

describe("PlanExecute plan-phase fixture replay — plan_phase_basic.jsonl", () => {
  it("captures the plain-JSON case identically to Rust", async () => {
    const exchanges = fixtureExchanges();
    expect(exchanges.length).toBeGreaterThanOrEqual(2);

    const state = emptySessionState();
    const harness = new StandardHarness(configFor(exchanges[0]!));
    const task = newTask("build something", SessionId.of("plan-fixture"), {
      kind: "plan_execute",
      plan_model: null,
    });

    const result = await harness.run({ task, session_state: state });
    // The plan turn is consumed + parsed into a non-empty list, so the run
    // enters the execute phase; the single-exchange replay then exhausts on the
    // first step and aborts with step_failed (issue #59, Q5).
    expect(result.kind).toBe("failure");
    if (result.kind === "failure") {
      expect(result.reason.kind).toBe("step_failed");
      if (result.reason.kind === "step_failed") {
        expect(result.reason.task_index).toBe(0);
      }
      expect(result.turns).toBeGreaterThanOrEqual(1); // at least the plan turn
    }

    // Stored artifact matches the recorded response (and the public capture).
    const expected = {
      tasks: ["scaffold the project", "add the argument parser", "write the integration tests"],
      rationale: "deliver a working CLI incrementally",
    };
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toEqual(expected);

    const captured = capturePlanArtifact(responseText(exchanges[0]!));
    expect(captured.ok).toBe(true);
    if (captured.ok) expect(captured.artifact).toEqual(expected);
  });

  it("captures the fenced-```json case identically to Rust (fence-strip)", async () => {
    const exchanges = fixtureExchanges();
    expect(exchanges.length).toBeGreaterThanOrEqual(2);

    const state = emptySessionState();
    const harness = new StandardHarness(configFor(exchanges[1]!));
    const task = newTask("build something", SessionId.of("plan-fixture"), {
      kind: "plan_execute",
      plan_model: null,
    });

    const result = await harness.run({ task, session_state: state });
    expect(result.kind).toBe("failure");
    if (result.kind === "failure") {
      expect(result.reason.kind).toBe("step_failed");
      if (result.reason.kind === "step_failed") {
        expect(result.reason.task_index).toBe(0);
      }
      expect(result.turns).toBeGreaterThanOrEqual(1);
    }

    const expected = {
      tasks: ["draft the outline", "write the reference section"],
      rationale: "docs follow the code",
    };
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toEqual(expected);

    const captured = capturePlanArtifact(responseText(exchanges[1]!));
    expect(captured.ok).toBe(true);
    if (captured.ok) expect(captured.artifact).toEqual(expected);
  });
});
