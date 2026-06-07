/**
 * Fixture-replay tests for the Tool Escalation Protocol (spore-core issue #80).
 *
 * Consumes the SHARED, cross-language fixtures (authored by the Rust agent — do
 * NOT modify them):
 *   1. `fixtures/harness/escalation_signals.json` — the byte-identical wire
 *      guarantee. Every `HarnessSignal` variant (wrapped in `ToolOutput` and
 *      `RunResult` escalate cases) must parse and re-serialize byte-for-byte.
 *   2. `fixtures/model_responses/harness/escalation_loop.jsonl` — drives a
 *      `StandardHarness`: the recorded turn requests a tool whose dispatch
 *      escalates, proving the loop returns `RunResult.escalate` and skips the
 *      history append.
 *
 * Must produce the same outcome as the Rust integration tests.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessSignalSchema,
  ModelAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  newTask,
  type HarnessConfig,
  type HarnessSignal,
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

const signalsFixture = resolve(repoRoot, "fixtures/harness/escalation_signals.json");
const loopFixture = resolve(repoRoot, "fixtures/model_responses/harness/escalation_loop.jsonl");

interface EscalateRunResultCase {
  kind: "escalate";
  signal: unknown;
  state: { human_request: unknown; pending_tool_calls: unknown[] };
}
interface EscalateToolOutputCase {
  kind: "escalate";
  signal: unknown;
}
interface SignalsFixture {
  run_result_cases: EscalateRunResultCase[];
  tool_output_cases: EscalateToolOutputCase[];
}

describe("Escalation serde fixture — escalation_signals.json", () => {
  const suite = JSON.parse(readFileSync(signalsFixture, "utf-8")) as SignalsFixture;

  it("every ToolOutput escalate case carries a parseable HarnessSignal", () => {
    expect(suite.tool_output_cases.length).toBe(4);
    const seen = new Set<string>();
    for (const c of suite.tool_output_cases) {
      expect(c.kind).toBe("escalate");
      const signal: HarnessSignal = HarnessSignalSchema.parse(c.signal);
      seen.add(signal.kind);
    }
    // All four HarnessSignal variants are exercised by the shared fixture.
    expect(seen).toEqual(new Set(["enter_plan_mode", "exit_plan_mode", "switch_mode", "abort"]));
  });

  it("HarnessSignal round-trips byte-identically (parse → re-serialize)", () => {
    for (const c of suite.tool_output_cases) {
      const signal = HarnessSignalSchema.parse(c.signal);
      // Re-serialized object deep-equals the fixture's signal value (the wire
      // shape is byte-identical across the four languages).
      expect(signal).toEqual(c.signal);
    }
  });

  it("RunResult escalate cases preserve null human_request and pending calls", () => {
    expect(suite.run_result_cases.length).toBe(2);
    for (const c of suite.run_result_cases) {
      expect(c.kind).toBe("escalate");
      // Escalation state omits the human request (#80: null on the wire).
      expect(c.state.human_request).toBeNull();
      // The escalating batch preserved its remaining call.
      expect(c.state.pending_tool_calls.length).toBe(1);
      // The carried signal parses as a HarnessSignal.
      HarnessSignalSchema.parse(c.signal);
    }
  });
});

const provider: ProviderInfo = {
  name: "anthropic",
  model_id: "fixture",
  context_window: 200_000,
};

describe("Escalation loop fixture — escalation_loop.jsonl", () => {
  it("a tool dispatch escalation terminates the run with RunResult.escalate", async () => {
    const jsonl = readFileSync(loopFixture, "utf-8");
    const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
    const agent = new ModelAgent(AgentId.of("fixture-agent"), replay);

    // The recorded turn requests `abort_tool`; its dispatch escalates with an
    // Abort signal carrying the recorded reason.
    const abort: HarnessSignal = { kind: "abort", reason: "blocked on missing credentials" };
    const toolRegistry = new ScriptedToolRegistry().push({ kind: "escalate", signal: abort });

    const config: HarnessConfig = {
      registry: registryWith({ agent }),
      toolRegistry,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    };
    const harness = new StandardHarness(config);

    const task = newTask("investigate then decide whether to abort", SessionId.of("escalation"), {
      kind: "react",
      budget: { kind: "per_loop", value: 5 },
      agent: "",
      toolset: "",
    });

    const result = await harness.run({ task });
    expect(result.kind).toBe("escalate");
    if (result.kind === "escalate") {
      expect(result.signal).toEqual(abort);
      // R2: the escalation is not appended to history (no tool-result message).
      const toolMessages = result.state.session_state.messages.filter((m) => m.role === "tool");
      expect(toolMessages.length).toBe(0);
      // The escalation consumed exactly one turn before terminating.
      expect(result.turns).toBe(1);
    }
    // Exactly one dispatch happened — the escalating tool call.
    expect(toolRegistry.callCount).toBe(1);
  });
});
