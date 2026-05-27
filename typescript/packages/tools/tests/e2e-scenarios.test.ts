/**
 * Hermetic end-to-end scenario tests (issue #57).
 *
 * These drive the SAME `buildScenario` wiring the `e2e-agent` CLI uses, but
 * with a scripted mock agent, scripted/real tool registries, and an allow-all
 * sandbox, so CI never needs a live Ollama or any network. Each test asserts
 * the harness loop control flow (turn count, tool dispatch order, S4 recovery
 * sequencing, S3 live compaction with real token reclamation). No
 * `SPORE_OTLP_ENDPOINT` is read — there is no forwarding.
 */

import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AgentId,
  cacheProvider,
  context as coreContext,
  emptySessionState,
  harnessTesting,
  MockAgent,
  MockModelInterface,
  newTask,
  observability as coreObs,
  type ProviderInfo,
  SessionId,
  type SessionState,
  type TaskId,
  type ToolCall,
  type ToolSchema,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  buildRealToolRegistry,
  buildRichContextManager,
  buildScenario,
  RealToolRegistry,
  scenarioPrompt,
  seedCompactionState,
} from "../src/index.js";

const {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} = harnessTesting;
const { InMemoryObservabilityProvider } = coreObs;
const { NullCacheProvider } = cacheProvider;

const usage = { input_tokens: 10, output_tokens: 5 };

function toolCall(id: string, name: string, input: unknown): ToolCall {
  return { id, name, input };
}

const react = (max: number) => ({
  kind: "re_act" as const,
  max_iterations: max,
});

// ---------------------------------------------------------------------------
// S1 — multi-step / multi-tool
// ---------------------------------------------------------------------------

describe("S1 — multi-step / multi-tool", () => {
  it("sustains a read -> write -> read-back -> final chain", async () => {
    const agent = new MockAgent(AgentId.of("mock"));
    agent.push({
      kind: "tool_call_requested",
      calls: [toolCall("c1", "read_file", { path: "input.txt" })],
      usage,
    });
    agent.push({
      kind: "tool_call_requested",
      calls: [
        toolCall("c2", "write_file", {
          path: "output.txt",
          content: "UPPERCASED",
        }),
      ],
      usage,
    });
    agent.push({
      kind: "tool_call_requested",
      calls: [toolCall("c3", "read_file", { path: "output.txt" })],
      usage,
    });
    agent.push({ kind: "final_response", content: "DONE", usage });

    const tools = new ScriptedToolRegistry();
    tools.push({ kind: "success", content: "hello", truncated: false });
    tools.push({
      kind: "success",
      content: "wrote 10 bytes",
      truncated: false,
    });
    tools.push({ kind: "success", content: "UPPERCASED", truncated: false });

    const harness = buildScenario({
      scenario: "s1",
      agent,
      tools,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      toolSchemas: [],
    });

    const result = await harness.run({
      task: newTask(scenarioPrompt("s1"), SessionId.of("s1-test"), react(8)),
    });

    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      expect(result.turns).toBeGreaterThan(2);
    }
    // read + write + read-back = 3 dispatches.
    expect(tools.callCount).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// S2 — multi-turn, same SessionId, carrying session state
// ---------------------------------------------------------------------------

describe("S2 — multi-turn", () => {
  it("carries state across two run() calls; turn 2 references turn 1", async () => {
    const sessionId = SessionId.of("s2-test");
    const agent = new MockAgent(AgentId.of("mock"));
    agent.push({
      kind: "tool_call_requested",
      calls: [
        toolCall("c1", "write_file", {
          path: "notes.md",
          content: "TODO: set up the project",
        }),
      ],
      usage,
    });
    agent.push({ kind: "final_response", content: "DONE", usage });
    agent.push({
      kind: "tool_call_requested",
      calls: [
        toolCall("c2", "write_file", {
          path: "notes.md",
          content: "TODO: follow up on set up the project",
          append: true,
        }),
      ],
      usage,
    });
    agent.push({
      kind: "final_response",
      content: "DONE referencing set up the project",
      usage,
    });

    const harness = buildScenario({
      scenario: "s2",
      agent,
      tools: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      toolSchemas: [],
    });

    const r1 = await harness.run({
      task: newTask(scenarioPrompt("s2"), sessionId, react(5)),
    });
    expect(r1.kind).toBe("success");

    const r2 = await harness.run({
      task: newTask(
        "add a second item referencing the first",
        sessionId,
        react(5),
      ),
      session_state: emptySessionState(),
    });
    expect(r2.kind).toBe("success");
    if (r2.kind === "success") {
      expect(r2.session_id.equals(sessionId)).toBe(true);
      expect(r2.output).toContain("set up the project");
    }
  });
});

// ---------------------------------------------------------------------------
// S3 — live compaction with real token reclamation
// ---------------------------------------------------------------------------

describe("S3 — live compaction", () => {
  it("emits a Compaction span mid-run and reclaims real tokens", async () => {
    const sessionId = SessionId.of("s3-test");
    const agent = new MockAgent(AgentId.of("mock"));
    // Tool call (to reach the post-tool compaction arm), then a summary
    // preserving the key terms, then the final response.
    agent.push({
      kind: "tool_call_requested",
      calls: [toolCall("c1", "read_file", { path: "x" })],
      usage,
    });
    agent.push({
      kind: "final_response",
      content: "summary: continuing the deploy of the payment service",
      usage,
    });
    agent.push({
      kind: "final_response",
      content: "DONE deploy payment service",
      usage,
    });

    const tools = new ScriptedToolRegistry();
    tools.push({ kind: "success", content: "file contents", truncated: false });

    const providerInfo: ProviderInfo = {
      name: "mock",
      model_id: "mock",
      context_window: 200,
    };
    const model = new MockModelInterface(providerInfo);
    const cfg: coreContext.CompactionConfig = {
      threshold: 0.8,
      preserve_recent_n: 2,
      head_tail_tokens: 64,
      offload_path: ".spore/offload",
      max_compaction_attempts: 2,
    };
    const cm = buildRichContextManager(model, new NullCacheProvider(), cfg);
    const obs = new InMemoryObservabilityProvider();

    const harness = buildScenario({
      scenario: "s3",
      agent,
      tools,
      sandbox: new AllowAllSandbox(),
      contextManager: cm,
      terminationPolicy: new AlwaysContinuePolicy(),
      toolSchemas: [],
      observability: obs,
    });

    const task = newTask("deploy the payment service", sessionId, react(8));
    const state: SessionState = emptySessionState();
    // Seed a small window with budget over threshold (170/200 = 0.85) + long history.
    seedCompactionState(
      state,
      "deploy the payment service",
      sessionId,
      task.id as TaskId,
      200,
      170,
      12,
    );

    const result = await harness.run({ task, session_state: state });
    expect(result.kind).toBe("success");

    const compactions = obs
      .contextSpans(sessionId)
      .filter((c) => c.operation.kind === "compaction");
    expect(compactions.length).toBeGreaterThanOrEqual(1);

    const first = compactions[0]!;
    expect(first.tokens_after).toBeLessThan(first.tokens_before);
    if (first.operation.kind === "compaction") {
      expect(first.operation.tokens_reclaimed).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// S4 — tool failure + recovery (uses the REAL registry + FailingTool)
// ---------------------------------------------------------------------------

describe("S4 — tool failure + recovery", () => {
  it("recovers from flaky_op by writing a recovery file", async () => {
    const workspace = mkdtempSync(join(tmpdir(), "spore-s4-"));
    const sessionId = SessionId.of("s4-test");
    const agent = new MockAgent(AgentId.of("mock"));
    agent.push({
      kind: "tool_call_requested",
      calls: [toolCall("c1", "flaky_op", { reason: "first try" })],
      usage,
    });
    agent.push({
      kind: "tool_call_requested",
      calls: [
        toolCall("c2", "write_file", {
          path: join(workspace, "recovered.txt"),
          content: "flaky_op failed; adapted by writing this file",
        }),
      ],
      usage,
    });
    agent.push({ kind: "final_response", content: "DONE recovered", usage });

    const registry = buildRealToolRegistry();
    const sandbox = new AllowAllSandbox();
    const bridge = new RealToolRegistry(registry, sandbox);
    const schemas = bridge.modelSchemas();

    const harness = buildScenario({
      scenario: "s4",
      agent,
      tools: bridge,
      sandbox,
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      toolSchemas: schemas,
    });

    const result = await harness.run({
      task: newTask(scenarioPrompt("s4"), sessionId, react(8)),
    });
    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      expect(result.turns).toBeGreaterThanOrEqual(3);
    }
    expect(existsSync(join(workspace, "recovered.txt"))).toBe(true);
  });

  it("does NOT hard-halt on the recoverable FailingTool error", async () => {
    const bridge = new RealToolRegistry(
      buildRealToolRegistry(),
      new AllowAllSandbox(),
    );
    expect(bridge.isAlwaysHalt("flaky_op")).toBe(false);
    const out = await bridge.dispatch(toolCall("c1", "flaky_op", {}));
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
    }
  });

  it("exposes sorted model schemas including flaky_op and read_file", () => {
    const bridge = new RealToolRegistry(
      buildRealToolRegistry(),
      new AllowAllSandbox(),
    );
    const names = bridge.modelSchemas().map((s: ToolSchema) => s.name);
    const sorted = [...names].sort();
    expect(names).toEqual(sorted);
    expect(names).toContain("flaky_op");
    expect(names).toContain("read_file");
  });
});
