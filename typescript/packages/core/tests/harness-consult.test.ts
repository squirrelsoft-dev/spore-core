/**
 * Unit tests for the mid-loop consult primitive — worker / harness side
 * (spore-core issue #114).
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs#tests` (the consult block) —
 * same rules, same verdicts, parallel structure. A worker-side tool returning
 * `ToolOutput.consult` pauses the loop and returns `RunResult.consult`; the
 * consult call is preserved as the head of `pending_tool_calls` with
 * `human_request` absent, and is NOT appended to history until resumed.
 *
 * Rules covered here: R1 (worker pauses → RunResult.consult, head call,
 * human_request null), R6 (empty handler map standalone → consult surfaces
 * unchanged), R9 (existing callers unaffected — exercised by the broad suite +
 * the empty-map default), R10 (consult not in history until resumed; resume
 * injects the answer as the tool result). R2–R5/R7 are covered in
 * `@spore/tools` subagent tests; R8 in the fixture-replay test.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  MockAgent,
  SessionId,
  StandardHarness,
  TaskId,
  emptyBudgetSnapshot,
  emptySessionState,
  newTask,
  observability,
  toolOutput,
  type ConsultRequest,
  type HarnessConfig,
  type LoopStrategy,
  type PausedState,
  type Task,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("test"));
}

function standardConfig(agent: MockAgent): HarnessConfig {
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
  };
}

function react(max: number): Task {
  const strategy: LoopStrategy = {
    kind: "react",
    budget: { kind: "per_loop", value: max },
    agent: "",
    toolset: "",
  };
  return newTask("audit the auth module", SessionId.of("s1"), strategy);
}

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

/** A turn that requests `n` tool calls (ids c0..c{n-1}), followed by a
 *  final_response so a resumed loop runs to success. */
function agentWithToolCalls(n: number): MockAgent {
  const a = makeAgent();
  const calls: ToolCall[] = [];
  for (let i = 0; i < n; i += 1) {
    calls.push({ id: `c${i}`, name: `t${i}`, input: {} });
  }
  a.push({ kind: "tool_call_requested", calls, usage: usage() } as TurnResult);
  a.push({ kind: "final_response", content: "resumed-done", usage: usage() } as TurnResult);
  return a;
}

function consultReq(): ConsultRequest {
  return {
    kind: "advice",
    situation: "stuck on auth",
    attempts: 3,
    question: "what next?",
  };
}

// R1 + R10: a worker-side tool returning ToolOutput.consult pauses the loop and
// returns RunResult.consult. The consult call is the head of pending_tool_calls,
// human_request is absent, there is no child_state, and the consult is NOT
// appended to message history (R10).
describe("Harness consult — R1 + R10: pauses and returns RunResult.consult", () => {
  it("pauses with the consult call as head, human_request absent, no history append", async () => {
    const a = agentWithToolCalls(2);
    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry().push(toolOutput.consult(consultReq()));
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });

    expect(r.kind).toBe("consult");
    if (r.kind === "consult") {
      expect(r.request.kind).toBe("advice");
      expect(r.request.question).toBe("what next?");
      // R1: human_request absent (null on the wire), no child_state.
      expect(r.state.human_request == null).toBe(true);
      expect(r.state.child_state).toBeNull();
      // R1: consulting call (c0) is head; c1 trails.
      expect(r.state.pending_tool_calls.length).toBe(2);
      expect(r.state.pending_tool_calls[0]!.id).toBe("c0");
      expect(r.state.pending_tool_calls[1]!.id).toBe("c1");
      // R10: no tool-result turn recorded yet.
      const toolTurns = r.state.session_state.messages.filter((m) => m.role === "tool");
      expect(toolTurns.length).toBe(0);
    }
    // Exactly one dispatch (the consulting call); c1 preserved, not run.
    expect(reg.callCount).toBe(1);
  });
});

// R6 (graceful degradation): an absent / empty consultHandlers map means a
// standalone worker simply surfaces RunResult.consult to its caller unchanged
// (existing callers unaffected — R9).
describe("Harness consult — R6/R9: empty handler map surfaces consult unchanged", () => {
  it("a standalone worker returns RunResult.consult with no handlers", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    expect(cfg.consultHandlers == null).toBe(true);
    cfg.toolRegistry = new ScriptedToolRegistry().push(toolOutput.consult(consultReq()));
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("consult");
  });
});

// R10 + resume seam: resumeConsult with an answer injects it as the TOOL RESULT
// for the head pending (consult) call, then continues to success.
describe("Harness consult — resume seam injects answer as tool result", () => {
  it("resumeConsult(answer) injects the answer and runs to success", async () => {
    const a = makeAgent();
    a.push({
      kind: "final_response",
      content: "consult-resumed-done",
      usage: usage(),
    } as TurnResult);
    const cfg = standardConfig(a);
    const h = new StandardHarness(cfg);
    const state: PausedState = {
      session_id: SessionId.of("s"),
      task_id: TaskId.of("t"),
      turn_number: 1,
      session_state: emptySessionState(),
      pending_tool_calls: [{ id: "consult", name: "ask_advice", input: { kind: "advice" } }],
      approved_results: [],
      task: react(5),
      budget_used: emptyBudgetSnapshot(),
      child_state: null,
      toolset: "",
    };
    const r = await h.resumeConsult(state, { kind: "answer", text: "the answer" });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("consult-resumed-done");
    }
    // R10: the injected answer is recorded as exactly one tool-result message.
    const toolTurns = state.session_state.messages.filter((m) => m.role === "tool");
    expect(toolTurns.length).toBe(1);
    expect(toolTurns[0]!.content).toEqual({ type: "text", text: "the answer" });
  });

  it("resumeConsult(budget_exhausted) injects the message and runs to success", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: "finished", usage: usage() } as TurnResult);
    const cfg = standardConfig(a);
    const h = new StandardHarness(cfg);
    const state: PausedState = {
      session_id: SessionId.of("s"),
      task_id: TaskId.of("t"),
      turn_number: 1,
      session_state: emptySessionState(),
      pending_tool_calls: [{ id: "consult", name: "ask_advice", input: { kind: "advice" } }],
      approved_results: [],
      task: react(5),
      budget_used: emptyBudgetSnapshot(),
      child_state: null,
      toolset: "",
    };
    const r = await h.resumeConsult(state, {
      kind: "budget_exhausted",
      message: "budget gone",
    });
    expect(r.kind).toBe("success");
    const toolTurns = state.session_state.messages.filter((m) => m.role === "tool");
    expect(toolTurns[0]!.content).toEqual({ type: "text", text: "budget gone" });
  });
});

// Observability: a consult pause emits a `consult_spawned` context event, and
// resume emits `consult_resumed` (answered=true for an answer).
describe("Harness consult — observability events", () => {
  it("emits consult_spawned on pause and consult_resumed on resume", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    const obs = new observability.InMemoryObservabilityProvider();
    cfg.observability = obs;
    cfg.toolRegistry = new ScriptedToolRegistry().push(toolOutput.consult(consultReq()));
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("consult");
    if (r.kind !== "consult") return;

    const spawnSpans = obs
      .contextSpans(r.state.session_id)
      .filter((s) => s.operation.kind === "consult_spawned");
    expect(spawnSpans.length).toBe(1);
    expect(
      spawnSpans[0]!.operation.kind === "consult_spawned" && spawnSpans[0]!.operation.consult_kind,
    ).toBe("advice");

    const resumeAgent = makeAgent();
    resumeAgent.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
    const resumeCfg = standardConfig(resumeAgent);
    resumeCfg.observability = obs;
    const resumeHarness = new StandardHarness(resumeCfg);
    await resumeHarness.resumeConsult(r.state, { kind: "answer", text: "x" });
    const resumeSpans = obs
      .contextSpans(r.state.session_id)
      .filter((s) => s.operation.kind === "consult_resumed");
    expect(resumeSpans.length).toBe(1);
    expect(
      resumeSpans[0]!.operation.kind === "consult_resumed" && resumeSpans[0]!.operation.answered,
    ).toBe(true);
  });
});
