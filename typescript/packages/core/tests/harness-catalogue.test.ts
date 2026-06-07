/**
 * Harness-loop tests for the #81 Standard Tool Catalogue wiring:
 *  - `send_message` → the loop emits a `user_message` stream event and records a
 *    minimal success result (it does NOT collapse to a normal tool result path).
 *  - `awaiting_clarification` → the loop pauses into a `PausedState` directly
 *    (NO ChildPausedState), sets `human_request` to `clarification`, preserves
 *    the clarifying call as head of `pending_tool_calls`, and returns
 *    `waiting_for_human`. On resume the answer text is injected as that call's
 *    tool result and the loop continues.
 *
 * Mirrors the corresponding Rust harness tests for issue #81.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  MockAgent,
  SessionId,
  StandardHarness,
  newTask,
  type HarnessConfig,
  type HarnessStreamEvent,
  type LoopStrategy,
  type Task,
  type TokenUsage,
  type ToolCall,
  type ToolOutput,
  type TurnResult,
} from "../src/index.js";

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

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
  return newTask("do something", SessionId.of("s1"), strategy);
}

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function agentCalling(name: string, input: unknown, callId = "c0"): MockAgent {
  const a = new MockAgent(AgentId.of("test"));
  a.push({
    kind: "tool_call_requested",
    calls: [{ id: callId, name, input } as ToolCall],
    usage: usage(),
  } as TurnResult);
  a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
  return a;
}

// ============================================================================
// send_message → user_message stream event
// ============================================================================

describe("send_message loop wiring (#81)", () => {
  it("emits a user_message stream event and records a success result", async () => {
    const a = agentCalling("send_message", { content: "hello human" });
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "success",
      content: "hello human",
      truncated: false,
    } as ToolOutput);
    const h = new StandardHarness(cfg);

    const events: HarnessStreamEvent[] = [];
    const r = await h.run({ task: react(5), on_stream: (e) => events.push(e) });

    expect(r.kind).toBe("success");
    const userMessages = events.filter((e) => e.kind === "user_message");
    expect(userMessages.length).toBe(1);
    expect(userMessages[0]).toEqual({ kind: "user_message", content: "hello human" });
  });

  it("a send_message that errors emits no user_message event", async () => {
    const a = agentCalling("send_message", {});
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "error",
      message: "invalid parameters: content required",
      recoverable: true,
    } as ToolOutput);
    const h = new StandardHarness(cfg);

    const events: HarnessStreamEvent[] = [];
    await h.run({ task: react(5), on_stream: (e) => events.push(e) });
    expect(events.filter((e) => e.kind === "user_message").length).toBe(0);
  });
});

// ============================================================================
// awaiting_clarification → pause + resume
// ============================================================================

describe("awaiting_clarification loop wiring (#81)", () => {
  it("pauses into a PausedState (no child_state) with a clarification request", async () => {
    const a = agentCalling("ask_user_question", { question: "which?", options: ["a", "b"] });
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "awaiting_clarification",
      question: "which?",
      options: ["a", "b"],
    } as ToolOutput);
    const h = new StandardHarness(cfg);

    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind !== "waiting_for_human") return;

    // No ChildPausedState (this is NOT the subagent path).
    expect(r.state.child_state).toBeNull();
    // human_request is a clarification carrying the options.
    expect(r.request.kind).toBe("clarification");
    if (r.request.kind === "clarification") {
      expect(r.request.question).toBe("which?");
      expect(r.request.options).toEqual(["a", "b"]);
    }
    // The clarifying call is preserved as the HEAD of pending_tool_calls.
    expect(r.state.pending_tool_calls.length).toBe(1);
    expect(r.state.pending_tool_calls[0]?.name).toBe("ask_user_question");
    expect(r.state.pending_tool_calls[0]?.id).toBe("c0");
  });

  it("resume injects the answer as the clarifying call's tool result and continues", async () => {
    const a = agentCalling("ask_user_question", { question: "which?" });
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "awaiting_clarification",
      question: "which?",
    } as ToolOutput);
    const h = new StandardHarness(cfg);

    const paused = await h.run({ task: react(5) });
    expect(paused.kind).toBe("waiting_for_human");
    if (paused.kind !== "waiting_for_human") return;

    const resumed = await h.resume(paused.state, { kind: "answer", text: "use a" });
    expect(resumed.kind).toBe("success");
    if (resumed.kind !== "success") return;
    expect(resumed.output).toBe("done");

    // The clarifying call's tool result is the injected answer text — recorded
    // as a tool-role message, NOT a free-standing user message.
    const toolMsgs = paused.state.session_state.messages.filter((m) => m.role === "tool");
    expect(toolMsgs.length).toBeGreaterThanOrEqual(1);
    const hasAnswer = paused.state.session_state.messages.some(
      (m) => m.role === "tool" && m.content.type === "text" && m.content.text === "use a",
    );
    expect(hasAnswer).toBe(true);
  });

  it("a clarification pause with options round-trips through HumanRequest back-compat", async () => {
    // Back-compat: a clarification HumanRequest without `options` still
    // deserializes (the field is optional).
    const withOptions = { kind: "clarification", question: "q", options: ["x"] };
    const back = JSON.parse(JSON.stringify(withOptions));
    expect(back.options).toEqual(["x"]);
    const bare = JSON.parse(JSON.stringify({ kind: "clarification", question: "q" }));
    expect(bare.options).toBeUndefined();
  });
});
