/**
 * Unit tests for the Harness (spore-core issue #3).
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs#tests` — same rules,
 * same verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  AgentModelError,
  EmptyResponse,
  HOOK_POINTS,
  MockAgent,
  SessionId,
  StandardHarness,
  TaskId,
  Timeout,
  emptyBudgetSnapshot,
  emptySessionState,
  newTask,
  type Context,
  type ContextManager,
  type HarnessConfig,
  type HumanResponse,
  type LoopStrategy,
  type Message,
  type PausedState,
  type SessionState,
  type Task,
  type TokenUsage,
  type ToolCall,
  type ToolResultRecord,
  type TurnResult,
} from "../src/index.js";

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedMiddleware,
  ScriptedSandbox,
  ScriptedTerminationPolicy,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("test"));
}

function standardConfig(agent: MockAgent): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
  };
}

function task(strategy: LoopStrategy): Task {
  return newTask("do something", SessionId.of("s1"), strategy);
}

function react(max: number): Task {
  return task({ kind: "re_act", max_iterations: max });
}

function usage(): TokenUsage {
  return {
    input_tokens: 1,
    output_tokens: 1,
    cache_read_tokens: null,
    cache_write_tokens: null,
  };
}

function toolCall(id: string, name = "x"): ToolCall {
  return { id, name, input: {} };
}

function tcr(call: ToolCall, u: TokenUsage = usage()): TurnResult {
  return { kind: "tool_call_requested", calls: [call], usage: u };
}

function fr(content: string, u: TokenUsage = usage()): TurnResult {
  return { kind: "final_response", content, usage: u };
}

describe("Harness — ReAct loop", () => {
  it("rule: harness owns the loop — final response on first turn returns Success", async () => {
    const a = makeAgent();
    a.push(fr("done"));
    const h = new StandardHarness(standardConfig(a));
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("done");
      expect(r.turns).toBe(1);
    }
  });

  it("rule: tool calls dispatched, then loop continues to a final response", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c1")));
    a.push(fr("after-tool"));
    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry().push({ kind: "success", content: "tool-ok" });
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("after-tool");
      expect(r.turns).toBe(2);
    }
    expect(reg.callCount).toBe(1);
  });

  it("rule: parallel tool calls all dispatched in one turn", async () => {
    const a = makeAgent();
    a.push({
      kind: "tool_call_requested",
      calls: [toolCall("a"), toolCall("b", "y")],
      usage: usage(),
    });
    a.push(fr("ok"));
    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry()
      .push({ kind: "success", content: "1" })
      .push({ kind: "success", content: "2" });
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);
    await h.run({ task: react(5) });
    expect(reg.callCount).toBe(2);
  });

  it("rule: budget overrun (max_turns) terminates with explicit BudgetExceeded", async () => {
    const a = makeAgent();
    for (let i = 0; i < 10; i++) a.push(tcr(toolCall("c")));
    const h = new StandardHarness(standardConfig(a));
    const t = react(100);
    t.budget.max_turns = 2;
    const r = await h.run({ task: t });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("budget_exceeded");
      if (r.reason.kind === "budget_exceeded") {
        expect(r.reason.limit_type).toBe("turns");
      }
      expect(r.turns).toBe(2);
    }
  });

  it("rule: a turn with neither tool call nor response is an error (AgentError halt)", async () => {
    const a = makeAgent();
    a.push({ kind: "error", error: new EmptyResponse(), usage: usage() });
    const h = new StandardHarness(standardConfig(a));
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") expect(r.reason.kind).toBe("agent_error");
  });

  it("rule: Layer-1 SandboxViolation::PathEscape halts unconditionally", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c", "read")));
    const cfg = standardConfig(a);
    cfg.sandbox = new ScriptedSandbox().push({ kind: "path_escape", path: "/etc/passwd" });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("sandbox_violation");
      if (r.reason.kind === "sandbox_violation") {
        expect(r.reason.violation.kind).toBe("path_escape");
      }
    }
  });

  it("rule: Layer-2 recoverable sandbox violation appended as tool error, loop continues", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c", "read")));
    a.push(fr("ack"));
    const cfg = standardConfig(a);
    cfg.sandbox = new ScriptedSandbox().push({ kind: "path_denied", path: "/p" });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.turns).toBe(2);
  });

  it("rule: TerminationPolicy::Halt overrides final response", async () => {
    const a = makeAgent();
    a.push(fr("done"));
    const cfg = standardConfig(a);
    cfg.terminationPolicy = new ScriptedTerminationPolicy().push({
      kind: "halt",
      reason: "not yet",
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "termination_policy_halt") {
      expect(r.reason.reason).toBe("not yet");
    } else {
      throw new Error(`unexpected: ${JSON.stringify(r)}`);
    }
  });

  it("rule: Middleware::Halt at BeforeTurn halts before model call", async () => {
    const a = makeAgent();
    a.push(fr("unused"));
    const cfg = standardConfig(a);
    cfg.middleware = new ScriptedMiddleware().push("before_turn", {
      kind: "halt",
      reason: "blocked",
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "middleware_halt") {
      expect(r.reason.hook).toBe("before_turn");
      expect(r.reason.reason).toBe("blocked");
      expect(r.turns).toBe(0);
    } else {
      throw new Error("expected middleware_halt");
    }
  });

  it("rule: Middleware::SurfaceToHuman at BeforeTool returns WaitingForHuman with pending calls", async () => {
    const a = makeAgent();
    const calls = [toolCall("c")];
    a.push({ kind: "tool_call_requested", calls, usage: usage() });
    const cfg = standardConfig(a);
    cfg.middleware = new ScriptedMiddleware().push("before_tool", {
      kind: "surface_to_human",
      request: { kind: "tool_approval", calls, risk_level: "medium" },
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind === "waiting_for_human") {
      expect(r.state.pending_tool_calls).toHaveLength(1);
      expect(r.state.child_state).toBeNull();
    }
  });

  it("rule: always_halt tool annotation halts the loop with UnrecoverableToolError", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c", "danger")));
    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry();
    reg.markAlwaysHalt("danger");
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "unrecoverable_tool_error") {
      expect(r.reason.tool).toBe("danger");
    } else {
      throw new Error("expected unrecoverable_tool_error");
    }
  });

  it("rule: unrecoverable tool error halts immediately", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c")));
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "error",
      message: "boom",
      recoverable: false,
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "unrecoverable_tool_error") {
      expect(r.reason.error).toBe("boom");
    } else {
      throw new Error("expected unrecoverable_tool_error");
    }
  });

  it("rule: WaitingForHuman from a tool dispatch propagates to RunResult", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c", "subagent")));
    const cfg = standardConfig(a);
    const childTask = newTask("child", SessionId.of("child"), {
      kind: "re_act",
      max_iterations: 1,
    });
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "waiting_for_human",
      child_state: {
        session_id: SessionId.of("child"),
        task_id: TaskId.of("ct"),
        turn_number: 1,
        session_state: emptySessionState(),
        pending_tool_calls: [],
        approved_results: [],
        human_request: { kind: "clarification", question: "?" },
        task: childTask,
        budget_used: emptyBudgetSnapshot(),
        parent_tool_call_id: "c",
      },
      request: { kind: "clarification", question: "?" },
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind === "waiting_for_human") {
      expect(r.state.child_state).not.toBeNull();
    }
  });

  it("rule: resume() with Halt returns Failure(HumanHalted)", async () => {
    const h = new StandardHarness(standardConfig(makeAgent()));
    const state: PausedState = {
      session_id: SessionId.of("s"),
      task_id: TaskId.of("t"),
      turn_number: 1,
      session_state: emptySessionState(),
      pending_tool_calls: [],
      approved_results: [],
      human_request: { kind: "clarification", question: "?" },
      task: react(5),
      budget_used: emptyBudgetSnapshot(),
      child_state: null,
    };
    const r = await h.resume(state, { kind: "halt" });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") expect(r.reason.kind).toBe("human_halted");
  });

  it("rule: resume() with Allow dispatches pending tool calls then continues loop", async () => {
    const a = makeAgent();
    a.push(fr("done"));
    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry().push({ kind: "success", content: "tool-ok" });
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);
    const state: PausedState = {
      session_id: SessionId.of("s"),
      task_id: TaskId.of("t"),
      turn_number: 1,
      session_state: emptySessionState(),
      pending_tool_calls: [toolCall("c")],
      approved_results: [],
      human_request: { kind: "tool_approval", calls: [], risk_level: "low" },
      task: react(5),
      budget_used: emptyBudgetSnapshot(),
      child_state: null,
    };
    const r = await h.resume(state, { kind: "allow" });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("done");
    expect(reg.callCount).toBe(1);
  });

  it("rule: non-ReAct strategies marked StrategyNotYetImplemented", async () => {
    // Q4 (issue #70): plan_execute no longer uses strategy_not_yet_implemented;
    // it produces an artifact then halts with execute_phase_not_implemented
    // (covered by the PlanExecute plan-phase tests below).
    const h = new StandardHarness(standardConfig(makeAgent()));
    const strategies: LoopStrategy[] = [
      { kind: "ralph" },
      { kind: "self_verifying" },
      {
        kind: "hill_climbing",
        direction: "maximize",
        max_stagnation: null,
        revert_on_no_improvement: false,
        min_improvement_delta: null,
      },
    ];
    for (const s of strategies) {
      const r = await h.run({ task: task(s) });
      expect(r.kind).toBe("failure");
      if (r.kind === "failure") expect(r.reason.kind).toBe("strategy_not_yet_implemented");
    }
  });

  it("rule: aggregate usage accumulates across turns", async () => {
    const a = makeAgent();
    a.push({
      kind: "tool_call_requested",
      calls: [toolCall("c")],
      usage: {
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: null,
        cache_write_tokens: null,
      },
    });
    a.push({
      kind: "final_response",
      content: "ok",
      usage: {
        input_tokens: 7,
        output_tokens: 3,
        cache_read_tokens: null,
        cache_write_tokens: null,
      },
    });
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "success", content: "x" });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.usage.input_tokens).toBe(17);
      expect(r.usage.output_tokens).toBe(8);
    }
  });

  it("rule: ModelError surfaces as AgentError → HaltReason::AgentError", async () => {
    const a = makeAgent();
    a.push({
      kind: "error",
      error: new AgentModelError(new Timeout()),
      usage: null,
    });
    const h = new StandardHarness(standardConfig(a));
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "agent_error") {
      expect(r.reason.error.kind).toBe("model_error");
    } else {
      throw new Error("expected agent_error");
    }
  });

  it("rule: budget max_input_tokens enforced", async () => {
    const a = makeAgent();
    a.push({
      kind: "tool_call_requested",
      calls: [toolCall("c")],
      usage: {
        input_tokens: 100,
        output_tokens: 1,
        cache_read_tokens: null,
        cache_write_tokens: null,
      },
    });
    a.push(fr("never reached"));
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "success", content: "ok" });
    const h = new StandardHarness(cfg);
    const t = react(10);
    t.budget.max_input_tokens = 10;
    const r = await h.run({ task: t });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "budget_exceeded") {
      expect(r.reason.limit_type).toBe("input_tokens");
    } else {
      throw new Error("expected budget_exceeded input_tokens");
    }
  });

  it("stream sink receives turn_start/turn_end/final_response", async () => {
    const a = makeAgent();
    a.push(fr("done"));
    const h = new StandardHarness(standardConfig(a));
    const events: string[] = [];
    await h.run({
      task: react(5),
      on_stream: (e) => events.push(e.kind),
    });
    expect(events).toContain("turn_start");
    expect(events).toContain("turn_end");
    expect(events).toContain("final_response");
  });

  it("HOOK_POINTS lists all four hook locations", () => {
    expect(HOOK_POINTS).toEqual(["before_turn", "before_tool", "after_tool", "before_completion"]);
  });

  // ── Assistant-turn recording (regression for lost conversation history) ──

  /**
   * A ContextManager that records every message it appends (assistant turns and
   * tool results) in order, so tests can assert the conversation the loop
   * builds. Tool results are recorded as a `tool_result` content block so the
   * `call_id` is preserved for ordering assertions — this is a test-only shape
   * and does NOT change the production tool-result shape (the standard adapter
   * keeps its `role:"tool"` text content).
   */
  class RecordingContextManager implements ContextManager {
    readonly recorded: Message[] = [];

    async assemble(session: SessionState, _task: Task): Promise<Context> {
      return { messages: session.messages.slice(), tools: [], params: { stop_sequences: [] } };
    }

    async appendToolResult(session: SessionState, result: ToolResultRecord): Promise<void> {
      const content =
        result.output.kind === "success"
          ? result.output.content
          : result.output.kind === "error"
            ? result.output.message
            : "";
      const msg: Message = {
        role: "tool",
        content: {
          type: "tool_result",
          tool_use_id: result.call_id,
          content,
          is_error: result.output.kind === "error",
        },
      };
      session.messages.push(msg);
      this.recorded.push(msg);
    }

    async appendUserMessage(session: SessionState, text: string): Promise<void> {
      const msg: Message = { role: "user", content: { type: "text", text } };
      session.messages.push(msg);
      this.recorded.push(msg);
    }

    async appendAssistantMessage(session: SessionState, message: Message): Promise<void> {
      session.messages.push(message);
      this.recorded.push(message);
    }

    shouldCompact(_session: SessionState): boolean {
      return false;
    }
  }

  it("regression: tool_call records an assistant tool-call message BEFORE its tool result", async () => {
    const a = makeAgent();
    a.push(tcr(toolCall("c1", "read_file")));
    a.push(fr("done"));
    const cfg = standardConfig(a);
    const cm = new RecordingContextManager();
    cfg.contextManager = cm;
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "success", content: "contents" });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");

    const assistantIdx = cm.recorded.findIndex(
      (m) => m.role === "assistant" && m.content.type === "tool_call" && m.content.id === "c1",
    );
    const toolIdx = cm.recorded.findIndex(
      (m) =>
        m.role === "tool" && m.content.type === "tool_result" && m.content.tool_use_id === "c1",
    );
    expect(assistantIdx, "assistant tool-call message must be recorded").toBeGreaterThanOrEqual(0);
    expect(toolIdx, "tool result must be recorded").toBeGreaterThanOrEqual(0);
    expect(assistantIdx).toBeLessThan(toolIdx);
  });

  it("regression: final_response records the assistant's final text in history", async () => {
    const a = makeAgent();
    a.push(fr("the final answer"));
    const cfg = standardConfig(a);
    const cm = new RecordingContextManager();
    cfg.contextManager = cm;
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");

    const hasFinalText = cm.recorded.some(
      (m) =>
        m.role === "assistant" &&
        m.content.type === "text" &&
        m.content.text === "the final answer",
    );
    expect(hasFinalText).toBe(true);
  });

  it("regression: resume after SurfaceToHuman records the assistant tool-call exactly once, before its result", async () => {
    const a = makeAgent();
    const calls = [toolCall("c1", "read_file")];
    a.push({ kind: "tool_call_requested", calls, usage: usage() });
    a.push(fr("done"));
    const cfg = standardConfig(a);
    const cm = new RecordingContextManager();
    cfg.contextManager = cm;
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "success", content: "contents" });
    cfg.middleware = new ScriptedMiddleware().push("before_tool", {
      kind: "surface_to_human",
      request: { kind: "tool_approval", calls, risk_level: "medium" },
    });
    const h = new StandardHarness(cfg);

    // Pause at BeforeTool — the assistant turn was recorded just before.
    const paused = await h.run({ task: react(5) });
    expect(paused.kind).toBe("waiting_for_human");
    if (paused.kind !== "waiting_for_human") throw new Error("expected waiting_for_human");

    // Resume with approval; the pending call is dispatched and its result
    // appended after the already-recorded assistant turn.
    const resumeResponse: HumanResponse = { kind: "allow" };
    const resumed = await h.resume(paused.state, resumeResponse);
    expect(resumed.kind).toBe("success");

    const assistantHits = cm.recorded.filter(
      (m) => m.role === "assistant" && m.content.type === "tool_call" && m.content.id === "c1",
    );
    expect(assistantHits.length, "assistant tool-call must be recorded exactly once").toBe(1);

    const assistantIdx = cm.recorded.findIndex(
      (m) => m.role === "assistant" && m.content.type === "tool_call" && m.content.id === "c1",
    );
    const toolIdx = cm.recorded.findIndex(
      (m) =>
        m.role === "tool" && m.content.type === "tool_result" && m.content.tool_use_id === "c1",
    );
    expect(toolIdx, "tool result must be recorded").toBeGreaterThanOrEqual(0);
    expect(assistantIdx).toBeLessThan(toolIdx);
  });
});
