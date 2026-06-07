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
  type Agent,
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

function task(strategy: LoopStrategy): Task {
  return newTask("do something", SessionId.of("s1"), strategy);
}

function react(max: number): Task {
  return task({ kind: "react", budget: { kind: "per_loop", value: max }, agent: "", toolset: "" });
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
      kind: "react",
      budget: { kind: "per_loop", value: 1 },
      agent: "",
      toolset: "",
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

  it("rule: every loop strategy is implemented — none returns StrategyNotYetImplemented", async () => {
    // plan_execute (#59), self_verifying (#61), ralph (#58), and hill_climbing
    // (#60) all run their full loops now, covered by their own test suites.
    // #124: with no metricEvaluator wired (and no inner output schema) the
    // single resolution path rejects the strategy at startup with a typed
    // configuration_error — NEVER strategy_not_yet_implemented.
    const h = new StandardHarness(standardConfig(makeAgent()));
    const strategies: LoopStrategy[] = [
      {
        kind: "hill_climbing",
        inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
        direction: "maximize",
        max_stagnation: 0,
        revert_on_no_improvement: false,
        min_improvement_delta: 0,
        evaluator: "",
      },
    ];
    for (const s of strategies) {
      const r = await h.run({ task: task(s) });
      expect(r.kind).toBe("failure");
      if (r.kind === "failure") {
        expect(r.reason.kind).not.toBe("strategy_not_yet_implemented");
        expect(r.reason.kind).toBe("configuration_error");
      }
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

// ── #93: modelParams reach every tool-requesting turn ──────────────────────
//
// `RecordingTurnAgent.turn` captures every `Context` it sees in `seen`, and the
// agent copies `Context.params` verbatim into the `ModelRequest` (see
// `ModelAgent.turn` / `intoRequest`). So asserting on a captured context's
// `params.structured_tool_calls` proves the configured params reached the
// request the model would have seen.
//
// Mirrors `rust/crates/spore-core/src/harness.rs#tests` (#93).

/** A context-capturing agent: records every `Context` it sees and pops the next
 *  scripted result, so we can assert which params reached each turn. */
class RecordingTurnAgent implements Agent {
  readonly seen: Context[] = [];
  private readonly results: TurnResult[] = [];
  constructor(private readonly agentId: AgentId) {}
  push(r: TurnResult): this {
    this.results.push(r);
    return this;
  }
  id(): AgentId {
    return this.agentId;
  }
  async turn(ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.seen.push(ctx);
    const next = this.results.shift();
    if (next == null) return { kind: "error", error: new EmptyResponse(), usage: null };
    return next;
  }
}

/** Build a config whose agent is a context-capturing `RecordingTurnAgent`, with
 *  the given (optionally non-default) model params. */
function recordingConfig(agent: RecordingTurnAgent, modelParams: HarnessConfig["modelParams"]) {
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams,
  } satisfies HarnessConfig;
}

const PLAN_EXECUTE_STRATEGY: LoopStrategy = {
  kind: "plan_execute",
  // #124: under genuine recursion the per-task turn cap is allocated by the
  // PlanExecute combinator and passed via the step task's budget; the execute
  // child's own `per_loop` budget must not gate below that (an absolute cap that
  // already includes the carried plan turn). MAX lets the combinator's allocation
  // be the effective gate — matching the Rust reference's `per_loop(u32::MAX)`.
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
};

function planTask93(): Task {
  return newTask("build a CLI", SessionId.of("p93"), PLAN_EXECUTE_STRATEGY);
}

describe("Harness — modelParams threading (#93)", () => {
  // No `.modelParams(...)` ⇒ each turn's context carries the default
  // (structured_tool_calls absent ⇒ false).
  it("default model params reach the request as default (structured false)", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec")).push(fr("done"));
    const h = new StandardHarness(recordingConfig(agent, { stop_sequences: [] }));
    await h.run({ task: react(5) });
    expect(agent.seen.length).toBeGreaterThan(0);
    expect(agent.seen[0].params.structured_tool_calls).not.toBe(true);
  });

  // (a) ReAct: the ReAct turn context carries the flag.
  it("model params reach the ReAct turn", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec")).push(fr("done"));
    const h = new StandardHarness(
      recordingConfig(agent, { stop_sequences: [], structured_tool_calls: true }),
    );
    await h.run({ task: react(5) });
    expect(agent.seen.length).toBeGreaterThan(0);
    expect(agent.seen[0].params.structured_tool_calls).toBe(true);
  });

  // (b) Plan phase: the plan-turn context carries the flag.
  it("model params reach the plan phase", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec")).push(
      fr('{"tasks":["one"],"rationale":"r"}'),
    );
    const h = new StandardHarness(
      recordingConfig(agent, { stop_sequences: [], structured_tool_calls: true }),
    );
    await h.run({ task: planTask93() });
    // First captured context is the plan turn.
    expect(agent.seen.length).toBeGreaterThan(0);
    expect(agent.seen[0].params.structured_tool_calls).toBe(true);
  });

  // (c) Execute sub-loops: a full PlanExecute run threads params through the
  // shared react seam used by the execute sub-loop — every captured context
  // (plan + execute steps) carries the flag.
  it("model params reach the execute sub-loops", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec"))
      .push(fr('{"tasks":["one","two"],"rationale":"r"}'))
      .push(fr("did one"))
      .push(fr("did two"));
    const h = new StandardHarness(
      recordingConfig(agent, { stop_sequences: [], structured_tool_calls: true }),
    );
    await h.run({ task: planTask93() });
    // 1 plan turn + 2 execute turns; every captured context carries it.
    expect(agent.seen.length).toBe(3);
    expect(agent.seen.every((c) => c.params.structured_tool_calls === true)).toBe(true);
  });

  // (d) Streaming path: it flows through `runReactInner`'s same seam — the
  // streamed turn's captured context carries the flag.
  it("model params reach the streaming path", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec")).push(fr("done"));
    const h = new StandardHarness(
      recordingConfig(agent, { stop_sequences: [], structured_tool_calls: true }),
    );
    await h.run({ task: react(5), on_stream: () => {} });
    expect(agent.seen.length).toBeGreaterThan(0);
    expect(agent.seen[0].params.structured_tool_calls).toBe(true);
  });

  // Concatenate the text of a captured context's user/tool messages so we can
  // assert which seeded directives/instructions reached a given turn.
  function ctxText(ctx: Context): string {
    return ctx.messages.map((m) => (m.content.type === "text" ? m.content.text : "")).join("\n");
  }

  // #93 regression: the plan-phase directive ("Produce a step-by-step plan…
  // Respond with a single JSON object…") must NOT leak into the SHARED
  // sessionState, otherwise every execute step re-sees it and an
  // instruction-following model re-emits a plan instead of calling tools.
  // Drive a full 2-step PlanExecute run with the context-capturing agent and
  // assert no execute-step context carries the directive, while the step
  // instructions DO reach those contexts (and a step can issue a tool call).
  it("plan directive does not leak into execute context", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec"))
      // Plan turn: produce a 2-step plan.
      .push(fr('{"tasks":["step one","step two"],"rationale":"r"}'))
      // Execute step 1: issue a tool call, then finalize.
      .push(tcr(toolCall("c1", "noop")))
      .push(fr("did step one"))
      // Execute step 2: finalize directly.
      .push(fr("did step two"));
    const h = new StandardHarness(recordingConfig(agent, { stop_sequences: [] }));
    await h.run({ task: planTask93() });

    // 1 plan turn + (tool call + final) for step one + 1 for step two.
    expect(agent.seen.length).toBe(4);

    // The PLAN turn (index 0) DOES carry the directive — that's correct.
    expect(ctxText(agent.seen[0]!)).toContain("Produce a step-by-step plan");
    expect(ctxText(agent.seen[0]!)).toContain("Respond with a single JSON object");

    // No EXECUTE-step context (indices 1..) may carry the directive.
    for (let i = 1; i < agent.seen.length; i++) {
      const text = ctxText(agent.seen[i]!);
      expect(text).not.toContain("Produce a step-by-step plan");
      expect(text).not.toContain("Respond with a single JSON object");
    }

    // The execute steps still receive their step instructions and can proceed
    // to a tool call. Step one's second turn (index 2) follows the dispatched
    // tool call: it still carries the step-one instruction plus the tool result
    // accumulated within the step's sub-loop.
    expect(ctxText(agent.seen[1]!)).toContain("step one");
    expect(ctxText(agent.seen[2]!)).toContain("step one");
    expect(ctxText(agent.seen[2]!)).toContain("ok");
    expect(ctxText(agent.seen[3]!)).toContain("step two");
  });

  // #93 regression: the execute phase maintains ONE accumulating context across
  // steps. After a successful step its conversation (instruction + tool calls +
  // TOOL RESULTS + assistant output) is folded back into the shared sessionState,
  // so the NEXT step's sub-loop sees prior steps' RESULTS — not just their
  // instructions. Drive a 2-step run where STEP 1 issues a tool call returning a
  // distinctive string and assert STEP 2's assembled context carries it.
  //
  // Mirrors `rust/crates/spore-core/src/harness.rs#execute_steps_accumulate_prior_results`.
  it("execute steps accumulate prior results", async () => {
    const agent = new RecordingTurnAgent(AgentId.of("rec"))
      // Plan turn: a 2-step plan (research -> summarize).
      .push(fr('{"tasks":["research tokio","summarize findings"],"rationale":"r"}'))
      // Step 1: call a tool, then finalize using its result.
      .push(tcr(toolCall("c1", "lookup")))
      .push(fr("researched"))
      // Step 2: finalize directly (it must SEE step 1's tool result).
      .push(fr("summarized"));
    const cfg = recordingConfig(agent, { stop_sequences: [] });
    // Step 1's tool call returns a distinctive result string.
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "success",
      content: "TOKIO_FACTS_123",
    });
    const h = new StandardHarness(cfg);
    const result = await h.run({ task: planTask93() });
    expect(result.kind).toBe("success");
    if (result.kind === "success") {
      expect(result.output).toBe("summarized");
    }

    // 1 plan turn + (tool call + final) for step 1 + 1 for step 2 = 4.
    expect(agent.seen.length).toBe(4);

    // Step 1's SECOND turn (index 2) sees the tool result — sanity check that the
    // result string is on the wire at all.
    expect(ctxText(agent.seen[2]!)).toContain("TOKIO_FACTS_123");

    // The accumulation guarantee: STEP 2's context (index 3) CONTAINS step 1's
    // tool result, proving the execute loop carried it forward.
    expect(ctxText(agent.seen[3]!)).toContain("TOKIO_FACTS_123");

    // Step 2 also sees step 1's prior instruction. (This harness's
    // ContextManager seam folds instructions + tool results into the shared
    // session but does not append the per-step final assistant text — that is
    // surfaced only as the step's `output` — so the accumulation guarantee is
    // proven by the carried instruction + tool result above.)
    expect(ctxText(agent.seen[3]!)).toContain("research tokio");
  });
});
