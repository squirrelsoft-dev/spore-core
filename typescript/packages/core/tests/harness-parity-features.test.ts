/**
 * Harness-loop feature tests for the Rust-parity additions:
 *  - G-C: `tool_result.revised` (AfterTool `continue_with_modification`).
 *  - G-D: steering source (mid-run user-message injection + `steered: true`).
 *  - G-E: tool-call repair (coerce + re-dispatch on recoverable error).
 *  - G-F: behavioral tool-call nudge (zero-tools-acted → nudge, bounded).
 *  - G-J: narration refund (send_message-only turns don't consume the budget).
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessBuilder,
  MockAgent,
  SessionId,
  StandardHarness,
  newTask,
  toolCallRepair,
  toolRegistry,
  wrapNodeSink,
  type HarnessConfig,
  type HarnessStreamEvent,
  type LoopStrategy,
  type Task,
  type TokenUsage,
  type ToolCall,
  type ToolOutput,
  type ToolSchema,
  type TurnResult,
} from "../src/index.js";

const { StandardToolCallRepair, coerceToolArgs } = toolCallRepair;

import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedMiddleware,
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

/** A registry that records every dispatched call and returns scripted outputs. */
class RecordingRegistry extends ScriptedToolRegistry {
  readonly dispatched: ToolCall[] = [];
  private readonly outputsByCall: ToolOutput[] = [];
  private i = 0;

  pushOutput(o: ToolOutput): this {
    this.outputsByCall.push(o);
    return this;
  }

  override async dispatch(call: ToolCall): Promise<ToolOutput> {
    this.dispatched.push(call);
    const next = this.outputsByCall[this.i];
    this.i += 1;
    return next ?? { kind: "success", content: "ok" };
  }
}

// ============================================================================
// G-A: NodeAttr stream-event attribution
// ============================================================================

describe("G-A NodeAttr stream-event attribution", () => {
  it("a bare react leaf stamps events with kind 'react'", async () => {
    const a = new MockAgent(AgentId.of("leaf-agent"));
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
    const cfg = standardConfig(a);
    const h = new StandardHarness(cfg);
    const events: HarnessStreamEvent[] = [];
    await h.run({
      task: newTask("hi", SessionId.of("s1"), {
        kind: "react",
        budget: { kind: "per_loop", value: 8 },
        agent: "",
        toolset: "",
      }),
      on_stream: (e) => events.push(e),
    });
    const turnStart = events.find((e) => e.kind === "turn_start");
    expect(turnStart).toBeDefined();
    expect(turnStart!.node).toEqual({
      kind: "react",
      agent_id: undefined,
      depth: 0,
      path: ["react"],
    });
  });

  it("wrapNodeSink: an enclosing combinator prepends its kind and bumps depth", () => {
    const seen: HarnessStreamEvent[] = [];
    // Outer combinator sink wraps the real sink.
    const combinatorSink = wrapNodeSink((e) => seen.push(e), "plan_execute");
    // Leaf wraps the combinator sink.
    const leafSink = wrapNodeSink(combinatorSink!, "react", "worker");
    leafSink!({ kind: "turn_start", turn: 1 });
    expect(seen[0]!.node).toEqual({
      kind: "react",
      agent_id: "worker",
      depth: 1,
      path: ["plan_execute", "react"],
    });
  });
});

// ============================================================================
// G-C: tool_result.revised
// ============================================================================

describe("G-C tool_result.revised (AfterTool continue_with_modification)", () => {
  it("emits a revised tool_result event when AfterTool rewrites a result", async () => {
    const a = new MockAgent(AgentId.of("test"));
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c0", name: "write_file", input: { path: "x", content: "hi" } } as ToolCall],
      usage: usage(),
    } as TurnResult);
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);

    const tr = new ScriptedToolRegistry();
    tr.push({ kind: "success", content: "wrote x", truncated: false });
    const cfg = standardConfig(a);
    cfg.toolRegistry = tr;
    const mw = new ScriptedMiddleware();
    // The first tool result is a success; AfterTool rewrites it to a recoverable
    // error (continue_with_modification).
    mw.push("after_tool", { kind: "continue_with_modification" });
    cfg.middleware = mw;

    const h = new StandardHarness(cfg);
    const events: HarnessStreamEvent[] = [];
    const r = await h.run({ task: react(5), on_stream: (e) => events.push(e) });
    expect(r.kind).toBe("success");

    const toolResults = events.filter((e) => e.kind === "tool_result") as Extract<
      HarnessStreamEvent,
      { kind: "tool_result" }
    >[];
    // The first tool_result is the original success (revised absent/false).
    // The second is the revised one with revised: true.
    expect(toolResults.length).toBe(2);
    expect(toolResults[0]!.revised ?? false).toBe(false);
    expect(toolResults[1]!.revised).toBe(true);
  });
});

// ============================================================================
// G-D: steering source
// ============================================================================

describe("G-D steering source", () => {
  it("injects queued user messages at the turn boundary with steered: true", async () => {
    const a = new MockAgent(AgentId.of("test"));
    a.push({ kind: "final_response", content: "first", usage: usage() } as TurnResult);
    a.push({ kind: "final_response", content: "after steer", usage: usage() } as TurnResult);

    const queue: string[] = ["steer now"];
    const cfg = standardConfig(a);
    cfg.steeringSource = () => queue.splice(0);

    const h = new StandardHarness(cfg);
    const events: HarnessStreamEvent[] = [];
    const r = await h.run({ task: react(5), on_stream: (e) => events.push(e) });
    expect(r.kind).toBe("success");

    const userMessages = events.filter((e) => e.kind === "user_message") as Extract<
      HarnessStreamEvent,
      { kind: "user_message" }
    >[];
    expect(userMessages.length).toBe(1);
    expect(userMessages[0]!.steered).toBe(true);
    expect(userMessages[0]!.content).toBe("steer now");
  });

  it("absent steeringSource is byte-identical (no user_message events)", async () => {
    const a = new MockAgent(AgentId.of("test"));
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
    const cfg = standardConfig(a);
    const h = new StandardHarness(cfg);
    const events: HarnessStreamEvent[] = [];
    await h.run({ task: react(5), on_stream: (e) => events.push(e) });
    expect(events.filter((e) => e.kind === "user_message").length).toBe(0);
  });
});

// ============================================================================
// G-E: tool-call repair
// ============================================================================

describe("G-E tool-call repair", () => {
  it("coerceToolArgs coerces string booleans/numbers and quoted keys", () => {
    const schema: ToolSchema = {
      name: "t",
      description: "",
      parameters: {
        type: "object",
        properties: {
          flag: { type: "boolean" },
          count: { type: "number" },
          path: { type: "string" },
        },
      },
      annotations: { read_only: false, destructive: false, idempotent: false, open_world: false },
    };
    const repaired = coerceToolArgs({ flag: "true", count: "42", '"path"': "x" }, schema) as Record<
      string,
      unknown
    >;
    expect(repaired).not.toBeNull();
    expect(repaired!.flag).toBe(true);
    expect(repaired!.count).toBe(42);
    expect(repaired!.path).toBe("x");
  });

  it("re-dispatches a coerced call and succeeds; default (no repair) is single-shot", async () => {
    const a = new MockAgent(AgentId.of("test"));
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c0", name: "t", input: { flag: "true" } } as ToolCall],
      usage: usage(),
    } as TurnResult);
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);

    const reg = new RecordingRegistry();
    // First dispatch: invalid_parameters (recoverable). Second: success.
    reg.pushOutput({ kind: "error", message: "invalid parameters: flag must be boolean", recoverable: true });
    reg.pushOutput({ kind: "success", content: "ok", truncated: false });
    reg.withSchema({
      name: "t",
      description: "",
      parameters: { type: "object", properties: { flag: { type: "boolean" } } },
      annotations: { read_only: false, destructive: false, idempotent: false, open_world: false },
    });

    const cfg = standardConfig(a);
    cfg.toolRegistry = reg;
    cfg.toolCallRepair = new StandardToolCallRepair();
    cfg.maxRepairAttempts = 1;

    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success");
    expect(reg.dispatched.length).toBe(2);
    // The second dispatch carried the coerced boolean.
    expect((reg.dispatched[1]!.input as { flag: unknown }).flag).toBe(true);
  });

  it("no repair configured → single dispatch, error propagates as recoverable", async () => {
    const a = new MockAgent(AgentId.of("test"));
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c0", name: "t", input: { flag: "true" } } as ToolCall],
      usage: usage(),
    } as TurnResult);
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);

    const reg = new RecordingRegistry();
    reg.pushOutput({ kind: "error", message: "invalid parameters", recoverable: true });
    const cfg = standardConfig(a);
    cfg.toolRegistry = reg;
    // No toolCallRepair configured.
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("success"); // recoverable error → loop continues → final_response
    expect(reg.dispatched.length).toBe(1);
  });
});

// ============================================================================
// G-F: behavioral tool-call nudge
// ============================================================================

describe("G-F behavioral tool-call nudge", () => {
  it("nudges when a final_response is produced with zero tools called (opt-in)", async () => {
    // Turn 1: final_response with no tool calls. Nudge fires → another turn.
    // Turn 2: a tool call (so the nudge stops), then turn 3: final_response.
    const a = new MockAgent(AgentId.of("test"));
    a.push({ kind: "final_response", content: "I think x", usage: usage() } as TurnResult);
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c0", name: "read_file", input: { path: "x" } } as ToolCall],
      usage: usage(),
    } as TurnResult);
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);

    const tr = new ScriptedToolRegistry();
    tr.push({ kind: "success", content: "file x", truncated: false });
    // Advertise a tool so toolsAdvertised is true.
    tr.withSchema({
      name: "read_file",
      description: "",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      annotations: { read_only: true, destructive: false, idempotent: false, open_world: false },
    });
    const cfg = standardConfig(a);
    cfg.toolRegistry = tr;
    cfg.toolCallNudge = true;

    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(8) });
    expect(r.kind).toBe("success");
    // The run took 3 turns (nudge forced an extra one after turn 1).
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.turns).toBe(3);
  });

  it("default (toolCallNudge off) → no nudge, completes on first final_response", async () => {
    const a = new MockAgent(AgentId.of("test"));
    a.push({ kind: "final_response", content: "I think x", usage: usage() } as TurnResult);
    const tr = new ScriptedToolRegistry();
    tr.withSchema({
      name: "read_file",
      description: "",
      parameters: { type: "object", properties: { path: { type: "string" } } },
      annotations: { read_only: true, destructive: false, idempotent: false, open_world: false },
    });
    const cfg = standardConfig(a);
    cfg.toolRegistry = tr;
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(8) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.turns).toBe(1);
  });
});

// ============================================================================
// G-J: narration refund
// ============================================================================

describe("G-J narration refund (send_message-only turns)", () => {
  it("a run with send_message narration and ample budget completes", async () => {
    // Smoke test: send_message-only turns don't break the loop; with ample
    // budget the run reaches the final_response and succeeds.
    const a = new MockAgent(AgentId.of("test"));
    for (let i = 0; i < 2; i += 1) {
      a.push({
        kind: "tool_call_requested",
        calls: [{ id: `c${i}`, name: "send_message", input: { content: `narrate ${i}` } } as ToolCall],
        usage: usage(),
      } as TurnResult);
    }
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);

    const tr = new ScriptedToolRegistry();
    for (let i = 0; i < 2; i += 1) {
      tr.push({ kind: "success", content: `narrate ${i}`, truncated: false });
    }
    const cfg = standardConfig(a);
    cfg.toolRegistry = tr;

    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(8) });
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.turns).toBe(3);
  });

  it("send_message-only narration is refunded at the window gate (vs productive turns)", async () => {
    // 2 send_message-only narration turns + 1 final_response, per_loop cap 2.
    // The narration refund inflates the window gate (cap + refund = 4) so the
    // window reaches the final_response turn (3) instead of halting at the
    // window gate (turns=2 >= cap=2). The bare leaf's per_loop charging then
    // catches the 3 turns > allowance 2 → the run fails at the LEAF budget
    // (budget_exhausted/budget_exceeded), NOT at the window gate. Compare to a
    // pure productive run (3 read_file turns) that fails at the WINDOW gate
    // (turns=2 >= cap=2) before reaching turn 3. The observable difference is
    // that the narration run reaches the final_response (the window produced
    // it) while the productive run does not — proving the refund let the window
    // run the extra narration turn.
    const a = new MockAgent(AgentId.of("test"));
    for (let i = 0; i < 2; i += 1) {
      a.push({
        kind: "tool_call_requested",
        calls: [{ id: `c${i}`, name: "send_message", input: { content: `narrate ${i}` } } as ToolCall],
        usage: usage(),
      } as TurnResult);
    }
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);
    const tr = new ScriptedToolRegistry();
    for (let i = 0; i < 2; i += 1) {
      tr.push({ kind: "success", content: `narrate ${i}`, truncated: false });
    }
    const cfg = standardConfig(a);
    cfg.toolRegistry = tr;
    const h = new StandardHarness(cfg);
    const events: HarnessStreamEvent[] = [];
    const r = await h.run({ task: react(2), on_stream: (e) => events.push(e) });
    // The window reached the final_response (the refund let it run turn 3);
    // the run ultimately fails at the leaf budget, but a final_response event
    // WAS emitted — proving the window gate admitted the narration-refunded turn.
    expect(events.some((e) => e.kind === "final_response")).toBe(true);
    expect(r.kind).toBe("failure");
  });
});

// ============================================================================
// G-B: ToolProgress live streaming
// ============================================================================

describe("G-B ToolProgress live streaming", () => {
  it("emits tool_progress events from a tool that calls ctx.commandProgress", async () => {
    // A custom tool that calls ctx.commandProgress(call) with tail snapshots,
    // then returns success. Registered via HarnessBuilder.tool so the run loop
    // bridges it through RealToolRegistry (which threads the progress sink).
    const progressTool: toolRegistry.StandardTool = {
      implementation: {
        name: "progress_tool",
        async execute(
          call: ToolCall,
          _sandbox: unknown,
          ctx: toolRegistry.ToolContext,
        ): Promise<ToolOutput> {
          const sink = ctx.commandProgress(call);
          if (sink != null) {
            sink("line 1\n");
            sink("line 1\nline 2\n");
            sink("line 1\nline 2\nline 3\n");
          }
          return { kind: "success", content: "line 1\nline 2\nline 3\n", truncated: false };
        },
      },
      schema: {
        name: "progress_tool",
        description: "a tool that streams progress",
        parameters: { type: "object", properties: {} },
        annotations: { read_only: true, destructive: false, idempotent: false, open_world: false },
      } as ToolSchema,
    };

    const a = new MockAgent(AgentId.of("test"));
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c0", name: "progress_tool", input: {} } as ToolCall],
      usage: usage(),
    } as TurnResult);
    a.push({ kind: "final_response", content: "done", usage: usage() } as TurnResult);

    const h = new HarnessBuilder(
      a,
      new ScriptedToolRegistry(),
      new AllowAllSandbox(),
      new NoopContextManager(),
      new AlwaysContinuePolicy(),
    )
      .tool(progressTool)
      .build();

    const events: HarnessStreamEvent[] = [];
    const r = await h.run({ task: react(8), on_stream: (e) => events.push(e) });
    expect(r.kind).toBe("success");

    const progress = events.filter((e) => e.kind === "tool_progress") as Extract<
      HarnessStreamEvent,
      { kind: "tool_progress" }
    >[];
    // Three previews, each REPLACING the previous (tail snapshots).
    expect(progress.length).toBe(3);
    expect(progress[0]!.call_id).toBe("c0");
    expect(progress[0]!.name).toBe("progress_tool");
    expect(progress[0]!.preview).toBe("line 1\n");
    expect(progress[2]!.preview).toBe("line 1\nline 2\nline 3\n");
  });
});
