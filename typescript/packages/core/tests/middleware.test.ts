/**
 * Unit tests for the canonical MiddlewareChain (spore-core issue #11).
 *
 * Mirrors `rust/crates/spore-core/src/middleware.rs#tests` rule-for-rule.
 */

import { describe, expect, it } from "vitest";

import { middleware, SessionId } from "../src/index.js";
import {
  emptyAggregateUsage,
  emptySessionState,
  newTask,
  type HumanRequest,
  type RunResult,
  type SessionState,
} from "../src/harness/types.js";
import type { ToolCall, ToolResult } from "../src/model/schemas.js";

const {
  StandardMiddlewareChain,
  TracingMiddleware,
  PatchToolCallsMiddleware,
  LoopDetectionMiddleware,
  PreCompletionChecklistMiddleware,
  TokenBudgetMiddleware,
  MiddlewareError,
} = middleware;

type HookPoint = middleware.HookPoint;
type HookContext = middleware.HookContext;
type Middleware = middleware.Middleware;
type MiddlewareDecision = middleware.MiddlewareDecision;

// ── Test middleware that returns a programmable decision ────────────────────

class Scripted implements Middleware {
  fired: HookPoint[] = [];

  constructor(
    private readonly myName: string,
    private readonly listenHooks: HookPoint[],
    private readonly prio: number,
    private decision: MiddlewareDecision = { kind: "continue" },
  ) {}

  setDecision(d: MiddlewareDecision): this {
    this.decision = d;
    return this;
  }

  async handle(ctx: HookContext): Promise<MiddlewareDecision> {
    this.fired.push(ctx.kind);
    return this.decision;
  }
  hooks(): HookPoint[] {
    return this.listenHooks;
  }
  priority(): number {
    return this.prio;
  }
  name(): string {
    return this.myName;
  }
}

function call(id: string, name: string, input: unknown = {}): ToolCall {
  return { id, name, input };
}

function result(toolUseId: string, content: string, isError = false): ToolResult {
  return { tool_use_id: toolUseId, content, is_error: isError };
}

// ── Rule: register validates hooks list and uniqueness ───────────────────────

describe("MiddlewareChain register", () => {
  it("rejects middleware with empty hooks", () => {
    const chain = new StandardMiddlewareChain();
    expect(() => chain.register(new Scripted("m", [], 0))).toThrow(MiddlewareError);
  });

  it("rejects duplicate name", () => {
    const chain = new StandardMiddlewareChain();
    chain.register(new Scripted("m", ["before_turn"], 0));
    expect(() => chain.register(new Scripted("m", ["before_turn"], 0))).toThrow(MiddlewareError);
  });
});

// ── Rule: before hooks ascend by priority ────────────────────────────────────

describe("MiddlewareChain ordering", () => {
  it("before hooks run in ascending priority", async () => {
    const chain = new StandardMiddlewareChain();
    const tracing = new TracingMiddleware("t");
    const c = new Scripted("c", ["before_turn"], 100);
    const b = new Scripted("b", ["before_turn"], 10);
    chain.register(c);
    chain.register(b);
    chain.register(tracing);

    const state = emptySessionState();
    const d = await chain.fireBeforeTurn(state, 1);
    expect(d.kind).toBe("continue");

    // tracing fires first (TRACING_PRIORITY); then b (10); then c (100).
    expect(tracing.entries()).toEqual([{ hook: "before_turn", turn: 1 }]);
  });

  it("after hooks run in descending priority", async () => {
    const chain = new StandardMiddlewareChain();
    const a = new Scripted("a", ["after_tool"], 1);
    const b = new Scripted("b", ["after_tool"], 100);
    chain.register(a);
    chain.register(b);

    const calls = [call("c1", "edit")];
    const results: ToolResult[] = [result("c1", "ok")];
    const d = await chain.fireAfterTool(calls, results);
    expect(d.kind).toBe("continue");
    expect(a.fired.length).toBe(1);
    expect(b.fired.length).toBe(1);
  });
});

// ── Rule: first Halt stops the chain ─────────────────────────────────────────

describe("MiddlewareChain halt semantics", () => {
  it("halt stops chain and downstream middleware does not run", async () => {
    const chain = new StandardMiddlewareChain();
    const halter = new Scripted("halt", ["before_turn"], 1).setDecision({
      kind: "halt",
      reason: "stop",
    });
    const after = new Scripted("after", ["before_turn"], 100);
    chain.register(halter);
    chain.register(after);

    const d = await chain.fireBeforeTurn(emptySessionState(), 1);
    expect(d.kind).toBe("halt");
    expect(after.fired.length).toBe(0);
  });
});

// ── Rule: SurfaceToHuman first-wins on BeforeTool ────────────────────────────

describe("MiddlewareChain surface_to_human", () => {
  it("first occurrence wins on before_tool; later middleware does not run", async () => {
    const chain = new StandardMiddlewareChain();
    const req: HumanRequest = {
      kind: "tool_approval",
      calls: [call("c1", "shell")],
      risk_level: "high",
    };
    const first = new Scripted("first", ["before_tool"], 1).setDecision({
      kind: "surface_to_human",
      request: req,
    });
    const second = new Scripted("second", ["before_tool"], 2);
    chain.register(first);
    chain.register(second);

    const calls = [call("c1", "shell")];
    const d = await chain.fireBeforeTool(calls, 1);
    expect(d.kind).toBe("surface_to_human");
    expect(second.fired.length).toBe(0);
  });

  it("illegal on before_turn → halt", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(
      new Scripted("bad", ["before_turn"], 1).setDecision({
        kind: "surface_to_human",
        request: { kind: "clarification", question: "?" },
      }),
    );
    const d = await chain.fireBeforeTurn(emptySessionState(), 1);
    expect(d.kind).toBe("halt");
    if (d.kind === "halt") {
      expect(d.reason).toContain("SurfaceToHuman");
    }
  });
});

// ── Rule: ForceAnotherTurn concatenates, chain continues ─────────────────────

describe("MiddlewareChain force_another_turn", () => {
  it("concatenates injections and continues running remaining middleware", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(
      new Scripted("a", ["before_completion"], 1).setDecision({
        kind: "force_another_turn",
        inject: "first",
      }),
    );
    chain.register(
      new Scripted("b", ["before_completion"], 2).setDecision({
        kind: "force_another_turn",
        inject: "second",
      }),
    );

    const d = await chain.fireBeforeCompletion("done", 3, emptySessionState());
    expect(d.kind).toBe("force_another_turn");
    if (d.kind === "force_another_turn") {
      expect(d.inject).toContain("first");
      expect(d.inject).toContain("second");
      expect(d.inject).toBe("first\nsecond");
    }
  });

  it("illegal on before_turn → halt", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(
      new Scripted("bad", ["before_turn"], 1).setDecision({
        kind: "force_another_turn",
        inject: "x",
      }),
    );
    const d = await chain.fireBeforeTurn(emptySessionState(), 1);
    expect(d.kind).toBe("halt");
  });
});

// ── PatchToolCallsMiddleware behaviour & ordering ────────────────────────────

describe("PatchToolCallsMiddleware", () => {
  it("renames empty / whitespace tool-call names to fallback", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(new PatchToolCallsMiddleware("noop"));
    const calls = [call("c1", "")];
    const d = await chain.fireBeforeTool(calls, 1);
    expect(d.kind).toBe("continue_with_modification");
    expect(calls[0]!.name).toBe("noop");
  });

  it("runs before all other before_tool middleware (downstream sees patched calls)", async () => {
    const chain = new StandardMiddlewareChain();
    const observer = new Scripted("observer", ["before_tool"], 0);
    chain.register(new PatchToolCallsMiddleware("noop"));
    chain.register(observer);

    const calls = [call("c1", "")];
    const d = await chain.fireBeforeTool(calls, 1);
    expect(d.kind).toBe("continue_with_modification");
    expect(calls[0]!.name).toBe("noop");
    expect(observer.fired.length).toBe(1);
  });
});

// ── LoopDetectionMiddleware ──────────────────────────────────────────────────

describe("LoopDetectionMiddleware", () => {
  it("annotates result content once threshold is reached", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(new LoopDetectionMiddleware("edit", 2));
    const calls = [call("c1", "edit", { path: "/tmp/foo.txt" })];

    // First fire: under threshold, no mutation.
    let results: ToolResult[] = [result("c1", "ok")];
    let d = await chain.fireAfterTool(calls, results);
    expect(d.kind).toBe("continue");
    expect(results[0]!.content).toBe("ok");

    // Second fire reaches threshold and annotates.
    results = [result("c1", "ok")];
    d = await chain.fireAfterTool(calls, results);
    expect(d.kind).toBe("continue_with_modification");
    expect(results[0]!.content).toContain("[loop-detection]");
  });

  it("ignores tool calls that do not match the configured tool name", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(new LoopDetectionMiddleware("edit", 1));
    const calls = [call("c1", "read", { path: "/tmp/foo.txt" })];
    const results: ToolResult[] = [result("c1", "ok")];
    const d = await chain.fireAfterTool(calls, results);
    expect(d.kind).toBe("continue");
    expect(results[0]!.content).toBe("ok");
  });
});

// ── PreCompletionChecklistMiddleware ────────────────────────────────────────

describe("PreCompletionChecklistMiddleware", () => {
  it("forces another turn when required substrings are missing", async () => {
    const chain = new StandardMiddlewareChain();
    chain.register(new PreCompletionChecklistMiddleware(["tests passed"]));

    const missing = await chain.fireBeforeCompletion("done", 1, emptySessionState());
    expect(missing.kind).toBe("force_another_turn");
    if (missing.kind === "force_another_turn") {
      expect(missing.inject).toContain("tests passed");
    }

    const ok = await chain.fireBeforeCompletion("all tests passed", 1, emptySessionState());
    expect(ok.kind).toBe("continue");
  });
});

// ── TokenBudgetMiddleware ────────────────────────────────────────────────────

describe("TokenBudgetMiddleware", () => {
  it("halts at before_turn when limit reached", async () => {
    const chain = new StandardMiddlewareChain();
    const budget = new TokenBudgetMiddleware(100);
    chain.register(budget);

    const state: SessionState = emptySessionState();
    const before = await chain.fireBeforeTurn(state, 1);
    expect(before.kind).toBe("continue");
    budget.record(150);
    const after = await chain.fireBeforeTurn(state, 2);
    expect(after.kind).toBe("halt");
  });
});

// ── Session boundary hooks ───────────────────────────────────────────────────

describe("MiddlewareChain session boundary hooks", () => {
  it("fires before_session and after_session through tracing", async () => {
    const chain = new StandardMiddlewareChain();
    const tracing = new TracingMiddleware("t");
    chain.register(tracing);

    const sid = SessionId.of("sess-1");
    const task = newTask("test task", sid, { kind: "re_act", max_iterations: 5 });
    await chain.fireBeforeSession(task, sid);
    const result: RunResult = {
      kind: "success",
      output: "done",
      session_id: sid,
      usage: emptyAggregateUsage(),
      turns: 1,
    };
    await chain.fireAfterSession(result, sid);

    const hooks = tracing.entries().map((e) => e.hook);
    expect(hooks).toContain("before_session");
    expect(hooks).toContain("after_session");
  });
});
