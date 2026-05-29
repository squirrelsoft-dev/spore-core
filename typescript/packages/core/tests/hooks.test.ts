/**
 * Unit tests for the lifecycle hook system (spore-core issue #69). One test per
 * rule R1–R26 where applicable, exercising each event/decision in isolation.
 */

import { describe, expect, it } from "vitest";

import { hooks, SessionId, TaskId, context as contextNs } from "../src/index.js";
import { emptyContext } from "../src/agent/types.js";

const {
  StandardHookChain,
  FunctionHook,
  CommandHook,
  HookError,
  validateDecisionForEvent,
  hookContextPayload,
  ALL_HOOK_EVENTS,
  hookEventIsPre,
  hookEventIsSyncOnly,
  hookEventIsAsyncOnly,
  hookEventCanBlock,
  hookEventCanDeny,
} = hooks;
type HookContext = hooks.HookContext;
type HookDecision = hooks.HookDecision;

const sid = () => SessionId.of("s1");

function fnHook(name: string, events: hooks.HookEvent[], d: HookDecision) {
  return new FunctionHook(name, events, () => d);
}

describe("HookEvent classification", () => {
  it("has 17 events", () => {
    expect(ALL_HOOK_EVENTS.length).toBe(17);
  });

  it("classifies pre / sync-only / async-only / block / deny", () => {
    expect(hookEventIsPre("pre_turn")).toBe(true);
    expect(hookEventIsPre("post_turn")).toBe(false);
    expect(hookEventIsSyncOnly("stop")).toBe(true);
    expect(hookEventIsSyncOnly("post_turn")).toBe(false);
    expect(hookEventIsAsyncOnly("on_pause")).toBe(true);
    expect(hookEventIsAsyncOnly("post_compact")).toBe(true);
    expect(hookEventCanBlock("stop")).toBe(true);
    expect(hookEventCanBlock("post_turn")).toBe(false);
    expect(hookEventCanDeny("pre_tool_use")).toBe(true);
    expect(hookEventCanDeny("on_subagent_spawn")).toBe(true);
    expect(hookEventCanDeny("pre_turn")).toBe(false);
  });
});

describe("StandardHookChain firing", () => {
  // R3 / R25: registration-order firing, per-event filtering.
  it("fires in registration order and only for the invoked event", async () => {
    const order: string[] = [];
    const chain = new StandardHookChain();
    for (const label of ["a", "b", "c"]) {
      chain.register(
        new FunctionHook(label, ["post_turn"], () => {
          order.push(label);
          return { decision: "continue" };
        }),
      );
    }
    // A hook on a different event must NOT fire (R25).
    chain.register(
      new FunctionHook("other", ["pre_turn"], () => {
        order.push("other");
        return { decision: "continue" };
      }),
    );
    const ctx: HookContext = {
      event: "post_turn",
      session_id: sid(),
      turn_number: 1,
      output: { text: "", had_tool_calls: false },
    };
    await chain.fire(ctx);
    expect(order).toEqual(["a", "b", "c"]);
  });

  // R1 / R16: pre-hook mutation in place.
  it("pre_tool_use mutates input in place", async () => {
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("mut", ["pre_tool_use"], (ctx) => {
        if (ctx.event === "pre_tool_use") ctx.tool_input = { path: "/safe" };
        return { decision: "continue" };
      }),
    );
    const ctx: HookContext = {
      event: "pre_tool_use",
      session_id: sid(),
      turn_number: 1,
      tool_name: "read_file",
      tool_input: { path: "/etc/passwd" },
    };
    const out = await chain.fire(ctx);
    expect(out).toEqual({ kind: "continue" });
    expect(ctx.tool_input).toEqual({ path: "/safe" });
  });

  // R2: pre-hook chain threads mutation to the next hook.
  it("threads mutation through the chain", async () => {
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("first", ["pre_tool_use"], (ctx) => {
        if (ctx.event === "pre_tool_use") ctx.tool_input = { v: 1 };
        return { decision: "continue" };
      }),
    );
    chain.register(
      new FunctionHook("second", ["pre_tool_use"], (ctx) => {
        if (ctx.event === "pre_tool_use") {
          const v = (ctx.tool_input as { v: number }).v;
          ctx.tool_input = { v: v + 1 };
        }
        return { decision: "continue" };
      }),
    );
    const ctx: HookContext = {
      event: "pre_tool_use",
      session_id: sid(),
      turn_number: 1,
      tool_name: "t",
      tool_input: {},
    };
    await chain.fire(ctx);
    expect(ctx.tool_input).toEqual({ v: 2 });
  });

  // R6: Mutate decision replaces the mutable field.
  it("mutate decision replaces the field", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("m", ["pre_tool_use"], { decision: "mutate", data: { replaced: true } }));
    const ctx: HookContext = {
      event: "pre_tool_use",
      session_id: sid(),
      turn_number: 1,
      tool_name: "t",
      tool_input: { orig: 1 },
    };
    await chain.fire(ctx);
    expect(ctx.tool_input).toEqual({ replaced: true });
  });

  // R15: PreToolUse deny.
  it("pre_tool_use deny short-circuits", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("deny", ["pre_tool_use"], { decision: "deny", reason: "blocked path" }));
    const ctx: HookContext = {
      event: "pre_tool_use",
      session_id: sid(),
      turn_number: 1,
      tool_name: "t",
      tool_input: {},
    };
    expect(await chain.fire(ctx)).toEqual({ kind: "deny", reason: "blocked path" });
  });

  // R10 / R12: sync post-hook (Stop) block.
  it("stop hook block returns block outcome", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("verify", ["stop"], { decision: "block", reason: "tests failing" }));
    const ctx: HookContext = {
      event: "stop",
      session_id: sid(),
      turn_number: 3,
      last_output: { text: "", had_tool_calls: false },
      task_instruction: "do it",
      session_state: null,
    };
    expect(await chain.fire(ctx)).toEqual({ kind: "block", reason: "tests failing" });
  });

  // R13: Stop all-continue terminates.
  it("stop hook all-continue returns continue", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("ok", ["stop"], { decision: "continue" }));
    const ctx: HookContext = {
      event: "stop",
      session_id: sid(),
      turn_number: 1,
      last_output: { text: "", had_tool_calls: false },
      task_instruction: "x",
      session_state: null,
    };
    expect(await chain.fire(ctx)).toEqual({ kind: "continue" });
  });

  // R7: Inject aggregation, newline-joined.
  it("aggregates inject decisions newline-joined", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("i1", ["pre_turn"], { decision: "inject", context: "one" }));
    chain.register(fnHook("i2", ["pre_turn"], { decision: "inject", context: "two" }));
    const ctx: HookContext = {
      event: "pre_turn",
      session_id: sid(),
      turn_number: 1,
      context_block: emptyContext(),
    };
    expect(await chain.fire(ctx)).toEqual({ kind: "inject", context: "one\ntwo" });
  });

  // R11: async post-hook is fire-and-forget — not awaited, never affects outcome.
  it("async post-hook is fire-and-forget", async () => {
    const chain = new StandardHookChain();
    let fired = false;
    chain.register(
      new FunctionHook("log", ["post_turn"], () => {
        fired = true;
        return { decision: "continue" };
      }).asyncMode(),
    );
    const ctx: HookContext = {
      event: "post_turn",
      session_id: sid(),
      turn_number: 1,
      output: { text: "", had_tool_calls: false },
    };
    // Returns continue synchronously; the async hook does not influence it.
    expect(await chain.fire(ctx)).toEqual({ kind: "continue" });
    void fired;
  });

  it("async post-hook failure is swallowed", async () => {
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("boom", ["post_turn"], () => {
        throw new Error("kaboom");
      }).asyncMode(),
    );
    const ctx: HookContext = {
      event: "post_turn",
      session_id: sid(),
      turn_number: 1,
      output: { text: "", had_tool_calls: false },
    };
    // Must not reject despite the async hook throwing.
    expect(await chain.fire(ctx)).toEqual({ kind: "continue" });
  });
});

describe("register-time validation", () => {
  // R8: Stop registered async is rejected.
  it("rejects async registration of sync-only stop", () => {
    const chain = new StandardHookChain();
    const hook = new FunctionHook("s", ["stop"], () => ({ decision: "continue" })).asyncMode();
    expect(() => chain.register(hook)).toThrowError(HookError);
    try {
      chain.register(hook);
    } catch (e) {
      expect((e as hooks.HookError).kind).toBe("sync_only_event");
    }
  });

  // R9: OnPause registered sync is rejected.
  it("rejects sync registration of async-only on_pause", () => {
    const chain = new StandardHookChain();
    const hook = fnHook("p", ["on_pause"], { decision: "continue" });
    try {
      chain.register(hook);
      expect.fail("should have thrown");
    } catch (e) {
      expect((e as hooks.HookError).kind).toBe("async_only_event");
    }
  });
});

describe("decision validity", () => {
  // R4 / R17 / R24: illegal Block on a non-blocking event rejected at fire.
  it("rejects block on a non-blocking event at fire time", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("bad", ["post_turn"], { decision: "block", reason: "no" }));
    const ctx: HookContext = {
      event: "post_turn",
      session_id: sid(),
      turn_number: 1,
      output: { text: "", had_tool_calls: false },
    };
    await expect(chain.fire(ctx)).rejects.toMatchObject({ kind: "illegal_decision" });
  });

  // R5: Deny outside PreToolUse / OnSubagentSpawn rejected.
  it("deny is legal only for pre_tool_use / on_subagent_spawn", () => {
    expect(validateDecisionForEvent({ decision: "deny", reason: "x" }, "pre_tool_use")).toBeNull();
    expect(
      validateDecisionForEvent({ decision: "deny", reason: "x" }, "on_subagent_spawn"),
    ).toBeNull();
    expect(validateDecisionForEvent({ decision: "deny", reason: "x" }, "pre_turn")).toBeInstanceOf(
      HookError,
    );
  });
});

describe("FunctionHook", () => {
  // R23: FunctionHook runs the closure (and may mutate).
  it("runs the closure and mutates the context", async () => {
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("f", ["on_loop_start"], (ctx) => {
        if (ctx.event === "on_loop_start") ctx.task_instruction += " [checked]";
        return { decision: "continue" };
      }),
    );
    const ctx: HookContext = {
      event: "on_loop_start",
      session_id: sid(),
      task_instruction: "do work",
      config: null,
    };
    await chain.fire(ctx);
    expect(ctx.task_instruction).toBe("do work [checked]");
  });
});

describe("CommandHook", () => {
  // R18-R21: stdin/stdout roundtrip.
  it("roundtrips stdin payload and parses stdout decision", async () => {
    // Echo a block decision (ignoring stdin).
    const hook = new CommandHook("cmd", ["stop"], "sh", [
      "-c",
      'cat >/dev/null; echo \'{"decision":"block","reason":"cmd says no"}\'',
    ]);
    const chain = new StandardHookChain();
    chain.register(hook);
    const ctx: HookContext = {
      event: "stop",
      session_id: sid(),
      turn_number: 1,
      last_output: { text: "", had_tool_calls: false },
      task_instruction: "x",
      session_state: null,
    };
    expect(await chain.fire(ctx)).toEqual({ kind: "block", reason: "cmd says no" });
  });

  // R20: nonzero exit → command_failed (NOT an implicit block).
  it("nonzero exit raises command_failed", async () => {
    const hook = new CommandHook("cmd", ["stop"], "sh", ["-c", "exit 7"]);
    const chain = new StandardHookChain();
    chain.register(hook);
    const ctx: HookContext = {
      event: "stop",
      session_id: sid(),
      turn_number: 1,
      last_output: { text: "", had_tool_calls: false },
      task_instruction: "x",
      session_state: null,
    };
    await expect(chain.fire(ctx)).rejects.toMatchObject({
      kind: "command_failed",
      detail: { kind: "command_failed", code: 7 },
    });
  });

  // R21: malformed stdout → command_output_invalid.
  it("malformed stdout raises command_output_invalid", async () => {
    const hook = new CommandHook("cmd", ["stop"], "sh", ["-c", "echo 'not json at all'"]);
    const chain = new StandardHookChain();
    chain.register(hook);
    const ctx: HookContext = {
      event: "stop",
      session_id: sid(),
      turn_number: 1,
      last_output: { text: "", had_tool_calls: false },
      task_instruction: "x",
      session_state: null,
    };
    await expect(chain.fire(ctx)).rejects.toMatchObject({ kind: "command_output_invalid" });
  });
});

describe("deferred (defined-but-not-loop-wired) events fire in isolation", () => {
  it("on_plan_created mutates the plan", async () => {
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("plan", ["on_plan_created"], (ctx) => {
        if (ctx.event === "on_plan_created") ctx.plan.tasks.push("extra");
        return { decision: "continue" };
      }),
    );
    const ctx: HookContext = {
      event: "on_plan_created",
      session_id: sid(),
      plan: { tasks: ["a"], rationale: "" },
    };
    await chain.fire(ctx);
    expect(ctx.plan.tasks).toEqual(["a", "extra"]);
  });

  it("on_subagent_spawn deny", async () => {
    const chain = new StandardHookChain();
    chain.register(fnHook("ss", ["on_subagent_spawn"], { decision: "deny", reason: "no spawn" }));
    const ctx: HookContext = {
      event: "on_subagent_spawn",
      session_id: sid(),
      child_task: "child task",
      strategy: "react",
    };
    expect(await chain.fire(ctx)).toEqual({ kind: "deny", reason: "no spawn" });
  });

  it("pre_compact mutates preserve hints", async () => {
    const chain = new StandardHookChain();
    chain.register(
      new FunctionHook("pc", ["pre_compact"], (ctx) => {
        if (ctx.event === "pre_compact") ctx.preserve_hints.keep_recent_file_list = false;
        return { decision: "continue" };
      }),
    );
    const ctx: HookContext = {
      event: "pre_compact",
      session_id: sid(),
      preserve_hints: contextNs.defaultCompactionPreserveHints(),
    };
    await chain.fire(ctx);
    expect(ctx.preserve_hints.keep_recent_file_list).toBe(false);
  });

  it("on_task_advance mutate replaces the task", async () => {
    const chain = new StandardHookChain();
    const replacement = {
      id: TaskId.of("t2"),
      instruction: "next",
      session_id: SessionId.of("s2"),
      budget: {},
      loop_strategy: { kind: "re_act", max_iterations: 5 } as const,
    };
    chain.register(fnHook("ta", ["on_task_advance"], { decision: "mutate", data: replacement }));
    const ctx: HookContext = {
      event: "on_task_advance",
      session_id: sid(),
      task: {
        id: TaskId.of("t1"),
        instruction: "first",
        session_id: sid(),
        budget: {},
        loop_strategy: { kind: "re_act", max_iterations: 5 },
      },
      task_index: 0,
      total_tasks: 2,
    };
    await chain.fire(ctx);
    expect(ctx.task.instruction).toBe("next");
  });
});

describe("command-handler stdin payload shape", () => {
  it("serializes the per-event context payload", () => {
    const ctx: HookContext = {
      event: "pre_tool_use",
      session_id: SessionId.of("sess-1"),
      turn_number: 1,
      tool_name: "read_file",
      tool_input: { path: "/etc/passwd" },
    };
    const payload = hookContextPayload(ctx);
    expect(payload).toEqual({
      session_id: SessionId.of("sess-1"),
      turn_number: 1,
      tool_name: "read_file",
      tool_input: { path: "/etc/passwd" },
    });
    // SessionId serializes to its string via toJSON.
    expect(JSON.parse(JSON.stringify({ event: ctx.event, context: payload }))).toEqual({
      event: "pre_tool_use",
      context: {
        session_id: "sess-1",
        turn_number: 1,
        tool_name: "read_file",
        tool_input: { path: "/etc/passwd" },
      },
    });
  });
});
