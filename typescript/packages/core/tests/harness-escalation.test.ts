/**
 * Unit tests for the Tool Escalation Protocol (spore-core issue #80).
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs#tests` (the escalation block)
 * — same rules R1–R9, same verdicts, parallel structure. A tool returning
 * `ToolOutput.escalate` terminates the run cleanly, surfaces the signal via
 * `RunResult.escalate`, preserves a resumable `PausedState`, and finalizes
 * observability as `Escalated` — the harness never acts on the signal itself.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  MockAgent,
  SessionId,
  StandardHarness,
  autoContinue,
  newTask,
  observability,
  type AutoGrantInfo,
  type HarnessConfig,
  type HarnessSignal,
  type LoopStrategy,
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
  return newTask("do something", SessionId.of("s1"), strategy);
}

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

/** A turn that requests `n` tool calls (ids c0..c{n-1}, names t0..t{n-1}),
 *  followed by a `FinalResponse` so a resumed loop runs to Success. */
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

const abortSignal: HarnessSignal = { kind: "abort", reason: "agent requested stop" };

// R1 + R8: a dispatched `escalate { abort }` terminates the run and returns
// `RunResult.escalate`, NOT `RunResult.failure`.
describe("Harness escalation — R1 + R8: terminates with Escalate, not Failure", () => {
  it("escalate(abort) returns escalate carrying the signal", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "escalate", signal: abortSignal });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("escalate");
    if (r.kind === "escalate") {
      expect(r.signal).toEqual(abortSignal);
    }
    expect(r.kind).not.toBe("failure");
  });
});

// R2: the escalation is NOT appended to message history. With one escalating
// call, no tool-result message is recorded in the preserved session state.
describe("Harness escalation — R2: not appended to history", () => {
  it("no tool-role message is appended on escalation", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "escalate", signal: abortSignal });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("escalate");
    if (r.kind === "escalate") {
      const toolMessages = r.state.session_state.messages.filter((m) => m.role === "tool");
      expect(toolMessages.length).toBe(0);
    }
  });
});

// R3: observability is finalized with `SessionOutcome.escalated`.
describe("Harness escalation — R3: finalizes observability as Escalated", () => {
  it("escalation flushes a finalized session with the escalated outcome", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    const obs = new observability.InMemoryObservabilityProvider();
    cfg.observability = obs;
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "escalate", signal: abortSignal });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("escalate");
    if (r.kind === "escalate") {
      const metrics = await obs.getSessionMetrics(r.session_id);
      expect(metrics).toBeDefined();
      expect(metrics?.outcome).toEqual({ kind: "escalated" });
    }
  });

  // R3 (contrast): `waiting_for_human` does NOT finalize observability — and in
  // particular never as Escalated. A subagent tool returns waiting_for_human.
  it("waiting_for_human is not terminal — never finalized as Escalated", async () => {
    const a = makeAgent();
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c", name: "subagent", input: {} }],
      usage: usage(),
    } as TurnResult);
    const cfg = standardConfig(a);
    const obs = new observability.InMemoryObservabilityProvider();
    cfg.observability = obs;
    cfg.toolRegistry = new ScriptedToolRegistry().push({
      kind: "waiting_for_human",
      child_state: {
        session_id: SessionId.of("child"),
        task_id: react(1).id,
        turn_number: 1,
        session_state: { messages: [], extras: {} },
        pending_tool_calls: [],
        approved_results: [],
        human_request: { kind: "clarification", question: "?" },
        task: react(1),
        budget_used: { turns: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0 },
        parent_tool_call_id: "c",
        toolset: "",
      },
      request: { kind: "clarification", question: "?" },
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind === "waiting_for_human") {
      const metrics = await obs.getSessionMetrics(r.state.session_id);
      // May exist (a turn span was emitted) but must NOT be Escalated.
      if (metrics) {
        expect(metrics.outcome).not.toEqual({ kind: "escalated" });
      }
    }
  });
});

// R4: `RunResult.escalate` carries all five fields populated, and
// `turns === budget_used.turns`.
describe("Harness escalation — R4: carries all five fields", () => {
  it("signal, state, session_id, usage, turns are all populated", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "escalate", signal: abortSignal });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("escalate");
    if (r.kind === "escalate") {
      expect(r.signal).toEqual(abortSignal);
      expect(r.session_id.asString()).toBe("s1");
      expect(r.state.session_id.asString()).toBe("s1");
      expect(r.turns).toBe(r.state.budget_used.turns);
      // One turn was consumed before the escalating dispatch.
      expect(r.turns).toBe(1);
      // Usage is populated from the consumed turn.
      expect(r.usage.input_tokens).toBe(1);
      expect(r.usage.output_tokens).toBe(1);
    }
  });
});

// R5 + R6: the preserved `state` is resumable, and the signal is discarded on
// resume — the harness just continues the original session.
describe("Harness escalation — R5 + R6: resumable state, signal discarded", () => {
  it("escalation state has no human_request and resumes to Success", async () => {
    const a = agentWithToolCalls(1);
    const cfg = standardConfig(a);
    const switchMode: HarnessSignal = { kind: "switch_mode", mode: "plan" };
    cfg.toolRegistry = new ScriptedToolRegistry().push({ kind: "escalate", signal: switchMode });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("escalate");
    if (r.kind !== "escalate") return;
    // R5 shape: escalation-derived state carries no human request.
    expect(r.state.human_request).toBeUndefined();
    // R6: resume continues the ORIGINAL session — it does NOT switch mode. The
    // MockAgent's next turn is a FinalResponse, so resume runs to Success.
    const resumed = await h.resume(r.state, { kind: "allow" });
    expect(resumed.kind).toBe("success");
    if (resumed.kind === "success") {
      expect(resumed.output).toBe("resumed-done");
      expect(resumed.session_id.asString()).toBe("s1");
    }
  });
});

// R7: each `HarnessSignal` variant round-trips through the documented wire
// shape; `switch_mode` carries the EXISTING `Mode` enum (no `HarnessMode`).
describe("Harness escalation — R7: wire round-trip + SwitchMode uses Mode", () => {
  it("every HarnessSignal variant round-trips wrapped in escalate", () => {
    const cases: HarnessSignal[] = [
      { kind: "enter_plan_mode", context: "ctx" },
      { kind: "exit_plan_mode", plan: { tasks: ["a", "b"], rationale: "why" } },
      { kind: "switch_mode", mode: "safe_auto" },
      abortSignal,
    ];
    for (const signal of cases) {
      const output = { kind: "escalate" as const, signal };
      const json = JSON.stringify(output);
      const back = JSON.parse(json) as typeof output;
      expect(back).toEqual(output);
    }
    // Spot-check the tag shape: snake_case `kind` on both layers.
    const value = JSON.parse(JSON.stringify({ kind: "escalate", signal: abortSignal }));
    expect(value.kind).toBe("escalate");
    expect(value.signal.kind).toBe("abort");
    expect(value.signal.reason).toBe("agent requested stop");
  });

  it("SwitchMode serializes the existing Mode enum value", () => {
    const value = JSON.parse(JSON.stringify({ kind: "switch_mode", mode: "safe_auto" }));
    expect(value.kind).toBe("switch_mode");
    expect(value.mode).toBe("safe_auto");
  });
});

// R9: remaining tool calls after the escalating call land in
// `state.pending_tool_calls` (escalate on call[0] of a 2-call batch →
// pending === [c1]).
describe("Harness escalation — R9: remaining calls preserved as pending", () => {
  it("the trailing batch call is preserved, not executed", async () => {
    const a = agentWithToolCalls(2);
    const cfg = standardConfig(a);
    const reg = new ScriptedToolRegistry().push({ kind: "escalate", signal: abortSignal });
    cfg.toolRegistry = reg;
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(5) });
    expect(r.kind).toBe("escalate");
    if (r.kind === "escalate") {
      expect(r.state.pending_tool_calls.length).toBe(1);
      expect(r.state.pending_tool_calls[0]?.id).toBe("c1");
      expect(r.state.pending_tool_calls[0]?.name).toBe("t1");
    }
    // Exactly one dispatch happened — c1 was preserved, not executed.
    expect(reg.callCount).toBe(1);
  });
});

// ============================================================================
// SC-5: EscalationMode AutoContinue ("autonomous but capped")
// ============================================================================
//
// Mirrors the Rust `auto_continue_*` tests in `harness.rs`. Under
// `auto_continue { maxGrants, stepsPerGrant }` a budget-exhausted bare ReAct leaf
// keeps working IN-PROCESS — no consumer drive loop — by auto-granting more steps
// at the Escalate fall-through, firing `onGrant` per grant, until the worker
// completes OR the grant cap is hit.

/** A ReAct leaf agent whose cap is binding: `turns` tool-call turns (one call each). */
function budgetExhaustingAgent(turns: number): MockAgent {
  const a = makeAgent();
  for (let i = 0; i < turns; i += 1) {
    const call: ToolCall = { id: `c${i}`, name: "x", input: {} };
    a.push({ kind: "tool_call_requested", calls: [call], usage: usage() } as TurnResult);
  }
  return a;
}

function toolReg(n: number): ScriptedToolRegistry {
  const reg = new ScriptedToolRegistry();
  for (let i = 0; i < n; i += 1) reg.push({ kind: "success", content: "ok", truncated: false });
  return reg;
}

describe("Harness escalation — SC-5: AutoContinue keeps working then completes", () => {
  it("auto-grants in-process at the Escalate fall-through, then completes (onGrant fired once)", async () => {
    const a = makeAgent();
    // First window: 2 tool turns exhaust the PerLoop{2} cap → Escalate.
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c0", name: "x", input: {} }],
      usage: usage(),
    } as TurnResult);
    a.push({
      kind: "tool_call_requested",
      calls: [{ id: "c1", name: "x", input: {} }],
      usage: usage(),
    } as TurnResult);
    // After one auto-grant refreshes the cap, the worker completes.
    a.push({
      kind: "final_response",
      content: "done after auto-continue",
      usage: usage(),
    } as TurnResult);

    let grants = 0;
    const cfg = standardConfig(a);
    cfg.toolRegistry = toolReg(3);
    cfg.escalationMode = autoContinue({
      maxGrants: 3,
      stepsPerGrant: 2,
      onGrant: (info: AutoGrantInfo) => {
        expect(info.stepsGranted).toBe(2);
        grants += 1;
      },
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(2) }); // PerLoop{2}, behavior defaults to escalate.
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("done after auto-continue");
    }
    // Exactly one auto-grant was needed to finish.
    expect(grants).toBe(1);
  });
});

describe("Harness escalation — SC-5: AutoContinue is capped at maxGrants, then fails", () => {
  it("falls through to Failure once grants are spent (onGrant fired exactly maxGrants times)", async () => {
    // The agent never emits a final response — far more tool turns than any
    // window can consume — so the run only ends when the grant cap is reached.
    const a = budgetExhaustingAgent(40);
    let grants = 0;
    const cfg = standardConfig(a);
    cfg.toolRegistry = toolReg(80);
    cfg.escalationMode = autoContinue({
      maxGrants: 2,
      stepsPerGrant: 2,
      onGrant: () => {
        grants += 1;
      },
    });
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(2) }); // PerLoop{2}, behavior defaults to escalate.
    expect(r.kind).toBe("failure");
    // Exactly maxGrants auto-grants fired before falling through to Failure.
    expect(grants).toBe(2);
  });
});
