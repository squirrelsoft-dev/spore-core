/**
 * Unit + integration tests for `HumanRequest::BudgetExhausted` + the `Escalate`
 * HITL resume seam (spore-core issue #130).
 *
 * Mirrors the Rust `#130` test module in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same verdicts, parallel
 * structure. Covers:
 *   - serde round-trips of the 3 new variants (EscalationAction,
 *     HumanRequest.budget_exhausted, HumanResponse.escalate) + existing-variant
 *     regression;
 *   - the `escalationMode()` accessor on StandardHarness;
 *   - SurfaceToHuman pause for a bare leaf (no Skip) and a combinator (Skip);
 *   - Autonomous: no pause, propagate budget_exceeded;
 *   - resume ContinueWithBudget / Fail / Skip-on-leaf / Skip-on-PlanExecute;
 *   - grantTaskBudget raises caps without shrinking.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EscalationActionSchema,
  ExecutionRegistry,
  HumanRequestSchema,
  HumanResponseSchema,
  MockAgent,
  SessionId,
  StandardHarness,
  autonomous,
  emptyBudgetSnapshot,
  emptySessionState,
  grantTaskBudget,
  newTask,
  reactPartialJson,
  surfaceToHuman,
  type EscalationAction,
  type HarnessConfig,
  type HumanRequest,
  type HumanResponse,
  type LoopStrategy,
  type PausedState,
  type Task,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import { EmptyToolRegistry } from "../src/tool-registry/empty.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

// ── config helpers ───────────────────────────────────────────────────────────

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("test"));
}

/** A `surface_to_human` config — budget escalation PAUSES (HITL). */
function surfaceConfig(agent: MockAgent): HarnessConfig {
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    escalationMode: surfaceToHuman,
  };
}

/** An `autonomous` config — budget escalation PROPAGATES (AFK). */
function autonomousConfig(agent: MockAgent): HarnessConfig {
  return { ...surfaceConfig(agent), escalationMode: autonomous };
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

/** A bare ReAct leaf agent whose cap is binding: `turns` tool-call turns. */
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

// ============================================================================
// Serde round-trips (the byte-identical wire guarantee, fork A/B)
// ============================================================================

describe("EscalationAction serde", () => {
  it("round-trips all 3 variants byte-identically (kind-tagged, snake_case)", () => {
    const actions: EscalationAction[] = [
      { kind: "continue_with_budget", steps: 7 },
      { kind: "skip" },
      { kind: "fail" },
    ];
    for (const a of actions) {
      const json = JSON.stringify(a);
      const back = EscalationActionSchema.parse(JSON.parse(json));
      expect(back).toEqual(a);
    }
  });

  it("ContinueWithBudget carries a NAMED `steps` field (fork A wire form)", () => {
    expect(JSON.stringify({ kind: "continue_with_budget", steps: 7 })).toBe(
      '{"kind":"continue_with_budget","steps":7}',
    );
    expect(JSON.stringify({ kind: "skip" })).toBe('{"kind":"skip"}');
    expect(JSON.stringify({ kind: "fail" })).toBe('{"kind":"fail"}');
  });
});

describe("HumanRequest.budget_exhausted serde", () => {
  it("round-trips with all fields", () => {
    const req: HumanRequest = {
      kind: "budget_exhausted",
      phase: "react",
      policy: { kind: "per_loop", value: 2 },
      steps_taken: 2,
      continues_used: 0,
      partial_output: reactPartialJson("partial"),
      available_actions: [
        { kind: "continue_with_budget", steps: 5 },
        { kind: "skip" },
        { kind: "fail" },
      ],
    };
    const back = HumanRequestSchema.parse(JSON.parse(JSON.stringify(req)));
    expect(back).toEqual(req);
  });

  it("existing variants are UNCHANGED (regression)", () => {
    const variants: HumanRequest[] = [
      { kind: "tool_approval", calls: [], risk_level: "high" },
      { kind: "clarification", question: "which?" },
      { kind: "review", content: "ship it?" },
    ];
    for (const v of variants) {
      const back = HumanRequestSchema.parse(JSON.parse(JSON.stringify(v)));
      expect(back).toEqual(v);
    }
  });
});

describe("HumanResponse.escalate serde", () => {
  it("round-trips the new escalate variant", () => {
    const r: HumanResponse = {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 3 },
    };
    const back = HumanResponseSchema.parse(JSON.parse(JSON.stringify(r)));
    expect(back).toEqual(r);
  });

  it("existing variants are UNCHANGED (regression)", () => {
    const variants: HumanResponse[] = [
      { kind: "allow" },
      { kind: "allow_with_modification", calls: [] },
      { kind: "deny", reason: "no" },
      { kind: "halt" },
      { kind: "answer", text: "x" },
      { kind: "approve_with_feedback", feedback: "lgtm" },
      { kind: "reject", reason: "redo" },
    ];
    for (const v of variants) {
      const back = HumanResponseSchema.parse(JSON.parse(JSON.stringify(v)));
      expect(back).toEqual(v);
    }
  });
});

// ============================================================================
// escalationMode() accessor (mirrors Rust escalation_mode)
// ============================================================================

describe("StandardHarness.escalationMode accessor", () => {
  it("reflects the configured mode", () => {
    const surface = new StandardHarness(surfaceConfig(makeAgent()));
    expect(surface.escalationMode()).toEqual(surfaceToHuman);
    const afk = new StandardHarness(autonomousConfig(makeAgent()));
    expect(afk.escalationMode()).toEqual(autonomous);
  });

  it("defaults to autonomous when the knob is omitted (legacy raw-config parity)", () => {
    const cfg = surfaceConfig(makeAgent());
    delete cfg.escalationMode;
    expect(new StandardHarness(cfg).escalationMode()).toEqual(autonomous);
  });
});

// ============================================================================
// SurfaceToHuman: bare leaf pauses (no Skip)
// ============================================================================

describe("SurfaceToHuman — bare leaf pauses with a budget request (no Skip)", () => {
  it("a binding leaf cap pauses with [continue_with_budget, fail]", async () => {
    const cfg = surfaceConfig(budgetExhaustingAgent(3));
    cfg.toolRegistry = toolReg(3);
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(2) }); // PerLoop{2}, no global cap → leaf binds.
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind === "waiting_for_human") {
      // The PausedState mirrors the request verbatim.
      expect(r.state.human_request).toEqual(r.request);
      expect(r.state.session_state.messages).toHaveLength(1);
      expect(r.request.kind).toBe("budget_exhausted");
      if (r.request.kind === "budget_exhausted") {
        expect(r.request.phase).toBe("react");
        expect(r.request.policy).toEqual({ kind: "per_loop", value: 2 });
        expect(r.request.steps_taken).toBe(2);
        // A bare leaf OMITS skip (fork C).
        expect(r.request.available_actions).toEqual([
          { kind: "continue_with_budget", steps: 2 },
          { kind: "fail" },
        ]);
        // The partial is the documented ReAct shape.
        expect(r.request.partial_output).toBe(reactPartialJson(""));
      }
    }
  });
});

// ============================================================================
// SurfaceToHuman: combinator (PlanExecute) pauses with Skip offered
// ============================================================================

describe("SurfaceToHuman — combinator (PlanExecute) offers Skip", () => {
  it("a PlanExecute whose execute leaf exhausts pauses with all three actions", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: '{"tasks":["x","y","z"]}', usage: usage() });
    // Execute step: keep requesting tools past the leaf cap of 2.
    for (let i = 0; i < 4; i += 1) {
      const call: ToolCall = { id: `e${i}`, name: "x", input: {} };
      a.push({ kind: "tool_call_requested", calls: [call], usage: usage() } as TurnResult);
    }
    const registry = ExecutionRegistry.builder()
      .agent("", a)
      .toolset("", new EmptyToolRegistry())
      .schema("plan-schema", {})
      .build();
    const cfg: HarnessConfig = {
      registry,
      toolRegistry: toolReg(4),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      escalationMode: surfaceToHuman,
    };
    const h = new StandardHarness(cfg);
    const strategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
        output: "plan-schema",
      },
      execute: { kind: "react", budget: { kind: "per_loop", value: 2 }, agent: "", toolset: "" },
    };
    const r = await h.run({ task: newTask("build", SessionId.of("pe"), strategy) });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind === "waiting_for_human" && r.request.kind === "budget_exhausted") {
      expect(r.request.phase).toBe("plan_execute");
      // A combinator offers all three actions (fork C).
      const kinds = r.request.available_actions.map((x) => x.kind);
      expect(kinds).toContain("skip");
      expect(kinds).toContain("fail");
      expect(kinds).toContain("continue_with_budget");
    }
  });
});

// ============================================================================
// Autonomous: no pause; existing propagate behavior
// ============================================================================

describe("Autonomous — no pause; propagate budget_exceeded", () => {
  it("a binding leaf cap fails with budget_exceeded (turns = leaf cap)", async () => {
    const cfg = autonomousConfig(budgetExhaustingAgent(3));
    cfg.toolRegistry = toolReg(3);
    const h = new StandardHarness(cfg);
    const r = await h.run({ task: react(2) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("budget_exceeded");
      expect(r.turns).toBe(2);
    }
  });
});

// ============================================================================
// Resume paths
// ============================================================================

async function pauseAtLeafCap(agent: MockAgent): Promise<PausedState> {
  const cfg = surfaceConfig(agent);
  cfg.toolRegistry = toolReg(3);
  const h = new StandardHarness(cfg);
  const r = await h.run({ task: react(2) });
  if (r.kind !== "waiting_for_human") throw new Error(`expected pause, got ${r.kind}`);
  return r.state;
}

describe("Resume — ContinueWithBudget grants N steps and resumes", () => {
  it("a granted continue runs the loop to Success", async () => {
    const a = makeAgent();
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
    a.push({ kind: "final_response", content: "finished after grant", usage: usage() });
    const state = await pauseAtLeafCap(a);
    const cfg = surfaceConfig(a);
    cfg.toolRegistry = toolReg(3);
    const h = new StandardHarness(cfg);
    const resumed = await h.resume(state, {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 5 },
    });
    expect(resumed.kind).toBe("success");
    if (resumed.kind === "success") expect(resumed.output).toBe("finished after grant");
  });
});

describe("Resume — Fail propagates budget_exceeded, partial discarded", () => {
  it("fails with budget_exceeded and an empty session state", async () => {
    const state = await pauseAtLeafCap(budgetExhaustingAgent(3));
    const h = new StandardHarness(surfaceConfig(makeAgent()));
    const resumed = await h.resume(state, { kind: "escalate", action: { kind: "fail" } });
    expect(resumed.kind).toBe("failure");
    if (resumed.kind === "failure") {
      expect(resumed.reason.kind).toBe("budget_exceeded");
      // Fail contract: the partial is discarded.
      expect(resumed.session_state?.messages ?? []).toHaveLength(0);
    }
  });
});

describe("Resume — Skip on a leaf resolves to clean Success", () => {
  it("a leaf skip yields Success (no sibling to advance to)", async () => {
    const state = await pauseAtLeafCap(budgetExhaustingAgent(3));
    const h = new StandardHarness(surfaceConfig(makeAgent()));
    const resumed = await h.resume(state, { kind: "escalate", action: { kind: "skip" } });
    expect(resumed.kind).toBe("success");
  });
});

describe("Resume — Skip on a PlanExecute advances the outer loop", () => {
  it("does NOT re-pause; re-enters the loop and runs to a terminal", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: "advanced", usage: usage() });
    const registry = ExecutionRegistry.builder()
      .agent("", a)
      .toolset("", new EmptyToolRegistry())
      .build();
    const cfg: HarnessConfig = {
      registry,
      toolRegistry: toolReg(0),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      escalationMode: surfaceToHuman,
    };
    const h = new StandardHarness(cfg);
    const pe: LoopStrategy = {
      kind: "plan_execute",
      plan: { kind: "react", budget: { kind: "unlimited" }, agent: "", toolset: "" },
      execute: { kind: "react", budget: { kind: "unlimited" }, agent: "", toolset: "" },
    };
    const t = newTask("build", SessionId.of("s1"), pe, { max_turns: 8 });
    const state: PausedState = {
      session_id: SessionId.of("s1"),
      task_id: t.id,
      turn_number: 1,
      session_state: emptySessionState(),
      pending_tool_calls: [],
      approved_results: [],
      human_request: {
        kind: "budget_exhausted",
        phase: "plan_execute",
        policy: { kind: "total_steps", value: 1 },
        steps_taken: 1,
        continues_used: 0,
        partial_output: reactPartialJson(""),
        available_actions: [
          { kind: "continue_with_budget", steps: 1 },
          { kind: "skip" },
          { kind: "fail" },
        ],
      },
      task: t,
      budget_used: emptyBudgetSnapshot(),
      child_state: null,
      toolset: "",
    };
    const resumed = await h.resume(state, { kind: "escalate", action: { kind: "skip" } });
    // Skip re-entered the PlanExecute loop rather than re-pausing.
    expect(resumed.kind).not.toBe("waiting_for_human");
  });
});

// ============================================================================
// grantTaskBudget raises caps without shrinking
// ============================================================================

describe("grantTaskBudget", () => {
  it("raises the leaf and global caps, never shrinks", () => {
    let t = grantTaskBudget(react(2), 9);
    expect(t.budget.max_turns).toBe(9);
    expect(t.loop_strategy.kind === "react" && t.loop_strategy.budget).toEqual({
      kind: "per_loop",
      value: 9,
    });
    // A lower grant never SHRINKS an existing allowance.
    t = grantTaskBudget(t, 3);
    expect(t.budget.max_turns).toBe(9);
    expect(t.loop_strategy.kind === "react" && t.loop_strategy.budget).toEqual({
      kind: "per_loop",
      value: 9,
    });
  });

  it("recurses into PlanExecute children", () => {
    const pe: LoopStrategy = {
      kind: "plan_execute",
      plan: { kind: "react", budget: { kind: "per_loop", value: 2 }, agent: "", toolset: "" },
      execute: { kind: "react", budget: { kind: "per_loop", value: 3 }, agent: "", toolset: "" },
    };
    const t = grantTaskBudget(newTask("x", SessionId.of("s"), pe), 9);
    if (t.loop_strategy.kind === "plan_execute") {
      expect(t.loop_strategy.plan).toEqual({
        kind: "react",
        budget: { kind: "per_loop", value: 9 },
        agent: "",
        toolset: "",
      });
      expect(t.loop_strategy.execute.kind === "react" && t.loop_strategy.execute.budget).toEqual({
        kind: "per_loop",
        value: 9,
      });
    }
  });

  it("leaves unlimited policies untouched", () => {
    const t = grantTaskBudget(
      newTask("x", SessionId.of("s"), {
        kind: "react",
        budget: { kind: "unlimited" },
        agent: "",
        toolset: "",
      }),
      9,
    );
    expect(t.loop_strategy.kind === "react" && t.loop_strategy.budget).toEqual({
      kind: "unlimited",
    });
  });
});
