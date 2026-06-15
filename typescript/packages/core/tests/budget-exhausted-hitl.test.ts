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
  leafEscalationActions,
  newTask,
  promoteBudgetExhaustedToHuman,
  reactPartialJson,
  surfaceToHuman,
  type Agent,
  type BudgetExhausted,
  type Content,
  type Context,
  type EscalationAction,
  type HarnessConfig,
  type HumanRequest,
  type HumanResponse,
  type LoopStrategy,
  type Message,
  type PausedState,
  type SessionState,
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
import { InMemoryStorageProvider, StorageProvider } from "../src/storage/index.js";
import { addTask, completeTask, defaultTaskList, updateTask } from "../src/tasklist/index.js";

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
      // #138 AC2-a: the pause now carries the FULL stalled worker session
      // (instruction + the budget-burning tool-call rounds), NOT a single
      // partial-only assistant stub — so a budget resume re-attaches the real
      // worker context. The session therefore has MORE than one message.
      expect(r.state.session_state.messages.length).toBeGreaterThan(1);
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

// ============================================================================
// #138 AC2-a: promoteBudgetExhaustedToHuman carries the FULL worker session
// ============================================================================

describe("#138 AC2-a — promoteBudgetExhaustedToHuman carries the full worker session", () => {
  // The pause helper carries the FULL stalled worker session (AC2-a) and the
  // worker leaf's toolset handle (AC4-a) — not a partial-only stub. A direct unit
  // on the boundary helper, decoupled from the surrounding strategy.
  it("carries the full worker session + the worker leaf's toolset handle", () => {
    const err: BudgetExhausted = {
      policy: { kind: "per_loop", value: 2 },
      behavior: { kind: "escalate" },
      stepsTaken: 2,
      continuesUsed: 0,
      phase: "react",
    };
    // A realistic worker conversation (instruction + a tool round).
    const worker: SessionState = {
      messages: [
        { role: "user", content: { type: "text", text: "worker: audit" } as Content },
        { role: "assistant", content: { type: "text", text: "looking" } as Content },
        { role: "tool", content: { type: "text", text: "listing" } as Content },
      ] as Message[],
      extras: {},
    };
    const waiting = promoteBudgetExhaustedToHuman(
      err,
      "partial",
      leafEscalationActions(err),
      SessionId.of("s1"),
      react(2),
      emptyBudgetSnapshot(),
      2,
      worker,
      "exec-tools",
    );
    expect(waiting.kind).toBe("waiting_for_human");
    if (waiting.kind === "waiting_for_human") {
      // AC2-a: the FULL worker session is carried (3 messages), NOT the single
      // partial-only assistant stub.
      expect(waiting.state.session_state.messages).toEqual(worker.messages);
      // AC4-a: the worker leaf's toolset handle rides the pause (#140 parity).
      expect(waiting.state.toolset).toBe("exec-tools");
    }

    // Back-compat: an EMPTY worker session falls back to the partial-only stub
    // (the pre-#138 behavior) so legacy/HillClimbing sites are unchanged.
    const waiting2 = promoteBudgetExhaustedToHuman(
      err,
      "just-the-partial",
      leafEscalationActions(err),
      SessionId.of("s1"),
      react(2),
      emptyBudgetSnapshot(),
      2,
      emptySessionState(),
      "",
    );
    if (waiting2.kind === "waiting_for_human") {
      expect(waiting2.state.session_state.messages).toHaveLength(1);
      const head = waiting2.state.session_state.messages[0]!;
      expect(head.content.type === "text" && head.content.text).toBe("just-the-partial");
    }
  });
});

// ============================================================================
// #138 AC1: skip-plan reconciles already-Completed tasks (dedup)
// ============================================================================

describe("#138 AC1 — skip-plan reconciles already-Completed tasks", () => {
  // A non-empty durable task_list whose task #1 is already Completed: a fresh run
  // SKIPS the plan phase (AC1) and reconcile does NOT re-run the completed task —
  // only the still-Pending task #2 runs (one model call, no plan turn).
  it("skips the plan phase and dedups the completed task", async () => {
    const a = makeAgent();
    // NO plan turn pushed (AC1 skips it). Only task #2 runs.
    a.push({ kind: "final_response", content: "did two", usage: usage() });
    const session = SessionId.of("s1");
    const storage = StorageProvider.single(new InMemoryStorageProvider());
    // The plan slot is STRUCTURED — its leaf needs an output schema registered
    // (#124 A.5), even though AC1 skips the plan phase at runtime.
    const registry = ExecutionRegistry.builder()
      .agent("", a)
      .toolset("", new EmptyToolRegistry())
      .schema("plan-schema", { type: "object" })
      .build();
    const cfg: HarnessConfig = {
      registry,
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      escalationMode: surfaceToHuman,
      storage,
    };
    const h = new StandardHarness(cfg);

    // Pre-seed: task #1 already Completed, task #2 Pending.
    const tl = defaultTaskList();
    addTask(tl, "one", []); // 1
    addTask(tl, "two", []); // 2
    updateTask(tl, 1, "in_progress");
    completeTask(tl, 1);
    await h.persistTaskList(session, tl);

    const planStrategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 12 },
        agent: "",
        toolset: "",
        output: "plan-schema",
      },
      execute: { kind: "react", budget: { kind: "per_loop", value: 12 }, agent: "", toolset: "" },
    };
    const r = await h.run({ task: newTask("audit", session, planStrategy) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("did two");

    // Exactly ONE model call: task #2 (no plan turn, task #1 not re-run).
    expect(a.callCount).toBe(1);

    // Both tasks are Completed in the durable store (1 deduped, 2 freshly run).
    const stored = (await storage.run().get(h.projectId().namespace(), "task_list")) as {
      tasks: { status: string }[];
    };
    expect(stored.tasks.every((t) => t.status === "completed")).toBe(true);
  });
});

// ============================================================================
// #138 AC3: plan-phase exhaustion resumes the PLAN session from the carried conv
// ============================================================================

describe("#138 AC3 — plan-phase exhaustion seeds the plan session from the carried conversation", () => {
  /** Scripts one final response per call and records every Context it sees. */
  class RecordingPlanner implements Agent {
    readonly contexts: Context[] = [];
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
      this.contexts.push(ctx);
      return this.results.shift() ?? { kind: "final_response", content: "done", usage: usage() };
    }
  }

  function ctxText(ctx: Context): string {
    return ctx.messages.map((m) => (m.content.type === "text" ? m.content.text : "")).join("\n");
  }

  // When a budget resume carries a worker session AND the durable task_list is
  // EMPTY (no `in_progress` task ⇒ the exhaustion happened in the PLAN phase),
  // `runPlanExecuteConfig` seeds the PLAN session from the carried conversation
  // instead of a fresh base session — so the planner CONTINUES on it. Observed via
  // the planner agent's RECORDED contexts (the replay harness would not otherwise
  // surface it through NoopContextManager).
  it("the plan session is seeded from the carried conversation", async () => {
    const planner = new RecordingPlanner(AgentId.of("planner")).push({
      kind: "final_response",
      content: '{"tasks":["only"],"rationale":"r"}',
      usage: usage(),
    });
    const worker = makeAgent();
    worker.push({ kind: "final_response", content: "did the work", usage: usage() });

    const storage = StorageProvider.single(new InMemoryStorageProvider());
    const registry = ExecutionRegistry.builder()
      .agent("planner", planner)
      .agent("", worker)
      .build();
    const cfg: HarnessConfig = {
      registry,
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      escalationMode: surfaceToHuman,
      storage,
    };
    const h = new StandardHarness(cfg);

    // A PlanExecute whose PLAN leaf resolves to "planner"; execute is a bare ReAct
    // on the default key.
    const pe: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 12 },
        agent: "planner",
        toolset: "",
        output: "",
      },
      execute: { kind: "react", budget: { kind: "per_loop", value: 8 }, agent: "", toolset: "" },
    };
    const t = newTask("audit", SessionId.of("s1"), pe, { max_turns: 32 });

    // A budget-exhausted pause carrying a worker session with a MARKER, and NO
    // durable task_list persisted (empty ⇒ plan-phase exhaustion, AC3).
    const MARKER = "CARRIED_PLAN_SESSION_MARKER";
    const carried: SessionState = {
      messages: [
        { role: "assistant", content: { type: "text", text: MARKER } as Content },
      ] as Message[],
      extras: {},
    };
    const state: PausedState = {
      session_id: SessionId.of("s1"),
      task_id: t.id,
      turn_number: 1,
      session_state: carried,
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

    await h.resume(state, {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 10 },
    });

    // AC3: the planner's FIRST context was seeded from the CARRIED session — the
    // marker is present, proving the plan session continued on it rather than
    // starting from a fresh base session.
    expect(planner.contexts.length).toBeGreaterThan(0);
    expect(ctxText(planner.contexts[0]!)).toContain(MARKER);
  });
});

// ============================================================================
// #144: PlanExecute budget-resume advances on the worker's consumed turns
// ============================================================================

describe("#144 — execute-phase budget grant advances and makes forward progress", () => {
  // Regression (cfffa40): a PlanExecute execute-phase worker that exhausts under
  // SurfaceToHuman and is granted more budget must make FORWARD PROGRESS across
  // grants — the run-wide turn cursor and the grant's `steps_taken` both STRICTLY
  // advance, and the worker eventually finishes. Before the fix the exhaustion
  // branch discarded the execute leaf's consumed-turn count and re-read
  // `stepsTaken = 0` from the (unlimited, because `max_turns` is unset)
  // `plan_execute` scope, and left `carried.turns` frozen at its pre-task value.
  // So every grant computed `granted = 0 + steps` (a no-op that never widened the
  // binding cap) and re-seeded the worker on the SAME window — the cursor and
  // `steps_taken` stuck, the worker re-ran the same turns, and a review-style
  // task thrashed through every auto-continue making zero progress.
  it("the turn cursor and steps_taken strictly advance across grants, then completes", async () => {
    // One execute task ("only"); the execute leaf cap is PerLoop{2} and the top
    // task leaves `max_turns` unset, so the `plan_execute` scope is unlimited —
    // the exact configuration that zeroed the grant accounting.
    const a = makeAgent();
    a.push({ kind: "final_response", content: '{"tasks":["only"]}', usage: usage() }); // plan turn
    // Execute worker keeps requesting tools so it never finishes until the final
    // grant; each tool_call_requested is one turn against the leaf cap.
    for (let i = 0; i < 3; i += 1) {
      const call: ToolCall = { id: `e${i}`, name: "x", input: {} };
      a.push({ kind: "tool_call_requested", calls: [call], usage: usage() } as TurnResult);
    }
    a.push({ kind: "final_response", content: "done only", usage: usage() }); // completes after 2nd grant

    // A real in-memory store (NOT the no-op default): the durable task-list path
    // must survive run -> resume -> resume, otherwise the in-progress task is
    // dropped and the test asserts the wrong path.
    const storage = StorageProvider.single(new InMemoryStorageProvider());
    const cfg = surfaceConfig(a);
    cfg.toolRegistry = toolReg(3);
    cfg.storage = storage;
    const h = new StandardHarness(cfg);

    const pe: LoopStrategy = {
      kind: "plan_execute",
      // plan: a structured ReAct leaf (output schema "" is registered by
      // registryWith); effectively unlimited so the plan turn never binds.
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
        output: "",
      },
      execute: { kind: "react", budget: { kind: "per_loop", value: 2 }, agent: "", toolset: "" },
    };
    // max_turns unset -> the plan_execute combinator scope is unlimited.
    const t = newTask("build", SessionId.of("s1"), pe);

    // First run -> pause #1. The worker consumed real turns past the plan floor,
    // so the cursor and `steps_taken` reflect that (NOT 0 / frozen).
    const r1 = await h.run({ task: t });
    expect(r1.kind).toBe("waiting_for_human");
    if (r1.kind !== "waiting_for_human") throw new Error(`expected pause #1, got ${r1.kind}`);
    expect(r1.request.kind).toBe("budget_exhausted");
    if (r1.request.kind !== "budget_exhausted") throw new Error("expected budget_exhausted");
    // The combinator scope owns the pause.
    expect(r1.request.phase).toBe("plan_execute");
    const turn1 = r1.state.turn_number;
    const steps1 = r1.request.steps_taken;
    // The bug froze the cursor at the pre-task value (1 = just the plan turn) and
    // read steps_taken=0 from the unlimited scope.
    expect(turn1).toBeGreaterThanOrEqual(2);
    expect(steps1).toBeGreaterThanOrEqual(2);

    // Grant +2 and resume -> pause #2. Forward progress: the cursor and
    // `steps_taken` must STRICTLY advance (the bug re-seeded the same window).
    const r2 = await h.resume(r1.state, {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 2 },
    });
    expect(r2.kind).toBe("waiting_for_human");
    if (r2.kind !== "waiting_for_human") throw new Error(`expected pause #2, got ${r2.kind}`);
    expect(r2.request.kind).toBe("budget_exhausted");
    if (r2.request.kind !== "budget_exhausted") throw new Error("expected budget_exhausted");
    const turn2 = r2.state.turn_number;
    const steps2 = r2.request.steps_taken;
    expect(turn2).toBeGreaterThan(turn1); // turn cursor advanced across grants
    expect(steps2).toBeGreaterThan(steps1); // steps_taken advanced across grants

    // Grant +2 again and resume -> Success: the worker finishes, having made real
    // progress rather than looping on the same window.
    const r3 = await h.resume(r2.state, {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 2 },
    });
    expect(r3.kind).toBe("success");
    if (r3.kind === "success") expect(r3.output).toBe("done only");
  });
});
