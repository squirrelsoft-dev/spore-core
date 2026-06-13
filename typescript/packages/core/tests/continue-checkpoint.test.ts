/**
 * #129 — Continue cross-process checkpoint (`continues_used` survives a pause).
 *
 * Mirrors the Rust `#129` test module in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same verdicts, parallel
 * structure. Covers:
 *   - AC1: the SHARED checkpoint utility (`serializeCheckpoint`/`loadCheckpoint`)
 *     round-trips a PausedState (Continue AND Ralph paused states);
 *   - AC2 (load-bearing): a Continue spanning a process pause resumes with the
 *     correct `continues_used` then falls through to `on_exhausted` after the
 *     REMAINING continues (end-to-end + the `BudgetContext.resumed` unit core);
 *   - the live in-process Continue completing without a pause;
 *   - AC3: the in-process Continue performs no serialization (no pause emitted);
 *   - AC4: a Continue resume PRESERVES the prior session context;
 *   - the `behavior` wire shape (Q1) and the new `continue_checkpoint.json`
 *     fixture replay (byte-identity ground truth);
 *   - nested-leaf-propagates vs bare-leaf-honors-behavior (Q1).
 *
 * Never edit a fixture to make a failing implementation pass.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  BudgetContext,
  HumanRequestSchema,
  MockAgent,
  SessionId,
  StandardHarness,
  TaskId,
  TaskSchema,
  autonomous,
  loadCheckpoint,
  serializeCheckpoint,
  surfaceToHuman,
  type BudgetExhaustedBehavior,
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

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..", "..", "..");
const fixturePath = (rel: string): string => resolve(repoRoot, "fixtures", rel);

// ── config helpers (parallel to budget-exhausted-hitl.test.ts) ──────────────

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function makeAgent(): MockAgent {
  return new MockAgent(AgentId.of("test"));
}

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

function autonomousConfig(agent: MockAgent): HarnessConfig {
  return { ...surfaceConfig(agent), escalationMode: autonomous };
}

/** A bare ReAct leaf agent that ALWAYS requests a tool (never finishes). */
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

const CONTINUE_FAIL: BudgetExhaustedBehavior = {
  kind: "continue",
  max_continues: 2,
  on_exhausted: { kind: "fail" },
};

/**
 * The canonical `continue_checkpoint.json` value: a bare ReAct leaf carrying
 * `continue{max_continues:2, on_exhausted:fail}` paused mid-loop with
 * `continues_used: 1` on its `budget_exhausted` request. Cross-language
 * byte-identity ground truth for the AC2 resume test.
 */
function continueCheckpointPausedState(): PausedState {
  const task: Task = {
    id: TaskId.of("task-129"),
    instruction: "iterate on the patch",
    session_id: SessionId.of("sess-129"),
    // Matches the fixture's explicit-null budget so the loadCheckpoint round-trip
    // (which deserializes via the schema) is structurally equal.
    budget: {
      max_turns: null,
      max_input_tokens: null,
      max_output_tokens: null,
      max_wall_time: null,
      max_cost_usd: null,
    },
    loop_strategy: {
      kind: "react",
      budget: { kind: "per_loop", value: 3 },
      behavior: CONTINUE_FAIL,
      agent: "worker",
      toolset: "patch-tools",
    },
  };
  return {
    session_id: SessionId.of("sess-129"),
    task_id: TaskId.of("task-129"),
    turn_number: 3,
    session_state: {
      messages: [
        {
          role: "assistant",
          content: { type: "text", text: '{"node":"react","last":""}' },
        },
      ],
      extras: {},
    },
    pending_tool_calls: [],
    approved_results: [],
    human_request: {
      kind: "budget_exhausted",
      phase: "react",
      policy: { kind: "per_loop", value: 3 },
      steps_taken: 3,
      continues_used: 1,
      partial_output: '{"node":"react","last":""}',
      available_actions: [{ kind: "continue_with_budget", steps: 3 }, { kind: "fail" }],
    },
    task,
    budget_used: { turns: 3, input_tokens: 0, output_tokens: 0, wall_time: null, cost_usd: 0 },
    child_state: null,
    // #140: matches the fixture's `"toolset": ""` so the loadCheckpoint round-trip
    // is structurally equal.
    toolset: "",
  };
}

// ============================================================================
// AC2 — wire: continue_checkpoint.json fixture replay (byte-identity)
// ============================================================================

describe("continue_checkpoint.json fixture replay (#129)", () => {
  const raw = JSON.parse(
    readFileSync(fixturePath("paused_states/continue_checkpoint.json"), "utf-8"),
  ) as Record<string, unknown>;

  it("the envelope + human_request match field-for-field", () => {
    expect(raw.session_id).toBe("sess-129");
    expect(raw.task_id).toBe("task-129");
    expect(raw.turn_number).toBe(3);
    const req = HumanRequestSchema.parse(raw.human_request);
    expect(req.kind).toBe("budget_exhausted");
    if (req.kind !== "budget_exhausted") throw new Error("not budget_exhausted");
    expect(req.phase).toBe("react");
    expect(req.policy).toEqual({ kind: "per_loop", value: 3 });
    expect(req.steps_taken).toBe(3);
    expect(req.continues_used).toBe(1);
    expect(req.partial_output).toBe('{"node":"react","last":""}');
    expect(req.available_actions).toEqual([
      { kind: "continue_with_budget", steps: 3 },
      { kind: "fail" },
    ]);
  });

  it("the embedded task parses with a Continue-behavior ReAct leaf", () => {
    const task = TaskSchema.parse(raw.task);
    expect(task.loop_strategy.kind).toBe("react");
    if (task.loop_strategy.kind !== "react") throw new Error("not react");
    expect(task.loop_strategy.behavior).toEqual(CONTINUE_FAIL);
  });

  it("loadCheckpoint round-trips the fixture to an EQUAL PausedState", () => {
    const restored = loadCheckpoint(
      readFileSync(fixturePath("paused_states/continue_checkpoint.json"), "utf-8"),
    );
    expect(restored).toEqual(continueCheckpointPausedState());
  });
});

// ============================================================================
// AC1 — the SHARED checkpoint utility round-trips
// ============================================================================

describe("shared checkpoint utility round-trips (#129, AC1)", () => {
  it("a Continue paused-state (with human_request) round-trips", () => {
    const state = continueCheckpointPausedState();
    const blob = serializeCheckpoint(state);
    expect(loadCheckpoint(blob)).toEqual(state);
  });

  it("a Ralph-style paused-state (no human_request) ALSO round-trips through the same utility", () => {
    // Proving the seam is shared, not Continue-specific.
    const state: PausedState = {
      ...continueCheckpointPausedState(),
      human_request: undefined,
    };
    const blob = serializeCheckpoint(state);
    expect(loadCheckpoint(blob)).toEqual(state);
  });
});

// ============================================================================
// AC2 — resume preserves continues_used (load-bearing)
// ============================================================================

describe("resume preserves continues_used then falls through (#129, AC2)", () => {
  /**
   * The unit core: `BudgetContext.resumed` seeds `continuesUsed` (NOT 0), and a
   * subsequent exhaustion falls through to `on_exhausted` after the REMAINING
   * continues — not a refreshed `max_continues`.
   */
  it("BudgetContext.resumed seeds continuesUsed and bounds the chain", () => {
    const scope = BudgetContext.resumed({ kind: "per_loop", value: 3 }, CONTINUE_FAIL, "react", 1);
    expect(scope.continuesUsed).toBe(1); // seeded, NOT zeroed
    expect(scope.stepsTaken).toBe(0); // fresh per-round step budget
    expect(scope.continuesRemaining()).toBe(1); // only ONE continue remains
    expect(scope.resolveExhausted()).toBe("continue");
    expect(scope.continuesUsed).toBe(2);
    // Continues now spent → fall through to on_exhausted=fail.
    expect(scope.resolveExhausted()).toBe("fail");

    // Contrast: a FRESH (pre-#129) scope would grant TWO continues — the bug.
    const fresh = new BudgetContext({ kind: "per_loop", value: 3 }, CONTINUE_FAIL, "react");
    expect(fresh.continuesRemaining()).toBe(2);
  });

  /**
   * End-to-end (load-bearing): a Continue that SPANS a process pause resumes with
   * the correct `continues_used` (NOT 0). FAILS on pre-#129 code, which zeroed
   * the counter and granted MORE in-process continues than `max_continues`.
   *
   * Setup: `continue{max_continues:2, on_exhausted:fail}`, checkpoint records
   * `continues_used: 1` (ONE spent). The worker never finishes, so every granted
   * window re-exhausts. Granted per-window cap is `steps_taken + steps = 3 + 1 =
   * 4`. On resume the operator's grant runs window A; with one in-process continue
   * left, window B runs, then fail → 2 windows × 4 = 8 turns. The bug (zeroed
   * counter) would grant a THIRD window → 12 turns. Assert turns === 8.
   */
  it("resumes with one continue left → 2 windows → 8 turns (bug would be 12)", async () => {
    const a = budgetExhaustingAgent(40);
    const cfg = surfaceConfig(a);
    cfg.toolRegistry = toolReg(40);
    const h = new StandardHarness(cfg);

    const state = continueCheckpointPausedState();
    // Bare leaf resolving via the default ("") registry handle on resume.
    state.task.loop_strategy = {
      kind: "react",
      budget: { kind: "per_loop", value: 3 },
      behavior: CONTINUE_FAIL,
      agent: "",
      toolset: "",
    };

    const resumed = await h.resume(state, {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 1 },
    });
    expect(resumed.kind).toBe("failure");
    if (resumed.kind !== "failure") return;
    expect(resumed.reason.kind).toBe("budget_exceeded");
    expect(resumed.turns).toBe(8);
  });
});

// ============================================================================
// Live in-process Continue (AC3: no serialization)
// ============================================================================

describe("live in-process Continue (#129)", () => {
  it("exhausts, gets a granted continue, loops, and completes — no pause", async () => {
    const a = makeAgent();
    // First window: 2 tool turns exhaust the per_loop{2} cap → Continue grant.
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
    // After the in-process continue refreshes the cap, the worker completes.
    a.push({ kind: "final_response", content: "done after in-process continue", usage: usage() });

    // Autonomous so an Escalate fall-through would NOT pause — proving the success
    // came from the Continue loop, not a HITL pause (AC3: no serialization).
    const cfg = autonomousConfig(a);
    cfg.toolRegistry = toolReg(3);
    const h = new StandardHarness(cfg);
    const strategy: LoopStrategy = {
      kind: "react",
      budget: { kind: "per_loop", value: 2 },
      behavior: { kind: "continue", max_continues: 1, on_exhausted: { kind: "fail" } },
      agent: "",
      toolset: "",
    };
    const r = await h.run({ task: newReactTask(strategy) });
    // AC3: no serialization on the in-process path — the run terminates directly,
    // never emitting a waiting_for_human PausedState.
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("done after in-process continue");
  });
});

// ============================================================================
// AC4 — Continue resume preserves session context
// ============================================================================

describe("Continue resume preserves session context (#129, AC4)", () => {
  it("builds ON the prior history (does not start from an empty session)", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: "resumed with context", usage: usage() });
    const cfg = surfaceConfig(a);
    cfg.toolRegistry = toolReg(1);
    const h = new StandardHarness(cfg);

    const state = continueCheckpointPausedState();
    state.task.loop_strategy = {
      kind: "react",
      budget: { kind: "per_loop", value: 3 },
      behavior: CONTINUE_FAIL,
      agent: "",
      toolset: "",
    };
    const priorMsgCount = state.session_state.messages.length;
    expect(priorMsgCount).toBeGreaterThanOrEqual(1);

    const resumed = await h.resume(state, {
      kind: "escalate",
      action: { kind: "continue_with_budget", steps: 3 },
    });
    expect(resumed.kind).toBe("success");
    if (resumed.kind !== "success") return;
    expect(resumed.output).toBe("resumed with context");
    // The resumed session retains the prior assistant message AND the new turn —
    // it did NOT start from an empty (re-seeded) session.
    expect((resumed.session_state?.messages ?? []).length).toBeGreaterThan(priorMsgCount);
  });
});

// ============================================================================
// Q1 — nested leaf propagates vs bare leaf honors behavior
// ============================================================================

describe("leaf behavior resolution (#129, Q1)", () => {
  /**
   * A BARE top-level leaf HONORS its `behavior: fail`: it resolves to a
   * budget_exceeded Failure with the partial DISCARDED — distinct from the
   * default `escalate` placeholder which (under surface_to_human) would PAUSE.
   */
  it("a bare leaf honors `fail` (partial discarded, no pause)", async () => {
    const cfg = surfaceConfig(budgetExhaustingAgent(3));
    cfg.toolRegistry = toolReg(3);
    const h = new StandardHarness(cfg);
    const strategy: LoopStrategy = {
      kind: "react",
      budget: { kind: "per_loop", value: 2 },
      behavior: { kind: "fail" },
      agent: "",
      toolset: "",
    };
    const r = await h.run({ task: newReactTask(strategy) });
    expect(r.kind).toBe("failure");
    if (r.kind !== "failure") return;
    expect(r.reason.kind).toBe("budget_exceeded");
    // Fail contract: the partial is DISCARDED (the bare leaf honored `fail`, not
    // the propagate/escalate path that would PAUSE under surface_to_human).
    expect(r.session_state?.messages ?? []).toHaveLength(0);
  });

  /**
   * A NESTED leaf does NOT self-resolve its `continue` behavior: it propagates to
   * the parent, so the PARENT combinator's `behavior` (default escalate) governs.
   * The nested leaf's exhaustion surfaces as the PARENT's `plan_execute` pause.
   */
  it("a nested leaf propagates (parent plan_execute pauses, not a leaf react)", async () => {
    const a = makeAgent();
    a.push({ kind: "final_response", content: '{"tasks":["x"]}', usage: usage() }); // plan
    for (let i = 0; i < 4; i += 1) {
      a.push({
        kind: "tool_call_requested",
        calls: [{ id: `e${i}`, name: "x", input: {} }],
        usage: usage(),
      } as TurnResult);
    }
    const cfg = surfaceConfig(a);
    cfg.toolRegistry = toolReg(4);
    const h = new StandardHarness(cfg);
    // The execute leaf carries `continue` — but NESTED, so the leaf must NOT
    // self-resolve it; the PlanExecute parent (escalate default) pauses.
    const strategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 1000000 },
        agent: "",
        toolset: "",
        output: "",
      },
      execute: {
        kind: "react",
        budget: { kind: "per_loop", value: 2 },
        behavior: { kind: "continue", max_continues: 1, on_exhausted: { kind: "fail" } },
        agent: "",
        toolset: "",
      },
    };
    const t: Task = {
      id: TaskId.generate(),
      instruction: "build",
      session_id: SessionId.of("s1"),
      budget: {},
      loop_strategy: strategy,
    };
    const r = await h.run({ task: t });
    expect(r.kind).toBe("waiting_for_human");
    if (r.kind !== "waiting_for_human") return;
    expect(r.state.human_request?.kind).toBe("budget_exhausted");
    if (r.state.human_request?.kind !== "budget_exhausted") return;
    // The PARENT resolved (phase == plan_execute); the nested leaf did NOT
    // self-resolve its own `continue` at the "react" phase.
    expect(r.state.human_request.phase).toBe("plan_execute");
  });
});

// ── local helper ────────────────────────────────────────────────────────────

function newReactTask(strategy: LoopStrategy): Task {
  return {
    id: TaskId.generate(),
    instruction: "do something",
    session_id: SessionId.of("s1"),
    budget: {},
    loop_strategy: strategy,
  };
}
