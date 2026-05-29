/**
 * Unit tests for the PlanExecute plan phase + plan-artifact capture (issue #70).
 *
 * Mirrors `rust/crates/spore-core/src/plan.rs#tests` and the plan-phase tests in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same verdicts.
 *
 * Rules covered:
 *   R1  plan phase runs exactly once
 *   R2  one-shot (tool call → plan_phase_failed / planning_turn_failed)
 *   R3  artifact captured from response text
 *   R4  stored in extras["plan_execute"]
 *   R5  plannerAgent set → planner ran, default did not
 *   R6  plannerAgent unset → default agent ran
 *   R7  plan turn counts against the budget
 *   R8  one turn span recorded
 *   R9  capture deterministic & total
 *   R10 budget exhausted before plan turn → failure, no artifact
 *   R11 on_plan_created mutation reflected in stored artifact
 *   Q4  PlanExecute hands the stored artifact off to the execute phase (#59)
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyResponse,
  PLAN_EXECUTE_EXTRAS_KEY,
  SessionId,
  StandardHarness,
  capturePlanArtifact,
  emptySessionState,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type SessionState,
  type TokenUsage,
  type TurnResult,
} from "../src/index.js";
import type { Hook, HookChain, HookContext, HookDecision, HookEvent } from "../src/hooks/index.js";
import { StandardHookChain } from "../src/hooks/standard.js";
import { InMemoryObservabilityProvider } from "../src/observability/in-memory.js";
import { InMemoryStorageProvider, StorageProvider } from "../src/storage/index.js";
import { TASK_LIST_EXTRAS_KEY } from "../src/tasklist/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

const PLAN_STRATEGY: LoopStrategy = { kind: "plan_execute", plan_model: null };

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

function tcr(): TurnResult {
  return {
    kind: "tool_call_requested",
    calls: [{ id: "c1", name: "x", input: {} }],
    usage: usage(),
  };
}

/** A recording planner: each `turn` pops the next scripted result and records
 *  that it was invoked, so R5/R6 can assert which agent ran. */
class RecordingAgent implements Agent {
  ran = 0;
  private readonly results: TurnResult[] = [];
  constructor(private readonly agentId: AgentId) {}
  push(r: TurnResult): this {
    this.results.push(r);
    return this;
  }
  id(): AgentId {
    return this.agentId;
  }
  async turn(_ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.ran += 1;
    const next = this.results.shift();
    if (next == null) return { kind: "error", error: new EmptyResponse(), usage: null };
    return next;
  }
}

function configWith(agent: Agent, overrides: Partial<HarnessConfig> = {}): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    ...overrides,
  };
}

const PLAN_SID = SessionId.of("plan-s1");

function planTask(instruction = "build a CLI", maxTurns?: number) {
  return newTask(instruction, PLAN_SID, PLAN_STRATEGY, {
    max_turns: maxTurns ?? null,
  });
}

/** #76: a fresh in-memory storage provider so tests can read the plan artifact
 *  back off the RunStore seam (it is no longer mirrored into
 *  `SessionState.extras`). */
function inMemoryStorage(): StorageProvider {
  return StorageProvider.single(new InMemoryStorageProvider());
}

const PLAN_JSON = '{"tasks":["a","b"],"rationale":"because"}';

/** A function hook for on_plan_created. */
function hook(events: HookEvent[], handle: Hook["handle"]): Hook {
  return {
    handle,
    events: () => events,
    name: () => "test-hook",
  };
}

// --------------------------------------------------------------------------
// capture grammar (R3 / R9) — byte-identical to plan.rs#tests
// --------------------------------------------------------------------------

describe("capturePlanArtifact — Q3 grammar", () => {
  it("R3/R9: captures a plain JSON object to exact tasks + rationale", () => {
    const r = capturePlanArtifact('{"tasks":["a","b","c"],"rationale":"because"}');
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.artifact.tasks).toEqual(["a", "b", "c"]);
      expect(r.artifact.rationale).toBe("because");
    }
  });

  it("trims surrounding ASCII whitespace", () => {
    const r = capturePlanArtifact('\n\t  {"tasks":["x"]}  \r\n');
    expect(r.ok && r.artifact.tasks).toEqual(["x"]);
    expect(r.ok && r.artifact.rationale).toBe("");
  });

  it("strips a ```json fence before parsing", () => {
    const r = capturePlanArtifact('```json\n{"tasks":["step 1","step 2"],"rationale":"r"}\n```');
    expect(r.ok && r.artifact.tasks).toEqual(["step 1", "step 2"]);
    expect(r.ok && r.artifact.rationale).toBe("r");
  });

  it("strips a bare ``` fence (no language tag)", () => {
    const r = capturePlanArtifact('```\n{"tasks":["only"]}\n```');
    expect(r.ok && r.artifact.tasks).toEqual(["only"]);
  });

  it("strips an uppercase ```JSON fence (language-tag agnostic)", () => {
    const r = capturePlanArtifact('```JSON\n{"tasks":["u"]}\n```');
    expect(r.ok && r.artifact.tasks).toEqual(["u"]);
  });

  it("rationale defaults to empty string", () => {
    const r = capturePlanArtifact('{"tasks":["a"]}');
    expect(r.ok && r.artifact.rationale).toBe("");
  });

  it("an empty tasks array is allowed", () => {
    const r = capturePlanArtifact('{"tasks":[]}');
    expect(r.ok && r.artifact.tasks).toEqual([]);
  });

  it("task strings are kept verbatim — no trimming or filtering", () => {
    const r = capturePlanArtifact('{"tasks":["  spaced  ",""]}');
    expect(r.ok && r.artifact.tasks).toEqual(["  spaced  ", ""]);
  });

  it("R9: malformed inputs return unparseable_plan, never throw", () => {
    for (const bad of [
      "not json at all",
      "[1,2,3]",
      '{"rationale":"x"}',
      '{"tasks":"a"}',
      '{"tasks":["a",2]}',
      '{"tasks":["a"],"rationale":5}',
      "   \n  ",
    ]) {
      const r = capturePlanArtifact(bad);
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.error.kind).toBe("unparseable_plan");
    }
  });

  it("R9: deterministic — identical input yields identical artifact", () => {
    const text = '```json\n{"tasks":["a","b"],"rationale":"r"}\n```';
    const a = capturePlanArtifact(text);
    const b = capturePlanArtifact(text);
    expect(a).toEqual(b);
  });
});

// --------------------------------------------------------------------------
// plan phase driver
// --------------------------------------------------------------------------

describe("PlanExecute plan phase", () => {
  it("Q4: produces+stores an artifact, then hands off to the execute phase", async () => {
    // Only the plan turn is scripted; the execute phase then drives the agent
    // again, exhausts the script, and aborts with step_failed (proving the
    // execute_phase_not_implemented halt is gone and the artifact was stored).
    const a = new RecordingAgent(AgentId.of("default")).push(fr(PLAN_JSON));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { storage }));
    const r = await h.run({ task: planTask(), session_state: state });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("step_failed");
    }
    // #76: the plan artifact was produced + persisted to the RunStore seam
    // before the execute phase ran (no longer mirrored into extras).
    expect(await storage.run().get(PLAN_SID, PLAN_EXECUTE_EXTRAS_KEY)).toEqual({
      tasks: ["a", "b"],
      rationale: "because",
    });
    // #76: neither persistence key is mirrored into SessionState.extras.
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  it("R1: the plan phase runs exactly once (single planner turn)", async () => {
    // Plan turn + one execute completion so the run progresses cleanly and only
    // the plan turn is attributable to the plan phase.
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["only"]}'))
      .push(fr("done"));
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    // 1 plan turn + 1 execute turn = 2 agent invocations.
    expect(a.ran).toBe(2);
  });

  it("R2: a tool call in the plan turn → plan_phase_failed (no dispatch loop)", async () => {
    const reg = new ScriptedToolRegistry();
    const a = new RecordingAgent(AgentId.of("default")).push(tcr());
    const h = new StandardHarness(configWith(a, { toolRegistry: reg }));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("plan_phase_failed");
      if (r.reason.kind === "plan_phase_failed") {
        expect(r.reason.error.kind).toBe("planning_turn_failed");
      }
    }
    expect(a.ran).toBe(1); // ran once, then stopped — no dispatch loop.
    expect(reg.callCount).toBe(0); // R2: no tool dispatch.
  });

  it("R3 + R4: artifact captured from response text and persisted to the RunStore", async () => {
    const a = new RecordingAgent(AgentId.of("default")).push(fr(PLAN_JSON));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { storage }));
    await h.run({ task: planTask(), session_state: state });
    const stored = await storage.run().get(PLAN_SID, PLAN_EXECUTE_EXTRAS_KEY);
    expect(stored).toEqual({ tasks: ["a", "b"], rationale: "because" });
    // #76: not mirrored into SessionState.extras.
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  it("R3: an unparseable plan surfaces plan_phase_failed / unparseable_plan, no artifact", async () => {
    const a = new RecordingAgent(AgentId.of("default")).push(fr("not json"));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { storage }));
    const r = await h.run({ task: planTask(), session_state: state });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "plan_phase_failed") {
      expect(r.reason.error.kind).toBe("unparseable_plan");
    }
    expect(await storage.run().get(PLAN_SID, PLAN_EXECUTE_EXTRAS_KEY)).toBeUndefined();
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  it("R5: when plannerAgent is set, the PLANNER runs the plan turn; the default runs the steps", async () => {
    // Single-task plan: the planner runs exactly the plan turn, the default runs
    // exactly the one execute step — proving the plan turn did NOT use the
    // default agent.
    const def = new RecordingAgent(AgentId.of("default")).push(fr("did step"));
    const planner = new RecordingAgent(AgentId.of("planner")).push(fr('{"tasks":["step"]}'));
    const h = new StandardHarness(configWith(def, { plannerAgent: planner }));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    expect(planner.ran).toBe(1); // planner ran exactly the plan turn
    expect(def.ran).toBe(1); // default ran exactly the execute step
  });

  it("R6: with no plannerAgent, the plan turn runs on the default agent", async () => {
    // No planner: the default agent runs both the plan turn and the one step.
    const def = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["step"]}'))
      .push(fr("did step"));
    const h = new StandardHarness(configWith(def));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    expect(def.ran).toBe(2); // plan turn + one execute step
  });

  it("R7: the plan turn counts against the budget (plan + one step ⇒ turns === 2)", async () => {
    // Plan turn (1) + one execute completion (1): the plan turn is counted in
    // the cumulative budget alongside the single execute step.
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["only"]}'))
      .push(fr("done"));
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind === "success" && r.turns).toBe(2);
  });

  it("R8: the plan turn records a turn span as the FIRST span of the run", async () => {
    const obs = new InMemoryObservabilityProvider();
    // Plan turn + one execute completion so the run progresses cleanly.
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["only"]}'))
      .push(fr("done"));
    const h = new StandardHarness(configWith(a, { observability: obs }));
    await h.run({ task: planTask() });
    const trace = await obs.getTrace(SessionId.of("plan-s1"));
    const turnSpans = trace.filter((s) => "turn_number" in s);
    // Plan turn (1) + the single execute step turn (1) = 2 spans total. A
    // dedicated #59 test (execute_phase_span_count) covers per-step accounting.
    expect(turnSpans.length).toBe(2);
    // The plan turn is the first span, turn_number 1.
    const first = turnSpans[0] as { turn_number: number };
    expect(first.turn_number).toBe(1);
  });

  it("R10: budget exhausted before the plan turn → budget_exceeded, no artifact", async () => {
    const a = new RecordingAgent(AgentId.of("default")).push(fr(PLAN_JSON));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    // max_turns: 0 means the turn cap is already met before any turn.
    const h = new StandardHarness(configWith(a, { storage }));
    const r = await h.run({ task: planTask("build a CLI", 0), session_state: state });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") expect(r.reason.kind).toBe("budget_exceeded");
    expect(a.ran).toBe(0); // never ran the plan turn.
    expect(await storage.run().get(PLAN_SID, PLAN_EXECUTE_EXTRAS_KEY)).toBeUndefined();
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  it("R11: on_plan_created mutation is reflected in the stored artifact", async () => {
    const a = new RecordingAgent(AgentId.of("default")).push(fr(PLAN_JSON));
    const chain: HookChain = new StandardHookChain();
    chain.register(
      hook(["on_plan_created"], async (ctx: HookContext): Promise<HookDecision> => {
        if (ctx.event === "on_plan_created") {
          // Mutate the plan in place — rewrite tasks + rationale.
          ctx.plan.tasks = ["rewritten"];
          ctx.plan.rationale = "by-hook";
        }
        return { decision: "continue" };
      }),
    );
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { hooks: chain, storage }));
    await h.run({ task: planTask(), session_state: state });
    expect(await storage.run().get(PLAN_SID, PLAN_EXECUTE_EXTRAS_KEY)).toEqual({
      tasks: ["rewritten"],
      rationale: "by-hook",
    });
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  // #76: after a full plan/execute run, BOTH persistence keys live on the
  // RunStore seam and NEITHER is mirrored into SessionState.extras. The
  // ephemeral extras keys (`__rich_state`, `subagent_handoff_summary`) are
  // owned by other components and untouched here.
  it("#76: plan_execute + task_list persistence live on the RunStore, not extras", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["one","two"],"rationale":"why"}'))
      .push(fr("did one"))
      .push(fr("did two"));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { storage }));
    const r = await h.run({ task: planTask(), session_state: state });
    expect(r.kind).toBe("success");

    // Both keys are durable in the RunStore.
    expect(await storage.run().get(PLAN_SID, PLAN_EXECUTE_EXTRAS_KEY)).not.toBeUndefined();
    expect(await storage.run().get(PLAN_SID, TASK_LIST_EXTRAS_KEY)).not.toBeUndefined();

    // Neither key is mirrored into SessionState.extras anymore.
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
    expect(state.extras[TASK_LIST_EXTRAS_KEY]).toBeUndefined();
  });
});
