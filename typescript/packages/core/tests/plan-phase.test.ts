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
  ExecutionRegistry,
  PLAN_EXECUTE_EXTRAS_KEY,
  SessionId,
  StandardHarness,
  capturePlanArtifact,
  capturePlanArtifactWithRepair,
  extractEmbeddedJsonObject,
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
  registryWith,
} from "../src/harness/testing.js";
import { EmptyToolRegistry } from "../src/tool-registry/empty.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

const PLAN_STRATEGY: LoopStrategy = {
  kind: "plan_execute",
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

function configWith(
  agent: Agent,
  overrides: Partial<HarnessConfig> = {},
  registry?: ExecutionRegistry,
): HarnessConfig {
  // #124: the worker agent folds into the registry under "". A caller may pass a
  // pre-built registry (e.g. to route the plan leaf to a distinct agent key).
  return {
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    ...overrides,
    registry: registry ?? registryWith({ agent }),
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
// prose-repair fallback (Item 1)
// --------------------------------------------------------------------------

describe("capturePlanArtifactWithRepair — prose-repair fallback", () => {
  // A clean object that the STRICT grammar already accepts is returned
  // unchanged by the repair wrapper (repair never runs on a success).
  it("passes a strict success through unchanged", () => {
    const r = capturePlanArtifactWithRepair('{"tasks":["a","b"],"rationale":"r"}');
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.artifact.tasks).toEqual(["a", "b"]);
      expect(r.artifact.rationale).toBe("r");
    }
  });

  // The live failure mode: the planner wraps its plan JSON in prose. The strict
  // grammar rejects it; the repair extracts the embedded object.
  it("extracts plan JSON wrapped in prose", () => {
    const text =
      'Sure! Here is the plan:\n{"tasks":["step 1","step 2"],"rationale":"because"}\nLet me know if that works.';
    // Strict path fails…
    expect(capturePlanArtifact(text).ok).toBe(false);
    // …repair rescues it.
    const r = capturePlanArtifactWithRepair(text);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.artifact.tasks).toEqual(["step 1", "step 2"]);
      expect(r.artifact.rationale).toBe("because");
    }
  });

  // Braces inside string values must NOT confuse the balanced-object scan.
  it("respects braces inside string values", () => {
    const text = 'prefix {"tasks":["use the { brace } char","b"]} suffix';
    const r = capturePlanArtifactWithRepair(text);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.artifact.tasks).toEqual(["use the { brace } char", "b"]);
    }
  });

  // The embedded object is captured to its FIRST balanced close (nested objects
  // are spanned correctly).
  it("spans nested objects to the first balanced close", () => {
    const text = 'x {"tasks":["a"],"meta":{"k":"v"}} y';
    expect(extractEmbeddedJsonObject(text)).toBe('{"tasks":["a"],"meta":{"k":"v"}}');
  });

  // Repair that still cannot parse a clean plan surfaces the ORIGINAL strict
  // error, not a repair-specific one.
  it("returns the original strict error when the embedded object is not a valid plan", () => {
    // Embedded object exists but is not a valid plan (tasks not an array).
    const text = 'here: {"tasks":"nope"} end';
    const r = capturePlanArtifactWithRepair(text);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.kind).toBe("unparseable_plan");
  });

  // No embedded object at all ⇒ still unparseable_plan, never throws.
  it("returns unparseable_plan when there is no embedded object", () => {
    const r = capturePlanArtifactWithRepair("no json here at all");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.kind).toBe("unparseable_plan");
  });

  // An unbalanced `{` (no matching close) extracts nothing.
  it("extracts nothing for an unbalanced `{`", () => {
    expect(extractEmbeddedJsonObject('{"tasks":["a"')).toBeUndefined();
  });
});

// --------------------------------------------------------------------------
// plan phase driver
// --------------------------------------------------------------------------

describe("PlanExecute plan phase", () => {
  it("Q4: produces+stores an artifact, then hands off to the execute phase", async () => {
    // Only the plan turn is scripted; the execute phase then drives the agent
    // again, exhausts the script, and (under #126) drains to
    // tasks_blocked_by_failure when the dry execute steps fail (proving the
    // execute_phase_not_implemented halt is gone and the artifact was stored).
    const a = new RecordingAgent(AgentId.of("default")).push(fr(PLAN_JSON));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { storage }));
    const r = await h.run({ task: planTask(), session_state: state });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("tasks_blocked_by_failure");
    }
    // #76: the plan artifact was produced + persisted to the RunStore seam
    // before the execute phase ran (no longer mirrored into extras).
    expect(await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY)).toEqual({
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

  // #124 recursion seam (was R2): the plan phase now dispatches the GENUINE
  // `plan` child (a ReAct loop) capped at ONE turn. A plan turn that requests a
  // tool instead of emitting the JSON plan cannot complete in its single turn, so
  // the plan child halts on the cap and the plan phase propagates a terminal
  // Failure WITHOUT capturing/storing an artifact. (The old one-shot primitive
  // special-cased a tool call as planning_turn_failed; under genuine recursion
  // the cap is what stops the loop — the observable contract that a non-plan plan
  // turn fails the run and stores nothing is preserved.)
  it("#124: a tool-only plan turn fails the run and stores no artifact", async () => {
    const reg = new ScriptedToolRegistry().push({ kind: "success", content: "ok" });
    const a = new RecordingAgent(AgentId.of("default")).push(tcr());
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { toolRegistry: reg, storage }));
    const r = await h.run({ task: planTask(), session_state: state });
    // The plan child never emitted a JSON plan, so the run halts terminally.
    expect(r.kind).toBe("failure");
    // Nothing captured/stored: no artifact reached the RunStore.
    expect(await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY)).toBeUndefined();
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  it("R3 + R4: artifact captured from response text and persisted to the RunStore", async () => {
    const a = new RecordingAgent(AgentId.of("default")).push(fr(PLAN_JSON));
    const state: SessionState = emptySessionState();
    const storage = inMemoryStorage();
    const h = new StandardHarness(configWith(a, { storage }));
    await h.run({ task: planTask(), session_state: state });
    const stored = await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY);
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
    expect(await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY)).toBeUndefined();
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
  });

  it("R5 (#124 Q1): the plan child's leaf agent runs the plan turn; the default worker runs the steps", async () => {
    // #124 Q1: the planner concept is DROPPED — the plan child's leaf
    // ReactConfig.agent is authoritative. Route the plan leaf to a distinct
    // "planner" key; the execute leaf (agent "") resolves the default worker.
    const def = new RecordingAgent(AgentId.of("default")).push(fr("did step"));
    const planner = new RecordingAgent(AgentId.of("planner")).push(fr('{"tasks":["step"]}'));
    const registry = ExecutionRegistry.builder()
      .agent("", def)
      .agent("planner", planner)
      .toolset("", new EmptyToolRegistry())
      .schema("", {})
      .build();
    const strategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "planner",
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
    const h = new StandardHarness(configWith(def, {}, registry));
    const r = await h.run({ task: newTask("build", PLAN_SID, strategy, { max_turns: null }) });
    expect(r.kind).toBe("success");
    expect(planner.ran).toBe(1); // the plan leaf's agent ran exactly the plan turn
    expect(def.ran).toBe(1); // default ran exactly the execute step
  });

  it("R6: with no distinct plan agent, the plan turn runs on the default worker", async () => {
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
    expect(await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY)).toBeUndefined();
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
    expect(await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY)).toEqual({
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
    expect(await storage.run().get(h.projectId().namespace(), PLAN_EXECUTE_EXTRAS_KEY)).not.toBeUndefined();
    expect(await storage.run().get(h.projectId().namespace(), TASK_LIST_EXTRAS_KEY)).not.toBeUndefined();

    // Neither key is mirrored into SessionState.extras anymore.
    expect(state.extras[PLAN_EXECUTE_EXTRAS_KEY]).toBeUndefined();
    expect(state.extras[TASK_LIST_EXTRAS_KEY]).toBeUndefined();
  });
});
