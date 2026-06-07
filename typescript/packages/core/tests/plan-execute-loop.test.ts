/**
 * Unit tests for the PlanExecute execute phase / two-phase loop (issue #59).
 *
 * Mirrors the inline `run_execute_phase` tests in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same verdicts.
 *
 * Rules covered:
 *   - happy path / task drain (plan then execute, last-step output, Q2)
 *   - planner-agent routing through the full run (Q1 model routing)
 *   - Pending → InProgress → Completed drain (persisted task list)
 *   - per-task turn allocation + shared budget carried forward (Q1)
 *   - budget exhaustion mid-execute → budget_exceeded (global hard stop)
 *   - observability span count (plan turn + one span per step)
 *   - compaction-in-loop (multi-turn step reuses the ReAct compaction path)
 *   - on_task_advance fires N times with correct indices/totals (Q1 mutable)
 *   - empty plan → empty_plan (Q3)
 *   - step failure → step_failed (Q5), later tasks do NOT run
 *   - RunStore persistence through the storage seam (Q4)
 *   - execute_phase_not_implemented is GONE
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyResponse,
  EmptyToolRegistry,
  ExecutionRegistry,
  SessionId,
  StandardHarness,
  emptySessionState,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type SessionState,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import type { Hook, HookChain, HookContext, HookDecision, HookEvent } from "../src/hooks/index.js";
import type { Verifier, VerifierInput, VerifierVerdict } from "../src/verifier/index.js";
import { StandardHookChain } from "../src/hooks/standard.js";
import { InMemoryObservabilityProvider } from "../src/observability/in-memory.js";
import { InMemoryStorageProvider, StorageProvider } from "../src/storage/index.js";
import {
  TASK_LIST_EXTRAS_KEY,
  planArtifactToTaskList,
  type TaskList,
} from "../src/tasklist/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

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

function tcr(name = "x"): TurnResult {
  const call: ToolCall = { id: `${Math.random()}`, name, input: {} };
  return { kind: "tool_call_requested", calls: [call], usage: usage() };
}

/** A recording agent: each `turn` pops the next scripted result and records the
 *  number of invocations so routing / drain assertions are exact. */
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
  overrides: Partial<HarnessConfig> & { verifier?: Verifier } = {},
): HarnessConfig {
  // #124: the worker agent + any verifier fold into the registry under "".
  const { verifier, ...rest } = overrides;
  return {
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    ...rest,
    registry: registryWith({ agent, verifier }),
  };
}

const SID = SessionId.of("s1");

function planTask(maxTurns?: number) {
  return newTask("build a CLI", SID, PLAN_STRATEGY, { max_turns: maxTurns ?? null });
}

function hook(events: HookEvent[], handle: Hook["handle"]): Hook {
  return { handle, events: () => events, name: () => "test-hook" };
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("PlanExecute execute phase (issue #59)", () => {
  it("happy path: plan then execute, drains all tasks, output is the last step (Q2)", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["t1","t2","t3"],"rationale":"r"}'))
      .push(fr("done t1"))
      .push(fr("done t2"))
      .push(fr("done t3"));
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("done t3"); // Q2: last step's final
      expect(r.turns).toBe(4); // one plan turn + one per task
    }
    expect(a.ran).toBe(4);
  });

  it("plan-leaf agent routing: the plan child's leaf agent runs the plan turn, the worker runs the steps (#124 Q1)", async () => {
    // #124 Q1: the planner concept is DROPPED — the plan child's leaf
    // ReactConfig.agent is authoritative. Route the plan leaf to a distinct
    // "planner" key; the execute leaf (agent "") resolves the default worker.
    const def = new RecordingAgent(AgentId.of("default")).push(fr("did the step"));
    const planner = new RecordingAgent(AgentId.of("planner")).push(fr('{"tasks":["step"]}'));
    const registry = ExecutionRegistry.builder()
      .agent("", def)
      .agent("planner", planner)
      .toolset("", new EmptyToolRegistry())
      .schema("plan-schema", {})
      .build();
    const strategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "planner",
        toolset: "",
        output: "plan-schema",
      },
      execute: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
      },
    };
    const h = new StandardHarness({
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      registry,
    });
    const r = await h.run({ task: newTask("build a CLI", SID, strategy, { max_turns: null }) });
    expect(r.kind === "success" && r.output).toBe("did the step");
    expect(planner.ran).toBe(1); // planner ran exactly the plan turn
    expect(def.ran).toBe(1); // default ran exactly the execute step
  });

  it("drains Pending → InProgress → Completed (persisted list all completed)", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["one","two"]}'))
      .push(fr("done one"))
      .push(fr("done two"));
    const state: SessionState = emptySessionState();
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    const r = await h.run({ task: planTask(), session_state: state });
    expect(r.kind).toBe("success");
    // #76: the task list is persisted to the RunStore seam, not mirrored into
    // SessionState.extras.
    const list = (await provider.run().get(SID, TASK_LIST_EXTRAS_KEY)) as TaskList;
    expect(list.tasks.length).toBe(2);
    expect(list.tasks.every((t) => t.status === "completed")).toBe(true);
    expect(state.extras[TASK_LIST_EXTRAS_KEY]).toBeUndefined();
  });

  it("per-task turn allocation + shared budget carried forward (Q1)", async () => {
    // Global cap 7; plan turn (1) spent ⇒ 3 tasks split the remaining 6 turns
    // (2 each). Task "a" makes 2 tool calls without finishing, so its sub-loop
    // hits the per-task cap (turn budget) and the run aborts — proving both the
    // allocation and the carried budget.
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["a","b","c"]}'))
      .push(tcr())
      .push(tcr());
    const reg = new ScriptedToolRegistry()
      .push({ kind: "success", content: "ok" })
      .push({ kind: "success", content: "ok" });
    const h = new StandardHarness(configWith(a, { toolRegistry: reg }));
    const r = await h.run({ task: planTask(7) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      // The per-task cap is a turn budget; the sub-loop hits the turn gate which
      // surfaces as budget_exceeded(turns) routed through the execute phase.
      expect(r.reason.kind).toBe("budget_exceeded");
      if (r.reason.kind === "budget_exceeded") expect(r.reason.limit_type).toBe("turns");
      expect(r.turns).toBe(3); // 1 plan + 2 task-a turns (shared budget carried)
    }
  });

  it("budget exhausted mid-execute → budget_exceeded (global hard stop)", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["x","y","z"]}'))
      .push(fr("did x"));
    // max_turns 2: plan(1) + exactly one execute turn, then the cap is hit.
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask(2) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("budget_exceeded");
      if (r.reason.kind === "budget_exceeded") expect(r.reason.limit_type).toBe("turns");
      expect(r.turns).toBe(2);
    }
  });

  it("observability span count: plan turn + one span per executed step", async () => {
    const obs = new InMemoryObservabilityProvider();
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["a","b"]}'))
      .push(fr("did a"))
      .push(fr("did b"));
    const h = new StandardHarness(configWith(a, { observability: obs }));
    await h.run({ task: planTask() });
    const trace = await obs.getTrace(SID);
    const turnSpans = trace.filter((s) => "turn_number" in s);
    // 1 plan turn + 2 execute step turns = 3 turn spans.
    expect(turnSpans.length).toBe(3);
  });

  it("compaction-in-loop: a multi-turn step reuses the ReAct path and still drains", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["only"]}'))
      .push(tcr())
      .push(fr("finished only"));
    const reg = new ScriptedToolRegistry().push({ kind: "success", content: "ok" });
    const h = new StandardHarness(configWith(a, { toolRegistry: reg }));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("finished only");
      expect(r.turns).toBe(3); // plan(1) + tool turn(1) + final(1)
    }
  });

  it("on_task_advance fires once per task with correct index/total (Q1)", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["s0","s1","s2"]}'))
      .push(fr("d0"))
      .push(fr("d1"))
      .push(fr("d2"));
    let fireCount = 0;
    const seenIndices: number[] = [];
    const seenTotals: number[] = [];
    const chain: HookChain = new StandardHookChain();
    chain.register(
      hook(["on_task_advance"], async (ctx: HookContext): Promise<HookDecision> => {
        if (ctx.event === "on_task_advance") {
          fireCount += 1;
          seenIndices.push(ctx.task_index);
          seenTotals.push(ctx.total_tasks);
        }
        return { decision: "continue" };
      }),
    );
    const h = new StandardHarness(configWith(a, { hooks: chain }));
    await h.run({ task: planTask() });
    expect(fireCount).toBe(3);
    expect(seenIndices).toEqual([0, 1, 2]);
    expect(seenTotals).toEqual([3, 3, 3]);
  });

  it("on_task_advance may rewrite the step instruction (Q1 mutable task)", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["original"]}'))
      .push(fr("done"));
    let seenInstruction = "";
    const chain: HookChain = new StandardHookChain();
    chain.register(
      hook(["on_task_advance"], async (ctx: HookContext): Promise<HookDecision> => {
        if (ctx.event === "on_task_advance") {
          seenInstruction = ctx.task.instruction;
          ctx.task.instruction = "rewritten step";
        }
        return { decision: "continue" };
      }),
    );
    const state: SessionState = emptySessionState();
    const h = new StandardHarness(configWith(a, { hooks: chain }));
    await h.run({ task: planTask(), session_state: state });
    expect(seenInstruction).toBe("original");
    // The rewritten instruction is what got seeded into the (parent) session.
    const seeded = state.messages.some(
      (m) => m.role === "user" && m.content.type === "text" && m.content.text === "rewritten step",
    );
    expect(seeded).toBe(true);
  });

  it("Q3: an empty plan → empty_plan (not a silent success)", async () => {
    const a = new RecordingAgent(AgentId.of("default")).push(
      fr('{"tasks":[],"rationale":"nothing"}'),
    );
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("empty_plan");
      expect(r.turns).toBe(1); // only the plan turn ran
    }
    expect(a.ran).toBe(1); // no execute steps
  });

  it("Q5: a step that errors aborts with step_failed; later tasks do NOT run", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["good","bad","never"]}'))
      .push(fr("did good"))
      .push({ kind: "error", error: new EmptyResponse(), usage: null });
    // "never" must not run.
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("step_failed");
      if (r.reason.kind === "step_failed") {
        expect(r.reason.task_index).toBe(1);
        expect(r.reason.task).toBe("bad");
      }
    }
    // plan(1) + good(1) + bad(1) = 3 agent calls; "never" never ran.
    expect(a.ran).toBe(3);
  });

  it("Q4: the task list persists through the RunStore seam (not the sandbox path)", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["one"]}'))
      .push(fr("did one"));
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    await h.run({ task: planTask() });
    const stored = await provider.run().get(SID, TASK_LIST_EXTRAS_KEY);
    expect(stored).not.toBeUndefined();
    const list = stored as TaskList;
    expect(list.tasks.length).toBe(1);
    expect(list.tasks[0]!.status).toBe("completed");
  });

  it("Q2: success output is the LAST step, not a concat or the rationale", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["a","b"],"rationale":"RATIONALE_TOKEN"}'))
      .push(fr("FIRST_STEP_OUTPUT"))
      .push(fr("LAST_STEP_OUTPUT"));
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("LAST_STEP_OUTPUT");
      expect(r.output).not.toContain("FIRST_STEP_OUTPUT");
      expect(r.output).not.toContain("RATIONALE_TOKEN");
    }
  });

  it("execute_phase_not_implemented is GONE: a full run returns Success", async () => {
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["only"]}'))
      .push(fr("done"));
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
  });

  // ── A.6 deep-resume (#124, Q2) ─────────────────────────────────────────────
  it("deep-resume: a task already Completed on the durable checkpoint is NOT re-run", async () => {
    // Mirrors the Rust `deep_resume_skips_already_completed_task`: a PRIOR run's
    // progress (task 1 Completed) lives on the durable RunStore checkpoint. The
    // execute phase reconciles the freshly-parsed (all-Pending) task list with
    // that checkpoint via serialize→reset→resume — task 1 is skipped, only the
    // not-yet-done task runs.
    const provider = StorageProvider.single(new InMemoryStorageProvider());

    // Pre-seed the durable checkpoint: task 1 ("one") Completed, task 2 Pending.
    const checkpoint: TaskList = {
      tasks: [
        { id: 1, description: "one", status: "completed", blockers: [] },
        { id: 2, description: "two", status: "pending", blockers: [] },
      ],
      next_id: 3,
    };
    await provider.run().put(SID, TASK_LIST_EXTRAS_KEY, JSON.parse(JSON.stringify(checkpoint)));

    // The freshly-parsed list (as the plan phase produces) is all-Pending.
    const fresh = planArtifactToTaskList({ tasks: ["one", "two"], rationale: "r" });

    // Only ONE scripted response: enough iff task 1 is skipped (else the second
    // step starves and the run fails).
    const a = new RecordingAgent(AgentId.of("default")).push(fr("done two"));
    const h = new StandardHarness(configWith(a, { storage: provider }));

    // Drive the execute phase directly (the StrategyExecutor seam) with a budget
    // that already reflects the prior plan turn (turns: 1).
    const r = await h.executePhase(
      planTask(),
      emptySessionState(),
      fresh,
      { turns: 1, input_tokens: 0, output_tokens: 0, cost_usd: 0 },
      {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        cost_usd: 0,
      },
      undefined,
      undefined,
    );

    expect(r.kind).toBe("success");
    // Exactly ONE agent turn: task 1 was resumed-from-checkpoint, not re-run.
    expect(a.ran).toBe(1);
    if (r.kind === "success") expect(r.output).toBe("done two");

    // Both tasks Completed on the final persisted checkpoint.
    const list = (await provider.run().get(SID, TASK_LIST_EXTRAS_KEY)) as TaskList;
    expect(list.tasks.every((t) => t.status === "completed")).toBe(true);
  });

  // ── #124 REGRESSION-PROOF: a NON-ReAct execute child genuinely runs per task ──
  it("plan_execute_runs_non_react_execute_child_per_task", async () => {
    // `PlanExecute[plan: ReAct, execute: SelfVerifying[ReAct]]` over a 2-task
    // plan. The execute child is SelfVerifying — its build↔evaluate loop runs
    // ONCE PER TASK, so the recording verifier is invoked exactly twice (once per
    // task). The pre-#124 implementation hardcoded a flat ReAct sub-loop in
    // `runExecutePhase` and silently DROPPED the SelfVerifying child — the
    // verifier would have been invoked ZERO times. This proves `c.execute` is
    // dispatched via `runStrategy(execute, cx)` and a non-ReAct child genuinely
    // executes its loop per task.

    /** A verifier that records every invocation; PASS each time. */
    class RecordingVerifier implements Verifier {
      readonly seen: VerifierInput[] = [];
      async verify(input: VerifierInput, _signal?: AbortSignal): Promise<VerifierVerdict> {
        this.seen.push(input);
        return { kind: "passed" };
      }
      maxIterations(): number {
        return 3;
      }
    }

    // #124 Q1c: ONE worker agent runs the plan turn (JSON), AND each task's
    // SelfVerifying build + evaluate phases (the evaluate agent defaults to the
    // inner worker's agent). Queue: plan, build t0, eval t0, build t1, eval t1.
    const a = new RecordingAgent(AgentId.of("default"))
      .push(fr('{"tasks":["t0","t1"],"rationale":"r"}'))
      .push(fr("built t0"))
      .push(fr("PASS"))
      .push(fr("built t1"))
      .push(fr("PASS"));
    const verifier = new RecordingVerifier();

    // The execute child is a genuine SelfVerifying combinator (NOT a ReAct).
    const strategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "",
        toolset: "",
        output: "",
      },
      execute: {
        kind: "self_verifying",
        inner: {
          kind: "react",
          budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
          agent: "",
          toolset: "",
          output: "",
        },
        evaluator: "",
      },
    };
    const task = newTask("build a CLI", SID, strategy);
    const h = new StandardHarness(configWith(a, { verifier }));

    const r = await h.run({ task });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      expect(r.output).toBe("built t1"); // Q2: last step's final output
    }

    // The smoking gun: the SelfVerifying evaluator ran ONCE PER TASK (2x). A
    // dropped execute child would record ZERO verifier invocations.
    expect(verifier.seen.length).toBe(2);
  });
});
