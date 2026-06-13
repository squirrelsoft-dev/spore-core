/**
 * Unit tests for the PlanExecute DAG executor (issue #126): ready-set task walk,
 * two-tier context, and failure cascade.
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs` (#126 tests) — same rules, same
 * verdicts. Covers all 6 acceptance criteria plus the ledger drop-oldest policy,
 * the Tier-1 lean (scoped) seed, and files_touched being harness-OBSERVED (never
 * self-reported).
 *
 * The pure DAG/ledger helpers (`nextReady`, `transitiveBlockers`,
 * `transitiveDependents`, `hasCycle`, `pushStepLedger`, `renderStepLedger`) are
 * also unit-tested directly.
 */

import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyResponse,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type Task,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import { InMemoryStorageProvider, StorageProvider } from "../src/storage/index.js";
import {
  STEP_LEDGER_ELISION_MARKER,
  STEP_LEDGER_MAX_ENTRIES,
  addTask,
  defaultTaskList,
  hasCycle,
  nextReady,
  pushStepLedger,
  renderStepLedger,
  transitiveBlockers,
  transitiveDependents,
  type StepLedgerEntry,
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

const SID = SessionId.of("dag");

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

function tcr(call: ToolCall): TurnResult {
  return { kind: "tool_call_requested", calls: [call], usage: usage() };
}

/** A context-capturing recording agent: records every Context it sees and the
 *  number of invocations, so routing / drain / Tier-1 isolation can be asserted. */
class RecordingAgent implements Agent {
  ran = 0;
  readonly seen: Context[] = [];
  private readonly results: TurnResult[] = [];
  constructor(private readonly agentId = AgentId.of("default")) {}
  push(r: TurnResult): this {
    this.results.push(r);
    return this;
  }
  id(): AgentId {
    return this.agentId;
  }
  async turn(ctx: Context): Promise<TurnResult> {
    this.ran += 1;
    this.seen.push(ctx);
    const next = this.results.shift();
    if (next == null) return { kind: "error", error: new EmptyResponse(), usage: null };
    return next;
  }
}

function configWith(agent: Agent, overrides: Partial<HarnessConfig> = {}): HarnessConfig {
  return {
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    ...overrides,
    registry: registryWith({ agent }),
  };
}

function planTask(session = SID): Task {
  return newTask("build it", session, PLAN_STRATEGY, { max_turns: null });
}

function ctxText(ctx: Context): string {
  return ctx.messages.map((m) => (m.content.type === "text" ? m.content.text : "")).join("\n");
}

/** Seed an authored DAG into the persisted task_list store (the #126 authoring
 *  path) so the executor's loadTaskList picks it up over the linear bridge. */
async function seedDag(h: StandardHarness, session: SessionId, list: TaskList): Promise<void> {
  await h.persistTaskList(session, list);
}

// ==========================================================================
// Pure DAG / ledger helpers
// ==========================================================================

describe("PlanExecute DAG helpers (#126, pure)", () => {
  it("nextReady picks the lowest-id pending task whose blockers are all completed", () => {
    const tl = defaultTaskList();
    const a = addTask(tl, "a", []);
    const b = addTask(tl, "b", []);
    expect(a.ok && b.ok).toBe(true);
    // Both pending, no blockers → lowest id (1).
    expect(nextReady(tl)).toBe(1);
    // Complete 1 → 2 becomes the lowest ready.
    tl.tasks[0]!.status = "completed";
    expect(nextReady(tl)).toBe(2);
    // A task blocked by an un-completed blocker is not ready.
    const c = addTask(tl, "c", [2]);
    expect(c.ok).toBe(true);
    expect(nextReady(tl)).toBe(2); // 3 waits on 2.
    tl.tasks[1]!.status = "completed";
    expect(nextReady(tl)).toBe(3);
    tl.tasks[2]!.status = "completed";
    expect(nextReady(tl)).toBeUndefined();
  });

  it("transitiveBlockers returns the sorted transitive-blocker closure, excluding self", () => {
    const tl = defaultTaskList();
    addTask(tl, "1", []);
    addTask(tl, "2", [1]);
    addTask(tl, "3", [2]);
    addTask(tl, "4", []); // independent
    expect(transitiveBlockers(tl, 3)).toEqual([1, 2]);
    expect(transitiveBlockers(tl, 4)).toEqual([]);
    expect(transitiveBlockers(tl, 1)).toEqual([]);
  });

  it("transitiveDependents returns the sorted transitive-dependent closure, excluding self", () => {
    const tl = defaultTaskList();
    addTask(tl, "1", []);
    addTask(tl, "2", [1]);
    addTask(tl, "3", [2]);
    addTask(tl, "4", []); // independent
    expect(transitiveDependents(tl, 1)).toEqual([2, 3]);
    expect(transitiveDependents(tl, 4)).toEqual([]);
    expect(transitiveDependents(tl, 3)).toEqual([]);
  });

  it("hasCycle detects a directed cycle in a hand-built cyclic graph", () => {
    const acyclic = defaultTaskList();
    addTask(acyclic, "1", []);
    addTask(acyclic, "2", [1]);
    expect(hasCycle(acyclic)).toBe(false);
    // Hand-build a cycle (addTask would reject it, so mutate directly — modeling
    // an out-of-band persisted store).
    const cyclic: TaskList = {
      tasks: [
        { id: 1, description: "a", status: "pending", blockers: [2] },
        { id: 2, description: "b", status: "pending", blockers: [1] },
      ],
      next_id: 3,
    };
    expect(hasCycle(cyclic)).toBe(true);
  });

  it("pushStepLedger drops the oldest past the bound (drop-oldest, deterministic)", () => {
    const ledger: StepLedgerEntry[] = [];
    let everDropped = false;
    for (let i = 1; i <= STEP_LEDGER_MAX_ENTRIES + 3; i += 1) {
      const dropped = pushStepLedger(ledger, { task_id: i, summary: `s${i}`, files_touched: [] });
      everDropped = everDropped || dropped;
    }
    expect(ledger.length).toBe(STEP_LEDGER_MAX_ENTRIES);
    expect(everDropped).toBe(true);
    // The OLDEST (1,2,3) were dropped; the most-recent remain in completion order.
    expect(ledger[0]!.task_id).toBe(4);
    expect(ledger[ledger.length - 1]!.task_id).toBe(STEP_LEDGER_MAX_ENTRIES + 3);
  });

  it("renderStepLedger renders compact lines, files suffix, and the elision marker", () => {
    expect(renderStepLedger([], false)).toBeUndefined();
    const ledger: StepLedgerEntry[] = [
      { task_id: 1, summary: "did one", files_touched: [] },
      { task_id: 2, summary: "did two", files_touched: ["a.ts", "b.ts"] },
    ];
    const rendered = renderStepLedger(ledger, false);
    expect(rendered).toBe("Progress ledger so far:\n#1 did one\n#2 did two [files: a.ts, b.ts]");
    const elided = renderStepLedger(ledger, true);
    expect(elided).toContain(STEP_LEDGER_ELISION_MARKER);
    expect(elided!.split("\n")[1]).toBe(STEP_LEDGER_ELISION_MARKER);
  });
});

// ==========================================================================
// AC1 — dependency-order execution + branch isolation
// ==========================================================================

describe("PlanExecute DAG executor (#126)", () => {
  it("AC1: a blocker DAG executes in dependency order with a lowest-id tiebreak", async () => {
    // DAG: 2 and 3 both block on 1; 4 blocks on 2 and 3. Expected run order by
    // lowest-ready-id: 1, 2, 3, 4.
    // #138 AC1: the durable task_list is pre-seeded below, so the plan phase is
    // SKIPPED — no plan turn is pushed. The first model call is task 1.
    const a = new RecordingAgent()
      .push(fr("did 1"))
      .push(fr("did 2"))
      .push(fr("did 3"))
      .push(fr("did 4"));
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    const tl = defaultTaskList();
    addTask(tl, "one", []);
    addTask(tl, "two", [1]);
    addTask(tl, "three", [1]);
    addTask(tl, "four", [2, 3]);
    await seedDag(h, SID, tl);

    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("did 4");
    // #138 AC1: plan SKIPPED → 4 steps in order, no plan turn.
    expect(a.ran).toBe(4);
    // Step contexts (indices 0..3) carry their own instructions in run order.
    expect(ctxText(a.seen[0]!)).toContain("one");
    expect(ctxText(a.seen[1]!)).toContain("two");
    expect(ctxText(a.seen[2]!)).toContain("three");
    expect(ctxText(a.seen[3]!)).toContain("four");
  });

  it("AC1: independent branches do NOT pollute each other's context (Tier-1 isolation)", async () => {
    // 1 (root) → 2 (child of 1); 3 is independent. Child (2) sees root's (1)
    // output; the independent task (3) must NOT see it, and the child must NOT
    // see the independent branch.
    // #138 AC1: the durable task_list is pre-seeded below, so the plan phase is
    // SKIPPED — no plan turn. The first model call is task 1.
    const a = new RecordingAgent()
      .push(fr("ROOT_OUTPUT_AAA")) // task 1
      .push(fr("CHILD_OUTPUT_BBB")) // task 2 (blocked by 1)
      .push(fr("INDEP_OUTPUT_CCC")); // task 3 (independent)
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    const tl = defaultTaskList();
    addTask(tl, "root", []);
    addTask(tl, "child of root", [1]);
    addTask(tl, "independent", []);
    await seedDag(h, SID, tl);

    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    // Run order by lowest-ready id: 1, 2, 3 (indices 0, 1, 2 — plan skipped).
    const childCtx = ctxText(a.seen[1]!); // task 2
    const indepCtx = ctxText(a.seen[2]!); // task 3
    // Child (Tier-1) sees its transitive blocker (root)'s output.
    expect(childCtx).toContain("ROOT_OUTPUT_AAA");
    // Child does NOT see the independent branch (task 3 has not run; not a blocker).
    expect(childCtx).not.toContain("INDEP_OUTPUT_CCC");
    // The independent task 3 has NO upstream blockers, so it must NOT carry a
    // Tier-1 "Results from upstream tasks" block. (The Tier-2 global ledger
    // summaries may still appear by design — this asserts Tier-1 isolation.)
    expect(indepCtx).not.toContain("Results from upstream tasks");
  });

  it("Tier-1 lean: a task's seed carries ONLY its transitive blockers (decision D)", async () => {
    // Diamond: 1 → {2,3} → 4. Task 4's Tier-1 seed should carry 1, 2, 3's outputs
    // (its transitive blockers) and nothing else. Task 2's seed should carry only
    // task 1 (NOT its sibling task 3).
    // #138 AC1: the durable task_list is pre-seeded below, so the plan phase is
    // SKIPPED — no plan turn. The first model call is task 1.
    const a = new RecordingAgent()
      .push(fr("OUT_ONE"))
      .push(fr("OUT_TWO"))
      .push(fr("OUT_THREE"))
      .push(fr("OUT_FOUR"));
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    const tl = defaultTaskList();
    addTask(tl, "one", []);
    addTask(tl, "two", [1]);
    addTask(tl, "three", [1]);
    addTask(tl, "four", [2, 3]);
    await seedDag(h, SID, tl);

    await h.run({ task: planTask() });
    const twoCtx = ctxText(a.seen[1]!); // task 2 (index 1 — plan skipped)
    const fourCtx = ctxText(a.seen[3]!); // task 4 (index 3 — plan skipped)
    // Task 2's seed carries task 1's output but NOT task 3's (a sibling, not a blocker).
    expect(twoCtx).toContain("OUT_ONE");
    expect(twoCtx).not.toContain("OUT_THREE");
    // Task 4's seed carries all three transitive blockers' outputs.
    expect(fourCtx).toContain("OUT_ONE");
    expect(fourCtx).toContain("OUT_TWO");
    expect(fourCtx).toContain("OUT_THREE");
  });

  // ========================================================================
  // AC2 — files_touched is harness-OBSERVED, never self-reported
  // ========================================================================

  it("AC2: ledger files_touched is observed from edit/write calls (not self-reported)", async () => {
    // Task 1 only NARRATES touching a file (no real write/edit call) → empty
    // files_touched. Task 2 issues a real edit_file call carrying a path → the
    // path is OBSERVED and recorded. Task 3 (blocked by 2) sees task 2's ledger
    // row including the observed file, proving the seam end-to-end.
    const editCall: ToolCall = { id: "e1", name: "edit_file", input: { path: "src/widget.ts" } };
    // #138 AC1: the durable task_list is pre-seeded below, so the plan phase is
    // SKIPPED — no plan turn. The first model call is task 1.
    const a = new RecordingAgent()
      // Task 1: narrate touching a file but make no tool call.
      .push(fr("I touched src/phantom.ts (narrated only, no tool call)"))
      // Task 2: real edit_file call, then finalize.
      .push(tcr(editCall))
      .push(fr("edited the widget"))
      // Task 3 (blocked by 2): finalize directly.
      .push(fr("done three"));
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const reg = new ScriptedToolRegistry().push({ kind: "success", content: "ok" });
    const h = new StandardHarness(configWith(a, { storage: provider, toolRegistry: reg }));
    const tl = defaultTaskList();
    addTask(tl, "narrate only", []);
    addTask(tl, "real edit", []);
    addTask(tl, "downstream", [2]);
    await seedDag(h, SID, tl);

    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("success");
    // Task 3's context carries the Tier-2 ledger. Task 2's row has the OBSERVED
    // file as a `[files: ...]` suffix; task 1 (narrate-only) has NO files suffix —
    // its narrated "src/phantom.ts" lives only in its summary text, never as an
    // observed file entry.
    const threeCtx = ctxText(a.seen[a.seen.length - 1]!);
    expect(threeCtx).toContain("#2 edited the widget [files: src/widget.ts]");
    // Task 1's row appears WITHOUT a files suffix (empty files_touched).
    expect(threeCtx).toContain("#1 I touched src/phantom.ts (narrated only, no tool call)\n");
    expect(threeCtx).not.toContain("[files: src/phantom.ts]");
  });

  it("AC2 (discriminating): the observed-write seam records ONLY real write/edit calls", async () => {
    // Drive the StrategyExecutor seam directly: a narrated path is NOT recorded;
    // a real write_file / edit_file call WITH a path IS recorded (de-duplicated).
    // Mirrors Rust's observed_writes_seam_records_only_real_write_calls.
    const h = new StandardHarness(configWith(new RecordingAgent()));
    // Nothing observed yet.
    expect(h.takeObservedWrites()).toEqual([]);
    // A non-write tool call is ignored.
    h.observeWriteCall({ id: "r", name: "read_file", input: { path: "src/read.ts" } });
    expect(h.takeObservedWrites()).toEqual([]);
    // A write_file / edit_file call WITH a path is recorded (de-duplicated).
    h.observeWriteCall({ id: "w", name: "write_file", input: { path: "src/a.ts" } });
    h.observeWriteCall({ id: "e", name: "edit_file", input: { path: "src/b.ts" } });
    h.observeWriteCall({ id: "e2", name: "edit_file", input: { path: "src/a.ts" } }); // dup
    expect(h.takeObservedWrites()).toEqual(["src/a.ts", "src/b.ts"]);
    // Draining reset the accumulator.
    expect(h.takeObservedWrites()).toEqual([]);
    // clearObservedWrites also empties it.
    h.observeWriteCall({ id: "w2", name: "write_file", input: { path: "src/c.ts" } });
    h.clearObservedWrites();
    expect(h.takeObservedWrites()).toEqual([]);
  });

  // ========================================================================
  // AC3 — failure cascade blocks only transitive dependents
  // ========================================================================

  it("AC3: a terminal task failure blocks only transitive dependents; unrelated tasks complete", async () => {
    // DAG: 1 "good" → completes; 2 "bad" → fails; 3 "dep" (blocked by 2) →
    // cascade-blocked; 4 "indep" → still completes. Run does NOT abort on first
    // failure; drains to tasks_blocked_by_failure with the full partition.
    // #138 AC1: the durable task_list is pre-seeded below, so the plan phase is
    // SKIPPED — no plan turn. The first model call is task 1.
    const a = new RecordingAgent()
      .push(fr("did good")) // task 1
      .push({ kind: "error", error: new EmptyResponse(), usage: null }) // task 2 fails
      .push(fr("did indep")); // task 4 (3 is blocked, never runs)
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    const tl = defaultTaskList();
    addTask(tl, "good", []);
    addTask(tl, "bad", []);
    addTask(tl, "dep", [2]);
    addTask(tl, "indep", []);
    await seedDag(h, SID, tl);

    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("tasks_blocked_by_failure");
      if (r.reason.kind === "tasks_blocked_by_failure") {
        expect(r.reason.failed_task).toBe(2);
        expect(r.reason.completed).toEqual([1, 4]);
        expect(r.reason.blocked).toEqual([2, 3]);
      }
    }
    // #138 AC1: plan SKIPPED — good + bad + indep = 3 agent calls; the dependent
    // "dep" never ran.
    expect(a.ran).toBe(3);
  });

  // ========================================================================
  // AC5 — cycle rejection at execute entry (defense in depth)
  // ========================================================================

  it("AC5: a cyclic persisted graph is rejected at execute entry; no task runs", async () => {
    // #138 AC1: a non-empty (cyclic) task_list is pre-seeded below, so the plan
    // phase is SKIPPED — the cycle is detected at execute entry with NO model turn
    // at all. No plan turn is pushed.
    const a = new RecordingAgent();
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const h = new StandardHarness(configWith(a, { storage: provider }));
    // Hand-build a cyclic graph (addTask would reject it) and persist it.
    const cyclic: TaskList = {
      tasks: [
        { id: 1, description: "a", status: "pending", blockers: [2] },
        { id: 2, description: "b", status: "pending", blockers: [1] },
      ],
      next_id: 3,
    };
    await seedDag(h, SID, cyclic);

    const r = await h.run({ task: planTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("task_graph_cycle");
    }
    // #138 AC1: skip-plan means NO model turn ran — the cycle is caught at execute
    // entry before any plan or execute dispatch.
    expect(a.ran).toBe(0);
  });

  // ========================================================================
  // AC6 — PlanArtifact bridge deprecated (still works as a fallback)
  // ========================================================================

  it("AC6: with no authored task_list, the executor falls back to the plan-artifact bridge", async () => {
    // No persisted task_list → the linear plan-artifact bridge (deprecated)
    // supplies the runnable list. A 2-task LINEAR plan drains to success.
    const a = new RecordingAgent()
      .push(fr('{"tasks":["first","second"]}'))
      .push(fr("did first"))
      .push(fr("did second"));
    const h = new StandardHarness(configWith(a));
    const r = await h.run({ task: planTask(SessionId.of("dag-fallback")) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("did second");
    expect(a.ran).toBe(3);
  });
});
