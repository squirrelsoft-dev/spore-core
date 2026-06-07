/**
 * Fixture-replay integration tests for the PlanExecute DAG executor (issue #126).
 *
 * Drives a `StandardHarness` with `LoopStrategy.plan_execute` against the shared
 * GROUND-TRUTH fixtures:
 *   - plan_execute_dag_order.jsonl            — dependency-order drain (AC1)
 *   - plan_execute_dag_branch_isolation.jsonl — Tier-1 branch isolation (AC1)
 *   - plan_execute_dag_failure_cascade.jsonl  — terminal-failure cascade (AC3)
 *   - plan_execute_dag_budget_fail_cascade.jsonl — budget-Fail cascade twin (AC4)
 *   - plan_execute_dag_cycle_rejection.jsonl  — cycle rejected at entry (AC5)
 *
 * The runnable task list (with real blockers) is authored via the persisted
 * `task_list` store (#126 decision C) — the fixtures supply the model turns. Must
 * produce the same outcomes as the Rust integration tests; never edit a fixture
 * to make a failing implementation pass.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  ModelAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  newTask,
  type HarnessConfig,
  type LoopStrategy,
  type ProviderInfo,
} from "../src/index.js";
import { InMemoryStorageProvider, StorageProvider } from "../src/storage/index.js";
import { addTask, defaultTaskList, type TaskList } from "../src/tasklist/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixtureDir = resolve(repoRoot, "fixtures/model_responses/harness");

const provider: ProviderInfo = { name: "anthropic", model_id: "fixture", context_window: 200_000 };

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

function configFor(fixture: string, storage: StorageProvider): HarnessConfig {
  const jsonl = readFileSync(resolve(fixtureDir, fixture), "utf-8");
  const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
  const agent = new ModelAgent(AgentId.of("dag"), replay);
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    storage,
  };
}

async function harnessFor(
  fixture: string,
  session: SessionId,
  dag: TaskList,
): Promise<StandardHarness> {
  const storage = StorageProvider.single(new InMemoryStorageProvider());
  const h = new StandardHarness(configFor(fixture, storage));
  await h.persistTaskList(session, dag);
  return h;
}

describe("PlanExecute DAG executor fixture replay (#126)", () => {
  it("order: a blocker DAG drains in dependency order to Success", async () => {
    // DAG 1 → 2 → 3 → 4 (a linear chain authored with blockers). The four
    // execute fixtures (one/two/three/four) replay in dependency order.
    const session = SessionId.of("dag-order");
    const tl = defaultTaskList();
    addTask(tl, "one", []);
    addTask(tl, "two", [1]);
    addTask(tl, "three", [2]);
    addTask(tl, "four", [3]);
    const h = await harnessFor("plan_execute_dag_order.jsonl", session, tl);
    const r = await h.run({ task: newTask("build a diamond", session, PLAN_STRATEGY) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("did 4");
  });

  it("branch_isolation: root → child + an independent task, all complete", async () => {
    // 1 root → 2 child(blocked by 1); 3 independent.
    const session = SessionId.of("dag-branch");
    const tl = defaultTaskList();
    addTask(tl, "root", []);
    addTask(tl, "child of root", [1]);
    addTask(tl, "independent", []);
    const h = await harnessFor("plan_execute_dag_branch_isolation.jsonl", session, tl);
    const r = await h.run({ task: newTask("root, child, independent", session, PLAN_STRATEGY) });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.output).toBe("INDEP_OUTPUT_CCC");
  });

  it("failure_cascade: a terminal failure blocks only dependents; unrelated branch completes", async () => {
    // 1 root → completes; 2 mid → fails terminally; 3 leaf (blocked by 2) →
    // cascade-blocked; 4 indep → completes.
    const session = SessionId.of("dag-fail");
    const tl = defaultTaskList();
    addTask(tl, "root", []);
    addTask(tl, "mid", []);
    addTask(tl, "leaf", [2]);
    addTask(tl, "indep", []);
    const h = await harnessFor("plan_execute_dag_failure_cascade.jsonl", session, tl);
    const r = await h.run({ task: newTask("root, mid, leaf, indep", session, PLAN_STRATEGY) });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("tasks_blocked_by_failure");
      if (r.reason.kind === "tasks_blocked_by_failure") {
        expect(r.reason.failed_task).toBe(2);
        expect(r.reason.completed).toEqual([1, 4]);
        expect(r.reason.blocked).toEqual([2, 3]);
      }
    }
  });

  it("budget_fail_cascade: cascades identically to the error-failed twin (AC4)", async () => {
    // Same DAG + same terminal failure as failure_cascade — proving the budget-
    // Fail resolution shares the cascade arm and produces an identical partition.
    const session = SessionId.of("dag-budget-fail");
    const tl = defaultTaskList();
    addTask(tl, "root", []);
    addTask(tl, "mid", []);
    addTask(tl, "leaf", [2]);
    addTask(tl, "indep", []);
    const h = await harnessFor("plan_execute_dag_budget_fail_cascade.jsonl", session, tl);
    const r = await h.run({
      task: newTask("root, mid, leaf, indep (budget-fail cascade twin)", session, PLAN_STRATEGY),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("tasks_blocked_by_failure");
      if (r.reason.kind === "tasks_blocked_by_failure") {
        expect(r.reason.failed_task).toBe(2);
        expect(r.reason.completed).toEqual([1, 4]);
        expect(r.reason.blocked).toEqual([2, 3]);
      }
    }
  });

  it("cycle_rejection: a cyclic persisted graph is rejected at execute entry", async () => {
    const session = SessionId.of("dag-cycle");
    // Hand-build a cyclic graph (addTask would reject it).
    const cyclic: TaskList = {
      tasks: [
        { id: 1, description: "a", status: "pending", blockers: [2] },
        { id: 2, description: "b", status: "pending", blockers: [1] },
      ],
      next_id: 3,
    };
    const h = await harnessFor("plan_execute_dag_cycle_rejection.jsonl", session, cyclic);
    const r = await h.run({
      task: newTask("cyclic graph (rejected at execute entry)", session, PLAN_STRATEGY),
    });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") expect(r.reason.kind).toBe("task_graph_cycle");
  });
});
