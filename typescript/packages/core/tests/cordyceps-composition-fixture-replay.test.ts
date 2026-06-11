/**
 * Fixture-replay integration tests for the cordyceps composition (#131):
 * `Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]]`, driven by the canonical
 * `fixtures/strategy/cordyceps_tree.json`.
 *
 * These exercise the SAME recorded-model harness as
 * `plan-execute-dag-fixture-replay.test.ts`, but against the FULL composed tree
 * with its real handles wired into an `ExecutionRegistry`: agents
 * `planner`/`executor`/`ralph-agent`, toolsets `plan-tools`/`exec-tools`,
 * schemas `plan-schema`/`worker-schema`, and the Default-FAIL `exec-evaluator`
 * verifier. Never edit a fixture to make a failing implementation pass — the
 * fixtures are ground truth and must stay internally consistent.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  AgentId,
  EmptyToolRegistry,
  ExecutionRegistry,
  LoopStrategySchema,
  ModelAgent,
  ReplayModelInterface,
  SessionId,
  StandardHarness,
  TaskSchema,
  emptyAggregateUsage,
  loopStrategyMaxSteps,
  newTask,
  type Agent,
  type ConsultRequest,
  type EscalationMode,
  type HarnessConfig,
  type LoopStrategy,
  type ProviderInfo,
  type RunResult,
  type Task,
  type ToolOutput,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";
import { InMemoryStorageProvider, StorageProvider } from "../src/storage/index.js";
import {
  EvaluatorResponseVerifier,
  type Verifier,
  type VerifierInput,
} from "../src/verifier/index.js";
import { addTask, defaultTaskList, type TaskList } from "../src/tasklist/index.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixtureDir = resolve(repoRoot, "fixtures/model_responses/harness");

const provider: ProviderInfo = { name: "anthropic", model_id: "fixture", context_window: 200_000 };

function fixturePath(name: string): string {
  return resolve(fixtureDir, name);
}

/**
 * The Default-FAIL evaluator the `exec-evaluator` handle resolves to — the same
 * construction the 12-cordyceps example registers (single read-only turn;
 * neither-pattern ⇒ Failed).
 */
function execEvaluator(): Verifier {
  return new EvaluatorResponseVerifier({
    pass_pattern: "(?i)\\bPASS\\b",
    fail_pattern: "(?i)\\bFAIL\\b",
    max_iterations: 1,
  });
}

/** A fresh `ExecutionRegistry` wired with the cordyceps handles, all node agents
 *  sharing ONE positional replay cursor. */
function cordycepsRegistry(replay: ReplayModelInterface): ExecutionRegistry {
  const agent = (id: string): Agent => new ModelAgent(AgentId.of(id), replay);
  return ExecutionRegistry.builder()
    .agent("planner", agent("planner"))
    .agent("executor", agent("executor"))
    .agent("ralph-agent", agent("ralph-agent"))
    .toolset("plan-tools", new EmptyToolRegistry())
    .toolset("exec-tools", new EmptyToolRegistry())
    .schema("plan-schema", { type: "object" })
    .schema("worker-schema", { type: "array" })
    .verifier("exec-evaluator", execEvaluator())
    .build();
}

/**
 * Build a harness whose plan/worker/evaluator turns all replay positionally from
 * ONE shared `ReplayModelInterface` (a single cursor across the whole composed
 * run), with the cordyceps handles wired into the registry and an optional
 * scripted tool registry (for the worker-side consult).
 */
function harnessFor(
  fixture: string,
  toolRegistry?: ScriptedToolRegistry,
): { h: StandardHarness; storage: StorageProvider } {
  const jsonl = readFileSync(fixturePath(fixture), "utf-8");
  const replay = ReplayModelInterface.fromJsonl(jsonl, provider);
  const storage = StorageProvider.single(new InMemoryStorageProvider());
  const escalationMode: EscalationMode = { kind: "autonomous" };
  const cfg: HarnessConfig = {
    registry: cordycepsRegistry(replay),
    toolRegistry: toolRegistry ?? new ScriptedToolRegistry(),
    sandbox: new AllowAllSandbox(),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    storage,
    escalationMode,
  };
  return { h: new StandardHarness(cfg), storage };
}

/** The canonical cordyceps tree, deserialized from the shared fixture (the same
 *  path the example uses). */
function cordycepsTree(): LoopStrategy {
  const json = readFileSync(resolve(repoRoot, "fixtures/strategy/cordyceps_tree.json"), "utf-8");
  return LoopStrategySchema.parse(JSON.parse(json));
}

/** The PlanExecute subtree of the cordyceps tree (drops the Ralph wrapper) so the
 *  positional fixture maps 1:1 to one window — Ralph's per-window reset loop would
 *  otherwise re-enter and re-consume the (exhausted) replay queue. */
function cordycepsPlanExecute(): LoopStrategy {
  const tree = cordycepsTree();
  if (tree.kind !== "ralph") throw new Error("root is Ralph");
  return tree.inner;
}

function peTask(session: string): Task {
  const t = newTask("audit the repo", SessionId.of(session), cordycepsPlanExecute());
  t.budget.max_turns = 64;
  return t;
}

async function storedList(storage: StorageProvider, session: SessionId): Promise<TaskList> {
  const value = await storage.run().get(session, "task_list");
  return value as TaskList;
}

describe("cordyceps composition fixture replay (#131)", () => {
  // AC5 (static): the canonical tree's per-window worst case is computable before
  // any run; an Unlimited anywhere collapses it to undefined.
  it("max_steps is 25; an Unlimited collapses it to undefined", () => {
    const tree = cordycepsTree();
    expect(loopStrategyMaxSteps(tree)).toBe(25);

    // Swap the worker leaf's PerLoop{12} for Unlimited ⇒ undefined.
    if (tree.kind !== "ralph") throw new Error("unreachable");
    const pe = tree.inner;
    if (pe.kind !== "plan_execute") throw new Error("unreachable");
    const sv = pe.execute;
    if (sv.kind !== "self_verifying") throw new Error("unreachable");
    const worker = sv.inner;
    if (worker.kind !== "react") throw new Error("unreachable");
    worker.budget = { kind: "unlimited" };
    expect(loopStrategyMaxSteps(tree)).toBeUndefined();
  });

  // AC6 (handle re-resolution): a paused cordyceps tree resumes by re-resolving
  // EVERY handle from a freshly-built registry, with no reconfiguration. Load the
  // paused-state fixture carrying the FULL tree, serde round-trip its Task, build
  // a fresh registry, and assert validate() succeeds + every handle resolves.
  it("resume re-resolves handles", () => {
    const raw = readFileSync(
      resolve(repoRoot, "fixtures/paused_states/cordyceps_budget_exhausted.json"),
      "utf-8",
    );
    const doc = JSON.parse(raw);

    // The paused state carries the FULL cordyceps tree in task.loop_strategy.
    const task = TaskSchema.parse(doc.task);

    // Round-trip the Task through the wire (no trait objects — handles are keys).
    const restored = TaskSchema.parse(JSON.parse(JSON.stringify(task)));

    // A fresh registry, built independently (as on a cold resume), re-resolves
    // every handle — proving no reconfiguration of the Task is needed.
    const jsonl = readFileSync(fixturePath("plan_execute_dag_order.jsonl"), "utf-8");
    const registry = cordycepsRegistry(ReplayModelInterface.fromJsonl(jsonl, provider));
    expect(() => registry.validate(restored)).not.toThrow();

    // Spot-check the load-bearing handles resolve concretely.
    if (restored.loop_strategy.kind !== "ralph") throw new Error("root is Ralph after round-trip");
    const ralph = restored.loop_strategy;
    expect(registry.resolveAgent(ralph.agent)).toBeDefined();
    if (ralph.inner.kind !== "plan_execute") throw new Error("unreachable");
    const pe = ralph.inner;
    if (pe.plan.kind !== "react") throw new Error("unreachable");
    expect(registry.resolveAgent(pe.plan.agent)).toBeDefined();
    expect(registry.resolveToolset(pe.plan.toolset)).toBeDefined();
    expect(registry.resolveSchema(pe.plan.output!)).toBeDefined();
    if (pe.execute.kind !== "self_verifying") throw new Error("unreachable");
    const sv = pe.execute;
    expect(registry.resolveVerifier(sv.evaluator)).toBeDefined();
    if (sv.inner.kind !== "react") throw new Error("unreachable");
    expect(registry.resolveAgent(sv.inner.agent)).toBeDefined();
    expect(registry.resolveToolset(sv.inner.toolset)).toBeDefined();

    // The fixture's available_actions advertise the combinator escalation menu.
    const kinds = doc.human_request.available_actions.map((a: { kind: string }) => a.kind);
    expect(kinds).toEqual(["continue_with_budget", "skip", "fail"]);
  });

  // AC2: the plan phase builds a blocker-aware task graph (seeded via task_list,
  // the decision-C authoring path) and the execute phase walks it as a READY-SET,
  // self-verifying each task. Two independent modules both complete in ready-set
  // order; the run succeeds.
  it("plan builds DAG; execute walks the ready-set", async () => {
    const { h, storage } = harnessFor("cordyceps_plan_execute_readyset.jsonl");
    const session = SessionId.of("cordyceps-pe");
    const tl = defaultTaskList();
    addTask(tl, "audit module one", []); // 1
    addTask(tl, "audit module two", []); // 2 (independent)
    await h.persistTaskList(session, tl);

    const r = await h.run({ task: peTask("cordyceps-pe") });
    expect(r.kind).toBe("success");

    // Every ready task was walked and self-verified to completion.
    const after = await storedList(storage, session);
    expect(after.tasks.every((t) => t.status === "completed")).toBe(true);
  });

  // AC4: a single runaway worker node exhausts its own PerLoop{12} budget and
  // FAILS its task; an INDEPENDENT module still completes. The PlanExecute drains
  // to tasks_blocked_by_failure with a partition that does NOT cascade to the
  // unrelated branch.
  it("runaway worker is bounded; unrelated branch completes", async () => {
    const { h } = harnessFor("cordyceps_runaway_bounded.jsonl");
    const session = SessionId.of("cordyceps-runaway");
    const tl = defaultTaskList();
    addTask(tl, "root module", []); // 1 (completes)
    addTask(tl, "runaway module", [1]); // 2 -> 1 (PerLoop{12} budget-Fail)
    addTask(tl, "dependent of runaway", [2]); // 3 -> 2 (cascade-blocked)
    addTask(tl, "independent module", []); // 4 (still completes)
    await h.persistTaskList(session, tl);

    const r = await h.run({ task: peTask("cordyceps-runaway") });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("tasks_blocked_by_failure");
      if (r.reason.kind === "tasks_blocked_by_failure") {
        expect(r.reason.failed_task).toBe(2);
        // The runaway (2) and its transitive dependent (3) are blocked; the root
        // (1) and the UNRELATED independent module (4) both complete — the
        // runaway's bounded failure does NOT cascade to unrelated tasks.
        expect(r.reason.completed).toEqual([1, 4]);
        expect(r.reason.blocked).toEqual([2, 3]);
      }
    }
  });

  // Consult ladder (#114, PRESERVED through the composed tree). A worker leaf
  // consult — with NO `SubagentTool` to mediate it — propagates all the way up to
  // a top-level `RunResult::Consult`. The host (this test) injects an answer via
  // `resumeConsult`, the worker finishes mid-loop, the evaluator passes, and the
  // run completes. This exercises the host-mediation seam the 12-cordyceps
  // example relies on.
  it("worker consult surfaces to the host and host resumes the composed tree", async () => {
    // The GLOBAL tool registry returns a worker-side consult on the first
    // dispatch (the worker's `consult_advisor` call), then defaults to plain
    // success for anything after.
    const toolRegistry = new ScriptedToolRegistry();
    const consultOutput: ToolOutput = {
      kind: "consult",
      request: {
        kind: "advice",
        situation: "found a suspicious unwrap in module one",
        attempts: 1,
        question: "is this a real defect and how severe?",
      },
    };
    toolRegistry.push(consultOutput);

    const { h, storage } = harnessFor("cordyceps_worker_consult.jsonl", toolRegistry);

    // Seed ONE ready task so the execute phase runs exactly one worker.
    const session = SessionId.of("cordyceps-consult");
    const tl = defaultTaskList();
    addTask(tl, "audit module one", []);
    await h.persistTaskList(session, tl);

    // First leg: drive to the consult pause.
    const first = await h.run({ task: peTask("cordyceps-consult") });
    if (first.kind !== "consult") {
      throw new Error(`expected RunResult.consult to surface to the host, got ${first.kind}`);
    }
    const request: ConsultRequest = first.request;
    expect(request.kind).toBe("advice");
    expect(request.question).toContain("real defect");

    // Host mediation: inject the advisor's answer and resume the composed tree.
    const resumed: RunResult = await h.resumeConsult(first.state, {
      kind: "answer",
      text: "Yes — unwrap on untrusted input is a real high-severity panic risk.",
    });
    // The worker continued mid-loop AFTER the consult (the finding it emitted
    // post-answer is the run output) — proving the answer was injected and the
    // SelfVerifying evaluator then cleared the task, not a bare leaf resume.
    expect(resumed.kind).toBe("success");
    if (resumed.kind === "success") {
      expect(resumed.output).toContain("advisor-confirmed");
    }

    // The worker's task self-verified and completed after the consult.
    const after = await storedList(storage, session);
    expect(after.tasks.every((t) => t.status === "completed")).toBe(true);
  });

  // AC3: the registered `exec-evaluator` is Default-FAIL — Passed only on an
  // explicit PASS, Failed on indeterminate output (proving the worker self-checks
  // before a task clears).
  it("self-verify is Default-FAIL", async () => {
    const v = execEvaluator();
    expect(v.maxIterations()).toBe(1);
    const success = (out: string): RunResult => ({
      kind: "success",
      output: out,
      session_id: SessionId.of("s"),
      usage: emptyAggregateUsage(),
      turns: 1,
      session_state: { messages: [], extras: {} },
    });
    const input = (evalText: string): VerifierInput => ({
      build_result: success("audited module"),
      eval_result: success(evalText),
      workspace: "/tmp",
      iteration: 0,
    });
    expect((await v.verify(input("verdict: PASS"))).kind).toBe("passed");
    expect((await v.verify(input("looks plausible"))).kind).toBe("failed");
  });
});
