/**
 * Regression tests for two composition fixes ported from the Rust reference
 * (`rust/crates/spore-core/src/harness.rs`):
 *
 *   Fix A — Ralph FILLS the worker leaf agent, never REPLACES it.
 *     `fill_empty_worker_agent` (formerly `override_worker_agent`) supplies the
 *     Ralph `agent` to a worker leaf ONLY where the leaf left its handle empty.
 *     An explicitly-declared leaf agent (the architect's node) is authoritative
 *     and must never be shadowed by Ralph's per-window agent.
 *
 *   Fix B — the plan phase honors the declared plan sub-strategy budget.
 *     The plan child now runs under its OWN budget (e.g. a ReAct `PerLoop{4}`);
 *     the global `max_turns` is only the outer backstop. The old `turns + 1`
 *     clamp pinned the planner to a SINGLE turn, starving multi-step task-graph
 *     authoring (the planner could not both call a tool and emit the plan).
 *
 * Each case is constructed to FAIL on the OLD behavior and PASS on the new.
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  ExecutionRegistry,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type SandboxProvider,
  type SandboxViolation,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";
import { EmptyToolRegistry } from "../src/tool-registry/empty.js";
import type { Verifier } from "../src/verifier/index.js";

/** An always-pass verifier so a `SelfVerifying[ReAct]` execute branch resolves
 *  its `evaluator: ""` slot and completes. */
const PASS_VERIFIER: Verifier = {
  async verify() {
    return { kind: "passed" };
  },
  maxIterations() {
    return 3;
  },
};

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

function fr(content: string): TurnResult {
  return { kind: "final_response", content, usage: usage() };
}

function tcr(name = "x"): TurnResult {
  return {
    kind: "tool_call_requested",
    calls: [{ id: "c1", name, input: {} }],
    usage: usage(),
  };
}

/** A recording agent: pops the next scripted result per `turn`, counts runs. */
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
    if (next == null) return fr("done");
    return next;
  }
}

/** Sandbox exposing a fixed workspace root so Ralph's `.spore/` files resolve. */
class WorkspaceSandbox implements SandboxProvider {
  constructor(private readonly root: string) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  workspaceRoot(): string {
    return this.root;
  }
}

function writeProgress(root: string, body: string): void {
  mkdirSync(join(root, ".spore"), { recursive: true });
  writeFileSync(join(root, ".spore", "progress.json"), body);
}

const COMPLETE = JSON.stringify({ complete: true, remaining: [] });

describe("composition: Ralph fills (Fix A) + plan budget honored (Fix B)", () => {
  // ────────────────────────────────────────────────────────────────────────
  // Fix A: Ralph fills empty leaves only — an explicit `executor` leaf wins.
  //
  // Tree: Ralph{agent:"ralph"}[ PlanExecute[ ReAct{planner}, SelfVerifying[
  //         ReAct{executor} ] ] ].
  //
  // The Ralph agent is NON-EMPTY ("ralph"). The execute worker leaf carries an
  // EXPLICIT agent "executor". Under the OLD `overrideWorkerAgent` the execute
  // leaf's handle would be REWRITTEN to "ralph", so the executor agent would
  // never run and every worker turn would dispatch as "ralph". Under the new
  // `fillEmptyWorkerAgent` the explicit "executor" leaf is authoritative and is
  // never shadowed — the executor agent runs the worker step, the ralph agent
  // runs nothing.
  // ────────────────────────────────────────────────────────────────────────
  it("Fix A: a non-empty Ralph agent does NOT shadow an explicit executor leaf", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-fill-"));
    // First worker window marks COMPLETE on disk so Ralph succeeds in one window.
    writeProgress(dir, COMPLETE);

    const planner = new RecordingAgent(AgentId.of("planner")).push(fr('{"tasks":["step"]}'));
    const executor = new RecordingAgent(AgentId.of("executor")).push(fr("did step"));
    const ralphAgent = new RecordingAgent(AgentId.of("ralph"));

    const registry = ExecutionRegistry.builder()
      .agent("planner", planner)
      .agent("executor", executor)
      .agent("ralph", ralphAgent)
      .toolset("", new EmptyToolRegistry())
      .schema("", {})
      .verifier("", PASS_VERIFIER)
      .build();

    const config: HarnessConfig = {
      registry,
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new WorkspaceSandbox(dir),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      maxResets: 2,
    };

    const strategy: LoopStrategy = {
      kind: "ralph",
      inner: {
        kind: "plan_execute",
        plan: {
          kind: "react",
          budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
          agent: "planner",
          toolset: "",
          output: "",
        },
        execute: {
          kind: "self_verifying",
          inner: {
            kind: "react",
            budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
            agent: "executor",
            toolset: "",
            output: "",
          },
          evaluator: "",
        },
      },
      agent: "ralph",
    };

    const h = new StandardHarness(config);
    const r = await h.run({
      task: newTask("implement", SessionId.of("fill-a"), strategy, { max_turns: 8 }),
    });

    expect(r.kind).toBe("success");
    // The explicit executor leaf ran the worker step; the planner authored the
    // plan. The Ralph agent ran NOTHING — it never shadowed a declared leaf.
    expect(executor.ran).toBeGreaterThanOrEqual(1);
    expect(planner.ran).toBeGreaterThanOrEqual(1);
    expect(ralphAgent.ran).toBe(0);
  });

  // ────────────────────────────────────────────────────────────────────────
  // Fix B: the plan phase honors the declared plan sub-strategy budget.
  //
  // The plan child is a ReAct `PerLoop{4}` planner that takes TWO turns: it
  // calls a tool first (turn 1), then emits the JSON plan (turn 2). Under the
  // OLD `min(global, turns + 1)` clamp the plan child was pinned to a SINGLE
  // turn (turns starts at 0 ⇒ cap 1), so the tool-call turn would exhaust the
  // cap before the plan was ever authored and the run would FAIL with nothing
  // stored. Under the new behavior the planner gets its declared 4-turn budget,
  // authors the plan across ≥2 turns, and the run succeeds.
  // ────────────────────────────────────────────────────────────────────────
  it("Fix B: a PerLoop{4} planner takes >1 turn to author the plan (not clamped to 1)", async () => {
    const planner = new RecordingAgent(AgentId.of("planner"))
      .push(tcr()) // turn 1: research via a tool call (would blow the old cap of 1)
      .push(fr('{"tasks":["step"]}')); // turn 2: author the plan
    const executor = new RecordingAgent(AgentId.of("executor")).push(fr("did step"));

    const reg = new ScriptedToolRegistry().push({ kind: "success", content: "ok" });
    const registry = ExecutionRegistry.builder()
      .agent("planner", planner)
      .agent("executor", executor)
      .toolset("", reg)
      .schema("", {})
      .build();

    const config: HarnessConfig = {
      registry,
      toolRegistry: reg,
      sandbox: new AllowAllSandbox(),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    };

    const strategy: LoopStrategy = {
      kind: "plan_execute",
      plan: {
        kind: "react",
        budget: { kind: "per_loop", value: 4 },
        agent: "planner",
        toolset: "",
        output: "",
      },
      execute: {
        kind: "react",
        budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
        agent: "executor",
        toolset: "",
      },
    };

    const h = new StandardHarness(config);
    // No global max_turns backstop, so ONLY the plan sub-strategy's PerLoop{4}
    // governs the plan phase. (A global cap, if present, would only be the
    // outer ceiling — never the per-turn clamp the old code imposed.)
    const r = await h.run({
      task: newTask("build", SessionId.of("plan-budget"), strategy, {}),
    });

    expect(r.kind).toBe("success");
    // The planner authored the plan across ≥2 turns — it was NOT clamped to one.
    expect(planner.ran).toBeGreaterThanOrEqual(2);
    expect(executor.ran).toBeGreaterThanOrEqual(1);
  });
});
