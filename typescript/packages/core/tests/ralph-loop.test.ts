/**
 * Unit tests for the Ralph loop strategy (issue #58).
 *
 * Mirrors the inline `run_ralph` tests in
 * `rust/crates/spore-core/src/harness.rs` — same rules, same `.spore/` contract.
 *
 * Rules covered (R1–R7):
 *   R1 the model's exit attempt RESETS the context window while incomplete.
 *   R2 each reset builds a FRESH SessionState — no message carryover.
 *   R3 the filesystem reload injects `.spore/` state into the fresh seed.
 *   R4 incomplete,incomplete,complete → Success at iteration 3.
 *   R5 always-incomplete → exactly maxResets windows → ralph_completion_unmet.
 *   R6 budgets fold across ALL context windows.
 *   R7 the registered Stop hook is INERT without a progress file (non-Ralph runs).
 */

import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  SessionId,
  StandardHarness,
  newTask,
  storage,
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

const { ProjectId, StorageProvider, InMemoryStorageProvider, RALPH_PROGRESS_KEY, RALPH_FEATURE_LIST_KEY } =
  storage;
import {
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
  registryWith,
} from "../src/harness/testing.js";
import type { Verifier, VerifierInput, VerifierVerdict } from "../src/verifier/index.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

// #125: the build leaf is UNBOUNDED (matching the Rust `ralph_task`'s
// `react(MAX)` inner); the per-window turn ceiling comes from the GLOBAL
// `max_turns: 1` backstop, NOT the leaf's own cap. With a tight `per_loop` leaf
// cap the leaf would propagate a budget_exhausted that Ralph treats as "window
// incomplete → reset" BEFORE consulting `.spore/`, masking the completion check
// these cases exercise.
const RALPH: LoopStrategy = {
  kind: "ralph",
  inner: {
    kind: "react",
    budget: { kind: "per_loop", value: Number.MAX_SAFE_INTEGER },
    agent: "",
    toolset: "",
  },
  agent: "",
};
const INCOMPLETE = JSON.stringify({ complete: false, remaining: ["task A"] });
const COMPLETE = JSON.stringify({ complete: true, remaining: [] });

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

/** A sandbox that exposes a fixed workspace root. #142: the Ralph checkpoint no
 *  longer lives on `.spore/` — it lives in the durable project-id RunStore — but
 *  the sandbox still exposes a workspace root the harness derives the default
 *  project id from. Tests PIN the project id explicitly so the writer and reader
 *  agree regardless of symlink/canonicalization. */
class WorkspaceSandbox implements SandboxProvider {
  constructor(private readonly root: string) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  workspaceRoot(): string {
    return this.root;
  }
}

/**
 * Per-test shared Ralph store + pinned project id (#142). The checkpoint moved
 * off the `.spore/` filesystem onto the durable project-id RunStore, so the
 * test's writer (the agent) and reader (the harness) MUST share ONE storage
 * provider and ONE project id — what the agent writes is what the harness reads.
 */
function ralphStore(): { storage: storage.StorageProvider; projectId: storage.ProjectId } {
  return {
    storage: StorageProvider.single(new InMemoryStorageProvider()),
    projectId: ProjectId.fromCanonicalPath("/ralph-test-project"),
  };
}

/** Write the Ralph progress checkpoint into the SHARED run store under the
 *  pinned project namespace (#142 relocated this off `.spore/`). `body` is the
 *  legacy JSON body string, parsed and stored under {@link RALPH_PROGRESS_KEY}. */
async function writeProgress(
  store: storage.StorageProvider,
  projectId: storage.ProjectId,
  body: string,
): Promise<void> {
  await store.run().put(projectId.namespace(), RALPH_PROGRESS_KEY, JSON.parse(body));
}

async function writeFeatureList(
  store: storage.StorageProvider,
  projectId: storage.ProjectId,
  body: string,
): Promise<void> {
  await store.run().put(projectId.namespace(), RALPH_FEATURE_LIST_KEY, JSON.parse(body));
}

/** An agent that, on each turn, writes the next scripted progress body to the
 *  shared project-id RunStore then returns a final response. Records every
 *  assembled context so seed/injection/no-carryover assertions are exact. */
class ProgressWritingAgent implements Agent {
  ran = 0;
  readonly contexts: Context[] = [];
  private i = 0;
  constructor(
    private readonly agentId: AgentId,
    private readonly store: storage.StorageProvider,
    private readonly projectId: storage.ProjectId,
    private readonly bodies: string[],
  ) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.ran += 1;
    this.contexts.push(ctx);
    const body = this.bodies[this.i] ?? this.bodies[this.bodies.length - 1] ?? INCOMPLETE;
    this.i += 1;
    await writeProgress(this.store, this.projectId, body);
    return { kind: "final_response", content: "window done", usage: usage() };
  }
}

/**
 * An agent that NEVER returns a final response — every turn requests a tool
 * call, so a bounded ReAct window can only ever exhaust its turn budget (it
 * never terminates cleanly). On its FIRST turn it writes `COMPLETE` to the shared
 * project-id RunStore, so any code path that DID consult Ralph's external
 * completion after the window would (wrongly) see "complete" and Success.
 * Mirrors the Rust `ToolLoopingAgent` (#125 F5).
 */
class ToolLoopingAgent implements Agent {
  calls = 0;
  constructor(
    private readonly agentId: AgentId,
    private readonly store: storage.StorageProvider,
    private readonly projectId: storage.ProjectId,
  ) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(_ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    if (this.calls === 0) {
      // Mark COMPLETE in the store up front — if Ralph (wrongly) consulted
      // completion after a budget-exhausted window it would Success.
      await writeProgress(this.store, this.projectId, COMPLETE);
    }
    const n = this.calls;
    this.calls += 1;
    const call: ToolCall = { id: `c${n}`, name: "x", input: {} };
    return { kind: "tool_call_requested", calls: [call], usage: usage() };
  }
}

/** Flatten a Context's text content for substring assertions. */
function contextText(ctx: Context): string {
  return ctx.messages
    .map((m) => {
      const c = m.content;
      if (Array.isArray(c)) return c.map((p) => ("text" in p ? p.text : "")).join(" ");
      if (typeof c === "object" && c != null && "text" in c) return (c as { text: string }).text;
      return typeof c === "string" ? c : "";
    })
    .join("\n");
}

function ralphConfig(
  root: string,
  agent: Agent,
  store: storage.StorageProvider,
  projectId: storage.ProjectId,
  maxResets = 3,
): HarnessConfig {
  return {
    registry: registryWith({ agent }),
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new WorkspaceSandbox(root),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    // #142: the harness and the test writer-agent MUST share ONE store + project
    // id so the checkpoint the agent writes is the one the harness reads.
    storage: store,
    projectId,
    maxResets,
  };
}

function ralphTask() {
  // One ReAct turn per context window keeps the per-window sub-loop bounded so
  // the OUTER reset loop drives the test deterministically (mirrors Rust's
  // `ralph_task`). The registered `ralph-stop` hook blocks on each incomplete
  // final response; max_turns=1 caps the window at one turn, then the OUTER
  // loop checks completion and resets.
  return newTask("implement the feature", SessionId.of("ralph-build"), RALPH, { max_turns: 1 });
}

/** An always-pass verifier that records every input it sees (#124). */
class CountingVerifier implements Verifier {
  readonly seen: VerifierInput[] = [];
  async verify(input: VerifierInput, _signal?: AbortSignal): Promise<VerifierVerdict> {
    this.seen.push(input);
    return { kind: "passed" };
  }
  maxIterations(): number {
    return 3;
  }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("Ralph loop strategy (issue #58)", () => {
  // R4: completion pattern incomplete,incomplete,complete → Success at it. 3.
  it("R1/R4: resets until complete → Success at the 3rd window", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [
      INCOMPLETE,
      INCOMPLETE,
      COMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, store, projectId, 3));
    const r = await h.run({ task: ralphTask() });
    expect(r.kind).toBe("success");
    // Exactly three context windows ran (one agent turn each).
    expect(agent.ran).toBe(3);
  });

  // R5: always-incomplete → exactly maxResets windows → ralph_completion_unmet.
  it("R5: always-incomplete → exactly maxResets windows → ralph_completion_unmet", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [
      INCOMPLETE,
      INCOMPLETE,
      INCOMPLETE,
      INCOMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, store, projectId, 3));
    const r = await h.run({ task: ralphTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("ralph_completion_unmet");
      if (r.reason.kind === "ralph_completion_unmet") {
        expect(r.reason.iterations).toBe(3);
        expect(r.reason.last_reason).toContain("task A");
      }
    }
    expect(agent.ran).toBe(3);
  });

  // R5 boundary: maxResets = 1 → a single window → ralph_completion_unmet.
  it("R5: maxResets=1, always-incomplete → single window → ralph_completion_unmet", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [INCOMPLETE]);
    const h = new StandardHarness(ralphConfig(dir, agent, store, projectId, 1));
    const r = await h.run({ task: ralphTask() });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure" && r.reason.kind === "ralph_completion_unmet") {
      expect(r.reason.iterations).toBe(1);
    }
    expect(agent.ran).toBe(1);
  });

  // R2: each reset builds a FRESH SessionState — no message carryover.
  it("R2: fresh session per reset (no carryover of window 1's output)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [
      INCOMPLETE,
      COMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, store, projectId, 3));
    await h.run({ task: ralphTask() });
    expect(agent.contexts.length).toBe(2);
    // Window 1's assistant "window done" output is NOT present in window 2's
    // fresh context — each window is re-seeded from instruction + reload only.
    expect(contextText(agent.contexts[1]!)).not.toContain("window done");
  });

  // R3: the durable checkpoint reload injects project-store state into the fresh
  // seed (#142: the reload now reads the project-id RunStore, not `.spore/`).
  it("R3: reload injects progress + feature_list into the fresh seed", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    await writeFeatureList(store, projectId, JSON.stringify([{ name: "login", passes: false }]));
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [
      INCOMPLETE,
      COMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, store, projectId, 3));
    await h.run({ task: ralphTask() });
    const w0 = contextText(agent.contexts[0]!);
    // The reload prefix is retained byte-stable across the relocation.
    expect(w0).toContain("Reloaded .spore/progress.json");
    expect(w0).toContain("Reloaded .spore/feature_list.json");
    expect(w0).toContain("login");
  });

  // R6: budgets fold across ALL context windows (each window adds usage).
  it("R6: budgets fold across all context windows", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [
      INCOMPLETE,
      INCOMPLETE,
      COMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, store, projectId, 3));
    const r = await h.run({ task: ralphTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      // Three windows × one turn × (1 in, 1 out) folded.
      expect(r.usage.input_tokens).toBe(3);
      expect(r.usage.output_tokens).toBe(3);
    }
  });

  // Completion-status helper: progress complete but a feature fails ⇒ still
  // incomplete (the feature list corroborates). #142: reads from the project-id
  // RunStore via the store-based helper.
  it("completion status: feature-list gate overrides a complete progress file", async () => {
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, COMPLETE);
    await writeFeatureList(store, projectId, JSON.stringify([{ name: "login", passes: false }]));
    expect(
      await StandardHarness.ralphCompletionStatusFromStore(store.run(), projectId),
    ).toContain("login");
    await writeFeatureList(store, projectId, JSON.stringify([{ name: "login", passes: true }]));
    expect(
      await StandardHarness.ralphCompletionStatusFromStore(store.run(), projectId),
    ).toBeNull();
  });

  // R7: the registered Stop hook is INERT for a workspace without a progress
  // file — a plain ReAct run terminates in one turn.
  it("R7: stop hook inert without a progress file (non-Ralph ReAct run)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    const agent: Agent = {
      id: () => AgentId.of("a"),
      turn: async () => ({ kind: "final_response", content: "done", usage: usage() }),
    };
    const h = new StandardHarness({
      registry: registryWith({ agent }),
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new WorkspaceSandbox(dir),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    });
    const task = newTask("do it", SessionId.of("react-1"), {
      kind: "react",
      budget: { kind: "per_loop", value: 5 },
      agent: "",
      toolset: "",
    });
    const r = await h.run({ task });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.turns).toBe(1);
  });

  it("ralph_runs_non_react_inner_per_window (#124): the inner SelfVerifying verifier fires once per window", async () => {
    // #124: Ralph GENUINELY recurses into `inner` per window. With a non-ReAct
    // inner (SelfVerifying[ReAct]) the inner verifier must fire at least once per
    // window. Always-incomplete progress ⇒ Ralph resets until max_resets (2) is
    // exhausted; the verifier fires once per window (>=2). A hardcoded-ReAct
    // window would record ZERO verifier invocations.
    const dir = mkdtempSync(join(tmpdir(), "ralph-nonreact-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), store, projectId, [INCOMPLETE]);
    const verifier = new CountingVerifier();
    const config: HarnessConfig = {
      registry: registryWith({ agent, verifier }),
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new WorkspaceSandbox(dir),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      storage: store,
      projectId,
      maxResets: 2,
    };
    const strategy: LoopStrategy = {
      kind: "ralph",
      inner: {
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
      agent: "",
    };
    const h = new StandardHarness(config);
    const task = newTask("implement", SessionId.of("ralph-nonreact"), strategy, { max_turns: 8 });
    const r = await h.run({ task });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      expect(r.reason.kind).toBe("ralph_completion_unmet");
      if (r.reason.kind === "ralph_completion_unmet") {
        expect(r.reason.iterations).toBe(2); // exactly max_resets windows ran
      }
    }
    // The inner SelfVerifying verifier fired at least once per window (>=2).
    expect(verifier.seen.length).toBeGreaterThanOrEqual(2);
  });

  // #125 F5: a Ralph window whose INNER LEAF exhausts its OWN budget mid-window
  // (the leaf's per_loop policy is the binding cap, no smaller global backstop).
  // The window surfaces StrategyOutcome `budget_exhausted`; Ralph must treat it
  // as "window incomplete → RESET and retry" — it must NOT consult external
  // completion (which is COMPLETE in the store here) and must NOT cascade the
  // child's exhaustion into its own terminal. So the run reaches max_resets
  // windows and ends `ralph_completion_unmet`, NOT Success.
  //
  // NOTE: the RALPH helper above uses an UNBOUNDED leaf to AVOID this path
  // (Deviation #14c); this case adds explicit coverage of the BOUNDED path.
  it("F5: a budget-exhausted window resets (no completion consult, no cascade)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-f5-"));
    const { storage: store, projectId } = ralphStore();
    await writeProgress(store, projectId, INCOMPLETE);
    const agent = new ToolLoopingAgent(AgentId.of("ralph-tool-loop"), store, projectId);
    // Provide tool outputs for the looping calls across all windows.
    const reg = new ScriptedToolRegistry();
    for (let i = 0; i < 32; i += 1) reg.push({ kind: "success", content: "ok" });
    const config: HarnessConfig = {
      registry: registryWith({ agent }),
      toolRegistry: reg,
      sandbox: new WorkspaceSandbox(dir),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
      storage: store,
      projectId,
      maxResets: 3,
    };
    const h = new StandardHarness(config);

    // Inner leaf carries its OWN binding cap (per_loop{2}); NO global max_turns
    // backstop, so the leaf policy — not the global cap — exhausts the window
    // and the new budget_exhausted path is taken.
    const strategy: LoopStrategy = {
      kind: "ralph",
      inner: {
        kind: "react",
        budget: { kind: "per_loop", value: 2 },
        agent: "",
        toolset: "",
      },
      agent: "",
    };
    // No global cap (empty BudgetLimits).
    const task = newTask("implement the feature", SessionId.of("ralph-build"), strategy, {});

    const r = await h.run({ task });
    expect(r.kind).toBe("failure");
    if (r.kind === "failure") {
      // Reached max_resets — completion was NEVER consulted (despite COMPLETE on
      // disk) and the child's exhaustion did NOT cascade into Ralph's terminal.
      expect(r.reason.kind).toBe("ralph_completion_unmet");
      if (r.reason.kind === "ralph_completion_unmet") {
        expect(r.reason.iterations).toBe(3); // exactly max_resets windows ran
      }
    }
    // Three windows × two leaf turns each = six agent turns total — proving each
    // exhausted window fully reset and re-ran, not short-circuited.
    expect(agent.calls).toBe(6);
  });
});
