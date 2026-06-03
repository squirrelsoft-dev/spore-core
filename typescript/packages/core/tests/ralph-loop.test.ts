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

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  AgentId,
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
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

const RALPH: LoopStrategy = { kind: "ralph" };
const INCOMPLETE = JSON.stringify({ complete: false, remaining: ["task A"] });
const COMPLETE = JSON.stringify({ complete: true, remaining: [] });

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

/** A sandbox that exposes a fixed workspace root so the Ralph `.spore/` files
 *  resolve to a real tempdir. */
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

function writeFeatureList(root: string, body: string): void {
  mkdirSync(join(root, ".spore"), { recursive: true });
  writeFileSync(join(root, ".spore", "feature_list.json"), body);
}

/** An agent that, on each turn, writes the next scripted progress body to
 *  `.spore/progress.json` then returns a final response. Records every
 *  assembled context so seed/injection/no-carryover assertions are exact. */
class ProgressWritingAgent implements Agent {
  ran = 0;
  readonly contexts: Context[] = [];
  private i = 0;
  constructor(
    private readonly agentId: AgentId,
    private readonly root: string,
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
    writeProgress(this.root, body);
    return { kind: "final_response", content: "window done", usage: usage() };
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

function ralphConfig(root: string, agent: Agent, maxResets = 3): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new WorkspaceSandbox(root),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
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

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describe("Ralph loop strategy (issue #58)", () => {
  // R4: completion pattern incomplete,incomplete,complete → Success at it. 3.
  it("R1/R4: resets until complete → Success at the 3rd window", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), dir, [
      INCOMPLETE,
      INCOMPLETE,
      COMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, 3));
    const r = await h.run({ task: ralphTask() });
    expect(r.kind).toBe("success");
    // Exactly three context windows ran (one agent turn each).
    expect(agent.ran).toBe(3);
  });

  // R5: always-incomplete → exactly maxResets windows → ralph_completion_unmet.
  it("R5: always-incomplete → exactly maxResets windows → ralph_completion_unmet", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), dir, [
      INCOMPLETE,
      INCOMPLETE,
      INCOMPLETE,
      INCOMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, 3));
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
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), dir, [INCOMPLETE]);
    const h = new StandardHarness(ralphConfig(dir, agent, 1));
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
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), dir, [INCOMPLETE, COMPLETE]);
    const h = new StandardHarness(ralphConfig(dir, agent, 3));
    await h.run({ task: ralphTask() });
    expect(agent.contexts.length).toBe(2);
    // Window 1's assistant "window done" output is NOT present in window 2's
    // fresh context — each window is re-seeded from instruction + reload only.
    expect(contextText(agent.contexts[1]!)).not.toContain("window done");
  });

  // R3: the filesystem reload injects `.spore/` state into the fresh seed.
  it("R3: reload injects progress + feature_list into the fresh seed", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    writeProgress(dir, INCOMPLETE);
    writeFeatureList(dir, JSON.stringify([{ name: "login", passes: false }]));
    const agent = new ProgressWritingAgent(AgentId.of("a"), dir, [INCOMPLETE, COMPLETE]);
    const h = new StandardHarness(ralphConfig(dir, agent, 3));
    await h.run({ task: ralphTask() });
    const w0 = contextText(agent.contexts[0]!);
    expect(w0).toContain("Reloaded .spore/progress.json");
    expect(w0).toContain("Reloaded .spore/feature_list.json");
    expect(w0).toContain("login");
  });

  // R6: budgets fold across ALL context windows (each window adds usage).
  it("R6: budgets fold across all context windows", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(AgentId.of("a"), dir, [
      INCOMPLETE,
      INCOMPLETE,
      COMPLETE,
    ]);
    const h = new StandardHarness(ralphConfig(dir, agent, 3));
    const r = await h.run({ task: ralphTask() });
    expect(r.kind).toBe("success");
    if (r.kind === "success") {
      // Three windows × one turn × (1 in, 1 out) folded.
      expect(r.usage.input_tokens).toBe(3);
      expect(r.usage.output_tokens).toBe(3);
    }
  });

  // Completion-status helper: progress complete but a feature fails ⇒ still
  // incomplete (the feature list corroborates).
  it("completion status: feature-list gate overrides a complete progress file", () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-ralph-"));
    writeProgress(dir, COMPLETE);
    writeFeatureList(dir, JSON.stringify([{ name: "login", passes: false }]));
    expect(StandardHarness.ralphCompletionStatus(dir)).toContain("login");
    writeFeatureList(dir, JSON.stringify([{ name: "login", passes: true }]));
    expect(StandardHarness.ralphCompletionStatus(dir)).toBeNull();
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
      agent,
      toolRegistry: new ScriptedToolRegistry(),
      sandbox: new WorkspaceSandbox(dir),
      contextManager: new NoopContextManager(),
      terminationPolicy: new AlwaysContinuePolicy(),
      modelParams: { stop_sequences: [] },
    });
    const task = newTask("do it", SessionId.of("react-1"), { kind: "re_act", max_iterations: 5 });
    const r = await h.run({ task });
    expect(r.kind).toBe("success");
    if (r.kind === "success") expect(r.turns).toBe(1);
  });
});
