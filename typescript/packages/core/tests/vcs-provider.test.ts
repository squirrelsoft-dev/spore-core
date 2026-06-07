/**
 * Unit tests for the VcsProvider seam (issue #58 v2).
 *
 * Mirrors the inline VcsProvider tests in
 * `rust/crates/spore-core/src/harness.rs` (commit 55f45e8):
 *   (a) FixtureVcsProvider.log returns the seeded string VERBATIM (args ignored).
 *   (b) GitVcsProvider builds the correct `git log` command, asserted via a mock
 *       sandbox capturing the invocation; status() runs `git status`.
 *   (c) Ralph with a FixtureVcsProvider injects the vcs_log string into the
 *       reloaded context across a reset.
 *   (d) Ralph with no vcs provider omits any git section (v1 unchanged).
 *   (e) status() round-trips the seeded string verbatim.
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  GitVcsProvider,
  SessionId,
  StandardHarness,
  newTask,
  type Agent,
  type CommandOutput,
  type Context,
  type HarnessConfig,
  type LoopStrategy,
  type SandboxProvider,
  type SandboxViolation,
  type TokenUsage,
  type ToolCall,
  type TurnResult,
  type VcsLogArgs,
} from "../src/index.js";
import {
  AlwaysContinuePolicy,
  FixtureVcsProvider,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const RALPH: LoopStrategy = {
  kind: "ralph",
  inner: { kind: "react", budget: { kind: "per_loop", value: 1 }, agent: "", toolset: "" },
  agent: "",
};
const INCOMPLETE = JSON.stringify({ complete: false, remaining: ["task A"] });
const COMPLETE = JSON.stringify({ complete: true, remaining: [] });

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

/** A sandbox that captures every executeCommand invocation and returns scripted
 *  stdout, so the built `git` command line can be asserted. */
class CapturingSandbox implements SandboxProvider {
  readonly captured: Array<{ command: string; args: string[] }> = [];
  constructor(private readonly stdout: string) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async executeCommand(
    command: string,
    args: readonly string[],
    _cwd?: string | null,
    _timeoutMs?: number | null,
  ): Promise<CommandOutput | SandboxViolation> {
    this.captured.push({ command, args: [...args] });
    return {
      stdout: this.stdout,
      stderr: "",
      exit_code: 0,
      timed_out: false,
      truncated: false,
    };
  }
  last(): { command: string; args: string[] } {
    return this.captured[this.captured.length - 1]!;
  }
}

/** Sandbox exposing a fixed workspace root so Ralph `.spore/` files resolve. */
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

/** Records every assembled context, writes the next scripted progress body. */
class ProgressWritingAgent implements Agent {
  readonly contexts: Context[] = [];
  private i = 0;
  constructor(
    private readonly root: string,
    private readonly bodies: string[],
  ) {}
  id(): AgentId {
    return AgentId.of("ralph");
  }
  async turn(ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    this.contexts.push(ctx);
    const body = this.bodies[this.i] ?? this.bodies[this.bodies.length - 1] ?? INCOMPLETE;
    this.i += 1;
    writeProgress(this.root, body);
    return { kind: "final_response", content: "window done", usage: usage() };
  }
}

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

function ralphConfig(root: string, agent: Agent): HarnessConfig {
  return {
    agent,
    toolRegistry: new ScriptedToolRegistry(),
    sandbox: new WorkspaceSandbox(root),
    contextManager: new NoopContextManager(),
    terminationPolicy: new AlwaysContinuePolicy(),
    modelParams: { stop_sequences: [] },
    maxResets: 3,
  };
}

function ralphTask() {
  return newTask("implement the feature", SessionId.of("ralph-vcs"), RALPH, { max_turns: 1 });
}

describe("VcsProvider seam (issue #58 v2)", () => {
  // (a) FixtureVcsProvider.log returns the seeded string verbatim, args ignored.
  it("(a) FixtureVcsProvider.log returns the seeded string verbatim", async () => {
    const log = "abc123 first\ndef456 second\n";
    const provider = new FixtureVcsProvider(log, "clean");
    const args: VcsLogArgs = { maxEntries: 5, sinceRef: "HEAD~3", format: "%h %s" };
    expect(await provider.log(args)).toBe(log);
  });

  // (e) status() round-trips the seeded string verbatim.
  it("(e) FixtureVcsProvider.status round-trips the seeded string", async () => {
    const status = "On branch main\nnothing to commit\n";
    const provider = new FixtureVcsProvider("", status);
    expect(await provider.status()).toBe(status);
  });

  // (b) GitVcsProvider builds the correct `git log` command via the sandbox.
  it("(b) GitVcsProvider builds the git log command from VcsLogArgs", async () => {
    const sandbox = new CapturingSandbox("log-output");
    const git = new GitVcsProvider(sandbox, "/work");
    const out = await git.log({ maxEntries: 7, sinceRef: "main", format: "%h %s" });
    expect(out).toBe("log-output");
    expect(sandbox.last().command).toBe("git");
    expect(sandbox.last().args).toEqual(["log", "-n", "7", "--format=%h %s", "main.."]);

    // status() runs `git status`.
    await git.status();
    expect(sandbox.last().command).toBe("git");
    expect(sandbox.last().args).toEqual(["status"]);
  });

  // (b') Minimal args produce just `git log -n <N>` (no format/range flags).
  it("(b') GitVcsProvider minimal args produce just `git log -n <N>`", async () => {
    const sandbox = new CapturingSandbox("");
    const git = new GitVcsProvider(sandbox, "/work");
    await git.log({ maxEntries: 3 });
    expect(sandbox.last().args).toEqual(["log", "-n", "3"]);
  });

  // (c) Ralph with a FixtureVcsProvider injects the vcs_log into the reload.
  it("(c) Ralph injects the vcs_log into the reloaded context across a reset", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-vcs-"));
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(dir, [INCOMPLETE, COMPLETE]);
    const config: HarnessConfig = {
      ...ralphConfig(dir, agent),
      vcsProvider: new FixtureVcsProvider("cafe123 implement login\nbeef456 add tests\n", "clean"),
    };
    const h = new StandardHarness(config);
    await h.run({ task: ralphTask() });
    const w0 = contextText(agent.contexts[0]!);
    expect(w0).toContain("Recent VCS history:");
    expect(w0).toContain("cafe123 implement login");
  });

  // (d) Ralph with no vcs provider omits any git section (v1 unchanged).
  it("(d) Ralph with no vcs provider omits any git section", async () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-vcs-"));
    writeProgress(dir, INCOMPLETE);
    const agent = new ProgressWritingAgent(dir, [COMPLETE]);
    const config = ralphConfig(dir, agent);
    expect(config.vcsProvider).toBeUndefined();
    const h = new StandardHarness(config);
    await h.run({ task: ralphTask() });
    expect(contextText(agent.contexts[0]!)).not.toContain("Recent VCS history:");
  });
});
