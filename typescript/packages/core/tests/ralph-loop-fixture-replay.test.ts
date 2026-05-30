/**
 * Fixture-replay test for the Ralph loop strategy (issue #58).
 *
 * Loads `fixtures/harness/ralph.json` — the shared cross-language fixture — and
 * replays each case. Each case scripts the per-window progress body the agent
 * writes to `.spore/progress.json`; the strategy resets the context window
 * (fresh SessionState, reload `.spore/`) until the completion check passes or
 * `max_resets` windows are exhausted:
 *   - first complete (with empty remaining) window at index i ⇒ Success, i+1
 *     windows run.
 *   - all windows incomplete up to max_resets ⇒
 *     Failure { ralph_completion_unmet { iterations: max_resets } }.
 *
 * Must produce the same outcome as the Rust/Python/Go replays — never edit the
 * fixture to make a failing implementation pass.
 */

import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

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
  FixtureVcsProvider,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/harness/ralph.json");

const RALPH: LoopStrategy = { kind: "ralph" };
const INCOMPLETE = JSON.stringify({ complete: false, remaining: ["task A"] });

interface Window {
  complete: boolean;
  remaining?: string[];
}
interface Case {
  name: string;
  windows: Window[];
  max_resets: number;
  /** Optional git-log string (issue #58 v2). When present a FixtureVcsProvider
   *  seeded with it is wired into the harness and the reloaded context of the
   *  first fresh window MUST contain it. Absent ⇒ no provider ⇒ no git section. */
  vcs_log?: string;
  expected: { kind: "success" | "completion_unmet"; iterations: number };
}

function usage(): TokenUsage {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

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

/** Writes the next scripted progress body each turn, then claims done. */
class ProgressWritingAgent implements Agent {
  ran = 0;
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
    this.ran += 1;
    this.contexts.push(ctx);
    const body = this.bodies[this.i] ?? this.bodies[this.bodies.length - 1] ?? INCOMPLETE;
    this.i += 1;
    writeProgress(this.root, body);
    return { kind: "final_response", content: "done", usage: usage() };
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

function loadCases(): Case[] {
  const json = JSON.parse(readFileSync(fixturePath, "utf-8")) as { cases: Case[] };
  return json.cases;
}

describe("Ralph loop fixture replay — ralph.json", () => {
  for (const c of loadCases()) {
    it(`${c.name} → ${c.expected.kind}`, async () => {
      const dir = mkdtempSync(join(tmpdir(), "spore-ralph-fx-"));
      // Seed an initial incomplete progress file so window 1 reloads state.
      writeProgress(dir, INCOMPLETE);
      const bodies = c.windows.map((w) =>
        JSON.stringify({ complete: w.complete, remaining: w.remaining ?? [] }),
      );
      const agent = new ProgressWritingAgent(dir, bodies);
      const config: HarnessConfig = {
        agent,
        toolRegistry: new ScriptedToolRegistry(),
        sandbox: new WorkspaceSandbox(dir),
        contextManager: new NoopContextManager(),
        terminationPolicy: new AlwaysContinuePolicy(),
        maxResets: c.max_resets,
        // issue #58 v2: when the case carries a `vcs_log`, wire a
        // FixtureVcsProvider seeded with it; absent ⇒ no provider ⇒ no git
        // section (v1 behavior).
        vcsProvider: c.vcs_log != null ? new FixtureVcsProvider(c.vcs_log) : undefined,
      };
      const h = new StandardHarness(config);
      // One ReAct turn per context window (mirrors Rust's `ralph_task`): the
      // registered `ralph-stop` hook blocks each incomplete window; max_turns=1
      // bounds the window so the OUTER reset loop drives the outcome.
      const task = newTask("do the work", SessionId.of(`ralph-${c.name}`), RALPH, {
        max_turns: 1,
      });
      const r = await h.run({ task });

      // issue #58 v2: when a vcs_log is present, the first fresh window must
      // include it under a delimited "Recent VCS history:" section.
      if (c.vcs_log != null) {
        const w0 = contextText(agent.contexts[0]!);
        expect(w0).toContain("Recent VCS history:");
        expect(w0).toContain(c.vcs_log.trim());
      }

      if (c.expected.kind === "success") {
        expect(r.kind).toBe("success");
        // `iterations` is the number of context windows run.
        expect(agent.ran).toBe(c.expected.iterations);
      } else {
        expect(r.kind).toBe("failure");
        if (r.kind === "failure") {
          expect(r.reason.kind).toBe("ralph_completion_unmet");
          if (r.reason.kind === "ralph_completion_unmet") {
            expect(r.reason.iterations).toBe(c.expected.iterations);
          }
        }
      }
    });
  }
});
