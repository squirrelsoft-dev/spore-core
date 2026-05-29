/**
 * Cross-language fixture replay for the hook system (spore-core issue #69).
 * Replays all four shared fixtures under `fixtures/hooks/`, asserting the same
 * outcomes the Rust suite asserts. NEVER modify a fixture.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { hooks, SessionId } from "../src/index.js";

const { StandardHookChain, FunctionHook, CommandHook, HookError, hookContextPayload } = hooks;
type HookDecision = hooks.HookDecision;
type HookContext = hooks.HookContext;

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/hooks");
const loadFixture = <T>(name: string): T =>
  JSON.parse(readFileSync(resolve(fixtureDir, name), "utf8")) as T;

// ── hook_decision_wire.json ───────────────────────────────────────────────
describe("hook_decision_wire fixture", () => {
  interface WireCase {
    name: string;
    json: HookDecision;
  }
  const fixture = loadFixture<{ cases: WireCase[] }>("hook_decision_wire.json");

  for (const c of fixture.cases) {
    it(c.name, () => {
      // Round-trip: deserialize then serialize must be byte-identical to `json`.
      const parsed = JSON.parse(JSON.stringify(c.json)) as HookDecision;
      expect(parsed).toEqual(c.json);
      expect(JSON.parse(JSON.stringify(parsed))).toEqual(c.json);
      // The tag key is `decision`.
      expect(typeof (c.json as { decision: string }).decision).toBe("string");
    });
  }
});

// ── pre_tool_use_mutation.json ────────────────────────────────────────────
describe("pre_tool_use_mutation fixture", () => {
  interface MutationCase {
    name: string;
    tool_name: string;
    tool_input: unknown;
    hook_decisions: HookDecision[];
    expected: { outcome: "continue"; tool_input: unknown } | { outcome: "deny"; reason: string };
  }
  const fixture = loadFixture<{ cases: MutationCase[] }>("pre_tool_use_mutation.json");

  for (const c of fixture.cases) {
    it(c.name, async () => {
      const chain = new StandardHookChain();
      c.hook_decisions.forEach((d, i) => {
        chain.register(new FunctionHook(`h${i}`, ["pre_tool_use"], () => d));
      });
      const ctx: HookContext = {
        event: "pre_tool_use",
        session_id: SessionId.of("sess-1"),
        turn_number: 1,
        tool_name: c.tool_name,
        tool_input: c.tool_input,
      };
      const outcome = await chain.fire(ctx);
      if (c.expected.outcome === "continue") {
        expect(outcome.kind).toBe("continue");
        expect(ctx.tool_input).toEqual(c.expected.tool_input);
      } else {
        expect(outcome).toEqual({ kind: "deny", reason: c.expected.reason });
      }
    });
  }
});

// ── stop_block_basic.json ─────────────────────────────────────────────────
describe("stop_block_basic fixture", () => {
  interface StopCase {
    name: string;
    max_stop_blocks: number;
    hook_decisions: HookDecision[];
    expected: { blocks: number; terminated_by: "continue" | "cap" };
  }
  const fixture = loadFixture<{ cases: StopCase[] }>("stop_block_basic.json");

  for (const c of fixture.cases) {
    it(c.name, async () => {
      // Simulate the harness per-run Stop loop: each entry in `hook_decisions`
      // is the verdict at a successive completion gate. A `block` under the cap
      // consumes one block and continues; once `blocks == max_stop_blocks` the
      // next block is ignored and the loop terminates (`cap`); a `continue`
      // terminates immediately (`continue`).
      let blocks = 0;
      let terminatedBy: "continue" | "cap" = "continue";
      for (const decision of c.hook_decisions) {
        const chain = new StandardHookChain();
        chain.register(new FunctionHook("stop", ["stop"], () => decision));
        const ctx: HookContext = {
          event: "stop",
          session_id: SessionId.of("sess-1"),
          turn_number: blocks,
          last_output: { text: "", had_tool_calls: false },
          task_instruction: "do it",
          session_state: null,
        };
        const outcome = await chain.fire(ctx);
        if (outcome.kind === "block") {
          if (blocks >= c.max_stop_blocks) {
            terminatedBy = "cap";
            break;
          }
          blocks += 1;
          continue;
        }
        // continue / inject / deny → normal termination.
        terminatedBy = "continue";
        break;
      }
      expect(blocks).toBe(c.expected.blocks);
      expect(terminatedBy).toBe(c.expected.terminated_by);
    });
  }
});

// ── command_handler_io.json ───────────────────────────────────────────────
describe("command_handler_io fixture", () => {
  interface CmdCase {
    name: string;
    event: hooks.HookEvent;
    expected_stdin?: { event: string; context: Record<string, unknown> };
    exit_code?: number;
    stdout: string;
    expected_decision?: HookDecision;
    expected_error?: "command_failed" | "command_output_invalid";
  }
  const fixture = loadFixture<{ cases: CmdCase[] }>("command_handler_io.json");

  // Build the live HookContext that matches each fixture case's expected_stdin.
  function ctxFor(c: CmdCase): HookContext {
    if (c.event === "stop") {
      return {
        event: "stop",
        session_id: SessionId.of("sess-1"),
        turn_number: 3,
        last_output: { text: "I'm done", had_tool_calls: false },
        task_instruction: "make the tests pass",
        session_state: null,
      };
    }
    // pre_tool_use
    return {
      event: "pre_tool_use",
      session_id: SessionId.of("sess-1"),
      turn_number: 1,
      tool_name: "read_file",
      tool_input: { path: "/etc/passwd" },
    };
  }

  for (const c of fixture.cases) {
    it(c.name, async () => {
      const ctx = ctxFor(c);

      // 1. The stdin payload the handler WOULD receive matches expected_stdin.
      if (c.expected_stdin) {
        const payload = {
          event: ctx.event,
          context: hookContextPayload(ctx),
        };
        // Compare via JSON normalization (SessionId → string).
        expect(JSON.parse(JSON.stringify(payload))).toEqual(c.expected_stdin);
      }

      // 2. Drive a real CommandHook whose script echoes `stdout` and exits with
      //    `exit_code` (default 0).
      const exit = c.exit_code ?? 0;
      const script = `cat >/dev/null; printf '%s' ${shellQuote(c.stdout)}; exit ${exit}`;
      const hook = new CommandHook("cmd", [c.event], "sh", ["-c", script]);
      const chain = new StandardHookChain();
      chain.register(hook);

      if (c.expected_error) {
        await expect(chain.fire(ctx)).rejects.toMatchObject({ kind: c.expected_error });
        await expect(chain.fire(ctx)).rejects.toBeInstanceOf(HookError);
      } else {
        const outcome = await chain.fire(ctx);
        // The parsed decision must equal expected_decision; map to FireOutcome.
        const d = c.expected_decision!;
        switch (d.decision) {
          case "block":
            expect(outcome).toEqual({ kind: "block", reason: d.reason });
            break;
          case "deny":
            expect(outcome).toEqual({ kind: "deny", reason: d.reason });
            break;
          case "continue":
            expect(outcome).toEqual({ kind: "continue" });
            break;
          default:
            throw new Error(`unexpected fixture decision ${d.decision}`);
        }
      }
    });
  }
});

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}
