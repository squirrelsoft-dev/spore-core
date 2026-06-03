/**
 * Fixture-replay test for the SelfVerifying loop strategy (issue #61, R12).
 *
 * Loads `fixtures/harness/self_verifying.json` — the shared cross-language
 * fixture — and replays each case. The build/evaluate agents are scripted to
 * always claim done, so the case's `verdicts` sequence fully determines the
 * outcome:
 *   - the first `pass` verdict at iteration i ⇒ Success.
 *   - all verdicts Failed up to `max_iterations` ⇒
 *     Failure { self_verify_exhausted { iterations: max_iterations } }.
 *   - `misconfigured` cases run with NO verifier ⇒
 *     Failure { self_verify_misconfigured }.
 *
 * Must produce the same outcome as the Rust/Python/Go replays — never edit the
 * fixture to make a failing implementation pass.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
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
  type TokenUsage,
  type TurnResult,
} from "../src/index.js";
import type { Verifier, VerifierInput, VerifierVerdict } from "../src/verifier/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..", "..", "..");
const fixturePath = resolve(repoRoot, "fixtures/harness/self_verifying.json");

interface Case {
  name: string;
  verdicts: string[];
  max_iterations: number;
  expected: { kind: "success" | "exhausted" | "misconfigured"; iterations?: number };
}

const SV_STRATEGY: LoopStrategy = { kind: "self_verifying" };

function usage(): TokenUsage {
  return { input_tokens: 0, output_tokens: 0, cache_read_tokens: null, cache_write_tokens: null };
}

/** Always claims done — the verdict sequence is what drives the outcome. */
class AlwaysDoneAgent implements Agent {
  constructor(private readonly agentId: AgentId) {}
  id(): AgentId {
    return this.agentId;
  }
  async turn(_ctx: Context, _signal?: AbortSignal): Promise<TurnResult> {
    return { kind: "final_response", content: "done", usage: usage() };
  }
}

/** Replays the fixture's `verdicts`: "pass" → passed, any other → failed(reason). */
class FixtureVerifier implements Verifier {
  private i = 0;
  constructor(
    private readonly verdicts: string[],
    private readonly _maxIterations: number,
  ) {}
  async verify(_input: VerifierInput, _signal?: AbortSignal): Promise<VerifierVerdict> {
    const v = this.verdicts[this.i] ?? "default-fail";
    this.i += 1;
    return v === "pass" ? { kind: "passed" } : { kind: "failed", reason: v };
  }
  maxIterations(): number {
    return this._maxIterations;
  }
}

function loadCases(): Case[] {
  const json = JSON.parse(readFileSync(fixturePath, "utf-8")) as { cases: Case[] };
  return json.cases;
}

describe("SelfVerifying loop fixture replay — self_verifying.json", () => {
  for (const c of loadCases()) {
    it(`${c.name} → ${c.expected.kind}`, async () => {
      const agent = new AlwaysDoneAgent(AgentId.of("sv"));
      const verifier =
        c.expected.kind === "misconfigured"
          ? undefined
          : new FixtureVerifier(c.verdicts, c.max_iterations);
      const config: HarnessConfig = {
        agent,
        toolRegistry: new ScriptedToolRegistry(),
        sandbox: new AllowAllSandbox(),
        contextManager: new NoopContextManager(),
        terminationPolicy: new AlwaysContinuePolicy(),
        modelParams: { stop_sequences: [] },
        verifier,
      };
      const h = new StandardHarness(config);
      const task = newTask("do the work", SessionId.of(`sv-${c.name}`), SV_STRATEGY, {
        max_turns: 100,
      });
      const r = await h.run({ task });

      switch (c.expected.kind) {
        case "success":
          expect(r.kind).toBe("success");
          break;
        case "exhausted":
          expect(r.kind).toBe("failure");
          if (r.kind === "failure") {
            expect(r.reason.kind).toBe("self_verify_exhausted");
            if (r.reason.kind === "self_verify_exhausted") {
              expect(r.reason.iterations).toBe(c.expected.iterations);
            }
          }
          break;
        case "misconfigured":
          expect(r.kind).toBe("failure");
          if (r.kind === "failure") {
            expect(r.reason.kind).toBe("self_verify_misconfigured");
          }
          break;
      }
    });
  }
});
