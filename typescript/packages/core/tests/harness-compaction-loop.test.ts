/**
 * Harness post-compaction verify→retry→warn loop (spore-core issue #46).
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs#tests::compaction` — same
 * rules, same verdicts, parallel structure. Drives the loop with a mock Agent,
 * a stub verifier, and the in-memory observability provider, plus a
 * fixture-replay against `fixtures/compaction_loop/cases.json`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  AgentId,
  HarnessBuilder,
  SessionId,
  TaskId,
  emptySessionState,
  newTask,
  type Agent,
  type CompactionTurn,
  type Context,
  type ContextManager,
  type SessionState as HarnessSessionState,
  type Task,
  type ToolResultRecord,
  type TurnResult,
  context as contextNs,
  observability as obsNs,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

type CompactionPreserveHints = contextNs.CompactionPreserveHints;
type CompactionVerificationResult = contextNs.CompactionVerificationResult;
type CompactionVerifier = contextNs.CompactionVerifier;
type ObservabilityProvider = obsNs.ObservabilityProvider;
const { InMemoryObservabilityProvider } = obsNs;
const newSessionState = contextNs.newSessionState;

const here = dirname(fileURLToPath(import.meta.url));

const SID = SessionId.of("s1");
const TID = TaskId.of("t1");

function usage() {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

/**
 * A ContextManager that always offers a compaction turn. Records how many times
 * `applyCompaction` ran and the contexts the agent saw (via the agent stub).
 */
class CompactingContextManager implements ContextManager {
  applied = 0;
  constructor(private readonly should: boolean) {}

  async assemble(session: HarnessSessionState): Promise<Context> {
    return { messages: session.messages.slice(), tools: [], params: { stop_sequences: [] } };
  }
  async appendToolResult(session: HarnessSessionState, _result: ToolResultRecord): Promise<void> {
    session.messages.push({ role: "tool", content: { type: "text", text: "tool" } });
  }
  async appendUserMessage(): Promise<void> {}
  shouldCompact(): boolean {
    return this.should;
  }
  prepareCompactionTurn(): CompactionTurn | undefined {
    const vs = newSessionState(SID, TID, "deploy the payment service");
    vs.token_budget_used = 1000;
    return {
      context: {
        messages: [{ role: "user", content: { type: "text", text: "please summarize" } }],
        tools: [],
        params: { stop_sequences: [] },
      },
      preserveHints: contextNs.defaultCompactionPreserveHints(),
      verificationState: vs,
      messagesRemoved: 3,
    };
  }
  applyCompaction(): void {
    this.applied += 1;
  }
}

/** Verifier that fails the first `failFirst` calls, then passes. */
class ScriptedVerifier implements CompactionVerifier {
  private remaining: number;
  private readonly missing: string[];
  constructor(failFirst: number, missing: string[]) {
    this.remaining = failFirst;
    this.missing = missing.slice();
  }
  verify(
    _summary: string,
    _hints: CompactionPreserveHints,
    _state: contextNs.SessionState,
  ): CompactionVerificationResult {
    if (this.remaining > 0) {
      this.remaining -= 1;
      return { passed: false, missingItems: this.missing.slice(), detail: "scripted fail" };
    }
    return { passed: true, missingItems: [], detail: "scripted pass" };
  }
}

function harnessWith(
  cm: CompactingContextManager,
  agent: Agent,
  verifier: CompactionVerifier,
  obs: ObservabilityProvider,
  maxAttempts: number,
) {
  return new HarnessBuilder(
    agent,
    new ScriptedToolRegistry(),
    new AllowAllSandbox(),
    cm,
    new AlwaysContinuePolicy(),
  )
    .observability(obs)
    .compactionVerifier(verifier)
    .maxCompactionAttempts(maxAttempts)
    .build();
}

/**
 * Drive the loop by running a full ReAct turn that ends with a final response
 * after one tool call. The compaction loop fires after the tool result is
 * appended (the agent's first non-compaction turn requests a tool, the second
 * returns done). The RecordingAgent's `seen` only records compaction-turn
 * contexts because the real loop assembles its own per-turn context via the
 * ContextManager; to isolate the compaction loop we feed the harness a small
 * task and let the loop's tool-call path trigger compaction.
 *
 * Simpler: exercise `runCompaction` exactly as the loop does via a single
 * helper that mirrors Rust's `drive`. We invoke the harness `run` with an agent
 * sequence (tool_call → final_response) so one compaction pass runs.
 */
function task(): Task {
  return newTask("do something", SID, { kind: "re_act", max_iterations: 5 });
}

/**
 * Agent that drives ONE tool call then a final response, while the compaction
 * agent (separate) is the RecordingAgent. We cannot share one agent for both
 * the loop turns and the compaction turns and still count compaction turns
 * cleanly, so the loop is exercised through the harness with the
 * RecordingAgent handling BOTH: first call = tool_call_requested (so a tool
 * result is appended and compaction fires), subsequent calls = the scripted
 * compaction summaries, final call = final_response to end the run.
 */
class LoopAgent implements Agent {
  readonly seen: Context[] = [];
  private firedTool = false;
  constructor(private readonly summaries: string[]) {}
  private isCompactionContext(context: Context): boolean {
    return context.messages.some(
      (m) =>
        m.content.type === "text" &&
        (m.content.text.includes("please summarize") ||
          m.content.text.includes("missing these items")),
    );
  }
  async turn(context: Context): Promise<TurnResult> {
    if (this.isCompactionContext(context)) {
      // A compaction turn: record the context, replay a scripted summary.
      this.seen.push(context);
      return { kind: "final_response", content: this.summaries.shift() ?? "", usage: usage() };
    }
    if (!this.firedTool) {
      // First main-loop turn: request a tool so the loop appends a result and
      // then runs compaction.
      this.firedTool = true;
      return {
        kind: "tool_call_requested",
        calls: [{ id: "c1", name: "x", input: {} }],
        usage: usage(),
      };
    }
    // Subsequent main-loop turn: end the run.
    return { kind: "final_response", content: "done", usage: usage() };
  }
  id(): AgentId {
    return AgentId.of("loop");
  }
}

async function drive(
  cm: CompactingContextManager,
  agent: LoopAgent,
  verifier: CompactionVerifier,
  obs: ObservabilityProvider,
  maxAttempts: number,
): Promise<void> {
  const h = harnessWith(cm, agent, verifier, obs, maxAttempts);
  await h.run({ task: task(), session_state: emptySessionState() });
}

describe("Harness — compaction verify→retry→warn loop (issue #46)", () => {
  it("shouldCompact false → no compaction turn, no apply", async () => {
    const cm = new CompactingContextManager(false);
    const agent = new LoopAgent(["summary"]);
    const obs = new InMemoryObservabilityProvider();
    await drive(cm, agent, new ScriptedVerifier(0, []), obs, 2);
    expect(agent.seen.length).toBe(0);
    expect(cm.applied).toBe(0);
    expect(obs.warnSpans(SID)).toHaveLength(0);
  });

  it("passing verifier → 1 turn, 1 apply, no warn", async () => {
    const cm = new CompactingContextManager(true);
    const agent = new LoopAgent(["good summary"]);
    const obs = new InMemoryObservabilityProvider();
    await drive(cm, agent, new ScriptedVerifier(0, []), obs, 2);
    expect(agent.seen.length).toBe(1);
    expect(cm.applied).toBe(1);
    expect(obs.warnSpans(SID)).toHaveLength(0);
  });

  it("failing-then-passing, max=2 → 2 turns; retry context carries injected missing items", async () => {
    const cm = new CompactingContextManager(true);
    const agent = new LoopAgent(["v1", "v2"]);
    const obs = new InMemoryObservabilityProvider();
    await drive(cm, agent, new ScriptedVerifier(1, ["payment", "deploy"]), obs, 2);
    expect(agent.seen.length).toBe(2);
    const retry = agent.seen[1]!;
    const injected = retry.messages.some(
      (m) =>
        m.content.type === "text" &&
        m.content.text.includes("missing these items") &&
        m.content.text.includes("payment") &&
        m.content.text.includes("deploy"),
    );
    expect(injected).toBe(true);
    expect(cm.applied).toBe(1);
    expect(obs.warnSpans(SID)).toHaveLength(0);
  });

  it("always-failing, max=2 → warn with missingItems + acceptedAnyway, apply still runs, metric==1", async () => {
    const cm = new CompactingContextManager(true);
    const agent = new LoopAgent(["v1", "v2"]);
    const obs = new InMemoryObservabilityProvider();
    await drive(cm, agent, new ScriptedVerifier(99, ["payment"]), obs, 2);
    expect(agent.seen.length).toBe(2);
    expect(cm.applied).toBe(1);
    const warns = obs.warnSpans(SID);
    expect(warns).toHaveLength(1);
    const ev = warns[0]!.event;
    expect(ev.warn).toBe("compaction_verification_failed");
    expect(ev.missing_items).toEqual(["payment"]);
    expect(ev.accepted_anyway).toBe(true);
    const metrics = await obs.getSessionMetrics(SID);
    expect(metrics?.compaction_verification_failures).toBe(1);
  });

  it("maxCompactionAttempts=1 honored → 1 attempt then warn+accept", async () => {
    const cm = new CompactingContextManager(true);
    const agent = new LoopAgent(["v1", "v2", "v3"]);
    const obs = new InMemoryObservabilityProvider();
    await drive(cm, agent, new ScriptedVerifier(99, ["payment"]), obs, 1);
    expect(agent.seen.length).toBe(1);
    expect(cm.applied).toBe(1);
    expect(obs.warnSpans(SID)).toHaveLength(1);
  });

  it("emitWarn default no-op does not break a provider that doesn't override it (W4)", async () => {
    // A bare provider that implements the required surface but NOT emitWarn.
    class BareProvider implements ObservabilityProvider {
      emitTurn(): void {}
      emitToolCall(): void {}
      emitSensor(): void {}
      emitContext(): void {}
      emitMiddleware(): void {}
      emitPatch(): void {}
      setSessionOutcome(): void {}
      async flushSession(): Promise<void> {}
      async getSessionMetrics() {
        return undefined;
      }
      async getSessions() {
        return [];
      }
      async getTrace() {
        return [];
      }
    }
    const cm = new CompactingContextManager(true);
    const agent = new LoopAgent(["v1", "v2"]);
    const obs = new BareProvider();
    // Reaching the warn path must not throw; the bare provider ignores it.
    await drive(cm, agent, new ScriptedVerifier(99, ["payment"]), obs, 2);
    expect(cm.applied).toBe(1);
  });
});

// ============================================================================
// Cross-language consistency fixture replay (issue #46)
// ============================================================================

interface Verdict {
  passed: boolean;
  missing_items: string[];
}
interface Expected {
  attempts: number;
  apply_compaction_calls: number;
  warn_emitted: boolean;
  retry_injected_missing?: string[];
  final_missing_items: string[];
  accepted_anyway?: boolean;
}
interface Case {
  name: string;
  max_compaction_attempts: number;
  verdicts: Verdict[];
  expected: Expected;
}
interface Suite {
  cases: Case[];
}

/** Verifier driven by a fixture verdict queue; repeats the last verdict once the
 *  queue is exhausted, matching the fixture contract. */
class FixtureVerifier implements CompactionVerifier {
  private idx = 0;
  constructor(private readonly verdicts: Verdict[]) {}
  verify(): CompactionVerificationResult {
    const v = this.verdicts[this.idx] ?? this.verdicts[this.verdicts.length - 1]!;
    this.idx += 1;
    return { passed: v.passed, missingItems: v.missing_items.slice(), detail: "" };
  }
}

describe("Harness — compaction loop fixture replay", () => {
  const fixturePath = resolve(here, "../../../../fixtures/compaction_loop/cases.json");
  const suite = JSON.parse(readFileSync(fixturePath, "utf8")) as Suite;

  it("fixture has >= 5 cases", () => {
    expect(suite.cases.length).toBeGreaterThanOrEqual(5);
  });

  for (const c of suite.cases) {
    it(`case: ${c.name}`, async () => {
      const cm = new CompactingContextManager(true);
      const agent = new LoopAgent(["s1", "s2", "s3", "s4"]);
      const obs = new InMemoryObservabilityProvider();
      const verifier = new FixtureVerifier(c.verdicts);
      await drive(cm, agent, verifier, obs, c.max_compaction_attempts);

      expect(agent.seen.length).toBe(c.expected.attempts);
      expect(cm.applied).toBe(c.expected.apply_compaction_calls);

      const warns = obs.warnSpans(SID);
      expect(warns.length > 0).toBe(c.expected.warn_emitted);
      if (c.expected.warn_emitted) {
        expect(warns).toHaveLength(1);
        const ev = warns[0]!.event;
        expect(ev.warn).toBe("compaction_verification_failed");
        expect(ev.missing_items).toEqual(c.expected.final_missing_items);
        expect(ev.accepted_anyway).toBe(true);
      }
      if (c.expected.retry_injected_missing) {
        const joined = c.expected.retry_injected_missing.join(", ");
        const found = agent.seen
          .slice(1)
          .some((ctx) =>
            ctx.messages.some(
              (m) =>
                m.content.type === "text" &&
                m.content.text.includes("missing these items") &&
                m.content.text.includes(joined),
            ),
          );
        expect(found).toBe(true);
      }
    });
  }
});
