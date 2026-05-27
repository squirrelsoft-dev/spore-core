/**
 * StandardCompactionAdapter — harness compaction-seam bridge (spore-core
 * issue #55).
 *
 * Mirrors `rust/crates/spore-core/src/compaction_adapter.rs#tests` — same
 * rules, parallel structure. The adapter bridges the rich
 * {@link StandardContextManager} onto the harness loop's compaction seam so a
 * {@link HarnessBuilder} built with a rich manager actually compacts.
 *
 * Coverage:
 *   - shouldCompact threshold behavior (rich-state driven; safe default when
 *     no rich state is seeded).
 *   - prepareCompactionTurn: undefined for short history; projects hints +
 *     verification state + count, with the summarization prompt appended.
 *   - applyCompaction shrinks the session and round-trips the rich state;
 *     swallows the missing-rich-state case without throwing.
 *   - end-to-end: a real CompactionTurn fires through the seam (mock model)
 *     and shrinks the session.
 *   - fixture parity against `fixtures/compaction_loop/cases.json` for the
 *     verify→retry→warn control flow with the real adapter.
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
  type Context,
  type ModelInterface,
  type ModelRequest,
  type ModelResponse,
  type ProviderInfo,
  type SessionState as HarnessSessionState,
  type StreamEvent,
  type Task,
  type TurnResult,
  cacheProvider as cacheNs,
  context as contextNs,
  observability as obsNs,
} from "../src/index.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";

const { StandardContextManager, StandardCompactionAdapter, intoHarnessAdapter, seedRichState } =
  contextNs;
const { NullCacheProvider } = cacheNs;
const { InMemoryObservabilityProvider } = obsNs;
const newSessionState = contextNs.newSessionState;

type RichSessionState = contextNs.SessionState;
type CompactionVerificationResult = contextNs.CompactionVerificationResult;
type CompactionVerifier = contextNs.CompactionVerifier;

const here = dirname(fileURLToPath(import.meta.url));
const SID = SessionId.of("s1");
const TID = TaskId.of("t1");

// ---- minimal model stub -----------------------------------------------------

class StubModel implements ModelInterface {
  async call(_req: ModelRequest): Promise<ModelResponse> {
    return { content: [], usage: usage(), stop_reason: "end_turn" };
  }
  async *callStreaming(_req: ModelRequest): AsyncIterable<StreamEvent> {
    // No events; never used in these tests.
  }
  async countTokens(_req: ModelRequest): Promise<number> {
    return 0;
  }
  provider(): ProviderInfo {
    return { name: "stub", model_id: "stub", context_window: 200_000 };
  }
}

function usage() {
  return { input_tokens: 1, output_tokens: 1, cache_read_tokens: null, cache_write_tokens: null };
}

// ---- helpers ----------------------------------------------------------------

function richManager(): InstanceType<typeof StandardContextManager> {
  return new StandardContextManager(new StubModel(), new NullCacheProvider(), {
    threshold: 0.8,
    preserve_recent_n: 2,
    head_tail_tokens: 64,
    offload_path: ".spore/offload",
    max_compaction_attempts: 2,
  });
}

function richState(messages: number, used: number, limit: number): RichSessionState {
  const s = newSessionState(SID, TID, "deploy the payment service");
  s.window_limit = limit;
  s.token_budget_used = used;
  s.message_history = Array.from({ length: messages }, (_, i) => ({
    role: "user" as const,
    content: { type: "text" as const, text: `m${i}` },
  }));
  return s;
}

function sessionWith(rich: RichSessionState): HarnessSessionState {
  const s = emptySessionState();
  seedRichState(s, rich);
  return s;
}

// ---- shouldCompact threshold ------------------------------------------------

describe("StandardCompactionAdapter — shouldCompact (issue #55)", () => {
  it("below threshold is false", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    expect(adapter.shouldCompact(sessionWith(richState(10, 70, 100)))).toBe(false);
  });

  it("at threshold is true", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    expect(adapter.shouldCompact(sessionWith(richState(10, 80, 100)))).toBe(true);
  });

  it("above threshold is true", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    expect(adapter.shouldCompact(sessionWith(richState(10, 95, 100)))).toBe(true);
  });

  it("false without seeded rich state", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    expect(adapter.shouldCompact(emptySessionState())).toBe(false);
  });
});

// ---- prepareCompactionTurn --------------------------------------------------

describe("StandardCompactionAdapter — prepareCompactionTurn (issue #55)", () => {
  it("returns undefined for history shorter than the preserve window", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    // preserve_recent_n = 2, history = 2 -> nothing to compact.
    expect(adapter.prepareCompactionTurn(sessionWith(richState(2, 95, 100)))).toBeUndefined();
  });

  it("projects hints, verification state, count, and appends the summarization prompt", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    const rich = richState(10, 95, 100); // 10 - 2 preserved = 8 removed
    const turn = adapter.prepareCompactionTurn(sessionWith(rich));
    expect(turn).toBeDefined();
    expect(turn!.messagesRemoved).toBe(8);
    // verification_state mirrors the rich state.
    expect(turn!.verificationState.task_instruction).toBe(rich.task_instruction);
    expect(turn!.verificationState.token_budget_used).toBe(95);
    // default hints projected.
    expect(turn!.preserveHints.keep_current_task_state).toBe(true);
    // the summarization instruction is appended after the compacted msgs.
    const hasSummarize = turn!.context.messages.some(
      (m) => m.content.type === "text" && m.content.text.includes("Summarize"),
    );
    expect(hasSummarize).toBe(true);
  });
});

// ---- applyCompaction --------------------------------------------------------

describe("StandardCompactionAdapter — applyCompaction (issue #55)", () => {
  it("shrinks the session and round-trips the rich state", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    const session = sessionWith(richState(10, 95, 100));
    const before = session.messages.length;
    adapter.applyCompaction(session, "summary preserving payment deploy");
    // 2 preserved + 1 summary = 3.
    expect(session.messages.length).toBeLessThan(before);
    expect(session.messages.length).toBe(3);
    // round-tripped rich state also shrank.
    const rich = contextNs.SessionStateSchema.parse(session.extras[contextNs.RICH_STATE_KEY]);
    expect(rich.message_history.length).toBe(3);
    expect(rich.message_history[0]!.role).toBe("assistant");
  });

  it("swallows the missing-rich-state case without throwing", () => {
    const adapter = new StandardCompactionAdapter(richManager());
    const session = emptySessionState();
    expect(() => adapter.applyCompaction(session, "summary")).not.toThrow();
    expect(session.messages.length).toBe(0);
  });
});

// ---- end-to-end mock-model harness ------------------------------------------

class SummaryAgent implements Agent {
  constructor(private readonly summary: string) {}
  private isCompaction(context: Context): boolean {
    return context.messages.some(
      (m) =>
        m.content.type === "text" &&
        (m.content.text.includes("Summarize") || m.content.text.includes("missing these items")),
    );
  }
  private firedTool = false;
  async turn(context: Context): Promise<TurnResult> {
    if (this.isCompaction(context)) {
      return { kind: "final_response", content: this.summary, usage: usage() };
    }
    if (!this.firedTool) {
      this.firedTool = true;
      return {
        kind: "tool_call_requested",
        calls: [{ id: "c1", name: "x", input: {} }],
        usage: usage(),
      };
    }
    return { kind: "final_response", content: "done", usage: usage() };
  }
  id(): AgentId {
    return AgentId.of("summary");
  }
}

function task(): Task {
  return newTask("do something", SID, { kind: "re_act", max_iterations: 5 });
}

describe("StandardCompactionAdapter — end-to-end through the seam (issue #55)", () => {
  it("drives a real compaction turn that shrinks the session", async () => {
    const adapter = intoHarnessAdapter(richManager());
    // A summary containing "payment" so the default KeyTermVerifier (sources
    // task_instruction "deploy the payment service") passes first attempt.
    const agent = new SummaryAgent("we are working on the deploy of the payment service");
    const obs = new InMemoryObservabilityProvider();
    const h = new HarnessBuilder(
      agent,
      new ScriptedToolRegistry(),
      new AllowAllSandbox(),
      adapter,
      new AlwaysContinuePolicy(),
    )
      .observability(obs)
      .maxCompactionAttempts(2)
      .build();

    // Drive utilization over threshold: 10 messages, 95/100 budget.
    const session = sessionWith(richState(10, 95, 100));
    const before = session.messages.length;
    expect(adapter.shouldCompact(session)).toBe(true);

    await h.run({ task: task(), session_state: session });

    expect(session.messages.length).toBeLessThan(before);
    expect(session.messages.length).toBe(3);
    obs.setSessionOutcome(SID, { kind: "success" });
    const metrics = await obs.getSessionMetrics(SID);
    expect(metrics?.compactions).toBe(1);
    // verification passed first time -> no warn.
    expect(obs.warnSpans(SID)).toHaveLength(0);
  });
});

// ---- compaction_loop fixture parity (verify->retry->warn) -------------------

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

/** Verifier scripted from the fixture's verdicts (last entry repeats). */
class FixtureVerifier implements CompactionVerifier {
  private idx = 0;
  constructor(private readonly verdicts: Verdict[]) {}
  verify(): CompactionVerificationResult {
    const v = this.verdicts[this.idx] ?? this.verdicts[this.verdicts.length - 1]!;
    this.idx += 1;
    return { passed: v.passed, missingItems: v.missing_items.slice(), detail: "" };
  }
}

/** Agent that records the compaction contexts it sees so retry injection can be
 *  asserted. First main-loop turn requests a tool (so compaction fires), then
 *  ends the run; compaction turns return a scripted summary. */
class RecordingAgent implements Agent {
  readonly seen: Context[] = [];
  private firedTool = false;
  private isCompaction(context: Context): boolean {
    return context.messages.some(
      (m) =>
        m.content.type === "text" &&
        (m.content.text.includes("Summarize") || m.content.text.includes("missing these items")),
    );
  }
  async turn(context: Context): Promise<TurnResult> {
    if (this.isCompaction(context)) {
      this.seen.push(context);
      return { kind: "final_response", content: "summary", usage: usage() };
    }
    if (!this.firedTool) {
      this.firedTool = true;
      return {
        kind: "tool_call_requested",
        calls: [{ id: "c1", name: "x", input: {} }],
        usage: usage(),
      };
    }
    return { kind: "final_response", content: "done", usage: usage() };
  }
  id(): AgentId {
    return AgentId.of("rec");
  }
}

describe("StandardCompactionAdapter — compaction loop fixture parity (issue #55)", () => {
  const fixturePath = resolve(here, "../../../../fixtures/compaction_loop/cases.json");
  const suite = JSON.parse(readFileSync(fixturePath, "utf8")) as Suite;

  it("fixture has >= 5 cases", () => {
    expect(suite.cases.length).toBeGreaterThanOrEqual(5);
  });

  for (const c of suite.cases) {
    it(`case: ${c.name}`, async () => {
      const adapter = intoHarnessAdapter(richManager());
      const agent = new RecordingAgent();
      const obs = new InMemoryObservabilityProvider();
      const verifier = new FixtureVerifier(c.verdicts);
      const h = new HarnessBuilder(
        agent,
        new ScriptedToolRegistry(),
        new AllowAllSandbox(),
        adapter,
        new AlwaysContinuePolicy(),
      )
        .observability(obs)
        .compactionVerifier(verifier)
        .maxCompactionAttempts(c.max_compaction_attempts)
        .build();

      // 10 messages, over threshold -> a real CompactionTurn is offered.
      const session = sessionWith(richState(10, 95, 100));
      await h.run({ task: task(), session_state: session });

      // attempts parity.
      expect(agent.seen.length).toBe(c.expected.attempts);

      // applyCompaction always runs exactly once -> session shrank to 3.
      expect(c.expected.apply_compaction_calls).toBe(1);
      const rich = contextNs.SessionStateSchema.parse(session.extras[contextNs.RICH_STATE_KEY]);
      expect(rich.message_history.length).toBe(3);

      // warn parity.
      const warns = obs.warnSpans(SID);
      expect(warns.length > 0).toBe(c.expected.warn_emitted);
      if (c.expected.warn_emitted) {
        expect(warns).toHaveLength(1);
        const ev = warns[0]!.event;
        expect(ev.warn).toBe("compaction_verification_failed");
        expect(ev.missing_items).toEqual(c.expected.final_missing_items);
        expect(ev.accepted_anyway).toBe(true);
      }

      // retry injection parity.
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
