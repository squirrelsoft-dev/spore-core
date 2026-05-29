/**
 * Compaction adapter — bridges the rich {@link StandardContextManager}
 * (issue #29) onto the harness-loop compaction seam (issue #46).
 *
 * Implements issue #55. Issue #46 wired the verify→retry→warn *machinery*
 * into the harness loop and proved it with test-double context managers. The
 * rich `StandardContextManager` operates on the *rich* `context` `SessionState`
 * / `CompactionResult` API and was never reachable from the loop seam. This
 * module is the production bridge so a {@link HarnessBuilder} built with a
 * rich manager *actually compacts*.
 *
 * ## Seam methods satisfied
 *
 * - {@link StandardCompactionAdapter.assemble} — minimal pass-through (NOT
 *   load-bearing for compaction; builds a harness `Context` from the session
 *   messages, mirroring the loop's test-double managers).
 * - {@link StandardCompactionAdapter.appendToolResult} /
 *   {@link StandardCompactionAdapter.appendUserMessage} — minimal: append to
 *   `harness` `SessionState.messages`.
 * - {@link StandardCompactionAdapter.shouldCompact} — reconstruct rich state
 *   from `session.extras`, delegate to rich `shouldCompact`.
 * - {@link StandardCompactionAdapter.prepareCompactionTurn} — reconstruct rich
 *   state → rich `prepareCompaction`; `undefined` when there is nothing to
 *   compact, else project hints + verification state + count.
 * - `injectMissingItems` — intentionally NOT overridden: the harness applies
 *   the exact "Your summary is missing these items: …" prompt the fixture
 *   asserts.
 * - {@link StandardCompactionAdapter.applyCompaction} — reconstruct rich state,
 *   delegate to rich `applyCompaction`, log+swallow any error (the loop must
 *   never halt on compaction), write the mutated rich state back into the
 *   session.
 *
 * ## Rules enforced
 *
 * 1. STATELESS bridge — the adapter holds no session state. Rich `context`
 *    `SessionState` is serialized into `harness` `SessionState.extras` under
 *    {@link RICH_STATE_KEY} on every mutating seam call and re-read on every
 *    read. No instance field/closure carries session state.
 * 2. Compaction never halts the loop — `applyCompaction` swallows the rich
 *    error (logged), and a malformed/absent rich-state blob degrades to a
 *    safe default (no compaction) rather than throwing.
 * 3. The summary is wrapped as an assistant {@link Message} for the rich
 *    {@link CompactionResult} so the rich manager prepends it as the summary
 *    turn.
 */

import type { Context } from "../agent/types.js";
import type { Message } from "../model/schemas.js";
import type {
  ContextManager as HarnessContextManager,
  CompactionTurn,
  SessionState as HarnessState,
  Task,
  ToolResultRecord,
} from "../harness/types.js";

import { StandardContextManager } from "./standard.js";
import {
  type CompactionResult,
  type SessionState as RichSessionState,
  SessionStateSchema,
} from "./types.js";

/**
 * Reserved key under `harness` `SessionState.extras` holding the serialized
 * rich `context` `SessionState`. The adapter is the only writer/reader.
 */
export const RICH_STATE_KEY = "spore.compaction_adapter.rich_state";

/**
 * Rough token estimate for a single message: the character length of its
 * textual content divided by four (the same chars/4 proxy
 * {@link StandardContextManager} uses for cache-marker placement). Used by the
 * adapter to compute real `tokens_reclaimed` from the messages a compaction
 * drops, since the synchronous harness seam cannot call the async
 * `countTokens`. Any non-empty message counts as at least one token so a drop
 * is never accounted as zero reclamation.
 */
export function estimateMessageTokens(message: Message): number {
  const c = message.content;
  let chars: number;
  switch (c.type) {
    case "text":
      chars = c.text.length;
      break;
    case "tool_call":
      chars = c.name.length + JSON.stringify(c.input ?? null).length;
      break;
    case "tool_result":
      chars = c.content.length;
      break;
    case "image":
      chars = c.data.length;
      break;
  }
  const tokens = Math.floor(chars / 4);
  return chars > 0 ? Math.max(1, tokens) : 0;
}

/** Sum {@link estimateMessageTokens} over a list of messages. */
export function estimateTokens(messages: Message[]): number {
  return messages.reduce((acc, m) => acc + estimateMessageTokens(m), 0);
}

/**
 * Reconstruct the rich session state from `extras`. Returns `undefined` when no
 * rich state has been seeded yet or the blob is malformed — callers treat that
 * as "nothing to compact" so the loop is never blocked.
 */
function readRichState(session: HarnessState): RichSessionState | undefined {
  const value = session.extras[RICH_STATE_KEY];
  if (value === undefined) return undefined;
  const parsed = SessionStateSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

/**
 * Serialize the rich session state back into `extras` and project its
 * `message_history` onto the harness-side `messages`.
 */
function writeRichState(session: HarnessState, rich: RichSessionState): void {
  session.messages = rich.message_history.slice();
  // Round-trip through JSON so the stored blob is plain wire data (ids become
  // strings via their `toJSON`), matching what `readRichState` parses back.
  session.extras[RICH_STATE_KEY] = JSON.parse(JSON.stringify(rich));
}

/**
 * Seed `extras` with a serialized rich session state. Callers that drive the
 * harness with {@link StandardCompactionAdapter} use this to project the rich
 * state into the harness session before the first turn.
 */
export function seedRichState(session: HarnessState, rich: RichSessionState): void {
  writeRichState(session, rich);
}

/**
 * Stateless bridge from the rich {@link StandardContextManager} onto the
 * harness-loop compaction seam ({@link HarnessContextManager}).
 *
 * Construct via {@link StandardCompactionAdapter} or
 * {@link intoHarnessAdapter}, then inject the result into a
 * {@link HarnessBuilder}.
 */
export class StandardCompactionAdapter implements HarnessContextManager {
  constructor(private readonly inner: StandardContextManager) {}

  async assemble(session: HarnessState, _task: Task, _signal?: AbortSignal): Promise<Context> {
    // NOT load-bearing for compaction. The rich `assemble` requires
    // `ContextSources` the seam does not supply, so we produce a minimal
    // context straight from the session messages (mirrors the loop's
    // test-double managers).
    return {
      messages: session.messages.slice(),
      tools: [],
      params: { stop_sequences: [] },
    };
  }

  async appendToolResult(session: HarnessState, result: ToolResultRecord): Promise<void> {
    const text = renderToolOutput(result);
    session.messages.push({ role: "tool", content: { type: "text", text } });
  }

  async appendAssistantMessage(session: HarnessState, message: Message): Promise<void> {
    // Record the assistant's turn (text and/or requested tool calls) so the
    // next assemble() reflects what the agent already did. Mirrors
    // appendToolResult: push onto the harness-side message list.
    session.messages.push(message);
  }

  async appendUserMessage(session: HarnessState, text: string): Promise<void> {
    session.messages.push({ role: "user", content: { type: "text", text } });
  }

  shouldCompact(session: HarnessState): boolean {
    const rich = readRichState(session);
    return rich !== undefined && this.inner.shouldCompact(rich);
  }

  prepareCompactionTurn(session: HarnessState): CompactionTurn | undefined {
    const rich = readRichState(session);
    if (rich === undefined) return undefined;

    const request = this.inner.prepareCompaction(rich);
    if (request.messages_to_compact.length === 0) return undefined;

    const messagesRemoved = request.messages_to_compact.length;

    // Build the summarization context: the messages to compact, followed by the
    // summarization instruction. The harness's default `injectMissingItems`
    // appends the retry instruction on verification failure.
    const messages: Message[] = [
      ...request.messages_to_compact,
      {
        role: "user",
        content: {
          type: "text",
          text: "Summarize the conversation above, preserving the items in the preservation hints.",
        },
      },
    ];

    return {
      context: { messages, tools: [], params: { stop_sequences: [] } },
      preserveHints: request.preserve_hints,
      verificationState: rich,
      messagesRemoved,
    };
  }

  // `injectMissingItems` is intentionally NOT implemented: the harness's
  // default produces the exact "Your summary is missing these items: …" prompt
  // the `compaction_loop` fixture asserts.

  applyCompaction(session: HarnessState, summary: string): void {
    const rich = readRichState(session);
    if (rich === undefined) {
      // No rich state to apply against — degrade safely; never throw.
      return;
    }

    const dropped = this.inner.prepareCompaction(rich).messages_to_compact;
    const messagesRemoved = dropped.length;
    const summaryMessage: Message = { role: "assistant", content: { type: "text", text: summary } };

    // Real token accounting (Known Deviation #2 fix): reclaim the tokens of the
    // messages we drop, net of the summary that replaces them, and clamp to the
    // live budget so `token_budget_used` never underflows. The rich
    // `applyCompaction` (standard.ts) decrements `token_budget_used` by this
    // amount, so utilization actually falls below threshold after a compaction
    // and a long session can compact repeatedly.
    const droppedTokens = estimateTokens(dropped);
    const summaryTokens = estimateMessageTokens(summaryMessage);
    const netReclaimed = Math.max(0, droppedTokens - summaryTokens);
    const tokensReclaimed = Math.min(netReclaimed, rich.token_budget_used);

    const result: CompactionResult = {
      summary_message: summaryMessage,
      tokens_reclaimed: tokensReclaimed,
      messages_removed: messagesRemoved,
    };

    try {
      this.inner.applyCompaction(rich, result);
    } catch (err) {
      // Compaction must never halt the loop — log and swallow.
      console.error(
        `spore.compaction: rich applyCompaction failed, leaving session unchanged: ${String(err)}`,
      );
      return;
    }
    writeRichState(session, rich);
  }

  tokenBudgetUsed(session: HarnessState): number | undefined {
    const rich = readRichState(session);
    return rich?.token_budget_used;
  }
}

/**
 * Ergonomic constructor: wrap a rich {@link StandardContextManager} as the
 * harness-seam adapter for injection into a {@link HarnessBuilder}.
 */
export function intoHarnessAdapter(inner: StandardContextManager): HarnessContextManager {
  return new StandardCompactionAdapter(inner);
}

function renderToolOutput(result: ToolResultRecord): string {
  switch (result.output.kind) {
    case "success":
      return result.output.content;
    case "error":
      return result.output.message;
    case "waiting_for_human":
      return "";
    case "escalate":
      return "";
  }
}
