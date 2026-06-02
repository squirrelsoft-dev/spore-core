/**
 * Public types for the Agent component (spore-core issue #2).
 *
 * The wire shape mirrors the Rust reference implementation byte-for-byte:
 *
 *   TurnResult is a tagged union with a `kind` discriminator in snake_case:
 *     - "tool_call_requested" { calls, usage }
 *     - "final_response"      { content, usage }
 *     - "error"               { error, usage }
 *
 * `AgentId` is a caller-assigned configuration name used for tracing when
 * multiple agents share a session.
 */

import type { AgentError } from "./errors.js";
import type {
  Message,
  ModelParams,
  ModelRequest,
  TokenUsage,
  ToolCall,
  ToolSchema,
} from "../model/schemas.js";

/** Caller-assigned agent configuration name used for trace correlation. */
export class AgentId {
  constructor(readonly value: string) {}

  static of(value: string): AgentId {
    return new AgentId(value);
  }

  asString(): string {
    return this.value;
  }

  toString(): string {
    return this.value;
  }

  equals(other: AgentId): boolean {
    return this.value === other.value;
  }

  toJSON(): string {
    return this.value;
  }
}

/**
 * Fully assembled per-turn input produced by the ContextManager (issue #7).
 *
 * The agent never modifies this — it is treated as an immutable snapshot.
 * Once issue #7 lands the canonical type lives there; this module's
 * `Context` is the minimum surface the agent requires.
 */
export interface Context {
  messages: Message[];
  tools: ToolSchema[];
  params: ModelParams;
}

export function emptyContext(): Context {
  return {
    messages: [],
    tools: [],
    params: { stop_sequences: [] },
  };
}

export function contextToRequest(context: Context, stream = false): ModelRequest {
  return {
    messages: context.messages,
    tools: context.tools,
    params: context.params,
    stream,
  };
}

/**
 * Result of one agent turn.
 *
 * The `reasoning` field on the `tool_call_requested` / `final_response`
 * variants carries accumulated thinking text produced during the turn
 * (issue #103, Q4). It is optional so pre-#103 serialized `TurnResult`s still
 * deserialize, and is omitted from the wire when absent (matching Rust's
 * `skip_serializing_if = "Option::is_none"`). Thinking is NOT preserved in
 * `SessionState` nor added as a `Content` variant — deferred to issue #104.
 */
export type TurnResult =
  | {
      kind: "tool_call_requested";
      calls: ToolCall[];
      usage: TokenUsage;
      reasoning?: string;
    }
  | {
      kind: "final_response";
      content: string;
      usage: TokenUsage;
      reasoning?: string;
    }
  | { kind: "error"; error: AgentError; usage: TokenUsage | null };
