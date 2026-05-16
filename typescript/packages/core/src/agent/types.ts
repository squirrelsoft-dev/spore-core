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

export function contextToRequest(context: Context): ModelRequest {
  return {
    messages: context.messages,
    tools: context.tools,
    params: context.params,
    stream: false,
  };
}

export type TurnResult =
  | { kind: "tool_call_requested"; calls: ToolCall[]; usage: TokenUsage }
  | { kind: "final_response"; content: string; usage: TokenUsage }
  | { kind: "error"; error: AgentError; usage: TokenUsage | null };
