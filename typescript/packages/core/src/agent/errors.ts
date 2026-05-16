/**
 * Typed errors for {@link Agent}.
 *
 * Mirrors the Rust `AgentError` enum byte-for-byte:
 *   - `model_error` wraps an underlying `ModelError`
 *   - `empty_response` — the model returned neither text nor tool calls
 *   - `malformed_tool_call` — `stop_reason=tool_use` without tool-use blocks
 *
 * Each variant is a class extending `Error` with a discriminant `kind` so
 * call-sites can exhaustively `switch` on `err.kind`.
 */

import { ModelError } from "../model/errors.js";

export type AgentErrorKind = "model_error" | "empty_response" | "malformed_tool_call";

export abstract class AgentError extends Error {
  abstract readonly kind: AgentErrorKind;

  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }

  /** JSON wire shape — matches Rust `#[serde(tag = "kind")]`. */
  abstract toJSON(): Record<string, unknown>;
}

export class AgentModelError extends AgentError {
  readonly kind = "model_error" as const;
  constructor(readonly error: ModelError) {
    super(error.message);
  }
  toJSON() {
    return { kind: this.kind, error: this.error.toJSON() };
  }
}

export class EmptyResponse extends AgentError {
  readonly kind = "empty_response" as const;
  constructor() {
    super("model returned neither text nor tool calls");
  }
  toJSON() {
    return { kind: this.kind };
  }
}

export class MalformedToolCall extends AgentError {
  readonly kind = "malformed_tool_call" as const;
  constructor(
    readonly toolName: string,
    readonly reason: string,
  ) {
    super(`malformed tool call from model (tool=${toolName}): ${reason}`);
  }
  toJSON() {
    return { kind: this.kind, tool_name: this.toolName, reason: this.reason };
  }
}
