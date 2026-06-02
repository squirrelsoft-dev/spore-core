/**
 * `EmptyToolRegistry` — a harness-loop {@link ToolRegistry} with no tools.
 *
 * `schemas()` is empty, so the model is offered no tools and replies in plain
 * text; `dispatch` is therefore never reached in normal operation. This is the
 * registry used by {@link "../harness/standard.js".HarnessBuilder.conversational}
 * and the right starting point for any agent that does not act on its
 * environment.
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs` `EmptyToolRegistry`.
 */

import type { ToolCall, ToolSchema } from "../model/schemas.js";
import type { ToolOutput, ToolRegistry } from "../harness/types.js";

export class EmptyToolRegistry implements ToolRegistry {
  /** No tools — the model is offered nothing and replies in plain text. */
  schemas(): ToolSchema[] {
    return [];
  }

  /** No tool can be marked always-halt because no tool is registered. */
  isAlwaysHalt(_toolName: string): boolean {
    return false;
  }

  /** Never reached in normal operation; a non-recoverable error if it is. */
  async dispatch(_call: ToolCall, _signal?: AbortSignal): Promise<ToolOutput> {
    return { kind: "error", message: "no tools are registered", recoverable: false };
  }
}
