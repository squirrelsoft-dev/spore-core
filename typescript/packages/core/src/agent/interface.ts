/**
 * `Agent` — executes a single turn against a `ModelInterface`.
 *
 * Implements spore-core issue #2. One call to `turn` performs exactly one
 * model call and returns a classified {@link TurnResult}. The agent does NOT:
 *   - Assemble context (that is the ContextManager — issue #7)
 *   - Execute tool calls (the harness dispatches via ToolRegistry — issue #4)
 *   - Validate tool call parameters (ToolRegistry — issue #4)
 *   - Decide termination (TerminationPolicy — issue #13)
 *   - Retry on transient errors (lives in the ModelInterface impl)
 */

import type { Context, TurnResult, AgentId } from "./types.js";

export interface Agent {
  /** Execute exactly one model call and classify the result. */
  turn(context: Context, signal?: AbortSignal): Promise<TurnResult>;

  /** Caller-assigned identity for tracing. */
  id(): AgentId;
}
