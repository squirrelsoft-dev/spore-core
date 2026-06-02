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
 *
 * ## Delta-level streaming (issue #103)
 *
 * {@link Agent.turnStreaming} forwards each RAW {@link StreamEvent} (the
 * *model*-layer event) to an {@link AgentStreamSink} as it arrives, then
 * returns the same {@link TurnResult} as `turn`. Per resolved spec decision
 * **Q1**, the agent boundary deals only in model-layer `StreamEvent`s; it does
 * NOT depend on the harness `StreamEvent` type. The harness owns the
 * `model StreamEvent → harness StreamEvent` mapping.
 */

import type { StreamEvent } from "../model/schemas.js";

import type { Context, TurnResult, AgentId } from "./types.js";

/**
 * Callback that receives RAW model-layer {@link StreamEvent}s as the agent
 * drains a streaming model call (issue #103, Q1).
 */
export type AgentStreamSink = (event: StreamEvent) => void;

export interface Agent {
  /** Execute exactly one model call and classify the result. */
  turn(context: Context, signal?: AbortSignal): Promise<TurnResult>;

  /**
   * Execute one turn while forwarding each raw model {@link StreamEvent} to
   * `sink` as it arrives (issue #103).
   *
   * Implementations are OPTIONAL: when absent the harness falls back to the
   * default in {@link defaultTurnStreaming}, which ignores the sink and
   * delegates to {@link Agent.turn}. Model-backed agents override this to call
   * `ModelInterface.callStreaming` and emit deltas.
   */
  turnStreaming?(
    context: Context,
    sink: AgentStreamSink,
    signal?: AbortSignal,
  ): Promise<TurnResult>;

  /** Caller-assigned identity for tracing. */
  id(): AgentId;
}

/**
 * Default `turnStreaming` behaviour for agents that don't implement it: ignore
 * the sink and delegate to {@link Agent.turn}, so every existing `Agent` impl
 * keeps working unchanged (issue #103). Callers should route through this
 * rather than calling `agent.turnStreaming?.()` directly.
 */
export function turnStreaming(
  agent: Agent,
  context: Context,
  sink: AgentStreamSink,
  signal?: AbortSignal,
): Promise<TurnResult> {
  if (agent.turnStreaming) {
    return agent.turnStreaming(context, sink, signal);
  }
  return agent.turn(context, signal);
}
