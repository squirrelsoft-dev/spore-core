/**
 * `ModelAgent` — standard `Agent` implementation that forwards a `Context`
 * to a `ModelInterface` and classifies the response.
 *
 * Classification rules (must match Rust byte-for-byte across all languages):
 *
 *   1. `stop_reason === "tool_use"` AND tool-use blocks present
 *        → `ToolCallRequested`
 *   2. `stop_reason === "tool_use"` AND no tool-use blocks
 *        → `Error(MalformedToolCall)`
 *   3. `stop_reason ∈ {"end_turn","max_tokens","stop_sequence"}` AND
 *      no text AND no tool calls → `Error(EmptyResponse)`
 *   4. `stop_reason ∈ {"end_turn","max_tokens","stop_sequence"}` AND
 *      tool-use blocks present → still `ToolCallRequested` (we never
 *      silently drop a tool call)
 *   5. `stop_reason ∈ {"end_turn","max_tokens","stop_sequence"}` AND text
 *        → `FinalResponse` (concatenated text blocks; thinking blocks
 *           discarded — observability only, not output)
 *   6. Model error → `Error(ModelError, usage: null)`
 */

import { ModelError } from "../model/errors.js";
import type { ModelInterface } from "../model/interface.js";
import type { ToolCall } from "../model/schemas.js";

import { AgentModelError, EmptyResponse, MalformedToolCall } from "./errors.js";
import type { Agent } from "./interface.js";
import { contextToRequest, type AgentId, type Context, type TurnResult } from "./types.js";

export class ModelAgent implements Agent {
  constructor(
    private readonly agentId: AgentId,
    private readonly model: ModelInterface,
  ) {}

  id(): AgentId {
    return this.agentId;
  }

  async turn(context: Context, signal?: AbortSignal): Promise<TurnResult> {
    const request = contextToRequest(context);
    let response;
    try {
      response = await this.model.call(request, signal);
    } catch (err) {
      if (err instanceof ModelError) {
        return {
          kind: "error",
          error: new AgentModelError(err),
          usage: null,
        };
      }
      // Non-typed errors are programming bugs, not domain errors — re-throw.
      throw err;
    }

    const usage = response.usage;

    // Extract tool-use blocks and text regardless of stop_reason. The
    // stop_reason determines the classification, but we want to surface
    // any tool calls present so they aren't silently dropped.
    const toolCalls: ToolCall[] = [];
    const textParts: string[] = [];
    for (const block of response.content) {
      switch (block.type) {
        case "tool_use":
          toolCalls.push({
            id: block.id,
            name: block.name,
            input: block.input,
          });
          break;
        case "text":
          textParts.push(block.text);
          break;
        case "thinking":
          // Observability only — discarded.
          break;
        default: {
          const _exhaustive: never = block;
          return _exhaustive;
        }
      }
    }

    switch (response.stop_reason) {
      case "tool_use":
        if (toolCalls.length === 0) {
          return {
            kind: "error",
            error: new MalformedToolCall("", "stop_reason=tool_use but no tool_use blocks present"),
            usage,
          };
        }
        return { kind: "tool_call_requested", calls: toolCalls, usage };

      case "end_turn":
      case "max_tokens":
      case "stop_sequence":
        if (textParts.length === 0 && toolCalls.length === 0) {
          return { kind: "error", error: new EmptyResponse(), usage };
        }
        if (toolCalls.length > 0) {
          return { kind: "tool_call_requested", calls: toolCalls, usage };
        }
        return {
          kind: "final_response",
          content: textParts.join(""),
          usage,
        };

      default: {
        const _exhaustive: never = response.stop_reason;
        return _exhaustive;
      }
    }
  }
}
