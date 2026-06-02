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
 *        → `FinalResponse` (concatenated text blocks)
 *   6. Model error → `Error(ModelError, usage: null)`
 *
 * ## Delta-level streaming (issue #103)
 *
 * {@link ModelAgent.turnStreaming} drives `ModelInterface.callStreaming`,
 * forwards each RAW model {@link StreamEvent} to the sink (Q1), reassembles a
 * complete {@link ModelResponse}, then runs the EXACT SAME
 * {@link classifyResponse} logic as {@link ModelAgent.turn} so the two paths
 * can never diverge. Thinking is accumulated into `reasoning` (Q4) rather than
 * discarded.
 *
 * ### Tool name + id in streamed turns
 *
 * The model-layer `StreamEvent` `tool_use_start` event carries the tool `name`
 * and call `id` — both arrive on the provider's block-start frame (Anthropic
 * `content_block_start`, Ollama / OpenAI's first `tool_calls` chunk) — followed
 * by `tool_use_delta` fragments for the argument JSON. The streaming
 * accumulator records the name/id from `tool_use_start`, so a tool call
 * reconstructed from a stream is faithful. A stable per-index id (`call_{index}`)
 * and empty name are synthesized only as a fallback if a stream somehow omitted
 * the start frame. The shared `fixtures/harness/streaming_events.json` golden
 * encodes this behaviour.
 */

import { ModelError } from "../model/errors.js";
import type { ModelInterface } from "../model/interface.js";
import type {
  ContentBlock,
  ModelResponse,
  StopReason,
  StreamEvent,
  ToolCall,
} from "../model/schemas.js";

import { AgentModelError, EmptyResponse, MalformedToolCall } from "./errors.js";
import type { Agent, AgentStreamSink } from "./interface.js";
import { contextToRequest, type AgentId, type Context, type TurnResult } from "./types.js";

/**
 * Classify an accumulated {@link ModelResponse} into a {@link TurnResult}.
 *
 * Single source of truth shared by {@link ModelAgent.turn} and
 * {@link ModelAgent.turnStreaming} (issue #103) so classification can never
 * diverge between the blocking and streaming paths. `thinking` blocks are
 * accumulated into the `reasoning` field (Q4) instead of being discarded.
 */
export function classifyResponse(response: ModelResponse): TurnResult {
  const usage = response.usage;

  const toolCalls: ToolCall[] = [];
  const textParts: string[] = [];
  const reasoningParts: string[] = [];
  for (const block of response.content) {
    switch (block.type) {
      case "tool_use":
        toolCalls.push({ id: block.id, name: block.name, input: block.input });
        break;
      case "text":
        textParts.push(block.text);
        break;
      case "thinking":
        // Q4: accumulate thinking text instead of discarding it.
        reasoningParts.push(block.text);
        break;
      default: {
        const _exhaustive: never = block;
        return _exhaustive;
      }
    }
  }
  const reasoning = reasoningParts.length > 0 ? reasoningParts.join("") : undefined;

  switch (response.stop_reason) {
    case "tool_use":
      if (toolCalls.length === 0) {
        return {
          kind: "error",
          error: new MalformedToolCall("", "stop_reason=tool_use but no tool_use blocks present"),
          usage,
        };
      }
      return { kind: "tool_call_requested", calls: toolCalls, usage, ...maybeReasoning(reasoning) };

    case "end_turn":
    case "max_tokens":
    case "stop_sequence":
      if (textParts.length === 0 && toolCalls.length === 0) {
        return { kind: "error", error: new EmptyResponse(), usage };
      }
      if (toolCalls.length > 0) {
        return {
          kind: "tool_call_requested",
          calls: toolCalls,
          usage,
          ...maybeReasoning(reasoning),
        };
      }
      return {
        kind: "final_response",
        content: textParts.join(""),
        usage,
        ...maybeReasoning(reasoning),
      };

    default: {
      const _exhaustive: never = response.stop_reason;
      return _exhaustive;
    }
  }
}

/**
 * Spread helper that adds `reasoning` only when present, so the field is
 * omitted from the wire (matching Rust's `skip_serializing_if`).
 */
function maybeReasoning(reasoning: string | undefined): { reasoning?: string } {
  return reasoning === undefined ? {} : { reasoning };
}

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
        return { kind: "error", error: new AgentModelError(err), usage: null };
      }
      // Non-typed errors are programming bugs, not domain errors — re-throw.
      throw err;
    }
    return classifyResponse(response);
  }

  /**
   * Streaming turn (issue #103). Builds a streaming request, drains the model
   * stream forwarding each raw {@link StreamEvent} to `sink`, reassembles a
   * complete {@link ModelResponse}, then runs the same {@link classifyResponse}
   * logic as {@link ModelAgent.turn}.
   */
  async turnStreaming(
    context: Context,
    sink: AgentStreamSink,
    signal?: AbortSignal,
  ): Promise<TurnResult> {
    const request = contextToRequest(context, true);
    const acc = new StreamAccumulator();
    try {
      for await (const event of this.model.callStreaming(request, signal)) {
        // Forward the RAW model event to the sink first (Q1), then fold it
        // into the in-progress response.
        sink(event);
        acc.fold(event);
      }
    } catch (err) {
      if (err instanceof ModelError) {
        return { kind: "error", error: new AgentModelError(err), usage: null };
      }
      throw err;
    }
    return classifyResponse(acc.intoResponse());
  }
}

type PartialBlock =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "tool_json"; id: string; name: string; json: string };

/**
 * Reassembles streamed model {@link StreamEvent}s into a {@link ModelResponse}
 * (issue #103). Tracks partial blocks keyed by their stream `index` in
 * first-seen order so the reconstructed `content` preserves emission order.
 */
class StreamAccumulator {
  private readonly blocks: Array<{ index: number; block: PartialBlock }> = [];
  private usage = { input_tokens: 0, output_tokens: 0 } as ModelResponse["usage"];
  private stopReason: StopReason | undefined = undefined;

  private entry(index: number, make: () => PartialBlock): PartialBlock {
    const found = this.blocks.find((b) => b.index === index);
    if (found) return found.block;
    const block = make();
    this.blocks.push({ index, block });
    return block;
  }

  fold(event: StreamEvent): void {
    switch (event.type) {
      case "message_start":
        break;
      case "content_block_delta": {
        const b = this.entry(event.index, () => ({ kind: "text", text: "" }));
        if (b.kind === "text") b.text += event.delta;
        break;
      }
      case "thinking_delta": {
        const b = this.entry(event.index, () => ({ kind: "thinking", text: "" }));
        if (b.kind === "thinking") b.text += event.delta;
        break;
      }
      case "tool_use_start": {
        const b = this.entry(event.index, () => ({
          kind: "tool_json",
          id: "",
          name: "",
          json: "",
        }));
        if (b.kind === "tool_json") {
          b.id = event.id;
          b.name = event.name;
        }
        break;
      }
      case "tool_use_delta": {
        const b = this.entry(event.index, () => ({
          kind: "tool_json",
          id: "",
          name: "",
          json: "",
        }));
        if (b.kind === "tool_json") b.json += event.partial_json;
        break;
      }
      case "content_block_stop":
        break;
      case "message_stop":
        this.usage = event.usage;
        this.stopReason = event.stop_reason;
        break;
      default: {
        const _exhaustive: never = event;
        return _exhaustive;
      }
    }
  }

  intoResponse(): ModelResponse {
    const content: ContentBlock[] = this.blocks.map(({ index, block }) => {
      switch (block.kind) {
        case "text":
          return { type: "text", text: block.text };
        case "thinking":
          return { type: "thinking", text: block.text };
        case "tool_json": {
          let input: unknown;
          try {
            input = JSON.parse(block.json);
          } catch {
            input = null;
          }
          // `id` / `name` come from the `tool_use_start` event every provider
          // emits at block start. Fall back to a stable per-index id (matching
          // the harness correlation key) and empty name only if a stream
          // somehow omitted the start frame, so reconstruction is well-formed.
          return {
            type: "tool_use",
            id: block.id === "" ? `call_${index}` : block.id,
            name: block.name,
            input,
          };
        }
        default: {
          const _exhaustive: never = block;
          return _exhaustive;
        }
      }
    });
    return {
      content,
      usage: this.usage,
      // Default to end_turn if the stream ended without message_stop.
      stop_reason: this.stopReason ?? "end_turn",
    };
  }
}
