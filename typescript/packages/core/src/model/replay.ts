/**
 * `ReplayModelInterface` — positional replay of recorded
 * `(request, response)` pairs.
 *
 * The n-th `call` returns the n-th recorded response. Matching is purely
 * positional (rather than request-hash matching) — this is the shared
 * contract across all four language implementations so fixtures stay
 * portable; deeper equality is a follow-up.
 */

import { ProviderError, type ModelError } from "./errors.js";
import type { ModelInterface } from "./interface.js";
import {
  ModelRequestSchema,
  RecordedExchangeSchema,
  type Content,
  type ContentBlock,
  type ModelRequest,
  type ModelResponse,
  type ProviderInfo,
  type RecordedExchange,
  type StreamEvent,
} from "./schemas.js";

export class ReplayModelInterface implements ModelInterface {
  private cursor = 0;

  constructor(
    private readonly exchanges: readonly RecordedExchange[],
    private readonly providerInfo: ProviderInfo,
  ) {}

  /**
   * Parse a JSONL string of {@link RecordedExchange} records and build a
   * replay model. Throws (synchronously) on malformed JSON or schema
   * violation — fixtures are part of the test inputs and a broken fixture
   * is a developer bug, not a runtime error.
   */
  static fromJsonl(jsonl: string, provider: ProviderInfo): ReplayModelInterface {
    const exchanges: RecordedExchange[] = [];
    const lines = jsonl.split(/\r?\n/);
    for (const line of lines) {
      if (line.trim() === "") continue;
      const parsed = JSON.parse(line);
      exchanges.push(RecordedExchangeSchema.parse(parsed));
    }
    return new ReplayModelInterface(exchanges, provider);
  }

  remaining(): number {
    return Math.max(0, this.exchanges.length - this.cursor);
  }

  provider(): ProviderInfo {
    return this.providerInfo;
  }

  async call(_request: ModelRequest, _signal?: AbortSignal): Promise<ModelResponse> {
    const next = this.exchanges[this.cursor];
    if (next == null) {
      // Replay exhaustion: surface as a typed ModelError; never throw raw.
      throw new ProviderError(0, "replay exhausted: no more recorded exchanges");
    }
    this.cursor += 1;
    return next.response;
  }

  async *callStreaming(
    request: ModelRequest,
    signal?: AbortSignal,
  ): AsyncIterable<StreamEvent> {
    const response = await this.call(request, signal);
    yield { type: "message_start" };
    for (let i = 0; i < response.content.length; i++) {
      const block = response.content[i]!;
      yield streamEventForBlock(block, i);
      yield { type: "content_block_stop", index: i };
    }
    yield {
      type: "message_stop",
      usage: response.usage,
      stop_reason: response.stop_reason,
    };
  }

  /**
   * Cheap deterministic estimate sufficient for fixture replay: ~4 chars
   * per token over the textual content. Real providers override this.
   */
  async countTokens(request: ModelRequest, _signal?: AbortSignal): Promise<number> {
    // Validate input shape so callers that hand-craft requests still get
    // the same defensive treatment they'd get from a real provider.
    const req = ModelRequestSchema.parse(request);
    let chars = 0;
    for (const m of req.messages) {
      chars += charLengthOfContent(m.content);
    }
    return Math.floor(chars / 4);
  }
}

function streamEventForBlock(block: ContentBlock, index: number): StreamEvent {
  switch (block.type) {
    case "text":
      return { type: "content_block_delta", index, delta: block.text };
    case "thinking":
      return { type: "thinking_delta", index, delta: block.text };
    case "tool_use": {
      let partial = "{}";
      try {
        partial = JSON.stringify(block.input ?? {});
      } catch {
        partial = "{}";
      }
      return { type: "tool_use_delta", index, partial_json: partial };
    }
    default: {
      const _exhaustive: never = block;
      return _exhaustive;
    }
  }
}

function charLengthOfContent(content: Content): number {
  switch (content.type) {
    case "text":
      return content.text.length;
    case "tool_call": {
      let inputLen = 0;
      try {
        inputLen = JSON.stringify(content.input ?? {}).length;
      } catch {
        inputLen = 0;
      }
      return content.name.length + inputLen;
    }
    case "tool_result":
      return content.content.length;
    case "image":
      return 0;
    default: {
      const _exhaustive: never = content;
      return _exhaustive;
    }
  }
}

// Re-export for convenience.
export type { ModelError };
