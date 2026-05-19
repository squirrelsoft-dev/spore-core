/**
 * `ReplayModelInterface` — replay of recorded `(request, response)` pairs.
 *
 * Two modes (spore-core issue #37):
 *   - `positional`: pre-#37 behavior; the n-th `call` returns the n-th
 *     recorded response. Fragile against loop-order changes but compatible
 *     with old fixtures.
 *   - `hash_matched`: each `call` hashes its request via `requestHash` and
 *     looks up the matching recorded entry. Order-independent.
 *
 * `new` (and `fromJsonl`) auto-detect: `hash_matched` when every entry has a
 * `request_hash` AND the list is non-empty; otherwise `positional`. Pass an
 * explicit `mode` to `withMode` to override.
 */

import { ProviderError, type ModelError } from "./errors.js";
import { requestHash } from "./hash.js";
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

export type ReplayMode = "positional" | "hash_matched";

export class ReplayModelInterface implements ModelInterface {
  private cursor = 0;
  private readonly modeValue: ReplayMode;

  /**
   * Construct with the auto-detected mode. Use this in new code.
   *
   * Pass a `mode` to force a specific mode (e.g. positional replay against a
   * hash-tagged fixture to test old behaviour).
   */
  constructor(
    private readonly exchanges: readonly RecordedExchange[],
    private readonly providerInfo: ProviderInfo,
    mode?: ReplayMode,
  ) {
    this.modeValue = mode ?? autoDetectMode(exchanges);
  }

  /**
   * Parse a JSONL string of {@link RecordedExchange} records and build a
   * replay model. Throws (synchronously) on malformed JSON or schema
   * violation — fixtures are part of the test inputs and a broken fixture
   * is a developer bug, not a runtime error.
   */
  static fromJsonl(jsonl: string, provider: ProviderInfo, mode?: ReplayMode): ReplayModelInterface {
    const exchanges: RecordedExchange[] = [];
    const lines = jsonl.split(/\r?\n/);
    for (const line of lines) {
      if (line.trim() === "") continue;
      const parsed = JSON.parse(line);
      exchanges.push(RecordedExchangeSchema.parse(parsed));
    }
    return new ReplayModelInterface(exchanges, provider, mode);
  }

  mode(): ReplayMode {
    return this.modeValue;
  }

  remaining(): number {
    return Math.max(0, this.exchanges.length - this.cursor);
  }

  provider(): ProviderInfo {
    return this.providerInfo;
  }

  async call(request: ModelRequest, _signal?: AbortSignal): Promise<ModelResponse> {
    if (this.modeValue === "hash_matched") {
      const want = requestHash(request);
      const match = this.exchanges.find((e) => e.request_hash === want);
      if (match == null) {
        throw new ProviderError(0, `no matching fixture for request_hash=${want}`);
      }
      return match.response;
    }
    const next = this.exchanges[this.cursor];
    if (next == null) {
      // Replay exhaustion: surface as a typed ModelError; never throw raw.
      throw new ProviderError(0, "replay exhausted: no more recorded exchanges");
    }
    this.cursor += 1;
    return next.response;
  }

  async *callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
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
   * Token count.
   *
   * When the fixture was recorded by `RecordingModelInterface` against a real
   * provider, the recorded response's `usage.input_tokens` carries the
   * provider's exact count — use that whenever we can match by hash. Fall
   * back to the bytes/4 heuristic only when no matching entry exists
   * (positional fixtures or unrecorded requests).
   */
  async countTokens(request: ModelRequest, _signal?: AbortSignal): Promise<number> {
    // Validate input shape so callers that hand-craft requests still get
    // the same defensive treatment they'd get from a real provider.
    const req = ModelRequestSchema.parse(request);
    if (this.modeValue === "hash_matched") {
      const want = requestHash(req);
      const match = this.exchanges.find((e) => e.request_hash === want);
      if (match != null) {
        return match.response.usage.input_tokens;
      }
    }
    let chars = 0;
    for (const m of req.messages) {
      chars += charLengthOfContent(m.content);
    }
    return Math.floor(chars / 4);
  }
}

function autoDetectMode(exchanges: readonly RecordedExchange[]): ReplayMode {
  if (exchanges.length === 0) return "positional";
  const allHashed = exchanges.every(
    (e) => typeof e.request_hash === "string" && e.request_hash.length > 0,
  );
  return allHashed ? "hash_matched" : "positional";
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
