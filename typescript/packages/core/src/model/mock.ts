/**
 * `MockModelInterface` — programmable mock for unit tests.
 *
 * Each `call` pops the next queued result (response or error). `callCount`
 * lets tests assert how many times the harness invoked the model.
 */

import { ProviderError, type ModelError } from "./errors.js";
import type { ModelInterface } from "./interface.js";
import type {
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StreamEvent,
} from "./schemas.js";

type Queued<T> = { ok: true; value: T } | { ok: false; error: ModelError };

export class MockModelInterface implements ModelInterface {
  private readonly responses: Queued<ModelResponse>[] = [];
  private readonly tokenCounts: Queued<number>[] = [];
  callCount = 0;

  constructor(private readonly providerInfo: ProviderInfo) {}

  pushResponse(value: ModelResponse): this {
    this.responses.push({ ok: true, value });
    return this;
  }

  pushError(error: ModelError): this {
    this.responses.push({ ok: false, error });
    return this;
  }

  pushTokenCount(value: number): this {
    this.tokenCounts.push({ ok: true, value });
    return this;
  }

  provider(): ProviderInfo {
    return this.providerInfo;
  }

  async call(_request: ModelRequest, _signal?: AbortSignal): Promise<ModelResponse> {
    this.callCount += 1;
    const next = this.responses.shift();
    if (next == null) {
      throw new ProviderError(0, "mock: no response queued");
    }
    if (!next.ok) throw next.error;
    return next.value;
  }

  async *callStreaming(
    request: ModelRequest,
    signal?: AbortSignal,
  ): AsyncIterable<StreamEvent> {
    const response = await this.call(request, signal);
    yield { type: "message_start" };
    yield {
      type: "message_stop",
      usage: response.usage,
      stop_reason: response.stop_reason,
    };
  }

  async countTokens(_request: ModelRequest, _signal?: AbortSignal): Promise<number> {
    const next = this.tokenCounts.shift();
    if (next == null) return 0;
    if (!next.ok) throw next.error;
    return next.value;
  }
}
