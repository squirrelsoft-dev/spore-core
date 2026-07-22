/**
 * Concrete `EmbeddingModel` providers — mirrors
 * `rust/crates/spore-core/src/embedding.rs`.
 *
 *   - {@link OllamaEmbeddingModel} — POST `/api/embed` (no API key; fails fast
 *     on connection errors with a "not running" hint; no retry).
 *   - {@link OpenAIEmbeddingModel} — POST `${base_url}/embeddings` (bearer
 *     auth; reuses the shared retry loop + `{error:{message}}` envelope).
 */

import { ProviderError, Timeout, Transport, type ModelError } from "../model/errors.js";
import {
  RETRYABLE_STATUSES,
  errorParts,
  keyFromEnv,
  sendWithRetry,
  type SleepFn,
} from "../model/provider-http.js";
import type { EmbeddingInfo, EmbeddingModel } from "./types.js";

const defaultSleep: SleepFn = (ms) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });

// ============================================================================
// Ollama
// ============================================================================

interface OllamaEmbedRequest {
  readonly model: string;
  readonly input: readonly string[];
}

interface OllamaEmbedResponse {
  readonly embeddings?: number[][];
}

/**
 * `EmbeddingModel` over a local Ollama server's `/api/embed` endpoint.
 *
 * No API key; default `baseUrl` is `http://localhost:11434`. Like the chat
 * `OllamaModelInterface` this fails fast on connection errors (no retry) with a
 * "not running" hint. Ollama does not report vector length except in the
 * embedding itself, so `dimensions` starts at `0` (unknown) — set it with
 * {@link OllamaEmbeddingModel.withDimensions} or discover it with
 * {@link OllamaEmbeddingModel.probeDimensions}.
 */
export class OllamaEmbeddingModel implements EmbeddingModel {
  static readonly DEFAULT_BASE_URL = "http://localhost:11434";
  static readonly DEFAULT_TIMEOUT_MS = 120_000;

  private readonly modelId: string;
  private readonly baseUrl: string;
  private readonly dims: number;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(modelId: string, opts: { baseUrl?: string; dimensions?: number; timeoutMs?: number; fetchImpl?: typeof fetch } = {}) {
    this.modelId = modelId;
    this.baseUrl = opts.baseUrl ?? OllamaEmbeddingModel.DEFAULT_BASE_URL;
    this.dims = opts.dimensions ?? 0;
    this.timeoutMs = opts.timeoutMs ?? OllamaEmbeddingModel.DEFAULT_TIMEOUT_MS;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  }

  withBaseUrl(baseUrl: string): this {
    return new OllamaEmbeddingModel(this.modelId, {
      baseUrl,
      dimensions: this.dims,
      timeoutMs: this.timeoutMs,
      fetchImpl: this.fetchImpl,
    }) as this;
  }

  /** Pin the vector length (so a `VectorStore` can be sized without a probe). */
  withDimensions(dimensions: number): this {
    return new OllamaEmbeddingModel(this.modelId, {
      baseUrl: this.baseUrl,
      dimensions,
      timeoutMs: this.timeoutMs,
      fetchImpl: this.fetchImpl,
    }) as this;
  }

  withTimeout(timeoutMs: number): this {
    return new OllamaEmbeddingModel(this.modelId, {
      baseUrl: this.baseUrl,
      dimensions: this.dims,
      timeoutMs,
      fetchImpl: this.fetchImpl,
    }) as this;
  }

  /** Discover the model's vector length by embedding a probe string. */
  async probeDimensions(signal?: AbortSignal): Promise<number> {
    const vector = await this.embedOne("dimension probe", signal);
    return vector.length;
  }

  async embed(inputs: string[], signal?: AbortSignal): Promise<number[][]> {
    if (inputs.length === 0) return [];
    const url = `${this.baseUrl}/api/embed`;
    const body: OllamaEmbedRequest = { model: this.modelId, input: inputs };
    let resp: Response;
    try {
      resp = await this.fetchImpl(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
    } catch (e) {
      throw this.mapTransportError(e);
    }
    if (!resp.ok) {
      const code = resp.status;
      let bodyText = "";
      try {
        bodyText = await resp.text();
      } catch {
        /* ignore */
      }
      throw new ProviderError(code, bodyText.slice(0, 500));
    }
    let parsed: OllamaEmbedResponse;
    try {
      parsed = (await resp.json()) as OllamaEmbedResponse;
    } catch (e) {
      throw new ProviderError(0, `decode embed response failed: ${formatError(e)}`);
    }
    return parsed.embeddings ?? [];
  }

  async embedOne(input: string, signal?: AbortSignal): Promise<number[]> {
    const vectors = await this.embed([input], signal);
    const v = vectors[0];
    if (v == null) {
      throw new ProviderError(0, "embedding provider returned no vectors");
    }
    return v;
  }

  dimensions(): number {
    return this.dims;
  }

  info(): EmbeddingInfo {
    return { provider: "ollama", modelId: this.modelId, dimensions: this.dims };
  }

  private mapTransportError(e: unknown): ModelError {
    if (e instanceof Error) {
      if (e.name === "AbortError") {
        return new ProviderError(0, `request aborted: ${e.message}`);
      }
      if (/timeout|ETIMEDOUT/i.test(e.message)) return new Timeout();
      // A connect failure (ECONNREFUSED, fetch failed to establish) → "not running".
      if (/ECONNREFUSED|Failed to fetch|fetch failed|NetworkError|connect/i.test(e.message)) {
        return new ProviderError(0, `Ollama not running at ${this.baseUrl}`);
      }
    }
    return new Transport(`HTTP transport error: ${formatError(e)}`);
  }
}

// ============================================================================
// OpenAI
// ============================================================================

interface OpenAIEmbedRequest {
  readonly model: string;
  readonly input: readonly string[];
  readonly dimensions?: number;
}

interface OpenAIEmbedResponse {
  readonly data: { embedding: number[]; index?: number }[];
}

/**
 * `EmbeddingModel` over `${base_url}/embeddings` (OpenAI + OpenAI-compatible
 * servers). Reuses the shared retry loop and `{error:{message}}` envelope,
 * exactly like the chat `OpenAIModelInterface`.
 *
 * `dimensions` defaults from the model id for the known `text-embedding-*`
 * families; {@link OpenAIEmbeddingModel.withDimensions} overrides it AND, for
 * the v3 families that accept the truncation parameter, asks the API for that
 * length. The id-prefix gate keeps older models (e.g. `ada-002`, which rejects
 * the parameter) working.
 */
export class OpenAIEmbeddingModel implements EmbeddingModel {
  static readonly DEFAULT_BASE_URL = "https://api.openai.com/v1";
  static readonly DEFAULT_TIMEOUT_MS = 120_000;
  static readonly DEFAULT_MAX_RETRIES = 3;

  private readonly apiKey: string;
  private readonly modelId: string;
  private readonly baseUrl: string;
  private readonly dims: number;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: SleepFn;

  constructor(
    apiKey: string,
    modelId: string,
    opts: { baseUrl?: string; dimensions?: number; timeoutMs?: number; maxRetries?: number; fetchImpl?: typeof fetch; sleep?: SleepFn } = {},
  ) {
    this.apiKey = apiKey;
    this.modelId = modelId;
    this.baseUrl = opts.baseUrl ?? OpenAIEmbeddingModel.DEFAULT_BASE_URL;
    this.dims = opts.dimensions ?? OpenAIEmbeddingModel.defaultDimensions(modelId);
    this.timeoutMs = opts.timeoutMs ?? OpenAIEmbeddingModel.DEFAULT_TIMEOUT_MS;
    this.maxRetries = opts.maxRetries ?? OpenAIEmbeddingModel.DEFAULT_MAX_RETRIES;
    this.fetchImpl = opts.fetchImpl ?? globalThis.fetch;
    this.sleep = opts.sleep ?? defaultSleep;
  }

  /** Read the key from `envVar` (rejects unset/empty). */
  static fromEnv(envVar: string, modelId: string, opts: { baseUrl?: string; dimensions?: number; timeoutMs?: number; maxRetries?: number; fetchImpl?: typeof fetch; sleep?: SleepFn } = {}): OpenAIEmbeddingModel {
    return new OpenAIEmbeddingModel(keyFromEnv(envVar), modelId, opts);
  }

  withBaseUrl(baseUrl: string): this {
    return new OpenAIEmbeddingModel(this.apiKey, this.modelId, {
      baseUrl,
      dimensions: this.dims,
      timeoutMs: this.timeoutMs,
      maxRetries: this.maxRetries,
      fetchImpl: this.fetchImpl,
      sleep: this.sleep,
    }) as this;
  }

  /**
   * Pin the vector length. For the `text-embedding-3-*` families this is also
   * sent as the request `dimensions` parameter (the API truncates to it); older
   * models that reject the parameter ignore this on the wire.
   */
  withDimensions(dimensions: number): this {
    return new OpenAIEmbeddingModel(this.apiKey, this.modelId, {
      baseUrl: this.baseUrl,
      dimensions,
      timeoutMs: this.timeoutMs,
      maxRetries: this.maxRetries,
      fetchImpl: this.fetchImpl,
      sleep: this.sleep,
    }) as this;
  }

  withTimeout(timeoutMs: number): this {
    return new OpenAIEmbeddingModel(this.apiKey, this.modelId, {
      baseUrl: this.baseUrl,
      dimensions: this.dims,
      timeoutMs,
      maxRetries: this.maxRetries,
      fetchImpl: this.fetchImpl,
      sleep: this.sleep,
    }) as this;
  }

  withMaxRetries(n: number): this {
    return new OpenAIEmbeddingModel(this.apiKey, this.modelId, {
      baseUrl: this.baseUrl,
      dimensions: this.dims,
      timeoutMs: this.timeoutMs,
      maxRetries: n,
      fetchImpl: this.fetchImpl,
      sleep: this.sleep,
    }) as this;
  }

  /** Vector length for the known `text-embedding-*` families; `0` otherwise. */
  static defaultDimensions(modelId: string): number {
    switch (modelId) {
      case "text-embedding-3-large":
        return 3072;
      case "text-embedding-3-small":
      case "text-embedding-ada-002":
        return 1536;
      default:
        return 0;
    }
  }

  /** Whether this model accepts the request `dimensions` truncation parameter. */
  private supportsDimensionsParam(): boolean {
    return this.modelId.startsWith("text-embedding-3");
  }

  async embed(inputs: string[], signal?: AbortSignal): Promise<number[][]> {
    if (inputs.length === 0) return [];
    const url = `${this.baseUrl}/embeddings`;
    const dimensions = this.dims > 0 && this.supportsDimensionsParam() ? this.dims : undefined;
    const body: OpenAIEmbedRequest = { model: this.modelId, input: inputs, dimensions };
    const bodyStr = JSON.stringify(body);
    const requestBuilder = (abortSignal: AbortSignal): Promise<Response> =>
      this.fetchImpl(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.apiKey}`,
        },
        body: bodyStr,
        signal: abortSignal,
      });
    const resp = await sendWithRetry(
      requestBuilder,
      this.maxRetries,
      this.timeoutMs,
      (code) => RETRYABLE_STATUSES.has(code),
      async (r) => {
        const parts = await errorParts(r);
        return new ProviderError(parts.code, parts.message);
      },
      signal,
      this.sleep,
    );
    let parsed: OpenAIEmbedResponse;
    try {
      parsed = (await resp.json()) as OpenAIEmbedResponse;
    } catch (e) {
      throw new ProviderError(0, `decode embeddings response failed: ${formatError(e)}`);
    }
    // The API returns data in request order, but `index` is authoritative —
    // sort by it so a non-conforming compatible server can't scramble the
    // input↔vector correspondence.
    const data = parsed.data.slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    return data.map((d) => d.embedding);
  }

  async embedOne(input: string, signal?: AbortSignal): Promise<number[]> {
    const vectors = await this.embed([input], signal);
    const v = vectors[0];
    if (v == null) {
      throw new ProviderError(0, "embedding provider returned no vectors");
    }
    return v;
  }

  dimensions(): number {
    return this.dims;
  }

  info(): EmbeddingInfo {
    return { provider: "openai", modelId: this.modelId, dimensions: this.dims };
  }
}

function formatError(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}