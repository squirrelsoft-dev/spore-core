/**
 * Issue #39 — `AnthropicModelInterface`: real Anthropic Messages API client.
 *
 * Implements {@link ModelInterface} against `https://api.anthropic.com/v1/messages`.
 * Translates {@link ModelRequest} / {@link ModelResponse} to and from
 * Anthropic's wire format, parses the SSE event stream for
 * {@link AnthropicModelInterface.callStreaming}, hits
 * `/v1/messages/count_tokens` for accurate token counts, and maps HTTP errors
 * to typed {@link ModelError} variants with retry/backoff for transient
 * failures.
 *
 * Mirrors `rust/crates/spore-core/src/anthropic.rs` rule-for-rule.
 *
 * Cache-cost wiring: per-model cache pricing lives in
 * {@link "../cache-provider/types.js".AnthropicCacheProvider.withModelPricing}.
 */

import { ProviderError, RateLimited, Timeout, type ModelError } from "./errors.js";
import type { ModelInterface } from "./interface.js";
import type {
  Content,
  ContentBlock,
  Message,
  ModelRequest,
  ModelResponse,
  ProviderInfo,
  StopReason,
  StreamEvent,
  TokenUsage,
  ToolSchema,
} from "./schemas.js";

// ============================================================================
// Options
// ============================================================================

export interface AnthropicModelInterfaceOptions {
  /** Override the base URL (used by tests pointing at a mock server). */
  baseUrl?: string;
  /** Request timeout in milliseconds. Defaults to 120_000 (120s). */
  timeoutMs?: number;
  /** Retry budget for transient 408/425/429/500/502/503/504/529. Defaults to 3. */
  maxRetries?: number;
  /**
   * Injected fetch implementation. Defaults to the global `fetch`. Tests pass
   * a Node http.Server-backed wrapper or stub fetch. Useful for unit testing
   * without spinning up a server.
   */
  fetchImpl?: typeof fetch;
  /**
   * Hook called before each retry sleep. Tests override to skip waits.
   * Receives the planned delay (ms) and the attempt index (0-based).
   */
  sleep?: (ms: number) => Promise<void>;
}

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_BASE_URL = "https://api.anthropic.com";
export const DEFAULT_TIMEOUT_MS = 120_000;
export const DEFAULT_MAX_RETRIES = 3;
export const ANTHROPIC_VERSION = "2023-06-01";

const RETRYABLE_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504, 529]);

// ============================================================================
// AnthropicModelInterface
// ============================================================================

/**
 * Reference Anthropic client. Constructed with an API key and a model id;
 * callers can override base URL (for proxying or mocking) and tune retry
 * behavior.
 */
export class AnthropicModelInterface implements ModelInterface {
  private readonly apiKey: string;
  private readonly modelId: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(apiKey: string, modelId: string, options: AnthropicModelInterfaceOptions = {}) {
    this.apiKey = apiKey;
    this.modelId = modelId;
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
    this.sleep =
      options.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  }

  /**
   * Read API key from environment variable. Throws `ProviderError` if the
   * variable is unset or empty.
   */
  static fromEnv(
    envVar: string,
    modelId: string,
    options: AnthropicModelInterfaceOptions = {},
  ): AnthropicModelInterface {
    const key = process.env[envVar];
    if (key == null) {
      throw new ProviderError(0, `env var \`${envVar}\` not set`);
    }
    if (key.trim() === "") {
      throw new ProviderError(0, `env var \`${envVar}\` is empty`);
    }
    return new AnthropicModelInterface(key, modelId, options);
  }

  /**
   * Context window for known model ids. Falls back to 200k for any
   * `claude-*` id and to 0 otherwise so that callers can detect "unknown
   * model" rather than silently getting a plausible-but-wrong value.
   */
  static contextWindow(modelId: string): number {
    switch (modelId) {
      case "claude-sonnet-4-5":
      case "claude-sonnet-4-6":
      case "claude-opus-4-5":
      case "claude-opus-4-6":
      case "claude-opus-4-7":
      case "claude-haiku-4-5":
      case "claude-haiku-4-5-20251001":
        return 200_000;
      default:
        return modelId.startsWith("claude-") ? 200_000 : 0;
    }
  }

  provider(): ProviderInfo {
    return {
      name: "anthropic",
      model_id: this.modelId,
      context_window: AnthropicModelInterface.contextWindow(this.modelId),
    };
  }

  /**
   * Debug helper. The API key MUST never appear in logs or traces.
   */
  toJSON(): Record<string, unknown> {
    return {
      api_key: "<redacted>",
      model_id: this.modelId,
      base_url: this.baseUrl,
      timeout_ms: this.timeoutMs,
      max_retries: this.maxRetries,
    };
  }

  async call(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse> {
    const body = JSON.stringify(buildRequest(this.modelId, request, false));
    const url = `${this.baseUrl}/v1/messages`;
    const resp = await this.sendWithRetry(url, body, false, signal);
    let parsed: AnthropicResponse;
    try {
      parsed = (await resp.json()) as AnthropicResponse;
    } catch (e) {
      throw new ProviderError(0, `response decode failed: ${formatError(e)}`);
    }
    return parseResponse(parsed);
  }

  async *callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = JSON.stringify(buildRequest(this.modelId, request, true));
    const url = `${this.baseUrl}/v1/messages`;
    // Streaming does not retry by spec — once the connection opens, we
    // surface whatever the server returns.
    let resp: Response;
    try {
      resp = await this.fetchOnce(url, body, true, signal);
    } catch (e) {
      throw toModelError(e);
    }
    if (!resp.ok) {
      throw await mapStatusError(resp);
    }
    if (resp.body == null) {
      throw new ProviderError(0, "streaming response had no body");
    }
    yield* sseToEvents(resp.body);
  }

  async countTokens(request: ModelRequest, signal?: AbortSignal): Promise<number> {
    const body = JSON.stringify(buildRequest(this.modelId, request, false));
    const url = `${this.baseUrl}/v1/messages/count_tokens`;
    const resp = await this.sendWithRetry(url, body, false, signal);
    let parsed: { input_tokens: number };
    try {
      parsed = (await resp.json()) as { input_tokens: number };
    } catch (e) {
      throw new ProviderError(0, `count_tokens decode failed: ${formatError(e)}`);
    }
    return parsed.input_tokens;
  }

  // ── HTTP plumbing ───────────────────────────────────────────────────────

  private async fetchOnce(
    url: string,
    body: string,
    streaming: boolean,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    const headers: Record<string, string> = {
      "x-api-key": this.apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
      "content-type": "application/json",
    };
    if (streaming) headers["accept"] = "text/event-stream";

    const controller = new AbortController();
    const timeoutHandle = setTimeout(() => controller.abort(new TimeoutSentinel()), this.timeoutMs);
    const onUserAbort = () => controller.abort(signal?.reason);
    if (signal != null) {
      if (signal.aborted) controller.abort(signal.reason);
      else signal.addEventListener("abort", onUserAbort, { once: true });
    }
    try {
      return await this.fetchImpl(url, {
        method: "POST",
        headers,
        body,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutHandle);
      if (signal != null) signal.removeEventListener("abort", onUserAbort);
    }
  }

  private async sendWithRetry(
    url: string,
    body: string,
    streaming: boolean,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    let attempt = 0;
    while (true) {
      let resp: Response;
      try {
        resp = await this.fetchOnce(url, body, streaming, signal);
      } catch (e) {
        const err = toModelError(e);
        if (err instanceof Timeout && attempt < this.maxRetries) {
          await this.sleep(backoffDelayMs(attempt));
          attempt += 1;
          continue;
        }
        throw err;
      }
      if (resp.ok) return resp;
      const code = resp.status;
      if (RETRYABLE_STATUSES.has(code) && attempt < this.maxRetries) {
        const retryAfter = parseRetryAfter(resp.headers.get("retry-after"));
        const delayMs = retryAfter != null ? retryAfter * 1000 : backoffDelayMs(attempt);
        // Drain the body so the connection is reusable.
        try {
          await resp.text();
        } catch {
          /* ignore */
        }
        await this.sleep(delayMs);
        attempt += 1;
        continue;
      }
      throw await mapStatusError(resp);
    }
  }
}

// ============================================================================
// Wire-format types
// ============================================================================

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  messages: AnthropicMessage[];
  system?: string;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  tools?: AnthropicTool[];
  stream?: boolean;
}

interface AnthropicMessage {
  role: "user" | "assistant";
  content: AnthropicContent[];
}

type AnthropicContent =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }
  | { type: "image"; source: { type: "base64"; media_type: string; data: string } };

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: unknown;
}

interface AnthropicResponse {
  content?: AnthropicResponseBlock[];
  stop_reason?: string | null;
  usage?: AnthropicUsage;
}

type AnthropicResponseBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | { type: "tool_use"; id: string; name: string; input: unknown };

interface AnthropicUsage {
  input_tokens?: number;
  output_tokens?: number;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
}

// ============================================================================
// Conversions (exported for tests)
// ============================================================================

/**
 * Translate a Spore `ModelRequest` to the Anthropic Messages API body.
 * System-role messages are extracted into the top-level `system` field; all
 * other messages keep their role and wrap their `Content` in a single-element
 * content-block array.
 */
export function buildRequest(
  modelId: string,
  req: ModelRequest,
  stream: boolean,
): AnthropicRequest {
  const systemParts: string[] = [];
  const messages: AnthropicMessage[] = [];
  for (const m of req.messages) {
    if (m.role === "system") {
      if (m.content.type === "text") systemParts.push(m.content.text);
      continue;
    }
    const role: "user" | "assistant" = m.role === "assistant" ? "assistant" : "user";
    messages.push({ role, content: [contentToAnthropic(m.content)] });
  }
  const out: AnthropicRequest = {
    model: modelId,
    max_tokens: req.params.max_tokens ?? 4096,
    messages,
  };
  if (systemParts.length > 0) out.system = systemParts.join("\n\n");
  if (req.params.temperature != null) out.temperature = req.params.temperature;
  if (req.params.top_p != null) out.top_p = req.params.top_p;
  if (req.params.stop_sequences.length > 0) out.stop_sequences = [...req.params.stop_sequences];
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t: ToolSchema) => ({
      name: t.name,
      description: t.description,
      input_schema: t.input_schema,
    }));
  }
  if (stream) out.stream = true;
  return out;
}

function contentToAnthropic(c: Content): AnthropicContent {
  switch (c.type) {
    case "text":
      return { type: "text", text: c.text };
    case "tool_call":
      return { type: "tool_use", id: c.id, name: c.name, input: c.input };
    case "tool_result": {
      const out: AnthropicContent = {
        type: "tool_result",
        tool_use_id: c.tool_use_id,
        content: c.content,
      };
      if (c.is_error) out.is_error = true;
      return out;
    }
    case "image":
      return {
        type: "image",
        source: { type: "base64", media_type: c.media_type, data: c.data },
      };
    default: {
      const _exhaustive: never = c;
      return _exhaustive;
    }
  }
}

export function parseResponse(body: AnthropicResponse): ModelResponse {
  const content: ContentBlock[] = (body.content ?? []).map((b) => {
    switch (b.type) {
      case "text":
        return { type: "text", text: b.text };
      case "thinking":
        return { type: "thinking", text: b.thinking };
      case "tool_use":
        return { type: "tool_use", id: b.id, name: b.name, input: b.input };
      default: {
        const _exhaustive: never = b;
        return _exhaustive;
      }
    }
  });
  const usage: TokenUsage = {
    input_tokens: body.usage?.input_tokens ?? 0,
    output_tokens: body.usage?.output_tokens ?? 0,
    cache_read_tokens: body.usage?.cache_read_input_tokens ?? null,
    cache_write_tokens: body.usage?.cache_creation_input_tokens ?? null,
  };
  return {
    content,
    usage,
    stop_reason: parseStopReason(body.stop_reason ?? null),
  };
}

export function parseStopReason(s: string | null | undefined): StopReason {
  switch (s) {
    case "tool_use":
      return "tool_use";
    case "max_tokens":
      return "max_tokens";
    case "stop_sequence":
      return "stop_sequence";
    default:
      return "end_turn";
  }
}

// ============================================================================
// Error mapping
// ============================================================================

interface AnthropicErrorBody {
  error?: { message?: string };
}

async function mapStatusError(resp: Response): Promise<ModelError> {
  const code = resp.status;
  const retryAfter = parseRetryAfter(resp.headers.get("retry-after"));
  let bodyText = "";
  try {
    bodyText = await resp.text();
  } catch {
    /* ignore */
  }
  let message = bodyText.slice(0, 500);
  try {
    const parsed = JSON.parse(bodyText) as AnthropicErrorBody;
    if (parsed?.error?.message != null) message = parsed.error.message;
  } catch {
    /* keep truncated body */
  }
  if (code === 429) return new RateLimited(retryAfter ?? null);
  if (code === 529) return new RateLimited(null);
  if (code === 408 || code === 504) return new Timeout();
  return new ProviderError(code, message);
}

function parseRetryAfter(header: string | null): number | null {
  if (header == null) return null;
  const n = parseInt(header.trim(), 10);
  if (!Number.isFinite(n) || n < 0) return null;
  return n;
}

class TimeoutSentinel extends Error {
  constructor() {
    super("anthropic-timeout");
    this.name = "TimeoutSentinel";
  }
}

function toModelError(e: unknown): ModelError {
  if (e instanceof Timeout || e instanceof RateLimited || e instanceof ProviderError) {
    return e;
  }
  if (e instanceof Error) {
    if (e.name === "TimeoutSentinel") return new Timeout();
    if (e.name === "AbortError") {
      // fetch aborted — if it was a user signal, surface as ProviderError.
      // Timeouts surface via TimeoutSentinel wrapped in the abort reason.
      const cause = (e as { cause?: unknown }).cause;
      if (cause instanceof TimeoutSentinel) return new Timeout();
      return new ProviderError(0, `request aborted: ${e.message}`);
    }
  }
  // Treat undici/Node timeout-flavored errors as Timeout.
  if (e instanceof Error && /timeout|ETIMEDOUT|AbortError/i.test(e.message)) {
    return new Timeout();
  }
  return new ProviderError(0, `HTTP transport error: ${formatError(e)}`);
}

function formatError(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

/**
 * Exponential backoff: 500ms, 1s, 2s, 4s, ... capped at 30s.
 * Pure function — exported for tests.
 */
export function backoffDelayMs(attempt: number): number {
  const shift = Math.min(Math.max(attempt, 0), 6);
  const base = 500 * (1 << shift);
  return Math.min(base, 30_000);
}

// ============================================================================
// SSE parsing
// ============================================================================

/**
 * Parse one SSE event block (`event: name\ndata: {...}`). Returns
 * `{ event, data }` or `null` if the block doesn't follow the expected shape.
 * Multi-line `data:` lines are joined with `\n`.
 *
 * Exported for tests.
 */
export function parseSseEvent(raw: string): { event: string; data: string } | null {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice("data:".length).replace(/^ /, ""));
    }
  }
  if (event == null) return null;
  return { event, data: data.join("\n") };
}

/**
 * Convert an Anthropic SSE response body into a stream of `StreamEvent`s.
 *
 * Exported for tests so unit tests can drive the parser without HTTP.
 */
export async function* sseToEvents(body: ReadableStream<Uint8Array>): AsyncIterable<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  const usage: TokenUsage = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: null,
    cache_write_tokens: null,
  };
  let stopReason: StopReason = "end_turn";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = parseSseEvent(raw);
        if (parsed == null) continue;
        const { event, data } = parsed;
        if (data === "" || data === "{}") continue;
        let json: unknown;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        const ev = jsonValue(json);
        if (event === "message_start") {
          const msg = ev?.message as Record<string, unknown> | undefined;
          const u = msg?.usage as Record<string, unknown> | undefined;
          if (typeof u?.input_tokens === "number") usage.input_tokens = u.input_tokens;
          yield { type: "message_start" };
        } else if (event === "content_block_start") {
          // A tool_use block opens here with its id + name; emit tool_use_start
          // so the accumulator captures them before the input_json_delta arg
          // fragments arrive.
          const index = typeof ev?.index === "number" ? ev.index : 0;
          const block = jsonValue(ev?.content_block);
          if (block?.type === "tool_use") {
            const id = typeof block.id === "string" ? block.id : "";
            const name = typeof block.name === "string" ? block.name : "";
            yield { type: "tool_use_start", index, id, name };
          }
        } else if (event === "content_block_delta") {
          const index = typeof ev?.index === "number" ? ev.index : 0;
          const delta = ev?.delta as Record<string, unknown> | undefined;
          const kind = typeof delta?.type === "string" ? delta.type : "";
          if (kind === "text_delta") {
            const text = typeof delta?.text === "string" ? delta.text : "";
            yield { type: "content_block_delta", index, delta: text };
          } else if (kind === "thinking_delta") {
            const text = typeof delta?.thinking === "string" ? delta.thinking : "";
            yield { type: "thinking_delta", index, delta: text };
          } else if (kind === "input_json_delta") {
            const partial = typeof delta?.partial_json === "string" ? delta.partial_json : "";
            yield { type: "tool_use_delta", index, partial_json: partial };
          }
        } else if (event === "content_block_stop") {
          const index = typeof ev?.index === "number" ? ev.index : 0;
          yield { type: "content_block_stop", index };
        } else if (event === "message_delta") {
          const delta = ev?.delta as Record<string, unknown> | undefined;
          const sr = delta?.stop_reason;
          if (typeof sr === "string") stopReason = parseStopReason(sr);
          const u = ev?.usage as Record<string, unknown> | undefined;
          if (typeof u?.output_tokens === "number") usage.output_tokens = u.output_tokens;
        } else if (event === "message_stop") {
          yield { type: "message_stop", usage, stop_reason: stopReason };
          return;
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* ignore */
    }
  }
}

function jsonValue(v: unknown): Record<string, unknown> | undefined {
  if (v != null && typeof v === "object" && !Array.isArray(v)) {
    return v as Record<string, unknown>;
  }
  return undefined;
}

// Re-export `Message` for tests that need it without pulling from schemas.
export type { Message };
