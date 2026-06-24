/**
 * Issue #40 — `OpenAIModelInterface`: real OpenAI Chat Completions client.
 *
 * Implements {@link ModelInterface} against `${baseUrl}/chat/completions`.
 * Translates {@link ModelRequest} / {@link ModelResponse} to and from the
 * OpenAI wire format, parses the OpenAI SSE event stream for
 * {@link OpenAIModelInterface.callStreaming}, handles tool-call delta
 * accumulation, and maps HTTP errors to typed {@link ModelError} variants
 * with retry/backoff for transient failures.
 *
 * Mirrors `rust/crates/spore-core/src/openai.rs` rule-for-rule.
 *
 * ## Provider-specific shape
 * - System messages become `{role: "system", content: ...}` entries in the
 *   `messages` array (Anthropic extracts them — OpenAI does not).
 * - Assistant tool calls travel in a `tool_calls` array on the assistant
 *   message. `function.arguments` is a JSON-encoded STRING (not an object).
 *   Tool results travel as standalone messages with `role: "tool"` and a
 *   `tool_call_id` linking back to the call.
 * - Reasoning models (`o1`, `o3`, `o4*`) do not accept `temperature` and
 *   replace `max_tokens` with `max_completion_tokens`. Detection by id prefix.
 * - Streaming SSE chunks contain `delta.content` (text), `delta.tool_calls`
 *   (partial tool calls indexed and accumulated across chunks), and end with
 *   a literal `data: [DONE]` line. Final `usage` only appears when the
 *   request set `stream_options: {include_usage: true}`.
 *
 * ## Token counting
 * OpenAI does not expose a counter endpoint. We use the bytes/4 heuristic —
 * sufficient for compaction decisions; exact counts come from response usage.
 * A future revision may pull in `tiktoken`.
 *
 * Cache-cost wiring: per-model cache pricing lives in
 * {@link "../cache-provider/types.js".OpenAICacheProvider.withModelPricing}.
 */

import {
  ProviderError,
  RateLimited,
  StreamInterrupted,
  Timeout,
  Transport,
  type ModelError,
} from "./errors.js";
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

export interface OpenAIModelInterfaceOptions {
  /** Override the base URL (used by tests + Azure / proxy deployments). */
  baseUrl?: string;
  /** Request timeout in milliseconds. Defaults to 120_000 (120s). */
  timeoutMs?: number;
  /** Retry budget for transient 408/425/429/500/502/503/504. Defaults to 3. */
  maxRetries?: number;
  /** Injected fetch implementation. Defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
  /** Hook called before each retry sleep. Tests override to skip waits. */
  sleep?: (ms: number) => Promise<void>;
}

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_BASE_URL = "https://api.openai.com/v1";
export const DEFAULT_TIMEOUT_MS = 120_000;
export const DEFAULT_MAX_RETRIES = 3;

const RETRYABLE_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504]);

// ============================================================================
// OpenAIModelInterface
// ============================================================================

export class OpenAIModelInterface implements ModelInterface {
  private readonly apiKey: string;
  private readonly modelId: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(apiKey: string, modelId: string, options: OpenAIModelInterfaceOptions = {}) {
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
    options: OpenAIModelInterfaceOptions = {},
  ): OpenAIModelInterface {
    const key = process.env[envVar];
    if (key == null) {
      throw new ProviderError(0, `env var \`${envVar}\` not set`);
    }
    if (key.trim() === "") {
      throw new ProviderError(0, `env var \`${envVar}\` is empty`);
    }
    return new OpenAIModelInterface(key, modelId, options);
  }

  /**
   * Context window for known model ids. Falls back to 0 for unknown ids so
   * callers can detect "unknown model" rather than silently getting a
   * plausible-but-wrong value.
   */
  static contextWindow(modelId: string): number {
    if (modelId.startsWith("gpt-4o")) return 128_000;
    if (modelId.startsWith("gpt-4.1")) return 1_000_000;
    if (modelId.startsWith("o3") || modelId.startsWith("o4")) return 200_000;
    if (modelId.startsWith("o1")) return 128_000;
    return 0;
  }

  /**
   * True if this is an o-series reasoning model. Reasoning models have
   * different parameter constraints (no `temperature`, use
   * `max_completion_tokens`).
   */
  static isReasoningModel(modelId: string): boolean {
    return modelId.startsWith("o1") || modelId.startsWith("o3") || modelId.startsWith("o4");
  }

  provider(): ProviderInfo {
    return {
      name: "openai",
      model_id: this.modelId,
      context_window: OpenAIModelInterface.contextWindow(this.modelId),
    };
  }

  /** Debug helper. The API key MUST never appear in logs or traces. */
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
    const url = `${this.baseUrl}/chat/completions`;
    const resp = await this.sendWithRetry(url, body, false, signal);
    let parsed: OpenAIResponse;
    try {
      parsed = (await resp.json()) as OpenAIResponse;
    } catch (e) {
      throw new ProviderError(0, `response decode failed: ${formatError(e)}`);
    }
    return parseResponse(parsed);
  }

  async *callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = JSON.stringify(buildRequest(this.modelId, request, true));
    const url = `${this.baseUrl}/chat/completions`;
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

  async countTokens(request: ModelRequest, _signal?: AbortSignal): Promise<number> {
    void _signal;
    // OpenAI has no count_tokens endpoint. Use the bytes/4 heuristic
    // consistent with ReplayModelInterface — sufficient for compaction
    // decisions; exact counts come back via response usage.
    let n = 0;
    for (const m of request.messages) {
      const c = m.content;
      switch (c.type) {
        case "text":
          n += c.text.length;
          break;
        case "tool_call":
          n += c.name.length + JSON.stringify(c.input ?? {}).length;
          break;
        case "tool_result":
          n += c.content.length;
          break;
        case "image":
          break;
      }
    }
    return Math.floor(n / 4);
  }

  // ── HTTP plumbing ───────────────────────────────────────────────────────

  private async fetchOnce(
    url: string,
    body: string,
    streaming: boolean,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    const headers: Record<string, string> = {
      authorization: `Bearer ${this.apiKey}`,
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
// Wire-format types (OpenAI Chat Completions API)
// ============================================================================

interface OpenAIRequest {
  model: string;
  messages: OpenAIMessage[];
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
  tools?: OpenAITool[];
  stream?: boolean;
  stream_options?: { include_usage: boolean };
}

interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface OpenAITool {
  type: "function";
  function: { name: string; description: string; parameters: unknown };
}

interface OpenAIResponse {
  choices?: OpenAIChoice[];
  usage?: OpenAIUsage;
}

interface OpenAIChoice {
  message?: OpenAIResponseMessage;
  finish_reason?: string | null;
}

interface OpenAIResponseMessage {
  content?: string | null;
  reasoning?: string | null;
  tool_calls?: OpenAIResponseToolCall[];
}

interface OpenAIResponseToolCall {
  id: string;
  function?: { name?: string; arguments?: string };
}

interface OpenAIUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number | null };
}

// ============================================================================
// Conversions (exported for tests)
// ============================================================================

export function buildRequest(modelId: string, req: ModelRequest, stream: boolean): OpenAIRequest {
  const messages: OpenAIMessage[] = req.messages.map(messageToOpenAI);

  const tools: OpenAITool[] = req.tools.map((t: ToolSchema) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
    },
  }));

  const isReasoning = OpenAIModelInterface.isReasoningModel(modelId);
  const out: OpenAIRequest = {
    model: modelId,
    messages,
  };
  if (req.params.max_tokens != null) {
    if (isReasoning) {
      out.max_completion_tokens = req.params.max_tokens;
    } else {
      out.max_tokens = req.params.max_tokens;
    }
  }
  if (!isReasoning && req.params.temperature != null) out.temperature = req.params.temperature;
  if (req.params.top_p != null) out.top_p = req.params.top_p;
  if (req.params.stop_sequences.length > 0) out.stop = [...req.params.stop_sequences];
  if (tools.length > 0) out.tools = tools;
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function messageToOpenAI(m: Message): OpenAIMessage {
  const role: OpenAIMessage["role"] =
    m.role === "system"
      ? "system"
      : m.role === "assistant"
        ? "assistant"
        : m.role === "tool"
          ? "tool"
          : "user";

  const c: Content = m.content;
  switch (c.type) {
    case "text":
      return { role, content: c.text };
    case "tool_call":
      return {
        role: "assistant",
        tool_calls: [
          {
            id: c.id,
            type: "function",
            function: {
              name: c.name,
              arguments: safeStringify(c.input),
            },
          },
        ],
      };
    case "tool_result":
      return {
        role: "tool",
        content: c.content,
        tool_call_id: c.tool_use_id,
      };
    case "image":
      // OpenAI's chat-completions image input uses a content-parts array.
      // The harness does not currently emit image content into requests, so
      // we serialize a textual placeholder rather than introduce a
      // heterogeneous shape.
      return { role, content: `[image: ${c.media_type}]` };
    default: {
      const _exhaustive: never = c;
      return _exhaustive;
    }
  }
}

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v ?? {});
  } catch {
    return "{}";
  }
}

export function parseResponse(body: OpenAIResponse): ModelResponse {
  const choice: OpenAIChoice = body.choices?.[0] ?? {};
  const msg: OpenAIResponseMessage = choice.message ?? {};

  const content: ContentBlock[] = [];
  if (msg.reasoning != null && msg.reasoning !== "") {
    content.push({ type: "thinking", text: msg.reasoning });
  }
  if (msg.content != null && msg.content !== "") {
    content.push({ type: "text", text: msg.content });
  }
  for (const tc of msg.tool_calls ?? []) {
    const argsStr = tc.function?.arguments ?? "";
    let input: unknown;
    if (argsStr === "") {
      input = {};
    } else {
      try {
        input = JSON.parse(argsStr);
      } catch {
        input = argsStr;
      }
    }
    content.push({
      type: "tool_use",
      id: tc.id,
      name: tc.function?.name ?? "",
      input,
    });
  }

  const usage: TokenUsage = {
    input_tokens: body.usage?.prompt_tokens ?? 0,
    output_tokens: body.usage?.completion_tokens ?? 0,
    cache_read_tokens: body.usage?.prompt_tokens_details?.cached_tokens ?? null,
    // OpenAI does not report cache writes directly.
    cache_write_tokens: null,
  };

  return {
    content,
    usage,
    stop_reason: parseStopReason(choice.finish_reason ?? null),
  };
}

export function parseStopReason(s: string | null | undefined): StopReason {
  switch (s) {
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "length":
      return "max_tokens";
    case "stop":
      return "end_turn";
    default:
      return "end_turn";
  }
}

// ============================================================================
// Error mapping
// ============================================================================

interface OpenAIErrorBody {
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
    const parsed = JSON.parse(bodyText) as OpenAIErrorBody;
    if (parsed?.error?.message != null) message = parsed.error.message;
  } catch {
    /* keep truncated body */
  }
  if (code === 429) return new RateLimited(retryAfter ?? null);
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
    super("openai-timeout");
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
      const cause = (e as { cause?: unknown }).cause;
      if (cause instanceof TimeoutSentinel) return new Timeout();
      return new ProviderError(0, `request aborted: ${e.message}`);
    }
  }
  if (e instanceof Error && /timeout|ETIMEDOUT|AbortError/i.test(e.message)) {
    return new Timeout();
  }
  // SC-3: a network/transport failure reaching the provider is a typed,
  // retryable Transport error — not a generic ProviderError.
  return new Transport(`HTTP transport error: ${formatError(e)}`);
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
 * Convert an OpenAI SSE response body into a stream of `StreamEvent`s.
 *
 * OpenAI streams chat-completion delta chunks. Each `data:` line carries a
 * JSON object shaped like
 * `{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`.
 *
 * Tool calls arrive as partial entries in `delta.tool_calls`, indexed; the
 * `id` and `function.name` arrive on the first chunk for a given index, and
 * subsequent chunks for the same index carry incremental `function.arguments`
 * JSON-fragment strings. The stream ends with `data: [DONE]`. When
 * `stream_options.include_usage` was set, the final non-`[DONE]` chunk also
 * carries `usage`.
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
  let started = false;
  // Tool call indices are independent of the text content index; shift by 1
  // to keep them disjoint from index 0 (which conventionally carries text).
  const toolIndicesSeen = new Set<number>();
  let contentIndexEmitted = false;
  let contentIndex = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        const rawLine = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        const line = rawLine.replace(/\r$/, "");
        if (!line.startsWith("data:")) continue;
        const data = line.slice("data:".length).replace(/^ /, "");
        if (data === "") continue;
        if (data === "[DONE]") {
          yield { type: "message_stop", usage, stop_reason: stopReason };
          return;
        }
        let json: unknown;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        const obj = jsonValue(json);
        if (obj == null) continue;
        if (!started) {
          started = true;
          yield { type: "message_start" };
        }
        const u = jsonValue(obj.usage);
        if (u != null) {
          if (typeof u.prompt_tokens === "number") usage.input_tokens = u.prompt_tokens;
          if (typeof u.completion_tokens === "number") usage.output_tokens = u.completion_tokens;
          const ptd = jsonValue(u.prompt_tokens_details);
          if (ptd != null && typeof ptd.cached_tokens === "number") {
            usage.cache_read_tokens = ptd.cached_tokens;
          }
        }
        const choices = Array.isArray(obj.choices) ? obj.choices : [];
        const choice = jsonValue(choices[0]);
        if (choice == null) continue;
        if (typeof choice.finish_reason === "string") {
          stopReason = parseStopReason(choice.finish_reason);
        }
        const delta = jsonValue(choice.delta);
        if (delta == null) continue;
        if (typeof delta.content === "string" && delta.content !== "") {
          if (!contentIndexEmitted) contentIndexEmitted = true;
          yield {
            type: "content_block_delta",
            index: contentIndex,
            delta: delta.content,
          };
        }
        if (typeof delta.reasoning === "string" && delta.reasoning !== "") {
          yield {
            type: "thinking_delta",
            index: contentIndex,
            delta: delta.reasoning,
          };
        }
        const tcs = Array.isArray(delta.tool_calls) ? delta.tool_calls : null;
        if (tcs != null) {
          for (const tcRaw of tcs) {
            const tc = jsonValue(tcRaw);
            if (tc == null) continue;
            const i = typeof tc.index === "number" ? tc.index : 0;
            const eventIndex = i + 1;
            const fn = jsonValue(tc.function);
            if (!toolIndicesSeen.has(eventIndex)) {
              toolIndicesSeen.add(eventIndex);
              if (contentIndexEmitted) {
                yield { type: "content_block_stop", index: contentIndex };
                contentIndexEmitted = false;
                contentIndex = eventIndex;
              }
              // The id + function.name arrive on this first chunk for the index;
              // emit tool_use_start so they aren't lost when only argument
              // fragments follow. A missing id is synthesized stably.
              const name = typeof fn?.name === "string" ? fn.name : "";
              const id = typeof tc.id === "string" ? tc.id : `call_${eventIndex}`;
              yield { type: "tool_use_start", index: eventIndex, id, name };
            }
            const argDelta = fn != null && typeof fn.arguments === "string" ? fn.arguments : "";
            if (argDelta !== "") {
              yield {
                type: "tool_use_delta",
                index: eventIndex,
                partial_json: argDelta,
              };
            }
          }
        }
      }
    }
    // If the stream ended without an explicit [DONE] marker we still emit
    // MessageStop so consumers see a terminator.
    yield { type: "message_stop", usage, stop_reason: stopReason };
  } catch (e) {
    // SC-3: a chunk read/decode failure while draining the streaming body is a
    // mid-flight interruption — surface the typed, retryable StreamInterrupted
    // variant rather than letting an untyped reader rejection escape.
    throw new StreamInterrupted(`stream chunk error: ${formatError(e)}`);
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
