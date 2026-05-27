/**
 * Issue #41 — `OllamaModelInterface`: real Ollama HTTP client.
 *
 * Implements {@link ModelInterface} against a local Ollama server's
 * `/api/chat`, `/api/tags`, and `/api/embed` endpoints. Translates
 * {@link ModelRequest} / {@link ModelResponse} to and from the Ollama wire
 * format, parses Ollama's NDJSON stream (one JSON object per line — not SSE)
 * for {@link OllamaModelInterface.callStreaming}, and maps HTTP/transport
 * errors to typed {@link ModelError} variants. Unlike the Anthropic and OpenAI
 * clients there is **no retry**: spec says fail fast on connection errors with
 * a helpful message ("Ollama not running", "Run: ollama pull <model>").
 *
 * Mirrors `rust/crates/spore-core/src/ollama.rs` rule-for-rule.
 *
 * ## Provider-specific shape
 * - No API key; default `baseUrl` is `http://localhost:11434`.
 * - Sampling parameters (`num_predict`, `temperature`, `top_p`, `stop`) are
 *   nested under `options` rather than top-level keys.
 * - `keepAlive` (default `"5m"`) controls how long Ollama keeps the model
 *   loaded after the call returns.
 * - Tool-call arguments are a JSON **object** in the wire format, not a
 *   JSON-encoded string like OpenAI.
 * - Ollama does not return tool-call ids; we synthesize `call-{i}` per index
 *   so downstream tool-result round-trips work.
 * - Thinking blocks are silently omitted from outgoing requests — Ollama
 *   has no structured reasoning shape.
 *
 * Cache-cost wiring: Ollama has no prefix caching;
 * {@link "../cache-provider/types.js".OllamaCacheProvider} is a no-op.
 */

import { ProviderError, Timeout, type ModelError } from "./errors.js";
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

export interface OllamaModelInterfaceOptions {
  /** Override the base URL. Defaults to `http://localhost:11434`. */
  baseUrl?: string;
  /** Request timeout in milliseconds. Defaults to 300_000 (300s). */
  timeoutMs?: number;
  /** How long Ollama should keep the model loaded. Defaults to "5m". */
  keepAlive?: string | null;
  /** Injected fetch implementation. Defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

// ============================================================================
// Constants
// ============================================================================

export const DEFAULT_BASE_URL = "http://localhost:11434";
export const DEFAULT_TIMEOUT_MS = 300_000;
export const DEFAULT_KEEP_ALIVE = "5m";

// ============================================================================
// Discovery metadata
// ============================================================================

/**
 * `/api/show`-discovered metadata for the model. Populated once, alongside the
 * `/api/tags` availability check. All fields are best-effort — `/api/show`
 * failures leave them unset rather than failing the call.
 */
interface ModelMeta {
  /** Discovered context window (`*.context_length` in `model_info`). */
  contextLength?: number;
  /** Top-level `capabilities` array (may contain `"tools"`). */
  capabilities: string[];
}

// ============================================================================
// OllamaModelInterface
// ============================================================================

export class OllamaModelInterface implements ModelInterface {
  private readonly modelId: string;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly keepAlive: string | null;
  private readonly fetchImpl: typeof fetch;
  /** Lazy availability + discovery check — populated after first successful probe. */
  private modelCheckPromise: Promise<ModelMeta> | null = null;
  /**
   * The `/api/show`-discovered metadata, set synchronously once the probe
   * resolves. Read non-blockingly by {@link provider}. `null` until the first
   * successful probe completes.
   */
  private discoveredMeta: ModelMeta | null = null;

  constructor(modelId: string, options: OllamaModelInterfaceOptions = {}) {
    this.modelId = modelId;
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.keepAlive = options.keepAlive === undefined ? DEFAULT_KEEP_ALIVE : options.keepAlive;
    this.fetchImpl = options.fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  /** Convenience constructor mirroring the Rust `with_base_url`. */
  static withBaseUrl(modelId: string, baseUrl: string): OllamaModelInterface {
    return new OllamaModelInterface(modelId, { baseUrl });
  }

  /**
   * Context window for known Ollama model id prefixes. Returns 0 for unknown
   * ids so callers can detect "unknown model" rather than silently getting a
   * plausible-but-wrong value.
   */
  static contextWindow(modelId: string): number {
    if (modelId.startsWith("llama3.2")) return 128_000;
    if (modelId.startsWith("qwen2.5-coder")) return 128_000;
    if (modelId.startsWith("mistral")) return 32_000;
    if (modelId.startsWith("gemma")) return 8_192;
    return 0;
  }

  /**
   * Best-effort capability table — `llama3.2` / `qwen2.5-coder` / `mistral`
   * are known to support native tool calling; others may or may not. Used as a
   * fallback when `/api/show` discovery is unavailable.
   */
  static supportsTools(modelId: string): boolean {
    return (
      modelId.startsWith("llama3.2") ||
      modelId.startsWith("qwen2.5-coder") ||
      modelId.startsWith("mistral")
    );
  }

  provider(): ProviderInfo {
    // `provider()` is synchronous so it cannot await `/api/show`. Read the probe
    // cache non-blockingly: prefer a discovered context length if the probe has
    // already run; otherwise fall back to the static table.
    const discovered = this.discoveredMeta?.contextLength;
    return {
      name: "ollama",
      model_id: this.modelId,
      context_window: discovered ?? OllamaModelInterface.contextWindow(this.modelId),
    };
  }

  toJSON(): Record<string, unknown> {
    return {
      model_id: this.modelId,
      base_url: this.baseUrl,
      timeout_ms: this.timeoutMs,
      keep_alive: this.keepAlive,
    };
  }

  async call(request: ModelRequest, signal?: AbortSignal): Promise<ModelResponse> {
    const meta = await this.ensureModelAvailable(signal);
    this.guardToolSupport(request, meta);
    const body = JSON.stringify(buildRequest(this.modelId, this.keepAlive, request, false));
    const url = `${this.baseUrl}/api/chat`;
    let resp: Response;
    try {
      resp = await this.fetchOnce(url, body, signal);
    } catch (e) {
      throw this.toTransportError(e);
    }
    if (!resp.ok) {
      throw await mapStatusError(resp, this.modelId);
    }
    let parsed: OllamaResponse;
    try {
      parsed = (await resp.json()) as OllamaResponse;
    } catch (e) {
      throw new ProviderError(0, `response decode failed: ${formatError(e)}`);
    }
    return parseResponse(parsed);
  }

  async *callStreaming(request: ModelRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const meta = await this.ensureModelAvailable(signal);
    this.guardToolSupport(request, meta);
    const body = JSON.stringify(buildRequest(this.modelId, this.keepAlive, request, true));
    const url = `${this.baseUrl}/api/chat`;
    let resp: Response;
    try {
      resp = await this.fetchOnce(url, body, signal);
    } catch (e) {
      throw this.toTransportError(e);
    }
    if (!resp.ok) {
      throw await mapStatusError(resp, this.modelId);
    }
    if (resp.body == null) {
      throw new ProviderError(0, "streaming response had no body");
    }
    yield* ndjsonToEvents(resp.body);
  }

  async countTokens(request: ModelRequest, signal?: AbortSignal): Promise<number> {
    // Try the embed endpoint; fall back to bytes/4 heuristic on missing field
    // or any transport failure. Matches `openai.rs` fallback strategy.
    const text = concatRequestText(request);
    const n = await this.tryEmbedCount(text, signal);
    if (n != null) return n;
    return Math.floor(text.length / 4);
  }

  // ── helpers ──────────────────────────────────────────────────────────────

  private async ensureModelAvailable(signal?: AbortSignal): Promise<ModelMeta> {
    if (this.modelCheckPromise != null) {
      return this.modelCheckPromise;
    }
    const promise = this.probeModel(signal);
    // Cache only on success — failures should be retryable.
    this.modelCheckPromise = promise.catch((e) => {
      this.modelCheckPromise = null;
      throw e;
    });
    return this.modelCheckPromise;
  }

  /**
   * One-time availability + discovery probe. Checks `/api/tags` (surfacing a
   * helpful "ollama pull" message when the model is missing), then —
   * best-effort — fetches `/api/show` for the context window and capabilities.
   * Resolves to the discovered {@link ModelMeta} (empty when `/api/show` was
   * unavailable). `/api/show` failures never fail the probe.
   */
  private async probeModel(signal?: AbortSignal): Promise<ModelMeta> {
    const url = `${this.baseUrl}/api/tags`;
    let resp: Response;
    try {
      resp = await this.fetchGet(url, signal);
    } catch (e) {
      throw this.toTransportError(e);
    }
    if (!resp.ok) {
      throw await mapStatusError(resp, this.modelId);
    }
    let body: TagsResponse;
    try {
      body = (await resp.json()) as TagsResponse;
    } catch (e) {
      throw new ProviderError(0, `tags decode failed: ${formatError(e)}`);
    }
    const found = (body.models ?? []).some((m) => nameMatches(m.name ?? "", this.modelId));
    if (!found) {
      throw new ProviderError(
        404,
        `Model ${this.modelId} not found. Run: ollama pull ${this.modelId}`,
      );
    }
    // Best-effort discovery — never fails the call.
    const meta = await this.discoverMeta(signal);
    this.discoveredMeta = meta;
    return meta;
  }

  /**
   * Best-effort `POST /api/show` discovery. Resolves to an empty
   * {@link ModelMeta} on any failure (404, transport error, decode error,
   * missing fields) so discovery being unavailable never errors the whole call.
   */
  private async discoverMeta(signal?: AbortSignal): Promise<ModelMeta> {
    const url = `${this.baseUrl}/api/show`;
    const body = JSON.stringify({ model: this.modelId });
    let resp: Response;
    try {
      resp = await this.fetchOnce(url, body, signal);
    } catch {
      return { capabilities: [] };
    }
    if (!resp.ok) {
      try {
        await resp.text();
      } catch {
        /* ignore */
      }
      return { capabilities: [] };
    }
    let parsed: ShowResponse;
    try {
      parsed = (await resp.json()) as ShowResponse;
    } catch {
      return { capabilities: [] };
    }
    const modelInfo = parsed.model_info ?? {};
    let contextLength: number | undefined;
    for (const [k, v] of Object.entries(modelInfo)) {
      if (k.endsWith(".context_length") && typeof v === "number") {
        contextLength = v;
        break;
      }
    }
    const capabilities = Array.isArray(parsed.capabilities)
      ? parsed.capabilities.filter((c): c is string => typeof c === "string")
      : [];
    return { contextLength, capabilities };
  }

  /**
   * Reject tool-bearing requests when the model does not support tools.
   * Capability source priority: the `/api/show` `capabilities` array when
   * discovery succeeded; otherwise the static {@link OllamaModelInterface.supportsTools}
   * table.
   */
  private guardToolSupport(request: ModelRequest, meta: ModelMeta): void {
    if (request.tools.length === 0) return;
    const supported =
      meta.capabilities.length === 0
        ? OllamaModelInterface.supportsTools(this.modelId)
        : meta.capabilities.includes("tools");
    if (!supported) {
      throw new ProviderError(0, `Model ${this.modelId} does not support tool calling`);
    }
  }

  private async tryEmbedCount(text: string, signal?: AbortSignal): Promise<number | null> {
    const url = `${this.baseUrl}/api/embed`;
    const body = JSON.stringify({ model: this.modelId, input: text });
    let resp: Response;
    try {
      resp = await this.fetchOnce(url, body, signal);
    } catch {
      return null;
    }
    if (!resp.ok) {
      try {
        await resp.text();
      } catch {
        /* ignore */
      }
      return null;
    }
    try {
      const parsed = (await resp.json()) as EmbedResponse;
      if (typeof parsed.prompt_eval_count === "number") {
        return parsed.prompt_eval_count;
      }
      return null;
    } catch {
      return null;
    }
  }

  private async fetchOnce(
    url: string,
    body: string,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    const { controller, cleanup } = this.makeAbortController(signal);
    try {
      return await this.fetchImpl(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        signal: controller.signal,
      });
    } finally {
      cleanup();
    }
  }

  private async fetchGet(url: string, signal: AbortSignal | undefined): Promise<Response> {
    const { controller, cleanup } = this.makeAbortController(signal);
    try {
      return await this.fetchImpl(url, { method: "GET", signal: controller.signal });
    } finally {
      cleanup();
    }
  }

  private makeAbortController(signal: AbortSignal | undefined): {
    controller: AbortController;
    cleanup: () => void;
  } {
    const controller = new AbortController();
    const timeoutHandle = setTimeout(() => controller.abort(new TimeoutSentinel()), this.timeoutMs);
    const onUserAbort = () => controller.abort(signal?.reason);
    if (signal != null) {
      if (signal.aborted) controller.abort(signal.reason);
      else signal.addEventListener("abort", onUserAbort, { once: true });
    }
    return {
      controller,
      cleanup: () => {
        clearTimeout(timeoutHandle);
        if (signal != null) signal.removeEventListener("abort", onUserAbort);
      },
    };
  }

  private toTransportError(e: unknown): ModelError {
    if (e instanceof ProviderError || e instanceof Timeout) return e;
    if (e instanceof Error) {
      if (e.name === "TimeoutSentinel") return new Timeout();
      if (e.name === "AbortError") {
        const cause = (e as { cause?: unknown }).cause;
        if (cause instanceof TimeoutSentinel) return new Timeout();
        return new ProviderError(0, `request aborted: ${e.message}`);
      }
      // Any other transport-level failure on the chat endpoint is treated as
      // "Ollama not running" — matches the Rust fail-fast behavior.
      return new ProviderError(0, `Ollama not running at ${this.baseUrl}`);
    }
    return new ProviderError(0, `Ollama not running at ${this.baseUrl}`);
  }
}

// ============================================================================
// Name-matching for /api/tags
// ============================================================================

/**
 * Ollama tag names often look like `"llama3.2:latest"` or `"llama3.2:3b"`.
 * Match if the request id equals the full tag or its bare-name prefix.
 */
export function nameMatches(tag: string, requested: string): boolean {
  if (tag === requested) return true;
  const colonIdx = tag.indexOf(":");
  const bare = colonIdx === -1 ? tag : tag.slice(0, colonIdx);
  return bare === requested;
}

// ============================================================================
// Wire-format types (Ollama Chat API)
// ============================================================================

export interface OllamaRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  keep_alive?: string;
  options?: OllamaOptions;
  tools?: OllamaTool[];
}

export interface OllamaOptions {
  num_predict?: number;
  temperature?: number;
  top_p?: number;
  stop?: string[];
}

export interface OllamaMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: OllamaToolCall[];
  /** Used by tool-result messages — mirrors OpenAI's `tool_call_id`. */
  tool_call_id?: string;
}

export interface OllamaToolCall {
  function: OllamaFunctionCall;
}

export interface OllamaFunctionCall {
  name: string;
  /** Object, NOT a JSON-encoded string (differs from OpenAI). */
  arguments: unknown;
}

export interface OllamaTool {
  type: "function";
  function: { name: string; description: string; parameters: unknown };
}

interface OllamaResponse {
  message?: OllamaResponseMessage;
  done?: boolean;
  done_reason?: string | null;
  prompt_eval_count?: number;
  eval_count?: number;
}

interface OllamaResponseMessage {
  role?: string;
  content?: string | null;
  tool_calls?: OllamaResponseToolCall[];
}

interface OllamaResponseToolCall {
  function?: { name?: string; arguments?: unknown };
}

interface TagsResponse {
  models?: { name?: string }[];
}

interface EmbedResponse {
  prompt_eval_count?: number;
}

interface ShowResponse {
  /** Map of architecture-specific keys; we look for `*.context_length`. */
  model_info?: Record<string, unknown>;
  /** Top-level capabilities array (may contain `"tools"`). */
  capabilities?: unknown[];
}

// ============================================================================
// Conversions (exported for tests)
// ============================================================================

export function buildRequest(
  modelId: string,
  keepAlive: string | null,
  req: ModelRequest,
  stream: boolean,
): OllamaRequest {
  const messages: OllamaMessage[] = req.messages.map(messageToOllama);

  const tools: OllamaTool[] = req.tools.map((t: ToolSchema) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
    },
  }));

  const options: OllamaOptions = {};
  if (req.params.max_tokens != null) options.num_predict = req.params.max_tokens;
  if (req.params.temperature != null) options.temperature = req.params.temperature;
  if (req.params.top_p != null) options.top_p = req.params.top_p;
  if (req.params.stop_sequences.length > 0) options.stop = [...req.params.stop_sequences];

  const out: OllamaRequest = {
    model: modelId,
    messages,
    stream,
  };
  if (keepAlive != null) out.keep_alive = keepAlive;
  if (
    options.num_predict != null ||
    options.temperature != null ||
    options.top_p != null ||
    (options.stop != null && options.stop.length > 0)
  ) {
    out.options = options;
  }
  if (tools.length > 0) out.tools = tools;
  return out;
}

function messageToOllama(m: Message): OllamaMessage {
  const role: OllamaMessage["role"] =
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
        content: "",
        tool_calls: [
          {
            function: {
              name: c.name,
              // Object — NOT JSON-encoded string.
              arguments: c.input ?? {},
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
      // Ollama supports images via a separate `images` field on user messages
      // (base64). The harness does not currently emit image content into
      // requests; emit a placeholder rather than introduce a heterogeneous
      // shape.
      return { role, content: `[image: ${c.media_type}]` };
    default: {
      const _exhaustive: never = c;
      return _exhaustive;
    }
  }
}

export function parseResponse(body: OllamaResponse): ModelResponse {
  const msg: OllamaResponseMessage = body.message ?? {};
  const content: ContentBlock[] = [];
  const text = msg.content;
  if (text != null && text !== "") {
    content.push({ type: "text", text });
  }
  const toolCalls = msg.tool_calls ?? [];
  for (let i = 0; i < toolCalls.length; i += 1) {
    const tc = toolCalls[i]!;
    const args = tc.function?.arguments;
    const input = args == null ? {} : args;
    content.push({
      type: "tool_use",
      id: `call-${i}`,
      name: tc.function?.name ?? "",
      input,
    });
  }

  const usage: TokenUsage = {
    input_tokens: body.prompt_eval_count ?? 0,
    output_tokens: body.eval_count ?? 0,
    cache_read_tokens: null,
    cache_write_tokens: null,
  };

  return {
    content,
    usage,
    stop_reason: parseStopReason(body.done_reason ?? null),
  };
}

export function parseStopReason(s: string | null | undefined): StopReason {
  switch (s) {
    case "tool_calls":
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

async function mapStatusError(resp: Response, modelId: string): Promise<ModelError> {
  const code = resp.status;
  let bodyText = "";
  try {
    bodyText = await resp.text();
  } catch {
    /* ignore */
  }
  if (code === 404) {
    const lower = bodyText.toLowerCase();
    if (lower.includes("not found") || lower.includes("model") || bodyText === "") {
      return new ProviderError(404, `Model ${modelId} not found. Run: ollama pull ${modelId}`);
    }
  }
  if (code === 408 || code === 504) return new Timeout();
  const message = bodyText === "" ? `HTTP ${code}` : bodyText.slice(0, 500);
  return new ProviderError(code, message);
}

class TimeoutSentinel extends Error {
  constructor() {
    super("ollama-timeout");
    this.name = "TimeoutSentinel";
  }
}

function formatError(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

// ============================================================================
// count_tokens helpers
// ============================================================================

function concatRequestText(request: ModelRequest): string {
  let out = "";
  for (const m of request.messages) {
    const c = m.content;
    switch (c.type) {
      case "text":
        out += c.text;
        break;
      case "tool_call":
        out += c.name;
        out += " ";
        out += JSON.stringify(c.input ?? {});
        break;
      case "tool_result":
        out += c.content;
        break;
      case "image":
        break;
    }
    out += "\n";
  }
  return out;
}

// ============================================================================
// NDJSON stream parsing — Ollama chat streaming
// ============================================================================

/**
 * Convert an Ollama NDJSON response body into a stream of `StreamEvent`s.
 *
 * Ollama streams chat results as **newline-delimited JSON** (one full JSON
 * object per line, NOT SSE). Each line carries an incremental
 * `message.content` delta; `tool_calls` arrive as full argument objects per
 * chunk (not partial-fragment strings like OpenAI); the terminator line
 * carries `done: true` plus `prompt_eval_count` and `eval_count`.
 *
 * Exported for tests so unit tests can drive the parser without HTTP.
 */
export async function* ndjsonToEvents(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let started = false;
  const toolIndicesSeen = new Set<number>();
  let contentIndex = 0;
  let contentOpen = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        const rawLine = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        const line = rawLine.replace(/\r$/, "").trim();
        if (line === "") continue;
        let value: unknown;
        try {
          value = JSON.parse(line);
        } catch {
          continue;
        }
        const obj = jsonValue(value);
        if (obj == null) continue;
        if (!started) {
          started = true;
          yield { type: "message_start" };
        }
        const message = jsonValue(obj.message);
        if (message != null) {
          if (typeof message.content === "string" && message.content !== "") {
            contentOpen = true;
            yield {
              type: "content_block_delta",
              index: contentIndex,
              delta: message.content,
            };
          }
          const tcs = Array.isArray(message.tool_calls) ? message.tool_calls : null;
          if (tcs != null) {
            for (let i = 0; i < tcs.length; i += 1) {
              const tc = jsonValue(tcs[i]);
              if (tc == null) continue;
              const eventIndex = i + 1;
              if (!toolIndicesSeen.has(eventIndex)) {
                toolIndicesSeen.add(eventIndex);
                if (contentOpen) {
                  yield { type: "content_block_stop", index: contentIndex };
                  contentOpen = false;
                  contentIndex = eventIndex;
                }
              }
              const fn = jsonValue(tc.function);
              if (fn != null && "arguments" in fn) {
                const args = fn.arguments;
                let partial: string;
                try {
                  partial = JSON.stringify(args ?? {});
                } catch {
                  partial = "{}";
                }
                yield {
                  type: "tool_use_delta",
                  index: eventIndex,
                  partial_json: partial,
                };
              }
            }
          }
        }
        if (obj.done === true) {
          const usage: TokenUsage = {
            input_tokens: typeof obj.prompt_eval_count === "number" ? obj.prompt_eval_count : 0,
            output_tokens: typeof obj.eval_count === "number" ? obj.eval_count : 0,
            cache_read_tokens: null,
            cache_write_tokens: null,
          };
          const stopReason = parseStopReason(
            typeof obj.done_reason === "string" ? obj.done_reason : null,
          );
          yield { type: "message_stop", usage, stop_reason: stopReason };
          return;
        }
      }
    }
    // Defensive: if the connection drops without `done:true`, still emit a
    // MessageStop so consumers see a terminator.
    yield {
      type: "message_stop",
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: null,
        cache_write_tokens: null,
      },
      stop_reason: "end_turn",
    };
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
