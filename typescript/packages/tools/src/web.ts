/**
 * Web tools (#81, net-new Tier-1 tools): `web_fetch`, `web_search`.
 *
 * Both use Node's global `fetch`. They are `open_world` (external effects) and
 * `read_only`.
 *
 * - {@link WebFetchTool} (`web_fetch`) — GET a URL, return the body text.
 * - {@link WebSearchTool} (`web_search`) — POST the query to a configurable
 *   search endpoint and return the structured response body verbatim. There is
 *   no live web-search backend in spore-core; the endpoint is injected at
 *   construction so tests drive it against a mock HTTP server (NEVER the live
 *   network). The default endpoint is absent, which yields a recoverable error
 *   until a real backend is configured.
 *
 * ## `web_search` configurability (#108)
 *
 * {@link WebSearchTool.withConfig} accepts a {@link WebSearchConfig} so the tool
 * can talk to real search APIs (Brave, Tavily, …) that need GET-style param
 * encoding and/or auth secrets. The original `new WebSearchTool(endpoint?)` and
 * {@link WebSearchTool.withEndpoint} constructors and their behavior are
 * **frozen and unchanged**: they still POST `{ "query": <q> }` as a JSON body
 * and return the response body verbatim.
 *
 * Env sourcing mirrors `AnthropicModelInterface.fromEnv`: the caller supplies
 * the env-var NAME; the value is resolved from `process.env` at CONSTRUCTION
 * time; an unset OR empty env var is a construction error
 * ({@link WebSearchConfigError}) — a request is NEVER sent with a missing or
 * empty secret. Resolved secret values are held only in a private field and are
 * never stored on the serializable {@link WebSearchConfig}.
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  parseParams,
  WebFetchParamsSchema,
  WebSearchParamsSchema,
} from "./params.js";
import { finishWithPossibleTruncation } from "./sandbox-defaults.js";

// ============================================================================
// WebFetch
// ============================================================================

/**
 * Apply `start_byte` slicing to a fetched response body.
 *
 * - `start_byte === 0` → return `body` unchanged (no header).
 * - `0 < start_byte < byteLength` → prepend `[starting at byte N of total]\n`
 *   and return the slice from `start_byte`.
 * - `start_byte >= byteLength` (for non-empty bodies) → recoverable error.
 * - Empty body + `start_byte > 0` → recoverable error.
 *
 * Works in UTF-8 bytes (via `Buffer`) to match Rust's byte semantics.
 */
export function applyWebFetchRange(
  body: string,
  startByte: number,
): { ok: true; value: string } | { ok: false; error: ToolOutput } {
  if (startByte === 0) {
    return { ok: true, value: body };
  }
  const buf = Buffer.from(body, "utf8");
  const total = buf.length;
  if (startByte >= total) {
    return {
      ok: false,
      error: {
        kind: "error",
        message: `start_byte ${startByte} exceeds response length ${total}`,
        recoverable: true,
      },
    };
  }
  const slice = buf.subarray(startByte).toString("utf8");
  return {
    ok: true,
    value: `[starting at byte ${startByte} of ${total}]\n${slice}`,
  };
}

export class WebFetchTool implements Tool {
  static readonly NAME = "web_fetch";
  readonly name = WebFetchTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: WebFetchTool.NAME,
      description: "Fetch the contents of a URL",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string" },
          start_byte: {
            type: "integer",
            description:
              "Byte offset into the response body to start reading from. Default 0 (no offset, output identical to a plain fetch). Use to page through responses larger than the 64 KB truncation window.",
            default: 0,
          },
        },
        required: ["url"],
      },
      annotations: {
        read_only: true,
        destructive: false,
        idempotent: false,
        open_world: true,
      },
    };
  }

  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    _ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(WebFetchParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    try {
      const resp = await fetch(p.value.url, { method: "GET", signal });
      const body = await resp.text();
      const ranged = applyWebFetchRange(body, p.value.start_byte ?? 0);
      if (!ranged.ok) return ranged.error;
      return finishWithPossibleTruncation(ranged.value, call.id, sandbox);
    } catch (e) {
      return {
        kind: "error",
        message: `web fetch failed: ${e instanceof Error ? e.message : String(e)}`,
        recoverable: true,
      };
    }
  }
}

// ============================================================================
// WebSearch
// ============================================================================

/**
 * HTTP method used to dispatch a web search. A string union (NOT a boolean) so
 * the wire shape is explicit and extensible. Default is `"POST"`.
 */
export type SearchMethod = "GET" | "POST";

/**
 * Construction-time configuration for {@link WebSearchTool.withConfig}.
 *
 * Env-var NAMES (not values) are stored here; values are resolved when
 * `withConfig` is called. Resolved secrets are never written back onto a config
 * object, so a `WebSearchConfig` stays safe to serialize.
 *
 * Fields:
 * - `endpoint` — the search backend URL.
 * - `method` — `"GET"` or `"POST"` (default `"POST"`).
 * - `queryParam` — field/param name the query is keyed under (default
 *   `"query"`; Brave uses `"q"`).
 * - `authHeaders` — `[headerName, envVar]` pairs. Each env var is resolved at
 *   construction time and attached as an HTTP header (by `headerName`) on every
 *   request, for both GET and POST.
 * - `bodyAuthParams` — `[fieldName, envVar]` pairs. Each env var is resolved at
 *   construction time and injected as a secret request parameter. For POST it
 *   goes into the JSON body alongside the query (e.g. Tavily's
 *   `{ "api_key": ..., "query": ... }`). For GET it is appended to the URL
 *   query string alongside the query param.
 */
export interface WebSearchConfig {
  endpoint: string;
  method?: SearchMethod;
  queryParam?: string;
  authHeaders?: ReadonlyArray<readonly [headerName: string, envVar: string]>;
  bodyAuthParams?: ReadonlyArray<readonly [fieldName: string, envVar: string]>;
}

/**
 * Error thrown by {@link WebSearchTool.withConfig} when a referenced env var is
 * unset or empty. Mirrors the `fromEnv` precedent: a request is never sent with
 * a missing/empty secret. `kind` discriminates the two cases for exhaustive
 * `switch`.
 */
export class WebSearchConfigError extends Error {
  override readonly name = "WebSearchConfigError";
  readonly kind: "env_var_not_set" | "env_var_empty";
  readonly envVar: string;

  private constructor(
    kind: "env_var_not_set" | "env_var_empty",
    envVar: string,
    message: string,
  ) {
    super(message);
    this.kind = kind;
    this.envVar = envVar;
  }

  static envVarNotSet(envVar: string): WebSearchConfigError {
    return new WebSearchConfigError(
      "env_var_not_set",
      envVar,
      `env var \`${envVar}\` not set`,
    );
  }

  static envVarEmpty(envVar: string): WebSearchConfigError {
    return new WebSearchConfigError(
      "env_var_empty",
      envVar,
      `env var \`${envVar}\` is empty`,
    );
  }
}

/**
 * Resolved backend: env-var names have been replaced with their secret values.
 * Private to this module — never serialized — so secrets do not leak.
 */
interface ResolvedBackend {
  endpoint: string;
  method: SearchMethod;
  queryParam: string;
  authHeaders: ReadonlyArray<readonly [name: string, value: string]>;
  bodyAuthParams: ReadonlyArray<readonly [field: string, value: string]>;
}

/** Resolve an env var by NAME at construction time. Unset or empty → throw. */
function resolveEnv(envVar: string): string {
  const value = process.env[envVar];
  if (value == null) {
    throw WebSearchConfigError.envVarNotSet(envVar);
  }
  if (value.trim() === "") {
    throw WebSearchConfigError.envVarEmpty(envVar);
  }
  return value;
}

/**
 * Web search tool. The search backend endpoint is injected so tests run against
 * a mock HTTP server. With no endpoint configured (the default), every call is
 * a recoverable error.
 */
export class WebSearchTool implements Tool {
  static readonly NAME = "web_search";
  readonly name = WebSearchTool.NAME;

  private readonly backend: ResolvedBackend | null;

  /** Construct with no backend configured (calls error until one is set).
   *
   *  Passing an `endpoint` is the FROZEN convenience path: POST `{ "query": q }`
   *  as JSON, no auth, default `"query"` param. {@link withConfig} builds the
   *  configurable backend directly via the private overload. */
  constructor(endpoint?: string | null);
  constructor(backend: ResolvedBackend | null, fromConfig: true);
  constructor(arg?: string | ResolvedBackend | null, fromConfig?: true) {
    if (fromConfig === true) {
      this.backend = (arg as ResolvedBackend | null) ?? null;
      return;
    }
    const endpoint = (arg as string | null | undefined) ?? null;
    this.backend =
      endpoint == null
        ? null
        : {
            endpoint,
            method: "POST",
            queryParam: "query",
            authHeaders: [],
            bodyAuthParams: [],
          };
  }

  /** Construct with a search endpoint (the query is POSTed as JSON
   *  `{ "query": ... }`; the response body is returned verbatim).
   *
   *  FROZEN behavior — kept compatible with the original tool. */
  static withEndpoint(endpoint: string): WebSearchTool {
    return new WebSearchTool(endpoint);
  }

  /**
   * Construct from a {@link WebSearchConfig}, resolving every referenced env var
   * at construction time. Throws {@link WebSearchConfigError} if any auth env
   * var is unset or empty — no request is ever attempted in that case.
   */
  static withConfig(config: WebSearchConfig): WebSearchTool {
    const authHeaders = (config.authHeaders ?? []).map(
      ([name, envVar]) => [name, resolveEnv(envVar)] as const,
    );
    const bodyAuthParams = (config.bodyAuthParams ?? []).map(
      ([field, envVar]) => [field, resolveEnv(envVar)] as const,
    );
    return new WebSearchTool(
      {
        endpoint: config.endpoint,
        method: config.method ?? "POST",
        queryParam: config.queryParam ?? "query",
        authHeaders,
        bodyAuthParams,
      },
      true,
    );
  }

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: WebSearchTool.NAME,
      description: "Search the web and return structured results",
      parameters: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
      annotations: {
        read_only: true,
        destructive: false,
        idempotent: false,
        open_world: true,
      },
    };
  }

  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    _ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(WebSearchParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const backend = this.backend;
    if (backend == null) {
      return {
        kind: "error",
        message: "web_search backend not configured",
        recoverable: true,
      };
    }
    try {
      const headers: Record<string, string> = {};
      for (const [name, value] of backend.authHeaders) {
        headers[name] = value;
      }
      let url = backend.endpoint;
      let init: RequestInit & { signal?: AbortSignal };
      if (backend.method === "GET") {
        // Query + body-auth params are URL-encoded into the query string;
        // `URLSearchParams` encodes spaces, `&`, etc. Any query string already
        // present on the endpoint (e.g. `?format=json`) is PRESERVED — the new
        // params are appended, not clobbered (matches Rust/Python/Go).
        const parsed = new URL(backend.endpoint);
        parsed.searchParams.set(backend.queryParam, p.value.query);
        for (const [field, value] of backend.bodyAuthParams) {
          parsed.searchParams.set(field, value);
        }
        url = parsed.toString();
        init = { method: "GET", headers, signal };
      } else {
        // Query + body-auth params go into the JSON body (Tavily shape:
        // { "api_key": ..., "query": ... }).
        const body: Record<string, string> = {
          [backend.queryParam]: p.value.query,
        };
        for (const [field, value] of backend.bodyAuthParams) {
          body[field] = value;
        }
        init = {
          method: "POST",
          headers: { "content-type": "application/json", ...headers },
          body: JSON.stringify(body),
          signal,
        };
      }
      const resp = await fetch(url, init);
      const respBody = await resp.text();
      return finishWithPossibleTruncation(respBody, call.id, sandbox);
    } catch (e) {
      return {
        kind: "error",
        message: `web search failed: ${e instanceof Error ? e.message : String(e)}`,
        recoverable: true,
      };
    }
  }
}
