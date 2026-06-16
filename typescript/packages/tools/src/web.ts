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

import { isIP } from "node:net";

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
// SSRF guard — outbound URL policy (validateFetchUrl seam)
// ============================================================================

/**
 * Policy consulted before any outbound `web_fetch` / `web_search` request.
 *
 * **Permissive by default** — {@link UrlPolicy.permissive} allows every URL, so
 * existing wiring, tests, and examples are unaffected. A deployment that wants
 * SSRF protection opts in with {@link UrlPolicy.denyPrivate} (wired via the
 * web-tool builders {@link WebFetchTool.withUrlPolicy} /
 * {@link WebSearchTool.withUrlPolicy}).
 *
 * `denyPrivate` rejects non-`http(s)` schemes and hosts given as an IP literal
 * in a loopback / link-local / private (RFC-1918) / cloud-metadata
 * (`169.254.169.254`) range, plus the `localhost` hostname. **Limitation:**
 * non-`localhost` hostnames are NOT DNS-resolved here, so a hostname that
 * resolves to a private address is not caught by this seam — deployments that
 * need resolution-time protection must enforce it at the network layer.
 */
export class UrlPolicy {
  private constructor(readonly denyPrivate: boolean) {}

  /** Allow every URL (the default). No SSRF restrictions. */
  static permissive(): UrlPolicy {
    return new UrlPolicy(false);
  }

  /**
   * Reject non-`http(s)` schemes and private/loopback/link-local/metadata
   * IP-literal hosts (and the `localhost` hostname).
   */
  static denyPrivate(): UrlPolicy {
    return new UrlPolicy(true);
  }
}

function urlDenied(message: string): ToolOutput {
  return { kind: "error", message, recoverable: true };
}

/** Parse the four octets of an IPv4 literal (already validated by `isIP`). */
function parseV4(host: string): [number, number, number, number] {
  const parts = host.split(".").map((p) => Number.parseInt(p, 10));
  return [parts[0]!, parts[1]!, parts[2]!, parts[3]!];
}

function isBlockedV4(host: string): boolean {
  const [a, b, c, d] = parseV4(host);
  // Loopback 127.0.0.0/8.
  if (a === 127) return true;
  // Private RFC-1918: 10/8, 172.16/12, 192.168/16.
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  // Link-local 169.254.0.0/16 (includes cloud metadata 169.254.169.254).
  if (a === 169 && b === 254) return true;
  // Unspecified 0.0.0.0 and broadcast 255.255.255.255.
  if (a === 0 && b === 0 && c === 0 && d === 0) return true;
  if (a === 255 && b === 255 && c === 255 && d === 255) return true;
  // Documentation ranges: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24.
  if (a === 192 && b === 0 && c === 2) return true;
  if (a === 198 && b === 51 && c === 100) return true;
  if (a === 203 && b === 0 && c === 113) return true;
  return false;
}

/**
 * Expand an IPv6 literal to its 8 16-bit groups, handling `::` compression.
 * `host` is assumed to be a valid IPv6 literal (already vetted by `isIP`).
 */
function expandV6Groups(host: string): number[] {
  // An embedded IPv4 tail (e.g. `::ffff:1.2.3.4`) — fold it into two groups.
  let head = host;
  let tail: number[] = [];
  const lastColon = head.lastIndexOf(":");
  const maybeV4 = head.slice(lastColon + 1);
  if (maybeV4.includes(".")) {
    const [a, b, c, d] = parseV4(maybeV4);
    tail = [(a << 8) | b, (c << 8) | d];
    head = head.slice(0, lastColon + 1);
  }
  const [left, right] = head.includes("::")
    ? head.split("::")
    : [head, undefined];
  const leftGroups = left ? left.split(":").filter((s) => s !== "") : [];
  const rightGroups =
    right !== undefined ? right.split(":").filter((s) => s !== "") : [];
  const leftParsed = leftGroups.map((g) => Number.parseInt(g, 16));
  const rightParsed = rightGroups.map((g) => Number.parseInt(g, 16));
  const known = leftParsed.length + rightParsed.length + tail.length;
  // `::` expands to however many all-zero groups are needed to reach 8.
  const fill = right !== undefined ? new Array<number>(8 - known).fill(0) : [];
  // Reassemble in order: left, gap-fill (for `::`), right, embedded-v4 tail.
  return [...leftParsed, ...fill, ...rightParsed, ...tail];
}

function isBlockedV6(host: string): boolean {
  const g = expandV6Groups(host);
  // Loopback ::1.
  if (g.every((x, i) => (i < 7 ? x === 0 : x === 1))) return true;
  // Unspecified ::.
  if (g.every((x) => x === 0)) return true;
  // Link-local fe80::/10.
  if ((g[0]! & 0xffc0) === 0xfe80) return true;
  // Unique-local fc00::/7.
  if ((g[0]! & 0xfe00) === 0xfc00) return true;
  // IPv4-mapped ::ffff:a.b.c.d → extract embedded v4 and apply the v4 rules.
  if (
    g[0] === 0 &&
    g[1] === 0 &&
    g[2] === 0 &&
    g[3] === 0 &&
    g[4] === 0 &&
    g[5] === 0xffff
  ) {
    const a = (g[6]! >> 8) & 0xff;
    const b = g[6]! & 0xff;
    const c = (g[7]! >> 8) & 0xff;
    const d = g[7]! & 0xff;
    return isBlockedV4(`${a}.${b}.${c}.${d}`);
  }
  return false;
}

/**
 * Validate `url` against `policy` before an outbound request. The permissive
 * policy always returns `{ ok: true }`; `denyPrivate` returns a recoverable
 * error {@link ToolOutput} when the URL is disallowed. It is RETURNED, never
 * thrown.
 */
export function validateFetchUrl(
  url: string,
  policy: UrlPolicy,
): { ok: true } | { ok: false; error: ToolOutput } {
  if (!policy.denyPrivate) {
    return { ok: true };
  }
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch (e) {
    return {
      ok: false,
      error: urlDenied(
        `invalid URL: ${e instanceof Error ? e.message : String(e)}`,
      ),
    };
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    // `protocol` includes the trailing colon (e.g. `ftp:`); strip it for the msg.
    const scheme = parsed.protocol.replace(/:$/, "");
    return {
      ok: false,
      error: urlDenied(`scheme '${scheme}' not allowed by URL policy`),
    };
  }
  if (parsed.hostname === "") {
    return { ok: false, error: urlDenied("URL has no host") };
  }
  // Node's `URL.hostname` KEEPS the brackets on an IPv6 literal (verified
  // empirically: `new URL("http://[::1]/").hostname === "[::1]"`, and
  // `isIP("[::1]") === 0`). Strip a single bracket pair so `isIP` recognizes the
  // literal — mirrors the Rust reference, which strips `[`/`]` before parsing.
  const host = parsed.hostname.replace(/^\[/, "").replace(/\]$/, "");
  const lowered = host.toLowerCase();
  if (lowered === "localhost" || lowered.endsWith(".localhost")) {
    return { ok: false, error: urlDenied(`host '${host}' is loopback`) };
  }
  // `isIP` returns 4/6 for an IP literal, 0 otherwise.
  const family = isIP(host);
  if (family !== 0) {
    const blocked = family === 4 ? isBlockedV4(host) : isBlockedV6(host);
    if (blocked) {
      return {
        ok: false,
        error: urlDenied(`host '${host}' is in a blocked address range`),
      };
    }
  }
  // Host is NOT an IP literal (a DNS name like example.com): ALLOW. We do not
  // resolve DNS here — a name that resolves to a private address is the
  // intentional, documented limitation of this seam (enforce at the network
  // layer if resolution-time protection is required).
  return { ok: true };
}

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

  private policy: UrlPolicy = UrlPolicy.permissive();

  /** Wire an opt-in SSRF {@link UrlPolicy} (default is permissive). The fetched
   *  URL is validated against this policy before each request is sent. */
  withUrlPolicy(policy: UrlPolicy): this {
    this.policy = policy;
    return this;
  }

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
    const allowed = validateFetchUrl(p.value.url, this.policy);
    if (!allowed.ok) return allowed.error;
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
  private policy: UrlPolicy = UrlPolicy.permissive();

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

  /** Wire an opt-in SSRF {@link UrlPolicy} (default is permissive). The
   *  configured endpoint is validated against this policy before each request is
   *  sent. */
  withUrlPolicy(policy: UrlPolicy): this {
    this.policy = policy;
    return this;
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
    // Validate the CONFIGURED endpoint (the raw backend URL), not the
    // post-param-merge request URL, before any outbound request.
    const allowed = validateFetchUrl(backend.endpoint, this.policy);
    if (!allowed.ok) return allowed.error;
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
