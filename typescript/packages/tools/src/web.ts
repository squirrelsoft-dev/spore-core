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
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import { parseParams, WebFetchParamsSchema, WebSearchParamsSchema } from "./params.js";
import { finishWithPossibleTruncation } from "./sandbox-defaults.js";

// ============================================================================
// WebFetch
// ============================================================================

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
        properties: { url: { type: "string" } },
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
      return finishWithPossibleTruncation(body, call.id, sandbox);
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
 * Web search tool. The search backend endpoint is injected so tests run against
 * a mock HTTP server. With no endpoint configured (the default), every call is
 * a recoverable error.
 */
export class WebSearchTool implements Tool {
  static readonly NAME = "web_search";
  readonly name = WebSearchTool.NAME;

  private readonly endpoint: string | null;

  /** Construct with no backend configured (calls error until one is set). */
  constructor(endpoint?: string | null) {
    this.endpoint = endpoint ?? null;
  }

  /** Construct with a search endpoint (the query is POSTed as JSON
   *  `{ "query": ... }`; the response body is returned verbatim). */
  static withEndpoint(endpoint: string): WebSearchTool {
    return new WebSearchTool(endpoint);
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
    if (this.endpoint == null) {
      return {
        kind: "error",
        message: "web_search backend not configured",
        recoverable: true,
      };
    }
    try {
      const resp = await fetch(this.endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: p.value.query }),
        signal,
      });
      const body = await resp.text();
      return finishWithPossibleTruncation(body, call.id, sandbox);
    } catch (e) {
      return {
        kind: "error",
        message: `web search failed: ${e instanceof Error ? e.message : String(e)}`,
        recoverable: true,
      };
    }
  }
}
