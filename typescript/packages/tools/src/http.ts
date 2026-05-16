/**
 * HTTP tools: HttpGet, HttpPost. Uses Node's global `fetch` (Node 18+).
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  HttpGetParamsSchema,
  HttpPostParamsSchema,
  parseParams,
} from "./params.js";
import { finishWithPossibleTruncation } from "./sandbox-defaults.js";

function buildHeaders(
  map: Record<string, string> | null | undefined,
): Record<string, string> | undefined {
  if (!map) return undefined;
  return { ...map };
}

function hasContentType(
  map: Record<string, string> | null | undefined,
): boolean {
  if (!map) return false;
  return Object.keys(map).some((k) => k.toLowerCase() === "content-type");
}

// ============================================================================
// HttpGet
// ============================================================================

export class HttpGetTool implements Tool {
  static readonly NAME = "http_get";
  readonly name = HttpGetTool.NAME;
  mayProduceLargeOutput(): boolean {
    return true;
  }
  static schema(): ToolSchema {
    return {
      name: HttpGetTool.NAME,
      description: "Perform an HTTP GET",
      parameters: {
        type: "object",
        properties: { url: { type: "string" }, headers: { type: "object" } },
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
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(HttpGetParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    try {
      const resp = await fetch(p.value.url, {
        method: "GET",
        headers: buildHeaders(p.value.headers),
        signal,
      });
      const body = await resp.text();
      return finishWithPossibleTruncation(body, call.id, sandbox);
    } catch (e) {
      return {
        kind: "error",
        message: `http get failed: ${e instanceof Error ? e.message : String(e)}`,
        recoverable: true,
      };
    }
  }
}

// ============================================================================
// HttpPost
// ============================================================================

export class HttpPostTool implements Tool {
  static readonly NAME = "http_post";
  readonly name = HttpPostTool.NAME;
  mayProduceLargeOutput(): boolean {
    return true;
  }
  static schema(): ToolSchema {
    return {
      name: HttpPostTool.NAME,
      description: "Perform an HTTP POST with a JSON body",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string" },
          body: {},
          headers: { type: "object" },
        },
        required: ["url", "body"],
      },
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: true,
      },
    };
  }
  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(HttpPostParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const headers: Record<string, string> = { ...(p.value.headers ?? {}) };
    if (!hasContentType(p.value.headers))
      headers["content-type"] = "application/json";
    try {
      const resp = await fetch(p.value.url, {
        method: "POST",
        headers,
        body: JSON.stringify(p.value.body ?? null),
        signal,
      });
      const body = await resp.text();
      return finishWithPossibleTruncation(body, call.id, sandbox);
    } catch (e) {
      return {
        kind: "error",
        message: `http post failed: ${e instanceof Error ? e.message : String(e)}`,
        recoverable: true,
      };
    }
  }
}
