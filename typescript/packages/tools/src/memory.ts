/**
 * Memory tool (#82, storage seam #78): the scope-aware read/write tool over the
 * persisted episodic {@link "@spore/core".storage.MemoryStore}.
 *
 * One tool, {@link MemoryTool} (`NAME = "memory"`), dispatched on an `operation`
 * discriminator (`write`, `read`). It is the agent-facing surface over the
 * scope-aware memory seam shipped in #78 ({@link "@spore/core".toolRegistry.ToolContext.memoryStore}).
 *
 * ## Types
 *   - {@link "@spore/core".storage.MemoryEntry} — `{ role, content, timestamp,
 *     metadata }`. UNCHANGED by #82 — `metadata` already exists; the tool
 *     exposes it as an optional `write` param (decision C).
 *   - `StorageScope` — `"user" | "project" | "local"`. `local` is rejected at
 *     runtime on BOTH ops (see Rules).
 *   - {@link MemoryToolParams} — the `operation`-tagged input union.
 *
 * ## Seam methods used (#78)
 *   - `ctx.memoryStore.appendMemory(scope, sessionId, entry)` for `write`.
 *   - `ctx.memoryStore.getMemories(scope, sessionId, limit)` for a scoped `read`.
 *   - `ctx.memoryStore.getMemoriesMerged(sessionId, limit)` for a merged `read` —
 *     the SINGLE merge implementation (#82 D2; User ∪ Project, newest-first, no
 *     dedup, Local excluded).
 *
 * ## Rules enforced
 *   - **R1 write→read roundtrip.** A `write` appends one entry to the given
 *     scope; a subsequent same-scope `read` returns it (newest-first).
 *   - **R2 write success content (decision A).** `write` returns the serialized
 *     just-written entry as the success content.
 *   - **R3 read default limit (decision B).** `limit` defaults to `50`,
 *     overridable; `read` returns the most-recent `limit` entries newest-first.
 *   - **R4 metadata on write (decision C).** `metadata` is optional, defaults to
 *     `{}`; stored verbatim on the entry. `MemoryEntry` is NOT changed.
 *   - **R5 scope isolation.** A non-merged `read` of one scope never sees the
 *     other scope's entries.
 *   - **R6 merged read (decision D2).** `read` with `merged: true` returns the
 *     User ∪ Project merge via the seam's single merge method.
 *   - **R7 Local rejected on BOTH ops.** `local` scope → recoverable error with
 *     the EXACT message
 *     `"Local scope is not supported by MemoryTool — use User or Project."`,
 *     checked BEFORE any storage access (nothing is written).
 *   - **R8 bad params recoverable.** Bad input → recoverable error.
 *   - **R9 storage error recoverable.** A storage failure from append/get maps
 *     to a recoverable error.
 *   - **R10 read does not write.** A `read` performs no append.
 *
 * ## Annotations (decision E)
 * NOT annotated `read_only`. A `read_only` tool would be run CONCURRENTLY by
 * `dispatchAll` and could race the shared append; like {@link "@spore/tools".TaskListTool}
 * this tool leaves all annotations false so the registry dispatches it
 * sequentially.
 *
 * ## Known v1 limitation (#78 Q7)
 * Memory is `SessionId`-keyed for v1: the tool always uses `ctx.sessionId` and
 * offers NO cross-session addressing param. v2 should add session-independent
 * memory keying — do not introduce it here.
 */

import {
  memory as coreMemory,
  type SandboxProvider,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import type { storage, toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;
type MemoryEntry = storage.MemoryEntry;
type StorageScope = storage.StorageScope;

import { toolExecutionErrorToOutput } from "./errors.js";
import { MemoryToolParamsSchema, parseParams } from "./params.js";

const { Timestamp } = coreMemory;

/** Exact error message returned when a `local`-scoped op is attempted. */
export const MEMORY_LOCAL_REJECTED_MESSAGE =
  "Local scope is not supported by MemoryTool — use User or Project.";

export class MemoryTool implements Tool {
  static readonly NAME = "memory";
  readonly name = MemoryTool.NAME;

  static schema(): ToolSchema {
    // Properties kept sorted/stable for cache stability. `scope` advertises only
    // user/project — `local` is rejected at runtime but intentionally omitted
    // from the advertised enum.
    return {
      name: MemoryTool.NAME,
      description: "Read or write scope-aware episodic memory for this session",
      parameters: {
        type: "object",
        properties: {
          content: { type: "string" },
          limit: { type: "integer" },
          merged: { type: "boolean" },
          metadata: { type: "object" },
          operation: {
            type: "string",
            enum: ["read", "write"],
          },
          role: { type: "string" },
          scope: {
            type: "string",
            enum: ["project", "user"],
          },
        },
        required: ["operation", "scope"],
      },
      // Intentionally NOT read_only: the shared append must dispatch
      // sequentially. See module docs (decision E).
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
      },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    ctx: ToolContext,
  ): Promise<ToolOutput> {
    const { sessionId, memoryStore } = ctx;

    // 1. Parse params (bad input → recoverable).
    const p = parseParams(MemoryToolParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const params = p.value;

    // R7: reject `local` on BOTH ops BEFORE touching storage; nothing written.
    if (params.scope === "local") {
      return {
        kind: "error",
        message: MEMORY_LOCAL_REJECTED_MESSAGE,
        recoverable: true,
      };
    }
    const scope: StorageScope = params.scope;

    if (params.operation === "write") {
      // Build the entry; the tool stamps "now" as an RFC-3339 string.
      const entry: MemoryEntry = {
        role: params.role,
        content: params.content,
        timestamp: Timestamp.now(),
        metadata: params.metadata as storage.JsonValue,
      };

      try {
        await memoryStore.appendMemory(scope, sessionId, entry);
      } catch (e) {
        return {
          kind: "error",
          message: `could not append memory: ${errMessage(e)}`,
          recoverable: true,
        };
      }

      // R2 (decision A): success content = the serialized just-written entry.
      return {
        kind: "success",
        content: JSON.stringify(entry),
        truncated: false,
      };
    }

    // operation === "read"
    let entries: MemoryEntry[];
    try {
      // R6 (decision D2): merged read drives the single seam merge; otherwise a
      // scoped read (R5 isolation). R10: neither path writes.
      entries = params.merged
        ? await memoryStore.getMemoriesMerged(sessionId, params.limit)
        : await memoryStore.getMemories(scope, sessionId, params.limit);
    } catch (e) {
      return {
        kind: "error",
        message: `could not read memory: ${errMessage(e)}`,
        recoverable: true,
      };
    }

    return {
      kind: "success",
      content: JSON.stringify(entries),
      truncated: false,
    };
  }
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
