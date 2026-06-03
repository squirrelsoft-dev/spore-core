/**
 * `recall(key)` — read a previously-remembered fact back out of the run store.
 *
 * This is the **read** half of the pair. Unlike {@link "./remember.js".RememberTool}
 * it is annotated `read_only` + `idempotent`: it only reads shared state, so the
 * registry may dispatch it concurrently with other read-only tools.
 *
 * Looking up a key that was never stored is a *recoverable* error — the agent
 * can adapt (try a different key, or remember the fact first) rather than
 * halting the run.
 */

import type {
  SandboxProvider,
  ToolCall,
  ToolOutput,
  toolRegistry,
  storage,
} from "@spore/core";
import { toolOutput } from "@spore/core";

import { FACT_PREFIX } from "./remember.js";

type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;
type JsonValue = storage.JsonValue;

export class RecallTool implements Tool {
  static readonly NAME = "recall";
  readonly name = RecallTool.NAME;

  /** The registry-side schema. `name` MUST equal {@link Tool.name}. */
  static schema(): ToolSchema {
    return {
      name: RecallTool.NAME,
      description:
        "Recall a fact previously stored with `remember`, by its key.",
      parameters: {
        type: "object",
        properties: {
          key: { type: "string" },
        },
        required: ["key"],
      },
      // Pure read of shared state: safe to mark read_only + idempotent.
      annotations: {
        read_only: true,
        destructive: false,
        idempotent: true,
        open_world: false,
      },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    ctx: ToolContext,
  ): Promise<ToolOutput> {
    const input = (call.input ?? {}) as Record<string, unknown>;
    const key = input.key;
    if (typeof key !== "string") {
      return toolOutput.error("recall: missing or non-string 'key'");
    }

    const storeKey = `${FACT_PREFIX}${key}`;
    let value: JsonValue | undefined;
    try {
      value = await ctx.runStore.get(ctx.sessionId, storeKey);
    } catch (e) {
      return toolOutput.error(
        `recall: could not read '${key}': ${errMessage(e)}`,
      );
    }
    if (value === undefined) {
      return toolOutput.error(`no fact stored under '${key}'`);
    }
    return toolOutput.success(valueToString(value));
  }
}

/**
 * `remember` always stores a JSON string, so render that back as plain text.
 * Fall back to the JSON encoding for anything unexpected.
 */
function valueToString(value: JsonValue): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
