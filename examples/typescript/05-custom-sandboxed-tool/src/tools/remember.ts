/**
 * `remember(key, value)` — persist a fact into the run store.
 *
 * This is the **write** half of the custom-tool pair. It demonstrates the
 * storage seam: a tool's only path to durable, per-run state is the
 * {@link ToolContext} — `ctx.runStore` keyed by `ctx.sessionId`. The `sandbox`
 * parameter is part of the {@link Tool.execute} signature but unused here —
 * these tools never touch the filesystem, so they ignore it (named `_sandbox`).
 *
 * Keys are namespaced under `fact:{key}` so the example cannot collide with the
 * reserved store keys the catalogue uses (`todo`, `task`, `memory`).
 */

import type {
  SandboxProvider,
  ToolCall,
  ToolOutput,
  toolRegistry,
} from "@spore/core";
import { toolOutput } from "@spore/core";

type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

/**
 * Prefix applied to every key so this example's facts live in their own
 * namespace inside the run store.
 */
export const FACT_PREFIX = "fact:";

export class RememberTool implements Tool {
  static readonly NAME = "remember";
  readonly name = RememberTool.NAME;

  /** The registry-side schema. `name` MUST equal {@link Tool.name}. */
  static schema(): ToolSchema {
    return {
      name: RememberTool.NAME,
      description:
        "Store a fact under a short key so it can be recalled later. " +
        "Use a stable, memorable key (e.g. 'habitat', 'lifespan').",
      parameters: {
        type: "object",
        properties: {
          key: { type: "string" },
          value: { type: "string" },
        },
        required: ["key", "value"],
      },
      // Intentionally NOT read_only: this mutates shared persisted state.
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
    const input = (call.input ?? {}) as Record<string, unknown>;
    const key = input.key;
    if (typeof key !== "string") {
      return toolOutput.error("remember: missing or non-string 'key'");
    }
    const value = input.value;
    if (typeof value !== "string") {
      return toolOutput.error("remember: missing or non-string 'value'");
    }

    const storeKey = `${FACT_PREFIX}${key}`;
    try {
      await ctx.runStore.put(ctx.sessionId, storeKey, value);
    } catch (e) {
      return toolOutput.error(
        `remember: could not persist '${key}': ${errMessage(e)}`,
      );
    }
    return toolOutput.success(`remembered ${key}`);
  }
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
