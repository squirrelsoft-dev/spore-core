/**
 * `remember(key, value)` — persist a fact into the run store.
 *
 * This is the **write** half of the custom-tool pair, defined with the
 * {@link toolRegistry.defineTool | defineTool} helper: a Zod input schema plus
 * an async `execute` body. `defineTool` DERIVES the advertised JSON schema from
 * that one schema and validates the model's arguments against it, so the schema
 * the model sees and the validation the tool performs can never drift apart.
 *
 * It demonstrates the storage seam: a tool's only path to durable, per-run state
 * is the {@link toolRegistry.ToolContext} — `ctx.runStore` keyed by
 * `ctx.sessionId`. The `sandbox` parameter is part of the execute signature but
 * unused here — these tools never touch the filesystem, so they ignore it
 * (named `_sandbox`).
 *
 * Keys are namespaced under `fact:{key}` so the example cannot collide with the
 * reserved store keys the catalogue uses (`todo`, `task`, `memory`).
 */

import { toolOutput, toolRegistry } from "@spore/core";
import { z } from "zod";

type StandardTool = toolRegistry.StandardTool;

/**
 * Prefix applied to every key so this example's facts live in their own
 * namespace inside the run store.
 */
export const FACT_PREFIX = "fact:";

/** Tool name — also used by tests and `recall` cross-checks. */
export const REMEMBER_NAME = "remember";

/**
 * Validated input for `remember`. This single Zod schema is the source of truth:
 * `defineTool` advertises a JSON schema derived from it AND validates the
 * model's arguments against it.
 */
export const RememberInput = z.object({
  key: z
    .string()
    .describe(
      "Short, stable key to file the fact under (e.g. 'habitat', 'lifespan').",
    ),
  value: z.string().describe("The fact to remember."),
});

/**
 * Build the `remember` tool. `defineTool` generates the {@link toolRegistry.Tool}
 * impl, derives the schema from {@link RememberInput}, and bundles them into a
 * {@link StandardTool} ready for `.tool(...)`.
 *
 * Annotations are omitted, so they default to all-`false` — `remember` MUTATES
 * shared persisted state, so (unlike `recall`) it is intentionally not
 * `read_only`.
 */
export function rememberTool(): StandardTool {
  return toolRegistry.defineTool({
    name: REMEMBER_NAME,
    description:
      "Store a fact under a short key so it can be recalled later. " +
      "Use a stable, memorable key (e.g. 'habitat', 'lifespan').",
    input: RememberInput,
    execute: async (input, _sandbox, ctx) => {
      const storeKey = `${FACT_PREFIX}${input.key}`;
      try {
        await ctx.runStore.put(ctx.sessionId, storeKey, input.value);
      } catch (e) {
        return toolOutput.error(
          `remember: could not persist '${input.key}': ${errMessage(e)}`,
        );
      }
      return toolOutput.success(`remembered ${input.key}`);
    },
  });
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
