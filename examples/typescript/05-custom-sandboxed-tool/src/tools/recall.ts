/**
 * `recall(key)` — read a previously-remembered fact back out of the run store.
 *
 * This is the **read** half of the pair, also defined with the
 * {@link toolRegistry.defineTool | defineTool} helper. Unlike
 * {@link "./remember.js".rememberTool} it passes `annotations` to mark itself
 * `read_only` + `idempotent`: it only reads shared state, so the registry may
 * dispatch it concurrently with other read-only tools.
 *
 * Looking up a key that was never stored is a *recoverable* error — the agent
 * can adapt (try a different key, or remember the fact first) rather than
 * halting the run.
 */

import { toolOutput, toolRegistry, type storage } from "@spore/core";
import { z } from "zod";

import { FACT_PREFIX } from "./remember.js";

type StandardTool = toolRegistry.StandardTool;
type JsonValue = storage.JsonValue;

/** Tool name. */
export const RECALL_NAME = "recall";

/**
 * Validated input for `recall`. The advertised JSON schema is derived from this
 * one Zod schema, which also validates the model's arguments.
 */
export const RecallInput = z.object({
  key: z
    .string()
    .describe("The key a fact was previously stored under with `remember`."),
});

/**
 * Build the `recall` tool. `annotations` marks it `read_only` + `idempotent`
 * (a pure read of shared state), in contrast to `remember`.
 */
export function recallTool(): StandardTool {
  return toolRegistry.defineTool({
    name: RECALL_NAME,
    description: "Recall a fact previously stored with `remember`, by its key.",
    input: RecallInput,
    annotations: { read_only: true, idempotent: true },
    execute: async (input, _sandbox, ctx) => {
      const storeKey = `${FACT_PREFIX}${input.key}`;
      let value: JsonValue | undefined;
      try {
        value = await ctx.runStore.get(ctx.sessionId, storeKey);
      } catch (e) {
        return toolOutput.error(
          `recall: could not read '${input.key}': ${errMessage(e)}`,
        );
      }
      if (value === undefined) {
        return toolOutput.error(`no fact stored under '${input.key}'`);
      }
      return toolOutput.success(valueToString(value));
    },
  });
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
