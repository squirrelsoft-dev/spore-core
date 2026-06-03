/**
 * `defineTool` — ergonomic, drift-proof tool definition.
 *
 * Mirrors Rust's `tool!` macro (`rust/crates/spore-core/src/macros.rs`): the
 * caller supplies a single Zod {@link z.ZodType | input schema} that is the one
 * source of truth, and {@link defineTool} does two things with it:
 *
 *   1. **Advertises** the tool by deriving its JSON Schema `parameters` from the
 *      Zod schema via `zod-to-json-schema` — never hand-written, so the schema
 *      the model sees always matches what the tool validates.
 *   2. **Validates** the model's raw arguments with the *same* Zod schema before
 *      `execute` ever runs. On a parse failure the tool returns a **recoverable**
 *      {@link ToolOutput} whose message contains the substring
 *      `invalid parameters` — so a configured `ToolCallRepair` can coerce the
 *      arguments and re-dispatch rather than halting the run.
 *
 * Because one schema serves both advertisement and validation, the classic
 * drift between a hand-written `parameters` blob and a separate validation
 * schema is eliminated by construction.
 *
 * @example
 * ```ts
 * import { z } from "zod";
 * import { toolRegistry, toolOutput } from "@spore/core";
 *
 * const echo = toolRegistry.defineTool({
 *   name: "echo",
 *   description: "Echoes the input message",
 *   input: z.object({
 *     message: z.string().describe("Text to echo back."),
 *     shout: z.boolean().default(false),
 *   }),
 *   execute: async (input) =>
 *     toolOutput.success(input.shout ? input.message.toUpperCase() : input.message),
 * });
 * // echo.schema.name === "echo"; echo.schema.parameters exposes `message`/`shout`.
 * ```
 */

import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

import type { SandboxProvider, ToolOutput } from "../harness/types.js";
import { toolOutput } from "../harness/types.js";
import type { ToolCall } from "../model/schemas.js";
import {
  defaultToolAnnotations,
  type StandardTool,
  type Tool,
  type ToolAnnotations,
  type ToolContext,
} from "./types.js";

/**
 * The async body of a {@link defineTool} tool. Receives the already-validated,
 * fully-typed `input` (parsed by the tool's Zod schema), plus the same
 * {@link SandboxProvider} (environment seam) and {@link ToolContext} (storage
 * seam) every {@link Tool.execute} gets. An optional {@link AbortSignal} is
 * threaded through for cancellation.
 */
export type DefineToolExecute<T> = (
  input: T,
  sandbox: SandboxProvider,
  ctx: ToolContext,
  signal?: AbortSignal,
) => Promise<ToolOutput>;

/** Options bag for {@link defineTool}. */
export interface DefineToolOptions<S extends z.ZodTypeAny> {
  /** The tool's registered name — must be unique within a registry. */
  name: string;
  /** Human/model-facing description of what the tool does. */
  description: string;
  /**
   * The single source of truth: a Zod schema for the tool's input. The
   * advertised JSON Schema is derived from it, and the model's raw arguments are
   * validated against it before {@link DefineToolOptions.execute} runs.
   */
  input: S;
  /** The tool body, invoked with the parsed, typed input. */
  execute: DefineToolExecute<z.infer<S>>;
  /**
   * Optional {@link ToolAnnotations}. Any omitted field defaults to `false`
   * (via {@link defaultToolAnnotations}), matching Rust's `ToolAnnotations`
   * default. Omit entirely for an all-`false` tool.
   */
  annotations?: Partial<ToolAnnotations>;
}

/**
 * Derive a JSON Schema `parameters` object from a Zod input schema. Produces a
 * draft-07 object schema with `properties`/`required` populated, then strips the
 * `$schema` meta-key so the result is a clean parameter object suitable for the
 * model. `$ref`s are inlined (`$refStrategy: "none"`) so the schema is
 * self-contained.
 */
function deriveParameters(schema: z.ZodTypeAny): Record<string, unknown> {
  const json = zodToJsonSchema(schema, {
    target: "jsonSchema7",
    $refStrategy: "none",
  }) as Record<string, unknown>;
  // Drop the JSON-Schema dialect marker — `parameters` is an inline object, not
  // a standalone document.
  delete json.$schema;
  return json;
}

/**
 * Define a {@link Tool} and bundle it with its derived {@link ToolSchema} into a
 * ready-to-register {@link StandardTool} — the TypeScript analogue of Rust's
 * `tool!` macro.
 *
 * The returned `StandardTool` plugs straight into the builder via `.tool(...)`
 * or into a registry's `register(...)`. Its `schema.parameters` is derived from
 * `input`; its `implementation.execute` validates against the same `input`
 * before calling your body, returning a recoverable `invalid parameters` error
 * on a mismatch.
 */
export function defineTool<S extends z.ZodTypeAny>(
  options: DefineToolOptions<S>,
): StandardTool {
  const { name, description, input, execute } = options;

  const annotations: ToolAnnotations = {
    ...defaultToolAnnotations(),
    ...options.annotations,
  };

  const implementation: Tool = {
    name,
    async execute(
      call: ToolCall,
      sandbox: SandboxProvider,
      ctx: ToolContext,
      signal?: AbortSignal,
    ): Promise<ToolOutput> {
      const parsed = input.safeParse(call.input ?? {});
      if (!parsed.success) {
        // Recoverable: a configured ToolCallRepair can coerce the args and
        // re-dispatch. The substring `invalid parameters` mirrors Rust's message.
        return toolOutput.error(
          `invalid parameters for tool \`${name}\`: ${formatZodError(parsed.error)}`,
        );
      }
      return execute(parsed.data as z.infer<S>, sandbox, ctx, signal);
    },
  };

  return {
    implementation,
    schema: {
      name,
      description,
      parameters: deriveParameters(input),
      annotations,
    },
  };
}

/**
 * Render a {@link z.ZodError} as a compact, single-line message naming the
 * offending paths — enough for a repair pass (and a human reading logs) to see
 * which fields were wrong.
 */
function formatZodError(error: z.ZodError): string {
  return error.issues
    .map((issue) => {
      const path = issue.path.join(".");
      return path ? `${path}: ${issue.message}` : issue.message;
    })
    .join("; ");
}
