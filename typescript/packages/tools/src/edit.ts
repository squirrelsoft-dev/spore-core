/**
 * EditFile tool (#81, net-new Tier-1 sandbox tool).
 *
 * `edit_file` replaces the FIRST and ONLY occurrence of `old_string` with
 * `new_string` in the file at `path`. The match must be UNIQUE:
 * - `old_string` not found      → recoverable error {@link ToolOutput}.
 * - `old_string` found >1 time  → recoverable error {@link ToolOutput}.
 *
 * Net-new tool that does NOT replace `write_file` (issue #81, Q5). Annotated
 * `destructive` (it mutates a file in place).
 */

import { promises as fs } from "node:fs";

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import { EditFileParamsSchema, parseParams } from "./params.js";
import { isSandboxViolation, sbResolvePath } from "./sandbox-defaults.js";

function countOccurrences(haystack: string, needle: string): number {
  if (needle.length === 0) return 0;
  let count = 0;
  let idx = haystack.indexOf(needle);
  while (idx !== -1) {
    count += 1;
    idx = haystack.indexOf(needle, idx + needle.length);
  }
  return count;
}

export class EditFileTool implements Tool {
  static readonly NAME = "edit_file";
  readonly name = EditFileTool.NAME;

  static schema(): ToolSchema {
    return {
      name: EditFileTool.NAME,
      description:
        "Replace the unique occurrence of old_string with new_string in a file",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          old_string: { type: "string" },
          new_string: { type: "string" },
        },
        required: ["path", "old_string", "new_string"],
      },
      annotations: {
        read_only: false,
        destructive: true,
        idempotent: false,
        open_world: false,
      },
    };
  }

  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(EditFileParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const { path, old_string, new_string } = p.value;

    const resolved = await sbResolvePath(sandbox, path, "write");
    if (isSandboxViolation(resolved))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: resolved,
      });

    let content: string;
    try {
      content = await fs.readFile(resolved, "utf8");
    } catch (e) {
      return {
        kind: "error",
        message: `read failed: ${errMessage(e)}`,
        recoverable: true,
      };
    }

    const count = countOccurrences(content, old_string);
    if (count === 0) {
      return {
        kind: "error",
        message: `old_string not found in ${path}`,
        recoverable: true,
      };
    }
    if (count > 1) {
      return {
        kind: "error",
        message: `old_string is not unique in ${path} (${count} occurrences); provide more context`,
        recoverable: true,
      };
    }

    // Replace exactly the first (and, since count === 1, only) occurrence. Use
    // a replacer function so `$`-sequences in `new_string` are NOT interpreted
    // as substitution patterns.
    const updated = content.replace(old_string, () => new_string);
    try {
      await fs.writeFile(resolved, updated);
    } catch (e) {
      return {
        kind: "error",
        message: `write failed: ${errMessage(e)}`,
        recoverable: true,
      };
    }
    return { kind: "success", content: `edited ${path}`, truncated: false };
  }
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
