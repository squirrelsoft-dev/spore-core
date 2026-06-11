/**
 * Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile.
 */

import { promises as fs } from "node:fs";
import { join, relative } from "node:path";

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  finishWithPossibleTruncation,
  isSandboxViolation,
  LARGE_OUTPUT_THRESHOLD,
  sbResolvePath,
} from "./sandbox-defaults.js";
import {
  DeleteFileParamsSchema,
  ListDirParamsSchema,
  MoveFileParamsSchema,
  parseParams,
  ReadFileParamsSchema,
  WriteFileParamsSchema,
} from "./params.js";

// ============================================================================
// ReadFile range helper (#132)
// ============================================================================

/**
 * Split `content` into lines, preserving each line's trailing `\n` (like
 * Rust's `split_inclusive('\n')`). The final line may or may not end in `\n`.
 */
function splitInclusive(content: string): string[] {
  if (content === "") return [];
  const lines: string[] = [];
  let start = 0;
  for (let i = 0; i < content.length; i++) {
    if (content[i] === "\n") {
      lines.push(content.slice(start, i + 1));
      start = i + 1;
    }
  }
  // Trailing fragment (no final newline).
  if (start < content.length) {
    lines.push(content.slice(start));
  }
  return lines;
}

/** Parsed-and-defaulted params used by {@link applyReadRange}. */
export interface ReadRangeOptions {
  offset: number;
  length: number;
  line_numbers: boolean;
}

/**
 * Apply the #132 range/line-number transform to a fully-read file body.
 *
 * With all params at their defaults the original `content` is returned
 * unchanged (byte-identical to the pre-#132 behavior). Any non-default param
 * prepends a `[lines {start}–{end} of {total}]\n` header (U+2013 en-dash).
 */
export function applyReadRange(
  content: string,
  params: ReadRangeOptions,
): { ok: true; value: string } | { ok: false; error: string } {
  const isDefault =
    params.offset === 1 && params.length === 0 && !params.line_numbers;
  if (isDefault) return { ok: true, value: content };

  if (params.offset === 0) {
    return { ok: false, error: "offset must be ≥ 1 (1-indexed)" };
  }

  // Empty file: any params still yield empty content with no header.
  if (content === "") return { ok: true, value: "" };

  const lines = splitInclusive(content);
  const total = lines.length;

  if (params.offset > total) {
    return {
      ok: false,
      error: `offset ${params.offset} exceeds file length ${total}`,
    };
  }

  const start = params.offset; // 1-indexed, validated >= 1 and <= total.
  const end =
    params.length === 0
      ? total
      : Math.min(start + params.length - 1, total);

  const startIdx = start - 1; // convert to 0-indexed
  const selected = lines.slice(startIdx, end);

  let out = `[lines ${start}–${end} of ${total}]\n`;
  if (params.line_numbers) {
    const width = String(total).length;
    for (let i = 0; i < selected.length; i++) {
      const n = start + i;
      out += `${String(n).padStart(width)} | ${selected[i]}`;
    }
  } else {
    for (const line of selected) {
      out += line;
    }
  }
  return { ok: true, value: out };
}

// ============================================================================
// ReadFile
// ============================================================================

export class ReadFileTool implements Tool {
  static readonly NAME = "read_file";
  readonly name = ReadFileTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: ReadFileTool.NAME,
      description:
        "Read a file's contents. Optionally read a line range " +
        "(offset is 1-indexed start, length is max lines, 0 = to EOF) " +
        "and/or prefix each line with its number via line_numbers. " +
        "With no optional params the whole file is returned verbatim.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          offset: {
            type: "integer",
            description: "1-indexed start line (default 1).",
          },
          length: {
            type: "integer",
            description:
              "Max lines to return; 0 = no limit / read to EOF (default 0).",
          },
          line_numbers: {
            type: "boolean",
            description:
              "Prefix each returned line with its 1-indexed number (default false).",
          },
        },
        required: ["path"],
      },
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
    sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(ReadFileParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const resolved = await sbResolvePath(sandbox, p.value.path, "read");
    if (isSandboxViolation(resolved))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: resolved,
      });
    try {
      const content = await fs.readFile(resolved, "utf8");
      const rangeResult = applyReadRange(content, {
        offset: p.value.offset ?? 1,
        length: p.value.length ?? 0,
        line_numbers: p.value.line_numbers ?? false,
      });
      if (rangeResult.ok) {
        return finishWithPossibleTruncation(rangeResult.value, call.id, sandbox);
      }
      return { kind: "error", message: rangeResult.error, recoverable: true };
    } catch (e) {
      return {
        kind: "error",
        message: `read failed: ${stringifyError(e)}`,
        recoverable: true,
      };
    }
  }
}

// ============================================================================
// WriteFile
// ============================================================================

export class WriteFileTool implements Tool {
  static readonly NAME = "write_file";
  readonly name = WriteFileTool.NAME;

  static schema(): ToolSchema {
    return {
      name: WriteFileTool.NAME,
      description:
        "Write content to a file (overwrites by default; set append=true to append)",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          content: { type: "string" },
          append: { type: "boolean" },
        },
        required: ["path", "content"],
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
    const p = parseParams(WriteFileParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const { path, content, append } = p.value;
    const resolved = await sbResolvePath(sandbox, path, "write");
    if (isSandboxViolation(resolved))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: resolved,
      });
    try {
      if (append) {
        await fs.appendFile(resolved, content, "utf8");
      } else {
        await fs.writeFile(resolved, content, "utf8");
      }
      const bytes = Buffer.byteLength(content, "utf8");
      return {
        kind: "success",
        content: `wrote ${bytes} bytes to ${path}`,
        truncated: false,
      };
    } catch (e) {
      return {
        kind: "error",
        message: `write failed: ${stringifyError(e)}`,
        recoverable: true,
      };
    }
  }
}

// ============================================================================
// ListDir
// ============================================================================

export class ListDirTool implements Tool {
  static readonly NAME = "list_dir";
  readonly name = ListDirTool.NAME;

  static schema(): ToolSchema {
    return {
      name: ListDirTool.NAME,
      description: "List directory entries (optionally recursive)",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string" },
          recursive: { type: "boolean" },
        },
        required: ["path"],
      },
      annotations: {
        read_only: true,
        destructive: false,
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
    const p = parseParams(ListDirParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const resolved = await sbResolvePath(sandbox, p.value.path, "read");
    if (isSandboxViolation(resolved))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: resolved,
      });
    // Emit paths relative to the workspace root so each entry can be fed
    // straight back into read_file/write_file. The sandbox treats every input
    // path as root-relative, so absolute paths would not round-trip (see #93).
    // `resolved` is the absolute path of the listed directory (= root-relative
    // `path`); each entry is under it. Relativize against `resolved`, then
    // re-anchor onto the caller-supplied (root-relative) `path`.
    const listed = p.value.path;
    const toRootRelative = (entryAbsolutePath: string): string | null => {
      // Path of the entry relative to the listed directory.
      const relToListed = relative(resolved, entryAbsolutePath);
      // Skip the listed directory itself (walk yields it first).
      if (relToListed === "") return null;
      // Re-anchor onto the caller-supplied path, drop any leading `./`, and
      // normalize to POSIX-style forward slashes so output is stable
      // cross-platform and round-trips through the sandbox. Preserve a leading
      // separator for absolute inputs (mirrors Rust keeping the RootDir
      // component while filtering CurDir).
      const joined = join(listed, relToListed);
      const isAbs = joined.startsWith("/") || joined.startsWith("\\");
      const body = joined
        .split(/[\\/]/)
        .filter((seg) => seg !== "" && seg !== ".")
        .join("/");
      return isAbs ? `/${body}` : body;
    };
    const entries: string[] = [];
    try {
      if (p.value.recursive) {
        await walk(resolved, toRootRelative, entries);
      } else {
        const items = await fs.readdir(resolved);
        for (const it of items) {
          const rel = toRootRelative(join(resolved, it));
          if (rel !== null) entries.push(rel);
        }
      }
    } catch (e) {
      return {
        kind: "error",
        message: `read_dir failed: ${stringifyError(e)}`,
        recoverable: true,
      };
    }
    entries.sort();
    const content = entries.join("\n");
    if (content.length > LARGE_OUTPUT_THRESHOLD) {
      return finishWithPossibleTruncation(content, call.id, sandbox);
    }
    return { kind: "success", content, truncated: false };
  }
}

async function walk(
  dir: string,
  toRootRelative: (entryAbsolutePath: string) => string | null,
  out: string[],
): Promise<void> {
  // Do not emit the listed directory itself; `toRootRelative` returns null for
  // it anyway, but skipping here avoids pushing intermediate directories.
  let items: import("node:fs").Dirent[];
  try {
    items = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const it of items) {
    const full = join(dir, it.name);
    if (it.isDirectory()) {
      const rel = toRootRelative(full);
      if (rel !== null) out.push(rel);
      await walk(full, toRootRelative, out);
    } else {
      const rel = toRootRelative(full);
      if (rel !== null) out.push(rel);
    }
  }
}

// ============================================================================
// DeleteFile
// ============================================================================

export class DeleteFileTool implements Tool {
  static readonly NAME = "delete_file";
  readonly name = DeleteFileTool.NAME;

  static schema(): ToolSchema {
    return {
      name: DeleteFileTool.NAME,
      description: "Delete a file",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
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
    const p = parseParams(DeleteFileParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const resolved = await sbResolvePath(sandbox, p.value.path, "write");
    if (isSandboxViolation(resolved))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: resolved,
      });
    try {
      await fs.unlink(resolved);
      return {
        kind: "success",
        content: `deleted ${p.value.path}`,
        truncated: false,
      };
    } catch (e) {
      return {
        kind: "error",
        message: `delete failed: ${stringifyError(e)}`,
        recoverable: true,
      };
    }
  }
}

// ============================================================================
// MoveFile
// ============================================================================

export class MoveFileTool implements Tool {
  static readonly NAME = "move_file";
  readonly name = MoveFileTool.NAME;

  static schema(): ToolSchema {
    return {
      name: MoveFileTool.NAME,
      description: "Move/rename a file",
      parameters: {
        type: "object",
        properties: { src: { type: "string" }, dst: { type: "string" } },
        required: ["src", "dst"],
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
    const p = parseParams(MoveFileParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const src = await sbResolvePath(sandbox, p.value.src, "write");
    if (isSandboxViolation(src))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: src,
      });
    const dst = await sbResolvePath(sandbox, p.value.dst, "write");
    if (isSandboxViolation(dst))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: dst,
      });
    try {
      await fs.rename(src, dst);
      return {
        kind: "success",
        content: `moved ${p.value.src} -> ${p.value.dst}`,
        truncated: false,
      };
    } catch (e) {
      return {
        kind: "error",
        message: `move failed: ${stringifyError(e)}`,
        recoverable: true,
      };
    }
  }
}

function stringifyError(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
