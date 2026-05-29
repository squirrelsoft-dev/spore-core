/**
 * Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile.
 */

import { promises as fs } from "node:fs";
import { join } from "node:path";

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
      description: "Read a file's contents",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
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
      return finishWithPossibleTruncation(content, call.id, sandbox);
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
    const entries: string[] = [];
    try {
      if (p.value.recursive) {
        await walk(resolved, entries);
      } else {
        const items = await fs.readdir(resolved);
        for (const it of items) entries.push(join(resolved, it));
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

async function walk(dir: string, out: string[]): Promise<void> {
  out.push(dir);
  let items: import("node:fs").Dirent[];
  try {
    items = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const it of items) {
    const full = join(dir, it.name);
    if (it.isDirectory()) {
      await walk(full, out);
    } else {
      out.push(full);
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
