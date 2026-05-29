/**
 * Search tools: GrepFiles, FindFiles.
 */

import { promises as fs } from "node:fs";
import { join, relative } from "node:path";

import picomatch from "picomatch";
import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  FindFilesParamsSchema,
  GrepFilesParamsSchema,
  GrepParamsSchema,
  parseParams,
} from "./params.js";
import {
  finishWithPossibleTruncation,
  isSandboxViolation,
  sbResolvePath,
} from "./sandbox-defaults.js";

interface GrepHit {
  file: string;
  line: number;
  text: string;
}

async function scanFile(
  path: string,
  re: RegExp,
  out: GrepHit[],
): Promise<void> {
  let content: string;
  try {
    content = await fs.readFile(path, "utf8");
  } catch {
    return;
  }
  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (re.test(lines[i]!))
      out.push({ file: path, line: i + 1, text: lines[i]! });
  }
}

async function walkFiles(dir: string, out: string[]): Promise<void> {
  let items: import("node:fs").Dirent[];
  try {
    items = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const it of items) {
    const full = join(dir, it.name);
    if (it.isDirectory()) await walkFiles(full, out);
    else if (it.isFile()) out.push(full);
  }
}

// ============================================================================
// GrepFiles
// ============================================================================

export class GrepFilesTool implements Tool {
  static readonly NAME = "grep_files";
  readonly name = GrepFilesTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: GrepFilesTool.NAME,
      description: "Search files for a regex pattern",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string" },
          path: { type: "string" },
          recursive: { type: "boolean" },
        },
        required: ["pattern", "path"],
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
    const p = parseParams(GrepFilesParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    let re: RegExp;
    try {
      re = new RegExp(p.value.pattern);
    } catch (e) {
      return toolExecutionErrorToOutput({
        kind: "invalid_parameters",
        reason: `invalid regex: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
    const root = await sbResolvePath(sandbox, p.value.path, "read");
    if (isSandboxViolation(root))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: root,
      });

    const matches: GrepHit[] = [];
    let stat: import("node:fs").Stats | null = null;
    try {
      stat = await fs.stat(root);
    } catch {
      // path missing — empty result
    }

    if (stat?.isFile()) {
      await scanFile(root, re, matches);
    } else if (stat?.isDirectory()) {
      if (p.value.recursive) {
        const files: string[] = [];
        await walkFiles(root, files);
        for (const f of files) await scanFile(f, re, matches);
      } else {
        const items = await fs.readdir(root, { withFileTypes: true });
        for (const it of items) {
          if (it.isFile()) await scanFile(join(root, it.name), re, matches);
        }
      }
    }

    matches.sort((a, b) =>
      a.file === b.file ? a.line - b.line : a.file < b.file ? -1 : 1,
    );
    const body = matches.map((m) => `${m.file}:${m.line}:${m.text}`).join("\n");
    return finishWithPossibleTruncation(body, call.id, sandbox);
  }
}

// ============================================================================
// FindFiles
// ============================================================================

export class FindFilesTool implements Tool {
  static readonly NAME = "find_files";
  readonly name = FindFilesTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: FindFilesTool.NAME,
      description: "Find files matching a glob",
      parameters: {
        type: "object",
        properties: { glob: { type: "string" }, path: { type: "string" } },
        required: ["glob", "path"],
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
    const p = parseParams(FindFilesParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const root = await sbResolvePath(sandbox, p.value.path, "read");
    if (isSandboxViolation(root))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: root,
      });

    let matcher: (s: string) => boolean;
    try {
      matcher = picomatch(p.value.glob);
    } catch (e) {
      return toolExecutionErrorToOutput({
        kind: "invalid_parameters",
        reason: `invalid glob: ${e instanceof Error ? e.message : String(e)}`,
      });
    }

    const all: string[] = [];
    await walkFiles(root, all);
    const hits: string[] = [];
    for (const f of all) {
      const rel = relative(root, f);
      if (matcher(rel) || matcher(f)) hits.push(f);
    }
    hits.sort();
    return finishWithPossibleTruncation(hits.join("\n"), call.id, sandbox);
  }
}

// ============================================================================
// Grep (#81, net-new — output modes)
// ============================================================================
//
// Net-new tool alongside the byte-identical {@link GrepFilesTool} (`grep_files`).
// It is `read_only` like `grep_files` but adds an `output_mode`:
//   - `content`            → `path:line:text` per matching line (default).
//   - `files_with_matches` → distinct file paths that contain a match.
//   - `count`              → `path:count` per file with matches.

export class GrepTool implements Tool {
  static readonly NAME = "grep";
  readonly name = GrepTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: GrepTool.NAME,
      description: "Search files for a regex pattern with selectable output mode",
      parameters: {
        type: "object",
        properties: {
          pattern: { type: "string" },
          path: { type: "string" },
          recursive: { type: "boolean" },
          output_mode: {
            type: "string",
            enum: ["content", "count", "files_with_matches"],
          },
        },
        required: ["pattern", "path"],
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
    const p = parseParams(GrepParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    let re: RegExp;
    try {
      re = new RegExp(p.value.pattern);
    } catch (e) {
      return toolExecutionErrorToOutput({
        kind: "invalid_parameters",
        reason: `invalid regex: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
    const root = await sbResolvePath(sandbox, p.value.path, "read");
    if (isSandboxViolation(root))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: root,
      });

    const matches: GrepHit[] = [];
    let stat: import("node:fs").Stats | null = null;
    try {
      stat = await fs.stat(root);
    } catch {
      // path missing — empty result
    }

    if (stat?.isFile()) {
      await scanFile(root, re, matches);
    } else if (stat?.isDirectory()) {
      if (p.value.recursive) {
        const files: string[] = [];
        await walkFiles(root, files);
        for (const f of files) await scanFile(f, re, matches);
      } else {
        const items = await fs.readdir(root, { withFileTypes: true });
        for (const it of items) {
          if (it.isFile()) await scanFile(join(root, it.name), re, matches);
        }
      }
    }

    matches.sort((a, b) =>
      a.file === b.file ? a.line - b.line : a.file < b.file ? -1 : 1,
    );

    let body = "";
    switch (p.value.output_mode) {
      case "content":
        body = matches.map((m) => `${m.file}:${m.line}:${m.text}`).join("\n");
        break;
      case "files_with_matches": {
        // matches are sorted by file; emit each distinct file once.
        const files: string[] = [];
        for (const m of matches) {
          if (files[files.length - 1] !== m.file) files.push(m.file);
        }
        body = files.join("\n");
        break;
      }
      case "count": {
        // matches are sorted by file; count per file.
        const counts: [string, number][] = [];
        for (const m of matches) {
          const last = counts[counts.length - 1];
          if (last && last[0] === m.file) last[1] += 1;
          else counts.push([m.file, 1]);
        }
        body = counts.map(([f, c]) => `${f}:${c}`).join("\n");
        break;
      }
    }
    return finishWithPossibleTruncation(body, call.id, sandbox);
  }
}
