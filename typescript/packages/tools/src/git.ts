/**
 * Git tools: GitLog, GitDiff, GitCommit, GitStatus, GitReset.
 */

import type {
  CommandOutput,
  SandboxProvider,
  SandboxViolation,
  ToolCall,
  ToolOutput,
} from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  GitCommitParamsSchema,
  GitDiffParamsSchema,
  GitLogParamsSchema,
  GitResetParamsSchema,
  parseParams,
} from "./params.js";
import {
  finishWithPossibleTruncation,
  isSandboxViolation,
  sbExecuteCommand,
} from "./sandbox-defaults.js";

export type { GitResetMode } from "./params.js";

async function runGit(
  args: readonly string[],
  sandbox: SandboxProvider,
  signal?: AbortSignal,
): Promise<CommandOutput | { violation: SandboxViolation }> {
  const out = await sbExecuteCommand(sandbox, "git", args, null, null, signal);
  if (isSandboxViolation(out)) return { violation: out };
  return out;
}

function classify(
  out: CommandOutput,
): { ok: true; content: string } | { ok: false; error: ToolOutput } {
  if (out.exit_code === 0) return { ok: true, content: out.stdout };
  return {
    ok: false,
    error: {
      kind: "error",
      message: `git exit ${out.exit_code} ; ${out.stderr.trimEnd()}`,
      recoverable: true,
    },
  };
}

// ============================================================================
// GitLog
// ============================================================================

export class GitLogTool implements Tool {
  static readonly NAME = "git_log";
  readonly name = GitLogTool.NAME;
  mayProduceLargeOutput(): boolean {
    return true;
  }
  static schema(): ToolSchema {
    return {
      name: GitLogTool.NAME,
      description: "Show recent git commits",
      parameters: {
        type: "object",
        properties: { n: { type: "integer" }, format: { type: "string" } },
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
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(GitLogParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const args = ["log", "-n", String(p.value.n)];
    if (p.value.format === "oneline") args.push("--oneline");
    else args.push(`--format=${p.value.format}`);
    const out = await runGit(args, sandbox, signal);
    if ("violation" in out)
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out.violation,
      });
    const c = classify(out);
    return c.ok
      ? finishWithPossibleTruncation(c.content, call.id, sandbox)
      : c.error;
  }
}

// ============================================================================
// GitDiff
// ============================================================================

export class GitDiffTool implements Tool {
  static readonly NAME = "git_diff";
  readonly name = GitDiffTool.NAME;
  mayProduceLargeOutput(): boolean {
    return true;
  }
  static schema(): ToolSchema {
    return {
      name: GitDiffTool.NAME,
      description: "Show a git diff",
      parameters: {
        type: "object",
        properties: { from: { type: "string" }, to: { type: "string" } },
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
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(GitDiffParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const args: string[] = ["diff"];
    if (p.value.from) args.push(p.value.from);
    if (p.value.to) args.push(p.value.to);
    const out = await runGit(args, sandbox, signal);
    if ("violation" in out)
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out.violation,
      });
    const c = classify(out);
    return c.ok
      ? finishWithPossibleTruncation(c.content, call.id, sandbox)
      : c.error;
  }
}

// ============================================================================
// GitCommit
// ============================================================================

export class GitCommitTool implements Tool {
  static readonly NAME = "git_commit";
  readonly name = GitCommitTool.NAME;
  static schema(): ToolSchema {
    return {
      name: GitCommitTool.NAME,
      description: "Stage files (if any) and create a git commit",
      parameters: {
        type: "object",
        properties: {
          message: { type: "string" },
          files: { type: "array", items: { type: "string" } },
        },
        required: ["message"],
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
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(GitCommitParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    let combined = "";
    const files = p.value.files ?? [];
    if (files.length > 0) {
      const out = await runGit(["add", ...files], sandbox, signal);
      if ("violation" in out)
        return toolExecutionErrorToOutput({
          kind: "sandbox_violation",
          violation: out.violation,
        });
      const c = classify(out);
      if (!c.ok) return c.error;
      combined += c.content;
    }
    const out = await runGit(
      ["commit", "-m", p.value.message],
      sandbox,
      signal,
    );
    if ("violation" in out)
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out.violation,
      });
    const c = classify(out);
    if (!c.ok) return c.error;
    combined += c.content;
    return { kind: "success", content: combined, truncated: false };
  }
}

// ============================================================================
// GitStatus
// ============================================================================

export class GitStatusTool implements Tool {
  static readonly NAME = "git_status";
  readonly name = GitStatusTool.NAME;
  static schema(): ToolSchema {
    return {
      name: GitStatusTool.NAME,
      description: "Show git status (porcelain)",
      parameters: { type: "object", properties: {} },
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
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const out = await runGit(["status", "--porcelain"], sandbox, signal);
    if ("violation" in out)
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out.violation,
      });
    const c = classify(out);
    return c.ok
      ? { kind: "success", content: c.content, truncated: false }
      : c.error;
  }
}

// ============================================================================
// GitReset
// ============================================================================

export class GitResetTool implements Tool {
  static readonly NAME = "git_reset";
  readonly name = GitResetTool.NAME;
  static schema(): ToolSchema {
    return {
      name: GitResetTool.NAME,
      description: "Reset to a target commit (hard/soft/mixed)",
      parameters: {
        type: "object",
        properties: {
          target: { type: "string" },
          mode: { type: "string", enum: ["hard", "soft", "mixed"] },
        },
        required: ["target", "mode"],
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
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(GitResetParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const flag = `--${p.value.mode}`;
    const out = await runGit(["reset", flag, p.value.target], sandbox, signal);
    if ("violation" in out)
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out.violation,
      });
    const c = classify(out);
    return c.ok
      ? { kind: "success", content: c.content, truncated: false }
      : c.error;
  }
}
