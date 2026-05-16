/**
 * Execution tools: BashCommand, RunTests.
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  BashCommandParamsSchema,
  parseParams,
  RunTestsParamsSchema,
} from "./params.js";
import {
  finishWithPossibleTruncation,
  isSandboxViolation,
  sbExecuteCommand,
  sbResolvePath,
} from "./sandbox-defaults.js";

// ============================================================================
// BashCommand
// ============================================================================

export class BashCommandTool implements Tool {
  static readonly NAME = "bash_command";
  readonly name = BashCommandTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: BashCommandTool.NAME,
      description: "Execute a shell command via the sandbox",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string" },
          args: { type: "array", items: { type: "string" } },
          timeout: { type: "integer" },
        },
        required: ["command"],
      },
      annotations: {
        read_only: false,
        destructive: true,
        idempotent: false,
        open_world: true,
      },
    };
  }

  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(BashCommandParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const timeoutMs = p.value.timeout != null ? p.value.timeout * 1000 : null;
    const out = await sbExecuteCommand(
      sandbox,
      p.value.command,
      p.value.args ?? [],
      null,
      timeoutMs,
      signal,
    );
    if (isSandboxViolation(out))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out,
      });
    if (out.timed_out) {
      const secs = timeoutMs != null ? Math.round(timeoutMs / 1000) : 0;
      return {
        kind: "error",
        message: `command timed out after ${secs}s`,
        recoverable: true,
      };
    }
    if (out.exit_code === 0) {
      return finishWithPossibleTruncation(out.stdout, call.id, sandbox);
    }
    return {
      kind: "error",
      message: `exit ${out.exit_code} ; stderr: ${out.stderr.trimEnd()}`,
      recoverable: true,
    };
  }
}

// ============================================================================
// RunTests
// ============================================================================

export class RunTestsTool implements Tool {
  static readonly NAME = "run_tests";
  readonly name = RunTestsTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: RunTestsTool.NAME,
      description: "Run a test command in a working directory",
      parameters: {
        type: "object",
        properties: {
          command: { type: "string" },
          working_dir: { type: "string" },
          timeout: { type: "integer" },
        },
        required: ["command", "working_dir"],
      },
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: true,
      },
    };
  }

  async execute(
    call: ToolCall,
    sandbox: SandboxProvider,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const p = parseParams(RunTestsParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const timeoutMs = p.value.timeout != null ? p.value.timeout * 1000 : null;
    const working = await sbResolvePath(sandbox, p.value.working_dir);
    if (isSandboxViolation(working))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: working,
      });
    // Split command into program + args (shell-free; spec says "no shell").
    const parts = p.value.command.split(/\s+/).filter((s) => s.length > 0);
    const program = parts.shift();
    if (!program) {
      return toolExecutionErrorToOutput({
        kind: "invalid_parameters",
        reason: "command must not be empty",
      });
    }
    const out = await sbExecuteCommand(
      sandbox,
      program,
      parts,
      working,
      timeoutMs,
      signal,
    );
    if (isSandboxViolation(out))
      return toolExecutionErrorToOutput({
        kind: "sandbox_violation",
        violation: out,
      });
    if (out.timed_out) {
      const secs = timeoutMs != null ? Math.round(timeoutMs / 1000) : 0;
      return {
        kind: "error",
        message: `tests timed out after ${secs}s`,
        recoverable: true,
      };
    }
    const combined = `${out.stdout}\n${out.stderr}`;
    if (out.exit_code === 0) {
      return finishWithPossibleTruncation(combined, call.id, sandbox);
    }
    return {
      kind: "error",
      message: `tests failed (exit ${out.exit_code}): ${combined}`,
      recoverable: true,
    };
  }
}
