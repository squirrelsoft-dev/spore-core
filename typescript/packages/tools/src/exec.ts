/**
 * Execution tools: `Exec`, `BashCommand`, `RunTests`.
 *
 * Two distinct ways to run a process, with deliberately different contracts:
 *
 * - {@link ExecTool} (tool name `"exec"`) runs **one program directly** — no
 *   shell. `command` + `args` are passed verbatim to `sbExecuteCommand`
 *   (`spawn(..., { shell: false })`), so there are no pipes, redirects,
 *   globbing, or `$(...)`. Every argument is literal. This is the
 *   path-validated, no-injection-surface option.
 * - {@link BashCommandTool} (tool name `"bash_command"`) runs a **shell command
 *   line** via `/bin/sh -c <script>`, so it supports pipes, redirects,
 *   globbing, and `$(...)`. It is sugar over the same `sbExecuteCommand`
 *   primitive (`sbExecuteCommand(sandbox, "/bin/sh", ["-c", script], …)`).
 *
 *   TRADEOFF: because the shell itself opens any files the script touches,
 *   `bash_command` does NOT get the per-path `validate()` / `resolvePath`
 *   enforcement that `read_file` / `write_file` / `exec` get — it relies on the
 *   outer sandbox/container for isolation. `exec` remains the path-validated
 *   choice. `/bin/sh` also assumes a Unix target (fine for this repo; no
 *   `cmd.exe`/PowerShell branch).
 *
 * - {@link RunTestsTool} (tool name `"run_tests"`) splits a command string on
 *   whitespace and runs it shell-free inside a working directory.
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import {
  ExecParamsSchema,
  parseParams,
  RunTestsParamsSchema,
  ShellCommandParamsSchema,
} from "./params.js";
import {
  finishWithPossibleTruncation,
  isSandboxViolation,
  sbExecuteCommand,
  sbResolvePath,
} from "./sandbox-defaults.js";

// ============================================================================
// Exec — shell-free: run one program directly
// ============================================================================

/**
 * Runs one program directly via `sbExecuteCommand`. No shell: `command` +
 * `args` are passed verbatim (no pipes, redirects, globbing, or `$(...)`).
 * Path-validated through the sandbox.
 */
export class ExecTool implements Tool {
  static readonly NAME = "exec";
  readonly name = ExecTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: ExecTool.NAME,
      description:
        "Run one program directly. No shell: no pipes, redirects, globbing, " +
        "or $(...). Args are passed verbatim.",
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
    const p = parseParams(ExecParamsSchema, call);
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
// BashCommand — real shell: /bin/sh -c <script>
// ============================================================================

/**
 * Runs a shell command line via `/bin/sh -c <script>`, supporting pipes,
 * redirects, globbing, and `$(...)`. Sugar over the same `sbExecuteCommand`
 * primitive `exec` uses (`sbExecuteCommand(sandbox, "/bin/sh", ["-c", script],
 * working_dir?, timeout?)`).
 *
 * TRADEOFF: the shell opens any files the script touches itself, so this tool
 * does NOT receive the per-path `validate()` / `resolvePath` enforcement that
 * `read_file` / `write_file` / {@link ExecTool} get — it relies on the outer
 * sandbox/container for isolation. `exec` remains the path-validated choice.
 * `/bin/sh` assumes a Unix target (no Windows shell branch).
 */
export class BashCommandTool implements Tool {
  static readonly NAME = "bash_command";
  readonly name = BashCommandTool.NAME;

  mayProduceLargeOutput(): boolean {
    return true;
  }

  static schema(): ToolSchema {
    return {
      name: BashCommandTool.NAME,
      description:
        "Execute a shell command line via /bin/sh -c. Supports pipes, " +
        "redirects, globbing, and $(...).",
      parameters: {
        type: "object",
        properties: {
          script: { type: "string" },
          working_dir: { type: "string" },
          timeout: { type: "integer" },
        },
        required: ["script"],
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
    const p = parseParams(ShellCommandParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const timeoutMs = p.value.timeout != null ? p.value.timeout * 1000 : null;
    // Only the optional working_dir is path-validated; the script's own file
    // accesses go through the shell, unvalidated (see class doc above).
    let working: string | null = null;
    if (p.value.working_dir != null) {
      const resolved = await sbResolvePath(
        sandbox,
        p.value.working_dir,
        "read",
      );
      if (isSandboxViolation(resolved))
        return toolExecutionErrorToOutput({
          kind: "sandbox_violation",
          violation: resolved,
        });
      working = resolved;
    }
    const out = await sbExecuteCommand(
      sandbox,
      "/bin/sh",
      ["-c", p.value.script],
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
    const working = await sbResolvePath(
      sandbox,
      p.value.working_dir,
      "execute",
    );
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
