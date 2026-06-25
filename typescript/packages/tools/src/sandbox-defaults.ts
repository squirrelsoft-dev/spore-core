/**
 * Default implementations of the optional {@link SandboxProvider} methods.
 *
 * Mirrors the defaulted methods on the Rust `SandboxProvider` trait
 * (`executeCommand`, `handleLargeOutput`, `resolvePath`). Tools call into
 * these helpers so lightweight test sandboxes only need to implement
 * `validate` while production sandboxes override each method.
 *
 * Issue #6 update: `resolvePath` now takes an `Operation`, `TruncatedOutput`
 * now carries `{ content, truncated, fullRef, originalSize }`, and
 * `CommandOutput` has a `truncated` field.
 */

import { spawn } from "node:child_process";
import { resolve as pathResolve } from "node:path";
import type {
  CommandOutput,
  Operation,
  SandboxProvider,
  SandboxViolation,
  TruncatedOutput,
} from "@spore/core";

/** Threshold (in chars) above which tool output is routed through
 * `SandboxProvider.handleLargeOutput` instead of returned inline. */
export const LARGE_OUTPUT_THRESHOLD = 64 * 1024;

/** Default head/tail token budgets when calling `handleLargeOutput`. */
export const DEFAULT_HEAD_TOKENS = 2000;
export const DEFAULT_TAIL_TOKENS = 2000;

/** Approximate characters-per-token used by the default truncator. */
const CHARS_PER_TOKEN = 4;

/**
 * Spawn `command` with `args`, no shell, captured stdout/stderr. Returns
 * a {@link CommandOutput} or — for cases the sandbox would normally
 * intercept — a {@link SandboxViolation}. The base impl never produces a
 * violation; production implementations can.
 */
export async function defaultExecuteCommand(
  command: string,
  args: readonly string[],
  cwd?: string | null,
  timeoutMs?: number | null,
  signal?: AbortSignal,
): Promise<CommandOutput | SandboxViolation> {
  return new Promise((resolve) => {
    const child = spawn(command, args as string[], {
      cwd: cwd ?? undefined,
      shell: false,
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let timer: NodeJS.Timeout | null = null;
    let aborted = false;

    const onAbort = (): void => {
      aborted = true;
      child.kill("SIGTERM");
    };
    if (signal) {
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }
    if (timeoutMs != null && timeoutMs > 0) {
      timer = setTimeout(() => {
        timedOut = true;
        child.kill("SIGKILL");
      }, timeoutMs);
    }

    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (err) => {
      if (timer) clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", onAbort);
      // SC-15: a genuine spawn failure (the binary never started) is a typed
      // violation, not a fake `exit_code: -1` success. A timeout/abort is a real
      // run that exceeded the clock / was cancelled — it KEEPS the legacy
      // `CommandOutput { exit_code: -1, timed_out }` (only never-started spawns
      // become the violation). Callers already narrow the union.
      const code = (err as NodeJS.ErrnoException).code;
      if (!timedOut && !aborted && (code === "ENOENT" || code === "EACCES")) {
        resolve({
          kind: "exec_spawn_failed",
          command,
          message: String(err),
        });
        return;
      }
      resolve({
        stdout,
        stderr: stderr + String(err),
        exit_code: -1,
        timed_out: timedOut,
        truncated: false,
      });
    });
    child.on("close", (code, sig) => {
      if (timer) clearTimeout(timer);
      if (signal) signal.removeEventListener("abort", onAbort);
      let exit = code ?? -1;
      if (sig) exit = -1;
      resolve({
        stdout,
        stderr,
        exit_code: exit,
        timed_out: timedOut || aborted,
        truncated: false,
      });
    });
  });
}

/**
 * Default truncation strategy. Approximates the head/tail budgets as
 * `tokens * CHARS_PER_TOKEN` characters and joins them with a marker.
 * Does not offload — `full_ref` is always `null`.
 */
export async function defaultHandleLargeOutput(
  content: string,
  callId: string,
  headTokens: number,
  tailTokens: number,
): Promise<TruncatedOutput> {
  const headChars = headTokens * CHARS_PER_TOKEN;
  const tailChars = tailTokens * CHARS_PER_TOKEN;
  const originalSize = content.length;
  if (originalSize <= headChars + tailChars) {
    return {
      content,
      truncated: false,
      full_ref: null,
      original_size: originalSize,
    };
  }
  const head = content.slice(0, headChars);
  const tail = content.slice(originalSize - tailChars);
  const omitted = originalSize - headChars - tailChars;
  const marker = `\n... [truncated ${omitted} chars; call_id=${callId}] ...\n`;
  return {
    content: head + marker + tail,
    truncated: true,
    full_ref: null,
    original_size: originalSize,
  };
}

/** Identity path resolution. Production sandboxes canonicalize and check
 * against root/allowlist/denylist. */
export async function defaultResolvePath(
  path: string,
  _operation: Operation,
): Promise<string | SandboxViolation> {
  return pathResolve(path);
}

// ============================================================================
// Helpers that route through the sandbox method when present, else default.
// ============================================================================

export async function sbExecuteCommand(
  sandbox: SandboxProvider,
  command: string,
  args: readonly string[],
  cwd?: string | null,
  timeoutMs?: number | null,
  signal?: AbortSignal,
): Promise<CommandOutput | SandboxViolation> {
  if (sandbox.executeCommand) {
    return sandbox.executeCommand(command, args, cwd, timeoutMs, signal);
  }
  return defaultExecuteCommand(command, args, cwd, timeoutMs, signal);
}

export async function sbHandleLargeOutput(
  sandbox: SandboxProvider,
  content: string,
  callId: string,
  headTokens: number,
  tailTokens: number,
): Promise<TruncatedOutput> {
  if (sandbox.handleLargeOutput) {
    return sandbox.handleLargeOutput(content, callId, headTokens, tailTokens);
  }
  return defaultHandleLargeOutput(content, callId, headTokens, tailTokens);
}

export async function sbResolvePath(
  sandbox: SandboxProvider,
  path: string,
  operation: Operation,
): Promise<string | SandboxViolation> {
  if (sandbox.resolvePath) return sandbox.resolvePath(path, operation);
  return defaultResolvePath(path, operation);
}

export function isSandboxViolation(v: unknown): v is SandboxViolation {
  if (typeof v !== "object" || v === null) return false;
  const k = (v as { kind?: unknown }).kind;
  return (
    k === "path_escape" ||
    k === "path_denied" ||
    k === "extension_denied" ||
    k === "read_only_violation" ||
    k === "file_size_exceeded" ||
    k === "disallowed_command" ||
    k === "network_violation" ||
    k === "exec_spawn_failed"
  );
}

/**
 * Common postlude: if content exceeds {@link LARGE_OUTPUT_THRESHOLD},
 * route through `sandbox.handleLargeOutput` and tag `truncated: true`.
 */
export async function finishWithPossibleTruncation(
  content: string,
  callId: string,
  sandbox: SandboxProvider,
): Promise<{ kind: "success"; content: string; truncated: boolean }> {
  if (content.length > LARGE_OUTPUT_THRESHOLD) {
    const t = await sbHandleLargeOutput(
      sandbox,
      content,
      callId,
      DEFAULT_HEAD_TOKENS,
      DEFAULT_TAIL_TOKENS,
    );
    return { kind: "success", content: t.content, truncated: t.truncated };
  }
  return { kind: "success", content, truncated: false };
}
