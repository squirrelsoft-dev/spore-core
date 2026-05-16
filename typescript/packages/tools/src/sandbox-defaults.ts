/**
 * Default implementations of the optional {@link SandboxProvider} methods.
 *
 * Mirrors the defaulted methods on the Rust `SandboxProvider` trait
 * (`executeCommand`, `handleLargeOutput`, `resolvePath`). Tools call into
 * these helpers so that lightweight test sandboxes only need to implement
 * `validate` while production sandboxes can override each method.
 */

import { spawn } from "node:child_process";
import { resolve as pathResolve } from "node:path";
import type {
  CommandOutput,
  SandboxProvider,
  SandboxViolation,
  TruncatedOutput,
} from "@spore/core";

/** Threshold (in bytes/chars) above which tool output is routed through
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
      resolve({
        stdout,
        stderr: stderr + String(err),
        exit_code: -1,
        timed_out: timedOut,
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
      });
    });
  });
}

/**
 * Default truncation strategy. Approximates the head/tail budgets as
 * `tokens * CHARS_PER_TOKEN` characters and joins them with a marker.
 */
export async function defaultHandleLargeOutput(
  content: string,
  callId: string,
  headTokens: number,
  tailTokens: number,
): Promise<TruncatedOutput> {
  const headChars = headTokens * CHARS_PER_TOKEN;
  const tailChars = tailTokens * CHARS_PER_TOKEN;
  if (content.length <= headChars + tailChars) {
    return { summary: content, full_ref: null };
  }
  const head = content.slice(0, headChars);
  const tail = content.slice(content.length - tailChars);
  const omitted = content.length - headChars - tailChars;
  const marker = `\n... [truncated ${omitted} chars; call_id=${callId}] ...\n`;
  return { summary: head + marker + tail, full_ref: null };
}

/** Identity path resolution. Production sandboxes canonicalize and check
 * against root/allowlist/denylist. */
export async function defaultResolvePath(
  path: string,
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
): Promise<string | SandboxViolation> {
  if (sandbox.resolvePath) return sandbox.resolvePath(path);
  return defaultResolvePath(path);
}

export function isSandboxViolation(v: unknown): v is SandboxViolation {
  if (typeof v !== "object" || v === null) return false;
  const k = (v as { kind?: unknown }).kind;
  return (
    k === "path_escape" ||
    k === "network_violation" ||
    k === "path_denied" ||
    k === "read_only_violation"
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
    return { kind: "success", content: t.summary, truncated: true };
  }
  return { kind: "success", content, truncated: false };
}
