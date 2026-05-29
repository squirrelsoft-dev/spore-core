/**
 * Concrete hook implementations (spore-core issue #69):
 *   - {@link StandardHookChain} — in-memory reference chain: registration-order
 *     fan-out, chained mutation through pre-hooks, sync aggregation, async
 *     fire-and-forget.
 *   - {@link FunctionHook} — inline closure handler (primary type, R23).
 *   - {@link CommandHook} — shell-command handler (R18–R22).
 */

import { spawn } from "node:child_process";

import {
  HookError,
  applyHookMutation,
  hookContextPayload,
  hookEventIsAsyncOnly,
  hookEventIsSyncOnly,
  hookEventOf,
  validateDecisionForEvent,
  type FireOutcome,
  type Hook,
  type HookChain,
  type HookContext,
  type HookDecision,
  type HookEvent,
  type HookSync,
} from "./types.js";

function hookSyncMode(hook: Hook): HookSync {
  return hook.syncMode?.() ?? "sync";
}

// ============================================================================
// StandardHookChain
// ============================================================================

/**
 * In-memory reference {@link HookChain}. Holds hooks in a flat list and fans
 * out in registration order. There is no borrow checker to satisfy — pre-hooks
 * mutate the live {@link HookContext} object directly, and the next hook in the
 * chain observes those mutations (R2). The Rust `unsafe transmute` reborrow has
 * no TypeScript counterpart.
 */
export class StandardHookChain implements HookChain {
  private readonly entries: Hook[] = [];

  register(hook: Hook): void {
    const mode = hookSyncMode(hook);
    for (const event of hook.events()) {
      if (hookEventIsSyncOnly(event) && mode === "async") {
        throw HookError.syncOnlyEvent(hook.name(), event);
      }
      if (hookEventIsAsyncOnly(event) && mode === "sync") {
        throw HookError.asyncOnlyEvent(hook.name(), event);
      }
    }
    this.entries.push(hook);
  }

  /** Snapshot the registered hooks subscribed to `event`, preserving
   *  registration order. */
  private hooksFor(event: HookEvent): Hook[] {
    return this.entries.filter((h) => h.events().includes(event));
  }

  async fire(ctx: HookContext, signal?: AbortSignal): Promise<FireOutcome> {
    const event = hookEventOf(ctx);
    const hooks = this.hooksFor(event);
    const injects: string[] = [];

    for (const hook of hooks) {
      if (hookSyncMode(hook) === "async") {
        // R11: async hooks are fire-and-forget. Snapshot the payload, spawn the
        // handler, and never await it. Its result and failure are swallowed.
        const payloadCtx = ctx;
        void Promise.resolve()
          .then(() => hook.handle(payloadCtx, signal))
          .catch(() => {
            /* swallowed: observability-only */
          });
        continue;
      }

      // Sync hook: await it, validate, then act. The hook may have mutated
      // `ctx` in place; the next hook sees that mutation (R2).
      const decision = await hook.handle(ctx, signal);
      const invalid = validateDecisionForEvent(decision, event); // R24
      if (invalid) throw invalid;

      switch (decision.decision) {
        case "continue":
          break;
        case "inject":
          injects.push(decision.context);
          break;
        case "block":
          return { kind: "block", reason: decision.reason };
        case "deny":
          return { kind: "deny", reason: decision.reason };
        case "mutate":
          applyHookMutation(ctx, hook.name(), decision.data);
          break;
        default: {
          const _exhaustive: never = decision;
          void _exhaustive;
          break;
        }
      }
    }

    if (injects.length === 0) return { kind: "continue" };
    return { kind: "inject", context: injects.join("\n") };
  }
}

// ============================================================================
// FunctionHook — inline closure handler
// ============================================================================

/** A synchronous-or-async closure handler. May mutate the context in place. */
export type HookFn = (
  ctx: HookContext,
  signal?: AbortSignal,
) => HookDecision | Promise<HookDecision>;

/**
 * A {@link Hook} backed by an inline closure — the primary handler type for
 * harness builders (R23). The closure runs inside {@link handle}.
 */
export class FunctionHook implements Hook {
  private readonly _name: string;
  private readonly _events: HookEvent[];
  private _sync: HookSync = "sync";
  private readonly func: HookFn;

  constructor(name: string, events: HookEvent[], func: HookFn) {
    this._name = name;
    this._events = events;
    this.func = func;
  }

  /** Mark this hook async (fire-and-forget). Only legal for events that are not
   *  sync-only; the chain enforces this at register time. Returns `this` for
   *  fluent construction. */
  asyncMode(): this {
    this._sync = "async";
    return this;
  }

  async handle(ctx: HookContext, signal?: AbortSignal): Promise<HookDecision> {
    return this.func(ctx, signal);
  }

  events(): HookEvent[] {
    return [...this._events];
  }

  name(): string {
    return this._name;
  }

  syncMode(): HookSync {
    return this._sync;
  }
}

// ============================================================================
// CommandHook — shell command handler
// ============================================================================

/**
 * A {@link Hook} that shells out to an external command. stdin receives
 * `{"event":"<snake_case>","context":<payload>}` (R18); stdout is parsed as a
 * tagged {@link HookDecision} (R19). Nonzero exit → {@link HookError}
 * `command_failed` (R20, NOT an implicit block); malformed stdout →
 * `command_output_invalid` (R21). No sandbox and no timeout in v1 (R22).
 */
export class CommandHook implements Hook {
  private readonly _name: string;
  private readonly _events: HookEvent[];
  private _sync: HookSync = "sync";
  private readonly program: string;
  private readonly args: string[];

  constructor(name: string, events: HookEvent[], program: string, args: string[] = []) {
    this._name = name;
    this._events = events;
    this.program = program;
    this.args = args;
  }

  asyncMode(): this {
    this._sync = "async";
    return this;
  }

  async handle(ctx: HookContext, _signal?: AbortSignal): Promise<HookDecision> {
    const stdinPayload = JSON.stringify({
      event: hookEventOf(ctx),
      context: hookContextPayload(ctx),
    });
    const { stdout } = await this.run(stdinPayload);
    const trimmed = stdout.trim();
    try {
      return JSON.parse(trimmed) as HookDecision;
    } catch (e) {
      throw HookError.commandOutputInvalid(this.program, (e as Error).message);
    }
  }

  /** Spawn the command, write `stdinPayload`, and collect stdout/stderr.
   *  Throws {@link HookError} `command_failed` on spawn error or nonzero exit. */
  private run(stdinPayload: string): Promise<{ stdout: string }> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.program, this.args, {
        stdio: ["pipe", "pipe", "pipe"],
      });
      let stdout = "";
      let stderr = "";
      let settled = false;

      const fail = (code: number, message: string): void => {
        if (settled) return;
        settled = true;
        reject(HookError.commandFailed(this.program, code, message.trim()));
      };

      child.stdout.on("data", (chunk: Buffer) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
      });
      child.on("error", (err) => fail(-1, err.message));
      child.on("close", (code) => {
        if (settled) return;
        if (code === 0) {
          settled = true;
          resolve({ stdout });
        } else {
          fail(code ?? -1, stderr);
        }
      });

      child.stdin.on("error", (err) => fail(-1, err.message));
      child.stdin.write(stdinPayload);
      child.stdin.end();
    });
  }

  events(): HookEvent[] {
    return [...this._events];
  }

  name(): string {
    return this._name;
  }

  syncMode(): HookSync {
    return this._sync;
  }
}
