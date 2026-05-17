/**
 * MiddlewareChain — canonical types (spore-core issue #11).
 *
 * Mirrors `rust/crates/spore-core/src/middleware.rs` byte-for-byte on the
 * wire: tagged unions use a `kind` discriminator in `snake_case`; struct
 * fields are `snake_case`.
 *
 * Middleware intercepts the agent loop at six hook points
 * (BeforeSession, BeforeTurn, BeforeTool, AfterTool, BeforeCompletion,
 * AfterSession) for cross-cutting concerns: budgets, permissions, PII
 * redaction, tracing, etc.
 *
 * Rules (enforced by {@link StandardMiddlewareChain}):
 *   - Before hooks run sorted by priority **ascending** (lowest first).
 *   - After hooks run sorted by priority **descending** (highest first,
 *     wrapping pattern).
 *   - First {@link MiddlewareDecision} of kind `halt` or `surface_to_human`
 *     stops the chain — downstream middleware do not run.
 *   - `force_another_turn` is valid only on `before_completion`. All
 *     injections are concatenated (newline-joined) and the chain continues
 *     to evaluate remaining middleware.
 *   - `surface_to_human` is valid only on `before_tool` and
 *     `before_completion`. Returning it elsewhere becomes an
 *     {@link IllegalDecision} error, surfaced as `halt` to the loop.
 *
 * Design constraints (not enforced — middleware authors must respect):
 *   - Middleware must not call `ModelInterface` or `ToolRegistry`. Neither
 *     is reachable from {@link HookContext}.
 *   - Middleware must not hold session state on `this` keyed by
 *     `SessionId`. Use an external map and clear it in `after_session`.
 */

import type { HumanRequest, RunResult, SessionId, SessionState, Task } from "../harness/types.js";
import type { ToolCall, ToolResult } from "../model/schemas.js";

// ============================================================================
// HookPoint
// ============================================================================

export type HookPoint =
  | "before_session"
  | "before_turn"
  | "before_tool"
  | "after_tool"
  | "before_completion"
  | "after_session";

export const ALL_HOOK_POINTS: readonly HookPoint[] = [
  "before_session",
  "before_turn",
  "before_tool",
  "after_tool",
  "before_completion",
  "after_session",
];

const BEFORE_HOOKS = new Set<HookPoint>([
  "before_session",
  "before_turn",
  "before_tool",
  "before_completion",
]);
const AFTER_HOOKS = new Set<HookPoint>(["after_tool", "after_session"]);

export function hookIsBefore(h: HookPoint): boolean {
  return BEFORE_HOOKS.has(h);
}

export function hookIsAfter(h: HookPoint): boolean {
  return AFTER_HOOKS.has(h);
}

export function hookAllowsSurfaceToHuman(h: HookPoint): boolean {
  return h === "before_tool" || h === "before_completion";
}

export function hookAllowsForceAnotherTurn(h: HookPoint): boolean {
  return h === "before_completion";
}

// ============================================================================
// HookContext — discriminated union; arrays are passed in by reference and
// may be mutated in place where the spec allows (BeforeTurn session,
// BeforeTool calls, AfterTool results).
// ============================================================================

export type HookContext =
  | { kind: "before_session"; task: Task; session_id: SessionId }
  | { kind: "before_turn"; session: SessionState; turn_number: number }
  | { kind: "before_tool"; calls: ToolCall[]; turn_number: number }
  | { kind: "after_tool"; calls: ToolCall[]; results: ToolResult[] }
  | {
      kind: "before_completion";
      response: string;
      turn_number: number;
      session_state: SessionState;
    }
  | { kind: "after_session"; result: RunResult; session_id: SessionId };

export function hookPointOf(ctx: HookContext): HookPoint {
  return ctx.kind;
}

// ============================================================================
// MiddlewareDecision (canonical, 5 variants)
// ============================================================================

export type MiddlewareDecision =
  | { kind: "continue" }
  /** The middleware mutated the borrowed context. Semantically equivalent
   *  to `continue` for control flow — exposed for observability. */
  | { kind: "continue_with_modification" }
  /** Valid only on `before_completion`. Injections from every middleware
   *  that returns this are newline-joined into one combined decision. */
  | { kind: "force_another_turn"; inject: string }
  | { kind: "halt"; reason: string }
  /** Valid only on `before_tool` and `before_completion`. First occurrence
   *  in priority order wins; downstream middleware do not run. */
  | { kind: "surface_to_human"; request: HumanRequest };

// ============================================================================
// Errors
// ============================================================================

export type MiddlewareErrorKind =
  | { kind: "already_registered"; name: string }
  | { kind: "no_hooks"; name: string }
  | { kind: "illegal_decision"; name: string; hook: HookPoint; decision: string };

export class MiddlewareError extends Error {
  override readonly name = "MiddlewareError";
  readonly kind: MiddlewareErrorKind["kind"];
  readonly detail: MiddlewareErrorKind;

  constructor(detail: MiddlewareErrorKind) {
    super(middlewareErrorMessage(detail));
    this.kind = detail.kind;
    this.detail = detail;
  }

  static alreadyRegistered(name: string): MiddlewareError {
    return new MiddlewareError({ kind: "already_registered", name });
  }
  static noHooks(name: string): MiddlewareError {
    return new MiddlewareError({ kind: "no_hooks", name });
  }
  static illegalDecision(name: string, hook: HookPoint, decision: string): MiddlewareError {
    return new MiddlewareError({ kind: "illegal_decision", name, hook, decision });
  }
}

function middlewareErrorMessage(e: MiddlewareErrorKind): string {
  switch (e.kind) {
    case "already_registered":
      return `middleware already registered: ${e.name}`;
    case "no_hooks":
      return `middleware ${e.name} declared zero hooks`;
    case "illegal_decision":
      return `middleware ${e.name} returned ${e.decision} from ${e.hook} which does not allow it`;
  }
}

// ============================================================================
// Interfaces
// ============================================================================

export interface Middleware {
  /** Invoked once per matched hook firing. Async to allow IO. */
  handle(ctx: HookContext, signal?: AbortSignal): Promise<MiddlewareDecision>;
  /** Hook points this middleware listens for. Must be non-empty. */
  hooks(): HookPoint[];
  /** Lower number = runs earlier on before hooks / later on after hooks.
   *  Defaults to 0 if omitted. */
  priority?(): number;
  /** Stable identifier; must be unique within a chain. */
  name(): string;
}

export interface MiddlewareChain {
  /** Throws {@link MiddlewareError} on duplicate name or empty hooks. */
  register(middleware: Middleware): void;

  fireBeforeSession(
    task: Task,
    sessionId: SessionId,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision>;

  fireBeforeTurn(
    session: SessionState,
    turnNumber: number,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision>;

  fireBeforeTool(
    calls: ToolCall[],
    turnNumber: number,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision>;

  fireAfterTool(
    calls: ToolCall[],
    results: ToolResult[],
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision>;

  fireBeforeCompletion(
    response: string,
    turnNumber: number,
    state: SessionState,
    signal?: AbortSignal,
  ): Promise<MiddlewareDecision>;

  fireAfterSession(result: RunResult, sessionId: SessionId, signal?: AbortSignal): Promise<void>;
}

// ============================================================================
// Standard priority sentinels
// ============================================================================

/** Tracing registers at the lowest possible priority so it fires first on
 *  every before-hook and last on every after-hook (wrapping pattern). */
export const TRACING_PRIORITY = Number.MIN_SAFE_INTEGER;

/** PatchToolCallsMiddleware registers at the second-lowest priority so it
 *  runs before all other before-tool middleware. */
export const PATCH_TOOL_CALLS_PRIORITY = Number.MIN_SAFE_INTEGER + 1;
