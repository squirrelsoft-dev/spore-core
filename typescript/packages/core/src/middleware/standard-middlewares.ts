/**
 * Standard middleware implementations (spore-core issue #11).
 *
 * Subset matching the Rust reference. Production deployments compose the
 * remaining middlewares (`PermissionMiddleware`, `PiiRedactionMiddleware`,
 * `RateLimitMiddleware`, `DirectoryMapMiddleware`, `TimeBudgetMiddleware`)
 * once the components they depend on land.
 */

import {
  PATCH_TOOL_CALLS_PRIORITY,
  TRACING_PRIORITY,
  type HookContext,
  type HookPoint,
  type Middleware,
  type MiddlewareDecision,
} from "./types.js";

// ============================================================================
// TracingMiddleware
// ============================================================================

/** Records every hook firing. The real implementation forwards to
 *  `ObservabilityProvider`; this version keeps an in-memory log so tests can
 *  assert ordering. */
export class TracingMiddleware implements Middleware {
  private readonly log: { hook: HookPoint; turn: number }[] = [];

  constructor(private readonly myName: string = "tracing") {}

  async handle(ctx: HookContext): Promise<MiddlewareDecision> {
    const turn = "turn_number" in ctx ? ctx.turn_number : 0;
    this.log.push({ hook: ctx.kind, turn });
    return { kind: "continue" };
  }
  hooks(): HookPoint[] {
    return [
      "before_session",
      "before_turn",
      "before_tool",
      "after_tool",
      "before_completion",
      "after_session",
    ];
  }
  priority(): number {
    return TRACING_PRIORITY;
  }
  name(): string {
    return this.myName;
  }

  entries(): readonly { hook: HookPoint; turn: number }[] {
    return this.log.slice();
  }
}

// ============================================================================
// PatchToolCallsMiddleware
// ============================================================================

/** Repairs empty / whitespace-only tool-call names to a configured fallback.
 *  Runs at the lowest BeforeTool priority so downstream middleware see
 *  clean calls (per spec). */
export class PatchToolCallsMiddleware implements Middleware {
  constructor(
    private readonly fallbackName: string,
    private readonly myName: string = "patch-tool-calls",
  ) {}

  async handle(ctx: HookContext): Promise<MiddlewareDecision> {
    if (ctx.kind !== "before_tool") return { kind: "continue" };
    let modified = false;
    for (const call of ctx.calls) {
      if (call.name.trim().length === 0) {
        call.name = this.fallbackName;
        modified = true;
      }
    }
    return modified ? { kind: "continue_with_modification" } : { kind: "continue" };
  }
  hooks(): HookPoint[] {
    return ["before_tool"];
  }
  priority(): number {
    return PATCH_TOOL_CALLS_PRIORITY;
  }
  name(): string {
    return this.myName;
  }
}

// ============================================================================
// LoopDetectionMiddleware
// ============================================================================

/** Tracks per-`path` edit counts and annotates the matching tool result
 *  with a `[loop-detection]` warning once `threshold` is reached.
 *
 *  The counts map is owned by the middleware instance — production code
 *  should key by `SessionId` in an external map and clear it on
 *  `after_session` (see spec). For the standalone reference impl one
 *  middleware instance is one session-equivalent; tests call {@link clear}
 *  to simulate session boundaries. */
export class LoopDetectionMiddleware implements Middleware {
  private readonly counts = new Map<string, number>();

  constructor(
    private readonly toolName: string,
    private readonly threshold: number,
    private readonly myName: string = "loop-detection",
  ) {}

  clear(): void {
    this.counts.clear();
  }

  async handle(ctx: HookContext): Promise<MiddlewareDecision> {
    if (ctx.kind !== "after_tool") return { kind: "continue" };
    let modified = false;
    const n = Math.min(ctx.calls.length, ctx.results.length);
    for (let i = 0; i < n; i += 1) {
      const call = ctx.calls[i]!;
      const result = ctx.results[i]!;
      if (call.name !== this.toolName) continue;
      const input = call.input as Record<string, unknown> | undefined;
      const pathVal = input && typeof input === "object" ? input["path"] : undefined;
      if (typeof pathVal !== "string" || pathVal.length === 0) continue;

      const next = (this.counts.get(pathVal) ?? 0) + 1;
      this.counts.set(pathVal, next);
      if (next >= this.threshold && !result.is_error) {
        const warning = `[loop-detection] ${pathVal} has been edited ${next} times — reconsider`;
        if (!result.content.includes("[loop-detection]")) {
          result.content = `${result.content}\n\n${warning}`;
          modified = true;
        }
      }
    }
    return modified ? { kind: "continue_with_modification" } : { kind: "continue" };
  }
  hooks(): HookPoint[] {
    return ["after_tool"];
  }
  name(): string {
    return this.myName;
  }
}

// ============================================================================
// PreCompletionChecklistMiddleware
// ============================================================================

/** Forces another turn at `before_completion` if the agent's response does
 *  not contain every required substring. Simplest possible reference impl. */
export class PreCompletionChecklistMiddleware implements Middleware {
  constructor(
    private readonly requiredSubstrings: string[],
    private readonly myName: string = "pre-completion-checklist",
  ) {}

  async handle(ctx: HookContext): Promise<MiddlewareDecision> {
    if (ctx.kind !== "before_completion") return { kind: "continue" };
    const missing = this.requiredSubstrings.filter((s) => !ctx.response.includes(s));
    if (missing.length === 0) return { kind: "continue" };
    return {
      kind: "force_another_turn",
      inject: `Verification incomplete. Required items not addressed: ${missing.join(", ")}`,
    };
  }
  hooks(): HookPoint[] {
    return ["before_completion"];
  }
  name(): string {
    return this.myName;
  }
}

// ============================================================================
// TokenBudgetMiddleware
// ============================================================================

/** Halts the session at `before_turn` when cumulative token spend hits the
 *  configured limit. Tests drive {@link record} directly; production wires
 *  this to `BudgetSnapshot`. */
export class TokenBudgetMiddleware implements Middleware {
  private spent = 0;

  constructor(
    private readonly limitTokens: number,
    private readonly myName: string = "token-budget",
  ) {}

  record(tokens: number): void {
    this.spent += tokens;
  }
  spentTokens(): number {
    return this.spent;
  }

  async handle(_ctx: HookContext): Promise<MiddlewareDecision> {
    if (this.spent >= this.limitTokens) {
      return {
        kind: "halt",
        reason: `token budget exhausted: ${this.spent}/${this.limitTokens}`,
      };
    }
    return { kind: "continue" };
  }
  hooks(): HookPoint[] {
    return ["before_turn"];
  }
  name(): string {
    return this.myName;
  }
}
