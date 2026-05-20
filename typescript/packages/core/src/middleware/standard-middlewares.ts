/**
 * Standard middleware implementations (spore-core issue #11).
 *
 * Subset matching the Rust reference. Production deployments compose the
 * remaining middlewares (`PermissionMiddleware`, `PiiRedactionMiddleware`,
 * `RateLimitMiddleware`, `DirectoryMapMiddleware`, `TimeBudgetMiddleware`)
 * once the components they depend on land.
 */

import { SessionId, TaskId } from "../harness/types.js";
import { Timestamp } from "../memory/types.js";
import {
  newPatchSpan,
  type ObservabilityProvider,
  type PatchType,
  type SpanBase,
} from "../observability/types.js";
import { SpanId } from "../observability/types.js";
import type { ToolCall } from "../model/schemas.js";

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

/**
 * Repairs empty / whitespace-only tool-call names to a configured fallback.
 * Runs at the lowest BeforeTool priority so downstream middleware see clean
 * calls (per spec).
 *
 * ## Observability (issue #28)
 *
 * This middleware is an always-on, highest-priority action mutator. To keep it
 * from silently rewriting calls, **every patch emits a warn-level
 * {@link PatchSpan}** via an injected {@link ObservabilityProvider} before the
 * patched call proceeds. The span carries the original and patched parameters
 * and a classified {@link PatchType} so the trace shows the diff, never just
 * the patched call.
 *
 * The shared `before_tool` {@link HookContext} does not carry `session_id` /
 * `task_id`, so this middleware captures identity at `before_session` into a
 * private field and reads it at `before_tool` — the same external-identity
 * pattern used by other middleware.
 */
export class PatchToolCallsMiddleware implements Middleware {
  /** Captured at `before_session`, read at `before_tool`. */
  private identity: { sessionId: SessionId; taskId: TaskId } | undefined;
  /** Monotonic counter so emitted patch spans get distinct ids. */
  private patchSeq = 0;

  constructor(
    private readonly fallbackName: string,
    /** Optional observability sink. `undefined` keeps the middleware a no-op
     *  observer when unwired and tests that don't care about spans simple. */
    private readonly observability?: ObservabilityProvider,
    private readonly myName: string = "patch-tool-calls",
  ) {}

  /** Tests expose this to simulate session boundaries. */
  clear(): void {
    this.identity = undefined;
  }

  async handle(ctx: HookContext): Promise<MiddlewareDecision> {
    if (ctx.kind === "before_session") {
      this.identity = { sessionId: ctx.session_id, taskId: ctx.task.id };
      return { kind: "continue" };
    }
    if (ctx.kind !== "before_tool") return { kind: "continue" };
    let modified = false;
    for (const call of ctx.calls) {
      if (call.name.trim().length === 0) {
        // Capture the original parameters before mutating the call.
        const original = call.input;
        call.name = this.fallbackName;
        modified = true;
        // R1/R4: classify the empty-name repair as a dangling tool call and
        // emit a warn-level event recording both the original and the patched
        // parameters.
        this.emitPatchEvent(call, original, {
          kind: "dangling_tool_call",
          reason: "empty tool name",
        });
      }
    }
    return modified ? { kind: "continue_with_modification" } : { kind: "continue" };
  }

  private emitPatchEvent(call: ToolCall, original: unknown, patchType: PatchType): void {
    const obs = this.observability;
    if (!obs) return;
    // Identity captured at before_session; fall back to empty ids if a tool
    // patch somehow fires before any session began (defensive — the span still
    // records the diff).
    const sessionId = this.identity?.sessionId ?? SessionId.of("");
    const taskId = this.identity?.taskId ?? TaskId.of("");
    const seq = this.patchSeq;
    this.patchSeq += 1;
    const ts = Timestamp.of("");
    const base: SpanBase = {
      span_id: SpanId.of(`patch-${seq}`),
      parent_span_id: null,
      session_id: sessionId,
      task_id: taskId,
      kind: "patch",
      started_at: ts,
      ended_at: ts,
      duration_ms: 0,
      status: { kind: "ok" },
    };
    obs.emitPatch(newPatchSpan(base, call.id, call.name, original, call.input, patchType));
  }

  hooks(): HookPoint[] {
    return ["before_session", "before_tool"];
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
