/**
 * Lifecycle hook system — canonical types (spore-core issue #69).
 *
 * A general-purpose extension layer that lets external code observe and shape
 * the harness at well-defined lifecycle moments. This is a NEW higher-level
 * sibling of {@link "../middleware/types.js".MiddlewareChain}: middleware shapes
 * the context block DURING assembly (a lower-level primitive); hooks fire at a
 * higher level on the already-assembled artifacts. The two layers are
 * intentionally distinct and this module does NOT modify or subsume middleware.
 *
 * The Rust reference (`rust/crates/spore-core/src/hooks.rs`) borrows mutable
 * fields `&mut` so pre-hooks rewrite them in place. TypeScript has no borrow
 * checker, so the idiomatic equivalent is a discriminated-union
 * {@link HookContext} whose mutable fields are mutated on the live object (the
 * same in-place pattern {@link "../middleware/types.js".HookContext} already
 * uses). The Rust `unsafe transmute` reborrow in `fire` is a borrow-checker
 * artifact with NO TypeScript counterpart.
 *
 * ## The 17 events (mutation / blocking / sync classification)
 *
 * | Event                | Pre/Post | Mutates              | Can block | Sync mode      |
 * |----------------------|----------|----------------------|-----------|----------------|
 * | `pre_turn`           | pre      | context_block        | yes       | sync           |
 * | `post_turn`          | post     | —                    | no        | sync or async  |
 * | `pre_tool_use`       | pre      | tool_input (or deny) | yes       | sync           |
 * | `post_tool_use`      | post     | —                    | no        | sync or async  |
 * | `post_tool_use_failure` | post  | —                    | no        | sync or async  |
 * | `post_tool_batch`    | post     | —                    | yes       | sync           |
 * | `on_loop_start`      | pre      | task_instruction     | yes       | sync           |
 * | `stop`               | post     | —                    | yes       | **sync only**  |
 * | `on_pause`           | post     | —                    | no        | **async only** |
 * | `on_resume`          | pre      | task_instruction     | no        | sync           |
 * | `on_error`           | post     | — (can suppress)     | yes       | sync or async  |
 * | `on_plan_created`    | post     | plan                 | yes       | sync           |
 * | `on_task_advance`    | pre      | task                 | yes       | sync           |
 * | `on_subagent_spawn`  | pre      | child_task (or deny) | yes       | sync           |
 * | `on_subagent_complete` | post   | —                    | no        | sync or async  |
 * | `pre_compact`        | pre      | preserve_hints       | yes       | sync           |
 * | `post_compact`       | post     | —                    | no        | async ok       |
 *
 * ## Rules enforced (R1–R26 from issue #69)
 * - R1  Pre-hooks may mutate the single mutable field of their context.
 * - R2  Pre-hook chains thread the mutated value to the next hook.
 * - R3  Hooks fire in REGISTRATION order (not middleware-style priority).
 * - R4  `block` is only legal on a can-block event.
 * - R5  `deny` is only legal on `pre_tool_use` / `on_subagent_spawn`.
 * - R6  `mutate` is only legal on a pre-event (and replaces the mutable field).
 * - R7  `inject` injects into the next turn's context block.
 * - R8  `stop` is SYNC ONLY — registering it async is rejected.
 * - R9  `on_pause` is ASYNC ONLY — registering it sync is rejected.
 * - R10 A sync post-hook block stops the chain and is reported to the loop.
 * - R11 Async post-hooks are fire-and-forget: spawned, never awaited, result
 *   and failure swallowed.
 * - R12 Stop `block` injects `reason` into the next turn via the same path
 *   `force_another_turn` used, and the loop continues.
 * - R13 Stop all-`continue` (or no hooks) terminates normally.
 * - R14 After `maxStopBlocks` consecutive Stop blocks in a run, the loop
 *   terminates anyway (per-run counter; resume starts fresh).
 * - R15 `pre_tool_use` deny rejects the tool call.
 * - R16 `pre_tool_use` may mutate `tool_input`.
 * - R17 Registering a hook for an event it cannot legally decide on is rejected
 *   at register time.
 * - R18 Command handler stdin = `{"event":"<snake_case>","context":<payload>}`.
 * - R19 Command handler stdout parsed as a tagged {@link HookDecision}.
 * - R20 Command nonzero exit → {@link HookError} `command_failed` (explicit
 *   error, NOT an implicit block).
 * - R21 Command malformed stdout → {@link HookError} `command_output_invalid`.
 * - R22 No sandbox, no timeout on command handlers in v1.
 * - R23 Function handler runs an inline closure synchronously.
 * - R24 Decision validity is checked at fire time as well as register time.
 * - R25 A hook that lists multiple events only fires for the event it is
 *   invoked with.
 * - R26 Firing order on Stop: registered Stop hooks first, THEN (when wired)
 *   the strategy verifier; either can block.
 *
 * ## Loop-wiring status
 *
 * Events whose loop machinery EXISTS and are wired into the ReAct loop in
 * `harness/standard.ts`: `stop` (the live one). The other turn/tool/compaction
 * events have fire methods that are unit-tested in isolation; only `stop` has a
 * live call site in the current loop (mirroring the Rust harness, which wires
 * Stop into the live ReAct loop).
 *
 * Events DEFINED-AND-UNIT-TESTED but NOT YET loop-wired (their strategy /
 * subagent / pause machinery is deferred elsewhere): `on_pause`, `on_resume`,
 * `on_plan_created`, `on_task_advance`, `on_subagent_spawn`,
 * `on_subagent_complete`. Each is exercised directly by unit tests.
 */

import type { Context } from "../agent/types.js";
import type {
  CompactionPreserveHints,
  SessionState as ContextSessionState,
} from "../context/types.js";
import type { PausedState, SessionId, Task } from "../harness/types.js";

// Forward-declared HarnessConfig reference for the `on_loop_start` context.
// Typed structurally as `unknown` to avoid a cycle with `harness/standard.ts`
// (which imports this module); the loop passes the live config object through.
export type HarnessConfigRef = unknown;

// ============================================================================
// Locally-defined payload types
//
// These artifacts are not yet modelled elsewhere in the package (TurnOutput,
// PlanArtifact, ToolCallSummary). `ContextBlock` reuses the agent's assembled
// `Context`; the rest are defined minimally here. They are intentionally
// additive — when the owning strategy / subagent issues land, the canonical
// shapes will replace these.
// ============================================================================

/** The assembled per-turn context a `pre_turn` hook may rewrite. */
export type ContextBlock = Context;

/** The output of a single completed turn, handed to post-turn / Stop hooks. */
export interface TurnOutput {
  /** The agent's final textual output for the turn (empty for tool turns). */
  text: string;
  /** Whether the turn requested tool calls rather than a final response. */
  had_tool_calls: boolean;
}

export function emptyTurnOutput(): TurnOutput {
  return { text: "", had_tool_calls: false };
}

/** A composite-strategy plan artifact, handed to `on_plan_created`. */
export interface PlanArtifact {
  tasks: string[];
  rationale: string;
}

/** A one-line summary of a tool call in a batch, handed to `post_tool_batch`. */
export interface ToolCallSummary {
  tool_name: string;
  succeeded: boolean;
}

// ============================================================================
// HookEvent
// ============================================================================

/** The 17 lifecycle events at which a {@link Hook} can fire (snake_case wire). */
export type HookEvent =
  | "pre_turn"
  | "post_turn"
  | "pre_tool_use"
  | "post_tool_use"
  | "post_tool_use_failure"
  | "post_tool_batch"
  | "on_loop_start"
  | "stop"
  | "on_pause"
  | "on_resume"
  | "on_error"
  | "on_plan_created"
  | "on_task_advance"
  | "on_subagent_spawn"
  | "on_subagent_complete"
  | "pre_compact"
  | "post_compact";

/** All 17 events, in catalogue order. */
export const ALL_HOOK_EVENTS: readonly HookEvent[] = [
  "pre_turn",
  "post_turn",
  "pre_tool_use",
  "post_tool_use",
  "post_tool_use_failure",
  "post_tool_batch",
  "on_loop_start",
  "stop",
  "on_pause",
  "on_resume",
  "on_error",
  "on_plan_created",
  "on_task_advance",
  "on_subagent_spawn",
  "on_subagent_complete",
  "pre_compact",
  "post_compact",
];

const PRE_EVENTS = new Set<HookEvent>([
  "pre_turn",
  "pre_tool_use",
  "on_loop_start",
  "on_resume",
  "on_task_advance",
  "on_subagent_spawn",
  "pre_compact",
]);

const SYNC_ONLY_EVENTS = new Set<HookEvent>([
  "stop",
  "pre_turn",
  "pre_tool_use",
  "post_tool_batch",
  "on_loop_start",
  "on_resume",
  "on_plan_created",
  "on_task_advance",
  "on_subagent_spawn",
  "pre_compact",
]);

const ASYNC_ONLY_EVENTS = new Set<HookEvent>(["on_pause", "post_compact"]);

const CAN_BLOCK_EVENTS = new Set<HookEvent>([
  "pre_turn",
  "post_tool_batch",
  "on_loop_start",
  "stop",
  "on_error",
  "on_plan_created",
  "on_task_advance",
]);

const CAN_DENY_EVENTS = new Set<HookEvent>(["pre_tool_use", "on_subagent_spawn"]);

/** Whether this is a pre-event (fires before its action; may mutate). */
export function hookEventIsPre(event: HookEvent): boolean {
  return PRE_EVENTS.has(event);
}

/** Whether this event carries a mutable field a pre-hook may rewrite.
 *  Equivalent to {@link hookEventIsPre} — every pre-event is mutable. */
export function hookEventIsMutable(event: HookEvent): boolean {
  return hookEventIsPre(event);
}

/** Whether this event may only run synchronously. */
export function hookEventIsSyncOnly(event: HookEvent): boolean {
  return SYNC_ONLY_EVENTS.has(event);
}

/** Whether this event may only run asynchronously (fire-and-forget). */
export function hookEventIsAsyncOnly(event: HookEvent): boolean {
  return ASYNC_ONLY_EVENTS.has(event);
}

/** Whether a hook on this event may return a `block` decision. */
export function hookEventCanBlock(event: HookEvent): boolean {
  return CAN_BLOCK_EVENTS.has(event);
}

/** Whether a hook on this event may return a `deny` decision. */
export function hookEventCanDeny(event: HookEvent): boolean {
  return CAN_DENY_EVENTS.has(event);
}

// ============================================================================
// HookSync
// ============================================================================

/** Whether a hook runs synchronously (blocking, result observed) or
 *  asynchronously (fire-and-forget). */
export type HookSync = "sync" | "async";

// ============================================================================
// HookDecision — serde-tagged on `decision`, snake_case values
// ============================================================================

/**
 * The control a hook exerts when it fires. Wire format is tagged on `decision`,
 * e.g. `{"decision":"block","reason":"x"}`. Round-trips byte-identically with
 * `fixtures/hooks/hook_decision_wire.json`.
 */
export type HookDecision =
  /** Proceed; no change. */
  | { decision: "continue" }
  /** Can-block events only — injects `reason` into the next turn. */
  | { decision: "block"; reason: string }
  /** Injects `context` into the next turn's context block. */
  | { decision: "inject"; context: string }
  /** `pre_tool_use` / `on_subagent_spawn` only — rejects the action. */
  | { decision: "deny"; reason: string }
  /** Pre-hooks only — replaces the mutable field with `data`. */
  | { decision: "mutate"; data: unknown };

/** Validate that `decision` is legal for `event`. Used at both register time
 *  (against a hook's declared events) and fire time. Returns `null` when legal
 *  or a {@link HookError} when not. */
export function validateDecisionForEvent(
  decision: HookDecision,
  event: HookEvent,
): HookError | null {
  let ok: boolean;
  switch (decision.decision) {
    case "continue":
    case "inject":
      ok = true;
      break;
    case "block":
      ok = hookEventCanBlock(event);
      break;
    case "deny":
      ok = hookEventCanDeny(event);
      break;
    case "mutate":
      ok = hookEventIsMutable(event);
      break;
    default: {
      const _exhaustive: never = decision;
      ok = false;
      void _exhaustive;
      break;
    }
  }
  return ok ? null : HookError.illegalDecision(event, decision.decision);
}

// ============================================================================
// HookError
// ============================================================================

export type HookErrorKind =
  | { kind: "illegal_decision"; event: HookEvent; decision: string }
  | { kind: "sync_only_event"; hook: string; event: HookEvent }
  | { kind: "async_only_event"; hook: string; event: HookEvent }
  | { kind: "command_failed"; command: string; code: number; stderr: string }
  | { kind: "command_output_invalid"; command: string; detail: string }
  | { kind: "handler_failed"; hook: string; detail: string };

export class HookError extends Error {
  override readonly name = "HookError";
  readonly kind: HookErrorKind["kind"];
  readonly detail: HookErrorKind;

  constructor(detail: HookErrorKind) {
    super(hookErrorMessage(detail));
    this.kind = detail.kind;
    this.detail = detail;
  }

  static illegalDecision(event: HookEvent, decision: string): HookError {
    return new HookError({ kind: "illegal_decision", event, decision });
  }
  static syncOnlyEvent(hook: string, event: HookEvent): HookError {
    return new HookError({ kind: "sync_only_event", hook, event });
  }
  static asyncOnlyEvent(hook: string, event: HookEvent): HookError {
    return new HookError({ kind: "async_only_event", hook, event });
  }
  static commandFailed(command: string, code: number, stderr: string): HookError {
    return new HookError({ kind: "command_failed", command, code, stderr });
  }
  static commandOutputInvalid(command: string, detail: string): HookError {
    return new HookError({ kind: "command_output_invalid", command, detail });
  }
  static handlerFailed(hook: string, detail: string): HookError {
    return new HookError({ kind: "handler_failed", hook, detail });
  }
}

function hookErrorMessage(e: HookErrorKind): string {
  switch (e.kind) {
    case "illegal_decision":
      return `hook '${e.decision}' decision is illegal for event '${e.event}'`;
    case "sync_only_event":
      return `hook '${e.hook}' cannot register for sync-only event '${e.event}' as async`;
    case "async_only_event":
      return `hook '${e.hook}' cannot register for async-only event '${e.event}' as sync`;
    case "command_failed":
      return `command hook '${e.command}' exited with status ${e.code}: ${e.stderr}`;
    case "command_output_invalid":
      return `command hook '${e.command}' produced invalid stdout: ${e.detail}`;
    case "handler_failed":
      return `hook '${e.hook}' failed: ${e.detail}`;
  }
}

// ============================================================================
// HookContext — discriminated union; one variant per event
//
// Mutable fields are mutated in place on the live object so pre-hooks rewrite
// them directly (the same in-place pattern the middleware HookContext uses).
// ============================================================================

export type HookContext =
  | {
      event: "pre_turn";
      session_id: SessionId;
      turn_number: number;
      /** Mutable: a pre-hook may rewrite this in place or via `mutate`. */
      context_block: ContextBlock;
    }
  | {
      event: "post_turn";
      session_id: SessionId;
      turn_number: number;
      output: TurnOutput;
    }
  | {
      event: "pre_tool_use";
      session_id: SessionId;
      turn_number: number;
      tool_name: string;
      /** Mutable. */
      tool_input: unknown;
    }
  | {
      event: "post_tool_use";
      session_id: SessionId;
      turn_number: number;
      tool_name: string;
      tool_input: unknown;
      tool_response: unknown;
      duration_ms: number;
    }
  | {
      event: "post_tool_use_failure";
      session_id: SessionId;
      turn_number: number;
      tool_name: string;
      tool_input: unknown;
      error: string;
      duration_ms: number;
    }
  | {
      event: "post_tool_batch";
      session_id: SessionId;
      turn_number: number;
      tool_calls: ToolCallSummary[];
    }
  | {
      event: "on_loop_start";
      session_id: SessionId;
      /** Mutable. */
      task_instruction: string;
      config: HarnessConfigRef;
    }
  | {
      event: "stop";
      session_id: SessionId;
      turn_number: number;
      last_output: TurnOutput;
      task_instruction: string;
      session_state: ContextSessionState | null;
    }
  | {
      event: "on_pause";
      session_id: SessionId;
      turn_number: number;
    }
  | {
      event: "on_resume";
      session_id: SessionId;
      /** Mutable. */
      task_instruction: string;
      paused_state: PausedState;
    }
  | {
      event: "on_error";
      session_id: SessionId;
      turn_number: number;
      error: string;
    }
  | {
      event: "on_plan_created";
      session_id: SessionId;
      /** Mutable. */
      plan: PlanArtifact;
    }
  | {
      event: "on_task_advance";
      session_id: SessionId;
      /** Mutable. */
      task: Task;
      task_index: number;
      total_tasks: number;
    }
  | {
      event: "on_subagent_spawn";
      session_id: SessionId;
      /** Mutable. */
      child_task: string;
      strategy: string;
    }
  | {
      event: "on_subagent_complete";
      session_id: SessionId;
      child_session_id: SessionId;
      result: unknown;
    }
  | {
      event: "pre_compact";
      session_id: SessionId;
      /** Mutable. */
      preserve_hints: CompactionPreserveHints;
    }
  | {
      event: "post_compact";
      session_id: SessionId;
      compact_summary: string;
    };

/** Which {@link HookEvent} a context corresponds to. */
export function hookEventOf(ctx: HookContext): HookEvent {
  return ctx.event;
}

/**
 * Serialize a context to the JSON payload a command handler receives on stdin
 * (the `context` field). Mutable fields are serialized by their current value.
 * Mirrors the Rust `to_payload` field selection exactly so the cross-language
 * command-handler fixture round-trips.
 */
export function hookContextPayload(ctx: HookContext): Record<string, unknown> {
  switch (ctx.event) {
    case "pre_turn":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        context_block: ctx.context_block,
      };
    case "post_turn":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        output: ctx.output,
      };
    case "pre_tool_use":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        tool_name: ctx.tool_name,
        tool_input: ctx.tool_input,
      };
    case "post_tool_use":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        tool_name: ctx.tool_name,
        tool_input: ctx.tool_input,
        tool_response: ctx.tool_response,
        duration_ms: ctx.duration_ms,
      };
    case "post_tool_use_failure":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        tool_name: ctx.tool_name,
        tool_input: ctx.tool_input,
        error: ctx.error,
        duration_ms: ctx.duration_ms,
      };
    case "post_tool_batch":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        tool_calls: ctx.tool_calls,
      };
    case "on_loop_start":
      return {
        session_id: ctx.session_id,
        task_instruction: ctx.task_instruction,
      };
    case "stop":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        last_output: ctx.last_output,
        task_instruction: ctx.task_instruction,
        session_state: ctx.session_state,
      };
    case "on_pause":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
      };
    case "on_resume":
      return {
        session_id: ctx.session_id,
        task_instruction: ctx.task_instruction,
        paused_state: ctx.paused_state,
      };
    case "on_error":
      return {
        session_id: ctx.session_id,
        turn_number: ctx.turn_number,
        error: ctx.error,
      };
    case "on_plan_created":
      return {
        session_id: ctx.session_id,
        plan: ctx.plan,
      };
    case "on_task_advance":
      return {
        session_id: ctx.session_id,
        task: ctx.task,
        task_index: ctx.task_index,
        total_tasks: ctx.total_tasks,
      };
    case "on_subagent_spawn":
      return {
        session_id: ctx.session_id,
        child_task: ctx.child_task,
        strategy: ctx.strategy,
      };
    case "on_subagent_complete":
      return {
        session_id: ctx.session_id,
        child_session_id: ctx.child_session_id,
        result: ctx.result,
      };
    case "pre_compact":
      return {
        session_id: ctx.session_id,
        preserve_hints: ctx.preserve_hints,
      };
    case "post_compact":
      return {
        session_id: ctx.session_id,
        compact_summary: ctx.compact_summary,
      };
    default: {
      const _exhaustive: never = ctx;
      return _exhaustive;
    }
  }
}

/**
 * Apply a `mutate` decision's `data` to a context's single mutable field, in
 * place. Throws {@link HookError} `handler_failed` if `data` cannot be coerced
 * into the target shape, or `illegal_decision` if the event is not mutable.
 *
 * `task_instruction` / `child_task` coerce a JSON string directly, else
 * stringify the value as JSON text (mirrors Rust's `string_from_value`).
 */
export function applyHookMutation(ctx: HookContext, hookName: string, data: unknown): void {
  const stringFromValue = (v: unknown): string => (typeof v === "string" ? v : JSON.stringify(v));

  switch (ctx.event) {
    case "pre_turn":
      ctx.context_block = data as ContextBlock;
      return;
    case "pre_tool_use":
      ctx.tool_input = data;
      return;
    case "on_loop_start":
    case "on_resume":
      ctx.task_instruction = stringFromValue(data);
      return;
    case "on_plan_created":
      ctx.plan = data as PlanArtifact;
      return;
    case "on_task_advance":
      ctx.task = data as Task;
      return;
    case "on_subagent_spawn":
      ctx.child_task = stringFromValue(data);
      return;
    case "pre_compact":
      ctx.preserve_hints = data as CompactionPreserveHints;
      return;
    default:
      throw HookError.illegalDecision(ctx.event, "mutate");
  }
}

// ============================================================================
// FireOutcome
// ============================================================================

/** Outcome of firing a chain back to the harness loop. */
export type FireOutcome =
  /** All hooks said continue (possibly after mutating in place). */
  | { kind: "continue" }
  /** A hook blocked; `reason` is to be injected into the next turn. */
  | { kind: "block"; reason: string }
  /** A hook denied the action (`pre_tool_use` / `on_subagent_spawn`). */
  | { kind: "deny"; reason: string }
  /** Hooks requested context injection; the newline-joined text follows. */
  | { kind: "inject"; context: string };

// ============================================================================
// Hook + HookChain interfaces
// ============================================================================

/** A single lifecycle hook handler. */
export interface Hook {
  /** Handle one firing. Pre-hooks may mutate the context's mutable field
   *  directly OR return a `mutate` decision. Async to allow IO. */
  handle(ctx: HookContext, signal?: AbortSignal): Promise<HookDecision>;
  /** The events this hook subscribes to. Must be non-empty in practice. */
  events(): HookEvent[];
  /** A stable name for diagnostics and error messages. */
  name(): string;
  /** Whether this hook runs sync (blocking) or async (fire-and-forget).
   *  Defaults to `"sync"` when omitted. */
  syncMode?(): HookSync;
}

/** Registry + dispatcher for {@link Hook}s. Implementations fan out to all
 *  hooks subscribed to an event in registration order. */
export interface HookChain {
  /** Register a hook. Throws {@link HookError} when a sync-only event is
   *  registered async (or an async-only event sync). Registration order is
   *  firing order. */
  register(hook: Hook): void;

  /** Fire the chain for `ctx`. Mutations thread through `ctx` in place; the
   *  aggregate outcome is returned (first block/deny wins; injects are
   *  newline-joined). */
  fire(ctx: HookContext, signal?: AbortSignal): Promise<FireOutcome>;
}
