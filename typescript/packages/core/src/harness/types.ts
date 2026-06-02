/**
 * Public types for the Harness runtime loop (spore-core issue #3).
 *
 * The wire shape mirrors the Rust reference implementation byte-for-byte:
 * tagged unions use a `kind` discriminator in `snake_case`. Static types
 * are derived from zod schemas for safe (de)serialization of `PausedState`
 * and `RunResult` across pause/resume boundaries.
 *
 * Component dependencies (#4–#13) ship in their own issues. Until those
 * land, this module defines minimal forward declarations of the trait
 * surface the loop consumes — each tagged with the owning issue.
 */

import { z } from "zod";

import type { Context } from "../agent/types.js";
import type { AgentError } from "../agent/errors.js";
import type { PlanPhaseErrorKind } from "../plan/types.js";
import type { PlanArtifact } from "../plan/types.js";
import type { Mode } from "../prompt-chunk-registry/types.js";
import type {
  CompactionPreserveHints,
  ContextError,
  SessionState as ContextSessionState,
} from "../context/types.js";
import {
  MessageSchema,
  ToolCallSchema,
  type Message,
  type StreamEvent as ModelStreamEvent,
  type TokenUsage,
  type ToolCall,
  type ToolSchema,
} from "../model/schemas.js";

// ============================================================================
// Identity newtypes
// ============================================================================

let __idCounter = 0;
function randomId(): string {
  __idCounter += 1;
  return __idCounter.toString(16).padStart(16, "0");
}

export class SessionId {
  constructor(readonly value: string) {}
  static of(value: string): SessionId {
    return new SessionId(value);
  }
  /** Fresh, opaque session id (used e.g. by SelfVerifying evaluator). */
  static generate(): SessionId {
    return new SessionId(`sess-${randomId()}`);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: SessionId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export class TaskId {
  constructor(readonly value: string) {}
  static of(value: string): TaskId {
    return new TaskId(value);
  }
  static generate(): TaskId {
    return new TaskId(`task-${randomId()}`);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: TaskId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

// Schemas (string) for SessionId / TaskId on the wire.
export const SessionIdSchema = z.string().transform((s) => new SessionId(s));
export const TaskIdSchema = z.string().transform((s) => new TaskId(s));

// ============================================================================
// Budget tracking
// ============================================================================

export const BudgetLimitsSchema = z.object({
  max_turns: z.number().int().nonnegative().nullable().optional(),
  max_input_tokens: z.number().int().nonnegative().nullable().optional(),
  max_output_tokens: z.number().int().nonnegative().nullable().optional(),
  /** Wall time limit, seconds. */
  max_wall_time: z.number().nonnegative().nullable().optional(),
  max_cost_usd: z.number().nonnegative().nullable().optional(),
});
export type BudgetLimits = z.infer<typeof BudgetLimitsSchema>;

export const BudgetLimitTypeSchema = z.enum([
  "turns",
  "input_tokens",
  "output_tokens",
  "wall_time",
  "cost_usd",
]);
export type BudgetLimitType = z.infer<typeof BudgetLimitTypeSchema>;

export const BudgetSnapshotSchema = z.object({
  turns: z.number().int().nonnegative().default(0),
  input_tokens: z.number().int().nonnegative().default(0),
  output_tokens: z.number().int().nonnegative().default(0),
  wall_time: z.number().nonnegative().nullable().optional(),
  cost_usd: z.number().nonnegative().default(0),
});
export type BudgetSnapshot = z.infer<typeof BudgetSnapshotSchema>;

export function emptyBudgetSnapshot(): BudgetSnapshot {
  return { turns: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0 };
}

export const AggregateUsageSchema = z.object({
  input_tokens: z.number().int().nonnegative().default(0),
  output_tokens: z.number().int().nonnegative().default(0),
  cache_read_tokens: z.number().int().nonnegative().default(0),
  cache_write_tokens: z.number().int().nonnegative().default(0),
  cost_usd: z.number().nonnegative().default(0),
});
export type AggregateUsage = z.infer<typeof AggregateUsageSchema>;

export function emptyAggregateUsage(): AggregateUsage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cost_usd: 0,
  };
}

export function addTurnUsage(agg: AggregateUsage, u: TokenUsage): void {
  agg.input_tokens += u.input_tokens;
  agg.output_tokens += u.output_tokens;
  agg.cache_read_tokens += u.cache_read_tokens ?? 0;
  agg.cache_write_tokens += u.cache_write_tokens ?? 0;
}

// ============================================================================
// Task + loop strategy
// ============================================================================

export const OptimizationDirectionSchema = z.enum(["minimize", "maximize"]);
export type OptimizationDirection = z.infer<typeof OptimizationDirectionSchema>;

export const ModelConfigSchema = z.object({
  provider: z.string(),
  model_id: z.string(),
});
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

/**
 * Loop strategy. Data shapes are canonical; only `react` is fully executable
 * in {@link StandardHarness}. Other variants return {@link HaltReason} of
 * `strategy_not_yet_implemented` until the trait dependencies (CompletionCheck,
 * Verifier, MetricEvaluator) ship in later component issues.
 */
export const LoopStrategySchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("re_act"), max_iterations: z.number().int().nonnegative() }),
  z.object({
    kind: z.literal("plan_execute"),
    plan_model: ModelConfigSchema.nullable().optional(),
  }),
  z.object({ kind: z.literal("ralph") }),
  z.object({ kind: z.literal("self_verifying") }),
  z.object({
    kind: z.literal("hill_climbing"),
    direction: OptimizationDirectionSchema,
    max_stagnation: z.number().int().nonnegative().nullable().optional(),
    revert_on_no_improvement: z.boolean(),
    min_improvement_delta: z.number().nullable().optional(),
  }),
]);
export type LoopStrategy = z.infer<typeof LoopStrategySchema>;

export const TaskSchema = z.object({
  id: z.string().transform((s) => new TaskId(s)),
  instruction: z.string(),
  session_id: z.string().transform((s) => new SessionId(s)),
  budget: BudgetLimitsSchema,
  loop_strategy: LoopStrategySchema,
});
export type Task = {
  id: TaskId;
  instruction: string;
  session_id: SessionId;
  budget: BudgetLimits;
  loop_strategy: LoopStrategy;
};

export function newTask(
  instruction: string,
  session_id: SessionId,
  loop_strategy: LoopStrategy,
  budget: BudgetLimits = {},
): Task {
  return {
    id: TaskId.generate(),
    instruction,
    session_id,
    budget,
    loop_strategy,
  };
}

// ============================================================================
// Streaming events emitted by the harness
// ============================================================================

/**
 * The kind of content block a {@link HarnessStreamEvent} `block_start` opens
 * (issue #103, resolved spec decision **Q2**: a single generic frame marker
 * carrying a `BlockKind` rather than typed-per-kind markers).
 */
export type BlockKind = "text" | "reasoning" | "tool_use";

/**
 * Events the harness surfaces on {@link StreamSink}.
 *
 * ## Delta-level streaming (issue #103)
 *
 * The harness maps each raw model `StreamEvent` produced by the agent through
 * `mapModelStreamEvent` into zero or more of the delta/frame variants below,
 * alongside the existing coarse lifecycle events. Resolution notes:
 *
 *   - **Q2**: frame markers are the generic `block_start` / `block_stop`
 *     carrying a {@link BlockKind}, NOT typed-per-kind markers.
 *   - **Q3**: model `message_start` / `message_stop` are DROPPED at the
 *     harness boundary. `turn_start` / `turn_end` already cover this.
 *   - **Q5**: the coarse `tool_call` now also carries the final `args`, and
 *     `tool_result` the result `content`. Both new fields are optional so
 *     pre-#103 serialized events round-trip.
 *
 * Tool lifecycle ordering per call:
 *   `tool_call_start` → `tool_args_delta`* → (`block_stop`) → coarse `tool_call`.
 */
export type HarnessStreamEvent =
  | { kind: "turn_start"; turn: number }
  | { kind: "turn_end"; turn: number }
  | {
      kind: "tool_call";
      call_id: string;
      name: string;
      /** Final, fully-accumulated tool-call arguments (issue #103, Q5). */
      args?: unknown;
    }
  | {
      kind: "tool_result";
      call_id: string;
      is_error: boolean;
      /** The tool result content (issue #103, Q5). */
      content?: string;
    }
  | { kind: "final_response"; content: string }
  | { kind: "budget_warning"; limit_type: BudgetLimitType }
  /**
   * Emitted by the harness loop when it dispatches the `send_message` tool
   * (issue #81 — `SendMessageTool`). The loop surfaces the message content as
   * this prominent event instead of collapsing it into a normal tool result;
   * rendering it prominently is the architect's UI concern. A minimal success
   * tool result is still recorded in history so the loop continues.
   */
  | { kind: "user_message"; content: string }
  // ── Delta-level streaming (issue #103) ──────────────────────────────────
  /** Streamed text fragment (model `content_block_delta`). */
  | { kind: "text_delta"; content: string }
  /** Streamed reasoning/thinking fragment (model `thinking_delta`). Q4. */
  | { kind: "reasoning_delta"; content: string }
  /**
   * Streamed tool-argument JSON fragment (model `tool_use_delta`), correlated
   * to a `call_id` via the open-block index.
   */
  | { kind: "tool_args_delta"; call_id: string; partial_json: string }
  /** A content block opened (Q2). Emitted on the first delta for an index. */
  | { kind: "block_start"; index: number; block: BlockKind }
  /** A content block closed (model `content_block_stop`). Q2. */
  | { kind: "block_stop"; index: number }
  /**
   * A tool-use block opened (issue #103). Emitted so consumers can correlate
   * the subsequent `tool_args_delta` fragments and the final coarse
   * `tool_call` by `call_id`. The `name` may be empty when the underlying
   * model stream does not surface it before args (a documented limitation —
   * the name is recovered on the coarse `tool_call`).
   */
  | { kind: "tool_call_start"; index: number; call_id: string; name: string };

export type StreamSink = (event: HarnessStreamEvent) => void;

/**
 * Per-turn state threaded through `mapModelStreamEvent` (issue #103). Tracks
 * which block indices are open and their kind so `tool_use_delta` /
 * `content_block_stop` events correlate back to a `call_id`, and so each
 * block's `block_start` is emitted exactly once.
 */
export class TurnStreamState {
  /** Open block index → its {@link BlockKind}. */
  readonly openBlocks = new Map<number, BlockKind>();
  /** Tool-use block index → its derived `call_id` (`call_{index}`). */
  readonly toolCalls = new Map<number, string>();

  static callIdFor(index: number): string {
    return `call_${index}`;
  }
}

/**
 * Map one raw model `StreamEvent` to zero or more harness
 * {@link HarnessStreamEvent}s (issue #103), threading {@link TurnStreamState}
 * so blocks and tool calls are correlated across events.
 *
 * Rules:
 *   - Q2: a block's `block_start` is emitted exactly once, the first time a
 *     delta for that index is observed; `content_block_stop` maps to
 *     `block_stop`.
 *   - Q3: model `message_start` / `message_stop` map to nothing (dropped).
 *   - A tool-use block additionally emits `tool_call_start` on open, then each
 *     fragment as `tool_args_delta` keyed by the derived `call_id`.
 */
export function mapModelStreamEvent(
  event: ModelStreamEvent,
  state: TurnStreamState,
): HarnessStreamEvent[] {
  switch (event.type) {
    // Q3: dropped at the harness boundary.
    case "message_start":
    case "message_stop":
      return [];
    case "content_block_delta": {
      const out: HarnessStreamEvent[] = [];
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "text");
        out.push({ kind: "block_start", index: event.index, block: "text" });
      }
      out.push({ kind: "text_delta", content: event.delta });
      return out;
    }
    case "thinking_delta": {
      const out: HarnessStreamEvent[] = [];
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "reasoning");
        out.push({ kind: "block_start", index: event.index, block: "reasoning" });
      }
      out.push({ kind: "reasoning_delta", content: event.delta });
      return out;
    }
    case "tool_use_delta": {
      const out: HarnessStreamEvent[] = [];
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "tool_use");
        const callId = TurnStreamState.callIdFor(event.index);
        state.toolCalls.set(event.index, callId);
        out.push({ kind: "block_start", index: event.index, block: "tool_use" });
        // Name is not carried by the model StreamEvent; recovered on the
        // coarse tool_call. Emit tool_call_start so consumers can begin
        // correlating args by call_id.
        out.push({ kind: "tool_call_start", index: event.index, call_id: callId, name: "" });
      }
      const callId = state.toolCalls.get(event.index) ?? TurnStreamState.callIdFor(event.index);
      out.push({ kind: "tool_args_delta", call_id: callId, partial_json: event.partial_json });
      return out;
    }
    case "content_block_stop":
      state.openBlocks.delete(event.index);
      return [{ kind: "block_stop", index: event.index }];
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
}

// ============================================================================
// Forward-declared sibling component types
// ============================================================================

/**
 * Tool Escalation Protocol — the typed channel by which a tool signals the
 * harness to terminate cleanly and pass a *structural* state change up to its
 * caller (issue #80).
 *
 * The harness is a pure intermediary: it never acts on a signal itself. Mode
 * switching, plan approval, and graceful abort are the caller's concern. The
 * harness terminates cleanly, surfaces the signal via the `escalate`
 * {@link RunResult}, and the caller (CLI, chat UI, REST API, parent harness)
 * owns the orchestration. This mirrors the `waiting_for_human` model — the
 * harness does not resume itself either.
 *
 * Variants:
 * - `enter_plan_mode` — agent requests entry into plan mode, carrying
 *   accumulated context as a seed for the planning harness.
 * - `exit_plan_mode` — planning agent's terminal signal, carrying the produced
 *   {@link PlanArtifact} for human approval before an execution harness is
 *   instantiated.
 * - `switch_mode` — agent requests a mode switch; carries the target
 *   {@link Mode} (the EXISTING mode enum — there is no separate `HarnessMode`).
 * - `abort` — agent requests a graceful, intentional stop with a reason.
 *   Distinct from a `HaltReason` `agent_error` — it surfaces as an `escalate`
 *   {@link RunResult}, NOT a `failure`.
 *
 * Wire format: tagged on `kind`, `snake_case`, byte-identical across the four
 * language implementations. Round-tripped by
 * `fixtures/harness/escalation_signals.json`.
 */
export const HarnessSignalSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("enter_plan_mode"), context: z.string() }),
  z.object({
    kind: z.literal("exit_plan_mode"),
    plan: z.object({ tasks: z.array(z.string()), rationale: z.string() }),
  }),
  z.object({
    kind: z.literal("switch_mode"),
    // `"yolo"` is intentionally absent — it is a dangerous-only mode (issue
    // #34) and cannot be named through the default build, mirroring the Rust
    // `SwitchMode { mode: Mode }` where `Mode::Yolo` does not exist without the
    // `dangerous` feature.
    mode: z.enum(["always_ask", "auto_edit", "plan", "safe_auto"]),
  }),
  z.object({ kind: z.literal("abort"), reason: z.string() }),
]);
export type HarnessSignal =
  | { kind: "enter_plan_mode"; context: string }
  | { kind: "exit_plan_mode"; plan: PlanArtifact }
  | { kind: "switch_mode"; mode: Mode }
  | { kind: "abort"; reason: string };

/**
 * Tool dispatch output. Full type lives in issue #4/#5; this covers loop routing.
 *
 * Prefer the {@link toolOutput} constructors over the object literals — they
 * spell out the common cases (`success` / recoverable `error` / `fatal`) and
 * document the field semantics below in one place.
 *
 *   - `truncated` (on `success`) — `true` ONLY when the tool itself clipped its
 *     output to fit an inline budget (large outputs routed through
 *     {@link SandboxProvider.handleLargeOutput} set this). Plain tool authors
 *     should leave it `false` (omit it) — use {@link toolOutput.success}.
 *   - `recoverable` (on `error`) — `true` if the agent may sensibly retry or
 *     adapt: the loop appends the error as a tool result and continues. `false`
 *     halts the run. Most tool failures are recoverable — prefer
 *     {@link toolOutput.error}; reach for {@link toolOutput.fatal} only when
 *     continuing is pointless.
 */
export type ToolOutput =
  | { kind: "success"; content: string; truncated?: boolean }
  | { kind: "error"; message: string; recoverable: boolean }
  | { kind: "waiting_for_human"; child_state: ChildPausedState; request: HumanRequest }
  /**
   * Tool requests a structural state change from the harness's parent (#80).
   * The harness terminates cleanly and passes the signal to the caller via the
   * `escalate` {@link RunResult}. NOT appended to message history.
   */
  | { kind: "escalate"; signal: HarnessSignal }
  /**
   * Tool asks the user a clarifying question (issue #81, Q4b —
   * `AskUserQuestionTool`). UNLIKE the subagent `waiting_for_human` path there
   * is NO {@link ChildPausedState}: the loop builds a {@link PausedState}
   * directly with `human_request` set to {@link HumanRequest} `clarification`,
   * preserves the clarifying call as the head of `pending_tool_calls`, and
   * returns a `waiting_for_human` {@link RunResult}. On resume the human's
   * answer text is injected as the tool RESULT for that clarifying call.
   */
  | { kind: "awaiting_clarification"; question: string; options?: string[] };

/**
 * Ergonomic constructors for the common {@link ToolOutput} cases. Mirrors Rust's
 * `ToolOutput::success` / `error` / `fatal` — see the field semantics on
 * {@link ToolOutput}.
 */
export const toolOutput = {
  /**
   * A successful, non-truncated result — the common case for a tool that
   * returns its full output. Saves spelling out `truncated: false`.
   */
  success(content: string): ToolOutput {
    return { kind: "success", content, truncated: false };
  },

  /**
   * A **recoverable** error: the harness loop appends it as a tool result and
   * lets the agent adapt or retry. The right default for almost every tool
   * failure (bad arguments, missing file, transient I/O).
   */
  error(message: string): ToolOutput {
    return { kind: "error", message, recoverable: true };
  },

  /**
   * A **fatal** error: continuing is pointless, so the run halts. Reserve for
   * genuinely unrecoverable conditions; prefer {@link toolOutput.error} when the
   * agent could reasonably do something different next turn.
   */
  fatal(message: string): ToolOutput {
    return { kind: "error", message, recoverable: false };
  },
} as const;

export interface ToolResultRecord {
  call_id: string;
  output: ToolOutput;
}

/** Sandbox violation — issue #6.
 *
 * Discriminated union, `kind` tag in `snake_case`. Wire-compatible with the
 * Rust `SandboxViolation` enum (which uses `#[serde(tag = "kind")]`).
 */
export type SandboxViolation =
  | { kind: "path_escape"; path: string }
  | { kind: "path_denied"; path: string; matched_rule: string }
  | { kind: "extension_denied"; path: string; extension: string }
  | { kind: "read_only_violation"; path: string }
  | { kind: "file_size_exceeded"; path: string; size: number; limit: number }
  | { kind: "disallowed_command"; command: string }
  | { kind: "network_violation"; host: string };

export function sandboxViolationIsAlwaysHalt(v: SandboxViolation): boolean {
  return v.kind === "path_escape" || v.kind === "network_violation";
}

// ============================================================================
// Sandbox isolation modes — issue #6
// ============================================================================

/** Read/write/execute operation tag — passed to `resolvePath`. */
export type Operation = "read" | "write" | "execute";

/** Bubblewrap profile — placeholder; backend not wired in v1. */
export interface BwrapProfile {
  /** Free-form profile name; semantics deferred to the backend. */
  name?: string;
}

/** Docker network policy. */
export type NetworkPolicy =
  | { kind: "none" }
  | { kind: "allowlist"; hosts: string[] }
  | { kind: "full" };

/**
 * Discriminated isolation mode (default build).
 *
 * `{ kind: "none" }` (no path enforcement) is a named safety footgun and is
 * **not** part of this type. It is reachable only through the dangerous opt-in
 * entry point (`@spore/core/dangerous`), which re-exposes it as
 * {@link DangerousIsolationMode}. This mirrors the Rust `dangerous` Cargo
 * feature that removes `IsolationMode::None` from the default build (issue #34).
 * The wire tag for the dangerous mode stays `"none"`.
 */
export type IsolationMode =
  | { kind: "workspace_scoped" }
  | { kind: "bubblewrap"; profile: BwrapProfile }
  | { kind: "docker"; image: string; network: NetworkPolicy };

/**
 * The dangerous-only isolation mode: no isolation, no path enforcement. Its
 * wire value is `{ kind: "none" }`, but the type is **branded**: a value can
 * only be minted by the dangerous opt-in entry point (`@spore/core/dangerous`).
 * A default-build caller cannot forge it by writing `{ kind: "none" }`, so the
 * dangerous sandbox constructor that consumes it is unreachable without
 * importing the dangerous module. Idiomatic TS analogue of the Rust `dangerous`
 * Cargo feature gating `IsolationMode::None` (issue #34).
 */
export type DangerousIsolationMode = { kind: "none" } & {
  readonly __sporeDangerous: unique symbol;
};

/**
 * Internal union of every isolation mode, including the dangerous
 * {@link DangerousIsolationMode}. Used by the sandbox switch bodies and by the
 * dangerous entry point. NOT part of the default public API — default callers
 * use {@link IsolationMode}, which cannot name the dangerous mode.
 */
export type AnyIsolationMode = IsolationMode | DangerousIsolationMode;

/** Configuration consumed by `WorkspaceScopedSandbox`. */
export interface WorkspaceConfig {
  /** Workspace root. Canonicalized at construction. */
  root: string;
  /** Explicit allowlist (relative to root, or absolute under root). */
  allowed_paths?: string[];
  /** Explicit denylist; evaluated after the allowlist. */
  denied_paths?: string[];
  /** Allowed extensions. `undefined` = allow all. Advisory in v1. */
  allowed_extensions?: string[];
  /** Denied extensions (e.g. `["env", "pem", "key"]`). Leading dot tolerated. */
  denied_extensions?: string[];
  /** If true, Write/Execute resolve to ReadOnlyViolation. */
  read_only?: boolean;
  /** Max file size (bytes) for reads. `0` disables. */
  max_file_size?: number;
}

export type HookPoint = "before_turn" | "before_tool" | "after_tool" | "before_completion";

export type TerminationDecision = { kind: "continue" } | { kind: "halt"; reason: string };

/**
 * Session state handed back and forth across pause/resume. The harness does
 * not interpret its contents; it round-trips opaquely so that #7
 * (ContextManager) and #8 (MemoryProvider) own the schema.
 */
export const SessionStateSchema = z.object({
  messages: z.array(MessageSchema).default([]),
  extras: z.record(z.unknown()).default({}),
});
export type SessionState = z.infer<typeof SessionStateSchema>;

export function emptySessionState(): SessionState {
  return { messages: [], extras: {} };
}

// ============================================================================
// Forward-declared sibling component interfaces
// ============================================================================

/** Issue #4 — ToolRegistry. */
export interface ToolRegistry {
  dispatch(call: ToolCall, signal?: AbortSignal): Promise<ToolOutput>;
  isAlwaysHalt(toolName: string): boolean;
  schemas(): ToolSchema[];
}

/** Output of a sandboxed command. */
export interface CommandOutput {
  stdout: string;
  stderr: string;
  exit_code: number;
  timed_out: boolean;
  /** Additive field (#6): true when stdout/stderr were truncated upstream. */
  truncated: boolean;
}

/** Reference to a file containing offloaded full content. */
export interface FileRef {
  path: string;
  size: number;
}

/** Result of routing oversized tool output through the sandbox (#6 shape). */
export interface TruncatedOutput {
  /** head + separator + tail (or original content when below threshold). */
  content: string;
  truncated: boolean;
  /** Path + size of the offloaded full content, when one was written. */
  full_ref: FileRef | null;
  /** Original (untruncated) size in characters. */
  original_size: number;
}

/** Issue #6 — SandboxProvider.
 *
 * `executeCommand`, `handleLargeOutput`, and `resolvePath` mirror the
 * Rust trait's defaulted methods (issue #5). They are optional here so
 * lightweight test stubs only need `validate`; tools fall back to
 * Node-based defaults when an implementation does not provide them.
 *
 * `isolationMode` and `workspaceRoot` are likewise optional for the same
 * reason — production sandboxes implement both.
 */
export interface SandboxProvider {
  /** Resolves with `null` on success, or a SandboxViolation. */
  validate(call: ToolCall, signal?: AbortSignal): Promise<SandboxViolation | null>;

  /** Execute a command (no shell). Default impl spawns via `child_process`. */
  executeCommand?(
    command: string,
    args: readonly string[],
    cwd?: string | null,
    timeoutMs?: number | null,
    signal?: AbortSignal,
  ): Promise<CommandOutput | SandboxViolation>;

  /** Truncate a large body head+tail. Default impl returns head+marker+tail. */
  handleLargeOutput?(
    content: string,
    callId: string,
    headTokens: number,
    tailTokens: number,
  ): Promise<TruncatedOutput>;

  /** Resolve a path against the workspace root. */
  resolvePath?(path: string, operation: Operation): Promise<string | SandboxViolation>;

  /**
   * Current isolation mode. Returns {@link AnyIsolationMode} so a sandbox
   * constructed through the dangerous opt-in can report `{ kind: "none" }`;
   * default-built sandboxes only ever return a safe {@link IsolationMode}.
   */
  isolationMode?(): AnyIsolationMode;

  /** Canonical workspace root. */
  workspaceRoot?(): string;
}

/**
 * Inputs the harness compaction loop (issue #46) needs to run one compaction
 * turn and verify its result.
 *
 * The harness loop operates on the opaque {@link SessionState} above; the rich
 * compaction/verification API ({@link "../context/types.js".ContextManager},
 * {@link "../context/types.js".CompactionVerifier}) operates on
 * {@link "../context/types.js".SessionState}. This struct is the bridge: a
 * compaction-capable {@link ContextManager} projects everything the loop needs
 * into one value, so the loop never has to know which concrete state type its
 * manager uses internally.
 *
 * `context` is fed straight to `Agent.turn` to produce the summary;
 * `preserveHints` and `verificationState` are passed to
 * {@link "../context/types.js".CompactionVerifier.verify}. On a verification
 * failure the loop re-runs the turn with {@link ContextManager.injectMissingItems}
 * applied to `context`.
 */
export interface CompactionTurn {
  /** Context to feed `Agent.turn` to elicit the summary. */
  context: Context;
  /** Preservation hints to hand the verifier. */
  preserveHints: CompactionPreserveHints;
  /** Verifier-facing session state (rich `context` `SessionState`). */
  verificationState: ContextSessionState;
  /** Messages about to be removed — used to stamp the compaction span. */
  messagesRemoved: number;
}

/**
 * Issue #7 — ContextManager.
 *
 * Issue #46 adds the optional compaction-loop surface
 * ({@link prepareCompactionTurn}, {@link injectMissingItems},
 * {@link applyCompaction}). All three are OPTIONAL so managers that do not
 * compact (the default `shouldCompact` returns `false`) need not implement
 * them; the harness applies the spec defaults when a method is absent.
 */
export interface ContextManager {
  assemble(session: SessionState, task: Task, signal?: AbortSignal): Promise<Context>;
  appendToolResult(session: SessionState, result: ToolResultRecord): Promise<void>;
  appendUserMessage(session: SessionState, text: string): Promise<void>;
  shouldCompact(session: SessionState): boolean;

  /** Append the assistant's turn (model output: text and/or the tool calls it
   *  requested) to the conversation so the next {@link assemble} reflects what
   *  the agent already did. Without this the model loses track of its own
   *  actions and the conversation is malformed (a tool result with no preceding
   *  assistant tool_use). OPTIONAL so test-double / non-recording managers need
   *  not implement it; the harness calls it via `?.` and treats absence as a
   *  no-op (mirrors the Rust trait's default-no-op method). */
  appendAssistantMessage?(session: SessionState, message: Message): Promise<void>;

  /** Build the inputs for one compaction turn (issue #46). Returns `undefined`
   *  when there is nothing to compact (e.g. history shorter than the preserve
   *  window), in which case the harness skips compaction entirely. Default
   *  (when absent): `undefined` — managers that never compact need not
   *  implement this. */
  prepareCompactionTurn?(session: SessionState): CompactionTurn | undefined;

  /** Mutate a compaction {@link Context} in place to request a revised summary
   *  on retry (issue #46). The harness calls this with the items the prior
   *  summary failed to preserve. Default (when absent): append the standard
   *  "Your summary is missing these items: {missing}. Please revise." user
   *  message. */
  injectMissingItems?(context: Context, missing: string[]): void;

  /** Accept a verified (or accepted-anyway) summary into the session, replacing
   *  the compacted span (issue #46). Default (when absent): no-op — only
   *  compaction-capable managers implement it. */
  applyCompaction?(session: SessionState, summary: string): void;

  /** Report the manager's current token-budget usage for the session, so the
   *  harness can stamp the post-compaction `tokens_after`/`tokens_reclaimed`
   *  on the compaction span with real values (issue #57 token-accounting fix).
   *  Default (when absent): `undefined` — the harness falls back to the
   *  pre-compaction budget. */
  tokenBudgetUsed?(session: SessionState): number | undefined;
}

/** Issue #13 — TerminationPolicy. */
export interface TerminationPolicy {
  evaluate(session: SessionState, budgetUsed: BudgetSnapshot): Promise<TerminationDecision>;
}

/** Issue #11 — Middleware decision. */
export type MiddlewareDecision =
  | { kind: "continue" }
  | { kind: "continue_with_modification"; calls: ToolCall[] }
  | { kind: "halt"; reason: string }
  | { kind: "surface_to_human"; request: HumanRequest };

export interface MiddlewareChain {
  fire(hook: HookPoint, session: SessionState): Promise<MiddlewareDecision>;
}

/** Issue #12 — ObservabilityProvider. Re-exported from the canonical
 *  definition in {@link ../observability/types.js} so the harness loop and the
 *  observability backends share one interface (emitTurn / emitToolCall / … /
 *  setSessionOutcome / flushSession). */
export type { ObservabilityProvider } from "../observability/types.js";

// ============================================================================
// Human-in-the-loop
// ============================================================================

export const RiskLevelSchema = z.enum(["low", "medium", "high", "critical"]);
export type RiskLevel = z.infer<typeof RiskLevelSchema>;

export const HumanRequestSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("tool_approval"),
    calls: z.array(ToolCallSchema),
    risk_level: RiskLevelSchema,
  }),
  z.object({
    kind: z.literal("clarification"),
    question: z.string(),
    // Optional (#81, Q4b): multiple-choice options carried with the
    // clarification. Absent on old `clarification` blobs (back-compat
    // deserialization) and on free-form questions.
    options: z.array(z.string()).nullable().optional(),
  }),
  z.object({ kind: z.literal("review"), content: z.string() }),
]);
export type HumanRequest = z.infer<typeof HumanRequestSchema>;

export type HumanResponse =
  | { kind: "allow" }
  | { kind: "allow_with_modification"; calls: ToolCall[] }
  | { kind: "deny"; reason: string }
  | { kind: "halt" }
  | { kind: "answer"; text: string }
  | { kind: "approve_with_feedback"; feedback: string }
  | { kind: "reject"; reason: string };

// ============================================================================
// PausedState / ChildPausedState
// ============================================================================

/**
 * Child paused state. **Deliberately has no `child_state` field** — the
 * type system enforces one-level subagent depth (spec rule #9).
 */
export interface ChildPausedState {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  session_state: SessionState;
  pending_tool_calls: ToolCall[];
  approved_results: ToolResultRecord[];
  /**
   * Optional (#80): a `waiting_for_human` pause always sets it; an `escalate`
   * pause omits it (the escalation carries no human request). Old
   * `waiting_for_human` blobs still deserialize (field present); escalation
   * blobs omit it (undefined/null on the wire).
   */
  human_request?: HumanRequest;
  task: Task;
  budget_used: BudgetSnapshot;
  parent_tool_call_id: string;
}

export interface PausedState {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  session_state: SessionState;
  pending_tool_calls: ToolCall[];
  approved_results: ToolResultRecord[];
  /**
   * Optional (#80): a `waiting_for_human` pause always sets it; an `escalate`
   * pause omits it (the escalation carries no human request). The `escalate`
   * {@link RunResult} preserves the full {@link PausedState} with this field
   * absent.
   */
  human_request?: HumanRequest;
  task: Task;
  budget_used: BudgetSnapshot;
  child_state: ChildPausedState | null;
}

// ============================================================================
// Halt reasons / RunResult
// ============================================================================

export type HaltReason =
  | { kind: "budget_exceeded"; limit_type: BudgetLimitType }
  | { kind: "termination_policy_halt"; reason: string }
  | { kind: "middleware_halt"; hook: HookPoint; reason: string }
  | { kind: "agent_error"; error: AgentError }
  /**
   * A {@link ContextError} surfaced by the {@link ContextManager} during
   * assembly halts the run (e.g. a cache-hash mismatch — both Block 1
   * `"static"` and, as of #32, Block 2 `"per_session"` halt mid-session). This
   * is the routing type; mirrors `agent_error`. The live {@link StandardHarness}
   * loop does not yet trigger it because its placeholder `ContextManager`
   * assemble is infallible pending the #7 migration.
   */
  | { kind: "context_error"; error: ContextError }
  | { kind: "sandbox_violation"; violation: SandboxViolation }
  | { kind: "unrecoverable_tool_error"; tool: string; error: string }
  | { kind: "human_halted" }
  | { kind: "stagnation_limit_reached"; iterations: number; best_metric: number }
  | { kind: "strategy_not_yet_implemented"; strategy: string }
  /**
   * Returned by {@link StandardHarness} for the `plan_execute` strategy (issue
   * #59) when the accepted plan parsed into an EMPTY task list (`tasks: []`).
   * Per Q3, an empty plan is a failure — the run does NOT silently succeed.
   */
  | { kind: "empty_plan" }
  /**
   * Returned by {@link StandardHarness} for the `plan_execute` strategy (issue
   * #59) when an execute step's bounded ReAct sub-loop errored or the agent
   * returned a blocked/failed outcome (Q5). A plan is a dependency chain by
   * assumption, so the whole run aborts at the failing step — execution does NOT
   * continue to the next task. Carries the failing step's positional index, its
   * instruction, and a human-readable reason derived from the underlying
   * {@link HaltReason}.
   */
  | { kind: "step_failed"; task_index: number; task: string; reason: string }
  /**
   * The `plan_execute` plan phase (issue #70) failed before producing an
   * artifact: the planner's response was unparseable, the planner requested a
   * tool call in the one-shot turn, or the agent returned an error. Carries the
   * underlying {@link "../plan/index.js".PlanPhaseError} detail.
   */
  | { kind: "plan_phase_failed"; error: PlanPhaseErrorKind }
  /**
   * Returned by {@link StandardHarness} for the `self_verifying` strategy
   * (issue #61, D4) when the build↔evaluate loop ran out of the verifier's
   * `maxIterations` round-trips without an explicit `passed` verdict. A RUNTIME
   * limit — the work was attempted in good faith but never verified; a caller
   * might retry with a different task decomposition. Carries the number of
   * round-trips run and the last failure reason the verifier gave. PEER to
   * {@link self_verify_misconfigured} (NOT a sub-case of it).
   */
  | { kind: "self_verify_exhausted"; iterations: number; last_reason: string }
  /**
   * Returned by {@link StandardHarness} for the `self_verifying` strategy
   * (issue #61, D4) when the strategy cannot run because it is misconfigured —
   * e.g. `config.verifier` is absent. Likely a BUILD-TIME bug in the caller's
   * wiring. Surfaced as a typed halt, NOT a throw. PEER to
   * {@link self_verify_exhausted} (NOT a sub-case of it).
   */
  | { kind: "self_verify_misconfigured"; reason: string }
  /**
   * Returned by {@link StandardHarness} for the `ralph` strategy (issue #58, B3)
   * when the multi-context-window continuation loop reached its `maxResets`
   * outer-loop cap with tasks still incomplete (the external completion check —
   * `.spore/progress.json` + `.spore/feature_list.json` — never passed). Carries
   * the number of context windows run and the last incompletion reason. PEER to
   * {@link self_verify_exhausted}.
   */
  | { kind: "ralph_completion_unmet"; iterations: number; last_reason: string }
  /**
   * Returned by {@link StandardHarness} for the `hill_climbing` strategy
   * (issue #60, Decision 6) when the strategy cannot run because it is
   * misconfigured — `config.metricEvaluator` is absent — OR the iteration-0
   * baseline evaluation itself errored (Decision 7: there is no `current_best`
   * to climb from, so a failed baseline is a misconfiguration of the experiment,
   * NOT a stagnation increment). Likely a BUILD-TIME bug in the caller's wiring.
   * Surfaced as a typed halt, NOT a throw.
   */
  | { kind: "hill_climbing_misconfigured"; reason: string };

export type RunResult =
  | {
      kind: "success";
      output: string;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
      /**
       * The post-run conversation history (issue #102). Carries the full
       * {@link SessionState.messages} the loop produced — assistant tool-call
       * turns and tool-result turns included — so an in-process caller can
       * resume losslessly via {@link HarnessRunOptions.session_state} without
       * reconstructing history from `output`. Optional so old serialized
       * `RunResult` blobs (and other languages mid-migration) still parse — the
       * TS analogue of Rust's `#[serde(default)]`; read it via
       * {@link runResultSessionState}, which defaults absence to an empty
       * {@link SessionState}.
       */
      session_state?: SessionState;
    }
  | {
      kind: "failure";
      reason: HaltReason;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
      /**
       * The post-run conversation history at the point of failure (issue #102).
       * Same contract as on the `success` variant: lossless resume, optional for
       * back-compat. Read via {@link runResultSessionState}.
       */
      session_state?: SessionState;
    }
  | { kind: "waiting_for_human"; state: PausedState; request: HumanRequest }
  /**
   * Harness terminated cleanly due to a tool escalation signal (#80). The
   * caller handles the signal and decides whether to resume, instantiate a new
   * harness, or present UI. `state` preserves the full {@link PausedState}
   * (with `human_request` absent) so the original harness can be resumed; the
   * signal is NOT stored in `state`, so it is discarded on resume — the harness
   * never re-acts on it.
   */
  | {
      kind: "escalate";
      signal: HarnessSignal;
      state: PausedState;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
    };

/**
 * The post-run {@link SessionState} carried by a terminal {@link RunResult}
 * (issue #102). For `success`/`failure` returns the carried history, defaulting
 * absence to an empty {@link SessionState} (back-compat with pre-#102 blobs). For
 * `waiting_for_human`/`escalate` returns the {@link PausedState.session_state}
 * those variants already carry. The one place to read post-run history from any
 * terminal/paused result.
 */
export function runResultSessionState(result: RunResult): SessionState {
  switch (result.kind) {
    case "success":
    case "failure":
      return result.session_state ?? emptySessionState();
    case "waiting_for_human":
    case "escalate":
      return result.state.session_state;
  }
}

// ============================================================================
// HarnessRunOptions
// ============================================================================

export interface HarnessRunOptions {
  task: Task;
  on_stream?: StreamSink;
  /** Optional starting session state (resumed history). */
  session_state?: SessionState;
  signal?: AbortSignal;
}

// ============================================================================
// Re-export of underlying schema types referenced in this file (for callers).
// ============================================================================
export type { Message, ToolCall, ToolSchema };
