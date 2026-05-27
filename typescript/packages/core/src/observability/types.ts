/**
 * ObservabilityProvider — canonical types (spore-core issue #12).
 *
 * Mirrors `rust/crates/spore-core/src/observability.rs` byte-for-byte on the
 * wire: tagged unions use a `kind` discriminator in `snake_case`; struct
 * fields are `snake_case`. Same fixture, same outcome — see
 * `/fixtures/README.md`.
 *
 * Every observable harness operation emits one {@link Span}. Spans carry
 * identity (session, task, parent span), timing, status, and operation-
 * specific payload. Aggregates roll up to {@link SessionMetrics} for the
 * improvement loop.
 *
 * Rules enforced by {@link InMemoryObservabilityProvider}:
 *   - `emit*` methods are fire-and-forget — synchronous, returning `void`;
 *     spans are buffered in memory.
 *   - `cost_usd` on {@link TurnSpan} is computed at emit time from the
 *     injected {@link PricingTable}; the harness does not estimate it.
 *   - {@link ObservabilityProvider.flushSession | flushSession} is idempotent
 *     — calling it twice for the same session is a no-op the second time,
 *     and spans remain queryable after flush.
 *   - {@link ObservabilityProvider.getTrace | getTrace} returns the full span
 *     list for a session in insertion order; the trace-analyzer reconstructs
 *     hierarchy via `parent_span_id`.
 *   - Observability is **passive**: the interface has no mutator that affects
 *     harness behavior. (Documented; not statically enforced.)
 */

import type { SessionId, TaskId } from "../harness/types.js";
import type { GuideId } from "../guide-registry/types.js";
import type { SessionOutcome } from "../guide-registry/types.js";
import type { Timestamp } from "../memory/types.js";
import type { StopReason } from "../model/schemas.js";
import type { HookPoint, MiddlewareDecision } from "../middleware/types.js";
import type { SensorId, SensorKind, SensorOutcome, SensorTrigger } from "../sensor/types.js";

// ============================================================================
// Identity
// ============================================================================

export class SpanId {
  constructor(readonly value: string) {}
  static of(value: string): SpanId {
    return new SpanId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: SpanId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

// ============================================================================
// Span enums and base
// ============================================================================

export type SpanKind =
  | "session"
  | "turn"
  | "tool_call"
  | "sensor_evaluation"
  | "context_assembly"
  | "compaction"
  | "middleware_hook"
  | "guide_selection"
  | "memory_query"
  | "memory_write"
  /** Emitted by `PatchToolCallsMiddleware` whenever it mutates a tool call
   *  (issue #28). Always carries a {@link PatchSpan} at {@link SpanLevel}
   *  `"warn"`. */
  | "patch"
  /** Emitted by the harness compaction loop when a summary is accepted despite
   *  failing verification (issue #46). Always carries a {@link WarnSpan} at
   *  {@link SpanLevel} `"warn"`. */
  | "warn";

export type SpanStatus =
  | { kind: "ok" }
  | { kind: "error"; message: string }
  | { kind: "halted"; reason: string };

export interface SpanBase {
  span_id: SpanId;
  parent_span_id?: SpanId | null;
  session_id: SessionId;
  task_id: TaskId;
  kind: SpanKind;
  started_at: Timestamp;
  ended_at: Timestamp;
  duration_ms: number;
  status: SpanStatus;
}

export function newRootSpanBase(
  spanId: SpanId,
  sessionId: SessionId,
  taskId: TaskId,
  kind: SpanKind,
  startedAt: Timestamp,
): SpanBase {
  return {
    span_id: spanId,
    parent_span_id: null,
    session_id: sessionId,
    task_id: taskId,
    kind,
    started_at: startedAt,
    ended_at: startedAt,
    duration_ms: 0,
    status: { kind: "ok" },
  };
}

export function newChildSpanBase(
  spanId: SpanId,
  parent: SpanBase,
  kind: SpanKind,
  startedAt: Timestamp,
): SpanBase {
  return {
    span_id: spanId,
    parent_span_id: parent.span_id,
    session_id: parent.session_id,
    task_id: parent.task_id,
    kind,
    started_at: startedAt,
    ended_at: startedAt,
    duration_ms: 0,
    status: { kind: "ok" },
  };
}

export function finishSpanBase(
  base: SpanBase,
  endedAt: Timestamp,
  status: SpanStatus,
  durationMs: number,
): SpanBase {
  return {
    ...base,
    ended_at: endedAt,
    status,
    duration_ms: durationMs,
  };
}

// ============================================================================
// Span payload types
// ============================================================================

export interface TurnSpan {
  base: SpanBase;
  turn_number: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens?: number | null;
  cache_write_tokens?: number | null;
  cost_usd: number;
  stop_reason: StopReason;
  tool_calls_requested: number;
}

export interface ToolCallSpan {
  base: SpanBase;
  tool_name: string;
  call_id: string;
  parameters_size_bytes: number;
  output_size_bytes: number;
  truncated: boolean;
  sandbox_mode: string;
  sandbox_violations: string[];
}

export interface SensorSpan {
  base: SpanBase;
  sensor_id: SensorId;
  sensor_kind: SensorKind;
  trigger: SensorTrigger;
  outcome: SensorOutcome;
  fired: boolean;
}

export type ContextOperation =
  | {
      kind: "assembly";
      guides_loaded: number;
      memory_items_loaded: number;
      tools_loaded: number;
    }
  | { kind: "tool_result_appended"; tool_name: string; truncated: boolean }
  | { kind: "compaction"; messages_removed: number; tokens_reclaimed: number }
  | { kind: "skill_injected"; guide_id: GuideId };

export interface ContextSpan {
  base: SpanBase;
  operation: ContextOperation;
  tokens_before: number;
  tokens_after: number;
  utilization_before: number;
  utilization_after: number;
}

export interface MiddlewareSpan {
  base: SpanBase;
  hook: HookPoint;
  decision: MiddlewareDecision;
}

// ============================================================================
// Patch observability (issue #28)
// ============================================================================
//
// `PatchToolCallsMiddleware` is an always-on, highest-priority `before_tool`
// action mutator that silently rewrites malformed or dangling tool calls
// before the sandbox and sensors see them. An always-on mutator with no
// observability is a footgun: the trace would show the patched call as if the
// model had sent it. Issue #28 closes that gap.

/**
 * Severity of an emitted span. Patch spans are always `"warn"` per issue #28;
 * this stays orthogonal to {@link SpanStatus} so a successful (`ok`) trace can
 * still surface warn-level patch events.
 */
export type SpanLevel = "info" | "warn";

/**
 * Classification of a tool-call patch. Production JSON-repair routines add
 * variants; downstream `switch`es must keep a default branch.
 */
export type PatchType =
  /** Raw tool-call arguments failed to parse as JSON; a repair was attempted.
   *  `error` is the parse error that was recovered from. */
  | { kind: "malformed_json"; error: string }
  /** The call was structurally incomplete (e.g. empty tool name) and was
   *  completed with defaults. `reason` explains what was missing. */
  | { kind: "dangling_tool_call"; reason: string }
  /** A parameter value was coerced from one type to another to satisfy the
   *  tool schema. */
  | { kind: "parameter_coercion"; field: string; from: string; to: string };

/**
 * One observability event per tool-call patch (issue #28). Carries both the
 * original parameters (what the model sent) and the patched parameters (what
 * was dispatched) so the trace shows the diff, never just the patched call.
 *
 * `level` is always `"warn"`; build via {@link newPatchSpan} so callers cannot
 * emit an `"info"`-level patch.
 */
export interface PatchSpan {
  base: SpanBase;
  call_id: string;
  tool_name: string;
  original_parameters: unknown;
  patched_parameters: unknown;
  patch_type: PatchType;
  /** Always `"warn"`. */
  level: SpanLevel;
}

/** Build a patch span. The level is forced to `"warn"`. */
export function newPatchSpan(
  base: SpanBase,
  callId: string,
  toolName: string,
  originalParameters: unknown,
  patchedParameters: unknown,
  patchType: PatchType,
): PatchSpan {
  return {
    base,
    call_id: callId,
    tool_name: toolName,
    original_parameters: originalParameters,
    patched_parameters: patchedParameters,
    patch_type: patchType,
    level: "warn",
  };
}

// ============================================================================
// Compaction-verification warn (issue #46)
// ============================================================================
//
// The harness compaction loop (issue #29 pseudocode, wired in #46) verifies
// every agent-produced summary with a {@link CompactionVerifier} before
// accepting it. After `maxCompactionAttempts` failed verifications the harness
// accepts the summary anyway — a blocked compaction is worse than an imperfect
// one — and emits exactly one warn-level {@link WarnSpan} recording the
// still-missing items and `accepted_anyway: true`.
//
// Rules (mirrored by harness tests):
//   W1  a successful (or first-try-passing) compaction emits NO warn span.
//   W2  exhausting attempts emits EXACTLY ONE warn span carrying the final
//       `missing_items` and `accepted_anyway = true`.
//   W3  `SessionMetrics.compaction_verification_failures` counts these spans
//       for the session (mirrors how `compactions` is derived from spans).
//   W4  `emitWarn` has a default no-op so providers predating #46 keep working.

/**
 * A warn-level, fire-and-forget observability event. Discriminated on `warn`;
 * future warn classes add members, so downstream `switch`es must keep a
 * default branch. The event-as-payload shape mirrors {@link PatchType} but
 * keeps warns that are not tied to a single tool call in their own type.
 *
 * Wire shape mirrors the Rust `WarnEvent` byte-for-byte: the tag key is
 * `warn` and field names are `snake_case`.
 */
export type WarnEvent = {
  /** A compaction summary failed verification on every attempt and was accepted
   *  as-is (issue #46). `missing_items` are the preservation-list terms still
   *  absent from the final summary; `accepted_anyway` is always `true` for this
   *  variant (the harness never blocks on compaction). */
  warn: "compaction_verification_failed";
  missing_items: string[];
  accepted_anyway: boolean;
};

/**
 * One warn-level observability span (issue #46). Carries a {@link SpanBase}
 * for trace correlation, the classified {@link WarnEvent}, and a hardcoded
 * `level: "warn"` (constructed via {@link newWarnSpan}). Modeled on
 * {@link PatchSpan}.
 */
export interface WarnSpan {
  base: SpanBase;
  event: WarnEvent;
  /** Always `"warn"`. */
  level: SpanLevel;
}

/** Build a warn span. The level is forced to `"warn"`. */
export function newWarnSpan(base: SpanBase, event: WarnEvent): WarnSpan {
  return { base, event, level: "warn" };
}

/** Heterogeneous return type for {@link ObservabilityProvider.getTrace}. */
export type Span =
  | TurnSpan
  | ToolCallSpan
  | SensorSpan
  | ContextSpan
  | MiddlewareSpan
  | PatchSpan
  | WarnSpan;

// ============================================================================
// SessionMetrics
// ============================================================================

export interface SessionMetrics {
  session_id: SessionId;
  task_id: TaskId;
  total_turns: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  total_duration_ms: number;
  tool_calls: number;
  sensor_fires: number;
  sensor_halts: number;
  compactions: number;
  outcome: SessionOutcome;
  guides_used: GuideId[];
  /** Number of tool-call patches in the session (issue #28). */
  patch_count: number;
  /** `patch_count / tool_calls`. `0.0` when there are no tool calls. */
  patch_rate: number;
  /** Patch count broken down by tool name. */
  patches_by_tool: Record<string, number>;
  /** Number of compactions whose summary failed verification on every attempt
   *  and was accepted anyway (issue #46). Derived from {@link WarnSpan}s
   *  carrying a `compaction_verification_failed` {@link WarnEvent}, mirroring
   *  how `compactions` is derived from compaction spans. */
  compaction_verification_failures: number;
}

// ============================================================================
// PricingTable
// ============================================================================

export interface PricingTable {
  /** USD per 1M input tokens. */
  input_per_million: number;
  /** USD per 1M output tokens. */
  output_per_million: number;
  /** USD per 1M cache-read tokens (typically 0.1× input price). */
  cache_read_per_million: number;
  /** USD per 1M cache-write tokens (typically 1.25× input price). */
  cache_write_per_million: number;
}

export const PricingTable = {
  /** Conservative zero-cost default. Production callers inject a real table. */
  DEFAULT: {
    input_per_million: 0,
    output_per_million: 0,
    cache_read_per_million: 0,
    cache_write_per_million: 0,
  } as PricingTable,

  costFor(
    table: PricingTable,
    input: number,
    output: number,
    cacheRead?: number | null,
    cacheWrite?: number | null,
  ): number {
    const perToken = (perMillion: number) => perMillion / 1_000_000;
    return (
      perToken(table.input_per_million) * input +
      perToken(table.output_per_million) * output +
      perToken(table.cache_read_per_million) * (cacheRead ?? 0) +
      perToken(table.cache_write_per_million) * (cacheWrite ?? 0)
    );
  },
} as const;

// ============================================================================
// ObservabilityProvider interface
// ============================================================================

/**
 * Structured observability surface. All `emit*` methods are fire-and-forget;
 * they must never block the harness loop. Implementations buffer internally
 * and flush asynchronously via {@link flushSession}.
 *
 * Observability is **passive** — no method on this interface may mutate
 * harness behavior. (Documented; not statically enforced.)
 */
export interface ObservabilityProvider {
  emitTurn(span: TurnSpan): void;
  emitToolCall(span: ToolCallSpan): void;
  emitSensor(span: SensorSpan): void;
  emitContext(span: ContextSpan): void;
  emitMiddleware(span: MiddlewareSpan): void;
  /** Record a warn-level tool-call patch event (issue #28). Fire-and-forget
   *  like the other `emit*` methods. */
  emitPatch(span: PatchSpan): void;

  /** Record a warn-level event not tied to a single tool call (issue #46) —
   *  e.g. an accepted-anyway compaction-verification failure. Fire-and-forget
   *  like the other `emit*` methods. OPTIONAL: providers predating #46 need not
   *  implement it; the harness treats a missing `emitWarn` as a no-op, so they
   *  keep compiling and behave unchanged (rule W4). */
  emitWarn?(span: WarnSpan): void;

  /** Record the terminal outcome for a session so {@link SessionMetrics} can
   *  surface it. The harness calls this once, on a terminal `run` outcome
   *  (never on a `WaitingForHuman` pause). */
  setSessionOutcome(sessionId: SessionId, outcome: SessionOutcome): void;

  flushSession(sessionId: SessionId): Promise<void>;

  getSessionMetrics(sessionId: SessionId): Promise<SessionMetrics | undefined>;

  getSessions(
    since: Timestamp,
    domain?: string,
    outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]>;

  getTrace(sessionId: SessionId): Promise<Span[]>;

  /**
   * Session ids whose durable outbox has a `trace.jsonl` but no `.flushed`
   * marker (issue #33). Optional: only the durable-outbox provider has
   * unflushed on-disk sessions. Providers without a durable outbox return `[]`.
   */
  listUnflushedSessions?(): Promise<SessionId[]>;

  /**
   * Delete a session's durable outbox (issue #33). The provider NEVER
   * auto-deletes; the caller drives cleanup. Optional: in-memory providers
   * have nothing to clean up and resolve to a no-op.
   */
  cleanupSession?(sessionId: SessionId): Promise<void>;
}
