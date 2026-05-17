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
  | "memory_write";

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

/** Heterogeneous return type for {@link ObservabilityProvider.getTrace}. */
export type Span = TurnSpan | ToolCallSpan | SensorSpan | ContextSpan | MiddlewareSpan;

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

  flushSession(sessionId: SessionId): Promise<void>;

  getSessionMetrics(sessionId: SessionId): Promise<SessionMetrics | undefined>;

  getSessions(
    since: Timestamp,
    domain?: string,
    outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]>;

  getTrace(sessionId: SessionId): Promise<Span[]>;
}
