/**
 * InMemoryObservabilityProvider — buffered in-memory reference backend for
 * {@link ObservabilityProvider} (spore-core issue #12).
 *
 * Mirrors `rust/crates/spore-core/src/observability.rs#InMemoryObservabilityProvider`.
 * OTLP / JSONL backends live in sibling packages; they implement the same
 * interface.
 *
 * Invariants enforced:
 *   - `emit*` is synchronous (fire-and-forget); spans land in per-kind buffers
 *     plus a per-session insertion-ordered span-id index.
 *   - `flushSession` is idempotent — calling it twice for the same session is
 *     a no-op the second time. Spans remain queryable after flush.
 *   - `getTrace` returns spans in insertion order across kinds.
 *   - `getSessionMetrics` aggregates totals from buffered spans; counts
 *     `sensor_fires` where `fired === true`, `sensor_halts` where
 *     `outcome === "halt"`, and `compactions` where the context operation
 *     kind is `compaction`.
 */

import type { SessionId } from "../harness/types.js";
import { TaskId } from "../harness/types.js";
import type { GuideId, SessionOutcome } from "../guide-registry/types.js";
import type { Timestamp } from "../memory/types.js";

import {
  type ContextSpan,
  type MiddlewareSpan,
  type ObservabilityProvider,
  type PatchSpan,
  type SensorSpan,
  type SessionMetrics,
  type Span,
  type SpanId,
  type SpanKind,
  type ToolCallSpan,
  type TurnSpan,
} from "./types.js";

interface OrderEntry {
  kind: SpanKind;
  span_id: SpanId;
}

interface Store {
  turns: TurnSpan[];
  toolCalls: ToolCallSpan[];
  sensors: SensorSpan[];
  contexts: ContextSpan[];
  middlewares: MiddlewareSpan[];
  patches: PatchSpan[];
  /** Per-session insertion order of (kind, span_id) tuples. */
  traceOrder: Map<string, OrderEntry[]>;
  flushed: Set<string>;
  /** Per-session terminal outcome — recorded by the harness post AfterSession. */
  outcomes: Map<string, SessionOutcome>;
  /** Per-session guides used — recorded by the harness at session start. */
  guidesUsed: Map<string, GuideId[]>;
}

function emptyStore(): Store {
  return {
    turns: [],
    toolCalls: [],
    sensors: [],
    contexts: [],
    middlewares: [],
    patches: [],
    traceOrder: new Map(),
    flushed: new Set(),
    outcomes: new Map(),
    guidesUsed: new Map(),
  };
}

export class InMemoryObservabilityProvider implements ObservabilityProvider {
  private readonly store: Store = emptyStore();

  /** Record the terminal outcome for a session so {@link SessionMetrics} can
   *  surface it. The harness calls this once, after `fireAfterSession`. */
  setSessionOutcome(sessionId: SessionId, outcome: SessionOutcome): void {
    this.store.outcomes.set(sessionId.asString(), outcome);
  }

  /** Record the guides selected for a session. Called once at session start. */
  recordGuidesUsed(sessionId: SessionId, guides: GuideId[]): void {
    this.store.guidesUsed.set(sessionId.asString(), guides.slice());
  }

  /** All recorded patch spans for a session, in insertion order (issue #28).
   *  Lets callers inspect the original/patched diff and classified
   *  {@link PatchType} without reconstructing them from the trace. */
  patchSpans(sessionId: SessionId): PatchSpan[] {
    return this.store.patches.filter((p) => p.base.session_id.equals(sessionId));
  }

  private pushOrder(sessionId: SessionId, kind: SpanKind, spanId: SpanId): void {
    const key = sessionId.asString();
    let order = this.store.traceOrder.get(key);
    if (!order) {
      order = [];
      this.store.traceOrder.set(key, order);
    }
    order.push({ kind, span_id: spanId });
  }

  emitTurn(span: TurnSpan): void {
    this.pushOrder(span.base.session_id, "turn", span.base.span_id);
    this.store.turns.push(span);
  }

  emitToolCall(span: ToolCallSpan): void {
    this.pushOrder(span.base.session_id, "tool_call", span.base.span_id);
    this.store.toolCalls.push(span);
  }

  emitSensor(span: SensorSpan): void {
    this.pushOrder(span.base.session_id, "sensor_evaluation", span.base.span_id);
    this.store.sensors.push(span);
  }

  emitContext(span: ContextSpan): void {
    const kind: SpanKind = span.operation.kind === "compaction" ? "compaction" : "context_assembly";
    this.pushOrder(span.base.session_id, kind, span.base.span_id);
    this.store.contexts.push(span);
  }

  emitMiddleware(span: MiddlewareSpan): void {
    this.pushOrder(span.base.session_id, "middleware_hook", span.base.span_id);
    this.store.middlewares.push(span);
  }

  emitPatch(span: PatchSpan): void {
    this.pushOrder(span.base.session_id, "patch", span.base.span_id);
    this.store.patches.push(span);
  }

  async flushSession(sessionId: SessionId): Promise<void> {
    const key = sessionId.asString();
    if (this.store.flushed.has(key)) {
      // Idempotent: second flush is a no-op.
      return;
    }
    this.store.flushed.add(key);
  }

  async getSessionMetrics(sessionId: SessionId): Promise<SessionMetrics | undefined> {
    const key = sessionId.asString();
    const turns = this.store.turns.filter((t) => t.base.session_id.equals(sessionId));
    if (turns.length === 0 && !this.store.outcomes.has(key)) {
      return undefined;
    }
    const taskId: TaskId = turns[0]?.base.task_id ?? TaskId.of("");
    const totalInput = turns.reduce((acc, t) => acc + t.input_tokens, 0);
    const totalOutput = turns.reduce((acc, t) => acc + t.output_tokens, 0);
    const totalCost = turns.reduce((acc, t) => acc + t.cost_usd, 0);

    const sessionToolCalls = this.store.toolCalls.filter((c) =>
      c.base.session_id.equals(sessionId),
    );
    const totalDuration =
      turns.reduce((acc, t) => acc + t.base.duration_ms, 0) +
      sessionToolCalls.reduce((acc, c) => acc + c.base.duration_ms, 0);

    const sessionSensors = this.store.sensors.filter((s) => s.base.session_id.equals(sessionId));
    const sensorFires = sessionSensors.filter((s) => s.fired).length;
    const sensorHalts = sessionSensors.filter((s) => s.outcome === "halt").length;

    const compactions = this.store.contexts.filter(
      (c) => c.base.session_id.equals(sessionId) && c.operation.kind === "compaction",
    ).length;

    const sessionPatches = this.store.patches.filter((p) => p.base.session_id.equals(sessionId));
    const patchCount = sessionPatches.length;
    // Guard divide-by-zero; denominator is all tool-call spans (issue #28).
    const patchRate = sessionToolCalls.length === 0 ? 0 : patchCount / sessionToolCalls.length;
    const patchesByTool: Record<string, number> = {};
    for (const p of sessionPatches) {
      patchesByTool[p.tool_name] = (patchesByTool[p.tool_name] ?? 0) + 1;
    }

    return {
      session_id: sessionId,
      task_id: taskId,
      total_turns: turns.length,
      total_input_tokens: totalInput,
      total_output_tokens: totalOutput,
      total_cost_usd: totalCost,
      total_duration_ms: totalDuration,
      tool_calls: sessionToolCalls.length,
      sensor_fires: sensorFires,
      sensor_halts: sensorHalts,
      compactions,
      outcome: this.store.outcomes.get(key) ?? { kind: "partial" },
      guides_used: this.store.guidesUsed.get(key)?.slice() ?? [],
      patch_count: patchCount,
      patch_rate: patchRate,
      patches_by_tool: patchesByTool,
    };
  }

  async getSessions(
    since: Timestamp,
    _domain?: string,
    outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]> {
    // Collect distinct session ids whose turn spans satisfy `started_at >= since`
    // (lexical RFC 3339 comparison — same convention as the Rust reference).
    const sinceStr = since.asString();
    const seen = new Set<string>();
    const sessionIds: SessionId[] = [];
    for (const t of this.store.turns) {
      if (t.base.started_at.asString() >= sinceStr) {
        const key = t.base.session_id.asString();
        if (!seen.has(key)) {
          seen.add(key);
          sessionIds.push(t.base.session_id);
        }
      }
    }
    sessionIds.sort((a, b) => a.asString().localeCompare(b.asString()));

    const out: SessionMetrics[] = [];
    for (const sid of sessionIds) {
      const metrics = await this.getSessionMetrics(sid);
      if (!metrics) continue;
      if (outcome && !sessionOutcomeEquals(metrics.outcome, outcome)) {
        continue;
      }
      out.push(metrics);
    }
    return out;
  }

  async getTrace(sessionId: SessionId): Promise<Span[]> {
    const order = this.store.traceOrder.get(sessionId.asString());
    if (!order) return [];
    const out: Span[] = [];
    for (const entry of order) {
      const span = this.lookupSpan(entry);
      if (span) out.push(span);
    }
    return out;
  }

  private lookupSpan(entry: OrderEntry): Span | undefined {
    const match = (s: { base: { span_id: SpanId } }) => s.base.span_id.equals(entry.span_id);
    switch (entry.kind) {
      case "turn":
        return this.store.turns.find(match);
      case "tool_call":
        return this.store.toolCalls.find(match);
      case "sensor_evaluation":
        return this.store.sensors.find(match);
      case "context_assembly":
      case "compaction":
        return this.store.contexts.find(match);
      case "middleware_hook":
        return this.store.middlewares.find(match);
      case "patch":
        return this.store.patches.find(match);
      default:
        return undefined;
    }
  }
}

function sessionOutcomeEquals(a: SessionOutcome, b: SessionOutcome): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === "failure" && b.kind === "failure") {
    return a.reason === b.reason;
  }
  return true;
}
