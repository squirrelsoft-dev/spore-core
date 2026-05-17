/**
 * Unit tests for the canonical ObservabilityProvider (spore-core issue #12).
 *
 * Mirrors `rust/crates/spore-core/src/observability.rs#tests` rule-for-rule.
 */

import { describe, expect, it } from "vitest";

import {
  guideRegistry,
  memory,
  middleware,
  observability,
  sensor,
  SessionId,
  TaskId,
} from "../src/index.js";

const { InMemoryObservabilityProvider, SpanId, PricingTable } = observability;
type SpanBase = observability.SpanBase;
type TurnSpan = observability.TurnSpan;
type ToolCallSpan = observability.ToolCallSpan;
type SensorSpan = observability.SensorSpan;
type ContextSpan = observability.ContextSpan;
type ContextOperation = observability.ContextOperation;
type MiddlewareSpan = observability.MiddlewareSpan;

const { Timestamp } = memory;
const { SensorId } = sensor;

function sid(s: string): SessionId {
  return SessionId.of(s);
}
function tid(s: string): TaskId {
  return TaskId.of(s);
}
function ts(s: string): memory.Timestamp {
  return Timestamp.of(s);
}

function turnSpan(
  session: string,
  spanId: string,
  turn: number,
  input: number,
  output: number,
): TurnSpan {
  const base: SpanBase = {
    span_id: SpanId.of(spanId),
    parent_span_id: null,
    session_id: sid(session),
    task_id: tid("t1"),
    kind: "turn",
    started_at: ts("2026-05-16T00:00:00Z"),
    ended_at: ts("2026-05-16T00:00:01Z"),
    duration_ms: 1000,
    status: { kind: "ok" },
  };
  return {
    base,
    turn_number: turn,
    input_tokens: input,
    output_tokens: output,
    cache_read_tokens: null,
    cache_write_tokens: null,
    cost_usd: 0,
    stop_reason: "end_turn",
    tool_calls_requested: 0,
  };
}

describe("InMemoryObservabilityProvider", () => {
  // ── Rule: emitTurn is fire-and-forget (sync) and span is queryable ────────
  it("emit_turn recorded and metrics aggregate", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "sp1", 1, 100, 50));
    obs.emitTurn(turnSpan("s1", "sp2", 2, 200, 80));
    obs.setSessionOutcome(sid("s1"), { kind: "success" });
    const m = await obs.getSessionMetrics(sid("s1"));
    expect(m).toBeDefined();
    expect(m!.total_turns).toBe(2);
    expect(m!.total_input_tokens).toBe(300);
    expect(m!.total_output_tokens).toBe(130);
    expect(m!.outcome).toEqual({ kind: "success" });
  });

  // ── Rule: emit_tool_call counted in metrics ───────────────────────────────
  it("emit_tool_call counted in metrics", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "t1", 1, 10, 5));
    const base: SpanBase = {
      span_id: SpanId.of("tc1"),
      parent_span_id: null,
      session_id: sid("s1"),
      task_id: tid("t1"),
      kind: "tool_call",
      started_at: ts("2026-05-16T00:00:00Z"),
      ended_at: ts("2026-05-16T00:00:00Z"),
      duration_ms: 250,
      status: { kind: "ok" },
    };
    const tc: ToolCallSpan = {
      base,
      tool_name: "shell",
      call_id: "c1",
      parameters_size_bytes: 12,
      output_size_bytes: 42,
      truncated: false,
      sandbox_mode: "workspace_scoped",
      sandbox_violations: [],
    };
    obs.emitToolCall(tc);
    const m = await obs.getSessionMetrics(sid("s1"));
    expect(m!.tool_calls).toBe(1);
    expect(m!.total_duration_ms).toBe(1250);
  });

  // ── Rule: sensor metrics — fires and halts ────────────────────────────────
  it("sensor metrics count fires and halts", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "t1", 1, 10, 5));
    const mk = (id: string, fired: boolean, outcome: sensor.SensorOutcome): SensorSpan => ({
      base: {
        span_id: SpanId.of(id),
        parent_span_id: null,
        session_id: sid("s1"),
        task_id: tid("t1"),
        kind: "sensor_evaluation",
        started_at: ts("2026-05-16T00:00:00Z"),
        ended_at: ts("2026-05-16T00:00:00Z"),
        duration_ms: 1,
        status: { kind: "ok" },
      },
      sensor_id: SensorId.of("lint"),
      sensor_kind: "computational",
      trigger: { kind: "post_turn" },
      outcome,
      fired,
    });
    obs.emitSensor(mk("sn1", true, "warn"));
    obs.emitSensor(mk("sn2", true, "halt"));
    obs.emitSensor(mk("sn3", false, "pass"));
    const m = await obs.getSessionMetrics(sid("s1"));
    expect(m!.sensor_fires).toBe(2);
    expect(m!.sensor_halts).toBe(1);
  });

  // ── Rule: compaction counted ──────────────────────────────────────────────
  it("compaction counted in metrics", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "t1", 1, 100, 50));
    const mkCtx = (op: ContextOperation): ContextSpan => ({
      base: {
        span_id: SpanId.of(op.kind === "compaction" ? "c1" : "c2"),
        parent_span_id: null,
        session_id: sid("s1"),
        task_id: tid("t1"),
        kind: op.kind === "compaction" ? "compaction" : "context_assembly",
        started_at: ts("2026-05-16T00:00:00Z"),
        ended_at: ts("2026-05-16T00:00:00Z"),
        duration_ms: 1,
        status: { kind: "ok" },
      },
      operation: op,
      tokens_before: 10_000,
      tokens_after: 5_000,
      utilization_before: 0.9,
      utilization_after: 0.5,
    });
    obs.emitContext(mkCtx({ kind: "compaction", messages_removed: 5, tokens_reclaimed: 5_000 }));
    obs.emitContext(
      mkCtx({
        kind: "assembly",
        guides_loaded: 2,
        memory_items_loaded: 3,
        tools_loaded: 5,
      }),
    );
    const m = await obs.getSessionMetrics(sid("s1"));
    expect(m!.compactions).toBe(1);
  });

  // ── Rule: pricing table computes cost_usd ─────────────────────────────────
  it("pricing table computes cost", () => {
    const table: observability.PricingTable = {
      input_per_million: 3.0,
      output_per_million: 15.0,
      cache_read_per_million: 0.3,
      cache_write_per_million: 3.75,
    };
    const cost = PricingTable.costFor(table, 1_000_000, 1_000_000, 1_000_000, 1_000_000);
    // 3 + 15 + 0.3 + 3.75 = 22.05
    expect(Math.abs(cost - 22.05)).toBeLessThan(1e-9);
  });

  it("pricing table DEFAULT is zero", () => {
    const cost = PricingTable.costFor(PricingTable.DEFAULT, 1000, 1000, 1000, 1000);
    expect(cost).toBe(0);
  });

  // ── Rule: flushSession idempotent + spans remain queryable ────────────────
  it("flushSession is idempotent and spans remain queryable", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "t1", 1, 10, 5));
    await obs.flushSession(sid("s1"));
    await obs.flushSession(sid("s1")); // second flush is a no-op
    const m = await obs.getSessionMetrics(sid("s1"));
    expect(m!.total_turns).toBe(1);
    const trace = await obs.getTrace(sid("s1"));
    expect(trace.length).toBe(1);
  });

  // ── Rule: getTrace returns spans in insertion order; parent linkage kept ──
  it("getTrace preserves insertion order and parent linkage", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "sp1", 1, 10, 5));
    const tc: ToolCallSpan = {
      base: {
        span_id: SpanId.of("sp2"),
        parent_span_id: SpanId.of("sp1"),
        session_id: sid("s1"),
        task_id: tid("t1"),
        kind: "tool_call",
        started_at: ts("2026-05-16T00:00:00Z"),
        ended_at: ts("2026-05-16T00:00:00Z"),
        duration_ms: 1,
        status: { kind: "ok" },
      },
      tool_name: "shell",
      call_id: "c1",
      parameters_size_bytes: 0,
      output_size_bytes: 0,
      truncated: false,
      sandbox_mode: "none",
      sandbox_violations: [],
    };
    obs.emitToolCall(tc);
    const trace = await obs.getTrace(sid("s1"));
    expect(trace.length).toBe(2);
    expect(trace[0].base.span_id.asString()).toBe("sp1");
    expect(trace[1].base.span_id.asString()).toBe("sp2");
    expect(trace[1].base.parent_span_id?.asString()).toBe("sp1");
  });

  // ── Rule: middleware span recorded with hook + decision ───────────────────
  it("middleware span recorded in trace", async () => {
    const obs = new InMemoryObservabilityProvider();
    const span: MiddlewareSpan = {
      base: {
        span_id: SpanId.of("mw1"),
        parent_span_id: null,
        session_id: sid("s1"),
        task_id: tid("t1"),
        kind: "middleware_hook",
        started_at: ts("2026-05-16T00:00:00Z"),
        ended_at: ts("2026-05-16T00:00:00Z"),
        duration_ms: 0,
        status: { kind: "ok" },
      },
      hook: "before_turn" as middleware.HookPoint,
      decision: { kind: "continue" } as middleware.MiddlewareDecision,
    };
    obs.emitMiddleware(span);
    const trace = await obs.getTrace(sid("s1"));
    expect(trace.length).toBe(1);
    expect(trace[0].base.kind).toBe("middleware_hook");
  });

  // ── Rule: getSessions filters by outcome ──────────────────────────────────
  it("getSessions filters by outcome", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("good", "sp1", 1, 10, 5));
    obs.emitTurn(turnSpan("bad", "sp2", 1, 10, 5));
    obs.setSessionOutcome(sid("good"), { kind: "success" });
    obs.setSessionOutcome(sid("bad"), { kind: "failure", reason: "x" });
    const success = await obs.getSessions(ts("2026-05-16T00:00:00Z"), undefined, {
      kind: "success",
    });
    expect(success.length).toBe(1);
    expect(success[0].session_id.asString()).toBe("good");
  });

  // ── Rule: getSessions filters by since timestamp ──────────────────────────
  it("getSessions filters by since", async () => {
    const obs = new InMemoryObservabilityProvider();
    const early = turnSpan("old", "sp1", 1, 10, 5);
    early.base = { ...early.base, started_at: ts("2026-01-01T00:00:00Z") };
    obs.emitTurn(early);
    obs.emitTurn(turnSpan("new", "sp2", 1, 10, 5));
    const recent = await obs.getSessions(ts("2026-05-15T00:00:00Z"));
    const ids = recent.map((m) => m.session_id.asString());
    expect(ids).toContain("new");
    expect(ids).not.toContain("old");
  });

  // ── Rule: recordGuidesUsed surfaces on metrics ────────────────────────────
  it("guides_used surfaces on metrics", async () => {
    const obs = new InMemoryObservabilityProvider();
    obs.emitTurn(turnSpan("s1", "sp1", 1, 10, 5));
    const g = guideRegistry.GuideId.of("g1");
    obs.recordGuidesUsed(sid("s1"), [g]);
    const m = await obs.getSessionMetrics(sid("s1"));
    expect(m!.guides_used.map((x) => x.asString())).toEqual(["g1"]);
  });

  // ── Rule: passive observer — no harness-mutating method on interface ──────
  // (type-level: ObservabilityProvider has no method that returns a value the
  //  harness uses to alter behavior. Asserted here by structural usage.)
  it("ObservabilityProvider can be used via the interface alone", async () => {
    const provider: observability.ObservabilityProvider = new InMemoryObservabilityProvider();
    provider.emitTurn(turnSpan("s1", "sp1", 1, 10, 5));
    const m = await provider.getSessionMetrics(sid("s1"));
    expect(m!.total_turns).toBe(1);
  });

  // ── SpanBase helpers ──────────────────────────────────────────────────────
  it("new root and child spans", () => {
    const root = observability.newRootSpanBase(
      SpanId.of("r"),
      sid("s"),
      tid("t"),
      "session",
      ts("2026-05-16T00:00:00Z"),
    );
    const child = observability.newChildSpanBase(
      SpanId.of("c"),
      root,
      "turn",
      ts("2026-05-16T00:00:01Z"),
    );
    expect(child.parent_span_id?.asString()).toBe("r");
    expect(child.session_id.asString()).toBe("s");
  });
});
