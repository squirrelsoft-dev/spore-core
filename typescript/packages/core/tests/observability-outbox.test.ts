/**
 * Unit + fixture-replay tests for the OutboxObservabilityProvider
 * (spore-core issue #33).
 *
 * Mirrors `rust/crates/spore-core/src/observability_outbox.rs#tests`
 * rule-for-rule and replays the cross-language golden fixtures under
 * `fixtures/observability/trace_line_*.json`. Hermetic: temp dirs, no live
 * OTLP / network (SPORE_OTLP_ENDPOINT is cleared before each provider).
 */

import { mkdtempSync, readFileSync, readdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  guideRegistry,
  memory,
  middleware,
  observability,
  sensor,
  SessionId,
  TaskId,
  type StopReason,
} from "../src/index.js";

const {
  OutboxObservabilityProvider,
  InMemoryObservabilityProvider,
  SessionNotFoundError,
  TraceLine,
  outboxConfig,
  SpanId,
} = observability;
type SpanBase = observability.SpanBase;
type SpanStatus = observability.SpanStatus;
type TurnSpan = observability.TurnSpan;
type ContextOperation = observability.ContextOperation;

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

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = resolve(here, "../../../../fixtures/observability");

beforeEach(() => {
  delete process.env.SPORE_OTLP_ENDPOINT;
});
afterEach(() => {
  delete process.env.SPORE_OTLP_ENDPOINT;
});

function tmpRoot(): string {
  return mkdtempSync(join(tmpdir(), "spore-outbox-"));
}

function base(
  session: string,
  spanId: string,
  kind: observability.SpanKind,
  status: SpanStatus = { kind: "ok" },
): SpanBase {
  return {
    span_id: SpanId.of(spanId),
    parent_span_id: null,
    session_id: sid(session),
    task_id: tid("task1"),
    kind,
    started_at: ts("2026-05-26T18:00:00.000Z"),
    ended_at: ts("2026-05-26T18:00:02.100Z"),
    duration_ms: 2100,
    status,
  };
}

function turn(session: string, spanId: string, status?: SpanStatus): TurnSpan {
  return {
    base: base(session, spanId, "turn", status),
    turn_number: 1,
    input_tokens: 1820,
    output_tokens: 140,
    cache_read_tokens: 1600,
    cache_write_tokens: 0,
    cost_usd: 0.0123,
    stop_reason: "tool_use" as StopReason,
    tool_calls_requested: 1,
  };
}

function provider(root: string, overrides: Partial<observability.OutboxConfig> = {}) {
  return new OutboxObservabilityProvider(outboxConfig(root, overrides));
}

function readLines(root: string, session: string): Record<string, unknown>[] {
  const path = join(root, "sessions", session, "trace.jsonl");
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.length > 0)
    .map((l) => JSON.parse(l) as Record<string, unknown>);
}

function attrs(line: Record<string, unknown>): Record<string, unknown> {
  return line.attributes as Record<string, unknown>;
}

describe("OutboxObservabilityProvider — durable JSONL outbox", () => {
  it("writes exactly one line per emit", () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "sp1"));
    obs.emitTurn(turn("s1", "sp2"));
    expect(readLines(root, "s1")).toHaveLength(2);
  });

  it("turn line matches the schema envelope", () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "sp1"));
    const l = readLines(root, "s1")[0];
    expect(l.kind).toBe("turn");
    expect(l.level).toBe("info");
    expect(l.span_id).toBe("sp1");
    expect(l.parent_span_id).toBeNull();
    expect(l.session_id).toBe("s1");
    expect(l.task_id).toBe("task1");
    expect(l.timestamp).toBe("2026-05-26T18:00:02.100Z");
    expect(l.started_at).toBe("2026-05-26T18:00:00.000Z");
    expect(l.duration_ms).toBe(2100);
    expect(l.status).toBe("ok");
    expect(l.status_detail).toBeNull();
    expect(attrs(l).turn_number).toBe(1);
    expect(attrs(l).input_tokens).toBe(1820);
    expect(attrs(l).cache_read_tokens).toBe(1600);
    expect(attrs(l).stop_reason).toBe("tool_use");
    expect(attrs(l).tool_calls_requested).toBe(1);
    expect((l.trace_id as string).length).toBe(32);
  });

  it("patch spans are warn-level with original + patched parameters", () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitPatch(
      observability.newPatchSpan(
        base("s1", "p1", "patch"),
        "c1",
        "shell",
        { a: "1" },
        { a: 1 },
        { kind: "parameter_coercion", field: "a", from: "string", to: "number" },
      ),
    );
    const l = readLines(root, "s1")[0];
    expect(l.kind).toBe("patch");
    expect(l.level).toBe("warn");
    expect((attrs(l).patch_type as Record<string, unknown>).kind).toBe("parameter_coercion");
    expect((attrs(l).original_parameters as Record<string, unknown>).a).toBe("1");
    expect((attrs(l).patched_parameters as Record<string, unknown>).a).toBe(1);
  });

  it("maps Ok / Error / Halted status to bare strings + detail", () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "ok"));
    obs.emitTurn(turn("s1", "err", { kind: "error", message: "boom" }));
    obs.emitTurn(turn("s1", "halt", { kind: "halted", reason: "stop" }));
    const lines = readLines(root, "s1");
    expect(lines[0].status).toBe("ok");
    expect(lines[0].status_detail).toBeNull();
    expect(lines[1].status).toBe("error");
    expect(lines[1].status_detail).toBe("boom");
    expect(lines[2].status).toBe("halted");
    expect(lines[2].status_detail).toBe("stop");
  });

  it("distinguishes context_assembly from compaction kind", () => {
    const root = tmpRoot();
    const obs = provider(root);
    const mk = (spanId: string, op: ContextOperation): observability.ContextSpan => ({
      base: base("s1", spanId, "context_assembly"),
      operation: op,
      tokens_before: 100,
      tokens_after: 50,
      utilization_before: 0.9,
      utilization_after: 0.5,
    });
    obs.emitContext(
      mk("asm", { kind: "assembly", guides_loaded: 1, memory_items_loaded: 2, tools_loaded: 3 }),
    );
    obs.emitContext(mk("comp", { kind: "compaction", messages_removed: 5, tokens_reclaimed: 50 }));
    const lines = readLines(root, "s1");
    expect(lines[0].kind).toBe("context_assembly");
    expect((attrs(lines[0]).operation as Record<string, unknown>).kind).toBe("assembly");
    expect(lines[1].kind).toBe("compaction");
    expect((attrs(lines[1]).operation as Record<string, unknown>).kind).toBe("compaction");
  });

  it("writes sensor and middleware lines (info level, verbatim attributes)", () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitSensor({
      base: base("s1", "sn1", "sensor_evaluation"),
      sensor_id: SensorId.of("test-runner"),
      sensor_kind: "computational",
      trigger: { kind: "post_turn" },
      outcome: "pass",
      fired: true,
    });
    obs.emitMiddleware({
      base: base("s1", "mw1", "middleware_hook"),
      hook: "before_turn" as middleware.HookPoint,
      decision: { kind: "continue" } as middleware.MiddlewareDecision,
    });
    const lines = readLines(root, "s1");
    expect(lines[0].kind).toBe("sensor_evaluation");
    expect(lines[0].level).toBe("info");
    expect(attrs(lines[0]).sensor_id).toBe("test-runner");
    expect((attrs(lines[0]).trigger as Record<string, unknown>).kind).toBe("post_turn");
    expect(attrs(lines[0]).fired).toBe(true);
    expect(lines[1].kind).toBe("middleware_hook");
    expect(attrs(lines[1]).hook).toBe("before_turn");
    expect((attrs(lines[1]).decision as Record<string, unknown>).kind).toBe("continue");
  });

  it("flushSession writes the session summary line + .flushed marker", async () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "sp1"));
    obs
      .innerProvider()
      .setSessionOutcome(sid("s1"), { kind: "success" } as guideRegistry.SessionOutcome);
    await obs.flushSession(sid("s1"));
    const lines = readLines(root, "s1");
    const last = lines[lines.length - 1];
    expect(last.kind).toBe("session");
    expect(attrs(last).outcome).toBe("success");
    expect(attrs(last).total_turns).toBe(1);
    expect(existsSync(join(root, "sessions", "s1", ".flushed"))).toBe(true);
  });

  it("flushSession skips the summary line when flushOnSessionEnd is false", async () => {
    const root = tmpRoot();
    const obs = provider(root, { flushOnSessionEnd: false });
    obs.emitTurn(turn("s1", "sp1"));
    obs
      .innerProvider()
      .setSessionOutcome(sid("s1"), { kind: "success" } as guideRegistry.SessionOutcome);
    await obs.flushSession(sid("s1"));
    const lines = readLines(root, "s1");
    expect(lines).toHaveLength(1);
    expect(lines[0].kind).toBe("turn");
    // Marker is still written.
    expect(existsSync(join(root, "sessions", "s1", ".flushed"))).toBe(true);
  });

  it("rotates the active file when it exceeds a tiny maxSizeBytes", () => {
    const root = tmpRoot();
    const obs = provider(root, { maxSizeBytes: 10 }); // each line >> 10 bytes
    obs.emitTurn(turn("s1", "sp1"));
    obs.emitTurn(turn("s1", "sp2"));
    obs.emitTurn(turn("s1", "sp3"));
    const dir = join(root, "sessions", "s1");
    const rotated = readdirSync(dir).filter((n) => /^trace-\d+\.jsonl$/.test(n));
    expect(rotated.length).toBeGreaterThan(0);
    expect(existsSync(join(dir, "trace-001.jsonl"))).toBe(true);
  });

  it("writes JSONL only (still one line) when SPORE_OTLP_ENDPOINT is unset", () => {
    const root = tmpRoot();
    delete process.env.SPORE_OTLP_ENDPOINT;
    const obs = new OutboxObservabilityProvider(outboxConfig(root));
    obs.emitTurn(turn("s1", "sp1"));
    expect(readLines(root, "s1")).toHaveLength(1);
  });

  it("treats a whitespace-only SPORE_OTLP_ENDPOINT as unset", () => {
    const root = tmpRoot();
    process.env.SPORE_OTLP_ENDPOINT = "   ";
    const obs = new OutboxObservabilityProvider(outboxConfig(root));
    obs.emitTurn(turn("s1", "sp1"));
    expect(readLines(root, "s1")).toHaveLength(1);
  });

  it("lists unflushed sessions before flush and drops them after", async () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "sp1"));
    const before = await obs.listUnflushedSessions();
    expect(before.map((s) => s.asString())).toEqual(["s1"]);
    obs
      .innerProvider()
      .setSessionOutcome(sid("s1"), { kind: "success" } as guideRegistry.SessionOutcome);
    await obs.flushSession(sid("s1"));
    const after = await obs.listUnflushedSessions();
    expect(after).toHaveLength(0);
  });

  it("cleanupSession deletes the dir and throws SessionNotFound for missing", async () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "sp1"));
    await obs.cleanupSession(sid("s1"));
    expect(existsSync(join(root, "sessions", "s1"))).toBe(false);
    await expect(obs.cleanupSession(sid("missing"))).rejects.toBeInstanceOf(SessionNotFoundError);
    await expect(obs.cleanupSession(sid("missing"))).rejects.toMatchObject({
      kind: "session_not_found",
      sessionId: "missing",
    });
  });

  it("uses a stable trace_id per session, distinct across sessions", () => {
    const root = tmpRoot();
    const obs = provider(root);
    obs.emitTurn(turn("s1", "a"));
    obs.emitTurn(turn("s1", "b"));
    obs.emitTurn(turn("s2", "c"));
    const s1 = readLines(root, "s1");
    const s2 = readLines(root, "s2");
    expect(s1[0].trace_id).toBe(s1[1].trace_id);
    expect(s1[0].trace_id).not.toBe(s2[0].trace_id);
  });
});

// ── Fixture replay ──────────────────────────────────────────────────────────

interface Fixture {
  trace_id: string;
  span: Record<string, unknown>;
  expected_line: Record<string, unknown>;
}

function buildLine(file: string, span: Record<string, unknown>, traceId: string): unknown {
  switch (file) {
    case "trace_line_turn.json":
      return TraceLine.fromTurn(span as unknown as observability.TurnSpan, traceId);
    case "trace_line_tool_call.json":
      return TraceLine.fromToolCall(span as unknown as observability.ToolCallSpan, traceId);
    case "trace_line_sensor.json":
      return TraceLine.fromSensor(span as unknown as observability.SensorSpan, traceId);
    case "trace_line_context_assembly.json":
    case "trace_line_compaction.json":
      return TraceLine.fromContext(span as unknown as observability.ContextSpan, traceId);
    case "trace_line_middleware.json":
      return TraceLine.fromMiddleware(span as unknown as observability.MiddlewareSpan, traceId);
    case "trace_line_patch.json":
      return TraceLine.fromPatch(span as unknown as observability.PatchSpan, traceId);
    case "trace_line_session_summary.json":
      return TraceLine.sessionSummary(
        span.metrics as unknown as observability.SessionMetrics,
        traceId,
        span.root as unknown as observability.SpanBase,
      );
    default:
      throw new Error(`unknown fixture ${file}`);
  }
}

describe("OutboxObservabilityProvider — cross-language fixture replay", () => {
  const files = [
    "trace_line_turn.json",
    "trace_line_tool_call.json",
    "trace_line_sensor.json",
    "trace_line_context_assembly.json",
    "trace_line_compaction.json",
    "trace_line_middleware.json",
    "trace_line_patch.json",
    "trace_line_session_summary.json",
  ];

  for (const file of files) {
    it(`replays ${file} to a JSON-equal expected_line`, () => {
      const fx = JSON.parse(readFileSync(join(fixturesDir, file), "utf8")) as Fixture;
      const got = buildLine(file, fx.span, fx.trace_id);
      // Round-trip through JSON to compare wire representations exactly.
      expect(JSON.parse(JSON.stringify(got))).toEqual(fx.expected_line);
    });
  }
});

// ── InMemory provider satisfies the optional interface defaults ──────────────

describe("InMemoryObservabilityProvider — optional outbox methods", () => {
  it("does not implement listUnflushedSessions / cleanupSession", () => {
    const obs = new InMemoryObservabilityProvider();
    expect(obs.listUnflushedSessions).toBeUndefined();
    expect(obs.cleanupSession).toBeUndefined();
  });
});

describe("attributesToOtelAttributes — per-span attribute flattening", () => {
  it("flattens scalars, skips null, and skips reserved envelope keys", () => {
    // Mirrors a turn span's `attributes` payload, plus a null field and a
    // reserved key that must never leak into the flattened output.
    const out = observability.attributesToOtelAttributes({
      input_tokens: 386,
      output_tokens: 102,
      stop_reason: "tool_use",
      turn_number: 1,
      cache_read_tokens: null,
      session_id: "should-be-skipped",
    });

    expect(out.input_tokens).toBe(386);
    expect(out.output_tokens).toBe(102);
    expect(out.turn_number).toBe(1);
    expect(typeof out.input_tokens).toBe("number");
    expect(typeof out.output_tokens).toBe("number");
    expect(typeof out.turn_number).toBe("number");
    expect(out.stop_reason).toBe("tool_use");
    // Null is skipped entirely.
    expect("cache_read_tokens" in out).toBe(false);
    // Reserved envelope key is skipped so the fixed tag wins.
    expect("session_id" in out).toBe(false);
    // 4 emitted, null + reserved skipped.
    expect(Object.keys(out)).toHaveLength(4);
  });

  it("JSON-stringifies nested objects/arrays and tolerates non-objects", () => {
    const out = observability.attributesToOtelAttributes({
      nested: { a: 1 },
      list: [1, "two"],
      flag: true,
    });
    expect(out.nested).toBe('{"a":1}');
    expect(out.list).toBe('[1,"two"]');
    expect(out.flag).toBe(true);
    expect(observability.attributesToOtelAttributes(null)).toEqual({});
    expect(observability.attributesToOtelAttributes(undefined)).toEqual({});
  });
});
