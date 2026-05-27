/**
 * OutboxObservabilityProvider — durable, JSONL-backed observability provider
 * (spore-core issue #33). Wraps {@link InMemoryObservabilityProvider}.
 *
 * Mirrors the Rust reference at
 * `rust/crates/spore-core/src/observability_outbox.rs`. It is NOT a
 * transliteration: the durable-outbox behavior and the on-disk schema are the
 * shared contract (see `observability/TRACE_SCHEMA.md` and the cross-language
 * golden fixtures under `fixtures/observability/`).
 *
 * ## What it adds on top of the in-memory provider
 *   1. **Durable outbox.** Every `emit*` writes exactly ONE JSONL line,
 *      synchronously appended and flushed (`fs.appendFileSync` to an
 *      `fsync`-backed file descriptor), to
 *      `{root}/sessions/{session_id}/trace.jsonl`. The local JSONL file is the
 *      **source of truth**. The wrapped {@link InMemoryObservabilityProvider}
 *      still handles all buffering, metrics, and query methods — this provider
 *      only adds the file write + OTLP hop in each emit.
 *   2. **OTLP forwarding.** When the env var `SPORE_OTLP_ENDPOINT` is set
 *      (non-empty / non-whitespace) at construction, each span is ALSO
 *      forwarded to OTLP gRPC, best-effort and non-blocking, and `flushSession`
 *      does a best-effort timed `forceFlush` that only logs (never throws) on
 *      failure. When unset/empty, JSONL only.
 *
 * ## Deviation — OTLP behind an internal interface
 * Per the issue's explicit allowance: the full OpenTelemetry JS SDK
 * (`@opentelemetry/sdk-trace-node` + `exporter-trace-otlp-grpc`) is a heavy,
 * version-churny dependency tree that is not part of this monorepo's dependency
 * set. To keep the reliability-critical durable-JSONL path fully implemented,
 * tested, and buildable without it, OTLP forwarding lives behind a small
 * internal {@link OtlpForwarder} interface with two implementations:
 *   - {@link NullOtlpForwarder} — the default; no-op (used whenever the env var
 *     is unset/empty, and as the fallback when the SDK is absent).
 *   - {@link OtelSdkOtlpForwarder} — wires the real OpenTelemetry batch span
 *     processor + OTLP-gRPC exporter via a lazy dynamic `import()`. If the SDK
 *     is not installed it logs a warning and the provider silently runs
 *     JSONL-only. This is the documented deviation; the durable path does not
 *     depend on this interface at all.
 *
 * The 8-byte OTLP span id is derived by SHA-256 hashing the harness SpanId
 * string and taking the first 8 bytes (matching the Rust reference); the
 * 32-hex `trace_id` is generated once per session and reused verbatim in every
 * JSONL line for that session AND as the OTLP trace id.
 */

import { createHash, randomBytes } from "node:crypto";
import {
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { join } from "node:path";

import type { SessionId } from "../harness/types.js";
import { SessionId as SessionIdClass } from "../harness/types.js";
import type { Timestamp } from "../memory/types.js";
import type { SessionOutcome } from "../guide-registry/types.js";

import { InMemoryObservabilityProvider } from "./in-memory.js";
import {
  type ContextSpan,
  type MiddlewareSpan,
  type ObservabilityProvider,
  type PatchSpan,
  type SensorSpan,
  type SessionMetrics,
  type Span,
  type SpanBase,
  type SpanStatus,
  type ToolCallSpan,
  type TurnSpan,
} from "./types.js";

const DEFAULT_MAX_SIZE_BYTES = 50 * 1024 * 1024;
const OTLP_FLUSH_TIMEOUT_MS = 2000;

// ============================================================================
// Errors
// ============================================================================

/**
 * Errors surfaced by the durable-outbox provider (issue #33). Follows the
 * conventions in `typescript/CONVENTIONS.md`: extends `Error`, sets `name` to
 * the class name, carries a discriminant `kind`.
 */
export class SessionNotFoundError extends Error {
  readonly kind = "session_not_found" as const;
  constructor(readonly sessionId: string) {
    super(`session not found: ${sessionId}`);
    this.name = "SessionNotFoundError";
  }
}

// ============================================================================
// Config
// ============================================================================

/**
 * Configuration for the durable outbox. `root` is the outbox root directory
 * (`.spore` by convention); per session the provider derives
 * `{root}/sessions/{session_id}/trace.jsonl`.
 */
export interface OutboxConfig {
  /** Outbox root directory (`.spore` by convention). */
  root: string;
  /** Rotate the active `trace.jsonl` once it exceeds this many bytes. */
  maxSizeBytes: number;
  /** When true, `flushSession` writes the trailing `session` summary line. */
  flushOnSessionEnd: boolean;
}

/** Build an {@link OutboxConfig} with the spec defaults. */
export function outboxConfig(root: string, overrides: Partial<OutboxConfig> = {}): OutboxConfig {
  return {
    root,
    maxSizeBytes: overrides.maxSizeBytes ?? DEFAULT_MAX_SIZE_BYTES,
    flushOnSessionEnd: overrides.flushOnSessionEnd ?? true,
  };
}

function sessionDir(config: OutboxConfig, sessionId: string): string {
  return join(config.root, "sessions", sessionId);
}

// ============================================================================
// Identity helpers (tolerate both string and {toJSON|asString} id wrappers)
// ============================================================================

/** Coerce a harness id (SpanId / SessionId / TaskId class, or a plain string
 *  as the fixtures supply) to its canonical string form. */
function idToString(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") {
    const o = value as { asString?: () => string; toJSON?: () => string; value?: string };
    if (typeof o.asString === "function") return o.asString();
    if (typeof o.toJSON === "function") return o.toJSON();
    if (typeof o.value === "string") return o.value;
  }
  return String(value);
}

function optIdToString(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  return idToString(value);
}

// ============================================================================
// Bare status mapping (the line format, NOT the tagged SpanStatus shape)
// ============================================================================

/** Map a {@link SpanStatus} to the bare `(status, status_detail)` pair the
 *  JSONL schema requires. The tagged `{ kind: ... }` form is NOT used. */
function statusPair(status: SpanStatus): { status: string; status_detail: string | null } {
  switch (status.kind) {
    case "ok":
      return { status: "ok", status_detail: null };
    case "error":
      return { status: "error", status_detail: status.message };
    case "halted":
      return { status: "halted", status_detail: status.reason };
  }
}

// ============================================================================
// TraceLine envelope
// ============================================================================

/**
 * One on-disk JSONL line. Common envelope fields are top-level; the per-kind
 * payload lives under `attributes`. Built by the `from*` / `sessionSummary`
 * constructors — these are the unit-tested, cross-language mapping surface.
 * The `attributes` values are the verbatim serde/JSON serialization of the
 * payload fields (structured unions stay tagged objects; scalars stay scalar;
 * keys are NOT renamed/flattened).
 */
export interface TraceLine {
  trace_id: string;
  span_id: string;
  parent_span_id: string | null;
  session_id: string;
  task_id: string;
  kind: string;
  level: string;
  timestamp: string;
  started_at: string;
  duration_ms: number;
  status: string;
  status_detail: string | null;
  attributes: Record<string, unknown>;
}

/**
 * Round-trip a payload field through JSON so id-wrappers (`SpanId`, etc.) and
 * tagged unions serialize exactly as they would on the wire — verbatim serde.
 */
function jsonValue(v: unknown): unknown {
  if (v === undefined) return null;
  return JSON.parse(JSON.stringify(v));
}

function envelopeFromBase(
  base: SpanBase,
  traceId: string,
  kind: string,
  level: string,
  attributes: Record<string, unknown>,
): TraceLine {
  const { status, status_detail } = statusPair(base.status);
  return {
    trace_id: traceId,
    span_id: idToString(base.span_id),
    parent_span_id: optIdToString(base.parent_span_id),
    session_id: idToString(base.session_id),
    task_id: idToString(base.task_id),
    kind,
    level,
    timestamp: idToString(base.ended_at),
    started_at: idToString(base.started_at),
    duration_ms: base.duration_ms,
    status,
    status_detail,
    attributes,
  };
}

export const TraceLine = {
  fromTurn(span: TurnSpan, traceId: string): TraceLine {
    return envelopeFromBase(span.base, traceId, "turn", "info", {
      turn_number: span.turn_number,
      input_tokens: span.input_tokens,
      output_tokens: span.output_tokens,
      cache_read_tokens: span.cache_read_tokens ?? null,
      cache_write_tokens: span.cache_write_tokens ?? null,
      cost_usd: span.cost_usd,
      stop_reason: jsonValue(span.stop_reason),
      tool_calls_requested: span.tool_calls_requested,
    });
  },

  fromToolCall(span: ToolCallSpan, traceId: string): TraceLine {
    return envelopeFromBase(span.base, traceId, "tool_call", "info", {
      tool_name: span.tool_name,
      call_id: span.call_id,
      parameters_size_bytes: span.parameters_size_bytes,
      output_size_bytes: span.output_size_bytes,
      truncated: span.truncated,
      sandbox_mode: span.sandbox_mode,
      sandbox_violations: jsonValue(span.sandbox_violations),
    });
  },

  fromSensor(span: SensorSpan, traceId: string): TraceLine {
    return envelopeFromBase(span.base, traceId, "sensor_evaluation", "info", {
      sensor_id: idToString(span.sensor_id),
      sensor_kind: jsonValue(span.sensor_kind),
      trigger: jsonValue(span.trigger),
      outcome: jsonValue(span.outcome),
      fired: span.fired,
    });
  },

  fromContext(span: ContextSpan, traceId: string): TraceLine {
    // Mirror emitContext: compaction → "compaction"; all else → "context_assembly".
    const kind = span.operation.kind === "compaction" ? "compaction" : "context_assembly";
    return envelopeFromBase(span.base, traceId, kind, "info", {
      operation: jsonValue(span.operation),
      tokens_before: span.tokens_before,
      tokens_after: span.tokens_after,
      utilization_before: span.utilization_before,
      utilization_after: span.utilization_after,
    });
  },

  fromMiddleware(span: MiddlewareSpan, traceId: string): TraceLine {
    return envelopeFromBase(span.base, traceId, "middleware_hook", "info", {
      hook: jsonValue(span.hook),
      decision: jsonValue(span.decision),
    });
  },

  fromPatch(span: PatchSpan, traceId: string): TraceLine {
    // Patch spans are ALWAYS warn-level.
    return envelopeFromBase(span.base, traceId, "patch", "warn", {
      tool_name: span.tool_name,
      call_id: span.call_id,
      patch_type: jsonValue(span.patch_type),
      original_parameters: jsonValue(span.original_parameters),
      patched_parameters: jsonValue(span.patched_parameters),
    });
  },

  /**
   * Build the trailing `session` summary line from rolled-up metrics. The
   * envelope identity (span/parent ids, task id, timing, status) come from
   * `root`; `attributes.outcome` is the bare-string serialization of the
   * {@link SessionOutcome} in `metrics`.
   */
  sessionSummary(metrics: SessionMetrics, traceId: string, root: SpanBase): TraceLine {
    const outcome = bareOutcome(metrics.outcome);
    return envelopeFromBase(root, traceId, "session", "info", {
      outcome,
      total_turns: metrics.total_turns,
      total_cost_usd: metrics.total_cost_usd,
      sensor_fires: metrics.sensor_fires,
      sensor_halts: metrics.sensor_halts,
      patch_count: metrics.patch_count,
    });
  },

  toJsonl(line: TraceLine): string {
    return `${JSON.stringify(line)}\n`;
  },
} as const;

function bareOutcome(outcome: SessionOutcome): string {
  switch (outcome.kind) {
    case "success":
      return "success";
    case "failure":
      return "failure";
    case "partial":
      return "partial";
  }
}

// ============================================================================
// id generation
// ============================================================================

/** Fresh 32-hex (16 random bytes) trace id, generated once per session. */
function newTraceId(): string {
  return randomBytes(16).toString("hex");
}

/** Derive an 8-byte OTLP span id from the harness SpanId string by SHA-256
 *  hashing and taking the first 8 bytes (matches the Rust reference). */
function deriveOtlpSpanId(spanId: string): Buffer {
  return createHash("sha256").update(spanId).digest().subarray(0, 8);
}

// ============================================================================
// OTLP forwarder (internal interface — see file-header deviation note)
// ============================================================================

/**
 * Internal abstraction over the OTLP forwarding hop. The durable JSONL path
 * does not depend on this interface; it exists only so the heavy/version-churny
 * OpenTelemetry SDK can be isolated and so tests run hermetically.
 */
interface OtlpForwarder {
  /** Forward one already-built line. Best-effort, non-blocking, never throws. */
  forward(line: TraceLine): void;
  /** Best-effort force-flush with a timeout. Logs on failure; never throws. */
  forceFlush(): Promise<void>;
}

/** No-op forwarder used when `SPORE_OTLP_ENDPOINT` is unset/empty (and as the
 *  fallback when the OpenTelemetry SDK is not installed). */
class NullOtlpForwarder implements OtlpForwarder {
  forward(_line: TraceLine): void {}
  async forceFlush(): Promise<void> {}
}

/**
 * Real OTLP forwarder backed by the OpenTelemetry JS SDK, loaded lazily via a
 * dynamic `import()`. A batch span processor makes export buffered and
 * non-blocking. If the SDK is not present this degrades to a no-op and logs a
 * warning — the durable JSONL file remains the source of truth either way.
 */
// Minimal structural typings for the slice of the OpenTelemetry JS surface
// used here. The SDK is dynamically imported, so these stand in for the
// `@opentelemetry/*` type packages (which are intentionally not build deps).
interface OtelSpan {
  setAttribute(key: string, value: unknown): void;
  setStatus(status: { code: number; message?: string }): void;
  end(): void;
}
interface OtelTracer {
  startSpan(name: string, options?: unknown, context?: unknown): OtelSpan;
}
interface OtelApi {
  trace: {
    setSpanContext(context: unknown, spanContext: unknown): unknown;
  };
  context: unknown;
  ROOT_CONTEXT: unknown;
  TraceFlags: { SAMPLED: number };
  SpanKind: { INTERNAL: number };
  SpanStatusCode: { OK: number; ERROR: number };
}

class OtelSdkOtlpForwarder implements OtlpForwarder {
  private provider: { forceFlush(): Promise<void> } | null = null;
  private tracer: OtelTracer | null = null;
  private api: OtelApi | null = null;
  private readonly ready: Promise<void>;

  constructor(private readonly endpoint: string) {
    this.ready = this.init();
  }

  private async init(): Promise<void> {
    try {
      const [{ NodeTracerProvider }, { BatchSpanProcessor }, { OTLPTraceExporter }, api] =
        await Promise.all([
          import(/* @vite-ignore */ "@opentelemetry/sdk-trace-node" as string),
          import(/* @vite-ignore */ "@opentelemetry/sdk-trace-base" as string),
          import(/* @vite-ignore */ "@opentelemetry/exporter-trace-otlp-grpc" as string),
          import(/* @vite-ignore */ "@opentelemetry/api" as string),
        ]);
      const exporter = new OTLPTraceExporter({ url: this.endpoint });
      const provider = new NodeTracerProvider();
      provider.addSpanProcessor(new BatchSpanProcessor(exporter));
      this.provider = provider;
      this.tracer = provider.getTracer("spore-core") as OtelTracer;
      this.api = api as OtelApi;
    } catch (err) {
      console.warn(
        `[spore-core] OpenTelemetry SDK unavailable for '${this.endpoint}'; JSONL only`,
        err instanceof Error ? err.message : err,
      );
      this.provider = null;
    }
  }

  forward(line: TraceLine): void {
    // Fire-and-forget: wait for lazy init, then emit into the batch processor.
    void this.ready.then(() => {
      const tracer = this.tracer;
      const api = this.api;
      if (!tracer || !api) return;
      try {
        // Force the emitted OTLP span onto the harness 32-hex `trace_id` so all
        // spans of a session collapse into one Tempo trace under that exact id
        // (resolved decision #3). Without an explicit parent SpanContext the SDK
        // would mint a fresh trace id per span and the Loki→Tempo derived-field
        // join — which opens the trace whose id == the JSONL `trace_id` — would
        // never resolve. The 8-byte parent span id is derived from the harness
        // SpanId string by hashing, matching the Rust reference's `forward()`.
        const parentSpanIdHex = deriveOtlpSpanId(line.span_id).toString("hex");
        const spanContext = {
          traceId: line.trace_id,
          spanId: parentSpanIdHex,
          traceFlags: api.TraceFlags.SAMPLED,
          isRemote: true,
        };
        const ctx = api.trace.setSpanContext(api.ROOT_CONTEXT, spanContext);
        const span = tracer.startSpan(line.kind, { kind: api.SpanKind.INTERNAL }, ctx);

        span.setAttribute("session_id", line.session_id);
        span.setAttribute("task_id", line.task_id);
        span.setAttribute("level", line.level);
        span.setAttribute("status", line.status);
        // Harmless cross-ref: the readable harness trace_id / span id string.
        span.setAttribute("trace_id", line.trace_id);
        span.setAttribute("span_id_hex", parentSpanIdHex);
        if (line.parent_span_id) span.setAttribute("parent_span_id", line.parent_span_id);

        if (line.status === "ok") {
          span.setStatus({ code: api.SpanStatusCode.OK });
        } else {
          span.setStatus({
            code: api.SpanStatusCode.ERROR,
            message: line.status_detail ?? "",
          });
        }
        span.end();
      } catch {
        // best-effort; JSONL is durable
      }
    });
  }

  async forceFlush(): Promise<void> {
    await this.ready;
    if (!this.provider) return;
    const provider = this.provider;
    try {
      await Promise.race([
        provider.forceFlush(),
        new Promise<void>((resolve) => setTimeout(resolve, OTLP_FLUSH_TIMEOUT_MS)),
      ]);
    } catch (err) {
      console.warn(
        "[spore-core] OTLP forceFlush failed (JSONL is durable):",
        err instanceof Error ? err.message : err,
      );
    }
  }
}

/** Read `SPORE_OTLP_ENDPOINT` once: empty/whitespace is treated as unset. */
function makeForwarder(): OtlpForwarder {
  const ep = process.env.SPORE_OTLP_ENDPOINT;
  if (ep && ep.trim().length > 0) {
    return new OtelSdkOtlpForwarder(ep.trim());
  }
  return new NullOtlpForwarder();
}

// ============================================================================
// SessionWriter — per-session open fd + rotation + trace_id
// ============================================================================

class SessionWriter {
  private fd: number;
  private bytesWritten: number;
  private nextSeq: number;
  readonly traceId: string;

  private constructor(
    fd: number,
    bytesWritten: number,
    nextSeq: number,
    private readonly dir: string,
    private readonly activePath: string,
    private readonly maxSizeBytes: number,
  ) {
    this.fd = fd;
    this.bytesWritten = bytesWritten;
    this.nextSeq = nextSeq;
    this.traceId = newTraceId();
  }

  static open(dir: string, maxSizeBytes: number): SessionWriter {
    mkdirSync(dir, { recursive: true });
    const activePath = join(dir, "trace.jsonl");
    const fd = openSync(activePath, "a");
    const bytesWritten = fstatSync(fd).size;
    const nextSeq = SessionWriter.scanNextSeq(dir);
    return new SessionWriter(fd, bytesWritten, nextSeq, dir, activePath, maxSizeBytes);
  }

  private static scanNextSeq(dir: string): number {
    let maxSeen = 0;
    for (const name of readdirSync(dir)) {
      const m = /^trace-(\d+)\.jsonl$/.exec(name);
      if (m) maxSeen = Math.max(maxSeen, Number.parseInt(m[1], 10));
    }
    return maxSeen + 1;
  }

  /** Append one line, fsync, then rotate if the active file is now over size. */
  append(line: string): void {
    const bytes = Buffer.byteLength(line, "utf8");
    writeSync(this.fd, line);
    fsyncSync(this.fd);
    this.bytesWritten += bytes;
    if (this.bytesWritten > this.maxSizeBytes) {
      this.rotate();
    }
  }

  private rotate(): void {
    const rotated = join(this.dir, `trace-${String(this.nextSeq).padStart(3, "0")}.jsonl`);
    this.nextSeq += 1;
    fsyncSync(this.fd);
    closeSync(this.fd);
    renameSync(this.activePath, rotated);
    this.fd = openSync(this.activePath, "a");
    this.bytesWritten = 0;
  }

  flush(): void {
    fsyncSync(this.fd);
  }

  close(): void {
    try {
      closeSync(this.fd);
    } catch {
      // already closed / removed
    }
  }
}

// ============================================================================
// OutboxObservabilityProvider
// ============================================================================

/**
 * A durable, JSONL-backed {@link ObservabilityProvider} (issue #33). Wraps an
 * {@link InMemoryObservabilityProvider} for all buffering / metrics / query
 * behavior and adds: one synchronous JSONL line per `emit*`, optional
 * best-effort OTLP forwarding, rotation, and `.flushed` markers.
 */
export class OutboxObservabilityProvider implements ObservabilityProvider {
  private readonly writers = new Map<string, SessionWriter>();
  private readonly otlp: OtlpForwarder;

  constructor(
    private readonly config: OutboxConfig,
    private readonly inner: InMemoryObservabilityProvider = new InMemoryObservabilityProvider(),
  ) {
    this.otlp = makeForwarder();
  }

  /** Access the wrapped in-memory provider (e.g. to call `setSessionOutcome` /
   *  `recordGuidesUsed`). */
  innerProvider(): InMemoryObservabilityProvider {
    return this.inner;
  }

  /** The per-session `trace_id`, opening the session writer if needed. */
  traceIdFor(sessionId: SessionId): string {
    return this.writerFor(sessionId.asString()).traceId;
  }

  private writerFor(key: string): SessionWriter {
    let w = this.writers.get(key);
    if (!w) {
      w = SessionWriter.open(sessionDir(this.config, key), this.config.maxSizeBytes);
      this.writers.set(key, w);
    }
    return w;
  }

  /** Append a built line to the session's JSONL file + forward to OTLP. */
  private writeLine(sessionId: string, line: TraceLine): void {
    try {
      this.writerFor(sessionId).append(TraceLine.toJsonl(line));
    } catch (err) {
      console.error("[spore-core] outbox append failed:", err instanceof Error ? err.message : err);
      return;
    }
    this.otlp.forward(line);
  }

  private traceIdLocked(sessionId: string): string {
    try {
      return this.writerFor(sessionId).traceId;
    } catch {
      return "";
    }
  }

  emitTurn(span: TurnSpan): void {
    const sid = idToString(span.base.session_id);
    const line = TraceLine.fromTurn(span, this.traceIdLocked(sid));
    this.inner.emitTurn(span);
    this.writeLine(sid, line);
  }

  emitToolCall(span: ToolCallSpan): void {
    const sid = idToString(span.base.session_id);
    const line = TraceLine.fromToolCall(span, this.traceIdLocked(sid));
    this.inner.emitToolCall(span);
    this.writeLine(sid, line);
  }

  emitSensor(span: SensorSpan): void {
    const sid = idToString(span.base.session_id);
    const line = TraceLine.fromSensor(span, this.traceIdLocked(sid));
    this.inner.emitSensor(span);
    this.writeLine(sid, line);
  }

  emitContext(span: ContextSpan): void {
    const sid = idToString(span.base.session_id);
    const line = TraceLine.fromContext(span, this.traceIdLocked(sid));
    this.inner.emitContext(span);
    this.writeLine(sid, line);
  }

  emitMiddleware(span: MiddlewareSpan): void {
    const sid = idToString(span.base.session_id);
    const line = TraceLine.fromMiddleware(span, this.traceIdLocked(sid));
    this.inner.emitMiddleware(span);
    this.writeLine(sid, line);
  }

  emitPatch(span: PatchSpan): void {
    const sid = idToString(span.base.session_id);
    const line = TraceLine.fromPatch(span, this.traceIdLocked(sid));
    this.inner.emitPatch(span);
    this.writeLine(sid, line);
  }

  /** Record the terminal outcome for a session; forwards to the wrapped
   *  in-memory provider so the trailing `session` summary line reflects it. */
  setSessionOutcome(sessionId: SessionId, outcome: SessionOutcome): void {
    this.inner.setSessionOutcome(sessionId, outcome);
  }

  async flushSession(sessionId: SessionId): Promise<void> {
    const sid = sessionId.asString();

    // (a) Write the trailing session summary line from the in-memory metrics.
    if (this.config.flushOnSessionEnd) {
      const metrics = await this.inner.getSessionMetrics(sessionId);
      if (metrics) {
        const traceId = this.traceIdLocked(sid);
        // Synthetic root base for the summary line's identity, mirroring Rust.
        const root: SpanBase = {
          span_id: sid as unknown as SpanBase["span_id"],
          parent_span_id: null,
          session_id: sessionId,
          task_id: metrics.task_id,
          kind: "session",
          started_at: "" as unknown as Timestamp,
          ended_at: "" as unknown as Timestamp,
          duration_ms: metrics.total_duration_ms,
          status: { kind: "ok" },
        };
        this.writeLine(sid, TraceLine.sessionSummary(metrics, traceId, root));
      }
    }

    // (b) Flush the JSONL file handle.
    const w = this.writers.get(sid);
    if (w) {
      try {
        w.flush();
      } catch {
        // best-effort
      }
    }

    // Best-effort OTLP force-flush; never throws out of flushSession.
    await this.otlp.forceFlush();

    // (c) Create the sibling `.flushed` marker.
    const dir = sessionDir(this.config, sid);
    if (existsSync(dir)) {
      try {
        writeFileSync(join(dir, ".flushed"), "");
      } catch (err) {
        console.error(
          "[spore-core] failed to write .flushed marker:",
          err instanceof Error ? err.message : err,
        );
      }
    }

    // Delegate to inner so its `flushed` bookkeeping stays consistent.
    await this.inner.flushSession(sessionId);
  }

  getSessionMetrics(sessionId: SessionId): Promise<SessionMetrics | undefined> {
    return this.inner.getSessionMetrics(sessionId);
  }

  getSessions(
    since: Timestamp,
    domain?: string,
    outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]> {
    return this.inner.getSessions(since, domain, outcome);
  }

  getTrace(sessionId: SessionId): Promise<Span[]> {
    return this.inner.getTrace(sessionId);
  }

  /** Session ids whose durable outbox has a `trace.jsonl` but no `.flushed`
   *  marker (issue #33). */
  async listUnflushedSessions(): Promise<SessionId[]> {
    const out: SessionId[] = [];
    const sessionsDir = join(this.config.root, "sessions");
    if (!existsSync(sessionsDir)) return out;
    for (const name of readdirSync(sessionsDir)) {
      const dir = join(sessionsDir, name);
      if (!statSync(dir).isDirectory()) continue;
      const hasTrace = existsSync(join(dir, "trace.jsonl"));
      const flushed = existsSync(join(dir, ".flushed"));
      if (hasTrace && !flushed) out.push(SessionIdClass.of(name));
    }
    out.sort((a, b) => a.asString().localeCompare(b.asString()));
    return out;
  }

  /** Delete a session's durable outbox (issue #33). Throws
   *  {@link SessionNotFoundError} if the dir does not exist. NEVER
   *  auto-deletes. */
  async cleanupSession(sessionId: SessionId): Promise<void> {
    const sid = sessionId.asString();
    const dir = sessionDir(this.config, sid);
    if (!existsSync(dir)) {
      throw new SessionNotFoundError(sid);
    }
    const w = this.writers.get(sid);
    if (w) {
      w.close();
      this.writers.delete(sid);
    }
    rmSync(dir, { recursive: true, force: true });
  }
}
