/**
 * OTLP multi-endpoint fan-out + ObservabilityStore-leg tests for the refactored
 * OutboxObservabilityProvider (issue #73).
 *
 * The OTLP leg is exercised through the hermetic `forwarders` test seam (the
 * constructor's third argument) with counting fakes — no live OTLP stack. The
 * routing matrix (both / otlp-only / store-only / dropped), failure isolation,
 * and bad-endpoint-skipped rules are all asserted here. Mirrors the Rust
 * reference's fan-out tests in `observability_outbox.rs`.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  OutboxObservabilityProvider,
  outboxConfig,
  type OtlpForwarder,
  type TraceLine,
} from "../src/observability/index.js";
import { SessionId, TaskId } from "../src/harness/types.js";
import { SpanId, type SpanBase, type TurnSpan } from "../src/observability/types.js";
import { Timestamp } from "../src/memory/types.js";
import { InMemoryStorageProvider, parseOtlpEndpoints } from "../src/storage/index.js";

function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), "spore-fanout-"));
}

function base(session: string, spanId: string): SpanBase {
  return {
    span_id: SpanId.of(spanId),
    parent_span_id: null,
    session_id: SessionId.of(session),
    task_id: TaskId.of("task1"),
    kind: "turn",
    started_at: Timestamp.of("2026-05-26T18:00:00.000Z"),
    ended_at: Timestamp.of("2026-05-26T18:00:02.100Z"),
    duration_ms: 2100,
    status: { kind: "ok" },
  };
}

function turn(session: string, spanId: string): TurnSpan {
  return {
    base: base(session, spanId),
    turn_number: 1,
    input_tokens: 1820,
    output_tokens: 140,
    cache_read_tokens: 1600,
    cache_write_tokens: 0,
    cost_usd: 0.0123,
    stop_reason: "tool_use",
    tool_calls_requested: 1,
  };
}

/** Counting fake forwarder — records every forwarded line. */
class CountingForwarder implements OtlpForwarder {
  readonly forwarded: TraceLine[] = [];
  flushCount = 0;
  forward(line: TraceLine): void {
    this.forwarded.push(line);
  }
  async forceFlush(): Promise<void> {
    this.flushCount += 1;
  }
}

/** Forwarder whose forward() throws — proves per-leg failure isolation. */
class ThrowingForwarder implements OtlpForwarder {
  forwardCount = 0;
  forward(_line: TraceLine): void {
    this.forwardCount += 1;
    throw new Error("boom");
  }
  async forceFlush(): Promise<void> {}
}

const ENV_KEY = "SPORE_OTLP_ENDPOINT";
let savedEnv: string | undefined;

beforeEach(() => {
  savedEnv = process.env[ENV_KEY];
  delete process.env[ENV_KEY];
});
afterEach(() => {
  if (savedEnv === undefined) delete process.env[ENV_KEY];
  else process.env[ENV_KEY] = savedEnv;
});

describe("OTLP fan-out routing matrix", () => {
  it("OTLP + store: every span goes to both", async () => {
    const root = tmpDir();
    const f1 = new CountingForwarder();
    const f2 = new CountingForwarder();
    const store = new InMemoryStorageProvider();
    const obs = new OutboxObservabilityProvider(outboxConfig(root), undefined, [f1, f2]).withStore(
      store,
    );

    obs.emitTurn(turn("s1", "sp1"));
    obs.emitTurn(turn("s1", "sp2"));
    await Promise.resolve();

    // Fanned out to both OTLP endpoints.
    expect(f1.forwarded).toHaveLength(2);
    expect(f2.forwarded).toHaveLength(2);
    // And to the store leg.
    expect(await store.getSpans(SessionId.of("s1"))).toHaveLength(2);
    // And the durable JSONL is still the source of truth.
    const jsonl = readFileSync(join(root, "sessions/s1/trace.jsonl"), "utf8");
    expect(jsonl.trim().split("\n")).toHaveLength(2);
  });

  it("OTLP only (no store): spans go to OTLP, store untouched", async () => {
    const root = tmpDir();
    const f1 = new CountingForwarder();
    const obs = new OutboxObservabilityProvider(outboxConfig(root), undefined, [f1]);
    obs.emitTurn(turn("s1", "sp1"));
    await Promise.resolve();
    expect(f1.forwarded).toHaveLength(1);
  });

  it("store only (no OTLP): spans go to the store", async () => {
    const root = tmpDir();
    const store = new InMemoryStorageProvider();
    const obs = new OutboxObservabilityProvider(outboxConfig(root), undefined, []).withStore(store);
    obs.emitTurn(turn("s1", "sp1"));
    await Promise.resolve();
    expect(await store.getSpans(SessionId.of("s1"))).toHaveLength(1);
  });

  it("dropped (no OTLP, no store): JSONL is still written, no throw", async () => {
    const root = tmpDir();
    const obs = new OutboxObservabilityProvider(outboxConfig(root), undefined, []);
    obs.emitTurn(turn("s1", "sp1"));
    await Promise.resolve();
    expect(existsSync(join(root, "sessions/s1/trace.jsonl"))).toBe(true);
  });
});

describe("OTLP fan-out failure isolation", () => {
  it("a throwing forwarder never blocks the others or the store", async () => {
    const root = tmpDir();
    const bad = new ThrowingForwarder();
    const good = new CountingForwarder();
    const store = new InMemoryStorageProvider();
    const obs = new OutboxObservabilityProvider(outboxConfig(root), undefined, [
      bad,
      good,
    ]).withStore(store);

    expect(() => obs.emitTurn(turn("s1", "sp1"))).not.toThrow();
    await Promise.resolve();

    expect(bad.forwardCount).toBe(1);
    expect(good.forwarded).toHaveLength(1);
    expect(await store.getSpans(SessionId.of("s1"))).toHaveLength(1);
    expect(existsSync(join(root, "sessions/s1/trace.jsonl"))).toBe(true);
  });

  it("flushSession force-flushes every forwarder and the store marker", async () => {
    const root = tmpDir();
    const f1 = new CountingForwarder();
    const f2 = new CountingForwarder();
    const store = new InMemoryStorageProvider();
    const obs = new OutboxObservabilityProvider(outboxConfig(root), undefined, [f1, f2]).withStore(
      store,
    );
    obs.emitTurn(turn("s1", "sp1"));
    await obs.flushSession(SessionId.of("s1"));
    expect(f1.flushCount).toBe(1);
    expect(f2.flushCount).toBe(1);
    // The outbox's own .flushed marker exists.
    expect(existsSync(join(root, "sessions/s1/.flushed"))).toBe(true);
  });
});

describe("bad endpoint skipped", () => {
  it("empty / all-empty-segment SPORE_OTLP_ENDPOINT yields no OTLP leg", () => {
    // The parse rule drops empty segments; an all-empty value parses to [], so
    // the provider builds zero forwarders and runs JSONL/store-only.
    expect(parseOtlpEndpoints("a,,b,")).toEqual(["a", "b"]);
    expect(parseOtlpEndpoints("")).toEqual([]);
    expect(parseOtlpEndpoints("  ")).toEqual([]);
  });
});
