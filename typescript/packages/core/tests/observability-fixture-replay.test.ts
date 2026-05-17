/**
 * Fixture-replay tests for the canonical ObservabilityProvider
 * (spore-core issue #12).
 *
 * Loads `fixtures/observability/session_metrics_basic.json` and asserts the
 * aggregated totals exactly match the Rust, Python, and Go suites.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { memory, observability, SessionId, TaskId, type guideRegistry } from "../src/index.js";

const { InMemoryObservabilityProvider, SpanId } = observability;
const { Timestamp } = memory;

interface FixtureTurn {
  span_id: string;
  turn: number;
  input: number;
  output: number;
}

interface FixtureCase {
  session_id: string;
  turns: FixtureTurn[];
  outcome: string;
  expected: {
    total_turns: number;
    total_input_tokens: number;
    total_output_tokens: number;
  };
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/observability/session_metrics_basic.json");

function outcomeOf(raw: string): guideRegistry.SessionOutcome {
  switch (raw) {
    case "success":
      return { kind: "success" };
    case "partial":
      return { kind: "partial" };
    default:
      return { kind: "failure", reason: raw };
  }
}

describe("Observability fixture replay", () => {
  it("session_metrics_basic.json aggregates match expected totals", async () => {
    const raw = readFileSync(fixturePath, "utf8");
    const fixture = JSON.parse(raw) as FixtureCase;

    const obs = new InMemoryObservabilityProvider();
    const sessionId = SessionId.of(fixture.session_id);
    const taskId = TaskId.of("fixture_task");

    for (const t of fixture.turns) {
      obs.emitTurn({
        base: {
          span_id: SpanId.of(t.span_id),
          parent_span_id: null,
          session_id: sessionId,
          task_id: taskId,
          kind: "turn",
          started_at: Timestamp.of("2026-05-16T00:00:00Z"),
          ended_at: Timestamp.of("2026-05-16T00:00:00Z"),
          duration_ms: 0,
          status: { kind: "ok" },
        },
        turn_number: t.turn,
        input_tokens: t.input,
        output_tokens: t.output,
        cache_read_tokens: null,
        cache_write_tokens: null,
        cost_usd: 0,
        stop_reason: "end_turn",
        tool_calls_requested: 0,
      });
    }
    obs.setSessionOutcome(sessionId, outcomeOf(fixture.outcome));

    const metrics = await obs.getSessionMetrics(sessionId);
    expect(metrics).toBeDefined();
    expect(metrics!.total_turns).toBe(fixture.expected.total_turns);
    expect(metrics!.total_input_tokens).toBe(fixture.expected.total_input_tokens);
    expect(metrics!.total_output_tokens).toBe(fixture.expected.total_output_tokens);
  });
});
