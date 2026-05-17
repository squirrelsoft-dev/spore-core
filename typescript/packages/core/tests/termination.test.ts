/**
 * Unit tests for the canonical {@link TerminationPolicy} (spore-core issue #13).
 *
 * Covers every rule in the spec, plus byte-equivalent fixture replay against
 * `fixtures/termination_policy/basic.json`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  SessionId,
  TaskId,
  emptyBudgetSnapshot,
  emptySessionState,
  type BudgetLimits,
  type BudgetSnapshot,
} from "../src/harness/types.js";
import { sensor, termination } from "../src/index.js";
import { EmptyResponse } from "../src/agent/errors.js";
import { SensorResultSchema, type SensorResult } from "../src/sensor/types.js";

const {
  BudgetValue,
  FixedCompletionCheck,
  NullCompletionCheck,
  StandardTerminationPolicy,
  checkBudgetDefault,
  newSessionStateSnapshot,
} = termination;

type TerminationInput = termination.TerminationInput;
type TerminationDecision = termination.TerminationDecision;
type TerminationFailureReason = termination.TerminationFailureReason;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function snapshot() {
  return newSessionStateSnapshot(new SessionId("s1"), new TaskId("t1"), emptySessionState());
}

function inputAt(turn: number, done: boolean): TerminationInput {
  return {
    session_id: new SessionId("s1"),
    task_id: new TaskId("t1"),
    turn_number: turn,
    agent_claims_done: done,
    agent_response: "ok",
    budget_used: emptyBudgetSnapshot(),
    budget_limits: {},
    sensor_results: [],
    session_state: snapshot(),
  };
}

function sensorResult(id: string, outcome: sensor.SensorOutcome): SensorResult {
  return {
    sensor_id: new sensor.SensorId(id),
    outcome,
    observation: null,
    detail: outcome,
    fired_at: new sensor.Timestamp("2026-05-17T00:00:00Z"),
  };
}

// ---------------------------------------------------------------------------
// Rule: budget is always checked first
// ---------------------------------------------------------------------------

describe("StandardTerminationPolicy — budget", () => {
  it("hard-stops when agent claims done", async () => {
    const p = StandardTerminationPolicy.withNullCheck();
    const input = inputAt(1, true);
    input.budget_used = { ...input.budget_used, turns: 5 };
    input.budget_limits = { max_turns: 5 };
    const d = await p.evaluate(input);
    expect(d.kind).toBe("halt_budget_exceeded");
    if (d.kind === "halt_budget_exceeded") {
      expect(d.limit_type).toBe("turns");
    }
  });

  it("hard-stops even when agent has not claimed done", async () => {
    // Budget is checked before agent_claims_done.
    const p = StandardTerminationPolicy.withNullCheck();
    const input = inputAt(1, false);
    input.budget_used = { ...input.budget_used, turns: 5 };
    input.budget_limits = { max_turns: 5 };
    const d = await p.evaluate(input);
    expect(d.kind).toBe("halt_budget_exceeded");
  });

  it("checkBudgetDefault covers every limit type", () => {
    const cases: Array<{
      snap: BudgetSnapshot;
      limits: BudgetLimits;
      want: termination.BudgetValue["kind"] | string;
      limitType: string;
    }> = [
      {
        snap: { ...emptyBudgetSnapshot(), turns: 3 },
        limits: { max_turns: 3 },
        want: "turns",
        limitType: "turns",
      },
      {
        snap: { ...emptyBudgetSnapshot(), input_tokens: 10 },
        limits: { max_input_tokens: 10 },
        want: "tokens",
        limitType: "input_tokens",
      },
      {
        snap: { ...emptyBudgetSnapshot(), output_tokens: 10 },
        limits: { max_output_tokens: 10 },
        want: "tokens",
        limitType: "output_tokens",
      },
      {
        snap: { ...emptyBudgetSnapshot(), wall_time: 10 },
        limits: { max_wall_time: 10 },
        want: "duration",
        limitType: "wall_time",
      },
      {
        snap: { ...emptyBudgetSnapshot(), cost_usd: 1.0 },
        limits: { max_cost_usd: 1.0 },
        want: "usd",
        limitType: "cost_usd",
      },
    ];
    for (const c of cases) {
      const got = checkBudgetDefault(c.snap, c.limits);
      expect(got?.kind).toBe("halt_budget_exceeded");
      if (got?.kind === "halt_budget_exceeded") {
        expect(got.limit_type).toBe(c.limitType);
        expect(got.used.kind).toBe(c.want);
        expect(got.limit.kind).toBe(c.want);
      }
    }
    expect(checkBudgetDefault(emptyBudgetSnapshot(), {})).toBeNull();
  });

  it("BudgetValue constructors produce the expected wire shape", () => {
    expect(BudgetValue.turns(3)).toEqual({ kind: "turns", value: 3 });
    expect(BudgetValue.tokens(10)).toEqual({ kind: "tokens", value: 10 });
    expect(BudgetValue.duration(5)).toEqual({ kind: "duration", value: 5 });
    expect(BudgetValue.usd(1.5)).toEqual({ kind: "usd", value: 1.5 });
  });
});

// ---------------------------------------------------------------------------
// Rule: not-done always continues (after budget)
// ---------------------------------------------------------------------------

describe("StandardTerminationPolicy — not-done", () => {
  it("continues when agent has not claimed done", async () => {
    const p = StandardTerminationPolicy.withNullCheck();
    const d = await p.evaluate(inputAt(1, false));
    expect(d.kind).toBe("continue");
  });
});

// ---------------------------------------------------------------------------
// Rule: sensor halt becomes UnrecoverableSensorHalt
// ---------------------------------------------------------------------------

describe("StandardTerminationPolicy — sensor results", () => {
  it("sensor halt overrides completion success", async () => {
    const p = StandardTerminationPolicy.withNullCheck();
    const input = inputAt(1, true);
    input.sensor_results.push(sensorResult("guardrail", "halt"));
    const d = await p.evaluate(input);
    expect(d.kind).toBe("halt_failure");
    if (d.kind === "halt_failure") {
      expect(d.reason.kind).toBe("unrecoverable_sensor_halt");
      if (d.reason.kind === "unrecoverable_sensor_halt") {
        expect(d.reason.sensor_id.asString()).toBe("guardrail");
      }
    }
  });

  it("sensor warn does not halt", async () => {
    const p = StandardTerminationPolicy.withNullCheck();
    const input = inputAt(1, true);
    input.sensor_results.push(sensorResult("guardrail", "warn"));
    const d = await p.evaluate(input);
    expect(d.kind).toBe("halt_success");
  });
});

// ---------------------------------------------------------------------------
// Rule: completion check Some(reason) ⇒ Continue
// ---------------------------------------------------------------------------

describe("StandardTerminationPolicy — completion check", () => {
  it("incomplete check continues with agent_claims_done=true", async () => {
    const p = new StandardTerminationPolicy(
      FixedCompletionCheck.incomplete("feature B not implemented"),
    );
    const d = await p.evaluate(inputAt(1, true));
    expect(d.kind).toBe("continue");
  });

  it("complete check halts with success and the agent's summary", async () => {
    const p = StandardTerminationPolicy.withNullCheck();
    const input = inputAt(1, true);
    input.agent_response = "all green";
    const d = await p.evaluate(input);
    expect(d.kind).toBe("halt_success");
    if (d.kind === "halt_success") expect(d.summary).toBe("all green");
  });

  it("halt_success summary is '' when agent_response is absent", async () => {
    const p = StandardTerminationPolicy.withNullCheck();
    const input = inputAt(1, true);
    input.agent_response = null;
    const d = await p.evaluate(input);
    expect(d.kind).toBe("halt_success");
    if (d.kind === "halt_success") expect(d.summary).toBe("");
  });

  it("FixedCompletionCheck and NullCompletionCheck expose descriptions", () => {
    expect(new NullCompletionCheck().description()).toBe("null (always complete)");
    expect(FixedCompletionCheck.complete().description()).toBe("fixed:complete");
    expect(FixedCompletionCheck.incomplete("x").description()).toBe("fixed:incomplete");
  });
});

// ---------------------------------------------------------------------------
// Rule: HaltFailure carries typed reason (round-trips on the wire)
// ---------------------------------------------------------------------------

describe("TerminationFailureReason — wire format", () => {
  it("every variant round-trips through JSON with snake_case kind", () => {
    const variants: TerminationFailureReason[] = [
      { kind: "completion_check_failed", detail: "nope" },
      { kind: "max_retries_exhausted", tool: "bash", attempts: 3 },
      {
        kind: "unrecoverable_sensor_halt",
        sensor_id: new sensor.SensorId("g"),
        detail: "tripped",
      },
      { kind: "middleware_halt", hook: "before_turn", reason: "veto" },
      { kind: "agent_error", error: new EmptyResponse() },
      { kind: "policy_violation", detail: "policy" },
      { kind: "human_halted" },
    ];
    for (const v of variants) {
      const json = JSON.parse(JSON.stringify(v)) as Record<string, unknown>;
      expect(json.kind).toBe(v.kind);
    }
  });
});

// ---------------------------------------------------------------------------
// Fixture replay
// ---------------------------------------------------------------------------

interface FixtureCase {
  name: string;
  agent_claims_done: boolean;
  agent_response: string | null;
  budget_used: BudgetSnapshot;
  budget_limits: BudgetLimits;
  sensor_results: unknown[];
  completion_check: { kind: "complete" } | { kind: "incomplete"; reason: string };
  expected: unknown;
}
interface FixtureSuite {
  description?: string;
  cases: FixtureCase[];
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/termination_policy/basic.json");

describe("StandardTerminationPolicy — fixture replay", () => {
  const raw = readFileSync(fixturePath, "utf-8");
  const suite = JSON.parse(raw) as FixtureSuite;

  for (const c of suite.cases) {
    it(c.name, async () => {
      const check =
        c.completion_check.kind === "complete"
          ? FixedCompletionCheck.complete()
          : FixedCompletionCheck.incomplete(c.completion_check.reason);
      const policy = new StandardTerminationPolicy(check);
      const input: TerminationInput = {
        session_id: new SessionId("fixture"),
        task_id: new TaskId("fixture-task"),
        turn_number: 1,
        agent_claims_done: c.agent_claims_done,
        agent_response: c.agent_response,
        budget_used: c.budget_used,
        budget_limits: c.budget_limits,
        sensor_results: c.sensor_results.map((r) => SensorResultSchema.parse(r)),
        session_state: snapshot(),
      };
      const got = await policy.evaluate(input);
      // Compare via JSON to bypass class instances on the wire side.
      const gotJson = JSON.parse(JSON.stringify(got)) as unknown;
      expect(gotJson).toEqual(c.expected);
    });
  }
});

// ---------------------------------------------------------------------------
// Discriminator-shape regression — TerminationDecision is snake_case on wire
// ---------------------------------------------------------------------------

describe("TerminationDecision — wire format", () => {
  it("uses snake_case discriminator values", () => {
    const cases: TerminationDecision[] = [
      { kind: "continue" },
      { kind: "halt_success", summary: "" },
      { kind: "halt_failure", reason: { kind: "human_halted" } },
      {
        kind: "halt_budget_exceeded",
        limit_type: "turns",
        used: BudgetValue.turns(1),
        limit: BudgetValue.turns(1),
      },
    ];
    const kinds = cases.map((d) => d.kind);
    expect(kinds).toEqual(["continue", "halt_success", "halt_failure", "halt_budget_exceeded"]);
  });
});
