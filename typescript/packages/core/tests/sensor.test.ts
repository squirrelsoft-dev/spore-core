/**
 * Unit tests for the canonical SensorChain (spore-core issue #10).
 *
 * Mirrors `rust/crates/spore-core/src/sensor.rs#tests` rule-for-rule.
 */

import { describe, expect, it } from "vitest";

import { sensor, SessionId } from "../src/index.js";
import { emptySessionState } from "../src/harness/types.js";

const { StandardSensorChain, SensorId, Timestamp, SensorError, newSensorInput } = sensor;

type Sensor = sensor.Sensor;
type SensorConfig = sensor.SensorConfig;
type SensorInput = sensor.SensorInput;
type SensorOutcome = sensor.SensorOutcome;
type SensorTrigger = sensor.SensorTrigger;
type SensorResult = sensor.SensorResult;

// ── Builders ────────────────────────────────────────────────────────────────

function ts(s: string): sensor.Timestamp {
  return new Timestamp(s);
}

class StubSensor implements Sensor {
  constructor(
    private readonly cfg: SensorConfig,
    private readonly outcome: SensorOutcome,
  ) {}
  async evaluate(_input: SensorInput): Promise<SensorResult> {
    return {
      sensor_id: new SensorId(this.cfg.id.value),
      outcome: this.outcome,
      observation: this.outcome === "warn" ? "warn-obs" : null,
      detail: this.outcome,
      fired_at: ts("2026-05-16T00:00:00Z"),
    };
  }
  config(): SensorConfig {
    return this.cfg;
  }
}

function computational(
  id: string,
  triggers: SensorTrigger[],
  outcome: SensorOutcome,
  overrides: Partial<SensorConfig> = {},
): StubSensor {
  return new StubSensor(
    {
      id: new SensorId(id),
      name: id,
      kind: "computational",
      triggers,
      run_every_n_turns: null,
      run_on_phases: null,
      low_signal_threshold: sensor.defaultSensorSignalThresholds(),
      ...overrides,
    },
    outcome,
  );
}

function inferential(
  id: string,
  triggers: SensorTrigger[],
  outcome: SensorOutcome,
  every_n: number | null,
  phases: sensor.SensorConfig["run_on_phases"],
): StubSensor {
  return new StubSensor(
    {
      id: new SensorId(id),
      name: id,
      kind: "inferential",
      triggers,
      run_every_n_turns: every_n,
      run_on_phases: phases,
      low_signal_threshold: sensor.defaultSensorSignalThresholds(),
    },
    outcome,
  );
}

function input(sid: string): SensorInput {
  return newSensorInput(new SessionId(sid), emptySessionState());
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("SensorChain — register validation", () => {
  it("register_rejects_empty_triggers", () => {
    const chain = new StandardSensorChain();
    const s = computational("s1", [], "pass");
    expect(() => chain.register(s)).toThrow(SensorError);
    try {
      chain.register(s);
    } catch (e) {
      expect((e as sensor.SensorError).kind).toBe("validation_failed");
    }
  });

  it("register_rejects_duplicate_ids", () => {
    const chain = new StandardSensorChain();
    chain.register(computational("s1", [{ kind: "post_turn" }], "pass"));
    try {
      chain.register(computational("s1", [{ kind: "post_turn" }], "pass"));
      expect.fail("expected duplicate registration to throw");
    } catch (e) {
      expect(e).toBeInstanceOf(SensorError);
      expect((e as sensor.SensorError).kind).toBe("already_registered");
    }
  });
});

describe("SensorChain — fire", () => {
  it("runs all matching sensors with no short circuit", async () => {
    const chain = new StandardSensorChain();
    chain.register(computational("pass", [{ kind: "post_turn" }], "pass"));
    chain.register(computational("warn", [{ kind: "post_turn" }], "warn"));
    chain.register(computational("halt", [{ kind: "post_turn" }], "halt"));
    const results = await chain.fire({ kind: "post_turn" }, input("s1"));
    expect(results).toHaveLength(3);
    const outcomes = new Set(results.map((r) => r.outcome));
    expect(outcomes.has("pass")).toBe(true);
    expect(outcomes.has("warn")).toBe(true);
    expect(outcomes.has("halt")).toBe(true);
  });

  it("filters by trigger", async () => {
    const chain = new StandardSensorChain();
    chain.register(computational("post-turn", [{ kind: "post_turn" }], "pass"));
    chain.register(computational("post-session", [{ kind: "post_session" }], "pass"));
    const results = await chain.fire({ kind: "post_turn" }, input("s1"));
    expect(results).toHaveLength(1);
    expect(results[0]!.sensor_id.value).toBe("post-turn");
  });

  it("post_tool wildcard and named matching", async () => {
    const chain = new StandardSensorChain();
    chain.register(computational("any", [{ kind: "post_tool", tool_name: "" }], "pass"));
    chain.register(computational("bash-only", [{ kind: "post_tool", tool_name: "bash" }], "pass"));
    const r1 = await chain.fire({ kind: "post_tool", tool_name: "bash" }, input("s1"));
    expect(r1).toHaveLength(2);
    const r2 = await chain.fire({ kind: "post_tool", tool_name: "edit" }, input("s2"));
    expect(r2).toHaveLength(1);
    expect(r2[0]!.sensor_id.value).toBe("any");
  });

  it("computational sensors ignore turn gating", async () => {
    const chain = new StandardSensorChain();
    // Even with `run_every_n_turns` set, computational fires every time.
    chain.register(computational("c", [{ kind: "post_turn" }], "pass", { run_every_n_turns: 99 }));
    const i = input("s1");
    i.turn_number = 1;
    const r = await chain.fire({ kind: "post_turn" }, i);
    expect(r).toHaveLength(1);
  });

  it("inferential gated by run_every_n_turns", async () => {
    const chain = new StandardSensorChain();
    chain.register(inferential("judge", [{ kind: "post_turn" }], "warn", 3, null));
    const i = input("s1");
    i.turn_number = 1;
    expect(await chain.fire({ kind: "post_turn" }, i)).toHaveLength(0);
    i.turn_number = 2;
    expect(await chain.fire({ kind: "post_turn" }, i)).toHaveLength(0);
    i.turn_number = 3;
    expect(await chain.fire({ kind: "post_turn" }, i)).toHaveLength(1);
    i.turn_number = 6;
    expect(await chain.fire({ kind: "post_turn" }, i)).toHaveLength(1);
  });

  it("inferential gated by run_on_phases", async () => {
    const chain = new StandardSensorChain();
    chain.register(inferential("judge", [{ kind: "post_turn" }], "pass", null, ["execution"]));
    const i = input("s1");
    i.phase = "planning";
    expect(await chain.fire({ kind: "post_turn" }, i)).toHaveLength(0);
    i.phase = "execution";
    expect(await chain.fire({ kind: "post_turn" }, i)).toHaveLength(1);
  });
});

describe("SensorChain — stats", () => {
  it("aggregates outcomes and fire_rate", async () => {
    const chain = new StandardSensorChain();
    chain.register(computational("warner", [{ kind: "post_turn" }], "warn"));
    for (let i = 0; i < 4; i++) {
      await chain.fire({ kind: "post_turn" }, input(`s${i}`));
    }
    const stats = await chain.stats();
    expect(stats).toHaveLength(1);
    const s = stats[0]!;
    expect(s.total_fires).toBe(4);
    expect(s.warn_count).toBe(4);
    expect(s.halt_count).toBe(0);
    expect(s.pass_count).toBe(0);
    expect(Math.abs(s.fire_rate - 1.0)).toBeLessThan(1e-6);
  });

  it("history is recorded in fire order", async () => {
    const chain = new StandardSensorChain();
    chain.register(computational("s1", [{ kind: "post_turn" }], "pass"));
    await chain.fire({ kind: "post_turn" }, input("a"));
    await chain.fire({ kind: "post_turn" }, input("b"));
    const stats = await chain.stats();
    expect(stats[0]!.total_fires).toBe(2);
  });
});

describe("SensorChain — signalQualityReport", () => {
  it("flags AlwaysFiring", async () => {
    const chain = new StandardSensorChain();
    chain.register(
      computational("noisy", [{ kind: "post_turn" }], "warn", {
        low_signal_threshold: { never_fired_after_n_sessions: 100, always_fired_rate: 0.5 },
      }),
    );
    for (let i = 0; i < 5; i++) {
      await chain.fire({ kind: "post_turn" }, input(`s${i}`));
    }
    const flags = await chain.signalQualityReport(5);
    expect(flags.some((f) => f.kind === "always_firing" && f.sensor_id.value === "noisy")).toBe(
      true,
    );
  });

  it("flags NeverFired", async () => {
    const chain = new StandardSensorChain();
    chain.register(
      computational("quiet", [{ kind: "post_session" }], "pass", {
        low_signal_threshold: { never_fired_after_n_sessions: 3, always_fired_rate: 0.9 },
      }),
    );
    // Fire PostTurn many times so chain observes the sessions; quiet never fires.
    for (let i = 0; i < 5; i++) {
      await chain.fire({ kind: "post_turn" }, input(`s${i}`));
    }
    const flags = await chain.signalQualityReport(3);
    expect(
      flags.some(
        (f) =>
          f.kind === "never_fired" && f.sensor_id.value === "quiet" && f.sessions_observed >= 3,
      ),
    ).toBe(true);
  });

  it("returns empty when below min_sessions", async () => {
    const chain = new StandardSensorChain();
    chain.register(computational("quiet", [{ kind: "post_session" }], "pass"));
    await chain.fire({ kind: "post_turn" }, input("s1"));
    const flags = await chain.signalQualityReport(10);
    expect(flags).toHaveLength(0);
  });
});
