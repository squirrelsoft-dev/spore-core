/**
 * Fixture-replay tests for the canonical SensorChain (spore-core issue #10).
 *
 * Loads `fixtures/sensor_chain/signal_quality_basic.json` and asserts the
 * NeverFired / AlwaysFiring sets exactly match the Rust, Python, and Go
 * suites.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { sensor, SessionId } from "../src/index.js";
import { emptySessionState } from "../src/harness/types.js";

const { StandardSensorChain, SensorId, newSensorInput } = sensor;

interface FixtureSensor {
  id: string;
  kind: sensor.SensorKind;
  triggers: sensor.SensorTrigger[];
  outcome: sensor.SensorOutcome;
  thresholds: sensor.SensorSignalThresholds;
}
interface FixtureEvent {
  trigger: sensor.SensorTrigger;
  session_id: string;
}
interface FixtureCase {
  name: string;
  description?: string;
  sensors: FixtureSensor[];
  events: FixtureEvent[];
  min_sessions: number;
  expected: { never_fired: string[]; always_firing: string[] };
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/sensor_chain/signal_quality_basic.json");

class StubSensor implements sensor.Sensor {
  constructor(
    private readonly cfg: sensor.SensorConfig,
    private readonly outcome: sensor.SensorOutcome,
  ) {}
  async evaluate(): Promise<sensor.SensorResult> {
    return {
      sensor_id: new SensorId(this.cfg.id.value),
      outcome: this.outcome,
      observation: this.outcome === "warn" ? "warn-obs" : null,
      detail: this.outcome,
      fired_at: new sensor.Timestamp("2026-05-16T00:00:00Z"),
    };
  }
  config(): sensor.SensorConfig {
    return this.cfg;
  }
}

describe("SensorChain fixture replay", () => {
  it("signal_quality_basic produces expected flags", async () => {
    const raw = readFileSync(fixturePath, "utf8");
    const fix = JSON.parse(raw) as FixtureCase;

    const chain = new StandardSensorChain();
    for (const s of fix.sensors) {
      const cfg: sensor.SensorConfig = {
        id: new SensorId(s.id),
        name: s.id,
        kind: s.kind,
        triggers: s.triggers,
        run_every_n_turns: null,
        run_on_phases: null,
        low_signal_threshold: s.thresholds,
      };
      chain.register(new StubSensor(cfg, s.outcome));
    }
    for (const ev of fix.events) {
      await chain.fire(
        ev.trigger,
        newSensorInput(new SessionId(ev.session_id), emptySessionState()),
      );
    }
    const flags = await chain.signalQualityReport(fix.min_sessions);

    const gotNever = new Set<string>();
    const gotAlways = new Set<string>();
    for (const f of flags) {
      if (f.kind === "never_fired") gotNever.add(f.sensor_id.value);
      else gotAlways.add(f.sensor_id.value);
    }
    expect([...gotNever].sort()).toEqual([...fix.expected.never_fired].sort());
    expect([...gotAlways].sort()).toEqual([...fix.expected.always_firing].sort());
  });
});
