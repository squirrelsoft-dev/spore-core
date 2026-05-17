/**
 * StandardSensorChain — in-memory reference implementation of
 * {@link SensorChain} (spore-core issue #10).
 *
 * Mirrors `rust/crates/spore-core/src/sensor.rs#StandardSensorChain`. Sensors
 * are registered with a {@link SensorConfig}; `fire` evaluates every sensor
 * whose triggers match (no short-circuit), gating inferential sensors by
 * `run_every_n_turns` and `run_on_phases`. `signalQualityReport` flags
 * `NeverFired` and `AlwaysFiring` sensors using the per-sensor thresholds.
 */

import {
  type Sensor,
  type SensorChain,
  type SensorConfig,
  SensorError,
  SensorId,
  type SensorInput,
  type SensorKind,
  type SensorOutcome,
  type SensorResult,
  type SensorSignalFlag,
  type SensorStats,
  type SensorTrigger,
  Timestamp,
  triggerMatches,
} from "./types.js";

interface SensorEntry {
  config: SensorConfig;
  sensor: Sensor;
}

interface HistoryRecord {
  sensor_id: SensorId;
  session_id: string;
  outcome: SensorOutcome;
  fired_at: Timestamp;
}

function inferentialGateOpen(cfg: SensorConfig, input: SensorInput): boolean {
  if (cfg.kind === ("computational" satisfies SensorKind)) return true;

  const allowed = cfg.run_on_phases ?? null;
  if (allowed && allowed.length > 0) {
    const phase = input.phase ?? null;
    if (!phase || !allowed.includes(phase)) {
      return false;
    }
  }

  const n = cfg.run_every_n_turns ?? null;
  if (n !== null && n !== undefined) {
    if (n === 0) return false;
    const t = input.turn_number ?? null;
    if (t !== null && t !== undefined) {
      if (t % n !== 0) return false;
    }
  }
  return true;
}

export class StandardSensorChain implements SensorChain {
  private readonly sensors: SensorEntry[] = [];
  private readonly history: HistoryRecord[] = [];
  private readonly sessionsSeen = new Set<string>();

  register(sensor: Sensor): void {
    const cfg = sensor.config();
    if (cfg.triggers.length === 0) {
      throw SensorError.validationFailed("sensor must declare at least one trigger");
    }
    if (this.sensors.some((e) => e.config.id.value === cfg.id.value)) {
      throw SensorError.alreadyRegistered(new SensorId(cfg.id.value));
    }
    this.sensors.push({ config: cfg, sensor });
  }

  async fire(
    trigger: SensorTrigger,
    input: SensorInput,
    signal?: AbortSignal,
  ): Promise<SensorResult[]> {
    this.sessionsSeen.add(input.session_id.value);

    const candidates = this.sensors.filter(
      (e) =>
        e.config.triggers.some((t) => triggerMatches(t, trigger)) &&
        inferentialGateOpen(e.config, input),
    );

    const results: SensorResult[] = [];
    for (const entry of candidates) {
      const result = await entry.sensor.evaluate(input, signal);
      this.history.push({
        sensor_id: new SensorId(entry.config.id.value),
        session_id: input.session_id.value,
        outcome: result.outcome,
        fired_at: result.fired_at,
      });
      results.push(result);
    }
    return results;
  }

  async stats(since?: Timestamp | null): Promise<SensorStats[]> {
    interface Agg {
      total: number;
      pass: number;
      warn: number;
      halt: number;
      last: Timestamp | null;
    }
    const bySensor = new Map<string, Agg>();
    for (const entry of this.sensors) {
      bySensor.set(entry.config.id.value, { total: 0, pass: 0, warn: 0, halt: 0, last: null });
    }

    const cutoff = since ?? null;
    for (const rec of this.history) {
      if (cutoff && rec.fired_at.value < cutoff.value) continue;
      const agg = bySensor.get(rec.sensor_id.value) ?? {
        total: 0,
        pass: 0,
        warn: 0,
        halt: 0,
        last: null,
      };
      agg.total += 1;
      if (rec.outcome === "pass") agg.pass += 1;
      else if (rec.outcome === "warn") agg.warn += 1;
      else agg.halt += 1;
      agg.last = rec.fired_at;
      bySensor.set(rec.sensor_id.value, agg);
    }

    const sessionsTotal = this.sessionsSeen.size;
    const out: SensorStats[] = [];
    for (const [id, agg] of bySensor) {
      const cfg = this.sensors.find((e) => e.config.id.value === id)?.config;
      const fireRate =
        sessionsTotal === 0 ? 0 : Math.max(0, Math.min(1, agg.total / sessionsTotal));
      let lowSignal = false;
      if (cfg) {
        lowSignal =
          fireRate > cfg.low_signal_threshold.always_fired_rate ||
          (agg.total === 0 &&
            sessionsTotal >= cfg.low_signal_threshold.never_fired_after_n_sessions);
      }
      out.push({
        sensor_id: new SensorId(id),
        total_fires: agg.total,
        warn_count: agg.warn,
        halt_count: agg.halt,
        pass_count: agg.pass,
        fire_rate: fireRate,
        last_fired: agg.last,
        low_signal_flag: lowSignal,
      });
    }
    out.sort((a, b) => a.sensor_id.value.localeCompare(b.sensor_id.value));
    return out;
  }

  async signalQualityReport(minSessions: number): Promise<SensorSignalFlag[]> {
    const sessionsObserved = this.sessionsSeen.size;
    const out: SensorSignalFlag[] = [];
    if (sessionsObserved < minSessions) return out;

    for (const entry of this.sensors) {
      const fires = this.history.filter((r) => r.sensor_id.value === entry.config.id.value);
      const total = fires.length;
      const fireRate =
        sessionsObserved === 0 ? 0 : Math.max(0, Math.min(1, total / sessionsObserved));

      if (
        total === 0 &&
        sessionsObserved >= entry.config.low_signal_threshold.never_fired_after_n_sessions
      ) {
        out.push({
          kind: "never_fired",
          sensor_id: new SensorId(entry.config.id.value),
          sessions_observed: sessionsObserved,
        });
      } else if (fireRate > entry.config.low_signal_threshold.always_fired_rate) {
        out.push({
          kind: "always_firing",
          sensor_id: new SensorId(entry.config.id.value),
          fire_rate: fireRate,
        });
      }
    }
    out.sort((a, b) => {
      const ka = a.kind === "never_fired" ? "a" : "b";
      const kb = b.kind === "never_fired" ? "a" : "b";
      if (ka !== kb) return ka < kb ? -1 : 1;
      return a.sensor_id.value.localeCompare(b.sensor_id.value);
    });
    return out;
  }
}
