/**
 * SensorChain — canonical types (spore-core issue #10).
 *
 * Mirrors `rust/crates/spore-core/src/sensor.rs` byte-for-byte on the wire:
 * tagged unions use a `kind` discriminator in `snake_case`; struct fields are
 * `snake_case`. Static types are derived from zod where helpful so the shared
 * fixture `fixtures/sensor_chain/signal_quality_basic.json` deserializes
 * identically in all four language targets.
 *
 * Sensors are post-action observers (linters, type checkers, LLM-as-judge,
 * etc.). The chain fan-outs every sensor matching a trigger and returns every
 * result — it never short-circuits. The harness decides routing for `warn`
 * (inject observation) and `halt` (stop).
 */

import { z } from "zod";

import { SessionId, SessionIdSchema, SessionStateSchema } from "../harness/types.js";
import { TaskPhaseSchema, type TaskPhase } from "../tool-registry/types.js";
import {
  ToolCallSchema,
  ToolResultSchema,
  type ToolCall,
  type ToolResult,
} from "../model/schemas.js";

// ============================================================================
// Identity & time
// ============================================================================

export class SensorId {
  constructor(readonly value: string) {}
  static of(value: string): SensorId {
    return new SensorId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: SensorId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export const SensorIdSchema = z.string().transform((s) => new SensorId(s));

/**
 * RFC 3339 / ISO 8601 timestamp. Stored as a string for cross-language fixture
 * portability. Duplicated locally (not re-imported) for the same reasons the
 * GuideRegistry module duplicates it.
 */
export class Timestamp {
  constructor(readonly value: string) {}
  static of(value: string): Timestamp {
    return new Timestamp(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: Timestamp): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export const TimestampSchema = z.string().transform((s) => new Timestamp(s));

// ============================================================================
// SensorTrigger
// ============================================================================

export const SensorTriggerSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("post_tool"), tool_name: z.string() }),
  z.object({ kind: z.literal("post_turn") }),
  z.object({ kind: z.literal("post_session") }),
  z.object({ kind: z.literal("continuous") }),
  z.object({ kind: z.literal("on_tool_error") }),
  z.object({ kind: z.literal("on_compaction") }),
]);
export type SensorTrigger =
  | { kind: "post_tool"; tool_name: string }
  | { kind: "post_turn" }
  | { kind: "post_session" }
  | { kind: "continuous" }
  | { kind: "on_tool_error" }
  | { kind: "on_compaction" };

/**
 * Match a configured trigger against the trigger that actually fired.
 * `post_tool` with empty `tool_name` is a wildcard; non-empty matches exact.
 */
export function triggerMatches(configured: SensorTrigger, fired: SensorTrigger): boolean {
  if (configured.kind === "post_tool" && fired.kind === "post_tool") {
    return configured.tool_name.length === 0 || configured.tool_name === fired.tool_name;
  }
  return configured.kind === fired.kind;
}

// ============================================================================
// SensorKind, SensorOutcome
// ============================================================================

export const SensorKindSchema = z.enum(["computational", "inferential"]);
export type SensorKind = z.infer<typeof SensorKindSchema>;

export const SensorOutcomeSchema = z.enum(["pass", "warn", "halt"]);
export type SensorOutcome = z.infer<typeof SensorOutcomeSchema>;

// ============================================================================
// SensorInput, SensorResult
// ============================================================================

export const SensorInputSchema = z.object({
  session_id: SessionIdSchema,
  turn_number: z.number().int().nonnegative().nullable().optional(),
  phase: TaskPhaseSchema.nullable().optional(),
  tool_call: ToolCallSchema.nullable().optional(),
  tool_result: ToolResultSchema.nullable().optional(),
  agent_response: z.string().nullable().optional(),
  session_state: SessionStateSchema,
});
export type SensorInput = {
  session_id: SessionId;
  turn_number?: number | null;
  phase?: TaskPhase | null;
  tool_call?: ToolCall | null;
  tool_result?: ToolResult | null;
  agent_response?: string | null;
  session_state: import("../harness/types.js").SessionState;
};

export function newSensorInput(
  session_id: SessionId,
  session_state: import("../harness/types.js").SessionState,
): SensorInput {
  return {
    session_id,
    turn_number: null,
    phase: null,
    tool_call: null,
    tool_result: null,
    agent_response: null,
    session_state,
  };
}

export const SensorResultSchema = z.object({
  sensor_id: SensorIdSchema,
  outcome: SensorOutcomeSchema,
  observation: z.string().nullable().optional(),
  detail: z.string(),
  fired_at: TimestampSchema,
});
export type SensorResult = {
  sensor_id: SensorId;
  outcome: SensorOutcome;
  observation?: string | null;
  detail: string;
  fired_at: Timestamp;
};

// ============================================================================
// SensorSignalThresholds, SensorConfig, SensorStats, SensorSignalFlag
// ============================================================================

export const SensorSignalThresholdsSchema = z.object({
  never_fired_after_n_sessions: z.number().int().nonnegative(),
  always_fired_rate: z.number(),
});
export type SensorSignalThresholds = z.infer<typeof SensorSignalThresholdsSchema>;

export function defaultSensorSignalThresholds(): SensorSignalThresholds {
  return { never_fired_after_n_sessions: 10, always_fired_rate: 0.9 };
}

export const SensorConfigSchema = z.object({
  id: SensorIdSchema,
  name: z.string(),
  kind: SensorKindSchema,
  triggers: z.array(SensorTriggerSchema),
  run_every_n_turns: z.number().int().nonnegative().nullable().optional(),
  run_on_phases: z.array(TaskPhaseSchema).nullable().optional(),
  low_signal_threshold: SensorSignalThresholdsSchema.default(defaultSensorSignalThresholds()),
});
export type SensorConfig = {
  id: SensorId;
  name: string;
  kind: SensorKind;
  triggers: SensorTrigger[];
  run_every_n_turns?: number | null;
  run_on_phases?: TaskPhase[] | null;
  low_signal_threshold: SensorSignalThresholds;
};

export const SensorStatsSchema = z.object({
  sensor_id: SensorIdSchema,
  total_fires: z.number().int().nonnegative(),
  warn_count: z.number().int().nonnegative(),
  halt_count: z.number().int().nonnegative(),
  pass_count: z.number().int().nonnegative(),
  fire_rate: z.number(),
  last_fired: TimestampSchema.nullable().optional(),
  low_signal_flag: z.boolean(),
});
export type SensorStats = {
  sensor_id: SensorId;
  total_fires: number;
  warn_count: number;
  halt_count: number;
  pass_count: number;
  fire_rate: number;
  last_fired?: Timestamp | null;
  low_signal_flag: boolean;
};

export type SensorSignalFlag =
  | { kind: "never_fired"; sensor_id: SensorId; sessions_observed: number }
  | { kind: "always_firing"; sensor_id: SensorId; fire_rate: number };

// ============================================================================
// SensorError
// ============================================================================

export type SensorErrorKind =
  | { kind: "already_registered"; sensor_id: SensorId }
  | { kind: "validation_failed"; reason: string };

export class SensorError extends Error {
  override readonly name = "SensorError";
  readonly kind: SensorErrorKind["kind"];
  readonly detail: SensorErrorKind;

  constructor(detail: SensorErrorKind) {
    super(sensorErrorMessage(detail));
    this.kind = detail.kind;
    this.detail = detail;
  }

  static alreadyRegistered(sensorId: SensorId): SensorError {
    return new SensorError({ kind: "already_registered", sensor_id: sensorId });
  }

  static validationFailed(reason: string): SensorError {
    return new SensorError({ kind: "validation_failed", reason });
  }
}

function sensorErrorMessage(e: SensorErrorKind): string {
  switch (e.kind) {
    case "already_registered":
      return `sensor already registered: ${e.sensor_id.asString()}`;
    case "validation_failed":
      return `validation failed: ${e.reason}`;
  }
}

// ============================================================================
// Sensor, SensorChain interfaces
// ============================================================================

export interface Sensor {
  evaluate(input: SensorInput, signal?: AbortSignal): Promise<SensorResult>;
  config(): SensorConfig;
}

export interface SensorChain {
  register(sensor: Sensor): void;
  fire(trigger: SensorTrigger, input: SensorInput, signal?: AbortSignal): Promise<SensorResult[]>;
  stats(since?: Timestamp | null): Promise<SensorStats[]>;
  signalQualityReport(minSessions: number): Promise<SensorSignalFlag[]>;
}
