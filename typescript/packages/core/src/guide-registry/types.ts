/**
 * GuideRegistry — canonical types (spore-core issue #9).
 *
 * Mirrors `rust/crates/spore-core/src/guide_registry.rs` byte-for-byte on the
 * wire: tagged unions use a `kind` discriminator in `snake_case`, struct fields
 * use `snake_case`. Static types are derived from zod where appropriate; the
 * wire field names are preserved on TS shapes for cross-language fixture
 * portability (matching the convention established in `../memory/types.ts`).
 *
 * Guides are feedforward artifacts injected before the agent acts: system
 * prompt fragments, on-demand skills, convention docs, schema annotations, and
 * safety rules.
 */

import { z } from "zod";

import { SessionId, SessionIdSchema } from "../harness/types.js";
import { TaskPhaseSchema, type TaskPhase } from "../tool-registry/types.js";

// ============================================================================
// Identity & time
// ============================================================================

export class GuideId {
  constructor(readonly value: string) {}
  static of(value: string): GuideId {
    return new GuideId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: GuideId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export const GuideIdSchema = z.string().transform((s) => new GuideId(s));

/**
 * RFC 3339 / ISO 8601 timestamp. Stored as a string for cross-language fixture
 * portability. Duplicated locally (not re-imported from memory) to avoid
 * pulling the entire memory namespace.
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
// GuideType
// ============================================================================

export const GuideTypeSchema = z.enum([
  "system_prompt_fragment",
  "skill",
  "convention_doc",
  "schema_annotation",
  "safety_rule",
]);
export type GuideType = z.infer<typeof GuideTypeSchema>;

// ============================================================================
// GuideSource
// ============================================================================

export const GuideSourceSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("manual") }),
  z.object({
    kind: z.literal("session_generated"),
    session_id: SessionIdSchema,
  }),
  z.object({
    kind: z.literal("trace_distilled"),
    session_ids: z.array(SessionIdSchema).default([]),
  }),
  z.object({
    kind: z.literal("meta_agent_proposed"),
    proposed_at: TimestampSchema,
  }),
]);
export type GuideSource =
  | { kind: "manual" }
  | { kind: "session_generated"; session_id: SessionId }
  | { kind: "trace_distilled"; session_ids: SessionId[] }
  | { kind: "meta_agent_proposed"; proposed_at: Timestamp };

// ============================================================================
// PendingReason
// ============================================================================

export const PendingReasonSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("automated_proposal") }),
  z.object({
    kind: z.literal("performance_degradation"),
    failure_rate_delta: z.number(),
  }),
  z.object({
    kind: z.literal("conflict_detected"),
    conflicts_with: z.array(GuideIdSchema).default([]),
  }),
  z.object({
    kind: z.literal("manual_flag"),
    note: z.string(),
  }),
]);
export type PendingReason =
  | { kind: "automated_proposal" }
  | { kind: "performance_degradation"; failure_rate_delta: number }
  | { kind: "conflict_detected"; conflicts_with: GuideId[] }
  | { kind: "manual_flag"; note: string };

// ============================================================================
// GuideStatus
// ============================================================================

export const GuideStatusSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("active") }),
  z.object({
    kind: z.literal("pending_review"),
    reason: PendingReasonSchema,
    since: TimestampSchema,
  }),
  z.object({
    kind: z.literal("deprecated"),
    reason: z.string(),
    at: TimestampSchema,
  }),
  z.object({
    kind: z.literal("stale"),
    last_used: TimestampSchema,
  }),
]);
export type GuideStatus =
  | { kind: "active" }
  | { kind: "pending_review"; reason: PendingReason; since: Timestamp }
  | { kind: "deprecated"; reason: string; at: Timestamp }
  | { kind: "stale"; last_used: Timestamp };

// ============================================================================
// SessionOutcome
// ============================================================================

export const SessionOutcomeSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("success") }),
  z.object({
    kind: z.literal("failure"),
    reason: z.string(),
  }),
  z.object({ kind: z.literal("partial") }),
  z.object({ kind: z.literal("escalated") }),
]);
export type SessionOutcome =
  | { kind: "success" }
  | { kind: "failure"; reason: string }
  | { kind: "partial" }
  /**
   * The session terminated cleanly because a tool escalated a structural
   * signal to the harness's caller (#80, Tool Escalation Protocol). Distinct
   * from `partial` — an escalation is an intentional, clean terminal outcome,
   * not a partial success.
   */
  | { kind: "escalated" };

// ============================================================================
// Guide
// ============================================================================

export const GuideSchema = z.object({
  id: GuideIdSchema,
  name: z.string(),
  content: z.string(),
  guide_type: GuideTypeSchema,
  domain: z.string().nullable().optional(),
  source: GuideSourceSchema,
  status: GuideStatusSchema,
  created_at: TimestampSchema,
  last_used: TimestampSchema.nullable().optional(),
  version: z.number().int().nonnegative(),
});
export type Guide = {
  id: GuideId;
  name: string;
  content: string;
  guide_type: GuideType;
  domain?: string | null;
  source: GuideSource;
  status: GuideStatus;
  created_at: Timestamp;
  last_used?: Timestamp | null;
  version: number;
};

// ============================================================================
// GuideUsageRecord
// ============================================================================

export const GuideUsageRecordSchema = z.object({
  guide_id: GuideIdSchema,
  session_id: SessionIdSchema,
  task_domain: z.string().nullable().optional(),
  outcome: SessionOutcomeSchema,
  recorded_at: TimestampSchema,
});
export type GuideUsageRecord = {
  guide_id: GuideId;
  session_id: SessionId;
  task_domain?: string | null;
  outcome: SessionOutcome;
  recorded_at: Timestamp;
};

// ============================================================================
// GuideQuery
// ============================================================================

export const GuideQuerySchema = z.object({
  task_instruction: z.string(),
  domain: z.string().nullable().optional(),
  phase: TaskPhaseSchema.nullable().optional(),
  guide_types: z.array(GuideTypeSchema).default([]),
});
export type GuideQuery = {
  task_instruction: string;
  domain?: string | null;
  phase?: TaskPhase | null;
  guide_types: GuideType[];
};

export function newGuideQuery(task_instruction: string): GuideQuery {
  return {
    task_instruction,
    domain: null,
    phase: null,
    guide_types: [],
  };
}

// ============================================================================
// GuideConflict
// ============================================================================

export const GuideConflictSchema = z.object({
  guide_a: GuideIdSchema,
  guide_b: GuideIdSchema,
  reason: z.string(),
});
export type GuideConflict = {
  guide_a: GuideId;
  guide_b: GuideId;
  reason: string;
};

// ============================================================================
// ImprovementSignal
// ============================================================================

export type ImprovementSignal =
  | { kind: "skill_generation_needed"; pattern: string; session_ids: SessionId[] }
  | { kind: "guide_deprecation_recommended"; guide_id: GuideId; reason: string }
  | { kind: "conflict_resolution_needed"; conflict: GuideConflict };

// ============================================================================
// GuideRegistryError
// ============================================================================

export type GuideRegistryErrorKind =
  | { kind: "not_found"; id: GuideId }
  | { kind: "conflict_detected"; conflict: GuideConflict }
  | { kind: "validation_failed"; reason: string }
  | { kind: "storage_error"; reason: string };

export class GuideRegistryError extends Error {
  override readonly name = "GuideRegistryError";
  readonly kind: GuideRegistryErrorKind["kind"];
  readonly detail: GuideRegistryErrorKind;

  constructor(detail: GuideRegistryErrorKind) {
    super(guideRegistryErrorMessage(detail));
    this.kind = detail.kind;
    this.detail = detail;
  }

  static notFound(id: GuideId): GuideRegistryError {
    return new GuideRegistryError({ kind: "not_found", id });
  }
  static conflictDetected(conflict: GuideConflict): GuideRegistryError {
    return new GuideRegistryError({ kind: "conflict_detected", conflict });
  }
  static validationFailed(reason: string): GuideRegistryError {
    return new GuideRegistryError({ kind: "validation_failed", reason });
  }
  static storageError(reason: string): GuideRegistryError {
    return new GuideRegistryError({ kind: "storage_error", reason });
  }
}

function guideRegistryErrorMessage(e: GuideRegistryErrorKind): string {
  switch (e.kind) {
    case "not_found":
      return `guide not found: ${e.id.asString()}`;
    case "conflict_detected":
      return `conflict detected: ${e.conflict.guide_a.asString()} vs ${e.conflict.guide_b.asString()}: ${e.conflict.reason}`;
    case "validation_failed":
      return `validation failed: ${e.reason}`;
    case "storage_error":
      return `storage error: ${e.reason}`;
  }
}

// ============================================================================
// GuideRegistry interface
// ============================================================================

export interface GuideRegistry {
  register(guide: Guide): Promise<GuideId>;
  select(query: GuideQuery): Promise<Guide[]>;
  recordUsage(record: GuideUsageRecord): Promise<void>;
  usageHistory(id: GuideId): Promise<GuideUsageRecord[]>;
  deprecate(id: GuideId, reason: string): Promise<void>;
  markPendingReview(id: GuideId, reason: PendingReason): Promise<void>;
  promoteToActive(id: GuideId): Promise<void>;
  /**
   * Analyze usage records within `window_secs` (seconds) of `now`. Emits:
   *   - `guide_deprecation_recommended` when a guide's failure rate exceeds
   *     the no-guide baseline by `min_failure_rate_delta`.
   *   - `skill_generation_needed` when a failure reason occurs at least
   *     `min_pattern_occurrences` times.
   *   - `conflict_resolution_needed` for guides currently pending review with
   *     `conflict_detected`.
   */
  analyzePerformance(
    window_secs: number,
    min_failure_rate_delta: number,
    min_pattern_occurrences: number,
  ): Promise<ImprovementSignal[]>;
  checkConflicts(content: string, domain: string | null | undefined): Promise<GuideConflict[]>;
}
