/**
 * MemoryProvider — canonical types (spore-core issue #8).
 *
 * Mirrors `rust/crates/spore-core/src/memory.rs` byte-for-byte on the wire:
 * tagged unions use a `kind` discriminator in `snake_case`, optional
 * collections default to empty arrays. Static types are derived from zod
 * schemas so external boundaries (fixtures, serialized records) round-trip
 * across language targets.
 *
 * Two distinct stores:
 *   - **Episodic** — what happened in a specific session.
 *   - **Semantic** — generalized knowledge distilled from episodes.
 */

import { z } from "zod";

import { SessionId, SessionIdSchema } from "../harness/types.js";

// ============================================================================
// Identity & time
// ============================================================================

export class MemoryId {
  constructor(readonly value: string) {}
  static of(value: string): MemoryId {
    return new MemoryId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: MemoryId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export const MemoryIdSchema = z.string().transform((s) => new MemoryId(s));

/**
 * RFC 3339 / ISO 8601 timestamp. Stored as a string for cross-language
 * fixture portability.
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
// MemorySource (discriminated union)
// ============================================================================

export const MemorySourceSchema = z.discriminatedUnion("kind", [
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
    approved_by: z.string().nullable().optional(),
  }),
]);
export type MemorySource =
  | { kind: "manual" }
  | { kind: "session_generated"; session_id: SessionId }
  | { kind: "trace_distilled"; session_ids: SessionId[] }
  | { kind: "meta_agent_proposed"; approved_by?: string | null };

// ============================================================================
// MemoryStatus (discriminated union)
// ============================================================================

export const MemoryStatusSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("active") }),
  z.object({
    kind: z.literal("deprecated"),
    reason: z.string(),
    at: TimestampSchema,
  }),
  z.object({
    kind: z.literal("pending_review"),
    proposed_at: TimestampSchema,
  }),
]);
export type MemoryStatus =
  | { kind: "active" }
  | { kind: "deprecated"; reason: string; at: Timestamp }
  | { kind: "pending_review"; proposed_at: Timestamp };

// ============================================================================
// Records
// ============================================================================

export const EpisodicMemorySchema = z.object({
  id: MemoryIdSchema,
  session_id: SessionIdSchema,
  content: z.string(),
  created_at: TimestampSchema,
  tags: z.array(z.string()).default([]),
});
export type EpisodicMemory = {
  id: MemoryId;
  session_id: SessionId;
  content: string;
  created_at: Timestamp;
  tags: string[];
};

export const SemanticMemorySchema = z.object({
  id: MemoryIdSchema,
  content: z.string(),
  source: MemorySourceSchema,
  domain: z.string().nullable().optional(),
  version: z.number().int().nonnegative(),
  previous_versions: z.array(MemoryIdSchema).default([]),
  created_at: TimestampSchema,
  updated_at: TimestampSchema,
  status: MemoryStatusSchema,
});
export type SemanticMemory = {
  id: MemoryId;
  content: string;
  source: MemorySource;
  domain?: string | null;
  version: number;
  previous_versions: MemoryId[];
  created_at: Timestamp;
  updated_at: Timestamp;
  status: MemoryStatus;
};

// ============================================================================
// MemoryItem / MemoryQuery
// ============================================================================

export interface MemoryItem {
  memory: SemanticMemory;
  relevance_score: number;
}

export const DEFAULT_MIN_RELEVANCE = 0.5;
export const DEFAULT_MAX_ITEMS = 10;

export interface MemoryQuery {
  task_instruction: string;
  domain?: string | null;
  session_id?: SessionId | null;
  /** Default {@link DEFAULT_MIN_RELEVANCE} (0.5). */
  min_relevance: number;
  /** Default {@link DEFAULT_MAX_ITEMS} (10). */
  max_items: number;
}

export function newMemoryQuery(task_instruction: string): MemoryQuery {
  return {
    task_instruction,
    domain: null,
    session_id: null,
    min_relevance: DEFAULT_MIN_RELEVANCE,
    max_items: DEFAULT_MAX_ITEMS,
  };
}

// ============================================================================
// MergeStrategy
// ============================================================================

export type MergeStrategy = "replace" | "append" | "reject";

export const MergeStrategySchema = z.enum(["replace", "append", "reject"]);

// ============================================================================
// MemoryError
// ============================================================================

export type MemoryErrorKind =
  | { kind: "not_found"; id: MemoryId }
  | { kind: "merge_conflict"; existing: MemoryId; reason: string }
  | { kind: "validation_failed"; reason: string }
  | { kind: "storage_error"; reason: string };

export class MemoryError extends Error {
  override readonly name = "MemoryError";
  readonly kind: MemoryErrorKind["kind"];
  readonly detail: MemoryErrorKind;

  constructor(detail: MemoryErrorKind) {
    super(memoryErrorMessage(detail));
    this.kind = detail.kind;
    this.detail = detail;
  }

  static notFound(id: MemoryId): MemoryError {
    return new MemoryError({ kind: "not_found", id });
  }
  static mergeConflict(existing: MemoryId, reason: string): MemoryError {
    return new MemoryError({ kind: "merge_conflict", existing, reason });
  }
  static validationFailed(reason: string): MemoryError {
    return new MemoryError({ kind: "validation_failed", reason });
  }
  static storageError(reason: string): MemoryError {
    return new MemoryError({ kind: "storage_error", reason });
  }
}

function memoryErrorMessage(e: MemoryErrorKind): string {
  switch (e.kind) {
    case "not_found":
      return `memory not found: ${e.id.asString()}`;
    case "merge_conflict":
      return `merge conflict on ${e.existing.asString()}: ${e.reason}`;
    case "validation_failed":
      return `validation failed: ${e.reason}`;
    case "storage_error":
      return `storage error: ${e.reason}`;
  }
}

// ============================================================================
// MemoryProvider interface
// ============================================================================

export interface MemoryProvider {
  // ── Episodic ───────────────────────────────────────────────────────────
  storeEpisodic(memory: EpisodicMemory): Promise<MemoryId>;
  getEpisodic(session_id: SessionId): Promise<EpisodicMemory[]>;

  // ── Semantic ───────────────────────────────────────────────────────────
  storeSemantic(memory: SemanticMemory, on_conflict: MergeStrategy): Promise<MemoryId>;
  getSemantic(id: MemoryId): Promise<SemanticMemory>;

  /**
   * Primary retrieval. Returns items with relevance >= `min_relevance`,
   * capped at `max_items`, sorted by score descending. Active memories only.
   */
  query(query: MemoryQuery): Promise<MemoryItem[]>;

  // ── Lifecycle ──────────────────────────────────────────────────────────
  deprecate(id: MemoryId, reason: string): Promise<void>;
  getVersionHistory(id: MemoryId): Promise<SemanticMemory[]>;
  markPendingReview(id: MemoryId): Promise<void>;
}
