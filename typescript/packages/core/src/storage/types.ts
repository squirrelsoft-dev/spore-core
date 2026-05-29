/**
 * StorageProvider — a pluggable, per-domain persistence layer (spore-core
 * issue #73).
 *
 * Mirrors `rust/crates/spore-core/src/storage.rs` on the wire: the four domain
 * stores, the `MemoryEntry` shape, and the OTLP endpoint parse rule are the
 * shared cross-language contract (same fixture, same outcome — see
 * `/fixtures/storage/`). This is NOT a transliteration of the Rust reference;
 * it is idiomatic TypeScript (interfaces + classes, `Promise`-based async).
 *
 * ## Domain stores (all async)
 *   - {@link SessionStore}  — pause/resume lifecycle, keyed by `SessionId`.
 *   - {@link MemoryStore}   — episodic memory; `getMemories(limit)` returns the
 *     MOST-RECENT `limit` entries, NEWEST-FIRST.
 *   - {@link RunStore}      — opaque per-run JSON blobs keyed by
 *     `(SessionId, key)`; the store never knows the schema.
 *   - {@link ObservabilityStore} — append-only span storage.
 *
 * ## Rules enforced
 *   - **No-op fallback.** Unconfigured domains fall back to
 *     {@link NoOpStorageProvider}; the harness never null-checks — it always
 *     calls the store and the store decides. No-op reads return
 *     `undefined`/`[]`; writes resolve.
 *   - **Single-provider-fills-all-slots.** {@link StorageProvider.single} places
 *     one provider implementing all four interfaces into all four slots.
 *   - **Composite per-domain routing.** {@link CompositeStorageProvider} holds an
 *     optional provider per domain; `.build()` fills each unset slot with a
 *     {@link NoOpStorageProvider}.
 *   - **Atomic write-rename.** {@link FileSystemStorageProvider} non-append
 *     writes write full bytes to a sibling `{target}.tmp`, fsync, then rename to
 *     the target. No leftover `.tmp` on success. Append writes (memory /
 *     observability JSONL) append + flush. `flushSession` creates a sibling
 *     `.flushed` marker.
 *   - **`getMemories` recency.** Returns the most-recent `limit` entries,
 *     newest-first.
 */

import type { SessionId } from "../harness/types.js";
import type { PausedState } from "../harness/types.js";
import type { Timestamp } from "../memory/types.js";
import type { SessionMetrics } from "../observability/types.js";
import type { SessionOutcome } from "../guide-registry/types.js";

// ============================================================================
// JsonValue
// ============================================================================

/**
 * An opaque JSON value — the cross-language equivalent of Rust's
 * `serde_json::Value` / Python's `Any` / Go's `json.RawMessage`. The
 * {@link RunStore} and {@link ObservabilityStore} treat their payloads as
 * opaque: callers own (de)serialization and the store never inspects the shape.
 */
export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

// ============================================================================
// MemoryEntry
// ============================================================================

/**
 * One episodic memory entry. Byte-identical cross-language: `{ role, content,
 * timestamp, metadata }` where `metadata` defaults to an empty object `{}`.
 * `timestamp` is serialized as the RFC-3339 string (the {@link Timestamp} class
 * serializes to its string value via `toJSON`).
 */
export interface MemoryEntry {
  role: string;
  content: string;
  timestamp: Timestamp;
  /** Free-form metadata; defaults to an empty JSON object `{}`. */
  metadata: JsonValue;
}

/** Build a {@link MemoryEntry} with an empty `metadata` object. */
export function newMemoryEntry(role: string, content: string, timestamp: Timestamp): MemoryEntry {
  return { role, content, timestamp, metadata: {} };
}

// ============================================================================
// Domain store interfaces
// ============================================================================

/** Pause/resume lifecycle store. Stores {@link PausedState} keyed by
 *  {@link SessionId}. */
export interface SessionStore {
  getSession(id: SessionId): Promise<PausedState | undefined>;
  putSession(id: SessionId, state: PausedState): Promise<void>;
  deleteSession(id: SessionId): Promise<void>;
  listSessions(): Promise<SessionId[]>;
}

/** Episodic memory store. Append-only log per session. */
export interface MemoryStore {
  appendMemory(sessionId: SessionId, entry: MemoryEntry): Promise<void>;
  /** Returns the MOST-RECENT `limit` entries, NEWEST-FIRST. */
  getMemories(sessionId: SessionId, limit: number): Promise<MemoryEntry[]>;
}

/** Per-run structured state keyed by `(SessionId, key)`. Values are opaque JSON
 *  blobs — the store does not know the schema; callers own serialization. */
export interface RunStore {
  get(sessionId: SessionId, key: string): Promise<JsonValue | undefined>;
  put(sessionId: SessionId, key: string, value: JsonValue): Promise<void>;
  delete(sessionId: SessionId, key: string): Promise<void>;
  listKeys(sessionId: SessionId): Promise<string[]>;
}

/** Append-only span storage. Distinct from the other three: no get-by-key,
 *  queried by session and time range. */
export interface ObservabilityStore {
  appendSpan(sessionId: SessionId, span: JsonValue): Promise<void>;
  getSpans(sessionId: SessionId): Promise<JsonValue[]>;
  getSessions(
    since: Timestamp,
    domain?: string,
    outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]>;
  flushSession(sessionId: SessionId): Promise<void>;
}

/** A provider that implements all four domain-store interfaces. Passing one to
 *  {@link StorageProvider.single} wires every domain to the same backend. */
export type FullStorageProvider = SessionStore & MemoryStore & RunStore & ObservabilityStore;
