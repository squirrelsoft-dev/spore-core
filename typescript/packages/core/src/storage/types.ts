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

import { createHash } from "node:crypto";

import type { SessionId } from "../harness/types.js";
import type { PausedState } from "../harness/types.js";
import type { Timestamp } from "../memory/types.js";
import type { SessionMetrics } from "../observability/types.js";
import type { SessionOutcome } from "../guide-registry/types.js";

/**
 * Re-export of the canonical {@link StorageScope} (its home is the
 * `prompt-assembly` module, decision A2). The storage module does NOT redefine
 * it — `storage.StorageScope` resolves to the same `"user" | "project" |
 * "local"` union, byte-identical on the wire (#78).
 */
export type { StorageScope } from "../prompt-assembly/types.js";
export { STORAGE_SCOPES } from "../prompt-assembly/types.js";

import type { StorageScope } from "../prompt-assembly/types.js";

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
// WorkspaceId (#78)
// ============================================================================

/**
 * A stable identifier for a workspace, derived purely from its canonical path
 * (#78). Form: `{sanitizedBasename}-{8hex}`, lowercased.
 *
 * This is the cross-language parity anchor — {@link WorkspaceId.fromCanonicalPath}
 * is a **pure string function** (it never touches the filesystem) so the pinned
 * fixture `fixtures/storage/workspace_id_derivation.json` is host-independent.
 *
 * Used at wiring time to partition the user-scope storage root:
 * `{userRoot}/projects/{workspaceId}`. Backends never see it.
 *
 * A branded `string` wrapper (class) so a raw `string` can never be passed where
 * a derived id is required.
 */
export class WorkspaceId {
  private constructor(private readonly id: string) {}

  /**
   * Derive a {@link WorkspaceId} from an already-OS-canonicalized path.
   *
   * Algorithm (pinned, byte-identical across languages):
   * 1. Normalize separators to `/`. On Windows strip the drive-letter prefix
   *    (e.g. `C:`) and convert `\` → `/`. The input is assumed already
   *    OS-canonicalized; this does NOT re-canonicalize or touch the filesystem.
   * 2. Build the canonical path string: forward slashes only, NO trailing
   *    slash, UTF-8.
   * 3. SHA-256 that string; take the first 8 hex chars (lowercase).
   * 4. Basename of the canonical path, lowercased; replace each non-alphanumeric
   *    char with `-`; collapse consecutive `-`; strip leading/trailing `-`. Empty
   *    basename (root `/`) → `root`.
   * 5. Concatenate `{sanitizedBasename}-{8hex}`.
   */
  static fromCanonicalPath(path: string): WorkspaceId {
    const canonical = canonicalizePathString(path);

    const digest = sha256Hex(canonical);
    // First 8 hex chars = first 4 bytes.
    const hex8 = digest.slice(0, 8);

    const slash = canonical.lastIndexOf("/");
    const basename = slash >= 0 ? canonical.slice(slash + 1) : canonical;
    let sanitized = sanitizeBasename(basename);
    if (sanitized.length === 0) sanitized = "root";

    return new WorkspaceId(`${sanitized}-${hex8}`);
  }

  /** The underlying derived id string. */
  asString(): string {
    return this.id;
  }

  toString(): string {
    return this.id;
  }

  toJSON(): string {
    return this.id;
  }

  equals(other: WorkspaceId): boolean {
    return this.id === other.id;
  }
}

/**
 * Steps 1–2 of the derivation: produce the canonical path string used for both
 * the hash input and the basename. Forward slashes only, no trailing slash.
 */
function canonicalizePathString(path: string): string {
  // Normalize Windows backslashes.
  let s = path.replace(/\\/g, "/");
  // Strip a leading drive-letter prefix like `C:` (only at the very start).
  if (s.length >= 2 && s[1] === ":" && /[A-Za-z]/.test(s[0]!)) {
    s = s.slice(2);
  }
  // Strip a trailing slash, but keep a lone root `/`.
  while (s.length > 1 && s.endsWith("/")) {
    s = s.slice(0, -1);
  }
  return s;
}

/**
 * Step 4 of the derivation: lowercase, replace each non-alphanumeric char with
 * `-`, collapse consecutive `-`, strip leading/trailing `-`. Only ASCII
 * alphanumerics are preserved, matching the Rust reference.
 */
function sanitizeBasename(basename: string): string {
  const lowered = basename.toLowerCase();
  let out = "";
  let prevDash = false;
  for (const ch of lowered) {
    if (/[a-z0-9]/.test(ch)) {
      out += ch;
      prevDash = false;
    } else if (!prevDash) {
      out += "-";
      prevDash = true;
    }
  }
  // Strip leading/trailing `-`.
  return out.replace(/^-+/, "").replace(/-+$/, "");
}

function sha256Hex(input: string): string {
  return createHash("sha256").update(input, "utf8").digest("hex");
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

/**
 * Episodic memory store. Append-only log per `(scope, session)` (#78).
 *
 * A leaf backend is **scope-dumb**: it stores under whatever root it was given.
 * The `scope` argument is carried for symmetry — the v1 wiring routes each scope
 * to its own backend via {@link CompositeStorageProvider}, so a leaf backend
 * receives a single scope's traffic. The cross-scope merge
 * ({@link StorageProvider.getMemoriesMerged}) lives in the routing layer, never
 * in a leaf.
 *
 * Known v1 limitation: memory addressing stays {@link SessionId}-keyed. v2
 * should address session-independent / cross-session keying — do not introduce
 * it here.
 */
export interface MemoryStore {
  appendMemory(scope: StorageScope, sessionId: SessionId, entry: MemoryEntry): Promise<void>;
  /** Returns the MOST-RECENT `limit` entries, NEWEST-FIRST, for `scope`. */
  getMemories(scope: StorageScope, sessionId: SessionId, limit: number): Promise<MemoryEntry[]>;
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
