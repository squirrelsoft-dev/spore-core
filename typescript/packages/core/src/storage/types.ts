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
import { realpathSync } from "node:fs";

import { z } from "zod";

import { SessionId } from "../harness/types.js";
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
// ProjectId (#142) — project-scoped durable storage namespace
// ============================================================================

/**
 * A **stable** identifier for a project, used as the durable storage namespace
 * (issue #142). Where {@link SessionId} is regenerated per Ralph context window
 * (`SessionId.generate()`), a `ProjectId` derived from the workspace root stays
 * constant across windows AND across process restarts — that stability is the
 * whole point: the `task_list`, plan artifact, Ralph checkpoint, and active-run
 * slot persist under it, so a window reset re-reads the prior window's work
 * instead of re-planning from scratch.
 *
 * Form: `{sanitizedBasename}-{8hex}`, lowercased — **identical** to
 * {@link WorkspaceId}. The two share the pure derivation
 * ({@link canonicalizePathString} + {@link sanitizeBasename}); a `ProjectId`
 * differs only in carrying FS-touching constructors that canonicalize first.
 *
 * ## `/a/b` vs `/a_b` collision policy (RESOLVED)
 * A naive "slashes → underscores" slug would map both `/a/b` and `/a_b` to the
 * same string. This derivation does NOT collide: it slugs ONLY the final
 * basename and appends the first 8 hex of the SHA-256 of the FULL canonical path
 * string. `/a/b` and `/a_b` have different canonical strings, hence different
 * hashes, hence distinct ids (`b-<h1>` vs `a-b-<h2>`). The fixture
 * `fixtures/storage/project_id_derivation.json` pins this distinct-id case.
 *
 * ## Namespace reuse
 * The {@link RunStore} interface is keyed by {@link SessionId}. Rather than
 * widening that interface, a `ProjectId` is projected onto the same string axis
 * via {@link ProjectId.namespace}, which yields a {@link SessionId}-typed key
 * whose string is the derived project id. Durable call sites pass that key in
 * place of the per-window session id, so the interface stays stable while the
 * value keyed is the stable project namespace. Ephemeral session/conversation
 * state keeps using the real per-window {@link SessionId}.
 *
 * A branded `string` wrapper (class) so a raw `string` can never be passed where
 * a derived project id is required.
 */
export class ProjectId {
  private constructor(private readonly id: string) {}

  /**
   * Derive a {@link ProjectId} from an already-OS-canonicalized path. **PURE and
   * infallible** — it never touches the filesystem. This is the cross-language
   * fixture anchor (`fixtures/storage/project_id_derivation.json`); it reuses the
   * EXACT same algorithm as {@link WorkspaceId.fromCanonicalPath}
   * ({@link canonicalizePathString} + {@link sanitizeBasename} + 8-hex SHA-256),
   * so the two derivations are byte-identical for the same input.
   */
  static fromCanonicalPath(path: string): ProjectId {
    // Reuse the WorkspaceId pure algorithm — do NOT duplicate it.
    return new ProjectId(WorkspaceId.fromCanonicalPath(path).asString());
  }

  /**
   * Derive a {@link ProjectId} from a path, **canonicalizing the filesystem
   * FIRST** (resolving symlinks, relative components, and macOS
   * case-insensitivity via {@link realpathSync}) before delegating to the pure
   * {@link ProjectId.fromCanonicalPath}. Throws {@link ProjectIdError} (kind
   * `canonicalize`) if the path cannot be canonicalized (does not exist, a
   * component is not a directory, a permission error, a broken symlink, …).
   */
  static fromPath(path: string): ProjectId {
    let canonical: string;
    try {
      canonical = realpathSync(path);
    } catch (err) {
      throw new ProjectIdError(path, err);
    }
    return ProjectId.fromCanonicalPath(canonical);
  }

  /**
   * Derive a {@link ProjectId} from the current working directory,
   * canonicalizing FIRST. Convenience wrapper over {@link ProjectId.fromPath} for
   * binaries that want the process cwd; the harness itself derives from the
   * sandbox workspace root, NOT process cwd (decision 5).
   */
  static fromCwd(): ProjectId {
    return ProjectId.fromPath(process.cwd());
  }

  /**
   * Project this `ProjectId` onto the {@link RunStore}'s {@link SessionId} string
   * axis (the namespace-reuse seam, #142). The returned {@link SessionId} is NOT
   * a real session — its string IS the derived project id — so durable
   * {@link RunStore.get} / {@link RunStore.put} calls key by the stable project
   * namespace without widening the interface. Ephemeral session-keyed state keeps
   * using the real per-window {@link SessionId}.
   */
  namespace(): SessionId {
    return SessionId.of(this.id);
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

  equals(other: ProjectId): boolean {
    return this.id === other.id;
  }
}

/**
 * Error surfaced while deriving a {@link ProjectId} from the live filesystem
 * (issue #142). The pure derivation {@link ProjectId.fromCanonicalPath} is
 * infallible — only the FS-touching constructors ({@link ProjectId.fromPath} /
 * {@link ProjectId.fromCwd}) can fail, and only because canonicalization touched
 * the filesystem. Follows the conventions error pattern (a `name`/`kind`).
 */
export class ProjectIdError extends Error {
  override readonly name = "ProjectIdError";
  readonly kind = "canonicalize" as const;
  constructor(
    readonly path: string,
    cause?: unknown,
  ) {
    super(`project id canonicalization failed for ${path}`, { cause });
  }
}

// ============================================================================
// Active-run lifecycle (#142, decision 2)
// ============================================================================

/**
 * Reserved {@link RunStore} key under the project namespace holding the
 * {@link ActiveRun} slot. The caller owns the lifecycle: start-new vs resume is
 * a deterministic match on the caller-supplied `runTag`, NOT instruction-diffing
 * and NOT auto-on-success. The harness stays stateless between runs.
 */
export const ACTIVE_RUN_KEY = "active_run";

/**
 * Reserved {@link RunStore} key under the project namespace holding the Ralph
 * progress checkpoint (issue #142, decision 3). The checkpoint content
 * previously lived at `{workspaceRoot}/.spore/progress.json` — it now lives in
 * the project-id store so it survives Ralph window resets and process restarts.
 */
export const RALPH_PROGRESS_KEY = "ralph_progress";

/**
 * Reserved {@link RunStore} key under the project namespace holding the Ralph
 * feature-list checkpoint (issue #142, decision 3). Mirrors the old
 * `{workspaceRoot}/.spore/feature_list.json`.
 */
export const RALPH_FEATURE_LIST_KEY = "ralph_feature_list";

/** Lifecycle status of the project's active run. */
export type ActiveRunStatus = "active" | "completed";

/**
 * The active-run slot persisted under {@link ACTIVE_RUN_KEY} in the project
 * store.
 *
 * `runTag` is **caller-supplied** and is the sole start-new-vs-resume
 * discriminator (decision 2): a {@link startOrResumeActiveRun} call whose tag
 * matches a live slot RESUMES; a different tag (or an absent / completed slot)
 * starts FRESH. `startedAt` is an **injected** timestamp (decision 2 — no
 * `Date.now()`-style nondeterminism) so tests are deterministic. Serialized
 * snake_case on the wire for cross-language byte parity.
 */
export interface ActiveRun {
  run_tag: string;
  started_at: Timestamp;
  status: ActiveRunStatus;
}

/** Zod schema for an {@link ActiveRun} slot read back from a {@link RunStore}. */
export const ActiveRunSchema = z.object({
  run_tag: z.string(),
  started_at: z.string(),
  status: z.enum(["active", "completed"]),
});

/**
 * Outcome of {@link startOrResumeActiveRun}: did the slot match (resume) or did
 * the call mint a fresh active slot (start-new)?
 */
export type ActiveRunDecision = "started_new" | "resumed";

/** Read the active-run slot for `project`, or `undefined` if absent /
 *  unparseable. A malformed slot is treated as "no live run". */
export async function loadActiveRun(
  runStore: RunStore,
  project: ProjectId,
): Promise<{ run_tag: string; started_at: string; status: ActiveRunStatus } | undefined> {
  const value = await runStore.get(project.namespace(), ACTIVE_RUN_KEY);
  if (value == null) return undefined;
  const parsed = ActiveRunSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

/**
 * Decide start-new vs resume for `project` on the caller-supplied `runTag`
 * (decision 2). Deterministic: a live (`active`) slot under the SAME tag ⇒
 * `"resumed"` (the slot is left intact); otherwise a fresh `active` slot stamped
 * with the injected `startedAt` is written and `"started_new"` is returned.
 * `startedAt` is injected (not read from a clock) so the result is deterministic
 * in tests.
 */
export async function startOrResumeActiveRun(
  runStore: RunStore,
  project: ProjectId,
  runTag: string,
  startedAt: Timestamp,
): Promise<ActiveRunDecision> {
  const existing = await loadActiveRun(runStore, project);
  if (existing != null && existing.status === "active" && existing.run_tag === runTag) {
    return "resumed";
  }
  const fresh: JsonValue = {
    run_tag: runTag,
    started_at: startedAt.asString(),
    status: "active",
  };
  await runStore.put(project.namespace(), ACTIVE_RUN_KEY, fresh);
  return "started_new";
}

/**
 * Mark the active run for `project` complete (decision 2): flips the slot's
 * status to `"completed"` so the next {@link startOrResumeActiveRun} (even under
 * the same tag) starts fresh. A no-op when there is no slot to complete.
 */
export async function completeActiveRun(runStore: RunStore, project: ProjectId): Promise<void> {
  const existing = await loadActiveRun(runStore, project);
  if (existing == null) return;
  const completed: JsonValue = {
    run_tag: existing.run_tag,
    started_at: existing.started_at,
    status: "completed",
  };
  await runStore.put(project.namespace(), ACTIVE_RUN_KEY, completed);
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
  /**
   * Cross-scope merged read (#78 R6 / #82 D2): **User ∪ Project, newest-first by
   * `timestamp`, NO dedup**. `Local` is excluded from the merge in v1.
   *
   * This is the SINGLE source of the merge algorithm. TypeScript interfaces
   * cannot carry default method bodies, so every {@link MemoryStore} implementation
   * delegates to the one shared helper {@link getMemoriesMergedDefault} — there
   * is exactly one merge implementation. {@link StorageProvider.getMemoriesMerged}
   * and {@link "@spore/tools".MemoryTool}'s merged `read` both reach this method.
   */
  getMemoriesMerged(sessionId: SessionId, limit: number): Promise<MemoryEntry[]>;
}

/**
 * The `timestamp` field of a {@link MemoryEntry} as a comparable string. Entries
 * from the in-memory backend carry a {@link Timestamp} instance; entries read
 * back from a JSONL backend carry a plain string. Both are handled so the merge
 * works regardless of backend (#78 R6).
 */
function memoryTimestampKey(entry: MemoryEntry): string {
  const t = entry.timestamp as unknown;
  if (typeof t === "string") return t;
  if (t != null && typeof (t as { asString?: () => string }).asString === "function") {
    return (t as { asString: () => string }).asString();
  }
  return String(t);
}

/**
 * The SINGLE cross-scope merge implementation (#78 R6 / #82 D2). Reads
 * `User` and `Project` from `store` (NEVER `Local`), concatenates them
 * (**no dedup**), sorts newest-first by `timestamp`, and truncates to `limit`.
 * A *stable* sort preserves input order among equal timestamps, keeping the
 * merge deterministic cross-language (`Array.prototype.sort` is stable in
 * modern V8). Every {@link MemoryStore.getMemoriesMerged} delegates here.
 */
export async function getMemoriesMergedDefault(
  store: MemoryStore,
  sessionId: SessionId,
  limit: number,
): Promise<MemoryEntry[]> {
  const user = await store.getMemories("user", sessionId, limit);
  const project = await store.getMemories("project", sessionId, limit);
  const combined = [...user, ...project];
  const sorted = combined.slice().sort((a, b) => {
    const ka = memoryTimestampKey(a);
    const kb = memoryTimestampKey(b);
    // Newest-first: descending by timestamp string.
    return ka < kb ? 1 : ka > kb ? -1 : 0;
  });
  return limit < 0 ? [] : sorted.slice(0, limit);
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
