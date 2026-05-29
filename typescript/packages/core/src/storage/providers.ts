/**
 * StorageProvider and the v1 concrete providers (spore-core issue #73):
 * {@link NoOpStorageProvider}, {@link InMemoryStorageProvider},
 * {@link FileSystemStorageProvider}, {@link CompositeStorageProvider}.
 *
 * Mirrors `rust/crates/spore-core/src/storage.rs`. See `./types.ts` for the
 * domain-store interfaces and the pinned rules.
 */

import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { dirname, join } from "node:path";

import { SessionId } from "../harness/types.js";
import type { PausedState } from "../harness/types.js";
import type { Timestamp } from "../memory/types.js";
import type { SessionMetrics } from "../observability/types.js";
import type { SessionOutcome } from "../guide-registry/types.js";

import type { StorageScope } from "../prompt-assembly/types.js";

import { StorageIoError, StorageSerializationError } from "./errors.js";
import type {
  FullStorageProvider,
  JsonValue,
  MemoryEntry,
  MemoryStore,
  ObservabilityStore,
  RunStore,
  SessionStore,
} from "./types.js";

// ============================================================================
// Shared helpers
// ============================================================================

/**
 * Return the most-recent `limit` items, newest-first, given a list in append
 * (oldest-first) order. Pure; never mutates the input.
 */
function mostRecentNewestFirst<T>(items: readonly T[], limit: number): T[] {
  const reversed = items.slice().reverse();
  return limit < 0 ? [] : reversed.slice(0, limit);
}

/**
 * The `timestamp` field of a {@link MemoryEntry} as a comparable string. Entries
 * from the in-memory backend carry a {@link Timestamp} instance; entries read
 * back from a JSONL backend carry a plain string. Both are handled so the merge
 * works regardless of backend (#78 R6).
 */
function timestampKey(entry: MemoryEntry): string {
  const t = entry.timestamp as unknown;
  if (typeof t === "string") return t;
  if (t != null && typeof (t as { asString?: () => string }).asString === "function") {
    return (t as { asString: () => string }).asString();
  }
  return String(t);
}

/**
 * Merge step for the cross-scope memory read (#78 R6): sort newest-first by
 * `timestamp` and truncate to `limit`. **No dedup** — identical-content entries
 * are all retained. A *stable* sort (the spec's order among equal timestamps)
 * keeps the merge deterministic cross-language: `Array.prototype.sort` is
 * guaranteed stable in modern V8.
 */
function mergeNewestFirst(entries: readonly MemoryEntry[], limit: number): MemoryEntry[] {
  const sorted = entries.slice().sort((a, b) => {
    const ka = timestampKey(a);
    const kb = timestampKey(b);
    // Newest-first: descending by timestamp string.
    return ka < kb ? 1 : ka > kb ? -1 : 0;
  });
  return limit < 0 ? [] : sorted.slice(0, limit);
}

/**
 * Round-trip a value through JSON so identity wrappers (`SessionId`, `Timestamp`,
 * etc.) serialize to their wire form and the result is a plain, opaque JSON
 * value — the cross-language ground truth for what lands on disk / in memory.
 */
function toJsonValue(v: unknown): JsonValue {
  return JSON.parse(JSON.stringify(v ?? null)) as JsonValue;
}

// ============================================================================
// NoOpStorageProvider
// ============================================================================

/**
 * Silent-discard provider. Reads return `undefined` / `[]`; writes resolve. The
 * default for any unconfigured domain.
 */
export class NoOpStorageProvider implements FullStorageProvider {
  // SessionStore
  async getSession(_id: SessionId): Promise<PausedState | undefined> {
    return undefined;
  }
  async putSession(_id: SessionId, _state: PausedState): Promise<void> {}
  async deleteSession(_id: SessionId): Promise<void> {}
  async listSessions(): Promise<SessionId[]> {
    return [];
  }

  // MemoryStore
  async appendMemory(
    _scope: StorageScope,
    _sessionId: SessionId,
    _entry: MemoryEntry,
  ): Promise<void> {}
  async getMemories(
    _scope: StorageScope,
    _sessionId: SessionId,
    _limit: number,
  ): Promise<MemoryEntry[]> {
    return [];
  }

  // RunStore
  async get(_sessionId: SessionId, _key: string): Promise<JsonValue | undefined> {
    return undefined;
  }
  async put(_sessionId: SessionId, _key: string, _value: JsonValue): Promise<void> {}
  async delete(_sessionId: SessionId, _key: string): Promise<void> {}
  async listKeys(_sessionId: SessionId): Promise<string[]> {
    return [];
  }

  // ObservabilityStore
  async appendSpan(_sessionId: SessionId, _span: JsonValue): Promise<void> {}
  async getSpans(_sessionId: SessionId): Promise<JsonValue[]> {
    return [];
  }
  async getSessions(
    _since: Timestamp,
    _domain?: string,
    _outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]> {
    return [];
  }
  async flushSession(_sessionId: SessionId): Promise<void> {}
}

// ============================================================================
// InMemoryStorageProvider
// ============================================================================

/** In-process map-backed provider. Used in tests and ephemeral runs. */
export class InMemoryStorageProvider implements FullStorageProvider {
  private readonly sessions = new Map<string, PausedState>();
  private readonly memories = new Map<string, MemoryEntry[]>();
  private readonly run = new Map<string, JsonValue>();
  private readonly spans = new Map<string, JsonValue[]>();

  private runKey(sessionId: string, key: string): string {
    // Length-prefix the session id so distinct (session, key) pairs never
    // collide regardless of the characters in either component.
    return `${sessionId.length}:${sessionId}/${key}`;
  }

  private memoryKey(scope: StorageScope, sessionId: string): string {
    // Key memory by (scope, sessionId) (#78). The scope is a closed enum string,
    // so a simple `scope/sessionId` join is collision-free.
    return `${scope}/${sessionId}`;
  }

  // SessionStore
  async getSession(id: SessionId): Promise<PausedState | undefined> {
    return this.sessions.get(id.asString());
  }
  async putSession(id: SessionId, state: PausedState): Promise<void> {
    this.sessions.set(id.asString(), state);
  }
  async deleteSession(id: SessionId): Promise<void> {
    this.sessions.delete(id.asString());
  }
  async listSessions(): Promise<SessionId[]> {
    return [...this.sessions.keys()].sort().map((s) => SessionId.of(s));
  }

  // MemoryStore — keyed by (scope, sessionId) (#78).
  async appendMemory(scope: StorageScope, sessionId: SessionId, entry: MemoryEntry): Promise<void> {
    const key = this.memoryKey(scope, sessionId.asString());
    const list = this.memories.get(key);
    if (list) list.push(entry);
    else this.memories.set(key, [entry]);
  }
  async getMemories(
    scope: StorageScope,
    sessionId: SessionId,
    limit: number,
  ): Promise<MemoryEntry[]> {
    const list = this.memories.get(this.memoryKey(scope, sessionId.asString())) ?? [];
    return mostRecentNewestFirst(list, limit);
  }

  // RunStore
  async get(sessionId: SessionId, key: string): Promise<JsonValue | undefined> {
    return this.run.get(this.runKey(sessionId.asString(), key));
  }
  async put(sessionId: SessionId, key: string, value: JsonValue): Promise<void> {
    this.run.set(this.runKey(sessionId.asString(), key), value);
  }
  async delete(sessionId: SessionId, key: string): Promise<void> {
    this.run.delete(this.runKey(sessionId.asString(), key));
  }
  async listKeys(sessionId: SessionId): Promise<string[]> {
    const prefix = this.runKey(sessionId.asString(), "");
    const out: string[] = [];
    for (const composite of this.run.keys()) {
      if (composite.startsWith(prefix)) out.push(composite.slice(prefix.length));
    }
    return out.sort();
  }

  // ObservabilityStore
  async appendSpan(sessionId: SessionId, span: JsonValue): Promise<void> {
    const key = sessionId.asString();
    const list = this.spans.get(key);
    if (list) list.push(span);
    else this.spans.set(key, [span]);
  }
  async getSpans(sessionId: SessionId): Promise<JsonValue[]> {
    return (this.spans.get(sessionId.asString()) ?? []).slice();
  }
  async getSessions(
    _since: Timestamp,
    _domain?: string,
    _outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]> {
    // SessionMetrics roll-up is owned by the ObservabilityProvider, not the raw
    // in-memory span store. Storage-only query returns empty.
    return [];
  }
  async flushSession(_sessionId: SessionId): Promise<void> {}
}

// ============================================================================
// FileSystemStorageProvider
// ============================================================================

/**
 * Atomic write-rename: ensure the parent dir, write full bytes to a sibling
 * `{target}.tmp`, fsync, then rename to the target. On any failure the `.tmp` is
 * removed so no partial sidecar is left behind. Byte-identical algorithm across
 * all four languages.
 */
function atomicWrite(target: string, bytes: string): void {
  mkdirSync(dirname(target), { recursive: true });
  const tmp = `${target}.tmp`;
  try {
    const fd = openSync(tmp, "w");
    try {
      writeSync(fd, bytes);
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
    renameSync(tmp, target);
  } catch (err) {
    // Best-effort cleanup; leave no leftover .tmp.
    try {
      rmSync(tmp, { force: true });
    } catch {
      // ignore
    }
    throw new StorageIoError(`atomic write to ${target} failed`, err);
  }
}

/** Append one JSONL line (the value plus a trailing `\n`), flushing the handle. */
function appendJsonl(path: string, value: JsonValue): void {
  mkdirSync(dirname(path), { recursive: true });
  let line: string;
  try {
    line = JSON.stringify(value);
  } catch (err) {
    throw new StorageSerializationError(`failed to serialize JSONL line for ${path}`, err);
  }
  let fd: number;
  try {
    fd = openSync(path, "a");
  } catch (err) {
    throw new StorageIoError(`failed to open ${path} for append`, err);
  }
  try {
    writeSync(fd, `${line}\n`);
    fsyncSync(fd);
  } catch (err) {
    throw new StorageIoError(`failed to append to ${path}`, err);
  } finally {
    closeSync(fd);
  }
}

/** Read every non-empty JSONL line from `path`. Missing file → empty list. */
function readJsonl(path: string): JsonValue[] {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw new StorageIoError(`failed to read ${path}`, err);
  }
  const out: JsonValue[] = [];
  for (const line of raw.split("\n")) {
    if (line.trim().length === 0) continue;
    try {
      out.push(JSON.parse(line) as JsonValue);
    } catch (err) {
      throw new StorageSerializationError(`failed to parse JSONL line in ${path}`, err);
    }
  }
  return out;
}

/**
 * Disk-backed provider rooted at `root`. Layout mirrors `.spore/`:
 *   - session → `{root}/sessions/{id}/state.json` (atomic write-rename)
 *   - run     → `{root}/sessions/{id}/run/{key}.json` (atomic write-rename)
 *   - memory  → `{root}/sessions/{id}/memory.jsonl` (append)
 *   - obs     → `{root}/sessions/{id}/trace.jsonl` (append)
 *
 * `flushSession` creates a sibling `.flushed` marker.
 */
export class FileSystemStorageProvider implements FullStorageProvider {
  constructor(private readonly rootDir: string) {}

  root(): string {
    return this.rootDir;
  }

  private sessionDir(id: SessionId): string {
    return join(this.rootDir, "sessions", id.asString());
  }
  private statePath(id: SessionId): string {
    return join(this.sessionDir(id), "state.json");
  }
  private runDir(id: SessionId): string {
    return join(this.sessionDir(id), "run");
  }
  private runPath(id: SessionId, key: string): string {
    return join(this.runDir(id), `${key}.json`);
  }
  private memoryPath(id: SessionId): string {
    return join(this.sessionDir(id), "memory.jsonl");
  }
  private tracePath(id: SessionId): string {
    return join(this.sessionDir(id), "trace.jsonl");
  }

  // SessionStore
  async getSession(id: SessionId): Promise<PausedState | undefined> {
    let raw: string;
    try {
      raw = readFileSync(this.statePath(id), "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return undefined;
      throw new StorageIoError(`failed to read session ${id.asString()}`, err);
    }
    try {
      return JSON.parse(raw) as PausedState;
    } catch (err) {
      throw new StorageSerializationError(`failed to parse session ${id.asString()}`, err);
    }
  }
  async putSession(id: SessionId, state: PausedState): Promise<void> {
    let bytes: string;
    try {
      bytes = JSON.stringify(state);
    } catch (err) {
      throw new StorageSerializationError(`failed to serialize session ${id.asString()}`, err);
    }
    atomicWrite(this.statePath(id), bytes);
  }
  async deleteSession(id: SessionId): Promise<void> {
    try {
      rmSync(this.statePath(id));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return;
      throw new StorageIoError(`failed to delete session ${id.asString()}`, err);
    }
  }
  async listSessions(): Promise<SessionId[]> {
    const dir = join(this.rootDir, "sessions");
    let names: string[];
    try {
      names = readdirSync(dir);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw new StorageIoError(`failed to list sessions`, err);
    }
    const out: SessionId[] = [];
    for (const name of names) {
      if (existsSync(join(dir, name, "state.json"))) out.push(SessionId.of(name));
    }
    return out.sort((a, b) => a.asString().localeCompare(b.asString()));
  }

  // MemoryStore — **scope-dumb** (#78): the user-scope backend is pointed at the
  // already-partitioned `{userRoot}/projects/{workspaceId}` at construction. The
  // provider just writes under whatever root it was given; `scope` is ignored at
  // the leaf.
  async appendMemory(
    _scope: StorageScope,
    sessionId: SessionId,
    entry: MemoryEntry,
  ): Promise<void> {
    appendJsonl(this.memoryPath(sessionId), toJsonValue(entry));
  }
  async getMemories(
    _scope: StorageScope,
    sessionId: SessionId,
    limit: number,
  ): Promise<MemoryEntry[]> {
    const values = readJsonl(this.memoryPath(sessionId));
    const entries = values as unknown as MemoryEntry[];
    return mostRecentNewestFirst(entries, limit);
  }

  // RunStore
  async get(sessionId: SessionId, key: string): Promise<JsonValue | undefined> {
    let raw: string;
    try {
      raw = readFileSync(this.runPath(sessionId, key), "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return undefined;
      throw new StorageIoError(`failed to read run key ${key}`, err);
    }
    try {
      return JSON.parse(raw) as JsonValue;
    } catch (err) {
      throw new StorageSerializationError(`failed to parse run key ${key}`, err);
    }
  }
  async put(sessionId: SessionId, key: string, value: JsonValue): Promise<void> {
    let bytes: string;
    try {
      bytes = JSON.stringify(value);
    } catch (err) {
      throw new StorageSerializationError(`failed to serialize run key ${key}`, err);
    }
    atomicWrite(this.runPath(sessionId, key), bytes);
  }
  async delete(sessionId: SessionId, key: string): Promise<void> {
    try {
      rmSync(this.runPath(sessionId, key));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return;
      throw new StorageIoError(`failed to delete run key ${key}`, err);
    }
  }
  async listKeys(sessionId: SessionId): Promise<string[]> {
    let names: string[];
    try {
      names = readdirSync(this.runDir(sessionId));
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw new StorageIoError(`failed to list run keys`, err);
    }
    const out: string[] = [];
    for (const name of names) {
      if (name.endsWith(".json")) out.push(name.slice(0, -".json".length));
    }
    return out.sort();
  }

  // ObservabilityStore
  async appendSpan(sessionId: SessionId, span: JsonValue): Promise<void> {
    appendJsonl(this.tracePath(sessionId), span);
  }
  async getSpans(sessionId: SessionId): Promise<JsonValue[]> {
    return readJsonl(this.tracePath(sessionId));
  }
  async getSessions(
    _since: Timestamp,
    _domain?: string,
    _outcome?: SessionOutcome,
  ): Promise<SessionMetrics[]> {
    // SessionMetrics roll-up is owned by the ObservabilityProvider, not the raw
    // on-disk span store. Storage-only query returns empty.
    return [];
  }
  async flushSession(sessionId: SessionId): Promise<void> {
    const dir = this.sessionDir(sessionId);
    mkdirSync(dir, { recursive: true });
    try {
      writeFileSync(join(dir, ".flushed"), "");
    } catch (err) {
      throw new StorageIoError(`failed to write .flushed marker`, err);
    }
  }
}

// ============================================================================
// StorageProvider
// ============================================================================

/**
 * A composed persistence layer: four independent domain stores. Built either
 * from a single backend (placed in all four slots via {@link StorageProvider.single})
 * or per-domain via {@link CompositeStorageProvider}.
 */
export class StorageProvider {
  private constructor(
    private readonly _session: SessionStore,
    private readonly _memory: MemoryStore,
    private readonly _run: RunStore,
    private readonly _observability: ObservabilityStore,
  ) {}

  /** Construct from four explicit per-domain stores. */
  static of(
    session: SessionStore,
    memory: MemoryStore,
    run: RunStore,
    observability: ObservabilityStore,
  ): StorageProvider {
    return new StorageProvider(session, memory, run, observability);
  }

  /** Place a single provider implementing all four domain interfaces into all
   *  four slots. */
  static single(provider: FullStorageProvider): StorageProvider {
    return new StorageProvider(provider, provider, provider, provider);
  }

  /** All-no-op provider. The default when `.storage(...)` is never set. */
  static noOp(): StorageProvider {
    return StorageProvider.single(new NoOpStorageProvider());
  }

  session(): SessionStore {
    return this._session;
  }
  memory(): MemoryStore {
    return this._memory;
  }

  /**
   * Merged memory read across scopes (#78 R6): **User ∪ Project, newest-first by
   * `timestamp`, NO dedup**. `Local` is excluded from the merge in v1.
   *
   * Routes through the memory slot — when built via {@link CompositeStorageProvider}
   * that slot is a {@link ScopedMemoryRouter} that fans out to the per-scope
   * backends and merges; for `single`/`of` the one backend serves both scopes
   * (keyed by scope) and merges identically. The merge always lives in this
   * routing layer, never in a leaf backend.
   */
  async getMemoriesMerged(sessionId: SessionId, limit: number): Promise<MemoryEntry[]> {
    const user = await this._memory.getMemories("user", sessionId, limit);
    const project = await this._memory.getMemories("project", sessionId, limit);
    const combined = [...user, ...project];
    return mergeNewestFirst(combined, limit);
  }

  run(): RunStore {
    return this._run;
  }
  observability(): ObservabilityStore {
    return this._observability;
  }
}

// ============================================================================
// CompositeStorageProvider
// ============================================================================

/**
 * Builder that routes each domain to its own backend — and, for the memory
 * domain, each {@link StorageScope} to its own backend (#78) — filling any unset
 * slot with {@link NoOpStorageProvider} on `.build()`.
 *
 * Only the `memory` domain varies by scope. `session`, `run`, and
 * `observability` are scope-flat — scope is wiring-only for them.
 *
 * @example
 * ```ts
 * new CompositeStorageProvider()
 *   .session(fs(userRoot))                            // scope-flat
 *   .run(fs(userRoot))                                // scope-flat
 *   .observability(fs(userRoot))                      // scope-flat
 *   .memory("user", fs(userWorkspaceRoot))            // scoped
 *   .memory("project", fs(projectRoot))               // scoped
 *   .memory("local", new NoOpStorageProvider())       // scoped (noop in v1)
 *   .build();
 * ```
 */
export class CompositeStorageProvider {
  private _session?: SessionStore;
  private readonly _memory = new Map<StorageScope, MemoryStore>();
  private _run?: RunStore;
  private _observability?: ObservabilityStore;

  session(store: SessionStore): this {
    this._session = store;
    return this;
  }
  /**
   * Configure the memory backend for one {@link StorageScope}. Unconfigured
   * `(memory, scope)` pairs fall back to {@link NoOpStorageProvider} on
   * `.build()` (#78 R7/R11 — `Local` may be wired to no-op in v1).
   */
  memory(scope: StorageScope, store: MemoryStore): this {
    this._memory.set(scope, store);
    return this;
  }
  run(store: RunStore): this {
    this._run = store;
    return this;
  }
  observability(store: ObservabilityStore): this {
    this._observability = store;
    return this;
  }

  /**
   * Build a {@link StorageProvider}, filling each unset domain — and each unset
   * `(memory, scope)` pair — with a {@link NoOpStorageProvider}.
   */
  build(): StorageProvider {
    const noop = new NoOpStorageProvider();
    return StorageProvider.of(
      this._session ?? noop,
      new ScopedMemoryRouter(this._memory),
      this._run ?? noop,
      this._observability ?? noop,
    );
  }
}

// ============================================================================
// ScopedMemoryRouter (#78) — the (memory, scope) routing layer
// ============================================================================

/**
 * Routes {@link MemoryStore} traffic to a per-{@link StorageScope} backend,
 * filling unconfigured scopes with {@link NoOpStorageProvider}. Leaf backends
 * stay scope-dumb; the cross-scope merge lives one level up in
 * {@link StorageProvider.getMemoriesMerged}.
 *
 * This is a {@link StorageProvider}'s memory slot when built via
 * {@link CompositeStorageProvider}: a caller that passes a `scope` is routed to
 * the right backend, and {@link StorageProvider.getMemoriesMerged} reaches each
 * scope through it.
 */
export class ScopedMemoryRouter implements MemoryStore {
  private readonly byScope: Map<StorageScope, MemoryStore>;
  private readonly noop: MemoryStore = new NoOpStorageProvider();

  constructor(byScope: Map<StorageScope, MemoryStore>) {
    // Copy so later mutation of the builder's map cannot affect a built router.
    this.byScope = new Map(byScope);
  }

  /** The backend for `scope`, or the shared no-op if unconfigured. */
  private backend(scope: StorageScope): MemoryStore {
    return this.byScope.get(scope) ?? this.noop;
  }

  async appendMemory(scope: StorageScope, sessionId: SessionId, entry: MemoryEntry): Promise<void> {
    return this.backend(scope).appendMemory(scope, sessionId, entry);
  }

  async getMemories(
    scope: StorageScope,
    sessionId: SessionId,
    limit: number,
  ): Promise<MemoryEntry[]> {
    return this.backend(scope).getMemories(scope, sessionId, limit);
  }
}

// ============================================================================
// OTLP endpoint parsing (cross-language ground truth)
// ============================================================================

/**
 * Parse the comma-separated `SPORE_OTLP_ENDPOINT` value: `split(',')`, trim each
 * segment, drop empty segments. This is the single most important cross-language
 * fixture (`fixtures/storage/otlp_endpoints_parse.json`) and MUST be
 * byte-identical in every language.
 */
export function parseOtlpEndpoints(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
