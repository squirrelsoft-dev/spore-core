/**
 * `MarkdownMemoryProvider` — a {@link storage.MemoryStore} that persists agent
 * memory to a single human-readable `memory.md` file on disk.
 *
 * ## What this demonstrates
 * The storage seam. The harness is **stateless** — every byte of durable state
 * lives behind a {@link storage.StorageProvider}. Memory is one of its four
 * domains (`MemoryStore`). This module implements *only* that domain and
 * composes it with {@link storage.NoOpStorageProvider} for the other three
 * (session / run / observability). That composed provider is what `main.ts`
 * hands to `HarnessBuilder.storage(...)`; the harness then threads
 * `storage.memory()` into the built-in `memory` tool's `ToolContext` on every
 * run. No custom harness plumbing — the seam is the whole integration surface.
 *
 * ## The seam
 * `MemoryStore` is the swap point. The built-in `FileSystemStorageProvider`
 * persists memory as a JSONL log; this provider persists the *same*
 * {@link storage.MemoryEntry} values to readable markdown instead. Same
 * interface, same agent, same tool — different on-disk shape. Anything
 * implementing `MemoryStore` slots in here.
 *
 * ## On-disk format (round-trips exactly)
 * Each {@link storage.MemoryEntry} is one markdown block. The header line
 * carries the round-trip fields; the body is the content:
 *
 * ```text
 * ## [project] [project-ironwood] 2026-06-02T12:00:00Z — assistant
 *
 * Postgres 15 is the system of record.
 * ```
 *
 * `appendMemory` writes such a block; `getMemories` parses them back, filters by
 * scope + session, sorts newest-first by timestamp, and takes `limit`. A
 * hand-edited file (extra prose, blank lines, reordered blocks) is tolerated:
 * anything that is not a recognized `## [scope] [session] timestamp — role`
 * header is treated as body for the preceding entry, and leading prose before
 * the first header is ignored.
 *
 * ## Pinned-session-id requirement (read this)
 * Memory is keyed by `SessionId`. The `memory` tool always uses
 * `ctx.sessionId`. For Run 2 (recall) to read Run 1's (store) memories, **both
 * runs MUST use the SAME `SessionId`** — see `main.ts`, which pins
 * `SessionId.of("project-ironwood")` rather than `SessionId.generate()`. This
 * provider also stores the session id in each header so a single `memory.md`
 * can hold multiple sessions without cross-talk.
 *
 * There are no SPEC QUESTION markers in this file.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import {
  SessionId,
  memory as coreMemory,
  storage as coreStorage,
} from "@spore/core";

type MemoryEntry = coreStorage.MemoryEntry;
type MemoryStore = coreStorage.MemoryStore;
type StorageScope = coreStorage.StorageScope;

const { Timestamp } = coreMemory;
const { StorageProvider, NoOpStorageProvider, getMemoriesMergedDefault } =
  coreStorage;

/** The header that introduces a fresh `memory.md`, written on first append. */
const FILE_PREAMBLE =
  "# Agent Memory\n\n" +
  "Human-readable working memory for this agent. Each `##` block below is one " +
  "remembered entry.\n";

/** How a {@link StorageScope} is spelled in a header line. */
function scopeToken(scope: StorageScope): string {
  return scope;
}

/** Parse a scope token back to a {@link StorageScope}; `undefined` if unknown. */
function parseScopeToken(s: string): StorageScope | undefined {
  return s === "user" || s === "project" || s === "local" ? s : undefined;
}

/**
 * Render one entry as a markdown block (header line + blank line + body). The
 * session id is encoded so one file can hold multiple sessions. The em-dash
 * separator is ` — ` (space, U+2014, space).
 */
function renderBlock(
  scope: StorageScope,
  sessionId: SessionId,
  entry: MemoryEntry,
): string {
  const ts = timestampString(entry);
  const content = entry.content.replace(/\s+$/, "");
  return (
    `## [${scopeToken(scope)}] [${sessionId.asString()}] ${ts} — ${entry.role}\n\n` +
    `${content}\n`
  );
}

/**
 * The `timestamp` field as a comparable string. The `memory` tool writes a
 * {@link coreMemory.Timestamp} instance; a hand-written entry may carry a plain
 * string. Both are handled.
 */
function timestampString(entry: MemoryEntry): string {
  const t = entry.timestamp as unknown;
  if (typeof t === "string") return t;
  if (
    t != null &&
    typeof (t as { asString?: () => string }).asString === "function"
  ) {
    return (t as { asString: () => string }).asString();
  }
  return String(t);
}

/** A parsed entry plus the scope + session it was filed under. */
interface ParsedBlock {
  scope: StorageScope;
  session: string;
  entry: MemoryEntry;
}

/** A header line's decoded fields. */
interface Header {
  scope: StorageScope;
  session: string;
  timestamp: string;
  role: string;
}

/**
 * Parse a header line of the form `## [scope] [session] timestamp — role`.
 * Returns `undefined` for any line that is not a recognized header (so prose and
 * hand-edits are tolerated).
 */
function parseHeader(line: string): Header | undefined {
  if (!line.startsWith("## [")) return undefined;
  let rest = line.slice("## ".length);

  // [scope]
  if (rest[0] !== "[") return undefined;
  const scopeEnd = rest.indexOf("] ");
  if (scopeEnd < 0) return undefined;
  const scope = parseScopeToken(rest.slice(1, scopeEnd).trim());
  if (scope === undefined) return undefined;
  rest = rest.slice(scopeEnd + 2);

  // [session]
  if (rest[0] !== "[") return undefined;
  const sessionEnd = rest.indexOf("] ");
  if (sessionEnd < 0) return undefined;
  const session = rest.slice(1, sessionEnd).trim();
  rest = rest.slice(sessionEnd + 2);

  // timestamp — role
  const sepIndex = rest.indexOf(" — ");
  if (sepIndex < 0) return undefined;
  const timestamp = rest.slice(0, sepIndex).trim();
  const role = rest.slice(sepIndex + " — ".length).trim();
  if (timestamp.length === 0 || role.length === 0) return undefined;

  return { scope, session, timestamp, role };
}

/**
 * Parse the whole file into blocks. Body lines accumulate under the most recent
 * header; text before the first header is discarded. A `## ` line that is NOT a
 * valid entry header (e.g. a human-added subheading) terminates the current
 * block and is otherwise ignored, so hand-edited headings never pollute an
 * entry's content.
 */
function parseFile(contents: string): ParsedBlock[] {
  const blocks: ParsedBlock[] = [];
  let current: { header: Header; body: string[] } | undefined;

  const flush = (): void => {
    if (current !== undefined) {
      blocks.push(finishBlock(current.header, current.body));
      current = undefined;
    }
  };

  for (const line of contents.split("\n")) {
    const header = parseHeader(line);
    if (header !== undefined) {
      flush();
      current = { header, body: [] };
    } else if (line.startsWith("## ")) {
      // A non-entry `## ` heading ends the current block and is ignored.
      flush();
    } else if (current !== undefined) {
      current.body.push(line);
    }
    // else: prose before the first header — ignored.
  }
  flush();
  return blocks;
}

function finishBlock(header: Header, body: string[]): ParsedBlock {
  // Trim the blank lines that bracket the body, then rejoin.
  const content = body.join("\n").trim();
  return {
    scope: header.scope,
    session: header.session,
    entry: {
      role: header.role,
      content,
      timestamp: Timestamp.of(header.timestamp),
      metadata: {},
    },
  };
}

/**
 * A {@link storage.MemoryStore} backed by a single human-readable `memory.md`
 * file. `appendMemory` is a read-modify-write; the synchronous `fs` calls below
 * are atomic enough for this single-agent demo and the harness dispatches the
 * `memory` tool sequentially, so appends never interleave a partial write.
 */
export class MarkdownMemoryProvider implements MemoryStore {
  constructor(private readonly path: string) {}

  /**
   * Compose this provider into a full {@link storage.StorageProvider}: the real
   * `MemoryStore` for the memory domain, {@link storage.NoOpStorageProvider}
   * for the other three. This is exactly what the example hands to the harness.
   */
  intoStorageProvider(): coreStorage.StorageProvider {
    const noop = new NoOpStorageProvider();
    return StorageProvider.of(noop, this, noop, noop);
  }

  async appendMemory(
    scope: StorageScope,
    sessionId: SessionId,
    entry: MemoryEntry,
  ): Promise<void> {
    // Read-modify-write: load existing text (or seed the preamble), append a
    // new block, and write it all back.
    let existing: string;
    try {
      existing = readFileSync(this.path, "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") {
        existing = FILE_PREAMBLE;
      } else {
        throw err;
      }
    }

    if (!existing.endsWith("\n")) existing += "\n";
    existing += "\n" + renderBlock(scope, sessionId, entry);

    const parent = dirname(this.path);
    if (parent.length > 0) mkdirSync(parent, { recursive: true });
    writeFileSync(this.path, existing);
  }

  async getMemories(
    scope: StorageScope,
    sessionId: SessionId,
    limit: number,
  ): Promise<MemoryEntry[]> {
    let contents: string;
    try {
      contents = readFileSync(this.path, "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw err;
    }

    const session = sessionId.asString();
    const entries = parseFile(contents)
      .filter((b) => b.scope === scope && b.session === session)
      .map((b) => b.entry);

    // Newest-first by timestamp. RFC-3339 strings sort lexically; a stable sort
    // keeps ties in append order, mirroring insertion order.
    entries.sort((a, b) => {
      const ka = timestampString(a);
      const kb = timestampString(b);
      return ka < kb ? 1 : ka > kb ? -1 : 0;
    });
    return limit < 0 ? [] : entries.slice(0, limit);
  }

  async getMemoriesMerged(
    sessionId: SessionId,
    limit: number,
  ): Promise<MemoryEntry[]> {
    // Delegate to the SINGLE shared merge implementation — never reimplement it.
    return getMemoriesMergedDefault(this, sessionId, limit);
  }
}
