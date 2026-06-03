/**
 * Unit tests for the `MarkdownMemoryProvider` shipped by example 07.
 *
 * They are driven directly over a temp `memory.md` file (no model, no Ollama),
 * proving each behavior the example demonstrates:
 *   - a missing file reads as empty.
 *   - append → get round-trips an entry, and writes real readable markdown.
 *   - multi-line content round-trips.
 *   - scope filtering isolates scopes.
 *   - session filtering isolates sessions.
 *   - get returns newest-first by timestamp.
 *   - limit takes the most-recent N.
 *   - a hand-edited file is tolerated (prose + extra headings ignored).
 *   - composition: the markdown provider fills the memory slot of a full
 *     StorageProvider, NoOp fills the rest.
 */

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  SessionId,
  memory as coreMemory,
  storage as coreStorage,
} from "@spore/core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { MarkdownMemoryProvider } from "../src/memory-provider.js";

type MemoryEntry = coreStorage.MemoryEntry;

const { Timestamp } = coreMemory;

function entry(role: string, content: string, ts: string): MemoryEntry {
  return { role, content, timestamp: Timestamp.of(ts), metadata: {} };
}

function sid(): SessionId {
  return SessionId.of("s1");
}

let dir: string;
let path: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "spore07-"));
  path = join(dir, "memory.md");
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe("MarkdownMemoryProvider", () => {
  it("reads a missing file as empty", async () => {
    const provider = new MarkdownMemoryProvider(path);
    const got = await provider.getMemories("project", sid(), 50);
    expect(got).toEqual([]);
  });

  it("round-trips an appended entry and writes readable markdown", async () => {
    const provider = new MarkdownMemoryProvider(path);
    await provider.appendMemory(
      "project",
      sid(),
      entry(
        "assistant",
        "Postgres is the system of record.",
        "2026-06-02T10:00:00Z",
      ),
    );

    const got = await provider.getMemories("project", sid(), 50);
    expect(got).toHaveLength(1);
    expect(got[0]!.role).toBe("assistant");
    expect(got[0]!.content).toBe("Postgres is the system of record.");
    expect(got[0]!.timestamp.asString()).toBe("2026-06-02T10:00:00Z");

    // The artifact is real, readable markdown on disk.
    const raw = readFileSync(path, "utf8");
    expect(raw).toContain("## [project] [s1] 2026-06-02T10:00:00Z — assistant");
    expect(raw).toContain("Postgres is the system of record.");
  });

  it("round-trips multi-line content", async () => {
    const provider = new MarkdownMemoryProvider(path);
    await provider.appendMemory(
      "project",
      sid(),
      entry(
        "assistant",
        "line one\nline two\n\nline four",
        "2026-06-02T10:00:00Z",
      ),
    );
    const got = await provider.getMemories("project", sid(), 50);
    expect(got[0]!.content).toBe("line one\nline two\n\nline four");
  });

  it("isolates scopes", async () => {
    const provider = new MarkdownMemoryProvider(path);
    await provider.appendMemory(
      "project",
      sid(),
      entry("user", "proj", "2026-06-02T10:00:00Z"),
    );
    await provider.appendMemory(
      "user",
      sid(),
      entry("user", "usr", "2026-06-02T10:00:01Z"),
    );

    const proj = await provider.getMemories("project", sid(), 50);
    expect(proj).toHaveLength(1);
    expect(proj[0]!.content).toBe("proj");

    const usr = await provider.getMemories("user", sid(), 50);
    expect(usr).toHaveLength(1);
    expect(usr[0]!.content).toBe("usr");
  });

  it("isolates sessions", async () => {
    const provider = new MarkdownMemoryProvider(path);
    const a = SessionId.of("alpha");
    const b = SessionId.of("beta");
    await provider.appendMemory(
      "project",
      a,
      entry("user", "from-alpha", "2026-06-02T10:00:00Z"),
    );
    await provider.appendMemory(
      "project",
      b,
      entry("user", "from-beta", "2026-06-02T10:00:01Z"),
    );

    const gotA = await provider.getMemories("project", a, 50);
    expect(gotA).toHaveLength(1);
    expect(gotA[0]!.content).toBe("from-alpha");
  });

  it("returns newest-first by timestamp", async () => {
    const provider = new MarkdownMemoryProvider(path);
    // Appended out of timestamp order on purpose.
    await provider.appendMemory(
      "project",
      sid(),
      entry("user", "middle", "2026-06-02T11:00:00Z"),
    );
    await provider.appendMemory(
      "project",
      sid(),
      entry("user", "oldest", "2026-06-02T10:00:00Z"),
    );
    await provider.appendMemory(
      "project",
      sid(),
      entry("user", "newest", "2026-06-02T12:00:00Z"),
    );

    const got = await provider.getMemories("project", sid(), 50);
    expect(got.map((e) => e.content)).toEqual(["newest", "middle", "oldest"]);
  });

  it("limit takes the most-recent entries", async () => {
    const provider = new MarkdownMemoryProvider(path);
    for (let i = 0; i < 5; i++) {
      await provider.appendMemory(
        "project",
        sid(),
        entry("user", `e${i}`, `2026-06-02T10:00:0${i}Z`),
      );
    }
    const got = await provider.getMemories("project", sid(), 2);
    expect(got.map((e) => e.content)).toEqual(["e4", "e3"]);
  });

  it("tolerates a hand-edited file", async () => {
    // A file a human authored/edited: prose before the first header, an extra
    // heading, blank lines, and a normal entry block.
    const hand =
      "# My Notes\n\n" +
      "Some rambling prose that is not an entry.\n\n" +
      "## [project] [s1] 2026-06-02T09:00:00Z — user\n\n" +
      "Hand-written fact about Ironwood.\n\n" +
      "## A non-entry heading the human added\n\n" +
      "more prose\n";
    writeFileSync(path, hand);

    const provider = new MarkdownMemoryProvider(path);
    let got = await provider.getMemories("project", sid(), 50);
    expect(got).toHaveLength(1);
    expect(got[0]!.content).toBe("Hand-written fact about Ironwood.");

    // And we can still append on top of the hand-edited file.
    await provider.appendMemory(
      "project",
      sid(),
      entry("assistant", "appended", "2026-06-02T10:00:00Z"),
    );
    got = await provider.getMemories("project", sid(), 50);
    expect(got).toHaveLength(2);
    expect(got[0]!.content).toBe("appended"); // newest-first
  });

  it("composes into the memory slot of a full StorageProvider", async () => {
    const provider = new MarkdownMemoryProvider(path);
    const storage = provider.intoStorageProvider();

    // The memory slot is the markdown provider; round-trips through it.
    await storage
      .memory()
      .appendMemory(
        "project",
        sid(),
        entry("user", "via-seam", "2026-06-02T10:00:00Z"),
      );
    const got = await storage.memory().getMemories("project", sid(), 50);
    expect(got).toHaveLength(1);
    expect(got[0]!.content).toBe("via-seam");

    // The other three domains are no-ops: a run read returns nothing.
    expect(await storage.run().get(sid(), "k")).toBeUndefined();

    // The artifact exists on disk.
    expect(existsSync(path)).toBe(true);
  });
});
