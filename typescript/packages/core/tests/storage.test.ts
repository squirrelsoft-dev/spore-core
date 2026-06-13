/**
 * Unit + fixture-replay tests for the StorageProvider abstraction (issue #73).
 *
 * Covers every pinned rule: no-op fallback, composite per-domain routing,
 * single-provider-fills-all-slots, the OTLP parse table, atomic write (no
 * leftover .tmp), append ordering (memory + spans), getMemories limit + recency,
 * run-store opaque-json roundtrip + listKeys + delete, session roundtrip + list
 * + delete, flushSession marker, MemoryEntry default metadata, and cross-language
 * fixture replay. Mirrors `rust/crates/spore-core/src/storage/tests.rs`.
 */

import { describe, expect, it } from "vitest";
import { mkdtempSync, existsSync, readdirSync, statSync, mkdirSync, realpathSync, symlinkSync } from "node:fs";
import { tmpdir, platform } from "node:os";
import { join, dirname, resolve } from "node:path";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  SessionId,
  TaskId,
  emptyBudgetSnapshot,
  emptySessionState,
  newTask,
  type PausedState,
} from "../src/harness/types.js";
import { Timestamp } from "../src/memory/types.js";
import {
  CompositeStorageProvider,
  FileSystemStorageProvider,
  InMemoryStorageProvider,
  NoOpStorageProvider,
  StorageProvider,
  ScopedMemoryRouter,
  WorkspaceId,
  ProjectId,
  ProjectIdError,
  ACTIVE_RUN_KEY,
  RALPH_PROGRESS_KEY,
  loadActiveRun,
  startOrResumeActiveRun,
  completeActiveRun,
  newMemoryEntry,
  parseOtlpEndpoints,
  type JsonValue,
  type MemoryEntry,
} from "../src/storage/index.js";
import { HarnessBuilder } from "../src/harness/standard.js";
import {
  AllowAllSandbox,
  AlwaysContinuePolicy,
  NoopContextManager,
  ScriptedToolRegistry,
} from "../src/harness/testing.js";
import { MockAgent, AgentId } from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = resolve(here, "../../../../fixtures/storage");

function sid(s: string): SessionId {
  return SessionId.of(s);
}
function ts(s: string): Timestamp {
  return Timestamp.of(s);
}
function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), "spore-storage-"));
}

/** Minimal valid PausedState for roundtrip tests. */
function paused(session: string, turn = 3): PausedState {
  return {
    session_id: sid(session),
    task_id: TaskId.of("task1"),
    turn_number: turn,
    session_state: emptySessionState(),
    pending_tool_calls: [],
    approved_results: [],
    human_request: { kind: "tool_approval", calls: [], risk_level: "low" },
    task: newTask("do the thing", sid(session), {
      kind: "react",
      budget: { kind: "per_loop", value: 1 },
      agent: "",
      toolset: "",
    }),
    budget_used: emptyBudgetSnapshot(),
    child_state: null,
    toolset: "",
  };
}

function mem(role: string, content: string, t: string): MemoryEntry {
  return newMemoryEntry(role, content, ts(t));
}

// ── OTLP endpoint parsing (the most important cross-language rule) ───────────

describe("parseOtlpEndpoints", () => {
  it("matches the pinned parse table", () => {
    expect(parseOtlpEndpoints("a")).toEqual(["a"]);
    expect(parseOtlpEndpoints("a,b,c")).toEqual(["a", "b", "c"]);
    expect(parseOtlpEndpoints(" a , b ")).toEqual(["a", "b"]);
    expect(parseOtlpEndpoints("a,,b,")).toEqual(["a", "b"]);
    expect(parseOtlpEndpoints("")).toEqual([]);
    expect(parseOtlpEndpoints("  ")).toEqual([]);
  });

  it("replays the cross-language fixture", () => {
    const raw = readFileSync(join(fixturesRoot, "otlp_endpoints_parse.json"), "utf8");
    const cases = JSON.parse(raw) as { input: string; expected: string[] }[];
    for (const c of cases) {
      expect(parseOtlpEndpoints(c.input), `input ${JSON.stringify(c.input)}`).toEqual(c.expected);
    }
  });
});

// ── No-op fallback ───────────────────────────────────────────────────────────

describe("NoOpStorageProvider", () => {
  it("reads empty / undefined and writes resolve", async () => {
    const p = new NoOpStorageProvider();
    expect(await p.getSession(sid("s"))).toBeUndefined();
    expect(await p.listSessions()).toEqual([]);
    await expect(p.putSession(sid("s"), paused("s"))).resolves.toBeUndefined();
    expect(await p.getMemories("project", sid("s"), 10)).toEqual([]);
    await expect(
      p.appendMemory("project", sid("s"), mem("user", "hi", "t")),
    ).resolves.toBeUndefined();
    expect(await p.get(sid("s"), "k")).toBeUndefined();
    await expect(p.put(sid("s"), "k", 1)).resolves.toBeUndefined();
    expect(await p.listKeys(sid("s"))).toEqual([]);
    expect(await p.getSpans(sid("s"))).toEqual([]);
    await expect(p.appendSpan(sid("s"), {})).resolves.toBeUndefined();
    expect(await p.getSessions(ts("t"))).toEqual([]);
    await expect(p.flushSession(sid("s"))).resolves.toBeUndefined();
  });

  it("StorageProvider.noOp() exposes all four slots", () => {
    const p = StorageProvider.noOp();
    expect(p.session()).toBeDefined();
    expect(p.memory()).toBeDefined();
    expect(p.run()).toBeDefined();
    expect(p.observability()).toBeDefined();
  });
});

// ── Single-provider-fills-all-slots ──────────────────────────────────────────

describe("StorageProvider.single", () => {
  it("fills all four slots with one backend", async () => {
    const backend = new InMemoryStorageProvider();
    const p = StorageProvider.single(backend);
    await p.session().putSession(sid("s"), paused("s"));
    await p.memory().appendMemory("project", sid("s"), mem("user", "hi", "t1"));
    await p.run().put(sid("s"), "plan", { x: 1 });
    await p.observability().appendSpan(sid("s"), { kind: "turn" });

    expect(await p.session().getSession(sid("s"))).toBeDefined();
    expect(await p.memory().getMemories("project", sid("s"), 10)).toHaveLength(1);
    expect(await p.run().get(sid("s"), "plan")).toEqual({ x: 1 });
    expect(await p.observability().getSpans(sid("s"))).toHaveLength(1);
  });
});

// ── Composite per-domain routing + no-op fallback ────────────────────────────

describe("CompositeStorageProvider", () => {
  it("routes per domain and falls back to no-op", async () => {
    const runBackend = new InMemoryStorageProvider();
    const p = new CompositeStorageProvider().run(runBackend).build();

    await p.run().put(sid("s"), "k", "v");
    expect(await p.run().get(sid("s"), "k")).toBe("v");

    // Unconfigured domains silently no-op.
    await p.session().putSession(sid("s"), paused("s"));
    expect(await p.session().getSession(sid("s"))).toBeUndefined();
    expect(await p.memory().getMemories("project", sid("s"), 5)).toEqual([]);
    expect(await p.observability().getSpans(sid("s"))).toEqual([]);
  });
});

// ── In-memory: session roundtrip + list + delete ─────────────────────────────

describe("InMemoryStorageProvider", () => {
  it("session roundtrip, sorted list, delete", async () => {
    const p = new InMemoryStorageProvider();
    await p.putSession(sid("b"), paused("b"));
    await p.putSession(sid("a"), paused("a"));
    const got = await p.getSession(sid("a"));
    expect(got?.session_id.asString()).toBe("a");
    expect((await p.listSessions()).map((s) => s.asString())).toEqual(["a", "b"]);
    await p.deleteSession(sid("a"));
    expect(await p.getSession(sid("a"))).toBeUndefined();
    expect((await p.listSessions()).map((s) => s.asString())).toEqual(["b"]);
  });

  it("run-store opaque-json roundtrip, listKeys, delete", async () => {
    const p = new InMemoryStorageProvider();
    const blob: JsonValue = { nested: { arr: [1, 2, 3], s: "x" }, n: 4.5 };
    await p.put(sid("s"), "plan", blob);
    await p.put(sid("s"), "tasks", [1, 2]);
    expect(await p.get(sid("s"), "plan")).toEqual(blob);
    expect(await p.listKeys(sid("s"))).toEqual(["plan", "tasks"]);
    // listKeys is scoped to the session.
    await p.put(sid("other"), "z", 1);
    expect(await p.listKeys(sid("s"))).toEqual(["plan", "tasks"]);
    await p.delete(sid("s"), "plan");
    expect(await p.get(sid("s"), "plan")).toBeUndefined();
    expect(await p.listKeys(sid("s"))).toEqual(["tasks"]);
  });

  it("memory append ordering + recency limit", async () => {
    const p = new InMemoryStorageProvider();
    const contents = ["m0", "m1", "m2", "m3"];
    for (const [i, content] of contents.entries()) {
      await p.appendMemory("project", sid("s"), mem("user", content, `t${i}`));
    }
    const got = await p.getMemories("project", sid("s"), 2);
    expect(got.map((e) => e.content)).toEqual(["m3", "m2"]);
    const all = await p.getMemories("project", sid("s"), 99);
    expect(all.map((e) => e.content)).toEqual(["m3", "m2", "m1", "m0"]);
  });

  it("spans append ordering", async () => {
    const p = new InMemoryStorageProvider();
    await p.appendSpan(sid("s"), { n: 0 });
    await p.appendSpan(sid("s"), { n: 1 });
    expect(await p.getSpans(sid("s"))).toEqual([{ n: 0 }, { n: 1 }]);
  });
});

// ── FileSystem ───────────────────────────────────────────────────────────────

describe("FileSystemStorageProvider", () => {
  it("atomic write leaves no leftover .tmp and uses the canonical layout", async () => {
    const root = tmpDir();
    const p = new FileSystemStorageProvider(root);
    await p.putSession(sid("s"), paused("s"));
    await p.put(sid("s"), "k", { a: 1 });

    const leftovers: string[] = [];
    const walk = (dir: string): void => {
      for (const name of readdirSync(dir)) {
        const full = join(dir, name);
        if (statSync(full).isDirectory()) walk(full);
        else if (name.endsWith(".tmp")) leftovers.push(full);
      }
    };
    walk(root);
    expect(leftovers).toEqual([]);
    expect(existsSync(join(root, "sessions/s/state.json"))).toBe(true);
    expect(existsSync(join(root, "sessions/s/run/k.json"))).toBe(true);
  });

  it("session roundtrip, list, delete (missing-delete ok)", async () => {
    const root = tmpDir();
    const p = new FileSystemStorageProvider(root);
    await p.putSession(sid("a"), paused("a"));
    await p.putSession(sid("b"), paused("b"));
    const got = await p.getSession(sid("a"));
    expect(got?.turn_number).toBe(3);
    expect((await p.listSessions()).map((s) => s.asString())).toEqual(["a", "b"]);
    await p.deleteSession(sid("a"));
    expect(await p.getSession(sid("a"))).toBeUndefined();
    await expect(p.deleteSession(sid("missing"))).resolves.toBeUndefined();
  });

  it("run-store roundtrip, listKeys, delete (missing read undefined)", async () => {
    const root = tmpDir();
    const p = new FileSystemStorageProvider(root);
    const blob: JsonValue = { deep: [true, null, "x"] };
    await p.put(sid("s"), "plan", blob);
    await p.put(sid("s"), "tasks", 7);
    expect(await p.get(sid("s"), "plan")).toEqual(blob);
    expect(await p.listKeys(sid("s"))).toEqual(["plan", "tasks"]);
    await p.delete(sid("s"), "plan");
    expect(await p.get(sid("s"), "plan")).toBeUndefined();
    expect(await p.get(sid("missing"), "x")).toBeUndefined();
  });

  it("memory append recency + jsonl path + default metadata", async () => {
    const root = tmpDir();
    const p = new FileSystemStorageProvider(root);
    const contents = ["a", "b", "c"];
    for (const [i, content] of contents.entries()) {
      await p.appendMemory("project", sid("s"), mem("user", content, `t${i}`));
    }
    expect(existsSync(join(root, "sessions/s/memory.jsonl"))).toBe(true);
    const got = await p.getMemories("project", sid("s"), 2);
    expect(got.map((e) => e.content)).toEqual(["c", "b"]);
    expect(got[0]?.metadata).toEqual({});
  });

  it("spans append + flush marker", async () => {
    const root = tmpDir();
    const p = new FileSystemStorageProvider(root);
    await p.appendSpan(sid("s"), { n: 0 });
    await p.appendSpan(sid("s"), { n: 1 });
    expect(existsSync(join(root, "sessions/s/trace.jsonl"))).toBe(true);
    expect(await p.getSpans(sid("s"))).toEqual([{ n: 0 }, { n: 1 }]);
    await p.flushSession(sid("s"));
    expect(existsSync(join(root, "sessions/s/.flushed"))).toBe(true);
  });
});

// ── MemoryEntry default metadata ─────────────────────────────────────────────

describe("MemoryEntry", () => {
  it("newMemoryEntry defaults metadata to an empty object", () => {
    const e = newMemoryEntry("user", "hi", ts("2026-05-28T00:00:00Z"));
    expect(e.metadata).toEqual({});
    const v = JSON.parse(JSON.stringify(e));
    expect(v.role).toBe("user");
    expect(v.content).toBe("hi");
    expect(v.metadata).toEqual({});
    expect(v.timestamp).toBe("2026-05-28T00:00:00Z");
  });
});

// ── Fixture replay: run_store_values + memory_entries ────────────────────────

describe("fixture replay", () => {
  it("run_store_values roundtrips in-memory and on-disk", async () => {
    const raw = readFileSync(join(fixturesRoot, "run_store_values.json"), "utf8");
    const cases = JSON.parse(raw) as { key: string; value: JsonValue }[];
    const inMem = new InMemoryStorageProvider();
    const fs = new FileSystemStorageProvider(tmpDir());
    for (const c of cases) {
      await inMem.put(sid("s"), c.key, c.value);
      expect(await inMem.get(sid("s"), c.key), `in-memory ${c.key}`).toEqual(c.value);
      await fs.put(sid("s"), c.key, c.value);
      expect(await fs.get(sid("s"), c.key), `fs ${c.key}`).toEqual(c.value);
    }
  });

  it("memory_entries replays with newest-first recency", async () => {
    const raw = readFileSync(join(fixturesRoot, "memory_entries.jsonl"), "utf8");
    const entries = raw
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map(
        (l) =>
          JSON.parse(l) as {
            role: string;
            content: string;
            timestamp: string;
            metadata: JsonValue;
          },
      );
    expect(entries.length).toBeGreaterThanOrEqual(3);

    const p = new InMemoryStorageProvider();
    for (const e of entries) {
      await p.appendMemory("project", sid("s"), {
        role: e.role,
        content: e.content,
        timestamp: ts(e.timestamp),
        metadata: e.metadata,
      });
    }
    const got = await p.getMemories("project", sid("s"), 2);
    expect(got).toHaveLength(2);
    expect(got[0]?.content).toBe(entries[entries.length - 1]?.content);
    expect(got[1]?.content).toBe(entries[entries.length - 2]?.content);

    const all = await p.getMemories("project", sid("s"), 999);
    expect(all.map((e) => e.content)).toEqual(entries.map((e) => e.content).reverse());
  });
});

// ── Harness wiring: default storage is no-op ─────────────────────────────────

describe("HarnessBuilder.storage wiring", () => {
  function builder(): HarnessBuilder {
    return new HarnessBuilder(
      new MockAgent(AgentId.of("test")),
      new ScriptedToolRegistry(),
      new AllowAllSandbox(),
      new NoopContextManager(),
      new AlwaysContinuePolicy(),
    );
  }

  it("defaults to an all-no-op StorageProvider when .storage() is never set", async () => {
    const harness = builder().build();
    const storage = harness.storage();
    // No-op reads return undefined/[]; writes resolve.
    expect(await storage.session().getSession(sid("s"))).toBeUndefined();
    expect(await storage.memory().getMemories("project", sid("s"), 5)).toEqual([]);
    expect(await storage.run().get(sid("s"), "k")).toBeUndefined();
    expect(await storage.observability().getSpans(sid("s"))).toEqual([]);
    await expect(storage.run().put(sid("s"), "k", 1)).resolves.toBeUndefined();
  });

  it("carries an injected StorageProvider through to the harness", async () => {
    const backend = new InMemoryStorageProvider();
    const provider = StorageProvider.single(backend);
    const harness = builder().storage(provider).build();
    await harness.storage().run().put(sid("s"), "plan", { ok: true });
    expect(await harness.storage().run().get(sid("s"), "plan")).toEqual({ ok: true });
  });
});

// ════════════════════════════════════════════════════════════════════════════
// #78 — scope + workspace-partitioning extension
// ════════════════════════════════════════════════════════════════════════════

// ── R2: WorkspaceId derivation ───────────────────────────────────────────────

describe("WorkspaceId", () => {
  it("is deterministic and pure (same input → same id)", () => {
    const a = WorkspaceId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core");
    const b = WorkspaceId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core");
    expect(a.asString()).toBe(b.asString());
    // Form is `{sanitizedBasename}-{8hex}`.
    expect(a.asString().startsWith("spore-core-")).toBe(true);
    expect(a.asString().length).toBe("spore-core-".length + 8);
  });

  it("root path collapses to the literal basename 'root'", () => {
    const w = WorkspaceId.fromCanonicalPath("/");
    expect(w.asString().startsWith("root-")).toBe(true);
  });

  it("sanitizes special chars and collapses dashes", () => {
    const w = WorkspaceId.fromCanonicalPath("/Users/me/My Project (v2)!");
    expect(w.asString().startsWith("my-project-v2-")).toBe(true);
    expect(w.asString().includes("--")).toBe(false);
  });

  it("ignores a trailing slash (same id as the no-slash form)", () => {
    const a = WorkspaceId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core");
    const b = WorkspaceId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core/");
    expect(a.asString()).toBe(b.asString());
  });

  it("strips the Windows drive prefix and normalizes separators", () => {
    const w = WorkspaceId.fromCanonicalPath("C:\\Users\\dev\\spore-core");
    expect(w.asString().startsWith("spore-core-")).toBe(true);
    // Distinct from the posix path (drive stripped, but the rest differs).
    const posix = WorkspaceId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core");
    expect(w.asString()).not.toBe(posix.asString());
  });

  it("fixture replay: matches workspace_id_derivation.json exactly", () => {
    const raw = readFileSync(join(fixturesRoot, "workspace_id_derivation.json"), "utf8");
    const cases = JSON.parse(raw) as {
      description: string;
      canonical_path: string;
      expected_workspace_id: string;
    }[];
    expect(cases.length).toBeGreaterThanOrEqual(4);
    for (const c of cases) {
      expect(WorkspaceId.fromCanonicalPath(c.canonical_path).asString()).toBe(
        c.expected_workspace_id,
      );
    }
  });
});

// ── ProjectId derivation + namespace (#142) ──────────────────────────────────

describe("ProjectId — pure derivation (#142)", () => {
  it("delegates to the SAME algorithm as WorkspaceId (byte-identical id)", () => {
    const path = "/Users/sbeardsley/dev/spore-core";
    expect(ProjectId.fromCanonicalPath(path).asString()).toBe(
      WorkspaceId.fromCanonicalPath(path).asString(),
    );
  });

  it("is deterministic + pure (same input → same id, form {slug}-{8hex})", () => {
    const a = ProjectId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core");
    const b = ProjectId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core");
    expect(a.asString()).toBe(b.asString());
    expect(a.equals(b)).toBe(true);
    expect(a.asString().startsWith("spore-core-")).toBe(true);
    expect(a.asString().length).toBe("spore-core-".length + 8);
  });

  it("slug derivation: root → 'root', special chars sanitized + collapsed", () => {
    expect(ProjectId.fromCanonicalPath("/").asString().startsWith("root-")).toBe(true);
    const w = ProjectId.fromCanonicalPath("/Users/me/My Project (v2)!");
    expect(w.asString().startsWith("my-project-v2-")).toBe(true);
    expect(w.asString().includes("--")).toBe(false);
  });

  it("trailing slash is normalized away (same id as the no-slash form)", () => {
    expect(ProjectId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core/").asString()).toBe(
      ProjectId.fromCanonicalPath("/Users/sbeardsley/dev/spore-core").asString(),
    );
  });

  // The whole collision policy: /a/b vs /a_b are DISTINCT ids. The 8-hex SHA-256
  // suffix of the FULL canonical path resolves what a naive slashes→underscores
  // slug would collide. NO extra logic needed — the inherited suffix does it.
  it("`/a/b` and `/a_b` derive DISTINCT ids (8-hex suffix resolves the collision)", () => {
    const ab = ProjectId.fromCanonicalPath("/a/b").asString();
    const aUnderscoreB = ProjectId.fromCanonicalPath("/a_b").asString();
    expect(ab).not.toBe(aUnderscoreB);
    // Distinct basenames too: `b-…` vs `a-b-…`.
    expect(ab.startsWith("b-")).toBe(true);
    expect(aUnderscoreB.startsWith("a-b-")).toBe(true);
  });

  it("namespace() projects onto the SessionId axis (string IS the project id)", () => {
    const p = ProjectId.fromCanonicalPath("/proj/x");
    expect(p.namespace().asString()).toBe(p.asString());
  });

  it("toJSON / toString surface the derived id string", () => {
    const p = ProjectId.fromCanonicalPath("/proj/x");
    expect(p.toString()).toBe(p.asString());
    expect(JSON.parse(JSON.stringify(p))).toBe(p.asString());
  });
});

describe("ProjectId — FS-touching constructors canonicalize FIRST (#142)", () => {
  it("fromPath canonicalizes a relative path component (resolves `.`/`..`)", () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-pid-"));
    const child = join(dir, "child");
    mkdirSync(child);
    // A path with a `..` component canonicalizes to the same id as the clean path.
    const viaDotDot = ProjectId.fromPath(join(child, "..", "child"));
    const viaClean = ProjectId.fromPath(child);
    expect(viaDotDot.equals(viaClean)).toBe(true);
    // And it equals the pure derivation over the OS-canonicalized path.
    expect(viaClean.equals(ProjectId.fromCanonicalPath(realpathSync(child)))).toBe(true);
  });

  // Gated: symlink resolution is FS-behavior-dependent.
  it("fromPath resolves symlinks before deriving (symlink → same id as target)", () => {
    const dir = mkdtempSync(join(tmpdir(), "spore-pid-link-"));
    const target = join(dir, "target");
    mkdirSync(target);
    const link = join(dir, "link");
    try {
      symlinkSync(target, link);
    } catch {
      // Symlink creation may be unsupported (e.g. restricted Windows) — skip.
      return;
    }
    expect(ProjectId.fromPath(link).equals(ProjectId.fromPath(target))).toBe(true);
  });

  // Gated: only meaningful on a case-insensitive filesystem (default macOS).
  it("fromPath resolves macOS case-insensitivity to one canonical id", () => {
    if (platform() !== "darwin") return;
    const dir = mkdtempSync(join(tmpdir(), "spore-pid-Case-"));
    const sub = join(dir, "MixedCase");
    mkdirSync(sub);
    // On a case-insensitive FS, the lowercased path resolves to the same inode;
    // realpath returns ONE canonical casing, so both derive the SAME id.
    let lowerExists = false;
    try {
      lowerExists = realpathSync(join(dir, "mixedcase")) === realpathSync(sub);
    } catch {
      lowerExists = false;
    }
    if (!lowerExists) return; // case-sensitive volume — skip.
    expect(ProjectId.fromPath(join(dir, "mixedcase")).equals(ProjectId.fromPath(sub))).toBe(true);
  });

  it("fromPath throws ProjectIdError on a non-existent path (canonicalize failure)", () => {
    const missing = join(tmpdir(), `spore-pid-missing-${Date.now()}-${Math.random()}`);
    expect(() => ProjectId.fromPath(missing)).toThrow(ProjectIdError);
    try {
      ProjectId.fromPath(missing);
    } catch (e) {
      expect(e).toBeInstanceOf(ProjectIdError);
      if (e instanceof ProjectIdError) {
        expect(e.kind).toBe("canonicalize");
        expect(e.path).toBe(missing);
        expect(e.name).toBe("ProjectIdError");
      }
    }
  });

  it("fromCwd derives from the (canonicalized) process cwd", () => {
    expect(ProjectId.fromCwd().equals(ProjectId.fromPath(process.cwd()))).toBe(true);
  });
});

describe("ProjectId — fixture replay project_id_derivation.json (#142)", () => {
  it("matches every pinned case exactly (cross-language anchor)", () => {
    const raw = readFileSync(join(fixturesRoot, "project_id_derivation.json"), "utf8");
    const cases = JSON.parse(raw) as {
      canonical_path: string;
      description: string;
      expected_project_id: string;
    }[];
    expect(cases.length).toBeGreaterThanOrEqual(4);
    for (const c of cases) {
      expect(ProjectId.fromCanonicalPath(c.canonical_path).asString()).toBe(c.expected_project_id);
    }
  });
});

// ── Project-scoped durable survival (#142) ───────────────────────────────────

describe("project-scoped durable survival (#142)", () => {
  const KEY = "task_list";

  it("task_list visible across DIFFERENT sessions with the SAME project id", async () => {
    const provider = StorageProvider.single(new InMemoryStorageProvider());
    const project = ProjectId.fromCanonicalPath("/work/audit-repo");
    // Window 1 writes under the project namespace (its session id is irrelevant).
    const payload: JsonValue = { tasks: [{ id: 1 }], next_id: 2 };
    await provider.run().put(project.namespace(), KEY, payload);
    // Window 2 = a FRESH session id, SAME project id → reads window 1's value.
    const w2 = await provider.run().get(project.namespace(), KEY);
    expect(w2).toEqual(payload);
    // A DIFFERENT session id keyed read finds NOTHING — proving project-keying.
    expect(await provider.run().get(sid("sess-window-2"), KEY)).toBeUndefined();
  });

  it("survives a fresh provider over the SAME on-disk root (process restart)", async () => {
    const root = tmpDir();
    const project = ProjectId.fromCanonicalPath("/work/audit-repo");
    const payload: JsonValue = { tasks: [{ id: 1 }], next_id: 2 };
    // Process 1: write through a FileSystemStorageProvider.
    await new FileSystemStorageProvider(root).put(project.namespace(), KEY, payload);
    // Process 2: a BRAND-NEW provider over the same root re-reads the same bytes.
    const reopened = new FileSystemStorageProvider(root);
    expect(await reopened.get(project.namespace(), KEY)).toEqual(payload);
  });

  it("fixture replay: project_durable_survival.json (cross-window + cross-process)", async () => {
    const raw = readFileSync(join(fixturesRoot, "project_durable_survival.json"), "utf8");
    const fx = JSON.parse(raw) as {
      project_canonical_path: string;
      expected_project_id: string;
      run_key: string;
      window_1: { session_id: string; task_list: JsonValue };
      window_2: { expected_task_list: JsonValue };
      cross_process: { expected_task_list: JsonValue };
    };
    // The derived project id matches the fixture's pinned id.
    const project = ProjectId.fromCanonicalPath(fx.project_canonical_path);
    expect(project.asString()).toBe(fx.expected_project_id);

    const root = tmpDir();
    // Window 1: write the list under the project namespace via FS provider.
    await new FileSystemStorageProvider(root).put(
      project.namespace(),
      fx.run_key,
      fx.window_1.task_list,
    );
    // Window 2: a fresh session id (mirrored by a fresh provider here) reads it.
    const w2 = await new FileSystemStorageProvider(root).get(project.namespace(), fx.run_key);
    expect(w2).toEqual(fx.window_2.expected_task_list);
    // Cross-process: yet another provider over the same root reads the same bytes.
    const xp = await new FileSystemStorageProvider(root).get(project.namespace(), fx.run_key);
    expect(xp).toEqual(fx.cross_process.expected_task_list);
  });
});

// ── Active-run lifecycle (#142) ──────────────────────────────────────────────

describe("active-run lifecycle (#142)", () => {
  const project = ProjectId.fromCanonicalPath("/work/active-run-project");
  const t0 = Timestamp.of("2026-06-12T00:00:00Z");
  const t1 = Timestamp.of("2026-06-12T01:00:00Z");

  it("start: an absent slot mints a fresh active run (started_new)", async () => {
    const store = new InMemoryStorageProvider();
    expect(await loadActiveRun(store, project)).toBeUndefined();
    const decision = await startOrResumeActiveRun(store, project, "tag-A", t0);
    expect(decision).toBe("started_new");
    const slot = await loadActiveRun(store, project);
    expect(slot).toEqual({ run_tag: "tag-A", started_at: t0.asString(), status: "active" });
  });

  it("resume: the SAME tag over a live slot resumes (slot left intact)", async () => {
    const store = new InMemoryStorageProvider();
    await startOrResumeActiveRun(store, project, "tag-A", t0);
    // A second call under the SAME tag with a DIFFERENT timestamp resumes and
    // does NOT overwrite the original started_at — the slot is left intact.
    const decision = await startOrResumeActiveRun(store, project, "tag-A", t1);
    expect(decision).toBe("resumed");
    const slot = await loadActiveRun(store, project);
    expect(slot?.started_at).toBe(t0.asString());
  });

  it("start-new: a DIFFERENT tag over a live slot starts fresh", async () => {
    const store = new InMemoryStorageProvider();
    await startOrResumeActiveRun(store, project, "tag-A", t0);
    const decision = await startOrResumeActiveRun(store, project, "tag-B", t1);
    expect(decision).toBe("started_new");
    const slot = await loadActiveRun(store, project);
    expect(slot).toEqual({ run_tag: "tag-B", started_at: t1.asString(), status: "active" });
  });

  it("complete: flips the slot to completed; next start (same tag) is fresh", async () => {
    const store = new InMemoryStorageProvider();
    await startOrResumeActiveRun(store, project, "tag-A", t0);
    await completeActiveRun(store, project);
    expect((await loadActiveRun(store, project))?.status).toBe("completed");
    // A completed slot does NOT resume even under the same tag — it starts fresh.
    const decision = await startOrResumeActiveRun(store, project, "tag-A", t1);
    expect(decision).toBe("started_new");
    const slot = await loadActiveRun(store, project);
    expect(slot).toEqual({ run_tag: "tag-A", started_at: t1.asString(), status: "active" });
  });

  it("complete on an absent slot is a no-op (no error, no write)", async () => {
    const store = new InMemoryStorageProvider();
    await expect(completeActiveRun(store, project)).resolves.toBeUndefined();
    expect(await loadActiveRun(store, project)).toBeUndefined();
  });

  it("a malformed slot is treated as no live run (next start mints fresh)", async () => {
    const store = new InMemoryStorageProvider();
    // Write garbage under the reserved key.
    await store.put(project.namespace(), ACTIVE_RUN_KEY, { not: "an active run" });
    expect(await loadActiveRun(store, project)).toBeUndefined();
    expect(await startOrResumeActiveRun(store, project, "tag-A", t0)).toBe("started_new");
  });

  it("the active run is keyed by project, not session (cross-window survival)", async () => {
    const store = new InMemoryStorageProvider();
    await startOrResumeActiveRun(store, project, "tag-A", t0);
    // A different project namespace has NO slot.
    const other = ProjectId.fromCanonicalPath("/work/other-project");
    expect(await loadActiveRun(store, other)).toBeUndefined();
    // The reserved keys are stable strings.
    expect(ACTIVE_RUN_KEY).toBe("active_run");
    expect(RALPH_PROGRESS_KEY).toBe("ralph_progress");
  });
});

// ── No-op fallback is scope-aware ────────────────────────────────────────────

describe("NoOpStorageProvider scoped memory", () => {
  it("scoped reads return [] and scoped writes resolve", async () => {
    const p = new NoOpStorageProvider();
    expect(await p.getMemories("user", sid("s"), 10)).toEqual([]);
    await expect(p.appendMemory("user", sid("s"), mem("user", "hi", "t"))).resolves.toBeUndefined();
  });
});

// ── R5: scope isolation — User and Project land in different backends ─────────

describe("CompositeStorageProvider scoped memory", () => {
  it("isolates scoped writes per scope; scoped reads return only own-scope", async () => {
    const user = new InMemoryStorageProvider();
    const project = new InMemoryStorageProvider();
    const p = new CompositeStorageProvider()
      .memory("user", user)
      .memory("project", project)
      .build();

    await p.memory().appendMemory("user", sid("s"), mem("user", "U", "t1"));
    await p.memory().appendMemory("project", sid("s"), mem("user", "P", "t1"));

    // Each backend physically holds only its own scope's entry.
    const u = await user.getMemories("user", sid("s"), 10);
    expect(u.map((e) => e.content)).toEqual(["U"]);
    const pr = await project.getMemories("project", sid("s"), 10);
    expect(pr.map((e) => e.content)).toEqual(["P"]);

    // Scoped reads through the router return only own-scope entries.
    const ru = await p.memory().getMemories("user", sid("s"), 10);
    expect(ru.map((e) => e.content)).toEqual(["U"]);
    const rp = await p.memory().getMemories("project", sid("s"), 10);
    expect(rp.map((e) => e.content)).toEqual(["P"]);
  });

  // ── R8: scoped read newest-first recency (append 4, limit=2 → newest two) ──
  it("scoped read is newest-first with a recency limit", async () => {
    const p = new CompositeStorageProvider()
      .memory("project", new InMemoryStorageProvider())
      .build();
    const contents = ["m0", "m1", "m2", "m3"];
    for (const [i, content] of contents.entries()) {
      await p.memory().appendMemory("project", sid("s"), mem("user", content, `t${i}`));
    }
    const got = await p.memory().getMemories("project", sid("s"), 2);
    expect(got.map((e) => e.content)).toEqual(["m3", "m2"]);
  });

  // ── R7: unconfigured (memory, scope) → NoOp returns [] ────────────────────
  it("unconfigured (memory, scope) falls back to no-op", async () => {
    // Only User wired; Project + Local fall back to no-op.
    const p = new CompositeStorageProvider().memory("user", new InMemoryStorageProvider()).build();
    // Writes to an unconfigured scope silently no-op.
    await expect(
      p.memory().appendMemory("project", sid("s"), mem("user", "x", "t")),
    ).resolves.toBeUndefined();
    // Reads from an unconfigured scope return [].
    expect(await p.memory().getMemories("project", sid("s"), 10)).toEqual([]);
  });

  // ── R11: Local falls back to NoOp when not wired ──────────────────────────
  it("Local defaults to no-op when not wired", async () => {
    const p = new CompositeStorageProvider()
      .memory("user", new InMemoryStorageProvider())
      .memory("project", new InMemoryStorageProvider())
      .build();
    await p.memory().appendMemory("local", sid("s"), mem("user", "l", "t"));
    expect(await p.memory().getMemories("local", sid("s"), 10)).toEqual([]);
  });
});

// ── R6: merged read = User ∪ Project, newest-first by timestamp, no dedup ─────

describe("StorageProvider.getMemoriesMerged", () => {
  it("unions User ∪ Project, newest-first by timestamp, with NO dedup", async () => {
    const p = new CompositeStorageProvider()
      .memory("user", new InMemoryStorageProvider())
      .memory("project", new InMemoryStorageProvider())
      .build();

    // Identical-content "dup" in BOTH scopes (same timestamp) proves no dedup.
    await p.memory().appendMemory("user", sid("s"), mem("user", "u-old", "2026-05-01T00:00:00Z"));
    await p.memory().appendMemory("user", sid("s"), mem("user", "dup", "2026-05-03T00:00:00Z"));
    await p.memory().appendMemory("user", sid("s"), mem("user", "u-new", "2026-05-05T00:00:00Z"));
    await p.memory().appendMemory("project", sid("s"), mem("a", "p-old", "2026-05-02T00:00:00Z"));
    await p.memory().appendMemory("project", sid("s"), mem("a", "dup", "2026-05-03T00:00:00Z"));
    await p.memory().appendMemory("project", sid("s"), mem("a", "p-new", "2026-05-06T00:00:00Z"));

    const merged = await p.getMemoriesMerged(sid("s"), 10);
    const contents = merged.map((e) => e.content);
    expect(contents).toEqual(["p-new", "u-new", "dup", "dup", "p-old", "u-old"]);
    // No dedup: the identical-content "dup" entry is present twice.
    expect(contents.filter((c) => c === "dup").length).toBe(2);
  });

  it("fixture replay: matches memory_scoped_merge.json (Local excluded)", async () => {
    const raw = readFileSync(join(fixturesRoot, "memory_scoped_merge.json"), "utf8");
    const f = JSON.parse(raw) as {
      limit: number;
      user: { role: string; content: string; timestamp: string; metadata: JsonValue }[];
      project: { role: string; content: string; timestamp: string; metadata: JsonValue }[];
      local: { role: string; content: string; timestamp: string; metadata: JsonValue }[];
      expected_merged_contents: string[];
    };

    const p = new CompositeStorageProvider()
      .memory("user", new InMemoryStorageProvider())
      .memory("project", new InMemoryStorageProvider())
      .memory("local", new InMemoryStorageProvider())
      .build();

    for (const [scope, rows] of [
      ["user", f.user],
      ["project", f.project],
      ["local", f.local],
    ] as const) {
      for (const r of rows) {
        await p.memory().appendMemory(scope, sid("s"), {
          role: r.role,
          content: r.content,
          timestamp: ts(r.timestamp),
          metadata: r.metadata,
        });
      }
    }

    const merged = await p.getMemoriesMerged(sid("s"), f.limit);
    const contents = merged.map((e) => e.content);
    expect(contents).toEqual(f.expected_merged_contents);
    // Local scope entries are excluded from the merge.
    expect(contents.some((c) => c.includes("should-not-appear"))).toBe(false);
  });
});

// ── ScopedMemoryRouter directly ──────────────────────────────────────────────

describe("ScopedMemoryRouter", () => {
  it("routes per scope and falls back to no-op for unconfigured scopes", async () => {
    const user = new InMemoryStorageProvider();
    const router = new ScopedMemoryRouter(new Map([["user", user]]));
    await router.appendMemory("user", sid("s"), mem("user", "U", "t1"));
    await router.appendMemory("project", sid("s"), mem("user", "P", "t1"));
    expect((await router.getMemories("user", sid("s"), 10)).map((e) => e.content)).toEqual(["U"]);
    // Unconfigured project scope → no-op.
    expect(await router.getMemories("project", sid("s"), 10)).toEqual([]);
  });
});

// ── R9: ToolContext exposes memoryStore threaded by the registry ─────────────
// The `RealToolRegistry` bridge graduated into core's `tool-registry` module
// (#91); per-run schema/dispatch bridging is covered by
// `packages/core/tests/harness-catalogue-wiring.test.ts`, and the catalogue
// tools' end-to-end storage threading by
// `typescript/packages/tools/tests/tool-context-memory-seam.test.ts`.
