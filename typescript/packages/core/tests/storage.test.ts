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
import { mkdtempSync, existsSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
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
    task: newTask("do the thing", sid(session), { kind: "re_act", max_iterations: 1 }),
    budget_used: emptyBudgetSnapshot(),
    child_state: null,
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
    expect(await p.getMemories(sid("s"), 10)).toEqual([]);
    await expect(p.appendMemory(sid("s"), mem("user", "hi", "t"))).resolves.toBeUndefined();
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
    await p.memory().appendMemory(sid("s"), mem("user", "hi", "t1"));
    await p.run().put(sid("s"), "plan", { x: 1 });
    await p.observability().appendSpan(sid("s"), { kind: "turn" });

    expect(await p.session().getSession(sid("s"))).toBeDefined();
    expect(await p.memory().getMemories(sid("s"), 10)).toHaveLength(1);
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
    expect(await p.memory().getMemories(sid("s"), 5)).toEqual([]);
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
      await p.appendMemory(sid("s"), mem("user", content, `t${i}`));
    }
    const got = await p.getMemories(sid("s"), 2);
    expect(got.map((e) => e.content)).toEqual(["m3", "m2"]);
    const all = await p.getMemories(sid("s"), 99);
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
      await p.appendMemory(sid("s"), mem("user", content, `t${i}`));
    }
    expect(existsSync(join(root, "sessions/s/memory.jsonl"))).toBe(true);
    const got = await p.getMemories(sid("s"), 2);
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
      await p.appendMemory(sid("s"), {
        role: e.role,
        content: e.content,
        timestamp: ts(e.timestamp),
        metadata: e.metadata,
      });
    }
    const got = await p.getMemories(sid("s"), 2);
    expect(got).toHaveLength(2);
    expect(got[0]?.content).toBe(entries[entries.length - 1]?.content);
    expect(got[1]?.content).toBe(entries[entries.length - 2]?.content);

    const all = await p.getMemories(sid("s"), 999);
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
    expect(await storage.memory().getMemories(sid("s"), 5)).toEqual([]);
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
