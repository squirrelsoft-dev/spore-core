/**
 * MemoryTool (#82) — rules R1–R10 plus fixture replays.
 *
 * Mirrors the Rust suite in `rust/crates/spore-core/src/tools/memory.rs`:
 * write→read roundtrip, metadata preservation, scope isolation, merged read
 * (driving `fixtures/storage/memory_scoped_merge.json`), Local rejection on both
 * ops (exact message + nothing written), session isolation, bad-params and
 * storage-error recoverability, read-does-not-write, schema-not-read_only, the
 * default limit of 50, and a step-by-step replay of `fixtures/tools/memory.json`
 * (honoring the per-step `unordered` flag).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  memory as coreMemory,
  SessionId,
  storage,
  toolRegistry,
  type SandboxProvider,
  type SandboxViolation,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import { MEMORY_LOCAL_REJECTED_MESSAGE, MemoryTool } from "../src/index.js";

type MemoryEntry = storage.MemoryEntry;
type MemoryStore = storage.MemoryStore;
type StorageScope = storage.StorageScope;

const { Timestamp } = coreMemory;
const { ToolContext } = toolRegistry;
const { InMemoryStorageProvider, CompositeStorageProvider, ProjectId } = storage;
const TEST_PROJECT = ProjectId.fromCanonicalPath("/test-project");

const here = dirname(fileURLToPath(import.meta.url));
const toolsFixtures = resolve(here, "../../../../fixtures/tools");
const storageFixtures = resolve(here, "../../../../fixtures/storage");

/** Permissive sandbox — the tool never touches the filesystem. */
class AllowAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
}

/** A MemoryStore that always fails — proves storage errors map to a recoverable error. */
class FailingMemoryStore implements MemoryStore {
  async appendMemory(
    _scope: StorageScope,
    _sessionId: SessionId,
    _entry: MemoryEntry,
  ): Promise<void> {
    throw new Error("boom");
  }
  async getMemories(
    _scope: StorageScope,
    _sessionId: SessionId,
    _limit: number,
  ): Promise<MemoryEntry[]> {
    throw new Error("boom");
  }
  async getMemoriesMerged(
    _sessionId: SessionId,
    _limit: number,
  ): Promise<MemoryEntry[]> {
    throw new Error("boom");
  }
}

function ctxWith(
  memoryStore: MemoryStore,
  session = "test-session",
): toolRegistry.ToolContext {
  // MemoryTool exercises the memory seam only; the run store is a no-op
  // in-memory backend here.
  return new ToolContext(
    SessionId.of(session),
    TEST_PROJECT,
    new InMemoryStorageProvider(),
    memoryStore,
  );
}

function inMemoryCtx(): toolRegistry.ToolContext {
  return ctxWith(new InMemoryStorageProvider());
}

function call(input: unknown): ToolCall {
  return { id: "c1", name: MemoryTool.NAME, input };
}

function parseEntries(out: ToolOutput): MemoryEntry[] {
  if (out.kind !== "success")
    throw new Error(`expected success, got ${JSON.stringify(out)}`);
  return JSON.parse(out.content) as MemoryEntry[];
}

function parseEntry(out: ToolOutput): MemoryEntry {
  if (out.kind !== "success")
    throw new Error(`expected success, got ${JSON.stringify(out)}`);
  return JSON.parse(out.content) as MemoryEntry;
}

const tool = new MemoryTool();
const sb = new AllowAllSandbox();

describe("MemoryTool", () => {
  // R1 + R2: write→read roundtrip; write returns the serialized entry.
  it("R1/R2: write then read roundtrip; write returns the entry", async () => {
    const ctx = inMemoryCtx();
    const w = await tool.execute(
      call({
        operation: "write",
        scope: "user",
        role: "user",
        content: "hello",
      }),
      sb,
      ctx,
    );
    const written = parseEntry(w);
    expect(written.role).toBe("user");
    expect(written.content).toBe("hello");
    expect(written.metadata).toEqual({}); // R4 default {}

    const r = await tool.execute(
      call({ operation: "read", scope: "user" }),
      sb,
      ctx,
    );
    const entries = parseEntries(r);
    expect(entries.length).toBe(1);
    expect(entries[0]!.content).toBe("hello");
  });

  // R4: metadata stored verbatim on the entry.
  it("R4: write preserves metadata", async () => {
    const ctx = inMemoryCtx();
    const w = await tool.execute(
      call({
        operation: "write",
        scope: "project",
        role: "assistant",
        content: "c",
        metadata: { k: "v", n: 3 },
      }),
      sb,
      ctx,
    );
    expect(parseEntry(w).metadata).toEqual({ k: "v", n: 3 });
  });

  // R5: non-merged scope isolation.
  it("R5: a scoped read does not see the other scope", async () => {
    const ctx = inMemoryCtx();
    await tool.execute(
      call({ operation: "write", scope: "user", role: "user", content: "u1" }),
      sb,
      ctx,
    );
    await tool.execute(
      call({
        operation: "write",
        scope: "project",
        role: "assistant",
        content: "p1",
      }),
      sb,
      ctx,
    );

    const user = parseEntries(
      await tool.execute(call({ operation: "read", scope: "user" }), sb, ctx),
    );
    expect(user.map((e) => e.content)).toEqual(["u1"]);

    const proj = parseEntries(
      await tool.execute(
        call({ operation: "read", scope: "project" }),
        sb,
        ctx,
      ),
    );
    expect(proj.map((e) => e.content)).toEqual(["p1"]);
  });

  // R6: merged read drives the shared merge fixture — both `dup` survive,
  // `local` absent, newest-first.
  it("R6: merged read replays memory_scoped_merge.json", async () => {
    const f = JSON.parse(
      readFileSync(
        resolve(storageFixtures, "memory_scoped_merge.json"),
        "utf8",
      ),
    ) as {
      limit: number;
      user: MemoryEntry[];
      project: MemoryEntry[];
      local: MemoryEntry[];
      expected_merged_contents: string[];
    };

    const provider = new CompositeStorageProvider()
      .memory("user", new InMemoryStorageProvider())
      .memory("project", new InMemoryStorageProvider())
      .memory("local", new InMemoryStorageProvider())
      .build();
    const memoryStore = provider.memory();
    const sid = SessionId.of("s");
    for (const [key, scope] of [
      ["user", "user"],
      ["project", "project"],
      ["local", "local"],
    ] as const) {
      for (const raw of f[key]) {
        await memoryStore.appendMemory(scope, sid, {
          role: raw.role,
          content: raw.content,
          timestamp: Timestamp.of(raw.timestamp as unknown as string),
          metadata: raw.metadata,
        });
      }
    }

    const ctx = new ToolContext(
      sid,
      TEST_PROJECT,
      new InMemoryStorageProvider(),
      memoryStore,
    );
    const out = await tool.execute(
      call({ operation: "read", scope: "user", merged: true, limit: f.limit }),
      sb,
      ctx,
    );
    const contents = parseEntries(out).map((e) => e.content);
    expect(contents).toEqual(f.expected_merged_contents);
    expect(contents.filter((c) => c === "dup").length).toBe(2);
    expect(contents.some((c) => c.includes("should-not-appear"))).toBe(false);
  });

  // merged read respects limit: 3 user entries, merged read limit=2.
  it("merged read respects limit (newest-first)", async () => {
    const provider = new CompositeStorageProvider()
      .memory("user", new InMemoryStorageProvider())
      .memory("project", new InMemoryStorageProvider())
      .build();
    const memoryStore = provider.memory();
    const sid = SessionId.of("s");
    const stamps = [
      "2026-05-01T00:00:00Z",
      "2026-05-02T00:00:00Z",
      "2026-05-03T00:00:00Z",
    ];
    for (let i = 0; i < stamps.length; i++) {
      await memoryStore.appendMemory("user", sid, {
        role: "user",
        content: `u${i}`,
        timestamp: Timestamp.of(stamps[i]!),
        metadata: {},
      });
    }
    const ctx = new ToolContext(
      sid,
      TEST_PROJECT,
      new InMemoryStorageProvider(),
      memoryStore,
    );
    const out = await tool.execute(
      call({ operation: "read", scope: "user", merged: true, limit: 2 }),
      sb,
      ctx,
    );
    const entries = parseEntries(out);
    expect(entries.map((e) => e.content)).toEqual(["u2", "u1"]);
  });

  // R7: Local rejected on write — exact message, nothing written.
  it("R7: local rejected on write writes nothing", async () => {
    const ctx = inMemoryCtx();
    const out = await tool.execute(
      call({ operation: "write", scope: "local", role: "user", content: "x" }),
      sb,
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toBe(MEMORY_LOCAL_REJECTED_MESSAGE);
    }
    // Nothing was written to ANY scope.
    for (const scope of ["user", "project", "local"] as const) {
      const got = await ctx.memoryStore.getMemories(scope, ctx.sessionId, 50);
      expect(got).toEqual([]);
    }
  });

  // R7: Local rejected on read — exact message.
  it("R7: local rejected on read", async () => {
    const ctx = inMemoryCtx();
    const out = await tool.execute(
      call({ operation: "read", scope: "local" }),
      sb,
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toBe(MEMORY_LOCAL_REJECTED_MESSAGE);
    }
  });

  // Session isolation: two sessions over the SAME store keep separate memory.
  it("memory is keyed by session id", async () => {
    const store = new InMemoryStorageProvider();
    const ctxA = ctxWith(store, "session-a");
    const ctxB = ctxWith(store, "session-b");

    await tool.execute(
      call({ operation: "write", scope: "user", role: "user", content: "a1" }),
      sb,
      ctxA,
    );
    await tool.execute(
      call({ operation: "write", scope: "user", role: "user", content: "b1" }),
      sb,
      ctxB,
    );

    const a = await store.getMemories("user", SessionId.of("session-a"), 50);
    const b = await store.getMemories("user", SessionId.of("session-b"), 50);
    expect(a.map((e) => e.content)).toEqual(["a1"]);
    expect(b.map((e) => e.content)).toEqual(["b1"]);
  });

  // R8: bad params → recoverable error.
  it("R8: bad params are a recoverable error", async () => {
    const ctx = inMemoryCtx();
    // Unknown operation.
    const r1 = await tool.execute(call({ operation: "nope" }), sb, ctx);
    expect(r1.kind).toBe("error");
    if (r1.kind === "error") expect(r1.recoverable).toBe(true);
    // Missing required field on write.
    const r2 = await tool.execute(
      call({ operation: "write", scope: "user", role: "user" }),
      sb,
      ctx,
    );
    expect(r2.kind).toBe("error");
    if (r2.kind === "error") expect(r2.recoverable).toBe(true);
  });

  // R9: storage failure → recoverable error (both write and read paths).
  it("R9: storage failure is a recoverable error", async () => {
    const ctx = ctxWith(new FailingMemoryStore());
    const w = await tool.execute(
      call({ operation: "write", scope: "user", role: "user", content: "x" }),
      sb,
      ctx,
    );
    expect(w.kind).toBe("error");
    if (w.kind === "error") expect(w.recoverable).toBe(true);

    const r = await tool.execute(
      call({ operation: "read", scope: "user" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind === "error") expect(r.recoverable).toBe(true);
  });

  // R10: read does not write — a read against a fresh store leaves it empty.
  it("R10: read does not write", async () => {
    const ctx = inMemoryCtx();
    const r = await tool.execute(
      call({ operation: "read", scope: "user" }),
      sb,
      ctx,
    );
    expect(parseEntries(r)).toEqual([]);
    const got = await ctx.memoryStore.getMemories("user", ctx.sessionId, 50);
    expect(got).toEqual([]);
  });

  // Schema is NOT read_only (decision E).
  it("schema is not read_only", () => {
    const s = MemoryTool.schema();
    expect(s.annotations.read_only).toBe(false);
    expect(s.annotations.destructive).toBe(false);
    expect(s.annotations.open_world).toBe(false);
    expect(s.name).toBe("memory");
    // Advertised scope enum omits `local`.
    const props = (
      s.parameters as { properties: { scope: { enum: string[] } } }
    ).properties;
    expect(props.scope.enum).toEqual(["project", "user"]);
  });

  // read default limit is 50 (decision B): more than 50 entries → 50 returned.
  it("R3: read default limit is 50", async () => {
    const ctx = inMemoryCtx();
    for (let i = 0; i < 60; i++) {
      await tool.execute(
        call({
          operation: "write",
          scope: "user",
          role: "user",
          content: `m${i}`,
        }),
        sb,
        ctx,
      );
    }
    const r = await tool.execute(
      call({ operation: "read", scope: "user" }),
      sb,
      ctx,
    );
    expect(parseEntries(r).length).toBe(50);
  });
});

// ============================================================================
// Fixture replay — fixtures/tools/memory.json
// ============================================================================

interface OpExpected {
  ok: boolean;
  contents?: string[];
  unordered?: boolean;
  error?: string;
}
interface OpStep {
  input: unknown;
  expected: OpExpected;
}
interface OpScenario {
  name: string;
  steps: OpStep[];
}

describe("MemoryTool fixture replay (memory.json)", () => {
  const scenarios = JSON.parse(
    readFileSync(resolve(toolsFixtures, "memory.json"), "utf8"),
  ) as OpScenario[];

  it("loads at least one scenario", () => {
    expect(scenarios.length).toBeGreaterThan(0);
  });

  for (const sc of scenarios) {
    it(`replays ${sc.name}`, async () => {
      // Fresh isolated scope-routing provider per scenario.
      const provider = new CompositeStorageProvider()
        .memory("user", new InMemoryStorageProvider())
        .memory("project", new InMemoryStorageProvider())
        .build();
      const ctx = new ToolContext(
        SessionId.of("fx"),
        TEST_PROJECT,
        new InMemoryStorageProvider(),
        provider.memory(),
      );

      for (let i = 0; i < sc.steps.length; i++) {
        const step = sc.steps[i]!;
        const out = await tool.execute(call(step.input), sb, ctx);
        if (step.expected.ok) {
          expect(out.kind, `${sc.name} step ${i}`).toBe("success");
          // A read asserts its content list; a write only asserts ok.
          if (step.expected.contents) {
            const got = parseEntries(out).map((e) => e.content);
            if (step.expected.unordered) {
              expect([...got].sort(), `${sc.name} step ${i}`).toEqual(
                [...step.expected.contents].sort(),
              );
            } else {
              expect(got, `${sc.name} step ${i}`).toEqual(
                step.expected.contents,
              );
            }
          }
        } else {
          expect(out.kind, `${sc.name} step ${i}`).toBe("error");
          if (out.kind === "error") {
            expect(
              out.recoverable,
              `${sc.name} step ${i}: errors must be recoverable`,
            ).toBe(true);
            expect(out.message, `${sc.name} step ${i}`).toBe(
              step.expected.error,
            );
          }
        }
      }
    });
  }
});
