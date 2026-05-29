/**
 * TaskListTool tests (#71, storage seam #75) + fixture replay against the
 * shared fixtures in `fixtures/tasklist/`. Mirrors
 * `rust/crates/spore-core/src/tools/tasklist.rs`.
 *
 * The tool persists via the {@link ToolContext}'s {@link RunStore}, NOT the
 * sandbox filesystem. These tests drive it over an in-memory run store and prove
 * the storage rules: persists to the run store (succeeds under a denying
 * sandbox), keyed by SessionId, persist→reload identity, storage failure and a
 * corrupt blob both map to recoverable errors, and `list_tasks` never writes.
 */

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  SessionId,
  tasklist,
  toolRegistry,
  type Operation,
  type SandboxProvider,
  type SandboxViolation,
  type ToolCall,
  type ToolOutput,
  storage,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import { TaskListTool } from "../src/index.js";

type TaskList = tasklist.TaskList;
type TaskStatus = tasklist.TaskStatus;
type RunStore = storage.RunStore;
type JsonValue = storage.JsonValue;

const { ToolContext } = toolRegistry;
const { InMemoryStorageProvider } = storage;
const TASK_LIST_KEY = tasklist.TASK_LIST_EXTRAS_KEY;

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = resolve(here, "../../../../fixtures/tasklist");

/** Permissive sandbox — the tool no longer touches the filesystem. */
class AllowAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
}

/**
 * Sandbox whose `resolvePath` denies everything. Proves the tool persists to the
 * RunStore, not the sandbox: `add_task` still succeeds even though the sandbox
 * would reject any filesystem path.
 */
class DenyPathSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async resolvePath(
    path: string,
    _op: Operation,
  ): Promise<string | SandboxViolation> {
    return { kind: "path_escape", path };
  }
}

/** A RunStore that always fails — proves storage errors map to a recoverable tool error. */
class FailingRunStore implements RunStore {
  async get(
    _sessionId: SessionId,
    _key: string,
  ): Promise<JsonValue | undefined> {
    throw new Error("boom");
  }
  async put(
    _sessionId: SessionId,
    _key: string,
    _value: JsonValue,
  ): Promise<void> {
    throw new Error("boom");
  }
  async delete(_sessionId: SessionId, _key: string): Promise<void> {}
  async listKeys(_sessionId: SessionId): Promise<string[]> {
    return [];
  }
}

/** A RunStore whose stored blob for the task_list key is malformed for a TaskList. */
class CorruptRunStore implements RunStore {
  async get(
    _sessionId: SessionId,
    _key: string,
  ): Promise<JsonValue | undefined> {
    return { not: "a task list" };
  }
  async put(
    _sessionId: SessionId,
    _key: string,
    _value: JsonValue,
  ): Promise<void> {}
  async delete(_sessionId: SessionId, _key: string): Promise<void> {}
  async listKeys(_sessionId: SessionId): Promise<string[]> {
    return [];
  }
}

function ctxWith(
  runStore: RunStore,
  session = "test-session",
): toolRegistry.ToolContext {
  return new ToolContext(
    SessionId.of(session),
    runStore,
    new InMemoryStorageProvider(),
  );
}

function inMemoryCtx(): toolRegistry.ToolContext {
  return ctxWith(new InMemoryStorageProvider());
}

function call(input: unknown): ToolCall {
  return { id: "c1", name: TaskListTool.NAME, input };
}

function parseList(out: ToolOutput): TaskList {
  if (out.kind !== "success") {
    throw new Error(`expected success, got ${JSON.stringify(out)}`);
  }
  return JSON.parse(out.content) as TaskList;
}

/** Read the persisted blob straight off a RunStore as a TaskList. */
async function loadFromStore(
  runStore: RunStore,
  session = "test-session",
): Promise<TaskList | undefined> {
  const v = await runStore.get(SessionId.of(session), TASK_LIST_KEY);
  return v === undefined ? undefined : (v as unknown as TaskList);
}

describe("TaskListTool", () => {
  it("add then list persists and assigns ids", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    const l1 = parseList(
      await tool.execute(
        call({ action: "add_task", description: "a" }),
        sb,
        ctx,
      ),
    );
    expect(l1.tasks).toHaveLength(1);
    expect(l1.tasks[0].id).toBe(1);
    expect(l1.next_id).toBe(2);

    const l2 = parseList(
      await tool.execute(
        call({ action: "add_task", description: "b" }),
        sb,
        ctx,
      ),
    );
    expect(l2.tasks.map((t) => t.id)).toEqual([1, 2]);

    // The blob actually exists in the run store under the shared key.
    const persisted = await loadFromStore(ctx.runStore);
    expect(persisted).toEqual(l2);

    // list_tasks returns the same list and does not mutate.
    const l3 = parseList(
      await tool.execute(call({ action: "list_tasks" }), sb, ctx),
    );
    expect(l3).toEqual(l2);
  });

  // Storage seam: persists to the RunStore, NOT the sandbox. Even with a sandbox
  // that denies every path, add_task succeeds and persists.
  it("persists to the run store, not the sandbox", async () => {
    const ctx = inMemoryCtx();
    const sb = new DenyPathSandbox();
    const tool = new TaskListTool();

    const list = parseList(
      await tool.execute(
        call({ action: "add_task", description: "via run store" }),
        sb,
        ctx,
      ),
    );
    expect(list.tasks).toHaveLength(1);
    const persisted = await loadFromStore(ctx.runStore);
    expect(persisted).toEqual(list);
  });

  // Keyed by SessionId: two sessions over the SAME run store keep separate lists.
  it("lists are keyed by SessionId", async () => {
    const runStore = new InMemoryStorageProvider();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    const ctxA = ctxWith(runStore, "session-a");
    const ctxB = ctxWith(runStore, "session-b");

    await tool.execute(
      call({ action: "add_task", description: "a1" }),
      sb,
      ctxA,
    );
    await tool.execute(
      call({ action: "add_task", description: "b1" }),
      sb,
      ctxB,
    );
    await tool.execute(
      call({ action: "add_task", description: "b2" }),
      sb,
      ctxB,
    );

    const a = await loadFromStore(runStore, "session-a");
    const b = await loadFromStore(runStore, "session-b");
    expect(a?.tasks).toHaveLength(1);
    expect(a?.tasks[0].description).toBe("a1");
    expect(b?.tasks).toHaveLength(2);
    expect(b?.tasks.map((t) => t.description)).toEqual(["b1", "b2"]);
  });

  // Persist then reload with a FRESH tool over the SAME ctx yields identical list.
  it("persist then reload yields identical list", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();

    const tool1 = new TaskListTool();
    await tool1.execute(
      call({ action: "add_task", description: "one" }),
      sb,
      ctx,
    );
    const fromTool = parseList(
      await tool1.execute(
        call({ action: "add_task", description: "two" }),
        sb,
        ctx,
      ),
    );

    const tool2 = new TaskListTool();
    const reloaded = parseList(
      await tool2.execute(call({ action: "list_tasks" }), sb, ctx),
    );
    expect(reloaded).toEqual(fromTool);
  });

  it("update status then complete", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();
    await tool.execute(call({ action: "add_task", description: "x" }), sb, ctx);

    const r = parseList(
      await tool.execute(
        call({ action: "update_task", id: 1, status: "in_progress" }),
        sb,
        ctx,
      ),
    );
    expect(r.tasks[0].status).toBe("in_progress");

    const c = parseList(
      await tool.execute(call({ action: "complete_task", id: 1 }), sb, ctx),
    );
    expect(c.tasks[0].status).toBe("completed");
  });

  it("unknown id is a recoverable error", async () => {
    const ctx = inMemoryCtx();
    const out = await new TaskListTool().execute(
      call({ action: "complete_task", id: 42 }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("invalid transition out of completed is a recoverable error", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();
    await tool.execute(call({ action: "add_task", description: "x" }), sb, ctx);
    await tool.execute(call({ action: "complete_task", id: 1 }), sb, ctx);
    const out = await tool.execute(
      call({ action: "update_task", id: 1, status: "pending" }),
      sb,
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("bad params is a recoverable error", async () => {
    const out = await new TaskListTool().execute(
      call({ action: "nope" }),
      new AllowAllSandbox(),
      inMemoryCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  // Storage failure (get/put) → recoverable error.
  it("storage failure is a recoverable error", async () => {
    const ctx = ctxWith(new FailingRunStore());
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "x" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  // Malformed persisted blob → recoverable parse error.
  it("corrupt blob is a recoverable error", async () => {
    const ctx = ctxWith(new CorruptRunStore());
    const out = await new TaskListTool().execute(
      call({ action: "list_tasks" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  // list_tasks does not write: a fresh ctx with a never-written store stays empty.
  it("list_tasks does not write", async () => {
    const ctx = inMemoryCtx();
    const out = await new TaskListTool().execute(
      call({ action: "list_tasks" }),
      new AllowAllSandbox(),
      ctx,
    );
    // Returns the empty default list.
    expect(parseList(out)).toEqual(tasklist.defaultTaskList());
    // Nothing was persisted (list_tasks must not write).
    expect(await loadFromStore(ctx.runStore)).toBeUndefined();
  });

  it("schema is not read_only / destructive / open_world", () => {
    const s = TaskListTool.schema();
    expect(s.annotations.read_only).toBe(false);
    expect(s.annotations.destructive).toBe(false);
    expect(s.annotations.open_world).toBe(false);
  });
});

// ============================================================================
// Fixture replay (driven over an in-memory RunStore, not the sandbox)
// ============================================================================

interface OpStep {
  action: unknown;
  expected: { ok: boolean; list?: TaskList; error?: string };
}
interface OpScenario {
  name: string;
  steps: OpStep[];
}

describe("fixture: tasklist operations", () => {
  const data = JSON.parse(
    readFileSync(join(fixturesRoot, "operations.json"), "utf8"),
  ) as OpScenario[];

  it("loads at least one scenario", () => {
    expect(data.length).toBeGreaterThan(0);
  });

  for (const sc of data) {
    it(sc.name, async () => {
      // Fresh isolated run store per scenario.
      const ctx = inMemoryCtx();
      const sb = new AllowAllSandbox();
      const tool = new TaskListTool();
      for (let i = 0; i < sc.steps.length; i++) {
        const step = sc.steps[i];
        const out = await tool.execute(call(step.action), sb, ctx);
        if (step.expected.ok) {
          expect(out.kind, `${sc.name} step ${i}`).toBe("success");
          const got = parseList(out);
          expect(got, `${sc.name} step ${i}`).toEqual(step.expected.list);
        } else {
          expect(out.kind, `${sc.name} step ${i}`).toBe("error");
          if (out.kind !== "error") throw new Error("unreachable");
          expect(out.recoverable, `${sc.name} step ${i}`).toBe(true);
          const kind = out.message.includes("not found")
            ? "task_not_found"
            : out.message.includes("invalid transition")
              ? "invalid_transition"
              : "other";
          expect(kind, `${sc.name} step ${i}: ${out.message}`).toBe(
            step.expected.error,
          );
        }
      }
    });
  }
});

interface TransitionCase {
  from: TaskStatus;
  to: TaskStatus;
  expected: string;
}

describe("fixture: tasklist transitions", () => {
  const cases = JSON.parse(
    readFileSync(join(fixturesRoot, "transitions.json"), "utf8"),
  ) as TransitionCase[];

  it("loads at least one case", () => {
    expect(cases.length).toBeGreaterThan(0);
  });

  for (const c of cases) {
    it(`${c.from} -> ${c.to} = ${c.expected}`, () => {
      const got = tasklist.validateTransition(1, c.from, c.to).ok
        ? "ok"
        : "invalid_transition";
      expect(got).toBe(c.expected);
    });
  }
});

interface SerCase {
  name: string;
  list: TaskList;
  json: string;
}

describe("fixture: tasklist serialization", () => {
  const cases = JSON.parse(
    readFileSync(join(fixturesRoot, "serialization.json"), "utf8"),
  ) as SerCase[];

  it("loads at least one case", () => {
    expect(cases.length).toBeGreaterThan(0);
  });

  for (const c of cases) {
    it(c.name, () => {
      expect(tasklist.serializeTaskList(c.list)).toBe(c.json);
      expect(tasklist.parseTaskList(c.json)).toEqual(c.list);
    });
  }
});
