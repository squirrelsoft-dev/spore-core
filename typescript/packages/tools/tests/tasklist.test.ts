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

const { ProjectId } = storage;

/** Derive the project namespace the tool keys the durable task_list by, from a
 *  test's distinguishing `key` string (#142). Tests that want SEPARATE lists pass
 *  DIFFERENT keys; tests that want the SAME list pass the SAME key. */
function projectNs(key: string): SessionId {
  return ProjectId.fromCanonicalPath(`/proj/${key}`).namespace();
}

function ctxWith(
  runStore: RunStore,
  key = "test-project",
): toolRegistry.ToolContext {
  // The `sessionId` is ephemeral and irrelevant to the durable task_list; the
  // `projectId` derived from `key` is the durable namespace the tool keys by.
  return new ToolContext(
    SessionId.of("ephemeral-session"),
    ProjectId.fromCanonicalPath(`/proj/${key}`),
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

/**
 * The TaskList portion of a success content. On `add_task` the content carries
 * an extra leading `added` key (#143); strip it so this always yields a bare
 * TaskList — mirroring the Rust helper that deserializes into a struct ignoring
 * `added`. Use {@link parseAdded} / {@link successContent} to assert on `added`.
 */
function parseList(out: ToolOutput): TaskList {
  if (out.kind !== "success") {
    throw new Error(`expected success, got ${JSON.stringify(out)}`);
  }
  const { added: _added, ...list } = JSON.parse(out.content) as TaskList & {
    added?: number;
  };
  return list;
}

/** Raw success-content string, for #143 byte-level and `added`-key asserts. */
function successContent(out: ToolOutput): string {
  if (out.kind !== "success") {
    throw new Error(`expected success, got ${JSON.stringify(out)}`);
  }
  return out.content;
}

/** The `added` field of a success content, or `undefined` if absent (#143). */
function parseAdded(out: ToolOutput): number | undefined {
  const v = JSON.parse(successContent(out)) as { added?: number };
  return v.added;
}

/** Read the persisted blob straight off a RunStore as a TaskList, keyed by the
 *  project namespace derived from `key` (#142 — durable artifacts are keyed by
 *  project_id, not the ephemeral session id). */
async function loadFromStore(
  runStore: RunStore,
  key = "test-project",
): Promise<TaskList | undefined> {
  const v = await runStore.get(projectNs(key), TASK_LIST_KEY);
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

  // #142: keyed by PROJECT id, not session id. Two DIFFERENT project ids over the
  // SAME run store keep separate lists.
  it("lists are keyed by project_id", async () => {
    const runStore = new InMemoryStorageProvider();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    const ctxA = ctxWith(runStore, "project-a");
    const ctxB = ctxWith(runStore, "project-b");

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

    const a = await loadFromStore(runStore, "project-a");
    const b = await loadFromStore(runStore, "project-b");
    expect(a?.tasks).toHaveLength(1);
    expect(a?.tasks[0].description).toBe("a1");
    expect(b?.tasks).toHaveLength(2);
    expect(b?.tasks.map((t) => t.description)).toEqual(["b1", "b2"]);
  });

  // #142 (the bug this issue fixes): the task_list is visible across DIFFERENT
  // sessions with the SAME project id. This mirrors the Ralph window-reset path —
  // each window mints a fresh SessionId but the project id is stable, so window 2
  // must see window 1's list.
  it("task_list is visible across sessions with the same project_id", async () => {
    const runStore = new InMemoryStorageProvider();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();
    const project = ProjectId.fromCanonicalPath("/proj/shared");

    // Window 1: a distinct session id, but the shared project id.
    const ctxW1 = new ToolContext(
      SessionId.of("window-1-session"),
      project,
      runStore,
      new InMemoryStorageProvider(),
    );
    await tool.execute(
      call({ action: "add_task", description: "from window 1" }),
      sb,
      ctxW1,
    );

    // Window 2: a DIFFERENT (freshly generated) session id, SAME project id.
    const ctxW2 = new ToolContext(
      SessionId.generate(),
      project,
      runStore,
      new InMemoryStorageProvider(),
    );
    const listed = parseList(
      await tool.execute(call({ action: "list_tasks" }), sb, ctxW2),
    );
    expect(listed.tasks).toHaveLength(1);
    expect(listed.tasks[0].description).toBe("from window 1");
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

  // #118: add_task passes blockers through to the list and stores them.
  it("add_task passes blockers through", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();
    await tool.execute(call({ action: "add_task", description: "a" }), sb, ctx);
    const list = parseList(
      await tool.execute(
        call({ action: "add_task", description: "b", blockers: [1] }),
        sb,
        ctx,
      ),
    );
    expect(list.tasks[1].blockers).toEqual([1]);
  });

  // #118: omitting blockers defaults to empty (backward-compatible call).
  it("add_task without blockers defaults to empty", async () => {
    const ctx = inMemoryCtx();
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "a" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(parseList(out).tasks[0].blockers).toEqual([]);
  });

  // #118: a self-blocking add maps to a recoverable tool error.
  it("self-block is a recoverable error", async () => {
    const ctx = inMemoryCtx();
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "a", blockers: [1] }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid blockers");
    }
  });

  // #118: an unknown blocker id maps to a recoverable tool error.
  it("unknown blocker is a recoverable error", async () => {
    const ctx = inMemoryCtx();
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "a", blockers: [99] }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  // #118: schema advertises blockers (kept in sorted property order).
  it("schema advertises blockers", () => {
    const s = TaskListTool.schema();
    const params = s.parameters as { properties: Record<string, unknown> };
    const props = params.properties;
    expect(props.blockers).toEqual({
      type: "array",
      items: { type: "integer" },
    });
    // Properties kept in sorted order: action, blockers, description, id, status.
    expect(Object.keys(props)).toEqual([
      "action",
      "blockers",
      "description",
      "id",
      "status",
    ]);
  });

  // ==========================================================================
  // #143: add_task surfaces the assigned id as a leading `added` key.
  // ==========================================================================

  // R-143.1: add success content carries `added` == the id `addTask` returned,
  // R-143.2: which is the persisted task's id, and
  // R-143.3: the full list is still present in the content.
  it("add success content carries the assigned id", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    const out = await tool.execute(
      call({ action: "add_task", description: "a" }),
      sb,
      ctx,
    );

    // R-143.1: `added` is present and equals the first assigned id (1).
    expect(parseAdded(out)).toBe(1);
    // R-143.3: the full list is still present (and parses, ignoring `added`).
    const list = parseList(out);
    expect(list.tasks).toHaveLength(1);
    expect(list.next_id).toBe(2);
    // R-143.2: `added` == the persisted task's id.
    expect(parseAdded(out)).toBe(list.tasks[0].id);
    const persisted = await loadFromStore(ctx.runStore);
    expect(parseAdded(out)).toBe(persisted?.tasks[0].id);
  });

  // R-143.4: two adds → `added` is 1 then 2, with next_id 2 then 3.
  it("two adds surface sequential ids", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    const r1 = await tool.execute(
      call({ action: "add_task", description: "a" }),
      sb,
      ctx,
    );
    expect(parseAdded(r1)).toBe(1);
    expect(parseList(r1).next_id).toBe(2);

    const r2 = await tool.execute(
      call({ action: "add_task", description: "b" }),
      sb,
      ctx,
    );
    expect(parseAdded(r2)).toBe(2);
    expect(parseList(r2).next_id).toBe(3);
  });

  // R-143.5: `added` appears ONLY on the add_task branch — never on
  // update_task, complete_task, or list_tasks.
  it("added appears only on the add branch", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    await tool.execute(call({ action: "add_task", description: "a" }), sb, ctx);

    const upd = await tool.execute(
      call({ action: "update_task", id: 1, status: "in_progress" }),
      sb,
      ctx,
    );
    expect(parseAdded(upd)).toBeUndefined();

    const comp = await tool.execute(
      call({ action: "complete_task", id: 1 }),
      sb,
      ctx,
    );
    expect(parseAdded(comp)).toBeUndefined();

    const listed = await tool.execute(call({ action: "list_tasks" }), sb, ctx);
    expect(parseAdded(listed)).toBeUndefined();
  });

  // R-143.6: a rejected add (self-block) is a recoverable error with NO `added`
  // and no list.
  it("rejected add has no added and no list", async () => {
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "a", blockers: [1] }),
      new AllowAllSandbox(),
      inMemoryCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid blockers");
      // No success content at all → no `added`, no list.
      expect(out.message).not.toContain("added");
      expect(out.message).not.toContain("tasks");
    }
  });

  // R-143.7: the PERSISTED blob never carries `added` — only the tool's success
  // content does. The PlanExecute executor depends on this shape.
  it("persisted blob has no added", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "a" }),
      sb,
      ctx,
    );
    // Success content DOES carry `added`...
    expect(parseAdded(out)).toBe(1);

    // ...but the raw persisted blob does NOT carry an `added` key. #142: the blob
    // is keyed by the project namespace, not the session id — read it straight off
    // the ctx's project_id namespace.
    const raw = await ctx.runStore.get(ctx.projectId.namespace(), TASK_LIST_KEY);
    expect(raw).toBeDefined();
    expect((raw as Record<string, unknown>).added).toBeUndefined();
    // The persisted blob round-trips to a bare TaskList whose canonical
    // serialization is exactly `{"tasks":[...],"next_id":N}` — no `added`.
    const persisted = tasklist.parseTaskList(JSON.stringify(raw));
    expect(tasklist.serializeTaskList(persisted)).toBe(
      '{"tasks":[{"id":1,"description":"a","status":"pending","blockers":[]}],"next_id":2}',
    );
  });

  // #143 EXACT BYTES: a known add scenario pins the success content
  // byte-for-byte. This is the cross-language contract the other three
  // languages must match exactly.
  it("add success content exact bytes", async () => {
    const out = await new TaskListTool().execute(
      call({ action: "add_task", description: "a" }),
      new AllowAllSandbox(),
      inMemoryCtx(),
    );
    expect(successContent(out)).toBe(
      '{"added":1,"tasks":[{"id":1,"description":"a","status":"pending","blockers":[]}],"next_id":2}',
    );
  });

  // #143 usability: the returned id is directly usable. Add A, use A's returned
  // `added` id as a blocker for B, then complete A — proving the surfaced id
  // round-trips through blockers/complete without prediction.
  it("returned id is usable as a blocker and for complete", async () => {
    const ctx = inMemoryCtx();
    const sb = new AllowAllSandbox();
    const tool = new TaskListTool();

    const ra = await tool.execute(
      call({ action: "add_task", description: "A" }),
      sb,
      ctx,
    );
    const aId = parseAdded(ra);
    expect(aId).toBeDefined();

    // Use the surfaced id as a blocker for B (no prediction).
    const rb = await tool.execute(
      call({ action: "add_task", description: "B", blockers: [aId] }),
      sb,
      ctx,
    );
    expect(parseList(rb).tasks[1].blockers).toEqual([aId]);

    // Complete A by the surfaced id.
    const rc = await tool.execute(
      call({ action: "complete_task", id: aId }),
      sb,
      ctx,
    );
    const c = parseList(rc);
    const aTask = c.tasks.find((t) => t.id === aId);
    expect(aTask?.status).toBe("completed");
    // complete_task is not an add branch → no `added`.
    expect(parseAdded(rc)).toBeUndefined();
  });
});

// ============================================================================
// Fixture replay (driven over an in-memory RunStore, not the sandbox)
// ============================================================================

interface OpStep {
  action: unknown;
  expected: {
    ok: boolean;
    list?: TaskList;
    error?: string;
    // #143: present on successful add_task steps; the id `addTask` assigned and
    // the tool must surface as a leading `added` key in the success content.
    added?: number;
  };
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
        const isAdd =
          (step.action as { action?: unknown } | null)?.action === "add_task";
        const out = await tool.execute(call(step.action), sb, ctx);
        if (step.expected.ok) {
          expect(out.kind, `${sc.name} step ${i}`).toBe("success");
          // The list portion always survives — parseList strips the extra
          // `added` key on add steps and yields a bare TaskList.
          const got = parseList(out);
          expect(got, `${sc.name} step ${i}`).toEqual(step.expected.list);

          // #143: add steps carry a leading `added` key equal to the fixture's
          // `expected.added`; non-add steps must NOT carry one.
          const contentAdded = parseAdded(out);
          if (isAdd) {
            expect(
              contentAdded,
              `${sc.name} step ${i}: surfaced added id`,
            ).toBe(step.expected.added);
          } else {
            expect(
              contentAdded,
              `${sc.name} step ${i}: non-add step must not carry added`,
            ).toBeUndefined();
          }
        } else {
          expect(out.kind, `${sc.name} step ${i}`).toBe("error");
          if (out.kind !== "error") throw new Error("unreachable");
          expect(out.recoverable, `${sc.name} step ${i}`).toBe(true);
          const kind = out.message.includes("not found")
            ? "task_not_found"
            : out.message.includes("invalid transition")
              ? "invalid_transition"
              : out.message.includes("invalid blockers")
                ? "invalid_blockers"
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

interface DeserCase {
  name: string;
  json: string;
  expected: TaskList;
  reserialized: string;
}

// #118 backward-compat: a pre-#118 blob WITHOUT a blockers key deserializes
// (blockers default to []), and re-serializing emits the canonical form WITH
// blockers:[]. Replayed byte-identically across all four languages.
describe("fixture: tasklist deserialize (backward compat)", () => {
  const cases = JSON.parse(
    readFileSync(join(fixturesRoot, "deserialize.json"), "utf8"),
  ) as DeserCase[];

  it("loads at least one case", () => {
    expect(cases.length).toBeGreaterThan(0);
  });

  for (const c of cases) {
    it(c.name, () => {
      const parsed = tasklist.parseTaskList(c.json);
      expect(parsed).toEqual(c.expected);
      expect(parsed.tasks.every((t) => t.blockers.length === 0)).toBe(true);
      expect(tasklist.serializeTaskList(parsed)).toBe(c.reserialized);
    });
  }
});
