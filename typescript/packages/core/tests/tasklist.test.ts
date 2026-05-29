/**
 * TaskList primitive tests (#71): types, transition matrix, mutation helpers,
 * serialization byte-identity, and disk persistence. Mirrors
 * `rust/crates/spore-core/src/tasklist.rs#tests`.
 */

import { mkdtemp } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  tasklist,
  type Operation,
  type SandboxProvider,
  type SandboxViolation,
  type ToolCall,
} from "../src/index.js";

const {
  TASK_LIST_PATH,
  defaultTaskList,
  serializeTaskList,
  parseTaskList,
  validateTransition,
  addTask,
  updateTask,
  completeTask,
  loadTaskList,
  storeTaskList,
} = tasklist;

type TaskStatus = tasklist.TaskStatus;
type TaskList = tasklist.TaskList;

/** Sandbox that roots `.spore/task_list.json` inside a tempdir. */
class TempRootSandbox implements SandboxProvider {
  constructor(private readonly root: string) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async resolvePath(path: string, _op: Operation): Promise<string | SandboxViolation> {
    return join(this.root, path);
  }
}

async function tmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), "spore-tasklist-"));
}

function listWith(statuses: TaskStatus[]): TaskList {
  const l = defaultTaskList();
  for (const _ of statuses) addTask(l, "t");
  statuses.forEach((s, i) => {
    l.tasks[i].status = s;
  });
  return l;
}

describe("tasklist helpers", () => {
  // R1
  it("assigns sequential ids from 1", () => {
    const l = defaultTaskList();
    expect(addTask(l, "a")).toBe(1);
    expect(addTask(l, "b")).toBe(2);
    expect(addTask(l, "c")).toBe(3);
    expect(l.next_id).toBe(4);
    expect(l.tasks.map((t) => t.id)).toEqual([1, 2, 3]);
  });

  // R2
  it("appends in order, new tasks pending", () => {
    const l = defaultTaskList();
    addTask(l, "first");
    addTask(l, "second");
    addTask(l, "third");
    expect(l.tasks.map((t) => t.description)).toEqual(["first", "second", "third"]);
    expect(l.tasks.every((t) => t.status === "pending")).toBe(true);
  });

  // R3 — serializing/listing does not mutate.
  it("serialize does not mutate", () => {
    const l = listWith(["pending", "in_progress"]);
    const before = structuredClone(l);
    serializeTaskList(l);
    expect(l).toEqual(before);
  });

  it("update applies a valid status transition", () => {
    const l = listWith(["pending"]);
    expect(updateTask(l, 1, "in_progress").ok).toBe(true);
    expect(l.tasks[0].status).toBe("in_progress");
  });

  it("update sets description only", () => {
    const l = listWith(["pending"]);
    expect(updateTask(l, 1, undefined, "rewritten").ok).toBe(true);
    expect(l.tasks[0].description).toBe("rewritten");
    expect(l.tasks[0].status).toBe("pending");
  });

  it("update sets status and description at once", () => {
    const l = listWith(["pending"]);
    expect(updateTask(l, 1, "blocked", "blocked on x").ok).toBe(true);
    expect(l.tasks[0].status).toBe("blocked");
    expect(l.tasks[0].description).toBe("blocked on x");
  });

  it("update with neither field is a no-op success", () => {
    const l = listWith(["in_progress"]);
    const before = structuredClone(l);
    expect(updateTask(l, 1).ok).toBe(true);
    expect(l).toEqual(before);
  });

  it("complete marks completed", () => {
    const l = listWith(["in_progress"]);
    expect(completeTask(l, 1).ok).toBe(true);
    expect(l.tasks[0].status).toBe("completed");
  });

  // R4
  it("unknown id is task_not_found on update and complete", () => {
    const l = listWith(["pending"]);
    const u = updateTask(l, 99, "completed");
    expect(u.ok).toBe(false);
    if (!u.ok) expect(u.error.kind).toBe("task_not_found");
    const c = completeTask(l, 99);
    expect(c.ok).toBe(false);
    if (!c.ok) expect(c.error.kind).toBe("task_not_found");
  });

  // R5/R6 — rejected transition leaves the task untouched.
  it("rejected transition does not mutate", () => {
    const l = listWith(["completed"]);
    const before = structuredClone(l);
    const r = updateTask(l, 1, "in_progress");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.kind).toBe("invalid_transition");
    expect(l).toEqual(before);
  });

  it("pending -> completed is allowed via complete", () => {
    const l = listWith(["pending"]);
    expect(completeTask(l, 1).ok).toBe(true);
    expect(l.tasks[0].status).toBe("completed");
  });

  it("blocked -> in_progress and blocked -> completed allowed", () => {
    const l = listWith(["blocked", "blocked"]);
    expect(updateTask(l, 1, "in_progress").ok).toBe(true);
    expect(completeTask(l, 2).ok).toBe(true);
    expect(l.tasks[0].status).toBe("in_progress");
    expect(l.tasks[1].status).toBe("completed");
  });

  // R7
  it("idempotent self-transition succeeds and is a no-op", () => {
    const l = listWith(["completed"]);
    expect(updateTask(l, 1, "completed").ok).toBe(true);
    expect(completeTask(l, 1).ok).toBe(true);
    expect(l.tasks[0].status).toBe("completed");
  });
});

describe("tasklist transition matrix (DECISION 1)", () => {
  const allowed: Array<[TaskStatus, TaskStatus]> = [
    ["pending", "in_progress"],
    ["pending", "completed"],
    ["pending", "blocked"],
    ["in_progress", "completed"],
    ["in_progress", "blocked"],
    ["blocked", "in_progress"],
    ["blocked", "completed"],
    ["pending", "pending"],
    ["in_progress", "in_progress"],
    ["completed", "completed"],
    ["blocked", "blocked"],
  ];
  for (const [from, to] of allowed) {
    it(`allows ${from} -> ${to}`, () => {
      expect(validateTransition(1, from, to).ok).toBe(true);
    });
  }

  // R6 — every transition out of completed (except self) rejected.
  for (const to of ["pending", "in_progress", "blocked"] as TaskStatus[]) {
    it(`rejects completed -> ${to}`, () => {
      const r = validateTransition(7, "completed", to);
      expect(r.ok).toBe(false);
      if (!r.ok) {
        expect(r.error.kind).toBe("invalid_transition");
        expect(r.error.detail).toEqual({
          kind: "invalid_transition",
          id: 7,
          from: "completed",
          to,
        });
      }
    });
  }

  it("rejects in_progress -> pending and blocked -> pending", () => {
    expect(validateTransition(1, "in_progress", "pending").ok).toBe(false);
    expect(validateTransition(1, "blocked", "pending").ok).toBe(false);
  });
});

describe("tasklist serialization", () => {
  it("reload preserves next_id (ids never reused)", () => {
    const l = defaultTaskList();
    addTask(l, "a");
    addTask(l, "b");
    const reloaded = parseTaskList(serializeTaskList(l));
    expect(reloaded.next_id).toBe(3);
    expect(addTask(reloaded, "c")).toBe(3);
  });

  it("serde round-trip is byte-identical", () => {
    const l = defaultTaskList();
    addTask(l, "alpha");
    addTask(l, "beta");
    updateTask(l, 2, "in_progress");
    const json1 = serializeTaskList(l);
    const parsed = parseTaskList(json1);
    const json2 = serializeTaskList(parsed);
    expect(json2).toBe(json1);
    expect(parsed).toEqual(l);
  });

  it("status snake_case spellings are exact", () => {
    const l = defaultTaskList();
    addTask(l, "x");
    updateTask(l, 1, "in_progress");
    expect(serializeTaskList(l)).toContain('"status":"in_progress"');
    for (const s of ["pending", "completed", "blocked"] as TaskStatus[]) {
      const t = listWith([s]);
      expect(serializeTaskList(t)).toContain(`"status":"${s}"`);
    }
  });

  it("default serializes canonically", () => {
    expect(serializeTaskList(defaultTaskList())).toBe('{"tasks":[],"next_id":1}');
  });

  it("populated serializes canonically", () => {
    const l = defaultTaskList();
    addTask(l, "write tests");
    updateTask(l, 1, "in_progress");
    expect(serializeTaskList(l)).toBe(
      '{"tasks":[{"id":1,"description":"write tests","status":"in_progress"}],"next_id":2}',
    );
  });
});

describe("tasklist persistence", () => {
  it("load on absent file yields default", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const r = await loadTaskList(sb);
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.list).toEqual(defaultTaskList());
  });

  it("persist then reload yields an identical list", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const l = defaultTaskList();
    addTask(l, "one");
    addTask(l, "two");
    updateTask(l, 1, "in_progress");
    const stored = await storeTaskList(l, sb);
    expect(stored.ok).toBe(true);
    expect(existsSync(join(dir, TASK_LIST_PATH))).toBe(true);
    const reloaded = await loadTaskList(sb);
    expect(reloaded.ok).toBe(true);
    if (reloaded.ok) expect(reloaded.list).toEqual(l);
  });

  it("load on malformed file is a recoverable parse error", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const { promises: fs } = await import("node:fs");
    await fs.mkdir(join(dir, ".spore"), { recursive: true });
    await fs.writeFile(join(dir, TASK_LIST_PATH), "{ not json", "utf8");
    const r = await loadTaskList(sb);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error.kind).toBe("parse");
  });
});
