/**
 * TaskList primitive tests (#71): types, transition matrix, mutation helpers,
 * and serialization byte-identity. Mirrors
 * `rust/crates/spore-core/src/tasklist.rs#tests`. Persistence moved to the
 * storage seam (#75) and is tested at the tool boundary in `@spore/tools`.
 */

import { describe, expect, it } from "vitest";

import { tasklist } from "../src/index.js";

const {
  defaultTaskList,
  serializeTaskList,
  parseTaskList,
  validateTransition,
  addTask,
  wouldCreateCycle,
  updateTask,
  completeTask,
} = tasklist;

type TaskStatus = tasklist.TaskStatus;
type TaskList = tasklist.TaskList;

/** Helper: addTask is fallible since #118; for the always-ok cases assert + return id. */
function add(l: TaskList, description: string, blockers: number[] = []): number {
  const r = addTask(l, description, blockers);
  expect(r.ok).toBe(true);
  if (!r.ok) throw new Error("unreachable");
  return r.id;
}

function listWith(statuses: TaskStatus[]): TaskList {
  const l = defaultTaskList();
  for (const _ of statuses) add(l, "t");
  statuses.forEach((s, i) => {
    l.tasks[i].status = s;
  });
  return l;
}

describe("tasklist helpers", () => {
  // R1
  it("assigns sequential ids from 1", () => {
    const l = defaultTaskList();
    expect(add(l, "a")).toBe(1);
    expect(add(l, "b")).toBe(2);
    expect(add(l, "c")).toBe(3);
    expect(l.next_id).toBe(4);
    expect(l.tasks.map((t) => t.id)).toEqual([1, 2, 3]);
  });

  // R2
  it("appends in order, new tasks pending", () => {
    const l = defaultTaskList();
    add(l, "first");
    add(l, "second");
    add(l, "third");
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
    add(l, "a");
    add(l, "b");
    const reloaded = parseTaskList(serializeTaskList(l));
    expect(reloaded.next_id).toBe(3);
    expect(add(reloaded, "c")).toBe(3);
  });

  it("serde round-trip is byte-identical", () => {
    const l = defaultTaskList();
    add(l, "alpha");
    add(l, "beta");
    updateTask(l, 2, "in_progress");
    const json1 = serializeTaskList(l);
    const parsed = parseTaskList(json1);
    const json2 = serializeTaskList(parsed);
    expect(json2).toBe(json1);
    expect(parsed).toEqual(l);
  });

  it("status snake_case spellings are exact", () => {
    const l = defaultTaskList();
    add(l, "x");
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
    add(l, "write tests");
    updateTask(l, 1, "in_progress");
    expect(serializeTaskList(l)).toBe(
      '{"tasks":[{"id":1,"description":"write tests","status":"in_progress","blockers":[]}],"next_id":2}',
    );
  });

  it("non-empty blockers serialize as the last field, byte-exact", () => {
    const l = defaultTaskList();
    add(l, "a");
    add(l, "b", [1]);
    expect(serializeTaskList(l)).toBe(
      '{"tasks":[{"id":1,"description":"a","status":"pending","blockers":[]},' +
        '{"id":2,"description":"b","status":"pending","blockers":[1]}],"next_id":3}',
    );
  });

  // Backward-compat: a pre-#118 blob WITHOUT a blockers key still parses, with
  // blockers defaulting to []. Re-serializing emits the canonical form with [].
  it("deserializes a pre-#118 blob without a blockers key", () => {
    const json = '{"tasks":[{"id":1,"description":"old","status":"pending"}],"next_id":2}';
    const l = parseTaskList(json);
    expect(l.tasks).toHaveLength(1);
    expect(l.tasks[0].blockers).toEqual([]);
    expect(serializeTaskList(l)).toBe(
      '{"tasks":[{"id":1,"description":"old","status":"pending","blockers":[]}],"next_id":2}',
    );
  });
});

// ============================================================================
// blockers (#118)
// ============================================================================

describe("tasklist blockers (#118)", () => {
  it("accepts and stores blockers referencing earlier real ids", () => {
    const l = defaultTaskList();
    expect(add(l, "a")).toBe(1);
    expect(add(l, "b")).toBe(2);
    expect(add(l, "c", [1, 2])).toBe(3);
    expect(l.tasks[2].blockers).toEqual([1, 2]);
    expect(l.next_id).toBe(4);
  });

  it("empty blockers never reject and store as an empty array", () => {
    const l = defaultTaskList();
    add(l, "a");
    expect(l.tasks[0].blockers).toEqual([]);
  });

  it("rejects a self-block (blocker == about-to-assign id)", () => {
    const l = defaultTaskList();
    const r = addTask(l, "a", [1]); // next_id is 1, blocker 1 == self
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error.kind).toBe("invalid_blockers");
      expect(r.error.detail).toEqual({
        kind: "invalid_blockers",
        id: 1,
        reason: { reason: "self_block" },
      });
    }
  });

  it("rejects an unknown blocker id", () => {
    const l = defaultTaskList();
    add(l, "a"); // id 1
    const r = addTask(l, "b", [99]);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error.kind).toBe("invalid_blockers");
      expect(r.error.detail).toEqual({
        kind: "invalid_blockers",
        id: 2,
        reason: { reason: "unknown_id", blocker: 99 },
      });
    }
  });

  // R9: a rejected add leaves the list completely untouched (mirrors update).
  it("a rejected add does not mutate the list", () => {
    const l = defaultTaskList();
    add(l, "a");
    const before = structuredClone(l);
    const r = addTask(l, "b", [99]);
    expect(r.ok).toBe(false);
    expect(l).toEqual(before);
    expect(l.next_id).toBe(2); // next_id did NOT advance
  });

  // self-block is checked before unknown-id per documented order.
  it("self-block takes precedence over unknown-id", () => {
    const l = defaultTaskList();
    const r = addTask(l, "a", [1, 99]); // next_id 1: self (1) and unknown (99)
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error.detail).toEqual({
        kind: "invalid_blockers",
        id: 1,
        reason: { reason: "self_block" },
      });
    }
  });

  it("rejects a cycle-introducing add", () => {
    const l = defaultTaskList();
    add(l, "a"); // id 1
    add(l, "b"); // id 2
    // Make task 1 depend on id 3 (the next id about to be assigned).
    l.tasks[0].blockers = [3];
    // Now add task 3 blocked by 1: path 3 -> 1 -> 3 is a cycle.
    const r = addTask(l, "c", [1]);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.error.detail).toEqual({
        kind: "invalid_blockers",
        id: 3,
        reason: { reason: "cycle" },
      });
    }
  });
});

// ============================================================================
// wouldCreateCycle helper (#118) — tested directly against hand-built graphs.
// ============================================================================

describe("wouldCreateCycle", () => {
  it("detects a back edge that closes a directed cycle", () => {
    // Edges (task -> blocker): 3 -> 2, 2 -> 1. From node 3 there is a path
    // 3 -> 2 -> 1 reaching node 1.
    const l = defaultTaskList();
    add(l, "a"); // 1
    add(l, "b"); // 2
    add(l, "c"); // 3
    l.tasks[2].blockers = [2]; // 3 -> 2
    l.tasks[1].blockers = [1]; // 2 -> 1
    // Re-adding node 1 with a blocker on 3 closes 1 -> 3 -> 2 -> 1.
    expect(wouldCreateCycle(l, 1, [3])).toBe(true);
    // Node 4 with blocker 3 has no path back to 4, so no cycle.
    expect(wouldCreateCycle(l, 4, [3])).toBe(false);
  });

  it("treats a direct self-edge as a cycle", () => {
    const l = defaultTaskList();
    expect(wouldCreateCycle(l, 5, [5])).toBe(true);
  });

  it("empty new edges are never a cycle", () => {
    const l = defaultTaskList();
    expect(wouldCreateCycle(l, 1, [])).toBe(false);
  });
});
