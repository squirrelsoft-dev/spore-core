/**
 * PlanArtifact → TaskList parser tests (#72): the deterministic, total bridge
 * between the plan phase (#70) and the persisted task list (#71). Mirrors
 * `rust/crates/spore-core/src/tasklist.rs#tests` (the `plan_artifact_to_task_list`
 * group) and replays the shared fixture `fixtures/plan_to_tasklist/cases.json`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { tasklist, type PlanArtifact } from "../src/index.js";

const { defaultTaskList, serializeTaskList, planArtifactToTaskList } = tasklist;

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/plan_to_tasklist");

function artifact(tasks: string[], rationale = ""): PlanArtifact {
  return { tasks, rationale };
}

describe("planArtifactToTaskList (#72)", () => {
  it("produces one task per plan step, in plan order, all pending", () => {
    const list = planArtifactToTaskList(artifact(["first", "second", "third"]));
    expect(list.tasks.map((t) => t.description)).toEqual(["first", "second", "third"]);
    expect(list.tasks.every((t) => t.status === "pending")).toBe(true);
  });

  it("assigns sequential ids [1,2,3] and next_id 4", () => {
    const list = planArtifactToTaskList(artifact(["a", "b", "c"]));
    expect(list.tasks.map((t) => t.id)).toEqual([1, 2, 3]);
    expect(list.next_id).toBe(4);
  });

  it("is deterministic — same artifact parsed twice is deep-equal", () => {
    const a = artifact(["x", "y"], "why");
    expect(planArtifactToTaskList(a)).toEqual(planArtifactToTaskList(a));
  });

  it("copies descriptions verbatim — whitespace and empty strings preserved", () => {
    const list = planArtifactToTaskList(artifact(["  spaced  ", ""]));
    expect(list.tasks[0].description).toBe("  spaced  ");
    expect(list.tasks[1].description).toBe("");
    expect(list.tasks[0].id).toBe(1);
    expect(list.tasks[1].id).toBe(2);
  });

  it("maps an empty plan to the canonical default list", () => {
    const list = planArtifactToTaskList(artifact([]));
    expect(list).toEqual(defaultTaskList());
    expect(list.tasks).toHaveLength(0);
    expect(list.next_id).toBe(1);
    expect(serializeTaskList(list)).toBe('{"tasks":[],"next_id":1}');
  });

  it("drops the rationale — it appears nowhere in the result", () => {
    const list = planArtifactToTaskList(artifact(["do thing"], "SECRET_RATIONALE_TOKEN"));
    const json = serializeTaskList(list);
    expect(json).not.toContain("SECRET_RATIONALE_TOKEN");
    expect(json).not.toContain("rationale");
  });

  it("serializes the parsed result byte-identical / canonical", () => {
    const list = planArtifactToTaskList(artifact(["alpha", "beta"], "r"));
    expect(serializeTaskList(list)).toBe(
      '{"tasks":[{"id":1,"description":"alpha","status":"pending","blockers":[]},' +
        '{"id":2,"description":"beta","status":"pending","blockers":[]}],"next_id":3}',
    );
  });

  it("never merges into an existing list — always builds fresh", () => {
    const first = planArtifactToTaskList(artifact(["a", "b"]));
    const second = planArtifactToTaskList(artifact(["c"]));
    // The second parse starts from the empty default; it does not continue
    // the ids of the first list.
    expect(second.tasks.map((t) => t.id)).toEqual([1]);
    expect(second.next_id).toBe(2);
    // The first list is untouched.
    expect(first.tasks.map((t) => t.id)).toEqual([1, 2]);
  });
});

// ----------------------------------------------------------------------------
// Fixture replay (#72) — GROUND TRUTH at fixtures/plan_to_tasklist/cases.json.
// ----------------------------------------------------------------------------

interface PlanCase {
  name: string;
  input: PlanArtifact;
  expected: unknown;
}

describe("planArtifactToTaskList fixture replay", () => {
  const raw = readFileSync(resolve(fixtureDir, "cases.json"), "utf-8");
  const cases = JSON.parse(raw) as PlanCase[];

  it("has at least one case", () => {
    expect(cases.length).toBeGreaterThan(0);
  });

  for (const c of cases) {
    it(`case ${c.name}: serialized TaskList matches expected byte-for-byte`, () => {
      const got = planArtifactToTaskList(c.input);
      // Serialize both sides through the canonical serializer so the comparison
      // is on bytes, independent of the JSON key order in the fixture file.
      const want = serializeTaskList(tasklist.TaskListSchema.parse(c.expected));
      expect(serializeTaskList(got)).toBe(want);
    });
  }
});
