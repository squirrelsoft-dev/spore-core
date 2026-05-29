/**
 * TaskListTool tests (#71) + fixture replay against the shared fixtures in
 * `fixtures/tasklist/`. Mirrors `rust/crates/spore-core/src/tools/tasklist.rs`.
 */

import { mkdtemp, readFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  tasklist,
  type Operation,
  type SandboxProvider,
  type SandboxViolation,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import { TaskListTool } from "../src/index.js";

type TaskList = tasklist.TaskList;
type TaskStatus = tasklist.TaskStatus;

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = resolve(here, "../../../../fixtures/tasklist");

/** Sandbox that roots `.spore/task_list.json` inside a tempdir. */
class TempRootSandbox implements SandboxProvider {
  constructor(private readonly root: string) {}
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
  async resolvePath(
    path: string,
    _op: Operation,
  ): Promise<string | SandboxViolation> {
    return join(this.root, path);
  }
}

async function tmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), "spore-tasklist-tool-"));
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

describe("TaskListTool", () => {
  it("add then list persists and assigns ids", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const tool = new TaskListTool();

    const l1 = parseList(
      await tool.execute(call({ action: "add_task", description: "a" }), sb),
    );
    expect(l1.tasks).toHaveLength(1);
    expect(l1.tasks[0].id).toBe(1);
    expect(l1.next_id).toBe(2);

    const l2 = parseList(
      await tool.execute(call({ action: "add_task", description: "b" }), sb),
    );
    expect(l2.tasks.map((t) => t.id)).toEqual([1, 2]);

    expect(existsSync(join(dir, tasklist.TASK_LIST_PATH))).toBe(true);

    const l3 = parseList(
      await tool.execute(call({ action: "list_tasks" }), sb),
    );
    expect(l3).toEqual(l2);
  });

  it("update status then complete", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const tool = new TaskListTool();
    await tool.execute(call({ action: "add_task", description: "x" }), sb);

    const r = parseList(
      await tool.execute(
        call({ action: "update_task", id: 1, status: "in_progress" }),
        sb,
      ),
    );
    expect(r.tasks[0].status).toBe("in_progress");

    const c = parseList(
      await tool.execute(call({ action: "complete_task", id: 1 }), sb),
    );
    expect(c.tasks[0].status).toBe("completed");
  });

  it("unknown id is a recoverable error", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const out = await new TaskListTool().execute(
      call({ action: "complete_task", id: 42 }),
      sb,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("invalid transition out of completed is a recoverable error", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const tool = new TaskListTool();
    await tool.execute(call({ action: "add_task", description: "x" }), sb);
    await tool.execute(call({ action: "complete_task", id: 1 }), sb);
    const out = await tool.execute(
      call({ action: "update_task", id: 1, status: "pending" }),
      sb,
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("bad params is a recoverable error", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const out = await new TaskListTool().execute(call({ action: "nope" }), sb);
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("schema is not read_only / destructive / open_world", () => {
    const s = TaskListTool.schema();
    expect(s.annotations.read_only).toBe(false);
    expect(s.annotations.destructive).toBe(false);
    expect(s.annotations.open_world).toBe(false);
  });

  it("persist then reload yields identical list (off disk)", async () => {
    const dir = await tmp();
    const sb = new TempRootSandbox(dir);
    const tool = new TaskListTool();
    await tool.execute(call({ action: "add_task", description: "one" }), sb);
    const fromTool = parseList(
      await tool.execute(call({ action: "add_task", description: "two" }), sb),
    );
    const onDisk = JSON.parse(
      await readFile(join(dir, tasklist.TASK_LIST_PATH), "utf8"),
    ) as TaskList;
    expect(onDisk).toEqual(fromTool);
  });
});

// ============================================================================
// Fixture replay
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
      const dir = await tmp();
      const sb = new TempRootSandbox(dir);
      const tool = new TaskListTool();
      for (let i = 0; i < sc.steps.length; i++) {
        const step = sc.steps[i];
        const out = await tool.execute(call(step.action), sb);
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
