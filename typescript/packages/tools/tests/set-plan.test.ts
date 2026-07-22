/**
 * SetPlanTool tests — mirror `rust/crates/spore-core/src/tools/set_plan.rs`.
 */

import {
  SessionId,
  storage,
  tasklist,
  toolRegistry,
  type SandboxProvider,
  type SandboxViolation,
  type ToolCall,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import { SetPlanTool } from "../src/index.js";

const { ToolContext } = toolRegistry;
const { InMemoryStorageProvider, ProjectId } = storage;
const TASK_LIST_KEY = tasklist.TASK_LIST_EXTRAS_KEY;

class AllowAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
}

function ctxWith(runStore: storage.RunStore, key = "set-plan-project"): toolRegistry.ToolContext {
  return new ToolContext(
    SessionId.of("ephemeral"),
    ProjectId.fromCanonicalPath(`/proj/${key}`),
    runStore,
    new InMemoryStorageProvider(),
  );
}

function call(input: unknown): ToolCall {
  return { id: "c1", name: "set_plan", input };
}

describe("SetPlanTool", () => {
  it("schema requires a tasks array", () => {
    const s = SetPlanTool.schema();
    expect(s.name).toBe("set_plan");
    expect((s.parameters as { required: string[] }).required).toContain("tasks");
    expect(((s.parameters as { properties: { tasks: { type: string } } }).properties).tasks.type).toBe("array");
  });

  it("records a linear plan into the durable task_list store and returns success", async () => {
    const runStore = new InMemoryStorageProvider();
    const ctx = ctxWith(runStore);
    const sb = new AllowAllSandbox();
    const r = await new SetPlanTool().execute(
      call({ tasks: ["step one", "step two", "step three"] }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("Recorded a 3-step plan.");

    const ns = ctx.projectId.namespace();
    const raw = await runStore.get(ns, TASK_LIST_KEY);
    expect(raw).toBeDefined();
    const list = tasklist.parseTaskList(JSON.stringify(raw));
    expect(list.tasks.map((t) => t.description)).toEqual([
      "step one",
      "step two",
      "step three",
    ]);
    expect(list.tasks.every((t) => t.status === "pending" && t.blockers.length === 0)).toBe(true);
    expect(list.next_id).toBe(4);
  });

  it("rejects an empty tasks array (recoverable)", async () => {
    const ctx = ctxWith(new InMemoryStorageProvider());
    const r = await new SetPlanTool().execute(call({ tasks: [] }), new AllowAllSandbox(), ctx);
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("rejects a non-string element (recoverable)", async () => {
    const ctx = ctxWith(new InMemoryStorageProvider());
    const r = await new SetPlanTool().execute(
      call({ tasks: ["ok", 5, "bad"] }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
    expect(r.message).toMatch(/task 1/);
  });

  it("rejects a blank-string element (recoverable)", async () => {
    const ctx = ctxWith(new InMemoryStorageProvider());
    const r = await new SetPlanTool().execute(
      call({ tasks: ["   "] }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("error");
  });

  it("accepts an optional rationale (ignored on the wire, persisted via tasks)", async () => {
    const runStore = new InMemoryStorageProvider();
    const ctx = ctxWith(runStore);
    const r = await new SetPlanTool().execute(
      call({ tasks: ["only"], rationale: "because" }),
      new AllowAllSandbox(),
      ctx,
    );
    expect(r.kind).toBe("success");
  });

  it("is not read_only (mutates shared persisted state)", () => {
    expect(SetPlanTool.schema().annotations.read_only).toBe(false);
  });
});