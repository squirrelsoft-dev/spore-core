/**
 * Unit tests for the canonical ToolRegistry (spore-core issue #4).
 *
 * Mirrors `rust/crates/spore-core/src/tool_registry.rs#tests` — same rules,
 * same verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";

import { toolRegistry } from "../src/index.js";
import type { ToolCall } from "../src/index.js";

const {
  StandardToolRegistry,
  toolRegistryMock: {
    EchoTool,
    FailingTool,
    SubagentMockTool,
    AllowAllSandbox,
    DenyAllSandbox,
    testCtx,
  },
} = toolRegistry;
type ToolAnnotations = toolRegistry.ToolAnnotations;
type ToolSchema = toolRegistry.ToolSchema;

function annotations(partial: Partial<ToolAnnotations> = {}): ToolAnnotations {
  return {
    read_only: false,
    destructive: false,
    idempotent: false,
    open_world: false,
    ...partial,
  };
}

function schema(name: string, anno: Partial<ToolAnnotations> = {}): ToolSchema {
  return {
    name,
    description: `${name} tool`,
    parameters: { type: "object", properties: {} },
    annotations: annotations(anno),
  };
}

function schemaWithRequired(name: string, required: string[]): ToolSchema {
  return {
    name,
    description: name,
    parameters: { type: "object", properties: {}, required },
    annotations: annotations({ read_only: true }),
  };
}

function call(name: string, id: string, input: unknown): ToolCall {
  return { id, name, input };
}

describe("StandardToolRegistry", () => {
  // Rule 1: tools dispatched via registry.
  it("dispatches a registered tool through the registry", async () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("echo"), schema("echo", { read_only: true }))).toBeNull();
    const out = await reg.dispatch(call("echo", "c1", { x: 1 }), new AllowAllSandbox(), testCtx());
    expect(out.ok).toBe(true);
    if (!out.ok) throw new Error("expected ok");
    expect(out.result.call_id).toBe("c1");
    expect(out.result.output.kind).toBe("success");
    if (out.result.output.kind === "success") {
      expect(out.result.output.content).toBe('{"x":1}');
    }
  });

  // Rule 3: duplicate registration fails.
  it("rejects duplicate registration", () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("echo"), schema("echo"))).toBeNull();
    const err = reg.register(new EchoTool("echo"), schema("echo"));
    expect(err?.kind).toBe("DuplicateName");
  });

  // Rule 2: schema validated at registration (missing top-level type).
  it("rejects invalid schemas at registration", () => {
    const reg = new StandardToolRegistry();
    const bad: ToolSchema = {
      name: "x",
      description: "x",
      parameters: { properties: {} },
      annotations: annotations(),
    };
    const err = reg.register(new EchoTool("x"), bad);
    expect(err?.kind).toBe("InvalidSchema");
  });

  // Rule 2 (empty name).
  it("rejects empty tool name at registration", () => {
    const reg = new StandardToolRegistry();
    const bad: ToolSchema = {
      name: "",
      description: "",
      parameters: { type: "object" },
      annotations: annotations(),
    };
    const err = reg.register(new EchoTool(""), bad);
    expect(err?.kind).toBe("InvalidSchema");
  });

  // Rule 4: conflicting annotations rejected.
  it("rejects read_only + destructive as conflicting", () => {
    const reg = new StandardToolRegistry();
    const err = reg.register(
      new EchoTool("rm"),
      schema("rm", { read_only: true, destructive: true }),
    );
    expect(err?.kind).toBe("ConflictingAnnotations");
  });

  // Tool/schema name mismatch.
  it("rejects when tool name does not match schema name", () => {
    const reg = new StandardToolRegistry();
    const err = reg.register(new EchoTool("a"), schema("b"));
    expect(err?.kind).toBe("InvalidSchema");
  });

  // Rule 6: dispatching unknown tool errors.
  it("errors when dispatching an unregistered tool", async () => {
    const reg = new StandardToolRegistry();
    const out = await reg.dispatch(call("missing", "c1", {}), new AllowAllSandbox(), testCtx());
    expect(out.ok).toBe(false);
    if (out.ok) throw new Error("expected err");
    expect(out.error.kind).toBe("UnregisteredTool");
  });

  // Rule 7: schema validation failure on missing required field.
  it("errors on missing required field", async () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("read"), schemaWithRequired("read", ["path"]))).toBeNull();
    const out = await reg.dispatch(call("read", "c1", {}), new AllowAllSandbox(), testCtx());
    expect(out.ok).toBe(false);
    if (out.ok) throw new Error("expected err");
    expect(out.error.kind).toBe("SchemaValidationFailed");
    if (out.error.kind === "SchemaValidationFailed") {
      expect(out.error.tool).toBe("read");
      expect(out.error.reason).toContain("path");
    }
  });

  // Sandbox violation surfaces as DispatchError.
  it("surfaces sandbox violations as DispatchError", async () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("echo"), schema("echo", { read_only: true }))).toBeNull();
    const out = await reg.dispatch(call("echo", "c1", {}), new DenyAllSandbox(), testCtx());
    expect(out.ok).toBe(false);
    if (out.ok) throw new Error("expected err");
    expect(out.error.kind).toBe("SandboxViolation");
    if (out.error.kind === "SandboxViolation") {
      expect(out.error.violation.kind).toBe("path_escape");
    }
  });

  // Recoverable tool error returned as ToolOutput.
  it("returns a recoverable tool error as ToolOutput", async () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new FailingTool("fail"), schema("fail"))).toBeNull();
    const out = await reg.dispatch(call("fail", "c1", {}), new AllowAllSandbox(), testCtx());
    expect(out.ok).toBe(true);
    if (!out.ok) throw new Error("expected ok");
    expect(out.result.output.kind).toBe("error");
    if (out.result.output.kind === "error") {
      expect(out.result.output.message).toBe("boom");
      expect(out.result.output.recoverable).toBe(true);
    }
  });

  // Rule 8: dispatchAll preserves caller-visible order.
  it("dispatchAll preserves input order across concurrent + sequential split", async () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("r"), schema("r", { read_only: true }))).toBeNull();
    expect(reg.register(new EchoTool("d"), schema("d", { destructive: true }))).toBeNull();
    const results = await reg.dispatchAll(
      [
        call("d", "1", { v: "a" }),
        call("r", "2", { v: "b" }),
        call("d", "3", { v: "c" }),
        call("r", "4", { v: "d" }),
      ],
      new AllowAllSandbox(),
      testCtx(),
    );
    const ids = results.map((r) => {
      if (!r.ok) throw new Error("expected ok");
      return r.result.call_id;
    });
    expect(ids).toEqual(["1", "2", "3", "4"]);
  });

  // Rule 8 (concurrency): read_only calls execute concurrently.
  it("dispatchAll runs read_only calls concurrently", async () => {
    const reg = new StandardToolRegistry();
    // Slow read-only tools — measure wall time to assert concurrency.
    type Sandbox = Parameters<InstanceType<typeof EchoTool>["execute"]>[1];
    type Ctx = Parameters<InstanceType<typeof EchoTool>["execute"]>[2];
    class SlowEcho extends EchoTool {
      override async execute(c: ToolCall, s: Sandbox, ctx: Ctx) {
        await new Promise<void>((resolve) => setTimeout(resolve, 60));
        return super.execute(c, s, ctx);
      }
    }
    expect(reg.register(new SlowEcho("r1"), schema("r1", { read_only: true }))).toBeNull();
    expect(reg.register(new SlowEcho("r2"), schema("r2", { read_only: true }))).toBeNull();
    expect(reg.register(new SlowEcho("r3"), schema("r3", { read_only: true }))).toBeNull();
    const start = Date.now();
    const results = await reg.dispatchAll(
      [call("r1", "1", {}), call("r2", "2", {}), call("r3", "3", {})],
      new AllowAllSandbox(),
      testCtx(),
    );
    const elapsed = Date.now() - start;
    expect(results.every((r) => r.ok)).toBe(true);
    // Three 60ms tools sequentially would be ≥180ms; concurrent should be well under.
    expect(elapsed).toBeLessThan(160);
  });

  // dispatchAll surfaces per-slot errors.
  it("dispatchAll surfaces per-slot errors", async () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("ok"), schema("ok", { read_only: true }))).toBeNull();
    const results = await reg.dispatchAll(
      [call("ok", "1", {}), call("missing", "2", {})],
      new AllowAllSandbox(),
      testCtx(),
    );
    expect(results[0]?.ok).toBe(true);
    expect(results[1]?.ok).toBe(false);
    if (results[1]?.ok === false) {
      expect(results[1].error.kind).toBe("UnregisteredTool");
    }
  });

  // Rule 10: hasSubagentTools reflects registration.
  it("hasSubagentTools reflects registration", () => {
    const reg = new StandardToolRegistry();
    expect(reg.hasSubagentTools()).toBe(false);
    expect(reg.register(new EchoTool("echo"), schema("echo"))).toBeNull();
    expect(reg.hasSubagentTools()).toBe(false);
    expect(reg.register(new SubagentMockTool("subagent"), schema("subagent"))).toBeNull();
    expect(reg.hasSubagentTools()).toBe(true);
  });

  // Rule 8 (active_schemas): filtered by phase and sorted.
  it("activeSchemas filters by phase and sorts by name", () => {
    const reg = new StandardToolRegistry();
    for (const n of ["zeta", "alpha", "beta"]) {
      expect(reg.register(new EchoTool(n), schema(n))).toBeNull();
    }
    expect(
      reg.registerSet({ name: "plan", tools: ["alpha", "zeta"], phase: "planning" }),
    ).toBeNull();
    expect(reg.registerSet({ name: "always", tools: ["beta"], phase: null })).toBeNull();

    const plan = reg.activeSchemas("planning").map((s) => s.name);
    expect(plan).toEqual(["alpha", "beta", "zeta"]);

    const exec = reg.activeSchemas("execution").map((s) => s.name);
    expect(exec).toEqual(["beta"]);
  });

  // activeSchemas with no sets returns every schema, sorted.
  it("activeSchemas with no sets returns the full catalog", () => {
    const reg = new StandardToolRegistry();
    expect(reg.register(new EchoTool("a"), schema("a"))).toBeNull();
    expect(reg.register(new EchoTool("b"), schema("b"))).toBeNull();
    expect(reg.activeSchemas(null).map((s) => s.name)).toEqual(["a", "b"]);
    expect(reg.activeSchemas("execution").map((s) => s.name)).toEqual(["a", "b"]);
  });

  it("rejects empty tool set name and duplicates", () => {
    const reg = new StandardToolRegistry();
    expect(reg.registerSet({ name: "", tools: [] })?.kind).toBe("InvalidSchema");
    expect(reg.registerSet({ name: "s", tools: [] })).toBeNull();
    expect(reg.registerSet({ name: "s", tools: [] })?.kind).toBe("DuplicateName");
  });
});
