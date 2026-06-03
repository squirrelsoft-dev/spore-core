/**
 * Unit tests for {@link defineTool} — the ergonomic, drift-proof tool helper.
 *
 * Mirrors Rust's `tool!` macro tests in
 * `rust/crates/spore-core/src/macros.rs#tests`:
 *   - the helper builds a tool whose advertised schema is DERIVED from the input
 *     Zod schema (so schema and validation can never drift);
 *   - optional annotations default to all-`false` and are otherwise respected;
 *   - bad/invalid input yields a RECOVERABLE error whose message contains the
 *     substring `invalid parameters`.
 */

import { describe, expect, it } from "vitest";
import { z } from "zod";

import { toolRegistry } from "../src/index.js";
import { toolOutput } from "../src/harness/types.js";
import type { ToolCall } from "../src/index.js";

const {
  defineTool,
  toolRegistryMock: { AllowAllSandbox, testCtx },
} = toolRegistry;

const sb = new AllowAllSandbox();

const EchoInput = z.object({
  /** Text to echo back. */
  message: z.string().describe("Text to echo back."),
  shout: z.boolean().default(false),
});

function echoTool() {
  return defineTool({
    name: "echo",
    description: "Echoes the input message",
    input: EchoInput,
    execute: async (input) =>
      toolOutput.success(input.shout ? input.message.toUpperCase() : input.message),
  });
}

function call(input: unknown): ToolCall {
  return { id: "c1", name: "echo", input };
}

describe("defineTool", () => {
  it("builds a tool with a schema derived from the input Zod schema", () => {
    const t = echoTool();
    expect(t.schema.name).toBe("echo");
    expect(t.schema.description).toBe("Echoes the input message");

    // Annotations default to all-false when omitted.
    expect(t.schema.annotations).toEqual({
      read_only: false,
      destructive: false,
      idempotent: false,
      open_world: false,
    });

    // The advertised schema exposes the input's properties — proof it was
    // DERIVED from the Zod schema, not hand-written.
    const params = t.schema.parameters as {
      type?: string;
      properties?: Record<string, unknown>;
    };
    expect(params.type).toBe("object");
    expect(params.properties).toBeDefined();
    expect(Object.keys(params.properties ?? {})).toContain("message");
    expect(Object.keys(params.properties ?? {})).toContain("shout");

    // Implementation name matches the schema name.
    expect(t.implementation.name).toBe("echo");
  });

  it("validates with the same schema and runs the body on good input", async () => {
    const t = echoTool();
    const out = await t.implementation.execute(
      call({ message: "hi", shout: true }),
      sb,
      testCtx(),
    );
    expect(out).toEqual(toolOutput.success("HI"));
  });

  it("respects explicitly-provided annotations", () => {
    const t = defineTool({
      name: "lookup",
      description: "Reads shared state",
      input: EchoInput,
      execute: async (input) => toolOutput.success(input.message),
      annotations: { read_only: true, idempotent: true },
    });
    expect(t.schema.annotations.read_only).toBe(true);
    expect(t.schema.annotations.idempotent).toBe(true);
    // Unspecified fields still default to false.
    expect(t.schema.annotations.destructive).toBe(false);
    expect(t.schema.annotations.open_world).toBe(false);
  });

  it("returns a recoverable `invalid parameters` error on a missing field", async () => {
    const t = echoTool();
    // `message` missing → parse fails → recoverable error.
    const out = await t.implementation.execute(call({ shout: true }), sb, testCtx());
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid parameters for tool `echo`");
      expect(out.message).toContain("message");
    }
  });

  it("returns a recoverable error on a wrong-typed field", async () => {
    const t = echoTool();
    const out = await t.implementation.execute(
      call({ message: 7, shout: true }),
      sb,
      testCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid parameters");
    }
  });

  it("treats a missing `input` (null/undefined) as a recoverable error", async () => {
    const t = echoTool();
    const out = await t.implementation.execute(
      { id: "c1", name: "echo", input: undefined as unknown },
      sb,
      testCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });
});
