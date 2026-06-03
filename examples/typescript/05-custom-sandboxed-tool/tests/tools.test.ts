/**
 * Unit tests for the two custom tools shipped by example 05.
 *
 * They are driven directly over an in-memory run store (no model, no Ollama),
 * proving each rule the example demonstrates:
 *   - `remember` stores under the `fact:` prefix, keyed by SessionId.
 *   - `recall` returns the stored value string.
 *   - a `recall` miss is a recoverable error with the exact message.
 *   - missing / wrong-typed args are recoverable `invalid parameters` errors
 *     (validated by the SAME Zod schema that derives the advertised schema).
 *   - a run-store failure is a recoverable error.
 *   - the read-only / idempotent annotations are correct.
 *   - the advertised schema is DERIVED from the input (exposes `key`/`value`).
 */

import {
  SessionId,
  storage,
  toolRegistry,
  type SandboxProvider,
  type SandboxViolation,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  FACT_PREFIX,
  REMEMBER_NAME,
  rememberTool,
} from "../src/tools/remember.js";
import { RECALL_NAME, recallTool } from "../src/tools/recall.js";

type RunStore = storage.RunStore;
type JsonValue = storage.JsonValue;

const { ToolContext } = toolRegistry;
const { InMemoryStorageProvider } = storage;

/** These tools never touch the filesystem — a permissive sandbox is plenty. */
class AllowAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
}

/** A RunStore that always fails — proves storage errors map to recoverable errors. */
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

function rememberCall(input: unknown): ToolCall {
  return { id: "c1", name: REMEMBER_NAME, input };
}

function recallCall(input: unknown): ToolCall {
  return { id: "c2", name: RECALL_NAME, input };
}

function expectSuccess(out: ToolOutput): string {
  if (out.kind !== "success") {
    throw new Error(`expected success, got ${JSON.stringify(out)}`);
  }
  return out.content;
}

const sb = new AllowAllSandbox();

// Build fresh tools per use — they are stateless StandardTool bundles.
const remember = () => rememberTool().implementation;
const recall = () => recallTool().implementation;

describe("remember", () => {
  it("stores the value under the fact: prefix, keyed by SessionId", async () => {
    const ctx = ctxWith(new InMemoryStorageProvider(), "sess-a");
    const out = await remember().execute(
      rememberCall({ key: "habitat", value: "coastal ocean waters" }),
      sb,
      ctx,
    );
    expect(expectSuccess(out)).toBe("remembered habitat");

    // The blob lives under fact:habitat in the run store for this session.
    const stored = await ctx.runStore.get(
      SessionId.of("sess-a"),
      `${FACT_PREFIX}habitat`,
    );
    expect(stored).toBe("coastal ocean waters");
    // And NOT under the bare key.
    expect(await ctx.runStore.get(SessionId.of("sess-a"), "habitat")).toBe(
      undefined,
    );
  });

  it("missing 'value' is a recoverable 'invalid parameters' error", async () => {
    const out = await remember().execute(
      rememberCall({ key: "habitat" }),
      sb,
      inMemoryCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid parameters");
      expect(out.message).toContain("value");
    }
  });

  it("non-string 'key' is a recoverable 'invalid parameters' error", async () => {
    const out = await remember().execute(
      rememberCall({ key: 7, value: "x" }),
      sb,
      inMemoryCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid parameters");
    }
  });

  it("a run-store failure is a recoverable error", async () => {
    const out = await remember().execute(
      rememberCall({ key: "k", value: "v" }),
      sb,
      ctxWith(new FailingRunStore()),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("is not read_only / destructive / idempotent", () => {
    const a = rememberTool().schema.annotations;
    expect(a.read_only).toBe(false);
    expect(a.destructive).toBe(false);
    expect(a.idempotent).toBe(false);
  });

  it("advertises a schema derived from the input (exposes key + value)", () => {
    const s = rememberTool().schema;
    expect(s.name).toBe(REMEMBER_NAME);
    const params = s.parameters as { properties?: Record<string, unknown> };
    expect(Object.keys(params.properties ?? {})).toEqual(
      expect.arrayContaining(["key", "value"]),
    );
  });
});

describe("recall", () => {
  it("returns the value a prior remember stored", async () => {
    const ctx = inMemoryCtx();
    await remember().execute(
      rememberCall({ key: "diet", value: "crabs and small fish" }),
      sb,
      ctx,
    );
    const out = await recall().execute(recallCall({ key: "diet" }), sb, ctx);
    expect(expectSuccess(out)).toBe("crabs and small fish");
  });

  it("a miss is a recoverable error with the exact message", async () => {
    const out = await recall().execute(
      recallCall({ key: "nope" }),
      sb,
      inMemoryCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toBe("no fact stored under 'nope'");
    }
  });

  it("missing 'key' is a recoverable 'invalid parameters' error", async () => {
    const out = await recall().execute(recallCall({}), sb, inMemoryCtx());
    expect(out.kind).toBe("error");
    if (out.kind === "error") {
      expect(out.recoverable).toBe(true);
      expect(out.message).toContain("invalid parameters");
    }
  });

  it("non-string 'key' is a recoverable error", async () => {
    const out = await recall().execute(
      recallCall({ key: 123 }),
      sb,
      inMemoryCtx(),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("a run-store failure is a recoverable error", async () => {
    const out = await recall().execute(
      recallCall({ key: "k" }),
      sb,
      ctxWith(new FailingRunStore()),
    );
    expect(out.kind).toBe("error");
    if (out.kind === "error") expect(out.recoverable).toBe(true);
  });

  it("read does not write: a recall never persists anything", async () => {
    const ctx = inMemoryCtx();
    await recall().execute(recallCall({ key: "k" }), sb, ctx);
    expect(await ctx.runStore.listKeys(ctx.sessionId)).toEqual([]);
  });

  it("is read_only + idempotent (and not destructive)", () => {
    const a = recallTool().schema.annotations;
    expect(a.read_only).toBe(true);
    expect(a.idempotent).toBe(true);
    expect(a.destructive).toBe(false);
  });
});
