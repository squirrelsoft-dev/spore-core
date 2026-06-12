/**
 * #78 R9 — the `RealToolRegistry` bridge threads the configured scope-aware
 * `MemoryStore` into the `ToolContext` it forwards on every dispatch. Mirrors
 * the Rust `tool_context_exposes_threaded_memory_store` test (which lives in the
 * storage crate alongside `RealToolRegistry`). In TypeScript the bridge lives in
 * `@spore/tools`, so the seam test lives here to avoid a core→tools dependency.
 */

import {
  memory as coreMemory,
  harnessTesting,
  SessionId,
  storage as coreStorage,
  toolRegistry,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import { RealToolRegistry } from "../src/index.js";

const { Timestamp } = coreMemory;
const { InMemoryStorageProvider } = coreStorage;
const { AllowAllSandbox } = harnessTesting;
const { StandardToolRegistry } = toolRegistry;

describe("ToolContext memoryStore seam (#78 R9)", () => {
  it("threads the configured memory store through ToolContext", async () => {
    // Wire a real memory backend through the registry and prove the seam is live
    // by writing through ToolContext's memoryStore and reading it back.
    const memory = new InMemoryStorageProvider();
    const inner = new StandardToolRegistry();
    const bridge = new RealToolRegistry(
      inner,
      new AllowAllSandbox(),
      SessionId.of("ctx-test"),
      coreStorage.ProjectId.fromCanonicalPath("/test-project"),
      new InMemoryStorageProvider(),
      memory,
    );

    const ctx = bridge.toolContext();
    await ctx.memoryStore.appendMemory("project", ctx.sessionId, {
      role: "user",
      content: "threaded",
      timestamp: Timestamp.of("t1"),
      metadata: {},
    });

    // Read back through the same Arc-equivalent backend the registry threaded in.
    const got = await memory.getMemories(
      "project",
      SessionId.of("ctx-test"),
      10,
    );
    expect(got.map((e) => e.content)).toEqual(["threaded"]);
  });
});
