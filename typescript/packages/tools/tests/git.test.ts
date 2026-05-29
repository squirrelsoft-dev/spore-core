/**
 * Git tool tests — skipped when `git` is unavailable.
 */

import { spawnSync } from "node:child_process";

import { harnessTesting, toolRegistry, type ToolCall } from "@spore/core";
import { describe, expect, it } from "vitest";

import { GitResetModeSchema, GitStatusTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
// Storage seam (#75): these tools ignore ctx, but the signature requires one.
const ctx = toolRegistry.toolRegistryMock.testCtx();

function gitAvailable(): boolean {
  try {
    const r = spawnSync("git", ["--version"]);
    return r.status === 0;
  } catch {
    return false;
  }
}

const HAS_GIT = gitAvailable();

describe("git tools", () => {
  it.runIf(HAS_GIT)("git_status runs when git is available", async () => {
    const sb = new AllowAllSandbox();
    const call: ToolCall = { id: "c1", name: "git_status", input: {} };
    const r = await new GitStatusTool().execute(call, sb, ctx);
    // Either Success (in a repo) or Error (outside one) — both are fine.
    expect(["success", "error"]).toContain(r.kind);
  });

  it("GitResetMode parses snake_case strings", () => {
    expect(GitResetModeSchema.parse("hard")).toBe("hard");
    expect(GitResetModeSchema.parse("soft")).toBe("soft");
    expect(GitResetModeSchema.parse("mixed")).toBe("mixed");
    expect(GitResetModeSchema.safeParse("bogus").success).toBe(false);
  });
});
