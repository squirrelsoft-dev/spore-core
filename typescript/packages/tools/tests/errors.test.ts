/**
 * Unit tests for the ToolExecutionError → ToolOutput mapping.
 * Mirrors `rust/crates/spore-core/src/tools/error.rs#tests`.
 */

import { describe, expect, it } from "vitest";

import { toolExecutionErrorToOutput } from "../src/index.js";

describe("toolExecutionErrorToOutput", () => {
  it("invalid_parameters is recoverable", () => {
    const o = toolExecutionErrorToOutput({
      kind: "invalid_parameters",
      reason: "x",
    });
    expect(o.kind).toBe("error");
    if (o.kind !== "error") throw new Error("unreachable");
    expect(o.recoverable).toBe(true);
  });

  it("execution_failed passes through flag", () => {
    const o = toolExecutionErrorToOutput({
      kind: "execution_failed",
      reason: "x",
      recoverable: false,
    });
    expect(o.kind).toBe("error");
    if (o.kind !== "error") throw new Error("unreachable");
    expect(o.recoverable).toBe(false);
  });

  it("sandbox_violation is not recoverable", () => {
    const o = toolExecutionErrorToOutput({
      kind: "sandbox_violation",
      violation: { kind: "path_escape", path: "/etc" },
    });
    expect(o.kind).toBe("error");
    if (o.kind !== "error") throw new Error("unreachable");
    expect(o.recoverable).toBe(false);
  });

  it("timeout is recoverable", () => {
    const o = toolExecutionErrorToOutput({ kind: "timeout", afterMs: 5000 });
    expect(o.kind).toBe("error");
    if (o.kind !== "error") throw new Error("unreachable");
    expect(o.recoverable).toBe(true);
    expect(o.message).toContain("5s");
  });
});
