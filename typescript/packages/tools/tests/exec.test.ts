/**
 * Execution tool tests — mirror `rust/crates/spore-core/src/tools/exec.rs#tests`.
 */

import { harnessTesting, type ToolCall } from "@spore/core";
import { describe, expect, it } from "vitest";

import { BashCommandTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

const IS_UNIX = process.platform !== "win32";

describe("BashCommandTool", () => {
  it.runIf(IS_UNIX)("echo returns stdout", async () => {
    const sb = new AllowAllSandbox();
    const r = await new BashCommandTool().execute(
      call("bash_command", { command: "echo", args: ["hi"] }),
      sb,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toContain("hi");
  });

  it.runIf(IS_UNIX)("nonzero exit is recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new BashCommandTool().execute(
      call("bash_command", { command: "false" }),
      sb,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it.runIf(IS_UNIX)(
    "timeout returns recoverable error",
    async () => {
      const sb = new AllowAllSandbox();
      const r = await new BashCommandTool().execute(
        call("bash_command", { command: "sleep", args: ["5"], timeout: 1 }),
        sb,
      );
      expect(r.kind).toBe("error");
      if (r.kind !== "error") throw new Error("unreachable");
      expect(r.recoverable).toBe(true);
      expect(r.message).toContain("timed out");
    },
    15000,
  );

  it("invalid params returns recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new BashCommandTool().execute(call("bash_command", {}), sb);
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });
});
