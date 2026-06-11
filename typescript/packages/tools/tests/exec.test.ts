/**
 * Execution tool tests — mirror `rust/crates/spore-core/src/tools/exec.rs#tests`.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { harnessTesting, toolRegistry, type ToolCall } from "@spore/core";
import { describe, expect, it } from "vitest";

import { BashCommandTool, ExecTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
// Storage seam (#75): these tools ignore ctx, but the signature requires one.
const ctx = toolRegistry.toolRegistryMock.testCtx();

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

const IS_UNIX = process.platform !== "win32";

// ---------------- ExecTool (shell-free) ----------------

describe("ExecTool", () => {
  it.runIf(IS_UNIX)("echo returns stdout", async () => {
    const sb = new AllowAllSandbox();
    const r = await new ExecTool().execute(
      call("exec", { command: "echo", args: ["hi"] }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toContain("hi");
  });

  // `exec` must NOT interpret shell syntax: pipe/`$(...)`/redirect tokens are
  // passed to `echo` as literal arguments, and no file is created.
  it.runIf(IS_UNIX)("has no shell semantics", async () => {
    const sb = new AllowAllSandbox();
    // Run in a temp dir so we can prove no `out` file appears.
    const dir = mkdtempSync(join(tmpdir(), "spore-exec-noshell-"));
    const prev = process.cwd();
    process.chdir(dir);
    let r;
    try {
      r = await new ExecTool().execute(
        call("exec", {
          command: "echo",
          args: ["a|b", "$(whoami)", ">out"],
        }),
        sb,
        ctx,
      );
    } finally {
      process.chdir(prev);
    }
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toContain("a|b $(whoami) >out");
    expect(existsSync(join(dir, "out"))).toBe(false);
    rmSync(dir, { recursive: true, force: true });
  });

  it.runIf(IS_UNIX)("nonzero exit is recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new ExecTool().execute(
      call("exec", { command: "false" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it.runIf(IS_UNIX)(
    "timeout returns recoverable error",
    async () => {
      const sb = new AllowAllSandbox();
      const r = await new ExecTool().execute(
        call("exec", { command: "sleep", args: ["5"], timeout: 1 }),
        sb,
        ctx,
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
    const r = await new ExecTool().execute(call("exec", {}), sb, ctx);
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });
});

// ---------------- BashCommandTool (real shell) ----------------

describe("BashCommandTool", () => {
  it.runIf(IS_UNIX)("supports a pipeline", async () => {
    const sb = new AllowAllSandbox();
    const r = await new BashCommandTool().execute(
      call("bash_command", { script: "printf 'hi' | tr a-z A-Z" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("HI");
  });

  it.runIf(IS_UNIX)("supports a redirect", async () => {
    const sb = new AllowAllSandbox();
    const dir = mkdtempSync(join(tmpdir(), "spore-bash-redirect-"));
    const target = join(dir, "out.txt");
    const r = await new BashCommandTool().execute(
      call("bash_command", { script: `printf 'data' > ${target}` }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    expect(readFileSync(target, "utf8")).toBe("data");
    rmSync(dir, { recursive: true, force: true });
  });

  it.runIf(IS_UNIX)("nonzero exit is recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new BashCommandTool().execute(
      call("bash_command", { script: "exit 3" }),
      sb,
      ctx,
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
        call("bash_command", { script: "sleep 5", timeout: 1 }),
        sb,
        ctx,
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
    const r = await new BashCommandTool().execute(
      call("bash_command", {}),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it.runIf(IS_UNIX)("large stderr is truncated in error message", async () => {
    const sb = new AllowAllSandbox();
    // awk writes 10 KB to stderr and exits non-zero; verify elision in message.
    const r = await new BashCommandTool().execute(
      call("bash_command", {
        script:
          "awk 'BEGIN{for(i=0;i<10240;i++)printf \"x\" > \"/dev/stderr\"; exit 1}'",
      }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.message).toContain("bytes elided");
    expect(r.message.length).toBeLessThan(10 * 1024);
  });
});
