/**
 * Filesystem tool tests — mirror `rust/crates/spore-core/src/tools/fs.rs#tests`.
 */

import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  harnessTesting,
  toolRegistry,
  WorkspaceScopedSandbox,
  type ToolCall,
} from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  DeleteFileTool,
  ListDirTool,
  MoveFileTool,
  ReadFileTool,
  WriteFileTool,
} from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
// Storage seam (#75): these tools ignore ctx, but the signature requires one.
const ctx = toolRegistry.toolRegistryMock.testCtx();

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

async function tmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), "spore-tools-"));
}

describe("filesystem tools", () => {
  it("write then read roundtrip", async () => {
    const dir = await tmp();
    const path = join(dir, "a.txt");
    const sb = new AllowAllSandbox();
    const w = await new WriteFileTool().execute(
      call("write_file", { path, content: "hello" }),
      sb,
      ctx,
    );
    expect(w.kind).toBe("success");
    const r = await new ReadFileTool().execute(
      call("read_file", { path }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("hello");
  });

  it("append mode concatenates", async () => {
    const dir = await tmp();
    const path = join(dir, "a.txt");
    const sb = new AllowAllSandbox();
    await new WriteFileTool().execute(
      call("write_file", { path, content: "a" }),
      sb,
      ctx,
    );
    await new WriteFileTool().execute(
      call("write_file", { path, content: "b", append: true }),
      sb,
      ctx,
    );
    expect(await readFile(path, "utf8")).toBe("ab");
  });

  it("list_dir is sorted", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "z"), "");
    await writeFile(join(dir, "a"), "");
    await writeFile(join(dir, "m"), "");
    const sb = new AllowAllSandbox();
    const r = await new ListDirTool().execute(
      call("list_dir", { path: dir }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    const lines = r.content.split("\n");
    expect(lines.length).toBe(3);
    const sorted = [...lines].sort();
    expect(lines).toEqual(sorted);
  });

  it("list_dir entries round-trip through the workspace sandbox", async () => {
    // Regression for #93: every entry list_dir returns must round-trip
    // straight back into read_file under the *real* WorkspaceScopedSandbox,
    // which treats all input paths as root-relative. Absolute paths (the old
    // behavior) would be rejected as a sandbox path_escape.
    const root = realpathSync(await tmp());
    await writeFile(join(root, "a.txt"), "alpha");
    await writeFile(join(root, "b.txt"), "beta");
    await mkdir(join(root, "sub"));
    await writeFile(join(root, "sub", "c.txt"), "gamma");
    const sb = new WorkspaceScopedSandbox({ root });

    // Recursive so we exercise both top-level files and a nested file.
    const r = await new ListDirTool().execute(
      call("list_dir", { path: ".", recursive: true }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    const entries = r.content.split("\n").filter((e) => e !== "");

    // Bare root-relative names for top-level files; nested file re-anchored.
    expect(entries).toContain("a.txt");
    expect(entries).toContain("sub/c.txt");
    // Must not emit the listed dir itself as an empty/`.` entry.
    expect(entries.every((e) => e !== "" && e !== ".")).toBe(true);

    // The actual bug check: feed each entry straight into read_file. None must
    // fail with a sandbox/path_escape violation.
    for (const entry of entries) {
      const rr = await new ReadFileTool().execute(
        call("read_file", { path: entry }),
        sb,
        ctx,
      );
      if (rr.kind === "error") {
        // A directory entry (e.g. `sub`) reads as an error but must NOT be a
        // sandbox violation — that's the regression.
        const msg = rr.message.toLowerCase();
        expect(msg.includes("escape") || msg.includes("sandbox")).toBe(false);
      } else {
        expect(rr.kind).toBe("success");
      }
    }
  });

  it("delete missing is recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new DeleteFileTool().execute(
      call("delete_file", { path: "/no/such/path/here-xyz" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("move_file renames", async () => {
    const dir = await tmp();
    const src = join(dir, "s");
    const dst = join(dir, "d");
    await writeFile(src, "hi");
    const sb = new AllowAllSandbox();
    const r = await new MoveFileTool().execute(
      call("move_file", { src, dst }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    expect(await readFile(dst, "utf8")).toBe("hi");
  });

  it("invalid params returns recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new ReadFileTool().execute(call("read_file", {}), sb, ctx);
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("read of missing in-workspace file is recoverable not-found, not path_escape", async () => {
    // Regression for #63: reading a not-yet-created file *inside* the
    // workspace must surface a recoverable not-found, not a sandbox
    // path_escape, end to end through the real WorkspaceScopedSandbox.
    const root = realpathSync(await tmp());
    const sb = new WorkspaceScopedSandbox({ root });
    const r = await new ReadFileTool().execute(
      call("read_file", { path: "output.txt" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
    expect(r.message).toContain("read failed");
  });

  it("read of path outside root is a sandbox path_escape error", async () => {
    // Counterpart: a path resolving outside the root is still a sandbox
    // violation, even when the file does not exist.
    const root = realpathSync(await tmp());
    const sb = new WorkspaceScopedSandbox({ root });
    const r = await new ReadFileTool().execute(
      call("read_file", { path: "../nonexistent_secret" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    const msg = r.message.toLowerCase();
    expect(msg.includes("escape") || msg.includes("sandbox")).toBe(true);
  });
});
