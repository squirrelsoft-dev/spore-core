/**
 * Filesystem tool tests — mirror `rust/crates/spore-core/src/tools/fs.rs#tests`.
 */

import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { harnessTesting, type ToolCall } from "@spore/core";
import { describe, expect, it } from "vitest";

import {
  DeleteFileTool,
  ListDirTool,
  MoveFileTool,
  ReadFileTool,
  WriteFileTool,
} from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;

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
    );
    expect(w.kind).toBe("success");
    const r = await new ReadFileTool().execute(call("read_file", { path }), sb);
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
    );
    await new WriteFileTool().execute(
      call("write_file", { path, content: "b", append: true }),
      sb,
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
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    const lines = r.content.split("\n");
    expect(lines.length).toBe(3);
    const sorted = [...lines].sort();
    expect(lines).toEqual(sorted);
  });

  it("delete missing is recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new DeleteFileTool().execute(
      call("delete_file", { path: "/no/such/path/here-xyz" }),
      sb,
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
    );
    expect(r.kind).toBe("success");
    expect(await readFile(dst, "utf8")).toBe("hi");
  });

  it("invalid params returns recoverable error", async () => {
    const sb = new AllowAllSandbox();
    const r = await new ReadFileTool().execute(call("read_file", {}), sb);
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });
});
