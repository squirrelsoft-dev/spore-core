/**
 * Search tool tests — mirror `rust/crates/spore-core/src/tools/search.rs#tests`.
 */

import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { harnessTesting, type ToolCall } from "@spore/core";
import { describe, expect, it } from "vitest";

import { FindFilesTool, GrepFilesTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

async function tmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), "spore-tools-search-"));
}

describe("search tools", () => {
  it("grep finds matches", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.txt"), "alpha\nbeta\nalpha2");
    const sb = new AllowAllSandbox();
    const r = await new GrepFilesTool().execute(
      call("grep_files", { pattern: "^alpha", path: dir }),
      sb,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toContain("alpha");
    expect(r.content).toContain("alpha2");
  });

  it("grep invalid regex returns invalid_params", async () => {
    const dir = await tmp();
    const sb = new AllowAllSandbox();
    const r = await new GrepFilesTool().execute(
      call("grep_files", { pattern: "(unclosed", path: dir }),
      sb,
    );
    expect(r.kind).toBe("error");
    if (r.kind !== "error") throw new Error("unreachable");
    expect(r.recoverable).toBe(true);
  });

  it("find_files glob", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.rs"), "");
    await writeFile(join(dir, "b.rs"), "");
    await writeFile(join(dir, "c.txt"), "");
    const sb = new AllowAllSandbox();
    const r = await new FindFilesTool().execute(
      call("find_files", { glob: "*.rs", path: dir }),
      sb,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    const lines = r.content.split("\n").filter((s) => s.length > 0);
    expect(lines.length).toBe(2);
  });
});
