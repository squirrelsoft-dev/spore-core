/**
 * Search tool tests — mirror `rust/crates/spore-core/src/tools/search.rs#tests`.
 */

import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { harnessTesting, toolRegistry, type ToolCall } from "@spore/core";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { FindFilesTool, GrepFilesTool, GrepTool } from "../src/index.js";

const { AllowAllSandbox } = harnessTesting;
// Storage seam (#75): these tools ignore ctx, but the signature requires one.
const ctx = toolRegistry.toolRegistryMock.testCtx();

function call(name: string, input: unknown): ToolCall {
  return { id: "c1", name, input };
}

async function tmp(): Promise<string> {
  return mkdtemp(join(tmpdir(), "spore-tools-search-"));
}

/** Run GrepTool and return the success content, or throw on error. */
async function grepOut(input: unknown): Promise<string> {
  const sb = new AllowAllSandbox();
  const r = await new GrepTool().execute(call("grep", input), sb, ctx);
  if (r.kind !== "success") throw new Error(`GrepTool error: ${JSON.stringify(r)}`);
  return r.content;
}

describe("search tools", () => {
  it("grep finds matches", async () => {
    const dir = await tmp();
    await writeFile(join(dir, "a.txt"), "alpha\nbeta\nalpha2");
    const sb = new AllowAllSandbox();
    const r = await new GrepFilesTool().execute(
      call("grep_files", { pattern: "^alpha", path: dir }),
      sb,
      ctx,
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
      ctx,
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
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    const lines = r.content.split("\n").filter((s) => s.length > 0);
    expect(lines.length).toBe(2);
  });
});

// ============================================================================
// GrepTool — context_lines (#133)
// ============================================================================

describe("GrepTool context_lines", () => {
  it("context_lines=0 is byte-identical to existing output", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "alpha\nbeta\ngamma\n");

    const withoutParam = await grepOut({ pattern: "beta", path: file });
    const withZero = await grepOut({ pattern: "beta", path: file, context_lines: 0 });

    expect(withZero).toBe(withoutParam);
    expect(withZero).toBe(`${file}:2:beta`);
  });

  it("single match with context — shows surrounding lines", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "one\ntwo\nthree\nfour\nfive\n");

    const out = await grepOut({ pattern: "three", path: file, context_lines: 1 });

    expect(out).toBe(
      `${file}:2-two\n${file}:3:three\n${file}:4-four`,
    );
  });

  it("overlapping windows are merged — no -- separator", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "a\nb\nc\nd\ne\n");

    // b (idx 1) and d (idx 3) each with context=2 → windows [0,3] and [1,4] → merged [0,4]
    const out = await grepOut({ pattern: "b|d", path: file, context_lines: 2 });

    expect(out).toBe(
      `${file}:1-a\n${file}:2:b\n${file}:3-c\n${file}:4:d\n${file}:5-e`,
    );
  });

  it("non-overlapping groups separated by --", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    // match1 at line 1, match10 at line 10; context=1 → windows [0,1] and [8,9] → non-adjacent
    await writeFile(
      file,
      "match1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nmatch10\nline11\nline12\n",
    );

    const out = await grepOut({ pattern: "match", path: file, context_lines: 1 });

    expect(out).toBe(
      `${file}:1:match1\n${file}:2-line2\n--\n${file}:9-line9\n${file}:10:match10\n${file}:11-line11`,
    );
  });

  it("context at file start — clamped to line 1", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "match\nline2\nline3\nline4\nline5\n");

    const out = await grepOut({ pattern: "match", path: file, context_lines: 3 });

    // match is at line 1 (0-based idx 0); no negative context; window [0, 3]
    expect(out).toBe(
      `${file}:1:match\n${file}:2-line2\n${file}:3-line3\n${file}:4-line4`,
    );
  });

  it("context at file end — clamped to last line", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "line1\nline2\nline3\nline4\nmatch\n");

    const out = await grepOut({ pattern: "match", path: file, context_lines: 3 });

    // match is at line 5 (0-based idx 4); window [1, 4] (clamped)
    expect(out).toBe(
      `${file}:2-line2\n${file}:3-line3\n${file}:4-line4\n${file}:5:match`,
    );
  });

  it("a context line that is also a match uses : separator", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "alpha\nbeta\ngamma\n");

    // Both alpha (idx 0) and beta (idx 1) match; context=1 → merged window [0,2]
    // alpha and beta → `:`, gamma → `-`
    const out = await grepOut({
      pattern: "alpha|beta",
      path: file,
      context_lines: 1,
    });

    expect(out).toBe(
      `${file}:1:alpha\n${file}:2:beta\n${file}:3-gamma`,
    );
  });

  it("no matches — empty output regardless of context_lines", async () => {
    const dir = await tmp();
    const file = join(dir, "f.txt");
    await writeFile(file, "line1\nline2\n");

    const out = await grepOut({ pattern: "NOMATCH", path: file, context_lines: 2 });

    expect(out).toBe("");
  });
});

// ============================================================================
// Fixture replay — grep_context_lines.json
// ============================================================================

const FixtureCaseSchema = z.object({
  name: z.string(),
  initial_content: z.string(),
  params: z.record(z.unknown()),
  expected: z.string(),
});
const FixtureSchema = z.array(FixtureCaseSchema);

describe("GrepTool context_lines fixture replay", () => {
  it("replays all cases in grep_context_lines.json", async () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const fixturePath = resolve(here, "../../../../fixtures/tools/grep_context_lines.json");
    const raw = JSON.parse(await readFile(fixturePath, "utf8")) as unknown;
    const cases = FixtureSchema.parse(raw);

    for (const tc of cases) {
      const dir = await tmp();
      const file = join(dir, "fixture.txt");
      await writeFile(file, tc.initial_content);

      // Replace <FIXTURE_PATH> placeholder in params and expected.
      const params = Object.fromEntries(
        Object.entries(tc.params).map(([k, v]) => [
          k,
          typeof v === "string" ? v.replace("<FIXTURE_PATH>", file) : v,
        ]),
      );
      const expected = tc.expected.replace(/<FIXTURE_PATH>/g, file);

      const out = await grepOut(params);
      expect(out, `fixture case "${tc.name}"`).toBe(expected);
    }
  });
});
