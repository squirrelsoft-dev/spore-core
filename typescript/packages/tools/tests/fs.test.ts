/**
 * Filesystem tool tests — mirror `rust/crates/spore-core/src/tools/fs.rs#tests`.
 */

import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { realpathSync, readFileSync } from "node:fs";
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
import { applyReadRange } from "../src/fs.js";

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
      // #150: a sandbox violation now surfaces as the typed `sandbox_violation`
      // variant, not a flattened `error` — so the regression is "none must be a
      // sandbox_violation".
      expect(rr.kind).not.toBe("sandbox_violation");
      if (rr.kind === "error") {
        // A directory entry (e.g. `sub`) reads as a recoverable error but must
        // NOT mention a sandbox/path_escape — that's the regression.
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

  it("read of path outside root surfaces the typed path_escape violation (#150)", async () => {
    // Counterpart: a path resolving outside the root is still a sandbox
    // violation, even when the file does not exist. The tool surfaces the TYPED
    // violation (`sandbox_violation`); the harness (not the tool) decides
    // recoverable-vs-halt via SandboxViolationPolicy.
    const root = realpathSync(await tmp());
    const sb = new WorkspaceScopedSandbox({ root });
    const r = await new ReadFileTool().execute(
      call("read_file", { path: "../nonexistent_secret" }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("sandbox_violation");
    if (r.kind !== "sandbox_violation") throw new Error("unreachable");
    expect(r.violation.kind).toBe("path_escape");
  });
});

// ============================================================================
// #134: list_dir gitignore-aware walk
// ============================================================================

describe("list_dir gitignore (#134)", () => {
  it("recursive default excludes gitignored files and .git/", async () => {
    // Build a temp tree:
    //   .gitignore        — "dist/\n*.log"
    //   src/main.ts       — tracked
    //   dist/bundle.js    — ignored via dist/
    //   .git/config       — always excluded
    //   logs/app.log      — ignored via *.log
    const root = realpathSync(await tmp());
    await writeFile(join(root, ".gitignore"), "dist/\n*.log\n");
    await mkdir(join(root, "src"), { recursive: true });
    await writeFile(join(root, "src", "main.ts"), "// main");
    await mkdir(join(root, "dist"), { recursive: true });
    await writeFile(join(root, "dist", "bundle.js"), "// bundle");
    await mkdir(join(root, ".git"), { recursive: true });
    await writeFile(join(root, ".git", "config"), "[core]");
    await mkdir(join(root, "logs"), { recursive: true });
    await writeFile(join(root, "logs", "app.log"), "log");

    const sb = new AllowAllSandbox();
    const r = await new ListDirTool().execute(
      call("list_dir", { path: root, recursive: true }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");

    const entries = r.content.split("\n").filter((e) => e !== "");

    // Tracked file must appear.
    expect(entries.some((e) => e.endsWith("src/main.ts"))).toBe(true);
    // Gitignored files must NOT appear.
    expect(entries.some((e) => e.includes("dist"))).toBe(false);
    // .git/ dir and its contents must not appear (use path-segment check to
    // avoid false positives on the ".gitignore" filename).
    expect(entries.some((e) => /(^|\/)\.git(\/|$)/.test(e))).toBe(false);
    expect(entries.some((e) => e.includes("app.log"))).toBe(false);
  });

  it("recursive include_ignored:true surfaces gitignored files but still skips .git/", async () => {
    const root = realpathSync(await tmp());
    await writeFile(join(root, ".gitignore"), "dist/\n*.log\n");
    await mkdir(join(root, "src"), { recursive: true });
    await writeFile(join(root, "src", "main.ts"), "// main");
    await mkdir(join(root, "dist"), { recursive: true });
    await writeFile(join(root, "dist", "bundle.js"), "// bundle");
    await mkdir(join(root, ".git"), { recursive: true });
    await writeFile(join(root, ".git", "config"), "[core]");
    await mkdir(join(root, "logs"), { recursive: true });
    await writeFile(join(root, "logs", "app.log"), "log");

    const sb = new AllowAllSandbox();
    const r = await new ListDirTool().execute(
      call("list_dir", { path: root, recursive: true, include_ignored: true }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");

    const entries = r.content.split("\n").filter((e) => e !== "");

    // Tracked file must appear.
    expect(entries.some((e) => e.endsWith("src/main.ts"))).toBe(true);
    // Previously-ignored files now appear.
    expect(entries.some((e) => e.includes("bundle.js"))).toBe(true);
    expect(entries.some((e) => e.includes("app.log"))).toBe(true);
    // .git/ must still be excluded unconditionally (path-segment check).
    expect(entries.some((e) => /(^|\/)\.git(\/|$)/.test(e))).toBe(false);
  });

  it("non-recursive listing always excludes .git/", async () => {
    const root = realpathSync(await tmp());
    await writeFile(join(root, ".gitignore"), "dist/\n");
    await mkdir(join(root, "src"), { recursive: true });
    await mkdir(join(root, "dist"), { recursive: true });
    await mkdir(join(root, ".git"), { recursive: true });
    await writeFile(join(root, "readme.md"), "# readme");

    const sb = new AllowAllSandbox();
    const r = await new ListDirTool().execute(
      call("list_dir", { path: root }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");

    const entries = r.content.split("\n").filter((e) => e !== "");
    // .git/ must not appear in non-recursive listing (path-segment check).
    expect(entries.some((e) => /(^|\/)\.git(\/|$)/.test(e))).toBe(false);
    // Non-recursive: files/dirs at the top level (excluding .git) appear.
    expect(entries.some((e) => e.includes("readme.md"))).toBe(true);
  });
});

// ============================================================================
// #132: read_file range scan + line numbers — unit tests
// ============================================================================

describe("applyReadRange (#132)", () => {
  const TEN_LINES =
    "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n";

  it("all defaults → byte-identical output, no header", () => {
    const body = "line1\nline2\nline3\n";
    const r = applyReadRange(body, { offset: 1, length: 0, line_numbers: false });
    expect(r).toEqual({ ok: true, value: body });
  });

  it("offset only → header + lines from offset to EOF", () => {
    const r = applyReadRange(TEN_LINES, { offset: 3, length: 0, line_numbers: false });
    expect(r).toEqual({
      ok: true,
      value:
        "[lines 3–10 of 10]\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n",
    });
  });

  it("length only → first N lines with header", () => {
    const r = applyReadRange(TEN_LINES, { offset: 1, length: 3, line_numbers: false });
    expect(r).toEqual({
      ok: true,
      value: "[lines 1–3 of 10]\nline1\nline2\nline3\n",
    });
  });

  it("offset + length together", () => {
    const r = applyReadRange(TEN_LINES, { offset: 4, length: 3, line_numbers: false });
    expect(r).toEqual({
      ok: true,
      value: "[lines 4–6 of 10]\nline4\nline5\nline6\n",
    });
  });

  it("line_numbers alone (whole file, single-digit total width)", () => {
    const body = "alpha\nbeta\ngamma\n";
    const r = applyReadRange(body, { offset: 1, length: 0, line_numbers: true });
    expect(r).toEqual({
      ok: true,
      value: "[lines 1–3 of 3]\n1 | alpha\n2 | beta\n3 | gamma\n",
    });
  });

  it("all three combined — right-pads line numbers to total-line digit width", () => {
    // total=10 → width 2 → single-digit lines get a leading space.
    const r = applyReadRange(TEN_LINES, { offset: 2, length: 3, line_numbers: true });
    expect(r).toEqual({
      ok: true,
      value: "[lines 2–4 of 10]\n 2 | line2\n 3 | line3\n 4 | line4\n",
    });
  });

  it("offset past EOF → recoverable error", () => {
    const body = "alpha\nbeta\ngamma\n";
    const r = applyReadRange(body, { offset: 11, length: 0, line_numbers: false });
    expect(r).toEqual({
      ok: false,
      error: "offset 11 exceeds file length 3",
    });
  });

  it("length trimmed at EOF silently (not an error)", () => {
    // offset 8 + length 5 would reach line 12, but only 10 lines exist.
    const r = applyReadRange(TEN_LINES, { offset: 8, length: 5, line_numbers: false });
    expect(r).toEqual({
      ok: true,
      value: "[lines 8–10 of 10]\nline8\nline9\nline10\n",
    });
  });

  it("offset=0 → recoverable error", () => {
    const r = applyReadRange("alpha\nbeta\n", { offset: 0, length: 0, line_numbers: false });
    expect(r.ok).toBe(false);
    if (r.ok) throw new Error("unreachable");
    expect(r.error).toContain("offset");
  });

  it("length=0 + offset > 1 → reads to EOF (no error)", () => {
    const body = "line1\nline2\nline3\nline4\nline5\n";
    const r = applyReadRange(body, { offset: 3, length: 0, line_numbers: false });
    expect(r).toEqual({
      ok: true,
      value: "[lines 3–5 of 5]\nline3\nline4\nline5\n",
    });
  });

  it("empty file + any params → empty content, no header", () => {
    const r = applyReadRange("", { offset: 1, length: 5, line_numbers: true });
    expect(r).toEqual({ ok: true, value: "" });
  });

  it("final line without trailing newline is preserved verbatim", () => {
    const body = "a\nb\nc"; // no trailing \n
    const r = applyReadRange(body, { offset: 2, length: 0, line_numbers: false });
    expect(r).toEqual({
      ok: true,
      value: "[lines 2–3 of 3]\nb\nc",
    });
  });

  it("end-to-end: read_file with offset emits header", async () => {
    const dir = await tmp();
    const path = join(dir, "r.txt");
    await writeFile(path, "l1\nl2\nl3\n");
    const sb = new AllowAllSandbox();
    const r = await new ReadFileTool().execute(
      call("read_file", { path, offset: 2 }),
      sb,
      ctx,
    );
    expect(r.kind).toBe("success");
    if (r.kind !== "success") throw new Error("unreachable");
    expect(r.content).toBe("[lines 2–3 of 3]\nl2\nl3\n");
  });
});

// ============================================================================
// #132: fixture replay
// ============================================================================

interface FixtureCase {
  name: string;
  initial_content: string;
  params: Record<string, unknown>;
  expected:
    | { kind: "success"; content: string }
    | { kind: "error"; recoverable: boolean; message_contains: string };
}

describe("read_file_range fixture replay (#132)", () => {
  // Resolve fixture path relative to this file's location (monorepo root → fixtures/).
  const fixtureDir = join(
    new URL("../../../..", import.meta.url).pathname,
    "fixtures",
    "tools",
  );
  const fixturePath = join(fixtureDir, "read_file_range.json");
  const cases: FixtureCase[] = JSON.parse(
    readFileSync(fixturePath, "utf8"),
  ) as FixtureCase[];

  for (const tc of cases) {
    it(`fixture: ${tc.name}`, async () => {
      const dir = await tmp();
      const filePath = join(dir, "fixture_file.txt");
      await writeFile(filePath, tc.initial_content);

      // Replace the <FIXTURE_PATH> placeholder with the real temp file path.
      const params = { ...tc.params, path: filePath };
      const sb = new AllowAllSandbox();
      const r = await new ReadFileTool().execute(
        call("read_file", params),
        sb,
        ctx,
      );

      if (tc.expected.kind === "success") {
        expect(r.kind).toBe("success");
        if (r.kind !== "success") throw new Error(`[${tc.name}] expected success, got ${JSON.stringify(r)}`);
        expect(r.content).toBe(tc.expected.content);
      } else {
        expect(r.kind).toBe("error");
        if (r.kind !== "error") throw new Error(`[${tc.name}] expected error, got ${JSON.stringify(r)}`);
        expect(r.recoverable).toBe(tc.expected.recoverable);
        expect(r.message).toContain(tc.expected.message_contains);
      }
    });
  }
});
