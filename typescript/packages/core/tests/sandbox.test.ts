/**
 * `WorkspaceScopedSandbox` unit tests — mirrors
 * `rust/crates/spore-core/src/sandbox.rs#tests`.
 */

import { mkdtempSync, realpathSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import {
  BuildError,
  WorkspaceScopedSandbox,
  type IsolationMode,
  type SandboxViolation,
  type WorkspaceConfig,
} from "../src/index.js";

function tmp(): string {
  return realpathSync(mkdtempSync(join(tmpdir(), "spore-sandbox-")));
}

function cfg(root: string): WorkspaceConfig {
  return { root };
}

describe("WorkspaceScopedSandbox construction", () => {
  it("fails when root does not exist", () => {
    expect(
      () =>
        new WorkspaceScopedSandbox({
          root: "/definitely/does/not/exist/spore-test",
        }),
    ).toThrowError(BuildError);
  });

  it("None isolation succeeds and warns", () => {
    const root = tmp();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const sb = new WorkspaceScopedSandbox(cfg(root), { kind: "none" });
    expect(sb.isolationMode()).toEqual({ kind: "none" });
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("workspaceRoot returns canonical root", () => {
    const root = tmp();
    const sb = new WorkspaceScopedSandbox(cfg(root));
    expect(sb.workspaceRoot()).toBe(root);
  });
});

describe("WorkspaceScopedSandbox path resolution", () => {
  it("escape via dotdot -> path_escape", async () => {
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    const r = await sb.resolvePath("../etc/passwd", "read");
    expect((r as SandboxViolation).kind).toBe("path_escape");
  });

  it("escape via absolute dotdot -> path_escape", async () => {
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    const r = await sb.resolvePath("/../../etc/passwd", "read");
    expect((r as SandboxViolation).kind).toBe("path_escape");
  });

  it("path_denied via denylist (matched_rule set)", async () => {
    const root = tmp();
    const secrets = join(root, "secrets");
    mkdirSync(secrets);
    writeFileSync(join(secrets, "k.txt"), "x");
    const sb = new WorkspaceScopedSandbox({
      ...cfg(root),
      denied_paths: ["secrets"],
    });
    const r = (await sb.resolvePath("secrets/k.txt", "read")) as SandboxViolation;
    expect(r.kind).toBe("path_denied");
    if (r.kind !== "path_denied") throw new Error("unreachable");
    expect(r.matched_rule).toContain("secrets");
  });

  it("path_denied via allowlist miss", async () => {
    const root = tmp();
    const allowed = join(root, "src");
    mkdirSync(allowed);
    writeFileSync(join(allowed, "a.rs"), "x");
    writeFileSync(join(root, "other.rs"), "x");
    const sb = new WorkspaceScopedSandbox({
      ...cfg(root),
      allowed_paths: ["src"],
    });
    const ok = await sb.resolvePath("src/a.rs", "read");
    expect(typeof ok).toBe("string");
    const r = (await sb.resolvePath("other.rs", "read")) as SandboxViolation;
    expect(r.kind).toBe("path_denied");
    if (r.kind !== "path_denied") throw new Error("unreachable");
    expect(r.matched_rule).toBe("not in allowlist");
  });

  it("extension_denied for dotfile", async () => {
    const root = tmp();
    writeFileSync(join(root, ".env"), "S=1");
    const sb = new WorkspaceScopedSandbox({
      ...cfg(root),
      denied_extensions: ["env"],
    });
    const r = (await sb.resolvePath(".env", "read")) as SandboxViolation;
    expect(r.kind).toBe("extension_denied");
  });

  it("read_only blocks write", async () => {
    const root = tmp();
    writeFileSync(join(root, "a.txt"), "x");
    const sb = new WorkspaceScopedSandbox({
      ...cfg(root),
      read_only: true,
    });
    expect(typeof (await sb.resolvePath("a.txt", "read"))).toBe("string");
    const r = (await sb.resolvePath("a.txt", "write")) as SandboxViolation;
    expect(r.kind).toBe("read_only_violation");
  });

  it("file_size_exceeded on read", async () => {
    const root = tmp();
    writeFileSync(join(root, "big.txt"), Buffer.alloc(1024, "A"));
    const sb = new WorkspaceScopedSandbox({
      ...cfg(root),
      max_file_size: 100,
    });
    const r = (await sb.resolvePath("big.txt", "read")) as SandboxViolation;
    expect(r.kind).toBe("file_size_exceeded");
    if (r.kind !== "file_size_exceeded") throw new Error("unreachable");
    expect(r.size).toBe(1024);
    expect(r.limit).toBe(100);
  });

  it("write to non-existent file canonicalizes parent", async () => {
    const root = tmp();
    const sb = new WorkspaceScopedSandbox(cfg(root));
    const r = await sb.resolvePath("new_file.txt", "write");
    expect(typeof r).toBe("string");
    expect(r as string).toBe(join(root, "new_file.txt"));
  });

  it("read of missing in-workspace file resolves (not path_escape)", async () => {
    // Regression for #63: a Read of a not-yet-created file *inside* the
    // workspace must resolve via its canonicalized parent, not be
    // misclassified as path_escape. The file is absent; resolution still
    // succeeds so the read can surface a recoverable not-found.
    const root = tmp();
    const sb = new WorkspaceScopedSandbox(cfg(root));
    const r = await sb.resolvePath("output.txt", "read");
    expect(typeof r).toBe("string");
    expect(r as string).toBe(join(root, "output.txt"));
  });

  it("read of missing file in existing subdir resolves", async () => {
    const root = tmp();
    mkdirSync(join(root, "sub"));
    const sb = new WorkspaceScopedSandbox(cfg(root));
    const r = await sb.resolvePath("sub/missing.txt", "read");
    expect(typeof r).toBe("string");
    expect(r as string).toBe(join(root, "sub", "missing.txt"));
  });

  it("read of missing file outside root still path_escape", async () => {
    // Regression for #63: a Read of a *non-existent* path that resolves
    // outside the workspace root must still be a path_escape, not a
    // not-found. (`..` makes the canonicalized parent escape the root.)
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    const r = await sb.resolvePath("../nonexistent_passwd", "read");
    expect((r as SandboxViolation).kind).toBe("path_escape");
  });

  it("read of existing in-workspace file resolves", async () => {
    const root = tmp();
    writeFileSync(join(root, "present.txt"), "hi");
    const sb = new WorkspaceScopedSandbox(cfg(root));
    const r = await sb.resolvePath("present.txt", "read");
    expect(typeof r).toBe("string");
    expect(r as string).toBe(join(root, "present.txt"));
  });

  it("disallowed_command for bubblewrap mode", async () => {
    const root = tmp();
    const mode: IsolationMode = {
      kind: "bubblewrap",
      profile: { name: "default" },
    };
    const sb = new WorkspaceScopedSandbox(cfg(root), mode);
    const r = await sb.executeCommand("echo", ["hi"]);
    expect((r as SandboxViolation).kind).toBe("disallowed_command");
  });
});

describe("executeCommand", () => {
  const IS_UNIX = process.platform !== "win32";
  it.runIf(IS_UNIX)("echo returns stdout and exit 0", async () => {
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    const out = await sb.executeCommand("echo", ["hello"]);
    if ("kind" in out) throw new Error("unexpected violation");
    expect(out.exit_code).toBe(0);
    expect(out.stdout).toContain("hello");
    expect(out.timed_out).toBe(false);
    expect(out.truncated).toBe(false);
  });

  it.runIf(IS_UNIX)(
    "timeout returns timed_out=true",
    async () => {
      const sb = new WorkspaceScopedSandbox(cfg(tmp()));
      const out = await sb.executeCommand("sleep", ["5"], null, 50);
      if ("kind" in out) throw new Error("unexpected violation");
      expect(out.timed_out).toBe(true);
    },
    15000,
  );
});

describe("handleLargeOutput", () => {
  it("below threshold returns content untouched", async () => {
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    const out = await sb.handleLargeOutput("short content", "c1", 100, 100);
    expect(out.truncated).toBe(false);
    expect(out.full_ref).toBeNull();
    expect(out.content).toBe("short content");
    expect(out.original_size).toBe("short content".length);
  });

  it("above threshold offloads to .spore/offload/", async () => {
    const root = tmp();
    const sb = new WorkspaceScopedSandbox(cfg(root));
    const content = "x".repeat(10_000);
    const out = await sb.handleLargeOutput(content, "call-1", 10, 10);
    expect(out.truncated).toBe(true);
    expect(out.content).toContain("...[truncated]...");
    expect(out.full_ref).not.toBeNull();
    if (out.full_ref == null) throw new Error("unreachable");
    expect(out.full_ref.size).toBe(content.length);
    expect(out.full_ref.path).toContain(".spore");
  });
});

describe("SandboxViolation variants — discriminated union", () => {
  // Compile-time exhaustiveness lives in the union. This test just makes
  // sure each variant constructs and round-trips through JSON.
  const cases: SandboxViolation[] = [
    { kind: "path_escape", path: "/p" },
    { kind: "path_denied", path: "/p", matched_rule: "r" },
    { kind: "extension_denied", path: "/p.env", extension: "env" },
    { kind: "read_only_violation", path: "/p" },
    { kind: "file_size_exceeded", path: "/p", size: 1024, limit: 100 },
    { kind: "disallowed_command", command: "rm" },
    { kind: "network_violation", host: "evil" },
  ];
  for (const c of cases) {
    it(`round-trips ${c.kind}`, () => {
      const back = JSON.parse(JSON.stringify(c));
      expect(back).toEqual(c);
    });
  }
});
