/**
 * `WorkspaceScopedSandbox` unit tests — mirrors
 * `rust/crates/spore-core/src/sandbox.rs#tests`.
 */

import {
  mkdtempSync,
  realpathSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  chmodSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  BuildError,
  WorkspaceScopedSandbox,
  sandboxViolationIsAlwaysHalt,
  type CommandOutput,
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

  it("defaults to workspace_scoped isolation (None is dangerous-only, issue #34)", () => {
    const root = tmp();
    const sb = new WorkspaceScopedSandbox(cfg(root));
    expect(sb.isolationMode()).toEqual({ kind: "workspace_scoped" });
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
    // SC-15: typed spawn failure.
    {
      kind: "exec_spawn_failed",
      command: "no-such-bin",
      message: "spawn no-such-bin ENOENT",
    },
  ];
  for (const c of cases) {
    it(`round-trips ${c.kind}`, () => {
      const back = JSON.parse(JSON.stringify(c));
      expect(back).toEqual(c);
    });
  }
});

// --------------------------------------------------------------------------
// SC-15 — typed spawn failure (not a fake exit_code: -1)
// --------------------------------------------------------------------------

describe("executeCommand spawn failure (SC-15)", () => {
  const IS_UNIX = process.platform !== "win32";

  it.runIf(IS_UNIX)("a missing binary yields exec_spawn_failed, never halt-eligible", async () => {
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    const out = await sb.executeCommand("spore-definitely-no-such-binary-xyz", []);
    expect("kind" in out).toBe(true);
    const v = out as SandboxViolation;
    expect(v.kind).toBe("exec_spawn_failed");
    if (v.kind === "exec_spawn_failed") {
      expect(v.command).toBe("spore-definitely-no-such-binary-xyz");
      expect(typeof v.message).toBe("string");
    }
    // Layer-2: always recoverable feedback, never halt-eligible.
    expect(sandboxViolationIsAlwaysHalt(v)).toBe(false);
  });

  // SC-15 breadth parity (#164): a start failure whose errno is NOT ENOENT/EACCES
  // must still become exec_spawn_failed (Rust/Python/Go convert ANY start error).
  // An empty 0755 file is "executable" to the OS but not a valid binary, so
  // exec(2) fails with ENOEXEC — which Node raises as a SYNCHRONOUS spawn() throw
  // (not the async `"error"` event). The handler converts that throw too.
  it.runIf(IS_UNIX)("a non-ENOENT start failure (ENOEXEC) yields exec_spawn_failed", async () => {
    const root = tmp();
    const notABinary = join(root, "not-a-binary");
    writeFileSync(notABinary, "");
    chmodSync(notABinary, 0o755);
    const out = await sb_run(root, notABinary);
    expect("kind" in out).toBe(true);
    const v = out as SandboxViolation;
    expect(v.kind).toBe("exec_spawn_failed");
    if (v.kind === "exec_spawn_failed") {
      expect(v.command).toBe(notABinary);
      // Prove the broadening: the errno is neither ENOENT nor EACCES.
      expect(v.message).not.toContain("ENOENT");
      expect(v.message).not.toContain("EACCES");
    }
    expect(sandboxViolationIsAlwaysHalt(v)).toBe(false);
  });

  it.runIf(IS_UNIX)(
    "a timeout keeps the legacy CommandOutput { exit_code: -1, timed_out }",
    async () => {
      const sb = new WorkspaceScopedSandbox(cfg(tmp()));
      const out = await sb.executeCommand("sleep", ["5"], null, 50);
      // A real run that exceeded the clock is NOT converted to exec_spawn_failed.
      if ("kind" in out) throw new Error("timeout must not be a spawn failure");
      expect(out.timed_out).toBe(true);
      expect(out.exit_code).toBe(-1);
    },
    15000,
  );
});

/** Run `command` through a sandbox rooted at `root`. */
async function sb_run(root: string, command: string): Promise<CommandOutput | SandboxViolation> {
  const sb = new WorkspaceScopedSandbox({ root });
  return sb.executeCommand(command, []);
}

// --------------------------------------------------------------------------
// SC-12 — ExecConfig exec-hardening knobs
// --------------------------------------------------------------------------

describe("ExecConfig exec-hardening (SC-12)", () => {
  const IS_UNIX = process.platform !== "win32";

  function execCfg(root: string, exec_config: WorkspaceConfig["exec_config"]): WorkspaceConfig {
    return { root, exec_config };
  }

  it.runIf(IS_UNIX)(
    "default_timeout applies when the per-call timeout is absent",
    async () => {
      const sb = new WorkspaceScopedSandbox(execCfg(tmp(), { default_timeout: 50 }));
      // No per-call timeout — the exec_config floor must still fire.
      const out = await sb.executeCommand("sleep", ["5"], null, null);
      if ("kind" in out) throw new Error("unexpected violation");
      expect(out.timed_out).toBe(true);
    },
    15000,
  );

  it.runIf(IS_UNIX)(
    "per-call timeout overrides a generous default_timeout",
    async () => {
      const sb = new WorkspaceScopedSandbox(execCfg(tmp(), { default_timeout: 30_000 }));
      const out = await sb.executeCommand("sleep", ["5"], null, 50);
      if ("kind" in out) throw new Error("unexpected violation");
      expect(out.timed_out).toBe(true);
    },
    15000,
  );

  it.runIf(IS_UNIX)("non_interactive_env is injected onto the child", async () => {
    const sb = new WorkspaceScopedSandbox(
      execCfg(tmp(), { non_interactive_env: { SPORE_SC12_ENV: "hardened" } }),
    );
    const out = await sb.executeCommand("/bin/sh", ["-c", "echo $SPORE_SC12_ENV"]);
    if ("kind" in out) throw new Error("unexpected violation");
    expect(out.exit_code).toBe(0);
    expect(out.stdout).toContain("hardened");
  });

  it.runIf(IS_UNIX)(
    "close_stdin yields prompt EOF (cat exits 0 instead of hanging)",
    async () => {
      const sb = new WorkspaceScopedSandbox(execCfg(tmp(), { close_stdin: true }));
      // `cat` with no args reads stdin to EOF; with stdin redirected to the null
      // device it returns immediately. A generous per-call timeout guards the
      // test against a hang if the knob regressed.
      const out = await sb.executeCommand("cat", [], null, 5000);
      if ("kind" in out) throw new Error("unexpected violation");
      expect(out.timed_out).toBe(false);
      expect(out.exit_code).toBe(0);
      expect(out.stdout).toBe("");
    },
    15000,
  );

  it.runIf(IS_UNIX)(
    "kill_on_drop reaps a child on timeout (sentinel never created)",
    async () => {
      const root = tmp();
      const sentinel = join(root, "kod_sentinel");
      const sb = new WorkspaceScopedSandbox(execCfg(root, { kill_on_drop: true }));
      // The shell sleeps, then would `touch` the sentinel. The 100ms timeout
      // aborts the exec; with kill_on_drop the shell is killed before it can run
      // `touch`, so the sentinel never appears.
      const out = await sb.executeCommand(
        "/bin/sh",
        ["-c", `sleep 1; touch ${sentinel}`],
        root,
        100,
      );
      if ("kind" in out) throw new Error("unexpected violation");
      expect(out.timed_out).toBe(true);
      // Wait past when the un-killed shell would have created the sentinel.
      await new Promise((r) => setTimeout(r, 1500));
      expect(existsSync(sentinel)).toBe(false);
    },
    15000,
  );

  it.runIf(IS_UNIX)("unset exec_config preserves legacy behavior", async () => {
    const sb = new WorkspaceScopedSandbox(cfg(tmp()));
    // No implicit timeout: a quick echo runs to completion, un-timed.
    const out = await sb.executeCommand("echo", ["ok"]);
    if ("kind" in out) throw new Error("unexpected violation");
    expect(out.exit_code).toBe(0);
    expect(out.stdout).toContain("ok");
    expect(out.timed_out).toBe(false);
  });
});

// --------------------------------------------------------------------------
// SC-13 — read-everywhere / write-scoped write_root
// --------------------------------------------------------------------------

describe("write_root read-everywhere / write-scoped (SC-13)", () => {
  it("read of a file under root but outside write_root succeeds; write is a PathEscape", async () => {
    const root = tmp();
    const out = join(root, "out");
    mkdirSync(out);
    writeFileSync(join(root, "secret.txt"), "s");
    const sb = new WorkspaceScopedSandbox({ root, write_root: out });

    // Read-everywhere: a file under `root` but outside `write_root` resolves.
    const read = await sb.resolvePath("secret.txt", "read");
    expect(typeof read).toBe("string");

    // Write-scoped: writing that same file is a PathEscape.
    const write = await sb.resolvePath("secret.txt", "write");
    expect((write as SandboxViolation).kind).toBe("path_escape");

    // A write under `write_root` (path strings stay root-relative) is OK.
    const ok = await sb.resolvePath("out/result.txt", "write");
    expect(typeof ok).toBe("string");
  });

  it("with no write_root, writes resolve anywhere under root (legacy)", async () => {
    const root = tmp();
    writeFileSync(join(root, "a.txt"), "x");
    const sb = new WorkspaceScopedSandbox(cfg(root));
    const write = await sb.resolvePath("a.txt", "write");
    expect(typeof write).toBe("string");
  });

  it("constructing with a missing write_root throws write_root_not_found", () => {
    const root = tmp();
    let err: unknown;
    try {
      new WorkspaceScopedSandbox({ root, write_root: join(root, "does-not-exist") });
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(BuildError);
    expect((err as BuildError).kind).toBe("write_root_not_found");
  });
});
