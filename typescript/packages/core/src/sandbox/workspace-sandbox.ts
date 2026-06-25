/**
 * `WorkspaceScopedSandbox` — canonical `SandboxProvider` implementation
 * (spore-core issue #6).
 *
 * Enforces a workspace root with allow/deny lists, extension filters, a
 * read-only mode, and per-file size limits. Subprocesses are spawned
 * directly via `node:child_process`; large outputs are offloaded to
 * `{workspaceRoot}/.spore/offload/{callId}.txt`.
 *
 * Mirrors `rust/crates/spore-core/src/sandbox.rs`.
 */

import { spawn } from "node:child_process";
import { promises as fs, realpathSync, statSync, existsSync } from "node:fs";
import {
  basename,
  dirname,
  isAbsolute,
  join,
  relative,
  resolve as pathResolve,
  sep,
} from "node:path";

import type { ToolCall } from "../model/schemas.js";
import type {
  AnyIsolationMode,
  CommandOutput,
  ExecConfig,
  FileRef,
  IsolationMode,
  Operation,
  SandboxProvider,
  SandboxViolation,
  TruncatedOutput,
  WorkspaceConfig,
} from "../harness/types.js";

// ============================================================================
// BuildError
// ============================================================================

/** Construction-time errors for `WorkspaceScopedSandbox`. */
export class BuildError extends Error {
  readonly name = "BuildError";
  constructor(
    readonly kind:
      | "root_not_found"
      | "root_not_canonical"
      | "root_io"
      // SC-13: an optional `write_root` was supplied but does not exist / could
      // not be canonicalized.
      | "write_root_not_found"
      | "write_root_io",
    readonly path: string,
    message: string,
  ) {
    super(message);
  }
}

// ============================================================================
// Helpers
// ============================================================================

const CHARS_PER_TOKEN = 4;

/** Warning emitted when a sandbox is constructed with no isolation (issue #34). */
function warnNoneIsolation(): void {
  console.warn(
    "spore-core: WorkspaceScopedSandbox constructed with IsolationMode::None — " +
      "trusted-dev use only; do not enable silently in production",
  );
}

function stripExtDot(ext: string): string {
  return ext.startsWith(".") ? ext.slice(1) : ext;
}

function pathStartsWith(child: string, parent: string): boolean {
  if (child === parent) return true;
  const p = parent.endsWith(sep) ? parent : parent + sep;
  return child.startsWith(p);
}

function sanitizeCallId(id: string): string {
  let out = "";
  for (const c of id) {
    if (/[A-Za-z0-9_-]/.test(c)) out += c;
    else out += "_";
  }
  return out;
}

/** Match either the extension or, for dotfiles (`.env`), the dotless name. */
function extensionMatches(canonical: string, denied: string): boolean {
  const trimmed = stripExtDot(denied).toLowerCase();
  const name = basename(canonical);
  const dotIdx = name.lastIndexOf(".");
  if (dotIdx > 0) {
    const ext = name.slice(dotIdx + 1).toLowerCase();
    if (ext === trimmed) return true;
  }
  if (name.startsWith(".") && !name.slice(1).includes(".")) {
    if (name.slice(1).toLowerCase() === trimmed) return true;
  }
  return false;
}

// ============================================================================
// WorkspaceScopedSandbox
// ============================================================================

export class WorkspaceScopedSandbox implements SandboxProvider {
  private readonly root: string;
  private readonly allowedPaths: string[];
  private readonly deniedPaths: string[];
  private readonly deniedExtensions: string[];
  private readonly readOnly: boolean;
  private readonly maxFileSize: number;
  private readonly mode: AnyIsolationMode;
  /** SC-13: canonical narrower write boundary, or `null` for legacy single-root. */
  private readonly writeRoot: string | null;
  /** SC-12: process-execution hardening knobs, or `null` for legacy behavior. */
  private readonly execConfig: ExecConfig | null;

  /**
   * Construct a sandbox with the given isolation mode.
   *
   * The public constructor only accepts a safe {@link IsolationMode}; the
   * dangerous `{ kind: "none" }` mode cannot be named here. To build a sandbox
   * with no isolation, use {@link unsafeWithMode} via the dangerous opt-in
   * entry point (`@spore/core/dangerous`). This mirrors the Rust `dangerous`
   * feature gate on `IsolationMode::None` (issue #34).
   */
  /**
   * Internal factory that accepts {@link AnyIsolationMode}, including the
   * dangerous `{ kind: "none" }`. Reached only through the dangerous entry
   * point (`@spore/core/dangerous`); not part of the default public API.
   */
  static unsafeWithMode(config: WorkspaceConfig, mode: AnyIsolationMode): WorkspaceScopedSandbox {
    // The constructor only accepts a safe IsolationMode; build with a safe
    // placeholder, then overwrite the private mode field with the dangerous
    // value. This keeps a single root-validation path while letting the
    // dangerous opt-in reach `{ kind: "none" }`.
    const sb = new WorkspaceScopedSandbox(config);
    (sb as unknown as { mode: AnyIsolationMode }).mode = mode;
    if (mode.kind === "none") warnNoneIsolation();
    return sb;
  }

  constructor(config: WorkspaceConfig, mode?: IsolationMode) {
    // Validate root exists, canonicalize.
    let canonical: string;
    try {
      if (!existsSync(config.root)) {
        throw new BuildError(
          "root_not_found",
          config.root,
          `workspace root does not exist: ${config.root}`,
        );
      }
      canonical = realpathSync(config.root);
    } catch (e) {
      if (e instanceof BuildError) throw e;
      throw new BuildError(
        "root_io",
        config.root,
        `workspace root io error: ${e instanceof Error ? e.message : String(e)}`,
      );
    }

    this.root = canonical;
    // Resolve allow/deny entries to absolute paths under the root.
    const toAbs = (p: string): string =>
      isAbsolute(p) ? pathResolve(p) : pathResolve(canonical, p);
    this.allowedPaths = (config.allowed_paths ?? []).map(toAbs);
    this.deniedPaths = (config.denied_paths ?? []).map(toAbs);
    this.deniedExtensions = config.denied_extensions ?? [];
    this.readOnly = config.read_only ?? false;
    this.maxFileSize = config.max_file_size ?? 0;
    // The public constructor only accepts a safe IsolationMode, so a default
    // build defaults to workspace-scoped isolation (issue #34). The dangerous
    // `{ kind: "none" }` mode is injected by `unsafeWithMode`, which emits the
    // warning itself.
    this.mode = mode ?? { kind: "workspace_scoped" };

    // SC-12: process-execution hardening knobs (or null for legacy behavior).
    this.execConfig = config.exec_config ?? null;

    // SC-13: canonicalize the optional write_root the same way as the root so
    // the boundary check compares canonical paths. It must exist — a missing or
    // unreadable write_root is a construction error.
    if (config.write_root != null) {
      try {
        if (!existsSync(config.write_root)) {
          throw new BuildError(
            "write_root_not_found",
            config.write_root,
            `workspace write_root does not exist: ${config.write_root}`,
          );
        }
        this.writeRoot = realpathSync(config.write_root);
      } catch (e) {
        if (e instanceof BuildError) throw e;
        throw new BuildError(
          "write_root_io",
          config.write_root,
          `workspace write_root io error: ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    } else {
      this.writeRoot = null;
    }
  }

  // --------------------------------------------------------------------------
  // SandboxProvider methods
  // --------------------------------------------------------------------------

  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }

  isolationMode(): AnyIsolationMode {
    return this.mode;
  }

  workspaceRoot(): string {
    return this.root;
  }

  async resolvePath(path: string, operation: Operation): Promise<string | SandboxViolation> {
    return this.resolveSync(path, operation);
  }

  /** Synchronous core; used by `resolvePath` and tests. */
  resolveSync(raw: string, operation: Operation): string | SandboxViolation {
    // 1. Join root + raw_path; treat absolute paths as relative-to-root.
    let joined: string;
    if (isAbsolute(raw)) {
      // Strip leading separator(s) so `/foo` becomes `<root>/foo`.
      let stripped = raw;
      while (stripped.startsWith(sep) || stripped.startsWith("/")) {
        stripped = stripped.slice(1);
      }
      joined = pathResolve(this.root, stripped);
    } else {
      joined = pathResolve(this.root, raw);
    }

    // 2. Canonicalize (resolves .., symlinks). The target file may not yet
    //    exist — for *any* operation, including Read — so canonicalize the
    //    parent and re-join the leaf. Resolution is operation-agnostic on
    //    purpose: existence is orthogonal to the boundary check. A missing
    //    in-workspace path still resolves (via its canonicalized parent) and
    //    passes the boundary check; the actual read then naturally fails with
    //    a not-found that the read tool surfaces as a recoverable error rather
    //    than a PathEscape. A missing path that resolves *outside* the root is
    //    still a PathEscape.
    let canonical: string;
    try {
      canonical = realpathSync(joined);
    } catch (e) {
      const enoent = (e as NodeJS.ErrnoException).code === "ENOENT" || !existsSync(joined);
      if (enoent) {
        const parent = dirname(joined);
        const leaf = basename(joined);
        if (!parent || !leaf) {
          return { kind: "path_escape", path: raw };
        }
        try {
          const parentCanon = realpathSync(parent);
          canonical = join(parentCanon, leaf);
        } catch {
          return { kind: "path_escape", path: raw };
        }
      } else {
        return { kind: "path_escape", path: raw };
      }
    }

    // 3. Boundary check. Reads and execute must stay under the read `root`;
    //    writes must stay under `writeRoot` when set (SC-13: read-everywhere,
    //    write-scoped). The path string was already joined onto `root` above —
    //    `writeRoot` only narrows where the resolved write target may land, it
    //    is NOT a separate join base — so a write under `root` but outside
    //    `writeRoot` is a PathEscape.
    const boundary = operation === "write" && this.writeRoot != null ? this.writeRoot : this.root;
    if (!pathStartsWith(canonical, boundary)) {
      return { kind: "path_escape", path: canonical };
    }

    // 4. Denylist.
    for (const denied of this.deniedPaths) {
      if (pathStartsWith(canonical, denied)) {
        return {
          kind: "path_denied",
          path: canonical,
          matched_rule: denied,
        };
      }
    }

    // 5. Allowlist (if non-empty).
    if (this.allowedPaths.length > 0) {
      const allowed = this.allowedPaths.some((a) => pathStartsWith(canonical, a));
      if (!allowed) {
        return {
          kind: "path_denied",
          path: canonical,
          matched_rule: "not in allowlist",
        };
      }
    }

    // 6. Denied extensions.
    for (const denied of this.deniedExtensions) {
      if (extensionMatches(canonical, denied)) {
        return {
          kind: "extension_denied",
          path: canonical,
          extension: stripExtDot(denied).toLowerCase(),
        };
      }
    }

    // 7. Read-only.
    if (this.readOnly && (operation === "write" || operation === "execute")) {
      return { kind: "read_only_violation", path: canonical };
    }

    // 8. File-size cap (read only).
    if (operation === "read" && this.maxFileSize > 0) {
      try {
        const st = statSync(canonical);
        if (st.isFile() && st.size > this.maxFileSize) {
          return {
            kind: "file_size_exceeded",
            path: canonical,
            size: st.size,
            limit: this.maxFileSize,
          };
        }
      } catch {
        // File may not exist — that's fine for Read; let the caller fail.
      }
    }

    return canonical;
  }

  async executeCommand(
    command: string,
    args: readonly string[],
    cwd?: string | null,
    timeoutMs?: number | null,
    signal?: AbortSignal,
  ): Promise<CommandOutput | SandboxViolation> {
    switch (this.mode.kind) {
      case "none":
      case "workspace_scoped":
        break;
      case "bubblewrap":
        return {
          kind: "disallowed_command",
          command: `bubblewrap isolation not implemented: ${command}`,
        };
      case "docker":
        return {
          kind: "disallowed_command",
          command: `docker isolation not implemented: ${command}`,
        };
    }

    // SC-12: apply exec-hardening knobs when configured. The per-call timeout
    // always wins; `default_timeout` is the floor for callers that pass none.
    const ec = this.execConfig;
    const effectiveTimeoutMs = timeoutMs ?? ec?.default_timeout ?? null;
    // `non_interactive_env` is forced onto the inherited environment in sorted
    // key order (deterministic, mirroring Rust's BTreeMap iteration).
    let childEnv: NodeJS.ProcessEnv | undefined;
    if (ec?.non_interactive_env != null) {
      const nie = ec.non_interactive_env;
      childEnv = { ...process.env };
      for (const key of Object.keys(nie).sort()) {
        childEnv[key] = nie[key];
      }
    }
    // `close_stdin` redirects the child's stdin to the null device (`"ignore"`)
    // so an input-blocked command hits EOF instead of hanging.
    const stdio: ["ignore" | "inherit", "pipe", "pipe"] | undefined = ec?.close_stdin
      ? ["ignore", "pipe", "pipe"]
      : undefined;

    return new Promise<CommandOutput | SandboxViolation>((resolveFn) => {
      const controller = new AbortController();
      const child = spawn(command, args as string[], {
        cwd: cwd ?? this.root,
        shell: false,
        signal: controller.signal,
        ...(childEnv != null ? { env: childEnv } : {}),
        ...(stdio != null ? { stdio } : {}),
      });

      let stdout = "";
      let stderr = "";
      let timedOut = false;
      let timer: NodeJS.Timeout | null = null;
      let aborted = false;

      const cleanup = (): void => {
        if (timer) clearTimeout(timer);
        if (signal) signal.removeEventListener("abort", onAbort);
      };
      // SC-12: `kill_on_drop` reaps the child when the exec is cancelled/timed
      // out. `controller.abort()` already kills the spawned process; with the
      // knob set we additionally SIGKILL to guarantee the child cannot outlive
      // the aborted exec (instead of being orphaned).
      const abortChild = (): void => {
        controller.abort();
        if (ec?.kill_on_drop) {
          try {
            child.kill("SIGKILL");
          } catch {
            // Child may already be gone — nothing to reap.
          }
        }
      };
      const onAbort = (): void => {
        aborted = true;
        abortChild();
      };
      if (signal) {
        if (signal.aborted) onAbort();
        else signal.addEventListener("abort", onAbort, { once: true });
      }
      if (effectiveTimeoutMs != null && effectiveTimeoutMs > 0) {
        timer = setTimeout(() => {
          timedOut = true;
          abortChild();
        }, effectiveTimeoutMs);
      }

      child.stdout?.on("data", (chunk: Buffer) => {
        stdout += chunk.toString("utf8");
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        stderr += chunk.toString("utf8");
      });
      child.on("error", (err) => {
        cleanup();
        // SC-15: a genuine spawn failure (the binary never started) is a typed
        // violation, not a fake `exit_code: -1` success. A timeout/abort is a
        // real run that exceeded the clock / was cancelled — it KEEPS the legacy
        // `CommandOutput { exit_code: -1, timed_out }` (only never-started spawns
        // become the violation). Callers already narrow the union.
        const code = (err as NodeJS.ErrnoException).code;
        if (!timedOut && !aborted && (code === "ENOENT" || code === "EACCES")) {
          resolveFn({ kind: "exec_spawn_failed", command, message: String(err) });
          return;
        }
        resolveFn({
          stdout,
          stderr: timedOut
            ? `command timed out after ${Math.round((effectiveTimeoutMs ?? 0) / 1000)}s`
            : stderr + (stderr ? "" : String(err)),
          exit_code: -1,
          timed_out: timedOut,
          truncated: false,
        });
      });
      child.on("close", (code, sig) => {
        cleanup();
        let exit = code ?? -1;
        if (sig) exit = -1;
        resolveFn({
          stdout,
          stderr,
          exit_code: exit,
          timed_out: timedOut || aborted,
          truncated: false,
        });
      });
    });
  }

  async handleLargeOutput(
    content: string,
    callId: string,
    headTokens: number,
    tailTokens: number,
  ): Promise<TruncatedOutput> {
    const headChars = Math.max(0, headTokens) * CHARS_PER_TOKEN;
    const tailChars = Math.max(0, tailTokens) * CHARS_PER_TOKEN;
    const originalSize = content.length;

    if (originalSize <= headChars + tailChars) {
      return {
        content,
        truncated: false,
        full_ref: null,
        original_size: originalSize,
      };
    }

    const head = content.slice(0, headChars);
    const tail = content.slice(originalSize - tailChars);
    const snippet = `${head}\n...[truncated]...\n${tail}`;

    // Offload the full content.
    const offloadDir = join(this.root, ".spore", "offload");
    let fullRef: FileRef | null = null;
    try {
      await fs.mkdir(offloadDir, { recursive: true });
      const safeId = sanitizeCallId(callId);
      const offloadPath = join(offloadDir, `${safeId}.txt`);
      await fs.writeFile(offloadPath, content, "utf8");
      fullRef = { path: offloadPath, size: originalSize };
    } catch {
      fullRef = null;
    }

    return {
      content: snippet,
      truncated: true,
      full_ref: fullRef,
      original_size: originalSize,
    };
  }
}

// Re-export for callers that prefer to import path helpers alongside.
export { relative as _relative };
