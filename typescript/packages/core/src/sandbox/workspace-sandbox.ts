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
  CommandOutput,
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
    readonly kind: "root_not_found" | "root_not_canonical" | "root_io",
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
  private readonly mode: IsolationMode;

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
    this.mode = mode ?? { kind: "workspace_scoped" };

    if (this.mode.kind === "none") {
      console.warn(
        "spore-core: WorkspaceScopedSandbox constructed with IsolationMode::None — " +
          "trusted-dev use only; do not enable silently in production",
      );
    }
  }

  // --------------------------------------------------------------------------
  // SandboxProvider methods
  // --------------------------------------------------------------------------

  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }

  isolationMode(): IsolationMode {
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

    // 2. Canonicalize (resolves .., symlinks). For Write/Execute on a
    //    non-existent file, canonicalize the parent and re-join the leaf.
    let canonical: string;
    try {
      canonical = realpathSync(joined);
    } catch (e) {
      const enoent = (e as NodeJS.ErrnoException).code === "ENOENT" || !existsSync(joined);
      if (enoent && (operation === "write" || operation === "execute")) {
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

    // 3. Boundary check.
    if (!pathStartsWith(canonical, this.root)) {
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

    return new Promise<CommandOutput>((resolveFn) => {
      const controller = new AbortController();
      const child = spawn(command, args as string[], {
        cwd: cwd ?? this.root,
        shell: false,
        signal: controller.signal,
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
      const onAbort = (): void => {
        aborted = true;
        controller.abort();
      };
      if (signal) {
        if (signal.aborted) onAbort();
        else signal.addEventListener("abort", onAbort, { once: true });
      }
      if (timeoutMs != null && timeoutMs > 0) {
        timer = setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, timeoutMs);
      }

      child.stdout?.on("data", (chunk: Buffer) => {
        stdout += chunk.toString("utf8");
      });
      child.stderr?.on("data", (chunk: Buffer) => {
        stderr += chunk.toString("utf8");
      });
      child.on("error", (err) => {
        cleanup();
        resolveFn({
          stdout,
          stderr: timedOut
            ? `command timed out after ${Math.round((timeoutMs ?? 0) / 1000)}s`
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
