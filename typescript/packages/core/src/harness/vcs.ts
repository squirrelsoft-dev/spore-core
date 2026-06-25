/**
 * VcsProvider seam (issue #58 v2) — git-log reload for the Ralph loop strategy.
 *
 * The v1 Ralph reload (commit `0e1f407`) re-seeded each fresh context window
 * from `.spore/progress.json` + `.spore/feature_list.json` only; the spec's
 * "reload git log" step was deferred (decision B4) because there was no
 * hermetic, cross-language-testable seam for VCS reads. This module IS that
 * seam.
 *
 * It mirrors how {@link "./types.js".SandboxProvider} abstracts filesystem/shell
 * access: define an interface, ship a real implementation ({@link GitVcsProvider})
 * and a deterministic fixture double ({@link "./testing.js".FixtureVcsProvider}),
 * and inject the chosen one at construction via
 * {@link "./standard.js".HarnessBuilder.vcsProvider}. The harness holds the
 * provider exactly as it does for every other component.
 *
 * `ralph` calls {@link VcsProvider.log} during its reload phase and injects the
 * output into the next window's seed as a clearly delimited "Recent VCS history:"
 * section. When NO provider is wired (the default) the git-log section is OMITTED
 * and Ralph behaves byte-for-byte like v1 — this is the B4→none decision.
 */

import type { CommandOutput, SandboxProvider, SandboxViolation } from "./types.js";

/**
 * Parameters shaping a {@link VcsProvider.log} read. Each field maps to a
 * `git log` flag in {@link GitVcsProvider}:
 *   - `maxEntries` → `-n <N>` (cap the number of commits returned),
 *   - `sinceRef`   → `<ref>..` (only commits AFTER `<ref>`),
 *   - `format`     → `--format=<fmt>` (custom pretty format).
 */
export interface VcsLogArgs {
  /** Maximum number of commits to return (`git log -n <maxEntries>`). */
  maxEntries: number;
  /** Only commits reachable after this ref (`git log <sinceRef>..`). Omitted
   *  returns the full history (subject to `maxEntries`). */
  sinceRef?: string;
  /** Custom `git log --format=<format>` string. Omitted uses git's default
   *  formatting. */
  format?: string;
}

/**
 * Read-only VCS abstraction the `ralph` loop strategy uses to reload git
 * history between context windows (issue #58 v2, decision B4). Both methods
 * reject with a {@link VcsError} on failure, per `typescript/CONVENTIONS.md`.
 */
export interface VcsProvider {
  /**
   * Return the project's commit log, shaped by `args`. The returned string is
   * the verbatim VCS output (e.g. `git log` stdout); the caller does not parse
   * it, it is injected into the reloaded context block as-is.
   */
  log(args: VcsLogArgs): Promise<string>;

  /** Return the working-tree status (e.g. `git status` stdout), verbatim. */
  status(): Promise<string>;

  /**
   * Revert the working tree to the last known-good state (SC-14). Called by the
   * HillClimbing loop to discard a no-improvement iteration's changes. OPTIONAL
   * (the additive-default analogue of Rust's defaulted trait method): a provider
   * that has no concept of revert — or whose workspace is not under version
   * control — simply omits it, and HillClimbing falls back to its hardcoded git
   * reset. {@link GitVcsProvider} implements it with `git reset --hard HEAD`
   * THROUGH the sandbox; a consumer climbing a non-git workspace (or one whose
   * "checkpoint" means something else) supplies its own. Best-effort: an error
   * is surfaced but does not halt the climb.
   */
  revert?(): Promise<void>;
}

/**
 * Error raised by a {@link VcsProvider}. Follows `typescript/CONVENTIONS.md`:
 * a class extending `Error` with `name` set to the class name and a discriminant
 * `kind` field for `switch` exhaustiveness. Mirrors the Rust `VcsError` variants
 * (`CommandFailed`, `Sandbox`) under a shared {@link VcsError} base.
 */
export abstract class VcsError extends Error {
  abstract readonly kind: "command_failed" | "sandbox";
}

/** The underlying VCS command failed (non-zero exit), carrying captured stderr. */
export class VcsCommandFailedError extends VcsError {
  readonly kind = "command_failed" as const;
  constructor(readonly stderr: string) {
    super(`vcs command failed: ${stderr}`);
    this.name = "VcsCommandFailedError";
  }
}

/** The VCS command was blocked or could not be spawned by the sandbox. */
export class VcsSandboxError extends VcsError {
  readonly kind = "sandbox" as const;
  constructor(readonly violation: SandboxViolation) {
    super(`vcs command blocked by sandbox: ${JSON.stringify(violation)}`);
    this.name = "VcsSandboxError";
  }
}

/** Discriminates a {@link CommandOutput} from a {@link SandboxViolation} — only
 *  the former carries `exit_code`. Mirrors the verifier's `isCommandOutput`. */
function isCommandOutput(v: CommandOutput | SandboxViolation): v is CommandOutput {
  return typeof v === "object" && v !== null && "exit_code" in v;
}

/**
 * Real {@link VcsProvider} that shells out to `git` THROUGH a
 * {@link SandboxProvider} (issue #58 v2). It wraps the sandbox and calls
 * {@link SandboxProvider.executeCommand} — it never bypasses sandboxing to spawn
 * `git` directly. The command line is built from {@link VcsLogArgs} (see that
 * type for the flag mapping); {@link status} runs `git status`. All commands run
 * in `workspaceRoot`.
 */
export class GitVcsProvider implements VcsProvider {
  constructor(
    private readonly sandbox: SandboxProvider,
    private readonly workspaceRoot: string,
  ) {}

  /**
   * Build the `git log` argument vector from `args` (static so the flag mapping
   * can be tested independently of process execution).
   */
  static logArgs(args: VcsLogArgs): string[] {
    const out = ["log", "-n", String(args.maxEntries)];
    if (args.format != null) {
      out.push(`--format=${args.format}`);
    }
    if (args.sinceRef != null) {
      out.push(`${args.sinceRef}..`);
    }
    return out;
  }

  async log(args: VcsLogArgs): Promise<string> {
    return this.run(GitVcsProvider.logArgs(args));
  }

  async status(): Promise<string> {
    return this.run(["status"]);
  }

  /**
   * Revert the working tree with `git reset --hard HEAD` THROUGH the sandbox
   * (SC-14) — never spawned directly. Relocated from the harness's hardcoded
   * HillClimbing revert. Reuses {@link run}, which rejects with a {@link VcsError}
   * on a sandbox refusal or non-zero exit.
   */
  async revert(): Promise<void> {
    await this.run(["reset", "--hard", "HEAD"]);
  }

  private async run(argv: string[]): Promise<string> {
    if (this.sandbox.executeCommand == null) {
      throw new VcsCommandFailedError("sandbox does not support executeCommand");
    }
    const result = await this.sandbox.executeCommand("git", argv, this.workspaceRoot, null);
    if (!isCommandOutput(result)) {
      throw new VcsSandboxError(result);
    }
    if (result.exit_code !== 0) {
      throw new VcsCommandFailedError(result.stderr);
    }
    return result.stdout;
  }
}
