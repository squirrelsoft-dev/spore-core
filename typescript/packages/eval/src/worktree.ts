/**
 * Workspace restore/teardown (Rules 2-3).
 *
 * Each task run gets a fresh workspace restored from its
 * {@link WorkspaceSnapshot}; it is torn down after the run regardless of
 * outcome. `files` writes the map into a fresh temp dir; `git_ref` adds a
 * worktree from a source repo; `empty` is a bare temp dir.
 */

import { spawn } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { EvalError, type WorkspaceSnapshot } from "./task.js";

/**
 * A live, restored workspace. Call {@link teardown} after the run to remove the
 * directory tree (Rule 3); the harness does this regardless of run outcome.
 */
export class Workspace {
  private constructor(readonly path: string) {}

  /** Restore a fresh workspace from a snapshot (Rule 2). */
  static async restore(snapshot: WorkspaceSnapshot): Promise<Workspace> {
    let dir: string;
    try {
      dir = await mkdtemp(join(tmpdir(), "spore-eval-"));
    } catch (e) {
      throw EvalError.io(`failed to create temp dir: ${(e as Error).message}`);
    }
    switch (snapshot.kind) {
      case "empty":
        break;
      case "files":
        for (const [rel, contents] of Object.entries(snapshot.files)) {
          const path = join(dir, rel);
          try {
            await mkdir(dirname(path), { recursive: true });
            await writeFile(path, contents);
          } catch (e) {
            throw EvalError.io(
              `failed to write ${rel}: ${(e as Error).message}`,
            );
          }
        }
        break;
      case "git_ref":
        await restoreGit(dir, snapshot.repo, snapshot.reference);
        break;
      default: {
        const _exhaustive: never = snapshot;
        return _exhaustive;
      }
    }
    return new Workspace(dir);
  }

  /** Tear down the workspace directory tree (Rule 3). Idempotent. */
  async teardown(): Promise<void> {
    await rm(this.path, { recursive: true, force: true });
  }
}

/** Restore from a git ref by adding a worktree from `repo` at `reference`. */
async function restoreGit(
  dest: string,
  repo: string,
  reference: string,
): Promise<void> {
  await runGit(["-C", repo, "worktree", "add", "--detach", dest, reference]);
}

function runGit(args: readonly string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn("git", [...args]);
    let stderr = "";
    child.stderr?.on("data", (d) => (stderr += d.toString()));
    child.on("error", (e) =>
      reject(EvalError.worktree(`git spawn failed: ${(e as Error).message}`)),
    );
    child.on("close", (code) => {
      if (code === 0) resolve();
      else
        reject(
          EvalError.worktree(`git ${JSON.stringify(args)} failed: ${stderr}`),
        );
    });
  });
}
