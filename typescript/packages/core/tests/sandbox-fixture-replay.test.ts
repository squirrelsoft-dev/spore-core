/**
 * Cross-language fixture replay for `WorkspaceScopedSandbox` (issue #6).
 *
 * Each fixture lives in `/fixtures/sandbox_violations/*.json`. We create a
 * `TempDir`, treat it as the workspace root (the fixture's literal
 * `workspace_root` is descriptive only), materialize the `filesystem`
 * under it, then resolve `raw_path` against the freshly built sandbox.
 *
 * Mirrors the Rust `sandbox_violation_fixtures` test.
 */

import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  WorkspaceScopedSandbox,
  type Operation,
  type SandboxViolation,
  type WorkspaceConfig,
} from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = resolve(here, "../../../../fixtures/sandbox_violations");

interface FsEntry {
  dir?: boolean;
  file?: string;
}

interface Scenario {
  name: string;
  workspace_root: string;
  raw_path: string;
  filesystem: Record<string, FsEntry>;
  allowed_paths: string[];
  denied_paths: string[];
  denied_extensions: string[];
  read_only: boolean;
  max_file_size: number;
  operation: Operation;
  expected: { kind: string };
}

function loadScenarios(): Scenario[] {
  const out: Scenario[] = [];
  for (const name of readdirSync(fixturesRoot)) {
    if (!name.endsWith(".json")) continue;
    const raw = readFileSync(join(fixturesRoot, name), "utf8");
    out.push(JSON.parse(raw) as Scenario);
  }
  return out;
}

function materialize(root: string, filesystem: Record<string, FsEntry>): void {
  // Sort to ensure parent directories are created before files.
  const keys = Object.keys(filesystem).sort();
  for (const rel of keys) {
    const target = join(root, rel);
    const entry = filesystem[rel]!;
    if (entry.dir) {
      mkdirSync(target, { recursive: true });
    } else if (entry.file != null) {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, entry.file);
    }
  }
}

describe("fixture: sandbox_violations", () => {
  const scenarios = loadScenarios();

  it("loads at least one scenario", () => {
    expect(scenarios.length).toBeGreaterThan(0);
  });

  for (const sc of scenarios) {
    it(`${sc.name} -> ${sc.expected.kind}`, async () => {
      const root = realpathSync(mkdtempSync(join(tmpdir(), "spore-sandbox-fx-")));
      materialize(root, sc.filesystem);

      const cfg: WorkspaceConfig = {
        root,
        allowed_paths: sc.allowed_paths,
        denied_paths: sc.denied_paths,
        denied_extensions: sc.denied_extensions,
        read_only: sc.read_only,
        max_file_size: sc.max_file_size,
      };
      const sb = new WorkspaceScopedSandbox(cfg);
      const result = await sb.resolvePath(sc.raw_path, sc.operation);

      let kind: string;
      if (typeof result === "string") kind = "ok";
      else kind = (result as SandboxViolation).kind;
      expect(kind).toBe(sc.expected.kind);
    });
  }
});
