/**
 * Dangerous-gated suite (spore-core issue #34).
 *
 * Exercises the footguns that the default build cannot reach:
 *   - `Mode.Yolo` (full autonomy) via `@spore/core/dangerous`,
 *   - `IsolationMode.None` (no path enforcement) via `@spore/core/dangerous`,
 *   - the cross-language `dangerous.json` fixture, replayed ONLY here.
 *
 * It also asserts the gate itself: the dangerous constructors are NOT exported
 * from the default barrel, and a default-built sandbox / registry is
 * safe-by-default.
 */

import { mkdtempSync, readFileSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it, vi } from "vitest";

import {
  DANGEROUS_MODE_YOLO,
  composeWithYolo,
  dangerousStandardChunks,
  dangerousWorkspaceSandbox,
  noneIsolationMode,
  yoloApprovalPolicy,
  yoloDefaultToolPhase,
  yoloPromptChunk,
} from "../src/dangerous/index.js";
import * as defaultBarrel from "../src/index.js";
import { promptChunkRegistry as pcr } from "../src/index.js";

const { ChunkId, StandardPromptChunkRegistry, renderComposed, standardChunks } = pcr;

function tmp(): string {
  return realpathSync(mkdtempSync(join(tmpdir(), "spore-dangerous-")));
}

// ── Mode.Yolo (gated) ────────────────────────────────────────────────────────

describe("Mode.Yolo (dangerous)", () => {
  it("yoloPromptChunk renders the Yolo mode chunk", () => {
    const c = yoloPromptChunk();
    expect(c.id.value).toBe("mode-yolo");
    expect(c.slot).toBe("mode");
    expect(c.cache_block).toBe("static");
    expect(c.content).toContain("Mode: Yolo");
  });

  it("yoloApprovalPolicy is none and yoloDefaultToolPhase is execution", () => {
    expect(yoloApprovalPolicy()).toBe("none");
    expect(yoloDefaultToolPhase()).toBe("execution");
  });

  it("DANGEROUS_MODE_YOLO carries the canonical wire tag", () => {
    expect(DANGEROUS_MODE_YOLO).toBe("yolo");
  });

  it("composeWithYolo places the Yolo mode chunk after the role", () => {
    const r = new StandardPromptChunkRegistry();
    expect(
      r.register(pcr.promptChunk("role-test", "you are a test agent", "role", "static")),
    ).toBeNull();
    const res = composeWithYolo(r, new ChunkId("role-test"), [], []);
    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.composed.chunks.map((c) => ({ slot: c.slot, id: c.id.value }))).toEqual([
        { slot: "role", id: "role-test" },
        { slot: "mode", id: "mode-yolo" },
      ]);
      expect(renderComposed(res.composed)).toContain("Mode: Yolo");
    }
  });

  it("dangerousStandardChunks adds mode-yolo to the default library", () => {
    const defaultModeIds = standardChunks()
      .filter((c) => c.slot === "mode")
      .map((c) => c.id.value);
    expect(defaultModeIds).not.toContain("mode-yolo");

    const dangerousModeIds = dangerousStandardChunks()
      .filter((c) => c.slot === "mode")
      .map((c) => c.id.value);
    expect(dangerousModeIds).toContain("mode-yolo");
  });
});

// ── IsolationMode.None (gated) ───────────────────────────────────────────────

describe("IsolationMode.None (dangerous)", () => {
  it("dangerousWorkspaceSandbox builds with none isolation and warns", () => {
    const root = tmp();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const sb = dangerousWorkspaceSandbox({ root });
    expect(sb.isolationMode()).toEqual({ kind: "none" });
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("noneIsolationMode serializes to the documented wire tag", () => {
    expect(JSON.parse(JSON.stringify(noneIsolationMode()))).toEqual({ kind: "none" });
  });
});

// ── Gate: the dangerous constructors are NOT on the default barrel ────────────

describe("gate — default barrel does not expose the footguns", () => {
  const exported = Object.keys(defaultBarrel);

  it("does not export dangerous-only constructors", () => {
    for (const name of [
      "yoloPromptChunk",
      "composeWithYolo",
      "dangerousStandardChunks",
      "dangerousWorkspaceSandbox",
      "noneIsolationMode",
      "DANGEROUS_MODE_YOLO",
    ]) {
      expect(exported).not.toContain(name);
    }
  });

  it("default standard library omits the Yolo mode chunk", () => {
    const modeIds = standardChunks()
      .filter((c) => c.slot === "mode")
      .map((c) => c.id.value);
    expect(modeIds.sort()).toEqual(
      ["mode-always-ask", "mode-auto-edit", "mode-plan", "mode-safe-auto"].sort(),
    );
  });
});

// ── Cross-language fixture replay — dangerous.json (gated) ────────────────────

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/prompt_chunk_registry");

interface FixtureChunk {
  id: string;
  content: string;
  slot: pcr.ChunkSlot;
  cache_block: pcr.CacheBlock;
}
interface FixtureCompose {
  role: string;
  mode: string;
  capabilities: string[];
  skills: string[];
}
interface FixtureExpected {
  slot: pcr.ChunkSlot;
  id: string;
}
interface FixtureCase {
  name: string;
  register_inputs: FixtureChunk[];
  compose: FixtureCompose;
  expected_chunks: FixtureExpected[];
  rendered_contains: string[];
}
interface FixtureFile {
  description?: string;
  cases: FixtureCase[];
}

describe("PromptChunkRegistry — fixture replay (dangerous.json)", () => {
  const raw = readFileSync(resolve(fixtureDir, "dangerous.json"), "utf-8");
  const suite = JSON.parse(raw) as FixtureFile;

  for (const c of suite.cases) {
    it(c.name, () => {
      // The dangerous fixture only exercises the Yolo mode; assert it so we
      // notice if the fixture ever drifts to a non-dangerous mode.
      expect(c.compose.mode).toBe("yolo");

      const r = new StandardPromptChunkRegistry();
      for (const input of c.register_inputs) {
        const err = r.register(
          pcr.promptChunk(input.id, input.content, input.slot, input.cache_block),
        );
        expect(err, `[${c.name}] register failed`).toBeNull();
      }

      const res = composeWithYolo(
        r,
        new ChunkId(c.compose.role),
        c.compose.capabilities.map((s) => new ChunkId(s)),
        c.compose.skills.map((s) => new ChunkId(s)),
      );
      expect(res.ok, `[${c.name}] compose failed`).toBe(true);
      if (!res.ok) return;

      const actual = res.composed.chunks.map((ch) => ({ slot: ch.slot, id: ch.id.value }));
      const expected = c.expected_chunks.map((e) => ({ slot: e.slot, id: e.id }));
      expect(actual).toEqual(expected);

      expect(r.validate(res.composed)).toEqual([]);

      const rendered = renderComposed(res.composed);
      for (const needle of c.rendered_contains) {
        expect(rendered).toContain(needle);
      }
    });
  }
});
