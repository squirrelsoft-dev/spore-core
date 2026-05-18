/**
 * Cross-language fixture replay for {@link PromptChunkRegistry} (spore-core
 * issue #24). Loads the shared JSON fixture and asserts the same outcomes
 * the Rust suite asserts:
 *
 *   - chunks registered from `register_inputs` compose into the expected
 *     `(slot, id)` sequence,
 *   - `validate()` returns an empty error list,
 *   - rendered text contains every `rendered_contains` string.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { promptChunkRegistry as pcr } from "../src/index.js";

const { ChunkId, StandardPromptChunkRegistry, promptChunk, renderComposed } = pcr;

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/prompt_chunk_registry");

// Fixture shape (snake_case slot/cache_block/mode strings).
interface FixtureChunk {
  id: string;
  content: string;
  slot: pcr.ChunkSlot;
  cache_block: pcr.CacheBlock;
}
interface FixtureCompose {
  role: string;
  mode: pcr.Mode;
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

describe("PromptChunkRegistry — fixture replay (basic.json)", () => {
  const raw = readFileSync(resolve(fixtureDir, "basic.json"), "utf-8");
  const suite = JSON.parse(raw) as FixtureFile;

  for (const c of suite.cases) {
    it(c.name, () => {
      const r = new StandardPromptChunkRegistry();
      for (const input of c.register_inputs) {
        const err = r.register(promptChunk(input.id, input.content, input.slot, input.cache_block));
        expect(err, `[${c.name}] register failed`).toBeNull();
      }

      const res = r.compose(
        new ChunkId(c.compose.role),
        c.compose.mode,
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
