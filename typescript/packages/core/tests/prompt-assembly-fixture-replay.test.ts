/**
 * Cross-language fixture replay for the prompt assembly engine (spore-core
 * issue #79). Loads the shared JSON fixtures and asserts the same outcomes the
 * Rust suite asserts:
 *
 *   - `condition_eval.json` — each serialized `condition` evaluated against its
 *     `assembly_context` matches `expected` (R1–R8).
 *   - `assembly_steps.json` — `registered_chunks` assembled against the
 *     `assembly_context` yield the expected per-bucket id lists (R10–R17).
 *
 * The fixtures are ground truth — a failure here means the serialization /
 * evaluation diverged; fix the code, never the fixture.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { promptAssembly as pa } from "../src/index.js";

const { ContextSourcesBuilder, deserializeCondition, parseAssemblyContext, parsePromptChunk } = pa;

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/prompt_assembly");

interface ConditionCase {
  name: string;
  condition: unknown;
  assembly_context: unknown;
  expected: boolean;
}
interface AssemblyCase {
  name: string;
  registered_chunks: unknown[];
  assembly_context: unknown;
  expected_static: string[];
  expected_per_session: string[];
  expected_per_turn: string[];
}
interface Suite<T> {
  description?: string;
  cases: T[];
}

describe("prompt assembly — condition_eval.json (R1–R8)", () => {
  const raw = readFileSync(resolve(fixtureDir, "condition_eval.json"), "utf-8");
  const suite = JSON.parse(raw) as Suite<ConditionCase>;
  const b = new ContextSourcesBuilder();

  expect(suite.cases.length).toBeGreaterThanOrEqual(8);

  for (const c of suite.cases) {
    it(c.name, () => {
      const condition = deserializeCondition(c.condition);
      const context = parseAssemblyContext(c.assembly_context);
      expect(b.evaluate(condition, context)).toBe(c.expected);
    });
  }
});

describe("prompt assembly — assembly_steps.json (R10–R17)", () => {
  const raw = readFileSync(resolve(fixtureDir, "assembly_steps.json"), "utf-8");
  const suite = JSON.parse(raw) as Suite<AssemblyCase>;

  expect(suite.cases.length).toBeGreaterThan(0);

  for (const c of suite.cases) {
    it(c.name, () => {
      const chunks = c.registered_chunks.map(parsePromptChunk);
      const context = parseAssemblyContext(c.assembly_context);
      const buckets = ContextSourcesBuilder.withChunks(chunks).assemble(context);
      expect(buckets.static_chunks.map((x) => x.id)).toEqual(c.expected_static);
      expect(buckets.per_session.map((x) => x.id)).toEqual(c.expected_per_session);
      expect(buckets.per_turn.map((x) => x.id)).toEqual(c.expected_per_turn);
    });
  }
});
