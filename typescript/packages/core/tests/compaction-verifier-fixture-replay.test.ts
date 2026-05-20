/**
 * Cross-language fixture replay for {@link KeyTermVerifier} (spore-core issue
 * #29). Loads the shared JSON fixture and asserts the same outcomes the Rust
 * suite asserts: build a SessionState from each case's `task_instruction`,
 * run KeyTermVerifier against `summary` with the given `hints`, and check
 * `passed` / `missing_items`.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { context, SessionId, TaskId } from "../src/index.js";

const { KeyTermVerifier, newSessionState } = context;

interface FixtureHints {
  keep_architectural_decisions: boolean;
  keep_open_problems: boolean;
  keep_current_task_state: boolean;
  keep_recent_file_list: boolean;
  keep_thinking_blocks: boolean;
}
interface FixtureExpected {
  passed: boolean;
  missing_items: string[];
}
interface FixtureCase {
  name: string;
  summary: string;
  hints: FixtureHints;
  task_instruction: string;
  expected: FixtureExpected;
}
interface FixtureFile {
  description?: string;
  cases: FixtureCase[];
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/compaction_verifier/cases.json");

const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as FixtureFile;

describe("KeyTermVerifier fixture replay", () => {
  const verifier = new KeyTermVerifier();

  for (const c of fixture.cases) {
    it(c.name, () => {
      const session = newSessionState(SessionId.of("s1"), TaskId.of("t1"), c.task_instruction);
      const res = verifier.verify(c.summary, c.hints, session);
      expect(res.passed).toBe(c.expected.passed);
      expect(res.missingItems).toEqual(c.expected.missing_items);
    });
  }
});
