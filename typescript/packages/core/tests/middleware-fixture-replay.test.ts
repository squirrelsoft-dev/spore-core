/**
 * Fixture-replay tests for the canonical MiddlewareChain (spore-core
 * issue #11).
 *
 * Loads `fixtures/middleware/checklist_basic.json` and asserts the
 * pre-completion checklist outcome matches the Rust, Python, and Go
 * suites byte-for-byte.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { middleware } from "../src/index.js";
import { emptySessionState } from "../src/harness/types.js";

const { StandardMiddlewareChain, PreCompletionChecklistMiddleware } = middleware;

interface FixtureCase {
  required: string[];
  response: string;
  expected: "continue" | "force_another_turn";
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/middleware/checklist_basic.json");

describe("MiddlewareChain fixture replay", () => {
  it("checklist_basic matches the cross-language expected outcome", async () => {
    const raw = readFileSync(fixturePath, "utf8");
    const fix = JSON.parse(raw) as FixtureCase;

    const chain = new StandardMiddlewareChain();
    chain.register(new PreCompletionChecklistMiddleware(fix.required));

    const d = await chain.fireBeforeCompletion(fix.response, 1, emptySessionState());
    expect(d.kind).toBe(fix.expected);
  });
});
