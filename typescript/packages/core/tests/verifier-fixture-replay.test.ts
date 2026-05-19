/**
 * Cross-language fixture replay for {@link EvaluatorResponseVerifier}
 * (spore-core issue #44). Loads the shared JSON fixture and asserts the
 * same outcomes the Rust suite asserts.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { SessionId, verifier } from "../src/index.js";
import type { AggregateUsage, HaltReason, RunResult } from "../src/index.js";

const { EvaluatorResponseVerifier } = verifier;

interface FixtureUsage {
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number;
}
interface FixtureSuccess {
  kind: "success";
  output: string;
  session_id: string;
  usage: FixtureUsage;
  turns: number;
}
interface FixtureFailure {
  kind: "failure";
  reason: HaltReason;
  session_id: string;
  usage: FixtureUsage;
  turns: number;
}
type FixtureRunResult = FixtureSuccess | FixtureFailure;

interface FixtureExpectedPassed {
  kind: "passed";
}
interface FixtureExpectedFailed {
  kind: "failed";
  contains: string;
}
type FixtureExpected = FixtureExpectedPassed | FixtureExpectedFailed;

interface FixtureCase {
  name: string;
  pass_pattern: string;
  fail_pattern: string;
  build_result: FixtureRunResult;
  eval_result: FixtureRunResult;
  expected: FixtureExpected;
}

interface FixtureFile {
  description?: string;
  cases: FixtureCase[];
}

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(here, "../../../../fixtures/verifier/evaluator_response.json");

function reifyUsage(u: FixtureUsage): AggregateUsage {
  return {
    input_tokens: u.input_tokens ?? 0,
    output_tokens: u.output_tokens ?? 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cost_usd: u.cost_usd ?? 0,
  };
}

function reifyRunResult(r: FixtureRunResult): RunResult {
  if (r.kind === "success") {
    return {
      kind: "success",
      output: r.output,
      session_id: SessionId.of(r.session_id),
      usage: reifyUsage(r.usage),
      turns: r.turns,
    };
  }
  return {
    kind: "failure",
    reason: r.reason,
    session_id: SessionId.of(r.session_id),
    usage: reifyUsage(r.usage),
    turns: r.turns,
  };
}

describe("EvaluatorResponseVerifier fixture replay", () => {
  const text = readFileSync(fixturePath, "utf8");
  const file = JSON.parse(text) as FixtureFile;

  for (const c of file.cases) {
    it(c.name, async () => {
      const v = new EvaluatorResponseVerifier({
        pass_pattern: c.pass_pattern,
        fail_pattern: c.fail_pattern,
      });
      const verdict = await v.verify({
        build_result: reifyRunResult(c.build_result),
        eval_result: reifyRunResult(c.eval_result),
        workspace: "/fixture",
        iteration: 0,
      });
      if (c.expected.kind === "passed") {
        expect(verdict.kind).toBe("passed");
      } else {
        expect(verdict.kind).toBe("failed");
        if (verdict.kind === "failed") {
          expect(verdict.reason).toContain(c.expected.contains);
        }
      }
    });
  }
});
