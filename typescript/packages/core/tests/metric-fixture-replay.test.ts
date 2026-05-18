/**
 * Cross-language fixture replay for {@link MetricEvaluator} (spore-core
 * issue #23). Loads the shared JSON fixtures and asserts the same outcomes
 * the Rust suite asserts:
 *
 *   - `fixtures/metric_evaluator/should_keep.json`   ⇒ {@link shouldKeep}
 *   - `fixtures/metric_evaluator/parse_metric.json`  ⇒ {@link parseMetric}
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { metric } from "../src/index.js";
import type { OptimizationDirection } from "../src/harness/types.js";

const { parseMetric, shouldKeep } = metric;

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = resolve(here, "../../../../fixtures/metric_evaluator");

// ---------------------------------------------------------------------------
// should_keep.json
// ---------------------------------------------------------------------------

interface ShouldKeepCase {
  name: string;
  new_value: number;
  current_best: number;
  direction: OptimizationDirection;
  min_delta: number | null;
  expected: boolean;
}
interface ShouldKeepSuite {
  description?: string;
  cases: ShouldKeepCase[];
}

describe("shouldKeep — fixture replay", () => {
  const raw = readFileSync(resolve(fixtureDir, "should_keep.json"), "utf-8");
  const suite = JSON.parse(raw) as ShouldKeepSuite;
  for (const c of suite.cases) {
    it(c.name, () => {
      const got = shouldKeep(c.new_value, c.current_best, c.direction, c.min_delta);
      expect(got).toBe(c.expected);
    });
  }
});

// ---------------------------------------------------------------------------
// parse_metric.json
// ---------------------------------------------------------------------------

type ParseExpected = { kind: "value"; value: number } | { kind: "parse_failed" };
interface ParseCase {
  name: string;
  output: string;
  pattern: string;
  expected: ParseExpected;
}
interface ParseSuite {
  description?: string;
  cases: ParseCase[];
}

describe("parseMetric — fixture replay", () => {
  const raw = readFileSync(resolve(fixtureDir, "parse_metric.json"), "utf-8");
  const suite = JSON.parse(raw) as ParseSuite;
  for (const c of suite.cases) {
    it(c.name, () => {
      if (c.expected.kind === "value") {
        const got = parseMetric(c.output, c.pattern);
        expect(got).toBeCloseTo(c.expected.value, 9);
      } else {
        try {
          parseMetric(c.output, c.pattern);
          throw new Error("expected ParseFailed");
        } catch (e) {
          expect(e).toBeInstanceOf(metric.MetricErrorException);
          expect((e as metric.MetricErrorException).error.kind).toBe("parse_failed");
        }
      }
    });
  }
});
