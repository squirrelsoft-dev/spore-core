/**
 * Native-statistics oracle tests (Rules 26-28) + the cross-language
 * `welch_bootstrap.json` replay (Rule 29).
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  bootstrapCi,
  metricStatsFromSamples,
  SplitMix64,
  welchTTest,
  DEFAULT_BOOTSTRAP_SEED,
} from "../src/index.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "../../../../fixtures/task_suites");

function approx(a: number, b: number, tol: number): boolean {
  return Math.abs(a - b) < tol;
}

describe("MetricStats", () => {
  it("aggregates mean/stddev/percentiles", () => {
    const s = metricStatsFromSamples([1, 2, 3, 4]);
    expect(approx(s.mean, 2.5, 1e-12)).toBe(true);
    expect(approx(s.stddev, Math.sqrt(5 / 3), 1e-9)).toBe(true);
    expect(s.n).toBe(4);
    // nearest-rank p50 of 4 elements: rank=ceil(0.5*4)=2 -> idx 1 -> 2.0
    expect(approx(s.p50, 2.0, 1e-12)).toBe(true);
    // p95: rank=ceil(0.95*4)=4 -> idx 3 -> 4.0
    expect(approx(s.p95, 4.0, 1e-12)).toBe(true);
  });

  it("single sample has zero stddev", () => {
    const s = metricStatsFromSamples([7]);
    expect(s.n).toBe(1);
    expect(s.stddev).toBe(0);
    expect(s.mean).toBe(7);
  });
});

describe("welchTTest", () => {
  it("matches the hand-computed oracle", () => {
    const r = welchTTest([27, 31, 29, 30], [20, 22, 24, 21]);
    expect(approx(r.t, 6.21063, 1e-4)).toBe(true);
    expect(approx(r.df, 6.0, 1e-3)).toBe(true);
    expect(r.pValue).toBeLessThan(0.005);
    expect(r.pValue).toBeGreaterThan(0);
  });

  it("identical samples give p=1", () => {
    const r = welchTTest([1, 2, 3, 4], [1, 2, 3, 4]);
    expect(approx(r.t, 0, 1e-12)).toBe(true);
    expect(approx(r.pValue, 1, 1e-6)).toBe(true);
  });

  it("differing constants give p=0", () => {
    const r = welchTTest([5, 5, 5], [3, 3, 3]);
    expect(r.pValue).toBe(0);
  });

  it("tiny samples give p=1", () => {
    const r = welchTTest([1], [2, 3]);
    expect(approx(r.pValue, 1, 1e-12)).toBe(true);
  });
});

describe("SplitMix64", () => {
  it("is deterministic and matches the known seed-0 constant", () => {
    const r1 = new SplitMix64(42n);
    const r2 = new SplitMix64(42n);
    for (let i = 0; i < 100; i++) {
      expect(r1.nextU64()).toBe(r2.nextU64());
    }
    const r0 = new SplitMix64(0n);
    expect(r0.nextU64()).toBe(16294208416658607535n);
  });
});

describe("bootstrapCi", () => {
  it("is seeded and brackets the mean", () => {
    const samples = [10, 12, 11, 13, 9, 14, 8, 15];
    const ci1 = bootstrapCi(samples, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)!;
    const ci2 = bootstrapCi(samples, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)!;
    expect(ci1).toEqual(ci2);
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(ci1.lower).toBeLessThanOrEqual(mean);
    expect(mean).toBeLessThanOrEqual(ci1.upper);
    expect(ci1.lower).toBeLessThan(ci1.upper);
    expect(approx(ci1.level, 0.95, 1e-12)).toBe(true);
  });

  it("empty sample returns undefined", () => {
    expect(bootstrapCi([], 100, 0.95, 1n)).toBeUndefined();
  });
});

interface StatsCase {
  name: string;
  baseline: number[];
  candidate: number[];
  welch_t: number;
  welch_df: number;
  welch_p_value: number;
  welch_p_tolerance: number;
  candidate_bootstrap_ci: { lower: number; upper: number };
  baseline_bootstrap_ci: { lower: number; upper: number };
}

describe("Rule 29 — welch_bootstrap.json fixture replay", () => {
  const oracle = JSON.parse(
    readFileSync(join(FIXTURES, "welch_bootstrap.json"), "utf8"),
  ) as {
    cases: StatsCase[];
  };

  for (const c of oracle.cases) {
    it(`reproduces case ${c.name}`, () => {
      const w = welchTTest(c.baseline, c.candidate);
      expect(approx(Math.abs(w.t), Math.abs(c.welch_t), 1e-9)).toBe(true);
      expect(approx(w.df, c.welch_df, 1e-9)).toBe(true);
      expect(approx(w.pValue, c.welch_p_value, c.welch_p_tolerance)).toBe(true);

      const cand = bootstrapCi(
        c.candidate,
        1000,
        0.95,
        DEFAULT_BOOTSTRAP_SEED,
      )!;
      expect(approx(cand.lower, c.candidate_bootstrap_ci.lower, 1e-12)).toBe(
        true,
      );
      expect(approx(cand.upper, c.candidate_bootstrap_ci.upper, 1e-12)).toBe(
        true,
      );

      const base = bootstrapCi(c.baseline, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED)!;
      expect(approx(base.lower, c.baseline_bootstrap_ci.lower, 1e-12)).toBe(
        true,
      );
      expect(approx(base.upper, c.baseline_bootstrap_ci.upper, 1e-12)).toBe(
        true,
      );
    });
  }
});
