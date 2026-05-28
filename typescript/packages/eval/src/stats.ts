/**
 * Native statistics for the EvalHarness — NO external stats library (Rules 26-28).
 *
 * Ships:
 *   - {@link MetricStats} aggregation (mean, stddev, p50, p95, n).
 *   - {@link welchTTest} — unequal-variance two-sample t with Welch–Satterthwaite
 *     df and a two-sided p-value via the regularized incomplete beta function
 *     (Lentz continued fraction). Rule 26.
 *   - {@link bootstrapCi} — percentile confidence interval, default 1000 iters,
 *     using an inline SplitMix64 PRNG so cross-language replay is byte-identical.
 *     Rule 27.
 *
 * Every public function has a hand-computed oracle test (Rule 28); the seeded
 * bootstrap reproduces the `welch_bootstrap.json` oracle byte-for-byte (Rule 29).
 */

// ============================================================================
// MetricStats
// ============================================================================

/** Aggregated sample statistics for one metric over a set of runs (Rule 19). */
export interface MetricStats {
  mean: number;
  stddev: number;
  p50: number;
  p95: number;
  n: number;
}

/**
 * Aggregate a slice of samples. `stddev` is the **sample** standard deviation
 * (Bessel's correction, n-1); with fewer than two samples it is `0`.
 * Percentiles use the nearest-rank method on the sorted samples.
 */
export function metricStatsFromSamples(
  samples: readonly number[],
): MetricStats {
  const n = samples.length;
  if (n === 0) {
    return { mean: 0, stddev: 0, p50: 0, p95: 0, n: 0 };
  }
  const mean = samples.reduce((acc, x) => acc + x, 0) / n;
  let stddev = 0;
  if (n >= 2) {
    const variance =
      samples.reduce((acc, x) => acc + (x - mean) ** 2, 0) / (n - 1);
    stddev = Math.sqrt(variance);
  }
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    mean,
    stddev,
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    n,
  };
}

/** Nearest-rank percentile on already-sorted data. `p` is in `[0, 100]`. */
export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0]!;
  // Nearest-rank: rank = ceil(p/100 * n), clamped to [1, n].
  const rank = Math.ceil((p / 100) * sorted.length);
  const idx = clamp(rank, 1, sorted.length) - 1;
  return sorted[idx]!;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

// ============================================================================
// Welch's t-test (Rule 26)
// ============================================================================

/** Result of a two-sample Welch t-test. */
export interface WelchResult {
  t: number;
  df: number;
  /** Two-sided p-value. */
  pValue: number;
}

/**
 * Welch's unequal-variance two-sample t-test with Welch–Satterthwaite degrees
 * of freedom and a two-sided p-value (Rule 26).
 *
 * Degenerate inputs (either sample with n < 2, or both variances zero) return
 * `t = 0`, `df = 0`, `pValue = 1` — "no detectable difference" rather than a
 * divide-by-zero.
 */
export function welchTTest(
  a: readonly number[],
  b: readonly number[],
): WelchResult {
  const na = a.length;
  const nb = b.length;
  if (na < 2 || nb < 2) {
    return { t: 0, df: 0, pValue: 1 };
  }
  const meanA = a.reduce((acc, x) => acc + x, 0) / na;
  const meanB = b.reduce((acc, x) => acc + x, 0) / nb;
  const varA = a.reduce((acc, x) => acc + (x - meanA) ** 2, 0) / (na - 1);
  const varB = b.reduce((acc, x) => acc + (x - meanB) ** 2, 0) / (nb - 1);

  const sa = varA / na;
  const sb = varB / nb;
  const denom = sa + sb;
  if (denom <= 0) {
    // Both samples are constant. Equal means → no difference; differing
    // constant means → an infinitely significant difference (p → 0).
    const p = Math.abs(meanA - meanB) <= Number.EPSILON ? 1 : 0;
    return { t: 0, df: 0, pValue: p };
  }

  const t = (meanA - meanB) / Math.sqrt(denom);
  // Welch–Satterthwaite df.
  const df = denom ** 2 / (sa ** 2 / (na - 1) + sb ** 2 / (nb - 1));
  const pValue = twoSidedP(t, df);
  return { t, df, pValue };
}

/**
 * Two-sided p-value for a t-statistic with `df` degrees of freedom, via the
 * regularized incomplete beta function:
 *   p = I_{df/(df+t^2)}(df/2, 1/2)
 */
function twoSidedP(t: number, df: number): number {
  if (df <= 0) return 1;
  const x = df / (df + t * t);
  return clamp(betai(df / 2, 0.5, x), 0, 1);
}

/**
 * Regularized incomplete beta function I_x(a, b) via the Lentz continued
 * fraction, the same routine used by Numerical Recipes' `betai`.
 */
function betai(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lnBeta = lnGamma(a + b) - lnGamma(a) - lnGamma(b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) + lnBeta);
  // Use the symmetry relation for faster continued-fraction convergence.
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betacf(a, b, x)) / a;
  }
  return 1 - (front * betacf(b, a, 1 - x)) / b;
}

/** Lentz continued fraction for the incomplete beta. */
function betacf(a: number, b: number, x: number): number {
  const MAX_ITER = 200;
  const EPS = 3.0e-12;
  const FPMIN = 1.0e-300;

  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= MAX_ITER; m++) {
    const m2 = 2 * m;
    // Even step.
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;
    // Odd step.
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

const LN_GAMMA_G = 7;
const LN_GAMMA_COEF = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531,
  -176.6150291621406, 12.507343278686905, -0.13857109526572012,
  9.984369578019572e-6, 1.5056327351493116e-7,
] as const;

/** Lanczos approximation of ln(Γ(x)) for x > 0. */
function lnGamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula.
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - lnGamma(1 - x);
  }
  const z = x - 1;
  let a = LN_GAMMA_COEF[0]!;
  const t = z + LN_GAMMA_G + 0.5;
  for (let i = 1; i < LN_GAMMA_COEF.length; i++) {
    a += LN_GAMMA_COEF[i]! / (z + i);
  }
  return (
    0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a)
  );
}

// ============================================================================
// Bootstrap CI (Rule 27)
// ============================================================================

/**
 * A bootstrap confidence interval. Mirrors the `ConfidenceInterval` reported on
 * a {@link import("./report.js").MetricComparison}.
 */
export interface ConfidenceInterval {
  lower: number;
  upper: number;
  level: number;
}

/**
 * Inline SplitMix64 PRNG (Rule 27): a fixed, seedable, byte-identical generator
 * so a bootstrap CI replays identically across all four languages.
 *
 * Implemented over `BigInt` u64 arithmetic so the wrapping multiplies match the
 * Rust reference exactly (JS `number` cannot hold 64-bit integer products).
 */
export class SplitMix64 {
  private state: bigint;

  constructor(seed: bigint) {
    this.state = BigInt.asUintN(64, seed);
  }

  /** Next 64-bit value (the canonical SplitMix64 step). */
  nextU64(): bigint {
    this.state = BigInt.asUintN(64, this.state + 0x9e3779b97f4a7c15n);
    let z = this.state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  }

  /** Uniform index in `[0, n)`. `n` must be a positive integer. */
  nextIndex(n: number): number {
    return Number(this.nextU64() % BigInt(n));
  }
}

/** Default bootstrap iteration count (Rule 27). */
export const DEFAULT_BOOTSTRAP_ITERATIONS = 1000;
/** Fixed bootstrap seed so replays are byte-identical cross-language. */
export const DEFAULT_BOOTSTRAP_SEED = 0x5eed5eed5eed5eedn;

/**
 * Percentile bootstrap CI for the mean (Rule 27). Resamples `samples` with
 * replacement `iterations` times using a SplitMix64 seeded with `seed`, then
 * takes the `[(1-level)/2, 1-(1-level)/2]` percentiles of the resample means.
 *
 * `level` is the confidence level (e.g. `0.95`). Returns `undefined` for an
 * empty sample.
 */
export function bootstrapCi(
  samples: readonly number[],
  iterations: number,
  level: number,
  seed: bigint,
): ConfidenceInterval | undefined {
  if (samples.length === 0) return undefined;
  const rng = new SplitMix64(seed);
  const means: number[] = new Array(iterations);
  for (let i = 0; i < iterations; i++) {
    let sum = 0;
    for (let j = 0; j < samples.length; j++) {
      sum += samples[rng.nextIndex(samples.length)]!;
    }
    means[i] = sum / samples.length;
  }
  means.sort((a, b) => a - b);
  const alpha = 1 - level;
  const lower = percentile(means, (alpha / 2) * 100);
  const upper = percentile(means, (1 - alpha / 2) * 100);
  return { lower, upper, level };
}
