//! Native statistics for the EvalHarness — NO external stats crate (Rules 26-28).
//!
//! Ships:
//!   - [`MetricStats`] aggregation (mean, stddev, p50, p95, n).
//!   - [`welch_t_test`] — unequal-variance two-sample t with Welch–Satterthwaite
//!     df and a two-sided p-value via the regularized incomplete beta function
//!     (Lentz continued fraction). Rule 26.
//!   - [`bootstrap_ci`] — percentile confidence interval, default 1000 iters,
//!     using an inline SplitMix64 PRNG so cross-language replay is byte-identical.
//!     Rule 27.
//!
//! Every public function has a hand-computed oracle test (Rule 28).

use serde::{Deserialize, Serialize};

// ============================================================================
// MetricStats
// ============================================================================

/// Aggregated sample statistics for one metric over a set of runs (Rule 19).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricStats {
    pub mean: f64,
    pub stddev: f64,
    pub p50: f64,
    pub p95: f64,
    pub n: u32,
}

impl MetricStats {
    /// Aggregate a slice of samples. `stddev` is the **sample** standard
    /// deviation (Bessel's correction, n-1); with fewer than two samples it is
    /// `0.0`. Percentiles use the nearest-rank method on the sorted samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                stddev: 0.0,
                p50: 0.0,
                p95: 0.0,
                n: 0,
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let stddev = if n < 2 {
            0.0
        } else {
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            var.sqrt()
        };
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            mean,
            stddev,
            p50: percentile(&sorted, 50.0),
            p95: percentile(&sorted, 95.0),
            n: n as u32,
        }
    }
}

/// Nearest-rank percentile on already-sorted data. `p` is in `[0, 100]`.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    // Nearest-rank: rank = ceil(p/100 * n), clamped to [1, n].
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = rank.clamp(1, sorted.len()) - 1;
    sorted[idx]
}

// ============================================================================
// Welch's t-test (Rule 26)
// ============================================================================

/// Result of a two-sample Welch t-test.
#[derive(Debug, Clone, PartialEq)]
pub struct WelchResult {
    pub t: f64,
    pub df: f64,
    /// Two-sided p-value.
    pub p_value: f64,
}

/// Welch's unequal-variance two-sample t-test with Welch–Satterthwaite degrees
/// of freedom and a two-sided p-value (Rule 26).
///
/// Degenerate inputs (either sample with n < 2, or both variances zero) return
/// `t = 0`, `df = 0`, `p_value = 1.0` — "no detectable difference" rather than
/// a divide-by-zero.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> WelchResult {
    let na = a.len();
    let nb = b.len();
    if na < 2 || nb < 2 {
        return WelchResult {
            t: 0.0,
            df: 0.0,
            p_value: 1.0,
        };
    }
    let mean_a = a.iter().sum::<f64>() / na as f64;
    let mean_b = b.iter().sum::<f64>() / nb as f64;
    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (na as f64 - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (nb as f64 - 1.0);

    let sa = var_a / na as f64;
    let sb = var_b / nb as f64;
    let denom = sa + sb;
    if denom <= 0.0 {
        // Both samples are constant. Equal means → no difference; differing
        // constant means → an infinitely significant difference (p → 0).
        let p = if (mean_a - mean_b).abs() <= f64::EPSILON {
            1.0
        } else {
            0.0
        };
        return WelchResult {
            t: 0.0,
            df: 0.0,
            p_value: p,
        };
    }

    let t = (mean_a - mean_b) / denom.sqrt();
    // Welch–Satterthwaite df.
    let df = denom.powi(2) / (sa.powi(2) / (na as f64 - 1.0) + sb.powi(2) / (nb as f64 - 1.0));
    let p_value = two_sided_p(t, df);
    WelchResult { t, df, p_value }
}

/// Two-sided p-value for a t-statistic with `df` degrees of freedom, via the
/// regularized incomplete beta function:
///   p = I_{df/(df+t^2)}(df/2, 1/2)
fn two_sided_p(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    let x = df / (df + t * t);
    betai(df / 2.0, 0.5, x).clamp(0.0, 1.0)
}

/// Regularized incomplete beta function I_x(a, b) via the Lentz continued
/// fraction, the same routine used by Numerical Recipes' `betai`.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (a * x.ln() + b * (1.0 - x).ln() + ln_beta).exp();
    // Use the symmetry relation for faster continued-fraction convergence.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// Lentz continued fraction for the incomplete beta (≈40 lines, no deps).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: u32 = 200;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Lanczos approximation of ln(Γ(x)) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        let t = x + G + 0.5;
        for (i, c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

// ============================================================================
// Bootstrap CI (Rule 27)
// ============================================================================

/// A bootstrap confidence interval. Mirrors the `ConfidenceInterval` reported on
/// a `MetricComparison`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

/// Inline SplitMix64 PRNG (Rule 27): a fixed, seedable, byte-identical
/// generator so a bootstrap CI replays identically across all four languages.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit value (the canonical SplitMix64 step).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, n)`. `n` must be non-zero.
    pub fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Percentile bootstrap CI for the mean (Rule 27). Resamples `samples` with
/// replacement `iterations` times using a SplitMix64 seeded with `seed`, then
/// takes the `[(1-level)/2, 1-(1-level)/2]` percentiles of the resample means.
///
/// `level` is the confidence level (e.g. `0.95`). Returns `None` for an empty
/// sample.
pub fn bootstrap_ci(
    samples: &[f64],
    iterations: u32,
    level: f64,
    seed: u64,
) -> Option<ConfidenceInterval> {
    if samples.is_empty() {
        return None;
    }
    let mut rng = SplitMix64::new(seed);
    let mut means: Vec<f64> = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..samples.len() {
            sum += samples[rng.next_index(samples.len())];
        }
        means.push(sum / samples.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = 1.0 - level;
    let lower = percentile(&means, alpha / 2.0 * 100.0);
    let upper = percentile(&means, (1.0 - alpha / 2.0) * 100.0);
    Some(ConfidenceInterval {
        lower,
        upper,
        level,
    })
}

/// Default bootstrap iteration count (Rule 27).
pub const DEFAULT_BOOTSTRAP_ITERATIONS: u32 = 1000;
/// Fixed bootstrap seed so replays are byte-identical cross-language.
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 0x5EED_5EED_5EED_5EED;

// ============================================================================
// Tests (Rule 28 — hand-computed oracles)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn metric_stats_basic() {
        let s = MetricStats::from_samples(&[1.0, 2.0, 3.0, 4.0]);
        assert!(approx(s.mean, 2.5, 1e-12));
        // sample stddev of 1,2,3,4 = sqrt(5/3) ≈ 1.290994
        assert!(approx(s.stddev, (5.0_f64 / 3.0).sqrt(), 1e-9));
        assert_eq!(s.n, 4);
        // nearest-rank p50 of 4 elements: rank=ceil(0.5*4)=2 -> idx 1 -> 2.0
        assert!(approx(s.p50, 2.0, 1e-12));
        // p95: rank=ceil(0.95*4)=4 -> idx 3 -> 4.0
        assert!(approx(s.p95, 4.0, 1e-12));
    }

    #[test]
    fn metric_stats_single_sample_zero_stddev() {
        let s = MetricStats::from_samples(&[7.0]);
        assert_eq!(s.n, 1);
        assert!(approx(s.stddev, 0.0, 1e-12));
        assert!(approx(s.mean, 7.0, 1e-12));
    }

    #[test]
    fn ln_gamma_known_values() {
        // ln(Γ(5)) = ln(24) ≈ 3.1780538
        assert!(approx(ln_gamma(5.0), 24.0_f64.ln(), 1e-7));
        // ln(Γ(0.5)) = ln(sqrt(pi)) ≈ 0.5723649
        assert!(approx(
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln(),
            1e-7
        ));
    }

    #[test]
    fn betai_symmetry_and_endpoints() {
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        let a = 2.5;
        let b = 3.5;
        let x = 0.3;
        let left = betai(a, b, x);
        let right = 1.0 - betai(b, a, 1.0 - x);
        assert!(approx(left, right, 1e-9));
        assert!(approx(betai(a, b, 0.0), 0.0, 1e-12));
        assert!(approx(betai(a, b, 1.0), 1.0, 1e-12));
    }

    #[test]
    fn welch_t_test_oracle() {
        // Hand-computed oracle.
        // a = [27,31,29,30], mean 29.25, var (sample) = ?
        //   deviations: -2.25,1.75,-0.25,0.75 -> sq: 5.0625,3.0625,0.0625,0.5625
        //   sum=8.75, /3 = 2.91667
        // b = [20,22,24,21], mean 21.75, var:
        //   dev: -1.75,0.25,2.25,-0.75 -> 3.0625,0.0625,5.0625,0.5625 sum=8.75 /3=2.91667
        // sa=sb=2.91667/4=0.729167; denom=1.458333; sqrt=1.207614
        // t=(29.25-21.75)/1.207614 = 7.5/1.207614 = 6.21063
        let a = [27.0, 31.0, 29.0, 30.0];
        let b = [20.0, 22.0, 24.0, 21.0];
        let r = welch_t_test(&a, &b);
        assert!(approx(r.t, 6.21063, 1e-4), "t={}", r.t);
        // df: denom^2 / (sa^2/3 + sb^2/3)
        //   denom^2 = 2.12674; sa^2=sb^2=0.531684; /3 each = 0.177228; sum=0.354456
        //   df = 2.12674/0.354456 = 6.0
        assert!(approx(r.df, 6.0, 1e-3), "df={}", r.df);
        // two-sided p for t=6.21, df=6 is small (≈ 0.0008).
        assert!(r.p_value < 0.005, "p={}", r.p_value);
        assert!(r.p_value > 0.0);
    }

    #[test]
    fn welch_identical_samples_p_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let r = welch_t_test(&a, &b);
        assert!(approx(r.t, 0.0, 1e-12));
        assert!(approx(r.p_value, 1.0, 1e-6));
    }

    #[test]
    fn welch_constant_differing_means_p_zero() {
        let a = [5.0, 5.0, 5.0];
        let b = [3.0, 3.0, 3.0];
        let r = welch_t_test(&a, &b);
        assert!(approx(r.p_value, 0.0, 1e-12));
    }

    #[test]
    fn welch_tiny_samples_p_one() {
        let r = welch_t_test(&[1.0], &[2.0, 3.0]);
        assert!(approx(r.p_value, 1.0, 1e-12));
    }

    #[test]
    fn splitmix64_is_deterministic() {
        let mut r1 = SplitMix64::new(42);
        let mut r2 = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
        // First SplitMix64 output for seed 0 is a well-known constant.
        let mut r0 = SplitMix64::new(0);
        assert_eq!(r0.next_u64(), 16294208416658607535);
    }

    #[test]
    fn bootstrap_ci_brackets_mean_and_is_seeded() {
        let samples = [10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 8.0, 15.0];
        let ci1 = bootstrap_ci(&samples, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED).unwrap();
        let ci2 = bootstrap_ci(&samples, 1000, 0.95, DEFAULT_BOOTSTRAP_SEED).unwrap();
        // Seeded → byte-identical across calls.
        assert_eq!(ci1, ci2);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(ci1.lower <= mean && mean <= ci1.upper);
        assert!(ci1.lower < ci1.upper);
        assert!(approx(ci1.level, 0.95, 1e-12));
    }

    #[test]
    fn bootstrap_ci_empty_is_none() {
        assert!(bootstrap_ci(&[], 100, 0.95, 1).is_none());
    }
}
