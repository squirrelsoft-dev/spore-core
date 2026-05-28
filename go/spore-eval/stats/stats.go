// Package stats implements the EvalHarness native statistics — NO external
// stats library (Rules 26-28).
//
// Ships:
//   - MetricStats aggregation (mean, stddev, p50, p95, n).
//   - WelchTTest — unequal-variance two-sample t with Welch–Satterthwaite df
//     and a two-sided p-value via the regularized incomplete beta function
//     (Lentz continued fraction). Rule 26.
//   - BootstrapCI — percentile confidence interval, default 1000 iterations,
//     using an inline SplitMix64 PRNG so cross-language replay is byte-identical
//     to welch_bootstrap.json. Rule 27.
//
// Every public function has a hand-computed oracle test (Rule 28).
package stats

import (
	"math"
	"sort"
)

// ============================================================================
// MetricStats
// ============================================================================

// MetricStats is the aggregated sample statistics for one metric over a set of
// runs (Rule 19).
type MetricStats struct {
	Mean   float64 `json:"mean"`
	Stddev float64 `json:"stddev"`
	P50    float64 `json:"p50"`
	P95    float64 `json:"p95"`
	N      uint32  `json:"n"`
}

// StatsFromSamples aggregates a slice of samples. Stddev is the sample standard
// deviation (Bessel's correction, n-1); with fewer than two samples it is 0.0.
// Percentiles use the nearest-rank method on the sorted samples.
func StatsFromSamples(samples []float64) MetricStats {
	n := len(samples)
	if n == 0 {
		return MetricStats{}
	}
	sum := 0.0
	for _, x := range samples {
		sum += x
	}
	mean := sum / float64(n)
	stddev := 0.0
	if n >= 2 {
		var ss float64
		for _, x := range samples {
			d := x - mean
			ss += d * d
		}
		stddev = math.Sqrt(ss / (float64(n) - 1.0))
	}
	sorted := append([]float64(nil), samples...)
	sort.Float64s(sorted)
	return MetricStats{
		Mean:   mean,
		Stddev: stddev,
		P50:    Percentile(sorted, 50.0),
		P95:    Percentile(sorted, 95.0),
		N:      uint32(n),
	}
}

// Percentile is the nearest-rank percentile on already-sorted data. p is in
// [0, 100].
func Percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0.0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := int(math.Ceil((p / 100.0) * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

// ============================================================================
// Welch's t-test (Rule 26)
// ============================================================================

// WelchResult is the result of a two-sample Welch t-test.
type WelchResult struct {
	T  float64
	DF float64
	// PValue is the two-sided p-value.
	PValue float64
}

// WelchTTest is Welch's unequal-variance two-sample t-test with
// Welch–Satterthwaite degrees of freedom and a two-sided p-value (Rule 26).
//
// Degenerate inputs (either sample with n < 2, or both variances zero) return
// t = 0, df = 0, p_value = 1.0 — "no detectable difference" rather than a
// divide-by-zero.
func WelchTTest(a, b []float64) WelchResult {
	na, nb := len(a), len(b)
	if na < 2 || nb < 2 {
		return WelchResult{T: 0.0, DF: 0.0, PValue: 1.0}
	}
	meanA := mean(a)
	meanB := mean(b)
	varA := sampleVar(a, meanA)
	varB := sampleVar(b, meanB)

	sa := varA / float64(na)
	sb := varB / float64(nb)
	denom := sa + sb
	if denom <= 0.0 {
		// Both samples constant. Equal means -> no difference; differing
		// constant means -> infinitely significant difference (p -> 0).
		p := 0.0
		if math.Abs(meanA-meanB) <= 2.220446049250313e-16 {
			p = 1.0
		}
		return WelchResult{T: 0.0, DF: 0.0, PValue: p}
	}

	t := (meanA - meanB) / math.Sqrt(denom)
	df := (denom * denom) / (sa*sa/(float64(na)-1.0) + sb*sb/(float64(nb)-1.0))
	return WelchResult{T: t, DF: df, PValue: twoSidedP(t, df)}
}

func mean(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func sampleVar(xs []float64, m float64) float64 {
	var ss float64
	for _, x := range xs {
		d := x - m
		ss += d * d
	}
	return ss / (float64(len(xs)) - 1.0)
}

// twoSidedP is the two-sided p-value for a t-statistic with df degrees of
// freedom, via the regularized incomplete beta function:
//
//	p = I_{df/(df+t^2)}(df/2, 1/2)
func twoSidedP(t, df float64) float64 {
	if df <= 0.0 {
		return 1.0
	}
	x := df / (df + t*t)
	return clamp01(betai(df/2.0, 0.5, x))
}

func clamp01(v float64) float64 {
	if v < 0.0 {
		return 0.0
	}
	if v > 1.0 {
		return 1.0
	}
	return v
}

// betai is the regularized incomplete beta function I_x(a, b) via the Lentz
// continued fraction (Numerical Recipes' betai).
func betai(a, b, x float64) float64 {
	if x <= 0.0 {
		return 0.0
	}
	if x >= 1.0 {
		return 1.0
	}
	lnBeta := lnGamma(a+b) - lnGamma(a) - lnGamma(b)
	front := math.Exp(a*math.Log(x) + b*math.Log(1.0-x) + lnBeta)
	if x < (a+1.0)/(a+b+2.0) {
		return front * betacf(a, b, x) / a
	}
	return 1.0 - front*betacf(b, a, 1.0-x)/b
}

// betacf is the Lentz continued fraction for the incomplete beta.
func betacf(a, b, x float64) float64 {
	const maxIter = 200
	const eps = 3.0e-12
	const fpmin = 1.0e-300

	qab := a + b
	qap := a + 1.0
	qam := a - 1.0
	c := 1.0
	d := 1.0 - qab*x/qap
	if math.Abs(d) < fpmin {
		d = fpmin
	}
	d = 1.0 / d
	h := d

	for m := 1; m <= maxIter; m++ {
		mf := float64(m)
		m2 := 2.0 * mf
		// Even step.
		aa := mf * (b - mf) * x / ((qam + m2) * (a + m2))
		d = 1.0 + aa*d
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = 1.0 + aa/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1.0 / d
		h *= d * c
		// Odd step.
		aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
		d = 1.0 + aa*d
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = 1.0 + aa/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.Abs(del-1.0) < eps {
			break
		}
	}
	return h
}

// lnGamma is the Lanczos approximation of ln(Γ(x)) for x > 0.
func lnGamma(x float64) float64 {
	coef := [9]float64{
		0.9999999999998099,
		676.5203681218851,
		-1259.1392167224028,
		771.3234287776531,
		-176.6150291621406,
		12.507343278686905,
		-0.13857109526572012,
		9.984369578019572e-6,
		1.5056327351493116e-7,
	}
	const g = 7.0
	if x < 0.5 {
		// Reflection formula.
		return math.Log(math.Pi) - math.Log(math.Sin(math.Pi*x)) - lnGamma(1.0-x)
	}
	x -= 1.0
	a := coef[0]
	tt := x + g + 0.5
	for i := 1; i < len(coef); i++ {
		a += coef[i] / (x + float64(i))
	}
	return 0.5*math.Log(2.0*math.Pi) + (x+0.5)*math.Log(tt) - tt + math.Log(a)
}

// ============================================================================
// Bootstrap CI (Rule 27)
// ============================================================================

// ConfidenceInterval is a bootstrap confidence interval. Mirrors the CI
// reported on a MetricComparison.
type ConfidenceInterval struct {
	Lower float64 `json:"lower"`
	Upper float64 `json:"upper"`
	Level float64 `json:"level"`
}

// SplitMix64 is an inline SplitMix64 PRNG (Rule 27): a fixed, seedable,
// byte-identical generator so a bootstrap CI replays identically across all
// four languages.
type SplitMix64 struct {
	state uint64
}

// NewSplitMix64 constructs a SplitMix64 seeded with seed.
func NewSplitMix64(seed uint64) *SplitMix64 { return &SplitMix64{state: seed} }

// NextU64 returns the next 64-bit value (the canonical SplitMix64 step).
func (r *SplitMix64) NextU64() uint64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// NextIndex returns a uniform index in [0, n). n must be non-zero.
func (r *SplitMix64) NextIndex(n int) int {
	return int(r.NextU64() % uint64(n))
}

// DefaultBootstrapIterations is the default bootstrap iteration count (Rule 27).
const DefaultBootstrapIterations uint32 = 1000

// DefaultBootstrapSeed is the fixed bootstrap seed so replays are byte-identical
// cross-language.
const DefaultBootstrapSeed uint64 = 0x5EED5EED5EED5EED

// BootstrapCI is the percentile bootstrap CI for the mean (Rule 27). It
// resamples samples with replacement iterations times using a SplitMix64
// seeded with seed, then takes the [(1-level)/2, 1-(1-level)/2] percentiles of
// the resample means.
//
// level is the confidence level (e.g. 0.95). Returns (zero, false) for an
// empty sample.
func BootstrapCI(samples []float64, iterations uint32, level float64, seed uint64) (ConfidenceInterval, bool) {
	if len(samples) == 0 {
		return ConfidenceInterval{}, false
	}
	rng := NewSplitMix64(seed)
	means := make([]float64, 0, iterations)
	n := len(samples)
	for i := uint32(0); i < iterations; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += samples[rng.NextIndex(n)]
		}
		means = append(means, sum/float64(n))
	}
	sort.Float64s(means)
	alpha := 1.0 - level
	lower := Percentile(means, alpha/2.0*100.0)
	upper := Percentile(means, (1.0-alpha/2.0)*100.0)
	return ConfidenceInterval{Lower: lower, Upper: upper, Level: level}, true
}
