package stats

import (
	"math"
	"testing"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) < tol }

func TestMetricStatsBasic(t *testing.T) {
	s := StatsFromSamples([]float64{1.0, 2.0, 3.0, 4.0})
	if !approx(s.Mean, 2.5, 1e-12) {
		t.Fatalf("mean=%v", s.Mean)
	}
	if !approx(s.Stddev, math.Sqrt(5.0/3.0), 1e-9) {
		t.Fatalf("stddev=%v", s.Stddev)
	}
	if s.N != 4 {
		t.Fatalf("n=%d", s.N)
	}
	if !approx(s.P50, 2.0, 1e-12) {
		t.Fatalf("p50=%v", s.P50)
	}
	if !approx(s.P95, 4.0, 1e-12) {
		t.Fatalf("p95=%v", s.P95)
	}
}

func TestMetricStatsSingleSampleZeroStddev(t *testing.T) {
	s := StatsFromSamples([]float64{7.0})
	if s.N != 1 || !approx(s.Stddev, 0.0, 1e-12) || !approx(s.Mean, 7.0, 1e-12) {
		t.Fatalf("got %+v", s)
	}
}

func TestLnGammaKnownValues(t *testing.T) {
	if !approx(lnGamma(5.0), math.Log(24.0), 1e-7) {
		t.Fatalf("lnGamma(5)=%v", lnGamma(5.0))
	}
	if !approx(lnGamma(0.5), math.Log(math.Sqrt(math.Pi)), 1e-7) {
		t.Fatalf("lnGamma(0.5)=%v", lnGamma(0.5))
	}
}

func TestBetaiSymmetryAndEndpoints(t *testing.T) {
	a, b, x := 2.5, 3.5, 0.3
	left := betai(a, b, x)
	right := 1.0 - betai(b, a, 1.0-x)
	if !approx(left, right, 1e-9) {
		t.Fatalf("symmetry: %v vs %v", left, right)
	}
	if !approx(betai(a, b, 0.0), 0.0, 1e-12) || !approx(betai(a, b, 1.0), 1.0, 1e-12) {
		t.Fatalf("endpoints wrong")
	}
}

func TestWelchTTestOracle(t *testing.T) {
	a := []float64{27, 31, 29, 30}
	b := []float64{20, 22, 24, 21}
	r := WelchTTest(a, b)
	if !approx(r.T, 6.21063, 1e-4) {
		t.Fatalf("t=%v", r.T)
	}
	if !approx(r.DF, 6.0, 1e-3) {
		t.Fatalf("df=%v", r.DF)
	}
	if !(r.PValue < 0.005 && r.PValue > 0.0) {
		t.Fatalf("p=%v", r.PValue)
	}
}

func TestWelchIdenticalSamplesPOne(t *testing.T) {
	r := WelchTTest([]float64{1, 2, 3, 4}, []float64{1, 2, 3, 4})
	if !approx(r.T, 0.0, 1e-12) || !approx(r.PValue, 1.0, 1e-6) {
		t.Fatalf("got %+v", r)
	}
}

func TestWelchConstantDifferingMeansPZero(t *testing.T) {
	r := WelchTTest([]float64{5, 5, 5}, []float64{3, 3, 3})
	if !approx(r.PValue, 0.0, 1e-12) {
		t.Fatalf("p=%v", r.PValue)
	}
}

func TestWelchTinySamplesPOne(t *testing.T) {
	r := WelchTTest([]float64{1}, []float64{2, 3})
	if !approx(r.PValue, 1.0, 1e-12) {
		t.Fatalf("p=%v", r.PValue)
	}
}

func TestSplitMix64IsDeterministic(t *testing.T) {
	r1 := NewSplitMix64(42)
	r2 := NewSplitMix64(42)
	for i := 0; i < 100; i++ {
		if r1.NextU64() != r2.NextU64() {
			t.Fatalf("diverged at %d", i)
		}
	}
	r0 := NewSplitMix64(0)
	if got := r0.NextU64(); got != 16294208416658607535 {
		t.Fatalf("seed-0 first output = %d", got)
	}
}

func TestBootstrapCIBracketsMeanAndIsSeeded(t *testing.T) {
	samples := []float64{10, 12, 11, 13, 9, 14, 8, 15}
	ci1, ok1 := BootstrapCI(samples, 1000, 0.95, DefaultBootstrapSeed)
	ci2, ok2 := BootstrapCI(samples, 1000, 0.95, DefaultBootstrapSeed)
	if !ok1 || !ok2 || ci1 != ci2 {
		t.Fatalf("not seeded: %+v vs %+v", ci1, ci2)
	}
	mean := 0.0
	for _, x := range samples {
		mean += x
	}
	mean /= float64(len(samples))
	if !(ci1.Lower <= mean && mean <= ci1.Upper && ci1.Lower < ci1.Upper) {
		t.Fatalf("ci doesn't bracket mean: %+v mean=%v", ci1, mean)
	}
}

func TestBootstrapCIEmptyIsNone(t *testing.T) {
	if _, ok := BootstrapCI(nil, 100, 0.95, 1); ok {
		t.Fatalf("empty should be none")
	}
}
