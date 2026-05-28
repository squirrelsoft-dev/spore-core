package stats

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// welch_bootstrap.json is the cross-language statistics oracle (Rule 29). Every
// language's WelchTTest + BootstrapCI must reproduce these values byte-for-byte.
// Never edit the fixture to make this pass.

type oracleFixture struct {
	Description         string  `json:"description"`
	BootstrapSeedHex    string  `json:"bootstrap_seed"`
	BootstrapIterations uint32  `json:"bootstrap_iterations"`
	CILevel             float64 `json:"ci_level"`
	Cases               []struct {
		Name                 string             `json:"name"`
		Baseline             []float64          `json:"baseline"`
		Candidate            []float64          `json:"candidate"`
		WelchT               float64            `json:"welch_t"`
		WelchDF              float64            `json:"welch_df"`
		WelchPValue          float64            `json:"welch_p_value"`
		WelchPTolerance      float64            `json:"welch_p_tolerance"`
		CandidateBootstrapCI ConfidenceInterval `json:"candidate_bootstrap_ci"`
		BaselineBootstrapCI  ConfidenceInterval `json:"baseline_bootstrap_ci"`
	} `json:"cases"`
}

func TestWelchBootstrapOracleFixture(t *testing.T) {
	path := filepath.Join("..", "..", "..", "fixtures", "task_suites", "welch_bootstrap.json")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx oracleFixture
	if err := json.Unmarshal(body, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	// Parse the seed hex (e.g. "0x5EED5EED5EED5EED").
	var seed uint64
	if _, err := scanHex(fx.BootstrapSeedHex, &seed); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if seed != DefaultBootstrapSeed {
		t.Fatalf("fixture seed %#x != DefaultBootstrapSeed %#x", seed, DefaultBootstrapSeed)
	}

	for _, c := range fx.Cases {
		welch := WelchTTest(c.Baseline, c.Candidate)
		// t sign depends on argument order; the oracle compares magnitudes
		// (the two-sided p-value is sign-invariant). Mirrors the Rust oracle.
		if !approx(math.Abs(welch.T), math.Abs(c.WelchT), 1e-9) {
			t.Fatalf("%s: welch |t| = %v want %v", c.Name, math.Abs(welch.T), math.Abs(c.WelchT))
		}
		if !approx(welch.DF, c.WelchDF, 1e-9) {
			t.Fatalf("%s: welch df = %v want %v", c.Name, welch.DF, c.WelchDF)
		}
		tol := c.WelchPTolerance
		if tol == 0 {
			tol = 1e-9
		}
		if !approx(welch.PValue, c.WelchPValue, tol) {
			t.Fatalf("%s: welch p = %.18g want %.18g", c.Name, welch.PValue, c.WelchPValue)
		}

		candCI, ok := BootstrapCI(c.Candidate, fx.BootstrapIterations, fx.CILevel, seed)
		if !ok {
			t.Fatalf("%s: candidate CI not computed", c.Name)
		}
		if candCI.Lower != c.CandidateBootstrapCI.Lower || candCI.Upper != c.CandidateBootstrapCI.Upper {
			t.Fatalf("%s: candidate CI = %+v want lower=%v upper=%v", c.Name, candCI, c.CandidateBootstrapCI.Lower, c.CandidateBootstrapCI.Upper)
		}
		baseCI, ok := BootstrapCI(c.Baseline, fx.BootstrapIterations, fx.CILevel, seed)
		if !ok {
			t.Fatalf("%s: baseline CI not computed", c.Name)
		}
		if baseCI.Lower != c.BaselineBootstrapCI.Lower || baseCI.Upper != c.BaselineBootstrapCI.Upper {
			t.Fatalf("%s: baseline CI = %+v want lower=%v upper=%v", c.Name, baseCI, c.BaselineBootstrapCI.Lower, c.BaselineBootstrapCI.Upper)
		}
	}
}

// scanHex parses a "0x..."-prefixed hex string into a uint64.
func scanHex(s string, out *uint64) (int, error) {
	var v uint64
	str := s
	if len(str) >= 2 && (str[:2] == "0x" || str[:2] == "0X") {
		str = str[2:]
	}
	for i := 0; i < len(str); i++ {
		c := str[i]
		var d uint64
		switch {
		case c >= '0' && c <= '9':
			d = uint64(c - '0')
		case c >= 'a' && c <= 'f':
			d = uint64(c-'a') + 10
		case c >= 'A' && c <= 'F':
			d = uint64(c-'A') + 10
		default:
			return 0, errBadHex
		}
		v = v<<4 | d
	}
	*out = v
	return 1, nil
}

var errBadHex = &hexErr{}

type hexErr struct{}

func (*hexErr) Error() string { return "bad hex digit" }
