package sensor

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// fixtureSensor / fixtureEvent / fixtureExpected mirror the JSON shape in
// fixtures/sensor_chain/*.json. The structures must match the Rust reference
// (`rust/crates/spore-core/src/sensor.rs::FixtureCase`) so the shared
// fixture deserializes here without modification.

type fixtureSensor struct {
	ID         string                 `json:"id"`
	Kind       SensorKind             `json:"kind"`
	Triggers   []SensorTrigger        `json:"triggers"`
	Outcome    SensorOutcome          `json:"outcome"`
	Thresholds SensorSignalThresholds `json:"thresholds"`
}

type fixtureEvent struct {
	Trigger   SensorTrigger `json:"trigger"`
	SessionID string        `json:"session_id"`
}

type fixtureExpected struct {
	NeverFired   []string `json:"never_fired"`
	AlwaysFiring []string `json:"always_firing"`
}

type fixtureCase struct {
	Sensors     []fixtureSensor `json:"sensors"`
	Events      []fixtureEvent  `json:"events"`
	MinSessions uint32          `json:"min_sessions"`
	Expected    fixtureExpected `json:"expected"`
}

// TestSensorChainFixtureReplay replays the shared signal_quality_basic.json
// fixture and asserts the same NeverFired / AlwaysFiring sets the Rust
// reference produces.
func TestSensorChainFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/sensor → ../../../fixtures/sensor_chain/...
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "sensor_chain", "signal_quality_basic.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fc fixtureCase
	if err := json.Unmarshal(raw, &fc); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	chain := NewStandardSensorChain()
	for _, s := range fc.Sensors {
		cfg := SensorConfig{
			ID:                 SensorID(s.ID),
			Name:               s.ID,
			Kind:               s.Kind,
			Triggers:           s.Triggers,
			LowSignalThreshold: s.Thresholds,
		}
		if err := chain.Register(context.Background(), &stubSensor{cfg: cfg, outcome: s.Outcome}); err != nil {
			t.Fatalf("register %s: %v", s.ID, err)
		}
	}
	for _, ev := range fc.Events {
		chain.Fire(context.Background(), ev.Trigger, input(ev.SessionID))
	}
	flags := chain.SignalQualityReport(context.Background(), fc.MinSessions)

	gotNever := []string{}
	gotAlways := []string{}
	for _, f := range flags {
		switch f.Kind {
		case FlagKindNeverFired:
			gotNever = append(gotNever, string(f.SensorID))
		case FlagKindAlwaysFiring:
			gotAlways = append(gotAlways, string(f.SensorID))
		}
	}
	sort.Strings(gotNever)
	sort.Strings(gotAlways)
	wantNever := append([]string(nil), fc.Expected.NeverFired...)
	wantAlways := append([]string(nil), fc.Expected.AlwaysFiring...)
	sort.Strings(wantNever)
	sort.Strings(wantAlways)

	if !equalStringSlices(gotNever, wantNever) {
		t.Errorf("never_fired mismatch: got %v want %v", gotNever, wantNever)
	}
	if !equalStringSlices(gotAlways, wantAlways) {
		t.Errorf("always_firing mismatch: got %v want %v", gotAlways, wantAlways)
	}
}

func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
