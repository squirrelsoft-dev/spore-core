package sensor

import (
	"context"
	"encoding/json"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// stubSensor is a programmable Sensor used across the test suite.
type stubSensor struct {
	cfg     SensorConfig
	outcome SensorOutcome
}

func (s *stubSensor) Evaluate(_ context.Context, _ *SensorInput) SensorResult {
	var obs *string
	if s.outcome == OutcomeWarn {
		v := "warn-obs"
		obs = &v
	}
	return SensorResult{
		SensorID:    s.cfg.ID,
		Outcome:     s.outcome,
		Observation: obs,
		Detail:      string(s.outcome),
		FiredAt:     "2026-05-16T00:00:00Z",
	}
}

func (s *stubSensor) Config() SensorConfig { return s.cfg }

func computational(id string, triggers []SensorTrigger, outcome SensorOutcome) *stubSensor {
	return &stubSensor{
		cfg: SensorConfig{
			ID:                 SensorID(id),
			Name:               id,
			Kind:               SensorKindComputational,
			Triggers:           triggers,
			LowSignalThreshold: DefaultSensorSignalThresholds(),
		},
		outcome: outcome,
	}
}

func inferential(
	id string,
	triggers []SensorTrigger,
	outcome SensorOutcome,
	everyN *uint32,
	phases []sporecore.TaskPhase,
) *stubSensor {
	return &stubSensor{
		cfg: SensorConfig{
			ID:                 SensorID(id),
			Name:               id,
			Kind:               SensorKindInferential,
			Triggers:           triggers,
			RunEveryNTurns:     everyN,
			RunOnPhases:        phases,
			LowSignalThreshold: DefaultSensorSignalThresholds(),
		},
		outcome: outcome,
	}
}

func input(sid string) *SensorInput {
	in := NewSensorInput(sporecore.SessionID(sid), sporecore.SessionState{})
	return &in
}

func u32(v uint32) *uint32 { return &v }

// ── Rule: register validates triggers ───────────────────────────────────────

func TestRegisterRejectsEmptyTriggers(t *testing.T) {
	chain := NewStandardSensorChain()
	err := chain.Register(context.Background(), computational("s1", nil, OutcomePass))
	if err == nil {
		t.Fatal("expected error")
	}
	se, ok := err.(*SensorError)
	if !ok || se.Kind != ErrKindValidationFailed {
		t.Fatalf("expected ValidationFailed, got %v", err)
	}
}

func TestRegisterRejectsDuplicateIDs(t *testing.T) {
	chain := NewStandardSensorChain()
	if err := chain.Register(context.Background(), computational("s1",
		[]SensorTrigger{NewTriggerPostTurn()}, OutcomePass)); err != nil {
		t.Fatal(err)
	}
	err := chain.Register(context.Background(), computational("s1",
		[]SensorTrigger{NewTriggerPostTurn()}, OutcomePass))
	se, ok := err.(*SensorError)
	if !ok || se.Kind != ErrKindAlreadyRegistered {
		t.Fatalf("expected AlreadyRegistered, got %v", err)
	}
}

// ── Rule: fire runs every matching sensor, returns all results ──────────────

func TestFireRunsAllMatchingSensorsNoShortCircuit(t *testing.T) {
	chain := NewStandardSensorChain()
	for _, o := range []SensorOutcome{OutcomePass, OutcomeWarn, OutcomeHalt} {
		if err := chain.Register(context.Background(), computational(string(o),
			[]SensorTrigger{NewTriggerPostTurn()}, o)); err != nil {
			t.Fatal(err)
		}
	}
	results := chain.Fire(context.Background(), NewTriggerPostTurn(), input("s1"))
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	seen := map[SensorOutcome]bool{}
	for _, r := range results {
		seen[r.Outcome] = true
	}
	for _, o := range []SensorOutcome{OutcomePass, OutcomeWarn, OutcomeHalt} {
		if !seen[o] {
			t.Errorf("missing outcome %q", o)
		}
	}
}

// ── Rule: triggers filter ───────────────────────────────────────────────────

func TestFireIgnoresSensorsWithoutMatchingTrigger(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), computational("post-turn",
		[]SensorTrigger{NewTriggerPostTurn()}, OutcomePass))
	_ = chain.Register(context.Background(), computational("post-session",
		[]SensorTrigger{NewTriggerPostSession()}, OutcomePass))
	results := chain.Fire(context.Background(), NewTriggerPostTurn(), input("s1"))
	if len(results) != 1 || results[0].SensorID != "post-turn" {
		t.Fatalf("expected 1 result for post-turn, got %+v", results)
	}
}

// ── Rule: PostTool wildcard / named matching ────────────────────────────────

func TestPostToolWildcardAndNamedMatching(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), computational("any",
		[]SensorTrigger{NewTriggerPostTool("")}, OutcomePass))
	_ = chain.Register(context.Background(), computational("bash-only",
		[]SensorTrigger{NewTriggerPostTool("bash")}, OutcomePass))

	r1 := chain.Fire(context.Background(), NewTriggerPostTool("bash"), input("s1"))
	if len(r1) != 2 {
		t.Fatalf("expected 2 results for bash, got %d", len(r1))
	}
	r2 := chain.Fire(context.Background(), NewTriggerPostTool("edit"), input("s2"))
	if len(r2) != 1 || r2[0].SensorID != "any" {
		t.Fatalf("expected only 'any' for edit, got %+v", r2)
	}
}

// ── Rule: computational ignores RunEveryNTurns ──────────────────────────────

func TestComputationalIgnoresTurnGating(t *testing.T) {
	chain := NewStandardSensorChain()
	s := computational("c", []SensorTrigger{NewTriggerPostTurn()}, OutcomePass)
	s.cfg.RunEveryNTurns = u32(99)
	if err := chain.Register(context.Background(), s); err != nil {
		t.Fatal(err)
	}
	in := input("s1")
	tn := uint32(1)
	in.TurnNumber = &tn
	if r := chain.Fire(context.Background(), NewTriggerPostTurn(), in); len(r) != 1 {
		t.Fatalf("expected computational to fire, got %d", len(r))
	}
}

// ── Rule: inferential gated by RunEveryNTurns ───────────────────────────────

func TestInferentialRunEveryNTurnsGating(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), inferential(
		"judge",
		[]SensorTrigger{NewTriggerPostTurn()},
		OutcomeWarn,
		u32(3),
		nil,
	))
	in := input("s1")
	for _, tc := range []struct {
		turn   uint32
		expect int
	}{
		{1, 0},
		{2, 0},
		{3, 1},
		{6, 1},
	} {
		tn := tc.turn
		in.TurnNumber = &tn
		r := chain.Fire(context.Background(), NewTriggerPostTurn(), in)
		if len(r) != tc.expect {
			t.Errorf("turn %d: expected %d, got %d", tc.turn, tc.expect, len(r))
		}
	}
}

// ── Rule: inferential gated by RunOnPhases ──────────────────────────────────

func TestInferentialRunOnPhasesGating(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), inferential(
		"judge",
		[]SensorTrigger{NewTriggerPostTurn()},
		OutcomePass,
		nil,
		[]sporecore.TaskPhase{sporecore.PhaseExecution},
	))
	in := input("s1")
	planning := sporecore.PhasePlanning
	in.Phase = &planning
	if r := chain.Fire(context.Background(), NewTriggerPostTurn(), in); len(r) != 0 {
		t.Fatalf("expected 0 when phase=Planning, got %d", len(r))
	}
	execution := sporecore.PhaseExecution
	in.Phase = &execution
	if r := chain.Fire(context.Background(), NewTriggerPostTurn(), in); len(r) != 1 {
		t.Fatalf("expected 1 when phase=Execution, got %d", len(r))
	}
}

// ── Rule: stats aggregates outcomes and fire_rate ───────────────────────────

func TestStatsAggregatesOutcomesAndFireRate(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), computational("warner",
		[]SensorTrigger{NewTriggerPostTurn()}, OutcomeWarn))
	for i := 0; i < 4; i++ {
		chain.Fire(context.Background(), NewTriggerPostTurn(), input(string(rune('a'+i))))
	}
	stats := chain.Stats(context.Background(), nil)
	if len(stats) != 1 {
		t.Fatalf("expected 1 stat, got %d", len(stats))
	}
	s := stats[0]
	if s.TotalFires != 4 || s.WarnCount != 4 || s.HaltCount != 0 || s.PassCount != 0 {
		t.Fatalf("aggregation off: %+v", s)
	}
	if d := s.FireRate - 1.0; d < -1e-6 || d > 1e-6 {
		t.Fatalf("fire_rate expected ~1.0, got %f", s.FireRate)
	}
}

// ── Rule: SignalQualityReport flags AlwaysFiring ────────────────────────────

func TestSignalQualityFlagsAlwaysFiring(t *testing.T) {
	chain := NewStandardSensorChain()
	s := computational("noisy", []SensorTrigger{NewTriggerPostTurn()}, OutcomeWarn)
	s.cfg.LowSignalThreshold = SensorSignalThresholds{
		NeverFiredAfterNSessions: 100,
		AlwaysFiredRate:          0.5,
	}
	_ = chain.Register(context.Background(), s)
	for i := 0; i < 5; i++ {
		chain.Fire(context.Background(), NewTriggerPostTurn(), input(string(rune('a'+i))))
	}
	flags := chain.SignalQualityReport(context.Background(), 5)
	found := false
	for _, f := range flags {
		if f.Kind == FlagKindAlwaysFiring && f.SensorID == "noisy" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected AlwaysFiring flag for 'noisy', got %+v", flags)
	}
}

// ── Rule: SignalQualityReport flags NeverFired ──────────────────────────────

func TestSignalQualityFlagsNeverFired(t *testing.T) {
	chain := NewStandardSensorChain()
	s := computational("quiet", []SensorTrigger{NewTriggerPostSession()}, OutcomePass)
	s.cfg.LowSignalThreshold = SensorSignalThresholds{
		NeverFiredAfterNSessions: 3,
		AlwaysFiredRate:          0.9,
	}
	_ = chain.Register(context.Background(), s)
	for i := 0; i < 5; i++ {
		chain.Fire(context.Background(), NewTriggerPostTurn(), input(string(rune('a'+i))))
	}
	flags := chain.SignalQualityReport(context.Background(), 3)
	found := false
	for _, f := range flags {
		if f.Kind == FlagKindNeverFired && f.SensorID == "quiet" && f.SessionsObserved >= 3 {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected NeverFired flag for 'quiet', got %+v", flags)
	}
}

// ── Rule: SignalQualityReport respects min_sessions ─────────────────────────

func TestSignalQualityRespectsMinSessions(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), computational("quiet",
		[]SensorTrigger{NewTriggerPostSession()}, OutcomePass))
	chain.Fire(context.Background(), NewTriggerPostTurn(), input("s1"))
	flags := chain.SignalQualityReport(context.Background(), 10)
	if len(flags) != 0 {
		t.Fatalf("expected empty flags below min_sessions, got %+v", flags)
	}
}

// ── Edge: history records every fire ────────────────────────────────────────

func TestHistoryIsRecordedInFireOrder(t *testing.T) {
	chain := NewStandardSensorChain()
	_ = chain.Register(context.Background(), computational("s1",
		[]SensorTrigger{NewTriggerPostTurn()}, OutcomePass))
	chain.Fire(context.Background(), NewTriggerPostTurn(), input("a"))
	chain.Fire(context.Background(), NewTriggerPostTurn(), input("b"))
	stats := chain.Stats(context.Background(), nil)
	if stats[0].TotalFires != 2 {
		t.Fatalf("expected 2 fires, got %d", stats[0].TotalFires)
	}
}

// ── Trigger JSON round-trip ─────────────────────────────────────────────────

func TestSensorTriggerJSONRoundTrip(t *testing.T) {
	cases := []SensorTrigger{
		NewTriggerPostTool("bash"),
		NewTriggerPostTool(""),
		NewTriggerPostTurn(),
		NewTriggerPostSession(),
		NewTriggerContinuous(),
		NewTriggerOnToolError(),
		NewTriggerOnCompaction(),
	}
	for _, c := range cases {
		b, err := json.Marshal(c)
		if err != nil {
			t.Fatal(err)
		}
		var got SensorTrigger
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatal(err)
		}
		if got != c {
			t.Errorf("round-trip mismatch: got %+v want %+v (raw=%s)", got, c, b)
		}
	}
}
