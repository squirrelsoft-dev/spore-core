package termination

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/memory"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/middleware"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/sensor"
)

func u32(v uint32) *uint32               { return &v }
func f64(v float64) *float64             { return &v }
func dur(d time.Duration) *time.Duration { return &d }

func snapshot() SessionStateSnapshot {
	return NewSessionStateSnapshot(
		sporecore.SessionID("s1"),
		sporecore.TaskID("t1"),
		sporecore.SessionState{},
	)
}

func inputAt(turn uint32, done bool) TerminationInput {
	resp := "ok"
	return TerminationInput{
		SessionID:       sporecore.SessionID("s1"),
		TaskID:          sporecore.TaskID("t1"),
		TurnNumber:      turn,
		AgentClaimsDone: done,
		AgentResponse:   &resp,
		BudgetUsed:      sporecore.BudgetSnapshot{},
		BudgetLimits:    sporecore.BudgetLimits{},
		SensorResults:   nil,
		SessionState:    snapshot(),
	}
}

func sensorResult(id string, outcome sensor.SensorOutcome, detail string) sensor.SensorResult {
	return sensor.SensorResult{
		SensorID: sensor.SensorID(id),
		Outcome:  outcome,
		Detail:   detail,
		FiredAt:  memory.Timestamp("2026-05-17T00:00:00Z"),
	}
}

// ── Rule: budget is always checked first ──────────────────────────────────

func TestBudgetHardStopWhenDone(t *testing.T) {
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, true)
	in.BudgetUsed.Turns = 5
	in.BudgetLimits.MaxTurns = u32(5)
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionHaltBudgetExceeded {
		t.Fatalf("expected HaltBudgetExceeded, got %v", d.Kind)
	}
	if d.LimitType != sporecore.BudgetLimitTurns {
		t.Fatalf("expected turns limit, got %v", d.LimitType)
	}
}

func TestBudgetHardStopWhenNotDone(t *testing.T) {
	// Budget is checked before AgentClaimsDone.
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, false)
	in.BudgetUsed.Turns = 5
	in.BudgetLimits.MaxTurns = u32(5)
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionHaltBudgetExceeded {
		t.Fatalf("expected HaltBudgetExceeded, got %v", d.Kind)
	}
}

func TestBudgetCheckCoversEveryLimitType(t *testing.T) {
	cases := []struct {
		name string
		snap sporecore.BudgetSnapshot
		lim  sporecore.BudgetLimits
		want sporecore.BudgetLimitType
	}{
		{"turns", sporecore.BudgetSnapshot{Turns: 3}, sporecore.BudgetLimits{MaxTurns: u32(3)}, sporecore.BudgetLimitTurns},
		{"input_tokens", sporecore.BudgetSnapshot{InputTokens: 10}, sporecore.BudgetLimits{MaxInputTokens: u32(10)}, sporecore.BudgetLimitInputTokens},
		{"output_tokens", sporecore.BudgetSnapshot{OutputTokens: 10}, sporecore.BudgetLimits{MaxOutputTokens: u32(10)}, sporecore.BudgetLimitOutputTokens},
		{"wall_time", sporecore.BudgetSnapshot{WallTime: dur(10 * time.Second)}, sporecore.BudgetLimits{MaxWallTime: dur(10 * time.Second)}, sporecore.BudgetLimitWallTime},
		{"cost_usd", sporecore.BudgetSnapshot{CostUSD: 1.0}, sporecore.BudgetLimits{MaxCostUSD: f64(1.0)}, sporecore.BudgetLimitCostUSD},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, ok := CheckBudgetDefault(&c.snap, &c.lim)
			if !ok {
				t.Fatalf("expected halt for %s", c.name)
			}
			if d.LimitType != c.want {
				t.Fatalf("expected limit_type %v, got %v", c.want, d.LimitType)
			}
		})
	}
	if _, ok := CheckBudgetDefault(&sporecore.BudgetSnapshot{}, &sporecore.BudgetLimits{}); ok {
		t.Fatalf("expected no halt for zero-valued snapshot/limits")
	}
}

// ── Rule: not-done always continues (after budget) ────────────────────────

func TestNotDoneContinues(t *testing.T) {
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, false)
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionContinue {
		t.Fatalf("expected Continue, got %v", d.Kind)
	}
}

// ── Rule: sensor halt becomes UnrecoverableSensorHalt ─────────────────────

func TestSensorHaltOverridesCompletionSuccess(t *testing.T) {
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, true)
	in.SensorResults = append(in.SensorResults, sensorResult("guardrail", sensor.OutcomeHalt, "tripped"))
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionHaltFailure {
		t.Fatalf("expected HaltFailure, got %v", d.Kind)
	}
	if d.Reason.Kind != ReasonUnrecoverableSensorHalt {
		t.Fatalf("expected UnrecoverableSensorHalt, got %v", d.Reason.Kind)
	}
	if d.Reason.SensorID != "guardrail" {
		t.Fatalf("unexpected sensor id: %q", d.Reason.SensorID)
	}
}

func TestSensorWarnDoesNotHalt(t *testing.T) {
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, true)
	in.SensorResults = append(in.SensorResults, sensorResult("guardrail", sensor.OutcomeWarn, "soft"))
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionHaltSuccess {
		t.Fatalf("expected HaltSuccess, got %v", d.Kind)
	}
}

// ── Rule: completion check returning incomplete ⇒ Continue ────────────────

func TestIncompleteCheckContinuesWithAgentClaimedDone(t *testing.T) {
	p := NewStandardTerminationPolicy(NewFixedIncomplete("feature B not implemented"))
	in := inputAt(1, true)
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionContinue {
		t.Fatalf("expected Continue, got %v", d.Kind)
	}
}

// ── Rule: completion check returning complete ⇒ HaltSuccess(summary) ──────

func TestCompleteCheckHaltsSuccessWithSummary(t *testing.T) {
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, true)
	s := "all green"
	in.AgentResponse = &s
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionHaltSuccess {
		t.Fatalf("expected HaltSuccess, got %v", d.Kind)
	}
	if d.Summary != "all green" {
		t.Fatalf("expected summary 'all green', got %q", d.Summary)
	}
}

func TestHaltSuccessSummaryEmptyWhenNoResponse(t *testing.T) {
	p := NewStandardTerminationPolicyWithNullCheck()
	in := inputAt(1, true)
	in.AgentResponse = nil
	d, err := p.Evaluate(context.Background(), &in)
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if d.Kind != DecisionHaltSuccess {
		t.Fatalf("expected HaltSuccess, got %v", d.Kind)
	}
	if d.Summary != "" {
		t.Fatalf("expected empty summary, got %q", d.Summary)
	}
}

// ── Wire format: TerminationFailureReason round-trips every variant ───────

func TestFailureReasonJSONRoundTrip(t *testing.T) {
	cases := []TerminationFailureReason{
		NewReasonCompletionCheckFailed("nope"),
		NewReasonMaxRetriesExhausted("bash", 3),
		NewReasonUnrecoverableSensorHalt(sensor.SensorID("g"), "tripped"),
		NewReasonMiddlewareHalt(middleware.HookBeforeTurn, "veto"),
		NewReasonAgentError(sporecore.NewEmptyResponseError()),
		NewReasonPolicyViolation("policy"),
		NewReasonHumanHalted(),
	}
	for _, r := range cases {
		raw, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal %v: %v", r.Kind, err)
		}
		var back TerminationFailureReason
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatalf("unmarshal %v: %v", r.Kind, err)
		}
		if back.Kind != r.Kind {
			t.Fatalf("round-trip kind mismatch: got %v, want %v", back.Kind, r.Kind)
		}
	}
}

// ── Wire format: TerminationDecision JSON shape matches Rust serde ────────

func TestDecisionJSONShape(t *testing.T) {
	d := NewDecisionHaltBudgetExceeded(
		sporecore.BudgetLimitTurns,
		NewBudgetValueTurns(10),
		NewBudgetValueTurns(10),
	)
	raw, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["kind"] != "halt_budget_exceeded" {
		t.Fatalf("expected kind halt_budget_exceeded, got %v", got["kind"])
	}
	if got["limit_type"] != "turns" {
		t.Fatalf("expected limit_type turns, got %v", got["limit_type"])
	}
	used := got["used"].(map[string]any)
	if used["kind"] != "turns" || used["value"].(float64) != 10 {
		t.Fatalf("unexpected used: %v", used)
	}
}

func TestBudgetValueJSONRoundTrip(t *testing.T) {
	cases := []BudgetValue{
		NewBudgetValueTurns(5),
		NewBudgetValueTokens(1234),
		NewBudgetValueDuration(60),
		NewBudgetValueUSD(2.5),
	}
	for _, v := range cases {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal %v: %v", v.Kind, err)
		}
		var back BudgetValue
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatalf("unmarshal %v: %v", v.Kind, err)
		}
		if back != v {
			t.Fatalf("round-trip mismatch: got %+v, want %+v", back, v)
		}
	}
}

// ── Completion checks ─────────────────────────────────────────────────────

func TestNullCompletionCheck(t *testing.T) {
	_, complete, err := NullCompletionCheck{}.Check(context.Background(), nil)
	if err != nil || !complete {
		t.Fatalf("expected complete with no error, got complete=%v err=%v", complete, err)
	}
	if (NullCompletionCheck{}).Description() == "" {
		t.Fatalf("description must be non-empty")
	}
}

func TestFixedCompletionCheck(t *testing.T) {
	c := NewFixedComplete()
	if reason, complete, _ := c.Check(context.Background(), nil); !complete || reason != "" {
		t.Fatalf("expected complete with empty reason")
	}
	ic := NewFixedIncomplete("nope")
	reason, complete, _ := ic.Check(context.Background(), nil)
	if complete || reason != "nope" {
		t.Fatalf("expected incomplete reason 'nope', got complete=%v reason=%q", complete, reason)
	}
}
