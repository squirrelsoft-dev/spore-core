package sporecore

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
)

// ============================================================================
// SC-5 — EscalationMode::AutoContinue ("autonomous but capped") (#155)
//
// Mirrors rust/crates/spore-core/src/{execution_registry,harness}.rs (f1c0beb).
// Same grant mechanics, same cap-then-fail terminal. EscalationMode is NEVER
// serialized in fixtures (no wire impact); OnGrant is runtime-only.
// ============================================================================

// ── EscalationMode JSON: the AutoContinue kind (OnGrant never serialized) ────

func TestEscalationModeAutoContinueJSONRoundTrip(t *testing.T) {
	fired := 0
	m := AutoContinueEscalation(3, 2, func(AutoGrantInfo) { fired++ })
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// OnGrant is NOT serialized (funcs aren't JSON-serializable).
	want := `{"kind":"auto_continue","max_grants":3,"steps_per_grant":2}`
	if string(b) != want {
		t.Fatalf("wire = %s, want %s", b, want)
	}
	var got EscalationMode
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Kind != EscalationAutoContinue || got.MaxGrants != 3 || got.StepsPerGrant != 2 {
		t.Fatalf("round-trip = %+v", got)
	}
	if got.OnGrant != nil {
		t.Fatal("OnGrant must not survive a round-trip (it is never serialized)")
	}
}

// ── BudgetContext.GrantAutoContinue grant mechanics ─────────────────────────

// The grant raises the scope cap to StepsTaken + stepsPerGrant (additive, NO
// StepsTaken rewind), so the loop gets exactly stepsPerGrant more steps —
// distinct from ConsumeContinue.
func TestGrantAutoContinueIsAdditiveAndCounted(t *testing.T) {
	scope := NewBudgetContext(
		BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		"react",
	)
	if scope.Charge(2) != nil {
		t.Fatal("first charge of 2 should fit")
	}
	if scope.Charge(1) == nil {
		t.Fatal("exhausts at the cap")
	}
	if scope.AutoGrantsRemaining(3) != 3 {
		t.Fatalf("AutoGrantsRemaining(3) = %d, want 3", scope.AutoGrantsRemaining(3))
	}
	scope.GrantAutoContinue(2) // cap → StepsTaken(2) + 2 = 4
	if scope.AutoGrantsUsed != 1 {
		t.Fatalf("AutoGrantsUsed = %d, want 1", scope.AutoGrantsUsed)
	}
	if scope.AutoGrantsRemaining(3) != 2 {
		t.Fatalf("AutoGrantsRemaining(3) = %d, want 2", scope.AutoGrantsRemaining(3))
	}
	if scope.Charge(2) != nil {
		t.Fatal("two more steps should fit after the grant")
	}
	if scope.Charge(1) == nil {
		t.Fatal("and it exhausts again at the raised cap")
	}
}

// ── ExecutionContext.TryAutoContinue grants until capped, then false ─────────

func TestTryAutoContinueGrantsUntilCappedThenFalse(t *testing.T) {
	registry := NewExecutionRegistryBuilder().Build()
	cx := NewExecutionContext(&registry)
	cx.PushBudget(
		BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		"react",
	)

	var count atomic.Uint32
	mode := AutoContinueEscalation(2, 2, func(info AutoGrantInfo) {
		if info.StepsGranted != 2 {
			t.Errorf("StepsGranted = %d, want 2", info.StepsGranted)
		}
		if info.Phase != "react" {
			t.Errorf("Phase = %q, want react", info.Phase)
		}
		count.Add(1)
	})

	if cx.ChargeCurrent(2) != nil {
		t.Fatal("first charge of 2 should fit")
	}
	if cx.ChargeCurrent(1) == nil {
		t.Fatal("exhausts at the cap")
	}
	if !cx.TryAutoContinue(mode) {
		t.Fatal("grant 1 should succeed")
	}
	if cx.ChargeCurrent(2) != nil {
		t.Fatal("two more steps after grant 1")
	}
	if cx.ChargeCurrent(1) == nil {
		t.Fatal("exhausts again")
	}
	if !cx.TryAutoContinue(mode) {
		t.Fatal("grant 2 should succeed")
	}
	if cx.TryAutoContinue(mode) {
		t.Fatal("grants spent (MaxGrants = 2) → must fall through to the autonomous terminal")
	}
	if count.Load() != 2 {
		t.Fatalf("OnGrant fired %d times, want 2", count.Load())
	}

	// Other modes never auto-continue.
	if cx.TryAutoContinue(AutonomousEscalation()) {
		t.Fatal("Autonomous must not auto-continue")
	}
	if cx.TryAutoContinue(SurfaceToHumanEscalation()) {
		t.Fatal("SurfaceToHuman must not auto-continue")
	}
	// MaxGrants = 0 behaves like Autonomous (no grant).
	if cx.TryAutoContinue(AutoContinueEscalation(0, 5, nil)) {
		t.Fatal("MaxGrants = 0 must not grant")
	}
}

// TryAutoContinue with no current scope returns false (defensive).
func TestTryAutoContinueNoScopeIsFalse(t *testing.T) {
	registry := NewExecutionRegistryBuilder().Build()
	cx := NewExecutionContext(&registry)
	if cx.TryAutoContinue(AutoContinueEscalation(3, 2, nil)) {
		t.Fatal("no current scope → TryAutoContinue must return false")
	}
}

// ── Full-run: AutoContinue keeps a bare leaf working in-process to completion ─

// SC-5 acceptance: EscalationMode AutoContinue keeps a budget-exhausted bare
// ReAct leaf working IN-PROCESS — no consumer drive loop — by auto-granting
// StepsPerGrant more steps at the Escalate fall-through, firing OnGrant per
// grant, until the worker completes.
func TestAutoContinueGrantsInProcessThenCompletes(t *testing.T) {
	a := NewMockAgent("leaf")
	// First window: 2 tool turns exhaust the PerLoop{2} cap → Escalate.
	a.Push(NewToolCallRequested([]ToolCall{{ID: "c0", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	// After one auto-grant refreshes the cap, the worker completes.
	a.Push(NewFinalResponse("done after auto-continue", turnUsage()))

	var grants atomic.Uint32
	cfg := standardCfg(a)
	cfg.ToolRegistry = toolReg(3)
	cfg.EscalationMode = AutoContinueEscalation(3, 2, func(info AutoGrantInfo) {
		if info.StepsGranted != 2 {
			t.Errorf("StepsGranted = %d, want 2", info.StepsGranted)
		}
		grants.Add(1)
	})
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success via AutoContinue, got %+v", r)
	}
	if r.Output != "done after auto-continue" {
		t.Fatalf("output = %q, want %q", r.Output, "done after auto-continue")
	}
	if grants.Load() != 1 {
		t.Fatalf("OnGrant fired %d times, want exactly 1 (one auto-grant needed to finish)", grants.Load())
	}
}

// SC-5: AutoContinue is CAPPED — once MaxGrants grants are spent it falls
// through to the autonomous terminal (Failure), firing OnGrant exactly
// MaxGrants times. The agent never emits a final response, so every window
// exhausts.
func TestAutoContinueCapsAtMaxGrantsThenFails(t *testing.T) {
	a := NewMockAgent("leaf")
	// Plenty of tool turns (far more than any window can consume) so the run only
	// ends when the grant cap is reached — never on an empty queue.
	for i := 0; i < 40; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	}
	var grants atomic.Uint32
	cfg := standardCfg(a)
	cfg.ToolRegistry = toolReg(80)
	cfg.EscalationMode = AutoContinueEscalation(2, 2, func(AutoGrantInfo) {
		grants.Add(1)
	})
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunFailure {
		t.Fatalf("expected Failure once grants are spent, got %+v", r)
	}
	if grants.Load() != 2 {
		t.Fatalf("OnGrant fired %d times, want exactly MaxGrants (2) before falling through to Failure", grants.Load())
	}
}
