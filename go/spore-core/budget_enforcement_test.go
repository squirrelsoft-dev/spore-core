// Per-node budget enforcement + failure isolation tests (issue #125).
//
// Covers every rule from the slice plan:
//  1. A node capped at N stops at N WITHOUT killing siblings.
//  2. In-process Continue resets the counter, honors MaxContinues, then falls
//     through; session/messages unchanged across resets.
//  3. Fail yields PartialOutput = nil.
//  4. A child BudgetExhausted reaches the parent as a StrategyOutcome, never
//     auto-propagated (parent's own scope unaffected; parent can then Complete).
//  5. PartialOutput concrete per node (4 shapes).
//  6. The ReAct leaf does not carry its own behavior; it propagates to parent.
//  7. Never auto-cascade a child exhaustion into a parent exhaustion.
//
// No new fixtures (fork #3): every type here is runtime-only.

package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

func continueThenFail(maxContinues uint32) BudgetExhaustedBehavior {
	return BudgetExhaustedBehavior{
		Kind:         BehaviorContinue,
		MaxContinues: maxContinues,
		OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
	}
}

// ── ResolveExhausted: Fail → Fail; Escalate → Escalate ──────────────────────

func TestResolveFailAndEscalateTerminal(t *testing.T) {
	fail := NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 1}, BudgetExhaustedBehavior{Kind: BehaviorFail}, "p")
	if got := fail.ResolveExhausted(); got != ExhaustedResolutionFail {
		t.Fatalf("Fail resolves to %q, want fail", got)
	}
	esc := NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 1}, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "p")
	if got := esc.ResolveExhausted(); got != ExhaustedResolutionEscalate {
		t.Fatalf("Escalate resolves to %q, want escalate", got)
	}
}

// ── Rule 2: Continue resets counter, honors MaxContinues, falls through ──────

func TestContinueResetsCounterThenFallsThroughToFail(t *testing.T) {
	cx := NewBudgetContext(BudgetPolicy{Kind: BudgetTotalSteps, Value: 3}, continueThenFail(2), "phase")

	// 1st exhaustion → Continue (counter resets to 0, ContinuesUsed = 1).
	if err := cx.Charge(3); err != nil {
		t.Fatalf("charge(3) #1: %v", err)
	}
	if cx.Charge(1) == nil {
		t.Fatal("charge(1) #1 should exhaust")
	}
	if got := cx.ResolveExhausted(); got != ExhaustedResolutionContinue {
		t.Fatalf("resolve #1 = %q, want continue", got)
	}
	if cx.StepsTaken != 0 {
		t.Fatalf("counter not reset on continue #1: %d", cx.StepsTaken)
	}
	if cx.ContinuesUsed != 1 {
		t.Fatalf("ContinuesUsed = %d, want 1", cx.ContinuesUsed)
	}

	// 2nd exhaustion → Continue (counter resets again, ContinuesUsed = 2).
	_ = cx.Charge(3)
	if cx.Charge(1) == nil {
		t.Fatal("charge(1) #2 should exhaust")
	}
	if got := cx.ResolveExhausted(); got != ExhaustedResolutionContinue {
		t.Fatalf("resolve #2 = %q, want continue", got)
	}
	if cx.StepsTaken != 0 || cx.ContinuesUsed != 2 {
		t.Fatalf("after continue #2: steps=%d continues=%d", cx.StepsTaken, cx.ContinuesUsed)
	}

	// 3rd exhaustion → continues spent → fall through to Fail.
	_ = cx.Charge(3)
	if cx.Charge(1) == nil {
		t.Fatal("charge(1) #3 should exhaust")
	}
	if got := cx.ResolveExhausted(); got != ExhaustedResolutionFail {
		t.Fatalf("resolve #3 = %q, want fail (continues spent)", got)
	}
	// ContinuesUsed does NOT advance past MaxContinues on the fall-through.
	if cx.ContinuesUsed != 2 {
		t.Fatalf("ContinuesUsed = %d, want 2 (no advance on fall-through)", cx.ContinuesUsed)
	}
}

// ── Continue chain shares ONE ContinuesUsed counter, then escalates ─────────

func TestContinueChainSharesCounterThenEscalates(t *testing.T) {
	behavior := BudgetExhaustedBehavior{
		Kind:         BehaviorContinue,
		MaxContinues: 2,
		OnExhausted: &BudgetExhaustedBehavior{
			Kind:         BehaviorContinue,
			MaxContinues: 2,
			OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		},
	}
	cx := NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 1}, behavior, "chain")

	if got := cx.ResolveExhausted(); got != ExhaustedResolutionContinue || cx.ContinuesUsed != 1 {
		t.Fatalf("continue #1: got %q continues=%d", got, cx.ContinuesUsed)
	}
	if got := cx.ResolveExhausted(); got != ExhaustedResolutionContinue || cx.ContinuesUsed != 2 {
		t.Fatalf("continue #2: got %q continues=%d", got, cx.ContinuesUsed)
	}
	// Outer spent → fall through to nested Continue{2}; the SHARED counter is
	// already 2 so the nested layer grants nothing → Escalate.
	if got := cx.ResolveExhausted(); got != ExhaustedResolutionEscalate {
		t.Fatalf("after outer spent: got %q, want escalate", got)
	}
}

// ── ConsumeContinue resets only the in-memory step counter ───────────────────

func TestConsumeContinueResetsStepsAndBumpsCounter(t *testing.T) {
	cx := NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 4}, continueThenFail(5), "c")
	if err := cx.Charge(4); err != nil {
		t.Fatalf("charge(4): %v", err)
	}
	if cx.StepsTaken != 4 {
		t.Fatalf("StepsTaken = %d, want 4", cx.StepsTaken)
	}
	cx.ConsumeContinue()
	if cx.StepsTaken != 0 {
		t.Fatalf("step counter not rewound: %d", cx.StepsTaken)
	}
	if cx.ContinuesUsed != 1 {
		t.Fatalf("ContinuesUsed = %d, want 1", cx.ContinuesUsed)
	}
	// The scope's allowance is intact — a fresh round can charge again.
	if err := cx.Charge(4); err != nil {
		t.Fatalf("post-continue charge(4): %v", err)
	}
}

// ── Rule 3: Fail → PartialOutput = nil; Escalate → non-nil partial ──────────

func TestPromoteFailDropsPartialEscalateKeepsIt(t *testing.T) {
	err := &BudgetExhausted{
		Policy:     BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		Behavior:   BudgetExhaustedBehavior{Kind: BehaviorFail},
		StepsTaken: 2,
		Phase:      "react",
	}
	// The Fail boundary supplies nil.
	failed := promoteBudgetExhausted(err, nil)
	if failed.Kind != StrategyOutcomeBudgetExhausted {
		t.Fatalf("kind = %q", failed.Kind)
	}
	if failed.Exhausted.PartialOutput != nil {
		t.Fatalf("Fail must discard the partial, got %q", *failed.Exhausted.PartialOutput)
	}
	if failed.Exhausted.StepsTaken != 2 {
		t.Fatalf("StepsTaken = %d, want 2", failed.Exhausted.StepsTaken)
	}
	// The Escalate boundary supplies a non-nil partial.
	p := reactPartialJSON("the answer so far")
	escalated := promoteBudgetExhausted(err, &p)
	if escalated.Exhausted.PartialOutput == nil {
		t.Fatal("Escalate must keep the partial")
	}
	if !json.Valid([]byte(*escalated.Exhausted.PartialOutput)) {
		t.Fatalf("partial is not valid JSON: %q", *escalated.Exhausted.PartialOutput)
	}
}

// ── Rule 5: each node's PartialOutput has its documented shape ──────────────

func TestReactPartialShape(t *testing.T) {
	var v map[string]any
	if err := json.Unmarshal([]byte(reactPartialJSON("hello world")), &v); err != nil {
		t.Fatal(err)
	}
	if v["node"] != "react" || v["last_final_response"] != "hello world" {
		t.Fatalf("react partial = %#v", v)
	}
}

func TestPlanExecutePartialShape(t *testing.T) {
	tl := DefaultTaskList()
	a, _ := tl.Add("task a", nil)
	_, _ = tl.Add("task b", nil)
	done := TaskStatusCompleted
	_ = tl.Update(a, &done, nil)

	var v map[string]any
	if err := json.Unmarshal([]byte(planExecutePartialJSON(tl)), &v); err != nil {
		t.Fatal(err)
	}
	if v["node"] != "plan_execute" {
		t.Fatalf("node = %v", v["node"])
	}
	if v["tasks"].(float64) != 2 {
		t.Fatalf("tasks = %v, want 2", v["tasks"])
	}
	ledger := v["ledger"].([]any)
	if len(ledger) != 2 {
		t.Fatalf("ledger len = %d, want 2", len(ledger))
	}
	row0 := ledger[0].(map[string]any)
	if row0["description"] != "task a" || row0["status"] != string(TaskStatusCompleted) {
		t.Fatalf("ledger[0] = %#v", row0)
	}
	row1 := ledger[1].(map[string]any)
	if row1["status"] != string(TaskStatusPending) {
		t.Fatalf("ledger[1] status = %v, want pending", row1["status"])
	}
}

func TestSelfVerifyingPartialShape(t *testing.T) {
	var v map[string]any
	if err := json.Unmarshal([]byte(selfVerifyingPartialJSON("worker output", "verdict: not yet")), &v); err != nil {
		t.Fatal(err)
	}
	if v["node"] != "self_verifying" || v["last_worker_result"] != "worker output" || v["last_verdict"] != "verdict: not yet" {
		t.Fatalf("self_verifying partial = %#v", v)
	}
}

func TestHillClimbingPartialShape(t *testing.T) {
	var v map[string]any
	if err := json.Unmarshal([]byte(hillClimbingPartialJSON(0.875)), &v); err != nil {
		t.Fatal(err)
	}
	if v["node"] != "hill_climbing" || v["best_candidate"].(float64) != 0.875 || v["score"].(float64) != 0.875 {
		t.Fatalf("hill_climbing partial = %#v", v)
	}
}

// ── Rule 1: a node capped at N stops at N without killing siblings ──────────

func TestSiblingIsolationFreshContextPerNode(t *testing.T) {
	var budgets BudgetStack
	// Sibling A: capped at 2, exhausts.
	budgets.Push(NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 2}, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "child-a"))
	if err := budgets.Current().Charge(2); err != nil {
		t.Fatalf("A charge(2): %v", err)
	}
	if budgets.Current().Charge(1) == nil {
		t.Fatal("A should exhaust at its own cap N=2")
	}
	a, _ := budgets.Pop()
	if a.StepsTaken != 2 {
		t.Fatalf("A StepsTaken = %d, want 2", a.StepsTaken)
	}

	// Sibling B gets a FRESH BudgetContext — A's exhaustion did not bleed in.
	budgets.Push(NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 4}, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "child-b"))
	b := budgets.Current()
	if b.StepsTaken != 0 {
		t.Fatalf("sibling B not fresh: StepsTaken = %d (rule 1)", b.StepsTaken)
	}
	if err := b.Charge(3); err != nil {
		t.Fatalf("B charge(3) under its own allowance: %v", err)
	}
}

// ── Rule 4 & 7: a child exhaustion does NOT auto-cascade to the parent ──────

func TestChildExhaustionDoesNotChargeParentScope(t *testing.T) {
	var budgets BudgetStack
	// Parent scope (TotalSteps{5}) that is ALREADY nearly exhausted: it has spent
	// 4 of its 5 steps, leaving EXACTLY ONE remaining. This is the adversarial
	// case for rule 4/7 — if a child's exhaustion auto-cascaded even a single step
	// onto the parent, the parent would be pushed over its own cap and could no
	// longer Complete.
	budgets.Push(NewBudgetContext(BudgetPolicy{Kind: BudgetTotalSteps, Value: 5}, BudgetExhaustedBehavior{Kind: BehaviorFail}, "parent"))
	if err := budgets.Current().Charge(4); err != nil {
		t.Fatalf("parent pre-charge(4): %v", err)
	}
	if rem, capped := budgets.Current().Remaining(); !capped || rem != 1 {
		t.Fatalf("parent should start with exactly 1 step left, got rem=%d capped=%v", rem, capped)
	}

	// Child descends with its OWN scope (capped at 1) and exhausts.
	budgets.Push(NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 1}, BudgetExhaustedBehavior{Kind: BehaviorEscalate}, "child"))
	if err := budgets.Current().Charge(1); err != nil {
		t.Fatalf("child charge(1): %v", err)
	}
	if budgets.Current().Charge(1) == nil {
		t.Fatal("child should exhaust at N=1")
	}
	// The child surfaces a BudgetExhausted value — modelled by popping its scope.
	// Crucially the parent scope is UNCHARGED by this.
	budgets.Pop()

	parent := budgets.Current()
	if parent.StepsTaken != 4 {
		t.Fatalf("rule 4/7: child exhaustion did NOT auto-charge the parent — its StepsTaken must be unchanged at 4 (not bumped to 5), got %d", parent.StepsTaken)
	}
	if rem, capped := parent.Remaining(); !capped || rem != 1 {
		t.Fatalf("the parent STILL has its 1 remaining step after the child exhausted, got rem=%d capped=%v", rem, capped)
	}
	// The parent can spend its last step and Complete — proving the child's
	// exhaustion did NOT push it over its own cap.
	if err := parent.Charge(1); err != nil {
		t.Fatalf("parent should spend its final step and Complete within its own budget: %v", err)
	}
	// And only now is the parent itself exhausted (at its own 5, by its own work
	// — never by the child's).
	if parent.Charge(1) == nil {
		t.Fatal("parent should now be exhausted at its own cap 5, by its own work")
	}
}

// ── ExecutionContext helpers: push / charge / resolve / pop round-trip ──────

func TestExecutionContextChargeAndResolveRoundTrip(t *testing.T) {
	registry := NewExecutionRegistry()
	cx := NewExecutionContext(&registry)
	if cx.Budgets.Depth() != 0 {
		t.Fatalf("initial depth = %d", cx.Budgets.Depth())
	}

	cx.PushBudget(BudgetPolicy{Kind: BudgetPerLoop, Value: 2}, continueThenFail(1), "node")
	if cx.Budgets.Depth() != 1 {
		t.Fatalf("depth after push = %d", cx.Budgets.Depth())
	}
	if cx.ChargeCurrent(2) != nil {
		t.Fatal("charge(2) within allowance should be nil")
	}
	if cx.ChargeCurrent(1) == nil {
		t.Fatal("charge(1) should exhaust at the cap")
	}
	if got := cx.ResolveCurrent(); got != ExhaustedResolutionContinue {
		t.Fatalf("resolve #1 = %q, want continue", got)
	}
	// After the continue, the counter reset → charging is possible again.
	if cx.ChargeCurrent(2) != nil {
		t.Fatal("post-continue charge(2) should be nil")
	}
	if cx.ChargeCurrent(1) == nil {
		t.Fatal("charge(1) #2 should exhaust")
	}
	if got := cx.ResolveCurrent(); got != ExhaustedResolutionFail {
		t.Fatalf("resolve #2 = %q, want fail (continues spent)", got)
	}
	if _, ok := cx.PopBudget(); !ok {
		t.Fatal("pop should return the scope")
	}
	if cx.Budgets.Depth() != 0 {
		t.Fatalf("depth after pop = %d", cx.Budgets.Depth())
	}
}

// ── ChargeCurrent with no scope is a no-op nil (scaffold contexts) ──────────

func TestChargeWithNoScopeNeverExhausts(t *testing.T) {
	registry := NewExecutionRegistry()
	cx := NewExecutionContext(&registry)
	if cx.ChargeCurrent(^uint32(0)) != nil {
		t.Fatal("charge with no scope should be a no-op nil")
	}
	if got := cx.ResolveCurrent(); got != ExhaustedResolutionFail {
		t.Fatalf("resolve with no scope = %q, want fail", got)
	}
}

// ── Rule 6: ReAct leaf cap-binding propagates a partial ─────────────────────
//
// When the ReAct LEAF's OWN policy is the binding cap (no smaller global
// backstop), the leaf PROPAGATES a typed BudgetExhausted carrying its last
// FinalResponse as the partial — it never self-resolves Continue/Fail at the
// leaf. driveStrategy surfaces it as a BudgetExceeded terminal whose surfaced
// turn count is the exhausted node's StepsTaken.
func TestReactLeafCapBindingPropagatesPartial(t *testing.T) {
	a := NewMockAgent("leaf")
	// The leaf cap is 2; push 3 tool-call turns so the window runs 2 turns and
	// hits the leaf's own cap (no global max_turns set).
	for i := 0; i < 3; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	}
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	for i := 0; i < 3; i++ {
		reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	}
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	// Leaf PerLoop{2}, NO global cap → the leaf policy is the binding cap.
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunFailure {
		t.Fatalf("expected Failure from leaf cap, got %+v", r)
	}
	if r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("expected BudgetExceeded(Turns), got %+v", r.Reason)
	}
	// The surfaced turn count is the EXHAUSTED NODE's own StepsTaken (#125
	// BudgetExhausted path), which equals the leaf cap N=2 here.
	if r.Turns != 2 {
		t.Fatalf("leaf stopped at its own cap N=2, surfaced turns = %d", r.Turns)
	}

	// The #125 discriminator: the partial flowed through
	// StrategyOutcome.BudgetExhausted{PartialOutput} → driveStrategy's
	// BudgetExhausted arm, which materializes the partial as a single assistant
	// Text message. On PRE-#125 code the leaf's recordTerminal(window_failure)
	// surfaced the window's RAW session (the tool-call / observation messages),
	// NOT this node-concrete partial JSON — so these assertions FAIL on the old
	// path and PROVE the new BudgetExhausted machinery is exercised.
	msgs := r.SessionState.Messages
	if len(msgs) != 1 {
		t.Fatalf("BudgetExhausted arm materializes exactly the partial as one assistant message (not the window's raw transcript), got %d messages", len(msgs))
	}
	m := msgs[0]
	if m.Role != RoleAssistant {
		t.Fatalf("partial message role = %q, want assistant", m.Role)
	}
	if m.Content.Type != ContentTypeText {
		t.Fatalf("partial content type = %q, want text", m.Content.Type)
	}
	// The documented ReAct partial shape (fork #2): the last FinalResponse text
	// as JSON. This window produced no FinalResponse, so the documented shape
	// carries an empty last_final_response.
	if m.Content.Text != reactPartialJSON("") {
		t.Fatalf("materialized partial = %q, want reactPartialJSON(\"\") = %q", m.Content.Text, reactPartialJSON(""))
	}
	// Sanity: the materialized text is genuinely the partial helper's output,
	// i.e. valid JSON with the node tag — not free-form prose.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(m.Content.Text), &parsed); err != nil {
		t.Fatalf("partial is not valid JSON: %v", err)
	}
	if parsed["node"] != "react" {
		t.Fatalf("partial node = %v, want \"react\"", parsed["node"])
	}
	if parsed["last_final_response"] != "" {
		t.Fatalf("partial last_final_response = %v, want \"\"", parsed["last_final_response"])
	}
}
