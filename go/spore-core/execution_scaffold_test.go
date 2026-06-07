package sporecore

import (
	"context"
	"testing"
)

// Composable Execution runtime scaffold tests (issue #123): BudgetContext.Charge
// / Remaining / ContinuesRemaining, StrategyOutcome variant discrimination,
// SpanStack / BudgetStack push/pop, recursive *ExecutionContext threading, and
// distinct sibling BudgetContexts. Runtime-only types — no fixtures.

func totalSteps(value uint32) BudgetPolicy {
	return BudgetPolicy{Kind: BudgetTotalSteps, Value: value}
}

func unlimited() BudgetPolicy {
	return BudgetPolicy{Kind: BudgetUnlimited}
}

func failBehavior() BudgetExhaustedBehavior {
	return BudgetExhaustedBehavior{Kind: BehaviorFail}
}

func escalateBehavior() BudgetExhaustedBehavior {
	return BudgetExhaustedBehavior{Kind: BehaviorEscalate}
}

func continueBehavior(maxContinues uint32) BudgetExhaustedBehavior {
	return BudgetExhaustedBehavior{
		Kind:         BehaviorContinue,
		MaxContinues: maxContinues,
		OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
	}
}

// ---- BudgetContext.Charge --------------------------------------------------

func TestChargeWithinAllowanceIncrementsStepsTaken(t *testing.T) {
	cx := NewBudgetContext(totalSteps(5), failBehavior(), "p")
	if exhausted := cx.Charge(2); exhausted != nil {
		t.Fatalf("charge(2) within allowance signalled exhaustion: %+v", exhausted)
	}
	if cx.StepsTaken != 2 {
		t.Fatalf("StepsTaken = %d, want 2", cx.StepsTaken)
	}
	if exhausted := cx.Charge(3); exhausted != nil {
		t.Fatalf("charge(3) to exactly the cap signalled exhaustion: %+v", exhausted)
	}
	if cx.StepsTaken != 5 {
		t.Fatalf("StepsTaken = %d, want 5", cx.StepsTaken)
	}
}

func TestChargeOverflowSignalsExhaustionWithoutMutation(t *testing.T) {
	cx := NewBudgetContext(totalSteps(3), continueBehavior(2), "phaseX")
	if exhausted := cx.Charge(2); exhausted != nil {
		t.Fatalf("charge(2) signalled exhaustion: %+v", exhausted)
	}
	// StepsTaken == 2; charging 2 more would exceed the cap of 3.
	exhausted := cx.Charge(2)
	if exhausted == nil {
		t.Fatal("charge overflow did not signal exhaustion")
	}
	// No mutation on overflow.
	if cx.StepsTaken != 2 {
		t.Fatalf("StepsTaken mutated on overflow: %d, want 2", cx.StepsTaken)
	}
	// Signal captures current state.
	if exhausted.StepsTaken != 2 {
		t.Fatalf("signal StepsTaken = %d, want 2", exhausted.StepsTaken)
	}
	if exhausted.Phase != "phaseX" {
		t.Fatalf("signal Phase = %q, want phaseX", exhausted.Phase)
	}
	if exhausted.Policy.Kind != BudgetTotalSteps || exhausted.Policy.Value != 3 {
		t.Fatalf("signal Policy = %+v, want total_steps/3", exhausted.Policy)
	}
	if exhausted.Behavior.Kind != BehaviorContinue {
		t.Fatalf("signal Behavior.Kind = %q, want continue", exhausted.Behavior.Kind)
	}
	if exhausted.ContinuesUsed != 0 {
		t.Fatalf("signal ContinuesUsed = %d, want 0", exhausted.ContinuesUsed)
	}
}

func TestChargeUnlimitedNeverExhausts(t *testing.T) {
	cx := NewBudgetContext(unlimited(), failBehavior(), "u")
	for i := 0; i < 1000; i++ {
		if exhausted := cx.Charge(1_000_000); exhausted != nil {
			t.Fatalf("unlimited charge signalled exhaustion at iter %d", i)
		}
	}
}

// ---- BudgetContext.Remaining -----------------------------------------------

func TestRemainingCappedAndUnlimited(t *testing.T) {
	capped := NewBudgetContext(totalSteps(5), failBehavior(), "c")
	if rem, ok := capped.Remaining(); !ok || rem != 5 {
		t.Fatalf("fresh capped Remaining = (%d,%v), want (5,true)", rem, ok)
	}
	_ = capped.Charge(2)
	if rem, ok := capped.Remaining(); !ok || rem != 3 {
		t.Fatalf("after charge(2) Remaining = (%d,%v), want (3,true)", rem, ok)
	}
	// Saturates at 0, never negative: charge to the cap then probe.
	_ = capped.Charge(3)
	if rem, ok := capped.Remaining(); !ok || rem != 0 {
		t.Fatalf("at cap Remaining = (%d,%v), want (0,true)", rem, ok)
	}

	unl := NewBudgetContext(unlimited(), failBehavior(), "u")
	if rem, ok := unl.Remaining(); ok {
		t.Fatalf("unlimited Remaining = (%d,%v), want (_, false)", rem, ok)
	}
}

// ---- BudgetContext.ContinuesRemaining --------------------------------------

func TestContinuesRemainingForContinueEscalateFail(t *testing.T) {
	cont := NewBudgetContext(unlimited(), continueBehavior(3), "c")
	if got := cont.ContinuesRemaining(); got != 3 {
		t.Fatalf("fresh Continue ContinuesRemaining = %d, want 3", got)
	}
	cont.ContinuesUsed = 1
	if got := cont.ContinuesRemaining(); got != 2 {
		t.Fatalf("after 1 used ContinuesRemaining = %d, want 2", got)
	}
	// Saturates at 0, never negative.
	cont.ContinuesUsed = 5
	if got := cont.ContinuesRemaining(); got != 0 {
		t.Fatalf("over-used ContinuesRemaining = %d, want 0 (saturating)", got)
	}

	esc := NewBudgetContext(unlimited(), escalateBehavior(), "e")
	if got := esc.ContinuesRemaining(); got != 0 {
		t.Fatalf("Escalate ContinuesRemaining = %d, want 0", got)
	}
	fail := NewBudgetContext(unlimited(), failBehavior(), "f")
	if got := fail.ContinuesRemaining(); got != 0 {
		t.Fatalf("Fail ContinuesRemaining = %d, want 0", got)
	}
}

// ---- StrategyOutcome variant discrimination --------------------------------

func TestStrategyOutcomeVariantDiscrimination(t *testing.T) {
	partial := "half"
	exhausted := StrategyBudgetExhausted(BudgetExhausted{
		Policy:     totalSteps(2),
		Behavior:   failBehavior(),
		StepsTaken: 2,
		Phase:      "x",
	}, &partial)
	failed := StrategyFailed(&StrategyNotFoundError{Key: "k"})
	complete := StrategyComplete("done")

	// A BudgetExhausted is NOT a Failed — callers can distinguish.
	if exhausted.Kind != StrategyOutcomeBudgetExhausted {
		t.Fatalf("exhausted.Kind = %q, want budget_exhausted", exhausted.Kind)
	}
	if exhausted.Kind == StrategyOutcomeFailed {
		t.Fatal("BudgetExhausted collapsed into Failed")
	}
	if exhausted.Exhausted == nil || exhausted.Exhausted.PartialOutput == nil || *exhausted.Exhausted.PartialOutput != "half" {
		t.Fatalf("exhausted payload/partial output wrong: %+v", exhausted.Exhausted)
	}
	if exhausted.Exhausted.StepsTaken != 2 {
		t.Fatalf("exhausted StepsTaken = %d, want 2", exhausted.Exhausted.StepsTaken)
	}

	if failed.Kind != StrategyOutcomeFailed {
		t.Fatalf("failed.Kind = %q, want failed", failed.Kind)
	}
	if failed.Failed == nil {
		t.Fatal("Failed outcome carries no error")
	}
	if complete.Kind != StrategyOutcomeComplete || complete.Complete != "done" {
		t.Fatalf("complete outcome wrong: %+v", complete)
	}
}

// ---- SpanStack push/pop depth ----------------------------------------------

func TestSpanStackPushPopDepth(t *testing.T) {
	var spans SpanStack
	if spans.Depth() != 0 {
		t.Fatalf("fresh SpanStack depth = %d, want 0", spans.Depth())
	}
	spans.Push("a")
	spans.Push("b")
	if spans.Depth() != 2 {
		t.Fatalf("depth after 2 pushes = %d, want 2", spans.Depth())
	}
	id, ok := spans.Pop()
	if !ok || id != "b" {
		t.Fatalf("pop = (%q,%v), want (b,true)", id, ok)
	}
	if spans.Depth() != 1 {
		t.Fatalf("depth after pop = %d, want 1", spans.Depth())
	}
	_, _ = spans.Pop()
	if _, ok := spans.Pop(); ok {
		t.Fatal("pop on empty SpanStack returned ok=true")
	}
}

// ---- BudgetStack push/pop --------------------------------------------------

func TestBudgetStackPushPopCurrent(t *testing.T) {
	var budgets BudgetStack
	if budgets.Current() != nil {
		t.Fatal("empty BudgetStack Current() != nil")
	}
	if _, ok := budgets.Pop(); ok {
		t.Fatal("pop on empty BudgetStack returned ok=true")
	}
	budgets.Push(NewBudgetContext(totalSteps(2), failBehavior(), "outer"))
	budgets.Push(NewBudgetContext(totalSteps(4), failBehavior(), "inner"))
	if budgets.Depth() != 2 {
		t.Fatalf("depth = %d, want 2", budgets.Depth())
	}
	// Current() is mutable and points at the innermost scope.
	cur := budgets.Current()
	if cur.Phase != "inner" {
		t.Fatalf("Current().Phase = %q, want inner", cur.Phase)
	}
	_ = cur.Charge(1)
	if budgets.Current().StepsTaken != 1 {
		t.Fatal("Current() did not yield a mutable pointer into the stack")
	}
	popped, ok := budgets.Pop()
	if !ok || popped.Phase != "inner" {
		t.Fatalf("pop = (%q,%v), want (inner,true)", popped.Phase, ok)
	}
	if budgets.Current().Phase != "outer" {
		t.Fatalf("after pop Current().Phase = %q, want outer", budgets.Current().Phase)
	}
}

// ---- recursive stub strategy threads *ExecutionContext + BudgetStack --------

// recursiveStub recurses depth times, pushing a fresh per-node BudgetContext on
// descent and popping it on ascent, so the stack returns to baseline. Each node
// (incl. siblings at the same level) gets its OWN BudgetContext.
type recursiveStub struct {
	depth    int
	maxDepth *int
}

func (s recursiveStub) Run(ctx context.Context, cx *ExecutionContext) StrategyOutcome {
	cx.Budgets.Push(NewBudgetContext(totalSteps(uint32(s.depth+1)), failBehavior(), "node"))
	if d := cx.Budgets.Depth(); d > *s.maxDepth {
		*s.maxDepth = d
	}
	if s.depth > 0 {
		// Two children per node — siblings share the parent context but each
		// pushes/pops its own BudgetContext.
		_ = recursiveStub{depth: s.depth - 1, maxDepth: s.maxDepth}.Run(ctx, cx)
		_ = recursiveStub{depth: s.depth - 1, maxDepth: s.maxDepth}.Run(ctx, cx)
	}
	cx.Budgets.Pop()
	return StrategyComplete("")
}

func TestRecursiveStubThreadsContextAndBudgetStack(t *testing.T) {
	registry := NewExecutionRegistryBuilder().Build()
	cx := NewExecutionContext(&registry)
	maxDepth := 0
	outcome := recursiveStub{depth: 3, maxDepth: &maxDepth}.Run(context.Background(), cx)

	if outcome.Kind != StrategyOutcomeComplete {
		t.Fatalf("recursive stub outcome = %q, want complete", outcome.Kind)
	}
	// Stack returned to baseline depth (every push had a matching pop).
	if cx.Budgets.Depth() != 0 {
		t.Fatalf("BudgetStack depth after run = %d, want 0", cx.Budgets.Depth())
	}
	// Recursion actually descended (depth 3 → at least 4 frames live at once).
	if maxDepth < 4 {
		t.Fatalf("max live BudgetStack depth = %d, want >= 4", maxDepth)
	}
	// The same registry pointer was threaded throughout.
	if cx.Registry != &registry {
		t.Fatal("ExecutionContext.Registry pointer changed during recursion")
	}
}

// ---- siblings get distinct BudgetContexts ----------------------------------

func TestSiblingsGetDistinctBudgetContexts(t *testing.T) {
	var budgets BudgetStack
	budgets.Push(NewBudgetContext(totalSteps(5), failBehavior(), "sibling-a"))
	// Charge the first sibling, then pop it.
	_ = budgets.Current().Charge(3)
	if budgets.Current().StepsTaken != 3 {
		t.Fatalf("sibling-a StepsTaken = %d, want 3", budgets.Current().StepsTaken)
	}
	budgets.Pop()

	// A second sibling at the same level starts fresh — no shared counters.
	budgets.Push(NewBudgetContext(totalSteps(5), failBehavior(), "sibling-b"))
	if budgets.Current().StepsTaken != 0 {
		t.Fatalf("sibling-b StepsTaken = %d, want 0 (siblings must not share)", budgets.Current().StepsTaken)
	}
	if budgets.Current().Phase != "sibling-b" {
		t.Fatalf("sibling-b Phase = %q, want sibling-b", budgets.Current().Phase)
	}
}
