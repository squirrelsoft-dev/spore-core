package sporecore

import (
	"context"
	"testing"
)

func TestPlanEmptyDefaultZeroImmediateEmptyPlan(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":[],"rationale":"nothing"}`))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunFailure || r.Reason.Kind != HaltEmptyPlan {
		t.Fatalf("result=%+v, want EmptyPlan", r)
	}
	if got := a.CallCount.Load(); got != 1 {
		t.Fatalf("model call count=%d, want 1 (default no retry)", got)
	}
}

func TestPlanEmptyRetryThenNonEmptyPlanProceeds(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":[],"rationale":"nothing"}`))
	a.Push(planFinal(`{"tasks":["one"],"rationale":"retry"}`))
	a.Push(planFinal("did one"))
	cfg := standardCfg(a).WithPlanEmptyMaxRetries(1)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess || r.Output != "did one" {
		t.Fatalf("result=%+v, want success after retry plan", r)
	}
	if got := a.CallCount.Load(); got != 3 {
		t.Fatalf("model call count=%d, want 3 (empty plan, retry plan, execute)", got)
	}
}
