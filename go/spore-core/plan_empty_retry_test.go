package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

type recordingPlanAgent struct {
	id       AgentID
	results  []TurnResult
	contexts []Context
}

func (a *recordingPlanAgent) ID() AgentID { return a.id }
func (a *recordingPlanAgent) Turn(_ context.Context, c Context) TurnResult {
	a.contexts = append(a.contexts, c)
	if len(a.results) == 0 {
		return NewTurnError(NewEmptyResponseError(), nil)
	}
	r := a.results[0]
	a.results = a.results[1:]
	return r
}

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

func TestPlanEmptyRetryForcesSchemaSuppressesToolsAndAccumulatesUsage(t *testing.T) {
	a := &recordingPlanAgent{id: "planner", results: []TurnResult{
		NewFinalResponse(`{"tasks":[],"rationale":"nothing"}`, TokenUsage{InputTokens: 11, OutputTokens: 3}),
		NewFinalResponse(`{"tasks":["one"],"rationale":"retry"}`, TokenUsage{InputTokens: 13, OutputTokens: 5}),
		NewFinalResponse("did one", TokenUsage{InputTokens: 17, OutputTokens: 7}),
	}}
	reg := NewStandardToolRegistry()
	registerNamedTool(t, reg, "planning_tool", "must be hidden on retry")
	cfg := standardCfg(a).WithPlanEmptyMaxRetries(1)
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess || r.Output != "did one" {
		t.Fatalf("result=%+v, want success after retry plan", r)
	}
	if len(a.contexts) != 3 {
		t.Fatalf("contexts=%d, want 3", len(a.contexts))
	}
	if len(a.contexts[0].Tools) == 0 {
		t.Fatal("first plan attempt should advertise configured tools")
	}
	if len(a.contexts[1].Tools) != 0 {
		t.Fatalf("retry plan tools=%v, want suppressed", a.contexts[1].Tools)
	}
	if len(a.contexts[1].Params.OutputSchema) == 0 || !json.Valid(a.contexts[1].Params.OutputSchema) {
		t.Fatalf("retry output schema=%q, want forced valid schema", string(a.contexts[1].Params.OutputSchema))
	}
	if r.Usage.InputTokens != 11+13+17 || r.Usage.OutputTokens != 3+5+7 {
		t.Fatalf("usage=%+v, want discarded plan + retry plan + execute usage", r.Usage)
	}
}
