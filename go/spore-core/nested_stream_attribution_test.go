package sporecore

import (
	"context"
	"testing"
)

type streamingPlanAgent struct {
	id      AgentID
	results []TurnResult
}

func (a *streamingPlanAgent) ID() AgentID { return a.id }
func (a *streamingPlanAgent) Turn(context.Context, Context) TurnResult {
	if len(a.results) == 0 {
		return NewTurnError(NewEmptyResponseError(), nil)
	}
	r := a.results[0]
	a.results = a.results[1:]
	return r
}
func (a *streamingPlanAgent) TurnStreaming(ctx context.Context, c Context, sink AgentStreamSink) TurnResult {
	sink(StreamEvent{Type: StreamMessageStart})
	sink(StreamEvent{Type: StreamContentBlockDelta, Index: 0, Delta: "live"})
	sink(StreamEvent{Type: StreamContentBlockStop, Index: 0})
	sink(StreamEvent{Type: StreamMessageStop, Usage: &TokenUsage{InputTokens: 1, OutputTokens: 1}})
	return a.Turn(ctx, c)
}

func TestPlanExecuteChildStreamEventsInheritNodeAttribution(t *testing.T) {
	a := &streamingPlanAgent{id: "worker", results: []TurnResult{
		planFinal(`{"tasks":["one"],"rationale":"r"}`),
		planFinal("done"),
	}}
	h := NewStandardHarness(standardCfg(a))
	var events []HarnessStreamEvent
	opts := NewHarnessRunOptions(planTask("build a CLI"))
	opts.OnStream = func(ev HarnessStreamEvent) { events = append(events, ev) }
	r := h.Run(context.Background(), opts)
	if r.Kind != RunSuccess {
		t.Fatalf("result=%+v", r)
	}
	for _, ev := range events {
		if ev.Kind == HarnessStreamTextDelta {
			if ev.Node == nil {
				t.Fatalf("text delta missing node: %+v", ev)
			}
			if len(ev.Node.Path) < 2 || ev.Node.Path[0] != "plan_execute" || ev.Node.Path[1] != "react" {
				t.Fatalf("text delta node path=%v, want plan_execute/react", ev.Node.Path)
			}
			return
		}
	}
	t.Fatalf("no child text delta observed; events=%+v", events)
}
