package sporecore

import (
	"context"
	"sync"
	"testing"
)

// ── #93: WithModelParams reaches every tool-requesting turn ─────────────────
//
// The agent copies Context.Params verbatim into the ModelRequest (IntoRequest /
// IntoRequestStreaming), so asserting on a captured context's
// Params.StructuredToolCalls proves the configured params reached the request
// the model would have seen. paramsCapturingAgent records the full Params of
// every Context it is handed.

type paramsCapturingAgent struct {
	id      AgentID
	mu      sync.Mutex
	seen    []ModelParams
	scripts []TurnResult
	calls   int
}

func (a *paramsCapturingAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	a.seen = append(a.seen, c.Params)
	r := a.next()
	a.mu.Unlock()
	return r
}

// next returns the scripted turn for this call, defaulting to a final response
// once the script is drained. Caller holds a.mu.
func (a *paramsCapturingAgent) next() TurnResult {
	var r TurnResult
	if a.calls < len(a.scripts) {
		r = a.scripts[a.calls]
	} else {
		r = NewFinalResponse("done", turnUsage())
	}
	a.calls++
	return r
}

func (a *paramsCapturingAgent) ID() AgentID { return a.id }

func (a *paramsCapturingAgent) seenParams() []ModelParams {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]ModelParams(nil), a.seen...)
}

func newParamsAgent(scripts ...TurnResult) *paramsCapturingAgent {
	return &paramsCapturingAgent{id: AgentID("cap"), scripts: scripts}
}

// No WithModelParams ⇒ each turn's context carries the zero/default value
// (StructuredToolCalls == false).
func TestModelParamsDefaultIsZero(t *testing.T) {
	agent := newParamsAgent()
	h := NewStandardHarness(standardCfg(agent))
	if r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}
	seen := agent.seenParams()
	if len(seen) == 0 {
		t.Fatal("agent saw no turns")
	}
	if seen[0].StructuredToolCalls {
		t.Fatalf("expected default StructuredToolCalls=false, got %+v", seen[0])
	}
}

// (a) WithModelParams(StructuredToolCalls: true) ⇒ the ReAct turn context
// carries it.
func TestModelParamsReachReactTurn(t *testing.T) {
	agent := newParamsAgent()
	cfg := standardCfg(agent)
	cfg.ModelParams = ModelParams{StructuredToolCalls: true}
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}
	seen := agent.seenParams()
	if len(seen) == 0 {
		t.Fatal("agent saw no turns")
	}
	if !seen[0].StructuredToolCalls {
		t.Fatalf("ReAct turn context did not carry StructuredToolCalls: %+v", seen[0])
	}
}

// (b) The PlanExecute plan phase replaces params on its own seam — the first
// (plan) turn's context carries the flag.
func TestModelParamsReachPlanPhase(t *testing.T) {
	agent := newParamsAgent(planFinal(`{"tasks":["one"],"rationale":"r"}`))
	cfg := standardCfg(agent)
	cfg.ModelParams = ModelParams{StructuredToolCalls: true}
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI"))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}
	seen := agent.seenParams()
	if len(seen) == 0 {
		t.Fatal("agent saw no turns")
	}
	// First captured context is the plan turn.
	if !seen[0].StructuredToolCalls {
		t.Fatalf("plan-phase turn context did not carry StructuredToolCalls: %+v", seen[0])
	}
}

// (c) A full PlanExecute run threads params through the shared react seam used
// by the execute sub-loop — every captured context (plan + execute steps)
// carries the flag.
func TestModelParamsReachExecuteSubLoop(t *testing.T) {
	agent := newParamsAgent(
		planFinal(`{"tasks":["one","two"],"rationale":"r"}`),
		planFinal("did one"),
		planFinal("did two"),
	)
	cfg := standardCfg(agent)
	cfg.ModelParams = ModelParams{StructuredToolCalls: true}
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI"))); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}
	seen := agent.seenParams()
	// 1 plan turn + 2 execute turns.
	if len(seen) != 3 {
		t.Fatalf("expected 3 captured turns (plan + 2 execute), got %d", len(seen))
	}
	for i, p := range seen {
		if !p.StructuredToolCalls {
			t.Fatalf("captured context %d did not carry StructuredToolCalls: %+v", i, p)
		}
	}
}

// (d) The streaming path flows through runReActInner's same seam — the streamed
// turn's captured context carries the flag.
func TestModelParamsReachStreamingTurn(t *testing.T) {
	agent := newParamsAgent()
	cfg := standardCfg(agent)
	cfg.ModelParams = ModelParams{StructuredToolCalls: true}
	h := NewStandardHarness(cfg)
	opts := NewHarnessRunOptions(reactTask(5))
	opts.OnStream = func(HarnessStreamEvent) {}
	if r := h.Run(context.Background(), opts); r.Kind != RunSuccess {
		t.Fatalf("run: %+v", r)
	}
	seen := agent.seenParams()
	if len(seen) == 0 {
		t.Fatal("agent saw no turns")
	}
	if !seen[0].StructuredToolCalls {
		t.Fatalf("streaming turn context did not carry StructuredToolCalls: %+v", seen[0])
	}
}
