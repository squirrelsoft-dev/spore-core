package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

// recompose_131_worker_fill_test.go — two regression tests for the #131 harness
// fixes ported from the Rust reference:
//
//	Fix A — Ralph FILLS the worker leaf agent only when empty; an explicitly
//	        declared leaf agent is authoritative and is never shadowed by Ralph's
//	        per-window agent (fillEmptyWorkerAgent, formerly overrideWorkerAgent).
//	Fix B — the plan phase honors the plan sub-strategy's OWN declared budget; the
//	        old min(global, turns+1) clamp (which pinned the plan to a single turn
//	        on a fresh run) is gone.
//
// Each test is written to FAIL on the OLD behavior.

// ----------------------------------------------------------------------------
// Fix A: a NON-EMPTY Ralph agent must NOT shadow an explicitly-declared worker
// leaf agent.
//
// Tree: Ralph{agent: "ralph-agent"}[ PlanExecute[ ReAct{planner},
//        SelfVerifying[ ReAct{executor} ] ] ].
//
// The execute worker leaf declares agent "executor". Under the OLD
// overrideWorkerAgent (unconditional replace), Ralph's per-window agent would
// REWRITE that leaf's handle to "ralph-agent", so the EXECUTOR agent would never
// run and the RALPH-AGENT agent would run the worker turns. The new
// fillEmptyWorkerAgent leaves the declared "executor" handle untouched.
// ----------------------------------------------------------------------------

// keyedCountingAgent records how many turns it ran and writes a COMPLETE Ralph
// progress file on its build turns (the turns that do NOT carry the evaluator
// role chunk), so a Ralph window that dispatches to it completes in one window.
type keyedCountingAgent struct {
	id    AgentID
	root  string
	mu    sync.Mutex
	calls int
}

func (a *keyedCountingAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.calls++
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	// On a build turn (not the fresh-evaluator turn) declare the work COMPLETE so
	// the Ralph window terminates successfully in a single window.
	if !strings.Contains(b.String(), RoleEvaluatorChunk) {
		writeRalphProgress(a.root, ralphWindow{complete: true})
	}
	return NewFinalResponse("done", TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *keyedCountingAgent) ID() AgentID { return a.id }

func (a *keyedCountingAgent) callCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.calls
}

var _ Agent = (*keyedCountingAgent)(nil)

func TestRalphAgentDoesNotShadowDeclaredWorkerLeaf(t *testing.T) {
	dir := t.TempDir()

	planner := newPlanTurnCounter("planner", `{"tasks":["t0"],"rationale":"r"}`, "plan-noop")
	executor := &keyedCountingAgent{id: AgentID("executor"), root: dir}
	// ralphAgentLeaf is registered under "ralph-agent" — the per-window Ralph
	// agent. If Fix A regresses (override semantics), THIS agent would run the
	// execute worker turns instead of `executor`.
	ralphAgentLeaf := &keyedCountingAgent{id: AgentID("ralph-agent"), root: dir}
	def := &keyedCountingAgent{id: AgentID("default"), root: dir}

	// Plan child routes to "planner"; execute worker leaf explicitly routes to
	// "executor". Both structured slots carry an output schema (A.5).
	planChild := withOutputSchema(^uint32(0))
	planChild.ReActCfg.Agent = AgentRef("planner")

	execWorker := withOutputSchema(^uint32(0))
	execWorker.ReActCfg.Agent = AgentRef("executor")
	sv := SelfVerifyingStrategy(SelfVerifyingConfig{Inner: &execWorker, Evaluator: SchemaRef("eval")})

	pe := PlanExecuteStrategy(PlanExecuteConfig{Plan: &planChild, Execute: &sv})

	v := newSVVerifier(3, "pass")
	cfg := standardCfg(def)
	cfg.Sandbox = rootedSandbox{root: dir}
	cfg.MaxResets = 2
	cfg = cfg.
		WithRegistryAgent("planner", planner).
		WithRegistryAgent("executor", executor).
		WithRegistryAgent("ralph-agent", ralphAgentLeaf).
		WithRegistryVerifier("eval", v)
	h := NewStandardHarness(cfg)

	// NON-EMPTY Ralph agent — the crux: it must FILL only empty leaves.
	task := NewTask("build the thing", SessionID("ralph-fill-sess"),
		RalphStrategy(RalphConfig{Inner: &pe, Agent: AgentRef("ralph-agent")}))

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}

	// The declared "executor" leaf must have run the worker turns.
	if executor.callCount() == 0 {
		t.Fatalf("the declared worker leaf agent \"executor\" ran 0 turns — Ralph's "+
			"per-window agent SHADOWED the explicit leaf (old override behavior)")
	}
	// Ralph's per-window agent must NOT have been dispatched as the worker.
	if ralphAgentLeaf.callCount() != 0 {
		t.Fatalf("the Ralph per-window agent \"ralph-agent\" ran %d worker turns — it "+
			"must only FILL empty leaves, never shadow a declared leaf agent",
			ralphAgentLeaf.callCount())
	}
}

// ----------------------------------------------------------------------------
// Fix B: the plan phase honors the plan sub-strategy's OWN declared budget.
//
// A plan node with PerLoop{4} authors its task_list across TWO turns (a tool
// call on turn 1, the JSON on turn 2). Under the OLD min(global, turns+1) clamp,
// a fresh run's plan cap collapses to turns(0)+1 = 1, so the planner is pinned
// to a single turn — it never reaches turn 2 to emit the task_list and the plan
// phase budget-exhausts. Under the fix the plan child's PerLoop{4} governs, so
// the planner authors the list across >= 2 turns and the run succeeds.
// ----------------------------------------------------------------------------

// twoTurnPlanner emits a tool call on its first plan turn, then the task_list
// JSON on the second; on execute turns it emits a plain FinalResponse. Counts
// plan turns so the test can assert the planner was NOT clamped to one turn.
type twoTurnPlanner struct {
	id        AgentID
	taskList  string
	execOut   string
	mu        sync.Mutex
	planTurns int
}

func (a *twoTurnPlanner) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	if !strings.Contains(b.String(), planDirectiveMarker) {
		// Execute step.
		return NewFinalResponse(a.execOut, TokenUsage{InputTokens: 1, OutputTokens: 1})
	}
	a.planTurns++
	if a.planTurns == 1 {
		// First plan turn: do a tool call so the ReAct plan loop continues to a
		// SECOND turn (only reachable if the plan budget allows > 1 turn).
		return NewToolCallRequested([]ToolCall{{
			ID: "p0", Name: "x", Input: json.RawMessage(`{}`),
		}}, TokenUsage{InputTokens: 1, OutputTokens: 1})
	}
	// Second plan turn: author the task_list.
	return NewFinalResponse(a.taskList, TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *twoTurnPlanner) ID() AgentID { return a.id }

func (a *twoTurnPlanner) planTurnCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.planTurns
}

var _ Agent = (*twoTurnPlanner)(nil)

func TestPlanPhaseHonorsDeclaredPlanBudgetNotClampedToOneTurn(t *testing.T) {
	agent := &twoTurnPlanner{
		id:       AgentID("planner"),
		taskList: `{"tasks":["step"],"rationale":"r"}`,
		execOut:  "did the step",
	}

	cfg := standardCfg(agent)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	// Plan child carries PerLoop{4} (its OWN budget). NO global MaxTurns backstop,
	// so under the OLD clamp the plan cap = budget.Turns(0)+1 = 1 and the planner
	// is pinned to a single turn.
	planChild := withOutputSchema(4)
	execChild := ReActStrategy(^uint32(0))
	strat := PlanExecuteStrategy(PlanExecuteConfig{Plan: &planChild, Execute: &execChild})
	task := NewTask("build a CLI", SessionID("plan-budget-sess"), strat)

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess || r.Output != "did the step" {
		t.Fatalf("expected Success with the execute step output, got %+v", r)
	}
	if got := agent.planTurnCount(); got < 2 {
		t.Fatalf("the plan child's declared PerLoop{4} budget must govern: the planner "+
			"authored its task_list across >= 2 turns, got %d — the old min(global, "+
			"turns+1) clamp pinned the plan to a single turn", got)
	}
}
