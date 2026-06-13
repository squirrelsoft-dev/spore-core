package sporecore

import (
	"context"
	"strings"
	"sync"
	"testing"
)

// recompose_124_test.go — the three non-ReAct-inner regression tests for the
// #124 recomposition: SelfVerifying / Ralph / HillClimbing must GENUINELY recurse
// into their `inner` child per iteration / window, not delegate to a monolithic
// loop that hardcodes a ReAct worker. Each test wires a NON-ReAct inner and
// asserts the inner combinator's distinctive turn fires the expected number of
// times. The deleted monolithic loops would have ignored `inner` entirely and
// recorded ZERO. Analogues of Rust's *_runs_non_react_inner_* tests.

// planDirectiveMarker is a stable substring of planDirective() used to detect a
// plan turn from the assembled context.
const planDirectiveMarker = "Respond with a "

// planTurnCounter returns the scripted plan JSON for any turn whose context
// carries the plan directive marker (counting those as PLAN turns), and a plain
// execOut FinalResponse otherwise. Lets the regression tests count how many
// genuine inner plan turns fired across the recursion.
type planTurnCounter struct {
	id        AgentID
	planJSON  string
	mu        sync.Mutex
	planTurns int
	execOut   string
}

func newPlanTurnCounter(id string, planJSON, execOut string) *planTurnCounter {
	return &planTurnCounter{id: AgentID(id), planJSON: planJSON, execOut: execOut}
}

func (a *planTurnCounter) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	if strings.Contains(b.String(), planDirectiveMarker) {
		a.planTurns++
		return NewFinalResponse(a.planJSON, TokenUsage{InputTokens: 1, OutputTokens: 1})
	}
	return NewFinalResponse(a.execOut, TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *planTurnCounter) ID() AgentID { return a.id }

func (a *planTurnCounter) planTurnCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.planTurns
}

var _ Agent = (*planTurnCounter)(nil)

// withOutputSchema returns a copy of a ReAct LoopStrategy carrying an empty
// output-schema handle (resolved under the default empty key folded by the
// harness), satisfying the A.5 structured-slot contract.
func withOutputSchema(maxIter uint32) LoopStrategy {
	s := ReActStrategy(maxIter)
	ref := SchemaRef("")
	s.ReActCfg.Output = &ref
	return s
}

// ----------------------------------------------------------------------------
// SelfVerifying[inner: PlanExecute[ReAct, ReAct]] — the inner plan turn fires
// once per SelfVerifying iteration.
// ----------------------------------------------------------------------------

func TestSelfVerifyingRunsNonReactInnerWorker(t *testing.T) {
	// The inner worker is a genuine PlanExecute (NOT a bare ReAct). Each
	// SelfVerifying iteration must run the WHOLE PlanExecute loop, so a plan turn
	// fires per iteration. The verifier fails once then passes => 2 iterations =>
	// the inner plan turn must fire EXACTLY twice. A dropped inner would record 0.
	agent := newPlanTurnCounter("sv-inner", `{"tasks":["t0"],"rationale":"r"}`, "done")
	v := newSVVerifier(3, "again", "pass")

	plan := withOutputSchema(^uint32(0))
	exec := ReActStrategy(^uint32(0))
	inner := PlanExecuteStrategy(PlanExecuteConfig{Plan: &plan, Execute: &exec})

	cfg := standardCfg(agent).WithRegistryVerifier("eval", v)
	h := NewStandardHarness(cfg)

	task := NewTask("build the widget", SessionID("sv-inner-sess"),
		SelfVerifyingStrategy(SelfVerifyingConfig{Inner: &inner, Evaluator: SchemaRef("eval")}))

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if v.calls != 2 {
		t.Fatalf("expected 2 verifier calls (one per iteration), got %d", v.calls)
	}
	if got := agent.planTurnCount(); got != 2 {
		t.Fatalf("the inner PlanExecute plan turn must fire once per SelfVerifying "+
			"iteration (2); got %d — a dropped inner would record 0", got)
	}
}

// ----------------------------------------------------------------------------
// Ralph[inner: SelfVerifying[ReAct]] over incomplete,complete — the inner
// verifier fires at least once per window.
// ----------------------------------------------------------------------------

// ralphIncompleteAgent always writes the SAME incomplete .spore/progress.json on
// its BUILD turns (those NOT carrying the evaluator role chunk), modelling an
// agent that never finishes — so every Ralph window exhausts its inner loop and
// the OUTER loop resets until MaxResets windows are spent. Each window's inner
// SelfVerifying nonetheless runs its build->evaluate->verify cycle, so the
// verifier fires per window.
type ralphIncompleteAgent struct {
	id    AgentID
	store *fakeRunStore
	mu    sync.Mutex
}

func (a *ralphIncompleteAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	if !strings.Contains(b.String(), RoleEvaluatorChunk) {
		writeRalphProgress(a.store, ralphWindow{complete: false, remaining: []string{"more"}})
	}
	return NewFinalResponse("done", TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *ralphIncompleteAgent) ID() AgentID { return a.id }

var _ Agent = (*ralphIncompleteAgent)(nil)

func TestRalphRunsNonReactInnerPerWindow(t *testing.T) {
	// The inner is a genuine SelfVerifying (NOT a bare ReAct). Each Ralph window
	// must run the WHOLE SelfVerifying loop — so its verifier fires per window. The
	// agent never writes a complete progress checkpoint, so each window exhausts
	// and the OUTER loop resets MaxResets (2) times => the inner verifier fires
	// >= 1 per window => >= 2 total. A monolithic Ralph that ignored `inner`
	// (hardcoded ReAct worker) would record ZERO verifier invocations.
	v := newSVVerifier(3, "pass") // PASS each iteration (1 verify per SV run)
	store := newFakeRunStore()
	agent := &ralphIncompleteAgent{id: AgentID("ralph-sv"), store: store}

	worker := withOutputSchema(^uint32(0))
	inner := SelfVerifyingStrategy(SelfVerifyingConfig{Inner: &worker, Evaluator: SchemaRef("eval")})

	cfg := standardCfg(agent)
	cfg.RunStore = store
	cfg.ProjectNamespace = ralphProjectNS
	cfg.MaxResets = 2
	cfg = cfg.WithRegistryVerifier("eval", v)
	h := NewStandardHarness(cfg)

	task := NewTask("build the thing", SessionID("ralph-sv-sess"),
		RalphStrategy(RalphConfig{Inner: &inner}))

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltRalphCompletionUnmet {
		t.Fatalf("expected RalphCompletionUnmet after exhausting windows, got %+v", r)
	}
	// One inner-SelfVerifying verifier call per window over MaxResets (2) windows.
	if v.calls < 2 {
		t.Fatalf("the inner SelfVerifying verifier must fire >= 1 per window (>= 2 over "+
			"two windows); got %d — a dropped inner would record 0", v.calls)
	}
}

// ----------------------------------------------------------------------------
// HillClimbing[inner: PlanExecute[ReAct, ReAct]] improve-then-stagnate — the
// inner plan turn fires once per HillClimbing iteration.
// ----------------------------------------------------------------------------

func TestHillClimbingRunsNonReactInnerPerIteration(t *testing.T) {
	// The inner is a genuine PlanExecute (NOT a bare ReAct). Each HillClimbing
	// iteration (after the iteration-0 baseline) runs the WHOLE PlanExecute loop —
	// its EXECUTE phase fires per iteration, which is what proves genuine #124
	// recursion. Metric: baseline 1.0, improve to 2.0 (iter1, kept), then 2.0
	// again (iter2, no improvement) with max_stagnation=1 => halt after iter2.
	//
	// #138 AC1: the durable task_list is project-scoped (a durable in-memory
	// RunStore is wired below), so iteration 1 authors the list and iteration 2
	// SKIPS re-planning. The plan phase therefore fires EXACTLY ONCE across the
	// whole climb — which still proves the recursion (a hardcoded-ReAct proposer
	// would fire the plan phase ZERO times).
	agent := newPlanTurnCounter("hc-inner", `{"tasks":["t0"],"rationale":"r"}`, "done")
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{
		res(1.0), // iteration 0 baseline (no agent turn)
		res(2.0), // iteration 1: improvement => kept, stagnation resets
		res(2.0), // iteration 2: no improvement => stagnation=1 => halt
	}}

	dir := t.TempDir()
	inner := PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    func() *LoopStrategy { p := withOutputSchema(^uint32(0)); return &p }(),
		Execute: PtrStrategy(ReActStrategy(^uint32(0))),
	})

	cfg, _ := hcConfigWith(t, dir, agent, eval)
	// #138 AC1: wire a DURABLE in-memory RunStore (project-scoped, #142) so the
	// task_list survives across HillClimbing iterations and the skip-replan guard
	// fires — mirroring the Rust/TS/Python siblings. (hcConfigWith is storeless and
	// has no other callers; scope the store to THIS test inline.)
	store := newFakeRunStore()
	cfg.RunStore = store
	cfg.ProjectNamespace = SessionID("hc-project")
	h := NewStandardHarness(cfg)

	maxTurns := uint32(64)
	task := NewTask("optimize", SessionID("hc-inner-sess"), HillClimbingStrategy(HillClimbingConfig{
		Inner:                 &inner,
		Direction:             OptimizationMaximize,
		MaxStagnation:         1,
		RevertOnNoImprovement: false,
		MinImprovementDelta:   0.0,
		Evaluator:             AgentRef("metric"),
	}))
	task.Budget = BudgetLimits{MaxTurns: &maxTurns}

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("expected StagnationLimitReached, got %+v", r)
	}
	if got := agent.planTurnCount(); got != 1 {
		t.Fatalf("#138 AC1: the inner PlanExecute plan turn fires EXACTLY ONCE (iter 1 "+
			"authors the durable task_list; later iterations skip re-planning); got %d — "+
			"a dropped inner would record 0", got)
	}
}

// hcConfigWith builds a HillClimbing config rooted at dir with the supplied agent
// and metric evaluator registered under the "metric" key.
func hcConfigWith(t *testing.T, dir string, agent Agent, eval MetricEvaluator) (HarnessConfig, *hcRootedSandbox) {
	t.Helper()
	sb := &hcRootedSandbox{root: dir}
	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           sb,
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	cfg = cfg.WithRegistryMetricEvaluator("metric", eval)
	return cfg, sb
}
