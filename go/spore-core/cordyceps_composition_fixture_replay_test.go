package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"testing"
)

// ============================================================================
// Fixture-replay integration tests for the cordyceps composition (#131):
// Ralph[PlanExecute[ReAct, SelfVerifying[ReAct]]], driven by the canonical
// fixtures/strategy/cordyceps_tree.json.
//
// Mirrors rust/crates/spore-core/tests/cordyceps_composition_fixture_replay.rs.
// These exercise the SAME recorded-model harness as the plan_execute_dag tests,
// but against the FULL composed tree with its real handles wired into an
// ExecutionRegistry: agents planner/executor/ralph-agent, toolsets
// plan-tools/exec-tools, schemas plan-schema/worker-schema, and the Default-FAIL
// exec-evaluator verifier. Never edit a fixture to make a failing implementation
// pass — the fixtures are ground truth.
// ============================================================================

func compFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "model_responses", "harness", name)
}

func compStrategyFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "strategy", name)
}

func compPausedStatePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "paused_states", name)
}

// compEvaluator is the Default-FAIL evaluator the exec-evaluator handle resolves
// to — the same construction the 12-cordyceps example registers (single
// read-only turn; neither-pattern => Failed). Mirrors Rust's exec_evaluator
// EvaluatorResponseVerifier(r"(?i)\bPASS\b", r"(?i)\bFAIL\b", 1).
type compEvaluator struct {
	pass *regexp.Regexp
	fail *regexp.Regexp
}

func newCompEvaluator() *compEvaluator {
	return &compEvaluator{
		pass: regexp.MustCompile(`(?i)\bPASS\b`),
		fail: regexp.MustCompile(`(?i)\bFAIL\b`),
	}
}

func (v *compEvaluator) Verify(_ context.Context, input SelfVerifyInput) SelfVerifyVerdict {
	text := ""
	if input.EvalResult.Kind == RunSuccess {
		text = input.EvalResult.Output
	}
	if v.pass.MatchString(text) {
		return SelfVerifyVerdict{Kind: SelfVerifyPassed}
	}
	return SelfVerifyVerdict{Kind: SelfVerifyFailed, Reason: "default-FAIL: no PASS token"}
}

func (v *compEvaluator) MaxIterations() uint32 { return 1 }

var _ Verifier = (*compEvaluator)(nil)

// compTree is the canonical cordyceps tree, deserialized from the shared fixture
// (the same path the example uses).
func compTree(t *testing.T) LoopStrategy {
	t.Helper()
	raw, err := os.ReadFile(compStrategyFixturePath(t, "cordyceps_tree.json"))
	if err != nil {
		t.Fatalf("read cordyceps_tree.json: %v", err)
	}
	var ls LoopStrategy
	if err := json.Unmarshal(raw, &ls); err != nil {
		t.Fatalf("parse cordyceps_tree.json: %v", err)
	}
	return ls
}

// compPlanExecute is the PlanExecute subtree of the cordyceps tree (drops the
// Ralph wrapper) so the positional fixture maps 1:1 to one window — Ralph's
// per-window reset loop would otherwise re-enter and re-consume the (exhausted)
// replay queue.
func compPlanExecute(t *testing.T) LoopStrategy {
	t.Helper()
	tree := compTree(t)
	if tree.Kind != StrategyRalph || tree.Ralph == nil {
		t.Fatal("root is Ralph")
	}
	return *tree.Ralph.Inner
}

// compPETask is the PlanExecute task the composition tests run; the budget is
// generous so the per-NODE worker bound fires first, not the global ceiling.
func compPETask(t *testing.T, session string) Task {
	t.Helper()
	task := NewTask("audit the repo", SessionID(session), compPlanExecute(t))
	cap := uint32(64)
	task.Budget.MaxTurns = &cap
	return task
}

// compHarness builds a harness whose plan/worker/evaluator turns all replay
// positionally from ONE shared replay backend (a single cursor across the whole
// composed run), with the cordyceps handles wired into the registry. consultReg,
// if non-nil, is used as the global tool registry (so a worker consult can
// surface); otherwise an empty scripted registry is used.
func compHarness(t *testing.T, fixture string, consultReg *ScriptedToolRegistry) (*StandardHarness, *fakeRunStore) {
	t.Helper()
	raw, err := os.ReadFile(compFixturePath(t, fixture))
	if err != nil {
		t.Fatalf("read fixture %s: %v", fixture, err)
	}
	// One replay backend, shared by all three node agents so the positional
	// cursor advances across plan -> worker -> evaluate uniformly.
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture %s: %v", fixture, err)
	}
	agent := func(id string) Agent { return NewModelAgent(AgentID(id), replay) }
	reg := NewExecutionRegistryBuilder().
		Agent("planner", agent("planner")).
		Agent("executor", agent("executor")).
		Agent("ralph-agent", agent("ralph-agent")).
		Toolset("plan-tools", NewScriptedToolRegistry()).
		Toolset("exec-tools", NewScriptedToolRegistry()).
		Schema("plan-schema", json.RawMessage(`{"type":"object"}`)).
		Schema("worker-schema", json.RawMessage(`{"type":"array"}`)).
		Verifier("exec-evaluator", newCompEvaluator()).
		Build()
	var toolReg ToolRegistry = NewScriptedToolRegistry()
	if consultReg != nil {
		toolReg = consultReg
	}
	store := newFakeRunStore()
	cfg := HarnessConfig{
		// NO ConsultHandlers: the composed tree has no SubagentTool, so a worker
		// consult must surface to the host (not be mediated inside the harness).
		ToolRegistry:      toolReg,
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		Registry:          reg,
		RunStore:          store,
		EscalationMode:    AutonomousEscalation(),
	}
	return NewStandardHarness(cfg), store
}

func compSeed(t *testing.T, store *fakeRunStore, sessionID SessionID, list TaskList) {
	t.Helper()
	value, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("marshal seed list: %v", err)
	}
	if err := store.Put(context.Background(), sessionID, TaskListExtrasKey, value); err != nil {
		t.Fatalf("seed put: %v", err)
	}
}

// AC5 (static): the canonical tree's per-window worst case is computable before
// any run; an Unlimited anywhere collapses it to (_, false).
func TestCordycepsMaxStepsIs17UnlimitedIsNone(t *testing.T) {
	tree := compTree(t)
	got, ok := tree.MaxSteps()
	if !ok || got != 17 {
		t.Fatalf("MaxSteps() = (%d, %v), want (17, true)", got, ok)
	}

	// Swap the worker leaf's PerLoop{12} for Unlimited => (_, false).
	worker := tree.Ralph.Inner.PlanExecute.Execute.SelfVerify.Inner.ReActCfg
	worker.Budget = BudgetPolicy{Kind: BudgetUnlimited}
	if _, ok := tree.MaxSteps(); ok {
		t.Fatal("an Unlimited worker leaf must collapse MaxSteps() to (_, false)")
	}
}

// AC6 (handle re-resolution): a paused cordyceps tree resumes by re-resolving
// EVERY handle from a freshly-built registry, with no reconfiguration. Load the
// paused-state fixture carrying the FULL tree, serde round-trip its Task, build a
// fresh registry, and assert Validate() succeeds + every handle resolves.
func TestCordycepsResumeReResolvesHandles(t *testing.T) {
	raw, err := os.ReadFile(compPausedStatePath(t, "cordyceps_budget_exhausted.json"))
	if err != nil {
		t.Fatalf("read cordyceps_budget_exhausted.json: %v", err)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse paused state: %v", err)
	}

	// The paused state carries the FULL cordyceps tree in task.loop_strategy.
	var task Task
	if err := json.Unmarshal(doc["task"], &task); err != nil {
		t.Fatalf("Task deserializes: %v", err)
	}

	// Serde round-trip the Task (trait objects never enter the wire).
	wire, err := json.Marshal(task)
	if err != nil {
		t.Fatalf("Task serializes: %v", err)
	}
	var restored Task
	if err := json.Unmarshal(wire, &restored); err != nil {
		t.Fatalf("Task re-deserializes: %v", err)
	}

	// A fresh registry, built independently (as on a cold resume), re-resolves
	// every handle — proving no reconfiguration of the Task is needed.
	reg := compFreshRegistry()
	if err := reg.Validate(restored); err != nil {
		t.Fatalf("every handle must re-resolve, got %v", err)
	}

	// Spot-check the load-bearing handles resolve concretely.
	if restored.LoopStrategy.Kind != StrategyRalph || restored.LoopStrategy.Ralph == nil {
		t.Fatal("root is Ralph after round-trip")
	}
	ralph := restored.LoopStrategy.Ralph
	if _, ok := reg.ResolveAgent(ralph.Agent); !ok {
		t.Fatal("ralph-agent resolves")
	}
	pe := ralph.Inner.PlanExecute
	plan := pe.Plan.ReActCfg
	if _, ok := reg.ResolveAgent(plan.Agent); !ok {
		t.Fatal("planner resolves")
	}
	if _, ok := reg.ResolveToolset(plan.Toolset); !ok {
		t.Fatal("plan-tools resolves")
	}
	if _, ok := reg.ResolveSchema(*plan.Output); !ok {
		t.Fatal("plan-schema resolves")
	}
	sv := pe.Execute.SelfVerify
	if _, ok := reg.ResolveVerifier(string(sv.Evaluator)); !ok {
		t.Fatal("exec-evaluator resolves")
	}
	worker := sv.Inner.ReActCfg
	if _, ok := reg.ResolveAgent(worker.Agent); !ok {
		t.Fatal("executor resolves")
	}
	if _, ok := reg.ResolveToolset(worker.Toolset); !ok {
		t.Fatal("exec-tools resolves")
	}

	// The fixture's available_actions advertise the combinator escalation menu.
	var hr struct {
		AvailableActions []struct {
			Kind string `json:"kind"`
		} `json:"available_actions"`
	}
	if err := json.Unmarshal(doc["human_request"], &hr); err != nil {
		t.Fatalf("parse human_request: %v", err)
	}
	var kinds []string
	for _, a := range hr.AvailableActions {
		kinds = append(kinds, a.Kind)
	}
	want := []string{"continue_with_budget", "skip", "fail"}
	if len(kinds) != len(want) {
		t.Fatalf("available_actions = %v, want %v", kinds, want)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("available_actions = %v, want %v", kinds, want)
		}
	}
}

// compFreshRegistry is a fresh ExecutionRegistry wired with the cordyceps handles
// (no model backend needed — handle resolution is structural).
func compFreshRegistry() ExecutionRegistry {
	stub := func(id string) Agent { return stubAgent{id: AgentID(id)} }
	return NewExecutionRegistryBuilder().
		Agent("planner", stub("planner")).
		Agent("executor", stub("executor")).
		Agent("ralph-agent", stub("ralph-agent")).
		Toolset("plan-tools", NewScriptedToolRegistry()).
		Toolset("exec-tools", NewScriptedToolRegistry()).
		Schema("plan-schema", json.RawMessage(`{"type":"object"}`)).
		Schema("worker-schema", json.RawMessage(`{"type":"array"}`)).
		Verifier("exec-evaluator", newCompEvaluator()).
		Build()
}

// AC2: the plan phase builds a blocker-aware task graph (seeded via task_list,
// the decision-C authoring path) and the execute phase walks it as a READY-SET,
// self-verifying each task. Two independent modules both complete in ready-set
// order; the run succeeds.
func TestCordycepsPlanBuildsDAGExecuteWalksReadyset(t *testing.T) {
	h, store := compHarness(t, "cordyceps_plan_execute_readyset.jsonl", nil)
	session := SessionID("cordyceps-pe")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "audit module one", nil) // 1
	mustAddBlk(t, &l, "audit module two", nil) // 2 (independent)
	compSeed(t, store, session, l)

	r := h.Run(context.Background(), NewHarnessRunOptions(compPETask(t, "cordyceps-pe")))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	after := runStoreTaskList(t, store, session)
	for _, x := range after.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("all ready-set tasks must complete; task %d = %q", x.ID, x.Status)
		}
	}
}

// AC4: a single runaway worker node exhausts its own PerLoop{12} budget and FAILS
// its task; an INDEPENDENT module still completes. The PlanExecute drains to
// TasksBlockedByFailure with a partition that does NOT cascade to the unrelated
// branch.
func TestCordycepsRunawayBounded(t *testing.T) {
	h, store := compHarness(t, "cordyceps_runaway_bounded.jsonl", nil)
	session := SessionID("cordyceps-runaway")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "root module", nil)               // 1 (completes)
	mustAddBlk(t, &l, "runaway module", []uint32{1})    // 2 -> 1 (PerLoop{12} budget-Fail)
	mustAddBlk(t, &l, "dependent of runaway", []uint32{2}) // 3 -> 2 (cascade-blocked)
	mustAddBlk(t, &l, "independent module", nil)        // 4 (still completes)
	compSeed(t, store, session, l)

	r := h.Run(context.Background(), NewHarnessRunOptions(compPETask(t, "cordyceps-runaway")))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTasksBlockedByFailure {
		t.Fatalf("expected TasksBlockedByFailure, got %+v", r)
	}
	if r.Reason.FailedTask != 2 {
		t.Fatalf("failed task = %d, want 2 (the runaway module)", r.Reason.FailedTask)
	}
	if want := []uint32{1, 4}; !equalU32s(r.Reason.Completed, want) {
		t.Fatalf("completed = %v, want %v (root + independent branch)", r.Reason.Completed, want)
	}
	if want := []uint32{2, 3}; !equalU32s(r.Reason.Blocked, want) {
		t.Fatalf("blocked = %v, want %v (runaway + its dependent)", r.Reason.Blocked, want)
	}
}

// AC3: the registered exec-evaluator is Default-FAIL — Passed only on an explicit
// PASS, Failed on indeterminate output (proving the worker self-checks before a
// task clears).
func TestCordycepsSelfVerifiedDefaultFail(t *testing.T) {
	v := newCompEvaluator()
	if v.MaxIterations() != 1 {
		t.Fatalf("single read-only evaluator turn, got MaxIterations() = %d", v.MaxIterations())
	}
	success := func(out string) RunResult {
		return RunResult{Kind: RunSuccess, Output: out, SessionID: SessionID("s")}
	}
	input := func(eval string) SelfVerifyInput {
		return SelfVerifyInput{
			BuildResult: success("audited module"),
			EvalResult:  success(eval),
			Workspace:   "/tmp",
			Iteration:   0,
		}
	}
	if got := v.Verify(context.Background(), input("verdict: PASS")); got.Kind != SelfVerifyPassed {
		t.Fatalf("explicit PASS must pass, got %+v", got)
	}
	if got := v.Verify(context.Background(), input("looks plausible")); got.Kind != SelfVerifyFailed {
		t.Fatalf("indeterminate output must default-FAIL, got %+v", got)
	}
}

// Consult ladder (#114, PRESERVED through the composed tree). A worker leaf
// consult — with NO SubagentTool to mediate it — propagates all the way up to a
// top-level RunResult.Consult. The host (this test) injects an answer via
// ResumeConsult, the worker finishes, the evaluator passes, and the run
// completes. This exercises the host-mediation seam the 12-cordyceps example
// relies on.
func TestCordycepsWorkerConsultSurfacesAndHostResumes(t *testing.T) {
	// The GLOBAL tool registry returns a worker-side consult on the first dispatch
	// (the worker's consult_advisor call), then defaults to plain success.
	toolReg := NewScriptedToolRegistry()
	toolReg.Push(NewToolOutputConsult(ConsultRequest{
		Kind:      "advice",
		Situation: "found a suspicious unwrap in module one",
		Attempts:  1,
		Question:  "is this a real defect and how severe?",
	}))

	h, store := compHarness(t, "cordyceps_worker_consult.jsonl", toolReg)

	// Seed ONE ready task so the execute phase runs exactly one worker.
	session := SessionID("cordyceps-consult")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "audit module one", nil)
	compSeed(t, store, session, l)

	// First leg: drive to the consult pause.
	first := h.Run(context.Background(), NewHarnessRunOptions(compPETask(t, "cordyceps-consult")))
	if first.Kind != RunConsult {
		t.Fatalf("expected RunResult.Consult to surface to the host, got %+v", first)
	}
	if first.ConsultRequest == nil || first.ConsultRequest.Kind != "advice" {
		t.Fatalf("the advice consult must reach the host, got %+v", first.ConsultRequest)
	}
	if first.State == nil {
		t.Fatal("consult pause must carry a PausedState")
	}
	if !contains(first.ConsultRequest.Question, "real defect") {
		t.Fatalf("the request must carry the worker's question verbatim, got %q", first.ConsultRequest.Question)
	}

	// Host mediation: inject the advisor's answer and resume the composed tree.
	resumed := h.ResumeConsult(
		context.Background(),
		*first.State,
		NewConsultAnswer("Yes — unwrap on untrusted input is a real high-severity panic risk."),
		nil,
	)
	// The worker continued mid-loop AFTER the consult (the finding it emitted
	// post-answer is the run output) — proving the answer was injected and the
	// SelfVerifying evaluator then cleared the task, not a bare leaf resume.
	if resumed.Kind != RunSuccess {
		t.Fatalf("expected Success after ResumeConsult, got %+v", resumed)
	}
	if !contains(resumed.Output, "advisor-confirmed") {
		t.Fatalf("run output must be the post-consult worker finding, got %q", resumed.Output)
	}

	// The worker's task self-verified and completed after the consult.
	after := runStoreTaskList(t, store, session)
	for _, x := range after.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("the consulted task must complete; task %d = %q", x.ID, x.Status)
		}
	}
}
