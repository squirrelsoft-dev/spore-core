package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

// ============================================================================
// PlanExecute execute-phase tests (issue #59). Mirror the Rust harness unit
// tests in rust/crates/spore-core/src/harness.rs.
// ============================================================================

// fakeRunStore is an in-memory RunStore double. It records every Put so the
// persistence test can read the durably-stored task list back. Defined locally
// rather than importing the storage package (which imports sporecore — a cycle).
type fakeRunStore struct {
	mu   sync.Mutex
	data map[string]json.RawMessage
}

func newFakeRunStore() *fakeRunStore { return &fakeRunStore{data: map[string]json.RawMessage{}} }

func (s *fakeRunStore) Put(_ context.Context, sessionID SessionID, key string, value json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[string(sessionID)+"/"+key] = append(json.RawMessage(nil), value...)
	return nil
}

func (s *fakeRunStore) get(sessionID SessionID, key string) (json.RawMessage, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.data[string(sessionID)+"/"+key]
	return v, ok
}

var _ RunStore = (*fakeRunStore)(nil)

// extrasTaskList decodes the TaskList mirrored into SessionState.Extras.
func extrasTaskList(t *testing.T, s SessionState) TaskList {
	t.Helper()
	raw, ok := s.Extras[TaskListExtrasKey]
	if !ok {
		t.Fatalf("no task list mirrored under %q; extras=%v", TaskListExtrasKey, s.Extras)
	}
	b := mustJSON(t, raw)
	var list TaskList
	if err := json.Unmarshal(b, &list); err != nil {
		t.Fatalf("decode task list: %v", err)
	}
	return list
}

func planTaskBudget(b BudgetLimits) Task {
	t := planTask("build a CLI")
	t.Budget = b
	return t
}

// Happy-path / drain: a full PlanExecute run over a 2-task plan succeeds with
// the last step's output and one turn per task plus the plan turn.
func TestPlanExecuteHappyPathDrains(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["one","two"],"rationale":"r"}`))
	a.Push(planFinal("did one"))
	a.Push(planFinal("did two"))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "did two" {
		t.Fatalf("output = %q, want %q", r.Output, "did two")
	}
	if r.Turns != 3 {
		t.Fatalf("turns = %d, want 3 (plan + 2 steps)", r.Turns)
	}
}

// Task drain Pending -> InProgress -> Completed: inspect the mirrored list after
// a run via runExecutePhase directly so we can read the final state.
func TestExecutePhaseDrainsPendingInProgressCompleted(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal("done one"))
	a.Push(planFinal("done two"))
	h := NewStandardHarness(standardCfg(a))
	tk := planTask("build a CLI")
	state := SessionState{}
	list := PlanArtifactToTaskList(PlanArtifact{Tasks: []string{"one", "two"}})
	for _, x := range list.Tasks {
		if x.Status != TaskStatusPending {
			t.Fatalf("task %d not Pending initially", x.ID)
		}
	}
	r := h.runExecutePhase(context.Background(), &tk, &state, list,
		BudgetSnapshot{Turns: 1}, AggregateUsage{}, nil)
	if r.Kind != RunSuccess || r.Output != "done two" {
		t.Fatalf("got %+v", r)
	}
	final := extrasTaskList(t, state)
	for _, x := range final.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("task %d status = %q, want completed", x.ID, x.Status)
		}
	}
}

// Q1 per-task turn allocation + shared budget: global cap 7, plan turn spent
// (1), 3 tasks split the remaining 6 turns (2 each). Task a needs >2 turns and
// is cut off by its per-task cap, proving allocation + shared-budget carry.
func TestExecutePhasePerTaskTurnAllocation(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["a","b","c"]}`))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "1", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "2", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	max := uint32(7)
	r := h.Run(context.Background(), NewHarnessRunOptions(planTaskBudget(BudgetLimits{MaxTurns: &max})))
	if r.Kind != RunFailure {
		t.Fatalf("expected Failure from turn cap, got %+v", r)
	}
	if r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("per-task turn cap enforced, got %+v", r.Reason)
	}
	if r.Turns != 3 {
		t.Fatalf("turns = %d, want 3 (1 plan + 2 task turns)", r.Turns)
	}
}

// Budget exhaustion MID-execute: a tight global turn cap stops the run partway
// with BudgetExceeded, not StepFailed.
func TestExecutePhaseBudgetExhaustedMidExecute(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["x","y","z"]}`))
	a.Push(planFinal("did x"))
	h := NewStandardHarness(standardCfg(a))
	max := uint32(2) // plan(1) + exactly one execute turn
	r := h.Run(context.Background(), NewHarnessRunOptions(planTaskBudget(BudgetLimits{MaxTurns: &max})))
	if r.Kind != RunFailure {
		t.Fatalf("expected Failure, got %+v", r)
	}
	if r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("global turn budget is the hard stop, got %+v", r.Reason)
	}
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2", r.Turns)
	}
}

// Observability span count: plan turn + one span per executed step.
func TestExecutePhaseSpanCount(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["a","b"]}`))
	a.Push(planFinal("did a"))
	a.Push(planFinal("did b"))
	cfg := standardCfg(a)
	obs := newCountingObserver()
	cfg.Observability = obs
	h := NewStandardHarness(cfg)
	_ = h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if obs.turns != 3 {
		t.Fatalf("turn spans = %d, want 3 (plan + one per step)", obs.turns)
	}
}

// Compaction-in-loop: a multi-turn step (tool call then final) reuses the
// shared ReAct machinery and the run still drains to completion.
func TestExecutePhaseCompactionInLoop(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["only"]}`))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "1", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(planFinal("finished only"))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess || r.Output != "finished only" {
		t.Fatalf("got %+v", r)
	}
	if r.Turns != 3 { // plan(1) + tool turn(1) + final(1)
		t.Fatalf("turns = %d, want 3", r.Turns)
	}
}

// OnTaskAdvance fires exactly N times with the correct task_index / total_tasks.
func TestExecutePhaseOnTaskAdvanceFiresPerTask(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["s0","s1","s2"]}`))
	a.Push(planFinal("d0"))
	a.Push(planFinal("d1"))
	a.Push(planFinal("d2"))

	var (
		mu          sync.Mutex
		fireCount   int
		seenIndices []int
		seenTotals  []int
	)
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("count-advance", []HookEvent{HookEventOnTaskAdvance},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			mu.Lock()
			defer mu.Unlock()
			fireCount++
			seenIndices = append(seenIndices, hctx.TaskIndex)
			seenTotals = append(seenTotals, hctx.TotalTasks)
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	cfg := standardCfg(a)
	cfg.Hooks = chain
	h := NewStandardHarness(cfg)
	_ = h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))

	if fireCount != 3 {
		t.Fatalf("fire count = %d, want 3", fireCount)
	}
	if want := []int{0, 1, 2}; !equalInts(seenIndices, want) {
		t.Fatalf("indices = %v, want %v", seenIndices, want)
	}
	if want := []int{3, 3, 3}; !equalInts(seenTotals, want) {
		t.Fatalf("totals = %v, want %v", seenTotals, want)
	}
}

// OnTaskAdvance may rewrite the step instruction (Q1, mutable hook).
func TestExecutePhaseOnTaskAdvanceRewritesInstruction(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["original"]}`))
	a.Push(planFinal("did rewritten"))
	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("rewrite", []HookEvent{HookEventOnTaskAdvance},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			hctx.Task.Instruction = "REWRITTEN"
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	cfg := standardCfg(a)
	// Capture what instruction the agent actually saw by recording messages.
	rec := &recordingContextManager{}
	cfg.ContextManager = rec
	cfg.Hooks = chain
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess {
		t.Fatalf("got %+v", r)
	}
	seenRewritten := false
	for i, role := range rec.roles {
		if role == RoleUser && rec.texts[i] == "REWRITTEN" {
			seenRewritten = true
		}
	}
	if !seenRewritten {
		t.Fatalf("rewritten instruction not seeded; user texts=%v", rec.texts)
	}
}

// Q3: an empty plan -> HaltEmptyPlan (not a silent success).
func TestExecutePhaseEmptyPlanHalts(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":[],"rationale":"nothing"}`))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunFailure || r.Reason.Kind != HaltEmptyPlan {
		t.Fatalf("expected EmptyPlan, got %+v", r)
	}
	if r.Turns != 1 {
		t.Fatalf("turns = %d, want 1 (only the plan turn ran)", r.Turns)
	}
}

// Q5: a step that errors aborts the whole run with StepFailed carrying the
// failing index + instruction; later tasks do NOT run.
func TestExecutePhaseStepFailureAbortsWithStepFailed(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["good","bad","never"]}`))
	a.Push(planFinal("did good"))
	a.Push(NewTurnError(NewEmptyResponseError(), nil)) // step "bad" errors
	a.Push(planFinal("never run"))                     // must NOT be consumed
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStepFailed {
		t.Fatalf("expected StepFailed, got %+v", r)
	}
	if r.Reason.TaskIndex != 1 {
		t.Fatalf("task index = %d, want 1", r.Reason.TaskIndex)
	}
	if r.Reason.Task != "bad" {
		t.Fatalf("task = %q, want %q", r.Reason.Task, "bad")
	}
	// plan + good + bad consumed; "never" remains queued (the third task never ran).
	if remaining := len(a.results); remaining != 1 {
		t.Fatalf("remaining queued responses = %d, want 1 (the 'never' task must not run)", remaining)
	}
}

// Q4: the task list is persisted through the RunStore seam (not the #71 sandbox
// path). Assert the durable RunStore holds the completed list after a run.
func TestExecutePhasePersistsThroughRunStore(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["one"]}`))
	a.Push(planFinal("did one"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)
	_ = h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))

	raw, ok := store.get(SessionID("plan-sess"), TaskListExtrasKey)
	if !ok {
		t.Fatal("task list not present in run store")
	}
	var list TaskList
	if err := json.Unmarshal(raw, &list); err != nil {
		t.Fatalf("decode stored list: %v", err)
	}
	if len(list.Tasks) != 1 {
		t.Fatalf("tasks = %d, want 1", len(list.Tasks))
	}
	if list.Tasks[0].Status != TaskStatusCompleted {
		t.Fatalf("task status = %q, want completed", list.Tasks[0].Status)
	}
}

// Q2: success output is the LAST step's final response, not a concatenation and
// not the plan rationale.
func TestExecutePhaseSuccessOutputIsLastStep(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["a","b"],"rationale":"RATIONALE_TOKEN"}`))
	a.Push(planFinal("FIRST_STEP_OUTPUT"))
	a.Push(planFinal("LAST_STEP_OUTPUT"))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "LAST_STEP_OUTPUT" {
		t.Fatalf("output = %q, want LAST_STEP_OUTPUT", r.Output)
	}
}

// Planner-agent routing through the FULL run: the planner runs the plan turn and
// the default agent runs the execute steps.
func TestExecutePhasePlannerAgentRouting(t *testing.T) {
	def := NewMockAgent("default")
	def.Push(planFinal("did the step"))
	planner := NewMockAgent("planner")
	planner.Push(planFinal(`{"tasks":["step"]}`))

	cfg := standardCfg(def)
	cfg.PlannerAgent = planner
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess || r.Output != "did the step" {
		t.Fatalf("got %+v", r)
	}
	if len(planner.results) != 0 {
		t.Fatalf("planner ran %d extra turns; want exactly the plan turn", len(planner.results))
	}
	if len(def.results) != 0 {
		t.Fatalf("default agent ran extra turns; want exactly the execute step")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ============================================================================
// Fixture-replay test against the shared plan_execute_loop.jsonl. Mirrors
// rust/crates/spore-core/tests/plan_execute_loop_fixture_replay.rs and must
// produce the SAME outcome. NEVER edit the fixture.
// ============================================================================

func planExecuteLoopFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "model_responses", "harness", "plan_execute_loop.jsonl")
}

func TestPlanExecuteLoopFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(planExecuteLoopFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	agent := NewModelAgent(AgentID("plan-execute"), replay)

	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
	}
	h := NewStandardHarness(cfg)
	task := NewTask("build a CLI", SessionID("plan-execute-fixture"), LoopStrategy{Kind: StrategyPlanExecute})

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	// Q2: the success handle is the LAST completed step's final text.
	if r.Output != "wrote the integration tests" {
		t.Fatalf("output = %q, want %q", r.Output, "wrote the integration tests")
	}
	// Q1: one plan turn + one turn per task (2) under the shared budget.
	if r.Turns != 3 {
		t.Fatalf("turns = %d, want 3 (plan + one per task)", r.Turns)
	}
}
