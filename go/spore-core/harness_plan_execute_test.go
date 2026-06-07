package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

// Get satisfies the RunStore seam (#124 deep-resume read path).
func (s *fakeRunStore) Get(_ context.Context, sessionID SessionID, key string) (json.RawMessage, bool, error) {
	v, ok := s.get(sessionID, key)
	return v, ok, nil
}

var _ RunStore = (*fakeRunStore)(nil)

// runStoreTaskList decodes the TaskList persisted to the RunStore seam under
// TaskListExtrasKey (#76 — no longer mirrored into SessionState.Extras).
func runStoreTaskList(t *testing.T, store *fakeRunStore, sessionID SessionID) TaskList {
	t.Helper()
	raw, ok := store.get(sessionID, TaskListExtrasKey)
	if !ok {
		t.Fatalf("no task list present in run store under %q", TaskListExtrasKey)
	}
	var list TaskList
	if err := json.Unmarshal(raw, &list); err != nil {
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
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)
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
	final := runStoreTaskList(t, store, tk.SessionID)
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

// #76: after a plan/execute run, BOTH persistence keys live on the RunStore
// seam and NEITHER is mirrored into SessionState.Extras. Drives the phases
// directly (rather than Run) so the post-run state.Extras is observable. The
// ephemeral extras keys (__rich_state, subagent_handoff_summary) are owned by
// other components and untouched here.
func TestPlanExecutePersistenceLivesOnRunStoreNotExtras(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["one","two"],"rationale":"why"}`))
	a.Push(planFinal("did one"))
	a.Push(planFinal("did two"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	tk := planTask("build a CLI")
	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), &tk, &state, BudgetSnapshot{}, nil)
	if failure != nil {
		t.Fatalf("unexpected plan-phase failure: %+v", *failure)
	}
	list := PlanArtifactToTaskList(outcome.artifact)
	r := h.runExecutePhase(context.Background(), &tk, &state, list,
		BudgetSnapshot{Turns: outcome.turns}, outcome.usage, nil)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}

	// Both keys are durable in the RunStore.
	if _, ok := store.get(tk.SessionID, PlanExecuteExtrasKey); !ok {
		t.Fatalf("plan artifact not present in run store under %q", PlanExecuteExtrasKey)
	}
	if _, ok := store.get(tk.SessionID, TaskListExtrasKey); !ok {
		t.Fatalf("task list not present in run store under %q", TaskListExtrasKey)
	}

	// Neither key is mirrored into SessionState.Extras anymore.
	if _, ok := state.Extras[PlanExecuteExtrasKey]; ok {
		t.Fatalf("%q must not be mirrored into Extras", PlanExecuteExtrasKey)
	}
	if _, ok := state.Extras[TaskListExtrasKey]; ok {
		t.Fatalf("%q must not be mirrored into Extras", TaskListExtrasKey)
	}
}

// #124: PlanExecute genuinely recurses into a NON-ReAct execute child. With a
// SelfVerifying execute child over a 2-task plan, the scripted verifier must be
// invoked exactly twice (once per task). The old hardcoded-ReAct execute impl
// dropped the SelfVerifying child entirely and would record ZERO invocations —
// this is the regression that proves the recursion is real. Mirrors Rust's
// plan_execute_runs_non_react_execute_child_per_task.
func TestPlanExecuteRunsNonReactExecuteChildPerTask(t *testing.T) {
	// cfg.Agent runs BOTH the plan turn (JSON) and the per-task build phase.
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["t0","t1"],"rationale":"r"}`))
	a.Push(planFinal("built t0"))
	a.Push(planFinal("built t1"))
	// The evaluate phase runs on a distinct agent; PASS each task.
	eval := newRecordingAgent("eval", "PASS")
	// The verifier records every invocation; PASS each time.
	v := newSVVerifier(3, "pass", "pass")
	cfg := standardCfg(a)
	cfg.EvaluatorAgent = eval
	cfg.Verifier = v
	h := NewStandardHarness(cfg)

	// The execute child is a genuine SelfVerifying combinator (NOT a ReAct).
	strat := PlanExecuteStrategy(PlanExecuteConfig{
		Plan: PtrStrategy(ReActStrategy(^uint32(0))),
		Execute: PtrStrategy(SelfVerifyingStrategy(SelfVerifyingConfig{
			Inner:     PtrStrategy(ReActStrategy(^uint32(0))),
			Evaluator: SchemaRef(""),
		})),
	})
	task := NewTask("build a CLI", SessionID("plan-sess"), strat)

	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "built t1" {
		t.Fatalf("Q2: output = %q, want last step's final output %q", r.Output, "built t1")
	}

	// The smoking gun: the SelfVerifying evaluator ran ONCE PER TASK (2x). A
	// dropped execute child would record ZERO verifier invocations.
	if v.calls != 2 {
		t.Fatalf("the SelfVerifying execute child must run its evaluator once per task; got %d invocations", v.calls)
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
	task := NewTask("build a CLI", SessionID("plan-execute-fixture"), PlanExecuteStrategy(PlanExecuteSimple(nil)))

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

// planRecordingAgent captures the assembled context of EVERY turn (in order) and
// yields queued TurnResults, so a test can assert what the model saw on the
// plan turn versus each execute-step turn.
type planRecordingAgent struct {
	id      AgentID
	mu      sync.Mutex
	results []TurnResult
	seen    [][]Message // one entry per Turn() call, in invocation order
}

func newPlanRecordingAgent(id AgentID) *planRecordingAgent { return &planRecordingAgent{id: id} }

func (a *planRecordingAgent) push(r TurnResult) *planRecordingAgent {
	a.results = append(a.results, r)
	return a
}

func (a *planRecordingAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.seen = append(a.seen, append([]Message(nil), c.Messages...))
	if len(a.results) == 0 {
		return NewTurnError(NewEmptyResponseError(), nil)
	}
	r := a.results[0]
	a.results = a.results[1:]
	return r
}

func (a *planRecordingAgent) ID() AgentID { return a.id }

// contextText flattens a captured context's text content into one string for
// substring assertions.
func contextText(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Content.Type {
		case ContentTypeText:
			b.WriteString(m.Content.Text)
			b.WriteByte('\n')
		case ContentTypeToolCall:
			if m.Content.ToolCall != nil {
				b.WriteString(m.Content.ToolCall.Name)
				b.WriteByte('\n')
			}
		case ContentTypeToolResult:
			// Tool results carry prior steps' outputs forward — surface their
			// content so accumulation is visible.
			if m.Content.ToolResult != nil {
				b.WriteString(m.Content.ToolResult.Content)
				b.WriteByte('\n')
			}
		}
	}
	return b.String()
}

// #93 regression: the execute phase maintains ONE accumulating context across
// steps. After a successful step its conversation (instruction + tool calls +
// TOOL RESULTS + assistant output) is folded back into the shared session, so
// the NEXT step's sub-loop sees prior steps' RESULTS — not just their
// instructions. Drive a 2-step run where STEP 1 issues a tool call returning a
// distinctive string and assert STEP 2's context carries it.
func TestExecuteStepsAccumulatePriorResults(t *testing.T) {
	agent := newPlanRecordingAgent("rec")
	// Plan turn: a 2-step plan (research -> use the result).
	agent.push(planFinal(`{"tasks":["research tokio","summarize findings"],"rationale":"r"}`))
	// Step 1: call a tool, then finalize using its result.
	agent.push(NewToolCallRequested([]ToolCall{{ID: "1", Name: "lookup", Input: json.RawMessage(`{}`)}}, turnUsage()))
	agent.push(planFinal("researched"))
	// Step 2: finalize directly (it must SEE step 1's tool result).
	agent.push(planFinal("summarized"))

	cfg := standardCfg(agent)
	// Use a context manager that records assistant turns + tool results into the
	// session (NoopContextManager drops both), so the accumulated execute context
	// actually carries step 1's tool result and assistant output forward.
	cfg.ContextManager = &recordingContextManager{}
	reg := NewScriptedToolRegistry()
	// Step 1's tool call returns a distinctive result string.
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "TOKIO_FACTS_123"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "summarized" {
		t.Fatalf("expected last step's output %q, got %q", "summarized", r.Output)
	}

	agent.mu.Lock()
	defer agent.mu.Unlock()
	// 1 plan turn + (tool call + final) for step 1 + 1 for step 2 = 4.
	if len(agent.seen) != 4 {
		t.Fatalf("captured %d turns, want 4 (plan turn + 3 execute turns)", len(agent.seen))
	}

	// Step 1's SECOND turn (index 2) sees the tool result — sanity check that
	// the result string is on the wire at all.
	if c := contextText(agent.seen[2]); !strings.Contains(c, "TOKIO_FACTS_123") {
		t.Fatalf("step-1 final turn should follow its own tool result:\n%s", c)
	}

	// The accumulation guarantee: STEP 2's context (index 3) CONTAINS step 1's
	// tool result, proving the execute loop carried it forward.
	step2 := contextText(agent.seen[3])
	if !strings.Contains(step2, "TOKIO_FACTS_123") {
		t.Fatalf("step 2 must see step 1's tool result (accumulating context):\n%s", step2)
	}
	// Step 2 also sees step 1's prior instruction + assistant output.
	if !strings.Contains(step2, "research tokio") || !strings.Contains(step2, "researched") {
		t.Fatalf("step 2 must see step 1's instruction and output:\n%s", step2)
	}
}

// #93 regression: the plan-phase directive ("Produce a step-by-step plan…
// Respond with a single JSON object…") must NOT leak into the SHARED session
// state, otherwise every execute step re-sees it and an instruction-following
// model re-emits a plan instead of calling tools. Drive a full 2-step
// PlanExecute run with a context-capturing agent and assert no execute-step
// context carries the directive, while the step instructions DO reach those
// contexts (and a step can issue a tool call).
func TestPlanDirectiveDoesNotLeakIntoExecuteContext(t *testing.T) {
	const (
		producePlan = "Produce a step-by-step plan"
		respondJSON = "Respond with a single JSON object"
	)

	agent := newPlanRecordingAgent("rec")
	// Plan turn: produce a 2-step plan.
	agent.push(planFinal(`{"tasks":["step one","step two"],"rationale":"r"}`))
	// Execute step 1: issue a tool call, then finalize.
	agent.push(NewToolCallRequested([]ToolCall{{ID: "1", Name: "noop", Input: json.RawMessage(`{}`)}}, turnUsage()))
	agent.push(planFinal("did step one"))
	// Execute step 2: finalize directly.
	agent.push(planFinal("did step two"))

	cfg := standardCfg(agent)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}

	agent.mu.Lock()
	defer agent.mu.Unlock()
	// 1 plan turn + (tool call + final) for step one + 1 for step two.
	if len(agent.seen) != 4 {
		t.Fatalf("captured %d turns, want 4 (plan turn + 3 execute turns)", len(agent.seen))
	}

	// The PLAN turn (index 0) DOES carry the directive — that's correct.
	plan := contextText(agent.seen[0])
	if !strings.Contains(plan, producePlan) || !strings.Contains(plan, respondJSON) {
		t.Fatalf("plan turn should see the directive, context was:\n%s", plan)
	}

	// No EXECUTE-step context (indices 1..) may carry the directive.
	for i := 1; i < len(agent.seen); i++ {
		c := contextText(agent.seen[i])
		if strings.Contains(c, producePlan) || strings.Contains(c, respondJSON) {
			t.Fatalf("execute-step context %d leaked the directive:\n%s", i, c)
		}
	}

	// The execute steps still receive their step instructions and can proceed
	// to a tool call.
	if c := contextText(agent.seen[1]); !strings.Contains(c, "step one") {
		t.Fatalf("step-one context should carry its instruction:\n%s", c)
	}
	if c := contextText(agent.seen[3]); !strings.Contains(c, "step two") {
		t.Fatalf("step-two context should carry its instruction:\n%s", c)
	}
}
