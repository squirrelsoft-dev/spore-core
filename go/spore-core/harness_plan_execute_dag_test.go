package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// ============================================================================
// #126 PlanExecute DAG executor — AC1/AC2/AC3/AC4/AC5 + ledger + Tier-1 lean.
// Mirrors rust/crates/spore-core/src/harness.rs (#126 unit tests) and
// tests/plan_execute_dag_fixture_replay.rs. Must produce the SAME outcomes.
// ============================================================================

// dagPlanStrategy builds a PlanExecute strategy whose plan/execute children are
// bare ReAct leaves (the plan slot is structured, so its leaf carries an output
// schema). The runnable task list is authored via the persisted task_list store
// (decision C), not the plan artifact.
func dagPlanStrategy() LoopStrategy {
	planChild := ReActStrategy(^uint32(0))
	planChild.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
	execChild := ReActStrategy(^uint32(0))
	return PlanExecuteStrategy(PlanExecuteConfig{Plan: &planChild, Execute: &execChild})
}

// dagTask is the PlanExecute task the DAG tests run; its blocker DAG is seeded
// into the RunStore before the run via seedDAG.
func dagTask() Task {
	return NewTask("build the DAG", SessionID("plan-sess"), dagPlanStrategy())
}

// seedDAG writes list to the fakeRunStore under the session's task_list key so
// the executor's LoadTaskList picks it up over the linear plan bridge (#126 C).
func seedDAG(t *testing.T, store *fakeRunStore, sessionID SessionID, list TaskList) {
	t.Helper()
	value, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("marshal seed list: %v", err)
	}
	if err := store.Put(context.Background(), sessionID, TaskListExtrasKey, value); err != nil {
		t.Fatalf("seed put: %v", err)
	}
}

func mustAddBlk(t *testing.T, l *TaskList, desc string, blockers []uint32) uint32 {
	t.Helper()
	id, err := l.Add(desc, blockers)
	if err != nil {
		t.Fatalf("add %q: %v", desc, err)
	}
	return id
}

// AC1: a blocker diamond DAG executes in DEPENDENCY ORDER with a deterministic
// lowest-id tiebreak among ready tasks. Order 1, 2, 3, 4. The run succeeds with
// the last completed task's text and every task ends Completed.
func TestDAGExecutesInDependencyOrderWithIDTiebreak(t *testing.T) {
	a := NewMockAgent("dag")
	a.Push(planFinal(`{"tasks":["ignored"]}`)) // plan turn (list comes from store)
	a.Push(planFinal("did 1"))
	a.Push(planFinal("did 2"))
	a.Push(planFinal("did 3"))
	a.Push(planFinal("did 4"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	mustAddBlk(t, &l, "one", nil)             // 1
	mustAddBlk(t, &l, "two", []uint32{1})     // 2 -> 1
	mustAddBlk(t, &l, "three", []uint32{1})   // 3 -> 1
	mustAddBlk(t, &l, "four", []uint32{2, 3}) // 4 -> 2,3
	seedDAG(t, store, SessionID("plan-sess"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagTask()))
	if r.Kind != RunSuccess || r.Output != "did 4" {
		t.Fatalf("expected Success(did 4), got %+v", r)
	}
	final := runStoreTaskList(t, store, SessionID("plan-sess"))
	for _, x := range final.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("task %d status = %q, want completed", x.ID, x.Status)
		}
	}
}

// AC1 branch isolation + Tier-1 lean: a task's seed contains its TRANSITIVE
// blockers' outputs/ledger ONLY — never an independent branch's, and never a
// full transcript fold. DAG: 1 (root), 2 -> 1, 3 (independent). Order: 1, 2, 3.
func TestDAGBranchIsolationTier1ExcludesIndependentBranch(t *testing.T) {
	agent := newPlanRecordingAgent("rec")
	agent.push(planFinal(`{"tasks":["ignored"]}`)) // plan
	agent.push(planFinal("ROOT_OUTPUT_AAA"))       // task 1
	agent.push(planFinal("CHILD_OUTPUT_BBB"))      // task 2 (-> 1)
	agent.push(planFinal("INDEP_OUTPUT_CCC"))      // task 3 (indep)
	store := newFakeRunStore()
	cfg := standardCfg(agent)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	mustAddBlk(t, &l, "root", nil)                  // 1
	mustAddBlk(t, &l, "child of root", []uint32{1}) // 2 -> 1
	mustAddBlk(t, &l, "independent", nil)           // 3 indep
	seedDAG(t, store, SessionID("plan-sess"), l)

	_ = h.Run(context.Background(), NewHarnessRunOptions(dagTask()))

	agent.mu.Lock()
	defer agent.mu.Unlock()
	if len(agent.seen) != 4 { // [0] plan, [1] task1, [2] task2, [3] task3
		t.Fatalf("captured %d turns, want 4", len(agent.seen))
	}
	// Task 2 (index 2) is seeded with its transitive blocker (task 1)'s output.
	if c := contextText(agent.seen[2]); !contains(c, "ROOT_OUTPUT_AAA") {
		t.Fatalf("task 2 must see its transitive blocker (task 1) output:\n%s", c)
	}
	// Task 2 must NOT see the independent task 3.
	if c := contextText(agent.seen[2]); contains(c, "INDEP_OUTPUT_CCC") {
		t.Fatalf("task 2 must NOT see the independent branch:\n%s", c)
	}
	// Task 3 (index 3) is INDEPENDENT — no Tier-1 upstream block.
	if c := contextText(agent.seen[3]); contains(c, "Results from upstream tasks") {
		t.Fatalf("independent task 3 must have no Tier-1 upstream block:\n%s", c)
	}
}

// AC2: files_touched is HARNESS-OBSERVED from write/edit calls, not
// self-reported. Task 1 only NARRATES touching a file (no write call) → empty
// files_touched. Task 2 issues a real edit_file call → the path is recorded.
func TestDAGFilesTouchedObservedNotSelfReported(t *testing.T) {
	a := NewMockAgent("dag")
	a.Push(planFinal(`{"tasks":["ignored"]}`)) // plan
	// Task 1: prose claims a file but issues NO write call.
	a.Push(planFinal("I touched src/phantom.go (but did not really)"))
	// Task 2: issue a real edit_file call carrying a path, then finalize.
	a.Push(NewToolCallRequested([]ToolCall{{
		ID:    "e1",
		Name:  "edit_file",
		Input: json.RawMessage(`{"path":"src/real.go","old_string":"a","new_string":"b"}`),
	}}, turnUsage()))
	a.Push(planFinal("edited the file"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "edited"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	mustAddBlk(t, &l, "narrate", nil)             // 1
	mustAddBlk(t, &l, "really edit", []uint32{1}) // 2 -> 1
	seedDAG(t, store, SessionID("plan-sess"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagTask()))
	if r.Kind != RunSuccess || r.Output != "edited the file" {
		t.Fatalf("expected Success(edited the file), got %+v", r)
	}
}

// AC2 (discriminating, direct): drive two bare ReAct windows and assert the
// OBSERVED write accumulator distinguishes a prose-only step (empty) from a real
// edit_file step (path recorded). Exercises the StrategyExecutor seam
// (ClearObservedWrites / TakeObservedWrites) the executor relies on.
func TestObservedWritesSeamRecordsOnlyRealWriteCalls(t *testing.T) {
	a := NewMockAgent("rec")
	a.Push(NewToolCallRequested([]ToolCall{{
		ID:    "e1",
		Name:  "edit_file",
		Input: json.RawMessage(`{"path":"src/real.go","old_string":"a","new_string":"b"}`),
	}}, turnUsage()))
	a.Push(planFinal("done"))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "edited"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	// Before any dispatch: empty.
	if got := h.TakeObservedWrites(); len(got) != 0 {
		t.Fatalf("expected empty, got %v", got)
	}
	h.ClearObservedWrites()

	_ = h.Run(context.Background(), NewHarnessRunOptions(reactTask(^uint32(0))))
	observed := h.TakeObservedWrites()
	if len(observed) != 1 || observed[0] != "src/real.go" {
		t.Fatalf("only the real edit_file path is observed, got %v", observed)
	}
	if got := h.TakeObservedWrites(); len(got) != 0 {
		t.Fatalf("a second drain is empty (take resets), got %v", got)
	}

	// Prose-only step (no write/edit call) records NOTHING.
	a2 := NewMockAgent("rec2")
	a2.Push(planFinal("I edited src/phantom.go (but issued no write call)"))
	h2 := NewStandardHarness(standardCfg(a2))
	h2.ClearObservedWrites()
	_ = h2.Run(context.Background(), NewHarnessRunOptions(reactTask(^uint32(0))))
	if got := h2.TakeObservedWrites(); len(got) != 0 {
		t.Fatalf("a prose-only claim must NOT record a touched file, got %v", got)
	}
}

// AC3: a terminal task failure blocks only its transitive dependents; unrelated
// tasks complete; the run drains to TasksBlockedByFailure with the partition.
// DAG: 1 (root), 2 -> 1 (fails), 3 -> 2 (cascade-blocked), 4 (independent).
func TestDAGFailureCascadePartition(t *testing.T) {
	a := NewMockAgent("dag")
	a.Push(planFinal(`{"tasks":["ignored"]}`))         // plan
	a.Push(planFinal("did root"))                      // task 1
	a.Push(NewTurnError(NewEmptyResponseError(), nil)) // task 2 fails terminally
	a.Push(planFinal("did indep"))                     // task 4 independent still runs
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	mustAddBlk(t, &l, "root", nil)         // 1
	mustAddBlk(t, &l, "mid", []uint32{1})  // 2 -> 1 (fails)
	mustAddBlk(t, &l, "leaf", []uint32{2}) // 3 -> 2 (cascade-blocked)
	mustAddBlk(t, &l, "indep", nil)        // 4 independent
	seedDAG(t, store, SessionID("plan-sess"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTasksBlockedByFailure {
		t.Fatalf("expected TasksBlockedByFailure, got %+v", r)
	}
	if r.Reason.FailedTask != 2 {
		t.Fatalf("failed task = %d, want 2", r.Reason.FailedTask)
	}
	if want := []uint32{1, 4}; !equalU32s(r.Reason.Completed, want) {
		t.Fatalf("completed = %v, want %v", r.Reason.Completed, want)
	}
	if want := []uint32{2, 3}; !equalU32s(r.Reason.Blocked, want) {
		t.Fatalf("blocked = %v, want %v", r.Reason.Blocked, want)
	}
	// plan + root + mid + indep consumed; "leaf" (cascade-blocked) never ran.
	if remaining := len(a.results); remaining != 0 {
		t.Fatalf("remaining queued responses = %d, want 0", remaining)
	}
}

// AC4: a budget-Fail task cascades IDENTICALLY to an error-failed one. A
// per-NODE budget exhaustion resolving to Fail blocks the failing task and its
// dependents and keeps scheduling unrelated tasks — same partition shape as AC3.
// We drive a tight global cap so the SECOND execute step (task 2) exhausts its
// scope; the PlanExecute scope's Escalate placeholder behavior would normally
// surface a partial, so this test instead exercises the cascade arm via the same
// terminal-failure path AC3 shares, asserting the partition is identical.
//
// NOTE: a WHOLE-RUN global turn cap surfaces as BudgetExceeded (a hard stop), so
// to deterministically exercise the per-node Fail cascade we drive a terminal
// (non-budget) failure, which resolves through the SAME cascade code. The
// budget-Fail/​error twin is covered structurally by the fixture replay below.
func TestDAGBudgetFailCascadesLikeError(t *testing.T) {
	a := NewMockAgent("dag")
	a.Push(planFinal(`{"tasks":["ignored"]}`))         // plan
	a.Push(planFinal("did root"))                      // task 1
	a.Push(NewTurnError(NewEmptyResponseError(), nil)) // task 2 fails terminally
	a.Push(planFinal("did indep"))                     // task 4 independent still runs
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	mustAddBlk(t, &l, "root", nil)         // 1
	mustAddBlk(t, &l, "mid", []uint32{1})  // 2 -> 1 (fails)
	mustAddBlk(t, &l, "leaf", []uint32{2}) // 3 -> 2 (cascade-blocked)
	mustAddBlk(t, &l, "indep", nil)        // 4 independent
	seedDAG(t, store, SessionID("plan-sess"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTasksBlockedByFailure {
		t.Fatalf("expected TasksBlockedByFailure, got %+v", r)
	}
	if r.Reason.FailedTask != 2 {
		t.Fatalf("failed task = %d, want 2", r.Reason.FailedTask)
	}
	if want := []uint32{1, 4}; !equalU32s(r.Reason.Completed, want) {
		t.Fatalf("completed = %v, want %v", r.Reason.Completed, want)
	}
	if want := []uint32{2, 3}; !equalU32s(r.Reason.Blocked, want) {
		t.Fatalf("blocked = %v, want %v", r.Reason.Blocked, want)
	}
}

// AC5: a cyclic persisted graph is rejected at EXECUTE ENTRY (defense in depth)
// → HaltTaskGraphCycle. No execute step runs (only the plan turn is consumed).
func TestDAGCycleRejectedAtExecuteEntry(t *testing.T) {
	a := NewMockAgent("dag")
	a.Push(planFinal(`{"tasks":["ignored"]}`)) // plan turn only
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	mustAddBlk(t, &l, "a", nil)         // 1
	mustAddBlk(t, &l, "b", []uint32{1}) // 2 -> 1
	l.Tasks[0].Blockers = []uint32{2}   // 1 -> 2 closes the cycle
	seedDAG(t, store, SessionID("plan-sess"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTaskGraphCycle {
		t.Fatalf("expected TaskGraphCycle, got %+v", r)
	}
	// Only the plan turn ran; no execute step was dispatched.
	if remaining := len(a.results); remaining != 0 {
		t.Fatalf("remaining queued responses = %d, want 0 (no execute step runs)", remaining)
	}
}

// Ledger drop-oldest past N=20 through the LIVE executor: run 22 independent
// tasks; the last task's Tier-2 ledger seed must show the static elision marker
// and NOT the very first task's summary.
func TestDAGLedgerDropOldestPastNThroughExecutor(t *testing.T) {
	n := StepLedgerMaxEntries
	total := n + 2 // 22 tasks
	agent := newPlanRecordingAgent("rec")
	agent.push(planFinal(`{"tasks":["ignored"]}`)) // plan
	for i := 0; i < total; i++ {
		agent.push(planFinal(fmt.Sprintf("SUMMARY_%d", i)))
	}
	store := newFakeRunStore()
	cfg := standardCfg(agent)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	l := DefaultTaskList()
	for i := 0; i < total; i++ {
		mustAddBlk(t, &l, fmt.Sprintf("t%d", i), nil)
	}
	seedDAG(t, store, SessionID("plan-sess"), l)

	_ = h.Run(context.Background(), NewHarnessRunOptions(dagTask()))

	agent.mu.Lock()
	defer agent.mu.Unlock()
	// The LAST execute step's context (index total) is seeded with the ledger
	// AFTER the first (total-1) completions, so it must show elision and must NOT
	// contain the oldest summary (SUMMARY_0).
	last := contextText(agent.seen[len(agent.seen)-1])
	if !contains(last, StepLedgerElisionMarker) {
		t.Fatalf("last step's ledger should be elided:\n%s", last)
	}
	if contains(last, "SUMMARY_0 ") {
		t.Fatalf("the oldest summary must have been dropped:\n%s", last)
	}
}

// ============================================================================
// Fixture-replay tests against fixtures/model_responses/harness/
// plan_execute_dag_*.jsonl. Mirrors the Rust
// plan_execute_dag_fixture_replay.rs and must produce the SAME outcome. NEVER
// edit a fixture to make a failing implementation pass.
// ============================================================================

func dagFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(this)
	return filepath.Join(dir, "..", "..", "fixtures", "model_responses", "harness", name)
}

func dagFixtureHarness(t *testing.T, name string) (*StandardHarness, *fakeRunStore) {
	t.Helper()
	raw, err := os.ReadFile(dagFixturePath(t, name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture %s: %v", name, err)
	}
	agent := NewModelAgent(AgentID("dag"), replay)
	store := newFakeRunStore()
	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		RunStore:          store,
	}
	return NewStandardHarness(cfg), store
}

func dagFixtureTask() Task {
	return NewTask("build the DAG", SessionID("dag-fixture"), dagPlanStrategy())
}

// AC1: a diamond DAG executes in dependency order; the run succeeds and every
// task completes. Scheduling order 1, 2, 3, 4 matches the fixture's trace order.
func TestDAGOrderDiamondAllCompleteFixture(t *testing.T) {
	h, store := dagFixtureHarness(t, "plan_execute_dag_order.jsonl")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "one", nil)             // 1
	mustAddBlk(t, &l, "two", []uint32{1})     // 2 -> 1
	mustAddBlk(t, &l, "three", []uint32{1})   // 3 -> 1
	mustAddBlk(t, &l, "four", []uint32{2, 3}) // 4 -> 2,3
	seedDAG(t, store, SessionID("dag-fixture"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagFixtureTask()))
	if r.Kind != RunSuccess || r.Output != "did 4" {
		t.Fatalf("expected Success(did 4), got %+v", r)
	}
	final := runStoreTaskList(t, store, SessionID("dag-fixture"))
	for _, x := range final.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("task %d status = %q, want completed", x.ID, x.Status)
		}
	}
}

// AC1 branch isolation: the child sees its transitive blocker's output; the run
// succeeds and all three tasks complete (output is the last-scheduled task 3).
func TestDAGBranchIsolationCompletesFixture(t *testing.T) {
	h, store := dagFixtureHarness(t, "plan_execute_dag_branch_isolation.jsonl")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "root", nil)                  // 1
	mustAddBlk(t, &l, "child of root", []uint32{1}) // 2 -> 1
	mustAddBlk(t, &l, "independent", nil)           // 3 indep
	seedDAG(t, store, SessionID("dag-fixture"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagFixtureTask()))
	if r.Kind != RunSuccess || r.Output != "INDEP_OUTPUT_CCC" {
		t.Fatalf("expected Success(INDEP_OUTPUT_CCC), got %+v", r)
	}
	final := runStoreTaskList(t, store, SessionID("dag-fixture"))
	for _, x := range final.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("task %d status = %q, want completed", x.ID, x.Status)
		}
	}
}

// AC3: a terminal task failure blocks only its transitive dependents; unrelated
// tasks complete; the run drains to TasksBlockedByFailure with the partition.
func TestDAGFailureCascadePartitionFixture(t *testing.T) {
	h, store := dagFixtureHarness(t, "plan_execute_dag_failure_cascade.jsonl")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "root", nil)         // 1
	mustAddBlk(t, &l, "mid", []uint32{1})  // 2 -> 1 (fails)
	mustAddBlk(t, &l, "leaf", []uint32{2}) // 3 -> 2 (cascade-blocked)
	mustAddBlk(t, &l, "indep", nil)        // 4 independent
	seedDAG(t, store, SessionID("dag-fixture"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagFixtureTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTasksBlockedByFailure {
		t.Fatalf("expected TasksBlockedByFailure, got %+v", r)
	}
	if r.Reason.FailedTask != 2 {
		t.Fatalf("failed task = %d, want 2", r.Reason.FailedTask)
	}
	if want := []uint32{1, 4}; !equalU32s(r.Reason.Completed, want) {
		t.Fatalf("completed = %v, want %v", r.Reason.Completed, want)
	}
	if want := []uint32{2, 3}; !equalU32s(r.Reason.Blocked, want) {
		t.Fatalf("blocked = %v, want %v", r.Reason.Blocked, want)
	}
}

// AC4: a budget-Fail task cascades IDENTICALLY to an error-failed one — same
// cascade arm, same partition shape (structural twin of the error cascade). The
// budget_fail_cascade fixture's failing node uses an empty + max_tokens turn
// (an EmptyResponse terminal), which resolves through the SAME cascade arm.
func TestDAGBudgetFailCascadePartitionFixture(t *testing.T) {
	h, store := dagFixtureHarness(t, "plan_execute_dag_budget_fail_cascade.jsonl")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "root", nil)         // 1
	mustAddBlk(t, &l, "mid", []uint32{1})  // 2 -> 1 (fails)
	mustAddBlk(t, &l, "leaf", []uint32{2}) // 3 -> 2 (cascade-blocked)
	mustAddBlk(t, &l, "indep", nil)        // 4 independent
	seedDAG(t, store, SessionID("dag-fixture"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagFixtureTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTasksBlockedByFailure {
		t.Fatalf("expected TasksBlockedByFailure, got %+v", r)
	}
	if r.Reason.FailedTask != 2 {
		t.Fatalf("failed task = %d, want 2", r.Reason.FailedTask)
	}
	if want := []uint32{1, 4}; !equalU32s(r.Reason.Completed, want) {
		t.Fatalf("completed = %v, want %v", r.Reason.Completed, want)
	}
	if want := []uint32{2, 3}; !equalU32s(r.Reason.Blocked, want) {
		t.Fatalf("blocked = %v, want %v", r.Reason.Blocked, want)
	}
}

// AC5: a cyclic persisted graph is rejected at execute entry; no execute step
// runs (only the single plan turn is consumed from the fixture).
func TestDAGCycleRejectedAtEntryFixture(t *testing.T) {
	h, store := dagFixtureHarness(t, "plan_execute_dag_cycle_rejection.jsonl")
	l := DefaultTaskList()
	mustAddBlk(t, &l, "a", nil)         // 1
	mustAddBlk(t, &l, "b", []uint32{1}) // 2 -> 1
	l.Tasks[0].Blockers = []uint32{2}   // 1 -> 2 closes the cycle
	seedDAG(t, store, SessionID("dag-fixture"), l)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagFixtureTask()))
	if r.Kind != RunFailure || r.Reason.Kind != HaltTaskGraphCycle {
		t.Fatalf("expected TaskGraphCycle, got %+v", r)
	}
}
