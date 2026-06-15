package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

// ============================================================================
// #138 resume seeding — seed the stalled worker + skip re-planning. Mirrors the
// Rust tests in cordyceps_composition_fixture_replay.rs (#138 additions) and the
// harness.rs inline #138 unit tests. The shared fixtures
// (cordyceps_budget_resume.jsonl, cordyceps_budget_exhausted.json) are ground
// truth — never edit a fixture to make a failing implementation pass.
// ============================================================================

// smallBudgetPE is a PlanExecute[ ReAct(plan), SelfVerifying[ ReAct(PerLoop{2}) ] ]
// tree whose worker leaf exhausts after exactly TWO turns — so a budget pause is
// reachable with a tiny fixture. Mirrors the cordyceps execute leaf's handles
// (executor / exec-tools / worker-schema / exec-evaluator).
func smallBudgetPE() LoopStrategy {
	workerSchema := SchemaRef("worker-schema")
	worker := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		Agent:    AgentRef("executor"),
		Toolset:  ToolsetRef("exec-tools"),
		Output:   &workerSchema,
	}}
	planSchema := SchemaRef("plan-schema")
	plan := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: 12},
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		Agent:    AgentRef("planner"),
		Toolset:  ToolsetRef("plan-tools"),
		Output:   &planSchema,
	}}
	return PlanExecuteStrategy(PlanExecuteConfig{
		Plan: &plan,
		Execute: PtrStrategy(SelfVerifyingStrategy(SelfVerifyingConfig{
			Inner:     &worker,
			Evaluator: SchemaRef("exec-evaluator"),
			Behavior:  BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		})),
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
	})
}

// surfaceBudgetHarness builds a SurfaceToHuman harness whose plan/worker/evaluate
// turns replay positionally from ONE shared replay backend plus a
// ScriptedToolRegistry that returns success for the worker's two budget-burning
// tool calls. Mirrors Rust's surface_harness_for.
func surfaceBudgetHarness(t *testing.T, fixture string) (*StandardHarness, *fakeRunStore) {
	t.Helper()
	raw, err := os.ReadFile(compFixturePath(t, fixture))
	if err != nil {
		t.Fatalf("read fixture %s: %v", fixture, err)
	}
	replay, err := ParseReplayJSONL(string(raw), ProviderInfo{
		Name: "anthropic", ModelID: "fixture", ContextWindow: 200_000,
	})
	if err != nil {
		t.Fatalf("parse fixture %s: %v", fixture, err)
	}
	agent := func(id string) Agent { return NewModelAgent(AgentID(id), replay) }
	// The worker's two tool calls each dispatch to a plain success (content is
	// irrelevant; they only burn the PerLoop{2} budget).
	toolReg := NewScriptedToolRegistry()
	toolReg.Push(NewToolOutputSuccess("src/one.rs\nsrc/two.rs"))
	toolReg.Push(NewToolOutputSuccess("fn one() { x.unwrap() }"))
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
	store := newFakeRunStore()
	cfg := HarnessConfig{
		ToolRegistry:      toolReg,
		Sandbox:           AllowAllSandbox{},
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		Registry:          reg,
		RunStore:          store,
		ProjectNamespace:  compProjectNS,
		// #138/#130: the worker leaf's budget exhaustion PAUSES (WaitingForHuman).
		EscalationMode: SurfaceToHumanEscalation(),
	}
	return NewStandardHarness(cfg), store
}

func smallPETask(session string) Task {
	t := NewTask("audit the repo", SessionID(session), smallBudgetPE())
	cap := uint32(64)
	t.Budget.MaxTurns = &cap
	return t
}

// #138 AC2 + AC1: a budget-resume of an execute-phase exhaustion SEEDS the
// stalled worker (carries its full session across the pause) and SKIPS
// re-planning. Leg 1 drives the worker leaf to its PerLoop{2} cap and PAUSES with
// a BudgetExhausted request whose PausedState carries the FULL worker session
// (AC2-a) and the exec-tools handle (AC4-a). Leg 2 (ContinueWithBudget) does NOT
// re-plan (the fixture has NO plan turn) and re-attaches the carried session so
// the worker continues mid-loop to a finding that the evaluator clears.
func TestCordycepsBudgetResumeSeedsStalledWorkerAndSkipsReplanning(t *testing.T) {
	h, store := surfaceBudgetHarness(t, "cordyceps_budget_resume.jsonl")
	session := SessionID("cordyceps-budget")
	// Pre-seed ONE ready task so AC1's skip-replan precondition holds (non-empty
	// durable list) and the execute phase runs exactly one worker.
	l := DefaultTaskList()
	mustAddBlk(t, &l, "audit module one", nil)
	compSeed(t, store, session, l)

	// Leg 1: drive to the budget-exhaustion pause.
	first := h.Run(context.Background(), NewHarnessRunOptions(smallPETask("cordyceps-budget")))
	if first.Kind != RunWaitingForHuman {
		t.Fatalf("expected WaitingForHuman budget pause, got %+v", first)
	}
	if first.State == nil || first.Request == nil {
		t.Fatal("budget pause must carry a PausedState + request")
	}
	if first.Request.Kind != HumanReqBudgetExhausted {
		t.Fatalf("expected BudgetExhausted request, got %q", first.Request.Kind)
	}
	// The combinator (PlanExecute) resolves the worker leaf's propagated
	// exhaustion, so the pause's phase is the resolving scope.
	if first.Request.Phase != "plan_execute" {
		t.Fatalf("phase = %q, want plan_execute (the combinator resolved the exhaustion)", first.Request.Phase)
	}
	// AC4-a (#140 parity): the pause carries the worker leaf's toolset handle.
	if first.State.Toolset != ToolsetRef("exec-tools") {
		t.Fatalf("AC4-a: budget pause toolset = %q, want exec-tools", first.State.Toolset)
	}
	// AC2-a: the pause carries the FULL worker session (instruction + the two
	// budget-burning tool-call rounds), NOT a partial-only stub.
	if len(first.State.SessionState.Messages) <= 1 {
		t.Fatalf("AC2-a: full worker session carried, got %d messages", len(first.State.SessionState.Messages))
	}
	// AC2 parity: the stalled task stays InProgress on the durable list at the
	// pause (the consult path's invariant) — NOT permanently Blocked — so the
	// resume can re-attach the carried session via InProgress->Pending->complete.
	paused := runStoreTaskList(t, store, compProjectNS)
	if paused.Tasks[0].Status != TaskStatusInProgress {
		t.Fatalf("the stalled task awaits a budget grant (InProgress, not Blocked), got %q", paused.Tasks[0].Status)
	}

	// Leg 2: grant more budget and resume. AC1: NO plan turn in the fixture, so a
	// re-plan would exhaust the positional replay and error — Success proves the
	// plan phase was skipped. AC2-b: the carried session re-attaches to the
	// InProgress task, so the worker continues to its finding and self-verifies.
	resumed := h.Resume(
		context.Background(),
		*first.State,
		EscalateResponse(ContinueWithBudgetAction(5)),
		nil,
	)
	if resumed.Kind != RunSuccess {
		t.Fatalf("expected Success after budget resume, got %+v", resumed)
	}
	if !contains(resumed.Output, "resume-continued") {
		t.Fatalf("run output must be the post-resume worker finding, got %q", resumed.Output)
	}

	// The resumed task self-verified and completed (InProgress->Pending->Completed
	// — the same transition machinery the consult path uses, AC2 parity).
	after := runStoreTaskList(t, store, compProjectNS)
	for _, x := range after.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("the resumed task completed; task %d = %q", x.ID, x.Status)
		}
	}
}

// #144 (Rust cfffa40 parity): a PlanExecute execute-phase worker that exhausts
// under SurfaceToHuman and is granted more budget must make FORWARD PROGRESS
// across grants — the run-wide turn cursor AND the grant's StepsTaken both
// STRICTLY advance, and the worker eventually finishes. Before the fix the
// exhaustion branch discarded the execute leaf's consumed-turn count and re-read
// StepsTaken=0 from the (Unlimited, because MaxTurns==nil) plan_execute scope,
// and left carried.Turns frozen at its pre-task value. So every grant computed
// granted = 0 + steps (a no-op that never widened the binding cap) and re-seeded
// the worker on the SAME window — the cursor and StepsTaken stuck, the worker
// re-ran the same turns, and a review-style task thrashed through every
// auto-continue making zero progress. Passes with the fix, fails without it.
//
// MUST wire a real in-memory store (not the no-op default): the durable
// task-list path the resume re-attaches through no-ops under the default store,
// which would let this pass while asserting the wrong path.
func TestExecutePhaseBudgetGrantAdvancesAndMakesProgress(t *testing.T) {
	// One execute task ("only"); the execute leaf cap is PerLoop{2} and the top
	// task leaves MaxTurns == nil, so the plan_execute scope is Unlimited — the
	// exact configuration that zeroed the grant accounting.
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["only"]}`)) // plan turn (carried.Turns -> 1)
	// Execute worker keeps requesting tools so it never finishes until the final
	// grant; each ToolCallRequested is one turn against the leaf cap.
	for i := 0; i < 3; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{
			ID:    "e" + string(rune('0'+i)),
			Name:  "x",
			Input: json.RawMessage(`{}`),
		}}, turnUsage()))
	}
	a.Push(planFinal("done only")) // completes after the 2nd grant

	// A real in-memory store + fixed project namespace so the durable task-list
	// re-attach path is exercised (not the no-op default RunStore).
	store := newFakeRunStore()
	cfg := surfaceCfg(a)
	cfg.RunStore = store
	cfg.ProjectNamespace = compProjectNS
	// tool_reg(3): three success outputs for the worker's three tool-burning turns.
	reg := NewScriptedToolRegistry()
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	// react_structured(MAX) plan slot + ReAct(PerLoop{2}) execute leaf, Escalate.
	plan := ReActStrategy(^uint32(0))
	plan.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
	exec := ReActStrategy(2)
	pe := PlanExecuteStrategy(PlanExecuteConfig{
		Plan:     &plan,
		Execute:  &exec,
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
	})
	tk := NewTask("do something", SessionID("s1"), pe)
	// MaxTurns intentionally UNSET -> the plan_execute scope is Unlimited.

	// First run -> pause #1. The worker consumed real turns past the plan floor,
	// so the cursor and StepsTaken reflect that (NOT 0 / frozen at 1).
	first := h.Run(context.Background(), NewHarnessRunOptions(tk))
	if first.Kind != RunWaitingForHuman {
		t.Fatalf("expected pause #1 (WaitingForHuman), got %+v", first)
	}
	if first.State == nil || first.Request == nil {
		t.Fatal("budget pause must carry a PausedState + request")
	}
	if first.Request.Kind != HumanReqBudgetExhausted {
		t.Fatalf("expected BudgetExhausted request, got %q", first.Request.Kind)
	}
	// The combinator scope owns the pause — phase must stay plan_execute so the
	// resume-seed ContinuesUsed restore matches.
	if first.Request.Phase != "plan_execute" {
		t.Fatalf("phase = %q, want plan_execute (the combinator scope owns the pause)", first.Request.Phase)
	}
	turn1 := first.State.TurnNumber
	steps1 := first.Request.StepsTaken
	// The bug froze the cursor at the pre-task value (1 = just the plan turn) and
	// read steps_taken=0 from the Unlimited scope.
	if turn1 < 2 {
		t.Fatalf("cursor advanced past the plan floor, got %d", turn1)
	}
	if steps1 < 2 {
		t.Fatalf("grant accounting uses the worker's consumed turns, got %d", steps1)
	}

	// Grant +2 and resume -> pause #2. Forward progress: the cursor and StepsTaken
	// must STRICTLY advance (the bug re-seeded the same window).
	second := h.Resume(
		context.Background(),
		*first.State,
		EscalateResponse(ContinueWithBudgetAction(2)),
		nil,
	)
	if second.Kind != RunWaitingForHuman {
		t.Fatalf("expected pause #2 (WaitingForHuman), got %+v", second)
	}
	if second.State == nil || second.Request == nil || second.Request.Kind != HumanReqBudgetExhausted {
		t.Fatalf("expected BudgetExhausted pause #2, got %+v", second)
	}
	turn2 := second.State.TurnNumber
	steps2 := second.Request.StepsTaken
	if turn2 <= turn1 {
		t.Fatalf("turn cursor must advance across grants (%d -> %d)", turn1, turn2)
	}
	if steps2 <= steps1 {
		t.Fatalf("steps_taken must advance across grants (%d -> %d)", steps1, steps2)
	}

	// Grant +2 again and resume -> Success: the worker finishes, having made real
	// progress rather than looping on the same window.
	third := h.Resume(
		context.Background(),
		*second.State,
		EscalateResponse(ContinueWithBudgetAction(2)),
		nil,
	)
	if third.Kind != RunSuccess {
		t.Fatalf("expected Success after the final grant, got %+v", third)
	}
	if third.Output != "done only" {
		t.Fatalf("output = %q, want %q", third.Output, "done only")
	}
}

// #138 AC4: the budget-exhausted PausedState fixture round-trips byte-identically
// — the carried worker session (AC2-a) and the exec-tools handle (AC4-a) survive
// a serde round-trip identically. This is the cross-language wire-parity lock.
// jsonEqual is SEMANTIC JSON equality: the ONLY divergence from the literal
// fixture bytes is the cross-language float representation of cost_usd (Go emits
// 0, the fixture writes 0.0) — the same established convention TestFixturePausedState
// uses. Field order, key set, and every value otherwise match exactly.
func TestBudgetExhaustedPausedStateRoundTrips(t *testing.T) {
	raw, err := os.ReadFile(compPausedStatePath(t, "cordyceps_budget_exhausted.json"))
	if err != nil {
		t.Fatalf("read cordyceps_budget_exhausted.json: %v", err)
	}

	var typed PausedState
	if err := json.Unmarshal(raw, &typed); err != nil {
		t.Fatalf("PausedState deserializes: %v", err)
	}
	reser, err := json.Marshal(typed)
	if err != nil {
		t.Fatalf("PausedState re-serializes: %v", err)
	}
	if !jsonEqual(t, reser, raw) {
		t.Fatalf("PausedState round-trips byte-structurally\n got: %s\nwant: %s", reser, raw)
	}

	// AC4-a: the toolset handle is the worker leaf's, always serialized.
	if typed.Toolset != ToolsetRef("exec-tools") {
		t.Fatalf("AC4-a: toolset = %q, want exec-tools", typed.Toolset)
	}
	// AC2-a: the carried session grew beyond the single partial-only stub.
	if len(typed.SessionState.Messages) <= 1 {
		t.Fatalf("AC2-a: budget-exhausted session carries the worker conversation, got %d messages",
			len(typed.SessionState.Messages))
	}
}

// #138 AC3: plan-phase exhaustion resumes the PLAN session. When a budget resume
// carries a worker session AND the durable task_list is EMPTY (no InProgress task
// ⇒ the exhaustion happened in the PLAN phase), PlanExecuteConfig.Run seeds the
// PLAN session from the carried conversation instead of a fresh base session — so
// the planner CONTINUES on it. Observed via the planner agent's RECORDED contexts.
func TestBudgetResumePlanPhaseSeedsPlanSessionFromCarried(t *testing.T) {
	// The planner records every context it sees; it authors a one-task plan so the
	// run can complete, then the worker + a clean final response run.
	planner := newPlanRecordingAgent("planner")
	planner.push(planFinal(`{"tasks":["only"],"rationale":"r"}`))
	worker := NewMockAgent("worker")
	worker.Push(planFinal("did the work"))

	store := newFakeRunStore()
	cfg := surfaceCfg(worker)
	cfg.RunStore = store
	cfg.ProjectNamespace = compProjectNS
	// Wire planner under "planner", worker under the default empty key (the bare
	// PlanExecute execute leaf carries an empty AgentRef).
	cfg.Registry = NewExecutionRegistryBuilder().
		Agent("planner", planner).
		Agent("", worker).
		Build()
	h := NewStandardHarness(cfg)

	// A PlanExecute whose PLAN leaf resolves to "planner"; execute is a bare ReAct
	// on the default key.
	planSchema := SchemaRef("")
	pe := PlanExecuteStrategy(PlanExecuteConfig{
		Plan: PtrStrategy(LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
			Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: 12},
			Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
			Agent:    AgentRef("planner"),
			Toolset:  ToolsetRef(""),
			Output:   &planSchema,
		}}),
		Execute:  PtrStrategy(ReActStrategy(8)),
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorEscalate},
	})
	task := NewTask("audit the repo", SessionID("s1"), pe)
	cap := uint32(32)
	task.Budget.MaxTurns = &cap

	// A budget-exhausted pause carrying a worker session with a MARKER, and NO
	// durable task_list persisted (empty ⇒ plan-phase exhaustion, AC3).
	const marker = "CARRIED_PLAN_SESSION_MARKER"
	carried := SessionState{Messages: []Message{{
		Role:    RoleAssistant,
		Content: NewTextContent(marker),
	}}}
	partial := reactPartialJSON("")
	state := PausedState{
		SessionID:    SessionID("s1"),
		TaskID:       task.ID,
		TurnNumber:   1,
		SessionState: carried,
		HumanRequest: &HumanRequest{
			Kind:          HumanReqBudgetExhausted,
			Phase:         "plan_execute",
			Policy:        BudgetPolicy{Kind: BudgetTotalSteps, Value: 1},
			StepsTaken:    1,
			ContinuesUsed: 0,
			PartialOutput: &partial,
			AvailableActions: []EscalationAction{
				ContinueWithBudgetAction(1), SkipAction(), FailAction(),
			},
		},
		Task:       task,
		BudgetUsed: BudgetSnapshot{},
		Toolset:    ToolsetRef(""),
	}

	_ = h.Resume(
		context.Background(),
		state,
		EscalateResponse(ContinueWithBudgetAction(10)),
		nil,
	)

	// AC3: the planner's FIRST context was seeded from the CARRIED session — the
	// marker is present, proving the plan session continued on it rather than
	// starting from a fresh base session.
	planner.mu.Lock()
	defer planner.mu.Unlock()
	if len(planner.seen) == 0 {
		t.Fatal("AC3: the planner must have run at least once")
	}
	if c := contextText(planner.seen[0]); !contains(c, marker) {
		t.Fatalf("AC3: the plan session must be seeded from the carried conversation:\n%s", c)
	}
}

// #138 AC1: skip-plan reconciles already-Completed tasks (dedup). A non-empty
// durable task_list whose task #1 is already Completed: a fresh run SKIPS the plan
// phase (AC1) and reconcile does NOT re-run the completed task — only the
// still-Pending task #2 runs (one model call, no plan turn).
func TestSkipPlanReconcilesCompletedTasks(t *testing.T) {
	a := NewMockAgent("dag")
	// NO plan turn pushed (AC1 skips it). Only task #2 runs.
	a.Push(planFinal("did two"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	cfg.ProjectNamespace = dagProjectNS
	h := NewStandardHarness(cfg)

	// Pre-seed: task #1 already Completed, task #2 Pending.
	tl := DefaultTaskList()
	mustAddBlk(t, &tl, "one", nil) // 1
	mustAddBlk(t, &tl, "two", nil) // 2
	ip := TaskStatusInProgress
	_ = tl.Update(1, &ip, nil)
	_ = tl.Complete(1)
	seedDAG(t, store, SessionID("plan-sess"), tl)

	r := h.Run(context.Background(), NewHarnessRunOptions(dagTask()))
	if r.Kind != RunSuccess || r.Output != "did two" {
		t.Fatalf("expected Success(did two), got %+v", r)
	}
	// Exactly ONE model call remains consumed: task #2 (no plan turn, task #1 not
	// re-run) — the queue is fully drained.
	if remaining := len(a.results); remaining != 0 {
		t.Fatalf("AC1: plan skipped + completed task #1 deduped — only task #2 should run, %d responses left", remaining)
	}
	// Both tasks are Completed in the durable store (1 deduped, 2 freshly run).
	stored := runStoreTaskList(t, store, dagProjectNS)
	for _, x := range stored.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("both tasks Completed (1 deduped, 2 run); task %d = %q", x.ID, x.Status)
		}
	}
}

// #138 AC2-a: promoteBudgetExhaustedToHuman carries the FULL stalled worker
// session (AC2-a) and the worker leaf's toolset handle (AC4-a) — not a
// partial-only stub. A direct unit on the boundary helper, decoupled from the
// surrounding strategy.
func TestPromoteBudgetPauseCarriesFullWorkerSessionAndHandle(t *testing.T) {
	err := &BudgetExhausted{
		Policy:        BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		Behavior:      BudgetExhaustedBehavior{Kind: BehaviorEscalate},
		StepsTaken:    2,
		ContinuesUsed: 0,
		Phase:         "react",
	}
	leafTask := NewTask("audit", SessionID("s1"), ReActStrategy(2))
	// A realistic worker conversation (instruction + a tool round).
	worker := SessionState{Messages: []Message{
		{Role: RoleUser, Content: NewTextContent("worker: audit")},
		{Role: RoleAssistant, Content: NewTextContent("looking")},
		{Role: RoleTool, Content: NewTextContent("listing")},
	}}
	partial := "partial"
	waiting := promoteBudgetExhaustedToHuman(
		err, &partial, leafEscalationActions(err),
		SessionID("s1"), leafTask, BudgetSnapshot{}, 2,
		worker, ToolsetRef("exec-tools"),
	)
	if waiting.Kind != RunWaitingForHuman || waiting.State == nil {
		t.Fatalf("expected WaitingForHuman, got %+v", waiting)
	}
	// AC2-a: the FULL worker session is carried (3 messages), NOT the single
	// partial-only assistant stub.
	if len(waiting.State.SessionState.Messages) != 3 {
		t.Fatalf("AC2-a: full worker session carried, got %d messages", len(waiting.State.SessionState.Messages))
	}
	// AC4-a: the worker leaf's toolset handle rides the pause (#140 parity).
	if waiting.State.Toolset != ToolsetRef("exec-tools") {
		t.Fatalf("AC4-a: toolset = %q, want exec-tools", waiting.State.Toolset)
	}

	// Back-compat: an EMPTY worker session falls back to the partial-only stub
	// (the pre-#138 behavior) so legacy/HillClimbing sites are unchanged.
	partial2 := "just-the-partial"
	waiting2 := promoteBudgetExhaustedToHuman(
		err, &partial2, leafEscalationActions(err),
		SessionID("s1"), leafTask, BudgetSnapshot{}, 2,
		SessionState{}, ToolsetRef(""),
	)
	if waiting2.State == nil || len(waiting2.State.SessionState.Messages) != 1 {
		t.Fatalf("back-compat: empty worker session falls back to the single partial stub, got %+v", waiting2.State)
	}
	if waiting2.State.SessionState.Messages[0].Content.Text != "just-the-partial" {
		t.Fatalf("back-compat stub content = %q, want just-the-partial", waiting2.State.SessionState.Messages[0].Content.Text)
	}
}
