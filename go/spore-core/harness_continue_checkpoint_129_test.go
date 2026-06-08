package sporecore

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// ============================================================================
// #129 — Continue cross-process checkpoint (continues_used survives a pause)
//
// Mirrors rust/crates/spore-core/src/harness.rs (#129 unit tests) and the shared
// fixtures fixtures/paused_states/continue_checkpoint.json +
// fixtures/strategy/*.json. Same types, same rules, byte-identical wire format.
// ============================================================================

func pausedStatesFixturePath(t *testing.T, name string) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "paused_states", name)
}

// continueCheckpointPausedState is the canonical continue_checkpoint.json value:
// a bare ReAct leaf carrying Continue{MaxContinues:2, OnExhausted:Fail} paused
// mid-loop with ContinuesUsed: 1 on its HumanRequest BudgetExhausted.
// Cross-language byte-identity ground truth for the AC2 resume test.
func continueCheckpointPausedState() PausedState {
	partial := `{"node":"react","last":""}`
	leaf := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget: BudgetPolicy{Kind: BudgetPerLoop, Value: 3},
		Behavior: BudgetExhaustedBehavior{
			Kind:         BehaviorContinue,
			MaxContinues: 2,
			OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
		},
		Agent:   AgentRef("worker"),
		Toolset: ToolsetRef("patch-tools"),
	}}
	task := NewTask("iterate on the patch", SessionID("sess-129"), leaf)
	task.ID = TaskID("task-129")
	return PausedState{
		SessionID:    SessionID("sess-129"),
		TaskID:       TaskID("task-129"),
		TurnNumber:   3,
		SessionState: SessionState{Messages: []Message{{Role: RoleAssistant, Content: NewTextContent(partial)}}},
		HumanRequest: &HumanRequest{
			Kind:          HumanReqBudgetExhausted,
			Phase:         "react",
			Policy:        BudgetPolicy{Kind: BudgetPerLoop, Value: 3},
			StepsTaken:    3,
			ContinuesUsed: 1,
			PartialOutput: &partial,
			AvailableActions: []EscalationAction{
				ContinueWithBudgetAction(3),
				FailAction(),
			},
		},
		Task:       task,
		BudgetUsed: BudgetSnapshot{Turns: 3},
	}
}

// AC2 (wire): continue_checkpoint.json (de)serializes byte-identically — the new
// fixture capturing a Continue node paused with ContinuesUsed>0.
func TestFixtureReplayContinueCheckpoint(t *testing.T) {
	raw, err := os.ReadFile(pausedStatesFixturePath(t, "continue_checkpoint.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var parsed PausedState
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	// Field-for-field assertions against the fixture ground truth.
	if parsed.SessionID != "sess-129" || parsed.TaskID != "task-129" || parsed.TurnNumber != 3 {
		t.Fatalf("envelope mismatch: %+v", parsed)
	}
	hr := parsed.HumanRequest
	if hr == nil || hr.Kind != HumanReqBudgetExhausted || hr.Phase != "react" ||
		hr.StepsTaken != 3 || hr.ContinuesUsed != 1 {
		t.Fatalf("human_request mismatch: %+v", hr)
	}
	leaf := parsed.Task.LoopStrategy.ReActCfg
	if leaf == nil || leaf.Behavior.Kind != BehaviorContinue || leaf.Behavior.MaxContinues != 2 ||
		leaf.Behavior.OnExhausted == nil || leaf.Behavior.OnExhausted.Kind != BehaviorFail {
		t.Fatalf("leaf behavior mismatch: %+v", leaf)
	}

	// Re-serialize and assert byte-identity. jsonEqual is the established
	// cross-language convention (normalizes the cost_usd 0 vs 0.0 float repr).
	out, err := json.Marshal(parsed)
	if err != nil {
		t.Fatalf("re-serialize: %v", err)
	}
	if !jsonEqual(t, out, raw) {
		var compact bytes.Buffer
		_ = json.Compact(&compact, raw)
		t.Fatalf("re-serialized form is NOT structurally identical:\n got: %s\nwant: %s", out, compact.Bytes())
	}

	// The hand-built value also matches the fixture (byte-for-byte modulo float).
	built, err := json.Marshal(continueCheckpointPausedState())
	if err != nil {
		t.Fatalf("marshal built: %v", err)
	}
	if !jsonEqual(t, built, raw) {
		var compact bytes.Buffer
		_ = json.Compact(&compact, raw)
		t.Fatalf("hand-built value NOT identical to fixture:\n got: %s\nwant: %s", built, compact.Bytes())
	}
}

// budgetExhaustingAgent queues n tool-call requests so the worker NEVER finishes
// (every granted window re-exhausts its refreshed cap).
func budgetExhaustingAgent(n int) *MockAgent {
	a := NewMockAgent("leaf")
	for i := 0; i < n; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	}
	return a
}

// toolReg returns a scripted tool registry that answers n calls with "ok".
func toolReg(n int) *ScriptedToolRegistry {
	reg := NewScriptedToolRegistry()
	for i := 0; i < n; i++ {
		reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	}
	return reg
}

// continueLeaf builds a bare ReAct leaf (default "" registry handle) with the
// given Continue behavior and PerLoop cap.
func continueLeaf(cap, maxContinues uint32, onExhausted BudgetExhaustedBehaviorKind) LoopStrategy {
	return LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget: BudgetPolicy{Kind: BudgetPerLoop, Value: cap},
		Behavior: BudgetExhaustedBehavior{
			Kind:         BehaviorContinue,
			MaxContinues: maxContinues,
			OnExhausted:  &BudgetExhaustedBehavior{Kind: onExhausted},
		},
		Agent:   AgentRef(""),
		Toolset: ToolsetRef(""),
	}}
}

// AC2 (LOAD-BEARING, end-to-end): a Continue that SPANS a process pause resumes
// with the correct ContinuesUsed (NOT 0). FAILS on pre-#129 code, which dropped
// continues_used in resumeInner and rebuilt the scope from steps_taken only —
// ZEROING the counter and granting MORE in-process continues than MaxContinues.
//
// DISCRIMINATING setup: Continue{MaxContinues:2, OnExhausted:Fail} leaf whose
// checkpoint records ContinuesUsed: 1 (ONE continue already spent). The resumed
// worker NEVER finishes, so every granted window re-exhausts and the run drains
// to Fail. The observable signal is HOW MANY windows ran, captured as turns:
//   - CORRECT (#129): ContinuesUsed==1 → ONE continue left → window A (operator
//     grant) + window B → 2 windows × cap 4 = 8 turns, then Fail.
//   - BUGGY (pre-#129): ContinuesUsed==0 → TWO continues → 3 windows = 12 turns.
func TestResumeContinuePreservesContinuesUsedThenFallsThrough(t *testing.T) {
	a := budgetExhaustingAgent(40)
	cfg := surfaceCfg(a)
	cfg.ToolRegistry = toolReg(40)
	h := NewStandardHarness(cfg)

	state := continueCheckpointPausedState()
	// Bare leaf resolving via the default ("") registry handle on resume.
	state.Task.LoopStrategy = continueLeaf(3, 2, BehaviorFail)
	// ContinuesUsed == 1: ONE continue already spent (only one remains).
	state.HumanRequest.ContinuesUsed = 1
	state.HumanRequest.StepsTaken = 3

	r := h.Resume(context.Background(), state, EscalateResponse(ContinueWithBudgetAction(1)), nil)
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("expected BudgetExceeded Failure, got %+v", r)
	}
	// window A (operator grant) + window B (the ONE remaining in-process
	// continue) → 2 windows × cap 4 = 8 turns. The bug (zeroed continues_used)
	// grants a THIRD window → 12 turns.
	if r.Turns != 8 {
		t.Fatalf("expected 2 windows (continues_used preserved at 1, one continue left) → 8 turns; "+
			"the bug zeroes it and grants an extra window → 12 turns; got %d", r.Turns)
	}
}

// AC2 (unit, load-bearing core): the resume seam seeds the reconstructed scope's
// ContinuesUsed from the request via ResumedBudgetContext (NOT 0), and a
// subsequent exhaustion falls through to OnExhausted after the REMAINING
// continues — not a refreshed MaxContinues.
func TestResumedBudgetContextSeedsContinuesUsedAndBoundsChain(t *testing.T) {
	behavior := BudgetExhaustedBehavior{
		Kind:         BehaviorContinue,
		MaxContinues: 2,
		OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
	}
	// Reconstruct as the resume seam does: ContinuesUsed seeded to 1.
	scope := ResumedBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 3}, behavior, "react", 1)
	if scope.ContinuesUsed != 1 {
		t.Fatalf("ContinuesUsed = %d, want 1 (seeded, NOT zeroed)", scope.ContinuesUsed)
	}
	if scope.StepsTaken != 0 {
		t.Fatalf("StepsTaken = %d, want 0 (fresh per-round step budget)", scope.StepsTaken)
	}
	if scope.ContinuesRemaining() != 1 {
		t.Fatalf("ContinuesRemaining = %d, want 1 (only one continue left)", scope.ContinuesRemaining())
	}
	if got := scope.ResolveExhausted(); got != ExhaustedResolutionContinue {
		t.Fatalf("first resolve = %q, want continue", got)
	}
	if scope.ContinuesUsed != 2 {
		t.Fatalf("ContinuesUsed after grant = %d, want 2", scope.ContinuesUsed)
	}
	// Continues now spent → fall through to OnExhausted = Fail.
	if got := scope.ResolveExhausted(); got != ExhaustedResolutionFail {
		t.Fatalf("second resolve = %q, want fail", got)
	}

	// Contrast: a FRESH (pre-#129) scope would grant TWO continues — the bug.
	fresh := NewBudgetContext(BudgetPolicy{Kind: BudgetPerLoop, Value: 3}, BudgetExhaustedBehavior{
		Kind:         BehaviorContinue,
		MaxContinues: 2,
		OnExhausted:  &BudgetExhaustedBehavior{Kind: BehaviorFail},
	}, "react")
	if fresh.ContinuesRemaining() != 2 {
		t.Fatalf("fresh ContinuesRemaining = %d, want 2 (the bug: full budget refreshed)", fresh.ContinuesRemaining())
	}
}

// LIVE Continue reachable (ExhaustedResolution::Continue no longer dead): a
// bare-leaf run with Behavior Continue{MaxContinues:1} exhausts in-process, gets
// a granted continue (counter resets, ContinuesUsed bumps), loops, and completes
// — all WITHOUT a pause (AC3: no serialization).
func TestLiveContinueLoopsInProcessThenCompletes(t *testing.T) {
	a := NewMockAgent("leaf")
	// First window: 2 tool turns exhaust the PerLoop{2} cap → Continue grant.
	a.Push(NewToolCallRequested([]ToolCall{{ID: "c0", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	// After the in-process continue refreshes the cap, the worker completes.
	a.Push(NewFinalResponse("done after in-process continue", turnUsage()))
	// Autonomous so an Escalate fall-through would NOT pause — proving the success
	// came from the Continue loop, not a HITL pause.
	cfg := standardCfg(a)
	cfg.ToolRegistry = toolReg(3)
	h := NewStandardHarness(cfg)
	task := NewTask("do something", SessionID("s1"), continueLeaf(2, 1, BehaviorFail))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success via in-process Continue, got %+v", r)
	}
	if r.Output != "done after in-process continue" {
		t.Fatalf("output = %q, want done after in-process continue", r.Output)
	}
}

// AC3: the in-process Continue path performs NO serialization — it never sets the
// cross-process resume seed. A live Continue run completes with the seed nil.
func TestInProcessContinueDoesNoSerialization(t *testing.T) {
	a := NewMockAgent("leaf")
	a.Push(NewToolCallRequested([]ToolCall{{ID: "c0", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewToolCallRequested([]ToolCall{{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	a.Push(NewFinalResponse("done", turnUsage()))
	cfg := standardCfg(a)
	cfg.ToolRegistry = toolReg(3)
	h := NewStandardHarness(cfg)
	task := NewTask("do something", SessionID("s1"), continueLeaf(2, 1, BehaviorFail))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	// A live Continue resolves to a terminal Success/Failure, NEVER a paused
	// WaitingForHuman state (which is the only path that serializes a checkpoint).
	if r.Kind == RunWaitingForHuman {
		t.Fatalf("in-process Continue must NOT pause/serialize, got %+v", r)
	}
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
}

// AC4: a Continue resume PRESERVES the prior session_state.messages (the
// conversation context survives the pause), while Ralph DISCARDS + re-seeds. The
// shared checkpoint utility did NOT unify the context policy.
func TestContinueResumePreservesSessionContext(t *testing.T) {
	a := NewMockAgent("leaf")
	a.Push(NewFinalResponse("resumed with context", turnUsage()))
	cfg := standardCfg(a) // Autonomous: a finish is a clean Success.
	cfg.ToolRegistry = toolReg(1)
	h := NewStandardHarness(cfg)

	state := continueCheckpointPausedState()
	state.Task.LoopStrategy = continueLeaf(3, 2, BehaviorFail)
	if len(state.SessionState.Messages) < 1 {
		t.Fatal("checkpoint must carry prior context")
	}
	priorMsg := state.SessionState.Messages[0]

	r := h.Resume(context.Background(), state, EscalateResponse(ContinueWithBudgetAction(3)), nil)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success on resume, got %+v", r)
	}
	if r.Output != "resumed with context" {
		t.Fatalf("output = %q, want resumed with context", r.Output)
	}
	// The resumed session retains the prior assistant message — it BUILT ON the
	// checkpoint's conversation rather than starting from an empty (re-seeded)
	// session the way Ralph would. (The NoopContextManager used here does not
	// append the resumed turn, so the discriminating signal is that the prior
	// context PERSISTS, not that the count grew.)
	if len(r.SessionState.Messages) < 1 {
		t.Fatalf("Continue must preserve prior context (Ralph discards it); got an empty session")
	}
	if r.SessionState.Messages[0] != priorMsg {
		t.Fatalf("Continue must preserve the prior message verbatim; got %+v, want %+v",
			r.SessionState.Messages[0], priorMsg)
	}
}

// AC1: the SHARED checkpoint utility round-trips a PausedState (the durable
// pause/resume seam reused by both the cross-process Continue path and Ralph's
// pause-propagation).
func TestSharedCheckpointUtilityRoundTrips(t *testing.T) {
	state := continueCheckpointPausedState()
	blob, err := state.SerializeCheckpoint()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	restored, err := LoadCheckpoint(blob)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got, err := json.Marshal(restored)
	if err != nil {
		t.Fatalf("re-marshal restored: %v", err)
	}
	if !jsonEqual(t, got, blob) {
		t.Fatalf("Continue checkpoint did not round-trip:\n got %s\nwant %s", got, blob)
	}

	// A Ralph-style paused state (no human_request) ALSO round-trips through the
	// same utility — proving it is shared, not Continue-specific.
	ralphState := PausedState{
		SessionID:    SessionID("ralph-sess"),
		TaskID:       TaskID("ralph-task"),
		TurnNumber:   2,
		SessionState: SessionState{},
		Task:         NewTask("ralph", SessionID("ralph-sess"), RalphStrategy(RalphConfig{Inner: PtrStrategy(ReActStrategy(1)), Agent: AgentRef("r"), Behavior: defaultBudgetBehavior()})),
		BudgetUsed:   BudgetSnapshot{Turns: 2},
	}
	rblob, err := ralphState.SerializeCheckpoint()
	if err != nil {
		t.Fatalf("ralph serialize: %v", err)
	}
	rrestored, err := LoadCheckpoint(rblob)
	if err != nil {
		t.Fatalf("ralph load: %v", err)
	}
	rgot, err := json.Marshal(rrestored)
	if err != nil {
		t.Fatalf("ralph re-marshal: %v", err)
	}
	if !jsonEqual(t, rgot, rblob) {
		t.Fatalf("ralph checkpoint did not round-trip:\n got %s\nwant %s", rgot, rblob)
	}
}

// Leaf behavior (Q1): a BARE top-level leaf HONORS its Behavior. A leaf with
// Behavior Fail resolves to a BudgetExceeded Failure with the partial DISCARDED
// — distinct from the default Escalate (which under SurfaceToHuman would PAUSE).
// Under Autonomous both reach Failure, so we assert the partial is discarded
// (the Fail contract) to prove the bare-leaf honored Fail.
func TestBareLeafHonorsFailBehavior(t *testing.T) {
	a := budgetExhaustingAgent(3)
	cfg := standardCfg(a) // Autonomous
	cfg.ToolRegistry = toolReg(3)
	h := NewStandardHarness(cfg)
	leaf := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:   BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		Behavior: BudgetExhaustedBehavior{Kind: BehaviorFail},
		Agent:    AgentRef(""),
		Toolset:  ToolsetRef(""),
	}}
	task := NewTask("do something", SessionID("s1"), leaf)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("expected BudgetExceeded Failure via leaf Fail, got %+v", r)
	}
	// Fail contract: the partial is DISCARDED (the bare leaf honored its Fail
	// behavior, not the propagate/Escalate path).
	if len(r.SessionState.Messages) != 0 {
		t.Fatalf("Fail must discard the partial, got %d messages", len(r.SessionState.Messages))
	}
}

// A NESTED leaf does NOT self-resolve: its Continue behavior is IGNORED by the
// leaf body (it propagates to the parent), so the PARENT combinator's behavior
// governs. Here the parent PlanExecute carries the default Escalate placeholder,
// so the nested leaf's exhaustion surfaces as the PARENT's plan_execute pause
// (phase == plan_execute), NOT a leaf-level react resolution of its own Continue.
func TestNestedLeafPropagatesDoesNotSelfResolve(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["x"]}`))
	for i := 0; i < 4; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{ID: "e", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	}
	cfg := surfaceCfg(a)
	cfg.ToolRegistry = toolReg(4)
	h := NewStandardHarness(cfg)

	// The execute leaf carries Continue — but NESTED, so the leaf must NOT
	// self-resolve it; the PlanExecute parent (Escalate placeholder) pauses.
	plan := ReActStrategy(^uint32(0))
	plan.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
	exec := continueLeaf(2, 1, BehaviorFail)
	task := NewTask("build", SessionID("s1"), PlanExecuteStrategy(PlanExecuteConfig{
		Plan:     &plan,
		Execute:  &exec,
		Behavior: defaultBudgetBehavior(),
	}))
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected parent plan_execute pause, got %+v", r)
	}
	if r.Request == nil || r.Request.Kind != HumanReqBudgetExhausted || r.Request.Phase != "plan_execute" {
		t.Fatalf("the PARENT must resolve (phase==plan_execute); the nested leaf must NOT self-resolve "+
			"its own Continue at the react phase; got %+v", r.Request)
	}
}
