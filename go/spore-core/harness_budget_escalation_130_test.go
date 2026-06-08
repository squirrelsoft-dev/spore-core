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
// #130 — HumanRequest::BudgetExhausted + Escalate HITL resume
//
// Mirrors rust/crates/spore-core/src/harness.rs (#130 unit tests) and the shared
// fixture fixtures/paused_states/budget_exhausted.json. Same types, same rules,
// byte-identical wire format.
// ============================================================================

// surfaceCfg is a standardCfg clone with EscalationMode = SurfaceToHuman (#130):
// the HITL path under test. Budget escalation PAUSES with a
// HumanRequest.BudgetExhausted rather than propagating up.
func surfaceCfg(agent Agent) HarnessConfig {
	cfg := standardCfg(agent)
	cfg.EscalationMode = SurfaceToHumanEscalation()
	return cfg
}

// ----------------------------------------------------------------------------
// JSON round-trips — the three new variants
// ----------------------------------------------------------------------------

func TestEscalationActionJSONRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		act  EscalationAction
		wire string
	}{
		{"continue_with_budget", ContinueWithBudgetAction(6), `{"kind":"continue_with_budget","steps":6}`},
		{"skip", SkipAction(), `{"kind":"skip"}`},
		{"fail", FailAction(), `{"kind":"fail"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.act)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) != tc.wire {
				t.Fatalf("wire = %s, want %s", b, tc.wire)
			}
			var got EscalationAction
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got != tc.act {
				t.Fatalf("round-trip = %+v, want %+v", got, tc.act)
			}
		})
	}
}

func TestEscalationActionRejectsUnknownKind(t *testing.T) {
	var a EscalationAction
	if err := json.Unmarshal([]byte(`{"kind":"bogus"}`), &a); err == nil {
		t.Fatal("expected error on unknown EscalationAction kind")
	}
}

func TestHumanRequestBudgetExhaustedJSONRoundTrip(t *testing.T) {
	partial := `{"node":"plan_execute","tasks":2,"ledger":[]}`
	req := HumanRequest{
		Kind:          HumanReqBudgetExhausted,
		Phase:         "plan_execute",
		Policy:        BudgetPolicy{Kind: BudgetTotalSteps, Value: 6},
		StepsTaken:    6,
		ContinuesUsed: 1,
		PartialOutput: &partial,
		AvailableActions: []EscalationAction{
			ContinueWithBudgetAction(6),
			SkipAction(),
			FailAction(),
		},
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"kind":"budget_exhausted","phase":"plan_execute","policy":{"kind":"total_steps","value":6},"steps_taken":6,"continues_used":1,"partial_output":"{\"node\":\"plan_execute\",\"tasks\":2,\"ledger\":[]}","available_actions":[{"kind":"continue_with_budget","steps":6},{"kind":"skip"},{"kind":"fail"}]}`
	if string(b) != want {
		t.Fatalf("wire =\n%s\nwant\n%s", b, want)
	}
	var got HumanRequest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Kind != HumanReqBudgetExhausted || got.Phase != "plan_execute" ||
		got.StepsTaken != 6 || got.ContinuesUsed != 1 ||
		got.Policy.Kind != BudgetTotalSteps || got.Policy.Value != 6 {
		t.Fatalf("scalar round-trip mismatch: %+v", got)
	}
	if got.PartialOutput == nil || *got.PartialOutput != partial {
		t.Fatalf("partial_output round-trip = %v, want %q", got.PartialOutput, partial)
	}
	if len(got.AvailableActions) != 3 || got.AvailableActions[0] != ContinueWithBudgetAction(6) ||
		got.AvailableActions[1] != SkipAction() || got.AvailableActions[2] != FailAction() {
		t.Fatalf("available_actions round-trip = %+v", got.AvailableActions)
	}
}

func TestHumanRequestBudgetExhaustedNilPartialIsNull(t *testing.T) {
	req := HumanRequest{
		Kind:             HumanReqBudgetExhausted,
		Phase:            "react",
		Policy:           BudgetPolicy{Kind: BudgetPerLoop, Value: 2},
		StepsTaken:       2,
		ContinuesUsed:    0,
		PartialOutput:    nil,
		AvailableActions: []EscalationAction{ContinueWithBudgetAction(2), FailAction()},
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// partial_output serializes as null (not omitted) when nil, mirroring the
	// Rust Option<String> with no skip-if-none.
	if !bytes.Contains(b, []byte(`"partial_output":null`)) {
		t.Fatalf("nil partial_output must serialize as null, got %s", b)
	}
	var got HumanRequest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.PartialOutput != nil {
		t.Fatalf("nil partial_output round-trip = %v, want nil", got.PartialOutput)
	}
}

func TestHumanResponseEscalateJSONRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		resp HumanResponse
		wire string
	}{
		{
			"continue_with_budget",
			EscalateResponse(ContinueWithBudgetAction(4)),
			`{"kind":"escalate","action":{"kind":"continue_with_budget","steps":4}}`,
		},
		{"skip", EscalateResponse(SkipAction()), `{"kind":"escalate","action":{"kind":"skip"}}`},
		{"fail", EscalateResponse(FailAction()), `{"kind":"escalate","action":{"kind":"fail"}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.resp)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) != tc.wire {
				t.Fatalf("wire = %s, want %s", b, tc.wire)
			}
			var got HumanResponse
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.Kind != HumanRespEscalate || got.Action != tc.resp.Action {
				t.Fatalf("round-trip = %+v, want %+v", got, tc.resp)
			}
		})
	}
}

// ----------------------------------------------------------------------------
// Existing-variant regression — the new fields/variants do NOT perturb the
// pre-#130 wire form of the existing HumanRequest / HumanResponse variants.
// ----------------------------------------------------------------------------

func TestExistingHumanRequestVariantsUnchanged(t *testing.T) {
	cases := []struct {
		name string
		req  HumanRequest
		wire string
	}{
		{
			"tool_approval",
			HumanRequest{Kind: HumanReqToolApproval, Calls: []ToolCall{}, RiskLevel: RiskHigh},
			`{"kind":"tool_approval","calls":[],"risk_level":"high"}`,
		},
		{
			"clarification_no_options",
			HumanRequest{Kind: HumanReqClarification, Question: "which?"},
			`{"kind":"clarification","question":"which?"}`,
		},
		{
			"review",
			HumanRequest{Kind: HumanReqReview, Content: "look"},
			`{"kind":"review","content":"look"}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.req)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) != tc.wire {
				t.Fatalf("wire = %s, want %s", b, tc.wire)
			}
			var got HumanRequest
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.Kind != tc.req.Kind {
				t.Fatalf("kind = %q, want %q", got.Kind, tc.req.Kind)
			}
		})
	}
}

func TestExistingHumanResponseVariantsUnchanged(t *testing.T) {
	cases := []struct {
		name string
		resp HumanResponse
		wire string
	}{
		{"allow", HumanResponse{Kind: HumanRespAllow}, `{"kind":"allow"}`},
		{"halt", HumanResponse{Kind: HumanRespHalt}, `{"kind":"halt"}`},
		{"deny", HumanResponse{Kind: HumanRespDeny, Reason: "no"}, `{"kind":"deny","reason":"no"}`},
		{"answer", HumanResponse{Kind: HumanRespAnswer, Text: "yes"}, `{"kind":"answer","text":"yes"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.resp)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) != tc.wire {
				t.Fatalf("wire = %s, want %s", b, tc.wire)
			}
		})
	}
}

// ----------------------------------------------------------------------------
// EscalationMode() accessor
// ----------------------------------------------------------------------------

func TestEscalationModeAccessor(t *testing.T) {
	a := NewMockAgent("x")
	if got := NewStandardHarness(surfaceCfg(a)).EscalationMode().Kind; got != EscalationSurfaceToHuman {
		t.Fatalf("surface accessor = %q, want surface_to_human", got)
	}
	if got := NewStandardHarness(standardCfg(a)).EscalationMode().Kind; got != EscalationAutonomous {
		t.Fatalf("autonomous accessor = %q, want autonomous", got)
	}
	// The zero EscalationMode defaults to surface_to_human.
	cfg := standardCfg(a)
	cfg.EscalationMode = EscalationMode{}
	if got := NewStandardHarness(cfg).EscalationMode().Kind; got != EscalationSurfaceToHuman {
		t.Fatalf("default accessor = %q, want surface_to_human", got)
	}
}

// ----------------------------------------------------------------------------
// SurfaceToHuman bare-leaf pause — omits Skip (fork C)
// ----------------------------------------------------------------------------

// leafExhaustionCfg drives a bare ReAct leaf to its own cap (PerLoop{2}, no
// global cap) so the leaf propagates a BudgetExhausted that resolves to Escalate.
func leafExhaustionCfg(t *testing.T) HarnessConfig {
	t.Helper()
	a := NewMockAgent("leaf")
	for i := 0; i < 3; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	}
	cfg := surfaceCfg(a)
	reg := NewScriptedToolRegistry()
	for i := 0; i < 3; i++ {
		reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	}
	cfg.ToolRegistry = reg
	return cfg
}

func TestSurfaceToHumanBareLeafPauses(t *testing.T) {
	h := NewStandardHarness(leafExhaustionCfg(t))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected WaitingForHuman, got %+v", r)
	}
	if r.Request == nil || r.Request.Kind != HumanReqBudgetExhausted {
		t.Fatalf("expected BudgetExhausted request, got %+v", r.Request)
	}
	if r.State == nil || r.State.HumanRequest == nil ||
		r.State.HumanRequest.Kind != HumanReqBudgetExhausted {
		t.Fatalf("paused state must carry the BudgetExhausted request, got %+v", r.State)
	}
	// fork E: the request carries the node's steps_taken so resume can rebuild
	// the budget context. The leaf stopped at its own cap of 2.
	if r.Request.StepsTaken != 2 {
		t.Fatalf("steps_taken = %d, want 2", r.Request.StepsTaken)
	}
	// fork C: a bare leaf OMITS Skip — offers [ContinueWithBudget, Fail].
	acts := r.Request.AvailableActions
	if len(acts) != 2 || acts[0].Kind != EscalationContinueWithBudget || acts[1].Kind != EscalationFail {
		t.Fatalf("bare leaf actions = %+v, want [continue_with_budget, fail]", acts)
	}
	for _, a := range acts {
		if a.Kind == EscalationSkip {
			t.Fatal("a bare leaf must NOT offer Skip (fork C)")
		}
	}
}

// ----------------------------------------------------------------------------
// SurfaceToHuman combinator pause (PlanExecute) — includes Skip (fork C)
// ----------------------------------------------------------------------------

// combinatorExhaustionTask builds a PlanExecute whose EXECUTE child is a ReAct
// leaf capped at PerLoop{1}. When the execute step runs more than one turn it
// exhausts its OWN leaf cap and propagates a BudgetExhausted to the PlanExecute
// parent's per-task Escalate site (the combinator escalate path under test).
func combinatorExhaustionTask() Task {
	plan := ReActStrategy(^uint32(0))
	plan.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
	exec := ReActStrategy(1) // PerLoop{1} — exhausts after one turn
	return NewTask("build a CLI", SessionID("plan-sess"), PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    &plan,
		Execute: &exec,
	}))
}

// planExhaustionHarness drives the execute child past its PerLoop{1} cap: the
// plan turn yields the list, then the first execute step issues two tool calls so
// the leaf exhausts and the PlanExecute combinator escalate site fires.
func planExhaustionHarness(t *testing.T) (*StandardHarness, Task) {
	t.Helper()
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["x","y","z"]}`))
	for i := 0; i < 3; i++ {
		a.Push(NewToolCallRequested([]ToolCall{{ID: "c", Name: "x", Input: json.RawMessage(`{}`)}}, turnUsage()))
	}
	cfg := surfaceCfg(a)
	reg := NewScriptedToolRegistry()
	for i := 0; i < 3; i++ {
		reg.Push(ToolOutput{Kind: ToolOutputSuccess, Content: "ok"})
	}
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	return h, combinatorExhaustionTask()
}

func TestSurfaceToHumanCombinatorPauses(t *testing.T) {
	h, task := planExhaustionHarness(t)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("expected WaitingForHuman, got %+v", r)
	}
	if r.Request == nil || r.Request.Kind != HumanReqBudgetExhausted {
		t.Fatalf("expected BudgetExhausted request, got %+v", r.Request)
	}
	if r.Request.Phase != "plan_execute" {
		t.Fatalf("phase = %q, want plan_execute", r.Request.Phase)
	}
	// fork C: a combinator OFFERS Skip — [ContinueWithBudget, Skip, Fail].
	acts := r.Request.AvailableActions
	if len(acts) != 3 || acts[0].Kind != EscalationContinueWithBudget ||
		acts[1].Kind != EscalationSkip || acts[2].Kind != EscalationFail {
		t.Fatalf("combinator actions = %+v, want [continue_with_budget, skip, fail]", acts)
	}
}

// ----------------------------------------------------------------------------
// Autonomous — no pause (existing propagate behavior unchanged)
// ----------------------------------------------------------------------------

func TestAutonomousBareLeafDoesNotPause(t *testing.T) {
	cfg := leafExhaustionCfg(t)
	cfg.EscalationMode = AutonomousEscalation()
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunFailure {
		t.Fatalf("autonomous must propagate to Failure, got %+v", r)
	}
	if r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("expected BudgetExceeded, got %+v", r.Reason)
	}
}

func TestAutonomousCombinatorDoesNotPause(t *testing.T) {
	h, task := planExhaustionHarness(t)
	// Flip the SAME setup to Autonomous: the combinator must propagate, not pause.
	h.config.EscalationMode = AutonomousEscalation()
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind == RunWaitingForHuman {
		t.Fatalf("autonomous combinator must NOT pause, got %+v", r)
	}
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("autonomous combinator must propagate to BudgetExceeded, got %+v", r)
	}
}

// ----------------------------------------------------------------------------
// Resume — ContinueWithBudget / Fail / Skip(leaf) / Skip(PlanExecute)
// ----------------------------------------------------------------------------

// budgetPausedLeafState builds a paused state for a bare-leaf BudgetExhausted
// pause, plus a fresh harness whose agent can complete the resumed run.
func budgetPausedLeafState(t *testing.T) PausedState {
	t.Helper()
	h := NewStandardHarness(leafExhaustionCfg(t))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(2)))
	if r.Kind != RunWaitingForHuman || r.State == nil {
		t.Fatalf("setup: expected a paused leaf state, got %+v", r)
	}
	return *r.State
}

func TestResumeContinueWithBudgetGrantsAndReenters(t *testing.T) {
	state := budgetPausedLeafState(t)
	// A resume harness whose agent finishes immediately once it gets more budget.
	a := NewMockAgent("leaf")
	a.Push(NewFinalResponse("finished", turnUsage()))
	cfg := surfaceCfg(a)
	h := NewStandardHarness(cfg)
	r := h.Resume(context.Background(), state, EscalateResponse(ContinueWithBudgetAction(3)), nil)
	if r.Kind != RunSuccess {
		t.Fatalf("ContinueWithBudget should re-enter and succeed, got %+v", r)
	}
	if r.Output != "finished" {
		t.Fatalf("output = %q, want finished", r.Output)
	}
}

func TestResumeContinueWithBudgetRaisesLeafCap(t *testing.T) {
	// The leaf paused at steps_taken=2 with a PerLoop{2} cap. A ContinueWithBudget
	// of 3 must raise the resumed task's leaf budget to 2+3=5 so the restored
	// scope has room for 3 more steps.
	state := budgetPausedLeafState(t)
	resumed := state.Task
	grantTaskBudget(&resumed, state.HumanRequest.StepsTaken+3)
	if resumed.LoopStrategy.ReActCfg.Budget.Value != 5 {
		t.Fatalf("granted leaf cap = %d, want 5", resumed.LoopStrategy.ReActCfg.Budget.Value)
	}
	if resumed.Budget.MaxTurns == nil || *resumed.Budget.MaxTurns != 5 {
		t.Fatalf("granted max_turns = %v, want 5", resumed.Budget.MaxTurns)
	}
}

func TestResumeFailPropagatesBudgetExceeded(t *testing.T) {
	state := budgetPausedLeafState(t)
	h := NewStandardHarness(surfaceCfg(NewMockAgent("x")))
	r := h.Resume(context.Background(), state, EscalateResponse(FailAction()), nil)
	if r.Kind != RunFailure {
		t.Fatalf("Fail should propagate Failure, got %+v", r)
	}
	if r.Reason.Kind != HaltBudgetExceeded || r.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("Fail reason = %+v, want BudgetExceeded(Turns)", r.Reason)
	}
	// The partial is discarded on Fail.
	if len(r.SessionState.Messages) != 0 {
		t.Fatalf("Fail must discard the partial, got %d messages", len(r.SessionState.Messages))
	}
}

func TestResumeSkipLeafResolvesCleanSuccess(t *testing.T) {
	state := budgetPausedLeafState(t)
	h := NewStandardHarness(surfaceCfg(NewMockAgent("x")))
	r := h.Resume(context.Background(), state, EscalateResponse(SkipAction()), nil)
	if r.Kind != RunSuccess {
		t.Fatalf("Skip on a leaf resolves to clean Success, got %+v", r)
	}
	if r.Output != "" {
		t.Fatalf("leaf Skip output = %q, want empty", r.Output)
	}
}

func TestResumeSkipPlanExecuteAdvancesOuterLoop(t *testing.T) {
	// Mirror the Rust test: build a paused PlanExecute state BY HAND with default
	// (uncapped) leaves + a generous budget, then resume with Skip. The contract
	// under test is that a PlanExecute Skip re-enters the loop via drive_strategy
	// (advancing the outer loop) rather than returning the same pause verbatim or
	// taking the leaf clean-Success shortcut.
	a := NewMockAgent("planner")
	a.Push(planFinal("advanced"))
	cfg := surfaceCfg(a)
	cfg.ToolRegistry = NewScriptedToolRegistry()
	h := NewStandardHarness(cfg)

	pe := PlanExecuteStrategy(PlanExecuteConfig{
		Plan: func() *LoopStrategy {
			p := ReActStrategy(^uint32(0))
			p.ReActCfg.Output = func() *SchemaRef { s := SchemaRef(""); return &s }()
			return &p
		}(),
		Execute: func() *LoopStrategy { e := ReActStrategy(^uint32(0)); return &e }(),
	})
	max := uint32(8)
	task := NewTask("build it", SessionID("s1"), pe).WithBudget(BudgetLimits{MaxTurns: &max})
	partial := reactPartialJSON("")
	state := PausedState{
		SessionID:    SessionID("s1"),
		TaskID:       task.ID,
		TurnNumber:   1,
		SessionState: SessionState{},
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
		Task: task,
	}
	got := h.Resume(context.Background(), state, EscalateResponse(SkipAction()), nil)
	if got.Kind == RunWaitingForHuman {
		t.Fatalf("Skip on PlanExecute should advance the outer loop, not re-pause: %+v", got)
	}
}

// ----------------------------------------------------------------------------
// Out-of-contract Escalate on a non-budget pause halts cleanly
// ----------------------------------------------------------------------------

func TestResumeEscalateOnNonBudgetPauseHalts(t *testing.T) {
	state := PausedState{
		SessionID:    SessionID("s1"),
		Task:         reactTask(5),
		SessionState: SessionState{},
		HumanRequest: &HumanRequest{Kind: HumanReqReview, Content: "look"},
	}
	h := NewStandardHarness(surfaceCfg(NewMockAgent("x")))
	r := h.Resume(context.Background(), state, EscalateResponse(FailAction()), nil)
	if r.Kind != RunFailure || r.Reason.Kind != HaltHumanHalted {
		t.Fatalf("Escalate on a non-budget pause must halt cleanly, got %+v", r)
	}
}

// ----------------------------------------------------------------------------
// Fixture replay — fixtures/paused_states/budget_exhausted.json
// ----------------------------------------------------------------------------

func budgetExhaustedFixturePath(t *testing.T) string {
	t.Helper()
	_, this, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(this), "..", "..", "fixtures", "paused_states", "budget_exhausted.json")
}

func TestBudgetExhaustedFixtureReplay(t *testing.T) {
	raw, err := os.ReadFile(budgetExhaustedFixturePath(t))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	var state PausedState
	if err := json.Unmarshal(raw, &state); err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	// Field-for-field assertions against the fixture ground truth.
	if state.SessionID != "sess-130" || state.TaskID != "task-130" || state.TurnNumber != 6 {
		t.Fatalf("envelope mismatch: %+v", state)
	}
	if state.ChildState != nil {
		t.Fatalf("child_state must be nil, got %+v", state.ChildState)
	}
	hr := state.HumanRequest
	if hr == nil || hr.Kind != HumanReqBudgetExhausted {
		t.Fatalf("human_request = %+v, want budget_exhausted", hr)
	}
	if hr.Phase != "plan_execute" || hr.StepsTaken != 6 || hr.ContinuesUsed != 1 {
		t.Fatalf("request scalars mismatch: %+v", hr)
	}
	if hr.Policy.Kind != BudgetTotalSteps || hr.Policy.Value != 6 {
		t.Fatalf("policy = %+v, want total_steps/6", hr.Policy)
	}
	wantPartial := `{"node":"plan_execute","tasks":2,"ledger":[]}`
	if hr.PartialOutput == nil || *hr.PartialOutput != wantPartial {
		t.Fatalf("partial_output = %v, want %q", hr.PartialOutput, wantPartial)
	}
	if len(hr.AvailableActions) != 3 ||
		hr.AvailableActions[0] != ContinueWithBudgetAction(6) ||
		hr.AvailableActions[1] != SkipAction() ||
		hr.AvailableActions[2] != FailAction() {
		t.Fatalf("available_actions = %+v", hr.AvailableActions)
	}
	if state.Task.LoopStrategy.Kind != StrategyPlanExecute {
		t.Fatalf("task strategy = %q, want plan_execute", state.Task.LoopStrategy.Kind)
	}

	// Re-serialize and assert byte-identity against the fixture. jsonEqual is the
	// established cross-language convention (hooks/consult/escalation fixtures): it
	// normalizes both sides through a value round-trip so the only legitimate
	// cross-language float-formatting divergence (Go's 0 vs serde's 0.0 for a
	// whole cost_usd) does not spuriously fail — field order and structure must
	// still match exactly.
	out, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("re-serialize: %v", err)
	}
	if !jsonEqual(t, out, raw) {
		var compact bytes.Buffer
		_ = json.Compact(&compact, raw)
		t.Fatalf("re-serialized form is NOT structurally identical:\n got: %s\nwant: %s", out, compact.Bytes())
	}
}
