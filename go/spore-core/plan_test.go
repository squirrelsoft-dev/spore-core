package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// ============================================================================
// CapturePlanArtifact — Q3 grammar (R3 / R9). Mirrors the Rust unit tests in
// rust/crates/spore-core/src/plan.rs.
// ============================================================================

func TestCapturePlainJSONObject(t *testing.T) {
	a, err := CapturePlanArtifact(`{"tasks":["a","b","c"],"rationale":"because"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := a.Tasks; len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
		t.Fatalf("tasks = %v", got)
	}
	if a.Rationale != "because" {
		t.Fatalf("rationale = %q", a.Rationale)
	}
}

func TestCaptureTrimsSurroundingWhitespace(t *testing.T) {
	a, err := CapturePlanArtifact("\n\t  {\"tasks\":[\"x\"]}  \r\n")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(a.Tasks) != 1 || a.Tasks[0] != "x" || a.Rationale != "" {
		t.Fatalf("got %+v", a)
	}
}

func TestCaptureStripsJSONFence(t *testing.T) {
	a, err := CapturePlanArtifact("```json\n{\"tasks\":[\"step 1\",\"step 2\"],\"rationale\":\"r\"}\n```")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(a.Tasks) != 2 || a.Tasks[0] != "step 1" || a.Tasks[1] != "step 2" || a.Rationale != "r" {
		t.Fatalf("got %+v", a)
	}
}

func TestCaptureStripsBareFence(t *testing.T) {
	a, err := CapturePlanArtifact("```\n{\"tasks\":[\"only\"]}\n```")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(a.Tasks) != 1 || a.Tasks[0] != "only" {
		t.Fatalf("got %+v", a)
	}
}

func TestCaptureStripsUppercaseJSONFence(t *testing.T) {
	a, err := CapturePlanArtifact("```JSON\n{\"tasks\":[\"u\"]}\n```")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(a.Tasks) != 1 || a.Tasks[0] != "u" {
		t.Fatalf("got %+v", a)
	}
}

func TestCaptureRationaleDefaultsToEmpty(t *testing.T) {
	a, err := CapturePlanArtifact(`{"tasks":["a"]}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if a.Rationale != "" {
		t.Fatalf("rationale = %q", a.Rationale)
	}
}

func TestCaptureEmptyTasksArrayIsAllowed(t *testing.T) {
	a, err := CapturePlanArtifact(`{"tasks":[]}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(a.Tasks) != 0 {
		t.Fatalf("tasks = %v", a.Tasks)
	}
}

func TestCaptureTasksKeptVerbatim(t *testing.T) {
	a, err := CapturePlanArtifact(`{"tasks":["  spaced  ",""]}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(a.Tasks) != 2 || a.Tasks[0] != "  spaced  " || a.Tasks[1] != "" {
		t.Fatalf("got %+v", a)
	}
}

// R9: malformed inputs return UnparseablePlan, never panic.
func TestCaptureUnparseableCases(t *testing.T) {
	cases := map[string]string{
		"invalid json":       "not json at all",
		"non-object top":     "[1,2,3]",
		"missing tasks":      `{"rationale":"x"}`,
		"tasks not array":    `{"tasks":"a"}`,
		"non-string element": `{"tasks":["a",2]}`,
		"non-string ration":  `{"tasks":["a"],"rationale":5}`,
		"empty input":        "   \n  ",
	}
	for name, text := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := CapturePlanArtifact(text)
			pe, ok := err.(*PlanPhaseError)
			if !ok {
				t.Fatalf("expected *PlanPhaseError, got %v", err)
			}
			if pe.Kind != PlanErrorUnparseablePlan {
				t.Fatalf("kind = %q", pe.Kind)
			}
		})
	}
}

// R9: deterministic — identical input yields identical artifact.
func TestCaptureIsDeterministic(t *testing.T) {
	text := "```json\n{\"tasks\":[\"a\",\"b\"],\"rationale\":\"r\"}\n```"
	a1, err := CapturePlanArtifact(text)
	if err != nil {
		t.Fatal(err)
	}
	a2, err := CapturePlanArtifact(text)
	if err != nil {
		t.Fatal(err)
	}
	if b1, b2 := mustJSON(t, a1), mustJSON(t, a2); string(b1) != string(b2) {
		t.Fatalf("not deterministic: %s vs %s", b1, b2)
	}
}

// ============================================================================
// Plan-phase driver (R1–R8, R10–R11 + Q4). Mirrors the Rust harness tests.
// ============================================================================

func planTask(instruction string) Task {
	return NewTask(instruction, SessionID("plan-sess"), LoopStrategy{Kind: StrategyPlanExecute})
}

func planFinal(text string) TurnResult {
	return NewFinalResponse(text, TokenUsage{InputTokens: 5, OutputTokens: 3})
}

// storedArtifact reads + decodes the artifact persisted to the RunStore seam
// under PlanExecuteExtrasKey (#76 — no longer mirrored into SessionState.Extras).
func storedArtifact(t *testing.T, store *fakeRunStore, sessionID SessionID) PlanArtifact {
	t.Helper()
	raw, ok := store.get(sessionID, PlanExecuteExtrasKey)
	if !ok {
		t.Fatalf("no artifact present in run store under %q", PlanExecuteExtrasKey)
	}
	var a PlanArtifact
	if err := json.Unmarshal(raw, &a); err != nil {
		t.Fatalf("decode artifact: %v", err)
	}
	return a
}

// R1 + R3 + R4: plan phase runs once, captures the artifact, stores it under
// extras (the full run continues into the execute phase — see harness_test.go).
func TestPlanPhaseRunsOnceAndStoresArtifact(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["one","two"],"rationale":"r"}`))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	task := planTask("build it")
	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), &task, &state, BudgetSnapshot{}, nil)
	if failure != nil {
		t.Fatalf("unexpected failure: %+v", *failure)
	}
	if outcome.turns != 1 { // R1 + R7
		t.Fatalf("turns = %d, want 1", outcome.turns)
	}
	// R3 / R4
	got := storedArtifact(t, store, task.SessionID)
	if len(got.Tasks) != 2 || got.Tasks[0] != "one" || got.Tasks[1] != "two" || got.Rationale != "r" {
		t.Fatalf("stored artifact = %+v", got)
	}
	// the mock agent had exactly one queued response consumed
	if remaining := len(a.results); remaining != 0 {
		t.Fatalf("planner ran %d extra turns", remaining)
	}
}

// Confirms ExecutePhaseNotImplemented is gone (#59): a full PlanExecute run with
// execute completions now SUCCEEDS (the old variant would have halted after the
// plan phase). Mirrors the Rust execute_phase_not_implemented_is_gone test.
func TestPlanExecuteExecutePhaseNotImplementedIsGone(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["only"],"rationale":""}`))
	a.Push(planFinal("done"))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("do")))
	if r.Kind != RunSuccess {
		t.Fatalf("got %+v", r)
	}
	// plan(1) + one execute step(1) = 2 turns.
	if r.Turns != 2 {
		t.Fatalf("turns = %d, want 2", r.Turns)
	}
}

// R2: a tool-call request in the one-shot plan turn is a planning failure
// (PlanningTurnFailed), never a dispatch loop.
func TestPlanPhaseToolCallIsPlanningTurnFailed(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
	}, TokenUsage{InputTokens: 1, OutputTokens: 1}))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	reg := NewScriptedToolRegistry()
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)

	task := planTask("do")
	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), &task, &state, BudgetSnapshot{}, nil)
	if outcome != nil || failure == nil {
		t.Fatalf("expected failure, got outcome=%+v failure=%v", outcome, failure)
	}
	if failure.Reason.Kind != HaltPlanPhaseFailed || failure.Reason.PlanError == nil ||
		failure.Reason.PlanError.Kind != PlanErrorPlanningTurnFailed {
		t.Fatalf("got %+v", failure.Reason)
	}
	// No tool dispatch happened (R2: not a dispatch loop).
	if reg.CallCount.Load() != 0 {
		t.Fatalf("tool registry dispatched %d times", reg.CallCount.Load())
	}
	// No artifact stored.
	if _, ok := store.get(task.SessionID, PlanExecuteExtrasKey); ok {
		t.Fatal("artifact stored despite planning failure")
	}
}

// R3 (failure path): an unparseable response surfaces PlanPhaseFailed /
// UnparseablePlan and stores no artifact.
func TestPlanPhaseUnparseableIsPlanPhaseFailed(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal("this is not json"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	task := planTask("do")
	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), &task, &state, BudgetSnapshot{}, nil)
	if outcome != nil || failure == nil {
		t.Fatalf("expected failure, got outcome=%+v failure=%v", outcome, failure)
	}
	if failure.Reason.Kind != HaltPlanPhaseFailed || failure.Reason.PlanError == nil ||
		failure.Reason.PlanError.Kind != PlanErrorUnparseablePlan {
		t.Fatalf("got %+v", failure.Reason)
	}
	if _, ok := store.get(task.SessionID, PlanExecuteExtrasKey); ok {
		t.Fatal("artifact stored despite unparseable plan")
	}
}

// R5: when PlannerAgent is set, the PLANNER runs the plan turn and the default
// agent does NOT.
func TestPlanPhaseRoutesToPlannerAgent(t *testing.T) {
	def := NewMockAgent("default")
	def.Push(planFinal(`{"tasks":["from default"]}`))
	planner := NewMockAgent("planner")
	planner.Push(planFinal(`{"tasks":["from planner"],"rationale":"p"}`))

	cfg := standardCfg(def)
	cfg.PlannerAgent = planner
	h := NewStandardHarness(cfg)

	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), ptrTask(planTask("do")), &state, BudgetSnapshot{}, nil)
	if failure != nil {
		t.Fatalf("unexpected failure: %+v", *failure)
	}
	if got := outcome.artifact; len(got.Tasks) != 1 || got.Tasks[0] != "from planner" {
		t.Fatalf("artifact = %+v (planner did not run)", got)
	}
	// Planner consumed its turn; default agent untouched.
	if len(planner.results) != 0 {
		t.Fatal("planner did not run")
	}
	if len(def.results) != 1 {
		t.Fatalf("default agent ran (remaining=%d)", len(def.results))
	}
}

// R6: with no PlannerAgent, the plan turn runs on the default agent.
func TestPlanPhaseRoutesToDefaultAgent(t *testing.T) {
	def := NewMockAgent("default")
	def.Push(planFinal(`{"tasks":["from default"],"rationale":"d"}`))
	cfg := standardCfg(def)
	h := NewStandardHarness(cfg)

	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), ptrTask(planTask("do")), &state, BudgetSnapshot{}, nil)
	if failure != nil {
		t.Fatalf("unexpected failure: %+v", *failure)
	}
	if got := outcome.artifact; len(got.Tasks) != 1 || got.Tasks[0] != "from default" {
		t.Fatalf("artifact = %+v", got)
	}
	if len(def.results) != 0 {
		t.Fatal("default agent did not run")
	}
}

// R7: the plan turn counts against the shared budget — outcome.turns reflects
// the prior budget plus one.
func TestPlanPhaseCountsAgainstBudget(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["x"]}`))
	h := NewStandardHarness(standardCfg(a))

	state := SessionState{}
	used := BudgetSnapshot{Turns: 2, InputTokens: 100, OutputTokens: 40}
	// Allow at least 3 turns so the gate doesn't trip.
	task := planTask("do")
	max := uint32(5)
	task.Budget.MaxTurns = &max
	outcome, failure := h.runPlanPhase(context.Background(), &task, &state, used, nil)
	if failure != nil {
		t.Fatalf("unexpected failure: %+v", *failure)
	}
	if outcome.turns != 3 {
		t.Fatalf("turns = %d, want 3 (2 prior + 1 plan turn)", outcome.turns)
	}
}

// R8: exactly one turn span is recorded for the plan turn.
func TestPlanPhaseRecordsOneTurnSpan(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["x"],"rationale":""}`))
	cfg := standardCfg(a)
	obs := newCountingObserver()
	cfg.Observability = obs
	h := NewStandardHarness(cfg)

	state := SessionState{}
	_, failure := h.runPlanPhase(context.Background(), ptrTask(planTask("do")), &state, BudgetSnapshot{}, nil)
	if failure != nil {
		t.Fatalf("unexpected failure: %+v", *failure)
	}
	if obs.turns != 1 {
		t.Fatalf("turn spans = %d, want 1", obs.turns)
	}
}

// R10: budget exhausted before the plan turn → budget-exceeded failure with no
// artifact stored and the agent never invoked.
func TestPlanPhaseBudgetExhaustedBeforeTurn(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["x"]}`)) // should never be consumed
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)

	task := planTask("do")
	max := uint32(1)
	task.Budget.MaxTurns = &max
	state := SessionState{}
	used := BudgetSnapshot{Turns: 1} // already at the cap

	outcome, failure := h.runPlanPhase(context.Background(), &task, &state, used, nil)
	if outcome != nil || failure == nil {
		t.Fatalf("expected failure, got outcome=%+v failure=%v", outcome, failure)
	}
	if failure.Reason.Kind != HaltBudgetExceeded || failure.Reason.LimitType != BudgetLimitTurns {
		t.Fatalf("got %+v", failure.Reason)
	}
	if _, ok := store.get(task.SessionID, PlanExecuteExtrasKey); ok {
		t.Fatal("artifact stored despite budget exhaustion")
	}
	if len(a.results) != 1 {
		t.Fatal("agent ran despite budget exhaustion")
	}
}

// R11: an OnPlanCreated hook mutation is reflected in the stored artifact.
func TestPlanPhaseOnPlanCreatedMutationIsStored(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["a"],"rationale":"orig"}`))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store

	chain := NewStandardHookChain()
	if err := chain.Register(NewFunctionHook("mutate", []HookEvent{HookEventOnPlanCreated},
		func(_ context.Context, hctx *HookContext) (HookDecision, error) {
			hctx.Plan.Tasks = append(hctx.Plan.Tasks, "injected")
			hctx.Plan.Rationale = "rewritten"
			return Continue(), nil
		})); err != nil {
		t.Fatal(err)
	}
	cfg.Hooks = chain
	h := NewStandardHarness(cfg)

	task := planTask("do")
	state := SessionState{}
	outcome, failure := h.runPlanPhase(context.Background(), &task, &state, BudgetSnapshot{}, nil)
	if failure != nil {
		t.Fatalf("unexpected failure: %+v", *failure)
	}
	// Outcome carries the mutated artifact...
	if len(outcome.artifact.Tasks) != 2 || outcome.artifact.Tasks[1] != "injected" {
		t.Fatalf("outcome artifact = %+v", outcome.artifact)
	}
	// ...and the stored artifact reflects the mutation.
	got := storedArtifact(t, store, task.SessionID)
	if len(got.Tasks) != 2 || got.Tasks[1] != "injected" || got.Rationale != "rewritten" {
		t.Fatalf("stored artifact = %+v", got)
	}
}

// ── helpers ─────────────────────────────────────────────────────────────────

func ptrTask(t Task) *Task { return &t }

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

// countingObserver counts EmitTurn calls; it embeds capturingObserver (defined
// in harness_compaction_test.go) for the rest of the HarnessObserver surface.
type countingObserver struct {
	*capturingObserver
	turns int
}

func newCountingObserver() *countingObserver {
	return &countingObserver{capturingObserver: newCapturingObserver()}
}

func (o *countingObserver) EmitTurn(string, SessionID, TaskID, uint32, string, uint64, TokenUsage, float64, StopReason, uint32, string, string, []ToolCall, []Message) {
	o.turns++
}

var _ HarnessObserver = (*countingObserver)(nil)
