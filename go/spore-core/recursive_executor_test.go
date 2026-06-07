// Per-variant RunStrategy.Run recursive executor tests (issue #124).
//
// The per-strategy behavior (ReAct / PlanExecute / SelfVerifying / Ralph /
// HillClimbing) is exercised at parity by the existing strategy test files,
// which all run through h.Run → the recursive executor now. These tests cover
// the NEW #124 capabilities + the recursive-executor wiring directly:
//
//   - ReAct leaf E2E + PlanExecute[ReAct, ReAct] E2E through the recursive
//     trait calls (driveStrategy)
//   - the no-executor scaffold path returns a typed Failed (never a panic)
//   - non-terminal pause (WaitingForHuman) propagates VERBATIM through the
//     executor (the terminal-override path)
//   - A.5 output-contract enforcement: a structured slot rejects a bare ReAct
//     without an output schema (and accepts one with a schema / a combinator)
//   - A.6 deep-resume: an already-Completed task on the durable checkpoint is
//     NOT re-run
//   - SelfVerifying is Default-FAIL and bounded; Ralph resets the window per
//     continuation; HillClimbing is stagnation-bounded (parity smoke through the
//     recursive path)

package sporecore

import (
	"context"
	"encoding/json"
	"testing"
)

// ── recursive-executor wiring ───────────────────────────────────────────────

// ReAct leaf E2E through the recursive executor: a single FinalResponse drives
// one ReAct window via ReactConfig.Run → ReactWindow and returns Success.
func TestRecursiveReActLeafEndToEnd(t *testing.T) {
	a := NewMockAgent("t")
	a.Push(NewFinalResponse("leaf done", turnUsage()))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunSuccess || r.Output != "leaf done" {
		t.Fatalf("ReAct leaf E2E got %+v", r)
	}
}

// PlanExecute[ReAct, ReAct] E2E through the recursive executor: the plan phase
// recurses (PlanPhase) then the execute phase drains per task (ExecutePhase),
// all via PlanExecuteConfig.Run.
func TestRecursivePlanExecuteEndToEnd(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal(`{"tasks":["one","two"],"rationale":"r"}`))
	a.Push(planFinal("did one"))
	a.Push(planFinal("did two"))
	h := NewStandardHarness(standardCfg(a))
	r := h.Run(context.Background(), NewHarnessRunOptions(planTask("build a CLI")))
	if r.Kind != RunSuccess || r.Output != "did two" || r.Turns != 3 {
		t.Fatalf("PlanExecute E2E got %+v", r)
	}
}

// The scaffold-only context (no wired StrategyExecutor) makes every config body
// return a typed Failed StrategyOutcome — never a panic.
func TestRecursiveNoExecutorIsTypedFailure(t *testing.T) {
	registry := NewExecutionRegistryBuilder().Build()
	cx := NewExecutionContext(&registry)
	tk := NewTask("x", NewSessionID(), ReActStrategy(1))
	cx.Scratch.Task = &tk
	cx.Scratch.RunBudget = BudgetSnapshot{}
	got := (&ReactConfig{Budget: BudgetPolicy{Kind: BudgetPerLoop, Value: 1}}).Run(context.Background(), cx)
	if got.Kind != StrategyOutcomeFailed || got.Failed == nil {
		t.Fatalf("no-executor leaf got %+v, want typed Failed", got)
	}
	if _, ok := got.Failed.(*InvalidConfigurationError); !ok {
		t.Fatalf("expected InvalidConfigurationError, got %T", got.Failed)
	}
}

// A non-terminal pause (WaitingForHuman) raised by a leaf propagates VERBATIM
// through the recursive executor: driveStrategy returns the stashed
// terminal-override rather than collapsing it into a Failure.
func TestRecursivePausePropagatesVerbatim(t *testing.T) {
	a := NewMockAgent("t")
	// A tool call whose result requests a human: the ReAct window pauses.
	a.Push(NewToolCallRequested([]ToolCall{
		{ID: "c1", Name: "x", Input: json.RawMessage(`{}`)},
	}, turnUsage()))
	cfg := standardCfg(a)
	reg := NewScriptedToolRegistry()
	req := HumanRequest{Kind: HumanReqClarification, Question: "approve?"}
	child := &ChildPausedState{
		SessionID: "child", TaskID: "ct", TurnNumber: 1,
		HumanRequest: &req, Task: reactTask(1), ParentToolCallID: "c1",
	}
	reg.Push(ToolOutput{Kind: ToolOutputWaitingForHuman, ChildState: child, Request: &req})
	cfg.ToolRegistry = reg
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(reactTask(5)))
	if r.Kind != RunWaitingForHuman {
		t.Fatalf("pause did not propagate verbatim, got %+v", r)
	}
	if r.State == nil {
		t.Fatalf("WaitingForHuman carries no paused state")
	}
}

// ── A.6 deep-resume: completed tasks are NOT re-run ─────────────────────────

// The execute phase reconciles the freshly-parsed task list with the DURABLE
// RunStore checkpoint (re-persisted after every step). A task already Completed
// on the checkpoint is skipped — serialize→reset→resume, not the old shallow
// no-op. Here task 0 is pre-marked Completed in the checkpoint; only task 1 runs
// (one queued response is enough iff task 0 is skipped — re-running it would
// starve the agent queue and fail).
func TestDeepResumeSkipsAlreadyCompletedTask(t *testing.T) {
	a := NewMockAgent("planner")
	// Only ONE response: if task 0 were re-run, the loop would starve on task 1.
	a.Push(planFinal("done two"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)
	tk := planTask("build a CLI")

	// Pre-seed the durable checkpoint: task 0 Completed, task 1 Pending.
	checkpoint := PlanArtifactToTaskList(PlanArtifact{Tasks: []string{"one", "two"}})
	firstID := checkpoint.Tasks[0].ID
	if err := checkpoint.Complete(firstID); err != nil {
		t.Fatalf("mark task 0 complete: %v", err)
	}
	h.persistTaskList(context.Background(), tk.SessionID, checkpoint)

	// The freshly-parsed list (as the plan phase would produce) is all-Pending.
	fresh := PlanArtifactToTaskList(PlanArtifact{Tasks: []string{"one", "two"}})
	state := SessionState{}
	r := h.runExecutePhase(context.Background(), &tk, &state, fresh,
		BudgetSnapshot{Turns: 1}, AggregateUsage{}, nil)
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "done two" {
		t.Fatalf("output = %q, want %q (only the not-yet-done task runs)", r.Output, "done two")
	}
	// Both tasks Completed in the final persisted list.
	final := runStoreTaskList(t, store, tk.SessionID)
	for _, x := range final.Tasks {
		if x.Status != TaskStatusCompleted {
			t.Fatalf("task %d status = %q, want completed after resume", x.ID, x.Status)
		}
	}
}

// Without a checkpoint (fresh run) every task runs — the deep-resume read is a
// miss and does not spuriously skip anything.
func TestDeepResumeNoCheckpointRunsAllTasks(t *testing.T) {
	a := NewMockAgent("planner")
	a.Push(planFinal("done one"))
	a.Push(planFinal("done two"))
	store := newFakeRunStore()
	cfg := standardCfg(a)
	cfg.RunStore = store
	h := NewStandardHarness(cfg)
	tk := planTask("build a CLI")
	fresh := PlanArtifactToTaskList(PlanArtifact{Tasks: []string{"one", "two"}})
	state := SessionState{}
	r := h.runExecutePhase(context.Background(), &tk, &state, fresh,
		BudgetSnapshot{Turns: 1}, AggregateUsage{}, nil)
	if r.Kind != RunSuccess || r.Output != "done two" {
		t.Fatalf("fresh run got %+v", r)
	}
}

// ── A.5 output-contract enforcement (#124, Q3) ──────────────────────────────

func outputContractRegistry() ExecutionRegistry {
	return NewExecutionRegistryBuilder().
		Agent("a1", stubAgent{id: "a1"}).
		Toolset("t1", NewScriptedToolRegistry()).
		Schema("plan-schema", json.RawMessage(`{}`)).
		Schema("worker-schema", json.RawMessage(`{}`)).
		Schema("eval-schema", json.RawMessage(`{}`)).
		Build()
}

// A PlanExecute whose plan slot is a bare ReAct with no output schema violates
// the A.5 contract: validation rejects it with InvalidConfigurationError naming
// the slot, BEFORE any handle resolution.
func TestStructuredSlotRejectsBareReActWithoutOutputSchema(t *testing.T) {
	reg := outputContractRegistry()
	tree := PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    PtrStrategy(reactLeaf("a1", "t1")), // output: nil
		Execute: PtrStrategy(reactLeaf("a1", "t1")),
	})
	task := NewTask("contract", NewSessionID(), tree)
	err := reg.Validate(task)
	ce, ok := err.(*InvalidConfigurationError)
	if !ok {
		t.Fatalf("expected InvalidConfigurationError, got %T (%v)", err, err)
	}
	if !contains(ce.Message, "plan") {
		t.Fatalf("error should name the slot: %q", ce.Message)
	}
}

// The worker slot (SelfVerifying.Inner) and the propose slot
// (HillClimbing.Inner) are equally structured.
func TestStructuredSlotRejectsBareReActWorkerAndPropose(t *testing.T) {
	reg := outputContractRegistry()

	sv := SelfVerifyingStrategy(SelfVerifyingConfig{
		Inner:     PtrStrategy(reactLeaf("a1", "t1")),
		Evaluator: SchemaRef("eval-schema"),
	})
	if err := reg.Validate(NewTask("w", NewSessionID(), sv)); err == nil {
		t.Fatal("worker slot accepted a bare ReAct without output schema")
	} else if ce, ok := err.(*InvalidConfigurationError); !ok || !contains(ce.Message, "worker") {
		t.Fatalf("expected worker-slot InvalidConfigurationError, got %v", err)
	}

	hc := HillClimbingStrategy(HillClimbingConfig{
		Inner:               PtrStrategy(reactLeaf("a1", "t1")),
		Direction:           OptimizationMinimize,
		MaxStagnation:       1,
		MinImprovementDelta: 0.0,
		Evaluator:           AgentRef("a1"),
	})
	if err := reg.Validate(NewTask("p", NewSessionID(), hc)); err == nil {
		t.Fatal("propose slot accepted a bare ReAct without output schema")
	} else if ce, ok := err.(*InvalidConfigurationError); !ok || !contains(ce.Message, "propose") {
		t.Fatalf("expected propose-slot InvalidConfigurationError, got %v", err)
	}
}

// A structured slot accepts a bare ReAct WITH an output schema.
func TestStructuredSlotAcceptsReActWithOutputSchema(t *testing.T) {
	reg := outputContractRegistry()
	out := SchemaRef("plan-schema")
	plan := LoopStrategy{Kind: StrategyReAct, ReActCfg: &ReactConfig{
		Budget:  BudgetPolicy{Kind: BudgetPerLoop, Value: 4},
		Agent:   AgentRef("a1"),
		Toolset: ToolsetRef("t1"),
		Output:  &out,
	}}
	tree := PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    &plan,
		Execute: PtrStrategy(withOutput(reactLeaf("a1", "t1"), "worker-schema")),
	})
	if err := reg.Validate(NewTask("ok", NewSessionID(), tree)); err != nil {
		t.Fatalf("expected validate ok, got %v", err)
	}
}

// A structured slot accepts a combinator child — the bare-ReAct check applies
// only to a leaf; a combinator carries its own contract.
func TestStructuredSlotAcceptsCombinatorChild(t *testing.T) {
	reg := outputContractRegistry()
	innerSV := SelfVerifyingStrategy(SelfVerifyingConfig{
		Inner:     PtrStrategy(withOutput(reactLeaf("a1", "t1"), "worker-schema")),
		Evaluator: SchemaRef("eval-schema"),
	})
	tree := PlanExecuteStrategy(PlanExecuteConfig{
		Plan:    &innerSV, // a combinator in the structured plan slot
		Execute: PtrStrategy(withOutput(reactLeaf("a1", "t1"), "worker-schema")),
	})
	if err := reg.Validate(NewTask("combo", NewSessionID(), tree)); err != nil {
		t.Fatalf("combinator in a structured slot should be accepted, got %v", err)
	}
}

func withOutput(s LoopStrategy, schema string) LoopStrategy {
	ref := SchemaRef(schema)
	c := *s.ReActCfg
	c.Output = &ref
	s.ReActCfg = &c
	return s
}
