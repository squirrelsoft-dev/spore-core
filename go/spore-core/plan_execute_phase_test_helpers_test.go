package sporecore

import "context"

// Test-only recursion drivers for the granular PlanExecute plan/execute phase
// tests (#124). These mirror the Rust #[cfg(test)] StandardHarness helpers
// run_plan_phase / run_execute_phase: rather than a stale parallel copy of the
// orchestration, they reproduce the MINIMAL driver around a single phase using
// the SAME scratch wiring + leaf primitives the real PlanExecuteConfig.Run uses,
// so the granular regression tests exercise the genuine per-task
// c.Plan.Run(ctx, cx) / c.Execute.Run(ctx, cx) dispatch — not a divergent path.
//
// The phase logic lives ONLY in PlanExecuteConfig.Run (strategy.go); these
// helpers exist purely to drive one phase in isolation for the existing granular
// tests.

// runPlanPhase drives the recursive plan phase for task (whose strategy MUST be a
// PlanExecute): seed the directive, dispatch plan.Run under the plan
// sub-strategy's own declared budget, then capture + persist the artifact.
// Mirrors the plan half of PlanExecuteConfig.Run
// and preserves the legacy (*planPhaseOutcome, *RunResult) test signature.
func (h *StandardHarness) runPlanPhase(
	ctx context.Context,
	task *Task,
	session *SessionState,
	budget BudgetSnapshot,
	onStream StreamSink,
) (*planPhaseOutcome, *RunResult) {
	_ = onStream
	cfg := task.LoopStrategy.PlanExecute
	if cfg == nil {
		panic("runPlanPhase test helper requires a PlanExecute strategy")
	}
	sessionID := task.SessionID

	directive := planDirective(task.Instruction)
	planSession := cloneSessionState(session)
	h.config.ContextManager.AppendUserMessage(ctx, planSession, directive)

	// The plan sub-strategy's own declared budget governs the plan phase (R1): an
	// authored task_list may take more than one turn. The task's global turn
	// ceiling remains the outer backstop (R10) — not a turns+1 clamp.
	planBudget := task.Budget
	planTask := Task{
		ID:           task.ID,
		Instruction:  directive,
		SessionID:    sessionID,
		Budget:       planBudget,
		LoopStrategy: *cfg.Plan,
	}

	planResult := h.RunPlanSubtree(ctx, cfg.Plan, planTask, *planSession, budget)
	if planResult == nil {
		return nil, &RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind: HaltPlanPhaseFailed,
				PlanError: &PlanPhaseError{
					Kind:    PlanErrorPlanningTurnFailed,
					Message: "plan sub-strategy produced no terminal",
				},
			},
			SessionID: sessionID,
			Turns:     budget.Turns,
		}
	}
	if planResult.Kind != RunSuccess {
		return nil, planResult
	}
	return h.captureAndPersistPlan(ctx, sessionID, planResult.Output, planResult.Usage, planResult.Turns)
}

// runExecutePhase drives the recursive execute phase for task (whose strategy
// MUST be a PlanExecute), draining taskList by dispatching execute.Run per task.
// Mirrors the execute half of PlanExecuteConfig.Run and preserves the legacy test
// signature.
func (h *StandardHarness) runExecutePhase(
	ctx context.Context,
	task *Task,
	session *SessionState,
	taskList TaskList,
	carried BudgetSnapshot,
	planUsage AggregateUsage,
	onStream StreamSink,
) RunResult {
	cfg := task.LoopStrategy.PlanExecute
	if cfg == nil {
		panic("runExecutePhase test helper requires a PlanExecute strategy")
	}
	cx := NewExecutionContext(&h.config.Registry)
	cx.Executor = h
	// #138: the A.6 deep-resume reconcile moved out of runExecuteLoop into
	// PlanExecuteConfig.Run; reproduce it here so this granular helper keeps the
	// same behavior as the real combinator (no resume seed in this isolated path).
	h.ReconcileCompletedTasks(ctx, task.SessionID, &taskList)
	result, exhausted := cfg.runExecuteLoop(ctx, cx, h, task, *session, taskList, carried, planUsage, nil, onStream)
	// #125: a BudgetExhausted is surfaced as a typed StrategyOutcome. Map it back
	// to the equivalent BudgetExceeded Failure RunResult so this legacy test
	// helper preserves its single-RunResult signature (mirrors driveStrategy).
	if exhausted != nil {
		var messages []Message
		var turns uint32
		if exhausted.Exhausted != nil {
			if exhausted.Exhausted.PartialOutput != nil {
				messages = []Message{{Role: RoleAssistant, Content: NewTextContent(*exhausted.Exhausted.PartialOutput)}}
			}
			turns = exhausted.Exhausted.StepsTaken
		}
		result = RunResult{
			Kind:         RunFailure,
			Reason:       HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
			SessionID:    task.SessionID,
			Turns:        turns,
			SessionState: SessionState{Messages: messages},
		}
	}
	// Fold the post-run session back into *session so existing assertions that
	// read the shared state observe the accumulated execute context (matching the
	// legacy in-place mutation).
	if result.Kind == RunSuccess || result.Kind == RunFailure {
		*session = result.SessionState
	}
	return result
}
