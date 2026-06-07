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
// PlanExecute): seed the directive, dispatch plan.Run capped at one turn, then
// capture + persist the artifact. Mirrors the plan half of PlanExecuteConfig.Run
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

	planCap := saturatingAddU32(budget.Turns, 1)
	if task.Budget.MaxTurns != nil && *task.Budget.MaxTurns < planCap {
		planCap = *task.Budget.MaxTurns
	}
	planBudget := task.Budget
	planBudget.MaxTurns = &planCap
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
	result := cfg.runExecuteLoop(ctx, cx, h, task, *session, taskList, carried, planUsage, onStream)
	// Fold the post-run session back into *session so existing assertions that
	// read the shared state observe the accumulated execute context (matching the
	// legacy in-place mutation).
	if result.Kind == RunSuccess || result.Kind == RunFailure {
		*session = result.SessionState
	}
	return result
}
