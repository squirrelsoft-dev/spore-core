// Per-variant RunStrategy.Run recursive executor (issue #124).
//
// This file owns the runtime seam that lets each strategy config OWN its loop
// while the model-touching machinery stays on the harness:
//
//   - StrategyExecutor — the harness-side primitives the per-variant
//     RunStrategy.Run bodies delegate to (implemented by StandardHarness in
//     harness.go). Mirrors the Rust StrategyExecutor trait.
//   - RunScratch — the per-run mutable orchestration state threaded across the
//     recursive strategy tree (task / session / budget / terminal override).
//     Runtime-only, NEVER serialized.
//   - The ExecutionContext helpers (executor / currentTask / takeSession /
//     takeStream / recordTerminal / finish) the config bodies call.
//   - outcomeFromRunResult — the Q5 RunResult→StrategyOutcome mapping.
//   - InvalidConfigurationError — the typed HarnessError used for the A.5
//     output-contract rejection and the no-executor scaffold failure.
//   - PlanPhaseOutcome — the public PlanExecute plan-phase result surfaced on
//     the StrategyExecutor.PlanPhase primitive.
//
// The central dispatch switch that used to live in StandardHarness.runInner is
// GONE (AC1): the harness entry now collapses to driveStrategy, which builds the
// shared ExecutionContext and calls task.LoopStrategy.Run(ctx, cx). The only
// switch left is the enum→config delegation in LoopStrategy.Run (strategy.go).

package sporecore

import (
	"context"
	"encoding/json"
	"fmt"
)

// ============================================================================
// InvalidConfigurationError — typed startup / configuration HarnessError
// ============================================================================

// InvalidConfigurationError is the typed HarnessError returned for an invalid
// strategy configuration: the A.5 output-contract rejection (a bare ReAct in a
// structured plan/worker/propose slot without an output schema), the
// no-executor scaffold failure, and the Q5 non-success terminal mapping.
// Mirrors Rust's HarnessError::InvalidConfiguration(String).
//
// Serializes as {"kind":"InvalidConfiguration","message":"<msg>"}.
type InvalidConfigurationError struct {
	Message string
}

func (e *InvalidConfigurationError) isHarnessError() {}

// Error implements error. Message mirrors the Rust display impl.
func (e *InvalidConfigurationError) Error() string {
	return fmt.Sprintf("invalid configuration: %s", e.Message)
}

// MarshalJSON serialises as {"kind":"InvalidConfiguration","message":...}.
func (e *InvalidConfigurationError) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Kind    string `json:"kind"`
		Message string `json:"message"`
	}{"InvalidConfiguration", e.Message})
}

// UnmarshalJSON decodes the "InvalidConfiguration" form.
func (e *InvalidConfigurationError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Message = probe.Message
	return nil
}

// ============================================================================
// PlanPhaseOutcome — public PlanExecute plan-phase result
// ============================================================================

// PlanPhaseOutcome is the result of a successful PlanExecute plan phase
// (issue #70), surfaced on the StrategyExecutor.PlanPhase primitive (#124).
// Carries the produced artifact plus the run accounting so the PlanExecuteConfig
// body can build the terminal RunResult. The artifact itself is also observable
// via SessionState.Extras["plan_execute"].
type PlanPhaseOutcome struct {
	Artifact PlanArtifact
	Usage    AggregateUsage
	Turns    uint32
}

// ============================================================================
// StrategyExecutor — the harness-side primitives the config bodies delegate to
// ============================================================================

// StrategyExecutor is the harness-side seam the per-variant RunStrategy.Run
// bodies delegate to (#124). Implemented by StandardHarness. This is what lets
// the recursive config bodies own their loops while the model-touching machinery
// (the ReAct turn-loop window, the evaluate phase, the plan phase, the metric
// machinery, the .spore/ checks) stays where it is tested.
//
// Every primitive returns a terminal RunResult for its phase; the config bodies
// translate the terminal into a StrategyOutcome (or recurse). ctx is the
// standard cancellation context (Go CONVENTIONS), threaded as the first arg.
type StrategyExecutor interface {
	// ReactWindow runs ONE bounded ReAct turn-loop window over session, carrying
	// the shared budget. The leaf primitive (the body of runReActInner). Does
	// NOT finalize observability — the caller (the leaf Run) does.
	ReactWindow(ctx context.Context, task Task, maxIterations uint32, session SessionState, budget BudgetSnapshot, onStream StreamSink) RunResult

	// PlanPhase runs the PlanExecute plan phase (runPlanPhase) — the constrained
	// planner turn that captures + persists a PlanArtifact. Returns the artifact
	// + accounting, or a non-nil terminal failure to propagate.
	PlanPhase(ctx context.Context, task *Task, session *SessionState, budget BudgetSnapshot, onStream StreamSink) (PlanPhaseOutcome, *RunResult)

	// ExecutePhase runs the PlanExecute execute phase (runExecutePhase) — drains
	// the task list, recursing the execute sub-strategy per task.
	ExecutePhase(ctx context.Context, task *Task, session *SessionState, taskList TaskList, carried BudgetSnapshot, planUsage AggregateUsage, onStream StreamSink) RunResult

	// PersistTaskList persists a parsed task list through the RunStore seam.
	PersistTaskList(ctx context.Context, sessionID SessionID, taskList TaskList)

	// SelfVerifyingLoop drives a whole SelfVerifying loop (runSelfVerifying). The
	// build phase reuses the leaf ReAct window; the loop is Default-FAIL and
	// bounded by the verifier's iteration cap (Q1).
	SelfVerifyingLoop(ctx context.Context, task Task, session SessionState, budget BudgetSnapshot, onStream StreamSink) RunResult

	// RalphLoop drives a whole Ralph continuation loop (runRalph). Resets the
	// context window per continuation and resumes from the durable .spore/
	// checkpoint (A.6 deep-resume).
	RalphLoop(ctx context.Context, task Task, budget BudgetSnapshot, onStream StreamSink) RunResult

	// HillClimbingLoop drives a whole HillClimbing loop (runHillClimbing). It
	// reads the config's direction / max_stagnation / delta off task.LoopStrategy.
	HillClimbingLoop(ctx context.Context, task Task, budget BudgetSnapshot, onStream StreamSink) RunResult

	// Finalize finalizes observability for a terminal outcome (the
	// finalizeObservability routing). No-op for non-terminal results.
	Finalize(ctx context.Context, result RunResult)
}

// ============================================================================
// RunScratch — per-run mutable orchestration state threaded through recursion
// ============================================================================

// RunScratch is the per-run mutable orchestration state threaded through the
// recursive strategy tree (#124). Runtime-only — NOT serialized. The combinator
// bodies set up the sub-phase Task here before recursing, and the leaf
// (*ReactConfig).Run reads it to drive the ReAct window.
type RunScratch struct {
	// Task is the task whose strategy is currently executing. Combinators swap in
	// a per-phase sub-task before recursing and restore it after.
	Task *Task
	// RunSession is the conversation/session state the current leaf turn-loop
	// builds on.
	RunSession SessionState
	// RunBudget is the shared budget snapshot threaded across every sub-loop.
	RunBudget BudgetSnapshot
	// StreamTaken records that the run's stream sink has been consumed by a leaf
	// / combinator (the sink is single-use and lives on ExecutionContext.Stream).
	streamTaken bool
	// TerminalOverride is a non-terminal pause (WaitingForHuman/Consult/Escalate)
	// or a fully-formed terminal that must propagate up the recursion VERBATIM as
	// a RunResult rather than being collapsed into a StrategyOutcome. The harness
	// entry (driveStrategy) returns this directly when set, preserving the
	// pause/escalate contract through the recursive executor (#124).
	TerminalOverride *RunResult
}

// ============================================================================
// ExecutionContext recursive-executor helpers (#124)
// ============================================================================

// executor returns the wired StrategyExecutor, or (nil, typed Failed outcome)
// for the scaffold-only contexts that have no real harness. Real harness runs
// always wire one.
func (cx *ExecutionContext) executor() (StrategyExecutor, StrategyOutcome) {
	if cx.Executor == nil {
		return nil, StrategyFailed(&InvalidConfigurationError{
			Message: "ExecutionContext has no StrategyExecutor wired",
		})
	}
	return cx.Executor, StrategyOutcome{}
}

// currentTask returns the per-run task. The harness always sets it before
// driving a strategy; a zero Task is returned only on misuse (the scaffold-only
// no-executor path returns before reading it).
func (cx *ExecutionContext) currentTask() Task {
	if cx.Scratch.Task == nil {
		return Task{}
	}
	return *cx.Scratch.Task
}

// takeSession takes (moves out) the current run session, leaving a zero value.
func (cx *ExecutionContext) takeSession() SessionState {
	s := cx.Scratch.RunSession
	cx.Scratch.RunSession = SessionState{}
	return s
}

// takeStream takes the run's stream sink once (it is single-use). Subsequent
// callers in the same recursion get nil.
func (cx *ExecutionContext) takeStream() StreamSink {
	if cx.Scratch.streamTaken {
		return nil
	}
	cx.Scratch.streamTaken = true
	s := cx.Stream
	cx.Stream = nil
	return s
}

// recordTerminal records a terminal/pause RunResult from a whole-loop primitive
// (ReAct / SelfVerifying / Ralph / HillClimbing): it carries the post-run
// session into the scratch (so a parent resumes losslessly) and stashes the FULL
// result in TerminalOverride so the harness entry returns it VERBATIM —
// preserving the strategy's typed HaltReason and accounting. It returns the
// matchable StrategyOutcome for any combinator that recurses into this node (a
// wrapping combinator clears the override and builds its own terminal via
// finish).
//
// Usage is NOT folded into cx.Usage here: the whole-loop primitive's RunResult
// already carries the cumulative usage for its subtree and is returned verbatim
// as the override, so folding would double-count.
func (cx *ExecutionContext) recordTerminal(result RunResult) StrategyOutcome {
	switch result.Kind {
	case RunSuccess, RunFailure:
		cx.Scratch.RunSession = result.SessionState
	}
	outcome := outcomeFromRunResult(result)
	r := result
	cx.Scratch.TerminalOverride = &r
	return outcome
}

// finish is a combinator's terminal seam: finalize observability for result,
// restore the parent task into scratch, stash result as the override so the
// harness entry returns it VERBATIM, and return the matching outcome.
func (cx *ExecutionContext) finish(ctx context.Context, executor StrategyExecutor, parentTask Task, result RunResult) StrategyOutcome {
	executor.Finalize(ctx, result)
	pt := parentTask
	cx.Scratch.Task = &pt
	switch result.Kind {
	case RunSuccess, RunFailure:
		cx.Scratch.RunSession = result.SessionState
	}
	outcome := outcomeFromRunResult(result)
	r := result
	cx.Scratch.TerminalOverride = &r
	return outcome
}

// outcomeFromRunResult translates a terminal RunResult into a StrategyOutcome
// (#124, Q5): Success → Complete(output); every non-success terminal → Failed. A
// budget-exceeded failure maps to Failed here (the budget-enforcement
// BudgetExhausted value is produced by BudgetContext.Charge at the boundary; full
// HITL-through-recursion is #130). The pause variants are handled separately via
// the override path and degrade to a typed failure only if they ever reach this
// mapping.
func outcomeFromRunResult(result RunResult) StrategyOutcome {
	switch result.Kind {
	case RunSuccess:
		return StrategyComplete(result.Output)
	case RunFailure:
		return StrategyFailed(&InvalidConfigurationError{Message: haltReasonString(result.Reason)})
	default:
		return StrategyFailed(&InvalidConfigurationError{
			Message: "non-terminal outcome reached strategy boundary",
		})
	}
}
