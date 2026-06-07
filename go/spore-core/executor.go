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
	"strconv"
	"time"
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
	// the shared budget, on the RESOLVED worker agent (#124 — the leaf no longer
	// reads a default config agent; the recursing ReactConfig.Run resolves
	// c.Agent from the registry and threads it here). The leaf primitive (the
	// body of runReActInner). Does NOT finalize observability — the caller (the
	// leaf Run) does.
	ReactWindow(ctx context.Context, task Task, maxIterations uint32, session SessionState, budget BudgetSnapshot, onStream StreamSink, agent Agent) RunResult

	// ResolveWorkerAgent resolves the worker agent for a LoopStrategy tree from
	// the ExecutionRegistry (#124): the agent on the LEAF reached by descending
	// the worker child chain (ReAct.agent; combinators descend into inner /
	// execute; a Ralph with a non-empty agent override resolves THAT — Q3).
	// Returns the resolved agent, or a typed UnresolvedHandle failure RunResult.
	ResolveWorkerAgent(ls *LoopStrategy) (Agent, *RunResult)

	// WorkspaceRoot returns the sandbox workspace root (the verifier-input and
	// HillClimbing TSV root). Empty when no sandbox is wired.
	WorkspaceRoot() string

	// AppendUserMessage seeds a user message onto session via the ContextManager
	// seam (alias of SeedUserMessage for the combinator bodies that thread the
	// session by value).
	AppendUserMessage(ctx context.Context, session *SessionState, text string)

	// EvaluatePhase runs a SelfVerifying evaluate phase (#124, Q1c): a fresh
	// evaluator RUN over a read-only sandbox in a never-shared session, on the
	// RESOLVED evalAgent (the inner worker's agent). Folds the evaluate run's
	// usage into totalUsage / carried (R8) and returns its terminal RunResult.
	EvaluatePhase(ctx context.Context, task *Task, evalAgent Agent, carried *BudgetSnapshot, totalUsage *AggregateUsage) RunResult

	// RalphSeedSession builds a FRESH per-window session re-seeded from the
	// .spore/ filesystem checkpoint (and the optional VCS history block) plus the
	// instruction (#124, Ralph R2/R3). Returns the seeded SessionState.
	RalphSeedSession(ctx context.Context, instruction string) SessionState

	// RalphCompletionStatus is the Ralph external completion check (#124): reads
	// the .spore/ state and reports (reason, incomplete). incomplete=false means
	// the task is complete (Success); true means reset into the next window.
	RalphCompletionStatus() (reason string, incomplete bool)

	// RalphMaxResets returns the configured Ralph outer-loop reset cap (B3).
	RalphMaxResets() uint32

	// HillEvaluate runs one HillClimbing metric evaluation on the resolved
	// evaluator over a fresh SessionState (#124). On success ok is true and
	// (value, dur) carry the result; on failure ok is false and (errStatus,
	// errMsg) carry the typed failure.
	HillEvaluate(ctx context.Context, evaluator MetricEvaluator, sessionID SessionID, taskID TaskID) (value float64, dur time.Duration, errStatus HillClimbIterationStatus, errMsg string, ok bool)

	// HillRevert reverts the working tree to HEAD through the sandbox for a
	// no-improvement HillClimbing iteration (Decision 1). Best-effort.
	HillRevert(ctx context.Context)

	// HillCommitHash resolves the commit_hash recorded on a TSV row (Decision 1;
	// v1 always empty).
	HillCommitHash(ctx context.Context) string

	// HillEmitIteration emits one fire-and-forget per-iteration HillClimbing
	// observability span. spanSeq is advanced. No-op when no provider is wired.
	HillEmitIteration(ctx context.Context, sessionID SessionID, taskID TaskID, spanSeq *uint64, iteration uint32, metricValue float64, hasMetric bool, delta float64, hasDelta bool, status HillClimbIterationStatus, reverted bool)

	// HillWriteTSV serializes the HillClimbing results log to
	// {workspace_root}/.spore/results/{task_id}.tsv (Decisions 2/3). Best-effort.
	HillWriteTSV(workspaceRoot string, taskID TaskID, rows []HillClimbRow)

	// SeedUserMessage seeds a user message onto session (the ContextManager seam).
	// Used by the recursive PlanExecuteConfig.Run to seed the planning directive
	// and each step instruction (#124).
	SeedUserMessage(ctx context.Context, session *SessionState, text string)

	// PlanDirective returns the planning directive seeded before the plan
	// sub-strategy runs (R1) — the "respond with a single JSON plan" instruction
	// wrapped around the task instruction.
	PlanDirective(instruction string) string

	// RunPlanSubtree dispatches the plan sub-strategy plan for planTask over
	// planSession, returning its terminal RunResult (#124). Genuinely recursive —
	// the child's Run drives its whole loop. Routes the configured PlannerAgent
	// (R5/R6) by running the child against an agent-swapped child harness when one
	// is set; otherwise the default agent runs the plan turn. Returns nil only if
	// the child produced no terminal.
	RunPlanSubtree(ctx context.Context, plan *LoopStrategy, planTask Task, planSession SessionState, budget BudgetSnapshot) *RunResult

	// CapturePlanArtifact captures + persists a PlanArtifact from the plan
	// sub-strategy's final output text (#124): R3 (parse), R11 (fire OnPlanCreated,
	// mutable), R4 (persist to the RunStore under PlanExecuteExtrasKey). The model
	// turn that produced planOutput ran elsewhere — the recursive plan child — so
	// this carries no agent call. Returns the captured outcome, or a non-nil
	// terminal failure to propagate.
	CapturePlanArtifact(ctx context.Context, sessionID SessionID, planOutput string, usage AggregateUsage, turns uint32) (PlanPhaseOutcome, *RunResult)

	// ReconcileCompletedTasks marks every task already Completed on the DURABLE
	// RunStore checkpoint as Completed in taskList so it is NOT re-run (A.6
	// deep-resume).
	ReconcileCompletedTasks(ctx context.Context, sessionID SessionID, taskList *TaskList)

	// FireTaskAdvance fires the OnTaskAdvance hook (pre, mutable) for an execute
	// step. The hook may rewrite stepTask.Instruction; the (possibly mutated)
	// instruction is what the execute sub-strategy then runs.
	FireTaskAdvance(ctx context.Context, sessionID SessionID, stepTask *Task, taskIndex, totalTasks int)

	// PersistTaskList persists a parsed task list through the RunStore seam.
	PersistTaskList(ctx context.Context, sessionID SessionID, taskList TaskList)

	// LoadTaskList reads the persisted runnable TaskList (with real blockers) from
	// the RunStore under TaskListExtrasKey (#126, decision C — the ONE authoring
	// path). Returns (list, true) on a hit, or (TaskList{}, false) on a storage
	// miss / decode failure (the executor then falls back to the linear plan
	// artifact bridge).
	LoadTaskList(ctx context.Context, sessionID SessionID) (TaskList, bool)

	// ClearObservedWrites clears the harness-observed write/edit accumulator
	// (#126, AC2). The DAG executor calls this before each step so a task's
	// files_touched reflect ONLY the writes that step issues.
	ClearObservedWrites()

	// TakeObservedWrites drains and returns the harness-observed write/edit paths
	// accumulated at the dispatch seam since the last clear (#126, AC2). Used by
	// the DAG executor on task completion to build a StepLedgerEntry's
	// files_touched — never a model-self-reported field.
	TakeObservedWrites() []string

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

// takeChildOverride takes the FULL terminal RunResult a child strategy stashed
// into Scratch.TerminalOverride when it returned from Run (#124). A combinator
// that recurses per-phase / per-task calls this immediately after each child Run
// to fold the child's usage / turns / session back into the shared execute
// context. Clearing the override is REQUIRED: the combinator builds its OWN
// terminal once the whole loop finishes (via finish), and a stale child override
// would otherwise propagate verbatim and mask it.
func (cx *ExecutionContext) takeChildOverride() *RunResult {
	r := cx.Scratch.TerminalOverride
	cx.Scratch.TerminalOverride = nil
	return r
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

// ============================================================================
// Per-node budget enforcement + failure isolation helpers (#125)
// ============================================================================
//
// promoteBudgetExhausted is the BudgetExhausted -> StrategyOutcome promotion
// boundary. The *PartialJSON helpers build the node-concrete partial_output
// (fork #2: a JSON-serialized string of the per-node partial). Per fork #1, an
// Escalate-resolved exhaustion carries a non-nil partial; a Fail-resolved one
// carries nil. lastFinalResponseText extracts the ReAct window's last
// FinalResponse text from a terminal RunResult.

// promoteBudgetExhausted promotes a charge-time *BudgetExhausted to a
// StrategyOutcome BudgetExhausted variant (#125 promotion boundary), attaching
// partialOutput. Per fork #1 the caller passes a non-nil partial for an Escalate
// resolution and nil for a Fail resolution.
func promoteBudgetExhausted(err *BudgetExhausted, partialOutput *string) StrategyOutcome {
	return StrategyBudgetExhausted(*err, partialOutput)
}

// lastFinalResponseText returns the last FinalResponse text from a ReAct window
// terminal (#125, fork #2): the Success.Output, or for a Failure the last
// assistant text message on its post-run session state (the partial captured
// before exhaustion). Empty for non-terminal pauses.
func lastFinalResponseText(result RunResult) string {
	switch result.Kind {
	case RunSuccess:
		return result.Output
	case RunFailure:
		msgs := result.SessionState.Messages
		for i := len(msgs) - 1; i >= 0; i-- {
			m := msgs[i]
			if m.Role == RoleAssistant && m.Content.Type == ContentTypeText {
				return m.Content.Text
			}
		}
		return ""
	default:
		return ""
	}
}

// reactPartialJSON builds the ReAct partial: the window's last FinalResponse text
// (#125, fork #2).
func reactPartialJSON(lastFinalResponse string) string {
	b, _ := json.Marshal(struct {
		Node              string `json:"node"`
		LastFinalResponse string `json:"last_final_response"`
	}{"react", lastFinalResponse})
	return string(b)
}

// planExecutePartialJSON builds the PlanExecute partial: the task list + per-task
// statuses + ledger (#125, fork #2). The ledger is one (id, description, status)
// row per task.
func planExecutePartialJSON(taskList TaskList) string {
	type ledgerRow struct {
		ID          string `json:"id"`
		Description string `json:"description"`
		Status      string `json:"status"`
	}
	ledger := make([]ledgerRow, 0, len(taskList.Tasks))
	for _, t := range taskList.Tasks {
		ledger = append(ledger, ledgerRow{
			ID:          strconv.FormatUint(uint64(t.ID), 10),
			Description: t.Description,
			Status:      string(t.Status),
		})
	}
	b, _ := json.Marshal(struct {
		Node   string      `json:"node"`
		Tasks  int         `json:"tasks"`
		Ledger []ledgerRow `json:"ledger"`
	}{"plan_execute", len(taskList.Tasks), ledger})
	return string(b)
}

// selfVerifyingPartialJSON builds the SelfVerifying partial: the last worker
// result summary + the last verdict reason (#125, fork #2).
func selfVerifyingPartialJSON(lastWorkerOutput, lastVerdict string) string {
	b, _ := json.Marshal(struct {
		Node             string `json:"node"`
		LastWorkerResult string `json:"last_worker_result"`
		LastVerdict      string `json:"last_verdict"`
	}{"self_verifying", lastWorkerOutput, lastVerdict})
	return string(b)
}

// hillClimbingPartialJSON builds the HillClimbing partial: the best candidate
// value + its score (#125, fork #2).
func hillClimbingPartialJSON(bestScore float64) string {
	b, _ := json.Marshal(struct {
		Node          string  `json:"node"`
		BestCandidate float64 `json:"best_candidate"`
		Score         float64 `json:"score"`
	}{"hill_climbing", bestScore, bestScore})
	return string(b)
}

// currentExhausted snapshots the current budget scope into a *BudgetExhausted for
// promotion (#125). It returns nil when no scope is pushed (defensive — the live
// bodies always push before charging).
func (cx *ExecutionContext) currentExhausted() *BudgetExhausted {
	scope := cx.Budgets.Current()
	if scope == nil {
		return nil
	}
	return &BudgetExhausted{
		Policy:        scope.Policy,
		Behavior:      scope.Behavior,
		StepsTaken:    scope.StepsTaken,
		ContinuesUsed: scope.ContinuesUsed,
		Phase:         scope.Phase,
	}
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
