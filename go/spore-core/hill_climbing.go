// hill_climbing.go — the HillClimbing loop strategy (issue #60).
//
// Iterative optimization loop. Establishes a baseline metric (iteration 0, NO
// agent turn), then for each subsequent iteration proposes a change via a
// bounded ReAct agent sub-run, evaluates the metric, and keeps or reverts the
// change based on whether it strictly improved the metric (ShouldKeep). Honors
// the strategy payload's direction, max_stagnation, revert_on_no_improvement,
// and min_improvement_delta. The HARNESS (not the agent) writes the per-run
// results log to {workspace_root}/.spore/results/{task_id}.tsv.
//
// Mirrors the Rust reference (rust/crates/spore-core/src/harness.rs
// run_hill_climbing and helpers, commit 5da525f) but is written idiomatically
// for Go. Go-specific divergences from the Rust reference, each following an
// established pattern in this repo:
//
//   - MetricEvaluator is a CONSUMER-SIDE interface defined here in root-package
//     terms (mirroring Verifier / VcsProvider / RunStore). The standard
//     metric.MetricEvaluator family (#23) lives in the `metric` package, which
//     imports sporecore; sporecore cannot import it back (cycle). The
//     metric.AsHarnessMetricEvaluator adapter bridges a metric.MetricEvaluator
//     into this seam. HillClimbMetricResult / HillClimbMetricError /
//     HillClimbIterationStatus are the root-package mirrors of
//     metric.MetricResult / metric.MetricError / metric.IterationStatus.
//
//   - The TSV is rendered by a pure root-package function over root-package
//     rows (hillClimbRow) so the exact byte content is unit-testable without an
//     import of the metric package.
//
//   - The per-iteration observability span is emitted through the OPTIONAL
//     EmitHillClimbingIteration method on the HarnessObserver seam (mirroring
//     EmitCompactionVerificationFailed), never affecting control flow.
//
// Seven pinned spec decisions (resolved with the user — NOT relitigated):
//   1. REVERT: revert_on_no_improvement: true reverts via the sandbox
//      `git reset --hard HEAD` directly from the harness. The harness NEVER git
//      commits; commit_hash in the TSV stays empty unless a VcsProvider supplies
//      it (v1: always empty).
//   2. TSV FLOAT FORMAT: both metric_value and duration_secs are formatted with
//      EXACTLY 6 decimal places for cross-language byte-identity.
//   3. TSV SCHEMA: {workspace_root}/.spore/results/{task_id}.tsv, tab-separated,
//      REQUIRED header then one row per iteration ascending. Columns: iteration,
//      commit_hash, metric_value, direction, status, duration_secs, description.
//      commit_hash empty when no VCS; metadata EXCLUDED; metric_value EMPTY on
//      crashed/timeout rows; direction/status snake_case.
//   4. DIRECTION: the strategy payload direction is authoritative for the
//      keep/revert decision via ShouldKeep; the evaluator's Direction() is
//      descriptive only. The TSV direction column records the payload direction.
//   5. BASELINE: iteration 0 is a pure baseline (no agent turn); its row has
//      status kept and duration_secs = the baseline evaluator-call time.
//   6. MISCONFIGURATION: a nil MetricEvaluator => halt
//      HaltHillClimbingMisconfigured. No panic.
//   7. BASELINE ERROR: if the iteration-0 baseline evaluation itself errors,
//      treat as HaltHillClimbingMisconfigured (no current best to climb from),
//      NOT a stagnation increment.

package sporecore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ============================================================================
// MetricEvaluator seam (consumer-side; issue #60 / #23 bridge)
// ============================================================================

// HillClimbIterationStatus is the per-iteration outcome the harness records in
// the results log. Mirrors metric.IterationStatus (snake_case wire values).
type HillClimbIterationStatus string

const (
	// HillClimbKept — metric improved, change kept.
	HillClimbKept HillClimbIterationStatus = "kept"
	// HillClimbDiscarded — metric did not improve, change reverted.
	HillClimbDiscarded HillClimbIterationStatus = "discarded"
	// HillClimbCrashed — evaluator returned a crashed (or any non-timeout) error.
	HillClimbCrashed HillClimbIterationStatus = "crashed"
	// HillClimbTimeout — evaluator returned a timeout error.
	HillClimbTimeout HillClimbIterationStatus = "timeout"
)

// HillClimbMetricResult is the successful outcome of a metric evaluation, in
// root-package terms. Mirrors the value/duration fields of metric.MetricResult
// the harness loop needs (raw output and metadata are not consumed here).
type HillClimbMetricResult struct {
	Value    float64
	Duration time.Duration
}

// HillClimbMetricError is the failure outcome of a metric evaluation, in
// root-package terms. Status is the iteration status the harness records for it
// (HillClimbCrashed or HillClimbTimeout); Message is a human-readable reason
// (used for the baseline-error halt, Decision 7).
type HillClimbMetricError struct {
	Status  HillClimbIterationStatus
	Message string
}

// MetricEvaluator is the consumer-side scoring seam for the HillClimbing loop
// strategy (issue #60). The harness calls Evaluate after each iteration's agent
// turn (and once up front for the iteration-0 baseline) and feeds the result
// into ShouldKeep.
//
// Defined in root-package terms so the harness loop needs no `metric` import
// (which would be a cycle — metric imports sporecore). The standard
// metric.MetricEvaluator family is adapted into this seam by
// metric.AsHarnessMetricEvaluator. On success Evaluate returns a
// *HillClimbMetricResult and a nil error; on failure it returns nil and a
// non-nil *HillClimbMetricError.
type MetricEvaluator interface {
	Evaluate(ctx context.Context, sandbox SandboxProvider, sessionID SessionID, taskID TaskID, state SessionState) (*HillClimbMetricResult, *HillClimbMetricError)
	// Description is the human-readable evaluator description recorded in the
	// TSV description column.
	Description() string
}

// ============================================================================
// HillClimbing strategy driver
// ============================================================================

// hillClimbRow is one row of the HillClimbing results log, in iteration order.
// A pure root-package mirror of metric.ResultsEntry's TSV-relevant fields so
// renderHillClimbingTSV is unit-testable without the metric package. hasMetric
// is false on crashed/timeout rows (metric_value renders EMPTY, Decision 3).
type hillClimbRow struct {
	iteration   uint32
	commitHash  string
	metricValue float64
	hasMetric   bool
	direction   OptimizationDirection
	status      HillClimbIterationStatus
	duration    time.Duration
	description string
}

// runHillClimbing drives the HillClimbing loop strategy (issue #60).
//
// Config fields read:
//   - config.MetricEvaluator — the scorer. REQUIRED: nil => Failure
//     {HaltHillClimbingMisconfigured} (Decision 6), a typed halt NOT a panic.
//   - config.VcsProvider — reserved; v1 records an EMPTY commit_hash regardless
//     (the harness never commits, Decision 1).
//
// Terminal HaltReason variants produced:
//   - HaltStagnationLimitReached{Iterations, BestMetric} — max_stagnation
//     configured and N consecutive non-improvements occurred.
//   - HaltBudgetExceeded — a turn/token/wall/cost cap tripped per iteration.
//   - HaltHillClimbingMisconfigured{Reason} — nil evaluator (Decision 6) or a
//     baseline-evaluation error (Decision 7).
func (h *StandardHarness) runHillClimbing(
	ctx context.Context,
	task Task,
	budget BudgetSnapshot,
	onStream StreamSink,
) RunResult {
	sessionID := task.SessionID
	workspaceRoot := h.config.Sandbox.WorkspaceRoot()

	// Decision 6: a missing evaluator is a typed halt, not a panic.
	if h.config.MetricEvaluator == nil {
		result := RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind:   HaltHillClimbingMisconfigured,
				Reason: "HillClimbing requires config.MetricEvaluator, but it is nil",
			},
			SessionID: sessionID,
		}
		h.finalizeObservability(ctx, sessionID, TerminalFailure, haltReasonString(result.Reason))
		return result
	}
	evaluator := h.config.MetricEvaluator
	direction := task.LoopStrategy.Direction
	description := evaluator.Description()

	// Cumulative usage + turns across ALL agent-turn iterations.
	var totalUsage AggregateUsage
	// Shared budget threaded across every agent sub-run.
	carried := budget
	// The TSV rows, in iteration order.
	var rows []hillClimbRow
	// Per-iteration observability span counter.
	var spanSeq uint64

	// ── Iteration 0: pure baseline. No agent turn (Decision 5).
	//    HillClimbing keeps no carried message state of its own (each iteration
	//    is a fresh sub-run), so a default SessionState is handed to the
	//    evaluator.
	baseRes, baseErr := evaluator.Evaluate(ctx, h.config.Sandbox, sessionID, task.ID, SessionState{})
	if baseErr != nil {
		// Decision 7: a baseline that cannot be measured is a misconfiguration of
		// the experiment, not a non-improvement to climb away from — there is no
		// current best to compare against. Record the failed row, write the TSV,
		// and halt.
		rows = append(rows, hillClimbRow{
			iteration:   0,
			commitHash:  h.hillClimbingCommitHash(ctx),
			hasMetric:   false,
			direction:   direction,
			status:      baseErr.Status,
			duration:    0,
			description: description,
		})
		h.emitHillClimbingIteration(ctx, sessionID, task.ID, &spanSeq, 0, 0, false, 0, false, baseErr.Status, false)
		h.writeHillClimbingTSV(workspaceRoot, task.ID, rows)
		result := RunResult{
			Kind: RunFailure,
			Reason: HaltReason{
				Kind:   HaltHillClimbingMisconfigured,
				Reason: fmt.Sprintf("baseline evaluation failed: %s", baseErr.Message),
			},
			SessionID: sessionID,
			Usage:     totalUsage,
			Turns:     carried.Turns,
		}
		h.finalizeObservability(ctx, sessionID, TerminalFailure, haltReasonString(result.Reason))
		return result
	}
	currentBest := baseRes.Value
	rows = append(rows, hillClimbRow{
		iteration:   0,
		commitHash:  h.hillClimbingCommitHash(ctx),
		metricValue: baseRes.Value,
		hasMetric:   true,
		direction:   direction,
		status:      HillClimbKept,
		duration:    baseRes.Duration,
		description: description,
	})
	h.emitHillClimbingIteration(ctx, sessionID, task.ID, &spanSeq, 0, baseRes.Value, true, 0, false, HillClimbKept, false)

	// Consecutive non-improvement counter (drives the stagnation halt).
	var stagnation uint32
	// The 0-based iteration index; agent turns begin at 1.
	iteration := uint32(1)

	for {
		// Budget gate before the iteration's agent turn (mirrors runReAct).
		turnCap := allTurns
		if task.Budget.MaxTurns != nil {
			turnCap = *task.Budget.MaxTurns
		}
		if carried.Turns >= turnCap {
			break
		}
		if lt, exceeded := budgetExceeded(task.Budget, carried, time.Now()); exceeded {
			h.writeHillClimbingTSV(workspaceRoot, task.ID, rows)
			result := RunResult{
				Kind:      RunFailure,
				Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: lt},
				SessionID: sessionID,
				Usage:     totalUsage,
				Turns:     carried.Turns,
			}
			h.finalizeObservability(ctx, sessionID, TerminalFailure, haltReasonString(result.Reason))
			return result
		}

		// ── One bounded agent turn proposes a change. The sub-run carries the
		//    shared budget so per-iteration turns count toward the cap.
		iterTask := Task{
			ID:           task.ID,
			Instruction:  task.Instruction,
			SessionID:    sessionID,
			Budget:       task.Budget,
			LoopStrategy: task.LoopStrategy,
		}
		var iterState SessionState
		h.config.ContextManager.AppendUserMessage(ctx, &iterState, task.Instruction)
		turnResult := h.runReActInner(ctx, iterTask, turnCap, iterState, carried, onStream)
		foldSelfVerifyUsage(&totalUsage, &carried, turnResult)

		// A turn that paused / escalated is propagated up unchanged.
		switch turnResult.Kind {
		case RunWaitingForHuman:
			// Not terminal — do not finalize or write the TSV.
			return turnResult
		case RunEscalate:
			h.writeHillClimbingTSV(workspaceRoot, task.ID, rows)
			h.finalizeObservability(ctx, turnResult.SessionID, TerminalEscalated, "")
			return turnResult
		}

		// ── Evaluate the metric after the change.
		evalRes, evalErr := evaluator.Evaluate(ctx, h.config.Sandbox, sessionID, task.ID, SessionState{})
		if evalErr != nil {
			// Crash/timeout/etc.: counts as a non-improvement. Optionally revert,
			// increment stagnation, record an empty-metric row.
			reverted := false
			if task.LoopStrategy.RevertOnNoImprovement {
				h.hillClimbingRevert(ctx)
				reverted = true
			}
			stagnation++
			rows = append(rows, hillClimbRow{
				iteration:   iteration,
				commitHash:  h.hillClimbingCommitHash(ctx),
				hasMetric:   false,
				direction:   direction,
				status:      evalErr.Status,
				duration:    0,
				description: description,
			})
			h.emitHillClimbingIteration(ctx, sessionID, task.ID, &spanSeq, iteration, 0, false, 0, false, evalErr.Status, reverted)
		} else {
			value := evalRes.Value
			kept := hillClimbShouldKeep(value, currentBest, direction, task.LoopStrategy.MinImprovementDelta)
			var delta float64
			switch direction {
			case OptimizationMinimize:
				delta = currentBest - value
			default:
				delta = value - currentBest
			}
			if kept {
				currentBest = value
				stagnation = 0
				rows = append(rows, hillClimbRow{
					iteration:   iteration,
					commitHash:  h.hillClimbingCommitHash(ctx),
					metricValue: value,
					hasMetric:   true,
					direction:   direction,
					status:      HillClimbKept,
					duration:    evalRes.Duration,
					description: description,
				})
				h.emitHillClimbingIteration(ctx, sessionID, task.ID, &spanSeq, iteration, value, true, delta, true, HillClimbKept, false)
			} else {
				// No improvement (Decision 1: optionally revert).
				reverted := false
				if task.LoopStrategy.RevertOnNoImprovement {
					h.hillClimbingRevert(ctx)
					reverted = true
				}
				stagnation++
				rows = append(rows, hillClimbRow{
					iteration:   iteration,
					commitHash:  h.hillClimbingCommitHash(ctx),
					metricValue: value,
					hasMetric:   true,
					direction:   direction,
					status:      HillClimbDiscarded,
					duration:    evalRes.Duration,
					description: description,
				})
				h.emitHillClimbingIteration(ctx, sessionID, task.ID, &spanSeq, iteration, value, true, delta, true, HillClimbDiscarded, reverted)
			}
		}

		// ── Stagnation halt (only when a cap is configured).
		if task.LoopStrategy.MaxStagnation != nil && stagnation >= *task.LoopStrategy.MaxStagnation {
			h.writeHillClimbingTSV(workspaceRoot, task.ID, rows)
			result := RunResult{
				Kind: RunFailure,
				Reason: HaltReason{
					Kind:       HaltStagnationLimitReached,
					Iterations: stagnation,
					BestMetric: currentBest,
				},
				SessionID: sessionID,
				Usage:     totalUsage,
				Turns:     carried.Turns,
			}
			h.finalizeObservability(ctx, sessionID, TerminalFailure, haltReasonString(result.Reason))
			return result
		}

		// iteration is bounded by the turn budget; saturating add guards overflow.
		if iteration < allTurns {
			iteration++
		}
	}

	// Budget/turn cap reached without a stagnation halt — clean budget halt.
	h.writeHillClimbingTSV(workspaceRoot, task.ID, rows)
	result := RunResult{
		Kind:      RunFailure,
		Reason:    HaltReason{Kind: HaltBudgetExceeded, LimitType: BudgetLimitTurns},
		SessionID: sessionID,
		Usage:     totalUsage,
		Turns:     carried.Turns,
	}
	h.finalizeObservability(ctx, sessionID, TerminalFailure, haltReasonString(result.Reason))
	return result
}

// hillClimbShouldKeep is the keep-or-revert decision the harness applies after
// each iteration (Decision 4). Mirrors metric.ShouldKeep exactly (the root
// package cannot import metric — cycle): returns true only when newValue
// STRICTLY beats currentBest by more than minDelta (default 0.0 when nil). An
// improvement of exactly minDelta (or an equal score) does NOT count.
func hillClimbShouldKeep(newValue, currentBest float64, direction OptimizationDirection, minDelta *float64) bool {
	var delta float64
	switch direction {
	case OptimizationMinimize:
		delta = currentBest - newValue
	default:
		delta = newValue - currentBest
	}
	threshold := 0.0
	if minDelta != nil {
		threshold = *minDelta
	}
	return delta > threshold
}

// hillClimbingRevert reverts the working tree to current HEAD for a
// no-improvement iteration (issue #60, Decision 1). Runs `git reset --hard HEAD`
// THROUGH the sandbox; the harness NEVER spawns git directly. A sandbox
// rejection / non-zero exit is best-effort: the loop continues (the next agent
// turn re-derives state).
func (h *StandardHarness) hillClimbingRevert(ctx context.Context) {
	_, _ = h.config.Sandbox.ExecuteCommand(ctx, "git", []string{"reset", "--hard", "HEAD"}, "", 0)
}

// hillClimbingCommitHash resolves the commit_hash recorded on a TSV row (issue
// #60, Decision 1). The harness never commits, so this is the EMPTY string
// unless a VcsProvider is wired to supply a hash. v1 has no per-keep commit, so
// it always returns "" (the VcsProvider seam is reserved for a later revision).
func (h *StandardHarness) hillClimbingCommitHash(_ context.Context) string {
	return ""
}

// emitHillClimbingIteration emits one fire-and-forget per-iteration
// observability span for a HillClimbing run (issue #60). No-op when no provider
// is configured. The span id mirrors the Rust reference's
// "{session}-hill-{seq}".
func (h *StandardHarness) emitHillClimbingIteration(
	_ context.Context,
	sessionID SessionID,
	taskID TaskID,
	spanSeq *uint64,
	iteration uint32,
	metricValue float64,
	hasMetric bool,
	delta float64,
	hasDelta bool,
	status HillClimbIterationStatus,
	reverted bool,
) {
	if h.config.Observability == nil {
		return
	}
	spanID := fmt.Sprintf("%s-hill-%d", sessionID, *spanSeq)
	h.config.Observability.EmitHillClimbingIteration(
		spanID, sessionID, taskID, nowRFC3339(),
		iteration, metricValue, hasMetric, delta, hasDelta, string(status), reverted,
	)
	*spanSeq++
}

// writeHillClimbingTSV serializes the HillClimbing results log and writes it to
// {workspace_root}/.spore/results/{task_id}.tsv (issue #60, Decisions 2/3).
// Best-effort: a filesystem error is swallowed (the run outcome is
// authoritative, the TSV is a diagnostic artifact).
func (h *StandardHarness) writeHillClimbingTSV(workspaceRoot string, taskID TaskID, rows []hillClimbRow) {
	body := renderHillClimbingTSV(rows)
	dir := filepath.Join(workspaceRoot, ".spore", "results")
	_ = os.MkdirAll(dir, 0o755)
	_ = os.WriteFile(filepath.Join(dir, string(taskID)+".tsv"), []byte(body), 0o644)
}

// renderHillClimbingTSV renders the HillClimbing results-log TSV body (issue
// #60, Decisions 2/3). Pure function over the rows so the exact byte content is
// unit-testable and cross-language-comparable. REQUIRED header, one row per
// iteration in ascending order, trailing newline after every row (including the
// last) so appends and diffs stay line-oriented. Floats use exactly 6 decimal
// places; metric_value is EMPTY on crashed/timeout rows.
func renderHillClimbingTSV(rows []hillClimbRow) string {
	var b strings.Builder
	b.WriteString("iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n")
	for _, r := range rows {
		metricValue := ""
		if r.hasMetric {
			metricValue = strconv.FormatFloat(r.metricValue, 'f', 6, 64)
		}
		durationSecs := strconv.FormatFloat(r.duration.Seconds(), 'f', 6, 64)
		b.WriteString(fmt.Sprintf(
			"%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
			r.iteration,
			r.commitHash,
			metricValue,
			string(r.direction),
			string(r.status),
			durationSecs,
			r.description,
		))
	}
	return b.String()
}
