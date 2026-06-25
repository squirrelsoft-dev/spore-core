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

// HillClimbRow is one row of the HillClimbing results log, in iteration order.
// A pure root-package mirror of metric.ResultsEntry's TSV-relevant fields so
// renderHillClimbingTSV is unit-testable without the metric package. HasMetric
// is false on crashed/timeout rows (metric_value renders EMPTY, Decision 3).
// Exported (#124) so the recursive HillClimbingConfig.Run, which now owns the
// optimization loop, can accumulate rows and hand them to the HillWriteTSV
// executor primitive.
type HillClimbRow struct {
	Iteration   uint32
	CommitHash  string
	MetricValue float64
	HasMetric   bool
	Direction   OptimizationDirection
	Status      HillClimbIterationStatus
	Duration    time.Duration
	Description string
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

// hillClimbingRevert reverts the working tree for a no-improvement iteration
// (issue #60, Decision 1). SC-14: when a VcsProvider is wired it owns the revert
// (e.g. GitVcsProvider runs `git reset --hard HEAD`, a custom provider does
// whatever its workspace needs); with no provider (the default) the harness
// falls back to the original hardcoded `git reset --hard HEAD` THROUGH the
// sandbox — byte-identical to the pre-SC-14 behavior. Either way the harness
// NEVER spawns git directly, and the revert is best-effort: an error / non-zero
// exit is swallowed and the loop continues (the next agent turn re-derives
// state).
func (h *StandardHarness) hillClimbingRevert(ctx context.Context) {
	if h.config.VcsProvider != nil {
		_ = h.config.VcsProvider.Revert(ctx)
		return
	}
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
func (h *StandardHarness) writeHillClimbingTSV(workspaceRoot string, taskID TaskID, rows []HillClimbRow) {
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
func renderHillClimbingTSV(rows []HillClimbRow) string {
	var b strings.Builder
	b.WriteString("iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n")
	for _, r := range rows {
		metricValue := ""
		if r.HasMetric {
			metricValue = strconv.FormatFloat(r.MetricValue, 'f', 6, 64)
		}
		durationSecs := strconv.FormatFloat(r.Duration.Seconds(), 'f', 6, 64)
		b.WriteString(fmt.Sprintf(
			"%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
			r.Iteration,
			r.CommitHash,
			metricValue,
			string(r.Direction),
			string(r.Status),
			durationSecs,
			r.Description,
		))
	}
	return b.String()
}
