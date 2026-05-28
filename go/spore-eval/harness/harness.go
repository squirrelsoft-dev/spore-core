// Package harness defines the EvalHarness runner (Rules 14-25), its fluent
// builder, and the deferred TraceAnalyzer interface (Rule 30).
package harness

import (
	"context"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/guideregistry"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/metricmap"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/report"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/stats"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/verifier"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/worktree"
)

// ============================================================================
// TraceAnalyzer (Rule 30 — interface only, no built-in impl ships)
// ============================================================================

// HarnessConfigDiff is a proposed change to a HarnessConfig produced by a
// TraceAnalyzer. Marker stub: the optimization loop (propose -> run -> compare
// -> open PR) is deferred (Rule 30).
type HarnessConfigDiff struct {
	// Description is a free-form human-readable description of the change.
	Description string
}

// TraceAnalyzer analyzes failure traces and proposes candidate config diffs
// (Rule 30). Interface only — no built-in implementation ships in the MVP.
type TraceAnalyzer interface {
	Analyze(ctx context.Context, traces []observability.Span) []HarnessConfigDiff
}

// ============================================================================
// EvalHarness
// ============================================================================

// candidateConfig pairs a candidate config id with its HarnessConfig.
type candidateConfig struct {
	id     task.ConfigID
	config sporecore.HarnessConfig
}

// EvalHarness runs a task suite against a baseline and candidate configs,
// aggregates metrics, and compares them.
type EvalHarness struct {
	taskSuite           task.TaskSuite
	baselineConfig      sporecore.HarnessConfig
	candidateConfigs    []candidateConfig
	nRunsPerConfig      uint32
	metrics             []metricmap.EvalMetric
	observability       observability.ObservabilityProvider
	bootstrapIterations uint32
	primaryMetric       metricmap.EvalMetric
	verifiers           map[sporecore.TaskID]verifier.TaskVerifier
}

const baselineConfigID = "baseline"

// Run executes the full comparison (Rules 14-25). It produces one
// ComparisonReport per candidate config.
func (h *EvalHarness) Run(ctx context.Context) ([]report.ComparisonReport, error) {
	if len(h.metrics) == 0 {
		return nil, &task.MissingMetricsError{Msg: "no metrics configured for comparison"}
	}

	baseline, err := h.runConfig(ctx, h.baselineConfig, baselineConfigID)
	if err != nil {
		return nil, err
	}

	reports := make([]report.ComparisonReport, 0, len(h.candidateConfigs))
	for _, cc := range h.candidateConfigs {
		candidate, err := h.runConfig(ctx, cc.config, string(cc.id))
		if err != nil {
			return nil, err
		}
		reports = append(reports, h.compare(baseline, candidate, cc.id))
	}
	return reports, nil
}

// runConfig runs every task nRunsPerConfig times for one config, collecting
// per-metric samples and trace links for interesting runs.
func (h *EvalHarness) runConfig(ctx context.Context, config sporecore.HarnessConfig, configID string) (*configSamples, error) {
	samples := newConfigSamples(h.metrics)
	for _, ct := range h.taskSuite.AllTasks() {
		for runIdx := uint32(0); runIdx < h.nRunsPerConfig; runIdx++ {
			if err := h.runOne(ctx, config, configID, ct.Task, runIdx, samples); err != nil {
				return nil, err
			}
		}
	}
	return samples, nil
}

// runOne executes a single (config, task) run (Rules 2-3, 14-18, 25).
func (h *EvalHarness) runOne(ctx context.Context, config sporecore.HarnessConfig, configID string, t *task.EvalTask, runIdx uint32, samples *configSamples) error {
	// Rule 2: fresh workspace restored from the snapshot.
	ws, err := worktree.Restore(ctx, t.WorkspaceSnapshot)
	if err != nil {
		return err
	}
	// Rule 3: workspace torn down regardless of outcome.
	defer func() { _ = ws.Close() }()

	maxTurns := uint32(20)
	if t.ExpectedTurns != nil {
		maxTurns = t.ExpectedTurns[1]
	}
	sessionID := sporecore.SessionID(configID + "-" + string(t.ID) + "-" + itoa(runIdx))
	timeout := time.Duration(t.TimeoutSecs) * time.Second
	if timeout <= 0 {
		timeout = time.Millisecond
	}

	coreTask := sporecore.Task{
		ID:          sporecore.TaskID(configID + "-" + string(t.ID) + "-" + itoa(runIdx)),
		Instruction: t.Instruction,
		SessionID:   sessionID,
		Budget: sporecore.BudgetLimits{
			MaxTurns:    &maxTurns,
			MaxWallTime: &timeout,
		},
		LoopStrategy: sporecore.LoopStrategy{Kind: sporecore.StrategyReAct, MaxIterations: maxTurns},
	}

	hs := sporecore.NewStandardHarness(config)

	// Rule 15: run the harness. Rule 4: timeout bounds a single run and yields a
	// failed run rather than a panic — guard with a context deadline + a
	// goroutine so a hung run still produces a Failure.
	runResult := runWithTimeout(ctx, hs, coreTask, timeout, sessionID)

	// Rule 16: read metrics from observability (do not recompute).
	sessionMetrics, _ := h.observability.GetSessionMetrics(ctx, sessionID)
	if sessionMetrics == nil {
		sessionMetrics = emptySessionMetrics(sessionID, coreTask.ID)
	}
	trace, _ := h.observability.GetTrace(ctx, sessionID)

	// Run the verifier (Rules 7-13).
	v := h.verifierFor(t)
	verification, err := v.Verify(ctx, t, runResult, ws.Path())
	if err != nil {
		return err
	}

	// Rule 18: WaitingForHuman counts as neither success nor failure; reported
	// separately and excluded from success-rate / score samples.
	waiting := runResult.Kind == sporecore.RunWaitingForHuman
	if waiting {
		samples.waitingForHuman++
	}

	inputs := metricmap.RunSampleInputs{VerifierPassed: verification.Passed, VerifierScore: verification.Score}
	for i, metric := range h.metrics {
		if waiting && (metric.Kind == metricmap.MetricTaskSuccessRate || metric.Kind == metricmap.MetricVerificationScore) {
			continue
		}
		samples.perMetric[i] = append(samples.perMetric[i], metricmap.SampleFor(metric, sessionMetrics, trace, inputs))
	}

	// Rule 25: collect trace links for failed or non-passing runs.
	if !verification.Passed || runResult.Kind == sporecore.RunFailure {
		samples.traceLinks = append(samples.traceLinks, string(sessionID))
	}
	return nil
}

// runWithTimeout runs the harness, returning a Failure RunResult if the run
// exceeds the timeout (Rule 4) rather than hanging or panicking.
func runWithTimeout(ctx context.Context, hs *sporecore.StandardHarness, t sporecore.Task, timeout time.Duration, sessionID sporecore.SessionID) sporecore.RunResult {
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	done := make(chan sporecore.RunResult, 1)
	go func() {
		done <- hs.Run(runCtx, sporecore.NewHarnessRunOptions(t))
	}()
	select {
	case r := <-done:
		return r
	case <-runCtx.Done():
		return sporecore.RunResult{
			Kind:      sporecore.RunFailure,
			Reason:    sporecore.HaltReason{Kind: sporecore.HaltBudgetExceeded, LimitType: sporecore.BudgetLimitWallTime},
			SessionID: sessionID,
		}
	}
}

// compare compares baseline vs candidate samples (Rules 19-25).
func (h *EvalHarness) compare(baseline, candidate *configSamples, configID task.ConfigID) report.ComparisonReport {
	comparisons := make([]report.MetricComparison, 0, len(h.metrics))
	for i, metric := range h.metrics {
		base := baseline.perMetric[i]
		cand := candidate.perMetric[i]
		baseStats := stats.StatsFromSamples(base) // Rule 19
		candStats := stats.StatsFromSamples(cand)
		delta := candStats.Mean - baseStats.Mean
		welch := stats.WelchTTest(base, cand)                                  // Rule 20
		direction := report.ClassifyDirection(delta, metric.Direction(), 1e-9) // Rule 22

		// Rule 21: bootstrap CI for metrics from non-deterministic verifiers.
		var ci *stats.ConfidenceInterval
		if h.metricIsNonDeterministic(metric) {
			if c, ok := stats.BootstrapCI(cand, h.bootstrapIterations, 0.95, stats.DefaultBootstrapSeed); ok {
				ci = &c
			}
		}

		comparisons = append(comparisons, report.MetricComparison{
			MetricName: metric.Name(),
			Baseline:   baseStats,
			Candidate:  candStats,
			Delta:      delta,
			PValue:     welch.PValue,
			CI:         ci,
			Direction:  direction,
		})
	}

	recommendation := report.DeriveRecommendation(string(configID), comparisons, h.primaryMetric) // Rules 23-24

	// Rule 25.
	traceLinks := append([]string{}, candidate.traceLinks...)
	traceLinks = append(traceLinks, baseline.traceLinks...)

	return report.ComparisonReport{
		BaselineConfigID:  baselineConfigID,
		CandidateConfigID: string(configID),
		Metrics:           comparisons,
		Recommendation:    recommendation,
		TraceLinks:        traceLinks,
	}
}

// metricIsNonDeterministic reports whether a metric should carry a bootstrap CI
// (Rule 21): metrics derived from non-deterministic verifiers (any task whose
// verifier reports IsDeterministic() == false).
func (h *EvalHarness) metricIsNonDeterministic(metric metricmap.EvalMetric) bool {
	if metric.Kind != metricmap.MetricTaskSuccessRate && metric.Kind != metricmap.MetricVerificationScore {
		return false
	}
	for _, ct := range h.taskSuite.AllTasks() {
		if !h.verifierFor(ct.Task).IsDeterministic() {
			return true
		}
	}
	return false
}

// verifierFor returns the resolved verifier for a task, using an injected
// override if present, otherwise building from the task's spec.
func (h *EvalHarness) verifierFor(t *task.EvalTask) verifier.TaskVerifier {
	if h.verifiers != nil {
		if v, ok := h.verifiers[t.ID]; ok {
			return v
		}
	}
	return verifier.BuildVerifier(t.VerifierSpec)
}

func emptySessionMetrics(sessionID sporecore.SessionID, taskID sporecore.TaskID) *observability.SessionMetrics {
	return &observability.SessionMetrics{
		SessionID:     sessionID,
		TaskID:        taskID,
		Outcome:       guideregistry.NewOutcomePartial(),
		GuidesUsed:    []observability.GuideID{},
		PatchesByTool: map[string]uint32{},
	}
}

// configSamples holds per-config metric samples + the metadata the comparison
// needs.
type configSamples struct {
	perMetric       [][]float64
	traceLinks      []string
	waitingForHuman uint32
}

func newConfigSamples(metrics []metricmap.EvalMetric) *configSamples {
	per := make([][]float64, len(metrics))
	return &configSamples{perMetric: per}
}

func itoa(n uint32) string {
	if n == 0 {
		return "0"
	}
	var buf [10]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
