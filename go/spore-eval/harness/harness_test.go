package harness

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/metricmap"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/report"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/verifier"
)

// stuntAgent is a stateless test Agent that returns the same TurnResult every
// turn — fresh per run, so the EvalHarness can re-run it n times without the
// queue draining (unlike sporecore.MockAgent).
type stuntAgent struct {
	id   sporecore.AgentID
	fail bool
}

func (a *stuntAgent) ID() sporecore.AgentID { return a.id }

func (a *stuntAgent) Turn(context.Context, sporecore.Context) sporecore.TurnResult {
	if a.fail {
		u := sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1}
		return sporecore.NewTurnError(sporecore.NewEmptyResponseError(), &u)
	}
	return sporecore.NewFinalResponse("done", sporecore.TokenUsage{InputTokens: 1, OutputTokens: 1})
}

// buildConfig wires a HarnessConfig whose loop emits through the supplied
// in-memory provider (so the EvalHarness reads metrics from the same provider).
func buildConfig(agent sporecore.Agent, provider observability.ObservabilityProvider) sporecore.HarnessConfig {
	return observability.NewHarnessBuilder(
		agent,
		sporecore.NewScriptedToolRegistry(),
		sporecore.AllowAllSandbox{},
		sporecore.NoopContextManager{},
		sporecore.AlwaysContinuePolicy{},
	).Observability(provider).BuildConfig()
}

// alwaysPassSuite is a small hermetic suite using always_pass on an empty
// workspace so verification does not depend on agent file output.
func alwaysPassSuite() task.TaskSuite {
	mk := func(id string) task.EvalTask {
		return task.EvalTask{
			ID:                sporecore.TaskID(id),
			Instruction:       "do it",
			WorkspaceSnapshot: task.WorkspaceSnapshot{Kind: task.SnapshotEmpty},
			VerifierSpec:      task.VerifierSpec{Kind: task.VerifierAlwaysPass},
			TimeoutSecs:       30,
		}
	}
	return task.TaskSuite{
		SuiteVersion: 1,
		Regression:   []task.EvalTask{mk("r1"), mk("r2"), mk("r3")},
	}
}

func TestMissingMetricsErrors(t *testing.T) {
	provider := observability.NewInMemoryObservabilityProvider()
	h := NewEvalHarnessBuilder(alwaysPassSuite(), buildConfig(&stuntAgent{id: "b"}, provider), provider).
		Metrics([]metricmap.EvalMetric{}).
		Build()
	if _, err := h.Run(context.Background()); err == nil {
		t.Fatal("expected MissingMetricsError")
	}
}

// Hermetic: baseline (passing agent) vs deliberately-worse candidate (failing
// agent) must flag the regression — Reject (or NeedsMoreRuns) with a sane
// p-value — on the success-rate primary metric (Rule 23, E2E requirement).
func TestE2ERegressionFlagged(t *testing.T) {
	suite := alwaysPassSuite()
	// The success-rate metric must be sensitive to the run outcome, so use the
	// metric_evaluator placeholder verifier (scores from run success) rather
	// than always_pass. The candidate agent always fails -> success rate 0.
	for i := range suite.Regression {
		suite.Regression[i].VerifierSpec = task.VerifierSpec{Kind: task.VerifierMetricEvaluator, Direction: task.DirectionMaximize}
	}

	// One in-memory provider holds both configs' sessions (session ids are
	// unique per config/task/run).
	provider := observability.NewInMemoryObservabilityProvider()
	baselineCfg := buildConfig(&stuntAgent{id: "base"}, provider)
	candidateCfg := buildConfig(&stuntAgent{id: "cand", fail: true}, provider)

	h := NewEvalHarnessBuilder(suite, baselineCfg, provider).
		Candidate("worse", candidateCfg).
		NRunsPerConfig(3).
		Metrics([]metricmap.EvalMetric{metricmap.TaskSuccessRate(), metricmap.MeanTurns()}).
		PrimaryMetric(metricmap.TaskSuccessRate()).
		Build()

	reports, err := h.Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("want 1 report, got %d", len(reports))
	}
	rep := reports[0]
	if rep.CandidateConfigID != "worse" {
		t.Fatalf("candidate id=%q", rep.CandidateConfigID)
	}

	// Find the success-rate comparison.
	var sr *report.MetricComparison
	for i := range rep.Metrics {
		if rep.Metrics[i].MetricName == metricmap.TaskSuccessRate().Name() {
			sr = &rep.Metrics[i]
		}
	}
	if sr == nil {
		t.Fatal("no success_rate metric")
	}
	// Baseline all pass (1.0), candidate all fail (0.0): delta -1, direction worse.
	if sr.Baseline.Mean != 1.0 {
		t.Fatalf("baseline mean=%v want 1.0", sr.Baseline.Mean)
	}
	if sr.Candidate.Mean != 0.0 {
		t.Fatalf("candidate mean=%v want 0.0", sr.Candidate.Mean)
	}
	if sr.Direction != report.DirectionWorse {
		t.Fatalf("direction=%v want worse", sr.Direction)
	}
	// Constant differing means -> p ~ 0 -> Reject.
	if sr.PValue >= 0.05 {
		t.Fatalf("p=%v should be significant", sr.PValue)
	}
	if rep.Recommendation.Kind != report.RecReject {
		t.Fatalf("recommendation=%+v want reject", rep.Recommendation)
	}
	// Rule 25: failing/non-passing candidate runs collect trace links.
	if len(rep.TraceLinks) == 0 {
		t.Fatal("expected trace links for failing runs (Rule 25)")
	}
}

// Hermetic: restore a Files snapshot into a tempdir and verify a real command
// against it via the EvalHarness end to end (workspace restore + teardown +
// test_suite verification + observability-sourced turn metric).
func TestE2EFilesSnapshotWithCommandVerifier(t *testing.T) {
	provider := observability.NewInMemoryObservabilityProvider()
	suite := task.TaskSuite{
		SuiteVersion: 1,
		Regression: []task.EvalTask{{
			ID:          "files_task",
			Instruction: "noop",
			WorkspaceSnapshot: task.WorkspaceSnapshot{
				Kind:  task.SnapshotFiles,
				Files: map[string]string{"present.txt": "hi\n"},
			},
			// The agent writes nothing; the check passes because the restored
			// file is present — exercises the Files snapshot restore path.
			VerifierSpec: task.VerifierSpec{
				Kind:        task.VerifierTestSuite,
				Command:     "sh",
				Args:        []string{"-c", "test -f present.txt"},
				TimeoutSecs: u64ptr(10),
			},
			TimeoutSecs: 30,
		}},
	}
	cfg := buildConfig(&stuntAgent{id: "base"}, provider)
	h := NewEvalHarnessBuilder(suite, cfg, provider).
		Candidate("c", cfg).
		NRunsPerConfig(2).
		Metrics([]metricmap.EvalMetric{metricmap.TaskSuccessRate()}).
		Build()
	reports, err := h.Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	sr := reports[0].Metrics[0]
	if sr.Baseline.Mean != 1.0 || sr.Candidate.Mean != 1.0 {
		t.Fatalf("expected all pass, got baseline=%v candidate=%v", sr.Baseline.Mean, sr.Candidate.Mean)
	}
}

func u64ptr(v uint64) *uint64 { return &v }

// Rule 29: load the shared core_suite.json and assert it drives the harness
// end-to-end with identical structural outcomes (the manifest is the
// cross-language oracle; never edited to pass).
func TestCoreSuiteFixtureReplayThroughHarness(t *testing.T) {
	path := filepath.Join("..", "..", "..", "fixtures", "task_suites", "core_suite.json")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var suite task.TaskSuite
	if err := json.Unmarshal(body, &suite); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(suite.AllTasks()) != 5 {
		t.Fatalf("want 5 tasks, got %d", len(suite.AllTasks()))
	}
	provider := observability.NewInMemoryObservabilityProvider()
	// Replace command verifiers (which depend on agent file output) with
	// always_pass so the replay is hermetic and deterministic; the shapes and
	// the canary's always_pass verifier already resolve. We keep the canary as
	// always_pass from the fixture and override the rest for determinism.
	verifiers := map[sporecore.TaskID]verifier.TaskVerifier{}
	for _, ct := range suite.AllTasks() {
		verifiers[ct.Task.ID] = verifier.AlwaysPass{}
	}
	cfg := buildConfig(&stuntAgent{id: "base"}, provider)
	b := NewEvalHarnessBuilder(suite, cfg, provider).
		Candidate("c", cfg).
		NRunsPerConfig(2).
		Metrics([]metricmap.EvalMetric{metricmap.TaskSuccessRate()})
	for id, v := range verifiers {
		b = b.VerifierFor(id, v)
	}
	reports, err := b.Build().Run(context.Background())
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if reports[0].Metrics[0].Baseline.Mean != 1.0 {
		t.Fatalf("baseline mean=%v", reports[0].Metrics[0].Baseline.Mean)
	}
}
