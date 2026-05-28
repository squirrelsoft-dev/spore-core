// Package eval implements the EvalHarness — the outer ring of the
// improvement flywheel (see docs/harness-engineering-concepts.md,
// "The Improvement Flywheel" rule 10). It runs the shared task suites
// in /fixtures/task_suites/ against the harness defined in
// github.com/squirrelsoft-dev/spore-core/go/spore-core (issues #1–#13).
//
// This root package is a thin facade re-exporting the most-used identifiers
// from the implementation subpackages (issue #26):
//
//   - task     — TaskSuite, EvalTask, WorkspaceSnapshot, VerifierSpec,
//     VerificationResult, ConfigID, error taxonomy (Rules 1, 5-8).
//   - verifier — TaskVerifier + standard impls (Rules 9-13).
//   - metricmap— EvalMetric + observability-sourced sampling (Rules 16-17, Resolution 1).
//   - stats    — native Welch t-test + bootstrap CI (Rules 26-28).
//   - report   — MetricComparison, Recommendation, ComparisonReport (Rules 19-25).
//   - worktree — fresh/torn-down workspace restore (Rules 2-3).
//   - manifest — suite loading + manual promotion (Rules 6, 29, 31).
//   - harness  — the EvalHarness runner + builder + TraceAnalyzer (Rules 14-25, 30).
package eval

import (
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/harness"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/manifest"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/metricmap"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/report"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/stats"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/verifier"
)

// Re-exported types (facade).
type (
	// TaskSuite is a versioned three-list task suite.
	TaskSuite = task.TaskSuite
	// EvalTask is one evaluation task.
	EvalTask = task.EvalTask
	// WorkspaceSnapshot is the workspace restore spec.
	WorkspaceSnapshot = task.WorkspaceSnapshot
	// VerifierSpec is the serializable verifier description.
	VerifierSpec = task.VerifierSpec
	// VerificationResult is the outcome of a TaskVerifier.
	VerificationResult = task.VerificationResult
	// ConfigID identifies a candidate config.
	ConfigID = task.ConfigID
	// EvalMetric is a metric the harness aggregates/compares.
	EvalMetric = metricmap.EvalMetric
	// MetricStats is aggregated sample statistics.
	MetricStats = stats.MetricStats
	// ConfidenceInterval is a bootstrap CI.
	ConfidenceInterval = stats.ConfidenceInterval
	// MetricComparison is a per-metric comparison.
	MetricComparison = report.MetricComparison
	// Recommendation is the runner's recommendation.
	Recommendation = report.Recommendation
	// ComparisonReport is the full per-candidate report.
	ComparisonReport = report.ComparisonReport
	// TaskVerifier verifies a task run.
	TaskVerifier = verifier.TaskVerifier
	// EvalHarness is the runner.
	EvalHarness = harness.EvalHarness
	// EvalHarnessBuilder is the fluent assembler.
	EvalHarnessBuilder = harness.EvalHarnessBuilder
	// TraceAnalyzer is the deferred trace-analysis interface (Rule 30).
	TraceAnalyzer = harness.TraceAnalyzer
)

// Re-exported functions (facade).
var (
	// LoadSuiteStr loads a suite from a JSON string (Rule 6).
	LoadSuiteStr = manifest.LoadSuiteStr
	// LoadSuitePath loads a suite from a file path.
	LoadSuitePath = manifest.LoadSuitePath
	// SuiteToJSON serialises a suite to pretty JSON.
	SuiteToJSON = manifest.SuiteToJSON
	// PromoteChallengeTask promotes a challenge task to regression (Rule 31).
	PromoteChallengeTask = manifest.PromoteChallengeTask
	// BuildVerifier resolves a VerifierSpec to a TaskVerifier.
	BuildVerifier = verifier.BuildVerifier
	// NewEvalHarnessBuilder starts a builder.
	NewEvalHarnessBuilder = harness.NewEvalHarnessBuilder
	// WelchTTest is the native Welch t-test (Rule 26).
	WelchTTest = stats.WelchTTest
	// BootstrapCI is the native bootstrap CI (Rule 27).
	BootstrapCI = stats.BootstrapCI
)
