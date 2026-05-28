package harness

import (
	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/observability"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/metricmap"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/stats"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/verifier"
)

// EvalHarnessBuilder is the fluent assembler for an EvalHarness, mirroring
// sporecore's HarnessBuilder. Defaults: NRunsPerConfig = 3, BootstrapIterations
// = 1000, metrics = [TaskSuccessRate], primaryMetric = TaskSuccessRate.
type EvalHarnessBuilder struct {
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

// NewEvalHarnessBuilder starts from the required pieces: a suite, the baseline
// config, and the observability provider the runner reads metrics from.
func NewEvalHarnessBuilder(suite task.TaskSuite, baseline sporecore.HarnessConfig, obs observability.ObservabilityProvider) *EvalHarnessBuilder {
	return &EvalHarnessBuilder{
		taskSuite:           suite,
		baselineConfig:      baseline,
		nRunsPerConfig:      3,
		metrics:             []metricmap.EvalMetric{metricmap.TaskSuccessRate()},
		observability:       obs,
		bootstrapIterations: stats.DefaultBootstrapIterations,
		primaryMetric:       metricmap.TaskSuccessRate(),
	}
}

// Candidate adds a candidate config under id.
func (b *EvalHarnessBuilder) Candidate(id string, config sporecore.HarnessConfig) *EvalHarnessBuilder {
	b.candidateConfigs = append(b.candidateConfigs, candidateConfig{id: task.ConfigID(id), config: config})
	return b
}

// NRunsPerConfig sets the per-(config,task) run count.
func (b *EvalHarnessBuilder) NRunsPerConfig(n uint32) *EvalHarnessBuilder {
	b.nRunsPerConfig = n
	return b
}

// Metrics sets the metrics to aggregate and compare.
func (b *EvalHarnessBuilder) Metrics(metrics []metricmap.EvalMetric) *EvalHarnessBuilder {
	b.metrics = metrics
	return b
}

// BootstrapIterations sets the bootstrap iteration count.
func (b *EvalHarnessBuilder) BootstrapIterations(n uint32) *EvalHarnessBuilder {
	b.bootstrapIterations = n
	return b
}

// PrimaryMetric sets the primary metric that drives the recommendation.
func (b *EvalHarnessBuilder) PrimaryMetric(metric metricmap.EvalMetric) *EvalHarnessBuilder {
	b.primaryMetric = metric
	return b
}

// VerifierFor injects a concrete TaskVerifier for a specific task id,
// overriding the spec-resolved verifier. Useful for wiring real evaluators or
// LLM judges that cannot be expressed in a manifest spec.
func (b *EvalHarnessBuilder) VerifierFor(taskID sporecore.TaskID, v verifier.TaskVerifier) *EvalHarnessBuilder {
	if b.verifiers == nil {
		b.verifiers = map[sporecore.TaskID]verifier.TaskVerifier{}
	}
	b.verifiers[taskID] = v
	return b
}

// Build assembles the EvalHarness.
func (b *EvalHarnessBuilder) Build() *EvalHarness {
	return &EvalHarness{
		taskSuite:           b.taskSuite,
		baselineConfig:      b.baselineConfig,
		candidateConfigs:    b.candidateConfigs,
		nRunsPerConfig:      b.nRunsPerConfig,
		metrics:             b.metrics,
		observability:       b.observability,
		bootstrapIterations: b.bootstrapIterations,
		primaryMetric:       b.primaryMetric,
		verifiers:           b.verifiers,
	}
}
