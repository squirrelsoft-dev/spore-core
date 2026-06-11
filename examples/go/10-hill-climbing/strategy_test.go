package main

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
)

// TestRegistryValidates is the regression guard that the post-#119 composed
// HillClimbing(inner: ReAct{propose-schema}, evaluator: "") tree stays
// validation-clean against current core. The propose slot's output schema
// resolves (structured-slot contract) and the empty-handle evaluator resolves to
// the metric evaluator registered under "". The leaves use the EMPTY
// agent/toolset handles that NewStandardHarness default-fills at build; here we
// mirror that fill (empty-key agent + toolset + metric evaluator) so the
// standalone registry validates exactly as the assembled harness would.
func TestRegistryValidates(t *testing.T) {
	model := ollama.New("test-model")
	evaluator := newReadmeQualityEvaluator(model)
	registry := sporecore.NewExecutionRegistryBuilder().
		Schema("propose-schema", proposeSchema()).
		MetricEvaluator("", evaluator).
		Agent("", sporecore.NewModelAgent(sporecore.AgentID("default"), model)).
		Toolset("", sporecore.NewStandardToolRegistry()).
		Build()
	task := sporecore.NewTask("refine the README", sporecore.NewSessionID(), hillClimbingStrategy(PerIterBudget))
	if err := registry.Validate(task); err != nil {
		t.Fatalf("the composed HillClimbing strategy must validate against the registry: %v", err)
	}
}
