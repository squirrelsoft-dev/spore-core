package main

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
)

// TestRegistryValidates is the regression guard that the post-#119 composed
// PlanExecute(plan: ReAct{plan-schema}, execute: ReAct) tree stays
// validation-clean against current core. The plan slot's output schema resolves
// and the structured-slot contract is satisfied. The leaves use the EMPTY
// agent/toolset handles that NewStandardHarness default-fills at build; here we
// mirror that fill (empty-key agent + toolset) so the standalone registry
// validates exactly as the assembled harness would.
func TestRegistryValidates(t *testing.T) {
	model := ollama.New("test-model")
	registry := sporecore.NewExecutionRegistryBuilder().
		Schema("plan-schema", planSchema()).
		Agent("", sporecore.NewModelAgent(sporecore.AgentID("default"), model)).
		Toolset("", sporecore.NewStandardToolRegistry()).
		Build()
	task := sporecore.NewTask("decompose and execute", sporecore.NewSessionID(), planExecuteStrategy())
	if err := registry.Validate(task); err != nil {
		t.Fatalf("the composed PlanExecute strategy must validate against the registry: %v", err)
	}
}
