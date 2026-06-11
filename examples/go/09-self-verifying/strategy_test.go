package main

import (
	"testing"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/ollama"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/verifier"
)

// TestRegistryValidates is the regression guard that the post-#119 composed
// SelfVerifying(inner: ReAct{worker-schema}, evaluator: "") tree stays
// validation-clean against current core. The worker slot's output schema resolves
// (structured-slot contract) and the empty-handle evaluator resolves to the
// verifier registered under "". The leaves use the EMPTY agent/toolset handles
// that NewStandardHarness default-fills at build; here we mirror that fill
// (empty-key agent + toolset + verifier) so the standalone registry validates
// exactly as the assembled harness would.
func TestRegistryValidates(t *testing.T) {
	model := ollama.New("test-model")
	inner, err := verifier.NewEvaluatorResponseVerifier(`(?im)^\s*PASS\s*$`, `(?im)FAIL:\s*.+`, 3)
	if err != nil {
		t.Fatalf("failed to construct verifier: %v", err)
	}
	harnessVerifier := newReportingVerifier(verifier.AsHarnessVerifier(inner), 3)
	registry := sporecore.NewExecutionRegistryBuilder().
		Schema("worker-schema", workerSchema()).
		Verifier("", harnessVerifier).
		Agent("", sporecore.NewModelAgent(sporecore.AgentID("default"), model)).
		Toolset("", sporecore.NewStandardToolRegistry()).
		Build()
	task := sporecore.NewTask("draft ParseIntList", sporecore.NewSessionID(), selfVerifyingStrategy(3))
	if err := registry.Validate(task); err != nil {
		t.Fatalf("the composed SelfVerifying strategy must validate against the registry: %v", err)
	}
}
