package promptassembly

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Fixture-replay tests: load the shared condition_eval.json and
// assembly_steps.json from fixtures/prompt_assembly/ and assert each outcome.
//
// The fixture files are byte-identical to the ones consumed by the Rust,
// TypeScript, and Python implementations. Do not edit a fixture to make a test
// pass — fix the implementation instead.

func fixtureDir(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/promptassembly -> ../../../fixtures/prompt_assembly
	return filepath.Join(wd, "..", "..", "..", "fixtures", "prompt_assembly")
}

func TestFixtureReplayConditionEval(t *testing.T) {
	type conditionCase struct {
		Name            string          `json:"name"`
		Condition       ChunkCondition  `json:"condition"`
		AssemblyContext AssemblyContext `json:"assembly_context"`
		Expected        bool            `json:"expected"`
	}
	type suite struct {
		Cases []conditionCase `json:"cases"`
	}

	path := filepath.Join(fixtureDir(t), "condition_eval.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var s suite
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("parse condition_eval.json: %v", err)
	}
	if len(s.Cases) < 8 {
		t.Fatalf("expected >=8 cases (R1-R8), got %d", len(s.Cases))
	}
	b := NewContextSourcesBuilder()
	for _, c := range s.Cases {
		ctx := c.AssemblyContext
		got := b.Evaluate(c.Condition, &ctx)
		if got != c.Expected {
			t.Errorf("case %q: got %v want %v", c.Name, got, c.Expected)
		}
	}
}

func TestFixtureReplayAssemblySteps(t *testing.T) {
	type stepCase struct {
		Name               string          `json:"name"`
		RegisteredChunks   []PromptChunk   `json:"registered_chunks"`
		AssemblyContext    AssemblyContext `json:"assembly_context"`
		ExpectedStatic     []string        `json:"expected_static"`
		ExpectedPerSession []string        `json:"expected_per_session"`
		ExpectedPerTurn    []string        `json:"expected_per_turn"`
	}
	type suite struct {
		Cases []stepCase `json:"cases"`
	}

	path := filepath.Join(fixtureDir(t), "assembly_steps.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var s suite
	if err := json.Unmarshal(raw, &s); err != nil {
		t.Fatalf("parse assembly_steps.json: %v", err)
	}
	if len(s.Cases) == 0 {
		t.Fatal("expected at least one case")
	}
	for _, c := range s.Cases {
		b := NewContextSourcesBuilderWithChunks(c.RegisteredChunks)
		ctx := c.AssemblyContext
		buckets := b.Assemble(&ctx)
		assertIDs(t, c.Name, "static", ids(buckets.Static), c.ExpectedStatic)
		assertIDs(t, c.Name, "per_session", ids(buckets.PerSession), c.ExpectedPerSession)
		assertIDs(t, c.Name, "per_turn", ids(buckets.PerTurn), c.ExpectedPerTurn)
	}
}

func assertIDs(t *testing.T, caseName, bucket string, got, want []string) {
	t.Helper()
	// Treat nil and empty as equal.
	if len(got) == 0 && len(want) == 0 {
		return
	}
	if len(got) != len(want) {
		t.Errorf("case %q %s: got %v want %v", caseName, bucket, got, want)
		return
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("case %q %s: got %v want %v", caseName, bucket, got, want)
			return
		}
	}
}
