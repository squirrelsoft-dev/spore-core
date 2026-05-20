package contextmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// Fixture-replay test: load the shared cases.json from
// fixtures/compaction_verifier/ and assert KeyTermVerifier matches each
// expected outcome. The fixture is byte-identical across Rust / TypeScript /
// Python / Go. Do not edit the fixture to make this test pass — fix the
// implementation instead.

type cvFixtureCase struct {
	Name            string                  `json:"name"`
	Summary         string                  `json:"summary"`
	Hints           CompactionPreserveHints `json:"hints"`
	TaskInstruction string                  `json:"task_instruction"`
	Expected        struct {
		Passed       bool     `json:"passed"`
		MissingItems []string `json:"missing_items"`
	} `json:"expected"`
}

type cvFixtureFile struct {
	Cases []cvFixtureCase `json:"cases"`
}

func TestCompactionVerifierFixtureReplay(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// go/spore-core/contextmgr → ../../../fixtures/compaction_verifier/cases.json
	path := filepath.Join(wd, "..", "..", "..", "fixtures", "compaction_verifier", "cases.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var file cvFixtureFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("expected at least one case")
	}

	v := KeyTermVerifier{}
	for _, c := range file.Cases {
		t.Run(c.Name, func(t *testing.T) {
			state := NewSessionState("sess", "task", c.TaskInstruction)
			res := v.Verify(c.Summary, c.Hints, &state)
			if res.Passed != c.Expected.Passed {
				t.Fatalf("Passed = %v, want %v (detail: %s)", res.Passed, c.Expected.Passed, res.Detail)
			}
			if !reflect.DeepEqual(res.MissingItems, c.Expected.MissingItems) {
				t.Fatalf("MissingItems = %#v, want %#v", res.MissingItems, c.Expected.MissingItems)
			}
		})
	}
}
