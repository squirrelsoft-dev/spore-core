package manifest

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
)

func TestMissingSuiteVersionRejected(t *testing.T) {
	_, err := LoadSuiteStr(`{"regression": []}`)
	if !errors.Is(err, task.ErrMissingSuiteVersion) {
		t.Fatalf("want ErrMissingSuiteVersion, got %v", err)
	}
}

func TestLoadMinimalSuite(t *testing.T) {
	s, err := LoadSuiteStr(`{"suite_version": 7}`)
	if err != nil {
		t.Fatal(err)
	}
	if s.SuiteVersion != 7 {
		t.Fatalf("version=%d", s.SuiteVersion)
	}
}

func TestTimeoutDefaultApplied(t *testing.T) {
	s, err := LoadSuiteStr(`{"suite_version":1,"regression":[{"id":"x","instruction":"i","workspace_snapshot":{"kind":"empty"},"verifier_spec":{"kind":"always_pass"}}]}`)
	if err != nil {
		t.Fatal(err)
	}
	if s.Regression[0].TimeoutSecs != task.DefaultTimeoutSecs {
		t.Fatalf("timeout=%d want default %d", s.Regression[0].TimeoutSecs, task.DefaultTimeoutSecs)
	}
}

func TestPromoteChallengeTask(t *testing.T) {
	s := &task.TaskSuite{
		SuiteVersion: 3,
		Challenge: []task.EvalTask{
			{ID: "c1", VerifierSpec: task.VerifierSpec{Kind: task.VerifierAlwaysPass}, WorkspaceSnapshot: task.WorkspaceSnapshot{Kind: task.SnapshotEmpty}},
		},
	}
	if err := PromoteChallengeTask(s, "c1"); err != nil {
		t.Fatal(err)
	}
	if len(s.Challenge) != 0 || len(s.Regression) != 1 || s.SuiteVersion != 4 {
		t.Fatalf("got challenge=%d regression=%d v=%d", len(s.Challenge), len(s.Regression), s.SuiteVersion)
	}
	if err := PromoteChallengeTask(s, "missing"); err == nil {
		t.Fatalf("expected error for missing task")
	}
}

// Rule 29: load the shared core_suite.json manifest and assert its shape; the
// JSON is the cross-language oracle and is never modified to make a test pass.
func TestCoreSuiteFixtureLoads(t *testing.T) {
	path := filepath.Join("..", "..", "..", "fixtures", "task_suites", "core_suite.json")
	s, err := LoadSuitePath(path)
	if err != nil {
		t.Fatalf("load core_suite: %v", err)
	}
	if s.SuiteVersion != 1 {
		t.Fatalf("version=%d", s.SuiteVersion)
	}
	if len(s.Regression) != 2 || len(s.Challenge) != 2 || len(s.Canary) != 1 {
		t.Fatalf("shape: reg=%d chal=%d can=%d", len(s.Regression), len(s.Challenge), len(s.Canary))
	}
	// Spot-check a tagged-union round-trip: regression[0] is a files snapshot
	// with a test_suite verifier.
	r0 := s.Regression[0]
	if r0.ID != "regression_s1_uppercase" {
		t.Fatalf("id=%q", r0.ID)
	}
	if r0.WorkspaceSnapshot.Kind != task.SnapshotFiles || r0.WorkspaceSnapshot.Files["input.txt"] == "" {
		t.Fatalf("snapshot=%+v", r0.WorkspaceSnapshot)
	}
	if r0.VerifierSpec.Kind != task.VerifierTestSuite || r0.VerifierSpec.Command != "sh" {
		t.Fatalf("verifier=%+v", r0.VerifierSpec)
	}
	if r0.ExpectedTurns == nil || r0.ExpectedTurns[0] != 2 || r0.ExpectedTurns[1] != 8 {
		t.Fatalf("expected_turns=%v", r0.ExpectedTurns)
	}
	if r0.TimeoutSecs != 60 {
		t.Fatalf("timeout=%d", r0.TimeoutSecs)
	}
	// Canary uses always_pass + empty snapshot.
	can := s.Canary[0]
	if can.VerifierSpec.Kind != task.VerifierAlwaysPass || can.WorkspaceSnapshot.Kind != task.SnapshotEmpty {
		t.Fatalf("canary=%+v", can)
	}
}

// Round-trip: marshal a loaded suite and re-load it, asserting structural
// equality of the tagged unions.
func TestSuiteRoundTrip(t *testing.T) {
	path := filepath.Join("..", "..", "..", "fixtures", "task_suites", "core_suite.json")
	s, err := LoadSuitePath(path)
	if err != nil {
		t.Fatal(err)
	}
	out, err := SuiteToJSON(s)
	if err != nil {
		t.Fatal(err)
	}
	s2, err := LoadSuiteStr(out)
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := json.Marshal(s)
	b2, _ := json.Marshal(s2)
	if string(b1) != string(b2) {
		t.Fatalf("round-trip mismatch:\n%s\n%s", b1, b2)
	}
}

var _ = os.Stdout
