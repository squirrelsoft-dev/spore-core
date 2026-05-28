package verifier

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
)

func successRun() sporecore.RunResult {
	return sporecore.RunResult{Kind: sporecore.RunSuccess, Output: "done", SessionID: "s"}
}

func failureRun() sporecore.RunResult {
	return sporecore.RunResult{Kind: sporecore.RunFailure, SessionID: "s"}
}

func TestAlwaysPassFail(t *testing.T) {
	ctx := context.Background()
	et := &task.EvalTask{ID: "t"}
	r, err := AlwaysPass{}.Verify(ctx, et, successRun(), "")
	if err != nil || !r.Passed || r.Score != 1.0 {
		t.Fatalf("pass: %+v %v", r, err)
	}
	r, err = AlwaysFail{}.Verify(ctx, et, successRun(), "")
	if err != nil || r.Passed || r.Score != 0.0 {
		t.Fatalf("fail: %+v %v", r, err)
	}
	if !(AlwaysPass{}).IsDeterministic() {
		t.Fatal("AlwaysPass should be deterministic")
	}
}

func TestScoreClampErrors(t *testing.T) {
	if _, err := task.NewVerificationResult(true, 1.5, "x"); err == nil {
		t.Fatal("score 1.5 should error (Rule 8)")
	}
	if _, err := task.NewVerificationResult(true, -0.1, "x"); err == nil {
		t.Fatal("score -0.1 should error (Rule 8)")
	}
	r := task.ClampedVerificationResult(true, 1.5, "x")
	if r.Score != 1.0 {
		t.Fatalf("clamped score=%v", r.Score)
	}
}

func TestTestSuiteVerifierPass(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "output.txt"), []byte("HELLO\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	v := &TestSuiteVerifier{Command: "sh", Args: []string{"-c", "grep -qx HELLO output.txt"}, Timeout: 5 * time.Second}
	r, err := v.Verify(context.Background(), &task.EvalTask{ID: "t"}, successRun(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Passed || r.Score != 1.0 {
		t.Fatalf("got %+v", r)
	}
}

func TestTestSuiteVerifierFail(t *testing.T) {
	dir := t.TempDir()
	v := &TestSuiteVerifier{Command: "sh", Args: []string{"-c", "test -f nope.txt"}, Timeout: 5 * time.Second}
	r, err := v.Verify(context.Background(), &task.EvalTask{ID: "t"}, successRun(), dir)
	if err != nil {
		t.Fatal(err)
	}
	if r.Passed || r.Score != 0.0 {
		t.Fatalf("got %+v", r)
	}
}

func TestCompositeVerifierWeightedMean(t *testing.T) {
	// One required pass (1.0, w=3) + one fail (0.0, w=1) => score 0.75, passed
	// false (required child failed only if it's required; here fail is required).
	c := NewCompositeVerifier(
		[]TaskVerifier{AlwaysPass{}, AlwaysFail{}},
		[]float64{3, 1},
		[]bool{true, true},
	)
	r, err := c.Verify(context.Background(), &task.EvalTask{ID: "t"}, successRun(), "")
	if err != nil {
		t.Fatal(err)
	}
	if r.Score != 0.75 {
		t.Fatalf("score=%v want 0.75", r.Score)
	}
	if r.Passed {
		t.Fatal("required child failed -> passed should be false")
	}
	if !c.IsDeterministic() {
		t.Fatal("composite of deterministic should be deterministic")
	}
}

func TestBuildVerifierFromSpec(t *testing.T) {
	v := BuildVerifier(task.VerifierSpec{Kind: task.VerifierAlwaysPass})
	if _, ok := v.(AlwaysPass); !ok {
		t.Fatalf("got %T", v)
	}
	v = BuildVerifier(task.VerifierSpec{Kind: task.VerifierLlmJudge, ScoreRange: [2]float64{0, 10}})
	if v.IsDeterministic() {
		t.Fatal("llm_judge stub should be non-deterministic (Rule 13)")
	}
	v = BuildVerifier(task.VerifierSpec{Kind: task.VerifierTestSuite, Command: "true"})
	if !v.IsDeterministic() {
		t.Fatal("test_suite should be deterministic")
	}
}

func TestNormalizingSuccessVerifierFromMetricSpec(t *testing.T) {
	v := BuildVerifier(task.VerifierSpec{Kind: task.VerifierMetricEvaluator, Direction: task.DirectionMaximize})
	r, _ := v.Verify(context.Background(), &task.EvalTask{ID: "t"}, successRun(), "")
	if !r.Passed || r.Score != 1.0 {
		t.Fatalf("success run -> %+v", r)
	}
	r, _ = v.Verify(context.Background(), &task.EvalTask{ID: "t"}, failureRun(), "")
	if r.Passed || r.Score != 0.0 {
		t.Fatalf("failure run -> %+v", r)
	}
	if !v.IsDeterministic() {
		t.Fatal("metric placeholder should be deterministic")
	}
}
