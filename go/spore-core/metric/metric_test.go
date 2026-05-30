package metric

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/termination"
)

// ============================================================================
// fakeSandbox — minimal SandboxProvider for evaluator tests.
// ============================================================================

type fakeSandbox struct {
	stdout   string
	stderr   string
	exitCode int
	timedOut bool
	root     string
}

func newFakeSandbox(t *testing.T, stdout string) *fakeSandbox {
	t.Helper()
	return &fakeSandbox{stdout: stdout, root: t.TempDir()}
}

func (f *fakeSandbox) Validate(context.Context, sporecore.ToolCall) *sporecore.SandboxViolation {
	return nil
}

func (f *fakeSandbox) ExecuteCommand(_ context.Context, _ string, _ []string, _ string, _ time.Duration) (sporecore.CommandOutput, *sporecore.SandboxViolation) {
	return sporecore.CommandOutput{
		Stdout:   f.stdout,
		Stderr:   f.stderr,
		ExitCode: f.exitCode,
		TimedOut: f.timedOut,
	}, nil
}

func (f *fakeSandbox) HandleLargeOutput(_ context.Context, content string, _ string, _ uint32, _ uint32) sporecore.TruncatedOutput {
	return sporecore.TruncatedOutput{Content: content, Truncated: false, OriginalSize: uint64(len(content))}
}

func (f *fakeSandbox) ResolvePath(_ context.Context, p string, _ sporecore.Operation) (string, *sporecore.SandboxViolation) {
	return filepath.Join(f.root, p), nil
}

func (f *fakeSandbox) IsolationMode() sporecore.IsolationMode { return sporecore.IsolationNone{} }
func (f *fakeSandbox) WorkspaceRoot() string                  { return f.root }

func snapshot() *termination.SessionStateSnapshot {
	s := termination.NewSessionStateSnapshot(
		sporecore.SessionID("sess"),
		sporecore.TaskID("task"),
		sporecore.SessionState{},
	)
	return &s
}

// ============================================================================
// ShouldKeep
// ============================================================================

func TestShouldKeepMinimizeLowerIsBetter(t *testing.T) {
	if !ShouldKeep(1.0, 2.0, sporecore.OptimizationMinimize, nil) {
		t.Fatal("expected keep")
	}
	if ShouldKeep(2.0, 1.0, sporecore.OptimizationMinimize, nil) {
		t.Fatal("expected discard")
	}
}

func TestShouldKeepMaximizeHigherIsBetter(t *testing.T) {
	if !ShouldKeep(2.0, 1.0, sporecore.OptimizationMaximize, nil) {
		t.Fatal("expected keep")
	}
	if ShouldKeep(1.0, 2.0, sporecore.OptimizationMaximize, nil) {
		t.Fatal("expected discard")
	}
}

func TestShouldKeepEqualIsDiscarded(t *testing.T) {
	if ShouldKeep(1.0, 1.0, sporecore.OptimizationMinimize, nil) {
		t.Fatal("equal must discard (minimize)")
	}
	if ShouldKeep(1.0, 1.0, sporecore.OptimizationMaximize, nil) {
		t.Fatal("equal must discard (maximize)")
	}
}

func TestShouldKeepRespectsMinDelta(t *testing.T) {
	d := 0.5
	if ShouldKeep(1.5, 2.0, sporecore.OptimizationMinimize, &d) {
		t.Fatal("delta == min_delta must discard")
	}
	if !ShouldKeep(1.49, 2.0, sporecore.OptimizationMinimize, &d) {
		t.Fatal("delta > min_delta must keep")
	}
}

// ============================================================================
// ParseMetric
// ============================================================================

func TestParseMetricExtractsCaptureGroup(t *testing.T) {
	v, err := ParseMetric("val_bpb:  3.125\nother", `val_bpb:\s+([\d.]+)`)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if v < 3.124999999 || v > 3.125000001 {
		t.Fatalf("got %v", v)
	}
}

func TestParseMetricNoMatchIsParseFailed(t *testing.T) {
	_, err := ParseMetric("no metric here", `val_bpb:\s+([\d.]+)`)
	if err == nil || err.Kind != MetricErrParseFailed {
		t.Fatalf("expected parse_failed, got %v", err)
	}
}

func TestParseMetricUnparseableCaptureIsParseFailed(t *testing.T) {
	_, err := ParseMetric("val_bpb: oops", `val_bpb:\s+(\S+)`)
	if err == nil || err.Kind != MetricErrParseFailed {
		t.Fatalf("expected parse_failed, got %v", err)
	}
}

func TestParseMetricInvalidRegexIsExecutionFailed(t *testing.T) {
	_, err := ParseMetric("x", "(unbalanced")
	if err == nil || err.Kind != MetricErrExecutionFailed {
		t.Fatalf("expected execution_failed, got %v", err)
	}
}

// ============================================================================
// IterationStatus
// ============================================================================

func TestIterationStatusFromErrorMapsTimeout(t *testing.T) {
	s := IterationStatusFromError(NewTimeout(time.Second))
	if s != IterationTimeout {
		t.Fatalf("got %s", s)
	}
}

func TestIterationStatusFromErrorMapsOthersToCrashed(t *testing.T) {
	for _, e := range []*MetricError{
		NewCrashed("x"),
		NewExecutionFailed("x"),
		NewParseFailed("", ""),
	} {
		if got := IterationStatusFromError(e); got != IterationCrashed {
			t.Fatalf("expected crashed for %v, got %s", e, got)
		}
	}
}

// ============================================================================
// CommandMetricEvaluator
// ============================================================================

func TestCommandEvaluatorHappyPathWritesLogAndParses(t *testing.T) {
	sb := newFakeSandbox(t, "val_bpb: 1.234\n")
	eval := &CommandMetricEvaluator{
		Command:       "uv",
		Args:          []string{"run", "train.py"},
		MetricPattern: `val_bpb:\s+([\d.]+)`,
		Timeout:       60 * time.Second,
		LogOutputTo:   "run.log",
		Dir:           sporecore.OptimizationMinimize,
		Desc:          "autoresearch val_bpb",
	}
	r, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if r.Value < 1.2339999 || r.Value > 1.2340001 {
		t.Fatalf("value: %v", r.Value)
	}
	body, ioerr := os.ReadFile(filepath.Join(sb.root, "run.log"))
	if ioerr != nil {
		t.Fatalf("read log: %v", ioerr)
	}
	if string(body) == "" || string(body) != "val_bpb: 1.234\n" {
		t.Fatalf("log: %q", body)
	}
	if eval.Direction() != sporecore.OptimizationMinimize {
		t.Fatal("direction")
	}
	if eval.Description() != "autoresearch val_bpb" {
		t.Fatal("description")
	}
}

func TestCommandEvaluatorTimeoutMapsToTimeoutError(t *testing.T) {
	sb := newFakeSandbox(t, "")
	sb.timedOut = true
	eval := &CommandMetricEvaluator{
		Command:       "x",
		MetricPattern: `v:(\d+)`,
		Timeout:       time.Millisecond,
		LogOutputTo:   "run.log",
		Dir:           sporecore.OptimizationMinimize,
	}
	_, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err == nil || err.Kind != MetricErrTimeout {
		t.Fatalf("expected timeout, got %v", err)
	}
}

func TestCommandEvaluatorNonzeroExitIsCrashed(t *testing.T) {
	sb := newFakeSandbox(t, "boom")
	sb.exitCode = 1
	eval := &CommandMetricEvaluator{
		Command:       "x",
		MetricPattern: `v:(\d+)`,
		Timeout:       time.Second,
		LogOutputTo:   "run.log",
		Dir:           sporecore.OptimizationMinimize,
	}
	_, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err == nil || err.Kind != MetricErrCrashed {
		t.Fatalf("expected crashed, got %v", err)
	}
}

func TestCommandEvaluatorParseFailedWhenRegexDoesntMatch(t *testing.T) {
	sb := newFakeSandbox(t, "no metric")
	eval := &CommandMetricEvaluator{
		Command:       "x",
		MetricPattern: `v:(\d+)`,
		Timeout:       time.Second,
		LogOutputTo:   "run.log",
		Dir:           sporecore.OptimizationMinimize,
	}
	_, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err == nil || err.Kind != MetricErrParseFailed {
		t.Fatalf("expected parse_failed, got %v", err)
	}
}

// ============================================================================
// TestPassRateEvaluator
// ============================================================================

func TestPassRateEvaluatorReturnsFraction(t *testing.T) {
	sb := newFakeSandbox(t, "passed 17 of 20")
	eval := &TestPassRateEvaluator{
		Command:      "pytest",
		Timeout:      60 * time.Second,
		PassPattern:  `passed (\d+)`,
		TotalPattern: `of (\d+)`,
	}
	r, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if r.Value < 0.84999 || r.Value > 0.85001 {
		t.Fatalf("value: %v", r.Value)
	}
	if eval.Direction() != sporecore.OptimizationMaximize {
		t.Fatal("direction")
	}
}

// ============================================================================
// LatencyEvaluator
// ============================================================================

func TestLatencyEvaluatorAveragesRuns(t *testing.T) {
	sb := newFakeSandbox(t, "ok")
	eval := &LatencyEvaluator{
		Command:      "echo",
		Args:         []string{"ok"},
		WarmupRuns:   1,
		MeasuredRuns: 2,
		Timeout:      5 * time.Second,
	}
	r, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if r.Value < 0.0 {
		t.Fatalf("value: %v", r.Value)
	}
	if eval.Direction() != sporecore.OptimizationMinimize {
		t.Fatal("direction")
	}
	if r.Metadata["measured_runs"] != "2" {
		t.Fatalf("metadata: %v", r.Metadata)
	}
}

func TestLatencyEvaluatorZeroMeasuredRunsRejects(t *testing.T) {
	sb := newFakeSandbox(t, "")
	eval := &LatencyEvaluator{
		Command:      "x",
		MeasuredRuns: 0,
		Timeout:      time.Second,
	}
	_, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err == nil || err.Kind != MetricErrExecutionFailed {
		t.Fatalf("expected execution_failed, got %v", err)
	}
}

// ============================================================================
// LlmJudgeEvaluator
// ============================================================================

type fakeModel struct {
	text string
}

func (f *fakeModel) Call(_ context.Context, _ sporecore.ModelRequest) (sporecore.ModelResponse, error) {
	return sporecore.ModelResponse{
		Content:    []sporecore.ContentBlock{sporecore.NewTextBlock(f.text)},
		Usage:      sporecore.TokenUsage{},
		StopReason: sporecore.StopEndTurn,
	}, nil
}

func (f *fakeModel) CallStreaming(_ context.Context, _ sporecore.ModelRequest) (<-chan sporecore.StreamEventOrErr, error) {
	ch := make(chan sporecore.StreamEventOrErr)
	close(ch)
	return ch, nil
}

func (f *fakeModel) CountTokens(_ context.Context, _ sporecore.ModelRequest) (uint32, error) {
	return 0, nil
}

func (f *fakeModel) Provider() sporecore.ProviderInfo {
	return sporecore.ProviderInfo{Name: "fake", ModelID: "fake", ContextWindow: 8000}
}

func TestLlmJudgeNormalizesScoreIntoUnitRange(t *testing.T) {
	sb := newFakeSandbox(t, "")
	eval := &LlmJudgeEvaluator{
		JudgeModel:  JudgeModelConfig{Provider: "fake", ModelID: "judge-1"},
		Rubric:      "rate this",
		ScoreRange:  [2]float64{0.0, 10.0},
		SampleInput: "the answer",
		Client:      &fakeModel{text: "score: 7.5"},
	}
	r, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if r.Value < 0.74999 || r.Value > 0.75001 {
		t.Fatalf("value: %v", r.Value)
	}
	if eval.Direction() != sporecore.OptimizationMaximize {
		t.Fatal("direction")
	}
}

func TestLlmJudgeClampsScoreOutsideRange(t *testing.T) {
	sb := newFakeSandbox(t, "")
	eval := &LlmJudgeEvaluator{
		JudgeModel:  JudgeModelConfig{Provider: "fake", ModelID: "judge-1"},
		Rubric:      "rate",
		ScoreRange:  [2]float64{0.0, 10.0},
		SampleInput: "x",
		Client:      &fakeModel{text: "score: 42"},
	}
	r, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if r.Value < 0.99999 || r.Value > 1.00001 {
		t.Fatalf("value: %v", r.Value)
	}
}

func TestLlmJudgeParseFailedWhenNoScore(t *testing.T) {
	sb := newFakeSandbox(t, "")
	eval := &LlmJudgeEvaluator{
		JudgeModel:  JudgeModelConfig{Provider: "fake", ModelID: "judge-1"},
		ScoreRange:  [2]float64{0.0, 10.0},
		SampleInput: "x",
		Client:      &fakeModel{text: "no score in here"},
	}
	_, err := eval.Evaluate(context.Background(), sb, snapshot())
	if err == nil || err.Kind != MetricErrParseFailed {
		t.Fatalf("expected parse_failed, got %v", err)
	}
}

// ============================================================================
// JSON round-trip — MetricError matches Rust serde shape.
// ============================================================================

func TestMetricErrorJSONRoundTrip(t *testing.T) {
	cases := []*MetricError{
		NewExecutionFailed("oops"),
		NewTimeout(2 * time.Second),
		NewParseFailed("output", "pattern"),
		NewCrashed("log"),
	}
	for _, e := range cases {
		b, err := json.Marshal(e)
		if err != nil {
			t.Fatalf("marshal %v: %v", e, err)
		}
		var got MetricError
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("unmarshal %v: %v", e, err)
		}
		if got.Kind != e.Kind {
			t.Fatalf("kind mismatch: got %s want %s", got.Kind, e.Kind)
		}
	}
}

func TestMetricErrorTimeoutWireFormat(t *testing.T) {
	b, err := json.Marshal(NewTimeout(2500 * time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	// Rust serializes Duration as fractional seconds via duration_secs.
	var probe map[string]any
	if err := json.Unmarshal(b, &probe); err != nil {
		t.Fatal(err)
	}
	if probe["kind"] != "timeout" {
		t.Fatalf("kind: %v", probe["kind"])
	}
	if probe["after"].(float64) < 2.49 || probe["after"].(float64) > 2.51 {
		t.Fatalf("after: %v", probe["after"])
	}
}

// ============================================================================
// AsHarnessMetricEvaluator bridge (issue #60)
// ============================================================================

// bridgeFakeEvaluator returns a scripted result/error so the harness-seam
// adapter's translation can be asserted.
type bridgeFakeEvaluator struct {
	result MetricResult
	err    *MetricError
	dir    sporecore.OptimizationDirection
}

func (e bridgeFakeEvaluator) Evaluate(context.Context, sporecore.SandboxProvider, *termination.SessionStateSnapshot) (MetricResult, *MetricError) {
	if e.err != nil {
		return MetricResult{}, e.err
	}
	return e.result, nil
}
func (e bridgeFakeEvaluator) Direction() sporecore.OptimizationDirection { return e.dir }
func (e bridgeFakeEvaluator) Description() string                        { return "bridge fake" }

func TestAsHarnessMetricEvaluatorNilStaysNil(t *testing.T) {
	if AsHarnessMetricEvaluator(nil) != nil {
		t.Fatal("nil evaluator should stay nil through the seam")
	}
}

func TestAsHarnessMetricEvaluatorSuccess(t *testing.T) {
	inner := bridgeFakeEvaluator{result: MetricResult{Value: 3.5, Duration: 2 * time.Second}, dir: sporecore.OptimizationMaximize}
	bridged := AsHarnessMetricEvaluator(inner)
	sb := newFakeSandbox(t, "")
	res, hcErr := bridged.Evaluate(context.Background(), sb, "s", "task", sporecore.SessionState{})
	if hcErr != nil {
		t.Fatalf("unexpected error: %+v", hcErr)
	}
	if res.Value != 3.5 || res.Duration != 2*time.Second {
		t.Fatalf("got %+v", res)
	}
	if bridged.Description() != "bridge fake" {
		t.Fatalf("description = %q", bridged.Description())
	}
}

func TestAsHarnessMetricEvaluatorCrashStatus(t *testing.T) {
	inner := bridgeFakeEvaluator{err: NewCrashed("boom")}
	bridged := AsHarnessMetricEvaluator(inner)
	sb := newFakeSandbox(t, "")
	_, hcErr := bridged.Evaluate(context.Background(), sb, "s", "task", sporecore.SessionState{})
	if hcErr == nil || hcErr.Status != sporecore.HillClimbCrashed {
		t.Fatalf("got %+v, want crashed", hcErr)
	}
}

func TestAsHarnessMetricEvaluatorTimeoutStatus(t *testing.T) {
	inner := bridgeFakeEvaluator{err: NewTimeout(time.Second)}
	bridged := AsHarnessMetricEvaluator(inner)
	sb := newFakeSandbox(t, "")
	_, hcErr := bridged.Evaluate(context.Background(), sb, "s", "task", sporecore.SessionState{})
	if hcErr == nil || hcErr.Status != sporecore.HillClimbTimeout {
		t.Fatalf("got %+v, want timeout", hcErr)
	}
}
