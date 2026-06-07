package sporecore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ============================================================================
// Test doubles
// ============================================================================

// hcRootedSandbox is an AllowAllSandbox with a real workspace root (so the TSV
// lands in a tempdir under test control) that records every ExecuteCommand call
// (so `git reset --hard HEAD` reverts can be asserted).
type hcRootedSandbox struct {
	AllowAllSandbox
	root     string
	commands [][]string // each entry: [command, args...]
}

func (s *hcRootedSandbox) WorkspaceRoot() string { return s.root }

func (s *hcRootedSandbox) ExecuteCommand(
	_ context.Context,
	command string,
	args []string,
	_ string,
	_ time.Duration,
) (CommandOutput, *SandboxViolation) {
	entry := append([]string{command}, args...)
	s.commands = append(s.commands, entry)
	return CommandOutput{ExitCode: 0}, nil
}

var _ SandboxProvider = (*hcRootedSandbox)(nil)

// scriptedMetricEvaluator is a root-package MetricEvaluator seam double driven
// by a scripted sequence of outcomes. A nil *HillClimbMetricResult entry means
// "return this error"; a non-nil result means "return this success". index 0 is
// the iteration-0 baseline.
type scriptedMetricEvaluator struct {
	results []*HillClimbMetricResult
	errors  []*HillClimbMetricError
	calls   int
	desc    string
}

func (e *scriptedMetricEvaluator) Evaluate(
	_ context.Context,
	_ SandboxProvider,
	_ SessionID,
	_ TaskID,
	_ SessionState,
) (*HillClimbMetricResult, *HillClimbMetricError) {
	i := e.calls
	e.calls++
	if i < len(e.errors) && e.errors[i] != nil {
		return nil, e.errors[i]
	}
	if i < len(e.results) && e.results[i] != nil {
		return e.results[i], nil
	}
	// Out of script: behave as a crash (defensive — tests should not over-run).
	return nil, &HillClimbMetricError{Status: HillClimbCrashed, Message: "out of script"}
}

func (e *scriptedMetricEvaluator) Description() string {
	if e.desc == "" {
		return "scripted metric"
	}
	return e.desc
}

var _ MetricEvaluator = (*scriptedMetricEvaluator)(nil)

// res builds a zero-duration successful metric result.
func res(v float64) *HillClimbMetricResult { return &HillClimbMetricResult{Value: v} }

// u32 returns a pointer to a uint32 literal.
func u32(v uint32) *uint32 { return &v }

// f64 returns a pointer to a float64 literal.
func f64(v float64) *float64 { return &v }

// hcConfig builds a HillClimbing harness config rooted at a tempdir, with an
// agent that always returns a FinalResponse (so each iteration's ReAct sub-run
// terminates cleanly) and the supplied evaluator wired in.
func hcConfig(t *testing.T, eval MetricEvaluator) (HarnessConfig, *hcRootedSandbox) {
	t.Helper()
	sb := &hcRootedSandbox{root: t.TempDir()}
	agent := NewMockAgent("hc")
	for i := 0; i < 64; i++ {
		agent.Push(NewFinalResponse("done", TokenUsage{}))
	}
	cfg := HarnessConfig{
		Agent:             agent,
		ToolRegistry:      NewScriptedToolRegistry(),
		Sandbox:           sb,
		ContextManager:    NoopContextManager{},
		TerminationPolicy: AlwaysContinuePolicy{},
		MetricEvaluator:   eval,
	}
	return cfg, sb
}

// hcTask builds a HillClimbing task with the given strategy payload.
func hcTask(direction OptimizationDirection, maxStagnation *uint32, revert bool, minDelta *float64, maxTurns *uint32) Task {
	// Map the legacy Option semantics onto the #119 required scalar config:
	// nil maxStagnation → the MaxUint32 "unbounded" sentinel; nil minDelta → 0.0.
	stag := ^uint32(0)
	if maxStagnation != nil {
		stag = *maxStagnation
	}
	var delta float64
	if minDelta != nil {
		delta = *minDelta
	}
	t := NewTask("optimize", SessionID("s1"), HillClimbingStrategy(HillClimbingConfig{
		Inner:                 PtrStrategy(ReActStrategy(^uint32(0))),
		Direction:             direction,
		MaxStagnation:         stag,
		RevertOnNoImprovement: revert,
		MinImprovementDelta:   delta,
		Evaluator:             AgentRef(""),
	}))
	if maxTurns != nil {
		t.Budget = BudgetLimits{MaxTurns: maxTurns}
	}
	return t
}

func readTSV(t *testing.T, sb *hcRootedSandbox, taskID TaskID) string {
	t.Helper()
	path := filepath.Join(sb.root, ".spore", "results", string(taskID)+".tsv")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading TSV: %v", err)
	}
	return string(b)
}

// ============================================================================
// Misconfiguration (Decision 6)
// ============================================================================

func TestHillClimbingNilEvaluatorMisconfigured(t *testing.T) {
	cfg, _ := hcConfig(t, nil)
	cfg.MetricEvaluator = nil
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(hcTask(OptimizationMaximize, u32(1), false, nil, nil)))
	if r.Kind != RunFailure || r.Reason.Kind != HaltHillClimbingMisconfigured {
		t.Fatalf("got %+v", r)
	}
}

// TestHillClimbingNotStrategyNotImplemented proves HillClimbing no longer
// returns StrategyNotYetImplemented.
func TestHillClimbingNotStrategyNotImplemented(t *testing.T) {
	cfg, _ := hcConfig(t, &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(1.0), res(2.0)}})
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(hcTask(OptimizationMaximize, u32(1), false, nil, nil)))
	if r.Kind == RunFailure && r.Reason.Kind == HaltStrategyNotYetImplemented {
		t.Fatalf("HillClimbing still returns StrategyNotYetImplemented")
	}
}

// ============================================================================
// Baseline-error misconfiguration (Decision 7)
// ============================================================================

func TestHillClimbingBaselineErrorMisconfigured(t *testing.T) {
	eval := &scriptedMetricEvaluator{
		results: []*HillClimbMetricResult{nil},
		errors:  []*HillClimbMetricError{{Status: HillClimbCrashed, Message: "boom"}},
	}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(1), false, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltHillClimbingMisconfigured {
		t.Fatalf("got %+v", r)
	}
	// The failed baseline row is still written, with an EMPTY metric_value.
	tsv := readTSV(t, sb, task.ID)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t\tmaximize\tcrashed\t0.000000\tscripted metric\n"
	if tsv != want {
		t.Fatalf("baseline-error TSV mismatch:\n got %q\nwant %q", tsv, want)
	}
	// No agent turn happened (no git/exec calls — baseline only).
	if len(sb.commands) != 0 {
		t.Fatalf("expected no commands, got %v", sb.commands)
	}
}

// ============================================================================
// Baseline-first / status-kept (Decision 5)
// ============================================================================

func TestHillClimbingBaselineFirstKept(t *testing.T) {
	// Baseline 1.0 kept; iter1 2.0 kept (improve); max_stagnation unset, so the
	// run ends on the 1-turn budget after iteration 1.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(1.0), res(2.0)}}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, nil, false, nil, u32(1))
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("got %+v", r)
	}
	tsv := readTSV(t, sb, task.ID)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
		"1\t\t2.000000\tmaximize\tkept\t0.000000\tscripted metric\n"
	if tsv != want {
		t.Fatalf("TSV mismatch:\n got %q\nwant %q", tsv, want)
	}
}

// ============================================================================
// Keep-on-improve / discard-on-regress
// ============================================================================

func TestHillClimbingKeepOnImprove(t *testing.T) {
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(1.0), res(2.0), res(3.0)}}
	cfg, _ := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(2), false, nil, u32(2))
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	// Two improvements; stagnation never reaches 2; ends on the turn budget.
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("got %+v", r)
	}
}

func TestHillClimbingDiscardOnRegress(t *testing.T) {
	// maximize: baseline 2.0 kept, iter1 1.0 regress -> discard. max_stagnation 1.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(2.0), res(1.0)}}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(1), false, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	if r.Reason.BestMetric != 2.0 {
		t.Fatalf("best_metric = %v, want 2.0", r.Reason.BestMetric)
	}
	tsv := readTSV(t, sb, task.ID)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t2.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
		"1\t\t1.000000\tmaximize\tdiscarded\t0.000000\tscripted metric\n"
	if tsv != want {
		t.Fatalf("TSV mismatch:\n got %q\nwant %q", tsv, want)
	}
}

// ============================================================================
// Strict min_delta boundary
// ============================================================================

func TestHillClimbingStrictMinDeltaBoundary(t *testing.T) {
	// minimize: baseline 2.0, iter1 1.5 with min_delta 0.5. Improvement of EXACTLY
	// min_delta is NOT progress -> discarded. max_stagnation 1 halts.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(2.0), res(1.5)}}
	cfg, _ := hcConfig(t, eval)
	task := hcTask(OptimizationMinimize, u32(1), false, f64(0.5), nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	if r.Reason.BestMetric != 2.0 {
		t.Fatalf("best_metric = %v, want 2.0 (exact-delta discarded)", r.Reason.BestMetric)
	}
}

func TestHillClimbingJustOverMinDeltaKept(t *testing.T) {
	// minimize: baseline 2.0, iter1 1.49 with min_delta 0.5 -> delta 0.51 > 0.5 -> kept.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(2.0), res(1.49)}}
	cfg, _ := hcConfig(t, eval)
	task := hcTask(OptimizationMinimize, u32(1), false, f64(0.5), u32(1))
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	// Improvement kept -> stagnation stays 0 -> ends on turn budget, not stagnation.
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("got %+v", r)
	}
}

// ============================================================================
// Revert on / off (Decision 1)
// ============================================================================

func TestHillClimbingRevertOnRegress(t *testing.T) {
	// minimize: baseline 2.0 kept, iter1 3.0 worse -> discard. revert true issues
	// exactly one `git reset --hard HEAD`.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(2.0), res(3.0)}}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMinimize, u32(1), true, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	resets := 0
	for _, c := range sb.commands {
		if len(c) == 4 && c[0] == "git" && c[1] == "reset" && c[2] == "--hard" && c[3] == "HEAD" {
			resets++
		}
	}
	if resets != 1 {
		t.Fatalf("expected exactly 1 git reset, got %d (commands: %v)", resets, sb.commands)
	}
}

func TestHillClimbingRevertOffRegress(t *testing.T) {
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(2.0), res(3.0)}}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMinimize, u32(1), false, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	for _, c := range sb.commands {
		if len(c) >= 2 && c[0] == "git" && c[1] == "reset" {
			t.Fatalf("revert OFF but a git reset was issued: %v", sb.commands)
		}
	}
}

// ============================================================================
// Stagnation halt / reset
// ============================================================================

func TestHillClimbingStagnationHalt(t *testing.T) {
	// maximize: baseline 5.0, then three regresses to 4.0; max_stagnation 3 halts
	// after the third consecutive non-improvement; best stays 5.0.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(5.0), res(4.0), res(4.0), res(4.0)}}
	cfg, _ := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(3), false, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	if r.Reason.Iterations != 3 || r.Reason.BestMetric != 5.0 {
		t.Fatalf("iterations=%d best=%v, want 3 / 5.0", r.Reason.Iterations, r.Reason.BestMetric)
	}
}

func TestHillClimbingStagnationResetOnImprove(t *testing.T) {
	// maximize: baseline 1.0, iter1 0.5 (discard), iter2 0.5 (discard, stag=2),
	// iter3 2.0 (kept -> stag resets to 0), iter4 1.0 (discard, stag=1). With
	// max_stagnation 3 and max_turns 4 the run ends on the turn budget.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(1.0), res(0.5), res(0.5), res(2.0), res(1.0)}}
	cfg, _ := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(3), false, nil, u32(4))
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("got %+v (stagnation should have reset on the improve)", r)
	}
}

// ============================================================================
// Crash / timeout counts as non-improvement
// ============================================================================

func TestHillClimbingCrashCountsAsNonImprovement(t *testing.T) {
	// maximize: baseline 1.0 kept, iter1 crash -> non-improvement; metric EMPTY;
	// status crashed. max_stagnation 1 halts.
	eval := &scriptedMetricEvaluator{
		results: []*HillClimbMetricResult{res(1.0), nil},
		errors:  []*HillClimbMetricError{nil, {Status: HillClimbCrashed, Message: "segfault"}},
	}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(1), false, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	if r.Reason.BestMetric != 1.0 {
		t.Fatalf("best_metric = %v, want 1.0", r.Reason.BestMetric)
	}
	tsv := readTSV(t, sb, task.ID)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
		"1\t\t\tmaximize\tcrashed\t0.000000\tscripted metric\n"
	if tsv != want {
		t.Fatalf("TSV mismatch:\n got %q\nwant %q", tsv, want)
	}
}

func TestHillClimbingTimeoutCountsAsNonImprovement(t *testing.T) {
	eval := &scriptedMetricEvaluator{
		results: []*HillClimbMetricResult{res(1.0), nil},
		errors:  []*HillClimbMetricError{nil, {Status: HillClimbTimeout, Message: "timed out"}},
	}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, u32(1), false, nil, nil)
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltStagnationLimitReached {
		t.Fatalf("got %+v", r)
	}
	tsv := readTSV(t, sb, task.ID)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric\n" +
		"1\t\t\tmaximize\ttimeout\t0.000000\tscripted metric\n"
	if tsv != want {
		t.Fatalf("TSV mismatch:\n got %q\nwant %q", tsv, want)
	}
}

// ============================================================================
// Budget gate
// ============================================================================

func TestHillClimbingBudgetGate(t *testing.T) {
	// max_turns 0: the budget gate trips before any agent turn; only the baseline
	// row is written.
	eval := &scriptedMetricEvaluator{results: []*HillClimbMetricResult{res(1.0)}}
	cfg, sb := hcConfig(t, eval)
	task := hcTask(OptimizationMaximize, nil, false, nil, u32(0))
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(task))
	if r.Kind != RunFailure || r.Reason.Kind != HaltBudgetExceeded {
		t.Fatalf("got %+v", r)
	}
	tsv := readTSV(t, sb, task.ID)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t1.000000\tmaximize\tkept\t0.000000\tscripted metric\n"
	if tsv != want {
		t.Fatalf("TSV mismatch:\n got %q\nwant %q", tsv, want)
	}
	if len(sb.commands) != 0 {
		t.Fatalf("expected no agent commands under a 0-turn budget, got %v", sb.commands)
	}
}

// ============================================================================
// Exact TSV byte content / float formatting (Decisions 2/3)
// ============================================================================

func TestRenderHillClimbingTSVExactBytes(t *testing.T) {
	rows := []hillClimbRow{
		{iteration: 0, commitHash: "", metricValue: 1.0, hasMetric: true, direction: OptimizationMaximize, status: HillClimbKept, duration: 0, description: "d"},
		{iteration: 1, commitHash: "", metricValue: 2.5, hasMetric: true, direction: OptimizationMaximize, status: HillClimbKept, duration: 1500 * time.Millisecond, description: "d"},
		{iteration: 2, commitHash: "abc123", hasMetric: false, direction: OptimizationMaximize, status: HillClimbCrashed, duration: 0, description: "d"},
	}
	got := renderHillClimbingTSV(rows)
	want := "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n" +
		"0\t\t1.000000\tmaximize\tkept\t0.000000\td\n" +
		"1\t\t2.500000\tmaximize\tkept\t1.500000\td\n" +
		"2\tabc123\t\tmaximize\tcrashed\t0.000000\td\n"
	if got != want {
		t.Fatalf("render mismatch:\n got %q\nwant %q", got, want)
	}
}

// ============================================================================
// ShouldKeep parity (root-package mirror of metric.ShouldKeep)
// ============================================================================

// ============================================================================
// HaltReason JSON round-trip (new variants)
// ============================================================================

func TestHillClimbingHaltReasonsRoundTrip(t *testing.T) {
	cases := []HaltReason{
		{Kind: HaltHillClimbingMisconfigured, Reason: "evaluator is nil"},
		{Kind: HaltStagnationLimitReached, Iterations: 3, BestMetric: 5.0},
	}
	for _, want := range cases {
		raw, err := json.Marshal(want)
		if err != nil {
			t.Fatalf("marshal %q: %v", want.Kind, err)
		}
		var got HaltReason
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("unmarshal %q: %v", want.Kind, err)
		}
		if got.Kind != want.Kind || got.Reason != want.Reason || got.Iterations != want.Iterations || got.BestMetric != want.BestMetric {
			t.Fatalf("round-trip mismatch for %q: got %+v want %+v (raw=%s)", want.Kind, got, want, raw)
		}
	}
}

func TestHillClimbShouldKeepStrictBoundary(t *testing.T) {
	// minimize: equal scores are NOT kept.
	if hillClimbShouldKeep(2.0, 2.0, OptimizationMinimize, nil) {
		t.Fatal("equal score should be discarded")
	}
	// minimize: strictly lower is kept.
	if !hillClimbShouldKeep(1.9, 2.0, OptimizationMinimize, nil) {
		t.Fatal("strictly lower should be kept")
	}
	// exactly min_delta is NOT kept; just over is kept.
	if hillClimbShouldKeep(1.5, 2.0, OptimizationMinimize, f64(0.5)) {
		t.Fatal("exactly min_delta should be discarded")
	}
	if !hillClimbShouldKeep(1.49, 2.0, OptimizationMinimize, f64(0.5)) {
		t.Fatal("just over min_delta should be kept")
	}
	// maximize: strictly higher is kept.
	if !hillClimbShouldKeep(2.1, 2.0, OptimizationMaximize, nil) {
		t.Fatal("strictly higher should be kept")
	}
}
