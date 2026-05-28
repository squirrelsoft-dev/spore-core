// Package verifier defines the TaskVerifier interface + standard
// implementations.
//
// Rules enforced here:
//   - 9  IsDeterministic() true for test-suite/result verifiers, false for the
//     LLM judge.
//   - 10 TestSuiteVerifier: command pass-rate; passed = score==1.0; deterministic.
//   - 11 CompositeVerifier: weighted mean; passed = all required; det = AND.
//   - 12 MetricEvaluatorVerifier: wraps a MetricEvaluator, normalizes value.
//   - 13 LlmJudgeVerifier: thin; non-deterministic; judge injected.
package verifier

import (
	"context"
	"fmt"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/metric"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/termination"
	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
)

// TaskVerifier verifies whether a task run satisfied its goal.
type TaskVerifier interface {
	// Verify a completed run against the task, with access to the restored
	// workspace directory.
	Verify(ctx context.Context, t *task.EvalTask, run sporecore.RunResult, workspace string) (task.VerificationResult, error)

	// IsDeterministic reports true for test-suite / result verifiers; false for
	// the LLM judge (Rule 9).
	IsDeterministic() bool
}

// ============================================================================
// BuildVerifier — resolve a VerifierSpec into a TaskVerifier
// ============================================================================

// BuildVerifier resolves a VerifierSpec to a concrete verifier. MetricEvaluator
// specs have no built-in concrete evaluator (it is injected for non-fixture
// use), so they resolve to a normalizing placeholder that scores from the run's
// success flag — adequate for manifest replay; real evaluators are wired via
// NewMetricEvaluatorVerifierWithRange/Threshold.
func BuildVerifier(spec task.VerifierSpec) TaskVerifier {
	switch spec.Kind {
	case task.VerifierTestSuite:
		secs := uint64(60)
		if spec.TimeoutSecs != nil {
			secs = *spec.TimeoutSecs
		}
		return &TestSuiteVerifier{
			Command: spec.Command,
			Args:    spec.Args,
			Timeout: time.Duration(secs) * time.Second,
		}
	case task.VerifierComposite:
		children := make([]compositeChild, 0, len(spec.Children))
		for _, c := range spec.Children {
			children = append(children, compositeChild{
				verifier: BuildVerifier(c.Spec),
				weight:   c.Weight,
				required: c.Required,
			})
		}
		return &CompositeVerifier{children: children}
	case task.VerifierMetricEvaluator:
		return &normalizingSuccessVerifier{
			direction: spec.Direction,
			min:       spec.Min,
			max:       spec.Max,
			threshold: spec.Threshold,
		}
	case task.VerifierLlmJudge:
		return &stubLlmJudgeVerifier{scoreRange: spec.ScoreRange}
	case task.VerifierAlwaysFail:
		return AlwaysFail{}
	case task.VerifierAlwaysPass:
		return AlwaysPass{}
	default:
		// Unknown spec resolves to a placeholder that fails closed.
		return AlwaysFail{}
	}
}

// ============================================================================
// AlwaysPass / AlwaysFail (test scaffolding)
// ============================================================================

// AlwaysPass always passes with score 1.0.
type AlwaysPass struct{}

// Verify implements TaskVerifier.
func (AlwaysPass) Verify(_ context.Context, _ *task.EvalTask, _ sporecore.RunResult, _ string) (task.VerificationResult, error) {
	return task.NewVerificationResult(true, 1.0, "always pass")
}

// IsDeterministic implements TaskVerifier.
func (AlwaysPass) IsDeterministic() bool { return true }

// AlwaysFail always fails with score 0.0.
type AlwaysFail struct{}

// Verify implements TaskVerifier.
func (AlwaysFail) Verify(_ context.Context, _ *task.EvalTask, _ sporecore.RunResult, _ string) (task.VerificationResult, error) {
	return task.NewVerificationResult(false, 0.0, "always fail")
}

// IsDeterministic implements TaskVerifier.
func (AlwaysFail) IsDeterministic() bool { return true }

// ============================================================================
// TestSuiteVerifier (Rule 10)
// ============================================================================

// TestSuiteVerifier runs a command in the workspace; score = pass rate parsed
// from the output, falling back to the exit code (0 => 1.0, nonzero => 0.0).
// passed = (score == 1.0). Deterministic.
type TestSuiteVerifier struct {
	Command string
	Args    []string
	Timeout time.Duration
}

// Verify implements TaskVerifier. It runs the check command directly in the
// restored workspace via a non-isolating direct sandbox.
func (v *TestSuiteVerifier) Verify(ctx context.Context, _ *task.EvalTask, _ sporecore.RunResult, workspace string) (task.VerificationResult, error) {
	sandbox := newDirectSandbox(workspace)
	out, viol := sandbox.ExecuteCommand(ctx, v.Command, v.Args, workspace, v.Timeout)
	if viol != nil {
		return task.VerificationResult{}, &task.VerifyError{Msg: fmt.Sprintf("command rejected: %+v", viol)}
	}
	combined := out.Stdout + out.Stderr
	score, ok := parsePassRate(combined)
	if !ok {
		if out.ExitCode == 0 {
			score = 1.0
		} else {
			score = 0.0
		}
	}
	passed := score == 1.0
	res := task.ClampedVerificationResult(passed, score, fmt.Sprintf("exit=%d pass_rate=%.3f", out.ExitCode, score))
	return res.WithSignal("exit_code", float64(out.ExitCode)).WithSignal("pass_rate", score), nil
}

// IsDeterministic implements TaskVerifier.
func (v *TestSuiteVerifier) IsDeterministic() bool { return true }

// parsePassRate parses a pass-rate from common test-runner output. Returns
// (_, false) if no recognizable counts are present.
func parsePassRate(output string) (float64, bool) {
	passed, okP := scanNumberBefore(output, " passed")
	total, okT := scanNumberBefore(output, " total")
	if !okT {
		total, okT = scanNumberAfter(output, "of ")
	}
	if okP && okT && total > 0.0 {
		r := passed / total
		if r < 0.0 {
			r = 0.0
		}
		if r > 1.0 {
			r = 1.0
		}
		return r, true
	}
	return 0.0, false
}

func scanNumberBefore(s, suffix string) (float64, bool) {
	idx := strings.Index(s, suffix)
	if idx < 0 {
		return 0, false
	}
	head := s[:idx]
	end := len(head)
	start := end
	for start > 0 && head[start-1] >= '0' && head[start-1] <= '9' {
		start--
	}
	if start == end {
		return 0, false
	}
	return parseFloat(head[start:end])
}

func scanNumberAfter(s, prefix string) (float64, bool) {
	idx := strings.Index(s, prefix)
	if idx < 0 {
		return 0, false
	}
	tail := s[idx+len(prefix):]
	end := 0
	for end < len(tail) && tail[end] >= '0' && tail[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0, false
	}
	return parseFloat(tail[:end])
}

func parseFloat(s string) (float64, bool) {
	var v float64
	if _, err := fmt.Sscanf(s, "%g", &v); err != nil {
		return 0, false
	}
	return v, true
}

// ============================================================================
// CompositeVerifier (Rule 11)
// ============================================================================

type compositeChild struct {
	verifier TaskVerifier
	weight   float64
	required bool
}

// CompositeVerifier combines children by weight: score = weighted mean; passed
// = all required children passed; IsDeterministic = AND of children (Rule 11).
type CompositeVerifier struct {
	children []compositeChild
}

// NewCompositeVerifier builds from already-resolved children.
func NewCompositeVerifier(verifiers []TaskVerifier, weights []float64, required []bool) *CompositeVerifier {
	children := make([]compositeChild, len(verifiers))
	for i := range verifiers {
		children[i] = compositeChild{verifier: verifiers[i], weight: weights[i], required: required[i]}
	}
	return &CompositeVerifier{children: children}
}

// Verify implements TaskVerifier.
func (c *CompositeVerifier) Verify(ctx context.Context, t *task.EvalTask, run sporecore.RunResult, workspace string) (task.VerificationResult, error) {
	var weightedSum, weightTotal float64
	allRequiredPassed := true
	details := make([]string, 0, len(c.children))
	for _, child := range c.children {
		r, err := child.verifier.Verify(ctx, t, run, workspace)
		if err != nil {
			return task.VerificationResult{}, err
		}
		weightedSum += r.Score * child.weight
		weightTotal += child.weight
		if child.required && !r.Passed {
			allRequiredPassed = false
		}
		details = append(details, fmt.Sprintf("[w=%v req=%v pass=%v score=%.3f]", child.weight, child.required, r.Passed, r.Score))
	}
	score := 0.0
	if weightTotal > 0.0 {
		score = weightedSum / weightTotal
	}
	return task.ClampedVerificationResult(allRequiredPassed, score, strings.Join(details, " ")), nil
}

// IsDeterministic implements TaskVerifier.
func (c *CompositeVerifier) IsDeterministic() bool {
	for _, child := range c.children {
		if !child.verifier.IsDeterministic() {
			return false
		}
	}
	return true
}

// ============================================================================
// MetricEvaluatorVerifier (Rule 12)
// ============================================================================

// MetricEvaluatorVerifier wraps a metric.MetricEvaluator: runs Evaluate,
// normalizes the value to [0,1] per Direction() and the configured min/max (or
// a threshold). Deterministic iff the wrapped evaluator is (defaults to
// deterministic).
type MetricEvaluatorVerifier struct {
	evaluator     metric.MetricEvaluator
	min           *float64
	max           *float64
	threshold     *float64
	deterministic bool
}

// NewMetricEvaluatorVerifierWithRange wraps an evaluator, normalizing by an
// explicit [min, max] range.
func NewMetricEvaluatorVerifierWithRange(evaluator metric.MetricEvaluator, min, max float64) *MetricEvaluatorVerifier {
	return &MetricEvaluatorVerifier{evaluator: evaluator, min: &min, max: &max, deterministic: true}
}

// NewMetricEvaluatorVerifierWithThreshold wraps an evaluator, scoring 1.0 when
// the value beats threshold in the evaluator's Direction(), else 0.0.
func NewMetricEvaluatorVerifierWithThreshold(evaluator metric.MetricEvaluator, threshold float64) *MetricEvaluatorVerifier {
	return &MetricEvaluatorVerifier{evaluator: evaluator, threshold: &threshold, deterministic: true}
}

// NonDeterministic marks the wrapped evaluator as non-deterministic (e.g. an
// LLM judge) and returns the verifier for chaining.
func (v *MetricEvaluatorVerifier) NonDeterministic() *MetricEvaluatorVerifier {
	v.deterministic = false
	return v
}

func (v *MetricEvaluatorVerifier) normalize(value float64, direction sporecore.OptimizationDirection) float64 {
	if v.threshold != nil {
		var beats bool
		switch direction {
		case sporecore.OptimizationMaximize:
			beats = value >= *v.threshold
		default:
			beats = value <= *v.threshold
		}
		if beats {
			return 1.0
		}
		return 0.0
	}
	if v.min != nil && v.max != nil {
		if *v.max-*v.min == 0 {
			return 0.0
		}
		unit := (value - *v.min) / (*v.max - *v.min)
		if unit < 0.0 {
			unit = 0.0
		}
		if unit > 1.0 {
			unit = 1.0
		}
		if direction == sporecore.OptimizationMinimize {
			return 1.0 - unit
		}
		return unit
	}
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

// Verify implements TaskVerifier.
func (v *MetricEvaluatorVerifier) Verify(ctx context.Context, t *task.EvalTask, run sporecore.RunResult, workspace string) (task.VerificationResult, error) {
	sandbox := newDirectSandbox(workspace)
	sid := sessionIDOf(run)
	snap := termination.NewSessionStateSnapshotWithRoot(sid, t.ID, sporecore.SessionState{}, workspace)
	result, mErr := v.evaluator.Evaluate(ctx, sandbox, &snap)
	if mErr != nil {
		return task.VerificationResult{}, &task.VerifyError{Msg: fmt.Sprintf("evaluator failed: %v", mErr)}
	}
	score := v.normalize(result.Value, v.evaluator.Direction())
	passed := score >= 1.0
	res := task.ClampedVerificationResult(passed, score, fmt.Sprintf("metric value=%v normalized=%.3f", result.Value, score))
	return res.WithSignal("metric_value", result.Value), nil
}

// IsDeterministic implements TaskVerifier.
func (v *MetricEvaluatorVerifier) IsDeterministic() bool { return v.deterministic }

// normalizingSuccessVerifier is the placeholder for a metric_evaluator spec
// resolved from a manifest (no concrete evaluator wired). Scores from the run's
// success flag, applying the spec's direction/range so the surface is exercised
// in replay.
type normalizingSuccessVerifier struct {
	direction task.MetricDirection
	min       *float64
	max       *float64
	threshold *float64
}

func (v *normalizingSuccessVerifier) Verify(_ context.Context, _ *task.EvalTask, run sporecore.RunResult, _ string) (task.VerificationResult, error) {
	success := run.Kind == sporecore.RunSuccess
	value := 0.0
	if success {
		value = 1.0
	}
	// direction/range are informational here; success is already in [0,1].
	_ = v.direction
	_ = v.min
	_ = v.max
	_ = v.threshold
	return task.NewVerificationResult(success, value, "metric-evaluator (manifest placeholder)")
}

func (v *normalizingSuccessVerifier) IsDeterministic() bool { return true }

// ============================================================================
// LlmJudgeVerifier (Rule 13)
// ============================================================================

// LlmJudgeVerifier is a thin LLM-judge verifier. IsDeterministic() == false.
// The concrete judge ModelInterface is injected at construction.
type LlmJudgeVerifier struct {
	Judge      sporecore.ModelInterface
	Rubric     string
	ScoreRange [2]float64
	Params     sporecore.ModelParams
}

// Verify implements TaskVerifier.
func (v *LlmJudgeVerifier) Verify(ctx context.Context, _ *task.EvalTask, run sporecore.RunResult, _ string) (task.VerificationResult, error) {
	output := ""
	if run.Kind == sporecore.RunSuccess {
		output = run.Output
	}
	prompt := fmt.Sprintf(
		"%s\n\nAgent output to evaluate:\n%s\n\nReply with a single line `score: <number>` within [%v, %v].",
		v.Rubric, output, v.ScoreRange[0], v.ScoreRange[1],
	)
	request := sporecore.ModelRequest{
		Messages: []sporecore.Message{{Role: sporecore.RoleUser, Content: sporecore.NewTextContent(prompt)}},
		Params:   v.Params,
	}
	resp, err := v.Judge.Call(ctx, request)
	if err != nil {
		return task.VerificationResult{}, &task.VerifyError{Msg: fmt.Sprintf("judge call failed: %v", err)}
	}
	var parts []string
	for _, b := range resp.Content {
		if b.Type == sporecore.ContentBlockTypeText {
			parts = append(parts, b.Text)
		}
	}
	text := strings.Join(parts, "\n")
	raw, ok := parseScore(text)
	if !ok {
		return task.VerificationResult{}, &task.VerifyError{Msg: fmt.Sprintf("no score in judge reply: %q", text)}
	}
	lo, hi := v.ScoreRange[0], v.ScoreRange[1]
	if hi <= lo {
		return task.VerificationResult{}, &task.VerifyError{Msg: fmt.Sprintf("invalid score_range (%v,%v)", lo, hi)}
	}
	if raw < lo {
		raw = lo
	}
	if raw > hi {
		raw = hi
	}
	score := (raw - lo) / (hi - lo)
	if score < 0.0 {
		score = 0.0
	}
	if score > 1.0 {
		score = 1.0
	}
	return task.NewVerificationResult(score >= 0.5, score, fmt.Sprintf("judge score=%v", raw))
}

// IsDeterministic implements TaskVerifier.
func (v *LlmJudgeVerifier) IsDeterministic() bool { return false }

// stubLlmJudgeVerifier is used when a manifest's llm_judge spec is resolved
// without an injected model. Non-deterministic; scores from the run's success
// flag so the non-deterministic comparison path (bootstrap CI) is exercised.
type stubLlmJudgeVerifier struct {
	scoreRange [2]float64
}

func (s *stubLlmJudgeVerifier) Verify(_ context.Context, _ *task.EvalTask, run sporecore.RunResult, _ string) (task.VerificationResult, error) {
	success := run.Kind == sporecore.RunSuccess
	score := 0.0
	if success {
		score = 1.0
	}
	return task.NewVerificationResult(success, score, "llm-judge (manifest stub)")
}

func (s *stubLlmJudgeVerifier) IsDeterministic() bool { return false }

// parseScore parses a "score: <number>" line (first match wins,
// case-insensitive).
func parseScore(text string) (float64, bool) {
	lower := strings.ToLower(text)
	idx := strings.Index(lower, "score")
	if idx < 0 {
		return 0, false
	}
	after := text[idx+len("score"):]
	after = strings.TrimLeft(after, ": \t")
	end := 0
	for end < len(after) {
		c := after[end]
		if (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' {
			end++
		} else {
			break
		}
	}
	if end == 0 {
		return 0, false
	}
	return parseFloat(after[:end])
}

// sessionIDOf extracts the session id from any RunResult variant.
func sessionIDOf(run sporecore.RunResult) sporecore.SessionID {
	switch run.Kind {
	case sporecore.RunWaitingForHuman:
		if run.State != nil {
			return run.State.SessionID
		}
		return ""
	default:
		return run.SessionID
	}
}
