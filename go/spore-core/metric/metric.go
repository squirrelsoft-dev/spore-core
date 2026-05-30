// Package metric — issue #23 `MetricEvaluator`: pluggable scoring for the
// `HillClimbing` loop strategy.
//
// See `docs/harness-engineering-concepts.md` § "Loop Strategies / HillClimbing"
// and issue #23 for the authoritative rules. This package ships:
//   - MetricError / MetricResult — the error and success surfaces.
//   - The MetricEvaluator interface — used by the harness to score each
//     iteration of a HillClimbing run.
//   - Standard evaluators: CommandMetricEvaluator, TestPassRateEvaluator,
//     LatencyEvaluator, LlmJudgeEvaluator.
//   - ResultsEntry / IterationStatus — the row format the harness writes to
//     `.spore/results/{task_id}.tsv`.
//   - ShouldKeep — the keep/revert decision the harness applies after each
//     iteration.
//
// Rules enforced
//   - Evaluate receives the SandboxProvider. All subprocess execution goes
//     through it; evaluators never touch os/exec directly.
//   - CommandMetricEvaluator writes captured stdout+stderr to LogOutputTo
//     *before* parsing the metric so a partial run is still diagnosable.
//   - A regex that does not match captured output is MetricErrParseFailed,
//     not a crash.
//   - Non-zero exit from the subprocess maps to MetricErrCrashed; an
//     exceeded timeout maps to MetricErrTimeout. Both are valid iteration
//     outcomes — the harness logs them and asks the agent to try a
//     different approach.
//   - ShouldKeep strictly compares against currentBest: a delta of exactly
//     minDelta (or 0.0 when unset) does NOT count as improvement. Equal
//     scores are discarded.
package metric

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/termination"
)

// ============================================================================
// MetricError (tagged union via Kind)
// ============================================================================

// MetricErrorKind discriminates MetricError variants. Tag values match the
// Rust serde rename_all = "snake_case" for cross-language fixture
// round-tripping.
type MetricErrorKind string

const (
	// MetricErrExecutionFailed — evaluator could not run (invalid config,
	// sandbox rejection, invalid regex, etc.).
	MetricErrExecutionFailed MetricErrorKind = "execution_failed"
	// MetricErrTimeout — evaluator timed out.
	MetricErrTimeout MetricErrorKind = "timeout"
	// MetricErrParseFailed — could not parse a metric from the captured output.
	MetricErrParseFailed MetricErrorKind = "parse_failed"
	// MetricErrCrashed — experiment ran but crashed (non-zero exit, OOM, …).
	MetricErrCrashed MetricErrorKind = "crashed"
)

// MetricError is the typed error returned by MetricEvaluator.Evaluate.
//
// Exactly one variant's fields are populated, selected by Kind. Field naming
// mirrors the Rust serde tags so the JSON wire format is byte-equivalent.
type MetricError struct {
	Kind MetricErrorKind

	// execution_failed
	Reason string
	// timeout
	After time.Duration
	// parse_failed
	Output  string
	Pattern string
	// crashed
	Log string
}

// NewExecutionFailed builds an execution_failed MetricError.
func NewExecutionFailed(reason string) *MetricError {
	return &MetricError{Kind: MetricErrExecutionFailed, Reason: reason}
}

// NewTimeout builds a timeout MetricError.
func NewTimeout(after time.Duration) *MetricError {
	return &MetricError{Kind: MetricErrTimeout, After: after}
}

// NewParseFailed builds a parse_failed MetricError.
func NewParseFailed(output, pattern string) *MetricError {
	return &MetricError{Kind: MetricErrParseFailed, Output: output, Pattern: pattern}
}

// NewCrashed builds a crashed MetricError.
func NewCrashed(log string) *MetricError {
	return &MetricError{Kind: MetricErrCrashed, Log: log}
}

// Error implements the error interface.
func (e *MetricError) Error() string {
	switch e.Kind {
	case MetricErrExecutionFailed:
		return fmt.Sprintf("execution failed: %s", e.Reason)
	case MetricErrTimeout:
		return fmt.Sprintf("evaluator timed out after %s", e.After)
	case MetricErrParseFailed:
		return fmt.Sprintf("could not parse metric from output (pattern: %s)", e.Pattern)
	case MetricErrCrashed:
		return fmt.Sprintf("experiment crashed: %s", e.Log)
	default:
		return fmt.Sprintf("metric error: unknown kind %q", e.Kind)
	}
}

// MarshalJSON serialises MetricError as a flat tagged object matching the
// Rust serde shape.
func (e MetricError) MarshalJSON() ([]byte, error) {
	switch e.Kind {
	case MetricErrExecutionFailed:
		return json.Marshal(struct {
			Kind   MetricErrorKind `json:"kind"`
			Reason string          `json:"reason"`
		}{e.Kind, e.Reason})
	case MetricErrTimeout:
		return json.Marshal(struct {
			Kind  MetricErrorKind `json:"kind"`
			After float64         `json:"after"`
		}{e.Kind, e.After.Seconds()})
	case MetricErrParseFailed:
		return json.Marshal(struct {
			Kind    MetricErrorKind `json:"kind"`
			Output  string          `json:"output"`
			Pattern string          `json:"pattern"`
		}{e.Kind, e.Output, e.Pattern})
	case MetricErrCrashed:
		return json.Marshal(struct {
			Kind MetricErrorKind `json:"kind"`
			Log  string          `json:"log"`
		}{e.Kind, e.Log})
	default:
		return nil, fmt.Errorf("MetricError: unknown kind %q", e.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (e *MetricError) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind    MetricErrorKind `json:"kind"`
		Reason  string          `json:"reason"`
		After   float64         `json:"after"`
		Output  string          `json:"output"`
		Pattern string          `json:"pattern"`
		Log     string          `json:"log"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Kind = probe.Kind
	switch probe.Kind {
	case MetricErrExecutionFailed:
		e.Reason = probe.Reason
	case MetricErrTimeout:
		e.After = time.Duration(probe.After * float64(time.Second))
	case MetricErrParseFailed:
		e.Output = probe.Output
		e.Pattern = probe.Pattern
	case MetricErrCrashed:
		e.Log = probe.Log
	default:
		return fmt.Errorf("MetricError: unknown kind %q", probe.Kind)
	}
	return nil
}

// ============================================================================
// MetricResult
// ============================================================================

// MetricResult is the successful outcome of a MetricEvaluator.Evaluate call.
type MetricResult struct {
	Value     float64           `json:"value"`
	RawOutput string            `json:"raw_output"`
	Duration  time.Duration     `json:"-"`
	Metadata  map[string]string `json:"metadata"`
}

// NewMetricResult constructs a MetricResult with sensible zero values.
func NewMetricResult(value float64) MetricResult {
	return MetricResult{Value: value, Metadata: map[string]string{}}
}

// MarshalJSON serialises Duration as fractional seconds (Rust-compatible)
// and ensures Metadata serialises as {} rather than null.
func (r MetricResult) MarshalJSON() ([]byte, error) {
	md := r.Metadata
	if md == nil {
		md = map[string]string{}
	}
	return json.Marshal(struct {
		Value     float64           `json:"value"`
		RawOutput string            `json:"raw_output"`
		Duration  float64           `json:"duration"`
		Metadata  map[string]string `json:"metadata"`
	}{r.Value, r.RawOutput, r.Duration.Seconds(), md})
}

// UnmarshalJSON decodes the fractional-seconds Duration shape.
func (r *MetricResult) UnmarshalJSON(data []byte) error {
	var probe struct {
		Value     float64           `json:"value"`
		RawOutput string            `json:"raw_output"`
		Duration  float64           `json:"duration"`
		Metadata  map[string]string `json:"metadata"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	r.Value = probe.Value
	r.RawOutput = probe.RawOutput
	r.Duration = time.Duration(probe.Duration * float64(time.Second))
	r.Metadata = probe.Metadata
	if r.Metadata == nil {
		r.Metadata = map[string]string{}
	}
	return nil
}

// ============================================================================
// MetricEvaluator interface
// ============================================================================

// MetricEvaluator is the pluggable scoring strategy for the HillClimbing
// loop. The harness calls Evaluate after the agent completes each iteration
// and feeds the result into ShouldKeep.
type MetricEvaluator interface {
	Evaluate(ctx context.Context, sandbox sporecore.SandboxProvider, sessionState *termination.SessionStateSnapshot) (MetricResult, *MetricError)
	Direction() sporecore.OptimizationDirection
	Description() string
}

// ============================================================================
// ShouldKeep
// ============================================================================

// ShouldKeep is the keep-or-revert decision the harness applies after every
// iteration.
//
// Returns true only when newValue strictly beats currentBest by more than
// minDelta (default 0.0 when nil). Equal scores are discarded — a flat run
// is not progress.
func ShouldKeep(newValue, currentBest float64, direction sporecore.OptimizationDirection, minDelta *float64) bool {
	var delta float64
	switch direction {
	case sporecore.OptimizationMinimize:
		delta = currentBest - newValue
	case sporecore.OptimizationMaximize:
		delta = newValue - currentBest
	default:
		delta = newValue - currentBest
	}
	threshold := 0.0
	if minDelta != nil {
		threshold = *minDelta
	}
	return delta > threshold
}

// ============================================================================
// ResultsEntry / IterationStatus
// ============================================================================

// IterationStatus is the per-iteration outcome the harness records in
// `.spore/results/{task_id}.tsv`.
type IterationStatus string

const (
	// IterationKept — metric improved, change kept.
	IterationKept IterationStatus = "kept"
	// IterationDiscarded — metric did not improve, change reverted.
	IterationDiscarded IterationStatus = "discarded"
	// IterationCrashed — evaluator returned MetricErrCrashed (or any non-timeout error).
	IterationCrashed IterationStatus = "crashed"
	// IterationTimeout — evaluator returned MetricErrTimeout.
	IterationTimeout IterationStatus = "timeout"
)

// IterationStatusFromError maps an evaluator error to the iteration status
// the harness records. Successful evaluations are routed through ShouldKeep
// to resolve Kept vs Discarded.
func IterationStatusFromError(err *MetricError) IterationStatus {
	if err == nil {
		return IterationCrashed
	}
	if err.Kind == MetricErrTimeout {
		return IterationTimeout
	}
	return IterationCrashed
}

// ResultsEntry is the structured row format the harness writes for each
// HillClimbing iteration.
type ResultsEntry struct {
	Iteration   uint32                          `json:"iteration"`
	CommitHash  *string                         `json:"commit_hash"`
	MetricValue float64                         `json:"metric_value"`
	Direction   sporecore.OptimizationDirection `json:"direction"`
	Status      IterationStatus                 `json:"status"`
	Duration    time.Duration                   `json:"-"`
	Description string                          `json:"description"`
	Metadata    map[string]string               `json:"metadata"`
}

// MarshalJSON serialises Duration as fractional seconds (Rust-compatible).
func (e ResultsEntry) MarshalJSON() ([]byte, error) {
	md := e.Metadata
	if md == nil {
		md = map[string]string{}
	}
	return json.Marshal(struct {
		Iteration   uint32                          `json:"iteration"`
		CommitHash  *string                         `json:"commit_hash"`
		MetricValue float64                         `json:"metric_value"`
		Direction   sporecore.OptimizationDirection `json:"direction"`
		Status      IterationStatus                 `json:"status"`
		Duration    float64                         `json:"duration"`
		Description string                          `json:"description"`
		Metadata    map[string]string               `json:"metadata"`
	}{e.Iteration, e.CommitHash, e.MetricValue, e.Direction, e.Status, e.Duration.Seconds(), e.Description, md})
}

// UnmarshalJSON decodes the fractional-seconds Duration shape.
func (e *ResultsEntry) UnmarshalJSON(data []byte) error {
	var probe struct {
		Iteration   uint32                          `json:"iteration"`
		CommitHash  *string                         `json:"commit_hash"`
		MetricValue float64                         `json:"metric_value"`
		Direction   sporecore.OptimizationDirection `json:"direction"`
		Status      IterationStatus                 `json:"status"`
		Duration    float64                         `json:"duration"`
		Description string                          `json:"description"`
		Metadata    map[string]string               `json:"metadata"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	e.Iteration = probe.Iteration
	e.CommitHash = probe.CommitHash
	e.MetricValue = probe.MetricValue
	e.Direction = probe.Direction
	e.Status = probe.Status
	e.Duration = time.Duration(probe.Duration * float64(time.Second))
	e.Description = probe.Description
	e.Metadata = probe.Metadata
	if e.Metadata == nil {
		e.Metadata = map[string]string{}
	}
	return nil
}

// ============================================================================
// Internal helpers
// ============================================================================

// ParseMetric extracts the first capture group of pattern from output and
// parses it as a float64.
//
//   - An invalid regex is MetricErrExecutionFailed.
//   - No match, no capture group, or unparseable capture is MetricErrParseFailed.
func ParseMetric(output, pattern string) (float64, *MetricError) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return 0, NewExecutionFailed(fmt.Sprintf("invalid regex %q: %s", pattern, err))
	}
	m := re.FindStringSubmatch(output)
	if m == nil || len(m) < 2 {
		return 0, NewParseFailed(output, pattern)
	}
	v, perr := strconv.ParseFloat(strings.TrimSpace(m[1]), 64)
	if perr != nil {
		return 0, NewParseFailed(output, pattern)
	}
	return v, nil
}

// writeLog best-effort writes body to {sandbox.WorkspaceRoot()}/path. Failures
// are silent — the diagnostic is non-critical and we never want a log-write
// failure to mask the underlying metric result.
func writeLog(sandbox sporecore.SandboxProvider, path, body string) {
	full := filepath.Join(sandbox.WorkspaceRoot(), path)
	_ = os.MkdirAll(filepath.Dir(full), 0o755)
	_ = os.WriteFile(full, []byte(body), 0o644)
}

// ============================================================================
// CommandMetricEvaluator
// ============================================================================

// CommandMetricEvaluator runs a shell command through the sandbox and
// parses a numeric metric out of its combined stdout+stderr via a
// single-capture-group regex. Models the autoresearch pattern
// (`uv run train.py` ⇒ `val_bpb`).
type CommandMetricEvaluator struct {
	Command       string
	Args          []string
	MetricPattern string
	Timeout       time.Duration
	LogOutputTo   string
	WorkingDir    string
	Dir           sporecore.OptimizationDirection
	Desc          string
}

// Evaluate runs the command, writes combined stdout+stderr to LogOutputTo
// (always, before parsing), then parses the metric value.
func (c *CommandMetricEvaluator) Evaluate(ctx context.Context, sandbox sporecore.SandboxProvider, _ *termination.SessionStateSnapshot) (MetricResult, *MetricError) {
	start := time.Now()
	out, viol := sandbox.ExecuteCommand(ctx, c.Command, c.Args, c.WorkingDir, c.Timeout)
	if viol != nil {
		return MetricResult{}, NewExecutionFailed(fmt.Sprintf("sandbox rejected command: %s", viol.Error()))
	}
	combined := out.Stdout + out.Stderr
	// Always redirect to LogOutputTo BEFORE parsing so a parse failure is
	// still diagnosable.
	writeLog(sandbox, c.LogOutputTo, combined)

	if out.TimedOut {
		return MetricResult{}, NewTimeout(c.Timeout)
	}
	if out.ExitCode != 0 {
		return MetricResult{}, NewCrashed(combined)
	}
	value, perr := ParseMetric(combined, c.MetricPattern)
	if perr != nil {
		return MetricResult{}, perr
	}
	return MetricResult{
		Value:     value,
		RawOutput: combined,
		Duration:  time.Since(start),
		Metadata: map[string]string{
			"command":   c.Command,
			"exit_code": strconv.Itoa(out.ExitCode),
		},
	}, nil
}

// Direction returns the configured optimisation direction.
func (c *CommandMetricEvaluator) Direction() sporecore.OptimizationDirection { return c.Dir }

// Description returns the configured human-readable description.
func (c *CommandMetricEvaluator) Description() string { return c.Desc }

// ============================================================================
// TestPassRateEvaluator
// ============================================================================

// TestPassRateEvaluator runs a test suite, extracts pass / total counts via
// two regexes, and reports the fraction of passing tests in [0.0, 1.0].
// Direction is fixed to Maximize.
type TestPassRateEvaluator struct {
	Command      string
	Args         []string
	Timeout      time.Duration
	PassPattern  string
	TotalPattern string
	WorkingDir   string
}

// Evaluate runs the command and computes pass/total.
func (t *TestPassRateEvaluator) Evaluate(ctx context.Context, sandbox sporecore.SandboxProvider, _ *termination.SessionStateSnapshot) (MetricResult, *MetricError) {
	start := time.Now()
	out, viol := sandbox.ExecuteCommand(ctx, t.Command, t.Args, t.WorkingDir, t.Timeout)
	if viol != nil {
		return MetricResult{}, NewExecutionFailed(fmt.Sprintf("sandbox rejected command: %s", viol.Error()))
	}
	combined := out.Stdout + out.Stderr
	if out.TimedOut {
		return MetricResult{}, NewTimeout(t.Timeout)
	}
	// A failing test run is a normal outcome here — we still want the
	// pass-rate. Only treat the run as crashed if we cannot parse it.
	pass, perr := ParseMetric(combined, t.PassPattern)
	if perr != nil {
		return MetricResult{}, perr
	}
	total, terr := ParseMetric(combined, t.TotalPattern)
	if terr != nil {
		return MetricResult{}, terr
	}
	if total <= 0.0 {
		return MetricResult{}, NewParseFailed(combined, t.TotalPattern)
	}
	return MetricResult{
		Value:     pass / total,
		RawOutput: combined,
		Duration:  time.Since(start),
		Metadata: map[string]string{
			"pass":  strconv.FormatFloat(pass, 'f', -1, 64),
			"total": strconv.FormatFloat(total, 'f', -1, 64),
		},
	}, nil
}

// Direction is fixed to Maximize.
func (t *TestPassRateEvaluator) Direction() sporecore.OptimizationDirection {
	return sporecore.OptimizationMaximize
}

// Description returns a human-readable description.
func (t *TestPassRateEvaluator) Description() string {
	return fmt.Sprintf("test pass rate (%s)", t.Command)
}

// ============================================================================
// LatencyEvaluator
// ============================================================================

// LatencyEvaluator measures wall-clock latency of Command, averaged over
// MeasuredRuns trials after WarmupRuns warm-ups. Direction is fixed to
// Minimize.
type LatencyEvaluator struct {
	Command      string
	Args         []string
	WarmupRuns   uint32
	MeasuredRuns uint32
	Timeout      time.Duration
	WorkingDir   string
}

// Evaluate runs warm-ups then averages the wall-clock duration over
// MeasuredRuns trials.
func (l *LatencyEvaluator) Evaluate(ctx context.Context, sandbox sporecore.SandboxProvider, _ *termination.SessionStateSnapshot) (MetricResult, *MetricError) {
	if l.MeasuredRuns == 0 {
		return MetricResult{}, NewExecutionFailed("measured_runs must be > 0")
	}
	start := time.Now()

	for i := uint32(0); i < l.WarmupRuns; i++ {
		_, viol := sandbox.ExecuteCommand(ctx, l.Command, l.Args, l.WorkingDir, l.Timeout)
		if viol != nil {
			return MetricResult{}, NewExecutionFailed(fmt.Sprintf("sandbox rejected command: %s", viol.Error()))
		}
	}

	var total time.Duration
	var lastOutput string
	for i := uint32(0); i < l.MeasuredRuns; i++ {
		trialStart := time.Now()
		out, viol := sandbox.ExecuteCommand(ctx, l.Command, l.Args, l.WorkingDir, l.Timeout)
		if viol != nil {
			return MetricResult{}, NewExecutionFailed(fmt.Sprintf("sandbox rejected command: %s", viol.Error()))
		}
		if out.TimedOut {
			return MetricResult{}, NewTimeout(l.Timeout)
		}
		if out.ExitCode != 0 {
			return MetricResult{}, NewCrashed(out.Stdout + out.Stderr)
		}
		total += time.Since(trialStart)
		lastOutput = out.Stdout + out.Stderr
	}

	avgSecs := total.Seconds() / float64(l.MeasuredRuns)
	return MetricResult{
		Value:     avgSecs,
		RawOutput: lastOutput,
		Duration:  time.Since(start),
		Metadata: map[string]string{
			"warmup_runs":   strconv.FormatUint(uint64(l.WarmupRuns), 10),
			"measured_runs": strconv.FormatUint(uint64(l.MeasuredRuns), 10),
		},
	}, nil
}

// Direction is fixed to Minimize.
func (l *LatencyEvaluator) Direction() sporecore.OptimizationDirection {
	return sporecore.OptimizationMinimize
}

// Description returns a human-readable description.
func (l *LatencyEvaluator) Description() string {
	return fmt.Sprintf("latency (%s)", l.Command)
}

// ============================================================================
// LlmJudgeEvaluator
// ============================================================================

// JudgeModelConfig describes the LLM judge for the results log. The actual
// dispatch goes through a ModelInterface supplied at construction time.
type JudgeModelConfig struct {
	Provider string                `json:"provider"`
	ModelID  string                `json:"model_id"`
	Params   sporecore.ModelParams `json:"params"`
}

// LlmJudgeEvaluator uses an LLM-as-judge to score SampleInput against
// Rubric. The judge is expected to emit a `score: <number>` line; that
// number is normalised into [0.0, 1.0] using ScoreRange. Direction is
// fixed to Maximize.
type LlmJudgeEvaluator struct {
	JudgeModel  JudgeModelConfig
	Rubric      string
	ScoreRange  [2]float64
	SampleInput string
	Client      sporecore.ModelInterface
}

// Evaluate calls the judge model, extracts the score, clamps it to
// ScoreRange, and normalises into [0, 1].
func (j *LlmJudgeEvaluator) Evaluate(ctx context.Context, _ sporecore.SandboxProvider, _ *termination.SessionStateSnapshot) (MetricResult, *MetricError) {
	start := time.Now()
	prompt := fmt.Sprintf(
		"%s\n\nInput to evaluate:\n%s\n\nReply with a single line `score: <number>` where the number is within [%g, %g].",
		j.Rubric, j.SampleInput, j.ScoreRange[0], j.ScoreRange[1],
	)
	req := sporecore.ModelRequest{
		Messages: []sporecore.Message{{
			Role:    sporecore.RoleUser,
			Content: sporecore.NewTextContent(prompt),
		}},
		Tools:  []sporecore.ToolSchema{},
		Params: j.JudgeModel.Params,
		Stream: false,
	}
	resp, err := j.Client.Call(ctx, req)
	if err != nil {
		return MetricResult{}, NewExecutionFailed(fmt.Sprintf("judge model call failed: %s", err))
	}
	var b strings.Builder
	for i, blk := range resp.Content {
		if blk.Type == sporecore.ContentBlockTypeText {
			if i > 0 && b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString(blk.Text)
		}
	}
	text := b.String()

	value, perr := j.parseScore(text)
	if perr != nil {
		return MetricResult{}, perr
	}
	return MetricResult{
		Value:     value,
		RawOutput: text,
		Duration:  time.Since(start),
		Metadata: map[string]string{
			"judge_model":    j.JudgeModel.ModelID,
			"judge_provider": j.JudgeModel.Provider,
		},
	}, nil
}

// parseScore extracts `score: <number>` (case-insensitive), clamps to
// ScoreRange, then normalises into [0, 1].
func (j *LlmJudgeEvaluator) parseScore(text string) (float64, *MetricError) {
	raw, perr := ParseMetric(text, `(?i)score\s*:\s*([-+]?\d+(?:\.\d+)?)`)
	if perr != nil {
		return 0, perr
	}
	lo, hi := j.ScoreRange[0], j.ScoreRange[1]
	if hi <= lo {
		return 0, NewExecutionFailed(fmt.Sprintf("invalid score_range: (%g, %g)", lo, hi))
	}
	clamped := raw
	if clamped < lo {
		clamped = lo
	} else if clamped > hi {
		clamped = hi
	}
	return (clamped - lo) / (hi - lo), nil
}

// Direction is fixed to Maximize.
func (j *LlmJudgeEvaluator) Direction() sporecore.OptimizationDirection {
	return sporecore.OptimizationMaximize
}

// Description returns a human-readable description.
func (j *LlmJudgeEvaluator) Description() string {
	return fmt.Sprintf("llm judge (%s/%s)", j.JudgeModel.Provider, j.JudgeModel.ModelID)
}

// ============================================================================
// Harness seam bridge (issue #60)
// ============================================================================

// harnessMetricEvaluator adapts a metric.MetricEvaluator to the consumer-side
// sporecore.MetricEvaluator seam the HillClimbing loop reads from config. The
// root sporecore package cannot import this package (metric imports sporecore —
// cycle), so it declares its own narrow MetricEvaluator interface; this adapter
// bridges the two and is the analogue of verifier.AsHarnessMetricEvaluator.
type harnessMetricEvaluator struct {
	inner MetricEvaluator
}

// AsHarnessMetricEvaluator wraps a metric.MetricEvaluator so it can be dropped
// straight into sporecore.HarnessConfig.MetricEvaluator. Returns nil when inner
// is nil so a nil evaluator stays nil through the seam (the HillClimbing
// strategy treats a nil config.MetricEvaluator as HillClimbingMisconfigured).
func AsHarnessMetricEvaluator(inner MetricEvaluator) sporecore.MetricEvaluator {
	if inner == nil {
		return nil
	}
	return harnessMetricEvaluator{inner: inner}
}

// Evaluate implements sporecore.MetricEvaluator. It rebuilds the
// SessionStateSnapshot the metric evaluators expect from the loose
// (sessionID, taskID, state) the harness threads through the seam, then
// translates the metric.MetricResult / metric.MetricError into the root-package
// mirror types.
func (h harnessMetricEvaluator) Evaluate(
	ctx context.Context,
	sandbox sporecore.SandboxProvider,
	sessionID sporecore.SessionID,
	taskID sporecore.TaskID,
	state sporecore.SessionState,
) (*sporecore.HillClimbMetricResult, *sporecore.HillClimbMetricError) {
	snapshot := termination.NewSessionStateSnapshotWithRoot(sessionID, taskID, state, sandbox.WorkspaceRoot())
	res, merr := h.inner.Evaluate(ctx, sandbox, &snapshot)
	if merr != nil {
		status := sporecore.HillClimbCrashed
		if IterationStatusFromError(merr) == IterationTimeout {
			status = sporecore.HillClimbTimeout
		}
		return nil, &sporecore.HillClimbMetricError{Status: status, Message: merr.Error()}
	}
	return &sporecore.HillClimbMetricResult{Value: res.Value, Duration: res.Duration}, nil
}

// Description implements sporecore.MetricEvaluator.
func (h harnessMetricEvaluator) Description() string { return h.inner.Description() }

// Compile-time checks.
var (
	_ MetricEvaluator           = (*CommandMetricEvaluator)(nil)
	_ MetricEvaluator           = (*TestPassRateEvaluator)(nil)
	_ MetricEvaluator           = (*LatencyEvaluator)(nil)
	_ MetricEvaluator           = (*LlmJudgeEvaluator)(nil)
	_ sporecore.MetricEvaluator = harnessMetricEvaluator{}
)
