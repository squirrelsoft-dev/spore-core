// Package task defines the EvalHarness task-suite types: TaskSuite, EvalTask,
// WorkspaceSnapshot, VerifierSpec, VerificationResult, ConfigID, and the
// EvalError taxonomy.
//
// Rules enforced here: 1 (three disjoint lists), 5 (tags free-form), 6
// (suite_version required), 7/8 (verification result shape + score clamp).
//
// The verifier resolution (VerifierSpec -> TaskVerifier) lives in the sibling
// verifier package to avoid an import cycle; EvalTask therefore holds the
// serializable VerifierSpec and the resolved verifier is attached by the
// manifest loader / harness at runtime.
package task

import (
	"errors"
	"fmt"
	"sort"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ConfigID identifies a candidate harness configuration in a comparison.
type ConfigID string

// ============================================================================
// EvalError taxonomy
// ============================================================================

// ErrMissingSuiteVersion is returned when a manifest is loaded without the
// required suite_version field (Rule 6). Match with errors.Is.
var ErrMissingSuiteVersion = errors.New("manifest is missing required field `suite_version`")

// ManifestParseError wraps a manifest parse / serialization failure.
type ManifestParseError struct{ Msg string }

func (e *ManifestParseError) Error() string { return "manifest parse error: " + e.Msg }

// VerifyError is a verifier failure that is not a normal "task failed"
// outcome — e.g. an out-of-range score (Rule 8) or a verifier command that
// could not be run.
type VerifyError struct{ Msg string }

func (e *VerifyError) Error() string { return "verification error: " + e.Msg }

// WorktreeError is a workspace restore / teardown failure (Rules 2-3).
type WorktreeError struct{ Msg string }

func (e *WorktreeError) Error() string { return "worktree error: " + e.Msg }

// MissingMetricsError is returned when an EvalHarness is built or run without
// the metrics it needs.
type MissingMetricsError struct{ Msg string }

func (e *MissingMetricsError) Error() string { return "missing metrics: " + e.Msg }

// ============================================================================
// VerificationResult (Rule 7)
// ============================================================================

// VerificationResult is the outcome of a TaskVerifier (Rule 7): a pass/fail
// flag, a score clamped to [0.0, 1.0], a human-readable detail, and granular
// signals.
type VerificationResult struct {
	Passed  bool               `json:"passed"`
	Score   float64            `json:"score"`
	Detail  string             `json:"detail"`
	Signals map[string]float64 `json:"signals"`
}

// NewVerificationResult builds a result, returning a *VerifyError when score is
// out of [0.0, 1.0] (Rule 8).
func NewVerificationResult(passed bool, score float64, detail string) (VerificationResult, error) {
	if score < 0.0 || score > 1.0 {
		return VerificationResult{}, &VerifyError{Msg: fmt.Sprintf("score %v out of range [0.0, 1.0]", score)}
	}
	return VerificationResult{Passed: passed, Score: score, Detail: detail, Signals: map[string]float64{}}, nil
}

// ClampedVerificationResult builds a result, clamping an out-of-range score
// into [0.0, 1.0] instead of erroring. Use for evaluator-derived scores that
// are guaranteed-finite but may drift slightly outside the unit interval.
func ClampedVerificationResult(passed bool, score float64, detail string) VerificationResult {
	if score < 0.0 {
		score = 0.0
	}
	if score > 1.0 {
		score = 1.0
	}
	return VerificationResult{Passed: passed, Score: score, Detail: detail, Signals: map[string]float64{}}
}

// WithSignal attaches a named signal and returns the result for chaining.
func (r VerificationResult) WithSignal(key string, value float64) VerificationResult {
	if r.Signals == nil {
		r.Signals = map[string]float64{}
	}
	r.Signals[key] = value
	return r
}

// ============================================================================
// MetricDirection (serializable mirror of OptimizationDirection)
// ============================================================================

// MetricDirection is the optimization direction for a metric-evaluator
// verifier spec. Mirrors sporecore.OptimizationDirection but is a
// self-contained serializable spec field.
type MetricDirection string

const (
	// DirectionMinimize — lower values are better.
	DirectionMinimize MetricDirection = "minimize"
	// DirectionMaximize — higher values are better.
	DirectionMaximize MetricDirection = "maximize"
)

// ToOptimizationDirection maps to the core OptimizationDirection.
func (d MetricDirection) ToOptimizationDirection() sporecore.OptimizationDirection {
	if d == DirectionMinimize {
		return sporecore.OptimizationMinimize
	}
	return sporecore.OptimizationMaximize
}

// ============================================================================
// WorkspaceSnapshot (Resolution 2) — tagged union via "kind"
// ============================================================================

// WorkspaceSnapshotKind discriminates WorkspaceSnapshot variants.
type WorkspaceSnapshotKind string

const (
	// SnapshotFiles — a path->contents map; the canonical hermetic form.
	SnapshotFiles WorkspaceSnapshotKind = "files"
	// SnapshotGitRef — a real git snapshot (repo + reference).
	SnapshotGitRef WorkspaceSnapshotKind = "git_ref"
	// SnapshotEmpty — a bare workspace.
	SnapshotEmpty WorkspaceSnapshotKind = "empty"
)

// WorkspaceSnapshot describes how a task's workspace is restored before a run
// (Rule 2). Files is the canonical hermetic form the shipped fixtures use;
// GitRef supports real snapshots; Empty is a bare workspace.
type WorkspaceSnapshot struct {
	Kind WorkspaceSnapshotKind `json:"kind"`
	// files
	Files map[string]string `json:"-"`
	// git_ref
	Repo      string `json:"-"`
	Reference string `json:"-"`
}

// MarshalJSON serialises as a flat tagged object keyed by "kind".
func (w WorkspaceSnapshot) MarshalJSON() ([]byte, error) {
	switch w.Kind {
	case SnapshotFiles:
		files := w.Files
		if files == nil {
			files = map[string]string{}
		}
		return jsonMarshal(struct {
			Kind  WorkspaceSnapshotKind `json:"kind"`
			Files map[string]string     `json:"files"`
		}{w.Kind, files})
	case SnapshotGitRef:
		return jsonMarshal(struct {
			Kind      WorkspaceSnapshotKind `json:"kind"`
			Repo      string                `json:"repo"`
			Reference string                `json:"reference"`
		}{w.Kind, w.Repo, w.Reference})
	case SnapshotEmpty:
		return jsonMarshal(struct {
			Kind WorkspaceSnapshotKind `json:"kind"`
		}{w.Kind})
	default:
		return nil, fmt.Errorf("WorkspaceSnapshot: unknown kind %q", w.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (w *WorkspaceSnapshot) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind      WorkspaceSnapshotKind `json:"kind"`
		Files     map[string]string     `json:"files"`
		Repo      string                `json:"repo"`
		Reference string                `json:"reference"`
	}
	if err := jsonUnmarshal(data, &probe); err != nil {
		return err
	}
	w.Kind = probe.Kind
	w.Files = probe.Files
	w.Repo = probe.Repo
	w.Reference = probe.Reference
	if w.Kind == "" {
		return fmt.Errorf("WorkspaceSnapshot: missing kind")
	}
	return nil
}

// ============================================================================
// VerifierSpec — serializable verifier description (tagged via "kind")
// ============================================================================

// VerifierSpecKind discriminates VerifierSpec variants.
type VerifierSpecKind string

const (
	// VerifierTestSuite — run a command; score = pass rate (Rule 10).
	VerifierTestSuite VerifierSpecKind = "test_suite"
	// VerifierComposite — combine children by weight (Rule 11).
	VerifierComposite VerifierSpecKind = "composite"
	// VerifierMetricEvaluator — adapt a metric evaluator (Rule 12).
	VerifierMetricEvaluator VerifierSpecKind = "metric_evaluator"
	// VerifierLlmJudge — an LLM-judge verifier; non-deterministic (Rule 13).
	VerifierLlmJudge VerifierSpecKind = "llm_judge"
	// VerifierAlwaysPass — test scaffolding: always passes with score 1.0.
	VerifierAlwaysPass VerifierSpecKind = "always_pass"
	// VerifierAlwaysFail — test scaffolding: always fails with score 0.0.
	VerifierAlwaysFail VerifierSpecKind = "always_fail"
)

// VerifierSpec is a serializable description of a verifier, resolved to a
// TaskVerifier by the verifier package.
type VerifierSpec struct {
	Kind VerifierSpecKind `json:"kind"`
	// test_suite
	Command     string   `json:"-"`
	Args        []string `json:"-"`
	TimeoutSecs *uint64  `json:"-"`
	// composite
	Children []CompositeChildSpec `json:"-"`
	// metric_evaluator
	Descriptor string          `json:"-"`
	Direction  MetricDirection `json:"-"`
	Min        *float64        `json:"-"`
	Max        *float64        `json:"-"`
	Threshold  *float64        `json:"-"`
	// llm_judge
	Rubric     string     `json:"-"`
	ScoreRange [2]float64 `json:"-"`
}

// CompositeChildSpec is one child of a composite verifier with its weight and
// required-ness.
type CompositeChildSpec struct {
	Spec     VerifierSpec `json:"spec"`
	Weight   float64      `json:"weight"`
	Required bool         `json:"required"`
}

// MarshalJSON serialises as a flat tagged object keyed by "kind".
func (s VerifierSpec) MarshalJSON() ([]byte, error) {
	switch s.Kind {
	case VerifierTestSuite:
		args := s.Args
		if args == nil {
			args = []string{}
		}
		return jsonMarshal(struct {
			Kind        VerifierSpecKind `json:"kind"`
			Command     string           `json:"command"`
			Args        []string         `json:"args"`
			TimeoutSecs *uint64          `json:"timeout_secs,omitempty"`
		}{s.Kind, s.Command, args, s.TimeoutSecs})
	case VerifierComposite:
		children := s.Children
		if children == nil {
			children = []CompositeChildSpec{}
		}
		return jsonMarshal(struct {
			Kind     VerifierSpecKind     `json:"kind"`
			Children []CompositeChildSpec `json:"children"`
		}{s.Kind, children})
	case VerifierMetricEvaluator:
		return jsonMarshal(struct {
			Kind       VerifierSpecKind `json:"kind"`
			Descriptor string           `json:"descriptor"`
			Direction  MetricDirection  `json:"direction"`
			Min        *float64         `json:"min,omitempty"`
			Max        *float64         `json:"max,omitempty"`
			Threshold  *float64         `json:"threshold,omitempty"`
		}{s.Kind, s.Descriptor, s.Direction, s.Min, s.Max, s.Threshold})
	case VerifierLlmJudge:
		return jsonMarshal(struct {
			Kind       VerifierSpecKind `json:"kind"`
			Rubric     string           `json:"rubric"`
			ScoreRange [2]float64       `json:"score_range"`
		}{s.Kind, s.Rubric, s.ScoreRange})
	case VerifierAlwaysPass, VerifierAlwaysFail:
		return jsonMarshal(struct {
			Kind VerifierSpecKind `json:"kind"`
		}{s.Kind})
	default:
		return nil, fmt.Errorf("VerifierSpec: unknown kind %q", s.Kind)
	}
}

// UnmarshalJSON decodes the flat tagged form.
func (s *VerifierSpec) UnmarshalJSON(data []byte) error {
	var probe struct {
		Kind        VerifierSpecKind     `json:"kind"`
		Command     string               `json:"command"`
		Args        []string             `json:"args"`
		TimeoutSecs *uint64              `json:"timeout_secs"`
		Children    []CompositeChildSpec `json:"children"`
		Descriptor  string               `json:"descriptor"`
		Direction   MetricDirection      `json:"direction"`
		Min         *float64             `json:"min"`
		Max         *float64             `json:"max"`
		Threshold   *float64             `json:"threshold"`
		Rubric      string               `json:"rubric"`
		ScoreRange  [2]float64           `json:"score_range"`
	}
	if err := jsonUnmarshal(data, &probe); err != nil {
		return err
	}
	s.Kind = probe.Kind
	s.Command = probe.Command
	s.Args = probe.Args
	s.TimeoutSecs = probe.TimeoutSecs
	s.Children = probe.Children
	s.Descriptor = probe.Descriptor
	s.Direction = probe.Direction
	s.Min = probe.Min
	s.Max = probe.Max
	s.Threshold = probe.Threshold
	s.Rubric = probe.Rubric
	s.ScoreRange = probe.ScoreRange
	if s.Kind == "" {
		return fmt.Errorf("VerifierSpec: missing kind")
	}
	return nil
}

// ============================================================================
// TaskCategory + EvalTask + TaskSuite
// ============================================================================

// TaskCategory is which of the three disjoint task lists a task belongs to
// (Rule 1).
type TaskCategory string

const (
	// CategoryRegression — must stay passing across versions.
	CategoryRegression TaskCategory = "regression"
	// CategoryChallenge — measure improvement over time.
	CategoryChallenge TaskCategory = "challenge"
	// CategoryCanary — detect breakthroughs.
	CategoryCanary TaskCategory = "canary"
)

// DefaultTimeoutSecs is the per-task timeout default when the manifest omits
// the timeout field.
const DefaultTimeoutSecs uint64 = 300

// EvalTask is one evaluation task. Timeout is serialized as whole seconds
// matching the fixture JSON.
type EvalTask struct {
	ID                sporecore.TaskID  `json:"id"`
	Instruction       string            `json:"instruction"`
	WorkspaceSnapshot WorkspaceSnapshot `json:"workspace_snapshot"`
	VerifierSpec      VerifierSpec      `json:"verifier_spec"`
	ExpectedTurns     *[2]uint32        `json:"expected_turns,omitempty"`
	ExpectedCostUSD   *float64          `json:"expected_cost_usd,omitempty"`
	Tags              []string          `json:"tags,omitempty"`
	// TimeoutSecs is the per-run timeout (Rule 4) in whole seconds, keyed
	// "timeout" in the manifest. Defaults to DefaultTimeoutSecs when absent.
	TimeoutSecs uint64 `json:"timeout"`
	// ModelFixture is an optional recorded-replay fixture path.
	ModelFixture string `json:"model_fixture,omitempty"`
}

// UnmarshalJSON applies the timeout default (Rule 4) when the field is absent.
func (t *EvalTask) UnmarshalJSON(data []byte) error {
	type alias EvalTask
	probe := alias{TimeoutSecs: DefaultTimeoutSecs}
	// Detect explicit presence of "timeout" so 0 stays 0 but absent => default.
	var raw map[string]jsonRaw
	if err := jsonUnmarshal(data, &raw); err != nil {
		return err
	}
	if err := jsonUnmarshal(data, &probe); err != nil {
		return err
	}
	if _, ok := raw["timeout"]; !ok {
		probe.TimeoutSecs = DefaultTimeoutSecs
	}
	*t = EvalTask(probe)
	return nil
}

// TaskSuite is a versioned task suite holding three disjoint task lists
// (Rule 1).
type TaskSuite struct {
	// SuiteVersion is required (Rule 6); the loader rejects a manifest without
	// it.
	SuiteVersion uint32     `json:"suite_version"`
	Regression   []EvalTask `json:"regression,omitempty"`
	Challenge    []EvalTask `json:"challenge,omitempty"`
	Canary       []EvalTask `json:"canary,omitempty"`
}

// CategorizedTask pairs a task with its category for iteration/reporting.
type CategorizedTask struct {
	Category TaskCategory
	Task     *EvalTask
}

// AllTasks returns every task across the three categories, tagged with its
// category, in regression -> challenge -> canary order.
func (s *TaskSuite) AllTasks() []CategorizedTask {
	out := make([]CategorizedTask, 0, len(s.Regression)+len(s.Challenge)+len(s.Canary))
	for i := range s.Regression {
		out = append(out, CategorizedTask{CategoryRegression, &s.Regression[i]})
	}
	for i := range s.Challenge {
		out = append(out, CategorizedTask{CategoryChallenge, &s.Challenge[i]})
	}
	for i := range s.Canary {
		out = append(out, CategorizedTask{CategoryCanary, &s.Canary[i]})
	}
	return out
}

// SortedTags returns a task's tags sorted (helper for deterministic reporting).
func (t *EvalTask) SortedTags() []string {
	out := append([]string(nil), t.Tags...)
	sort.Strings(out)
	return out
}
