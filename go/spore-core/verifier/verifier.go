// verifier.go — issue #44 Verifier interface and standard implementations.
//
// Mirrors the Rust reference at rust/crates/spore-core/src/verifier.rs:
//   - EvaluatorResponseVerifier  (pattern-matches the evaluator's final text)
//   - TestSuiteVerifier          (runs a command via an injected SandboxProvider)
//   - CompositeVerifier          (passes only when all children pass)
//
// Ambiguity resolutions (see issue #44):
//  1. EvaluatorResponseVerifier when neither pass_pattern nor fail_pattern
//     matches → Failed with a descriptive reason including a truncated copy
//     of the output. Default-FAIL is NOT configurable.
//  2. Any non-Success RunResult → Failed. WaitingForHuman is treated as a
//     misconfiguration signal and surfaced in the reason.
//  3. CompositeVerifier concatenates all child failure reasons (joined by
//     "\n"), capped at 2000 characters total. Children that pass are not
//     mentioned.
//  4. LoopStrategy.SelfVerifying wiring is deferred to #45 — the strategy
//     continues to return StrategyNotYetImplemented in the harness loop.
package verifier

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// VerifierVerdict
// ============================================================================

// VerdictKind discriminates VerifierVerdict variants.
type VerdictKind string

const (
	VerdictPassed VerdictKind = "passed"
	VerdictFailed VerdictKind = "failed"
)

// VerifierVerdict is the output of Verifier.Verify.
type VerifierVerdict struct {
	Kind   VerdictKind `json:"kind"`
	Reason string      `json:"reason,omitempty"`
}

// Passed returns the Passed verdict.
func Passed() VerifierVerdict { return VerifierVerdict{Kind: VerdictPassed} }

// Failed returns a Failed verdict with the given reason.
func Failed(reason string) VerifierVerdict {
	return VerifierVerdict{Kind: VerdictFailed, Reason: reason}
}

// ============================================================================
// VerifierInput
// ============================================================================

// VerifierInput is the input to Verifier.Verify.
type VerifierInput struct {
	BuildResult sporecore.RunResult `json:"build_result"`
	EvalResult  sporecore.RunResult `json:"eval_result"`
	Workspace   string              `json:"workspace"`
	// Which build-evaluate cycle this is (0-indexed).
	Iteration uint32 `json:"iteration"`
}

// ============================================================================
// Verifier interface
// ============================================================================

// Verifier is the oracle for the SelfVerifying loop strategy. It translates
// (build_result, eval_result) into a VerifierVerdict — either Passed (halt
// with success) or Failed (re-enter the build loop with reason injected into
// the next turn's context).
type Verifier interface {
	Verify(ctx context.Context, input *VerifierInput) VerifierVerdict
	// MaxIterations is the maximum number of build-evaluate cycles before the
	// harness halts the loop regardless of verdict. Spec default: 3.
	MaxIterations() uint32
}

// DefaultMaxIterations is the spec default cap on build-evaluate cycles.
const DefaultMaxIterations uint32 = 3

// ============================================================================
// Helpers
// ============================================================================

// resultView reduces a RunResult to either its Success output or a
// descriptive failure reason.
type resultView struct {
	output string
	failed string
	ok     bool
}

func viewResult(label string, r sporecore.RunResult) resultView {
	switch r.Kind {
	case sporecore.RunSuccess:
		return resultView{output: r.Output, ok: true}
	case sporecore.RunFailure:
		return resultView{failed: fmt.Sprintf("%s run halted: %s", label, describeHalt(r.Reason))}
	case sporecore.RunWaitingForHuman:
		return resultView{failed: fmt.Sprintf(
			"%s run is WaitingForHuman — verifier received a paused harness; "+
				"this is a misconfiguration signal (the %s should run to completion "+
				"before being verified)",
			label, label,
		)}
	default:
		return resultView{failed: fmt.Sprintf("%s run returned unknown kind: %q", label, r.Kind)}
	}
}

// describeHalt produces an opaque diagnostic string for a HaltReason. The
// exact shape is not load-bearing for verifier output.
func describeHalt(reason sporecore.HaltReason) string {
	switch reason.Kind {
	case sporecore.HaltBudgetExceeded:
		return fmt.Sprintf("BudgetExceeded{limit_type=%s}", reason.LimitType)
	case sporecore.HaltTerminationPolicyHalt:
		return fmt.Sprintf("TerminationPolicyHalt{reason=%q}", reason.Reason)
	case sporecore.HaltMiddlewareHalt:
		return fmt.Sprintf("MiddlewareHalt{hook=%s, reason=%q}", reason.Hook, reason.Reason)
	case sporecore.HaltAgentError:
		return fmt.Sprintf("AgentError{%+v}", reason.AgentError)
	case sporecore.HaltSandboxViolation:
		return fmt.Sprintf("SandboxViolation{%+v}", reason.Violation)
	case sporecore.HaltUnrecoverableToolError:
		return fmt.Sprintf("UnrecoverableToolError{tool=%s, error=%q}", reason.Tool, reason.Error)
	case sporecore.HaltHumanHalted:
		return "HumanHalted"
	case sporecore.HaltStagnationLimitReached:
		return fmt.Sprintf("StagnationLimitReached{iterations=%d, best_metric=%v}", reason.Iterations, reason.BestMetric)
	case sporecore.HaltStrategyNotYetImplemented:
		return fmt.Sprintf("StrategyNotYetImplemented{strategy=%s}", reason.Strategy)
	case sporecore.HaltSelfVerifyExhausted:
		return fmt.Sprintf("SelfVerifyExhausted{iterations=%d, last_reason=%q}", reason.Iterations, reason.Reason)
	case sporecore.HaltSelfVerifyMisconfigured:
		return fmt.Sprintf("SelfVerifyMisconfigured{reason=%q}", reason.Reason)
	default:
		return fmt.Sprintf("%+v", reason)
	}
}

// truncateForReason truncates s to at most max bytes, appending "... [truncated]".
func truncateForReason(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "... [truncated]"
}

func tailLines(s string, n int) string {
	if s == "" {
		return ""
	}
	lines := strings.Split(s, "\n")
	start := len(lines) - n
	if start < 0 {
		start = 0
	}
	return strings.Join(lines[start:], "\n")
}

// ============================================================================
// EvaluatorResponseVerifier
// ============================================================================

// EvaluatorResponseVerifier pattern-matches the evaluator harness's final
// text response. The simplest verifier — trusts whatever the evaluator wrote.
//
// Rules:
//   - If BuildResult is not Success → Failed with the halt reason.
//   - If EvalResult is not Success → Failed with the halt reason.
//   - If PassPattern matches the eval output → Passed.
//   - If FailPattern matches the eval output → Failed with the matched
//     line(s) as the reason.
//   - Neither matches → Failed with a descriptive default reason
//     (Default-FAIL contract; not configurable).
type EvaluatorResponseVerifier struct {
	PassPattern   *regexp.Regexp
	FailPattern   *regexp.Regexp
	maxIterations uint32
}

// NewEvaluatorResponseVerifier compiles the patterns and returns a
// configured EvaluatorResponseVerifier. Returns an error if either regex
// fails to compile.
func NewEvaluatorResponseVerifier(passPattern, failPattern string, maxIterations uint32) (*EvaluatorResponseVerifier, error) {
	pp, err := regexp.Compile(passPattern)
	if err != nil {
		return nil, fmt.Errorf("compile pass_pattern: %w", err)
	}
	fp, err := regexp.Compile(failPattern)
	if err != nil {
		return nil, fmt.Errorf("compile fail_pattern: %w", err)
	}
	return &EvaluatorResponseVerifier{
		PassPattern:   pp,
		FailPattern:   fp,
		maxIterations: maxIterations,
	}, nil
}

// Verify implements Verifier.
func (v *EvaluatorResponseVerifier) Verify(_ context.Context, input *VerifierInput) VerifierVerdict {
	if bv := viewResult("build", input.BuildResult); !bv.ok {
		return Failed(bv.failed)
	}
	ev := viewResult("evaluator", input.EvalResult)
	if !ev.ok {
		return Failed(ev.failed)
	}
	output := ev.output
	if m := v.PassPattern.FindString(output); m != "" {
		return Passed()
	}
	if m := v.FailPattern.FindString(output); m != "" {
		return Failed(fmt.Sprintf(
			"evaluator reported failure: %s",
			truncateForReason(m, 500),
		))
	}
	return Failed(fmt.Sprintf(
		"evaluator output matched neither pass_pattern (`%s`) nor "+
			"fail_pattern (`%s`). Output was:\n%s",
		v.PassPattern.String(),
		v.FailPattern.String(),
		truncateForReason(output, 1000),
	))
}

// MaxIterations implements Verifier.
func (v *EvaluatorResponseVerifier) MaxIterations() uint32 { return v.maxIterations }

// ============================================================================
// TestSuiteVerifier
// ============================================================================

// TestSuiteVerifier runs a test command via the injected SandboxProvider and
// uses the exit code as the verdict. Ignores the evaluator's text output —
// ground truth is the tests.
//
// Rules:
//   - If BuildResult is not Success → Failed with the halt reason.
//   - Run Command in WorkingDir via Sandbox.ExecuteCommand.
//   - Exit 0, not timed out → Passed.
//   - Anything else → Failed with a stderr/stdout tail.
type TestSuiteVerifier struct {
	Command       string
	WorkingDir    string
	Timeout       time.Duration
	Sandbox       sporecore.SandboxProvider
	maxIterations uint32
}

// NewTestSuiteVerifier constructs a TestSuiteVerifier.
func NewTestSuiteVerifier(
	command, workingDir string,
	timeout time.Duration,
	sandbox sporecore.SandboxProvider,
	maxIterations uint32,
) *TestSuiteVerifier {
	return &TestSuiteVerifier{
		Command:       command,
		WorkingDir:    workingDir,
		Timeout:       timeout,
		Sandbox:       sandbox,
		maxIterations: maxIterations,
	}
}

// Verify implements Verifier.
func (t *TestSuiteVerifier) Verify(ctx context.Context, input *VerifierInput) VerifierVerdict {
	if bv := viewResult("build", input.BuildResult); !bv.ok {
		return Failed(bv.failed)
	}
	parts := strings.Fields(t.Command)
	if len(parts) == 0 {
		return Failed("empty test command")
	}
	program := parts[0]
	args := parts[1:]
	out, viol := t.Sandbox.ExecuteCommand(ctx, program, args, t.WorkingDir, t.Timeout)
	if viol != nil {
		return Failed(fmt.Sprintf("sandbox refused test command: %v", viol))
	}
	if out.ExitCode == 0 && !out.TimedOut {
		return Passed()
	}
	tail := tailLines(out.Stderr, 20)
	if strings.TrimSpace(tail) == "" {
		tail = tailLines(out.Stdout, 20)
	}
	return Failed(fmt.Sprintf(
		"test suite failed (exit %d, timed_out=%t):\n%s",
		out.ExitCode, out.TimedOut, tail,
	))
}

// MaxIterations implements Verifier.
func (t *TestSuiteVerifier) MaxIterations() uint32 { return t.maxIterations }

// ============================================================================
// CompositeVerifier
// ============================================================================

// compositeReasonCap caps the joined failure reason of CompositeVerifier.
const compositeReasonCap = 2000

// CompositeVerifier passes only when all child verifiers pass. On failure,
// concatenates every child's failure reason (joined by "\n"), capped at 2000
// characters total. Children that pass are not mentioned in the failure reason.
type CompositeVerifier struct {
	Verifiers     []Verifier
	maxIterations uint32
}

// NewCompositeVerifier constructs a CompositeVerifier.
func NewCompositeVerifier(verifiers []Verifier, maxIterations uint32) *CompositeVerifier {
	return &CompositeVerifier{Verifiers: verifiers, maxIterations: maxIterations}
}

// Verify implements Verifier.
func (c *CompositeVerifier) Verify(ctx context.Context, input *VerifierInput) VerifierVerdict {
	var failures []string
	for i, v := range c.Verifiers {
		verdict := v.Verify(ctx, input)
		if verdict.Kind == VerdictFailed {
			failures = append(failures, fmt.Sprintf("[verifier %d] %s", i, verdict.Reason))
		}
	}
	if len(failures) == 0 {
		return Passed()
	}
	joined := strings.Join(failures, "\n")
	return Failed(truncateForReason(joined, compositeReasonCap))
}

// MaxIterations implements Verifier.
func (c *CompositeVerifier) MaxIterations() uint32 { return c.maxIterations }

// ============================================================================
// Harness seam adapter (issue #61)
// ============================================================================

// harnessVerifier adapts a verifier.Verifier into the consumer-side
// sporecore.Verifier seam the SelfVerifying strategy reads. The sporecore root
// package cannot import this package (cycle), so it defines a narrow Verifier
// interface in root-package terms (sporecore.SelfVerifyInput /
// sporecore.SelfVerifyVerdict); this adapter translates between the two.
type harnessVerifier struct {
	inner Verifier
}

// AsHarnessVerifier wraps a verifier.Verifier so it can be dropped straight into
// sporecore.HarnessConfig.Verifier. Returns nil when v is nil so a nil verifier
// stays nil through the seam (the SelfVerifying strategy treats a nil
// config.Verifier as HaltSelfVerifyMisconfigured, D4).
func AsHarnessVerifier(v Verifier) sporecore.Verifier {
	if v == nil {
		return nil
	}
	return harnessVerifier{inner: v}
}

// Verify implements sporecore.Verifier.
func (h harnessVerifier) Verify(ctx context.Context, input sporecore.SelfVerifyInput) sporecore.SelfVerifyVerdict {
	verdict := h.inner.Verify(ctx, &VerifierInput{
		BuildResult: input.BuildResult,
		EvalResult:  input.EvalResult,
		Workspace:   input.Workspace,
		Iteration:   input.Iteration,
	})
	if verdict.Kind == VerdictPassed {
		return sporecore.SelfVerifyVerdict{Kind: sporecore.SelfVerifyPassed}
	}
	return sporecore.SelfVerifyVerdict{Kind: sporecore.SelfVerifyFailed, Reason: verdict.Reason}
}

// MaxIterations implements sporecore.Verifier.
func (h harnessVerifier) MaxIterations() uint32 { return h.inner.MaxIterations() }

// ============================================================================
// Compile-time interface checks.
// ============================================================================

var (
	_ Verifier           = (*EvaluatorResponseVerifier)(nil)
	_ Verifier           = (*TestSuiteVerifier)(nil)
	_ Verifier           = (*CompositeVerifier)(nil)
	_ sporecore.Verifier = harnessVerifier{}
)
