package verifier

import (
	"context"
	"strings"
	"testing"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

func successResult(output string) sporecore.RunResult {
	return sporecore.RunResult{
		Kind:      sporecore.RunSuccess,
		Output:    output,
		SessionID: sporecore.SessionID("s"),
		Turns:     1,
	}
}

func failureResult() sporecore.RunResult {
	return sporecore.RunResult{
		Kind: sporecore.RunFailure,
		Reason: sporecore.HaltReason{
			Kind:     sporecore.HaltStrategyNotYetImplemented,
			Strategy: "x",
		},
		SessionID: sporecore.SessionID("s"),
	}
}

func waitingForHumanResult() sporecore.RunResult {
	return sporecore.RunResult{Kind: sporecore.RunWaitingForHuman}
}

func inputWith(build, eval sporecore.RunResult) *VerifierInput {
	return &VerifierInput{
		BuildResult: build,
		EvalResult:  eval,
		Workspace:   "/tmp",
		Iteration:   0,
	}
}

func makeRespVerifier(t *testing.T) *EvaluatorResponseVerifier {
	t.Helper()
	v, err := NewEvaluatorResponseVerifier(`(?i)\bPASS\b`, `(?i)\bFAIL: .+`, 3)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	return v
}

// ── EvaluatorResponseVerifier ────────────────────────────────────────────────

func TestResponseVerifierPassPatternMatches(t *testing.T) {
	v := makeRespVerifier(t)
	in := inputWith(successResult("ok"), successResult("all checks PASS, ready to ship"))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictPassed {
		t.Fatalf("expected Passed, got %+v", got)
	}
}

func TestResponseVerifierFailPatternMatchesWithReason(t *testing.T) {
	v := makeRespVerifier(t)
	in := inputWith(successResult("ok"), successResult("FAIL: missing edge case in handler.rs"))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.Contains(got.Reason, "missing edge case") {
		t.Fatalf("expected reason to contain 'missing edge case', got %q", got.Reason)
	}
}

func TestResponseVerifierNeitherPatternDefaultFails(t *testing.T) {
	v := makeRespVerifier(t)
	in := inputWith(successResult("ok"), successResult("indeterminate output"))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.Contains(got.Reason, "matched neither") {
		t.Fatalf("expected 'matched neither' in reason, got %q", got.Reason)
	}
	if !strings.Contains(got.Reason, "indeterminate output") {
		t.Fatalf("expected eval output in reason, got %q", got.Reason)
	}
}

func TestResponseVerifierBuildFailurePropagates(t *testing.T) {
	v := makeRespVerifier(t)
	in := inputWith(failureResult(), successResult("PASS"))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.HasPrefix(got.Reason, "build run halted") {
		t.Fatalf("expected 'build run halted' prefix, got %q", got.Reason)
	}
}

func TestResponseVerifierEvalFailurePropagates(t *testing.T) {
	v := makeRespVerifier(t)
	in := inputWith(successResult("ok"), failureResult())
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.HasPrefix(got.Reason, "evaluator run halted") {
		t.Fatalf("expected 'evaluator run halted' prefix, got %q", got.Reason)
	}
}

func TestResponseVerifierWaitingForHumanIsMisconfiguration(t *testing.T) {
	v := makeRespVerifier(t)
	in := inputWith(successResult("ok"), waitingForHumanResult())
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.Contains(got.Reason, "WaitingForHuman") {
		t.Fatalf("expected WaitingForHuman in reason, got %q", got.Reason)
	}
	if !strings.Contains(got.Reason, "misconfiguration") {
		t.Fatalf("expected misconfiguration mention, got %q", got.Reason)
	}
}

func TestResponseVerifierMaxIterations(t *testing.T) {
	v := makeRespVerifier(t)
	if v.MaxIterations() != 3 {
		t.Fatalf("expected 3, got %d", v.MaxIterations())
	}
	v2, err := NewEvaluatorResponseVerifier("a", "b", 10)
	if err != nil {
		t.Fatal(err)
	}
	if v2.MaxIterations() != 10 {
		t.Fatalf("expected 10, got %d", v2.MaxIterations())
	}
}

func TestResponseVerifierInvalidRegexErrors(t *testing.T) {
	if _, err := NewEvaluatorResponseVerifier("(unclosed", "b", 3); err == nil {
		t.Fatalf("expected error for invalid pass_pattern")
	}
	if _, err := NewEvaluatorResponseVerifier("a", "(unclosed", 3); err == nil {
		t.Fatalf("expected error for invalid fail_pattern")
	}
}

// ── TestSuiteVerifier ────────────────────────────────────────────────────────

type stubSandbox struct {
	out  sporecore.CommandOutput
	root string
}

func (s *stubSandbox) Validate(context.Context, sporecore.ToolCall) *sporecore.SandboxViolation {
	return nil
}
func (s *stubSandbox) ExecuteCommand(
	ctx context.Context,
	command string,
	args []string,
	workingDir string,
	timeout time.Duration,
) (sporecore.CommandOutput, *sporecore.SandboxViolation) {
	return s.out, nil
}
func (s *stubSandbox) HandleLargeOutput(
	ctx context.Context, content, callID string, head, tail uint32,
) sporecore.TruncatedOutput {
	return sporecore.TruncatedOutput{Content: content}
}
func (s *stubSandbox) ResolvePath(
	ctx context.Context, p string, op sporecore.Operation,
) (string, *sporecore.SandboxViolation) {
	return p, nil
}
func (s *stubSandbox) IsolationMode() sporecore.IsolationMode {
	return sporecore.IsolationNone{}
}
func (s *stubSandbox) WorkspaceRoot() string { return s.root }

func newStubSandbox(exit int, stderr string) sporecore.SandboxProvider {
	return &stubSandbox{
		out: sporecore.CommandOutput{
			Stdout:   "",
			Stderr:   stderr,
			ExitCode: exit,
			TimedOut: false,
		},
		root: "/",
	}
}

func TestSuiteVerifierPass(t *testing.T) {
	v := NewTestSuiteVerifier("cargo test", "/work", 60*time.Second, newStubSandbox(0, ""), 3)
	in := inputWith(successResult("ok"), successResult(""))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictPassed {
		t.Fatalf("expected Passed, got %+v", got)
	}
}

func TestSuiteVerifierFailIncludesStderr(t *testing.T) {
	v := NewTestSuiteVerifier("cargo test", "/work", 60*time.Second, newStubSandbox(1, "test foo ... FAILED"), 3)
	in := inputWith(successResult("ok"), successResult(""))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.Contains(got.Reason, "FAILED") {
		t.Fatalf("expected FAILED in reason, got %q", got.Reason)
	}
}

func TestSuiteVerifierBuildFailureShortCircuits(t *testing.T) {
	v := NewTestSuiteVerifier("cargo test", "/work", 60*time.Second, newStubSandbox(0, ""), 3)
	in := inputWith(failureResult(), successResult(""))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.HasPrefix(got.Reason, "build run halted") {
		t.Fatalf("expected 'build run halted' prefix, got %q", got.Reason)
	}
}

func TestSuiteVerifierEmptyCommandFails(t *testing.T) {
	v := NewTestSuiteVerifier("", "/work", 60*time.Second, newStubSandbox(0, ""), 3)
	in := inputWith(successResult("ok"), successResult(""))
	got := v.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.Contains(got.Reason, "empty test command") {
		t.Fatalf("expected 'empty test command', got %q", got.Reason)
	}
}

func TestSuiteVerifierMaxIterations(t *testing.T) {
	v := NewTestSuiteVerifier("x", "/work", time.Second, newStubSandbox(0, ""), 7)
	if v.MaxIterations() != 7 {
		t.Fatalf("expected 7, got %d", v.MaxIterations())
	}
}

// ── CompositeVerifier ────────────────────────────────────────────────────────

type fixedVerifier struct {
	v VerifierVerdict
}

func (f *fixedVerifier) Verify(context.Context, *VerifierInput) VerifierVerdict { return f.v }
func (f *fixedVerifier) MaxIterations() uint32                                  { return 3 }

func passV() Verifier              { return &fixedVerifier{v: Passed()} }
func failV(reason string) Verifier { return &fixedVerifier{v: Failed(reason)} }

func TestCompositeAllPassReturnsPassed(t *testing.T) {
	c := NewCompositeVerifier([]Verifier{passV(), passV(), passV()}, 3)
	in := inputWith(successResult("ok"), successResult("ok"))
	got := c.Verify(context.Background(), in)
	if got.Kind != VerdictPassed {
		t.Fatalf("expected Passed, got %+v", got)
	}
}

func TestCompositeOneFailReturnsThatReason(t *testing.T) {
	c := NewCompositeVerifier([]Verifier{passV(), failV("oops"), passV()}, 3)
	in := inputWith(successResult("ok"), successResult("ok"))
	got := c.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if !strings.Contains(got.Reason, "oops") {
		t.Fatalf("expected 'oops' in reason, got %q", got.Reason)
	}
	if !strings.Contains(got.Reason, "[verifier 1]") {
		t.Fatalf("expected '[verifier 1]' in reason, got %q", got.Reason)
	}
}

func TestCompositeManyFailsConcatenated(t *testing.T) {
	c := NewCompositeVerifier(
		[]Verifier{failV("first"), passV(), failV("second"), failV("third")}, 3,
	)
	in := inputWith(successResult("ok"), successResult("ok"))
	got := c.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	for _, sub := range []string{"first", "second", "third"} {
		if !strings.Contains(got.Reason, sub) {
			t.Fatalf("expected %q in reason, got %q", sub, got.Reason)
		}
	}
	if strings.Contains(got.Reason, "[verifier 1]") {
		t.Fatalf("pass-verifier should not appear in reason: %q", got.Reason)
	}
}

func TestCompositeTruncatesAt2000Chars(t *testing.T) {
	long := strings.Repeat("x", 5000)
	c := NewCompositeVerifier([]Verifier{failV(long)}, 3)
	in := inputWith(successResult("ok"), successResult("ok"))
	got := c.Verify(context.Background(), in)
	if got.Kind != VerdictFailed {
		t.Fatalf("expected Failed, got %+v", got)
	}
	if len(got.Reason) > compositeReasonCap+len("... [truncated]") {
		t.Fatalf("reason exceeded cap: len=%d", len(got.Reason))
	}
	if !strings.HasSuffix(got.Reason, "... [truncated]") {
		t.Fatalf("expected truncation suffix, got %q", got.Reason[len(got.Reason)-30:])
	}
}

func TestCompositeMaxIterations(t *testing.T) {
	c := NewCompositeVerifier(nil, 9)
	if c.MaxIterations() != 9 {
		t.Fatalf("expected 9, got %d", c.MaxIterations())
	}
}

// Compile-time guarantee that the Verifier interface is satisfied by an
// interface-typed value.
func TestVerifierInterfaceUsable(t *testing.T) {
	v, err := NewEvaluatorResponseVerifier("PASS", "FAIL", 3)
	if err != nil {
		t.Fatal(err)
	}
	var iface Verifier = v
	_ = iface.MaxIterations()
}
