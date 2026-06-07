package sporecore

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

// ============================================================================
// Test doubles
// ============================================================================

// svRecordingAgent always claims done (returns a FinalResponse) and records, for
// every Turn, the session id it ran under and the concatenated text of the
// messages it was handed. Safe for concurrent use. Used to assert build-vs-
// evaluate distinguishability (R9), the injected reason in the build context
// (R6), and the role-evaluator chunk in the evaluate seed (R4).
type svRecordingAgent struct {
	id     AgentID
	output string
	mu     sync.Mutex
	seen   []svTurn
}

type svTurn struct {
	messages string
}

func newRecordingAgent(id string, output string) *svRecordingAgent {
	return &svRecordingAgent{id: AgentID(id), output: output}
}

func (a *svRecordingAgent) Turn(_ context.Context, c Context) TurnResult {
	a.mu.Lock()
	defer a.mu.Unlock()
	var b strings.Builder
	for _, m := range c.Messages {
		b.WriteString(m.Content.Text)
		b.WriteString("\n")
	}
	a.seen = append(a.seen, svTurn{messages: b.String()})
	return NewFinalResponse(a.output, TokenUsage{InputTokens: 1, OutputTokens: 1})
}

func (a *svRecordingAgent) ID() AgentID { return a.id }

func (a *svRecordingAgent) turns() []svTurn {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]svTurn, len(a.seen))
	copy(out, a.seen)
	return out
}

var _ Agent = (*svRecordingAgent)(nil)

// svVerifier yields a queued sequence of verdicts; "pass" => Passed, any
// other string => Failed with that reason. It records every input it saw so
// tests can assert distinct build/eval session ids (R9) and iteration counts.
type svVerifier struct {
	verdicts []string
	maxIter  uint32
	mu       sync.Mutex
	inputs   []SelfVerifyInput
	calls    int
}

func newSVVerifier(maxIter uint32, verdicts ...string) *svVerifier {
	return &svVerifier{verdicts: verdicts, maxIter: maxIter}
}

func (v *svVerifier) Verify(_ context.Context, input SelfVerifyInput) SelfVerifyVerdict {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.inputs = append(v.inputs, input)
	idx := v.calls
	v.calls++
	verdict := "pass"
	if idx < len(v.verdicts) {
		verdict = v.verdicts[idx]
	}
	if verdict == "pass" {
		return SelfVerifyVerdict{Kind: SelfVerifyPassed}
	}
	return SelfVerifyVerdict{Kind: SelfVerifyFailed, Reason: verdict}
}

func (v *svVerifier) MaxIterations() uint32 { return v.maxIter }

func (v *svVerifier) seenInputs() []SelfVerifyInput {
	v.mu.Lock()
	defer v.mu.Unlock()
	out := make([]SelfVerifyInput, len(v.inputs))
	copy(out, v.inputs)
	return out
}

var _ Verifier = (*svVerifier)(nil)

func selfVerifyTask() Task {
	return NewTask("build the widget", SessionID("build-sess"),
		SelfVerifyingStrategy(SelfVerifyingConfig{Inner: PtrStrategy(ReActStrategy(^uint32(0))), Evaluator: SchemaRef("evaluator")}))
}

func selfVerifyCfg(agent Agent, v Verifier) HarnessConfig {
	cfg := standardCfg(agent)
	cfg.Verifier = v
	return cfg
}

// ============================================================================
// R10: SelfVerifying no longer returns the not-yet-implemented halt
// ============================================================================

func TestSelfVerifyingNotStrategyNotImplemented(t *testing.T) {
	agent := newRecordingAgent("a", "done")
	v := newSVVerifier(3, "pass")
	h := NewStandardHarness(selfVerifyCfg(agent, v))
	r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if r.Kind == RunFailure && r.Reason.Kind == HaltStrategyNotYetImplemented {
		t.Fatalf("SelfVerifying still returns StrategyNotYetImplemented")
	}
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
}

// ============================================================================
// R11 / D4: nil verifier => SelfVerifyMisconfigured (not a panic)
// ============================================================================

func TestSelfVerifyingNilVerifierMisconfigured(t *testing.T) {
	agent := newRecordingAgent("a", "done")
	cfg := standardCfg(agent) // Verifier left nil
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if r.Kind != RunFailure {
		t.Fatalf("expected Failure, got %+v", r)
	}
	if r.Reason.Kind != HaltSelfVerifyMisconfigured {
		t.Fatalf("expected SelfVerifyMisconfigured, got %q", r.Reason.Kind)
	}
	if r.Reason.Reason == "" {
		t.Fatalf("expected a non-empty misconfigured reason")
	}
	if len(agent.turns()) != 0 {
		t.Fatalf("agent should never run when verifier is nil")
	}
}

// ============================================================================
// R1 / pass-first: a single passing verdict succeeds after one build
// ============================================================================

func TestSelfVerifyingPassFirstIteration(t *testing.T) {
	agent := newRecordingAgent("a", "build-output")
	v := newSVVerifier(3, "pass")
	h := NewStandardHarness(selfVerifyCfg(agent, v))
	r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	if r.Output != "build-output" {
		t.Fatalf("expected build output reused, got %q", r.Output)
	}
	if v.calls != 1 {
		t.Fatalf("expected exactly one verify, got %d", v.calls)
	}
}

// ============================================================================
// R2 / R9: evaluate uses a FRESH session id distinct from build
// ============================================================================

func TestSelfVerifyingEvaluateFreshSession(t *testing.T) {
	build := newRecordingAgent("build", "built")
	eval := newRecordingAgent("eval", "PASS")
	cfg := selfVerifyCfg(build, newSVVerifier(3, "pass"))
	cfg.EvaluatorAgent = eval
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	v := cfg.Verifier.(*svVerifier)
	inputs := v.seenInputs()
	if len(inputs) != 1 {
		t.Fatalf("expected 1 verifier input, got %d", len(inputs))
	}
	buildSID := inputs[0].BuildResult.SessionID
	evalSID := inputs[0].EvalResult.SessionID
	if buildSID != SessionID("build-sess") {
		t.Fatalf("expected build session id build-sess, got %q", buildSID)
	}
	if evalSID == buildSID {
		t.Fatalf("evaluate session id %q must differ from build session id (R2/R9)", evalSID)
	}
	if evalSID == "" {
		t.Fatalf("evaluate session id must be a fresh generated id")
	}
}

// ============================================================================
// R4: role-evaluator chunk present in the evaluate seed (presence-only)
// ============================================================================

func TestSelfVerifyingRoleEvaluatorChunkInSeed(t *testing.T) {
	build := newRecordingAgent("build", "built")
	eval := newRecordingAgent("eval", "PASS")
	cfg := selfVerifyCfg(build, newSVVerifier(3, "pass"))
	cfg.EvaluatorAgent = eval
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask())); r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	turns := eval.turns()
	if len(turns) == 0 {
		t.Fatalf("evaluator agent never ran")
	}
	if !strings.Contains(turns[0].messages, RoleEvaluatorChunk) {
		t.Fatalf("evaluate seed missing role-evaluator chunk; got:\n%s", turns[0].messages)
	}
	// The build agent's seed must NOT carry the evaluator role chunk.
	bturns := build.turns()
	if strings.Contains(bturns[0].messages, RoleEvaluatorChunk) {
		t.Fatalf("build seed must not carry the role-evaluator chunk")
	}
}

// ============================================================================
// R3: evaluate read-only sandbox rejects a write; build sandbox unaffected
// ============================================================================

func TestSelfVerifyingEvaluateReadOnlySandbox(t *testing.T) {
	inner := AllowAllSandbox{}
	ro := NewReadOnlySandbox(inner)
	// A mutating tool is rejected with a ReadOnlyViolation.
	if v := ro.Validate(context.Background(), ToolCall{Name: "write_file"}); v == nil || v.Kind != SandboxReadOnly {
		t.Fatalf("expected ReadOnlyViolation for write_file, got %+v", v)
	}
	// A read tool delegates and is allowed.
	if v := ro.Validate(context.Background(), ToolCall{Name: "read_file"}); v != nil {
		t.Fatalf("read_file should be allowed, got %+v", v)
	}
	// ExecuteCommand is forbidden outright.
	if _, v := ro.ExecuteCommand(context.Background(), "ls", nil, "", 0); v == nil || v.Kind != SandboxReadOnly {
		t.Fatalf("expected ReadOnlyViolation for ExecuteCommand, got %+v", v)
	}
	// Write/Execute path resolution is rejected; reads delegate.
	if _, v := ro.ResolvePath(context.Background(), "a.txt", OperationWrite); v == nil || v.Kind != SandboxReadOnly {
		t.Fatalf("expected ReadOnlyViolation for write ResolvePath, got %+v", v)
	}
	// The build (inner) sandbox is unaffected — it still allows writes.
	if v := inner.Validate(context.Background(), ToolCall{Name: "write_file"}); v != nil {
		t.Fatalf("build sandbox must still allow writes, got %+v", v)
	}
}

// ============================================================================
// R6: fail iter0 (reason X) then pass iter1 => iter1 build context has X
// ============================================================================

func TestSelfVerifyingFailThenPassInjectsReason(t *testing.T) {
	const finding = "needs a null check on line 42"
	build := newRecordingAgent("build", "built")
	eval := newRecordingAgent("eval", "PASS")
	v := newSVVerifier(3, finding, "pass")
	cfg := selfVerifyCfg(build, v)
	cfg.EvaluatorAgent = eval
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success on iter1, got %+v", r)
	}
	if v.calls != 2 {
		t.Fatalf("expected 2 verify calls, got %d", v.calls)
	}
	bturns := build.turns()
	if len(bturns) < 2 {
		t.Fatalf("expected the build agent to run at least twice, got %d", len(bturns))
	}
	// The first build turn must NOT have seen the finding; the last one MUST.
	if strings.Contains(bturns[0].messages, finding) {
		t.Fatalf("iter0 build context must not contain the finding")
	}
	last := bturns[len(bturns)-1]
	if !strings.Contains(last.messages, finding) {
		t.Fatalf("iter1 build context must contain injected reason %q; got:\n%s", finding, last.messages)
	}
}

// ============================================================================
// R5 / R7: indeterminate (Default-FAIL) and always-Fail exhaust the cap
// ============================================================================

func TestSelfVerifyingExhaustsCap(t *testing.T) {
	cases := []struct {
		name     string
		max      uint32
		verdicts []string
	}{
		{"always_fail_default_cap", 3, []string{"nope", "still wrong", "no good"}},
		{"single_iteration_cap", 1, []string{"wrong"}},
		{"indeterminate_default_fail", 2, []string{"indeterminate", "indeterminate"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			build := newRecordingAgent("build", "built")
			v := newSVVerifier(tc.max, tc.verdicts...)
			h := NewStandardHarness(selfVerifyCfg(build, v))
			r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
			if r.Kind != RunFailure {
				t.Fatalf("expected Failure, got %+v", r)
			}
			if r.Reason.Kind != HaltSelfVerifyExhausted {
				t.Fatalf("expected SelfVerifyExhausted, got %q", r.Reason.Kind)
			}
			if r.Reason.Iterations != tc.max {
				t.Fatalf("expected %d iterations, got %d", tc.max, r.Reason.Iterations)
			}
			if v.calls != int(tc.max) {
				t.Fatalf("expected exactly %d verify cycles, got %d", tc.max, v.calls)
			}
			// The last failure reason is carried on the halt.
			wantReason := tc.verdicts[len(tc.verdicts)-1]
			if r.Reason.Reason != wantReason {
				t.Fatalf("expected last_reason %q, got %q", wantReason, r.Reason.Reason)
			}
		})
	}
}

// ============================================================================
// R8: budgets fold BOTH build and evaluate phases across ALL iterations
// ============================================================================

func TestSelfVerifyingBudgetsFoldBothPhases(t *testing.T) {
	// 2 iterations: each iteration runs build (1 turn, 1+1 tokens) + evaluate
	// (1 turn, 1+1 tokens). Total input/output tokens = 4 each.
	build := newRecordingAgent("build", "built")
	eval := newRecordingAgent("eval", "PASS")
	v := newSVVerifier(3, "again", "pass")
	cfg := selfVerifyCfg(build, v)
	cfg.EvaluatorAgent = eval
	h := NewStandardHarness(cfg)
	r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask()))
	if r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	// 2 build turns + 2 evaluate turns = 4 input + 4 output tokens.
	if r.Usage.InputTokens != 4 || r.Usage.OutputTokens != 4 {
		t.Fatalf("expected folded usage 4/4, got %d/%d", r.Usage.InputTokens, r.Usage.OutputTokens)
	}
}

// ============================================================================
// R9: build vs evaluate distinguishable — each evaluate has its own session
// ============================================================================

func TestSelfVerifyingEvaluateSessionsDistinctPerIteration(t *testing.T) {
	build := newRecordingAgent("build", "built")
	eval := newRecordingAgent("eval", "PASS")
	v := newSVVerifier(3, "more", "more", "pass")
	cfg := selfVerifyCfg(build, v)
	cfg.EvaluatorAgent = eval
	h := NewStandardHarness(cfg)
	if r := h.Run(context.Background(), NewHarnessRunOptions(selfVerifyTask())); r.Kind != RunSuccess {
		t.Fatalf("expected Success, got %+v", r)
	}
	inputs := v.seenInputs()
	if len(inputs) != 3 {
		t.Fatalf("expected 3 iterations, got %d", len(inputs))
	}
	seen := map[SessionID]bool{}
	for i, in := range inputs {
		evalSID := in.EvalResult.SessionID
		if evalSID == in.BuildResult.SessionID {
			t.Fatalf("iter %d: evaluate session id must differ from build", i)
		}
		if seen[evalSID] {
			t.Fatalf("iter %d: evaluate session id %q reused across iterations", i, evalSID)
		}
		seen[evalSID] = true
		if in.Iteration != uint32(i) {
			t.Fatalf("iter %d: expected Iteration=%d, got %d", i, i, in.Iteration)
		}
	}
}

// ============================================================================
// HaltReason JSON round-trips for the two new variants (D4)
// ============================================================================

func TestSelfVerifyHaltReasonsRoundTrip(t *testing.T) {
	cases := []HaltReason{
		{Kind: HaltSelfVerifyExhausted, Iterations: 3, Reason: "still wrong"},
		{Kind: HaltSelfVerifyMisconfigured, Reason: "verifier is nil"},
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
		if got.Kind != want.Kind || got.Iterations != want.Iterations || got.Reason != want.Reason {
			t.Fatalf("round-trip mismatch for %q: got %+v want %+v (raw=%s)", want.Kind, got, want, raw)
		}
	}
}
