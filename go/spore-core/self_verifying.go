// self_verifying.go — the SelfVerifying loop strategy (issue #61).
//
// Loop-within-a-loop. Each round-trip runs a bounded BUILD ReAct sub-loop (the
// agent does work until it claims done), then a fresh EVALUATE run (a separate
// evaluator agent on a read-only sandbox in a never-shared session), then asks
// the injected Verifier to translate (build_result, eval_result) into a
// verdict. Passed => Success; Failed => inject the reason into the build context
// and loop. Running out of the verifier's MaxIterations round-trips without a
// pass is a clean Default-FAIL exhaustion (SelfVerifyExhausted). A nil verifier
// is a build-time misconfiguration (SelfVerifyMisconfigured) — both are typed
// halts, never panics.
//
// Mirrors the Rust reference (rust/crates/spore-core/src/harness.rs
// run_self_verifying / run_evaluate_phase, and the ReadOnlySandbox decorator in
// sandbox.rs) but is written idiomatically for Go. Go-specific divergences from
// the Rust reference, all following established patterns in this repo:
//
//   - Verifier is a CONSUMER-SIDE interface defined here in root-package terms
//     (mirroring CompactionVerifier / RunStore / HarnessObserver). The standard
//     verifier.Verifier family (#44) lives in the `verifier` package, which
//     imports sporecore; sporecore cannot import it back (cycle). The
//     verifier.AsHarnessVerifier adapter bridges a verifier.Verifier into this
//     seam. SelfVerifyInput / SelfVerifyVerdict are the root-package mirrors of
//     verifier.VerifierInput / VerifierVerdict.
//
//   - The role-evaluator chunk (R4, presence-only) is prepended to the evaluate
//     seed directly from the RoleEvaluatorChunk constant. Go's sporecore cannot
//     import promptchunkregistry (cycle), and HarnessConfig carries no
//     chunk-provider field; the constant is the single source of truth and is
//     kept byte-identical to promptchunkregistry's "role-evaluator" content.

package sporecore

import (
	"context"
	"fmt"
	"time"
)

// ============================================================================
// Verifier seam (consumer-side; issue #61 / #44 bridge)
// ============================================================================

// SelfVerifyVerdictKind discriminates SelfVerifyVerdict variants.
type SelfVerifyVerdictKind string

const (
	// SelfVerifyPassed halts the SelfVerifying loop with Success.
	SelfVerifyPassed SelfVerifyVerdictKind = "passed"
	// SelfVerifyFailed re-enters the build loop with Reason injected into the
	// next turn's context.
	SelfVerifyFailed SelfVerifyVerdictKind = "failed"
)

// SelfVerifyVerdict is the output of Verifier.Verify, in root-package terms.
// Mirrors verifier.VerifierVerdict.
type SelfVerifyVerdict struct {
	Kind   SelfVerifyVerdictKind
	Reason string
}

// SelfVerifyInput is the input to Verifier.Verify, in root-package terms.
// Mirrors verifier.VerifierInput.
type SelfVerifyInput struct {
	BuildResult RunResult
	EvalResult  RunResult
	Workspace   string
	// Iteration is which build-evaluate cycle this is (0-indexed).
	Iteration uint32
}

// Verifier is the consumer-side oracle seam for the SelfVerifying loop strategy
// (issue #61). It translates (build_result, eval_result) into a verdict —
// either Passed (halt with success) or Failed (re-enter the build loop with the
// reason injected into the next turn's context).
//
// Defined in root-package terms so the harness loop needs no `verifier` import
// (which would be a cycle). The standard verifier.Verifier family is adapted
// into this seam by verifier.AsHarnessVerifier.
type Verifier interface {
	Verify(ctx context.Context, input SelfVerifyInput) SelfVerifyVerdict
	// MaxIterations is the maximum number of build-evaluate round-trips before
	// the harness halts the loop regardless of verdict (D3). Spec default: 3.
	MaxIterations() uint32
}

// RoleEvaluatorChunk is the presence-only content prepended to the evaluate
// seed (R4). Kept byte-identical to the promptchunkregistry "role-evaluator"
// chunk; sporecore cannot import that package (cycle), so the content is
// duplicated here as the single source of truth for the harness loop.
const RoleEvaluatorChunk = "You are a fresh evaluator. You did not write the code under review. Default to FAIL."

// ============================================================================
// ReadOnlySandbox decorator (R3)
// ============================================================================

// readOnlyWriteTools is the standard-catalogue set of mutating tool names a
// read-only sandbox blocks. Mirrors Rust's ReadOnlySandbox::DEFAULT_WRITE_TOOLS.
var readOnlyWriteTools = map[string]struct{}{
	"write_file":   {},
	"edit_file":    {},
	"delete_file":  {},
	"move_file":    {},
	"exec":         {},
	"bash_command": {},
	"run_tests":    {},
}

// ReadOnlySandbox wraps a SandboxProvider and rejects mutating tool calls,
// subprocess execution, and write/execute path resolutions with a recoverable
// ReadOnlyViolation; all other operations delegate to the inner provider.
//
// ReadOnlyViolation is a Layer-2 (recoverable) violation, so in the harness loop
// a blocked write surfaces to the evaluator agent as a recoverable tool error —
// it does NOT halt the evaluate run.
type ReadOnlySandbox struct {
	inner      SandboxProvider
	writeTools map[string]struct{}
}

// NewReadOnlySandbox wraps inner, blocking the standard mutating tools.
func NewReadOnlySandbox(inner SandboxProvider) *ReadOnlySandbox {
	return &ReadOnlySandbox{inner: inner, writeTools: readOnlyWriteTools}
}

func (s *ReadOnlySandbox) isWrite(toolName string) bool {
	_, ok := s.writeTools[toolName]
	return ok
}

// Validate rejects mutating tool calls; everything else delegates.
func (s *ReadOnlySandbox) Validate(ctx context.Context, call ToolCall) *SandboxViolation {
	if s.isWrite(call.Name) {
		return &SandboxViolation{Kind: SandboxReadOnly, Path: call.Name}
	}
	return s.inner.Validate(ctx, call)
}

// ExecuteCommand is forbidden outright — subprocesses may have arbitrary write
// side effects.
func (s *ReadOnlySandbox) ExecuteCommand(_ context.Context, command string, _ []string, _ string, _ time.Duration) (CommandOutput, *SandboxViolation) {
	return CommandOutput{}, &SandboxViolation{Kind: SandboxReadOnly, Path: command}
}

// HandleLargeOutput delegates (a read-side concern).
func (s *ReadOnlySandbox) HandleLargeOutput(ctx context.Context, content string, callID string, headTokens uint32, tailTokens uint32) TruncatedOutput {
	return s.inner.HandleLargeOutput(ctx, content, callID, headTokens, tailTokens)
}

// ResolvePath rejects write/execute resolutions; read resolutions delegate.
func (s *ReadOnlySandbox) ResolvePath(ctx context.Context, path string, op Operation) (string, *SandboxViolation) {
	if op == OperationWrite || op == OperationExecute {
		return "", &SandboxViolation{Kind: SandboxReadOnly, Path: path}
	}
	return s.inner.ResolvePath(ctx, path, op)
}

// IsolationMode delegates to the inner provider.
func (s *ReadOnlySandbox) IsolationMode() IsolationMode { return s.inner.IsolationMode() }

// WorkspaceRoot delegates to the inner provider.
func (s *ReadOnlySandbox) WorkspaceRoot() string { return s.inner.WorkspaceRoot() }

var _ SandboxProvider = (*ReadOnlySandbox)(nil)

// ============================================================================
// SelfVerifying strategy driver
// ============================================================================

// allTurns is the "no turn cap" sentinel (max uint32), mirroring the execute
// phase's ^uint32(0).
const allTurns = ^uint32(0)

// EvaluatePhase runs the SelfVerifying evaluate phase (issue #61, #124): a fresh
// evaluator RUN over a read-only sandbox in a never-shared session, on the
// RESOLVED evalAgent (Q1c — the inner worker's resolved agent, threaded by the
// recursive SelfVerifyingConfig.Run; the harness no longer reads config.Agent /
// EvaluatorAgent here).
//
// Builds a child StandardHarness from a copy of h.config with the Agent swapped
// to evalAgent and the Sandbox wrapped in a ReadOnlySandbox (R3). The evaluator
// runs a fresh ReAct loop seeded with the role-evaluator chunk (R4,
// presence-only) plus a review directive, in a freshly generated session
// (R2/R9). Folds the evaluate run's usage into totalUsage / carried (R8) and
// returns its terminal RunResult.
func (h *StandardHarness) EvaluatePhase(
	ctx context.Context,
	task *Task,
	evalAgent Agent,
	carried *BudgetSnapshot,
	totalUsage *AggregateUsage,
) RunResult {
	// R4 (presence-only): prepend the role-evaluator chunk content to the review
	// directive.
	directive := fmt.Sprintf(
		"%s\n\nReview the work produced for the following task and report whether "+
			"it is correct. You did NOT write this code; default to FAIL unless you "+
			"can confirm it is right.\n\nTask:\n%s",
		RoleEvaluatorChunk, task.Instruction,
	)

	// R2/R9: fresh, never-shared session id for the evaluate run.
	evalSessionID := NewSessionID()

	cap := allTurns
	if task.Budget.MaxTurns != nil {
		cap = *task.Budget.MaxTurns
	}
	evalTask := Task{
		ID:           NewTaskID(),
		Instruction:  directive,
		SessionID:    evalSessionID,
		Budget:       task.Budget,
		LoopStrategy: ReActStrategy(cap),
	}

	// Child harness: copy the config, swap sandbox. The copy shares the same
	// observability / context-manager / tool seams so the evaluate run's spans
	// land in the SAME trace stream (distinguished by its distinct session id,
	// R9). The resolved evalAgent runs the fresh ReAct window directly.
	evalConfig := h.config
	evalConfig.Sandbox = NewReadOnlySandbox(h.config.Sandbox)
	evalHarness := NewStandardHarness(evalConfig)

	var evalState SessionState
	evalHarness.config.ContextManager.AppendUserMessage(ctx, &evalState, directive)

	// Issue 2: the evaluate phase carries no per-node toolset handle, so it threads
	// the EMPTY handle → global-catalogue fallback, byte-for-byte.
	evalResult := evalHarness.runReActInner(ctx, evalTask, cap, evalState, BudgetSnapshot{}, nil, evalAgent, ToolsetRef(""))
	foldSelfVerifyUsage(totalUsage, carried, evalResult)
	return evalResult
}

// foldSelfVerifyUsage folds a sub-run's token usage / turn count into the
// cumulative totalUsage and the shared carried budget snapshot (R8). Mirrors the
// PlanExecute budget fold and Rust's fold_usage: carried.Turns becomes the
// build sub-loop's ABSOLUTE cumulative turn count (max), while the fresh-session
// evaluate run's turns are also max-folded in. WaitingForHuman carries no usage.
func foldSelfVerifyUsage(totalUsage *AggregateUsage, carried *BudgetSnapshot, r RunResult) {
	if r.Kind == RunWaitingForHuman {
		return
	}
	totalUsage.InputTokens += r.Usage.InputTokens
	totalUsage.OutputTokens += r.Usage.OutputTokens
	totalUsage.CacheReadTokens += r.Usage.CacheReadTokens
	totalUsage.CacheWriteTokens += r.Usage.CacheWriteTokens
	totalUsage.CostUSD += r.Usage.CostUSD
	carried.InputTokens += r.Usage.InputTokens
	carried.OutputTokens += r.Usage.OutputTokens
	if r.Turns > carried.Turns {
		carried.Turns = r.Turns
	}
}
