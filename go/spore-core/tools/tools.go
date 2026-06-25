// Package tools houses the standard tool catalogue for spore-core (issue #5).
//
// Each tool conforms to the canonical sporecore.Tool interface defined in
// the parent package by issue #4. Tools are stateless and receive a
// SandboxProvider on every dispatch — they never touch the environment
// directly.
//
// Families:
//   - fs.go       — ReadFile, WriteFile, ListDir, DeleteFile, MoveFile
//   - exec.go     — BashCommand, RunTests
//   - search.go   — GrepFiles, FindFiles
//   - git.go      — GitLog, GitDiff, GitCommit, GitStatus, GitReset
//   - http.go     — HttpGet, HttpPost
//   - subagent.go — SubagentTool (wraps a child Harness)
//
// Output larger than LargeOutputThreshold is routed through
// SandboxProvider.HandleLargeOutput. Tools that may produce large output
// report MayProduceLargeOutput() == true.
package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// LargeOutputThreshold is the byte length above which tool output is routed
// through SandboxProvider.HandleLargeOutput instead of returned inline.
const LargeOutputThreshold = 64 * 1024

// DefaultHeadTokens / DefaultTailTokens are the head/tail budgets passed to
// HandleLargeOutput by default.
const (
	DefaultHeadTokens uint32 = 2000
	DefaultTailTokens uint32 = 2000
)

// ============================================================================
// ToolExecutionError — typed error class for tool implementations.
// ============================================================================

// ToolExecutionErrorKind discriminates ToolExecutionError variants.
type ToolExecutionErrorKind string

const (
	ToolExecErrInvalidParameters ToolExecutionErrorKind = "invalid_parameters"
	ToolExecErrExecutionFailed   ToolExecutionErrorKind = "execution_failed"
	ToolExecErrSandboxViolation  ToolExecutionErrorKind = "sandbox_violation"
	ToolExecErrTimeout           ToolExecutionErrorKind = "timeout"
)

// ToolExecutionError is the typed error returned by tool helpers. It is
// always convertible to a sporecore.ToolOutput via ToToolOutput(), which is
// how tools surface recoverable failures without panicking.
type ToolExecutionError struct {
	Kind        ToolExecutionErrorKind
	Reason      string
	Recoverable bool                        // ExecutionFailed only
	Violation   *sporecore.SandboxViolation // SandboxViolation only
	AfterSecs   uint64                      // Timeout only
}

// Error implements the error interface.
func (e *ToolExecutionError) Error() string {
	switch e.Kind {
	case ToolExecErrInvalidParameters:
		return "invalid parameters: " + e.Reason
	case ToolExecErrExecutionFailed:
		return "execution failed: " + e.Reason
	case ToolExecErrSandboxViolation:
		if e.Violation != nil {
			return "sandbox violation: " + e.Violation.Error()
		}
		return "sandbox violation"
	case ToolExecErrTimeout:
		return fmt.Sprintf("timed out after %ds", e.AfterSecs)
	default:
		return "tool execution error: " + string(e.Kind)
	}
}

// ToToolOutput maps a ToolExecutionError to a ToolOutput.Error with the
// recoverability rules from issue #5.
func (e *ToolExecutionError) ToToolOutput() sporecore.ToolOutput {
	switch e.Kind {
	case ToolExecErrInvalidParameters:
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     "invalid parameters: " + e.Reason,
			Recoverable: true,
		}
	case ToolExecErrExecutionFailed:
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     e.Reason,
			Recoverable: e.Recoverable,
		}
	case ToolExecErrSandboxViolation:
		// issue #150: carry the TYPED violation to the harness rather than
		// flattening it into a non-recoverable Error here. The harness applies its
		// SandboxViolationPolicy — recoverable feedback by default (the boundary
		// still holds; the access was refused), halt only on opt-in.
		return sporecore.ToolOutput{
			Kind:      sporecore.ToolOutputSandboxViolation,
			Violation: e.Violation,
		}
	case ToolExecErrTimeout:
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     fmt.Sprintf("timed out after %ds", e.AfterSecs),
			Recoverable: true,
		}
	default:
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     e.Error(),
			Recoverable: false,
		}
	}
}

// InvalidParameters constructs an InvalidParameters error.
func InvalidParameters(reason string) *ToolExecutionError {
	return &ToolExecutionError{Kind: ToolExecErrInvalidParameters, Reason: reason}
}

// ExecutionFailed constructs an ExecutionFailed error with explicit
// recoverability.
func ExecutionFailed(reason string, recoverable bool) *ToolExecutionError {
	return &ToolExecutionError{Kind: ToolExecErrExecutionFailed, Reason: reason, Recoverable: recoverable}
}

// SandboxViolationError wraps a *sporecore.SandboxViolation into a
// ToolExecutionError.
func SandboxViolationError(v *sporecore.SandboxViolation) *ToolExecutionError {
	return &ToolExecutionError{Kind: ToolExecErrSandboxViolation, Violation: v}
}

// TimeoutError constructs a Timeout error.
func TimeoutError(afterSecs uint64) *ToolExecutionError {
	return &ToolExecutionError{Kind: ToolExecErrTimeout, AfterSecs: afterSecs}
}

// ============================================================================
// Param parsing helper
// ============================================================================

// parseParams decodes call.Input into v. Any JSON decoding failure surfaces
// as an InvalidParameters ToolExecutionError.
func parseParams(call sporecore.ToolCall, v any) *ToolExecutionError {
	if len(call.Input) == 0 {
		return InvalidParameters("input must be a JSON object")
	}
	if err := json.Unmarshal(call.Input, v); err != nil {
		return InvalidParameters(err.Error())
	}
	return nil
}

// ============================================================================
// Truncation helper
// ============================================================================

// finishWithPossibleTruncation returns a Success output. If content exceeds
// LargeOutputThreshold, it is routed through sandbox.HandleLargeOutput first
// and the result is marked truncated.
func finishWithPossibleTruncation(
	ctx context.Context,
	content string,
	callID string,
	sandbox sporecore.SandboxProvider,
) sporecore.ToolOutput {
	if len(content) > LargeOutputThreshold {
		t := sandbox.HandleLargeOutput(ctx, content, callID, DefaultHeadTokens, DefaultTailTokens)
		return sporecore.ToolOutput{
			Kind:      sporecore.ToolOutputSuccess,
			Content:   t.Content,
			Truncated: true,
		}
	}
	return sporecore.ToolOutput{
		Kind:    sporecore.ToolOutputSuccess,
		Content: content,
	}
}
