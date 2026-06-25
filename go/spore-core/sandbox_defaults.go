// Default sandbox helpers.
//
// DefaultSandbox supplies non-sandboxed default implementations of the
// SandboxProvider methods that are not specific to a workspace root. Test
// stubs and the in-tree AllowAllSandbox embed it so they don't have to spell
// out each method individually.
//
// These defaults are deliberately not production-safe:
//   - ExecuteCommand spawns processes directly via os/exec with no isolation.
//   - ResolvePath returns the raw path unchanged.
//   - HandleLargeOutput truncates head+tail in memory and never offloads.
//   - IsolationMode is IsolationWorkspaceScoped (safe-by-default, issue #34)
//     and WorkspaceRoot is empty.
//
// Issue #6 (SandboxProvider) ships the canonical WorkspaceScopedSandbox in
// sandbox.go for production use.

package sporecore

import (
	"context"
	"fmt"
	"os/exec"
	"time"
)

// DefaultSandbox supplies the non-sandboxed defaults for the
// SandboxProvider methods that are reused across test stubs.
type DefaultSandbox struct{}

// ExecuteCommand runs a subprocess via os/exec.CommandContext. A non-zero
// timeout bounds execution; on expiry, TimedOut is set and ExitCode is -1. A
// spawn failure (missing binary, OS refusal) returns a typed
// SandboxExecSpawnFailed violation (SC-15) — never started is no longer
// conflated with ran-and-exited -1; timeouts still land in CommandOutput.
func (DefaultSandbox) ExecuteCommand(
	ctx context.Context,
	command string,
	args []string,
	workingDir string,
	timeout time.Duration,
) (CommandOutput, *SandboxViolation) {
	runCtx := ctx
	var cancel context.CancelFunc
	if timeout > 0 {
		runCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	cmd := exec.CommandContext(runCtx, command, args...)
	if workingDir != "" {
		cmd.Dir = workingDir
	}
	stdout, err := cmd.Output()
	stderr := ""
	if ee, ok := err.(*exec.ExitError); ok {
		stderr = string(ee.Stderr)
	}
	exitCode := 0
	timedOut := false
	if err != nil {
		if runCtx.Err() == context.DeadlineExceeded {
			return CommandOutput{
				Stdout:   "",
				Stderr:   fmt.Sprintf("command timed out after %s", timeout),
				ExitCode: -1,
				TimedOut: true,
			}, nil
		}
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			// SC-15: a failed spawn is a typed violation, not a fake
			// CommandOutput{ExitCode: -1}. Callers already handle the arm.
			return CommandOutput{}, &SandboxViolation{
				Kind:    SandboxExecSpawnFailed,
				Command: command,
				Message: err.Error(),
			}
		}
	}
	return CommandOutput{
		Stdout:   string(stdout),
		Stderr:   stderr,
		ExitCode: exitCode,
		TimedOut: timedOut,
	}, nil
}

// HandleLargeOutput head+tail-truncates content using a 4-chars-per-token
// approximation. Returns the original content unchanged if it already fits.
// Never offloads — FullRef is always nil.
func (DefaultSandbox) HandleLargeOutput(
	_ context.Context,
	content string,
	_ string,
	headTokens uint32,
	tailTokens uint32,
) TruncatedOutput {
	headChars := int(headTokens) * 4
	tailChars := int(tailTokens) * 4
	// Operate on runes so the truncation is UTF-8 safe.
	runes := []rune(content)
	originalSize := uint64(len(content))
	if len(runes) <= headChars+tailChars {
		return TruncatedOutput{Content: content, Truncated: false, OriginalSize: originalSize}
	}
	head := string(runes[:headChars])
	tail := string(runes[len(runes)-tailChars:])
	elided := len(runes) - headChars - tailChars
	summary := fmt.Sprintf("%s\n... [%d chars elided] ...\n%s", head, elided, tail)
	return TruncatedOutput{Content: summary, Truncated: true, OriginalSize: originalSize}
}

// ResolvePath is an identity pass-through — the default sandbox does not
// enforce a workspace root.
func (DefaultSandbox) ResolvePath(_ context.Context, path string, _ Operation) (string, *SandboxViolation) {
	return path, nil
}

// IsolationMode reports IsolationWorkspaceScoped — the safe-by-default
// isolation mode (issue #34). The default sandbox is a non-sandboxed stub, but
// it must not advertise the gated no-enforcement IsolationNone; that mode only
// exists under the `dangerous` build tag and requires an explicit opt-in.
func (DefaultSandbox) IsolationMode() IsolationMode { return IsolationWorkspaceScoped{} }

// WorkspaceRoot returns the empty string — the default sandbox does not
// enforce a workspace root.
func (DefaultSandbox) WorkspaceRoot() string { return "" }
