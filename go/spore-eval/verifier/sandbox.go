package verifier

import (
	"context"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// directSandbox runs commands directly in a workspace dir. Used by the
// TestSuiteVerifier and MetricEvaluatorVerifier for the verification command
// only; it inherits the non-isolating default ExecuteCommand / HandleLargeOutput
// / ResolvePath from sporecore.DefaultSandbox and approves every Validate call.
type directSandbox struct {
	sporecore.DefaultSandbox
	root string
}

func newDirectSandbox(root string) *directSandbox { return &directSandbox{root: root} }

// Validate approves every call (verification runs in the restored workspace).
func (s *directSandbox) Validate(context.Context, sporecore.ToolCall) *sporecore.SandboxViolation {
	return nil
}

// WorkspaceRoot returns the restored workspace root.
func (s *directSandbox) WorkspaceRoot() string { return s.root }

var _ sporecore.SandboxProvider = (*directSandbox)(nil)
