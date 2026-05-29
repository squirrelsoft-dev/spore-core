// Deliberately-failing tool (issue #57).
//
// FailingTool always returns a *recoverable* tool error so the harness loop
// appends it as a tool result and lets the agent adapt rather than crashing or
// hanging (scenario S4). It must NOT be annotated as a Layer-1 always-halt tool.

package tools

import (
	"context"
	"encoding/json"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// FailingToolName is the registered name of the flaky operation.
const FailingToolName = "flaky_op"

// FailingTool is a tool whose Execute always fails with a recoverable error.
type FailingTool struct{}

// NewFailingTool constructs a FailingTool.
func NewFailingTool() *FailingTool { return &FailingTool{} }

// Name returns the registered tool name.
func (*FailingTool) Name() string { return FailingToolName }

// IsSubagentTool reports false — FailingTool wraps no child harness.
func (*FailingTool) IsSubagentTool() bool { return false }

// MayProduceLargeOutput reports false.
func (*FailingTool) MayProduceLargeOutput() bool { return false }

// Schema is the registry schema for the failing tool.
func (*FailingTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        FailingToolName,
		Description: "A flaky operation that fails recoverably; adapt and try another approach",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"reason": {"type": "string"}}
		}`),
		Annotations: sporecore.ToolAnnotations{Idempotent: true},
	}
}

// Execute always returns a recoverable ToolOutputError so the loop surfaces it
// to the agent instead of halting.
func (*FailingTool) Execute(_ context.Context, _ sporecore.ToolCall, _ sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	return sporecore.ToolOutput{
		Kind:        sporecore.ToolOutputError,
		Message:     "flaky_op is unavailable right now; try a different approach",
		Recoverable: true,
	}
}

var _ sporecore.Tool = (*FailingTool)(nil)
