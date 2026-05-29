// EditFile tool (#81, net-new Tier-1 sandbox tool).
//
// edit_file replaces the FIRST and ONLY occurrence of OldString with NewString
// in the file at Path. The match must be UNIQUE:
//   - OldString not found     → recoverable ToolOutput.Error.
//   - OldString found >1 time → recoverable ToolOutput.Error.
//
// This is a net-new tool that does NOT replace write_file (issue #81, Q5).
// Annotated destructive (it mutates a file in place).

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// EditFileToolName is the registered tool name.
const EditFileToolName = "edit_file"

// EditFileTool replaces the unique occurrence of a string in a file.
type EditFileTool struct{}

// NewEditFileTool constructs an EditFileTool.
func NewEditFileTool() *EditFileTool { return &EditFileTool{} }

func (*EditFileTool) Name() string                { return EditFileToolName }
func (*EditFileTool) IsSubagentTool() bool        { return false }
func (*EditFileTool) MayProduceLargeOutput() bool { return false }

func (*EditFileTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        EditFileToolName,
		Description: "Replace the unique occurrence of old_string with new_string in a file",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"old_string": {"type": "string"},
				"new_string": {"type": "string"}
			},
			"required": ["path", "old_string", "new_string"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true},
	}
}

func (t *EditFileTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params EditFileParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	resolved, v := sandbox.ResolvePath(ctx, params.Path, sporecore.OperationWrite)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	data, err := os.ReadFile(resolved)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("read failed: %s", err), true).ToToolOutput()
	}
	content := string(data)
	count := strings.Count(content, params.OldString)
	if count == 0 {
		return ExecutionFailed(fmt.Sprintf("old_string not found in %s", params.Path), true).ToToolOutput()
	}
	if count > 1 {
		return ExecutionFailed(
			fmt.Sprintf("old_string is not unique in %s (%d occurrences); provide more context", params.Path, count),
			true,
		).ToToolOutput()
	}
	updated := strings.Replace(content, params.OldString, params.NewString, 1)
	if err := os.WriteFile(resolved, []byte(updated), 0o644); err != nil {
		return ExecutionFailed(fmt.Sprintf("write failed: %s", err), true).ToToolOutput()
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: fmt.Sprintf("edited %s", params.Path)}
}

var _ sporecore.Tool = (*EditFileTool)(nil)
