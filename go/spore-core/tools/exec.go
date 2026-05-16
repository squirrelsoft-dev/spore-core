// Execution tools: BashCommand, RunTests.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ============================================================================
// BashCommand
// ============================================================================

type BashCommandTool struct{}

func NewBashCommandTool() *BashCommandTool { return &BashCommandTool{} }

const BashCommandToolName = "bash_command"

func (*BashCommandTool) Name() string                { return BashCommandToolName }
func (*BashCommandTool) IsSubagentTool() bool        { return false }
func (*BashCommandTool) MayProduceLargeOutput() bool { return true }

func (*BashCommandTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        BashCommandToolName,
		Description: "Execute a shell command via the sandbox",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"command": {"type": "string"},
				"args": {"type": "array", "items": {"type": "string"}},
				"timeout": {"type": "integer"}
			},
			"required": ["command"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true, OpenWorld: true},
	}
}

func (t *BashCommandTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params BashCommandParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	var timeout time.Duration
	if params.Timeout != nil {
		timeout = time.Duration(*params.Timeout) * time.Second
	}
	out, v := sandbox.ExecuteCommand(ctx, params.Command, params.Args, "", timeout)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	if out.TimedOut {
		secs := uint64(0)
		if params.Timeout != nil {
			secs = *params.Timeout
		}
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     fmt.Sprintf("command timed out after %ds", secs),
			Recoverable: true,
		}
	}
	if out.ExitCode == 0 {
		return finishWithPossibleTruncation(ctx, out.Stdout, call.ID, sandbox)
	}
	return sporecore.ToolOutput{
		Kind:        sporecore.ToolOutputError,
		Message:     fmt.Sprintf("exit %d ; stderr: %s", out.ExitCode, strings.TrimRight(out.Stderr, "\n")),
		Recoverable: true,
	}
}

// ============================================================================
// RunTests
// ============================================================================

type RunTestsTool struct{}

func NewRunTestsTool() *RunTestsTool { return &RunTestsTool{} }

const RunTestsToolName = "run_tests"

func (*RunTestsTool) Name() string                { return RunTestsToolName }
func (*RunTestsTool) IsSubagentTool() bool        { return false }
func (*RunTestsTool) MayProduceLargeOutput() bool { return true }

func (*RunTestsTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        RunTestsToolName,
		Description: "Run a test command in a working directory",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"command": {"type": "string"},
				"working_dir": {"type": "string"},
				"timeout": {"type": "integer"}
			},
			"required": ["command", "working_dir"]
		}`),
		Annotations: sporecore.ToolAnnotations{OpenWorld: true},
	}
}

func (t *RunTestsTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params RunTestsParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	var timeout time.Duration
	if params.Timeout != nil {
		timeout = time.Duration(*params.Timeout) * time.Second
	}
	working, v := sandbox.ResolvePath(ctx, params.WorkingDir, sporecore.OperationExecute)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	parts := strings.Fields(params.Command)
	if len(parts) == 0 {
		return InvalidParameters("command must not be empty").ToToolOutput()
	}
	program := parts[0]
	args := parts[1:]
	out, sv := sandbox.ExecuteCommand(ctx, program, args, working, timeout)
	if sv != nil {
		return SandboxViolationError(sv).ToToolOutput()
	}
	if out.TimedOut {
		secs := uint64(0)
		if params.Timeout != nil {
			secs = *params.Timeout
		}
		return sporecore.ToolOutput{
			Kind:        sporecore.ToolOutputError,
			Message:     fmt.Sprintf("tests timed out after %ds", secs),
			Recoverable: true,
		}
	}
	combined := out.Stdout + "\n" + out.Stderr
	if out.ExitCode == 0 {
		return finishWithPossibleTruncation(ctx, combined, call.ID, sandbox)
	}
	return sporecore.ToolOutput{
		Kind:        sporecore.ToolOutputError,
		Message:     fmt.Sprintf("tests failed (exit %d): %s", out.ExitCode, combined),
		Recoverable: true,
	}
}

var (
	_ sporecore.Tool = (*BashCommandTool)(nil)
	_ sporecore.Tool = (*RunTestsTool)(nil)
)
