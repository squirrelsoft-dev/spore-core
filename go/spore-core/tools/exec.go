// Execution tools: Exec, BashCommand, RunTests.
//
// Two distinct ways to run a process, with deliberately different contracts:
//
//   - ExecTool (tool name "exec") runs ONE program directly — no shell. command
//     + args are passed verbatim to SandboxProvider.ExecuteCommand, so there are
//     no pipes, redirects, globbing, or $(...). Every argument is literal. This
//     is the path-validated, no-injection-surface option.
//   - BashCommandTool (tool name "bash_command") runs a shell command line via
//     /bin/sh -c <script>, so it supports pipes, redirects, globbing, and
//     $(...). It is sugar over the same ExecuteCommand primitive
//     (ExecuteCommand("/bin/sh", ["-c", script], …)).
//
//     TRADEOFF: because the shell itself opens any files the script touches,
//     bash_command does NOT get the per-path sandbox validation that read_file /
//     write_file / exec get — it relies on the outer sandbox/container for
//     isolation. exec remains the path-validated choice. /bin/sh also assumes a
//     Unix target (fine for this repo; no cmd.exe/PowerShell branch).
//   - RunTestsTool (tool name "run_tests") splits a command string on whitespace
//     and runs it shell-free inside a working directory.

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
// Exec — shell-free: run one program directly
// ============================================================================

// ExecTool runs one program directly via SandboxProvider.ExecuteCommand. No
// shell: command + args are passed verbatim (no pipes, redirects, globbing, or
// $(...)). Path-validated through the sandbox.
type ExecTool struct{}

func NewExecTool() *ExecTool { return &ExecTool{} }

const ExecToolName = "exec"

func (*ExecTool) Name() string                { return ExecToolName }
func (*ExecTool) IsSubagentTool() bool        { return false }
func (*ExecTool) MayProduceLargeOutput() bool { return true }

func (*ExecTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        ExecToolName,
		Description: "Run one program directly. No shell: no pipes, redirects, globbing, or $(...). Args are passed verbatim.",
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

func (t *ExecTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params ExecParams
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
// BashCommand — real shell: /bin/sh -c <script>
// ============================================================================

// BashCommandTool runs a shell command line via /bin/sh -c <script>, supporting
// pipes, redirects, globbing, and $(...). It is sugar over the same
// SandboxProvider.ExecuteCommand primitive ExecTool uses
// (ExecuteCommand("/bin/sh", ["-c", script], workingDir?, timeout?)).
//
// TRADEOFF: the shell opens any files the script touches itself, so this tool
// does NOT receive the per-path sandbox validation that read_file / write_file /
// ExecTool get — it relies on the outer sandbox/container for isolation. exec
// remains the path-validated choice. /bin/sh assumes a Unix target (no Windows
// shell branch).
type BashCommandTool struct{}

func NewBashCommandTool() *BashCommandTool { return &BashCommandTool{} }

const BashCommandToolName = "bash_command"

func (*BashCommandTool) Name() string                { return BashCommandToolName }
func (*BashCommandTool) IsSubagentTool() bool        { return false }
func (*BashCommandTool) MayProduceLargeOutput() bool { return true }

func (*BashCommandTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        BashCommandToolName,
		Description: "Execute a shell command line via /bin/sh -c. Supports pipes, redirects, globbing, and $(...).",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"script": {"type": "string"},
				"working_dir": {"type": "string"},
				"timeout": {"type": "integer"}
			},
			"required": ["script"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true, OpenWorld: true},
	}
}

func (t *BashCommandTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
	var params ShellCommandParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	var timeout time.Duration
	if params.Timeout != nil {
		timeout = time.Duration(*params.Timeout) * time.Second
	}
	// Only the optional working_dir is path-validated; the script's own file
	// accesses go through the shell, unvalidated (see the type doc-comment).
	working := ""
	if params.WorkingDir != "" {
		resolved, v := sandbox.ResolvePath(ctx, params.WorkingDir, sporecore.OperationRead)
		if v != nil {
			return SandboxViolationError(v).ToToolOutput()
		}
		working = resolved
	}
	args := []string{"-c", params.Script}
	out, v := sandbox.ExecuteCommand(ctx, "/bin/sh", args, working, timeout)
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

func (t *RunTestsTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider, _ *sporecore.ToolContext) sporecore.ToolOutput {
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
	_ sporecore.Tool = (*ExecTool)(nil)
	_ sporecore.Tool = (*BashCommandTool)(nil)
	_ sporecore.Tool = (*RunTestsTool)(nil)
)
