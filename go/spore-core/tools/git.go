// Git tools: GitLog, GitDiff, GitCommit, GitStatus, GitReset.
//
// Every git invocation goes through sandbox.ExecuteCommand — the tool never
// touches os/exec directly. That keeps the sandboxing contract intact.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// runGit shells out to `git <args...>` via the sandbox.
func runGit(
	ctx context.Context,
	args []string,
	sandbox sporecore.SandboxProvider,
) (sporecore.CommandOutput, *ToolExecutionError) {
	out, v := sandbox.ExecuteCommand(ctx, "git", args, "", 0)
	if v != nil {
		return sporecore.CommandOutput{}, SandboxViolationError(v)
	}
	return out, nil
}

// classify maps a git CommandOutput onto either its stdout (exit 0) or a
// recoverable error ToolOutput.
func classify(out sporecore.CommandOutput) (string, *sporecore.ToolOutput) {
	if out.ExitCode == 0 {
		return out.Stdout, nil
	}
	err := sporecore.ToolOutput{
		Kind:        sporecore.ToolOutputError,
		Message:     fmt.Sprintf("git exit %d ; %s", out.ExitCode, strings.TrimRight(out.Stderr, "\n")),
		Recoverable: true,
	}
	return "", &err
}

// ============================================================================
// GitLog
// ============================================================================

type GitLogTool struct{}

func NewGitLogTool() *GitLogTool { return &GitLogTool{} }

const GitLogToolName = "git_log"

func (*GitLogTool) Name() string                { return GitLogToolName }
func (*GitLogTool) IsSubagentTool() bool        { return false }
func (*GitLogTool) MayProduceLargeOutput() bool { return true }

func (*GitLogTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        GitLogToolName,
		Description: "Show recent git commits",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"n": {"type": "integer"},
				"format": {"type": "string"}
			}
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func (t *GitLogTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	params := GitLogParams{N: 20, Format: "oneline"}
	if len(call.Input) > 0 {
		if e := parseParams(call, &params); e != nil {
			return e.ToToolOutput()
		}
	}
	args := []string{"log", "-n", fmt.Sprintf("%d", params.N)}
	if params.Format == "oneline" {
		args = append(args, "--oneline")
	} else {
		args = append(args, "--format="+params.Format)
	}
	out, e := runGit(ctx, args, sandbox)
	if e != nil {
		return e.ToToolOutput()
	}
	body, errOut := classify(out)
	if errOut != nil {
		return *errOut
	}
	return finishWithPossibleTruncation(ctx, body, call.ID, sandbox)
}

// ============================================================================
// GitDiff
// ============================================================================

type GitDiffTool struct{}

func NewGitDiffTool() *GitDiffTool { return &GitDiffTool{} }

const GitDiffToolName = "git_diff"

func (*GitDiffTool) Name() string                { return GitDiffToolName }
func (*GitDiffTool) IsSubagentTool() bool        { return false }
func (*GitDiffTool) MayProduceLargeOutput() bool { return true }

func (*GitDiffTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        GitDiffToolName,
		Description: "Show a git diff",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"from": {"type": "string"},
				"to": {"type": "string"}
			}
		}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func (t *GitDiffTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params GitDiffParams
	if len(call.Input) > 0 {
		if e := parseParams(call, &params); e != nil {
			return e.ToToolOutput()
		}
	}
	args := []string{"diff"}
	if params.From != nil {
		args = append(args, *params.From)
	}
	if params.To != nil {
		args = append(args, *params.To)
	}
	out, e := runGit(ctx, args, sandbox)
	if e != nil {
		return e.ToToolOutput()
	}
	body, errOut := classify(out)
	if errOut != nil {
		return *errOut
	}
	return finishWithPossibleTruncation(ctx, body, call.ID, sandbox)
}

// ============================================================================
// GitCommit
// ============================================================================

type GitCommitTool struct{}

func NewGitCommitTool() *GitCommitTool { return &GitCommitTool{} }

const GitCommitToolName = "git_commit"

func (*GitCommitTool) Name() string                { return GitCommitToolName }
func (*GitCommitTool) IsSubagentTool() bool        { return false }
func (*GitCommitTool) MayProduceLargeOutput() bool { return false }

func (*GitCommitTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        GitCommitToolName,
		Description: "Stage files (if any) and create a git commit",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"message": {"type": "string"},
				"files": {"type": "array", "items": {"type": "string"}}
			},
			"required": ["message"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true},
	}
}

func (t *GitCommitTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params GitCommitParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	combined := ""
	if len(params.Files) > 0 {
		args := append([]string{"add"}, params.Files...)
		out, e := runGit(ctx, args, sandbox)
		if e != nil {
			return e.ToToolOutput()
		}
		body, errOut := classify(out)
		if errOut != nil {
			return *errOut
		}
		combined += body
	}
	out, e := runGit(ctx, []string{"commit", "-m", params.Message}, sandbox)
	if e != nil {
		return e.ToToolOutput()
	}
	body, errOut := classify(out)
	if errOut != nil {
		return *errOut
	}
	combined += body
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: combined}
}

// ============================================================================
// GitStatus
// ============================================================================

type GitStatusTool struct{}

func NewGitStatusTool() *GitStatusTool { return &GitStatusTool{} }

const GitStatusToolName = "git_status"

func (*GitStatusTool) Name() string                { return GitStatusToolName }
func (*GitStatusTool) IsSubagentTool() bool        { return false }
func (*GitStatusTool) MayProduceLargeOutput() bool { return false }

func (*GitStatusTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        GitStatusToolName,
		Description: "Show git status (porcelain)",
		Parameters:  json.RawMessage(`{"type": "object", "properties": {}}`),
		Annotations: sporecore.ToolAnnotations{ReadOnly: true},
	}
}

func (t *GitStatusTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	out, e := runGit(ctx, []string{"status", "--porcelain"}, sandbox)
	if e != nil {
		return e.ToToolOutput()
	}
	body, errOut := classify(out)
	if errOut != nil {
		return *errOut
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: body}
}

// ============================================================================
// GitReset
// ============================================================================

type GitResetTool struct{}

func NewGitResetTool() *GitResetTool { return &GitResetTool{} }

const GitResetToolName = "git_reset"

func (*GitResetTool) Name() string                { return GitResetToolName }
func (*GitResetTool) IsSubagentTool() bool        { return false }
func (*GitResetTool) MayProduceLargeOutput() bool { return false }

func (*GitResetTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        GitResetToolName,
		Description: "Reset to a target commit (hard/soft/mixed)",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"target": {"type": "string"},
				"mode": {"type": "string", "enum": ["hard", "soft", "mixed"]}
			},
			"required": ["target", "mode"]
		}`),
		Annotations: sporecore.ToolAnnotations{Destructive: true},
	}
}

func (t *GitResetTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	var params GitResetParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	var flag string
	switch params.Mode {
	case GitResetHard:
		flag = "--hard"
	case GitResetSoft:
		flag = "--soft"
	case GitResetMixed:
		flag = "--mixed"
	default:
		return InvalidParameters(fmt.Sprintf("unknown mode %q", params.Mode)).ToToolOutput()
	}
	out, e := runGit(ctx, []string{"reset", flag, params.Target}, sandbox)
	if e != nil {
		return e.ToToolOutput()
	}
	body, errOut := classify(out)
	if errOut != nil {
		return *errOut
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: body}
}

var (
	_ sporecore.Tool = (*GitLogTool)(nil)
	_ sporecore.Tool = (*GitDiffTool)(nil)
	_ sporecore.Tool = (*GitCommitTool)(nil)
	_ sporecore.Tool = (*GitStatusTool)(nil)
	_ sporecore.Tool = (*GitResetTool)(nil)
)
