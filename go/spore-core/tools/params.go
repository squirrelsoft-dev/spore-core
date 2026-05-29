// Per-tool input parameter structs (issue #5).
//
// One struct per tool. Each maps 1:1 onto the Rust serde struct in
// rust/crates/spore-core/src/tools/params.rs. Wire-compatible JSON tags use
// snake_case.

package tools

import (
	"encoding/json"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// ----- Filesystem -----

type ReadFileParams struct {
	Path string `json:"path"`
}

type WriteFileParams struct {
	Path    string `json:"path"`
	Content string `json:"content"`
	Append  bool   `json:"append,omitempty"`
}

type ListDirParams struct {
	Path      string `json:"path"`
	Recursive bool   `json:"recursive,omitempty"`
}

type DeleteFileParams struct {
	Path string `json:"path"`
}

type MoveFileParams struct {
	Src string `json:"src"`
	Dst string `json:"dst"`
}

// ----- Exec -----

// ExecParams are the parameters for the shell-free ExecTool: a program name
// plus a verbatim argument vector. No shell is involved, so the args are passed
// literally (no pipes, redirects, globbing, or $(...)).
type ExecParams struct {
	Command string   `json:"command"`
	Args    []string `json:"args,omitempty"`
	// Timeout in whole seconds. *uint64 so the absence of the field is
	// distinguishable from 0.
	Timeout *uint64 `json:"timeout,omitempty"`
}

// ShellCommandParams are the parameters for the real BashCommandTool: a single
// shell script run via /bin/sh -c, with an optional working directory.
type ShellCommandParams struct {
	Script string `json:"script"`
	// Optional working directory; the only path that gets sandbox validation.
	WorkingDir string `json:"working_dir,omitempty"`
	// Timeout in whole seconds. *uint64 so the absence of the field is
	// distinguishable from 0.
	Timeout *uint64 `json:"timeout,omitempty"`
}

type RunTestsParams struct {
	Command    string  `json:"command"`
	WorkingDir string  `json:"working_dir"`
	Timeout    *uint64 `json:"timeout,omitempty"`
}

// ----- Search -----

type GrepFilesParams struct {
	Pattern   string `json:"pattern"`
	Path      string `json:"path"`
	Recursive bool   `json:"recursive,omitempty"`
}

type FindFilesParams struct {
	Glob string `json:"glob"`
	Path string `json:"path"`
}

// ----- EditFile (#81, new) -----

// EditFileParams are the parameters for EditFileTool: replace the FIRST and
// ONLY occurrence of OldString with NewString in the file at Path. The match
// must be unique — an absent or non-unique OldString is a recoverable error.
type EditFileParams struct {
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

// ----- Grep (#81, new — output modes) -----

// GrepOutputMode selects GrepTool's output shape. One of "content" (default),
// "files_with_matches", or "count".
type GrepOutputMode string

const (
	// GrepOutputContent emits each matching line as `path:line:text` (default).
	GrepOutputContent GrepOutputMode = "content"
	// GrepOutputFilesWithMatches emits the distinct file paths with a match.
	GrepOutputFilesWithMatches GrepOutputMode = "files_with_matches"
	// GrepOutputCount emits `path:count` for each file with matches.
	GrepOutputCount GrepOutputMode = "count"
)

// GrepParams are the parameters for the net-new GrepTool. Distinct from
// GrepFilesParams: adds OutputMode (defaulting to content).
type GrepParams struct {
	Pattern    string         `json:"pattern"`
	Path       string         `json:"path"`
	Recursive  bool           `json:"recursive,omitempty"`
	OutputMode GrepOutputMode `json:"output_mode,omitempty"`
}

// UnmarshalJSON applies the default OutputMode=content when absent or empty.
func (p *GrepParams) UnmarshalJSON(data []byte) error {
	type alias GrepParams
	a := alias{}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	if a.OutputMode == "" {
		a.OutputMode = GrepOutputContent
	}
	*p = GrepParams(a)
	return nil
}

// ----- SendMessage (#81, new) -----

// SendMessageParams are the parameters for SendMessageTool.
type SendMessageParams struct {
	Content string `json:"content"`
}

// ----- TodoWrite (#81, new) -----

// TodoStatus is one of "pending" | "in_progress" | "completed".
type TodoStatus string

const (
	TodoStatusPending    TodoStatus = "pending"
	TodoStatusInProgress TodoStatus = "in_progress"
	TodoStatusCompleted  TodoStatus = "completed"
)

// TodoItem is a single todo entry managed by TodoWriteTool.
type TodoItem struct {
	Content string     `json:"content"`
	Status  TodoStatus `json:"status"`
}

// TodoWriteParams are the parameters for TodoWriteTool: the agent supplies the
// FULL desired todo list, which replaces the persisted list wholesale.
type TodoWriteParams struct {
	Todos []TodoItem `json:"todos"`
}

// ----- WebFetch / WebSearch (#81, new) -----

// WebFetchParams are the parameters for WebFetchTool.
type WebFetchParams struct {
	URL string `json:"url"`
}

// WebSearchParams are the parameters for WebSearchTool.
type WebSearchParams struct {
	Query string `json:"query"`
}

// ----- Tier 3: plan / clarify / abort (#81, new) -----

// EnterPlanModeParams are the parameters for EnterPlanModeTool. Context is
// optional (defaults to empty).
type EnterPlanModeParams struct {
	Context string `json:"context,omitempty"`
}

// ExitPlanModeParams are the parameters for ExitPlanModeTool. The agent supplies
// the plan as a structured object that deserializes DIRECTLY into the existing
// sporecore.PlanArtifact (issue #81, Q4a — no stub).
type ExitPlanModeParams struct {
	Plan sporecore.PlanArtifact `json:"plan"`
}

// AskUserQuestionParams are the parameters for AskUserQuestionTool. Options is
// nil for a free-form clarification.
type AskUserQuestionParams struct {
	Question string    `json:"question"`
	Options  *[]string `json:"options,omitempty"`
}

// AbortParams are the parameters for AbortTool.
type AbortParams struct {
	Reason string `json:"reason"`
}

// ----- Git -----

type GitLogParams struct {
	N      uint32 `json:"n"`
	Format string `json:"format"`
}

// UnmarshalJSON applies the defaults n=20, format="oneline".
func (p *GitLogParams) UnmarshalJSON(data []byte) error {
	type alias GitLogParams
	a := alias{N: 20, Format: "oneline"}
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	if a.N == 0 {
		a.N = 20
	}
	if a.Format == "" {
		a.Format = "oneline"
	}
	*p = GitLogParams(a)
	return nil
}

type GitDiffParams struct {
	From *string `json:"from,omitempty"`
	To   *string `json:"to,omitempty"`
}

type GitCommitParams struct {
	Message string   `json:"message"`
	Files   []string `json:"files,omitempty"`
}

type GitStatusParams struct{}

// GitResetMode is one of "hard" | "soft" | "mixed".
type GitResetMode string

const (
	GitResetHard  GitResetMode = "hard"
	GitResetSoft  GitResetMode = "soft"
	GitResetMixed GitResetMode = "mixed"
)

type GitResetParams struct {
	Target string       `json:"target"`
	Mode   GitResetMode `json:"mode"`
}

// ----- HTTP -----

type HttpGetParams struct {
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers,omitempty"`
}

type HttpPostParams struct {
	URL     string            `json:"url"`
	Body    json.RawMessage   `json:"body"`
	Headers map[string]string `json:"headers,omitempty"`
}

// ----- Subagent -----

type SubagentParams struct {
	Instruction string `json:"instruction"`
}

// ----- TaskList (#71) -----

// TaskListAction is the action discriminator for TaskListParams. One of
// "add_task" | "update_task" | "complete_task" | "list_tasks".
type TaskListAction string

const (
	TaskListActionAddTask      TaskListAction = "add_task"
	TaskListActionUpdateTask   TaskListAction = "update_task"
	TaskListActionCompleteTask TaskListAction = "complete_task"
	TaskListActionListTasks    TaskListAction = "list_tasks"
)

// TaskListParams are the parameters for the TaskListTool. Mirrors the Rust
// internally-tagged TaskListParams enum: an `action` discriminator plus the
// union of per-action fields. Pointer fields distinguish present-vs-absent so
// update_task can apply status and/or description independently.
//
// Field requirements per action (validated in the tool, not by serde):
//   - add_task:      description (required)
//   - update_task:   id (required); status and/or description (both optional)
//   - complete_task: id (required)
//   - list_tasks:    no fields
type TaskListParams struct {
	Action      TaskListAction        `json:"action"`
	Description *string               `json:"description,omitempty"`
	ID          *uint32               `json:"id,omitempty"`
	Status      *sporecore.TaskStatus `json:"status,omitempty"`
}
