// Per-tool input parameter structs (issue #5).
//
// One struct per tool. Each maps 1:1 onto the Rust serde struct in
// rust/crates/spore-core/src/tools/params.rs. Wire-compatible JSON tags use
// snake_case.

package tools

import (
	"encoding/json"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
	"github.com/squirrelsoft-dev/spore-core/go/spore-core/storage"
)

// StorageScope is re-exported from the storage package (its canonical home is
// promptassembly; storage re-exports it). The MemoryTool params name it without
// forcing every params caller to import storage directly.
type StorageScope = storage.StorageScope

// Scope constants re-exported for use at the MemoryTool call site.
const (
	StorageScopeUser    = storage.StorageScopeUser
	StorageScopeProject = storage.StorageScopeProject
	StorageScopeLocal   = storage.StorageScopeLocal
)

// ----- Filesystem -----

// ReadFileParams are the parameters for ReadFileTool. With all optional fields
// at their defaults (Offset=nil, Length=0, LineNumbers=false) the read is
// byte-identical to reading the whole file — no header, no line numbers (#132).
// Any non-default param prepends a one-line header
// "[lines {start}–{end} of {total}]" (U+2013 en-dash).
//
// Offset is a *uint64 (pointer) so we can distinguish "not provided" (nil,
// default=1) from explicitly-zero (0, which is a recoverable error per spec).
type ReadFileParams struct {
	Path string `json:"path"`
	// 1-indexed start line. nil means "not provided" (default = 1).
	// Explicitly 0 is a recoverable error.
	Offset *uint64 `json:"offset,omitempty"`
	// Max lines to return. 0 = no limit / read to EOF (default 0).
	// A length that runs past EOF silently returns through the last line.
	Length uint64 `json:"length,omitempty"`
	// When true, prefix each returned line with its 1-indexed number,
	// right-padded to the digit-width of the file's total line count.
	LineNumbers bool `json:"line_numbers,omitempty"`
}

type WriteFileParams struct {
	Path    string `json:"path"`
	Content string `json:"content"`
	Append  bool   `json:"append,omitempty"`
}

type ListDirParams struct {
	Path           string `json:"path"`
	Recursive      bool   `json:"recursive,omitempty"`
	IncludeIgnored bool   `json:"include_ignored,omitempty"` // default false
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
// GrepFilesParams: adds OutputMode (defaulting to content) and ContextLines.
type GrepParams struct {
	Pattern      string         `json:"pattern"`
	Path         string         `json:"path"`
	Recursive    bool           `json:"recursive,omitempty"`
	OutputMode   GrepOutputMode `json:"output_mode,omitempty"`
	ContextLines uint32         `json:"context_lines,omitempty"` // default 0
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

// ----- Memory (#82) -----

// MemoryOperation is the operation discriminator for MemoryToolParams. One of
// "write" | "read".
type MemoryOperation string

const (
	// MemoryOperationWrite appends one MemoryEntry to a scope.
	MemoryOperationWrite MemoryOperation = "write"
	// MemoryOperationRead returns the most-recent entries for a scope (or the
	// cross-scope merged view).
	MemoryOperationRead MemoryOperation = "read"
)

// MemoryDefaultReadLimit is the recency cap applied to a read when `limit` is
// omitted (decision B).
const MemoryDefaultReadLimit = 50

// MemoryToolParams are the parameters for the MemoryTool. It mirrors the Rust
// internally-tagged MemoryToolParams enum (tagged on `operation`): an
// `operation` discriminator plus the union of per-operation fields. `scope` is
// explicit and required on BOTH operations.
//
// Per-operation fields (validated in the tool, not by serde):
//   - write: scope (required), role (required), content (required),
//     metadata (optional, defaults to {})
//   - read:  scope (required), merged (optional, defaults false),
//     limit (optional, defaults to MemoryDefaultReadLimit)
//
// Scope is decoded as a raw promptassembly StorageScope string ("user" /
// "project" / "local"). StorageScopeLocal decodes fine here so a bad-scope call
// reaches the tool body, where it is rejected at runtime with a recoverable
// error (the advertised schema enum omits "local").
//
// Limit uses a *int so an omitted `limit` (nil) is distinguishable from an
// explicit 0; the tool applies MemoryDefaultReadLimit when nil. Metadata uses
// json.RawMessage; UnmarshalJSON fills it with {} when absent or null so the
// stored entry's metadata default matches Rust's serde default.
type MemoryToolParams struct {
	Operation MemoryOperation `json:"operation"`
	Scope     StorageScope    `json:"scope"`
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
	Merged    bool            `json:"merged,omitempty"`
	Limit     *int            `json:"limit,omitempty"`
}

// UnmarshalJSON fills Metadata with the empty object {} when the field is absent
// or null, mirroring serde's #[serde(default)] on the write metadata param.
func (p *MemoryToolParams) UnmarshalJSON(data []byte) error {
	type alias MemoryToolParams
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	if len(a.Metadata) == 0 || string(a.Metadata) == "null" {
		a.Metadata = json.RawMessage("{}")
	}
	*p = MemoryToolParams(a)
	return nil
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
//   - add_task:      description (required); blockers (optional, defaults empty)
//   - update_task:   id (required); status and/or description (both optional)
//   - complete_task: id (required)
//   - list_tasks:    no fields
//
// Blockers (#118) are the ids that must be completed before the added task runs;
// omitting the field defaults to empty. They are threaded through to
// TaskList.Add, which validates them.
type TaskListParams struct {
	Action      TaskListAction        `json:"action"`
	Description *string               `json:"description,omitempty"`
	ID          *uint32               `json:"id,omitempty"`
	Status      *sporecore.TaskStatus `json:"status,omitempty"`
	Blockers    []uint32              `json:"blockers,omitempty"`
}
