// TaskList tool (#71, storage seam #75): the single mutating tool over the
// persisted task list.
//
// One tool, TaskListTool (NAME = "task_list"), dispatched on an `action`
// discriminator (add_task, update_task, complete_task, list_tasks). See the
// parent package's tasklist.go for the types and the transition matrix.
//
// # Storage seam (#75)
//
// The tool persists via the *ToolContext's RunStore — NOT the sandbox
// filesystem. It is read-modify-write keyed by the STABLE project namespace
// (#142 — ToolContext.ProjectNamespace, falling back to SessionID when unset)
// under TaskListExtrasKey ("task_list"):
//  1. parse params (bad input → recoverable error),
//  2. toolCtx.GetDurable(ctx, "task_list") (absent → DefaultTaskList),
//  3. apply the action (domain errors → recoverable),
//  4. on a mutating action, toolCtx.PutDurable(ctx, "task_list", value),
//  5. return the serialized current list as success content (on add_task,
//     prefixed with the assigned id as a leading `added` key — see #143 below).
//
// Keying by the project namespace (not the per-window SessionID) is the #142 fix:
// the Ralph wrapper mints a fresh SessionID each context window, so a
// session-keyed list was orphaned at every window boundary; a project-keyed list
// survives window resets AND process restarts (with a durable RunStore backend).
//
// # add_task surfaces the assigned id (#143)
//
// On a successful add_task, the success content is the canonical TaskList object
// with ONE extra top-level key `added` placed FIRST — the id just assigned by
// TaskList.Add:
//
//	{"added":3,"tasks":[...],"next_id":4}
//
// The field order is EXACTLY `added`, then `tasks`, then `next_id`, and is
// byte-identical across all four languages so a model can reference a just-added
// task without re-parsing the whole list or predicting ids. The `added` key
// appears ONLY on the add_task success branch — update_task, complete_task, and
// list_tasks keep returning the bare serialized TaskList ({"tasks":[...],
// "next_id":N}), unchanged. A rejected add_task (self-block / unknown blocker /
// cycle) still returns a recoverable error with NO `added` and no list.
//
// CRITICAL: the PERSISTED RunStore blob stays EXACTLY {"tasks":[...],"next_id":N}
// — NO `added` key. `added` lives only in the tool's success content, never in
// what is persisted; the PlanExecute executor depends on the persisted blob shape.
//
// Shared key: this standalone tool and the harness-side PlanExecute execute loop
// (#76) persist under the SAME RunStore key ("task_list"), keyed by the project
// namespace (#142). A standalone tool call and a PlanExecute run on the same
// project intentionally share one blob. The JSON shape is the canonical TaskList
// serialization ({"tasks":[...],"next_id":N}), unchanged.
//
// Behavior change vs the retired sandbox path: previously the tool persisted to
// .spore/task_list.json via the sandbox. That path is GONE. With the library's
// default storage (storage.NoOp / a nil RunStore on the ToolContext) a standalone
// tool call persists NOTHING across processes — the no-op run store silently
// discards writes and returns "not found" on read. This is an accepted behavior
// change: durable cross-process persistence now requires configuring a real
// StorageProvider (and injecting it via StandardToolRegistry.SetToolContext).
// There is NO migration shim for old on-disk .spore/task_list.json files.
//
// Storage-error mapping: a storage Get/Put error maps to a recoverable
// ToolOutput error, as does a present-but-malformed blob (parse failure).
// list_tasks never writes.
//
// CRITICAL: this tool is NOT annotated ReadOnly. Read-only tools are run
// CONCURRENTLY by DispatchAll, and a concurrent read-modify-write over the same
// key would race. Leaving ReadOnly false makes the registry dispatch it
// sequentially. Destructive / OpenWorld are also left false so it is not
// treated as an irreversible side effect.

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// TaskListToolName is the registered tool name.
const TaskListToolName = "task_list"

// TaskListTool is the single mutating tool over the persisted task list.
type TaskListTool struct{}

// NewTaskListTool constructs a TaskListTool.
func NewTaskListTool() *TaskListTool { return &TaskListTool{} }

func (*TaskListTool) Name() string                { return TaskListToolName }
func (*TaskListTool) IsSubagentTool() bool        { return false }
func (*TaskListTool) MayProduceLargeOutput() bool { return false }

func (*TaskListTool) Schema() sporecore.RegistryToolSchema {
	// Fields kept sorted/stable for cache stability: `action` (required) plus the
	// union of per-action fields, listed alphabetically.
	return sporecore.RegistryToolSchema{
		Name:        TaskListToolName,
		Description: "Manage the persisted task list: add, update, complete, or list tasks",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"action": {
					"type": "string",
					"enum": ["add_task", "complete_task", "list_tasks", "update_task"]
				},
				"blockers": {"type": "array", "items": {"type": "integer"}},
				"description": {"type": "string"},
				"id": {"type": "integer"},
				"status": {
					"type": "string",
					"enum": ["blocked", "completed", "in_progress", "pending"]
				}
			},
			"required": ["action"]
		}`),
		// Intentionally NOT ReadOnly: this tool mutates shared on-disk state and
		// must dispatch sequentially. See package docs.
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *TaskListTool) Execute(ctx context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	// 1. Parse params (bad input → recoverable).
	var params TaskListParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	if e := validateTaskListParams(&params); e != nil {
		return e.ToToolOutput()
	}

	// 2. Load current list from the run store, keyed by the STABLE project
	//    namespace (#142) under the shared TaskListExtrasKey (absent → default).
	//    Keying by project_id (not the per-window SessionID the Ralph wrapper
	//    regenerates) is what lets a window reset re-read the prior window's list
	//    instead of re-planning under a session it has never seen. A storage error
	//    or a malformed blob is recoverable.
	list := sporecore.DefaultTaskList()
	value, found, err := toolCtx.GetDurable(ctx, sporecore.TaskListExtrasKey)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("could not load task list: %s", err), true).ToToolOutput()
	}
	if found {
		if err := json.Unmarshal(value, &list); err != nil {
			return ExecutionFailed(fmt.Sprintf("could not parse task list: %s", err), true).ToToolOutput()
		}
	}

	// 3. Apply the action. Domain errors → recoverable. list_tasks does not
	//    mutate. `added` carries the id assigned by Add so the add branch can
	//    surface it in the success content (#143); it stays nil for the non-add
	//    (and read-only) actions, which return the bare serialized TaskList.
	mutated := false
	var added *uint32
	switch params.Action {
	case TaskListActionAddTask:
		// Capture the assigned id (#143) instead of discarding it. A rejected
		// blocker set still maps to a recoverable error and leaves the list
		// untouched.
		id, domErr := list.Add(*params.Description, params.Blockers)
		if domErr != nil {
			return ExecutionFailed(domErr.Error(), true).ToToolOutput()
		}
		added = &id
		mutated = true
	case TaskListActionUpdateTask:
		if domErr := list.Update(*params.ID, params.Status, params.Description); domErr != nil {
			return ExecutionFailed(domErr.Error(), true).ToToolOutput()
		}
		mutated = true
	case TaskListActionCompleteTask:
		if domErr := list.Complete(*params.ID); domErr != nil {
			return ExecutionFailed(domErr.Error(), true).ToToolOutput()
		}
		mutated = true
	case TaskListActionListTasks:
		// No mutation.
	}

	// 4. Persist the (possibly mutated) list through the run store under the
	//    shared TaskListExtrasKey, keyed by the STABLE project namespace (#142).
	//    list_tasks skips the write.
	if mutated {
		encoded, err := json.Marshal(list)
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("could not serialize task list: %s", err), true).ToToolOutput()
		}
		if err := toolCtx.PutDurable(ctx, sporecore.TaskListExtrasKey, encoded); err != nil {
			return ExecutionFailed(fmt.Sprintf("could not persist task list: %s", err), true).ToToolOutput()
		}
	}

	// 5. Return the serialized current list. On add_task (#143) splice the
	//    assigned id in as a leading `added` key so the success content is
	//    {"added":N,"tasks":[...],"next_id":M} — exactly that field order,
	//    byte-identical across languages. The canonical TaskList marshaling is
	//    {"tasks":[...],"next_id":M} (struct fields marshal in declaration order,
	//    NOT alphabetically), so splicing `"added":N,` right after the opening
	//    brace yields the pinned order deterministically. Re-marshaling a map or
	//    a new struct is NOT used: encoding/json sorts map keys alphabetically
	//    (added,next_id,tasks — wrong order), so we splice onto the existing
	//    canonical bytes. Other actions return the bare TaskList unchanged, and
	//    the PERSISTED blob (step 4) never carries `added`.
	bare, err := json.Marshal(list)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("could not serialize task list: %s", err), true).ToToolOutput()
	}
	content := string(bare)
	if added != nil {
		// `bare` is the canonical form starting with `{"tasks":`; insert
		// `"added":N,` after the leading `{` (index 1). encoding/json never
		// emits leading whitespace, so byte 0 is always `{`.
		content = fmt.Sprintf("{\"added\":%d,%s", *added, content[1:])
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: content}
}

// validateTaskListParams enforces the per-action field requirements that the
// internally-tagged Rust enum gets for free from serde. An unknown action or a
// missing required field is an InvalidParameters error.
func validateTaskListParams(p *TaskListParams) *ToolExecutionError {
	switch p.Action {
	case TaskListActionAddTask:
		if p.Description == nil {
			return InvalidParameters("add_task requires `description`")
		}
	case TaskListActionUpdateTask:
		if p.ID == nil {
			return InvalidParameters("update_task requires `id`")
		}
	case TaskListActionCompleteTask:
		if p.ID == nil {
			return InvalidParameters("complete_task requires `id`")
		}
	case TaskListActionListTasks:
		// No required fields.
	case "":
		return InvalidParameters("missing required field `action`")
	default:
		return InvalidParameters(fmt.Sprintf("unknown action %q", p.Action))
	}
	return nil
}

// Compile-time interface check.
var _ sporecore.Tool = (*TaskListTool)(nil)
