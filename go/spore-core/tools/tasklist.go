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
// filesystem. It is read-modify-write keyed by the run's SessionID under
// TaskListExtrasKey ("task_list"):
//  1. parse params (bad input → recoverable error),
//  2. toolCtx.Get(ctx, "task_list") (absent → DefaultTaskList),
//  3. apply the action (domain errors → recoverable),
//  4. on a mutating action, toolCtx.Put(ctx, "task_list", value),
//  5. return the serialized current list as success content.
//
// Shared key: this standalone tool and the harness-side PlanExecute execute loop
// (#76) persist under the SAME RunStore key ("task_list"), keyed by SessionID. A
// standalone tool call and a PlanExecute run on the same session intentionally
// share one blob. The JSON shape is the canonical TaskList serialization
// ({"tasks":[...],"next_id":N}), unchanged.
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

	// 2. Load current list from the run store, keyed by SessionID under the
	//    shared TaskListExtrasKey (absent → default). A storage error or a
	//    malformed blob is recoverable.
	list := sporecore.DefaultTaskList()
	value, found, err := toolCtx.Get(ctx, sporecore.TaskListExtrasKey)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("could not load task list: %s", err), true).ToToolOutput()
	}
	if found {
		if err := json.Unmarshal(value, &list); err != nil {
			return ExecutionFailed(fmt.Sprintf("could not parse task list: %s", err), true).ToToolOutput()
		}
	}

	// 3. Apply the action. Domain errors → recoverable. list_tasks does not mutate.
	mutated := false
	switch params.Action {
	case TaskListActionAddTask:
		list.Add(*params.Description)
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
	//    shared TaskListExtrasKey. list_tasks skips the write.
	if mutated {
		encoded, err := json.Marshal(list)
		if err != nil {
			return ExecutionFailed(fmt.Sprintf("could not serialize task list: %s", err), true).ToToolOutput()
		}
		if err := toolCtx.Put(ctx, sporecore.TaskListExtrasKey, encoded); err != nil {
			return ExecutionFailed(fmt.Sprintf("could not persist task list: %s", err), true).ToToolOutput()
		}
	}

	// 5. Return the serialized current list.
	content, err := json.Marshal(list)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("could not serialize task list: %s", err), true).ToToolOutput()
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: string(content)}
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
