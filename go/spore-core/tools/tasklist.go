// TaskList tool (#71): the single mutating tool over the persisted task list.
//
// One tool, TaskListTool (NAME = "task_list"), dispatched on an `action`
// discriminator (add_task, update_task, complete_task, list_tasks). See the
// parent package's tasklist.go for the types, the transition matrix, and the
// disk-persistence helpers this tool drives.
//
// The tool is read-modify-write over the on-disk TaskListPath:
//  1. parse params (bad input → recoverable error),
//  2. load the current list (absent → default),
//  3. apply the action (domain errors → recoverable),
//  4. persist the (possibly mutated) list,
//  5. return the serialized current list as success content.
//
// CRITICAL: this tool is NOT annotated ReadOnly. Read-only tools are run
// CONCURRENTLY by DispatchAll, and a concurrent read-modify-write over the same
// file would race. Leaving ReadOnly false makes the registry dispatch it
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

func (t *TaskListTool) Execute(ctx context.Context, call sporecore.ToolCall, sandbox sporecore.SandboxProvider) sporecore.ToolOutput {
	// 1. Parse params (bad input → recoverable).
	var params TaskListParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	if e := validateTaskListParams(&params); e != nil {
		return e.ToToolOutput()
	}

	// 2. Load current list (absent → default).
	list, v, err := sporecore.LoadTaskList(ctx, sandbox)
	if v != nil {
		return SandboxViolationError(v).ToToolOutput()
	}
	if err != nil {
		return ExecutionFailed(err.Error(), true).ToToolOutput()
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

	// 4. Persist the (possibly mutated) list. list_tasks skips the write.
	if mutated {
		if v, err := sporecore.StoreTaskList(ctx, list, sandbox); v != nil {
			return SandboxViolationError(v).ToToolOutput()
		} else if err != nil {
			return ExecutionFailed(err.Error(), true).ToToolOutput()
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
