// TodoWrite tool (#81, net-new Tier-2 storage tool).
//
// todo_write persists an agent-managed todo list via the *ToolContext's
// RunStore under the key TodoStoreKey ("todo"), keyed by the run's SessionID.
// The agent supplies the FULL desired list on every call; it REPLACES the
// persisted list wholesale (no per-item diffing). The current list is returned
// as JSON success content.
//
// Like TaskListTool it is NOT annotated ReadOnly: it mutates shared persisted
// state and must dispatch sequentially (a concurrent read-modify-write would
// race).

package tools

import (
	"context"
	"encoding/json"
	"fmt"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// TodoStoreKey is the RunStore key under which the todo list is persisted
// (issue #81, Q5). Mirrors Rust's TODO_STORE_KEY.
const TodoStoreKey = "todo"

// TodoWriteToolName is the registered tool name.
const TodoWriteToolName = "todo_write"

// TodoWriteTool replaces the persisted todo list wholesale.
type TodoWriteTool struct{}

// NewTodoWriteTool constructs a TodoWriteTool.
func NewTodoWriteTool() *TodoWriteTool { return &TodoWriteTool{} }

func (*TodoWriteTool) Name() string                { return TodoWriteToolName }
func (*TodoWriteTool) IsSubagentTool() bool        { return false }
func (*TodoWriteTool) MayProduceLargeOutput() bool { return false }

func (*TodoWriteTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        TodoWriteToolName,
		Description: "Replace the persisted todo list with the supplied full list",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"todos": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {
							"content": {"type": "string"},
							"status": {
								"type": "string",
								"enum": ["completed", "in_progress", "pending"]
							}
						},
						"required": ["content", "status"]
					}
				}
			},
			"required": ["todos"]
		}`),
		// Intentionally NOT ReadOnly — mutates shared persisted state and must
		// dispatch sequentially. See module docs / TaskListTool.
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *TodoWriteTool) Execute(ctx context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	var params TodoWriteParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	// Normalize nil to an empty slice so an empty list persists as [] and the
	// returned content is "[]" (not "null"), matching the fixture.
	if params.Todos == nil {
		params.Todos = []TodoItem{}
	}
	value, err := json.Marshal(params.Todos)
	if err != nil {
		return ExecutionFailed(fmt.Sprintf("could not serialize todos: %s", err), true).ToToolOutput()
	}
	if err := toolCtx.Put(ctx, TodoStoreKey, value); err != nil {
		return ExecutionFailed(fmt.Sprintf("could not persist todos: %s", err), true).ToToolOutput()
	}
	return sporecore.ToolOutput{Kind: sporecore.ToolOutputSuccess, Content: string(value)}
}

var _ sporecore.Tool = (*TodoWriteTool)(nil)
