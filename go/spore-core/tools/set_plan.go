// SetPlan tool — the planner authors its whole plan in ONE atomic call.
//
// `set_plan(tasks=[...], rationale="...")` replaces the persisted task list with
// the given ordered steps. Because the plan arrives as a TOOL CALL — consumed by
// the harness, never rendered as a message — it cannot leak as a raw
// `{"tasks":...}` assistant message, and the execute phase (which is not told to
// author plans and is not offered this tool) won't emit one either.
//
// It writes the SAME RunStore blob the `task_list` tool authors (durable,
// project-scoped, under TaskListExtrasKey), so the existing plan-capture
// fallback (load_task_list) and the execute-phase runnable list read it
// unchanged, and the plan-start clear_task_list clears any stale plan for free.
// Like `task_list` it is NOT ReadOnly — it mutates that shared blob and must
// dispatch sequentially.
//
// Mirrors the Rust reference (tools/set_plan.rs). Its terminal signal is
// HarnessSignal::PlanRecorded (the plan-phase analogue of plan_completed),
// consumed inside PlanExecute — see the parent package's plan-execute phase.

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sporecore "github.com/squirrelsoft-dev/spore-core/go/spore-core"
)

// SetPlanToolName is the registered tool name.
const SetPlanToolName = "set_plan"

// SetPlanParams is the parameter shape for set_plan.
type SetPlanParams struct {
	Tasks     []string `json:"tasks,omitempty"`
	Rationale string   `json:"rationale,omitempty"`
}

// SetPlanTool records an ordered plan in one atomic call.
type SetPlanTool struct{}

// NewSetPlanTool constructs a SetPlanTool.
func NewSetPlanTool() *SetPlanTool { return &SetPlanTool{} }

func (*SetPlanTool) Name() string                { return SetPlanToolName }
func (*SetPlanTool) IsSubagentTool() bool        { return false }
func (*SetPlanTool) MayProduceLargeOutput() bool { return false }

func (*SetPlanTool) Schema() sporecore.RegistryToolSchema {
	return sporecore.RegistryToolSchema{
		Name:        SetPlanToolName,
		Description: "Record your plan in ONE call: `tasks` = ordered step strings. Use this instead of writing the plan as text.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"tasks": {
					"type": "array",
					"items": {"type": "string"},
					"description": "Ordered, concrete step strings."
				},
				"rationale": {
					"type": "string",
					"description": "Optional one-line reason for the plan."
				}
			},
			"required": ["tasks"]
		}`),
		// Intentionally NOT ReadOnly: this tool mutates the shared durable blob
		// and must dispatch sequentially (a concurrent read-modify-write would
		// race). Mirrors task_list.
		Annotations: sporecore.ToolAnnotations{},
	}
}

func (t *SetPlanTool) Execute(ctx context.Context, call sporecore.ToolCall, _ sporecore.SandboxProvider, toolCtx *sporecore.ToolContext) sporecore.ToolOutput {
	// 1. Parse params (bad input → recoverable).
	var params SetPlanParams
	if e := parseParams(call, &params); e != nil {
		return e.ToToolOutput()
	}
	if len(params.Tasks) == 0 {
		return sporecore.NewToolOutputError("`tasks` must be a non-empty array of step strings")
	}
	tasks := make([]string, 0, len(params.Tasks))
	for i, desc := range params.Tasks {
		if strings.TrimSpace(desc) == "" {
			return sporecore.NewToolOutputError(fmt.Sprintf("task %d must be a non-empty string", i))
		}
		tasks = append(tasks, desc)
	}

	// 2. Build a linear TaskList (no blockers) from the ordered steps, starting
	//    from the canonical empty list (next_id == 1) — matches Rust's
	//    TaskList::default().
	list := sporecore.DefaultTaskList()
	for _, desc := range tasks {
		if _, err := list.Add(desc, nil); err != nil {
			return sporecore.NewToolOutputError(fmt.Sprintf("could not record task %q: %s", desc, err.Error()))
		}
	}

	// 3. Persist under the SAME durable, project-scoped key `task_list` uses
	//    (#142), so the plan-capture fallback and execute phase read it.
	value, err := json.Marshal(list)
	if err != nil {
		return sporecore.NewToolOutputError(fmt.Sprintf("could not serialize plan: %s", err.Error()))
	}
	if err := toolCtx.PutDurable(ctx, sporecore.TaskListExtrasKey, value); err != nil {
		return sporecore.NewToolOutputError(fmt.Sprintf("could not persist plan: %s", err.Error()))
	}

	return sporecore.NewToolOutputSuccess(fmt.Sprintf("Recorded a %d-step plan.", len(tasks)))
}
