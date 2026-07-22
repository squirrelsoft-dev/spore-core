/**
 * SetPlan tool — the planner authors its whole plan in ONE atomic call.
 *
 * Mirrors `rust/crates/spore-core/src/tools/set_plan.rs`.
 * `set_plan(tasks=[...], rationale="...")` replaces the persisted task list
 * with the given ordered steps. Because the plan arrives as a TOOL CALL —
 * consumed by the harness, never rendered as a message — it cannot leak as a
 * raw `{"tasks":...}` assistant message, and the execute phase (which is not
 * told to author plans and is not offered this tool) won't emit one either.
 *
 * It writes the SAME RunStore blob the `task_list` tool authors (durable,
 * project-scoped, under `TASK_LIST_EXTRAS_KEY`), so the plan-capture fallback
 * (`loadTaskList`) and the execute-phase runnable list read it unchanged. Like
 * `task_list` it is NOT `read_only` — it mutates that shared blob and must
 * dispatch sequentially.
 *
 * NOTE: the Rust `SetPlanTool` returns `ToolOutput::success` — it does NOT
 * escalate `PlanRecorded`. The `plan_recorded` {@link HarnessSignal} variant is
 * a forward-compatible seam handled in the PlanExecute plan phase but not yet
 * emitted by the ReAct loop; the plan is captured via the durable `task_list`
 * store fallback (SC-28) instead.
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import { tasklist } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

const { TASK_LIST_EXTRAS_KEY, defaultTaskList, serializeTaskList } = tasklist;

export class SetPlanTool implements Tool {
  static readonly NAME = "set_plan";
  readonly name = SetPlanTool.NAME;

  static schema(): ToolSchema {
    return {
      name: SetPlanTool.NAME,
      description:
        "Record your plan in ONE call: `tasks` = ordered step strings. " +
        "Use this instead of writing the plan as text.",
      parameters: {
        type: "object",
        properties: {
          tasks: {
            type: "array",
            items: { type: "string" },
            description: "Ordered, concrete step strings.",
          },
          rationale: {
            type: "string",
            description: "Optional one-line reason for the plan.",
          },
        },
        required: ["tasks"],
      },
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
      },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    ctx: ToolContext,
  ): Promise<ToolOutput> {
    // Parse `tasks`: a non-empty array of non-empty strings (bad → recoverable).
    const input = call.input as Record<string, unknown>;
    const tasksArr = input.tasks;
    if (!Array.isArray(tasksArr) || tasksArr.length === 0) {
      return {
        kind: "error",
        message: "`tasks` must be a non-empty array of step strings",
        recoverable: true,
      };
    }
    const tasks: string[] = [];
    for (let i = 0; i < tasksArr.length; i += 1) {
      const el = tasksArr[i];
      if (typeof el !== "string" || el.trim() === "") {
        return {
          kind: "error",
          message: `task ${i} must be a non-empty string`,
          recoverable: true,
        };
      }
      tasks.push(el);
    }

    // Build a linear TaskList (no blockers) from the ordered steps.
    const list = defaultTaskList();
    for (const desc of tasks) {
      const res = tasklist.addTask(list, desc, []);
      if (!res.ok) {
        return {
          kind: "error",
          message: `could not record task ${JSON.stringify(desc)}: ${tasklist.taskListErrorMessage(
            res.error,
          )}`,
          recoverable: true,
        };
      }
    }

    // Persist under the SAME durable, project-scoped key `task_list` uses (#142).
    const durableNs = ctx.projectId.namespace();
    let value: unknown;
    try {
      value = JSON.parse(serializeTaskList(list));
    } catch (e) {
      return {
        kind: "error",
        message: `could not serialize plan: ${errMessage(e)}`,
        recoverable: true,
      };
    }
    try {
      await ctx.runStore.put(durableNs, TASK_LIST_EXTRAS_KEY, value as never);
    } catch (e) {
      return {
        kind: "error",
        message: `could not persist plan: ${errMessage(e)}`,
        recoverable: true,
      };
    }

    return {
      kind: "success",
      content: `Recorded a ${tasks.length}-step plan.`,
      truncated: false,
    };
  }
}

function errMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}