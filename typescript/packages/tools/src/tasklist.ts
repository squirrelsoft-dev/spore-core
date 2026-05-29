/**
 * TaskList tool (#71): the single mutating tool over the persisted task list.
 *
 * One tool, {@link TaskListTool} (`NAME = "task_list"`), dispatched on an
 * `action` discriminator (`add_task`, `update_task`, `complete_task`,
 * `list_tasks`). The types, transition matrix, and disk-persistence helpers it
 * drives live in `@spore/core`'s `tasklist` module.
 *
 * The tool is read-modify-write over the on-disk
 * {@link "@spore/core".tasklist.TASK_LIST_PATH}:
 * 1. parse params (bad input → recoverable error),
 * 2. load the current list (absent → default),
 * 3. apply the action (domain errors → recoverable),
 * 4. persist the (possibly mutated) list,
 * 5. return the serialized current list as success content.
 *
 * CRITICAL: this tool is NOT annotated `read_only`. `read_only` tools may be run
 * concurrently by `dispatchAll`, and a concurrent read-modify-write over the
 * same file would race. Leaving `read_only` false makes the registry dispatch it
 * sequentially. `destructive` / `open_world` are also left false so it is not
 * treated as an irreversible side effect.
 */

import {
  tasklist,
  type SandboxProvider,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import { parseParams, TaskListParamsSchema } from "./params.js";

const {
  loadTaskList,
  storeTaskList,
  serializeTaskList,
  addTask,
  updateTask,
  completeTask,
} = tasklist;

export class TaskListTool implements Tool {
  static readonly NAME = "task_list";
  readonly name = TaskListTool.NAME;

  static schema(): ToolSchema {
    // Fields kept sorted/stable for cache stability: `action` (required) plus
    // the union of per-action fields.
    return {
      name: TaskListTool.NAME,
      description:
        "Manage the persisted task list: add, update, complete, or list tasks",
      parameters: {
        type: "object",
        properties: {
          action: {
            type: "string",
            enum: ["add_task", "complete_task", "list_tasks", "update_task"],
          },
          description: { type: "string" },
          id: { type: "integer" },
          status: {
            type: "string",
            enum: ["blocked", "completed", "in_progress", "pending"],
          },
        },
        required: ["action"],
      },
      // Intentionally NOT read_only: this tool mutates shared on-disk state and
      // must dispatch sequentially. See module docs.
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
      },
    };
  }

  async execute(call: ToolCall, sandbox: SandboxProvider): Promise<ToolOutput> {
    // 1. Parse params (bad input → recoverable).
    const p = parseParams(TaskListParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const params = p.value;

    // 2. Load current list (absent → default).
    const loaded = await loadTaskList(sandbox);
    if (!loaded.ok) {
      switch (loaded.error.kind) {
        case "sandbox":
          return toolExecutionErrorToOutput({
            kind: "sandbox_violation",
            violation: loaded.error.violation,
          });
        case "parse":
          return {
            kind: "error",
            message: `could not parse task list: ${loaded.error.reason}`,
            recoverable: true,
          };
      }
    }
    const list = loaded.list;

    // 3. Apply the action. Domain errors → recoverable. `list_tasks` does not
    //    mutate.
    let mutated = false;
    switch (params.action) {
      case "add_task":
        addTask(list, params.description);
        mutated = true;
        break;
      case "update_task": {
        const r = updateTask(
          list,
          params.id,
          params.status,
          params.description,
        );
        if (!r.ok) {
          return { kind: "error", message: r.error.message, recoverable: true };
        }
        mutated = true;
        break;
      }
      case "complete_task": {
        const r = completeTask(list, params.id);
        if (!r.ok) {
          return { kind: "error", message: r.error.message, recoverable: true };
        }
        mutated = true;
        break;
      }
      case "list_tasks":
        mutated = false;
        break;
    }

    // 4. Persist the (possibly mutated) list. `list_tasks` skips the write.
    if (mutated) {
      const stored = await storeTaskList(list, sandbox);
      if (!stored.ok) {
        switch (stored.error.kind) {
          case "sandbox":
            return toolExecutionErrorToOutput({
              kind: "sandbox_violation",
              violation: stored.error.violation,
            });
          case "serialize":
            return {
              kind: "error",
              message: `could not serialize task list: ${stored.error.reason}`,
              recoverable: true,
            };
          case "io":
            return {
              kind: "error",
              message: `could not persist task list: ${stored.error.reason}`,
              recoverable: true,
            };
        }
      }
    }

    // 5. Return the serialized current list.
    return {
      kind: "success",
      content: serializeTaskList(list),
      truncated: false,
    };
  }
}
