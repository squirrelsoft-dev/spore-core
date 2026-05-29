/**
 * TaskList tool (#71, storage seam #75): the single mutating tool over the
 * persisted task list.
 *
 * One tool, {@link TaskListTool} (`NAME = "task_list"`), dispatched on an
 * `action` discriminator (`add_task`, `update_task`, `complete_task`,
 * `list_tasks`). The types, transition matrix, and mutation helpers it drives
 * live in `@spore/core`'s `tasklist` module.
 *
 * ## Storage seam (#75)
 *
 * The tool persists via the {@link "@spore/core".toolRegistry.ToolContext}'s
 * {@link "@spore/core".storage.RunStore} — NOT the sandbox filesystem. It is
 * read-modify-write keyed by the run's
 * {@link "@spore/core".SessionId} under
 * {@link "@spore/core".tasklist.TASK_LIST_EXTRAS_KEY} (`"task_list"`):
 * 1. parse params (bad input → recoverable error),
 * 2. `ctx.runStore.get(sessionId, "task_list")` (absent → default empty list),
 * 3. apply the action (domain errors → recoverable),
 * 4. on a mutating action, `ctx.runStore.put(sessionId, "task_list", value)`,
 * 5. return the serialized current list as success content.
 *
 * ### Shared key
 * This standalone tool and the harness-side PlanExecute execute loop persist
 * under the SAME `RunStore` key (`"task_list"`), keyed by `SessionId`. A
 * standalone tool call and a PlanExecute run on the same session intentionally
 * share one blob. The JSON shape is the canonical serialized {@link TaskList}
 * (`{"tasks":[...],"next_id":N}`), unchanged.
 *
 * ### Behavior change vs the retired sandbox path
 * Previously the tool persisted to `.spore/task_list.json` via the sandbox.
 * That path is GONE. With the library's default storage
 * ({@link "@spore/core".storage.NoOpStorageProvider}) a standalone tool call
 * persists NOTHING across processes — the no-op run store silently discards
 * writes and returns `undefined` on read. This is an accepted behavior change:
 * durable cross-process persistence now requires configuring a real
 * `StorageProvider` (e.g. `FileSystemStorageProvider`). There is NO migration
 * shim for old on-disk `.spore/task_list.json` files.
 *
 * ## Storage-error mapping
 * A storage failure from a get/put maps to a recoverable error
 * {@link ToolOutput}. A present-but-malformed blob (parse failure) is likewise
 * recoverable. `list_tasks` never writes.
 *
 * CRITICAL: this tool is NOT annotated `read_only`. `read_only` tools may be run
 * concurrently by `dispatchAll`, and a concurrent read-modify-write over the
 * same key would race. Leaving `read_only` false makes the registry dispatch it
 * sequentially. `destructive` / `open_world` are also left false so it is not
 * treated as an irreversible side effect.
 */

import {
  tasklist,
  type SandboxProvider,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";
import type { storage, toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import { parseParams, TaskListParamsSchema } from "./params.js";

const {
  TASK_LIST_EXTRAS_KEY,
  defaultTaskList,
  parseTaskList,
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
      // Intentionally NOT read_only: this tool mutates shared persisted state
      // and must dispatch sequentially. See module docs.
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
    const { sessionId, runStore } = ctx;

    // 1. Parse params (bad input → recoverable).
    const p = parseParams(TaskListParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const params = p.value;

    // 2. Load current list from the run store (absent → default). A storage
    //    error or a malformed blob is recoverable.
    let raw: storage.JsonValue | undefined;
    try {
      raw = await runStore.get(sessionId, TASK_LIST_EXTRAS_KEY);
    } catch (e) {
      return {
        kind: "error",
        message: `could not load task list: ${errMessage(e)}`,
        recoverable: true,
      };
    }
    let list: tasklist.TaskList;
    if (raw === undefined) {
      list = defaultTaskList();
    } else {
      try {
        list = parseTaskList(JSON.stringify(raw));
      } catch (e) {
        return {
          kind: "error",
          message: `could not parse task list: ${errMessage(e)}`,
          recoverable: true,
        };
      }
    }

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

    // 4. Persist the (possibly mutated) list to the run store, keyed by
    //    SessionId under the shared TASK_LIST_EXTRAS_KEY. We always persist on a
    //    mutating action; list_tasks skips the write.
    if (mutated) {
      let value: storage.JsonValue;
      try {
        value = JSON.parse(serializeTaskList(list)) as storage.JsonValue;
      } catch (e) {
        return {
          kind: "error",
          message: `could not serialize task list: ${errMessage(e)}`,
          recoverable: true,
        };
      }
      try {
        await runStore.put(sessionId, TASK_LIST_EXTRAS_KEY, value);
      } catch (e) {
        return {
          kind: "error",
          message: `could not persist task list: ${errMessage(e)}`,
          recoverable: true,
        };
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

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
