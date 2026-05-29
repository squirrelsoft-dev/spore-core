/**
 * TodoWrite tool (#81, net-new Tier-2 storage tool).
 *
 * `todo_write` persists an agent-managed todo list via the {@link ToolContext}'s
 * {@link "@spore/core".storage.RunStore} under the key {@link TODO_STORE_KEY}
 * (`"todo"`), keyed by the run's {@link "@spore/core".SessionId}. The agent
 * supplies the FULL desired list on every call; it REPLACES the persisted list
 * wholesale (no per-item diffing). The current list is returned as JSON success
 * content.
 *
 * Like {@link import("./tasklist.js").TaskListTool} it is NOT annotated
 * `read_only`: it mutates shared persisted state and must dispatch sequentially
 * (a concurrent read-modify-write would race).
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { storage, toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import { parseParams, TodoWriteParamsSchema } from "./params.js";

/** RunStore key under which the todo list is persisted (issue #81, Q5). */
export const TODO_STORE_KEY = "todo";

export class TodoWriteTool implements Tool {
  static readonly NAME = "todo_write";
  readonly name = TodoWriteTool.NAME;

  static schema(): ToolSchema {
    return {
      name: TodoWriteTool.NAME,
      description: "Replace the persisted todo list with the supplied full list",
      parameters: {
        type: "object",
        properties: {
          todos: {
            type: "array",
            items: {
              type: "object",
              properties: {
                content: { type: "string" },
                status: {
                  type: "string",
                  enum: ["completed", "in_progress", "pending"],
                },
              },
              required: ["content", "status"],
            },
          },
        },
        required: ["todos"],
      },
      // Intentionally NOT read_only — mutates shared persisted state and must
      // dispatch sequentially. See module docs / TaskListTool.
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
    const p = parseParams(TodoWriteParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    const { todos } = p.value;

    const value = todos as unknown as storage.JsonValue;
    try {
      await ctx.runStore.put(ctx.sessionId, TODO_STORE_KEY, value);
    } catch (e) {
      return {
        kind: "error",
        message: `could not persist todos: ${e instanceof Error ? e.message : String(e)}`,
        recoverable: true,
      };
    }
    return {
      kind: "success",
      content: JSON.stringify(todos),
      truncated: false,
    };
  }
}
