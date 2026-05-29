/**
 * SendMessage tool (#81, net-new Tier-1 tool).
 *
 * `send_message` surfaces an out-of-band message to the user. The TOOL itself
 * is trivial: it echoes the `content` back as a success {@link ToolOutput}. The
 * harness loop is what gives it meaning — it recognizes the `send_message` tool
 * name, emits a `user_message` {@link "@spore/core".HarnessStreamEvent} with the
 * content, and records a minimal success tool result so the loop continues. The
 * tool does NOT touch the sandbox or storage.
 */

import type { SandboxProvider, ToolCall, ToolOutput } from "@spore/core";
import type { toolRegistry } from "@spore/core";
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolSchema = toolRegistry.ToolSchema;

import { toolExecutionErrorToOutput } from "./errors.js";
import { parseParams, SendMessageParamsSchema } from "./params.js";

export class SendMessageTool implements Tool {
  static readonly NAME = "send_message";
  readonly name = SendMessageTool.NAME;

  static schema(): ToolSchema {
    return {
      name: SendMessageTool.NAME,
      description: "Send a message to the user",
      parameters: {
        type: "object",
        properties: { content: { type: "string" } },
        required: ["content"],
      },
      annotations: {
        read_only: true,
        destructive: false,
        idempotent: false,
        open_world: false,
      },
    };
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const p = parseParams(SendMessageParamsSchema, call);
    if (!p.ok) return toolExecutionErrorToOutput(p.error);
    // Content returned verbatim; the harness loop reads it off this success and
    // emits a `user_message` stream event.
    return { kind: "success", content: p.value.content, truncated: false };
  }
}
