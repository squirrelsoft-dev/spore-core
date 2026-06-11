/**
 * `send_user_message(message)` — let an agent narrate its plan to the human.
 *
 * Pure side-effecting tool: it prints the message to stdout (prefixed with a
 * per-agent marker emoji so you can tell who is speaking) and returns a short
 * confirmation so the model keeps going. It captures the marker, so it is a
 * hand-written {@link Tool} rather than a {@link toolRegistry.defineTool} tool
 * (defineTool produces a stateless closure; the marker is closed-over state —
 * same reason `load_skill` was hand-written before it was dropped).
 */

import {
  toolOutput,
  toolRegistry,
  type SandboxProvider,
  type ToolCall,
  type ToolOutput,
} from "@spore/core";

type StandardTool = toolRegistry.StandardTool;
type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;

/** The registered name of the tool. */
export const NAME = "send_user_message";

/** `send_user_message`, holding the marker emoji that prefixes its output. */
class SendUserMessageTool implements Tool {
  readonly name = NAME;

  constructor(private readonly marker: string) {}

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    const raw = (call.input as Record<string, unknown>)?.message;
    const message = typeof raw === "string" ? raw.trim() : "";
    if (message === "") {
      return toolOutput.error(
        "invalid parameters: `message` (non-empty string) is required",
      );
    }
    // Leading newline to break from the stream banners, the marker, then a
    // trailing blank line to give the message room.
    process.stdout.write(`\n${this.marker} ${message}\n\n`);
    return toolOutput.success("Message shown to the user.");
  }
}

/**
 * Build a `send_user_message` {@link toolRegistry.StandardTool} that prefixes its
 * output with `marker` (e.g. "🤖" for the worker).
 */
export function sendUserMessageTool(marker: string): StandardTool {
  return {
    implementation: new SendUserMessageTool(marker),
    schema: {
      name: NAME,
      description:
        "Tell the watching human what you are about to do and why, in one short " +
        "sentence, BEFORE you act. Call this at the start of each step so your plan " +
        "is visible. Pass a single `message` string. This does not pause the run.",
      parameters: {
        type: "object",
        properties: {
          message: {
            type: "string",
            description:
              "What you are about to do and why, in one short sentence.",
          },
        },
        required: ["message"],
      },
      // Side-effecting (prints), but harmless and never destructive — defaults.
      annotations: {
        read_only: false,
        destructive: false,
        idempotent: false,
        open_world: false,
      },
    },
  };
}
