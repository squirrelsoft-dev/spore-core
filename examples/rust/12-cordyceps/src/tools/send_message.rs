//! `send_user_message(message)` — let an agent narrate its plan to the human.
//!
//! Pure side-effecting tool: it prints the message to stdout (prefixed with a
//! per-agent marker emoji so you can tell who is speaking — 🍄 orchestrator,
//! 🤖 subagent) and returns a short confirmation so the model keeps going. It
//! captures the marker, so it is a hand-written [`Tool`] rather than a
//! `tool!`-macro tool (the macro produces a zero-sized struct that cannot
//! close over state — same reason as `load_skill`).

use std::io::Write;

use serde_json::json;

use spore_core::harness::{BoxFut, SandboxProvider, ToolOutput};
use spore_core::tool_registry::{Tool, ToolContext};
use spore_core::{RegisteredToolSchema, StandardTool, ToolAnnotations};

/// The registered name of the tool.
pub const NAME: &str = "send_user_message";

/// `send_user_message`, holding the marker emoji that prefixes its output.
struct SendUserMessageTool {
    marker: String,
}

impl Tool for SendUserMessageTool {
    fn name(&self) -> &str {
        NAME
    }

    fn execute<'a>(
        &'a self,
        call: &'a spore_core::ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let message = match call.input.get("message").and_then(|v| v.as_str()) {
                Some(s) if !s.trim().is_empty() => s.trim(),
                _ => {
                    return ToolOutput::error(
                        "invalid parameters: `message` (non-empty string) is required",
                    );
                }
            };
            // Leading newline to break from the stream banners, the marker, then
            // a trailing blank line to give the message room.
            print!("\n{} {}\n\n", self.marker, message);
            let _ = std::io::stdout().flush();
            ToolOutput::success("Message shown to the user.")
        })
    }
}

/// Build a `send_user_message` [`StandardTool`] that prefixes its output with
/// `marker` (e.g. "🍄" for the orchestrator, "🤖" for a subagent).
pub fn send_user_message_tool(marker: impl Into<String>) -> StandardTool {
    StandardTool::new(
        Box::new(SendUserMessageTool {
            marker: marker.into(),
        }),
        RegisteredToolSchema {
            name: NAME.into(),
            description: "Tell the watching human what you are about to do and why, in one short \
                          sentence, BEFORE you act. Call this at the start of each step so your \
                          plan is visible. Pass a single `message` string. This does not pause \
                          the run."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "What you are about to do and why, in one short sentence."
                    }
                },
                "required": ["message"]
            }),
            annotations: ToolAnnotations {
                // Side-effecting (prints), but harmless and never destructive.
                ..Default::default()
            },
        },
    )
}
