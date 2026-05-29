//! SendMessage tool (#81, net-new Tier-1 tool).
//!
//! `send_message` surfaces an out-of-band message to the user. The TOOL itself
//! is trivial: it echoes the `content` back as a [`ToolOutput::Success`]. The
//! harness loop is what gives it meaning — it recognizes the `send_message`
//! tool name, emits a [`StreamEvent::UserMessage`](crate::harness::StreamEvent)
//! with the content, and records a minimal success tool result so the loop
//! continues (see the harness `run_react` loop). The tool does NOT touch the
//! sandbox or storage.

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::params::{parse_params, SendMessageParams};

pub struct SendMessageTool;

impl SendMessageTool {
    pub const NAME: &'static str = "send_message";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Send a message to the user".into(),
            parameters: json!({
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for SendMessageTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: SendMessageParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            // The content is returned verbatim; the harness loop reads it off
            // this Success and emits a StreamEvent::UserMessage.
            ToolOutput::Success {
                content: params.content,
                truncated: false,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: SendMessageTool::NAME.into(),
            input,
        }
    }

    #[tokio::test]
    async fn send_message_echoes_content() {
        let sb = AllowAllSandbox;
        let r = SendMessageTool::new()
            .execute(&call(json!({"content": "hi user"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "hi user"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn send_message_bad_params_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = SendMessageTool::new()
            .execute(&call(json!({})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
