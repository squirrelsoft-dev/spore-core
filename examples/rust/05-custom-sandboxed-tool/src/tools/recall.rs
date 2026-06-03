//! `recall(key)` — read a previously-remembered fact back out of the run store.
//!
//! This is the **read** half of the pair. Unlike [`remember`](super::remember)
//! it is annotated `read_only` + `idempotent`: it only reads shared state, so
//! the harness may dispatch it concurrently with other read-only tools.
//!
//! Looking up a key that was never stored is a *recoverable* error — the agent
//! can adapt (try a different key, or remember the fact first) rather than
//! halting the run.

use serde_json::json;

use spore_core::harness::BoxFut;
use spore_core::{
    RegisteredToolSchema, SandboxProvider, Tool, ToolAnnotations, ToolCall, ToolContext, ToolOutput,
};

use super::remember::FACT_PREFIX;

pub struct RecallTool;

impl RecallTool {
    pub const NAME: &'static str = "recall";

    pub fn new() -> Self {
        Self
    }

    /// The registry-side schema. `name` MUST equal [`Tool::name`].
    pub fn schema() -> RegisteredToolSchema {
        RegisteredToolSchema {
            name: Self::NAME.into(),
            description: "Recall a fact previously stored with `remember`, by its key.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            }),
            // Pure read of shared state: safe to mark read_only + idempotent.
            annotations: ToolAnnotations {
                read_only: true,
                idempotent: true,
                ..Default::default()
            },
        }
    }
}

impl Default for RecallTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for RecallTool {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        _sandbox: &'a (dyn SandboxProvider + 'a),
        ctx: &'a ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let Some(key) = call.input.get("key").and_then(|v| v.as_str()) else {
                return ToolOutput::error("recall: missing or non-string 'key'");
            };

            let store_key = format!("{FACT_PREFIX}{key}");
            match ctx.run_store().get(ctx.session_id(), &store_key).await {
                Ok(Some(value)) => ToolOutput::success(value_to_string(&value)),
                Ok(None) => ToolOutput::error(format!("no fact stored under '{key}'")),
                Err(e) => ToolOutput::error(format!("recall: could not read '{key}': {e}")),
            }
        })
    }
}

/// `remember` always stores a JSON string, so render that back as plain text.
/// Fall back to the JSON encoding for anything unexpected.
fn value_to_string(value: &serde_json::Value) -> String {
    match value.as_str() {
        Some(s) => s.to_string(),
        None => value.to_string(),
    }
}
