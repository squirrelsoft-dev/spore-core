//! `remember(key, value)` — persist a fact into the run store.
//!
//! This is the **write** half of the custom-tool pair. It demonstrates the
//! storage seam: [`ToolContext::run_store`] + [`ToolContext::session_id`] are the
//! only path to durable, per-run state. The `sandbox` parameter is part of the
//! `Tool::execute` signature but unused here — these tools never touch the
//! filesystem, so they ignore it (named `_sandbox`).
//!
//! Keys are namespaced under `fact:{key}` so the example cannot collide with
//! reserved store keys the catalogue uses (`todo`, `task`, `memory`).

use serde_json::json;

use spore_core::harness::BoxFut;
use spore_core::{
    RegisteredToolSchema, SandboxProvider, Tool, ToolAnnotations, ToolCall, ToolContext, ToolOutput,
};

/// Prefix applied to every key so this example's facts live in their own
/// namespace inside the run store.
pub const FACT_PREFIX: &str = "fact:";

pub struct RememberTool;

impl RememberTool {
    pub const NAME: &'static str = "remember";

    pub fn new() -> Self {
        Self
    }

    /// The registry-side schema. `name` MUST equal [`Tool::name`].
    pub fn schema() -> RegisteredToolSchema {
        RegisteredToolSchema {
            name: Self::NAME.into(),
            description: "Store a fact under a short key so it can be recalled later. \
                          Use a stable, memorable key (e.g. 'habitat', 'lifespan')."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            }),
            // Intentionally NOT read_only: this mutates shared persisted state.
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for RememberTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for RememberTool {
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
                return ToolOutput::error("remember: missing or non-string 'key'");
            };
            let Some(value) = call.input.get("value").and_then(|v| v.as_str()) else {
                return ToolOutput::error("remember: missing or non-string 'value'");
            };

            let store_key = format!("{FACT_PREFIX}{key}");
            match ctx
                .run_store()
                .put(ctx.session_id(), &store_key, json!(value))
                .await
            {
                Ok(()) => ToolOutput::success(format!("remembered {key}")),
                Err(e) => ToolOutput::error(format!("remember: could not persist '{key}': {e}")),
            }
        })
    }
}
