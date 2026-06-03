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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use serde_json::Value as JsonValue;
    use spore_core::storage::{InMemoryStorageProvider, RunStore, StorageError};
    use spore_core::{SandboxProvider, SandboxViolation, SessionId};

    use crate::tools::recall::RecallTool;

    const SESSION: &str = "fact-session";

    /// These tools never touch the filesystem — a permissive sandbox is plenty.
    struct AllowAllSandbox;
    impl SandboxProvider for AllowAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
    }

    /// A RunStore whose every operation fails — proves storage errors map to a
    /// recoverable tool error (mirrors the TS/Go `FailingRunStore`).
    struct FailingRunStore;
    impl RunStore for FailingRunStore {
        fn get<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
        ) -> BoxFut<'a, Result<Option<JsonValue>, StorageError>> {
            Box::pin(async move {
                Err(StorageError::Backend {
                    message: "boom".into(),
                })
            })
        }
        fn put<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
            _value: JsonValue,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move {
                Err(StorageError::Backend {
                    message: "boom".into(),
                })
            })
        }
        fn delete<'a>(
            &'a self,
            _session_id: &'a SessionId,
            _key: &'a str,
        ) -> BoxFut<'a, Result<(), StorageError>> {
            Box::pin(async move { Ok(()) })
        }
        fn list_keys<'a>(
            &'a self,
            _session_id: &'a SessionId,
        ) -> BoxFut<'a, Result<Vec<String>, StorageError>> {
            Box::pin(async move { Ok(Vec::new()) })
        }
    }

    fn ctx_with(run_store: Arc<dyn RunStore>) -> ToolContext {
        // remember/recall exercise the run-store seam only; an in-memory backend
        // covers the (unused) memory seam.
        ToolContext::new(
            SessionId::new(SESSION),
            run_store,
            Arc::new(InMemoryStorageProvider::new()),
        )
    }

    fn in_memory_ctx() -> ToolContext {
        ctx_with(Arc::new(InMemoryStorageProvider::new()))
    }

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: RememberTool::NAME.into(),
            input,
        }
    }

    async fn execute(tool: &RememberTool, call: ToolCall, ctx: &ToolContext) -> ToolOutput {
        tool.execute(&call, &AllowAllSandbox, ctx).await
    }

    #[tokio::test]
    async fn stores_under_fact_prefix() {
        let ctx = in_memory_ctx();
        let out = execute(
            &RememberTool::new(),
            call(json!({"key": "habitat", "value": "coastal ocean waters"})),
            &ctx,
        )
        .await;
        match out {
            ToolOutput::Success { content, .. } => assert_eq!(content, "remembered habitat"),
            other => panic!("expected success, got {other:?}"),
        }

        // Persisted under the namespaced key, JSON-encoded as a string …
        let stored = ctx
            .run_store()
            .get(ctx.session_id(), &format!("{FACT_PREFIX}habitat"))
            .await
            .unwrap();
        assert_eq!(stored, Some(json!("coastal ocean waters")));
        // … and NOT under the bare key.
        let bare = ctx
            .run_store()
            .get(ctx.session_id(), "habitat")
            .await
            .unwrap();
        assert_eq!(bare, None);
    }

    #[tokio::test]
    async fn recall_reads_back_remembered_value() {
        let ctx = in_memory_ctx();
        execute(
            &RememberTool::new(),
            call(json!({"key": "diet", "value": "crabs and shrimp"})),
            &ctx,
        )
        .await;
        let recall = RecallTool::new();
        let out = recall
            .execute(
                &ToolCall {
                    id: "c2".into(),
                    name: RecallTool::NAME.into(),
                    input: json!({"key": "diet"}),
                },
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match out {
            ToolOutput::Success { content, .. } => assert_eq!(content, "crabs and shrimp"),
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_value_is_recoverable_error() {
        let out = execute(
            &RememberTool::new(),
            call(json!({"key": "habitat"})),
            &in_memory_ctx(),
        )
        .await;
        match out {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("value"), "message: {message}");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_string_key_is_recoverable_error() {
        let out = execute(
            &RememberTool::new(),
            call(json!({"key": 7, "value": "x"})),
            &in_memory_ctx(),
        )
        .await;
        match out {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("key"), "message: {message}");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn store_failure_is_recoverable_error() {
        let ctx = ctx_with(Arc::new(FailingRunStore));
        let out = execute(
            &RememberTool::new(),
            call(json!({"key": "k", "value": "v"})),
            &ctx,
        )
        .await;
        match out {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn schema_is_not_read_only() {
        let s = RememberTool::schema();
        assert_eq!(s.name, RememberTool::NAME);
        assert!(!s.annotations.read_only);
        assert!(!s.annotations.destructive);
        assert!(!s.annotations.idempotent);
    }
}
