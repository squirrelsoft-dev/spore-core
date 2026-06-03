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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use spore_core::storage::InMemoryStorageProvider;
    use spore_core::{SandboxProvider, SandboxViolation, SessionId};

    use crate::tools::remember::RememberTool;

    const SESSION: &str = "fact-session";

    /// These tools never touch the filesystem — a permissive sandbox is plenty.
    struct AllowAllSandbox;
    impl SandboxProvider for AllowAllSandbox {
        fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
            Box::pin(async move { Ok(()) })
        }
    }

    fn in_memory_ctx() -> ToolContext {
        let backend = Arc::new(InMemoryStorageProvider::new());
        ToolContext::new(SessionId::new(SESSION), backend.clone(), backend)
    }

    fn recall_call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: RecallTool::NAME.into(),
            input,
        }
    }

    async fn remember(ctx: &ToolContext, key: &str, value: &str) {
        let tool = RememberTool::new();
        tool.execute(
            &ToolCall {
                id: "setup".into(),
                name: RememberTool::NAME.into(),
                input: json!({"key": key, "value": value}),
            },
            &AllowAllSandbox,
            ctx,
        )
        .await;
    }

    #[tokio::test]
    async fn returns_stored_value() {
        let ctx = in_memory_ctx();
        remember(&ctx, "diet", "crabs and shrimp").await;
        let out = RecallTool::new()
            .execute(&recall_call(json!({"key": "diet"})), &AllowAllSandbox, &ctx)
            .await;
        match out {
            ToolOutput::Success { content, .. } => assert_eq!(content, "crabs and shrimp"),
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn miss_is_recoverable_error_with_exact_message() {
        let ctx = in_memory_ctx();
        let out = RecallTool::new()
            .execute(
                &recall_call(json!({"key": "unknown"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match out {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert_eq!(message, "no fact stored under 'unknown'");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_key_is_recoverable_error() {
        let ctx = in_memory_ctx();
        let out = RecallTool::new()
            .execute(&recall_call(json!({})), &AllowAllSandbox, &ctx)
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
    async fn non_string_key_is_recoverable_error() {
        let ctx = in_memory_ctx();
        let out = RecallTool::new()
            .execute(&recall_call(json!({"key": 123})), &AllowAllSandbox, &ctx)
            .await;
        match out {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_does_not_write() {
        let ctx = in_memory_ctx();
        RecallTool::new()
            .execute(&recall_call(json!({"key": "k"})), &AllowAllSandbox, &ctx)
            .await;
        let keys = ctx.run_store().list_keys(ctx.session_id()).await.unwrap();
        assert!(
            keys.is_empty(),
            "recall must not persist anything, got {keys:?}"
        );
    }

    #[test]
    fn schema_is_read_only_and_idempotent() {
        let s = RecallTool::schema();
        assert_eq!(s.name, RecallTool::NAME);
        assert!(s.annotations.read_only);
        assert!(s.annotations.idempotent);
        assert!(!s.annotations.destructive);
    }
}
