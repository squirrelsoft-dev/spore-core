//! `recall(key)` — read a previously-remembered fact back out of the run store.
//!
//! This is the **read** half of the pair, also defined with the
//! [`tool!`](spore_core::tool) macro. Unlike [`remember`](super::remember) it
//! passes `annotations` to mark itself `read_only` + `idempotent`: it only reads
//! shared state, so the harness may dispatch it concurrently with other
//! read-only tools.
//!
//! Looking up a key that was never stored is a *recoverable* error — the agent
//! can adapt (try a different key, or remember the fact first) rather than
//! halting the run.

use schemars::JsonSchema;
use serde::Deserialize;

use spore_core::harness::ToolOutput;
use spore_core::{tool, StandardTool, ToolAnnotations};

use super::remember::FACT_PREFIX;

/// Tool name.
pub const NAME: &str = "recall";

/// Validated input for `recall`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallInput {
    /// The key a fact was previously stored under with `remember`.
    pub key: String,
}

/// Build the `recall` tool. `annotations` marks it `read_only` + `idempotent`
/// (a pure read of shared state), in contrast to `remember`.
pub fn recall_tool() -> StandardTool {
    tool! {
        name: NAME,
        description: "Recall a fact previously stored with `remember`, by its key.",
        input: RecallInput,
        execute: |input, _sandbox, ctx| async move {
            let store_key = format!("{FACT_PREFIX}{}", input.key);
            match ctx.run_store().get(ctx.session_id(), &store_key).await {
                Ok(Some(value)) => ToolOutput::success(value_to_string(&value)),
                Ok(None) => ToolOutput::error(format!("no fact stored under '{}'", input.key)),
                Err(e) => ToolOutput::error(format!("recall: could not read '{}': {e}", input.key)),
            }
        },
        annotations: ToolAnnotations {
            read_only: true,
            idempotent: true,
            ..Default::default()
        },
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

    use serde_json::json;
    use spore_core::harness::BoxFut;
    use spore_core::storage::InMemoryStorageProvider;
    use spore_core::{
        SandboxProvider, SandboxViolation, SessionId, ToolCall, ToolContext, ToolOutput,
    };

    use crate::tools::remember::remember_tool;

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
            name: NAME.into(),
            input,
        }
    }

    async fn recall(call: ToolCall, ctx: &ToolContext) -> ToolOutput {
        recall_tool()
            .implementation
            .execute(&call, &AllowAllSandbox, ctx)
            .await
    }

    async fn remember(ctx: &ToolContext, key: &str, value: &str) {
        remember_tool()
            .implementation
            .execute(
                &ToolCall {
                    id: "setup".into(),
                    name: "remember".into(),
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
        let out = recall(recall_call(json!({"key": "diet"})), &ctx).await;
        match out {
            ToolOutput::Success { content, .. } => assert_eq!(content, "crabs and shrimp"),
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn miss_is_recoverable_error_with_exact_message() {
        let ctx = in_memory_ctx();
        let out = recall(recall_call(json!({"key": "unknown"})), &ctx).await;
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
        let out = recall(recall_call(json!({})), &ctx).await;
        match out {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("invalid parameters"), "message: {message}");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_string_key_is_recoverable_error() {
        let ctx = in_memory_ctx();
        let out = recall(recall_call(json!({"key": 123})), &ctx).await;
        match out {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_does_not_write() {
        let ctx = in_memory_ctx();
        recall(recall_call(json!({"key": "k"})), &ctx).await;
        let keys = ctx.run_store().list_keys(ctx.session_id()).await.unwrap();
        assert!(
            keys.is_empty(),
            "recall must not persist anything, got {keys:?}"
        );
    }

    #[test]
    fn schema_is_read_only_and_idempotent() {
        let s = recall_tool().schema;
        assert_eq!(s.name, NAME);
        assert!(s.annotations.read_only);
        assert!(s.annotations.idempotent);
        assert!(!s.annotations.destructive);
    }
}
