//! `remember(key, value)` — persist a fact into the run store.
//!
//! This is the **write** half of the custom-tool pair, defined with the
//! [`tool!`](spore_core::tool) macro: a typed input struct plus an async
//! `execute` closure, and the macro derives the advertised JSON schema from the
//! struct (via `schemars`) so the schema and the deserialization can never
//! drift.
//!
//! It demonstrates the storage seam: [`ToolContext::run_store`] +
//! [`ToolContext::session_id`] are the only path to durable, per-run state. The
//! `sandbox` parameter is part of the closure signature but unused here — these
//! tools never touch the filesystem, so they ignore it (`_sandbox`).
//!
//! Keys are namespaced under `fact:{key}` so the example cannot collide with
//! reserved store keys the catalogue uses (`todo`, `task`, `memory`).

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use spore_core::harness::ToolOutput;
use spore_core::{tool, StandardTool};

/// Prefix applied to every key so this example's facts live in their own
/// namespace inside the run store.
pub const FACT_PREFIX: &str = "fact:";

/// Tool name — also used by tests and `recall` cross-checks.
pub const NAME: &str = "remember";

/// Validated input for `remember`. Deriving `JsonSchema` lets `tool!` advertise
/// a schema generated from exactly this struct.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RememberInput {
    /// Short, stable key to file the fact under (e.g. "habitat", "lifespan").
    pub key: String,
    /// The fact to remember.
    pub value: String,
}

/// Build the `remember` tool. The `tool!` macro generates the `Tool` impl,
/// derives the schema from [`RememberInput`], and bundles them into a
/// [`StandardTool`] ready for `.tool(...)`.
///
/// Annotations are omitted, so they default to all-`false` — `remember` MUTATES
/// shared persisted state, so (unlike `recall`) it is intentionally not
/// `read_only`.
pub fn remember_tool() -> StandardTool {
    tool! {
        name: NAME,
        description: "Store a fact under a short key so it can be recalled later. \
                      Use a stable, memorable key (e.g. 'habitat', 'lifespan').",
        input: RememberInput,
        execute: |input, _sandbox, ctx| async move {
            let store_key = format!("{FACT_PREFIX}{}", input.key);
            match ctx
                .run_store()
                .put(ctx.session_id(), &store_key, json!(input.value))
                .await
            {
                Ok(()) => ToolOutput::success(format!("remembered {}", input.key)),
                Err(e) => {
                    ToolOutput::error(format!("remember: could not persist '{}': {e}", input.key))
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use serde_json::Value as JsonValue;
    use spore_core::harness::BoxFut;
    use spore_core::storage::{InMemoryStorageProvider, RunStore, StorageError};
    use spore_core::{
        SandboxProvider, SandboxViolation, SessionId, ToolCall, ToolContext, ToolOutput,
    };

    use crate::tools::recall::recall_tool;

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
            name: NAME.into(),
            input,
        }
    }

    /// Dispatch through the macro-built tool's `Tool` impl.
    async fn execute(call: ToolCall, ctx: &ToolContext) -> ToolOutput {
        remember_tool()
            .implementation
            .execute(&call, &AllowAllSandbox, ctx)
            .await
    }

    #[tokio::test]
    async fn stores_under_fact_prefix() {
        let ctx = in_memory_ctx();
        let out = execute(
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
            call(json!({"key": "diet", "value": "crabs and shrimp"})),
            &ctx,
        )
        .await;
        let out = recall_tool()
            .implementation
            .execute(
                &ToolCall {
                    id: "c2".into(),
                    name: "recall".into(),
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
        // The macro deserializes into `RememberInput`; a missing field is a
        // recoverable "invalid parameters" error (so repair can retry).
        let out = execute(call(json!({"key": "habitat"})), &in_memory_ctx()).await;
        match out {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("invalid parameters"), "message: {message}");
                assert!(message.contains("value"), "message: {message}");
            }
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_string_key_is_recoverable_error() {
        let out = execute(call(json!({"key": 7, "value": "x"})), &in_memory_ctx()).await;
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
    async fn store_failure_is_recoverable_error() {
        let ctx = ctx_with(Arc::new(FailingRunStore));
        let out = execute(call(json!({"key": "k", "value": "v"})), &ctx).await;
        match out {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("expected error, got {other:?}"),
        }
    }

    #[test]
    fn schema_is_not_read_only() {
        let s = remember_tool().schema;
        assert_eq!(s.name, NAME);
        assert!(!s.annotations.read_only);
        assert!(!s.annotations.destructive);
        assert!(!s.annotations.idempotent);
        // Derived from `RememberInput`.
        let props = s
            .parameters
            .get("properties")
            .and_then(|p| p.as_object())
            .expect("object schema");
        assert!(props.contains_key("key"));
        assert!(props.contains_key("value"));
    }
}
