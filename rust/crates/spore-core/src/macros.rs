//! Ergonomic tool-definition macros.
//!
//! [`tool!`](crate::tool) collapses the boilerplate of a hand-written
//! [`Tool`](crate::tool_registry::Tool) impl — name, schema, input
//! deserialization, and an async `execute` body — into a single expression that
//! evaluates to a ready-to-register [`StandardTool`](crate::tools::StandardTool).
//!
//! The input schema is derived automatically from the input type via
//! [`schemars`], so the JSON Schema the model sees always matches the struct the
//! tool deserializes — they can never drift.
//!
//! ```
//! use spore_core::tool;
//! use spore_core::harness::ToolOutput;
//!
//! #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
//! struct CalculatorInput {
//!     /// The expression to evaluate.
//!     expression: String,
//! }
//!
//! let calculator = tool! {
//!     name: "calculator",
//!     description: "Evaluates a mathematical expression and returns the result",
//!     input: CalculatorInput,
//!     execute: |input, _sandbox, _ctx| async move {
//!         ToolOutput::success(format!("evaluated: {}", input.expression))
//!     }
//! };
//! assert_eq!(calculator.schema.name, "calculator");
//! ```
//!
//! The generated tool plugs straight into a builder via
//! [`HarnessBuilder::tool`](crate::HarnessBuilder::tool).

// Re-export the crate-root `tool!` under `macros::` too, so both
// `spore_core::tool!` and `spore_core::macros::tool!` resolve.
pub use crate::tool;

/// Hidden runtime support for [`tool!`](crate::tool). Not part of the stable
/// surface — call paths here may change between releases. Re-exporting
/// `serde_json` lets the macro expand at call sites that do not themselves
/// depend on `serde_json`.
#[doc(hidden)]
pub mod __support {
    pub use serde_json;

    /// Derive a tool's `parameters` JSON Schema from its input type. Used by
    /// [`tool!`](crate::tool) so the advertised schema is generated from the
    /// exact type the tool deserializes. Falls back to a permissive
    /// `{"type":"object"}` if serialization of the schema ever fails (it does
    /// not for any well-formed `JsonSchema` derive).
    pub fn schema_for_input<T: schemars::JsonSchema>() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(T))
            .unwrap_or_else(|_| serde_json::json!({ "type": "object" }))
    }
}

/// Define a [`Tool`](crate::tool_registry::Tool) and bundle it with its derived
/// schema into a [`StandardTool`](crate::tools::StandardTool) in one expression.
///
/// # Fields
/// - `name` — string literal (or `&str` expression); the tool's registered name.
/// - `description` — string literal (or `&str` expression) shown to the model.
/// - `input` — a type deriving [`serde::Deserialize`] and
///   [`schemars::JsonSchema`]. The macro deserializes the model's tool arguments
///   into this type and derives the advertised JSON Schema from it.
/// - `execute` — an async closure `|input, sandbox, ctx| async move { ... }`
///   returning a [`ToolOutput`](crate::harness::ToolOutput). `input` is the
///   deserialized `input` type; `sandbox` is `&dyn SandboxProvider`; `ctx` is
///   `&ToolContext`.
///
/// If the model's arguments fail to deserialize into `input`, the generated
/// tool returns a **recoverable** [`ToolOutput::error`](crate::harness::ToolOutput::error)
/// — so a configured [`ToolCallRepair`](crate::tool_call_repair::ToolCallRepair)
/// gets a chance to coerce the arguments and re-dispatch.
///
/// See the [module docs](crate::macros) for a full example.
#[macro_export]
macro_rules! tool {
    (
        name: $name:expr,
        description: $description:expr,
        input: $input:ty,
        execute: $execute:expr $(,)?
    ) => {{
        // Anonymous, zero-sized tool implementation. Scoped to this block, so
        // multiple `tool!` invocations never collide on the type name.
        struct __SporeMacroTool;

        impl $crate::tool_registry::Tool for __SporeMacroTool {
            fn name(&self) -> &str {
                $name
            }

            fn execute<'a>(
                &'a self,
                call: &'a $crate::model::ToolCall,
                sandbox: &'a (dyn $crate::harness::SandboxProvider + 'a),
                ctx: &'a $crate::tool_registry::ToolContext,
            ) -> $crate::harness::BoxFut<'a, $crate::harness::ToolOutput> {
                // Pins the `execute` closure's parameter types via an explicit
                // `FnOnce(..)` bound, so untyped closure params (`|input, ..|`)
                // are inferred even though the closure returns an `async` block
                // (where call-site inference alone is insufficient).
                async fn __spore_invoke<'b, F, Fut>(
                    f: F,
                    input: $input,
                    sandbox: &'b (dyn $crate::harness::SandboxProvider + 'b),
                    ctx: &'b $crate::tool_registry::ToolContext,
                ) -> $crate::harness::ToolOutput
                where
                    F: ::core::ops::FnOnce(
                        $input,
                        &'b (dyn $crate::harness::SandboxProvider + 'b),
                        &'b $crate::tool_registry::ToolContext,
                    ) -> Fut,
                    Fut: ::core::future::Future<Output = $crate::harness::ToolOutput> + 'b,
                {
                    f(input, sandbox, ctx).await
                }

                ::std::boxed::Box::pin(async move {
                    let input: $input =
                        match $crate::macros::__support::serde_json::from_value(call.input.clone())
                        {
                            ::core::result::Result::Ok(v) => v,
                            ::core::result::Result::Err(e) => {
                                return $crate::harness::ToolOutput::error(::std::format!(
                                    "invalid parameters for tool `{}`: {}",
                                    $name,
                                    e
                                ));
                            }
                        };
                    __spore_invoke($execute, input, sandbox, ctx).await
                })
            }
        }

        let __spore_schema = $crate::tool_registry::ToolSchema {
            name: ::std::string::String::from($name),
            description: ::std::string::String::from($description),
            parameters: $crate::macros::__support::schema_for_input::<$input>(),
            annotations: ::core::default::Default::default(),
        };

        $crate::tools::StandardTool::new(::std::boxed::Box::new(__SporeMacroTool), __spore_schema)
    }};
}

#[cfg(test)]
mod tests {
    use crate::harness::{NullSandbox, SandboxProvider, ToolOutput};
    use crate::model::ToolCall;
    use crate::storage::{InMemoryStorageProvider, StorageProvider};
    use crate::tool_registry::ToolContext;
    use std::sync::Arc;

    #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    struct EchoInput {
        /// Text to echo back.
        message: String,
        #[serde(default)]
        shout: bool,
    }

    fn ctx() -> ToolContext {
        let storage = StorageProvider::single(Arc::new(InMemoryStorageProvider::new()));
        ToolContext::new(
            crate::harness::SessionId::new("s1"),
            storage.run().clone(),
            storage.memory().clone(),
        )
    }

    #[tokio::test]
    async fn macro_builds_tool_with_derived_schema() {
        let t = tool! {
            name: "echo",
            description: "Echoes the input message",
            input: EchoInput,
            execute: |input, _sandbox, _ctx| async move {
                if input.shout {
                    ToolOutput::success(input.message.to_uppercase())
                } else {
                    ToolOutput::success(input.message)
                }
            }
        };

        // Schema carries the macro metadata and a derived object schema.
        assert_eq!(t.schema.name, "echo");
        assert_eq!(t.schema.description, "Echoes the input message");
        let props = t
            .schema
            .parameters
            .get("properties")
            .and_then(|p| p.as_object())
            .expect("derived schema should expose properties");
        assert!(props.contains_key("message"));
        assert!(props.contains_key("shout"));

        // Implementation reports the right name and runs.
        assert_eq!(t.implementation.name(), "echo");
        let sandbox = NullSandbox;
        let call = ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            input: serde_json::json!({ "message": "hi", "shout": true }),
        };
        let out = t
            .implementation
            .execute(&call, &sandbox as &dyn SandboxProvider, &ctx())
            .await;
        assert_eq!(out, ToolOutput::success("HI"));
    }

    #[tokio::test]
    async fn macro_tool_returns_recoverable_error_on_bad_input() {
        let t = tool! {
            name: "echo",
            description: "Echoes the input message",
            input: EchoInput,
            execute: |input, _sandbox, _ctx| async move {
                ToolOutput::success(input.message)
            }
        };
        let sandbox = NullSandbox;
        // `message` missing → deserialize fails → recoverable error.
        let call = ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            input: serde_json::json!({ "shout": true }),
        };
        let out = t
            .implementation
            .execute(&call, &sandbox as &dyn SandboxProvider, &ctx())
            .await;
        match out {
            ToolOutput::Error {
                recoverable: true,
                message,
            } => assert!(message.contains("invalid parameters for tool `echo`")),
            other => panic!("expected recoverable error, got {other:?}"),
        }
    }
}
