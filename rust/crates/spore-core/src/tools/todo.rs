//! TodoWrite tool (#81, net-new Tier-2 storage tool).
//!
//! `todo_write` persists an agent-managed todo list via the [`ToolContext`]'s
//! [`RunStore`](crate::storage::RunStore) under the key
//! [`TODO_STORE_KEY`] (`"todo"`), keyed by the run's
//! [`SessionId`](crate::harness::SessionId). The agent supplies the FULL desired
//! list on every call; it REPLACES the persisted list wholesale (no per-item
//! diffing). The current list is returned as JSON success content.
//!
//! Like [`TaskListTool`](crate::tools::tasklist::TaskListTool) it is NOT
//! annotated `read_only`: it mutates shared persisted state and must dispatch
//! sequentially (a concurrent read-modify-write would race).

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolContext, ToolSchema};
use crate::tools::params::{parse_params, TodoWriteParams};

/// RunStore key under which the todo list is persisted (issue #81, Q5).
pub const TODO_STORE_KEY: &str = "todo";

pub struct TodoWriteTool;

impl TodoWriteTool {
    pub const NAME: &'static str = "todo_write";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Replace the persisted todo list with the supplied full list".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["completed", "in_progress", "pending"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    },
                },
                "required": ["todos"],
            }),
            // Intentionally NOT read_only — mutates shared persisted state and
            // must dispatch sequentially. See module docs / TaskListTool.
            annotations: ToolAnnotations::default(),
        }
    }
}

impl Default for TodoWriteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for TodoWriteTool {
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
            let params: TodoWriteParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let value = match serde_json::to_value(&params.todos) {
                Ok(v) => v,
                Err(e) => {
                    return ToolOutput::Error {
                        message: format!("could not serialize todos: {e}"),
                        recoverable: true,
                    };
                }
            };
            if let Err(e) = ctx
                .run_store()
                .put(ctx.session_id(), TODO_STORE_KEY, value)
                .await
            {
                return ToolOutput::Error {
                    message: format!("could not persist todos: {e}"),
                    recoverable: true,
                };
            }
            match serde_json::to_string(&params.todos) {
                Ok(content) => ToolOutput::Success {
                    content,
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("could not serialize todos: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::SessionId;
    use crate::storage::InMemoryStorageProvider;
    use crate::tool_registry::mock::AllowAllSandbox;
    use crate::tools::params::{TodoItem, TodoStatus};
    use serde_json::json;
    use std::sync::Arc;

    fn ctx() -> ToolContext {
        let backend = Arc::new(InMemoryStorageProvider::new());
        ToolContext::new(
            SessionId::new("todo-session"),
            crate::storage::ProjectId::from_canonical_path("/todo-test-project"),
            backend.clone(),
            backend,
        )
    }

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: TodoWriteTool::NAME.into(),
            input,
        }
    }

    #[tokio::test]
    async fn writes_and_persists_under_todo_key() {
        let ctx = ctx();
        let sb = AllowAllSandbox;
        let r = TodoWriteTool::new()
            .execute(
                &call(json!({"todos": [
                    {"content": "a", "status": "pending"},
                    {"content": "b", "status": "in_progress"},
                ]})),
                &sb,
                &ctx,
            )
            .await;
        let got: Vec<TodoItem> = match r {
            ToolOutput::Success { content, .. } => serde_json::from_str(&content).unwrap(),
            other => panic!("{other:?}"),
        };
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].status, TodoStatus::InProgress);

        // Persisted under the "todo" key.
        let blob = ctx
            .run_store()
            .get(ctx.session_id(), TODO_STORE_KEY)
            .await
            .unwrap()
            .expect("todo blob present");
        let persisted: Vec<TodoItem> = serde_json::from_value(blob).unwrap();
        assert_eq!(persisted, got);
    }

    #[tokio::test]
    async fn replaces_list_wholesale() {
        let ctx = ctx();
        let sb = AllowAllSandbox;
        let tool = TodoWriteTool::new();
        tool.execute(
            &call(json!({"todos": [{"content": "old1", "status": "pending"}, {"content": "old2", "status": "pending"}]})),
            &sb,
            &ctx,
        )
        .await;
        let r = tool
            .execute(
                &call(json!({"todos": [{"content": "new", "status": "completed"}]})),
                &sb,
                &ctx,
            )
            .await;
        let got: Vec<TodoItem> = match r {
            ToolOutput::Success { content, .. } => serde_json::from_str(&content).unwrap(),
            other => panic!("{other:?}"),
        };
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].content, "new");
    }

    #[tokio::test]
    async fn bad_params_is_recoverable_error() {
        let ctx = ctx();
        let r = TodoWriteTool::new()
            .execute(
                &call(json!({"todos": "not-an-array"})),
                &AllowAllSandbox,
                &ctx,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn schema_not_read_only() {
        let s = TodoWriteTool::schema();
        assert!(!s.annotations.read_only);
        assert!(!s.annotations.destructive);
    }
}
