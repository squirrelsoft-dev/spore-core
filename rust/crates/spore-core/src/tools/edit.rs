//! EditFile tool (#81, net-new Tier-1 sandbox tool).
//!
//! `edit_file` replaces the FIRST and ONLY occurrence of `old_string` with
//! `new_string` in the file at `path`. The match must be UNIQUE:
//! - `old_string` not found      → recoverable [`ToolOutput::Error`].
//! - `old_string` found >1 time  → recoverable [`ToolOutput::Error`].
//!
//! This is a net-new tool that does NOT replace `write_file` (issue #81, Q5).
//! Annotated `destructive` (it mutates a file in place).

use serde_json::json;

use crate::harness::{BoxFut, Operation, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::params::{parse_params, EditFileParams};

pub struct EditFileTool;

impl EditFileTool {
    pub const NAME: &'static str = "edit_file";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Replace the unique occurrence of old_string with new_string in a file"
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for EditFileTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: EditFileParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let resolved = match sandbox.resolve_path(&params.path, Operation::Write).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            let content = match tokio::fs::read_to_string(&resolved).await {
                Ok(c) => c,
                Err(e) => {
                    return ToolOutput::Error {
                        message: format!("read failed: {e}"),
                        recoverable: true,
                    };
                }
            };
            let count = content.matches(&params.old_string).count();
            if count == 0 {
                return ToolOutput::Error {
                    message: format!("old_string not found in {}", params.path),
                    recoverable: true,
                };
            }
            if count > 1 {
                return ToolOutput::Error {
                    message: format!(
                        "old_string is not unique in {} ({count} occurrences); provide more context",
                        params.path
                    ),
                    recoverable: true,
                };
            }
            let updated = content.replacen(&params.old_string, &params.new_string, 1);
            match tokio::fs::write(&resolved, updated.as_bytes()).await {
                Ok(_) => ToolOutput::Success {
                    content: format!("edited {}", params.path),
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("write failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;
    use tempfile::TempDir;

    fn call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: EditFileTool::NAME.into(),
            input,
        }
    }

    #[tokio::test]
    async fn edit_replaces_unique_occurrence() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello world\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": "world",
                    "new_string": "there",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        assert!(matches!(r, ToolOutput::Success { .. }));
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "hello there\n"
        );
    }

    #[tokio::test]
    async fn edit_not_found_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": "absent",
                    "new_string": "x",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("not found"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_non_unique_is_recoverable_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "x x x\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": path.to_str().unwrap(),
                    "old_string": "x",
                    "new_string": "y",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable);
                assert!(message.contains("not unique"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_missing_file_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(
                &call(json!({
                    "path": "/no/such/file",
                    "old_string": "a",
                    "new_string": "b",
                })),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn edit_bad_params_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = EditFileTool::new()
            .execute(&call(json!({"path": "/x"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn schema_is_destructive() {
        let s = EditFileTool::schema();
        assert!(s.annotations.destructive);
        assert!(!s.annotations.read_only);
    }
}
