//! Execution tools: BashCommand, RunTests.

use std::time::Duration;

use serde_json::json;

use crate::harness::{BoxFut, Operation, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::finish_with_possible_truncation;
use crate::tools::params::{parse_params, BashCommandParams, RunTestsParams};

// ============================================================================
// BashCommand
// ============================================================================

pub struct BashCommandTool;

impl BashCommandTool {
    pub const NAME: &'static str = "bash_command";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Execute a shell command via the sandbox".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                open_world: true,
                ..Default::default()
            },
        }
    }
}

impl Default for BashCommandTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for BashCommandTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn may_produce_large_output(&self) -> bool {
        true
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: BashCommandParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let timeout = params.timeout.map(Duration::from_secs);
            let out = match sandbox
                .execute_command(&params.command, &params.args, None, timeout)
                .await
            {
                Ok(o) => o,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            if out.timed_out {
                let secs = timeout.map(|d| d.as_secs()).unwrap_or(0);
                return ToolOutput::Error {
                    message: format!("command timed out after {secs}s"),
                    recoverable: true,
                };
            }
            if out.exit_code == 0 {
                finish_with_possible_truncation(out.stdout, &call.id, sandbox).await
            } else {
                ToolOutput::Error {
                    message: format!("exit {} ; stderr: {}", out.exit_code, out.stderr.trim_end()),
                    recoverable: true,
                }
            }
        })
    }
}

// ============================================================================
// RunTests
// ============================================================================

pub struct RunTestsTool;

impl RunTestsTool {
    pub const NAME: &'static str = "run_tests";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Run a test command in a working directory".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_dir": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command", "working_dir"],
            }),
            annotations: ToolAnnotations {
                open_world: true,
                ..Default::default()
            },
        }
    }
}

impl Default for RunTestsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for RunTestsTool {
    fn name(&self) -> &str {
        Self::NAME
    }
    fn may_produce_large_output(&self) -> bool {
        true
    }
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
        sandbox: &'a (dyn SandboxProvider + 'a),
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: RunTestsParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let timeout = params.timeout.map(Duration::from_secs);
            let working = match sandbox
                .resolve_path(&params.working_dir, Operation::Read)
                .await
            {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            // Split command into program + args (shell-free; spec says "no shell").
            let mut parts = params.command.split_whitespace();
            let Some(program) = parts.next() else {
                return ToolExecutionError::InvalidParameters {
                    reason: "command must not be empty".into(),
                }
                .into();
            };
            let args: Vec<String> = parts.map(String::from).collect();
            let out = match sandbox
                .execute_command(program, &args, Some(working.as_path()), timeout)
                .await
            {
                Ok(o) => o,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            if out.timed_out {
                let secs = timeout.map(|d| d.as_secs()).unwrap_or(0);
                return ToolOutput::Error {
                    message: format!("tests timed out after {secs}s"),
                    recoverable: true,
                };
            }
            let combined = format!("{}\n{}", out.stdout, out.stderr);
            if out.exit_code == 0 {
                finish_with_possible_truncation(combined, &call.id, sandbox).await
            } else {
                ToolOutput::Error {
                    message: format!("tests failed (exit {}): {}", out.exit_code, combined),
                    recoverable: true,
                }
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
    use crate::tool_registry::mock::AllowAllSandbox;
    use serde_json::json;

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn echo_runs_and_returns_stdout() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(
                &call("bash_command", json!({"command": "echo", "args": ["hi"]})),
                &sb,
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert!(content.contains("hi")),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nonzero_exit_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(&call("bash_command", json!({"command": "false"})), &sb)
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(
                &call(
                    "bash_command",
                    json!({"command": "sleep", "args": ["5"], "timeout": 1}),
                ),
                &sb,
            )
            .await;
        match r {
            ToolOutput::Error {
                message,
                recoverable,
            } => {
                assert!(recoverable);
                assert!(message.contains("timed out"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_params_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(&call("bash_command", json!({})), &sb)
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
