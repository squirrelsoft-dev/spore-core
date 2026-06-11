//! Execution tools: `Exec`, `BashCommand`, `RunTests`.
//!
//! Two distinct ways to run a process, with deliberately different contracts:
//!
//! - [`ExecTool`] (tool name `"exec"`) runs **one program directly** — no
//!   shell. `command` + `args` are passed verbatim to
//!   `SandboxProvider::execute_command`, so there are no pipes, redirects,
//!   globbing, or `$(...)`. Every argument is literal. This is the
//!   path-validated, no-injection-surface option.
//! - [`BashCommandTool`] (tool name `"bash_command"`) runs a **shell command
//!   line** via `/bin/sh -c <script>`, so it supports pipes, redirects,
//!   globbing, and `$(...)`. It is sugar over the same `execute_command`
//!   primitive (`execute_command("/bin/sh", ["-c", script], …)`).
//!
//!   TRADEOFF: because the shell itself opens any files the script touches,
//!   `bash_command` does NOT get the per-path `validate()` / `resolve_path`
//!   enforcement that `read_file` / `write_file` / `exec` get. It relies on the
//!   outer sandbox/container for isolation. `exec` remains the path-validated
//!   choice. `/bin/sh` also assumes a Unix target (fine for this repo; no
//!   `cmd.exe`/PowerShell branch).
//!
//! - [`RunTestsTool`] (tool name `"run_tests"`) splits a command string on
//!   whitespace and runs it shell-free inside a working directory.

use std::time::Duration;

use serde_json::json;

use crate::harness::{BoxFut, Operation, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::finish_with_possible_truncation;
use crate::tools::params::{parse_params, ExecParams, RunTestsParams, ShellCommandParams};

// ============================================================================
// Exec — shell-free: run one program directly
// ============================================================================

/// Runs one program directly via `SandboxProvider::execute_command`. No shell:
/// `command` + `args` are passed verbatim (no pipes, redirects, globbing, or
/// `$(...)`). Path-validated through the sandbox.
pub struct ExecTool;

impl ExecTool {
    pub const NAME: &'static str = "exec";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Run one program directly. No shell: no pipes, redirects, globbing, or \
                          $(...). Args are passed verbatim."
                .into(),
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

impl Default for ExecTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ExecTool {
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
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: ExecParams = match parse_params(call) {
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
                    message: format!(
                        "exit {} ; stderr: {}",
                        out.exit_code,
                        truncate_for_message(out.stderr.trim_end())
                    ),
                    recoverable: true,
                }
            }
        })
    }
}

// ============================================================================
// BashCommand — real shell: /bin/sh -c <script>
// ============================================================================

/// Runs a shell command line via `/bin/sh -c <script>`, supporting pipes,
/// redirects, globbing, and `$(...)`. Sugar over the same
/// `SandboxProvider::execute_command` primitive `exec` uses
/// (`execute_command("/bin/sh", ["-c", script], working_dir?, timeout?)`).
///
/// TRADEOFF: the shell opens any files the script touches itself, so this tool
/// does NOT receive the per-path `validate()` / `resolve_path` enforcement that
/// `read_file` / `write_file` / [`ExecTool`] get — it relies on the outer
/// sandbox/container for isolation. `exec` remains the path-validated choice.
/// `/bin/sh` assumes a Unix target (no Windows shell branch).
pub struct BashCommandTool;

impl BashCommandTool {
    pub const NAME: &'static str = "bash_command";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Execute a shell command line via /bin/sh -c. Supports pipes, redirects, \
                          globbing, and $(...)."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "script": {"type": "string"},
                    "working_dir": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["script"],
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
        _ctx: &'a crate::tool_registry::ToolContext,
    ) -> BoxFut<'a, ToolOutput> {
        Box::pin(async move {
            let params: ShellCommandParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let timeout = params.timeout.map(Duration::from_secs);
            // Only the optional working_dir is path-validated; the script's own
            // file accesses go through the shell, unvalidated (see doc above).
            let working = match &params.working_dir {
                Some(dir) => match sandbox.resolve_path(dir, Operation::Read).await {
                    Ok(p) => Some(p),
                    Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
                },
                None => None,
            };
            let args = vec!["-c".to_string(), params.script];
            let out = match sandbox
                .execute_command("/bin/sh", &args, working.as_deref(), timeout)
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
                    message: format!(
                        "exit {} ; stderr: {}",
                        out.exit_code,
                        truncate_for_message(out.stderr.trim_end())
                    ),
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
        _ctx: &'a crate::tool_registry::ToolContext,
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
                    message: format!(
                        "tests failed (exit {}): {}",
                        out.exit_code,
                        truncate_for_message(&combined)
                    ),
                    recoverable: true,
                }
            }
        })
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Truncate a string before embedding it in an error message. The threshold
/// is intentionally smaller than the general 64 KB output threshold because
/// this string is embedded inside an error `message` field, not returned as
/// a standalone `content` field — large stderr should not flood context.
fn truncate_for_message(s: &str) -> String {
    const THRESHOLD: usize = 8 * 1024;
    const HEAD: usize = 2 * 1024;
    const TAIL: usize = 2 * 1024;
    if s.len() <= THRESHOLD {
        return s.to_string();
    }
    let head_end = floor_char_boundary(s, HEAD);
    let tail_start = ceil_char_boundary(s, s.len().saturating_sub(TAIL));
    let elided = tail_start - head_end;
    format!(
        "{}\n... [{elided} bytes elided] ...\n{}",
        &s[..head_end],
        &s[tail_start..]
    )
}

fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn ceil_char_boundary(s: &str, mut idx: usize) -> usize {
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    // ---------------- ExecTool (shell-free) ----------------

    #[cfg(unix)]
    #[tokio::test]
    async fn echo_runs_and_returns_stdout() {
        let sb = AllowAllSandbox;
        let r = ExecTool::new()
            .execute(
                &call("exec", json!({"command": "echo", "args": ["hi"]})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert!(content.contains("hi")),
            other => panic!("{other:?}"),
        }
    }

    /// `exec` must NOT interpret shell syntax: pipe/`$(...)`/redirect tokens are
    /// passed to `echo` as literal arguments, and no file is created.
    #[cfg(unix)]
    #[tokio::test]
    async fn exec_has_no_shell_semantics() {
        let sb = AllowAllSandbox;
        // Run in a temp dir so we can prove no `out` file appears.
        let dir = std::env::temp_dir().join(format!(
            "spore-exec-noshell-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let r = ExecTool::new()
            .execute(
                &call(
                    "exec",
                    json!({"command": "echo", "args": ["a|b", "$(whoami)", ">out"]}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        std::env::set_current_dir(&prev).unwrap();
        match r {
            ToolOutput::Success { content, .. } => {
                assert!(
                    content.contains("a|b $(whoami) >out"),
                    "args must be literal, got {content:?}"
                );
            }
            other => panic!("{other:?}"),
        }
        assert!(
            !dir.join("out").exists(),
            "no redirect: `out` must not be created"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn exec_nonzero_exit_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = ExecTool::new()
            .execute(&call("exec", json!({"command": "false"})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn exec_timeout_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = ExecTool::new()
            .execute(
                &call(
                    "exec",
                    json!({"command": "sleep", "args": ["5"], "timeout": 1}),
                ),
                &sb,
                &test_ctx(),
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
    async fn exec_invalid_params_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = ExecTool::new()
            .execute(&call("exec", json!({})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    // ---------------- BashCommandTool (real shell) ----------------

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_command_supports_pipeline() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(
                &call(
                    "bash_command",
                    json!({"script": "printf 'hi' | tr a-z A-Z"}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "HI"),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_command_supports_redirect() {
        let sb = AllowAllSandbox;
        let tmp = std::env::temp_dir().join(format!(
            "spore-bash-redirect-{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let script = format!("printf 'data' > {}", tmp.display());
        let r = BashCommandTool::new()
            .execute(
                &call("bash_command", json!({"script": script})),
                &sb,
                &test_ctx(),
            )
            .await;
        assert!(matches!(r, ToolOutput::Success { .. }), "got {r:?}");
        assert_eq!(std::fs::read_to_string(&tmp).unwrap(), "data");
        let _ = std::fs::remove_file(&tmp);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_command_nonzero_exit_is_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(
                &call("bash_command", json!({"script": "exit 3"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn bash_command_timeout_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(
                &call("bash_command", json!({"script": "sleep 5", "timeout": 1})),
                &sb,
                &test_ctx(),
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
    async fn bash_command_invalid_params_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = BashCommandTool::new()
            .execute(&call("bash_command", json!({})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    // ---------------- truncate_for_message ----------------

    #[test]
    fn truncate_for_message_passthrough_when_short() {
        let s = "small error output";
        assert_eq!(truncate_for_message(s), s);
    }

    #[test]
    fn truncate_for_message_elides_middle_when_large() {
        let long = "x".repeat(10 * 1024);
        let result = truncate_for_message(&long);
        assert!(
            result.contains("bytes elided"),
            "expected elision marker in {result:?}"
        );
        assert!(
            result.len() < 8 * 1024,
            "message too long: {}",
            result.len()
        );
    }
}
