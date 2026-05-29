//! Git tools: GitLog, GitDiff, GitCommit, GitStatus, GitReset.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::finish_with_possible_truncation;
use crate::tools::params::{
    parse_params, GitCommitParams, GitDiffParams, GitLogParams, GitResetParams, GitStatusParams,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitResetMode {
    Hard,
    Soft,
    Mixed,
}

async fn run_git(
    args: &[String],
    sandbox: &(dyn SandboxProvider + '_),
) -> Result<crate::harness::CommandOutput, ToolExecutionError> {
    sandbox
        .execute_command("git", args, None, None)
        .await
        .map_err(ToolExecutionError::SandboxViolation)
}

fn classify(out: &crate::harness::CommandOutput) -> Result<String, ToolOutput> {
    if out.exit_code == 0 {
        Ok(out.stdout.clone())
    } else {
        Err(ToolOutput::Error {
            message: format!("git exit {} ; {}", out.exit_code, out.stderr.trim_end()),
            recoverable: true,
        })
    }
}

// ============================================================================
// GitLog
// ============================================================================

pub struct GitLogTool;

impl GitLogTool {
    pub const NAME: &'static str = "git_log";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Show recent git commits".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "format": {"type": "string"},
                },
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for GitLogTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GitLogTool {
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
            let params: GitLogParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let mut args = vec!["log".to_string(), "-n".to_string(), params.n.to_string()];
            if params.format == "oneline" {
                args.push("--oneline".into());
            } else {
                args.push(format!("--format={}", params.format));
            }
            let out = match run_git(&args, sandbox).await {
                Ok(o) => o,
                Err(e) => return e.into(),
            };
            match classify(&out) {
                Ok(s) => finish_with_possible_truncation(s, &call.id, sandbox).await,
                Err(e) => e,
            }
        })
    }
}

// ============================================================================
// GitDiff
// ============================================================================

pub struct GitDiffTool;

impl GitDiffTool {
    pub const NAME: &'static str = "git_diff";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Show a git diff".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for GitDiffTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GitDiffTool {
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
            let params: GitDiffParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let mut args = vec!["diff".to_string()];
            if let Some(f) = params.from {
                args.push(f);
            }
            if let Some(t) = params.to {
                args.push(t);
            }
            let out = match run_git(&args, sandbox).await {
                Ok(o) => o,
                Err(e) => return e.into(),
            };
            match classify(&out) {
                Ok(s) => finish_with_possible_truncation(s, &call.id, sandbox).await,
                Err(e) => e,
            }
        })
    }
}

// ============================================================================
// GitCommit
// ============================================================================

pub struct GitCommitTool;

impl GitCommitTool {
    pub const NAME: &'static str = "git_commit";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Stage files (if any) and create a git commit".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["message"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for GitCommitTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GitCommitTool {
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
            let params: GitCommitParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let mut combined = String::new();
            if !params.files.is_empty() {
                let mut args = vec!["add".to_string()];
                args.extend(params.files.clone());
                let out = match run_git(&args, sandbox).await {
                    Ok(o) => o,
                    Err(e) => return e.into(),
                };
                match classify(&out) {
                    Ok(s) => {
                        combined.push_str(&s);
                    }
                    Err(e) => return e,
                }
            }
            let args = vec![
                "commit".to_string(),
                "-m".to_string(),
                params.message.clone(),
            ];
            let out = match run_git(&args, sandbox).await {
                Ok(o) => o,
                Err(e) => return e.into(),
            };
            match classify(&out) {
                Ok(s) => {
                    combined.push_str(&s);
                    ToolOutput::Success {
                        content: combined,
                        truncated: false,
                    }
                }
                Err(e) => e,
            }
        })
    }
}

// ============================================================================
// GitStatus
// ============================================================================

pub struct GitStatusTool;

impl GitStatusTool {
    pub const NAME: &'static str = "git_status";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Show git status (porcelain)".into(),
            parameters: json!({"type": "object", "properties": {}}),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for GitStatusTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GitStatusTool {
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
            let _: GitStatusParams = parse_params(call).unwrap_or_default();
            let args = vec!["status".to_string(), "--porcelain".to_string()];
            let out = match run_git(&args, sandbox).await {
                Ok(o) => o,
                Err(e) => return e.into(),
            };
            match classify(&out) {
                Ok(s) => ToolOutput::Success {
                    content: s,
                    truncated: false,
                },
                Err(e) => e,
            }
        })
    }
}

// ============================================================================
// GitReset
// ============================================================================

pub struct GitResetTool;

impl GitResetTool {
    pub const NAME: &'static str = "git_reset";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Reset to a target commit (hard/soft/mixed)".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "mode": {"type": "string", "enum": ["hard", "soft", "mixed"]},
                },
                "required": ["target", "mode"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for GitResetTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for GitResetTool {
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
            let params: GitResetParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let flag = match params.mode {
                GitResetMode::Hard => "--hard",
                GitResetMode::Soft => "--soft",
                GitResetMode::Mixed => "--mixed",
            };
            let args = vec!["reset".to_string(), flag.to_string(), params.target.clone()];
            let out = match run_git(&args, sandbox).await {
                Ok(o) => o,
                Err(e) => return e.into(),
            };
            match classify(&out) {
                Ok(s) => ToolOutput::Success {
                    content: s,
                    truncated: false,
                },
                Err(e) => e,
            }
        })
    }
}

// ============================================================================
// Tests (gated on `git` availability)
// ============================================================================

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use serde_json::json;

    fn git_available() -> bool {
        std::process::Command::new("git")
            .arg("--version")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn git_status_runs_when_git_available() {
        if !git_available() {
            return;
        }
        let sb = AllowAllSandbox;
        let r = GitStatusTool::new()
            .execute(&call("git_status", json!({})), &sb, &test_ctx())
            .await;
        // Either Success (in a repo) or Error (outside a repo) — both fine.
        match r {
            ToolOutput::Success { .. } | ToolOutput::Error { .. } => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn reset_mode_roundtrips_snake_case() {
        let m = GitResetMode::Hard;
        let j = serde_json::to_string(&m).unwrap();
        assert_eq!(j, "\"hard\"");
        let back: GitResetMode = serde_json::from_str(&j).unwrap();
        assert_eq!(back, m);
    }
}
