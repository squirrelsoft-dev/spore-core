//! Standard tool implementations.
//!
//! Implements issue #5. Each submodule houses a family of tools that all
//! conform to the canonical [`crate::tool_registry::Tool`] trait shipped in
//! issue #4. Tools are stateless and receive a `SandboxProvider` on every
//! dispatch — they never touch the environment directly.
//!
//! ## Families
//! - [`fs`]       — `ReadFile`, `WriteFile`, `ListDir`, `DeleteFile`, `MoveFile`
//! - [`exec`]     — `Exec`, `BashCommand`, `RunTests`
//! - [`search`]   — `GrepFiles`, `FindFiles`
//! - [`git`]      — `GitLog`, `GitDiff`, `GitCommit`, `GitStatus`, `GitReset`
//! - [`http`]     — `HttpGet`, `HttpPost`
//! - [`subagent`] — `SubagentTool` (wraps a child `Harness`)
//!
//! All tools that may produce large output set `may_produce_large_output() ==
//! true` and cooperate with `SandboxProvider::handle_large_output`.

pub mod error;
pub mod exec;
pub mod fs;
pub mod git;
pub mod http;
pub mod params;
pub mod search;
pub mod subagent;

pub use error::ToolExecutionError;
pub use exec::{BashCommandTool, ExecTool, RunTestsTool};
pub use fs::{DeleteFileTool, ListDirTool, MoveFileTool, ReadFileTool, WriteFileTool};
pub use git::{GitCommitTool, GitDiffTool, GitLogTool, GitResetMode, GitResetTool, GitStatusTool};
pub use http::{HttpGetTool, HttpPostTool};
pub use search::{FindFilesTool, GrepFilesTool};
pub use subagent::{BuildError, ContextSharing, SubagentTool};

/// Threshold (in bytes/chars) above which tool output is routed through
/// `SandboxProvider::handle_large_output` instead of returned inline.
pub const LARGE_OUTPUT_THRESHOLD: usize = 64 * 1024;

/// Head/tail token budgets passed to `handle_large_output` by default.
pub const DEFAULT_HEAD_TOKENS: u32 = 2000;
pub const DEFAULT_TAIL_TOKENS: u32 = 2000;

/// Helper: return `Success { truncated }` for a string body, routing through
/// the sandbox if the body exceeds [`LARGE_OUTPUT_THRESHOLD`].
pub(crate) async fn finish_with_possible_truncation(
    content: String,
    call_id: &str,
    sandbox: &(dyn crate::harness::SandboxProvider + '_),
) -> crate::harness::ToolOutput {
    if content.len() > LARGE_OUTPUT_THRESHOLD {
        let truncated = sandbox
            .handle_large_output(content, call_id, DEFAULT_HEAD_TOKENS, DEFAULT_TAIL_TOKENS)
            .await;
        crate::harness::ToolOutput::Success {
            content: truncated.content,
            truncated: true,
        }
    } else {
        crate::harness::ToolOutput::Success {
            content,
            truncated: false,
        }
    }
}

#[cfg(test)]
mod fixture_tests {
    use crate::harness::SandboxProvider;
    use crate::tool_registry::mock::AllowAllSandbox;
    use serde::Deserialize;
    use serde_json::Value;

    #[derive(Deserialize)]
    struct ParamValidationScenario {
        tool: String,
        input: Value,
        expected: String, // "ok" | "invalid_parameters"
    }

    /// Try to parse a tool's parameters. Returns `true` if the input
    /// passes parameter-validation, `false` if it is rejected with
    /// `InvalidParameters`.
    fn param_parse_ok(tool: &str, input: &Value) -> bool {
        use crate::tools::params::*;
        fn ok<T: serde::de::DeserializeOwned>(v: &Value) -> bool {
            serde_json::from_value::<T>(v.clone()).is_ok()
        }
        match tool {
            "read_file" => ok::<ReadFileParams>(input),
            "write_file" => ok::<WriteFileParams>(input),
            "list_dir" => ok::<ListDirParams>(input),
            "delete_file" => ok::<DeleteFileParams>(input),
            "move_file" => ok::<MoveFileParams>(input),
            "grep_files" => {
                if !ok::<GrepFilesParams>(input) {
                    return false;
                }
                // Regex compile also gates as InvalidParameters.
                let p: GrepFilesParams = serde_json::from_value(input.clone()).unwrap();
                regex::Regex::new(&p.pattern).is_ok()
            }
            "find_files" => ok::<FindFilesParams>(input),
            "git_status" => true,
            "git_log" => ok::<GitLogParams>(input),
            "git_diff" => ok::<GitDiffParams>(input),
            "git_commit" => ok::<GitCommitParams>(input),
            "git_reset" => ok::<GitResetParams>(input),
            "http_get" => ok::<HttpGetParams>(input),
            "http_post" => ok::<HttpPostParams>(input),
            _ => true,
        }
    }

    #[tokio::test]
    async fn fixture_replay_param_validation() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tools/param_validation.json");
        let Ok(data) = std::fs::read_to_string(&path) else {
            return;
        };
        let scenarios: Vec<ParamValidationScenario> = serde_json::from_str(&data).unwrap();
        assert!(!scenarios.is_empty(), "expected ≥1 scenario");
        for sc in scenarios {
            let actual = if param_parse_ok(&sc.tool, &sc.input) {
                "ok"
            } else {
                "invalid_parameters"
            };
            assert_eq!(
                actual, sc.expected,
                "scenario tool={} got {actual} expected {}",
                sc.tool, sc.expected
            );
        }
    }

    #[derive(Deserialize)]
    struct TruncationScenario {
        content_length: usize,
        head_tokens: u32,
        tail_tokens: u32,
        expects_truncated: bool,
    }

    #[tokio::test]
    async fn fixture_replay_output_truncation() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tools/output_truncation.json");
        let Ok(data) = std::fs::read_to_string(&path) else {
            return;
        };
        let scenarios: Vec<TruncationScenario> = serde_json::from_str(&data).unwrap();
        let sandbox = AllowAllSandbox;
        for sc in scenarios {
            let content = "x".repeat(sc.content_length);
            let truncated = sandbox
                .handle_large_output(content.clone(), "fx", sc.head_tokens, sc.tail_tokens)
                .await;
            let actually_truncated = truncated.content != content;
            assert_eq!(
                actually_truncated, sc.expects_truncated,
                "truncation mismatch at content_length={}",
                sc.content_length
            );
        }
    }
}
