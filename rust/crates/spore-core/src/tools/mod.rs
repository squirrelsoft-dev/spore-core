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

pub mod catalogue;
pub mod control;
pub mod edit;
pub mod error;
pub mod exec;
pub mod fs;
pub mod git;
pub mod http;
pub mod memory;
pub mod message;
pub mod params;
pub mod search;
pub mod subagent;
pub mod tasklist;
pub mod todo;
pub mod web;

pub use catalogue::{StandardTool, StandardTools};
pub use control::{AbortTool, AskUserQuestionTool, EnterPlanModeTool, ExitPlanModeTool};
pub use edit::EditFileTool;
pub use error::ToolExecutionError;
pub use exec::{BashCommandTool, ExecTool, RunTestsTool};
pub use fs::{DeleteFileTool, ListDirTool, MoveFileTool, ReadFileTool, WriteFileTool};
pub use git::{GitCommitTool, GitDiffTool, GitLogTool, GitResetMode, GitResetTool, GitStatusTool};
pub use http::{HttpGetTool, HttpPostTool};
pub use memory::MemoryTool;
pub use message::SendMessageTool;
pub use search::{FindFilesTool, GrepFilesTool, GrepTool};
pub use subagent::{BuildError, ContextSharing, SubagentTool};
pub use tasklist::TaskListTool;
pub use todo::{TodoWriteTool, TODO_STORE_KEY};
pub use web::{SearchMethod, WebFetchTool, WebSearchConfig, WebSearchConfigError, WebSearchTool};

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
    use crate::harness::{HarnessSignal, SandboxProvider, ToolOutput};
    use crate::model::ToolCall;
    use crate::tool_registry::mock::{test_ctx, AllowAllSandbox};
    use crate::tool_registry::Tool;
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
            "edit_file" => ok::<EditFileParams>(input),
            "grep" => {
                if !ok::<GrepParams>(input) {
                    return false;
                }
                let p: GrepParams = serde_json::from_value(input.clone()).unwrap();
                regex::Regex::new(&p.pattern).is_ok()
            }
            "send_message" => ok::<SendMessageParams>(input),
            "memory" => ok::<MemoryToolParams>(input),
            "todo_write" => ok::<TodoWriteParams>(input),
            "web_fetch" => ok::<WebFetchParams>(input),
            "web_search" => ok::<WebSearchParams>(input),
            "enter_plan_mode" => ok::<EnterPlanModeParams>(input),
            "exit_plan_mode" => ok::<ExitPlanModeParams>(input),
            "ask_user_question" => ok::<AskUserQuestionParams>(input),
            "abort" => ok::<AbortParams>(input),
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

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/tools")
            .join(name)
    }

    fn call(name: &str, input: Value) -> ToolCall {
        ToolCall {
            id: "fx".into(),
            name: name.into(),
            input,
        }
    }

    // ---- edit_file_cases.json ----
    #[derive(Deserialize)]
    struct EditExpected {
        kind: String,
        #[serde(default)]
        final_content: Option<String>,
        #[serde(default)]
        recoverable: Option<bool>,
        #[serde(default)]
        reason: Option<String>,
    }
    #[derive(Deserialize)]
    struct EditCase {
        name: String,
        initial_content: String,
        old_string: String,
        new_string: String,
        expected: EditExpected,
    }

    #[tokio::test]
    async fn fixture_replay_edit_file_cases() {
        use crate::tools::edit::EditFileTool;
        use tempfile::TempDir;
        let data = std::fs::read_to_string(fixture_path("edit_file_cases.json")).unwrap();
        let cases: Vec<EditCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty());
        let sb = AllowAllSandbox;
        for c in cases {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("f.txt");
            tokio::fs::write(&p, &c.initial_content).await.unwrap();
            let out = EditFileTool::new()
                .execute(
                    &call(
                        "edit_file",
                        serde_json::json!({
                            "path": p.to_str().unwrap(),
                            "old_string": c.old_string,
                            "new_string": c.new_string,
                        }),
                    ),
                    &sb,
                    &test_ctx(),
                )
                .await;
            match (&out, c.expected.kind.as_str()) {
                (ToolOutput::Success { .. }, "success") => {
                    let got = tokio::fs::read_to_string(&p).await.unwrap();
                    assert_eq!(got, c.expected.final_content.unwrap(), "{}", c.name);
                }
                (
                    ToolOutput::Error {
                        recoverable,
                        message,
                    },
                    "error",
                ) => {
                    assert_eq!(*recoverable, c.expected.recoverable.unwrap(), "{}", c.name);
                    match c.expected.reason.as_deref() {
                        Some("not_found") => assert!(message.contains("not found"), "{}", c.name),
                        Some("not_unique") => assert!(message.contains("not unique"), "{}", c.name),
                        _ => {}
                    }
                }
                (other, k) => panic!("{}: expected {k}, got {other:?}", c.name),
            }
        }
    }

    // ---- grep_output_modes.json ----
    #[derive(Deserialize)]
    struct GrepModeCase {
        name: String,
        files: std::collections::BTreeMap<String, String>,
        pattern: String,
        output_mode: String,
        expected_lines: usize,
        expected_contains: Vec<String>,
    }

    #[tokio::test]
    async fn fixture_replay_grep_output_modes() {
        use crate::tools::search::GrepTool;
        use tempfile::TempDir;
        let data = std::fs::read_to_string(fixture_path("grep_output_modes.json")).unwrap();
        let cases: Vec<GrepModeCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty());
        let sb = AllowAllSandbox;
        for c in cases {
            let dir = TempDir::new().unwrap();
            for (fname, content) in &c.files {
                tokio::fs::write(dir.path().join(fname), content)
                    .await
                    .unwrap();
            }
            let out = GrepTool::new()
                .execute(
                    &call(
                        "grep",
                        serde_json::json!({
                            "pattern": c.pattern,
                            "path": dir.path().to_str().unwrap(),
                            "recursive": true,
                            "output_mode": c.output_mode,
                        }),
                    ),
                    &sb,
                    &test_ctx(),
                )
                .await;
            match out {
                ToolOutput::Success { content, .. } => {
                    assert_eq!(content.lines().count(), c.expected_lines, "{}", c.name);
                    for needle in &c.expected_contains {
                        assert!(content.contains(needle), "{}: missing {needle}", c.name);
                    }
                }
                other => panic!("{}: {other:?}", c.name),
            }
        }
    }

    // ---- send_message_event.json ----
    #[derive(Deserialize)]
    struct SendExpectedOutput {
        kind: String,
        #[serde(default)]
        content: Option<String>,
        #[serde(default)]
        recoverable: Option<bool>,
    }
    #[derive(Deserialize)]
    struct SendEvent {
        content: String,
    }
    #[derive(Deserialize)]
    struct SendCase {
        name: String,
        input: Value,
        expected_tool_output: SendExpectedOutput,
        #[serde(default)]
        expected_stream_event: Option<SendEvent>,
    }

    #[tokio::test]
    async fn fixture_replay_send_message_event() {
        use crate::tools::message::SendMessageTool;
        let data = std::fs::read_to_string(fixture_path("send_message_event.json")).unwrap();
        let cases: Vec<SendCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty());
        let sb = AllowAllSandbox;
        for c in cases {
            let out = SendMessageTool::new()
                .execute(&call("send_message", c.input.clone()), &sb, &test_ctx())
                .await;
            match (&out, c.expected_tool_output.kind.as_str()) {
                (ToolOutput::Success { content, .. }, "success") => {
                    assert_eq!(
                        content,
                        &c.expected_tool_output.content.unwrap(),
                        "{}",
                        c.name
                    );
                    // The loop derives the UserMessage event content from this
                    // success content (see harness run_react).
                    let ev_content = c.expected_stream_event.unwrap().content;
                    assert_eq!(content, &ev_content, "{}", c.name);
                }
                (ToolOutput::Error { recoverable, .. }, "error") => {
                    assert_eq!(
                        *recoverable,
                        c.expected_tool_output.recoverable.unwrap(),
                        "{}",
                        c.name
                    );
                    assert!(c.expected_stream_event.is_none(), "{}", c.name);
                }
                (other, k) => panic!("{}: expected {k}, got {other:?}", c.name),
            }
        }
    }

    // ---- escalation_tools.json ----
    #[derive(Deserialize)]
    struct EscExpected {
        tool_output_kind: String,
        #[serde(default)]
        signal: Option<Value>,
        #[serde(default)]
        question: Option<String>,
        #[serde(default)]
        options: Option<Vec<String>>,
    }
    #[derive(Deserialize)]
    struct EscCase {
        name: String,
        tool: String,
        input: Value,
        expected: EscExpected,
    }

    #[tokio::test]
    async fn fixture_replay_escalation_tools() {
        use crate::tools::control::{
            AbortTool, AskUserQuestionTool, EnterPlanModeTool, ExitPlanModeTool,
        };
        let data = std::fs::read_to_string(fixture_path("escalation_tools.json")).unwrap();
        let cases: Vec<EscCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty());
        let sb = AllowAllSandbox;
        for c in cases {
            let out = match c.tool.as_str() {
                "enter_plan_mode" => {
                    EnterPlanModeTool::new()
                        .execute(&call(&c.tool, c.input.clone()), &sb, &test_ctx())
                        .await
                }
                "exit_plan_mode" => {
                    ExitPlanModeTool::new()
                        .execute(&call(&c.tool, c.input.clone()), &sb, &test_ctx())
                        .await
                }
                "ask_user_question" => {
                    AskUserQuestionTool::new()
                        .execute(&call(&c.tool, c.input.clone()), &sb, &test_ctx())
                        .await
                }
                "abort" => {
                    AbortTool::new()
                        .execute(&call(&c.tool, c.input.clone()), &sb, &test_ctx())
                        .await
                }
                other => panic!("unknown tool {other}"),
            };
            match c.expected.tool_output_kind.as_str() {
                "escalate" => match out {
                    ToolOutput::Escalate { signal } => {
                        let got = serde_json::to_value(&signal).unwrap();
                        assert_eq!(got, c.expected.signal.unwrap(), "{}", c.name);
                        // Sanity: signal deserializes back to a HarnessSignal.
                        let _back: HarnessSignal = serde_json::from_value(got).unwrap();
                    }
                    other => panic!("{}: expected Escalate, got {other:?}", c.name),
                },
                "awaiting_clarification" => match out {
                    ToolOutput::AwaitingClarification { question, options } => {
                        assert_eq!(question, c.expected.question.unwrap(), "{}", c.name);
                        assert_eq!(options, c.expected.options, "{}", c.name);
                    }
                    other => panic!("{}: expected AwaitingClarification, got {other:?}", c.name),
                },
                k => panic!("{}: unknown expected kind {k}", c.name),
            }
        }
    }

    // ---- todo_write.json ----
    #[derive(Deserialize)]
    struct TodoStep {
        input: Value,
        expected_persisted: Vec<crate::tools::params::TodoItem>,
    }
    #[derive(Deserialize)]
    struct TodoScenario {
        name: String,
        steps: Vec<TodoStep>,
    }

    #[tokio::test]
    async fn fixture_replay_todo_write() {
        use crate::harness::SessionId;
        use crate::storage::{InMemoryStorageProvider, MemoryStore, RunStore};
        use crate::tool_registry::ToolContext;
        use crate::tools::params::TodoItem;
        use crate::tools::todo::{TodoWriteTool, TODO_STORE_KEY};
        use std::sync::Arc;
        let data = std::fs::read_to_string(fixture_path("todo_write.json")).unwrap();
        let scenarios: Vec<TodoScenario> = serde_json::from_str(&data).unwrap();
        assert!(!scenarios.is_empty());
        let sb = AllowAllSandbox;
        for sc in scenarios {
            let run_store: Arc<dyn RunStore> = Arc::new(InMemoryStorageProvider::new());
            let memory_store: Arc<dyn MemoryStore> = Arc::new(InMemoryStorageProvider::new());
            let ctx = ToolContext::new(SessionId::new("fx"), run_store.clone(), memory_store);
            let tool = TodoWriteTool::new();
            for step in &sc.steps {
                let out = tool
                    .execute(&call("todo_write", step.input.clone()), &sb, &ctx)
                    .await;
                assert!(
                    matches!(out, ToolOutput::Success { .. }),
                    "{}: {out:?}",
                    sc.name
                );
                let blob = run_store
                    .get(&SessionId::new("fx"), TODO_STORE_KEY)
                    .await
                    .unwrap()
                    .expect("persisted");
                let persisted: Vec<TodoItem> = serde_json::from_value(blob).unwrap();
                assert_eq!(persisted, step.expected_persisted, "{}", sc.name);
            }
        }
    }

    // ---- read_file_range.json (#132) ----
    #[derive(Deserialize)]
    struct ReadRangeExpected {
        kind: String,
        #[serde(default)]
        content: Option<String>,
        #[serde(default)]
        recoverable: Option<bool>,
        #[serde(default)]
        message_contains: Option<String>,
    }
    #[derive(Deserialize)]
    struct ReadRangeCase {
        name: String,
        initial_content: String,
        params: Value,
        expected: ReadRangeExpected,
    }

    #[tokio::test]
    async fn fixture_replay_read_file_range() {
        use crate::tools::fs::ReadFileTool;
        use tempfile::TempDir;
        let data = std::fs::read_to_string(fixture_path("read_file_range.json")).unwrap();
        let cases: Vec<ReadRangeCase> = serde_json::from_str(&data).unwrap();
        assert!(!cases.is_empty());
        let sb = AllowAllSandbox;
        for c in cases {
            let dir = TempDir::new().unwrap();
            let p = dir.path().join("f.txt");
            tokio::fs::write(&p, &c.initial_content).await.unwrap();
            // The fixture uses "<FIXTURE_PATH>" as a placeholder; swap in the
            // real temp-file path at runtime (same pattern as edit_file_cases).
            let mut input = c.params.clone();
            input["path"] = Value::String(p.to_str().unwrap().to_string());
            let out = ReadFileTool::new()
                .execute(&call("read_file", input), &sb, &test_ctx())
                .await;
            match (&out, c.expected.kind.as_str()) {
                (ToolOutput::Success { content, .. }, "success") => {
                    assert_eq!(content, &c.expected.content.unwrap(), "{}", c.name);
                }
                (
                    ToolOutput::Error {
                        recoverable,
                        message,
                    },
                    "error",
                ) => {
                    assert_eq!(*recoverable, c.expected.recoverable.unwrap(), "{}", c.name);
                    if let Some(needle) = c.expected.message_contains.as_deref() {
                        assert!(
                            message.contains(needle),
                            "{}: message {message:?} missing {needle:?}",
                            c.name
                        );
                    }
                }
                (other, k) => panic!("{}: expected {k}, got {other:?}", c.name),
            }
        }
    }
}
