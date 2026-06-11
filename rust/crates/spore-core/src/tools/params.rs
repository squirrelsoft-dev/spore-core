//! Per-tool serde input structs and parsing helper.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::model::ToolCall;
use crate::tools::error::ToolExecutionError;

/// Parse `call.input` into a typed parameter struct. Any deserialization
/// failure is mapped to [`ToolExecutionError::InvalidParameters`].
pub fn parse_params<T: serde::de::DeserializeOwned>(
    call: &ToolCall,
) -> Result<T, ToolExecutionError> {
    serde_json::from_value::<T>(call.input.clone()).map_err(|e| {
        ToolExecutionError::InvalidParameters {
            reason: e.to_string(),
        }
    })
}

// ---------- Filesystem ----------

/// Parameters for [`crate::tools::fs::ReadFileTool`]. With all three optional
/// params at their defaults (`offset = 1`, `length = 0`, `line_numbers =
/// false`) the read is **byte-identical** to reading the whole file — no
/// header, no line numbers (#132). Any non-default param prepends a one-line
/// header `[lines {start}–{end} of {total}]` (U+2013 en-dash).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileParams {
    pub path: String,
    /// 1-indexed start line. Default `1` (read from the beginning).
    #[serde(default = "default_read_offset")]
    pub offset: u64,
    /// Max lines to return. Default `0` = no limit (read to EOF). A `length`
    /// that runs past EOF silently returns through the last line.
    #[serde(default)]
    pub length: u64,
    /// When `true`, prefix each returned line with its 1-indexed number,
    /// right-padded to the digit-width of the file's total line count.
    #[serde(default)]
    pub line_numbers: bool,
}
fn default_read_offset() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteFileParams {
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub append: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListDirParams {
    pub path: String,
    #[serde(default)]
    pub recursive: bool,
    /// When `false` (default), a recursive walk honors `.gitignore`/`.ignore`
    /// and skips VCS dirs (`.git/`) — keeping the listing focused on source and
    /// out of the way of build artifacts (`target/`, `node_modules/`). Set
    /// `true` to walk everything, ignored files included.
    #[serde(default)]
    pub include_ignored: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteFileParams {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveFileParams {
    pub src: String,
    pub dst: String,
}

// ---------- Exec ----------

/// Parameters for the shell-free [`crate::tools::exec::ExecTool`]: a program
/// name plus verbatim argument vector. No shell is involved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecParams {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Timeout in whole seconds.
    #[serde(default)]
    pub timeout: Option<u64>,
}

/// Parameters for the real [`crate::tools::exec::BashCommandTool`]: a single
/// shell `script` run via `/bin/sh -c`, with an optional working directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCommandParams {
    pub script: String,
    #[serde(default)]
    pub working_dir: Option<String>,
    /// Timeout in whole seconds.
    #[serde(default)]
    pub timeout: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTestsParams {
    pub command: String,
    pub working_dir: String,
    #[serde(default)]
    pub timeout: Option<u64>,
}

// ---------- Search ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepFilesParams {
    pub pattern: String,
    pub path: String,
    #[serde(default)]
    pub recursive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindFilesParams {
    pub glob: String,
    pub path: String,
}

// ---------- Git ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitLogParams {
    #[serde(default = "default_log_n")]
    pub n: u32,
    #[serde(default = "default_log_format")]
    pub format: String,
}
fn default_log_n() -> u32 {
    20
}
fn default_log_format() -> String {
    "oneline".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitDiffParams {
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub to: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCommitParams {
    pub message: String,
    #[serde(default)]
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GitStatusParams {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitResetParams {
    pub target: String,
    pub mode: super::git::GitResetMode,
}

// ---------- HTTP ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpGetParams {
    pub url: String,
    #[serde(default)]
    pub headers: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpPostParams {
    pub url: String,
    pub body: Value,
    #[serde(default)]
    pub headers: Option<Map<String, Value>>,
}

// ---------- EditFile (#81, new) ----------

/// Parameters for [`crate::tools::edit::EditFileTool`]: replace the FIRST and
/// ONLY occurrence of `old_string` with `new_string` in the file at `path`.
/// The match must be unique — an absent or non-unique `old_string` is a
/// recoverable error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditFileParams {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
}

// ---------- Grep (#81, new — output modes) ----------

/// Output modes for [`crate::tools::search::GrepTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GrepOutputMode {
    /// Each matching line as `path:line:text` (default).
    #[default]
    Content,
    /// Distinct file paths that contain at least one match.
    FilesWithMatches,
    /// `path:count` for each file with matches.
    Count,
}

/// Parameters for the net-new [`crate::tools::search::GrepTool`]. Distinct from
/// [`GrepFilesParams`]: adds `output_mode` and `context_lines`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepParams {
    pub pattern: String,
    pub path: String,
    #[serde(default)]
    pub recursive: bool,
    #[serde(default)]
    pub output_mode: GrepOutputMode,
    /// Lines of context to show before and after each match (default: 0).
    /// When 0 the output is byte-identical to the pre-context behaviour.
    /// When > 0 the output uses standard `grep -C N` format:
    /// match lines use `:` separator, context lines use `-`, and non-adjacent
    /// groups are separated by a `--` line.
    #[serde(default)]
    pub context_lines: u32,
}

// ---------- SendMessage (#81, new) ----------

/// Parameters for [`crate::tools::message::SendMessageTool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendMessageParams {
    pub content: String,
}

// ---------- TodoWrite (#81, new) ----------

/// A single todo item managed by [`crate::tools::todo::TodoWriteTool`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

/// Parameters for [`crate::tools::todo::TodoWriteTool`]: the agent supplies the
/// FULL desired todo list, which replaces the persisted list wholesale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoWriteParams {
    pub todos: Vec<TodoItem>,
}

// ---------- WebFetch / WebSearch (#81, new) ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebFetchParams {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchParams {
    pub query: String,
}

// ---------- Tier 3: plan / clarify / abort (#81, new) ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterPlanModeParams {
    #[serde(default)]
    pub context: String,
}

/// Parameters for [`crate::tools::plan::ExitPlanModeTool`]. The agent supplies
/// the plan as a structured object that deserializes DIRECTLY into the existing
/// [`PlanArtifact`](crate::hooks::PlanArtifact) (issue #81, Q4a — no stub).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitPlanModeParams {
    pub plan: crate::hooks::PlanArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AskUserQuestionParams {
    pub question: String,
    #[serde(default)]
    pub options: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbortParams {
    pub reason: String,
}

// ---------- Subagent ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentParams {
    pub instruction: String,
}

// ---------- TaskList (#71) ----------

/// Parameters for the [`crate::tools::tasklist::TaskListTool`], internally
/// tagged on `action`. Each variant carries exactly the fields that action
/// consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum TaskListParams {
    AddTask {
        description: String,
        /// Ids of tasks that must be `Completed` before this one runs (#118).
        /// Optional; defaults to empty. Validated by [`crate::tasklist::TaskList::add`].
        #[serde(default)]
        blockers: Vec<u32>,
    },
    UpdateTask {
        id: u32,
        #[serde(default)]
        status: Option<crate::tasklist::TaskStatus>,
        #[serde(default)]
        description: Option<String>,
    },
    CompleteTask {
        id: u32,
    },
    ListTasks {},
}

// ---------- Memory (#82) ----------

/// Parameters for the [`crate::tools::memory::MemoryTool`], internally tagged on
/// `operation`. `scope` is an explicit field on BOTH variants. `write` carries
/// the entry payload; `read` carries the recency `limit` and a `merged` flag.
///
/// `StorageScope::Local` is accepted by serde (so a bad-scope call reaches the
/// tool body) but rejected at runtime with a recoverable error — the advertised
/// schema enum omits `local`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum MemoryToolParams {
    Write {
        scope: crate::storage::StorageScope,
        role: String,
        content: String,
        /// Free-form metadata stored on the entry; defaults to `{}`.
        #[serde(default = "memory_empty_metadata")]
        metadata: Value,
    },
    Read {
        scope: crate::storage::StorageScope,
        /// When `true`, return the cross-scope merged view (User ∪ Project,
        /// newest-first, no dedup) instead of just `scope`. Defaults to `false`.
        #[serde(default)]
        merged: bool,
        /// Recency cap; most-recent `limit` entries, newest-first. Defaults to 50.
        #[serde(default = "memory_default_limit")]
        limit: usize,
    },
}

fn memory_empty_metadata() -> Value {
    Value::Object(Map::new())
}

fn memory_default_limit() -> usize {
    50
}
