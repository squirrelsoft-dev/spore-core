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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadFileParams {
    pub path: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashCommandParams {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
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

// ---------- Subagent ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentParams {
    pub instruction: String,
}
