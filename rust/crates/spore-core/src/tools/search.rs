//! Search tools: GrepFiles, FindFiles.

use std::path::Path;

use serde_json::json;

use crate::harness::{BoxFut, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::finish_with_possible_truncation;
use crate::tools::params::{parse_params, FindFilesParams, GrepFilesParams};

// ============================================================================
// GrepFiles
// ============================================================================

pub struct GrepFilesTool;

impl GrepFilesTool {
    pub const NAME: &'static str = "grep_files";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Search files for a regex pattern".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["pattern", "path"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for GrepFilesTool {
    fn default() -> Self {
        Self::new()
    }
}

fn scan_file(path: &Path, re: &regex::Regex, out: &mut Vec<(String, usize, String)>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    for (i, line) in content.lines().enumerate() {
        if re.is_match(line) {
            out.push((path.display().to_string(), i + 1, line.to_string()));
        }
    }
}

impl Tool for GrepFilesTool {
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
            let params: GrepFilesParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let re = match regex::Regex::new(&params.pattern) {
                Ok(r) => r,
                Err(e) => {
                    return ToolExecutionError::InvalidParameters {
                        reason: format!("invalid regex: {e}"),
                    }
                    .into()
                }
            };
            let root = match sandbox.resolve_path(&params.path).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            let mut matches: Vec<(String, usize, String)> = Vec::new();
            if params.recursive {
                for entry in walkdir::WalkDir::new(&root)
                    .into_iter()
                    .filter_map(Result::ok)
                {
                    if entry.file_type().is_file() {
                        scan_file(entry.path(), &re, &mut matches);
                    }
                }
            } else if root.is_file() {
                scan_file(&root, &re, &mut matches);
            } else if let Ok(rd) = std::fs::read_dir(&root) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_file() {
                        scan_file(&p, &re, &mut matches);
                    }
                }
            }
            matches.sort_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));
            let body = matches
                .iter()
                .map(|(p, n, t)| format!("{p}:{n}:{t}"))
                .collect::<Vec<_>>()
                .join("\n");
            finish_with_possible_truncation(body, &call.id, sandbox).await
        })
    }
}

// ============================================================================
// FindFiles
// ============================================================================

pub struct FindFilesTool;

impl FindFilesTool {
    pub const NAME: &'static str = "find_files";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Find files matching a glob".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "glob": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["glob", "path"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for FindFilesTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for FindFilesTool {
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
            let params: FindFilesParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let root = match sandbox.resolve_path(&params.path).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            let pattern = root.join(&params.glob).display().to_string();
            let mut entries: Vec<String> = Vec::new();
            match glob::glob(&pattern) {
                Ok(paths) => {
                    for p in paths.flatten() {
                        entries.push(p.display().to_string());
                    }
                }
                Err(e) => {
                    return ToolExecutionError::InvalidParameters {
                        reason: format!("invalid glob: {e}"),
                    }
                    .into()
                }
            }
            entries.sort();
            finish_with_possible_truncation(entries.join("\n"), &call.id, sandbox).await
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
    use tempfile::TempDir;

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn grep_finds_matches() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("a.txt");
        std::fs::write(&p, "alpha\nbeta\nalpha2").unwrap();
        let sb = AllowAllSandbox;
        let r = GrepFilesTool::new()
            .execute(
                &call(
                    "grep_files",
                    json!({"pattern": "^alpha", "path": dir.path().to_str().unwrap()}),
                ),
                &sb,
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => {
                assert!(content.contains("alpha"));
                assert!(content.contains("alpha2"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn grep_invalid_regex_returns_invalid_params() {
        let dir = TempDir::new().unwrap();
        let sb = AllowAllSandbox;
        let r = GrepFilesTool::new()
            .execute(
                &call(
                    "grep_files",
                    json!({"pattern": "(unclosed", "path": dir.path().to_str().unwrap()}),
                ),
                &sb,
            )
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn find_files_glob() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.rs"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();
        let sb = AllowAllSandbox;
        let r = FindFilesTool::new()
            .execute(
                &call(
                    "find_files",
                    json!({"glob": "*.rs", "path": dir.path().to_str().unwrap()}),
                ),
                &sb,
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => {
                let lines: Vec<&str> = content.lines().collect();
                assert_eq!(lines.len(), 2);
                let mut sorted = lines.clone();
                sorted.sort();
                assert_eq!(lines, sorted);
            }
            other => panic!("{other:?}"),
        }
    }
}
