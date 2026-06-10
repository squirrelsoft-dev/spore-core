//! Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile.

use serde_json::json;

use crate::harness::{BoxFut, Operation, SandboxProvider, ToolOutput};
use crate::model::ToolCall;
use crate::tool_registry::{Tool, ToolAnnotations, ToolSchema};
use crate::tools::error::ToolExecutionError;
use crate::tools::params::{
    parse_params, DeleteFileParams, ListDirParams, MoveFileParams, ReadFileParams, WriteFileParams,
};
use crate::tools::{finish_with_possible_truncation, LARGE_OUTPUT_THRESHOLD};

// ============================================================================
// ReadFile
// ============================================================================

pub struct ReadFileTool;

impl ReadFileTool {
    pub const NAME: &'static str = "read_file";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Read a file's contents".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                idempotent: true,
                ..Default::default()
            },
        }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ReadFileTool {
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
            let params: ReadFileParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let resolved = match sandbox.resolve_path(&params.path, Operation::Read).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            match tokio::fs::read_to_string(&resolved).await {
                Ok(content) => finish_with_possible_truncation(content, &call.id, sandbox).await,
                Err(e) => ToolOutput::Error {
                    message: format!("read failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// WriteFile
// ============================================================================

pub struct WriteFileTool;

impl WriteFileTool {
    pub const NAME: &'static str = "write_file";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description:
                "Write content to a file (overwrites by default; set append=true to append)".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for WriteFileTool {
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
            let params: WriteFileParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let resolved = match sandbox.resolve_path(&params.path, Operation::Write).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            let bytes = params.content.len();
            let result = if params.append {
                use tokio::io::AsyncWriteExt;
                match tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&resolved)
                    .await
                {
                    Ok(mut f) => f.write_all(params.content.as_bytes()).await,
                    Err(e) => Err(e),
                }
            } else {
                tokio::fs::write(&resolved, params.content.as_bytes()).await
            };
            match result {
                Ok(_) => ToolOutput::Success {
                    content: format!("wrote {bytes} bytes to {}", params.path),
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
// ListDir
// ============================================================================

pub struct ListDirTool;

impl ListDirTool {
    pub const NAME: &'static str = "list_dir";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "List directory entries (optionally recursive). A recursive \
                          listing honors .gitignore and skips VCS/build dirs by default; \
                          set include_ignored to walk everything."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "include_ignored": {
                        "type": "boolean",
                        "description": "Recursive only: when true, include .gitignore-matched and VCS files (default false).",
                    },
                },
                "required": ["path"],
            }),
            annotations: ToolAnnotations {
                read_only: true,
                ..Default::default()
            },
        }
    }
}

impl Default for ListDirTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ListDirTool {
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
            let params: ListDirParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let resolved = match sandbox.resolve_path(&params.path, Operation::Read).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            // Emit paths relative to the workspace root so each entry can be
            // fed straight back into read_file/write_file. The sandbox treats
            // every input path as root-relative, so absolute paths would not
            // round-trip (see #93). `resolved` is the absolute path of the
            // listed directory (= root-relative `params.path`); each entry is
            // under it. Relativize against `resolved`, then re-anchor onto the
            // root-relative `params.path`.
            let listed = std::path::Path::new(&params.path);
            let to_root_relative = |entry_path: &std::path::Path| -> Option<String> {
                // Path of the entry relative to the listed directory.
                let rel_to_listed = entry_path.strip_prefix(&resolved).ok()?;
                // Skip the listed directory itself (WalkDir yields it first).
                if rel_to_listed.as_os_str().is_empty() {
                    return None;
                }
                // Re-anchor onto the caller-supplied (root-relative) path.
                let anchored = listed.join(rel_to_listed);
                // Drop a leading `./` so `.`/empty inputs yield bare names.
                let normalized: std::path::PathBuf = anchored
                    .components()
                    .filter(|c| !matches!(c, std::path::Component::CurDir))
                    .collect();
                Some(normalized.display().to_string())
            };
            let mut entries: Vec<String> = Vec::new();
            if params.recursive {
                // By default the walk honors `.gitignore`/`.ignore` and skips
                // VCS dirs, so a recursive listing stays focused on source and
                // is not buried under build artifacts (`target/`, `node_modules/`)
                // — which alphabetically precede and would truncate away real
                // source before the model ever sees it. `include_ignored` opts
                // back into walking everything.
                let honor_ignores = !params.include_ignored;
                let walker = ignore::WalkBuilder::new(&resolved)
                    .standard_filters(honor_ignores)
                    // Respect `.gitignore` even when the listed tree is not a git
                    // checkout (the default only honors it inside a `.git` repo).
                    .require_git(false)
                    .build();
                for entry in walker.filter_map(Result::ok) {
                    if let Some(rel) = to_root_relative(entry.path()) {
                        entries.push(rel);
                    }
                }
            } else {
                match tokio::fs::read_dir(&resolved).await {
                    Ok(mut rd) => loop {
                        match rd.next_entry().await {
                            Ok(Some(e)) => {
                                if let Some(rel) = to_root_relative(&e.path()) {
                                    entries.push(rel);
                                }
                            }
                            Ok(None) => break,
                            Err(e) => {
                                return ToolOutput::Error {
                                    message: format!("read_dir failed: {e}"),
                                    recoverable: true,
                                };
                            }
                        }
                    },
                    Err(e) => {
                        return ToolOutput::Error {
                            message: format!("read_dir failed: {e}"),
                            recoverable: true,
                        };
                    }
                }
            }
            entries.sort();
            let content = entries.join("\n");
            if content.len() > LARGE_OUTPUT_THRESHOLD {
                finish_with_possible_truncation(content, &call.id, sandbox).await
            } else {
                ToolOutput::Success {
                    content,
                    truncated: false,
                }
            }
        })
    }
}

// ============================================================================
// DeleteFile
// ============================================================================

pub struct DeleteFileTool;

impl DeleteFileTool {
    pub const NAME: &'static str = "delete_file";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Delete a file".into(),
            parameters: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for DeleteFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for DeleteFileTool {
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
            let params: DeleteFileParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let resolved = match sandbox.resolve_path(&params.path, Operation::Write).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            match tokio::fs::remove_file(&resolved).await {
                Ok(_) => ToolOutput::Success {
                    content: format!("deleted {}", params.path),
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("delete failed: {e}"),
                    recoverable: true,
                },
            }
        })
    }
}

// ============================================================================
// MoveFile
// ============================================================================

pub struct MoveFileTool;

impl MoveFileTool {
    pub const NAME: &'static str = "move_file";
    pub fn new() -> Self {
        Self
    }
    pub fn schema() -> ToolSchema {
        ToolSchema {
            name: Self::NAME.into(),
            description: "Move/rename a file".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                },
                "required": ["src", "dst"],
            }),
            annotations: ToolAnnotations {
                destructive: true,
                ..Default::default()
            },
        }
    }
}

impl Default for MoveFileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for MoveFileTool {
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
            let params: MoveFileParams = match parse_params(call) {
                Ok(p) => p,
                Err(e) => return e.into(),
            };
            let src = match sandbox.resolve_path(&params.src, Operation::Write).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            let dst = match sandbox.resolve_path(&params.dst, Operation::Write).await {
                Ok(p) => p,
                Err(v) => return ToolExecutionError::SandboxViolation(v).into(),
            };
            match tokio::fs::rename(&src, &dst).await {
                Ok(_) => ToolOutput::Success {
                    content: format!("moved {} -> {}", params.src, params.dst),
                    truncated: false,
                },
                Err(e) => ToolOutput::Error {
                    message: format!("move failed: {e}"),
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

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input,
        }
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        let sb = AllowAllSandbox;
        let w = WriteFileTool::new()
            .execute(
                &call(
                    "write_file",
                    json!({"path": path.to_str().unwrap(), "content": "hello"}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        assert!(matches!(w, ToolOutput::Success { .. }));
        let r = ReadFileTool::new()
            .execute(
                &call("read_file", json!({"path": path.to_str().unwrap()})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => assert_eq!(content, "hello"),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn append_mode_concatenates() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        let sb = AllowAllSandbox;
        WriteFileTool::new()
            .execute(
                &call(
                    "write_file",
                    json!({"path": path.to_str().unwrap(), "content": "a"}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        WriteFileTool::new()
            .execute(
                &call(
                    "write_file",
                    json!({"path": path.to_str().unwrap(), "content": "b", "append": true}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(content, "ab");
    }

    #[tokio::test]
    async fn list_dir_sorted() {
        let dir = TempDir::new().unwrap();
        tokio::fs::write(dir.path().join("z"), "").await.unwrap();
        tokio::fs::write(dir.path().join("a"), "").await.unwrap();
        tokio::fs::write(dir.path().join("m"), "").await.unwrap();
        let sb = AllowAllSandbox;
        let r = ListDirTool::new()
            .execute(
                &call("list_dir", json!({"path": dir.path().to_str().unwrap()})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => {
                let lines: Vec<&str> = content.lines().collect();
                assert_eq!(lines.len(), 3);
                let mut sorted = lines.clone();
                sorted.sort();
                assert_eq!(lines, sorted);
            }
            other => panic!("{other:?}"),
        }
    }

    /// Regression for #93: every entry list_dir returns must round-trip
    /// straight back into read_file under the *real* WorkspaceScopedSandbox,
    /// which treats all input paths as root-relative. Absolute paths (the old
    /// behavior) would be rejected as PathEscape.
    #[tokio::test]
    async fn list_dir_entries_roundtrip_through_workspace_sandbox() {
        use crate::sandbox::{WorkspaceConfig, WorkspaceScopedSandbox};
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        tokio::fs::write(root.join("a.txt"), "alpha").await.unwrap();
        tokio::fs::write(root.join("b.txt"), "beta").await.unwrap();
        tokio::fs::create_dir(root.join("sub")).await.unwrap();
        tokio::fs::write(root.join("sub").join("c.txt"), "gamma")
            .await
            .unwrap();
        let sb = WorkspaceScopedSandbox::new(WorkspaceConfig {
            root: root.clone(),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        })
        .unwrap();

        // Recursive so we exercise both top-level files and a nested file.
        let r = ListDirTool::new()
            .execute(
                &call("list_dir", json!({"path": ".", "recursive": true})),
                &sb,
                &test_ctx(),
            )
            .await;
        let entries: Vec<String> = match r {
            ToolOutput::Success { content, .. } => content.lines().map(str::to_string).collect(),
            other => panic!("list_dir failed: {other:?}"),
        };
        assert!(
            entries.iter().any(|e| e == "a.txt"),
            "expected bare root-relative names, got {entries:?}"
        );
        assert!(
            entries.iter().any(|e| e == "sub/c.txt"),
            "expected nested entry as sub/c.txt, got {entries:?}"
        );
        assert!(
            !entries.iter().any(|e| e.is_empty() || e == "."),
            "must not emit the listed dir itself, got {entries:?}"
        );

        // The actual bug check: feed each entry straight into read_file.
        for entry in &entries {
            let rr = ReadFileTool::new()
                .execute(&call("read_file", json!({"path": entry})), &sb, &test_ctx())
                .await;
            match rr {
                ToolOutput::Success { .. } => {}
                ToolOutput::Error { message, .. } => {
                    // A directory entry (e.g. `sub`) reads as an error but must
                    // NOT be a sandbox violation — that's the regression.
                    assert!(
                        !message.contains("Sandbox") && !message.contains("PathEscape"),
                        "entry {entry:?} did not round-trip: {message}"
                    );
                }
                other => panic!("unexpected read_file output for {entry:?}: {other:?}"),
            }
        }
    }

    /// A recursive listing honors `.gitignore` by default — build artifacts
    /// (`target/`) that alphabetically precede real source must not flood/
    /// truncate the listing — but `include_ignored: true` opts back in.
    #[tokio::test]
    async fn list_dir_recursive_respects_gitignore() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        tokio::fs::write(root.join(".gitignore"), "target/\n")
            .await
            .unwrap();
        tokio::fs::write(root.join("lib.rs"), "").await.unwrap();
        tokio::fs::create_dir(root.join("target")).await.unwrap();
        tokio::fs::write(root.join("target").join("junk.rs"), "")
            .await
            .unwrap();
        let list = |include_ignored: bool| {
            let path = root.to_str().unwrap().to_string();
            async move {
                let mut params = json!({"path": path, "recursive": true});
                if include_ignored {
                    params["include_ignored"] = json!(true);
                }
                let r = ListDirTool::new()
                    .execute(&call("list_dir", params), &AllowAllSandbox, &test_ctx())
                    .await;
                match r {
                    ToolOutput::Success { content, .. } => content,
                    other => panic!("{other:?}"),
                }
            }
        };

        // Default: target/ (gitignored) is excluded; source survives.
        let default = list(false).await;
        assert!(default.contains("lib.rs"), "source missing: {default:?}");
        assert!(
            !default.contains("target"),
            "gitignored build dir leaked into default listing: {default:?}"
        );

        // Opt-in: include_ignored walks everything, build artifacts included.
        let everything = list(true).await;
        assert!(
            everything.contains("target/junk.rs"),
            "include_ignored should surface ignored files: {everything:?}"
        );
    }

    #[tokio::test]
    async fn delete_missing_is_recoverable() {
        let sb = AllowAllSandbox;
        let r = DeleteFileTool::new()
            .execute(
                &call("delete_file", json!({"path": "/no/such/path/here"})),
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
    async fn move_file_renames() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("s");
        let dst = dir.path().join("d");
        tokio::fs::write(&src, "hi").await.unwrap();
        let sb = AllowAllSandbox;
        let r = MoveFileTool::new()
            .execute(
                &call(
                    "move_file",
                    json!({"src": src.to_str().unwrap(), "dst": dst.to_str().unwrap()}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        assert!(matches!(r, ToolOutput::Success { .. }));
        assert!(!src.exists());
        assert!(dst.exists());
    }

    #[tokio::test]
    async fn read_missing_in_workspace_file_is_recoverable_not_found() {
        // Regression for #63: reading a not-yet-created file *inside* the
        // workspace must surface a recoverable not-found, not a sandbox
        // PathEscape, end to end through the real WorkspaceScopedSandbox.
        use crate::sandbox::{WorkspaceConfig, WorkspaceScopedSandbox};
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(WorkspaceConfig {
            root: root.clone(),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        })
        .unwrap();
        let r = ReadFileTool::new()
            .execute(
                &call("read_file", json!({"path": "output.txt"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error {
                recoverable,
                message,
            } => {
                assert!(recoverable, "missing-file read must be recoverable");
                assert!(
                    message.contains("read failed"),
                    "expected a not-found read error, got: {message}"
                );
            }
            other => panic!("expected recoverable not-found error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_outside_workspace_is_path_escape() {
        // Counterpart to the above: a path that resolves outside the root is
        // still a sandbox violation, even when the file does not exist.
        use crate::sandbox::{WorkspaceConfig, WorkspaceScopedSandbox};
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(WorkspaceConfig {
            root: root.clone(),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        })
        .unwrap();
        let r = ReadFileTool::new()
            .execute(
                &call("read_file", json!({"path": "../nonexistent_secret"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Error { message, .. } => {
                assert!(
                    message.to_lowercase().contains("escape")
                        || message.to_lowercase().contains("sandbox"),
                    "expected a sandbox/path-escape error, got: {message}"
                );
            }
            other => panic!("expected sandbox violation error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_params_returns_recoverable_error() {
        let sb = AllowAllSandbox;
        let r = ReadFileTool::new()
            .execute(&call("read_file", json!({})), &sb, &test_ctx())
            .await;
        match r {
            ToolOutput::Error { recoverable, .. } => assert!(recoverable),
            other => panic!("{other:?}"),
        }
    }
}
