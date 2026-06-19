//! Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile.

use serde_json::json;

use crate::harness::{BoxFut, Operation, SandboxProvider, SandboxViolation, ToolOutput};
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
            description: "Read a file's contents. Optionally read a line range \
                          (offset is 1-indexed start, length is max lines, 0 = \
                          to EOF) and/or prefix each line with its number via \
                          line_numbers. With no optional params the whole file \
                          is returned verbatim."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed start line (default 1).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Max lines to return; 0 = no limit / read to EOF (default 0).",
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": "Prefix each returned line with its 1-indexed number (default false).",
                    },
                },
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

/// Apply the #132 range/line-number transform to a fully-read file body.
///
/// Returns the content to surface, or a recoverable error message. With all
/// params at their defaults the original `content` is returned unchanged
/// (byte-identical to the pre-#132 behavior). Any non-default param prepends a
/// `[lines {start}–{end} of {total}]\n` header (U+2013 en-dash).
fn apply_read_range(content: &str, params: &ReadFileParams) -> Result<String, String> {
    let is_default = params.offset == 1 && params.length == 0 && !params.line_numbers;
    if is_default {
        return Ok(content.to_string());
    }
    if params.offset == 0 {
        return Err("offset must be \u{2265} 1 (1-indexed)".to_string());
    }
    // Empty file: any params still yield empty content with no header.
    if content.is_empty() {
        return Ok(String::new());
    }
    // `split_inclusive` preserves each line's trailing '\n'; the final line may
    // or may not end in '\n'. This keeps the slice byte-faithful to the source.
    let lines: Vec<&str> = content.split_inclusive('\n').collect();
    let total = lines.len() as u64;
    if params.offset > total {
        return Err(format!(
            "offset {} exceeds file length {}",
            params.offset, total
        ));
    }
    let start = params.offset; // 1-indexed, validated >= 1 and <= total.
    let end = if params.length == 0 {
        total
    } else {
        // offset + length - 1, clamped to total (length past EOF is silent).
        (start + params.length - 1).min(total)
    };
    let start_idx = (start - 1) as usize;
    let end_idx = end as usize; // exclusive
    let selected = &lines[start_idx..end_idx];

    let mut out = String::new();
    out.push_str(&format!("[lines {start}\u{2013}{end} of {total}]\n"));
    if params.line_numbers {
        let width = total.to_string().len();
        for (i, line) in selected.iter().enumerate() {
            let n = start + i as u64;
            out.push_str(&format!("{n:>width$} | {line}"));
        }
    } else {
        for line in selected {
            out.push_str(line);
        }
    }
    Ok(out)
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
                Ok(content) => match apply_read_range(&content, &params) {
                    Ok(out) => finish_with_possible_truncation(out, &call.id, sandbox).await,
                    Err(message) => ToolOutput::Error {
                        message,
                        recoverable: true,
                    },
                },
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
            // Enforce the sandbox's write cap on the payload (reads are capped in
            // `resolve_path`; writes/appends are capped here, since the resolver
            // never sees the content).
            if let Some(limit) = sandbox.max_write_size() {
                if bytes as u64 > limit {
                    return ToolExecutionError::SandboxViolation(SandboxViolation::FileSizeExceeded {
                        path: params.path.clone(),
                        size: bytes as u64,
                        limit,
                    })
                    .into();
                }
            }
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
    async fn write_over_size_cap_is_rejected() {
        use crate::sandbox::{WorkspaceConfig, WorkspaceScopedSandbox};
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(WorkspaceConfig {
            root,
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 8,
        })
        .unwrap();
        // 9 bytes > 8-byte cap → rejected on the write path.
        let over = WriteFileTool::new()
            .execute(
                &call("write_file", json!({"path": "big.txt", "content": "123456789"})),
                &sb,
                &test_ctx(),
            )
            .await;
        match over {
            ToolOutput::SandboxViolation { violation } => assert!(
                matches!(violation, SandboxViolation::FileSizeExceeded { .. }),
                "expected FileSizeExceeded, got {violation:?}"
            ),
            other => panic!("expected size violation, got {other:?}"),
        }
        // 8 bytes == cap → allowed; the appended byte (1 byte ≤ cap) is allowed too.
        let ok = WriteFileTool::new()
            .execute(
                &call("write_file", json!({"path": "ok.txt", "content": "12345678"})),
                &sb,
                &test_ctx(),
            )
            .await;
        assert!(matches!(ok, ToolOutput::Success { .. }), "{ok:?}");
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
        // The tool surfaces the TYPED violation; the harness (not the tool)
        // decides recoverable-vs-halt via SandboxViolationPolicy.
        match r {
            ToolOutput::SandboxViolation { violation } => {
                assert!(
                    matches!(violation, SandboxViolation::PathEscape { .. }),
                    "expected PathEscape, got {violation:?}"
                );
            }
            other => panic!("expected SandboxViolation, got {other:?}"),
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

    #[tokio::test]
    async fn write_outside_workspace_surfaces_typed_violation() {
        // A write that escapes the workspace root surfaces the TYPED violation
        // (`ToolOutput::SandboxViolation`), end to end through the real sandbox.
        // The tool does NOT decide recoverable-vs-halt — the harness does, via
        // SandboxViolationPolicy (see the harness-level policy tests).
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
        let r = WriteFileTool::new()
            .execute(
                &call(
                    "write_file",
                    json!({"path": "../escape.txt", "content": "x"}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::SandboxViolation { violation } => {
                assert!(
                    matches!(violation, SandboxViolation::PathEscape { .. }),
                    "expected PathEscape, got {violation:?}"
                );
            }
            other => panic!("expected SandboxViolation, got {other:?}"),
        }
    }

    // ---- #132: read_file range scan + line numbers ----

    fn read_params(v: serde_json::Value) -> ReadFileParams {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn read_range_defaults_are_byte_identical() {
        let body = "line1\nline2\nline3\n";
        // Only `path` supplied → all three params default → unchanged content.
        let params = read_params(json!({"path": "f"}));
        assert_eq!(apply_read_range(body, &params).unwrap(), body);
    }

    #[test]
    fn read_range_offset_header_runs_to_eof() {
        let body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n";
        let params = read_params(json!({"path": "f", "offset": 3}));
        assert_eq!(
            apply_read_range(body, &params).unwrap(),
            "[lines 3\u{2013}10 of 10]\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
        );
    }

    #[test]
    fn read_range_length_trims_at_eof_silently() {
        let body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n";
        // offset 8 + length 5 would reach line 12, but only 10 lines exist.
        let params = read_params(json!({"path": "f", "offset": 8, "length": 5}));
        assert_eq!(
            apply_read_range(body, &params).unwrap(),
            "[lines 8\u{2013}10 of 10]\nline8\nline9\nline10\n"
        );
    }

    #[test]
    fn read_range_line_numbers_pad_to_total_width() {
        let body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n";
        // total = 10 → width 2 → single-digit numbers are right-padded.
        let params =
            read_params(json!({"path": "f", "offset": 2, "length": 3, "line_numbers": true}));
        assert_eq!(
            apply_read_range(body, &params).unwrap(),
            "[lines 2\u{2013}4 of 10]\n 2 | line2\n 3 | line3\n 4 | line4\n"
        );
    }

    #[test]
    fn read_range_line_numbers_no_pad_when_single_digit_total() {
        let body = "alpha\nbeta\ngamma\n";
        let params = read_params(json!({"path": "f", "line_numbers": true}));
        assert_eq!(
            apply_read_range(body, &params).unwrap(),
            "[lines 1\u{2013}3 of 3]\n1 | alpha\n2 | beta\n3 | gamma\n"
        );
    }

    #[test]
    fn read_range_length_zero_always_means_no_limit() {
        let body = "line1\nline2\nline3\nline4\nline5\n";
        // length: 0 with offset > 1 is never an error — it reads to EOF.
        let params = read_params(json!({"path": "f", "offset": 3, "length": 0}));
        assert_eq!(
            apply_read_range(body, &params).unwrap(),
            "[lines 3\u{2013}5 of 5]\nline3\nline4\nline5\n"
        );
    }

    #[test]
    fn read_range_offset_zero_is_error() {
        let body = "alpha\nbeta\n";
        let params = read_params(json!({"path": "f", "offset": 0}));
        let err = apply_read_range(body, &params).unwrap_err();
        assert!(err.contains("offset"), "{err}");
    }

    #[test]
    fn read_range_offset_past_eof_is_error() {
        let body = "alpha\nbeta\ngamma\n";
        let params = read_params(json!({"path": "f", "offset": 11}));
        let err = apply_read_range(body, &params).unwrap_err();
        assert_eq!(err, "offset 11 exceeds file length 3");
    }

    #[test]
    fn read_range_empty_file_any_params_no_header() {
        let params =
            read_params(json!({"path": "f", "offset": 1, "length": 5, "line_numbers": true}));
        assert_eq!(apply_read_range("", &params).unwrap(), "");
    }

    #[test]
    fn read_range_final_line_without_newline_preserved() {
        // Last line lacks a trailing '\n'; split_inclusive keeps it verbatim.
        let body = "a\nb\nc";
        let params = read_params(json!({"path": "f", "offset": 2}));
        assert_eq!(
            apply_read_range(body, &params).unwrap(),
            "[lines 2\u{2013}3 of 3]\nb\nc"
        );
    }

    #[tokio::test]
    async fn read_file_with_offset_emits_header_end_to_end() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "l1\nl2\nl3\n").await.unwrap();
        let sb = AllowAllSandbox;
        let r = ReadFileTool::new()
            .execute(
                &call(
                    "read_file",
                    json!({"path": path.to_str().unwrap(), "offset": 2}),
                ),
                &sb,
                &test_ctx(),
            )
            .await;
        match r {
            ToolOutput::Success { content, .. } => {
                assert_eq!(content, "[lines 2\u{2013}3 of 3]\nl2\nl3\n");
            }
            other => panic!("{other:?}"),
        }
    }
}
