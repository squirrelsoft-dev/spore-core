//! Canonical `SandboxProvider` implementation — issue #6.
//!
//! [`WorkspaceScopedSandbox`] is the default in-tree sandbox. It enforces a
//! workspace root with allow/deny lists, extension filters, a read-only
//! mode, and per-file size limits. It runs subprocesses directly via
//! `tokio::process::Command` and offloads large outputs to
//! `{workspace_root}/.spore/offload/{call_id}.txt`.
//!
//! The [`IsolationMode`], [`NetworkPolicy`], [`BwrapProfile`], and
//! [`Operation`] enums live in [`crate::harness`] so the `SandboxProvider`
//! trait can reference them without a circular dependency. They are
//! re-exported from `crate` for convenience.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::harness::{
    BoxFut, CommandOutput, FileRef, IsolationMode, Operation, SandboxProvider, SandboxViolation,
    TruncatedOutput,
};
use crate::model::ToolCall;

// ============================================================================
// WorkspaceConfig
// ============================================================================

/// Configuration injected at harness construction time.
///
/// Paths are kept as `PathBuf` here (Rust-side ergonomics); the wire format
/// for `SandboxViolation` still uses `String` so it stays portable across
/// the four reference implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    /// Canonical workspace root. All resolved paths must live under it.
    pub root: PathBuf,
    /// Explicit allowlist. Empty means "allow everything under `root`".
    #[serde(default)]
    pub allowed_paths: Vec<PathBuf>,
    /// Explicit denylist. Evaluated after the allowlist.
    #[serde(default)]
    pub denied_paths: Vec<PathBuf>,
    /// Allowed extensions. `None` means "allow all". Currently advisory —
    /// the v1 sandbox enforces only [`Self::denied_extensions`].
    #[serde(default)]
    pub allowed_extensions: Option<Vec<String>>,
    /// Denied extensions (e.g. `["env", "pem", "key"]`). Leading dots are
    /// tolerated and stripped during comparison.
    #[serde(default)]
    pub denied_extensions: Vec<String>,
    /// If `true`, [`Operation::Write`] and [`Operation::Execute`] yield
    /// [`SandboxViolation::ReadOnlyViolation`].
    #[serde(default)]
    pub read_only: bool,
    /// Maximum file size (bytes) for read operations. `0` disables the
    /// check.
    #[serde(default)]
    pub max_file_size: u64,
}

impl WorkspaceConfig {
    /// A read-write config scoped to `root` with no allow/deny lists, no
    /// extension filters, and no size cap — the common starting point for a
    /// tool-using agent. Tighten individual fields as needed.
    pub fn scoped(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        }
    }
}

// ============================================================================
// BuildError
// ============================================================================

/// Construction-time errors for [`WorkspaceScopedSandbox`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BuildError {
    #[error("workspace root does not exist: {path}")]
    RootNotFound { path: PathBuf },
    #[error("workspace root is not canonical: {path}")]
    RootNotCanonical { path: PathBuf },
    #[error("workspace root io error: {source}")]
    RootIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

// ============================================================================
// WorkspaceScopedSandbox
// ============================================================================

/// Path-enforcing sandbox. Canonicalizes every resolved path against the
/// workspace root and applies allow/deny + extension + read-only policies.
#[derive(Debug)]
pub struct WorkspaceScopedSandbox {
    config: WorkspaceConfig,
    isolation_mode: IsolationMode,
}

impl WorkspaceScopedSandbox {
    /// Build a sandbox with [`IsolationMode::WorkspaceScoped`].
    pub fn new(config: WorkspaceConfig) -> Result<Self, BuildError> {
        Self::with_mode(config, IsolationMode::WorkspaceScoped)
    }

    /// Build a sandbox with the given isolation mode. Emits an `eprintln!`
    /// warning if `mode == IsolationMode::None` — that mode is for trusted
    /// dev use only and must not be enabled silently in production.
    pub fn with_mode(mut config: WorkspaceConfig, mode: IsolationMode) -> Result<Self, BuildError> {
        // Validate root exists, then canonicalize it for stable comparisons.
        let canonical = match std::fs::canonicalize(&config.root) {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(BuildError::RootNotFound { path: config.root });
            }
            Err(e) => {
                return Err(BuildError::RootIo {
                    path: config.root,
                    source: e,
                });
            }
        };
        if canonical != config.root && !path_equiv(&canonical, &config.root) {
            // Accept non-canonical input but record the canonical form so
            // downstream comparisons are stable.
            config.root = canonical;
        } else {
            config.root = canonical;
        }

        #[cfg(feature = "dangerous")]
        if matches!(mode, IsolationMode::None) {
            eprintln!(
                "spore-core: WorkspaceScopedSandbox constructed with IsolationMode::None — \
                 trusted-dev use only; do not enable silently in production",
            );
        }

        Ok(Self {
            config,
            isolation_mode: mode,
        })
    }

    /// Borrow the underlying workspace config.
    pub fn config(&self) -> &WorkspaceConfig {
        &self.config
    }

    /// Path-resolution core. Public-but-internal so the trait impl and the
    /// fixture replay test can share the same logic.
    fn resolve(&self, raw: &str, operation: Operation) -> Result<PathBuf, SandboxViolation> {
        // 1. Treat raw as relative even if absolute — strip a single leading
        //    separator so callers can pass `/foo/bar` and mean
        //    `<root>/foo/bar`.
        let raw_path = Path::new(raw);
        let joined = if raw_path.is_absolute() {
            let stripped = raw_path
                .strip_prefix("/")
                .ok()
                .or_else(|| raw_path.strip_prefix(std::path::MAIN_SEPARATOR_STR).ok())
                .unwrap_or(raw_path);
            self.config.root.join(stripped)
        } else {
            self.config.root.join(raw_path)
        };

        // 2. Canonicalize. The target file may not yet exist — for *any*
        //    operation, including Read — so canonicalize the parent and
        //    re-join the filename. Resolution is operation-agnostic on
        //    purpose: existence is orthogonal to the boundary check. A
        //    missing in-workspace path still resolves (via its canonicalized
        //    parent) and passes the boundary check; the actual read then
        //    naturally returns NotFound, surfaced as a recoverable error by
        //    the read tool rather than a PathEscape. A missing path that
        //    resolves *outside* the root is still a PathEscape.
        let canonical = match std::fs::canonicalize(&joined) {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let parent = joined
                    .parent()
                    .ok_or_else(|| SandboxViolation::PathEscape {
                        path: raw.to_string(),
                    })?;
                let canonical_parent =
                    std::fs::canonicalize(parent).map_err(|_| SandboxViolation::PathEscape {
                        path: raw.to_string(),
                    })?;
                let file_name = joined
                    .file_name()
                    .ok_or_else(|| SandboxViolation::PathEscape {
                        path: raw.to_string(),
                    })?;
                canonical_parent.join(file_name)
            }
            Err(_) => {
                return Err(SandboxViolation::PathEscape {
                    path: raw.to_string(),
                });
            }
        };

        // 3. Boundary check.
        if !canonical.starts_with(&self.config.root) {
            return Err(SandboxViolation::PathEscape {
                path: canonical.display().to_string(),
            });
        }

        // 4. Denylist.
        for denied in &self.config.denied_paths {
            if canonical.starts_with(denied) {
                return Err(SandboxViolation::PathDenied {
                    path: canonical.display().to_string(),
                    matched_rule: denied.display().to_string(),
                });
            }
        }

        // 5. Allowlist (if non-empty).
        if !self.config.allowed_paths.is_empty() {
            let allowed = self
                .config
                .allowed_paths
                .iter()
                .any(|a| canonical.starts_with(a));
            if !allowed {
                return Err(SandboxViolation::PathDenied {
                    path: canonical.display().to_string(),
                    matched_rule: "not in allowlist".into(),
                });
            }
        }

        // 6. Denied extensions. Match either the path's real extension or,
        //    for dotfiles like `.env`, the filename itself (which `Path`
        //    reports as the file_name and an empty extension).
        let ext_from_ext = canonical
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase());
        let ext_from_dotfile = canonical
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix('.'))
            .filter(|n| !n.contains('.'))
            .map(|n| n.to_ascii_lowercase());
        for denied_ext in &self.config.denied_extensions {
            let trimmed = denied_ext.trim_start_matches('.').to_ascii_lowercase();
            let matched = ext_from_ext.as_deref() == Some(&trimmed)
                || ext_from_dotfile.as_deref() == Some(&trimmed);
            if matched {
                return Err(SandboxViolation::ExtensionDenied {
                    path: canonical.display().to_string(),
                    extension: trimmed,
                });
            }
        }

        // 7. Read-only.
        if self.config.read_only && matches!(operation, Operation::Write | Operation::Execute) {
            return Err(SandboxViolation::ReadOnlyViolation {
                path: canonical.display().to_string(),
            });
        }

        // 8. File-size cap (read ops only — write ops use `content.len()`
        //    upstream).
        if matches!(operation, Operation::Read)
            && self.config.max_file_size > 0
            && canonical.is_file()
        {
            if let Ok(meta) = std::fs::metadata(&canonical) {
                if meta.len() > self.config.max_file_size {
                    return Err(SandboxViolation::FileSizeExceeded {
                        path: canonical.display().to_string(),
                        size: meta.len(),
                        limit: self.config.max_file_size,
                    });
                }
            }
        }

        Ok(canonical)
    }
}

fn path_equiv(a: &Path, b: &Path) -> bool {
    a == b
}

// ============================================================================
// SandboxProvider impl
// ============================================================================

impl SandboxProvider for WorkspaceScopedSandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async move { Ok(()) })
    }

    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        working_dir: Option<&'a Path>,
        timeout: Option<Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        Box::pin(async move {
            match self.isolation_mode {
                #[cfg(feature = "dangerous")]
                IsolationMode::None => {}
                IsolationMode::WorkspaceScoped => {}
                IsolationMode::Bubblewrap { .. } => {
                    // TODO(#6): wire bubblewrap backend.
                    return Err(SandboxViolation::DisallowedCommand {
                        command: format!("bubblewrap isolation not implemented: {command}"),
                    });
                }
                IsolationMode::Docker { .. } => {
                    // TODO(#6): wire docker backend.
                    return Err(SandboxViolation::DisallowedCommand {
                        command: format!("docker isolation not implemented: {command}"),
                    });
                }
            }

            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args);
            cmd.current_dir(working_dir.unwrap_or(&self.config.root));

            let fut = cmd.output();
            let result = match timeout {
                Some(t) => match tokio::time::timeout(t, fut).await {
                    Ok(r) => r,
                    Err(_) => {
                        return Ok(CommandOutput {
                            stdout: String::new(),
                            stderr: format!("command timed out after {}s", t.as_secs()),
                            exit_code: -1,
                            timed_out: true,
                            truncated: false,
                        });
                    }
                },
                None => fut.await,
            };

            match result {
                Ok(out) => Ok(CommandOutput {
                    stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&out.stderr).to_string(),
                    exit_code: out.status.code().unwrap_or(-1),
                    timed_out: false,
                    truncated: false,
                }),
                Err(e) => Ok(CommandOutput {
                    stdout: String::new(),
                    stderr: format!("spawn failed: {e}"),
                    exit_code: -1,
                    timed_out: false,
                    truncated: false,
                }),
            }
        })
    }

    fn handle_large_output<'a>(
        &'a self,
        content: String,
        call_id: &'a str,
        head_tokens: u32,
        tail_tokens: u32,
    ) -> BoxFut<'a, TruncatedOutput> {
        Box::pin(async move {
            let head_chars = (head_tokens as usize).saturating_mul(4);
            let tail_chars = (tail_tokens as usize).saturating_mul(4);
            let total_chars = content.chars().count();
            let original_size = content.len() as u64;

            if total_chars <= head_chars + tail_chars {
                return TruncatedOutput {
                    content,
                    truncated: false,
                    full_ref: None,
                    original_size,
                };
            }

            let head: String = content.chars().take(head_chars).collect();
            let tail: String = content.chars().skip(total_chars - tail_chars).collect();
            let snippet = format!("{head}\n...[truncated]...\n{tail}");

            // Offload the full original content.
            let offload_dir = self.config.root.join(".spore").join("offload");
            let full_ref = match tokio::fs::create_dir_all(&offload_dir).await {
                Ok(()) => {
                    let safe_id = sanitize_call_id(call_id);
                    let offload_path = offload_dir.join(format!("{safe_id}.txt"));
                    match tokio::fs::write(&offload_path, content.as_bytes()).await {
                        Ok(()) => Some(FileRef {
                            path: offload_path.display().to_string(),
                            byte_len: original_size,
                        }),
                        Err(_) => None,
                    }
                }
                Err(_) => None,
            };

            TruncatedOutput {
                content: snippet,
                truncated: true,
                full_ref,
                original_size,
            }
        })
    }

    fn resolve_path<'a>(
        &'a self,
        path: &'a str,
        operation: Operation,
    ) -> BoxFut<'a, Result<PathBuf, SandboxViolation>> {
        let res = self.resolve(path, operation);
        Box::pin(async move { res })
    }

    fn isolation_mode(&self) -> IsolationMode {
        self.isolation_mode.clone()
    }

    fn workspace_root(&self) -> &Path {
        &self.config.root
    }
}

fn sanitize_call_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn cfg(root: &Path) -> WorkspaceConfig {
        WorkspaceConfig {
            root: root.to_path_buf(),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        }
    }

    #[tokio::test]
    async fn build_fails_when_root_missing() {
        let err = WorkspaceScopedSandbox::new(WorkspaceConfig {
            root: PathBuf::from("/definitely/does/not/exist/spore-test"),
            allowed_paths: vec![],
            denied_paths: vec![],
            allowed_extensions: None,
            denied_extensions: vec![],
            read_only: false,
            max_file_size: 0,
        })
        .unwrap_err();
        assert!(matches!(err, BuildError::RootNotFound { .. }));
    }

    #[cfg(feature = "dangerous")]
    #[tokio::test]
    async fn build_none_isolation_succeeds_and_warns() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::with_mode(cfg(dir.path()), IsolationMode::None).unwrap();
        assert_eq!(sb.isolation_mode(), IsolationMode::None);
    }

    #[tokio::test]
    async fn workspace_root_returns_canonical_root() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        assert!(sb
            .workspace_root()
            .starts_with(std::fs::canonicalize(dir.path()).unwrap()));
    }

    #[tokio::test]
    async fn escape_via_dotdot() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let err = sb
            .resolve_path("../etc/passwd", Operation::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
    }

    #[tokio::test]
    async fn escape_via_absolute_path() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        // Absolute path is rebased onto the workspace root, so an attempt
        // to read `/tmp/foo` becomes `<root>/tmp/foo` — which lives under
        // the workspace. The interesting absolute-path escape is
        // `/../..`-style, which canonicalizes outside the root.
        let err = sb
            .resolve_path("/../../etc/passwd", Operation::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
    }

    #[tokio::test]
    async fn path_denied_via_denylist() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let secrets = root.join("secrets");
        std::fs::create_dir(&secrets).unwrap();
        std::fs::write(secrets.join("k.txt"), "x").unwrap();
        let mut c = cfg(&root);
        c.denied_paths = vec![secrets.clone()];
        let sb = WorkspaceScopedSandbox::new(c).unwrap();
        let err = sb
            .resolve_path("secrets/k.txt", Operation::Read)
            .await
            .unwrap_err();
        match err {
            SandboxViolation::PathDenied { matched_rule, .. } => {
                assert!(matched_rule.contains("secrets"))
            }
            other => panic!("expected PathDenied, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn path_denied_via_allowlist_miss() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let allowed = root.join("src");
        std::fs::create_dir(&allowed).unwrap();
        std::fs::write(allowed.join("a.rs"), "x").unwrap();
        std::fs::write(root.join("other.rs"), "x").unwrap();
        let mut c = cfg(&root);
        c.allowed_paths = vec![allowed];
        let sb = WorkspaceScopedSandbox::new(c).unwrap();
        // Inside allowlist — OK.
        sb.resolve_path("src/a.rs", Operation::Read).await.unwrap();
        // Outside allowlist — denied.
        let err = sb
            .resolve_path("other.rs", Operation::Read)
            .await
            .unwrap_err();
        match err {
            SandboxViolation::PathDenied { matched_rule, .. } => {
                assert_eq!(matched_rule, "not in allowlist")
            }
            other => panic!("expected PathDenied, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extension_denied() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join(".env"), "SECRET=1").unwrap();
        let mut c = cfg(&root);
        c.denied_extensions = vec!["env".into()];
        let sb = WorkspaceScopedSandbox::new(c).unwrap();
        let err = sb.resolve_path(".env", Operation::Read).await.unwrap_err();
        assert!(matches!(err, SandboxViolation::ExtensionDenied { .. }));
    }

    #[tokio::test]
    async fn read_only_violation() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("a.txt"), "x").unwrap();
        let mut c = cfg(&root);
        c.read_only = true;
        let sb = WorkspaceScopedSandbox::new(c).unwrap();
        sb.resolve_path("a.txt", Operation::Read).await.unwrap();
        let err = sb
            .resolve_path("a.txt", Operation::Write)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::ReadOnlyViolation { .. }));
    }

    #[tokio::test]
    async fn file_size_exceeded() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("big.txt"), vec![0u8; 1024]).unwrap();
        let mut c = cfg(&root);
        c.max_file_size = 100;
        let sb = WorkspaceScopedSandbox::new(c).unwrap();
        let err = sb
            .resolve_path("big.txt", Operation::Read)
            .await
            .unwrap_err();
        match err {
            SandboxViolation::FileSizeExceeded { size, limit, .. } => {
                assert_eq!(size, 1024);
                assert_eq!(limit, 100);
            }
            other => panic!("expected FileSizeExceeded, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn write_to_nonexistent_file_works() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        // File does not exist yet, but its parent (root) does — Write must
        // canonicalize the parent and synthesize the leaf.
        let resolved = sb
            .resolve_path("new_file.txt", Operation::Write)
            .await
            .unwrap();
        assert_eq!(resolved.parent().unwrap(), root.as_path());
    }

    #[tokio::test]
    async fn read_of_missing_in_workspace_file_resolves_not_path_escape() {
        // Regression for #63: a Read of a not-yet-created file *inside* the
        // workspace must resolve via its canonicalized parent (not be
        // misclassified as PathEscape). The file is absent; resolution still
        // succeeds so the actual read can surface a recoverable not-found.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let resolved = sb
            .resolve_path("output.txt", Operation::Read)
            .await
            .expect("missing in-workspace read must resolve, not PathEscape");
        assert_eq!(resolved.parent().unwrap(), root.as_path());
        assert_eq!(resolved.file_name().unwrap(), "output.txt");
        // And it really is absent on disk.
        assert!(!resolved.exists());
    }

    #[tokio::test]
    async fn read_of_missing_file_in_subdir_resolves() {
        // Parent dir exists, leaf file does not — still resolves for Read.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::create_dir(root.join("sub")).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let resolved = sb
            .resolve_path("sub/missing.txt", Operation::Read)
            .await
            .unwrap();
        assert_eq!(resolved.parent().unwrap(), root.join("sub").as_path());
        assert!(!resolved.exists());
    }

    #[tokio::test]
    async fn read_of_missing_file_outside_root_still_path_escape() {
        // Regression for #63: a Read of a *non-existent* path that resolves
        // outside the workspace root must still be a PathEscape, not a
        // not-found. (`..` makes the canonicalized parent escape the root.)
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let err = sb
            .resolve_path("../nonexistent_passwd", Operation::Read)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
    }

    #[tokio::test]
    async fn read_of_existing_in_workspace_file_resolves() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("present.txt"), "hi").unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let resolved = sb
            .resolve_path("present.txt", Operation::Read)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("present.txt"));
    }

    #[tokio::test]
    async fn violations_round_trip_through_serde() {
        for v in [
            SandboxViolation::PathEscape { path: "/p".into() },
            SandboxViolation::PathDenied {
                path: "/p".into(),
                matched_rule: "r".into(),
            },
            SandboxViolation::ExtensionDenied {
                path: "/p.env".into(),
                extension: "env".into(),
            },
            SandboxViolation::ReadOnlyViolation { path: "/p".into() },
            SandboxViolation::FileSizeExceeded {
                path: "/p".into(),
                size: 1024,
                limit: 100,
            },
            SandboxViolation::DisallowedCommand {
                command: "rm".into(),
            },
            SandboxViolation::NetworkViolation {
                host: "evil".into(),
            },
        ] {
            let s = serde_json::to_string(&v).unwrap();
            let back: SandboxViolation = serde_json::from_str(&s).unwrap();
            assert_eq!(v, back, "round-trip failed for {v:?}");
        }
    }

    #[tokio::test]
    async fn execute_command_runs_echo() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let out = sb
            .execute_command("echo", &["hello".into()], None, None)
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("hello"));
        assert!(!out.timed_out);
    }

    #[tokio::test]
    async fn execute_command_timeout() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let out = sb
            .execute_command(
                "sleep",
                &["5".into()],
                None,
                Some(Duration::from_millis(50)),
            )
            .await
            .unwrap();
        assert!(out.timed_out);
    }

    #[tokio::test]
    async fn handle_large_output_below_threshold_untouched() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let out = sb
            .handle_large_output("short content".into(), "c1", 100, 100)
            .await;
        assert!(!out.truncated);
        assert!(out.full_ref.is_none());
        assert_eq!(out.content, "short content");
    }

    // ------------------------------------------------------------------
    // Cross-language fixture replay
    // ------------------------------------------------------------------

    #[derive(serde::Deserialize)]
    struct FsEntry {
        #[serde(default)]
        dir: bool,
        #[serde(default)]
        file: Option<String>,
    }

    #[derive(serde::Deserialize)]
    struct Expected {
        kind: String,
    }

    #[derive(serde::Deserialize)]
    struct Scenario {
        name: String,
        #[serde(default)]
        filesystem: std::collections::BTreeMap<String, FsEntry>,
        raw_path: String,
        #[serde(default)]
        allowed_paths: Vec<String>,
        #[serde(default)]
        denied_paths: Vec<String>,
        #[serde(default)]
        denied_extensions: Vec<String>,
        #[serde(default)]
        read_only: bool,
        #[serde(default)]
        max_file_size: u64,
        operation: String,
        expected: Expected,
    }

    fn parse_op(s: &str) -> Operation {
        match s {
            "read" => Operation::Read,
            "write" => Operation::Write,
            "execute" => Operation::Execute,
            other => panic!("unknown operation in fixture: {other}"),
        }
    }

    #[tokio::test]
    async fn sandbox_violation_fixtures() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/sandbox_violations");
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        let mut ran = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(&path).unwrap();
            let sc: Scenario = serde_json::from_str(&raw).unwrap_or_else(|e| {
                panic!("failed to parse {}: {e}", path.display());
            });

            // Materialize the virtual filesystem under a TempDir treated as
            // the workspace root.
            let tmp = TempDir::new().unwrap();
            let root = std::fs::canonicalize(tmp.path()).unwrap();
            for (rel, entry) in &sc.filesystem {
                let target = root.join(rel);
                if entry.dir {
                    std::fs::create_dir_all(&target).unwrap();
                } else if let Some(content) = &entry.file {
                    if let Some(parent) = target.parent() {
                        std::fs::create_dir_all(parent).unwrap();
                    }
                    std::fs::write(&target, content).unwrap();
                }
            }

            let cfg = WorkspaceConfig {
                root: root.clone(),
                allowed_paths: sc.allowed_paths.iter().map(|s| root.join(s)).collect(),
                denied_paths: sc.denied_paths.iter().map(|s| root.join(s)).collect(),
                allowed_extensions: None,
                denied_extensions: sc.denied_extensions.clone(),
                read_only: sc.read_only,
                max_file_size: sc.max_file_size,
            };
            let sb = WorkspaceScopedSandbox::new(cfg).unwrap();
            let op = parse_op(&sc.operation);
            let result = sb.resolve_path(&sc.raw_path, op).await;
            let actual_kind = match &result {
                Ok(_) => "ok".to_string(),
                Err(v) => {
                    let j = serde_json::to_value(v).unwrap();
                    j.get("kind").and_then(|k| k.as_str()).unwrap().to_string()
                }
            };
            assert_eq!(
                actual_kind, sc.expected.kind,
                "fixture {} expected kind={} got kind={}",
                sc.name, sc.expected.kind, actual_kind
            );
            ran += 1;
        }
        assert!(ran > 0, "expected at least one sandbox-violation fixture");
    }

    #[tokio::test]
    async fn handle_large_output_above_threshold_offloads() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let content = "x".repeat(10_000);
        let out = sb
            .handle_large_output(content.clone(), "call-1", 10, 10)
            .await;
        assert!(out.truncated);
        assert!(out.content.contains("...[truncated]..."));
        let r = out.full_ref.expect("expected offloaded file");
        assert_eq!(r.byte_len, content.len() as u64);
        assert!(r.path.contains(".spore"));
        let written = std::fs::read_to_string(&r.path).unwrap();
        assert_eq!(written.len(), content.len());
    }
}
