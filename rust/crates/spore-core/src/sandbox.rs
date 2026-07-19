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
    BoxFut, CommandOutput, CommandProgressSink, FileRef, IsolationMode, Operation, SandboxProvider,
    SandboxViolation, TruncatedOutput,
};
use crate::model::ToolCall;

// ============================================================================
// ExecConfig
// ============================================================================

/// Process-execution hardening knobs for [`WorkspaceScopedSandbox::execute_command`]
/// (SC-12). `None` on [`WorkspaceConfig::exec_config`] keeps the legacy
/// behavior: the child inherits the parent's stdin and full environment, no
/// implicit timeout is applied, and a dropped `execute_command` future leaks
/// the child. Setting it tightens subprocess execution — the hardening the
/// looper coding agent needs so build/test commands fail fast on a prompt
/// instead of hanging, and orphaned processes are reaped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecConfig {
    /// Timeout applied when the per-call `timeout` argument is `None`. The
    /// per-call value always wins; this is the floor for callers that pass no
    /// timeout. `None` leaves such commands un-timed (legacy).
    #[serde(default)]
    pub default_timeout: Option<Duration>,
    /// Redirect the child's stdin from the null device so a command that blocks
    /// reading input fails fast instead of hanging on the inherited terminal.
    #[serde(default)]
    pub close_stdin: bool,
    /// Environment variables forced onto every child (e.g.
    /// `GIT_TERMINAL_PROMPT=0`, `DEBIAN_FRONTEND=noninteractive`, `CI=1`) on top
    /// of the inherited environment, so would-be interactive prompts turn into
    /// non-interactive failures. Applied in `BTreeMap` order (deterministic).
    #[serde(default)]
    pub non_interactive_env: std::collections::BTreeMap<String, String>,
    /// Kill the child if the `execute_command` future is dropped (e.g. an outer
    /// cancellation, or the timeout firing) instead of orphaning it.
    #[serde(default)]
    pub kill_on_drop: bool,
}

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
    /// Optional process-execution hardening (SC-12). `None` keeps the legacy
    /// inherit-stdin / no-timeout / inherit-env / leak-on-drop behavior.
    #[serde(default)]
    pub exec_config: Option<ExecConfig>,
    /// Optional narrower boundary for [`Operation::Write`] (SC-13). When
    /// `Some`, writes must resolve under `write_root` while reads (and execute)
    /// may range over the wider [`root`](Self::root) — "read-everywhere,
    /// write-scoped". `None` gates writes by `root` exactly like reads (the
    /// legacy single-root behavior). Path strings stay root-relative for both
    /// reads and writes; only the boundary check tightens. Canonicalized at
    /// construction and must exist; typically a subdirectory of `root`.
    #[serde(default)]
    pub write_root: Option<PathBuf>,
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
            exec_config: None,
            write_root: None,
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
    #[error("workspace write_root does not exist: {path}")]
    WriteRootNotFound { path: PathBuf },
    #[error("workspace write_root io error: {source}")]
    WriteRootIo {
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
        // Accept non-canonical input but record the canonical form so
        // downstream comparisons are stable.
        config.root = canonical;

        // SC-13: canonicalize the optional write_root the same way so the
        // boundary `starts_with` check compares canonical paths.
        if let Some(write_root) = config.write_root.take() {
            let canonical_wr = match std::fs::canonicalize(&write_root) {
                Ok(p) => p,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(BuildError::WriteRootNotFound { path: write_root });
                }
                Err(e) => {
                    return Err(BuildError::WriteRootIo {
                        path: write_root,
                        source: e,
                    });
                }
            };
            config.write_root = Some(canonical_wr);
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
        // 1. Treat raw as relative even if absolute. An absolute path that is
        //    already inside the workspace maps to itself: models routinely
        //    echo the full workspace path from their prompt
        //    (`/Users/me/Downloads/x`), and re-rooting that under the root
        //    would silently nest a junk copy of the whole prefix
        //    (`<root>/Users/me/Downloads/x`). Any other absolute path keeps
        //    the historical lenient treatment — strip a single leading
        //    separator so `/foo/bar` means `<root>/foo/bar`.
        let raw_path = Path::new(raw);
        let joined = if raw_path.is_absolute() {
            if raw_path.starts_with(&self.config.root) {
                raw_path.to_path_buf()
            } else {
                let stripped = raw_path
                    .strip_prefix("/")
                    .ok()
                    .or_else(|| raw_path.strip_prefix(std::path::MAIN_SEPARATOR_STR).ok())
                    .unwrap_or(raw_path);
                self.config.root.join(stripped)
            }
        } else {
            self.config.root.join(raw_path)
        };

        // 2. Canonicalize. The target — and any run of its trailing
        //    directories — may not yet exist, for *any* operation including
        //    Read: canonicalize the nearest existing ancestor and re-join the
        //    missing components. Resolution is operation-agnostic on
        //    purpose: existence is orthogonal to the boundary check. A
        //    missing in-workspace path still resolves (via its canonicalized
        //    ancestor) and passes the boundary check; the actual read then
        //    naturally returns NotFound, surfaced as a recoverable error by
        //    the read tool rather than a PathEscape. A missing path that
        //    resolves *outside* the root is still a PathEscape.
        let canonical = match std::fs::canonicalize(&joined) {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Walk up to the nearest EXISTING ancestor, canonicalize it,
                // and re-join the missing components — so a target several
                // levels inside a not-yet-created subtree still resolves
                // (write_file / create_dir then materialize the directories).
                // Symlinks can only live in the existing prefix (a missing
                // path contains none), so canonicalizing that prefix is as
                // strong as canonicalizing a direct parent. `file_name()`
                // returning `None` (a `..`-terminated ancestor) aborts the
                // walk: a lexical re-join across `..` cannot be trusted once
                // the directories exist.
                let mut ancestor = joined.as_path();
                let mut missing: Vec<std::ffi::OsString> = Vec::new();
                loop {
                    let name =
                        ancestor
                            .file_name()
                            .ok_or_else(|| SandboxViolation::PathEscape {
                                path: raw.to_string(),
                            })?;
                    missing.push(name.to_os_string());
                    let parent = ancestor
                        .parent()
                        .ok_or_else(|| SandboxViolation::PathEscape {
                            path: raw.to_string(),
                        })?;
                    match std::fs::canonicalize(parent) {
                        Ok(base) => break missing.iter().rev().fold(base, |p, c| p.join(c)),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ancestor = parent,
                        Err(_) => {
                            return Err(SandboxViolation::PathEscape {
                                path: raw.to_string(),
                            });
                        }
                    }
                }
            }
            Err(_) => {
                return Err(SandboxViolation::PathEscape {
                    path: raw.to_string(),
                });
            }
        };

        // 3. Boundary check. Reads and execute must stay under the read
        //    `root`; writes must stay under `write_root` when set (SC-13:
        //    read-everywhere, write-scoped). The path string was already
        //    joined onto `root` above — `write_root` only narrows where the
        //    resolved target is allowed to land, it is not a separate join
        //    base — so a write outside `write_root` is a PathEscape even
        //    though it lives under `root`.
        let boundary_root: &Path = match operation {
            Operation::Write => self
                .config
                .write_root
                .as_deref()
                .unwrap_or(&self.config.root),
            _ => &self.config.root,
        };
        if !canonical.starts_with(boundary_root) {
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

// ============================================================================
// Subprocess plumbing: incremental capture, group kill, output sanitizing
// ============================================================================

type SharedBuf = std::sync::Arc<std::sync::Mutex<Vec<u8>>>;

fn new_shared_buf() -> SharedBuf {
    std::sync::Arc::new(std::sync::Mutex::new(Vec::new()))
}

fn take_buf(buf: &SharedBuf) -> String {
    String::from_utf8_lossy(&buf.lock().expect("output buffer lock")).into_owned()
}

/// Non-destructive sibling of [`take_buf`]: snapshot the bytes drained so far
/// without consuming them (the buffers keep accumulating until exit).
fn peek_buf(buf: &SharedBuf) -> String {
    String::from_utf8_lossy(&buf.lock().expect("output buffer lock")).into_owned()
}

/// How often the executor's wait loop snapshots the drain buffers for a live
/// progress preview (pi streams its subprocess tail at the same cadence).
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// Preview budget: the last N bytes of sanitized output. Enough for a screenful
/// of tail context in a UI without shipping megabytes every tick.
const PROGRESS_TAIL_BYTES: usize = 2048;

/// Build one live-progress snapshot: both buffers (stderr appended after
/// stdout), sanitized like the final output, cut to the tail budget on a char
/// boundary. Interleaving is approximate — the true merge order of the two
/// pipes is unknowable post-hoc — but a tail preview only needs "what is it
/// printing right now", not a faithful transcript (the `ToolResult` carries
/// the real streams separately).
fn progress_preview(stdout_buf: &SharedBuf, stderr_buf: &SharedBuf) -> String {
    let mut combined = peek_buf(stdout_buf);
    let err = peek_buf(stderr_buf);
    if !err.is_empty() {
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&err);
    }
    let sanitized = sanitize_command_output(&combined);
    let trimmed = sanitized.trim_end();
    if trimmed.len() <= PROGRESS_TAIL_BYTES {
        return trimmed.to_string();
    }
    let mut start = trimmed.len() - PROGRESS_TAIL_BYTES;
    while !trimmed.is_char_boundary(start) {
        start += 1;
    }
    trimmed[start..].to_string()
}

/// Read a child pipe to EOF, appending into a shared buffer so the bytes stay
/// reachable even if the awaiting future is cancelled by a timeout.
async fn drain_pipe(mut pipe: impl tokio::io::AsyncRead + Unpin + Send + 'static, buf: SharedBuf) {
    use tokio::io::AsyncReadExt;
    let mut chunk = [0u8; 8192];
    loop {
        match pipe.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buf
                .lock()
                .expect("output buffer lock")
                .extend_from_slice(&chunk[..n]),
        }
    }
}

/// Await the drain tasks with a short bound. A child can only exit once its
/// remaining output fits the OS pipe buffer, so post-exit drains finish in
/// milliseconds — the bound exists to cut off a BACKGROUND grandchild that
/// inherited the pipe and would otherwise hold EOF open indefinitely
/// (`/bin/sh -c 'server &'`).
async fn join_drains(
    stdout_task: Option<tokio::task::JoinHandle<()>>,
    stderr_task: Option<tokio::task::JoinHandle<()>>,
) {
    const DRAIN_GRACE: Duration = Duration::from_millis(500);
    for task in [stdout_task, stderr_task].into_iter().flatten() {
        let _ = tokio::time::timeout(DRAIN_GRACE, task).await;
    }
}

/// SIGKILL an entire Unix process group (the child is spawned as group
/// leader). No-op on non-Unix and for a child that never reported a pid.
#[cfg(unix)]
fn kill_process_group(pgid: Option<i32>) {
    if let Some(pgid) = pgid {
        // SAFETY: plain syscall; a stale/invalid pgid returns ESRCH harmlessly.
        unsafe {
            libc::kill(-pgid, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pgid: Option<i32>) {}

/// Kills the child's whole process group if the owning future is dropped
/// mid-run (user abort). `kill_on_drop` only covers the direct child.
struct GroupKillGuard {
    pgid: Option<i32>,
    armed: bool,
}

impl Drop for GroupKillGuard {
    fn drop(&mut self) {
        if self.armed {
            kill_process_group(self.pgid);
        }
    }
}

/// Sanitize subprocess output for model consumption: strip ANSI escape
/// sequences and non-printing control characters, then apply carriage-return
/// overwrite semantics per line so progress bars collapse to their final
/// state instead of flooding context with every intermediate frame.
pub fn sanitize_command_output(s: &str) -> String {
    let stripped = strip_ansi(s);
    stripped
        .split('\n')
        .map(resolve_cr_overwrites)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Remove ANSI CSI (`ESC[…m`), OSC (`ESC]…BEL`), and two-byte escape
/// sequences, plus non-printing control characters other than `\n`/`\t`/`\r`.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            match chars.peek() {
                // CSI: `ESC [` params/intermediates, terminated by 0x40–0x7E.
                Some('[') => {
                    chars.next();
                    for n in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&n) {
                            break;
                        }
                    }
                }
                // OSC: `ESC ]` … BEL or `ESC \`.
                Some(']') => {
                    chars.next();
                    while let Some(n) = chars.next() {
                        if n == '\u{7}' {
                            break;
                        }
                        if n == '\u{1b}' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                // Two-byte escape (`ESC c`, `ESC =`, …).
                Some(_) => {
                    chars.next();
                }
                None => {}
            }
        } else if c == '\n' || c == '\t' || c == '\r' || !c.is_control() {
            out.push(c);
        }
        // Other control chars (BEL, BS, VT, …) are dropped.
    }
    out
}

/// Terminal CR semantics for one line: a carriage return moves the cursor to
/// column 0, so only the text after the LAST `\r` survives. A trailing `\r`
/// (CRLF line ending) is stripped first so it doesn't erase the line.
fn resolve_cr_overwrites(line: &str) -> &str {
    let line = line.strip_suffix('\r').unwrap_or(line);
    match line.rfind('\r') {
        Some(i) => &line[i + 1..],
        None => line,
    }
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
        self.execute_command_streaming(command, args, working_dir, timeout, None)
    }

    fn execute_command_streaming<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        working_dir: Option<&'a Path>,
        timeout: Option<Duration>,
        progress: Option<CommandProgressSink>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        Box::pin(async move {
            match self.isolation_mode {
                #[cfg(feature = "dangerous")]
                IsolationMode::None => {}
                // CAVEAT (issue #6): `WorkspaceScoped` only scopes FILESYSTEM
                // paths (via `resolve_path`); it does NOT contain process
                // execution. Commands dispatched here run directly on the host
                // with the agent's privileges — no namespace/cgroup/seccomp
                // isolation, full network access, and `working_dir` is only a
                // default, not a jail. Deployments that run untrusted agents must
                // select `Bubblewrap`/`Docker` (tracked under #6) once those
                // backends land. Until then, treat the coding/exec tools wired to
                // this sandbox as running uncontained.
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
                IsolationMode::MicroVm { .. } => {
                    // MicroVM execution lives in `sandbox_petri::PetriSandboxProvider`
                    // (feature `petri`); this host-exec sandbox cannot honor it.
                    return Err(SandboxViolation::DisallowedCommand {
                        command: format!(
                            "micro_vm isolation requires the petri sandbox provider: {command}"
                        ),
                    });
                }
            }

            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args);
            cmd.current_dir(working_dir.unwrap_or(&self.config.root));
            // Pipe both streams and read them INCREMENTALLY (instead of the old
            // buffered `.output()`) so partial output survives a timeout or an
            // aborted (dropped) future — a timed-out test run still shows the
            // model what it printed before hanging.
            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::piped());
            // Unix: make the child a process-GROUP leader so timeout/abort can
            // kill the whole tree. `kill_on_drop` only reaps the direct child
            // (`/bin/sh`), orphaning grandchildren like `npm → node`.
            #[cfg(unix)]
            cmd.process_group(0);

            // SC-12: apply exec-hardening knobs when configured. The per-call
            // `timeout` always wins; `default_timeout` is the floor for callers
            // that pass `None`.
            let effective_timeout = if let Some(ec) = &self.config.exec_config {
                if ec.close_stdin {
                    cmd.stdin(std::process::Stdio::null());
                }
                for (k, v) in &ec.non_interactive_env {
                    cmd.env(k, v);
                }
                if ec.kill_on_drop {
                    cmd.kill_on_drop(true);
                }
                timeout.or(ec.default_timeout)
            } else {
                timeout
            };

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                // SC-15: a failed spawn is a typed violation, not a fake
                // exit_code: -1 success. Callers already handle `Err`.
                Err(e) => {
                    return Err(SandboxViolation::ExecSpawnFailed {
                        command: command.to_string(),
                        message: e.to_string(),
                    })
                }
            };
            // On Unix the group id == the child pid (it is the group leader).
            let pgid = child.id().map(|pid| pid as i32);
            // If this future is DROPPED mid-run (user abort), kill the whole
            // group — kill_on_drop alone would orphan grandchildren. Disarmed
            // on every normal exit path.
            let mut guard = GroupKillGuard { pgid, armed: true };

            // Drain both pipes into shared buffers from spawned tasks, so the
            // bytes captured so far remain reachable after a timeout.
            let stdout_buf = new_shared_buf();
            let stderr_buf = new_shared_buf();
            let stdout_task = child
                .stdout
                .take()
                .map(|p| tokio::spawn(drain_pipe(p, stdout_buf.clone())));
            let stderr_task = child
                .stderr
                .take()
                .map(|p| tokio::spawn(drain_pipe(p, stderr_buf.clone())));

            // Wait for exit, racing a timeout deadline and (when a progress
            // sink is wired) a ~100ms preview ticker that snapshots the drain
            // buffers into a sanitized tail preview. The block scopes the
            // pinned `child.wait()` borrow so the kill paths below can take
            // `&mut child` again.
            enum Waited {
                Exited(std::io::Result<std::process::ExitStatus>),
                TimedOut,
            }
            let waited = {
                let mut last_preview = String::new();
                let mut ticker = tokio::time::interval(PROGRESS_INTERVAL);
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                let wait = child.wait();
                tokio::pin!(wait);
                let deadline = async {
                    match effective_timeout {
                        Some(t) => tokio::time::sleep(t).await,
                        None => std::future::pending::<()>().await,
                    }
                };
                tokio::pin!(deadline);
                loop {
                    tokio::select! {
                        biased;
                        res = &mut wait => break Waited::Exited(res),
                        _ = &mut deadline => break Waited::TimedOut,
                        _ = ticker.tick(), if progress.is_some() => {
                            let preview = progress_preview(&stdout_buf, &stderr_buf);
                            if !preview.is_empty() && preview != last_preview {
                                if let Some(p) = &progress {
                                    p(preview.clone());
                                }
                                last_preview = preview;
                            }
                        }
                    }
                }
            };

            match waited {
                // Timed out: kill the whole process group, then return whatever
                // output was captured before the hang.
                Waited::TimedOut => {
                    kill_process_group(pgid);
                    let _ = child.kill().await; // non-unix fallback + reaps the zombie
                    guard.armed = false;
                    join_drains(stdout_task, stderr_task).await;
                    let secs = effective_timeout.map(|t| t.as_secs()).unwrap_or(0);
                    let mut stderr = sanitize_command_output(&take_buf(&stderr_buf));
                    if !stderr.is_empty() {
                        stderr.push('\n');
                    }
                    stderr.push_str(&format!("command timed out after {secs}s"));
                    Ok(CommandOutput {
                        stdout: sanitize_command_output(&take_buf(&stdout_buf)),
                        stderr,
                        exit_code: -1,
                        timed_out: true,
                        truncated: false,
                    })
                }
                Waited::Exited(Ok(status)) => {
                    guard.armed = false;
                    join_drains(stdout_task, stderr_task).await;
                    Ok(CommandOutput {
                        stdout: sanitize_command_output(&take_buf(&stdout_buf)),
                        stderr: sanitize_command_output(&take_buf(&stderr_buf)),
                        exit_code: status.code().unwrap_or(-1),
                        timed_out: false,
                        truncated: false,
                    })
                }
                Waited::Exited(Err(e)) => {
                    kill_process_group(pgid);
                    guard.armed = false;
                    Err(SandboxViolation::ExecSpawnFailed {
                        command: command.to_string(),
                        message: format!("wait failed: {e}"),
                    })
                }
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

    fn max_write_size(&self) -> Option<u64> {
        // Mirror the read-side cap on writes/appends (0 = unlimited, matching
        // `resolve`'s `max_file_size > 0` guard).
        (self.config.max_file_size > 0).then_some(self.config.max_file_size)
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
        WorkspaceConfig::scoped(root.to_path_buf())
    }

    #[tokio::test]
    async fn build_fails_when_root_missing() {
        let err = WorkspaceScopedSandbox::new(WorkspaceConfig::scoped(
            "/definitely/does/not/exist/spore-test",
        ))
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
    async fn write_into_missing_nested_subdirs_resolves() {
        // The whole `new/sub` run is absent — resolution walks up to the
        // existing root and re-joins the missing components, so write_file /
        // create_dir can materialize the directories afterwards.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let resolved = sb
            .resolve_path("new/sub/file.txt", Operation::Write)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("new/sub/file.txt"));
    }

    #[tokio::test]
    async fn absolute_path_inside_workspace_maps_to_itself() {
        // Models echo the full workspace path from the prompt; that must NOT
        // be re-rooted into `<root>/<root>/…`.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let raw = root.join("images/cat.jpg");
        let resolved = sb
            .resolve_path(raw.to_str().unwrap(), Operation::Write)
            .await
            .unwrap();
        assert_eq!(resolved, root.join("images/cat.jpg"));
    }

    #[tokio::test]
    async fn absolute_workspace_prefix_with_dotdot_is_still_path_escape() {
        // `<root>/../…` passes the lexical prefix check but must die at the
        // canonical boundary check.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let raw = format!("{}/../escaped.txt", root.display());
        let err = sb.resolve_path(&raw, Operation::Write).await.unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
    }

    #[tokio::test]
    async fn missing_subdir_with_dotdot_is_path_escape() {
        // `..` inside a missing run would resolve differently once the
        // directories exist, so a lexical re-join is refused outright.
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        let err = sb
            .resolve_path("ghost/../../outside.txt", Operation::Write)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
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
            SandboxViolation::ExecSpawnFailed {
                command: "no-such-bin".into(),
                message: "No such file or directory (os error 2)".into(),
            },
        ] {
            let s = serde_json::to_string(&v).unwrap();
            let back: SandboxViolation = serde_json::from_str(&s).unwrap();
            assert_eq!(v, back, "round-trip failed for {v:?}");
        }
    }

    #[tokio::test]
    async fn execute_command_spawn_failure_is_typed_err() {
        // SC-15: a command that cannot be spawned yields Err(ExecSpawnFailed),
        // not a fake Ok { exit_code: -1 }.
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let err = sb
            .execute_command("spore-definitely-no-such-binary-xyz", &[], None, None)
            .await
            .unwrap_err();
        match err {
            SandboxViolation::ExecSpawnFailed { command, .. } => {
                assert_eq!(command, "spore-definitely-no-such-binary-xyz");
            }
            other => panic!("expected ExecSpawnFailed, got {other:?}"),
        }
        // And it is never halt-eligible (always recoverable feedback).
        assert!(!SandboxViolation::ExecSpawnFailed {
            command: "x".into(),
            message: "y".into(),
        }
        .is_always_halt());
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
    async fn execute_command_streaming_emits_tail_previews() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let previews: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let sink_previews = previews.clone();
        let sink: CommandProgressSink =
            std::sync::Arc::new(move |p| sink_previews.lock().unwrap().push(p));
        let out = sb
            .execute_command_streaming(
                "/bin/sh",
                &["-c".into(), "echo one; sleep 0.35; echo two".into()],
                None,
                None,
                Some(sink),
            )
            .await
            .unwrap();
        // Final output is complete and unaffected by streaming.
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.contains("one") && out.stdout.contains("two"));
        // At least one preview fired WHILE the command was still running: the
        // first snapshot carries the early line but cannot carry the late one
        // (printed 350ms in, past several 100ms ticks).
        let previews = previews.lock().unwrap();
        assert!(!previews.is_empty(), "no live previews were emitted");
        assert!(
            previews[0].contains("one"),
            "first preview missing early output: {:?}",
            previews[0]
        );
        assert!(
            !previews[0].contains("two"),
            "first preview already has late output — not live: {:?}",
            previews[0]
        );
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

    /// Output printed BEFORE a hang must survive the timeout (the old
    /// buffered `.output()` returned an empty stdout on timeout).
    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_preserves_partial_output() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let out = sb
            .execute_command(
                "/bin/sh",
                &["-c".into(), "echo started; echo warn >&2; sleep 5".into()],
                None,
                Some(Duration::from_millis(300)),
            )
            .await
            .unwrap();
        assert!(out.timed_out);
        assert!(
            out.stdout.contains("started"),
            "partial stdout lost: {:?}",
            out.stdout
        );
        assert!(
            out.stderr.contains("warn"),
            "partial stderr lost: {:?}",
            out.stderr
        );
        assert!(
            out.stderr.contains("timed out"),
            "timeout note missing: {:?}",
            out.stderr
        );
    }

    /// A timed-out shell's GRANDCHILD must die with it: the child is spawned
    /// as a process-group leader and the whole group is SIGKILLed.
    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_the_whole_process_group() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        // Print the grandchild's pid, then hang the direct child.
        let out = sb
            .execute_command(
                "/bin/sh",
                &["-c".into(), "sleep 30 & echo pid:$!; sleep 30".into()],
                None,
                Some(Duration::from_millis(300)),
            )
            .await
            .unwrap();
        assert!(out.timed_out);
        let pid: i32 = out
            .stdout
            .lines()
            .find_map(|l| l.strip_prefix("pid:"))
            .expect("grandchild pid printed")
            .trim()
            .parse()
            .unwrap();
        // Give the SIGKILL a moment to land, then probe with signal 0.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let alive = unsafe { libc::kill(pid, 0) } == 0;
        assert!(!alive, "grandchild {pid} survived the group kill");
    }

    // ------------------------------------------------------------------
    // Output sanitizing (ANSI / carriage-return overwrite)
    // ------------------------------------------------------------------

    #[test]
    fn sanitize_strips_ansi_color_codes() {
        assert_eq!(
            sanitize_command_output("\u{1b}[31mred\u{1b}[0m plain \u{1b}[1;32mbold\u{1b}[m"),
            "red plain bold"
        );
    }

    #[test]
    fn sanitize_strips_osc_title_sequence() {
        assert_eq!(
            sanitize_command_output("\u{1b}]0;window title\u{7}visible"),
            "visible"
        );
    }

    #[test]
    fn sanitize_collapses_cr_progress_bars() {
        assert_eq!(
            sanitize_command_output("downloading 10%\rdownloading 55%\rdone.\nnext"),
            "done.\nnext"
        );
    }

    #[test]
    fn sanitize_keeps_crlf_lines_intact() {
        // The trailing \r of a CRLF line must not erase the line.
        assert_eq!(
            sanitize_command_output("alpha\r\nbeta\r\n"),
            "alpha\nbeta\n"
        );
    }

    #[test]
    fn sanitize_drops_stray_control_chars_keeps_tabs() {
        assert_eq!(sanitize_command_output("a\u{7}b\tc\u{8}"), "ab\tc");
    }

    /// Live command output really is sanitized at the execute_command boundary.
    #[cfg(unix)]
    #[tokio::test]
    async fn execute_command_sanitizes_ansi_output() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(dir.path())).unwrap();
        let out = sb
            .execute_command(
                "/bin/sh",
                &["-c".into(), r"printf '\033[31mred\033[0m\n'".into()],
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(out.stdout, "red\n");
    }

    // ------------------------------------------------------------------
    // SC-12 — ExecConfig exec-hardening knobs
    // ------------------------------------------------------------------

    fn exec_cfg(root: &Path, ec: ExecConfig) -> WorkspaceConfig {
        WorkspaceConfig {
            exec_config: Some(ec),
            ..WorkspaceConfig::scoped(root.to_path_buf())
        }
    }

    #[tokio::test]
    async fn exec_config_default_timeout_applies_when_call_passes_none() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(exec_cfg(
            dir.path(),
            ExecConfig {
                default_timeout: Some(Duration::from_millis(50)),
                ..ExecConfig::default()
            },
        ))
        .unwrap();
        // No per-call timeout — the exec_config floor must still fire.
        let out = sb
            .execute_command("sleep", &["5".into()], None, None)
            .await
            .unwrap();
        assert!(out.timed_out, "default_timeout did not apply");
    }

    #[tokio::test]
    async fn exec_config_per_call_timeout_overrides_default() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(exec_cfg(
            dir.path(),
            ExecConfig {
                // A generous default that must NOT veto the tight per-call one.
                default_timeout: Some(Duration::from_secs(30)),
                ..ExecConfig::default()
            },
        ))
        .unwrap();
        let out = sb
            .execute_command(
                "sleep",
                &["5".into()],
                None,
                Some(Duration::from_millis(50)),
            )
            .await
            .unwrap();
        assert!(out.timed_out, "per-call timeout should win over default");
    }

    #[tokio::test]
    async fn exec_config_non_interactive_env_is_injected() {
        let dir = TempDir::new().unwrap();
        let mut env = std::collections::BTreeMap::new();
        env.insert("SPORE_SC12_ENV".to_string(), "hardened".to_string());
        let sb = WorkspaceScopedSandbox::new(exec_cfg(
            dir.path(),
            ExecConfig {
                non_interactive_env: env,
                ..ExecConfig::default()
            },
        ))
        .unwrap();
        let out = sb
            .execute_command(
                "/bin/sh",
                &["-c".into(), "echo $SPORE_SC12_ENV".into()],
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0);
        assert!(
            out.stdout.contains("hardened"),
            "env var not injected; stdout={:?}",
            out.stdout
        );
    }

    #[tokio::test]
    async fn exec_config_close_stdin_yields_eof() {
        let dir = TempDir::new().unwrap();
        let sb = WorkspaceScopedSandbox::new(exec_cfg(
            dir.path(),
            ExecConfig {
                close_stdin: true,
                ..ExecConfig::default()
            },
        ))
        .unwrap();
        // `cat` with no args reads stdin to EOF; with stdin redirected to the
        // null device it returns immediately with exit 0 and empty output. A
        // generous per-call timeout guards the test against a hang if the knob
        // regressed (an inherited terminal would block `cat` forever).
        let out = sb
            .execute_command("cat", &[], None, Some(Duration::from_secs(5)))
            .await
            .unwrap();
        assert!(!out.timed_out, "cat hung — stdin was not closed");
        assert_eq!(out.exit_code, 0);
        assert!(out.stdout.is_empty());
    }

    #[tokio::test]
    async fn exec_config_kill_on_drop_reaps_child_on_timeout() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let sentinel = root.join("kod_sentinel");
        let sb = WorkspaceScopedSandbox::new(exec_cfg(
            &root,
            ExecConfig {
                kill_on_drop: true,
                ..ExecConfig::default()
            },
        ))
        .unwrap();
        // The shell sleeps, then would `touch` the sentinel. The 100ms timeout
        // drops the `cmd.output()` future; with kill_on_drop the shell is killed
        // before it can run `touch`, so the sentinel never appears.
        let script = format!("sleep 1; touch {}", sentinel.display());
        let out = sb
            .execute_command(
                "/bin/sh",
                &["-c".into(), script],
                None,
                Some(Duration::from_millis(100)),
            )
            .await
            .unwrap();
        assert!(out.timed_out);
        // Wait past when the un-killed shell would have created the sentinel.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert!(
            !sentinel.exists(),
            "child was not reaped on drop — sentinel was created"
        );
    }

    // ------------------------------------------------------------------
    // SC-13 — read-everywhere / write-scoped (write_root)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn write_root_allows_read_everywhere_but_scopes_writes() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let out = root.join("out");
        std::fs::create_dir(&out).unwrap();
        std::fs::write(root.join("secret.txt"), "s").unwrap();
        let sb = WorkspaceScopedSandbox::new(WorkspaceConfig {
            write_root: Some(out.clone()),
            ..WorkspaceConfig::scoped(root.clone())
        })
        .unwrap();

        // Read-everywhere: a file under `root` but outside `write_root` resolves.
        sb.resolve_path("secret.txt", Operation::Read)
            .await
            .unwrap();
        // Write-scoped: writing that same file is a PathEscape.
        let err = sb
            .resolve_path("secret.txt", Operation::Write)
            .await
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathEscape { .. }));
        // A write under `write_root` (path strings stay root-relative) is OK.
        sb.resolve_path("out/result.txt", Operation::Write)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn no_write_root_gates_writes_by_root() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        std::fs::write(root.join("a.txt"), "x").unwrap();
        let sb = WorkspaceScopedSandbox::new(cfg(&root)).unwrap();
        // With write_root unset, writes resolve anywhere under `root` (legacy).
        sb.resolve_path("a.txt", Operation::Write).await.unwrap();
    }

    #[tokio::test]
    async fn build_fails_when_write_root_missing() {
        let dir = TempDir::new().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let err = WorkspaceScopedSandbox::new(WorkspaceConfig {
            write_root: Some(root.join("does-not-exist")),
            ..WorkspaceConfig::scoped(root)
        })
        .unwrap_err();
        assert!(matches!(err, BuildError::WriteRootNotFound { .. }));
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
                allowed_paths: sc.allowed_paths.iter().map(|s| root.join(s)).collect(),
                denied_paths: sc.denied_paths.iter().map(|s| root.join(s)).collect(),
                denied_extensions: sc.denied_extensions.clone(),
                read_only: sc.read_only,
                max_file_size: sc.max_file_size,
                ..WorkspaceConfig::scoped(root.clone())
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
