//! Sibling component traits and built-in providers extracted from the harness
//! god-module (Phase 6.1): the tool-registry, sandbox, and VCS abstractions plus
//! their built-in implementations (`EmptyToolRegistry`, `NullSandbox`,
//! `ReadOnlySandbox`, `GitVcsProvider`). Pure move — no behavior change.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::*;
use crate::model::{ToolCall, ToolSchema};
// ============================================================================
// Forward-declared sibling component traits
// ============================================================================

/// Issue #4 — ToolRegistry: dispatches tool calls.
pub trait ToolRegistry: Send + Sync {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput>;

    /// Whether this tool is annotated `always_halt` (e.g. dangerous external
    /// effects). The harness short-circuits on the first such call in a batch.
    fn is_always_halt(&self, tool_name: &str) -> bool {
        let _ = tool_name;
        false
    }

    /// Tool schemas to feed into context assembly.
    fn schemas(&self) -> Vec<ToolSchema> {
        Vec::new()
    }
}

/// A [`ToolRegistry`] with no tools.
///
/// `schemas()` is empty, so the model is offered no tools and replies in plain
/// text; `dispatch` is therefore never reached. This is the registry used by
/// [`HarnessBuilder::conversational`] and the right starting point for any
/// agent that does not act on its environment.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyToolRegistry;

impl ToolRegistry for EmptyToolRegistry {
    fn dispatch<'a>(&'a self, _call: ToolCall) -> BoxFut<'a, ToolOutput> {
        Box::pin(async {
            ToolOutput::Error {
                message: "no tools are registered".into(),
                recoverable: false,
            }
        })
    }
}

/// SC-30: a read-only VIEW over an inner harness-loop [`ToolRegistry`]. It
/// advertises ([`schemas`](ToolRegistry::schemas)) and dispatches ONLY tools
/// whose name is in `allow` — the INTERSECTION of the wrapped catalogue with a
/// read-only allow-list (`StandardTools::readonly_set()` names). Used internally
/// for the SelfVerifying evaluate phase so a reviewer cannot reach
/// write / exec / side-effecting tools (web/MCP the read-only sandbox does not
/// gate) even though the work phase could — WITHOUT the consumer registering a
/// scoped read-only toolset. A non-allow-listed dispatch (which the model should
/// never request, since it is never advertised) returns a recoverable
/// [`ToolOutput::error`]. A consumer that wants a different reviewer toolset sets
/// an explicit `eval_toolset` handle, which bypasses this view entirely.
pub(crate) struct ReadOnlyToolView {
    inner: Arc<dyn ToolRegistry>,
    allow: std::collections::HashSet<String>,
}

impl ReadOnlyToolView {
    pub(crate) fn new(inner: Arc<dyn ToolRegistry>, allow: std::collections::HashSet<String>) -> Self {
        Self { inner, allow }
    }
}

impl ToolRegistry for ReadOnlyToolView {
    fn dispatch<'a>(&'a self, call: ToolCall) -> BoxFut<'a, ToolOutput> {
        if self.allow.contains(&call.name) {
            self.inner.dispatch(call)
        } else {
            let name = call.name;
            Box::pin(async move {
                ToolOutput::error(format!(
                    "tool `{name}` is not available in the read-only evaluate phase"
                ))
            })
        }
    }

    fn is_always_halt(&self, tool_name: &str) -> bool {
        self.inner.is_always_halt(tool_name)
    }

    fn schemas(&self) -> Vec<ToolSchema> {
        self.inner
            .schemas()
            .into_iter()
            .filter(|s| self.allow.contains(&s.name))
            .collect()
    }
}

/// Issue #6 — SandboxProvider: validates tool calls against sandbox policy.
///
/// Issue #5 adds default implementations for `execute_command`,
/// `handle_large_output`, and `resolve_path` so the standard tool catalogue
/// can be built before #6 lands its canonical sandbox. **These defaults are
/// NOT production-safe**: `execute_command` spawns processes directly with no
/// sandboxing, `resolve_path` returns the input as-is, and
/// `handle_large_output` truncates inline without offloading. Issue #6 must
/// override these.
pub trait SandboxProvider: Send + Sync {
    fn validate<'a>(&'a self, call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>>;

    /// Run a subprocess. **Default impl spawns directly with no sandboxing —
    /// production sandboxes must override.**
    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        args: &'a [String],
        working_dir: Option<&'a std::path::Path>,
        timeout: Option<std::time::Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        Box::pin(async move {
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(args);
            if let Some(dir) = working_dir {
                cmd.current_dir(dir);
            }
            let fut = cmd.output();
            let output_res = if let Some(t) = timeout {
                match tokio::time::timeout(t, fut).await {
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
                }
            } else {
                fut.await
            };
            match output_res {
                Ok(out) => Ok(CommandOutput {
                    stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&out.stderr).to_string(),
                    exit_code: out.status.code().unwrap_or(-1),
                    timed_out: false,
                    truncated: false,
                }),
                // SC-15: a failed spawn is a typed violation, not a fake
                // exit_code: -1 success. Callers already handle `Err`.
                Err(e) => Err(SandboxViolation::ExecSpawnFailed {
                    command: command.to_string(),
                    message: e.to_string(),
                }),
            }
        })
    }

    /// Truncate large output head+tail. Default treats `head_tokens * 4` as
    /// an approximate char budget per half. Production sandboxes should
    /// offload the full content to a file and return a `FileRef`.
    fn handle_large_output<'a>(
        &'a self,
        content: String,
        _call_id: &'a str,
        head_tokens: u32,
        tail_tokens: u32,
    ) -> BoxFut<'a, TruncatedOutput> {
        Box::pin(async move {
            let head_chars = (head_tokens as usize).saturating_mul(4);
            let tail_chars = (tail_tokens as usize).saturating_mul(4);
            let total = content.chars().count();
            let original_size = content.len() as u64;
            if total <= head_chars + tail_chars {
                return TruncatedOutput {
                    content,
                    truncated: false,
                    full_ref: None,
                    original_size,
                };
            }
            let head: String = content.chars().take(head_chars).collect();
            let tail: String = content.chars().skip(total - tail_chars).collect();
            let elided = total - head_chars - tail_chars;
            let snippet = format!("{head}\n... [{elided} chars elided] ...\n{tail}");
            TruncatedOutput {
                content: snippet,
                truncated: true,
                full_ref: None,
                original_size,
            }
        })
    }

    /// Resolve / canonicalize a path against the sandbox root. Default is
    /// an identity pass-through; production sandboxes must enforce roots.
    ///
    /// `operation` lets implementations distinguish reads from writes so they
    /// can apply `read_only`-style policies and skip canonicalization for
    /// not-yet-existing files on writes.
    fn resolve_path<'a>(
        &'a self,
        path: &'a str,
        operation: Operation,
    ) -> BoxFut<'a, Result<std::path::PathBuf, SandboxViolation>> {
        let _ = operation;
        Box::pin(async move { Ok(std::path::PathBuf::from(path)) })
    }

    /// Active isolation mode. Used by observability and middleware. Default
    /// returns `IsolationMode::WorkspaceScoped` (safe-by-default, issue #34).
    /// A provider that genuinely wants no isolation must override this and opt
    /// in to `IsolationMode::None`, which only exists under the `dangerous`
    /// feature.
    fn isolation_mode(&self) -> IsolationMode {
        IsolationMode::WorkspaceScoped
    }

    /// Workspace root used by `ContextManager` for directory-map injection.
    /// Default returns `/`, which is **not** safe — production sandboxes
    /// must override.
    fn workspace_root(&self) -> &std::path::Path {
        std::path::Path::new("/")
    }

    /// Maximum number of bytes a single write/append may produce, or `None` for
    /// no cap. Write tools consult this BEFORE writing (`resolve_path` only sees
    /// the path, not the payload, so the read-side `max_file_size` cap cannot
    /// cover writes). Default `None` — production sandboxes override to mirror
    /// their read cap.
    fn max_write_size(&self) -> Option<u64> {
        None
    }
}

/// A [`SandboxProvider`] with no filesystem.
///
/// `validate` allows everything because nothing is meant to run against it: it
/// is the sandbox for tool-less agents (e.g. [`HarnessBuilder::conversational`]),
/// where no tool is ever dispatched and the boundary is never exercised. Agents
/// that actually touch the filesystem or shell must use a real sandbox such as
/// [`WorkspaceScopedSandbox`](crate::sandbox::WorkspaceScopedSandbox).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSandbox;

impl SandboxProvider for NullSandbox {
    fn validate<'a>(&'a self, _call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async { Ok(()) })
    }
}

/// Filesystem operation being performed on a path. Passed to
/// [`SandboxProvider::resolve_path`] so the sandbox can apply read-only
/// policies and adjust canonicalization behavior for not-yet-existing files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Read,
    Write,
    Execute,
}

/// Isolation strategy for sandboxed subprocess execution.
///
/// Lives on the trait so middleware and observability can route on it
/// without depending on the concrete sandbox impl.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IsolationMode {
    /// No isolation. Trusted-dev use only; the sandbox must emit a warning
    /// at construction time. Gated behind the `dangerous` feature (issue #34);
    /// absent from the default build. The wire tag stays `"none"`.
    #[cfg(feature = "dangerous")]
    None,
    /// Path enforcement only — no process or network isolation. Default for
    /// the canonical `WorkspaceScopedSandbox`. Commands run UNCONTAINED on the
    /// host (full privileges + network); real process isolation
    /// (`Bubblewrap`/`Docker`) is tracked under issue #6. Do not run untrusted
    /// agents under this mode.
    WorkspaceScoped,
    /// Linux process isolation via bubblewrap. Reserved; not implemented in
    /// the v1 reference sandbox.
    Bubblewrap { profile: BwrapProfile },
    /// Full Docker isolation including network policy. Reserved; not
    /// implemented in the v1 reference sandbox.
    Docker {
        image: String,
        network: NetworkPolicy,
    },
}

/// Network egress policy for `IsolationMode::Docker`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NetworkPolicy {
    /// No network access at all.
    None,
    /// Allow egress to the listed hosts only.
    Allowlist { hosts: Vec<String> },
    /// Unrestricted egress. Use with caution.
    Full,
}

/// Bubblewrap profile descriptor. Opaque in v1 — the bubblewrap backend is
/// not yet implemented, but the shape lives on the public surface so the
/// `IsolationMode` enum can be exhaustive across all four implementations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BwrapProfile {}

/// Read-only decorator over any [`SandboxProvider`] (issue #61). Wraps an
/// inner sandbox and additionally rejects every WRITE/EXECUTE-classified tool
/// call in [`validate`](SandboxProvider::validate) with
/// [`SandboxViolation::ReadOnlyViolation`], while delegating reads (and all the
/// other `SandboxProvider` methods) to the inner sandbox unchanged.
///
/// The `SelfVerifying` strategy derives one of these INTERNALLY for its
/// evaluate phase (the Default-FAIL contract: the evaluator must not be able to
/// mutate the workspace it is reviewing). Write-intent is classified by tool
/// name against [`Self::DEFAULT_WRITE_TOOLS`] (the standard catalogue's
/// mutating tools: `write_file`, `edit_file`, `delete_file`, `move_file`,
/// `exec`, `bash_command`, `run_tests`). Callers with bespoke tool names can
/// supply their own set via [`with_write_tools`](Self::with_write_tools).
///
/// `ReadOnlyViolation` is a Layer-2 (recoverable) violation, so in the harness
/// loop a blocked write surfaces to the evaluator agent as a recoverable tool
/// error — it does NOT halt the evaluate run.
pub struct ReadOnlySandbox {
    inner: Arc<dyn SandboxProvider>,
    write_tools: std::collections::HashSet<String>,
}

impl ReadOnlySandbox {
    /// Standard-catalogue tool names that MUTATE the workspace and are therefore
    /// blocked by a read-only sandbox.
    pub const DEFAULT_WRITE_TOOLS: &'static [&'static str] = &[
        "write_file",
        "edit_file",
        "delete_file",
        "move_file",
        "exec",
        "bash_command",
        "run_tests",
    ];

    /// Wrap `inner`, blocking the standard mutating tools.
    pub fn new(inner: Arc<dyn SandboxProvider>) -> Self {
        Self {
            inner,
            write_tools: Self::DEFAULT_WRITE_TOOLS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// Wrap `inner`, blocking exactly the supplied tool names (overrides the
    /// default set).
    pub fn with_write_tools(
        inner: Arc<dyn SandboxProvider>,
        write_tools: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            inner,
            write_tools: write_tools.into_iter().collect(),
        }
    }

    fn is_write(&self, tool_name: &str) -> bool {
        self.write_tools.contains(tool_name)
    }
}

impl SandboxProvider for ReadOnlySandbox {
    fn validate<'a>(&'a self, call: &'a ToolCall) -> BoxFut<'a, Result<(), SandboxViolation>> {
        Box::pin(async move {
            if self.is_write(&call.name) {
                return Err(SandboxViolation::ReadOnlyViolation {
                    path: call.name.clone(),
                });
            }
            self.inner.validate(call).await
        })
    }

    fn execute_command<'a>(
        &'a self,
        command: &'a str,
        _args: &'a [String],
        _working_dir: Option<&'a std::path::Path>,
        _timeout: Option<std::time::Duration>,
    ) -> BoxFut<'a, Result<CommandOutput, SandboxViolation>> {
        // A read-only sandbox forbids subprocess execution outright (commands
        // may have arbitrary write side effects).
        Box::pin(async move {
            Err(SandboxViolation::ReadOnlyViolation {
                path: command.to_string(),
            })
        })
    }

    fn resolve_path<'a>(
        &'a self,
        path: &'a str,
        operation: Operation,
    ) -> BoxFut<'a, Result<std::path::PathBuf, SandboxViolation>> {
        Box::pin(async move {
            if matches!(operation, Operation::Write | Operation::Execute) {
                return Err(SandboxViolation::ReadOnlyViolation {
                    path: path.to_string(),
                });
            }
            self.inner.resolve_path(path, operation).await
        })
    }

    fn isolation_mode(&self) -> IsolationMode {
        self.inner.isolation_mode()
    }

    fn workspace_root(&self) -> &std::path::Path {
        self.inner.workspace_root()
    }
}

// ============================================================================
// VcsProvider seam (issue #58 v2) — git-log reload for the Ralph loop strategy
// ============================================================================

/// Read-only VCS abstraction the `Ralph` loop strategy uses to reload git
/// history between context windows (issue #58 v2, decision B4).
///
/// The v1 Ralph reload (commit `927cc57`) re-seeded each fresh context window
/// from `.spore/progress.json` + `.spore/feature_list.json` only; the spec's
/// "reload git log" step was deferred (B4) because there was no hermetic,
/// cross-language-testable seam for VCS reads. This trait IS that seam.
///
/// It mirrors how [`SandboxProvider`] abstracts filesystem/shell access:
/// define a trait, ship a real implementation ([`GitVcsProvider`]) and a
/// deterministic fixture double ([`FixtureVcsProvider`]), and inject the chosen
/// one at construction via [`HarnessBuilder::vcs_provider`]. The harness owns an
/// `Arc<dyn VcsProvider>` clone exactly as it does for every other component.
///
/// `Ralph` calls [`log`](Self::log) during its reload phase and injects the
/// output into the next window's seed as a clearly delimited
/// "Recent VCS history:" section. When NO provider is wired
/// (`vcs_provider == None`, the default) the git-log section is OMITTED and
/// Ralph behaves byte-for-byte like v1 — this is the B4→None decision.
pub trait VcsProvider: Send + Sync {
    /// Return the project's commit log, shaped by `args`. The returned string
    /// is the verbatim VCS output (e.g. `git log` stdout); the caller does not
    /// parse it, it is injected into the reloaded context block as-is.
    fn log<'a>(&'a self, args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>>;

    /// Return the working-tree status (e.g. `git status` stdout), verbatim.
    fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>>;

    /// Revert the working tree to the last known-good state (SC-14). Called by
    /// the HillClimbing loop to discard a no-improvement iteration's changes.
    /// The default is a NO-OP (`Ok(())`) so a provider that has no concept of
    /// revert — or whose workspace is not under version control — keeps
    /// compiling and simply leaves the tree as-is. [`GitVcsProvider`] overrides
    /// it with `git reset --hard HEAD`; a consumer climbing a non-git workspace
    /// (or one whose "checkpoint" means something else) supplies its own. Like
    /// the HillClimbing loop's use of it, this is best-effort: an error is
    /// surfaced but does not halt the climb.
    fn revert<'a>(&'a self) -> BoxFut<'a, Result<(), VcsError>> {
        Box::pin(async move { Ok(()) })
    }

    /// Return the SHA (or opaque revision id) the workspace currently points at
    /// (`hill-climbing-loop-correctness` §4.5). [`GitVcsProvider`] runs
    /// `git rev-parse HEAD`. The default returns the empty string — a provider
    /// whose workspace has no commit concept reports "no SHA", and the
    /// HillClimbing loop then degrades to its no-VCS fallback (revert to HEAD).
    fn head<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move { Ok(String::new()) })
    }

    /// Commit the current working tree as an accepted candidate and return its
    /// SHA (`hill-climbing-loop-correctness` §4.5). Called by HillClimbing after
    /// a KEPT iteration so the accepted artifact has a durable, revertible
    /// pointer. [`GitVcsProvider`] runs `git add -A && git commit` then
    /// `rev-parse HEAD`. The default returns the empty string (no-op providers
    /// cannot checkpoint); the loop then records an empty `commit_hash` and
    /// keeps the pre-fix behaviour. `label` is the commit message.
    fn commit_checkpoint<'a>(&'a self, label: &'a str) -> BoxFut<'a, Result<String, VcsError>> {
        let _ = label;
        Box::pin(async move { Ok(String::new()) })
    }

    /// Reset the workspace to a SPECIFIC accepted candidate SHA
    /// (`hill-climbing-loop-correctness` §4.5) — the load-bearing fix: a
    /// no-improvement revert targets the last KEPT state, not the original
    /// `HEAD`. [`GitVcsProvider`] runs `git reset --hard <sha>`. The default is a
    /// no-op (a provider with no SHA concept leaves the tree as-is); the
    /// HillClimbing loop only calls this with a non-empty SHA, falling back to
    /// [`revert`](Self::revert) when it has none.
    fn reset_to<'a>(&'a self, sha: &'a str) -> BoxFut<'a, Result<(), VcsError>> {
        let _ = sha;
        Box::pin(async move { Ok(()) })
    }
}

/// Parameters shaping a [`VcsProvider::log`] read. Each field maps to a
/// `git log` flag in [`GitVcsProvider`]:
///   - `max_entries` → `-n <N>` (cap the number of commits returned),
///   - `since_ref`   → `<ref>..` (only commits AFTER `<ref>`),
///   - `format`      → `--format=<fmt>` (custom pretty format).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VcsLogArgs {
    /// Maximum number of commits to return (`git log -n <max_entries>`).
    pub max_entries: usize,
    /// Only commits reachable after this ref (`git log <since_ref>..`). `None`
    /// returns the full history (subject to `max_entries`).
    pub since_ref: Option<String>,
    /// Custom `git log --format=<format>` string. `None` uses git's default
    /// formatting.
    pub format: Option<String>,
}

/// Error raised by a [`VcsProvider`]. Mirrors the per-component
/// `<Component>Error` convention (`thiserror`, `#[non_exhaustive]`).
#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum VcsError {
    /// The underlying VCS command failed (non-zero exit), carrying the captured
    /// stderr.
    #[error("vcs command failed: {message}")]
    CommandFailed { message: String },
    /// The VCS command was blocked or could not be spawned by the sandbox.
    #[error("vcs command blocked by sandbox: {0}")]
    Sandbox(SandboxViolation),
}

/// Real [`VcsProvider`] that shells out to `git` THROUGH a [`SandboxProvider`]
/// (issue #58 v2). It wraps the sandbox and calls
/// [`SandboxProvider::execute_command`] — it never bypasses sandboxing to spawn
/// `git` directly. The command line is built from [`VcsLogArgs`] (see that
/// type for the flag mapping); [`status`](VcsProvider::status) runs
/// `git status`. All commands run in `workspace_root`.
pub struct GitVcsProvider {
    sandbox: Arc<dyn SandboxProvider>,
    workspace_root: std::path::PathBuf,
}

impl GitVcsProvider {
    /// Wrap `sandbox`, running `git` invocations in `workspace_root`.
    pub fn new(
        sandbox: Arc<dyn SandboxProvider>,
        workspace_root: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            sandbox,
            workspace_root: workspace_root.into(),
        }
    }

    /// Build the `git log` argument vector from `args` (visible for testing the
    /// flag mapping independently of process execution).
    fn log_args(args: &VcsLogArgs) -> Vec<String> {
        let mut out = vec![
            "log".to_string(),
            "-n".to_string(),
            args.max_entries.to_string(),
        ];
        if let Some(fmt) = &args.format {
            out.push(format!("--format={fmt}"));
        }
        if let Some(since) = &args.since_ref {
            out.push(format!("{since}.."));
        }
        out
    }
}

impl VcsProvider for GitVcsProvider {
    fn log<'a>(&'a self, args: &'a VcsLogArgs) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move {
            let argv = Self::log_args(args);
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(out.stdout)
        })
    }

    fn status<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move {
            let argv = vec!["status".to_string()];
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(out.stdout)
        })
    }

    fn revert<'a>(&'a self) -> BoxFut<'a, Result<(), VcsError>> {
        Box::pin(async move {
            // `git reset --hard HEAD` THROUGH the sandbox — never spawned
            // directly. Relocated from the harness's hardcoded revert (SC-14).
            let argv = vec![
                "reset".to_string(),
                "--hard".to_string(),
                "HEAD".to_string(),
            ];
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(())
        })
    }

    fn head<'a>(&'a self) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move {
            // `git rev-parse HEAD` THROUGH the sandbox (hill-climbing-loop-
            // correctness §4.5). Returns the trimmed SHA.
            let argv = vec!["rev-parse".to_string(), "HEAD".to_string()];
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(out.stdout.trim().to_string())
        })
    }

    fn commit_checkpoint<'a>(&'a self, label: &'a str) -> BoxFut<'a, Result<String, VcsError>> {
        Box::pin(async move {
            // `git add -A && git commit -m <label>` then resolve HEAD (hill-
            // climbing-loop-correctness §4.5). The commit is best-effort: a
            // "nothing to commit" exit is NOT an error — HEAD still resolves to
            // the (unchanged) accepted SHA. A blocked spawn is surfaced. The
            // harness's `.spore/` diagnostics are excluded so they stay untracked
            // and survive a later `git reset --hard`.
            let add = vec![
                "add".to_string(),
                "-A".to_string(),
                "--".to_string(),
                HILL_CLIMB_COMMIT_EXCLUDE.to_string(),
            ];
            let _ = self
                .sandbox
                .execute_command("git", &add, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            let commit = vec!["commit".to_string(), "-m".to_string(), label.to_string()];
            let _ = self
                .sandbox
                .execute_command("git", &commit, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            let head = vec!["rev-parse".to_string(), "HEAD".to_string()];
            let out = self
                .sandbox
                .execute_command("git", &head, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(out.stdout.trim().to_string())
        })
    }

    fn reset_to<'a>(&'a self, sha: &'a str) -> BoxFut<'a, Result<(), VcsError>> {
        Box::pin(async move {
            // `git reset --hard <sha>` THROUGH the sandbox (hill-climbing-loop-
            // correctness §4.3): the no-improvement revert targets the last
            // ACCEPTED candidate, not the original HEAD.
            let argv = vec!["reset".to_string(), "--hard".to_string(), sha.to_string()];
            let out = self
                .sandbox
                .execute_command("git", &argv, Some(self.workspace_root.as_path()), None)
                .await
                .map_err(VcsError::Sandbox)?;
            if out.exit_code != 0 {
                return Err(VcsError::CommandFailed {
                    message: out.stderr,
                });
            }
            Ok(())
        })
    }
}

/// Output of a subprocess executed through the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    #[serde(default)]
    pub timed_out: bool,
    /// `true` if `stdout`/`stderr` were truncated by `handle_large_output`
    /// before being returned. Kept alongside `timed_out` so existing callers
    /// continue to compile.
    #[serde(default)]
    pub truncated: bool,
}
