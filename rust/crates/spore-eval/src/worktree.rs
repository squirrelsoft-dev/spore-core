//! Workspace restore/teardown (Rules 2-3).
//!
//! Each task run gets a fresh workspace restored from its
//! [`WorkspaceSnapshot`]; it is torn down after the run regardless of outcome.
//! `Files` writes the map into a `TempDir`; `GitRef` inits a throwaway repo and
//! adds a worktree; `Empty` is a bare `TempDir`.

use std::path::Path;

use tempfile::TempDir;

use crate::task::{EvalError, WorkspaceSnapshot};

/// A live, restored workspace. Dropping it tears the workspace down (Rule 3) —
/// `TempDir`'s `Drop` removes the directory tree.
pub struct Workspace {
    dir: TempDir,
}

impl Workspace {
    /// The restored workspace root.
    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// Restore a fresh workspace from a snapshot (Rule 2).
    pub async fn restore(snapshot: &WorkspaceSnapshot) -> Result<Self, EvalError> {
        let dir = tempfile::tempdir().map_err(EvalError::Io)?;
        match snapshot {
            WorkspaceSnapshot::Empty => {}
            WorkspaceSnapshot::Files { files } => {
                for (rel, contents) in files {
                    let path = dir.path().join(rel);
                    if let Some(parent) = path.parent() {
                        tokio::fs::create_dir_all(parent)
                            .await
                            .map_err(EvalError::Io)?;
                    }
                    tokio::fs::write(&path, contents)
                        .await
                        .map_err(EvalError::Io)?;
                }
            }
            WorkspaceSnapshot::GitRef { repo, reference } => {
                restore_git(dir.path(), repo, reference).await?;
            }
        }
        Ok(Self { dir })
    }
}

/// Restore from a git ref by adding a worktree from `repo` at `reference`.
async fn restore_git(dest: &Path, repo: &str, reference: &str) -> Result<(), EvalError> {
    let run = |args: Vec<String>, cwd: Option<std::path::PathBuf>| async move {
        let mut cmd = tokio::process::Command::new("git");
        cmd.args(&args);
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        let out = cmd
            .output()
            .await
            .map_err(|e| EvalError::Worktree(format!("git spawn failed: {e}")))?;
        if !out.status.success() {
            return Err(EvalError::Worktree(format!(
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        Ok::<_, EvalError>(())
    };
    // `git worktree add <dest> <reference>` run inside the source repo.
    run(
        vec![
            "-C".into(),
            repo.into(),
            "worktree".into(),
            "add".into(),
            "--detach".into(),
            dest.to_string_lossy().into_owned(),
            reference.into(),
        ],
        None,
    )
    .await
}
