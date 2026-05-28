// Package worktree handles workspace restore/teardown (Rules 2-3).
//
// Each task run gets a fresh workspace restored from its WorkspaceSnapshot; it
// is torn down after the run regardless of outcome via Close. Files writes the
// map into a temp dir; GitRef inits a worktree from a real repo; Empty is a
// bare temp dir.
package worktree

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/squirrelsoft-dev/spore-core/go/spore-eval/task"
)

// Workspace is a live, restored workspace. Close tears the workspace down
// (Rule 3) by removing the temp directory tree.
type Workspace struct {
	dir string
}

// Path returns the restored workspace root.
func (w *Workspace) Path() string { return w.dir }

// Close removes the workspace directory tree (Rule 3). Idempotent.
func (w *Workspace) Close() error {
	if w.dir == "" {
		return nil
	}
	err := os.RemoveAll(w.dir)
	w.dir = ""
	return err
}

// Restore creates a fresh workspace from a snapshot (Rule 2).
func Restore(ctx context.Context, snapshot task.WorkspaceSnapshot) (*Workspace, error) {
	dir, err := os.MkdirTemp("", "spore-eval-ws-*")
	if err != nil {
		return nil, &task.WorktreeError{Msg: err.Error()}
	}
	switch snapshot.Kind {
	case task.SnapshotEmpty:
		// bare directory
	case task.SnapshotFiles:
		for rel, contents := range snapshot.Files {
			p := filepath.Join(dir, rel)
			if parent := filepath.Dir(p); parent != "" {
				if err := os.MkdirAll(parent, 0o755); err != nil {
					_ = os.RemoveAll(dir)
					return nil, &task.WorktreeError{Msg: err.Error()}
				}
			}
			if err := os.WriteFile(p, []byte(contents), 0o644); err != nil {
				_ = os.RemoveAll(dir)
				return nil, &task.WorktreeError{Msg: err.Error()}
			}
		}
	case task.SnapshotGitRef:
		if err := restoreGit(ctx, dir, snapshot.Repo, snapshot.Reference); err != nil {
			_ = os.RemoveAll(dir)
			return nil, err
		}
	default:
		_ = os.RemoveAll(dir)
		return nil, &task.WorktreeError{Msg: fmt.Sprintf("unknown workspace snapshot kind %q", snapshot.Kind)}
	}
	return &Workspace{dir: dir}, nil
}

// restoreGit restores from a git ref by adding a worktree from repo at
// reference. The destination dir must not already exist for `git worktree add`,
// so it is removed first and recreated by git.
func restoreGit(ctx context.Context, dest, repo, reference string) error {
	if err := os.RemoveAll(dest); err != nil {
		return &task.WorktreeError{Msg: err.Error()}
	}
	cmd := exec.CommandContext(ctx, "git", "-C", repo, "worktree", "add", "--detach", dest, reference)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return &task.WorktreeError{Msg: fmt.Sprintf("git worktree add failed: %s: %s", err, string(out))}
	}
	return nil
}
