"""Workspace restore/teardown (Rules 2-3).

Mirrors ``rust/crates/spore-eval/src/worktree.rs``.

Each task run gets a fresh workspace restored from its
:data:`WorkspaceSnapshot`; it is torn down after the run regardless of outcome.
``files`` writes the map into a temp dir; ``git_ref`` inits a throwaway repo and
adds a worktree; ``empty`` is a bare temp dir.

Use as an async context manager so teardown is guaranteed::

    async with Workspace.restore(snapshot) as ws:
        ...  # ws.path is the restored root
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from types import TracebackType

from .task import (
    WorktreeError,
    WorkspaceSnapshot,
    WorkspaceSnapshotEmpty,
    WorkspaceSnapshotFiles,
    WorkspaceSnapshotGitRef,
)


class Workspace:
    """A live, restored workspace. Use as an async context manager (or call
    :meth:`teardown`); teardown removes the directory tree (Rule 3)."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def path(self) -> Path:
        """The restored workspace root."""
        return self._root

    @classmethod
    async def restore(cls, snapshot: WorkspaceSnapshot) -> Workspace:
        """Restore a fresh workspace from a snapshot (Rule 2)."""
        root = Path(tempfile.mkdtemp(prefix="spore-eval-"))
        try:
            if isinstance(snapshot, WorkspaceSnapshotEmpty):
                pass
            elif isinstance(snapshot, WorkspaceSnapshotFiles):
                for rel, contents in snapshot.files.items():
                    target = root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(contents, encoding="utf-8")
            elif isinstance(snapshot, WorkspaceSnapshotGitRef):
                await _restore_git(root, snapshot.repo, snapshot.reference)
        except OSError as e:
            shutil.rmtree(root, ignore_errors=True)
            raise WorktreeError(f"workspace restore failed: {e}") from e
        return cls(root)

    def teardown(self) -> None:
        """Remove the workspace directory tree (Rule 3)."""
        shutil.rmtree(self._root, ignore_errors=True)

    async def __aenter__(self) -> Workspace:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.teardown()


async def _restore_git(dest: Path, repo: str, reference: str) -> None:
    """Restore from a git ref by adding a detached worktree from ``repo`` at
    ``reference``. ``git worktree add`` requires the destination to not already
    exist, so the freshly-made temp dir is removed first."""
    shutil.rmtree(dest, ignore_errors=True)
    args = [
        "-C",
        repo,
        "worktree",
        "add",
        "--detach",
        str(dest),
        reference,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise WorktreeError(f"git spawn failed: {e}") from e
    _, stderr_b = await proc.communicate()
    if proc.returncode != 0:
        raise WorktreeError(f"git {args} failed: {stderr_b.decode('utf-8', errors='replace')}")
