"""Tests for filesystem tools."""

from __future__ import annotations

from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess, WorkspaceConfig
from spore_core.model import ToolCall
from spore_core.sandbox import WorkspaceScopedSandbox
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.fs import (
    DeleteFileTool,
    ListDirTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)

_CTX = make_test_ctx()


def _workspace_sandbox(root: Path) -> WorkspaceScopedSandbox:
    return WorkspaceScopedSandbox(
        WorkspaceConfig(
            root=root,
            allowed_paths=[],
            denied_paths=[],
            allowed_extensions=None,
            denied_extensions=[],
            read_only=False,
            max_file_size=0,
        )
    )


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    sb = AllowAllSandbox()
    p = tmp_path / "a.txt"
    w = await WriteFileTool().execute(
        _call("write_file", {"path": str(p), "content": "hello"}), sb, _CTX
    )
    assert isinstance(w, ToolOutputSuccess)
    r = await ReadFileTool().execute(_call("read_file", {"path": str(p)}), sb, _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "hello"


async def test_append_mode_concatenates(tmp_path: Path) -> None:
    sb = AllowAllSandbox()
    p = tmp_path / "a.txt"
    await WriteFileTool().execute(_call("write_file", {"path": str(p), "content": "a"}), sb, _CTX)
    await WriteFileTool().execute(
        _call("write_file", {"path": str(p), "content": "b", "append": True}), sb, _CTX
    )
    assert p.read_text() == "ab"


async def test_list_dir_sorted(tmp_path: Path) -> None:
    (tmp_path / "z").write_text("")
    (tmp_path / "a").write_text("")
    (tmp_path / "m").write_text("")
    sb = AllowAllSandbox()
    r = await ListDirTool().execute(_call("list_dir", {"path": str(tmp_path)}), sb, _CTX)
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 3
    assert lines == sorted(lines)


async def test_list_dir_entries_roundtrip_through_workspace_sandbox(
    tmp_path: Path,
) -> None:
    # Regression for #93: every entry list_dir returns must round-trip straight
    # back into read_file under the *real* WorkspaceScopedSandbox, which treats
    # all input paths as root-relative. Absolute paths (the old behavior) would
    # be rejected as PathEscape.
    root = tmp_path.resolve()
    (root / "a.txt").write_text("alpha")
    (root / "b.txt").write_text("beta")
    (root / "sub").mkdir()
    (root / "sub" / "c.txt").write_text("gamma")
    sb = _workspace_sandbox(root)

    # Recursive so we exercise both top-level files and a nested file.
    r = await ListDirTool().execute(_call("list_dir", {"path": ".", "recursive": True}), sb, _CTX)
    assert isinstance(r, ToolOutputSuccess)
    entries = r.content.splitlines()
    assert "a.txt" in entries, f"expected bare root-relative names, got {entries}"
    assert "sub/c.txt" in entries, f"expected nested entry as sub/c.txt, got {entries}"
    assert not any(e == "" or e == "." for e in entries), (
        f"must not emit the listed dir itself, got {entries}"
    )

    # The actual bug check: feed each entry straight into read_file.
    for entry in entries:
        rr = await ReadFileTool().execute(_call("read_file", {"path": entry}), sb, _CTX)
        if isinstance(rr, ToolOutputError):
            # A directory entry (e.g. ``sub``) reads as an error but must NOT be
            # a sandbox violation — that's the regression.
            lowered = rr.message.lower()
            assert "sandbox" not in lowered and "escape" not in lowered, (
                f"entry {entry!r} did not round-trip: {rr.message}"
            )
        else:
            assert isinstance(rr, ToolOutputSuccess)


async def test_delete_missing_is_recoverable() -> None:
    sb = AllowAllSandbox()
    r = await DeleteFileTool().execute(
        _call("delete_file", {"path": "/no/such/path/here"}), sb, _CTX
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_move_file_renames(tmp_path: Path) -> None:
    src = tmp_path / "s"
    dst = tmp_path / "d"
    src.write_text("hi")
    sb = AllowAllSandbox()
    r = await MoveFileTool().execute(
        _call("move_file", {"src": str(src), "dst": str(dst)}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess)
    assert not src.exists()
    assert dst.exists()


async def test_invalid_params_returns_recoverable_error() -> None:
    sb = AllowAllSandbox()
    r = await ReadFileTool().execute(_call("read_file", {}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_read_missing_in_workspace_file_is_recoverable_not_found(
    tmp_path: Path,
) -> None:
    # Regression for #63: reading a not-yet-created file *inside* the
    # workspace must surface a recoverable not-found, not a sandbox
    # PathEscape, end to end through the real WorkspaceScopedSandbox.
    sb = _workspace_sandbox(tmp_path)
    r = await ReadFileTool().execute(_call("read_file", {"path": "output.txt"}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "read failed" in r.message


async def test_read_outside_workspace_is_path_escape(tmp_path: Path) -> None:
    # Counterpart: a path that resolves outside the root is still a sandbox
    # violation, even when the file does not exist.
    sb = _workspace_sandbox(tmp_path)
    r = await ReadFileTool().execute(
        _call("read_file", {"path": "../nonexistent_secret"}), sb, _CTX
    )
    assert isinstance(r, ToolOutputError)
    lowered = r.message.lower()
    assert "escape" in lowered or "sandbox" in lowered
