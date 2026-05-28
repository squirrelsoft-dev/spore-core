"""Tests for filesystem tools."""

from __future__ import annotations

from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess, WorkspaceConfig
from spore_core.model import ToolCall
from spore_core.sandbox import WorkspaceScopedSandbox
from spore_core.tool_registry import AllowAllSandbox
from spore_tools.tools.fs import (
    DeleteFileTool,
    ListDirTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)


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
    w = await WriteFileTool().execute(_call("write_file", {"path": str(p), "content": "hello"}), sb)
    assert isinstance(w, ToolOutputSuccess)
    r = await ReadFileTool().execute(_call("read_file", {"path": str(p)}), sb)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "hello"


async def test_append_mode_concatenates(tmp_path: Path) -> None:
    sb = AllowAllSandbox()
    p = tmp_path / "a.txt"
    await WriteFileTool().execute(_call("write_file", {"path": str(p), "content": "a"}), sb)
    await WriteFileTool().execute(
        _call("write_file", {"path": str(p), "content": "b", "append": True}), sb
    )
    assert p.read_text() == "ab"


async def test_list_dir_sorted(tmp_path: Path) -> None:
    (tmp_path / "z").write_text("")
    (tmp_path / "a").write_text("")
    (tmp_path / "m").write_text("")
    sb = AllowAllSandbox()
    r = await ListDirTool().execute(_call("list_dir", {"path": str(tmp_path)}), sb)
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 3
    assert lines == sorted(lines)


async def test_delete_missing_is_recoverable() -> None:
    sb = AllowAllSandbox()
    r = await DeleteFileTool().execute(_call("delete_file", {"path": "/no/such/path/here"}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_move_file_renames(tmp_path: Path) -> None:
    src = tmp_path / "s"
    dst = tmp_path / "d"
    src.write_text("hi")
    sb = AllowAllSandbox()
    r = await MoveFileTool().execute(_call("move_file", {"src": str(src), "dst": str(dst)}), sb)
    assert isinstance(r, ToolOutputSuccess)
    assert not src.exists()
    assert dst.exists()


async def test_invalid_params_returns_recoverable_error() -> None:
    sb = AllowAllSandbox()
    r = await ReadFileTool().execute(_call("read_file", {}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_read_missing_in_workspace_file_is_recoverable_not_found(
    tmp_path: Path,
) -> None:
    # Regression for #63: reading a not-yet-created file *inside* the
    # workspace must surface a recoverable not-found, not a sandbox
    # PathEscape, end to end through the real WorkspaceScopedSandbox.
    sb = _workspace_sandbox(tmp_path)
    r = await ReadFileTool().execute(_call("read_file", {"path": "output.txt"}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "read failed" in r.message


async def test_read_outside_workspace_is_path_escape(tmp_path: Path) -> None:
    # Counterpart: a path that resolves outside the root is still a sandbox
    # violation, even when the file does not exist.
    sb = _workspace_sandbox(tmp_path)
    r = await ReadFileTool().execute(_call("read_file", {"path": "../nonexistent_secret"}), sb)
    assert isinstance(r, ToolOutputError)
    lowered = r.message.lower()
    assert "escape" in lowered or "sandbox" in lowered
