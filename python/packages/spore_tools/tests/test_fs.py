"""Tests for filesystem tools."""

from __future__ import annotations

from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox
from spore_tools.tools.fs import (
    DeleteFileTool,
    ListDirTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
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
