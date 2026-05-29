"""Tests for the net-new ``edit_file`` tool (#81)."""

from __future__ import annotations

from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.edit import EditFileTool

_CTX = make_test_ctx()


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=EditFileTool.NAME, input=input_)


async def test_edit_replaces_unique_occurrence(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello world\n")
    r = await EditFileTool().execute(
        _call({"path": str(p), "old_string": "world", "new_string": "there"}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    assert p.read_text() == "hello there\n"


async def test_edit_not_found_is_recoverable_error(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello\n")
    r = await EditFileTool().execute(
        _call({"path": str(p), "old_string": "absent", "new_string": "x"}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "not found" in r.message


async def test_edit_non_unique_is_recoverable_error(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("x x x\n")
    r = await EditFileTool().execute(
        _call({"path": str(p), "old_string": "x", "new_string": "y"}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "not unique" in r.message


async def test_edit_missing_file_is_recoverable_error() -> None:
    r = await EditFileTool().execute(
        _call({"path": "/no/such/file", "old_string": "a", "new_string": "b"}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_edit_bad_params_is_recoverable_error() -> None:
    r = await EditFileTool().execute(_call({"path": "/x"}), AllowAllSandbox(), _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


def test_schema_is_destructive() -> None:
    s = EditFileTool.schema()
    assert s.annotations.destructive is True
    assert s.annotations.read_only is False
