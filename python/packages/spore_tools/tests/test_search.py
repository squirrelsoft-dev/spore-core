"""Tests for search tools."""

from __future__ import annotations

from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.search import FindFilesTool, GrepFilesTool, GrepTool

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


# ---- GrepTool output modes (#81) ------------------------------------------


async def test_grep_content_mode(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\nbeta\nalpha2\n")
    r = await GrepTool().execute(
        _call("grep", {"pattern": "alpha", "path": str(tmp_path), "output_mode": "content"}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 2
    assert lines[0].endswith(":1:alpha")
    assert lines[1].endswith(":3:alpha2")


async def test_grep_default_mode_is_content(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n")
    r = await GrepTool().execute(
        _call("grep", {"pattern": "alpha", "path": str(tmp_path)}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(":1:alpha")


async def test_grep_files_with_matches_mode(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\nalpha\n")
    (tmp_path / "b.txt").write_text("nope\n")
    r = await GrepTool().execute(
        _call(
            "grep",
            {"pattern": "alpha", "path": str(tmp_path), "output_mode": "files_with_matches"},
        ),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("a.txt")


async def test_grep_count_mode(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\nalpha\nx\n")
    r = await GrepTool().execute(
        _call("grep", {"pattern": "alpha", "path": str(tmp_path), "output_mode": "count"}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(":2")


async def test_grep_invalid_regex_recoverable(tmp_path: Path) -> None:
    r = await GrepTool().execute(
        _call("grep", {"pattern": "(unclosed", "path": str(tmp_path)}),
        AllowAllSandbox(),
        _CTX,
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


def test_grep_schema_read_only() -> None:
    assert GrepTool.schema().annotations.read_only is True


async def test_grep_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\nbeta\nalpha2")
    sb = AllowAllSandbox()
    r = await GrepFilesTool().execute(
        _call("grep_files", {"pattern": "^alpha", "path": str(tmp_path)}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess)
    assert "alpha" in r.content
    assert "alpha2" in r.content


async def test_grep_invalid_regex_returns_invalid_params(tmp_path: Path) -> None:
    sb = AllowAllSandbox()
    r = await GrepFilesTool().execute(
        _call("grep_files", {"pattern": "(unclosed", "path": str(tmp_path)}), sb, _CTX
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_find_files_glob(tmp_path: Path) -> None:
    (tmp_path / "a.rs").write_text("")
    (tmp_path / "b.rs").write_text("")
    (tmp_path / "c.txt").write_text("")
    sb = AllowAllSandbox()
    r = await FindFilesTool().execute(
        _call("find_files", {"glob": "*.rs", "path": str(tmp_path)}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 2
    assert lines == sorted(lines)
