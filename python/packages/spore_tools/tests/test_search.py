"""Tests for search tools."""

from __future__ import annotations

from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox
from spore_tools.tools.search import FindFilesTool, GrepFilesTool


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_grep_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\nbeta\nalpha2")
    sb = AllowAllSandbox()
    r = await GrepFilesTool().execute(
        _call("grep_files", {"pattern": "^alpha", "path": str(tmp_path)}), sb
    )
    assert isinstance(r, ToolOutputSuccess)
    assert "alpha" in r.content
    assert "alpha2" in r.content


async def test_grep_invalid_regex_returns_invalid_params(tmp_path: Path) -> None:
    sb = AllowAllSandbox()
    r = await GrepFilesTool().execute(
        _call("grep_files", {"pattern": "(unclosed", "path": str(tmp_path)}), sb
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_find_files_glob(tmp_path: Path) -> None:
    (tmp_path / "a.rs").write_text("")
    (tmp_path / "b.rs").write_text("")
    (tmp_path / "c.txt").write_text("")
    sb = AllowAllSandbox()
    r = await FindFilesTool().execute(
        _call("find_files", {"glob": "*.rs", "path": str(tmp_path)}), sb
    )
    assert isinstance(r, ToolOutputSuccess)
    lines = r.content.splitlines()
    assert len(lines) == 2
    assert lines == sorted(lines)
