"""Tests for search tools."""

from __future__ import annotations

import json
from pathlib import Path

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.search import FindFilesTool, GrepFilesTool, GrepTool, _build_context_output

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


# ---- _build_context_output unit tests (#133) ----------------------------------


def test_context_output_empty_matches() -> None:
    assert _build_context_output("/f", ["a", "b"], [], 1) == ""


def test_context_output_zero_context_single_match() -> None:
    # context=0 is handled upstream; helper still works when called with 0
    result = _build_context_output("/f", ["alpha", "beta", "gamma"], [1], 0)
    assert result == "/f:2:beta"


def test_context_output_single_match_with_context() -> None:
    lines = ["one", "two", "three", "four", "five"]
    result = _build_context_output("/f", lines, [2], 1)
    assert result == "/f:2-two\n/f:3:three\n/f:4-four"


def test_context_output_overlapping_windows_merged() -> None:
    # matches at indices 1 and 3 with context 2 → windows [0,3] and [1,4] → merged [0,4]
    lines = ["a", "b", "c", "d", "e"]
    result = _build_context_output("/f", lines, [1, 3], 2)
    assert result == "/f:1-a\n/f:2:b\n/f:3-c\n/f:4:d\n/f:5-e"


def test_context_output_non_adjacent_groups_separated() -> None:
    # match at index 0, context 1 → window [0,1]
    # match at index 9, context 1 → window [8,10]
    # gap between index 1 and 8 → separated by --
    lines = [
        "match1",
        "line2",
        "line3",
        "line4",
        "line5",
        "line6",
        "line7",
        "line8",
        "line9",
        "match10",
        "line11",
    ]
    result = _build_context_output("/f", lines, [0, 9], 1)
    expected = "/f:1:match1\n/f:2-line2\n--\n/f:9-line9\n/f:10:match10\n/f:11-line11"
    assert result == expected


def test_context_output_clamped_at_file_start() -> None:
    lines = ["match", "line2", "line3", "line4", "line5"]
    result = _build_context_output("/f", lines, [0], 3)
    assert result == "/f:1:match\n/f:2-line2\n/f:3-line3\n/f:4-line4"


def test_context_output_clamped_at_file_end() -> None:
    lines = ["line1", "line2", "line3", "line4", "match"]
    result = _build_context_output("/f", lines, [4], 3)
    assert result == "/f:2-line2\n/f:3-line3\n/f:4-line4\n/f:5:match"


def test_context_output_context_line_is_also_match_uses_colon() -> None:
    # both alpha (idx 0) and beta (idx 1) match; context 1 → single merged window [0,2]
    # idx 0 → match → `:`, idx 1 → match → `:`, idx 2 → context → `-`
    lines = ["alpha", "beta", "gamma"]
    result = _build_context_output("/f", lines, [0, 1], 1)
    assert result == "/f:1:alpha\n/f:2:beta\n/f:3-gamma"


# ---- GrepTool context_lines integration tests (#133) -------------------------


async def test_grep_context_lines_zero_unchanged(tmp_path: Path) -> None:
    """context_lines=0 produces byte-identical output to the default path."""
    (tmp_path / "a.txt").write_text("alpha\nbeta\ngamma\n")
    sb = AllowAllSandbox()
    r_default = await GrepTool().execute(
        _call("grep", {"pattern": "beta", "path": str(tmp_path / "a.txt")}),
        sb,
        _CTX,
    )
    r_explicit = await GrepTool().execute(
        _call("grep", {"pattern": "beta", "path": str(tmp_path / "a.txt"), "context_lines": 0}),
        sb,
        _CTX,
    )
    assert isinstance(r_default, ToolOutputSuccess)
    assert isinstance(r_explicit, ToolOutputSuccess)
    assert r_default.content == r_explicit.content


async def test_grep_context_lines_single_match(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\nthree\nfour\nfive\n")
    sb = AllowAllSandbox()
    r = await GrepTool().execute(
        _call("grep", {"pattern": "three", "path": str(tmp_path / "f.txt"), "context_lines": 1}),
        sb,
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    p = str(tmp_path / "f.txt")
    assert r.content == f"{p}:2-two\n{p}:3:three\n{p}:4-four"


async def test_grep_context_lines_non_adjacent_separator(tmp_path: Path) -> None:
    content = (
        "match1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nmatch10\nline11\nline12\n"
    )
    (tmp_path / "f.txt").write_text(content)
    sb = AllowAllSandbox()
    r = await GrepTool().execute(
        _call("grep", {"pattern": "match", "path": str(tmp_path / "f.txt"), "context_lines": 1}),
        sb,
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    p = str(tmp_path / "f.txt")
    expected = f"{p}:1:match1\n{p}:2-line2\n--\n{p}:9-line9\n{p}:10:match10\n{p}:11-line11"
    assert r.content == expected


async def test_grep_context_lines_ignored_for_non_content_modes(tmp_path: Path) -> None:
    """context_lines has no effect on count/files_with_matches modes."""
    (tmp_path / "f.txt").write_text("alpha\nbeta\nalpha2\n")
    sb = AllowAllSandbox()
    r = await GrepTool().execute(
        _call(
            "grep",
            {
                "pattern": "alpha",
                "path": str(tmp_path / "f.txt"),
                "output_mode": "count",
                "context_lines": 5,
            },
        ),
        sb,
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content.endswith(":2")


# ---- Fixture replay test (#133) ----------------------------------------------

_FIXTURE_PATH = Path(__file__).parents[4] / "fixtures" / "tools" / "grep_context_lines.json"


async def test_grep_context_lines_fixture_replay(tmp_path: Path) -> None:
    """Replay every case in grep_context_lines.json byte-identically."""
    fixture_data = json.loads(_FIXTURE_PATH.read_text())
    sb = AllowAllSandbox()

    for case in fixture_data:
        fixture_file = tmp_path / f"{case['name']}.txt"
        fixture_file.write_text(case["initial_content"])

        raw_params = dict(case["params"])
        # Replace the placeholder with the actual tmp file path.
        raw_params["path"] = str(fixture_file)
        expected = case["expected"].replace("<FIXTURE_PATH>", str(fixture_file))

        r = await GrepTool().execute(_call("grep", raw_params), sb, _CTX)
        assert isinstance(r, ToolOutputSuccess), f"case {case['name']!r} failed: {r}"
        assert r.content == expected, (
            f"case {case['name']!r}:\n  got:      {r.content!r}\n  expected: {expected!r}"
        )
