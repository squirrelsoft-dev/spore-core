"""Tests for filesystem tools."""

from __future__ import annotations

import json
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
    _apply_read_range,
)
from spore_tools.tools.params import ReadFileParams

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


# ============================================================================
# #132: read_file range scan + line numbers (unit tests)
# ============================================================================


def _params(**kwargs) -> ReadFileParams:  # type: ignore[no-untyped-def]
    return ReadFileParams(path="f", **kwargs)


def test_read_range_defaults_are_byte_identical() -> None:
    body = "line1\nline2\nline3\n"
    result = _apply_read_range(body, _params())
    assert result == body


def test_read_range_offset_only_header_runs_to_eof() -> None:
    body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
    result = _apply_read_range(body, _params(offset=3))
    assert result == "[lines 3–10 of 10]\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"


def test_read_range_length_only() -> None:
    body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
    result = _apply_read_range(body, _params(length=3))
    assert result == "[lines 1–3 of 10]\nline1\nline2\nline3\n"


def test_read_range_offset_and_length() -> None:
    body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
    result = _apply_read_range(body, _params(offset=4, length=3))
    assert result == "[lines 4–6 of 10]\nline4\nline5\nline6\n"


def test_read_range_line_numbers_alone() -> None:
    body = "alpha\nbeta\ngamma\n"
    result = _apply_read_range(body, _params(line_numbers=True))
    assert result == "[lines 1–3 of 3]\n1 | alpha\n2 | beta\n3 | gamma\n"


def test_read_range_all_three_combined() -> None:
    body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
    result = _apply_read_range(body, _params(offset=2, length=3, line_numbers=True))
    assert result == "[lines 2–4 of 10]\n 2 | line2\n 3 | line3\n 4 | line4\n"


def test_read_range_line_numbers_pad_to_total_width() -> None:
    # 10-line file → width 2 → single-digit numbers are right-padded
    body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
    result = _apply_read_range(body, _params(offset=2, length=3, line_numbers=True))
    assert result == "[lines 2–4 of 10]\n 2 | line2\n 3 | line3\n 4 | line4\n"


def test_read_range_offset_past_eof_is_error() -> None:
    body = "alpha\nbeta\ngamma\n"
    result = _apply_read_range(body, _params(offset=11))
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True
    assert "offset 11 exceeds file length 3" in result.message


def test_read_range_length_trimmed_at_eof_silently() -> None:
    body = "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n"
    result = _apply_read_range(body, _params(offset=8, length=5))
    assert result == "[lines 8–10 of 10]\nline8\nline9\nline10\n"


def test_read_range_offset_zero_is_error() -> None:
    body = "alpha\nbeta\n"
    result = _apply_read_range(body, _params(offset=0))
    assert isinstance(result, ToolOutputError)
    assert result.recoverable is True
    assert "offset" in result.message


def test_read_range_length_zero_with_offset_means_no_limit() -> None:
    body = "line1\nline2\nline3\nline4\nline5\n"
    result = _apply_read_range(body, _params(offset=3, length=0))
    assert result == "[lines 3–5 of 5]\nline3\nline4\nline5\n"


def test_read_range_empty_file_any_params_no_header() -> None:
    result = _apply_read_range("", _params(offset=1, length=5, line_numbers=True))
    assert result == ""


def test_read_range_final_line_without_newline_preserved() -> None:
    body = "a\nb\nc"
    result = _apply_read_range(body, _params(offset=2))
    assert result == "[lines 2–3 of 3]\nb\nc"


async def test_read_file_with_offset_emits_header_end_to_end(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("l1\nl2\nl3\n")
    sb = AllowAllSandbox()
    r = await ReadFileTool().execute(_call("read_file", {"path": str(p), "offset": 2}), sb, _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "[lines 2–3 of 3]\nl2\nl3\n"


# ============================================================================
# #132: fixture replay
# ============================================================================

_FIXTURE_PATH = Path(__file__).parents[4] / "fixtures" / "tools" / "read_file_range.json"


# ============================================================================
# #134: list_dir gitignore parity
# ============================================================================


def _has_git_dir(entries: list[str]) -> bool:
    """Return True if any entry is inside the .git/ directory (not just .gitignore)."""
    return any(
        "/.git/" in e or e.endswith("/.git") or e == ".git" or e.startswith(".git/")
        for e in entries
    )


async def test_list_dir_recursive_excludes_gitignored_files(tmp_path: Path) -> None:
    """Recursive list_dir respects .gitignore by default."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("// bundle")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "app.log").write_text("log entry")
    (tmp_path / ".gitignore").write_text("dist/\n*.log\n")

    sb = AllowAllSandbox()
    r = await ListDirTool().execute(
        _call("list_dir", {"path": str(tmp_path), "recursive": True}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess)
    entries = r.content.splitlines()

    # Tracked file must appear
    assert any("main.py" in e for e in entries), f"src/main.py missing from {entries}"
    # Gitignored files must NOT appear
    assert not any("bundle.js" in e for e in entries), (
        f"dist/bundle.js should be excluded: {entries}"
    )
    assert not _has_git_dir(entries), f".git dir should be excluded: {entries}"
    assert not any("app.log" in e for e in entries), f"logs/app.log should be excluded: {entries}"


async def test_list_dir_recursive_include_ignored_restores_all(tmp_path: Path) -> None:
    """Recursive list_dir with include_ignored=True walks everything (except .git/)."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_text("// bundle")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "app.log").write_text("log entry")
    (tmp_path / ".gitignore").write_text("dist/\n*.log\n")

    sb = AllowAllSandbox()
    r = await ListDirTool().execute(
        _call(
            "list_dir",
            {"path": str(tmp_path), "recursive": True, "include_ignored": True},
        ),
        sb,
        _CTX,
    )
    assert isinstance(r, ToolOutputSuccess)
    entries = r.content.splitlines()

    # All non-.git paths must appear
    assert any("main.py" in e for e in entries), f"src/main.py missing from {entries}"
    assert any("bundle.js" in e for e in entries), f"dist/bundle.js should be included: {entries}"
    assert any("app.log" in e for e in entries), f"logs/app.log should be included: {entries}"
    # .git still excluded
    assert not _has_git_dir(entries), f".git dir should still be excluded: {entries}"


async def test_list_dir_non_recursive_excludes_git_dir(tmp_path: Path) -> None:
    """Non-recursive list_dir never returns .git/ entries."""
    (tmp_path / "readme.txt").write_text("hi")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")

    sb = AllowAllSandbox()
    r = await ListDirTool().execute(_call("list_dir", {"path": str(tmp_path)}), sb, _CTX)
    assert isinstance(r, ToolOutputSuccess)
    entries = r.content.splitlines()
    assert not _has_git_dir(entries), f".git dir should be excluded: {entries}"
    assert any("readme.txt" in e for e in entries), f"readme.txt missing: {entries}"


async def test_read_file_range_fixture_replay(tmp_path: Path) -> None:
    """Replay every case in fixtures/tools/read_file_range.json byte-identically."""
    cases = json.loads(_FIXTURE_PATH.read_text())
    sb = AllowAllSandbox()
    fixture_file = tmp_path / "fixture.txt"

    for case in cases:
        name: str = case["name"]
        initial_content: str = case["initial_content"]
        params: dict = dict(case["params"])
        expected: dict = case["expected"]

        # Write the fixture file with the initial content for this case.
        fixture_file.write_text(initial_content, encoding="utf-8")

        # Replace the placeholder path with the real temp file path.
        params["path"] = str(fixture_file)

        r = await ReadFileTool().execute(_call("read_file", params), sb, _CTX)

        if expected["kind"] == "success":
            assert isinstance(r, ToolOutputSuccess), f"[{name}] expected success, got error: {r}"
            assert r.content == expected["content"], (
                f"[{name}] content mismatch:\n"
                f"  expected: {expected['content']!r}\n"
                f"  got:      {r.content!r}"
            )
        else:
            assert isinstance(r, ToolOutputError), (
                f"[{name}] expected error, got success: {r.content!r}"
            )
            assert r.recoverable is expected.get("recoverable", True), (
                f"[{name}] recoverable mismatch"
            )
            if "message_contains" in expected:
                assert expected["message_contains"] in r.message, (
                    f"[{name}] message {r.message!r} missing {expected['message_contains']!r}"
                )
