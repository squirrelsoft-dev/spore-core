"""Search tools: GrepFiles, FindFiles."""

from __future__ import annotations

import re
from pathlib import Path

import anyio

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from ._common import finish_with_possible_truncation
from .error import InvalidParameters, ToolExecutionError
from .params import (
    FindFilesParams,
    GrepFilesParams,
    GrepOutputMode,
    GrepParams,
    parse_params,
)


def _build_context_output(
    path: str,
    all_lines: list[str],
    match_indices: list[int],
    context: int,
) -> str:
    """Build standard ``grep -C N`` output for a single file.

    ``all_lines`` is the full file as a list of lines (not newline-terminated).
    ``match_indices`` is a sorted list of 0-based line indices that matched.
    ``context`` is the number of context lines on each side.
    Returns the formatted output string; empty string when no matches.
    """
    if not match_indices:
        return ""

    last = len(all_lines) - 1

    # Expand each match into a [start, end] window (inclusive, 0-based).
    windows: list[tuple[int, int]] = []
    for mi in match_indices:
        start = max(0, mi - context)
        end = min(last, mi + context)
        windows.append((start, end))

    # Merge overlapping/adjacent windows (input is already sorted).
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    match_set = set(match_indices)
    parts: list[str] = []
    for gi, (win_start, win_end) in enumerate(merged):
        if gi > 0:
            parts.append("--")
        for line_idx in range(win_start, win_end + 1):
            line_num = line_idx + 1  # 1-indexed
            text = all_lines[line_idx]
            if line_idx in match_set:
                parts.append(f"{path}:{line_num}:{text}")
            else:
                parts.append(f"{path}:{line_num}-{text}")

    return "\n".join(parts)


class GrepFilesTool:
    NAME = "grep_files"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return True

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Search files for a regex pattern",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["pattern", "path"],
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(GrepFilesParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            regex = re.compile(params.pattern)
        except re.error as e:
            return InvalidParameters(reason=f"invalid regex: {e}").to_tool_output()
        root = await sandbox.resolve_path(params.path, "read")

        def _scan() -> list[str]:
            matches: list[tuple[str, int, str]] = []
            root_path = Path(root)
            files: list[Path]
            if params.recursive:
                files = [p for p in root_path.rglob("*") if p.is_file()]
            elif root_path.is_file():
                files = [root_path]
            else:
                files = [p for p in root_path.iterdir() if p.is_file()]
            for f in files:
                try:
                    text = f.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        matches.append((str(f), i, line))
            matches.sort(key=lambda m: (m[0], m[1]))
            return [f"{p}:{n}:{t}" for p, n, t in matches]

        try:
            lines = await anyio.to_thread.run_sync(_scan)
        except OSError as e:
            return InvalidParameters(reason=f"scan failed: {e}").to_tool_output()
        body = "\n".join(lines)
        return await finish_with_possible_truncation(body, call.id, sandbox)


class GrepTool:
    """Net-new regex search with selectable output mode (#81), alongside the
    byte-identical :class:`GrepFilesTool` (``grep_files``). ``read_only`` like
    ``grep_files`` but adds an ``output_mode``:

    * ``content``            → ``path:line:text`` per matching line (default).
    * ``files_with_matches`` → distinct file paths that contain a match.
    * ``count``              → ``path:count`` per file with matches.
    """

    NAME = "grep"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return True

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Search files for a regex pattern with selectable output mode",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "count", "files_with_matches"],
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": (
                            "Lines of context to show before and after each match"
                            " (default 0). When > 0, uses standard grep -C N format:"
                            " match lines use `:` separator, context lines use `-`,"
                            " non-adjacent groups separated by `--`."
                        ),
                    },
                },
                "required": ["pattern", "path"],
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(GrepParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            regex = re.compile(params.pattern)
        except re.error as e:
            return InvalidParameters(reason=f"invalid regex: {e}").to_tool_output()
        root = await sandbox.resolve_path(params.path, "read")

        use_context = params.context_lines > 0 and params.output_mode is GrepOutputMode.CONTENT

        if use_context:
            # Per-file scan: collect full file lines + match indices for context output.
            def _scan_with_context() -> list[str]:
                root_path = Path(root)
                files: list[Path]
                if params.recursive:
                    files = sorted(p for p in root_path.rglob("*") if p.is_file())
                elif root_path.is_file():
                    files = [root_path]
                else:
                    files = sorted(p for p in root_path.iterdir() if p.is_file())
                file_parts: list[str] = []
                for f in files:
                    try:
                        text = f.read_text()
                    except (OSError, UnicodeDecodeError):
                        continue
                    all_lines = text.splitlines()
                    match_indices = [i for i, line in enumerate(all_lines) if regex.search(line)]
                    part = _build_context_output(
                        str(f), all_lines, match_indices, params.context_lines
                    )
                    if part:
                        file_parts.append(part)
                return file_parts

            try:
                file_parts = await anyio.to_thread.run_sync(_scan_with_context)
            except OSError as e:
                return InvalidParameters(reason=f"scan failed: {e}").to_tool_output()

            body = "\n--\n".join(file_parts)
            return await finish_with_possible_truncation(body, call.id, sandbox)

        def _scan() -> list[tuple[str, int, str]]:
            matches: list[tuple[str, int, str]] = []
            root_path = Path(root)
            files: list[Path]
            if params.recursive:
                files = [p for p in root_path.rglob("*") if p.is_file()]
            elif root_path.is_file():
                files = [root_path]
            else:
                files = [p for p in root_path.iterdir() if p.is_file()]
            for f in files:
                try:
                    text = f.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        matches.append((str(f), i, line))
            matches.sort(key=lambda m: (m[0], m[1]))
            return matches

        try:
            matches = await anyio.to_thread.run_sync(_scan)
        except OSError as e:
            return InvalidParameters(reason=f"scan failed: {e}").to_tool_output()

        if params.output_mode is GrepOutputMode.CONTENT:
            lines = [f"{p}:{n}:{t}" for p, n, t in matches]
        elif params.output_mode is GrepOutputMode.FILES_WITH_MATCHES:
            # matches are sorted by path; collapse to distinct consecutive files.
            files_seen: list[str] = []
            for p, _n, _t in matches:
                if not files_seen or files_seen[-1] != p:
                    files_seen.append(p)
            lines = files_seen
        else:  # COUNT — per-file counts over the path-sorted matches.
            counts: list[tuple[str, int]] = []
            for p, _n, _t in matches:
                if counts and counts[-1][0] == p:
                    counts[-1] = (p, counts[-1][1] + 1)
                else:
                    counts.append((p, 1))
            lines = [f"{p}:{c}" for p, c in counts]

        body = "\n".join(lines)
        return await finish_with_possible_truncation(body, call.id, sandbox)


class FindFilesTool:
    NAME = "find_files"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return True

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Find files matching a glob",
            parameters={
                "type": "object",
                "properties": {
                    "glob": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["glob", "path"],
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(FindFilesParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        root = await sandbox.resolve_path(params.path, "read")

        def _glob() -> list[str]:
            return sorted(str(p) for p in Path(root).glob(params.glob))

        try:
            entries = await anyio.to_thread.run_sync(_glob)
        except ValueError as e:
            return InvalidParameters(reason=f"invalid glob: {e}").to_tool_output()
        body = "\n".join(entries)
        return await finish_with_possible_truncation(body, call.id, sandbox)


__all__ = ["FindFilesTool", "GrepFilesTool", "GrepTool"]
