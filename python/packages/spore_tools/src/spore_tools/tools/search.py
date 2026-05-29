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
