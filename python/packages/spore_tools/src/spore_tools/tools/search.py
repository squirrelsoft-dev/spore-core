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
from spore_core.tool_registry import ToolAnnotations, ToolSchema

from ._common import finish_with_possible_truncation
from .error import InvalidParameters, ToolExecutionError
from .params import FindFilesParams, GrepFilesParams, parse_params


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

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
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

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
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


__all__ = ["FindFilesTool", "GrepFilesTool"]
