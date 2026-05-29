"""Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile."""

from __future__ import annotations

from pathlib import Path

import anyio

from spore_core.harness import (
    Operation,
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.sandbox import SandboxViolationException
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from ._common import LARGE_OUTPUT_THRESHOLD, finish_with_possible_truncation
from .error import (
    InvalidParameters,
    SandboxViolationError,
    ToolExecutionError,
)
from .params import (
    DeleteFileParams,
    ListDirParams,
    MoveFileParams,
    ReadFileParams,
    WriteFileParams,
    parse_params,
)


# ============================================================================
# ReadFile
# ============================================================================


class ReadFileTool:
    NAME = "read_file"

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
            description="Read a file's contents",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            annotations=ToolAnnotations(read_only=True, idempotent=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(ReadFileParams, call)
            resolved = await _resolve(sandbox, params.path, "read")
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            content = await anyio.to_thread.run_sync(Path(resolved).read_text)
        except OSError as e:
            return ToolOutputError(message=f"read failed: {e}", recoverable=True)
        return await finish_with_possible_truncation(content, call.id, sandbox)


# ============================================================================
# WriteFile
# ============================================================================


class WriteFileTool:
    NAME = "write_file"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description=(
                "Write content to a file (overwrites by default; set append=true to append)"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
            annotations=ToolAnnotations(destructive=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(WriteFileParams, call)
            resolved = await _resolve(sandbox, params.path, "write")
        except ToolExecutionError as e:
            return e.to_tool_output()

        def _write() -> None:
            mode = "ab" if params.append else "wb"
            with open(resolved, mode) as f:
                f.write(params.content.encode("utf-8"))

        try:
            await anyio.to_thread.run_sync(_write)
        except OSError as e:
            return ToolOutputError(message=f"write failed: {e}", recoverable=True)
        return ToolOutputSuccess(
            content=f"wrote {len(params.content)} bytes to {params.path}",
            truncated=False,
        )


# ============================================================================
# ListDir
# ============================================================================


class ListDirTool:
    NAME = "list_dir"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="List directory entries (optionally recursive)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
                "required": ["path"],
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(ListDirParams, call)
            resolved = await _resolve(sandbox, params.path, "read")
        except ToolExecutionError as e:
            return e.to_tool_output()

        def _gather() -> list[str]:
            root = Path(resolved)
            out: list[str] = []
            if params.recursive:
                for p in root.rglob("*"):
                    out.append(str(p))
                # include root itself to mirror Rust's WalkDir behavior
                out.append(str(root))
            else:
                for p in root.iterdir():
                    out.append(str(p))
            out.sort()
            return out

        try:
            entries = await anyio.to_thread.run_sync(_gather)
        except OSError as e:
            return ToolOutputError(message=f"read_dir failed: {e}", recoverable=True)
        content = "\n".join(entries)
        if len(content) > LARGE_OUTPUT_THRESHOLD:
            return await finish_with_possible_truncation(content, call.id, sandbox)
        return ToolOutputSuccess(content=content, truncated=False)


# ============================================================================
# DeleteFile
# ============================================================================


class DeleteFileTool:
    NAME = "delete_file"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Delete a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            annotations=ToolAnnotations(destructive=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(DeleteFileParams, call)
            resolved = await _resolve(sandbox, params.path, "write")
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            await anyio.to_thread.run_sync(Path(resolved).unlink)
        except OSError as e:
            return ToolOutputError(message=f"delete failed: {e}", recoverable=True)
        return ToolOutputSuccess(content=f"deleted {params.path}", truncated=False)


# ============================================================================
# MoveFile
# ============================================================================


class MoveFileTool:
    NAME = "move_file"

    def name(self) -> str:
        return self.NAME

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    @classmethod
    def schema(cls) -> ToolSchema:
        return ToolSchema(
            name=cls.NAME,
            description="Move/rename a file",
            parameters={
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                },
                "required": ["src", "dst"],
            },
            annotations=ToolAnnotations(destructive=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(MoveFileParams, call)
            src = await _resolve(sandbox, params.src, "write")
            dst = await _resolve(sandbox, params.dst, "write")
        except ToolExecutionError as e:
            return e.to_tool_output()
        try:
            await anyio.to_thread.run_sync(lambda: Path(src).rename(dst))
        except OSError as e:
            return ToolOutputError(message=f"move failed: {e}", recoverable=True)
        return ToolOutputSuccess(content=f"moved {params.src} -> {params.dst}", truncated=False)


# ============================================================================
# helpers
# ============================================================================


async def _resolve(sandbox: SandboxProvider, path: str, operation: Operation = "read") -> Path:
    """Resolve a path through the sandbox, raising :class:`SandboxViolationError`."""

    try:
        return await sandbox.resolve_path(path, operation)
    except SandboxViolationError:
        raise
    except SandboxViolationException as e:
        raise SandboxViolationError(violation=e.violation) from e
    except Exception as e:  # noqa: BLE001 — keep contract loose for now
        raise InvalidParameters(reason=f"path resolve failed: {e}") from e


__all__ = [
    "DeleteFileTool",
    "ListDirTool",
    "MoveFileTool",
    "ReadFileTool",
    "WriteFileTool",
]
