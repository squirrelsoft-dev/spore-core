"""Filesystem tools: ReadFile, WriteFile, ListDir, DeleteFile, MoveFile."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

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
            description=(
                "Read a file's contents. Optionally read a line range "
                "(offset is 1-indexed start, length is max lines, 0 = "
                "to EOF) and/or prefix each line with its number via "
                "line_numbers. With no optional params the whole file "
                "is returned verbatim."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "description": "1-indexed start line (default 1).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Max lines to return; 0 = no limit / read to EOF (default 0).",
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": "Prefix each returned line with its 1-indexed number (default false).",
                    },
                },
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
        result = _apply_read_range(content, params)
        if isinstance(result, str):
            return await finish_with_possible_truncation(result, call.id, sandbox)
        return result


# ============================================================================
# _apply_read_range (#132)
# ============================================================================


def _apply_read_range(content: str, params: ReadFileParams) -> str | ToolOutputError:
    """Apply the #132 range/line-number transform to a fully-read file body.

    Returns the transformed content string, or a :class:`ToolOutputError` for
    recoverable errors. With all params at their defaults the original
    ``content`` is returned unchanged (byte-identical to the pre-#132 behavior).
    Any non-default param prepends a ``[lines {start}–{end} of {total}]`` header
    (U+2013 en-dash).
    """
    is_default = params.offset == 1 and params.length == 0 and not params.line_numbers
    if is_default:
        return content
    if params.offset == 0:
        return ToolOutputError(message="offset must be ≥ 1 (1-indexed)", recoverable=True)
    # Empty file: any params still yield empty content with no header.
    if not content:
        return ""
    # splitlines(keepends=True) preserves each line's trailing '\n'; the final
    # line may or may not end in '\n'. This keeps the slice byte-faithful to the
    # source (Python equivalent of Rust's split_inclusive('\n')).
    lines = content.splitlines(keepends=True)
    total = len(lines)
    if params.offset > total:
        return ToolOutputError(
            message=f"offset {params.offset} exceeds file length {total}",
            recoverable=True,
        )
    start = params.offset  # 1-indexed, validated >= 1 and <= total
    end = total if params.length == 0 else min(start + params.length - 1, total)
    selected = lines[start - 1 : end]

    out = f"[lines {start}–{end} of {total}]\n"
    if params.line_numbers:
        width = len(str(total))
        for i, line in enumerate(selected):
            n = start + i
            out += f"{n:>{width}} | {line}"
    else:
        out += "".join(selected)
    return out


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
            # Emit paths relative to the workspace root so each entry can be fed
            # straight back into read_file/write_file. The sandbox treats every
            # input path as root-relative, so absolute paths would not round-trip
            # (see #93). ``resolved`` is the absolute path of the listed directory
            # (= root-relative ``params.path``); each entry is under it.
            # Relativize against ``resolved``, then re-anchor onto the
            # root-relative ``params.path``, dropping any leading ``./``.
            listed = PurePosixPath(*Path(params.path).parts)

            def to_root_relative(entry: Path) -> str | None:
                try:
                    rel_to_listed = entry.relative_to(root)
                except ValueError:
                    return None
                # Skip the listed directory itself (an empty relative path).
                if rel_to_listed == Path():
                    return None
                anchored = listed / PurePosixPath(*rel_to_listed.parts)
                # Drop ``CurDir`` (``.``) components so ``.``/empty inputs yield
                # bare names.
                normalized = PurePosixPath(*(part for part in anchored.parts if part != "."))
                return normalized.as_posix()

            out: list[str] = []
            if params.recursive:
                for p in root.rglob("*"):
                    rel = to_root_relative(p)
                    if rel is not None:
                        out.append(rel)
                # mirror Rust's WalkDir behavior, which yields the root itself;
                # to_root_relative skips it (empty relative path), so this is a
                # no-op for the listed dir but keeps the branches symmetric.
                rel = to_root_relative(root)
                if rel is not None:
                    out.append(rel)
            else:
                for p in root.iterdir():
                    rel = to_root_relative(p)
                    if rel is not None:
                        out.append(rel)
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
