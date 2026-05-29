"""EditFile tool (#81, net-new Tier-1 sandbox tool).

Mirrors ``rust/crates/spore-core/src/tools/edit.rs``.

``edit_file`` replaces the FIRST and ONLY occurrence of ``old_string`` with
``new_string`` in the file at ``path``. The match must be UNIQUE:

* ``old_string`` not found     → recoverable :class:`ToolOutputError`.
* ``old_string`` found > 1 time → recoverable :class:`ToolOutputError`.

This is a net-new tool that does NOT replace ``write_file`` (issue #81, Q5). It
is annotated ``destructive`` (it mutates a file in place).
"""

from __future__ import annotations

from pathlib import Path

import anyio

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .error import ToolExecutionError
from .fs import _resolve
from .params import EditFileParams, parse_params


class EditFileTool:
    NAME = "edit_file"

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
            description="Replace the unique occurrence of old_string with new_string in a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            annotations=ToolAnnotations(destructive=True),
        )

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            params = parse_params(EditFileParams, call)
            resolved = await _resolve(sandbox, params.path, "write")
        except ToolExecutionError as e:
            return e.to_tool_output()

        try:
            content = await anyio.to_thread.run_sync(Path(resolved).read_text)
        except OSError as e:
            return ToolOutputError(message=f"read failed: {e}", recoverable=True)

        count = content.count(params.old_string)
        if count == 0:
            return ToolOutputError(
                message=f"old_string not found in {params.path}",
                recoverable=True,
            )
        if count > 1:
            return ToolOutputError(
                message=(
                    f"old_string is not unique in {params.path} ({count} occurrences); "
                    "provide more context"
                ),
                recoverable=True,
            )

        updated = content.replace(params.old_string, params.new_string, 1)

        def _write() -> None:
            with open(resolved, "wb") as f:
                f.write(updated.encode("utf-8"))

        try:
            await anyio.to_thread.run_sync(_write)
        except OSError as e:
            return ToolOutputError(message=f"write failed: {e}", recoverable=True)
        return ToolOutputSuccess(content=f"edited {params.path}", truncated=False)


__all__ = ["EditFileTool"]
