"""Execution tools: BashCommand, RunTests."""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolSchema

from ._common import finish_with_possible_truncation
from .error import InvalidParameters, ToolExecutionError
from .params import BashCommandParams, RunTestsParams, parse_params


class BashCommandTool:
    NAME = "bash_command"

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
            description="Execute a shell command via the sandbox",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
            annotations=ToolAnnotations(destructive=True, open_world=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(BashCommandParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        timeout = float(params.timeout) if params.timeout is not None else None
        out = await sandbox.execute_command(params.command, params.args, None, timeout)
        if out.timed_out:
            secs = int(timeout) if timeout is not None else 0
            return ToolOutputError(message=f"command timed out after {secs}s", recoverable=True)
        if out.exit_code == 0:
            return await finish_with_possible_truncation(out.stdout, call.id, sandbox)
        return ToolOutputError(
            message=f"exit {out.exit_code} ; stderr: {out.stderr.rstrip()}",
            recoverable=True,
        )


class RunTestsTool:
    NAME = "run_tests"

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
            description="Run a test command in a working directory",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_dir": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command", "working_dir"],
            },
            annotations=ToolAnnotations(open_world=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(RunTestsParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        working = await sandbox.resolve_path(params.working_dir, "execute")
        parts = params.command.split()
        if not parts:
            return InvalidParameters(reason="command must not be empty").to_tool_output()
        program, *args = parts
        timeout = float(params.timeout) if params.timeout is not None else None
        out = await sandbox.execute_command(program, args, working, timeout)
        if out.timed_out:
            secs = int(timeout) if timeout is not None else 0
            return ToolOutputError(message=f"tests timed out after {secs}s", recoverable=True)
        combined = f"{out.stdout}\n{out.stderr}"
        if out.exit_code == 0:
            return await finish_with_possible_truncation(combined, call.id, sandbox)
        return ToolOutputError(
            message=f"tests failed (exit {out.exit_code}): {combined}",
            recoverable=True,
        )


__all__ = ["BashCommandTool", "RunTestsTool"]
