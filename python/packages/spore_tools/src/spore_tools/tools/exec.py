"""Execution tools: Exec, BashCommand, RunTests.

Two distinct ways to run a process, with deliberately different contracts
(mirroring ``rust/crates/spore-core/src/tools/exec.rs``):

* :class:`ExecTool` (tool name ``"exec"``) runs **one program directly** — no
  shell. ``command`` + ``args`` are passed verbatim to
  :meth:`SandboxProvider.execute_command`, so there are no pipes, redirects,
  globbing, or ``$(...)``. Every argument is literal. This is the
  path-validated, no-injection-surface option.
* :class:`BashCommandTool` (tool name ``"bash_command"``) runs a **shell
  command line** via ``/bin/sh -c <script>``, so it supports pipes, redirects,
  globbing, and ``$(...)``. It is sugar over the same ``execute_command``
  primitive ``exec`` uses (``execute_command("/bin/sh", ["-c", script], …)``).
* :class:`RunTestsTool` (tool name ``"run_tests"``) splits a command string on
  whitespace and runs it shell-free inside a working directory.
"""

from __future__ import annotations

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
)
from spore_core.model import ToolCall
from spore_core.sandbox import SandboxViolationException
from spore_core.tool_registry import ToolAnnotations, ToolSchema

from ._common import finish_with_possible_truncation
from .error import InvalidParameters, SandboxViolationError, ToolExecutionError
from .params import ExecParams, RunTestsParams, ShellCommandParams, parse_params


class ExecTool:
    """Runs one program directly via :meth:`SandboxProvider.execute_command`.

    No shell: ``command`` + ``args`` are passed verbatim (no pipes, redirects,
    globbing, or ``$(...)``). Path-validated through the sandbox.
    """

    NAME = "exec"

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
                "Run one program directly. No shell: no pipes, redirects, "
                "globbing, or $(...). Args are passed verbatim."
            ),
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
            params = parse_params(ExecParams, call)
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


class BashCommandTool:
    """Runs a shell command line via ``/bin/sh -c <script>``.

    Supports pipes, redirects, globbing, and ``$(...)``. Sugar over the same
    :meth:`SandboxProvider.execute_command` primitive :class:`ExecTool` uses
    (``execute_command("/bin/sh", ["-c", script], working_dir?, timeout?)``).

    TRADEOFF: because the shell itself opens any files the script touches, this
    tool does NOT receive the per-path ``resolve_path``/``validate`` enforcement
    that ``read_file``/``write_file``/``exec`` get — only the optional
    ``working_dir`` is path-validated. It relies on the outer sandbox/container
    for isolation; :class:`ExecTool` remains the path-validated choice.
    ``/bin/sh`` assumes a Unix target (no Windows shell branch).
    """

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
            description=(
                "Execute a shell command line via /bin/sh -c. Supports pipes, "
                "redirects, globbing, and $(...)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script": {"type": "string"},
                    "working_dir": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["script"],
            },
            annotations=ToolAnnotations(destructive=True, open_world=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(ShellCommandParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        timeout = float(params.timeout) if params.timeout is not None else None
        # Only the optional working_dir is path-validated; the script's own file
        # accesses go through the shell, unvalidated (see class docstring).
        working = None
        if params.working_dir is not None:
            try:
                working = await sandbox.resolve_path(params.working_dir, "read")
            except SandboxViolationException as e:
                return SandboxViolationError(violation=e.violation).to_tool_output()
        out = await sandbox.execute_command("/bin/sh", ["-c", params.script], working, timeout)
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


__all__ = ["BashCommandTool", "ExecTool", "RunTestsTool"]
