"""Git tools: GitLog, GitDiff, GitCommit, GitStatus, GitReset."""

from __future__ import annotations

from spore_core.harness import (
    CommandOutput,
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolSchema

from ._common import finish_with_possible_truncation
from .error import ToolExecutionError
from .params import (
    GitCommitParams,
    GitDiffParams,
    GitLogParams,
    GitResetMode,
    GitResetParams,
    GitStatusParams,
    parse_params,
)


async def _run_git(args: list[str], sandbox: SandboxProvider) -> CommandOutput:
    return await sandbox.execute_command("git", args, None, None)


def _classify(out: CommandOutput) -> tuple[bool, str]:
    """Return ``(is_error, content_or_message)``."""

    if out.exit_code == 0:
        return False, out.stdout
    return True, f"git exit {out.exit_code} ; {out.stderr.rstrip()}"


class GitLogTool:
    NAME = "git_log"

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
            description="Show recent git commits",
            parameters={
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "format": {"type": "string"},
                },
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(GitLogParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        args = ["log", "-n", str(params.n)]
        if params.format == "oneline":
            args.append("--oneline")
        else:
            args.append(f"--format={params.format}")
        out = await _run_git(args, sandbox)
        is_err, body = _classify(out)
        if is_err:
            return ToolOutputError(message=body, recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


class GitDiffTool:
    NAME = "git_diff"

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
            description="Show a git diff",
            parameters={
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
            },
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(GitDiffParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        args = ["diff"]
        if params.from_ is not None:
            args.append(params.from_)
        if params.to is not None:
            args.append(params.to)
        out = await _run_git(args, sandbox)
        is_err, body = _classify(out)
        if is_err:
            return ToolOutputError(message=body, recoverable=True)
        return await finish_with_possible_truncation(body, call.id, sandbox)


class GitCommitTool:
    NAME = "git_commit"

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
            description="Stage files (if any) and create a git commit",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["message"],
            },
            annotations=ToolAnnotations(destructive=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(GitCommitParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        combined = ""
        if params.files:
            out = await _run_git(["add", *params.files], sandbox)
            is_err, body = _classify(out)
            if is_err:
                return ToolOutputError(message=body, recoverable=True)
            combined += body
        out = await _run_git(["commit", "-m", params.message], sandbox)
        is_err, body = _classify(out)
        if is_err:
            return ToolOutputError(message=body, recoverable=True)
        combined += body
        return ToolOutputSuccess(content=combined, truncated=False)


class GitStatusTool:
    NAME = "git_status"

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
            description="Show git status (porcelain)",
            parameters={"type": "object", "properties": {}},
            annotations=ToolAnnotations(read_only=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        # GitStatusParams accepts {} or extra-rejecting empty object.
        try:
            parse_params(GitStatusParams, call)
        except ToolExecutionError:
            # status takes no params — be lenient like the Rust reference.
            pass
        out = await _run_git(["status", "--porcelain"], sandbox)
        is_err, body = _classify(out)
        if is_err:
            return ToolOutputError(message=body, recoverable=True)
        return ToolOutputSuccess(content=body, truncated=False)


class GitResetTool:
    NAME = "git_reset"

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
            description="Reset to a target commit (hard/soft/mixed)",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["hard", "soft", "mixed"],
                    },
                },
                "required": ["target", "mode"],
            },
            annotations=ToolAnnotations(destructive=True),
        )

    async def execute(self, call: ToolCall, sandbox: SandboxProvider) -> ToolOutput:
        try:
            params = parse_params(GitResetParams, call)
        except ToolExecutionError as e:
            return e.to_tool_output()
        flag = {
            GitResetMode.HARD: "--hard",
            GitResetMode.SOFT: "--soft",
            GitResetMode.MIXED: "--mixed",
        }[params.mode]
        out = await _run_git(["reset", flag, params.target], sandbox)
        is_err, body = _classify(out)
        if is_err:
            return ToolOutputError(message=body, recoverable=True)
        return ToolOutputSuccess(content=body, truncated=False)


__all__ = [
    "GitCommitTool",
    "GitDiffTool",
    "GitLogTool",
    "GitResetMode",
    "GitResetTool",
    "GitStatusTool",
]
