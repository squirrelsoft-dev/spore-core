"""Tests for git tools — skipped if ``git`` binary is unavailable."""

from __future__ import annotations

import shutil

import pytest

from pathlib import Path

from spore_core.harness import (
    CommandOutput,
    SandboxExecSpawnFailed,
    ToolOutputError,
    ToolOutputSandboxViolation,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.sandbox import SandboxViolationException
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.git import GitResetMode, GitStatusTool
from spore_tools.tools.params import GitResetParams

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


class _SpawnFailSandbox(AllowAllSandbox):
    """Sandbox whose ``execute_command`` always raises a spawn violation, to
    exercise the SC-15 caller-audit path without depending on a real binary."""

    async def execute_command(
        self,
        command: str,
        args: list[str],
        working_dir: Path | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        raise SandboxViolationException(
            SandboxExecSpawnFailed(command=command, message="No such file or directory")
        )


async def test_git_status_runs() -> None:
    sb = AllowAllSandbox()
    r = await GitStatusTool().execute(_call("git_status", {}), sb, _CTX)
    # Either Success (in a repo) or Error (outside a repo) — both fine.
    assert isinstance(r, (ToolOutputSuccess, ToolOutputError))


async def test_git_spawn_failure_surfaces_typed_violation() -> None:
    # SC-15: a git spawn failure flows _run_git → SandboxViolationError →
    # ToolOutputSandboxViolation carrying the typed SandboxExecSpawnFailed (the
    # harness applies its policy; recoverable feedback by default).
    sb = _SpawnFailSandbox()
    r = await GitStatusTool().execute(_call("git_status", {}), sb, _CTX)
    assert isinstance(r, ToolOutputSandboxViolation)
    assert isinstance(r.violation, SandboxExecSpawnFailed)
    assert r.violation.command == "git"


def test_reset_mode_roundtrips_snake_case() -> None:
    p = GitResetParams.model_validate({"target": "HEAD", "mode": "hard"})
    assert p.mode is GitResetMode.HARD
    assert p.model_dump()["mode"] == "hard"
