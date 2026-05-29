"""Tests for git tools — skipped if ``git`` binary is unavailable."""

from __future__ import annotations

import shutil

import pytest

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.git import GitResetMode, GitStatusTool
from spore_tools.tools.params import GitResetParams

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_git_status_runs() -> None:
    sb = AllowAllSandbox()
    r = await GitStatusTool().execute(_call("git_status", {}), sb, _CTX)
    # Either Success (in a repo) or Error (outside a repo) — both fine.
    assert isinstance(r, (ToolOutputSuccess, ToolOutputError))


def test_reset_mode_roundtrips_snake_case() -> None:
    p = GitResetParams.model_validate({"target": "HEAD", "mode": "hard"})
    assert p.mode is GitResetMode.HARD
    assert p.model_dump()["mode"] == "hard"
