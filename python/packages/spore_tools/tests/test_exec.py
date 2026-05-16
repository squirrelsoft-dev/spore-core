"""Tests for execution tools."""

from __future__ import annotations

import shutil
import sys

import pytest

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox
from spore_tools.tools.exec import BashCommandTool

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX tools only")


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


async def test_echo_runs_and_returns_stdout() -> None:
    if not shutil.which("echo"):
        pytest.skip("no echo binary")
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(
        _call("bash_command", {"command": "echo", "args": ["hi"]}), sb
    )
    assert isinstance(r, ToolOutputSuccess)
    assert "hi" in r.content


async def test_nonzero_exit_is_recoverable_error() -> None:
    if not shutil.which("false"):
        pytest.skip("no false binary")
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(_call("bash_command", {"command": "false"}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_timeout_returns_recoverable_error() -> None:
    if not shutil.which("sleep"):
        pytest.skip("no sleep binary")
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(
        _call("bash_command", {"command": "sleep", "args": ["5"], "timeout": 1}), sb
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "timed out" in r.message


async def test_invalid_params_returns_recoverable_error() -> None:
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(_call("bash_command", {}), sb)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
