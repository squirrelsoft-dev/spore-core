"""Tests for execution tools (exec = shell-free, bash_command = real shell)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from spore_core.harness import ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.tool_registry import AllowAllSandbox, make_test_ctx
from spore_tools.tools.exec import BashCommandTool, ExecTool, _truncate_for_message

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX tools only")

_CTX = make_test_ctx()


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


# ---------------- ExecTool (shell-free) ----------------


async def test_exec_echo_runs_and_returns_stdout() -> None:
    if not shutil.which("echo"):
        pytest.skip("no echo binary")
    sb = AllowAllSandbox()
    r = await ExecTool().execute(_call("exec", {"command": "echo", "args": ["hi"]}), sb, _CTX)
    assert isinstance(r, ToolOutputSuccess)
    assert "hi" in r.content


async def test_exec_has_no_shell_semantics(tmp_path: Path) -> None:
    """`exec` must NOT interpret shell syntax: pipe/`$(...)`/redirect tokens are
    passed to `echo` as literal arguments, and no file is created."""
    if not shutil.which("echo"):
        pytest.skip("no echo binary")
    sb = AllowAllSandbox()
    out_file = tmp_path / "out"
    r = await ExecTool().execute(
        _call("exec", {"command": "echo", "args": ["a|b", "$(whoami)", ">out"]}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess), r
    assert "a|b $(whoami) >out" in r.content, f"args must be literal, got {r.content!r}"
    assert not out_file.exists(), "no redirect: `out` must not be created"


async def test_exec_nonzero_exit_is_recoverable_error() -> None:
    if not shutil.which("false"):
        pytest.skip("no false binary")
    sb = AllowAllSandbox()
    r = await ExecTool().execute(_call("exec", {"command": "false"}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_exec_timeout_returns_recoverable_error() -> None:
    if not shutil.which("sleep"):
        pytest.skip("no sleep binary")
    sb = AllowAllSandbox()
    r = await ExecTool().execute(
        _call("exec", {"command": "sleep", "args": ["5"], "timeout": 1}), sb, _CTX
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "timed out" in r.message


async def test_exec_invalid_params_returns_recoverable_error() -> None:
    sb = AllowAllSandbox()
    r = await ExecTool().execute(_call("exec", {}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


# ---------------- BashCommandTool (real shell) ----------------


async def test_bash_command_supports_pipeline() -> None:
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(
        _call("bash_command", {"script": "printf 'hi' | tr a-z A-Z"}), sb, _CTX
    )
    assert isinstance(r, ToolOutputSuccess), r
    assert r.content == "HI"


async def test_bash_command_supports_redirect() -> None:
    sb = AllowAllSandbox()
    tmp = Path(tempfile.gettempdir()) / f"spore-bash-redirect-{time.time_ns()}.txt"
    try:
        r = await BashCommandTool().execute(
            _call("bash_command", {"script": f"printf 'data' > {tmp}"}), sb, _CTX
        )
        assert isinstance(r, ToolOutputSuccess), r
        assert tmp.read_text() == "data"
    finally:
        tmp.unlink(missing_ok=True)


async def test_bash_command_nonzero_exit_is_recoverable_error() -> None:
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(_call("bash_command", {"script": "exit 3"}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_bash_command_timeout_returns_recoverable_error() -> None:
    if not shutil.which("sleep"):
        pytest.skip("no sleep binary")
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(
        _call("bash_command", {"script": "sleep 5", "timeout": 1}), sb, _CTX
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "timed out" in r.message


async def test_bash_command_invalid_params_returns_recoverable_error() -> None:
    sb = AllowAllSandbox()
    r = await BashCommandTool().execute(_call("bash_command", {}), sb, _CTX)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_bash_command_large_stderr_is_truncated_in_error_message() -> None:
    if not shutil.which("awk"):
        pytest.skip("no awk binary")
    sb = AllowAllSandbox()
    # awk writes 10 KB to stderr and exits non-zero; verify elision in message.
    r = await BashCommandTool().execute(
        _call(
            "bash_command",
            {"script": "awk 'BEGIN{for(i=0;i<10240;i++)printf \"x\" > \"/dev/stderr\"; exit 1}'"},
        ),
        sb,
        _CTX,
    )
    assert isinstance(r, ToolOutputError)
    assert "bytes elided" in r.message
    assert len(r.message) < 10 * 1024


# ---------------- _truncate_for_message unit tests ----------------


def test_truncate_for_message_passthrough_when_short() -> None:
    s = "small error output"
    assert _truncate_for_message(s) == s


def test_truncate_for_message_elides_middle_when_large() -> None:
    long_s = "x" * (10 * 1024)
    result = _truncate_for_message(long_s)
    assert "bytes elided" in result
    assert len(result) < 8 * 1024
