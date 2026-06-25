"""Tests for :mod:`spore_tools.tools.error` — ToolExecutionError mapping."""

from __future__ import annotations

from spore_core.harness import (
    SandboxPathEscape,
    ToolOutputError,
    ToolOutputSandboxViolation,
)
from spore_tools.tools.error import (
    ExecutionFailed,
    InvalidParameters,
    SandboxViolationError,
    Timeout,
)


def test_invalid_parameters_is_recoverable() -> None:
    out = InvalidParameters(reason="missing path").to_tool_output()
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True


def test_execution_failed_passes_through_flag() -> None:
    out = ExecutionFailed(reason="x", recoverable=False).to_tool_output()
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is False


def test_sandbox_violation_carries_typed_violation() -> None:
    # The conversion does NOT pre-decide recoverability — it carries the typed
    # violation through as :class:`ToolOutputSandboxViolation` so the harness can
    # apply its ``SandboxViolationPolicy`` (recoverable by default; halt on
    # opt-in). #150.
    v = SandboxPathEscape(path="/etc")
    out = SandboxViolationError(violation=v).to_tool_output()
    assert isinstance(out, ToolOutputSandboxViolation)
    assert out.violation == v


def test_timeout_is_recoverable() -> None:
    out = Timeout(after_seconds=5).to_tool_output()
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True
