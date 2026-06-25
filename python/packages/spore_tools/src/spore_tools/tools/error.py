"""``ToolExecutionError`` — typed error hierarchy for tool implementations.

Mirrors ``rust/crates/spore-core/src/tools/error.rs``. Error → :class:`ToolOutput`
mapping:

* :class:`InvalidParameters` — :class:`ToolOutputError` (``recoverable=True``)
* :class:`Timeout` — :class:`ToolOutputError` (``recoverable=True``)
* :class:`ExecutionFailed` — :class:`ToolOutputError` (caller-specified ``recoverable``)
* :class:`SandboxViolationError` — :class:`ToolOutputSandboxViolation` (typed violation)

A tool-surfaced sandbox violation is NOT flattened into a recoverable-or-not
:class:`ToolOutputError` here — that decision is the HARNESS's, not the tool's.
The conversion carries the typed violation through as
:class:`ToolOutputSandboxViolation`, and the harness applies its
``SandboxViolationPolicy``: by default the model is fed a recoverable error and
retries (the boundary still holds — the access was refused); under ``HALT`` an
always-halt-eligible violation ends the run with a typed
``HaltReasonSandboxViolation``. Keeping the violation typed all the way to the
harness is what makes the policy uniform across every tool and both surfacing
paths (this one and the pre-dispatch ``validate`` check). See issue #150.

Tools convert errors via :meth:`ToolExecutionError.to_tool_output` so the
registry can stay on its happy path.
"""

from __future__ import annotations

from dataclasses import dataclass

from spore_core.errors import SporeError
from spore_core.harness import (
    SandboxViolation as HarnessSandboxViolation,
)
from spore_core.harness import (
    ToolOutput,
    ToolOutputError,
    ToolOutputSandboxViolation,
)


class ToolExecutionError(SporeError):
    """Base class for typed tool-execution errors."""

    def to_tool_output(self) -> ToolOutput:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class InvalidParameters(ToolExecutionError):
    """Parameter validation failed. Recoverable."""

    reason: str

    def __post_init__(self) -> None:
        super().__init__(f"invalid parameters: {self.reason}")

    def to_tool_output(self) -> ToolOutput:
        return ToolOutputError(message=f"invalid parameters: {self.reason}", recoverable=True)


@dataclass
class ExecutionFailed(ToolExecutionError):
    """Tool execution failed. ``recoverable`` is caller-specified."""

    reason: str
    recoverable: bool = True

    def __post_init__(self) -> None:
        super().__init__(f"execution failed: {self.reason}")

    def to_tool_output(self) -> ToolOutput:
        return ToolOutputError(message=self.reason, recoverable=self.recoverable)


@dataclass
class SandboxViolationError(ToolExecutionError):
    """Sandbox rejected the operation. The conversion carries the TYPED
    violation to the harness, which applies the configured
    ``SandboxViolationPolicy`` (recoverable feedback by default; halt on opt-in)
    — the tool does NOT pre-decide recoverability. See the module docstring."""

    violation: HarnessSandboxViolation

    def __post_init__(self) -> None:
        super().__init__(f"sandbox violation: {self.violation}")

    def to_tool_output(self) -> ToolOutput:
        # Carry the typed violation to the harness, which applies the configured
        # ``SandboxViolationPolicy`` (recoverable feedback by default; halt on
        # opt-in). See the module docstring (#150).
        return ToolOutputSandboxViolation(violation=self.violation)


@dataclass
class Timeout(ToolExecutionError):
    """Tool timed out. Recoverable."""

    after_seconds: float

    def __post_init__(self) -> None:
        super().__init__(f"timed out after {self.after_seconds}s")

    def to_tool_output(self) -> ToolOutput:
        return ToolOutputError(
            message=f"timed out after {int(self.after_seconds)}s",
            recoverable=True,
        )


__all__ = [
    "ExecutionFailed",
    "InvalidParameters",
    "SandboxViolationError",
    "Timeout",
    "ToolExecutionError",
]
