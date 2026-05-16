"""``ToolExecutionError`` — typed error hierarchy for tool implementations.

Mirrors ``rust/crates/spore-core/src/tools/error.rs``. Spec (issue #5):

* :class:`InvalidParameters` — ``recoverable=True``
* :class:`Timeout` — ``recoverable=True``
* :class:`ExecutionFailed` — caller-specified ``recoverable``
* :class:`SandboxViolation` — ``recoverable=False``

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
    """Sandbox rejected the operation. Not recoverable."""

    violation: HarnessSandboxViolation

    def __post_init__(self) -> None:
        super().__init__(f"sandbox violation: {self.violation}")

    def to_tool_output(self) -> ToolOutput:
        return ToolOutputError(
            message=f"sandbox violation: {self.violation.kind}",
            recoverable=False,
        )


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
