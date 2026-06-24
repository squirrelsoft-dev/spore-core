"""SC-30: ``ReadOnlyToolView`` — the eval-phase read-only view filters the
wrapped catalogue to the read-only allow-list.

Mirrors the Rust unit test ``read_only_tool_view_filters_to_readonly_allowlist``:
an inner registry advertising read + write + exec tools is wrapped so that only
the INTERSECTION with ``READONLY_EVAL_TOOL_NAMES`` is advertised and dispatchable.
A non-allow-listed dispatch returns a recoverable error and never reaches inner.
"""

from __future__ import annotations

import pytest

from spore_core.harness import ToolOutput, ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.model import ToolSchema as ModelToolSchema
from spore_core.tool_registry import READONLY_EVAL_TOOL_NAMES, ReadOnlyToolView


class _Inner:
    """Inner harness-loop registry advertising read + write + exec tools, recording
    every dispatch so we can assert blocked calls never reach it."""

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def dispatch(self, call: ToolCall) -> ToolOutput:
        self.dispatched.append(call.name)
        return ToolOutputSuccess.success("ok")

    def is_always_halt(self, tool_name: str) -> bool:
        _ = tool_name
        return False

    def schemas(self) -> list[ModelToolSchema]:
        return [
            ModelToolSchema(name=n, description="", input_schema={"type": "object"})
            for n in ("read_file", "write_file", "bash")
        ]


@pytest.mark.asyncio
async def test_read_only_tool_view_filters_to_readonly_allowlist() -> None:
    inner = _Inner()
    view = ReadOnlyToolView(inner, set(READONLY_EVAL_TOOL_NAMES))

    # Schemas: only read_file survives the intersection; write_file/bash gone.
    names = {s.name for s in view.schemas()}
    assert "read_file" in names, "read_file advertised"
    assert "write_file" not in names, "write_file hidden"
    assert "bash" not in names, "bash hidden"

    # read_file dispatches through to the inner registry.
    ok = await view.dispatch(ToolCall(id="1", name="read_file", input={}))
    assert isinstance(ok, ToolOutputSuccess)

    # write_file is blocked with a recoverable error and never reaches inner.
    blocked = await view.dispatch(ToolCall(id="2", name="write_file", input={}))
    assert isinstance(blocked, ToolOutputError)
    assert blocked.recoverable is True
    assert inner.dispatched == ["read_file"]


def test_is_always_halt_delegates_to_inner() -> None:
    inner = _Inner()
    view = ReadOnlyToolView(inner, set(READONLY_EVAL_TOOL_NAMES))
    # Inner never halts; the view forwards the question unchanged.
    assert view.is_always_halt("read_file") is False
