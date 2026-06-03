"""Tests for :func:`spore_tools.define_tool` — the Python analogue of Rust's
``tool!`` macro.

Mirrors ``rust/crates/spore-core/src/macros.rs`` tests: the schema is *derived*
from the pydantic input model (never hand-written), the tool runs with a
validated model, optional annotations default to all-``False``, and invalid
arguments produce a **recoverable** ``invalid parameters`` error so tool-call
repair can retry.
"""

from __future__ import annotations

from pydantic import BaseModel

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import (
    AllowAllSandbox,
    ToolAnnotations,
    ToolContext,
    make_test_ctx,
)
from spore_tools import StandardTool, define_tool

_CTX = make_test_ctx()


class EchoInput(BaseModel):
    message: str
    shout: bool = False


def _echo_tool(**kwargs: object) -> StandardTool:
    async def run(input: EchoInput, sandbox: SandboxProvider, ctx: ToolContext) -> ToolOutput:
        content = input.message.upper() if input.shout else input.message
        return ToolOutputSuccess.success(content)

    return define_tool(
        name="echo",
        description="Echoes the input message",
        input_model=EchoInput,
        execute=run,
        **kwargs,  # type: ignore[arg-type]
    )


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name="echo", input=input_)


def test_returns_standard_tool() -> None:
    t = _echo_tool()
    assert isinstance(t, StandardTool)
    assert t.schema.name == "echo"
    assert t.implementation.name() == "echo"


def test_schema_is_derived_from_input_model() -> None:
    t = _echo_tool()
    params = t.schema.parameters
    assert params["type"] == "object"
    props = params["properties"]
    assert "message" in props
    assert "shout" in props


def test_annotations_default_to_all_false() -> None:
    t = _echo_tool()
    a = t.schema.annotations
    assert a == ToolAnnotations()
    assert a.read_only is False
    assert a.destructive is False
    assert a.idempotent is False
    assert a.open_world is False


def test_annotations_passed_through() -> None:
    t = _echo_tool(annotations=ToolAnnotations(read_only=True, idempotent=True))
    a = t.schema.annotations
    assert a.read_only is True
    assert a.idempotent is True
    assert a.destructive is False


async def test_runs_with_validated_input() -> None:
    t = _echo_tool()
    out = await t.implementation.execute(
        _call({"message": "hi", "shout": True}), AllowAllSandbox(), _CTX
    )
    assert isinstance(out, ToolOutputSuccess)
    assert out.content == "HI"


async def test_missing_field_is_recoverable_invalid_parameters_error() -> None:
    t = _echo_tool()
    out = await t.implementation.execute(_call({"shout": True}), AllowAllSandbox(), _CTX)
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True
    assert "invalid parameters for tool `echo`" in out.message
    assert "message" in out.message


async def test_wrong_type_is_recoverable_error() -> None:
    t = _echo_tool()
    out = await t.implementation.execute(
        _call({"message": 7, "shout": True}), AllowAllSandbox(), _CTX
    )
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True
    assert "invalid parameters" in out.message


def test_may_produce_large_output_flag() -> None:
    assert _echo_tool().implementation.may_produce_large_output() is False
    assert (
        _echo_tool(may_produce_large_output=True).implementation.may_produce_large_output() is True
    )


def test_is_not_subagent_tool() -> None:
    assert _echo_tool().implementation.is_subagent_tool() is False
