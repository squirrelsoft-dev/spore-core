"""Ergonomic tool definition — :func:`define_tool`.

Mirrors Rust's ``tool!`` macro (``rust/crates/spore-core/src/macros.rs``):
:func:`define_tool` collapses the boilerplate of a hand-written
:class:`~spore_core.tool_registry.Tool` implementation — name, schema, input
validation, and an async ``execute`` body — into a single call that returns a
ready-to-register :class:`~spore_tools.StandardTool`.

The headline property is the same as the Rust macro: **the input model is the
single source of truth.** The advertised JSON Schema the model sees is *derived*
from the pydantic input model via ``model.model_json_schema()`` — never
hand-written — so the schema and the validation can never drift apart.

If the model's tool arguments fail to validate against the input model, the
generated tool returns a **recoverable** :class:`~spore_core.harness.ToolOutputError`
(message ``invalid parameters for tool `{name}`: {e}``) so a configured tool-call
repair pass gets a chance to coerce the arguments and re-dispatch, rather than
halting the run.

Example::

    from pydantic import BaseModel
    from spore_core.harness import ToolOutputSuccess
    from spore_tools import define_tool

    class CalculatorInput(BaseModel):
        expression: str

    async def run(input: CalculatorInput, sandbox, ctx):
        return ToolOutputSuccess.success(f"evaluated: {input.expression}")

    calculator = define_tool(
        name="calculator",
        description="Evaluates a mathematical expression and returns the result",
        input_model=CalculatorInput,
        execute=run,
    )
    # calculator.schema.name == "calculator"
    # calculator.schema.parameters is derived from CalculatorInput

The returned :class:`~spore_tools.StandardTool` plugs straight into a builder via
:meth:`~spore_core.HarnessBuilder.tool`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from spore_core.harness import (
    SandboxProvider,
    ToolOutput,
    ToolOutputError,
)
from spore_core.model import ToolCall
from spore_core.tool_registry import ToolAnnotations, ToolContext, ToolSchema

from .catalogue import StandardTool

InputT = TypeVar("InputT", bound=BaseModel)

#: An async ``execute`` body: it receives the *validated* input model plus the
#: two seams (the environment ``sandbox`` and the storage ``ctx``) and returns a
#: :class:`~spore_core.harness.ToolOutput`. This mirrors the Rust closure
#: signature ``|input, sandbox, ctx| async move { .. }``.
ExecuteFn = Callable[[InputT, SandboxProvider, ToolContext], Awaitable[ToolOutput]]


def _schema_for_model(model: type[BaseModel]) -> dict[str, Any]:
    """Derive a tool's ``parameters`` JSON Schema from its pydantic input model.

    Used by :func:`define_tool` so the advertised schema is generated from the
    exact model the tool validates. Falls back to a permissive
    ``{"type": "object"}`` if the model ever fails to produce a schema (it does
    not for any well-formed :class:`pydantic.BaseModel`)."""
    try:
        schema = model.model_json_schema()
    except Exception:  # noqa: BLE001 — never let schema derivation crash registration
        return {"type": "object"}
    # pydantic always emits ``type: object`` for a model, but guard so the
    # registry's structural check (a top-level ``type``) always passes.
    if "type" not in schema:
        schema["type"] = "object"
    return schema


class _DefinedTool:
    """Anonymous :class:`~spore_core.tool_registry.Tool` implementation backing a
    :func:`define_tool` call. Validates ``call.input`` into the input model and,
    on failure, returns a recoverable ``invalid parameters`` error; otherwise it
    forwards the validated model to the user's ``execute`` body."""

    def __init__(
        self,
        name: str,
        input_model: type[BaseModel],
        execute: ExecuteFn[Any],
        *,
        may_produce_large_output: bool,
    ) -> None:
        self._name = name
        self._input_model = input_model
        self._execute = execute
        self._may_produce_large_output = may_produce_large_output

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return self._may_produce_large_output

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        try:
            parsed = self._input_model.model_validate(call.input)
        except ValidationError as e:
            # Recoverable so a tool-call-repair pass can coerce the arguments and
            # re-dispatch — mirrors the Rust macro's deserialize-failure branch.
            return ToolOutputError(
                message=f"invalid parameters for tool `{self._name}`: {e}",
                recoverable=True,
            )
        return await self._execute(parsed, sandbox, ctx)


def define_tool(
    *,
    name: str,
    description: str,
    input_model: type[InputT],
    execute: ExecuteFn[InputT],
    annotations: ToolAnnotations | None = None,
    may_produce_large_output: bool = False,
) -> StandardTool:
    """Define a :class:`~spore_core.tool_registry.Tool` and bundle it with its
    *derived* schema into a :class:`~spore_tools.StandardTool` in one call.

    The Python analogue of Rust's ``tool!`` macro. The advertised JSON Schema is
    derived from ``input_model`` via ``model_json_schema()`` so it can never
    drift from the model the tool actually validates.

    Args:
        name: The tool's registered name (must be non-empty).
        description: Human-readable description shown to the model.
        input_model: A :class:`pydantic.BaseModel` subclass. The tool validates
            the model's tool arguments into this and derives the advertised JSON
            Schema from it.
        execute: An async callable ``(input, sandbox, ctx) -> ToolOutput``.
            ``input`` is the validated ``input_model`` instance; ``sandbox`` is
            the :class:`~spore_core.harness.SandboxProvider` (environment seam);
            ``ctx`` is the :class:`~spore_core.tool_registry.ToolContext`
            (storage seam).
        annotations: Optional :class:`~spore_core.tool_registry.ToolAnnotations`.
            Defaults to all-``False`` (a read-write, non-idempotent tool) when
            omitted — matching the Rust macro's ``ToolAnnotations::default``.
        may_produce_large_output: Whether the tool may emit large output (drives
            large-output routing). Defaults to ``False``.

    Returns:
        A :class:`~spore_tools.StandardTool` ready for
        :meth:`~spore_core.HarnessBuilder.tool`.

    If the model's arguments fail to validate against ``input_model``, the
    generated tool returns a **recoverable**
    :class:`~spore_core.harness.ToolOutputError` whose message contains
    ``invalid parameters`` — so tool-call repair gets a chance to retry.
    """
    impl = _DefinedTool(
        name,
        input_model,
        execute,
        may_produce_large_output=may_produce_large_output,
    )
    schema = ToolSchema(
        name=name,
        description=description,
        parameters=_schema_for_model(input_model),
        annotations=annotations if annotations is not None else ToolAnnotations(),
    )
    return StandardTool(implementation=impl, schema=schema)


__all__ = ["ExecuteFn", "define_tool"]
