"""Tests for the conversational preset + ``Task.simple`` (parity with Rust's
``HarnessBuilder::conversational`` / ``Task::simple``).

Mirrors ``rust/crates/spore-core/src/harness.rs``: a tool-less chat harness
built from a model should drive a single turn to a final response and succeed,
and ``Task.simple`` should default to a fresh session id + ReAct(8).
"""

from __future__ import annotations

from spore_core import (
    CompleteOnFinalResponse,
    EmptyToolRegistry,
    HarnessBuilder,
    HarnessRunOptions,
    LoopStrategyReAct,
    NullSandbox,
    ProviderInfo,
    RunResultSuccess,
    Task,
    TerminationContinue,
    ToolCall,
)
from spore_core.harness import BudgetSnapshot, SessionState
from spore_core.model import (
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    TokenUsage,
)


class _GreetingModel:
    """Minimal :class:`ModelInterface` that always returns one final text
    response. Enough to drive the conversational loop to success without a live
    provider."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    async def call(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content=[TextBlock(text=self._reply)],
            usage=TokenUsage(input_tokens=3, output_tokens=2),
            stop_reason=StopReason.END_TURN,
        )

    async def call_streaming(self, request: ModelRequest):  # pragma: no cover - unused
        yield  # type: ignore[misc]

    async def count_tokens(self, request: ModelRequest) -> int:
        return 1

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="test", model_id="greeting", context_window=8192)


async def test_conversational_single_turn_success() -> None:
    """A conversational harness over a scripted model drives one turn to a
    final response and returns ``RunResultSuccess`` with that text as output."""
    model = _GreetingModel("Hello there, friend!")
    harness = HarnessBuilder.conversational(model).build()
    result = await harness.run(
        HarnessRunOptions(Task.simple("Reply with a friendly one-line greeting."))
    )
    assert isinstance(result, RunResultSuccess)
    assert result.output == "Hello there, friend!"
    assert result.turns == 1
    assert model.calls == 1


def test_conversational_defaults_components() -> None:
    """``conversational`` wires the five Rust defaults: a ModelAgent named
    ``agent``, an empty tool registry, a ``NullSandbox`` (permissive validate,
    mirroring Rust), a standard context manager, and ``CompleteOnFinalResponse``
    termination."""
    model = _GreetingModel("hi")
    config = HarnessBuilder.conversational(model).build_config()
    assert config.agent.id() == "agent"
    assert isinstance(config.tool_registry, EmptyToolRegistry)
    assert isinstance(config.termination_policy, CompleteOnFinalResponse)
    assert isinstance(config.sandbox, NullSandbox)
    # Empty registry: advertises nothing and errors on dispatch.
    assert config.tool_registry.schemas() == []
    assert config.tool_registry.is_always_halt("anything") is False


async def test_complete_on_final_response_always_continues() -> None:
    """``CompleteOnFinalResponse`` always returns Continue (the loop accepts the
    first final response)."""
    decision = await CompleteOnFinalResponse().evaluate(SessionState(), BudgetSnapshot())
    assert isinstance(decision, TerminationContinue)


async def test_empty_tool_registry_dispatch_errors() -> None:
    """The empty registry returns a recoverable error for any tool call."""
    out = await EmptyToolRegistry().dispatch(ToolCall(id="1", name="nope", input={}))
    assert out.kind == "error"
    assert out.recoverable is True
    assert "nope" in out.message


def test_task_simple_defaults() -> None:
    """``Task.simple`` mints a fresh session id and a ReAct(8) loop."""
    a = Task.simple("do a thing")
    b = Task.simple("do a thing")
    assert a.instruction == "do a thing"
    assert isinstance(a.loop_strategy, LoopStrategyReAct)
    assert a.loop_strategy.max_iterations == 8
    # Fresh, distinct session ids per call.
    assert a.session_id
    assert a.session_id != b.session_id
    # Fresh, distinct task ids too.
    assert a.id != b.id
