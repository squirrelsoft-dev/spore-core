"""Tests for the conversational preset + ``Task.simple`` (parity with Rust's
``HarnessBuilder::conversational`` / ``Task::simple``).

Mirrors ``rust/crates/spore-core/src/harness.rs``: a tool-less chat harness
built from a model should drive a single turn to a final response and succeed,
and ``Task.simple`` should default to a fresh session id + ReAct(8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spore_core import (
    CompleteOnFinalResponse,
    EmptyToolRegistry,
    HarnessBuilder,
    HarnessRunOptions,
    InMemoryStorageProvider,
    PlanExecuteConfig,
    ReactConfig,
    RegistryToolSchema,
    NoopContextManager,
    NullSandbox,
    ProviderInfo,
    RunResultSuccess,
    StorageProvider,
    Task,
    TerminationContinue,
    ToolCall,
    ToolOutputSuccess,
)
from spore_core.harness import BudgetSnapshot, SessionState
from spore_core.model import (
    MockModelInterface,
    ModelInterface,
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


class _ScriptedRecordingModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def call(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self._responses:
            raise AssertionError("scripted model ran out of responses")
        return ModelResponse(
            content=[TextBlock(text=self._responses.pop(0))],
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            stop_reason=StopReason.END_TURN,
        )

    async def call_streaming(self, request: ModelRequest):  # pragma: no cover - unused
        yield  # type: ignore[misc]

    async def count_tokens(self, request: ModelRequest) -> int:
        _ = request
        return 1

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="test", model_id="scripted", context_window=8192)


class _InspectTool:
    def name(self) -> str:
        return "inspect"

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(self, call: Any, sandbox: Any, ctx: Any) -> Any:
        _ = (call, sandbox, ctx)
        return ToolOutputSuccess.success("inspected")


@dataclass
class _ToolBundle:
    implementation: Any
    schema: Any


def _inspect_bundle() -> _ToolBundle:
    return _ToolBundle(
        implementation=_InspectTool(),
        schema=RegistryToolSchema(
            name="inspect",
            description="inspect project state",
            parameters={"type": "object", "properties": {}},
        ),
    )


async def test_conversational_structural_nudge_activates_prompt_tool_mode() -> None:
    model = _ScriptedRecordingModel(
        [
            '{"tasks":["inspect repo"],"rationale":"r"}',
            "guessed without tools",
            "<tool_call><name>inspect</name><input>{}</input></tool_call>",
            "grounded",
        ]
    )
    harness = (
        HarnessBuilder.conversational(model)
        .tool(_inspect_bundle())
        .tool_call_nudge(True)
        .storage(StorageProvider.single(InMemoryStorageProvider()))
        .build()
    )

    result = await harness.run(
        HarnessRunOptions(Task.new("inspect", "conv-nudge", PlanExecuteConfig.simple()))
    )

    assert isinstance(result, RunResultSuccess)
    assert result.output == "grounded"
    assert len(model.requests) == 4
    # Request 2 is the first execute answer and is still native mode. The
    # structural nudge flips the shared adaptive flag, so request 3 has the
    # prompt-tool instructions injected before the inner model is called.
    assert "<available_tools>" not in _request_text(model.requests[1])
    assert "<available_tools>" in _request_text(model.requests[2])


def _request_text(request: ModelRequest) -> str:
    parts: list[str] = []
    for message in request.messages:
        text = getattr(message.content, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


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
    # #124: the legacy ``config.agent`` field is gone — the worker resolves from
    # the registry under the DEFAULT empty key (the builder folds it there).
    assert config.registry.resolve_agent("").id() == "agent"
    assert isinstance(config.tool_registry, EmptyToolRegistry)
    assert isinstance(config.termination_policy, CompleteOnFinalResponse)
    assert isinstance(config.sandbox, NullSandbox)
    # Empty registry: advertises nothing and errors on dispatch.
    assert config.tool_registry.schemas() == []
    assert config.tool_registry.is_always_halt("anything") is False


def test_context_manager_setter_overrides_the_configured_manager() -> None:
    """``.context_manager(...)`` replaces the default StandardContextManager so
    callers can install a compaction config with a lower threshold. Mirrors
    Rust's ``context_manager_setter_overrides_the_configured_manager``."""
    model = _GreetingModel("hi")
    cm = NoopContextManager()
    config = HarnessBuilder.conversational(model).context_manager(cm).build_config()
    assert config.context_manager is cm


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
    assert isinstance(a.loop_strategy, ReactConfig)
    assert a.loop_strategy.max_iterations() == 8
    # Fresh, distinct session ids per call.
    assert a.session_id
    assert a.session_id != b.session_id
    # Fresh, distinct task ids too.
    assert a.id != b.id


# ---------------------------------------------------------------------------
# SC-2: object-safe ModelInterface — construct from an interface-typed model
# ---------------------------------------------------------------------------


async def test_harness_constructs_from_interface_typed_model() -> None:
    """SC-2: a conversational harness can be built from a model held under the
    interface type (``ModelInterface``), not a concrete generic — the equivalent
    of Rust's ``Arc<dyn ModelInterface>`` injection. The queued response comes
    back through the harness. Mirrors Rust's
    ``harness_constructs_from_boxed_model_interface``.
    """
    # The model is held as the structural interface type, not a concrete class.
    model: ModelInterface = MockModelInterface(
        ProviderInfo(name="test", model_id="boxed", context_window=8192)
    )
    model.push_response(
        ModelResponse(
            content=[TextBlock(text="boxed reply")],
            usage=TokenUsage(input_tokens=3, output_tokens=2),
            stop_reason=StopReason.END_TURN,
        )
    )
    harness = HarnessBuilder.conversational(model).build()
    result = await harness.run(HarnessRunOptions(Task.simple("hi")))
    assert isinstance(result, RunResultSuccess)
    assert result.output == "boxed reply"


def test_conversational_shares_one_model_instance() -> None:
    """SC-2 acceptance: the agent wrapper and the context manager reference the
    SAME model instance — the model is held once and shared, not re-minted per
    component."""
    model = _GreetingModel("hi")
    config = HarnessBuilder.conversational(model).build_config()
    agent = config.registry.resolve_agent("")
    # The agent wraps the model in an AdaptiveToolCallModelInterface holding the
    # original instance; the context manager keeps the RAW model. Both are the
    # one shared instance.
    assert agent._model._inner is model
    assert config.context_manager._inner._model is model
