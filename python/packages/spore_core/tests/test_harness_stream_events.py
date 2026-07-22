"""Parity tests for harness stream-event wire format (parity plan §5.1/§5.7/§5.9).

Covers:
- ``NodeAttr`` attribution on react-leaf events (``kind``/``agent_id``/``path``/``depth``).
- ``StreamTurnEnd.usage`` carries the turn's ``TokenUsage``.
- ``StreamToolResult.revised=True`` re-emitted after an ``AfterTool``
  ``ContinueWithModification`` rewrite (only for results that actually changed).
- ``node`` is OMITTED from serialized output when ``None`` (byte-identical
  back-compat, mirroring Rust's ``skip_serializing_if = "Option::is_none"``).
- ``StreamUserMessage.steered`` defaults to ``False``.
- ``StreamToolProgress`` event shape.
"""

from __future__ import annotations

from typing import Any

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessConfig,
    HarnessRunOptions,
    HarnessStreamEvent,
    MiddlewareContinue,
    MiddlewareContinueWithModification,
    ModelAgent,
    ModelResponse,
    NoopContextManager,
    NodeAttr,
    ProviderInfo,
    ReactConfig,
    RecordedExchange,
    ReplayModelInterface,
    RunResultSuccess,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    StandardMiddlewareChain,
    StopReason,
    StreamToolProgress,
    StreamToolResult,
    StreamTurnEnd,
    StreamUserMessage,
    Task,
    TextBlock,
    TokenUsage,
    ToolOutputError,
    ToolOutputSuccess,
    ToolUseBlock,
)
from spore_core.middleware import HookContext, HookPoint, MiddlewareDecision


def _provider() -> ProviderInfo:
    return ProviderInfo(name="anthropic", model_id="replay", context_window=200_000)


def _usage(inp: int = 3, out: int = 4) -> TokenUsage:
    return TokenUsage(input_tokens=inp, output_tokens=out)


def _replay_agent(responses: list[ModelResponse]) -> ModelAgent:
    from spore_core import ModelRequest

    exchanges = [
        RecordedExchange(request=ModelRequest(stream=True), response=r, provider="anthropic")
        for r in responses
    ]
    replay = ReplayModelInterface(exchanges, _provider())
    return ModelAgent(AgentId("leaf-agent"), replay)


def _config(agent: ModelAgent, **overrides: Any) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=overrides.get("tool_registry", ScriptedToolRegistry()),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
        middleware=overrides.get("middleware"),
    )


def _react_task(max_iter: int = 5) -> Task:
    return Task.new("do something", SessionId("s1"), ReactConfig.per_loop(max_iter))


def _collect() -> tuple[list[HarnessStreamEvent], Any]:
    events: list[HarnessStreamEvent] = []
    return events, events.append


# ---------------------------------------------------------------------------
# NodeAttr attribution (react leaf)
# ---------------------------------------------------------------------------


async def test_react_leaf_events_carry_node_attribution() -> None:
    agent = _replay_agent(
        [
            ModelResponse(
                content=[TextBlock(text="hello")],
                usage=_usage(),
                stop_reason=StopReason.END_TURN,
            )
        ]
    )
    events, sink = _collect()
    h = StandardHarness(_config(agent))
    r = await h.run(HarnessRunOptions(_react_task(), on_stream=sink))
    assert isinstance(r, RunResultSuccess)

    attributed = [e for e in events if e.node is not None]
    assert attributed, "expected react-leaf events to carry node attribution"
    for e in attributed:
        assert e.node is not None
        assert e.node.kind == "react"
        # Empty agent handle ⇒ ``agent_id`` is ``None`` (mirrors Rust's
        # `(!self.agent.0.is_empty()).then(...)`).
        assert e.node.agent_id is None
        assert e.node.depth == 0
        assert e.node.path == ["react"]


# ---------------------------------------------------------------------------
# StreamTurnEnd.usage
# ---------------------------------------------------------------------------


async def test_turn_end_carries_per_turn_usage() -> None:
    agent = _replay_agent(
        [
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=_usage(inp=11, out=22),
                stop_reason=StopReason.END_TURN,
            )
        ]
    )
    events, sink = _collect()
    h = StandardHarness(_config(agent))
    await h.run(HarnessRunOptions(_react_task(), on_stream=sink))
    ends = [e for e in events if isinstance(e, StreamTurnEnd)]
    assert ends, "expected a StreamTurnEnd"
    assert ends[-1].usage.input_tokens == 11
    assert ends[-1].usage.output_tokens == 22


# ---------------------------------------------------------------------------
# StreamToolResult.revised after AfterTool rewrite
# ---------------------------------------------------------------------------


class _RewriteAfterTool:
    """A middleware that rewrites the first tool result to an error on AfterTool."""

    async def handle(self, ctx: HookContext) -> MiddlewareDecision:
        if ctx.kind == "after_tool" and ctx.results:
            ctx.results[0].output = ToolOutputError(
                message="inverted by middleware", recoverable=True
            )
            return MiddlewareContinueWithModification()
        return MiddlewareContinue()

    def hooks(self) -> list[HookPoint]:
        return [HookPoint.AFTER_TOOL]

    def name(self) -> str:
        return "rewrite-after-tool"


async def test_tool_result_revised_after_aftertool_rewrite() -> None:
    reg = ScriptedToolRegistry()
    reg.push(ToolOutputSuccess(content="original"))
    agent = _replay_agent(
        [
            ModelResponse(
                content=[ToolUseBlock(id="toolu_1", name="lookup", input={"q": "x"})],
                usage=_usage(),
                stop_reason=StopReason.TOOL_USE,
            ),
            ModelResponse(
                content=[TextBlock(text="done")],
                usage=_usage(),
                stop_reason=StopReason.END_TURN,
            ),
        ]
    )
    chain = StandardMiddlewareChain()
    await chain.register(_RewriteAfterTool())
    events, sink = _collect()
    h = StandardHarness(_config(agent, tool_registry=reg, middleware=chain))
    await h.run(HarnessRunOptions(_react_task(), on_stream=sink))

    results = [e for e in events if isinstance(e, StreamToolResult)]
    assert results, "expected StreamToolResult events"
    # The first emission is the original (revised=False); the rewrite re-emits
    # with revised=True and the inverted content.
    assert any(e.revised is False for e in results)
    revised = [e for e in results if e.revised is True]
    assert revised, "expected a revised StreamToolResult after the AfterTool rewrite"
    assert revised[-1].is_error is True
    assert revised[-1].content == "inverted by middleware"


# ---------------------------------------------------------------------------
# Serialization back-compat: node omitted when None
# ---------------------------------------------------------------------------


def test_node_omitted_from_serialization_when_none() -> None:
    ev = StreamToolResult(call_id="c1", is_error=False, content="ok")
    assert ev.node is None
    blob = ev.model_dump_json()
    assert '"node"' not in blob, blob
    # Round-trips back with node=None.
    back = StreamToolResult.model_validate_json(blob)
    assert back.node is None


def test_node_present_in_serialization_when_set() -> None:
    ev = StreamToolResult(
        call_id="c1",
        is_error=False,
        content="ok",
        node=NodeAttr(kind="react", agent_id="a1", depth=0, path=["react"]),
    )
    blob = ev.model_dump_json()
    assert '"node"' in blob
    assert '"agent_id":"a1"' in blob


# ---------------------------------------------------------------------------
# StreamUserMessage.steered default + StreamToolProgress shape
# ---------------------------------------------------------------------------


def test_stream_user_message_steered_defaults_false() -> None:
    ev = StreamUserMessage(content="hi")
    assert ev.steered is False


def test_stream_tool_progress_shape() -> None:
    ev = StreamToolProgress(call_id="c1", name="bash", preview="building...")
    assert ev.kind == "tool_progress"
    assert ev.preview == "building..."
    assert ev.node is None
