"""Tests for Agent (issue #2).

Mirrors ``rust/crates/spore-core/src/agent.rs`` unit tests and the
fixture-replay integration test
``rust/crates/spore-core/tests/agent_fixture_replay.rs``. Both consume the
same shared JSONL fixture under
``fixtures/model_responses/agent/turn_classification.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from spore_core import (
    Agent,
    AgentErrorEmpty,
    AgentErrorMalformed,
    AgentErrorModel,
    AgentId,
    Context,
    FinalResponse,
    Message,
    MockModelInterface,
    ModelAgent,
    ModelResponse,
    ProviderInfo,
    ReplayModelInterface,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    ThinkingBlock,
    TimeoutError,
    TokenUsage,
    ToolCall,
    ToolCallRequested,
    ToolUseBlock,
    TurnError,
    TurnResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider() -> ProviderInfo:
    return ProviderInfo(name="test", model_id="test-1", context_window=1000)


def _ctx_user(text: str) -> Context:
    return Context(
        messages=[Message(role=Role.USER, content=TextContent(text=text))],
    )


def _usage(in_t: int, out_t: int) -> TokenUsage:
    return TokenUsage(input_tokens=in_t, output_tokens=out_t)


def _text_resp(text: str) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=text)],
        usage=_usage(3, 4),
        stop_reason=StopReason.END_TURN,
    )


def _tool_resp(calls: list[ToolCall]) -> ModelResponse:
    return ModelResponse(
        content=[ToolUseBlock(id=c.id, name=c.name, input=c.input) for c in calls],
        usage=_usage(5, 6),
        stop_reason=StopReason.TOOL_USE,
    )


def _make_agent(model: MockModelInterface) -> ModelAgent:
    return ModelAgent(AgentId("coding-agent"), model)


# ---------------------------------------------------------------------------
# Protocol / identity
# ---------------------------------------------------------------------------


def test_model_agent_satisfies_protocol() -> None:
    m = MockModelInterface(_provider())
    agent = _make_agent(m)
    assert isinstance(agent, Agent)


def test_agent_id_reported() -> None:
    m = MockModelInterface(_provider())
    agent = ModelAgent(AgentId("initializer"), m)
    assert agent.id() == AgentId("initializer")
    assert str(agent.id()) == "initializer"


# ---------------------------------------------------------------------------
# Classification rules
# ---------------------------------------------------------------------------


async def test_turn_makes_exactly_one_model_call() -> None:
    m = MockModelInterface(_provider())
    m.push_response(_text_resp("ok"))
    agent = _make_agent(m)
    await agent.turn(_ctx_user("hi"))
    assert m.call_count == 1


async def test_final_response_on_end_turn_with_text() -> None:
    m = MockModelInterface(_provider())
    m.push_response(_text_resp("hello world"))
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("hi"))
    assert isinstance(result, FinalResponse)
    assert result.content == "hello world"
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 4


async def test_tool_call_requested_on_tool_use_stop() -> None:
    m = MockModelInterface(_provider())
    m.push_response(_tool_resp([ToolCall(id="call_1", name="read_file", input={"path": "/x"})]))
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("read /x"))
    assert isinstance(result, ToolCallRequested)
    assert len(result.calls) == 1
    assert result.calls[0].id == "call_1"
    assert result.calls[0].name == "read_file"
    assert result.usage.input_tokens == 5


async def test_tool_call_requested_carries_multiple_calls() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        _tool_resp(
            [
                ToolCall(id="a", name="read_file", input={"path": "/a"}),
                ToolCall(id="b", name="read_file", input={"path": "/b"}),
            ]
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("read both"))
    assert isinstance(result, ToolCallRequested)
    assert len(result.calls) == 2
    assert result.calls[0].id == "a"
    assert result.calls[1].id == "b"


async def test_empty_response_when_no_content_blocks() -> None:
    m = MockModelInterface(_provider())
    m.push_response(ModelResponse(content=[], usage=_usage(1, 0), stop_reason=StopReason.END_TURN))
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, TurnError)
    assert isinstance(result.error, AgentErrorEmpty)
    assert result.usage is not None
    assert result.usage.input_tokens == 1


async def test_thinking_blocks_do_not_satisfy_final_response() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        ModelResponse(
            content=[ThinkingBlock(text="musing")],
            usage=_usage(1, 2),
            stop_reason=StopReason.END_TURN,
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, TurnError)
    assert isinstance(result.error, AgentErrorEmpty)


async def test_model_error_surfaces_wrapped() -> None:
    m = MockModelInterface(_provider())
    m.push_response(TimeoutError())
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("hi"))
    assert isinstance(result, TurnError)
    assert isinstance(result.error, AgentErrorModel)
    assert result.error.error.kind == "Timeout"
    assert result.usage is None


async def test_malformed_when_tool_use_stop_but_no_tool_blocks() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        ModelResponse(
            content=[TextBlock(text="hmm")],
            usage=_usage(2, 2),
            stop_reason=StopReason.TOOL_USE,
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, TurnError)
    assert isinstance(result.error, AgentErrorMalformed)
    assert result.usage is not None


async def test_tool_calls_dispatched_even_when_stop_is_end_turn() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        ModelResponse(
            content=[ToolUseBlock(id="x", name="noop", input={})],
            usage=_usage(1, 1),
            stop_reason=StopReason.END_TURN,
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, ToolCallRequested)


async def test_max_tokens_stop_classifies_as_final_response() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        ModelResponse(
            content=[TextBlock(text="truncated")],
            usage=_usage(2, 5),
            stop_reason=StopReason.MAX_TOKENS,
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, FinalResponse)


async def test_stop_sequence_classifies_as_final_response() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        ModelResponse(
            content=[TextBlock(text="done.")],
            usage=_usage(2, 1),
            stop_reason=StopReason.STOP_SEQUENCE,
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, FinalResponse)


async def test_multiple_text_blocks_are_concatenated() -> None:
    m = MockModelInterface(_provider())
    m.push_response(
        ModelResponse(
            content=[TextBlock(text="foo"), TextBlock(text="bar")],
            usage=_usage(1, 1),
            stop_reason=StopReason.END_TURN,
        )
    )
    agent = _make_agent(m)
    result = await agent.turn(_ctx_user("?"))
    assert isinstance(result, FinalResponse)
    assert result.content == "foobar"


# ---------------------------------------------------------------------------
# Wire-format round-trip (cross-language)
# ---------------------------------------------------------------------------


_TurnResultAdapter = TypeAdapter(TurnResult)


def test_turn_result_tool_call_roundtrips_json() -> None:
    r = ToolCallRequested(
        calls=[ToolCall(id="1", name="x", input={"a": 1})],
        usage=_usage(2, 3),
    )
    blob = _TurnResultAdapter.dump_json(r)
    back = _TurnResultAdapter.validate_json(blob)
    assert back == r


def test_turn_result_final_response_roundtrips_json() -> None:
    r = FinalResponse(content="hi", usage=_usage(1, 1))
    blob = _TurnResultAdapter.dump_json(r)
    back = _TurnResultAdapter.validate_json(blob)
    assert back == r


def test_turn_result_error_empty_roundtrips_json() -> None:
    r = TurnError(error=AgentErrorEmpty(), usage=None)
    blob = _TurnResultAdapter.dump_json(r)
    assert b'"kind":"error"' in blob
    assert b'"kind":"empty_response"' in blob
    back = _TurnResultAdapter.validate_json(blob)
    assert back == r


def test_turn_result_error_malformed_roundtrips_json() -> None:
    r = TurnError(
        error=AgentErrorMalformed(tool_name="x", reason="y"),
        usage=_usage(1, 1),
    )
    blob = _TurnResultAdapter.dump_json(r)
    back = _TurnResultAdapter.validate_json(blob)
    assert back == r


def test_agent_error_variants_displayable() -> None:
    e1 = AgentErrorEmpty()
    e2 = AgentErrorMalformed(tool_name="x", reason="y")
    e3 = AgentErrorModel(error=__import__("spore_core").ModelErrorPayload(kind="Timeout"))
    assert e1.kind == "empty_response"
    assert e2.kind == "malformed_tool_call"
    assert e3.kind == "model_error"


# ---------------------------------------------------------------------------
# Fixture replay — cross-language consistency
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "fixtures").is_dir() and (parent / "rust").is_dir():
            return parent
    raise RuntimeError("could not locate spore-core repo root from test file")


def _fixture_path() -> Path:
    return _repo_root() / "fixtures/model_responses/agent/turn_classification.jsonl"


async def test_agent_classifies_recorded_turns_consistently() -> None:
    jsonl = _fixture_path().read_text()
    replay = ReplayModelInterface.from_jsonl(
        jsonl,
        ProviderInfo(name="anthropic", model_id="fixture", context_window=200_000),
    )
    agent = ModelAgent(AgentId("fixture-agent"), replay)

    # 1. Plain text → FinalResponse("hello")
    r1 = await agent.turn(Context())
    assert isinstance(r1, FinalResponse)
    assert r1.content == "hello"
    assert r1.usage.input_tokens == 5
    assert r1.usage.output_tokens == 1

    # 2. Single tool call → ToolCallRequested(1)
    r2 = await agent.turn(Context())
    assert isinstance(r2, ToolCallRequested)
    assert len(r2.calls) == 1
    assert r2.calls[0].name == "read_file"
    assert r2.calls[0].id == "toolu_a"
    assert r2.usage.input_tokens == 20

    # 3. Parallel tool calls → ToolCallRequested(2)
    r3 = await agent.turn(Context())
    assert isinstance(r3, ToolCallRequested)
    assert len(r3.calls) == 2
    assert r3.calls[0].id == "toolu_b1"
    assert r3.calls[1].id == "toolu_b2"

    # 4. Empty content + end_turn → EmptyResponse
    r4 = await agent.turn(Context())
    assert isinstance(r4, TurnError)
    assert isinstance(r4.error, AgentErrorEmpty)
    assert r4.usage is not None
    assert r4.usage.input_tokens == 3
    assert r4.usage.output_tokens == 0
