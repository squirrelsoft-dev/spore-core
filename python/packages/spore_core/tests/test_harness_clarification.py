"""Tests for the AwaitingClarification pause/resume path and the send_message
StreamUserMessage event (issue #81, Q4b + SendMessage routing).

Mirrors the Rust harness unit tests in ``harness.rs``.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    FinalResponse,
    HarnessConfig,
    HarnessRunOptions,
    HumanRequestClarification,
    HumanResponseAnswer,
    LoopStrategyReAct,
    MockAgent,
    RunResultSuccess,
    RunResultWaitingForHuman,
    ScriptedToolRegistry,
    SEND_MESSAGE_TOOL_NAME,
    SessionId,
    SessionState,
    StandardHarness,
    StreamUserMessage,
    Task,
    ToolCall,
    ToolCallRequested,
    ToolOutputAwaitingClarification,
    ToolOutputSuccess,
    TokenUsage,
)
from spore_core.agent import Context
from spore_core.harness import HarnessToolResult
from spore_core.model import Message, ModelParams


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=1, output_tokens=1)


def _tc(call_id: str, name: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, input={})


def _react_task(session: str = "s1") -> Task:
    return Task.new("clarify", SessionId(session), LoopStrategyReAct(max_iterations=5))


class _RecordingCM:
    """Records appended tool results / messages structurally."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.tool_results: list[HarnessToolResult] = []

    async def assemble(self, session: SessionState, task: Task) -> Context:
        _ = task
        return Context(messages=list(self.messages), tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: HarnessToolResult) -> None:
        _ = session
        self.tool_results.append(result)

    async def append_assistant_message(self, session: SessionState, message: Message) -> None:
        _ = session
        self.messages.append(message)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        _ = session

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


def _config(agent: MockAgent, reg: ScriptedToolRegistry, cm: object | None = None) -> HarnessConfig:
    return HarnessConfig(
        agent=agent,
        tool_registry=reg,
        sandbox=AllowAllSandbox(),
        context_manager=cm or _RecordingCM(),
        termination_policy=AlwaysContinuePolicy(),
    )


# ---------------------------------------------------------------------------
# AwaitingClarification pauses into a PausedState (no ChildPausedState).
# ---------------------------------------------------------------------------


async def test_awaiting_clarification_pauses_with_clarification_request() -> None:
    agent = MockAgent(AgentId("clar"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "ask_user_question")], usage=_usage()))
    reg = ScriptedToolRegistry().push(
        ToolOutputAwaitingClarification(question="which?", options=["a", "b"])
    )
    harness = StandardHarness(_config(agent, reg))

    r = await harness.run(HarnessRunOptions(_react_task("clar-session")))
    assert isinstance(r, RunResultWaitingForHuman)
    assert isinstance(r.request, HumanRequestClarification)
    assert r.request.question == "which?"
    assert r.request.options == ["a", "b"]
    # NO child_state — built directly as a PausedState.
    assert r.state.child_state is None
    assert isinstance(r.state.human_request, HumanRequestClarification)
    # The clarifying call is preserved as the HEAD of pending_tool_calls.
    assert [c.id for c in r.state.pending_tool_calls] == ["c1"]


# ---------------------------------------------------------------------------
# Resuming a clarification injects the answer as the clarifying call's result.
# ---------------------------------------------------------------------------


async def test_resume_clarification_injects_answer_as_tool_result() -> None:
    agent = MockAgent(AgentId("clar"))
    agent.push(ToolCallRequested(calls=[_tc("c1", "ask_user_question")], usage=_usage()))
    agent.push(FinalResponse(content="done", usage=_usage()))
    cm = _RecordingCM()
    reg = ScriptedToolRegistry().push(
        ToolOutputAwaitingClarification(question="which?", options=None)
    )
    harness = StandardHarness(_config(agent, reg, cm))

    paused = await harness.run(HarnessRunOptions(_react_task("resume-clar")))
    assert isinstance(paused, RunResultWaitingForHuman)

    resumed = await harness.resume(paused.state, HumanResponseAnswer(text="the answer"))
    assert isinstance(resumed, RunResultSuccess)
    assert resumed.output == "done"
    # The human's answer was injected as the clarifying call's tool result.
    injected = [tr for tr in cm.tool_results if tr.call_id == "c1"]
    assert len(injected) == 1
    out = injected[0].output
    assert isinstance(out, ToolOutputSuccess)
    assert out.content == "the answer"


# ---------------------------------------------------------------------------
# send_message emits a StreamUserMessage event and records a success result.
# ---------------------------------------------------------------------------


async def test_send_message_emits_user_message_event() -> None:
    agent = MockAgent(AgentId("msg"))
    agent.push(ToolCallRequested(calls=[_tc("c1", SEND_MESSAGE_TOOL_NAME)], usage=_usage()))
    agent.push(FinalResponse(content="done", usage=_usage()))
    cm = _RecordingCM()
    reg = ScriptedToolRegistry().push(ToolOutputSuccess(content="hello human"))
    harness = StandardHarness(_config(agent, reg, cm))

    events: list[object] = []
    r = await harness.run(HarnessRunOptions(_react_task("msg-session"), on_stream=events.append))
    assert isinstance(r, RunResultSuccess)
    # The loop emitted a StreamUserMessage with the content.
    user_msgs = [e for e in events if isinstance(e, StreamUserMessage)]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "hello human"
    # A (minimal) success tool result was still recorded so the loop continues.
    msg_results = [tr for tr in cm.tool_results if tr.call_id == "c1"]
    assert len(msg_results) == 1
    assert isinstance(msg_results[0].output, ToolOutputSuccess)
