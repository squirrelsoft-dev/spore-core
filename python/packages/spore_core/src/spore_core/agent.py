"""Agent — executes a single turn against a :class:`ModelInterface`.

Mirrors the Rust reference at ``rust/crates/spore-core/src/agent.rs``. The
agent is **one turn**: it accepts a fully assembled :class:`Context`
(produced upstream by the ``ContextManager``, issue #7) and returns a
:class:`TurnResult` classifying what the model wants to do next.

What this component does:

* Translate :class:`Context` to :class:`ModelRequest`.
* Invoke ``ModelInterface.call``.
* Classify the response as ``ToolCallRequested``, ``FinalResponse``, or
  ``Error``.

What this component does NOT do:

* Assemble context (``ContextManager`` — issue #7).
* Execute or dispatch tool calls (the harness loop dispatches via the
  ``ToolRegistry`` — issues #3, #4).
* Validate tool call parameters against tool schemas (``ToolRegistry``).
* Decide termination (``TerminationPolicy`` — issue #13).
* Retry on transient errors (lives in the ``ModelInterface`` impl).

Classification rules (must match Rust/TS/Go byte-for-byte):

1. ``stop_reason == tool_use`` and tool-use blocks present →
   ``ToolCallRequested``.
2. ``stop_reason == tool_use`` and no tool-use blocks → ``Error`` with
   :class:`MalformedToolCall`.
3. ``stop_reason in {end_turn, max_tokens, stop_sequence}``:

   * Tool-use blocks present → still ``ToolCallRequested`` (do not drop).
   * No text and no tool calls → ``Error`` with :class:`EmptyResponse`.
   * Otherwise → ``FinalResponse`` with concatenated text (``Thinking``
     blocks discarded — observability, not output).

4. Model error → ``Error`` with :class:`ModelErrorAgent` and ``usage=None``.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .errors import SporeError
from .model import (
    Message,
    ModelError,
    ModelInterface,
    ModelParams,
    ModelRequest,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolCall,
    ToolSchema,
    ToolUseBlock,
)

# ============================================================================
# Identity
# ============================================================================


AgentId = NewType("AgentId", str)


# ============================================================================
# Context — the assembled input handed to the agent
# ============================================================================


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Context(_Model):
    """Fully assembled per-turn input produced by the ``ContextManager``.

    The agent treats this as an immutable snapshot — never mutates it.
    """

    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSchema] = Field(default_factory=list)
    params: ModelParams = Field(default_factory=ModelParams)

    def into_request(self) -> ModelRequest:
        return ModelRequest(
            messages=list(self.messages),
            tools=list(self.tools),
            params=self.params,
            stream=False,
        )


# ============================================================================
# Errors
# ============================================================================


class AgentErrorException(SporeError):
    """Base class for agent-side exception types.

    Mirrors the Rust ``AgentError`` enum. Note: the agent never *raises*
    these across its public boundary — they are reported inside
    :class:`TurnResult.Error` as values. The class hierarchy exists so
    callers can pattern-match by ``isinstance`` and so the wire format
    aligns with the Rust serde-tagged union.
    """

    kind: ClassVar[str] = "AgentError"


class EmptyResponseError(AgentErrorException):
    kind: ClassVar[str] = "empty_response"

    def __init__(self) -> None:
        super().__init__("model returned neither text nor tool calls")


class MalformedToolCallError(AgentErrorException):
    kind: ClassVar[str] = "malformed_tool_call"

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"malformed tool call from model (tool={tool_name}): {reason}")


# ----- Serialized AgentError tagged union (wire format) ---------------------


class ModelErrorPayload(_Model):
    """Serialised view of a wrapped :class:`ModelError`.

    Mirrors Rust's ``ModelError`` serde representation: tagged on ``kind``.
    Concrete provider details vary per variant; we keep the ``kind`` tag
    plus a free-form ``message`` so cross-language round-trip is lossless
    for the variants we currently exercise.
    """

    kind: str
    message: str | None = None


class AgentErrorModel(_Model):
    """``AgentError::ModelError`` wire variant."""

    kind: Literal["model_error"] = "model_error"
    error: ModelErrorPayload


class AgentErrorEmpty(_Model):
    """``AgentError::EmptyResponse`` wire variant."""

    kind: Literal["empty_response"] = "empty_response"


class AgentErrorMalformed(_Model):
    """``AgentError::MalformedToolCall`` wire variant."""

    kind: Literal["malformed_tool_call"] = "malformed_tool_call"
    tool_name: str
    reason: str


AgentError = Annotated[
    AgentErrorModel | AgentErrorEmpty | AgentErrorMalformed,
    Field(discriminator="kind"),
]


def _wrap_model_error(err: ModelError) -> AgentErrorModel:
    return AgentErrorModel(error=ModelErrorPayload(kind=err.kind, message=str(err)))


# ============================================================================
# TurnResult — tagged union mirroring Rust's serde tag = "kind"
# ============================================================================


class ToolCallRequested(_Model):
    kind: Literal["tool_call_requested"] = "tool_call_requested"
    calls: list[ToolCall]
    usage: TokenUsage


class FinalResponse(_Model):
    kind: Literal["final_response"] = "final_response"
    content: str
    usage: TokenUsage


class TurnError(_Model):
    """``TurnResult::Error`` — the model could not be classified."""

    kind: Literal["error"] = "error"
    error: AgentError
    usage: TokenUsage | None = None


TurnResult = Annotated[
    ToolCallRequested | FinalResponse | TurnError,
    Field(discriminator="kind"),
]


# ============================================================================
# Agent protocol
# ============================================================================


@runtime_checkable
class Agent(Protocol):
    """Executes a single turn given a fully assembled :class:`Context`."""

    async def turn(self, context: Context) -> TurnResult: ...

    def id(self) -> AgentId: ...


# ============================================================================
# ModelAgent — standard implementation
# ============================================================================


class ModelAgent:
    """Standard :class:`Agent`: forward :class:`Context` to a
    :class:`ModelInterface` and classify the response per the module rules.
    """

    def __init__(self, agent_id: AgentId, model: ModelInterface) -> None:
        self._id = agent_id
        self._model = model

    def id(self) -> AgentId:
        return self._id

    async def turn(self, context: Context) -> TurnResult:
        request = context.into_request()
        try:
            response = await self._model.call(request)
        except ModelError as e:
            return TurnError(error=_wrap_model_error(e), usage=None)

        usage = response.usage

        # Extract any tool-use blocks regardless of stop_reason; the model
        # may, in principle, request tool use without setting stop_reason.
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in response.content:
            if isinstance(block, ToolUseBlock):
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
            elif isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                # observability only — discard
                pass

        stop = response.stop_reason.value

        if stop == "tool_use":
            if not tool_calls:
                return TurnError(
                    error=AgentErrorMalformed(
                        tool_name="",
                        reason="stop_reason=ToolUse but no ToolUse blocks present",
                    ),
                    usage=usage,
                )
            return ToolCallRequested(calls=tool_calls, usage=usage)

        # end_turn | max_tokens | stop_sequence
        if not text_parts and not tool_calls:
            return TurnError(error=AgentErrorEmpty(), usage=usage)
        if tool_calls:
            return ToolCallRequested(calls=tool_calls, usage=usage)
        return FinalResponse(content="".join(text_parts), usage=usage)


# ============================================================================
# Mock implementation (test utility)
# ============================================================================


class MockAgent:
    """Programmable mock for unit tests. Each entry pushed via
    :meth:`push` is yielded on successive calls to :meth:`turn`.
    """

    def __init__(self, agent_id: AgentId) -> None:
        self._id = agent_id
        self._results: list[TurnResult] = []
        self.call_count: int = 0

    def push(self, result: TurnResult) -> MockAgent:
        self._results.append(result)
        return self

    async def turn(self, context: Context) -> TurnResult:
        self.call_count += 1
        if not self._results:
            return TurnError(error=AgentErrorEmpty(), usage=None)
        return self._results.pop(0)

    def id(self) -> AgentId:
        return self._id


__all__ = [
    "Agent",
    "AgentError",
    "AgentErrorEmpty",
    "AgentErrorException",
    "AgentErrorMalformed",
    "AgentErrorModel",
    "AgentId",
    "Context",
    "EmptyResponseError",
    "FinalResponse",
    "MalformedToolCallError",
    "MockAgent",
    "ModelAgent",
    "ModelErrorPayload",
    "ToolCallRequested",
    "TurnError",
    "TurnResult",
]
