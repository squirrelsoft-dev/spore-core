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
   * No text and no tool calls → interpreted by ``stop_reason``: a clean
     ``end_turn`` is the model's completion signal → a (possibly empty)
     terminal ``FinalResponse``; a ``max_tokens`` / ``stop_sequence`` empty
     is an abnormal/truncated stop → ``Error`` with :class:`EmptyResponse`.
   * Otherwise → ``FinalResponse`` with concatenated text (``Thinking``
     blocks discarded — observability, not output).

4. Model error → ``Error`` with :class:`ModelErrorAgent` and ``usage=None``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .errors import SporeError
from .model import (
    ContentBlockDelta,
    Message,
    MessageStart,
    MessageStop,
    ModelError,
    ModelInterface,
    ModelParams,
    ModelRequest,
    ModelResponse,
    StopReason,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ThinkingDelta,
    TokenUsage,
    ToolCall,
    ToolSchema,
    ToolUseBlock,
    ToolUseDelta,
    ToolUseStart,
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
        return self.into_request_with_stream(stream=False)

    def into_request_with_stream(self, *, stream: bool) -> ModelRequest:
        """Build a :class:`ModelRequest`, setting the ``stream`` flag. The
        streaming turn path (issue #103) builds the request with
        ``stream=True``."""

        return ModelRequest(
            messages=list(self.messages),
            tools=list(self.tools),
            params=self.params,
            stream=stream,
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
    # Accumulated reasoning (``Thinking``) text produced in this turn, if any
    # (issue #103, Q4). Defaults to ``None`` and is omitted from serialized
    # output when absent, so pre-#103 serialized ``TurnResult``s round-trip.
    reasoning: str | None = Field(default=None)


class FinalResponse(_Model):
    kind: Literal["final_response"] = "final_response"
    content: str
    usage: TokenUsage
    # Accumulated reasoning (``Thinking``) text produced in this turn, if any
    # (issue #103, Q4). See note on :class:`ToolCallRequested`.
    reasoning: str | None = Field(default=None)


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


# An owned callback that receives **raw** :class:`spore_core.model.StreamEvent`
# values as the agent drains a streaming model call (issue #103, Q1). The agent
# boundary deals only in ``model`` stream events; it does NOT depend on the
# harness ``StreamEvent`` type. The harness wraps its own sink in an adapter
# that maps ``model.StreamEvent`` -> ``harness.StreamEvent``.
AgentStreamSink = Callable[[StreamEvent], None]


@runtime_checkable
class Agent(Protocol):
    """Executes a single turn given a fully assembled :class:`Context`."""

    async def turn(self, context: Context) -> TurnResult: ...

    def id(self) -> AgentId: ...


async def turn_streaming(
    agent: Agent,
    context: Context,
    sink: AgentStreamSink,
) -> TurnResult:
    """Execute one turn, forwarding each raw model ``StreamEvent`` to ``sink``.

    This is the streaming counterpart to :meth:`Agent.turn` (issue #103).
    Because :class:`Agent` is a structural :class:`~typing.Protocol`, the
    streaming entry point is a free function rather than a defaulted method:
    if ``agent`` provides its own ``turn_streaming`` coroutine (e.g.
    :class:`ModelAgent`) it is used; otherwise this **default** ignores the
    sink and delegates to :meth:`Agent.turn`, so every existing ``Agent`` impl
    (e.g. :class:`MockAgent`) keeps working with zero changes.
    """

    own = getattr(agent, "turn_streaming", None)
    if own is not None:
        return await own(context, sink)
    return await agent.turn(context)


def classify_response(response: ModelResponse) -> TurnResult:
    """Classify an accumulated :class:`ModelResponse` into a :class:`TurnResult`.

    Single source of truth shared by :meth:`ModelAgent.turn` and
    :meth:`ModelAgent.turn_streaming` (issue #103) — both the blocking and
    streaming paths buffer a complete ``ModelResponse`` and then run this
    identical logic so classification can never diverge between them.

    ``Thinking`` blocks are accumulated into the ``reasoning`` field (Q4)
    instead of being discarded.
    """

    usage = response.usage

    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
        elif isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ThinkingBlock):
            # Q4: accumulate thinking text instead of discarding it.
            reasoning_parts.append(block.text)

    reasoning = "".join(reasoning_parts) if reasoning_parts else None

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
        return ToolCallRequested(calls=tool_calls, usage=usage, reasoning=reasoning)

    # end_turn | max_tokens | stop_sequence
    if not text_parts and not tool_calls:
        # The meaning of a no-content stop depends on *why* the model stopped.
        # A clean ``end_turn`` is the model's voluntary completion signal: it
        # chose to stop and did not request a tool, so an empty ``end_turn`` is
        # a (possibly empty) terminal FinalResponse, not an error. A
        # ``max_tokens`` / ``stop_sequence`` empty is an abnormal/truncated stop
        # and remains genuinely suspect → EmptyResponse. (Thinking-only output
        # is still empty: thinking is not a terminal response.)
        if stop == "end_turn":
            return FinalResponse(content="", usage=usage, reasoning=reasoning)
        return TurnError(error=AgentErrorEmpty(), usage=usage)
    if tool_calls:
        return ToolCallRequested(calls=tool_calls, usage=usage, reasoning=reasoning)
    return FinalResponse(content="".join(text_parts), usage=usage, reasoning=reasoning)


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
        return classify_response(response)

    async def turn_streaming(self, context: Context, sink: AgentStreamSink) -> TurnResult:
        """Streaming turn (issue #103).

        Builds a streaming request, drains the model stream forwarding each
        **raw** ``StreamEvent`` to ``sink``, reassembles a complete
        :class:`ModelResponse`, then runs the EXACT SAME
        :func:`classify_response` logic as :meth:`turn`.

        Reassembly rules:

        * :class:`ContentBlockDelta` text deltas concatenate per block index
          into a ``TextBlock``.
        * :class:`ThinkingDelta` deltas accumulate into a ``ThinkingBlock``
          (Q4 — surfaced via ``reasoning``, NOT discarded).
        * :class:`ToolUseDelta` fragments concatenate and parse into a
          ``ToolUseBlock``'s ``input`` on block stop.
        * :class:`MessageStop` carries the final ``usage`` + ``stop_reason``.

        Block ordering is preserved by first-seen stream index.

        Tool name + id are recovered from :class:`ToolUseStart`, which every
        provider emits at the tool block's start frame (Anthropic
        ``content_block_start``, Ollama / OpenAI's first ``tool_calls`` chunk)
        before the :class:`ToolUseDelta` argument fragments arrive. The
        accumulator records them and reconstructs the ``ToolCall`` faithfully;
        it only falls back to a synthesized per-index id ``call_{index}`` and
        empty name if a stream somehow omitted the start frame.
        """

        request = context.into_request_with_stream(stream=True)
        acc = _StreamAccumulator()
        try:
            async for event in self._model.call_streaming(request):
                # Forward the RAW model event to the sink first (Q1), then
                # fold it into the in-progress response.
                sink(event)
                acc.fold(event)
        except ModelError as e:
            return TurnError(error=_wrap_model_error(e), usage=None)
        return classify_response(acc.into_response())


# ============================================================================
# Streaming reassembly (issue #103)
# ============================================================================


class _StreamAccumulator:
    """Reassembles streamed ``StreamEvent``s into a :class:`ModelResponse`.

    Tracks an ordered set of partial blocks keyed by their stream ``index`` so
    the reconstructed ``content`` preserves emission (first-seen) order
    regardless of interleaving.
    """

    def __init__(self) -> None:
        # index -> [kind, payload]; kind is "text" | "thinking" | "tool_json".
        self._blocks: list[tuple[int, str, list[str]]] = []
        # index -> (id, name) captured from ToolUseStart for tool-use blocks.
        self._tool_meta: dict[int, tuple[str, str]] = {}
        self._usage: TokenUsage = TokenUsage()
        self._stop_reason: StopReason | None = None

    def _buf(self, index: int, kind: str) -> list[str]:
        for i, k, payload in self._blocks:
            if i == index:
                return payload
        payload = []
        self._blocks.append((index, kind, payload))
        return payload

    def fold(self, event: StreamEvent) -> None:
        if isinstance(event, MessageStart):
            return
        if isinstance(event, ContentBlockDelta):
            self._buf(event.index, "text").append(event.delta)
        elif isinstance(event, ThinkingDelta):
            self._buf(event.index, "thinking").append(event.delta)
        elif isinstance(event, ToolUseStart):
            # Record id + name; ensure the block exists in first-seen order.
            self._buf(event.index, "tool_json")
            self._tool_meta[event.index] = (event.id, event.name)
        elif isinstance(event, ToolUseDelta):
            self._buf(event.index, "tool_json").append(event.partial_json)
        elif isinstance(event, MessageStop):
            self._usage = event.usage
            self._stop_reason = event.stop_reason
        # ContentBlockStop: nothing to accumulate.

    def into_response(self) -> ModelResponse:
        content: list[Any] = []
        for index, kind, payload in self._blocks:
            joined = "".join(payload)
            if kind == "text":
                content.append(TextBlock(text=joined))
            elif kind == "thinking":
                content.append(ThinkingBlock(text=joined))
            else:  # tool_json
                try:
                    parsed = json.loads(joined) if joined else {}
                except json.JSONDecodeError:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
                # id + name come from the ToolUseStart event every provider
                # emits at block start. Fall back to a stable per-index id and
                # empty name only if a stream somehow omitted the start frame,
                # so reconstruction is always well-formed.
                tool_id, tool_name = self._tool_meta.get(index, ("", ""))
                content.append(
                    ToolUseBlock(
                        id=tool_id or f"call_{index}",
                        name=tool_name,
                        input=parsed,
                    )
                )
        return ModelResponse(
            content=content,
            usage=self._usage,
            # Default to END_TURN if the stream ended without MessageStop.
            stop_reason=self._stop_reason or StopReason.END_TURN,
        )


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
    "AgentStreamSink",
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
    "classify_response",
    "turn_streaming",
]
