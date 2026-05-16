"""ModelInterface — boundary between the harness and the underlying LLM.

Mirrors the Rust reference at ``rust/crates/spore-core/src/model.rs``. Wire
formats (JSON field names and discriminator tags) match byte-for-byte so the
shared fixtures under ``fixtures/model_responses/`` replay identically across
all four language implementations.

Rules enforced here (per spec issue #1):

1. :class:`TokenUsage` is reported on every successful ``call`` and on the
   final ``message_stop`` of ``call_streaming``.
2. :class:`ContextLimitExceeded` is raised before the provider is contacted
   when the harness detects that the request exceeds the context window
   (see :func:`enforce_context_limit`).
3. :class:`BudgetExceeded` is a harness-side check against
   ``ModelParams.max_tokens`` (see :func:`enforce_budget`).
4. Provider-specific retry/backoff lives in implementations, not the
   protocol.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import AsyncIterator
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Protocol, runtime_checkable

import anyio
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .errors import AlwaysHaltError, SporeError

# ============================================================================
# Roles, content, messages
# ============================================================================


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _Model(BaseModel):
    """Project-wide pydantic base — strict, frozen-ish defaults."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolCall(_Model):
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(_Model):
    tool_use_id: str
    content: str
    is_error: bool = False


# Content variants — tagged-union on ``type``. Rust serializes
# ``Content::ToolCall(ToolCall)`` by flattening the inner struct alongside the
# tag, so the wire shape is ``{"type":"tool_call","id":...,"name":...,"input":...}``.
class TextContent(_Model):
    type: Literal["text"] = "text"
    text: str


class ToolCallContent(_Model):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContent(_Model):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class ImageContent(_Model):
    type: Literal["image"] = "image"
    media_type: str
    data: str


Content = Annotated[
    TextContent | ToolCallContent | ToolResultContent | ImageContent,
    Field(discriminator="type"),
]


class Message(_Model):
    role: Role
    content: Content


# ============================================================================
# Tool schema (subset — canonical type lives with ToolRegistry, #4)
# ============================================================================


class ToolSchema(_Model):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Request / params / response
# ============================================================================


class ModelParams(_Model):
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_budget: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] = Field(default_factory=list)


class ModelRequest(_Model):
    messages: list[Message] = Field(default_factory=list)
    tools: list[ToolSchema] = Field(default_factory=list)
    params: ModelParams = Field(default_factory=ModelParams)
    stream: bool = False


class StopReason(str, Enum):
    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"


# ContentBlock — tagged on ``type``. Rust's ``ToolUse(ToolCall)`` flattens the
# ToolCall struct alongside the tag → ``{"type":"tool_use","id":...,"name":...,"input":...}``.
class TextBlock(_Model):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(_Model):
    type: Literal["thinking"] = "thinking"
    text: str


class ToolUseBlock(_Model):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock,
    Field(discriminator="type"),
]


class TokenUsage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


class ModelResponse(_Model):
    content: list[ContentBlock]
    usage: TokenUsage
    stop_reason: StopReason


# ============================================================================
# Streaming
# ============================================================================


class MessageStart(_Model):
    type: Literal["message_start"] = "message_start"


class ContentBlockDelta(_Model):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: str


class ThinkingDelta(_Model):
    type: Literal["thinking_delta"] = "thinking_delta"
    index: int
    delta: str


class ToolUseDelta(_Model):
    type: Literal["tool_use_delta"] = "tool_use_delta"
    index: int
    partial_json: str


class ContentBlockStop(_Model):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageStop(_Model):
    type: Literal["message_stop"] = "message_stop"
    usage: TokenUsage
    stop_reason: StopReason


StreamEvent = Annotated[
    MessageStart
    | ContentBlockDelta
    | ThinkingDelta
    | ToolUseDelta
    | ContentBlockStop
    | MessageStop,
    Field(discriminator="type"),
]


# ============================================================================
# Provider identity
# ============================================================================


class ProviderInfo(_Model):
    name: str
    model_id: str
    context_window: int


# ============================================================================
# Errors
# ============================================================================


class ModelError(SporeError):
    """Root of every typed error raised by a ``ModelInterface`` implementation.

    Subclasses mirror the Rust ``ModelError`` enum variants; the ``kind``
    class attribute matches the serde tag so cross-language JSON
    representations agree.
    """

    kind: ClassVar[str] = "ModelError"


class ProviderError(ModelError):
    kind: ClassVar[str] = "ProviderError"

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"provider error {code}: {message}")


class RateLimited(ModelError):
    kind: ClassVar[str] = "RateLimited"

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(f"rate limited (retry_after={retry_after!r})")


class ContextLimitExceeded(ModelError, AlwaysHaltError):
    kind: ClassVar[str] = "ContextLimitExceeded"

    def __init__(self, limit: int, actual: int) -> None:
        self.limit = limit
        self.actual = actual
        super().__init__(f"context limit exceeded: {actual} tokens > limit {limit}")


class BudgetExceeded(ModelError, AlwaysHaltError):
    kind: ClassVar[str] = "BudgetExceeded"

    def __init__(self, budget: int, used: int) -> None:
        self.budget = budget
        self.used = used
        super().__init__(f"budget exceeded: {used} > budget {budget}")


class TimeoutError(ModelError):  # noqa: A001 — intentional shadow of builtin
    kind: ClassVar[str] = "Timeout"

    def __init__(self, message: str = "model call timed out") -> None:
        super().__init__(message)


# ============================================================================
# Shared pre/post-call validation
# ============================================================================


def enforce_context_limit(actual: int, limit: int) -> None:
    """Raise :class:`ContextLimitExceeded` when the request would overflow."""

    if actual > limit:
        raise ContextLimitExceeded(limit=limit, actual=actual)


def enforce_budget(used: int, budget: int | None = None) -> None:
    """Raise :class:`BudgetExceeded` when ``used`` overruns ``budget``."""

    if budget is not None and used > budget:
        raise BudgetExceeded(budget=budget, used=used)


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class ModelInterface(Protocol):
    """Structural boundary between harness and LLM."""

    async def call(self, request: ModelRequest) -> ModelResponse: ...

    def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]: ...

    async def count_tokens(self, request: ModelRequest) -> int: ...

    def provider(self) -> ProviderInfo: ...


# ============================================================================
# Recorded exchange + replay
# ============================================================================


class RecordedExchange(_Model):
    request: ModelRequest
    response: ModelResponse
    provider: str
    recorded_at: str | None = None


_RecordedExchangeAdapter = TypeAdapter(RecordedExchange)


class ReplayModelInterface:
    """Positional replay of recorded ``(request, response)`` pairs.

    Matching is positional — the n-th call returns the n-th recorded
    response. Mirrors the Rust reference implementation; the shared
    fixtures under ``fixtures/model_responses/`` exercise this code in
    every language.
    """

    def __init__(self, exchanges: list[RecordedExchange], provider: ProviderInfo) -> None:
        self._exchanges: list[RecordedExchange] = list(exchanges)
        self._cursor: int = 0
        self._lock = anyio.Lock()
        self._provider = provider

    @classmethod
    def from_jsonl(cls, text: str, provider: ProviderInfo) -> ReplayModelInterface:
        exchanges: list[RecordedExchange] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            exchanges.append(_RecordedExchangeAdapter.validate_json(line))
        return cls(exchanges, provider)

    def remaining(self) -> int:
        return max(0, len(self._exchanges) - self._cursor)

    async def call(self, request: ModelRequest) -> ModelResponse:
        async with self._lock:
            if self._cursor >= len(self._exchanges):
                raise ProviderError(
                    code=0,
                    message="replay exhausted: no more recorded exchanges",
                )
            exchange = self._exchanges[self._cursor]
            self._cursor += 1
            return exchange.response.model_copy(deep=True)

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        response = await self.call(request)
        yield MessageStart()
        for idx, block in enumerate(response.content):
            if isinstance(block, TextBlock):
                yield ContentBlockDelta(index=idx, delta=block.text)
            elif isinstance(block, ThinkingBlock):
                yield ThinkingDelta(index=idx, delta=block.text)
            elif isinstance(block, ToolUseBlock):
                yield ToolUseDelta(
                    index=idx,
                    partial_json=json.dumps(block.input, separators=(",", ":")),
                )
            yield ContentBlockStop(index=idx)
        yield MessageStop(usage=response.usage, stop_reason=response.stop_reason)

    async def count_tokens(self, request: ModelRequest) -> int:
        # Cheap deterministic estimate sufficient for fixture replay;
        # mirrors the Rust ~4-chars/token rule of thumb.
        total = 0
        for msg in request.messages:
            c = msg.content
            if isinstance(c, TextContent):
                total += len(c.text)
            elif isinstance(c, ToolCallContent):
                total += len(c.name) + len(json.dumps(c.input, separators=(",", ":")))
            elif isinstance(c, ToolResultContent):
                total += len(c.content)
            elif isinstance(c, ImageContent):
                total += 0
        return total // 4

    def provider(self) -> ProviderInfo:
        return self._provider


# ============================================================================
# Mock implementation (test utility)
# ============================================================================


class MockModelInterface:
    """Programmable mock for unit tests.

    Each entry pushed via :meth:`push_response` is yielded on successive
    calls; pushed exceptions are raised instead. ``call_count`` tracks
    invocations.
    """

    def __init__(self, provider: ProviderInfo) -> None:
        self._provider = provider
        self._responses: deque[ModelResponse | ModelError] = deque()
        self._token_counts: deque[int | ModelError] = deque()
        self.call_count: int = 0

    def push_response(self, value: ModelResponse | ModelError) -> MockModelInterface:
        self._responses.append(value)
        return self

    def push_token_count(self, value: int | ModelError) -> MockModelInterface:
        self._token_counts.append(value)
        return self

    async def call(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        if not self._responses:
            raise ProviderError(code=0, message="mock: no response queued")
        item = self._responses.popleft()
        if isinstance(item, ModelError):
            raise item
        return item

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        response = await self.call(request)
        yield MessageStart()
        yield MessageStop(usage=response.usage, stop_reason=response.stop_reason)

    async def count_tokens(self, request: ModelRequest) -> int:
        if not self._token_counts:
            return 0
        item = self._token_counts.popleft()
        if isinstance(item, ModelError):
            raise item
        return item

    def provider(self) -> ProviderInfo:
        return self._provider


__all__ = [
    "BudgetExceeded",
    "Content",
    "ContentBlock",
    "ContentBlockDelta",
    "ContentBlockStop",
    "ContextLimitExceeded",
    "ImageContent",
    "Message",
    "MessageStart",
    "MessageStop",
    "MockModelInterface",
    "ModelError",
    "ModelInterface",
    "ModelParams",
    "ModelRequest",
    "ModelResponse",
    "ProviderError",
    "ProviderInfo",
    "RateLimited",
    "RecordedExchange",
    "ReplayModelInterface",
    "Role",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "TextContent",
    "ThinkingBlock",
    "ThinkingDelta",
    "TimeoutError",
    "TokenUsage",
    "ToolCall",
    "ToolCallContent",
    "ToolResult",
    "ToolResultContent",
    "ToolSchema",
    "ToolUseBlock",
    "ToolUseDelta",
    "enforce_budget",
    "enforce_context_limit",
]
