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

import hashlib
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path
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


class ToolUseStart(_Model):
    """Start of a tool-use block. Carries the tool ``name`` and call ``id`` —
    both arrive on the provider's block-start frame (Anthropic
    ``content_block_start``, Ollama / OpenAI's first ``tool_calls`` chunk) and
    would otherwise be lost, since :class:`ToolUseDelta` carries only argument
    JSON. The streaming accumulator uses this to reconstruct the tool call
    faithfully.
    """

    type: Literal["tool_use_start"] = "tool_use_start"
    index: int
    id: str
    name: str


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
    | ToolUseStart
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
# Cross-language request hashing (#37, #38)
# ============================================================================


def _canonicalize_json(v: Any) -> str:
    """Canonicalize a JSON-compatible value to a stable string.

    Object keys are sorted lexicographically by ASCII codepoint and there is
    no insignificant whitespace. Mirrors the Rust ``canonicalize_json`` in
    ``rust/crates/spore-core/src/model.rs`` byte-for-byte.
    """

    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ",".join(_canonicalize_json(x) for x in v) + "]"
    if isinstance(v, dict):
        keys = sorted(v.keys())
        return (
            "{"
            + ",".join(
                json.dumps(k, ensure_ascii=False) + ":" + _canonicalize_json(v[k]) for k in keys
            )
            + "}"
        )
    raise TypeError(f"non-JSON value in canonicalize_json: {type(v).__name__}")


def request_hash(request: ModelRequest) -> str:
    """Stable content hash of a :class:`ModelRequest`.

    Canonicalizes the request to JSON (object keys sorted lexicographically,
    no insignificant whitespace), SHA-256 hashes the UTF-8 bytes, and
    hex-encodes the first 8 bytes (16 lowercase hex characters representing
    the leading u64). Identical across Rust / TypeScript / Python / Go for
    the same request — see ``fixtures/model_hashing/cases.json``.
    """

    value = request.model_dump(mode="json")
    canonical = _canonicalize_json(value)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return digest[:8].hex()


# ============================================================================
# Recorded exchange + replay
# ============================================================================


class RecordedExchange(_Model):
    """A ``(ModelRequest, ModelResponse)`` pair as serialised in the shared
    fixtures under ``fixtures/model_responses/``.

    ``request_hash`` (issue #37) is populated by
    :class:`RecordingModelInterface` (#38) to enable content-addressed
    replay. Fixtures recorded before #37 do not include it; absence
    triggers positional fallback in :class:`ReplayModelInterface`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, ser_json_inf_nan="constants")

    request_hash: str | None = Field(default=None)
    request: ModelRequest
    response: ModelResponse
    provider: str
    model_id: str | None = Field(default=None)
    recorded_at: str | None = Field(default=None)
    duration_ms: int | None = Field(default=None)


_RecordedExchangeAdapter = TypeAdapter(RecordedExchange)


class ReplayMode(str, Enum):
    """How a :class:`ReplayModelInterface` matches incoming requests."""

    POSITIONAL = "positional"
    HASH_MATCHED = "hash_matched"


class ReplayModelInterface:
    """Replay of recorded ``(request, response)`` pairs.

    Defaults to :attr:`ReplayMode.HASH_MATCHED` when every entry has a
    ``request_hash`` and the list is non-empty; otherwise falls back to
    :attr:`ReplayMode.POSITIONAL` so pre-#37 fixtures continue to work.
    """

    def __init__(
        self,
        exchanges: list[RecordedExchange],
        provider: ProviderInfo,
        mode: ReplayMode | None = None,
    ) -> None:
        self._exchanges: list[RecordedExchange] = list(exchanges)
        self._cursor: int = 0
        self._lock = anyio.Lock()
        self._provider = provider
        if mode is None:
            if self._exchanges and all(e.request_hash is not None for e in self._exchanges):
                mode = ReplayMode.HASH_MATCHED
            else:
                mode = ReplayMode.POSITIONAL
        self._mode = mode

    @classmethod
    def from_jsonl(
        cls,
        text: str,
        provider: ProviderInfo,
        mode: ReplayMode | None = None,
    ) -> ReplayModelInterface:
        exchanges: list[RecordedExchange] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            exchanges.append(_RecordedExchangeAdapter.validate_json(line))
        return cls(exchanges, provider, mode)

    def mode(self) -> ReplayMode:
        return self._mode

    def remaining(self) -> int:
        return max(0, len(self._exchanges) - self._cursor)

    async def call(self, request: ModelRequest) -> ModelResponse:
        if self._mode is ReplayMode.HASH_MATCHED:
            want = request_hash(request)
            for exchange in self._exchanges:
                if exchange.request_hash == want:
                    return exchange.response.model_copy(deep=True)
            raise ProviderError(
                code=0,
                message=f"no matching fixture for request_hash={want}",
            )
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
                yield ToolUseStart(index=idx, id=block.id, name=block.name)
                yield ToolUseDelta(
                    index=idx,
                    partial_json=json.dumps(block.input, separators=(",", ":")),
                )
            yield ContentBlockStop(index=idx)
        yield MessageStop(usage=response.usage, stop_reason=response.stop_reason)

    async def count_tokens(self, request: ModelRequest) -> int:
        # When the fixture was recorded by RecordingModelInterface against a
        # real provider, the recorded response's ``usage.input_tokens``
        # carries the provider's exact count. Use that whenever we can
        # match by hash; fall back to the bytes/4 heuristic only when no
        # matching entry exists (positional fixtures or unrecorded
        # requests).
        if self._mode is ReplayMode.HASH_MATCHED:
            want = request_hash(request)
            for exchange in self._exchanges:
                if exchange.request_hash == want:
                    return exchange.response.usage.input_tokens
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
# RecordingModelInterface (issue #38)
# ============================================================================


class RecordingMode(str, Enum):
    """Modes for :class:`RecordingModelInterface`."""

    RECORD = "record"
    RECORD_IF_NEW = "record_if_new"
    PASSTHROUGH = "passthrough"


class RecordingModelInterface:
    """Transparent wrapper around a real :class:`ModelInterface` that
    appends each ``(request, response)`` pair to a JSONL fixture file as a
    :class:`RecordedExchange` with a stable :func:`request_hash`.
    """

    def __init__(
        self,
        inner: ModelInterface,
        output_path: str | Path,
        mode: RecordingMode,
    ) -> None:
        self._inner = inner
        self._output_path = Path(output_path)
        self._mode = mode
        self._lock = anyio.Lock()

    @property
    def output_path(self) -> Path:
        return self._output_path

    def mode(self) -> RecordingMode:
        return self._mode

    async def _record(
        self,
        request: ModelRequest,
        response: ModelResponse,
        duration_ms: int,
    ) -> None:
        async with self._lock:
            if self._mode is RecordingMode.PASSTHROUGH:
                return
            if self._mode is RecordingMode.RECORD_IF_NEW and self._output_path.exists():
                return
            parent = self._output_path.parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            provider_info = self._inner.provider()
            entry = RecordedExchange(
                request_hash=request_hash(request),
                request=request,
                response=response,
                provider=provider_info.name,
                model_id=provider_info.model_id,
                recorded_at=None,
                duration_ms=duration_ms,
            )
            line = entry.model_dump_json(exclude_none=True)
            with self._output_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")

    async def call(self, request: ModelRequest) -> ModelResponse:
        start = time.monotonic()
        response = await self._inner.call(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            await self._record(request, response, duration_ms)
        except OSError as e:
            raise ProviderError(code=0, message=f"recorder write failed: {e}") from e
        return response

    def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        # Streaming recording is not implemented; pass through unchanged.
        return self._inner.call_streaming(request)

    async def count_tokens(self, request: ModelRequest) -> int:
        return await self._inner.count_tokens(request)

    def provider(self) -> ProviderInfo:
        return self._inner.provider()


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
    "RecordingMode",
    "RecordingModelInterface",
    "ReplayMode",
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
    "ToolUseStart",
    "enforce_budget",
    "enforce_context_limit",
    "request_hash",
]
