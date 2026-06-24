"""Issue #39 — :class:`AnthropicModelInterface`: real Anthropic Messages API
client.

Implements the :class:`spore_core.model.ModelInterface` protocol against
``https://api.anthropic.com/v1/messages``. Translates
:class:`ModelRequest` / :class:`ModelResponse` to and from Anthropic's
wire format, parses the SSE event stream for ``call_streaming``, hits
``/v1/messages/count_tokens`` for accurate token counts, and maps HTTP
errors to typed :class:`ModelError` subclasses with retry/backoff for
transient failures.

Mirrors the Rust reference at
``rust/crates/spore-core/src/anthropic.rs`` and the analogous
TypeScript/Go implementations.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .model import (
    ContentBlock,
    ImageContent,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ProviderInfo,
    RateLimited,
    Role,
    StopReason,
    StreamEvent,
    StreamInterrupted,
    TextBlock,
    TextContent,
    ThinkingBlock,
    TokenUsage,
    ToolCallContent,
    ToolResultContent,
    ToolUseBlock,
    Transport,
)
from .model import (
    ContentBlockDelta as _ContentBlockDelta,
)
from .model import (
    ContentBlockStop as _ContentBlockStop,
)
from .model import (
    MessageStart as _MessageStart,
)
from .model import (
    MessageStop as _MessageStop,
)
from .model import (
    ThinkingDelta as _ThinkingDelta,
)
from .model import (
    TimeoutError as _ModelTimeoutError,
)
from .model import (
    ToolUseDelta as _ToolUseDelta,
)
from .model import (
    ToolUseStart as _ToolUseStart,
)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


# ============================================================================
# Helpers
# ============================================================================


def context_window(model_id: str) -> int:
    """Context window for a given model id.

    Falls back to 200k for any ``claude-*`` id (current family default) and
    to 0 otherwise so that callers can detect "unknown model" rather than
    silently getting a plausible-but-wrong value.
    """

    if model_id.startswith("claude-"):
        return 200_000
    return 0


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff: 0.5, 1, 2, 4, 8, 16, 30 (cap) seconds."""

    base = 0.5 * (1 << min(attempt, 6))
    return min(base, 30.0)


def _content_to_anthropic(content: Any) -> dict[str, Any]:
    if isinstance(content, TextContent):
        return {"type": "text", "text": content.text}
    if isinstance(content, ToolCallContent):
        return {
            "type": "tool_use",
            "id": content.id,
            "name": content.name,
            "input": content.input,
        }
    if isinstance(content, ToolResultContent):
        out: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": content.tool_use_id,
            "content": content.content,
        }
        if content.is_error:
            out["is_error"] = True
        return out
    if isinstance(content, ImageContent):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": content.media_type,
                "data": content.data,
            },
        }
    raise TypeError(f"unsupported Content variant: {type(content).__name__}")


def _role_to_anthropic(role: Role) -> str:
    if role is Role.USER:
        return "user"
    if role is Role.ASSISTANT:
        return "assistant"
    if role is Role.TOOL:
        # Anthropic carries tool_result in user-role messages.
        return "user"
    raise ValueError(f"unexpected non-anthropic role: {role}")


def build_request_body(model_id: str, request: ModelRequest, stream: bool) -> dict[str, Any]:
    """Translate a Spore :class:`ModelRequest` to the Anthropic Messages API
    request body.

    System-role messages are extracted into the top-level ``system`` field;
    all other messages keep their role and have their :class:`Content`
    wrapped in a single-element content-block array.
    """

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role is Role.SYSTEM:
            if isinstance(m.content, TextContent):
                system_parts.append(m.content.text)
            continue
        messages.append(
            {
                "role": _role_to_anthropic(m.role),
                "content": [_content_to_anthropic(m.content)],
            }
        )
    body: dict[str, Any] = {
        "model": model_id,
        "max_tokens": request.params.max_tokens
        if request.params.max_tokens is not None
        else DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if request.params.temperature is not None:
        body["temperature"] = request.params.temperature
    if request.params.top_p is not None:
        body["top_p"] = request.params.top_p
    if request.params.stop_sequences:
        body["stop_sequences"] = list(request.params.stop_sequences)
    if request.tools:
        body["tools"] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in request.tools
        ]
    if stream:
        body["stream"] = True
    return body


def parse_stop_reason(raw: str | None) -> StopReason:
    if raw == "tool_use":
        return StopReason.TOOL_USE
    if raw == "max_tokens":
        return StopReason.MAX_TOKENS
    if raw == "stop_sequence":
        return StopReason.STOP_SEQUENCE
    return StopReason.END_TURN


def parse_response_body(body: dict[str, Any]) -> ModelResponse:
    """Translate an Anthropic Messages API response body into a Spore
    :class:`ModelResponse`."""

    content: list[ContentBlock] = []
    for block in body.get("content", []) or []:
        kind = block.get("type")
        if kind == "text":
            content.append(TextBlock(text=block.get("text", "")))
        elif kind == "thinking":
            content.append(ThinkingBlock(text=block.get("thinking", "")))
        elif kind == "tool_use":
            content.append(
                ToolUseBlock(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input", {}) or {},
                )
            )
        # Unknown block types ignored — forward-compatible.
    usage_raw = body.get("usage") or {}
    usage = TokenUsage(
        input_tokens=int(usage_raw.get("input_tokens") or 0),
        output_tokens=int(usage_raw.get("output_tokens") or 0),
        cache_read_tokens=(
            int(usage_raw["cache_read_input_tokens"])
            if usage_raw.get("cache_read_input_tokens") is not None
            else None
        ),
        cache_write_tokens=(
            int(usage_raw["cache_creation_input_tokens"])
            if usage_raw.get("cache_creation_input_tokens") is not None
            else None
        ),
    )
    return ModelResponse(
        content=content,
        usage=usage,
        stop_reason=parse_stop_reason(body.get("stop_reason")),
    )


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _extract_error_message(body_text: str) -> str:
    try:
        data = json.loads(body_text)
    except (ValueError, TypeError):
        return body_text[:500]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg
    return body_text[:500]


def _map_status_error(status: int, body_text: str, retry_after: float | None) -> Exception:
    if status == 429:
        return RateLimited(retry_after=retry_after)
    if status == 529:
        return RateLimited(retry_after=None)
    if status in (408, 504):
        return _ModelTimeoutError()
    return ProviderError(code=status, message=_extract_error_message(body_text))


# ============================================================================
# SSE parsing
# ============================================================================


def _parse_sse_event(raw: str) -> tuple[str, str] | None:
    """Parse one SSE event block. Returns ``(event_name, data_payload)`` or
    ``None`` if the block does not include an ``event:`` line."""

    event_name: str | None = None
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            # Strip a single leading space if present, matching SSE conventions.
            data_lines.append(line[len("data:") :].lstrip(" "))
    if event_name is None:
        return None
    return event_name, "\n".join(data_lines)


async def _sse_to_events(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    """Convert an Anthropic SSE response into a stream of :class:`StreamEvent`s.

    Emits:

    - ``message_start`` → :class:`MessageStart`
    - ``content_block_delta`` text/thinking/input_json deltas →
      :class:`ContentBlockDelta` / :class:`ThinkingDelta` /
      :class:`ToolUseDelta`
    - ``content_block_stop`` → :class:`ContentBlockStop`
    - ``message_delta`` → buffers usage / stop_reason
    - ``message_stop`` → :class:`MessageStop` with accumulated usage and
      stop_reason
    """

    buf = ""
    usage_input = 0
    usage_output = 0
    stop_reason: StopReason = StopReason.END_TURN
    try:
        async for chunk in response.aiter_text():
            buf += chunk
            while True:
                idx = buf.find("\n\n")
                if idx < 0:
                    break
                raw = buf[:idx]
                buf = buf[idx + 2 :]
                parsed = _parse_sse_event(raw)
                if parsed is None:
                    continue
                event_name, data = parsed
                if not data or data == "{}":
                    continue
                try:
                    value: Any = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(value, dict):
                    continue
                if event_name == "message_start":
                    msg = value.get("message")
                    if isinstance(msg, dict):
                        u = msg.get("usage")
                        if isinstance(u, dict):
                            it = u.get("input_tokens")
                            if isinstance(it, int):
                                usage_input = it
                    yield _MessageStart()
                elif event_name == "content_block_start":
                    # A tool_use block opens here with its id + name; emit
                    # ToolUseStart so the accumulator captures them before the
                    # input_json_delta arg fragments arrive.
                    index = int(value.get("index") or 0)
                    block = value.get("content_block")
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        yield _ToolUseStart(
                            index=index,
                            id=str(block.get("id") or ""),
                            name=str(block.get("name") or ""),
                        )
                elif event_name == "content_block_delta":
                    index = int(value.get("index") or 0)
                    delta = value.get("delta") or {}
                    kind = delta.get("type") if isinstance(delta, dict) else None
                    if kind == "text_delta":
                        yield _ContentBlockDelta(index=index, delta=str(delta.get("text") or ""))
                    elif kind == "thinking_delta":
                        yield _ThinkingDelta(index=index, delta=str(delta.get("thinking") or ""))
                    elif kind == "input_json_delta":
                        yield _ToolUseDelta(
                            index=index,
                            partial_json=str(delta.get("partial_json") or ""),
                        )
                elif event_name == "content_block_stop":
                    yield _ContentBlockStop(index=int(value.get("index") or 0))
                elif event_name == "message_delta":
                    d = value.get("delta")
                    if isinstance(d, dict):
                        sr = d.get("stop_reason")
                        if isinstance(sr, str):
                            stop_reason = parse_stop_reason(sr)
                    u = value.get("usage")
                    if isinstance(u, dict):
                        ot = u.get("output_tokens")
                        if isinstance(ot, int):
                            usage_output = ot
                elif event_name == "message_stop":
                    yield _MessageStop(
                        usage=TokenUsage(input_tokens=usage_input, output_tokens=usage_output),
                        stop_reason=stop_reason,
                    )
                    return
    except httpx.TimeoutException as e:
        # A read timeout mid-stream is still a timeout (maps to Timeout), not a
        # generic transport drop — placed before the HTTPError handler since
        # TimeoutException is a subclass of HTTPError.
        raise _ModelTimeoutError() from e
    except httpx.HTTPError as e:
        # SC-3: a chunk read/decode failure while draining the body interrupts
        # the stream — a typed, retryable StreamInterrupted, not a generic
        # ProviderError. Mirrors Rust's sse_to_events stream-chunk-error site.
        raise StreamInterrupted(f"stream chunk error: {e}") from e
    finally:
        await response.aclose()


# ============================================================================
# AnthropicModelInterface
# ============================================================================


class AnthropicModelInterface:
    """Reference Anthropic client.

    Construct with an API key and a model id; callers can override
    ``base_url`` (for proxying or mocking) and tune retry behavior. An
    optional :class:`httpx.AsyncClient` may be injected to share connection
    pools across multiple interfaces — tests pass a transport-mocked client.

    The API key is redacted from :meth:`__repr__` so it never leaks into
    logs.
    """

    DEFAULT_BASE_URL = DEFAULT_BASE_URL
    DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
    DEFAULT_MAX_RETRIES = DEFAULT_MAX_RETRIES
    ANTHROPIC_VERSION = ANTHROPIC_VERSION

    def __init__(
        self,
        api_key: str,
        model_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._http_client = http_client
        self._owns_client = http_client is None

    def __repr__(self) -> str:  # pragma: no cover — trivial
        return (
            "AnthropicModelInterface("
            "api_key=<redacted>, "
            f"model_id={self._model_id!r}, "
            f"base_url={self._base_url!r}, "
            f"timeout={self._timeout!r}, "
            f"max_retries={self._max_retries!r})"
        )

    @classmethod
    def from_env(
        cls,
        env_var: str,
        model_id: str,
        **kwargs: Any,
    ) -> AnthropicModelInterface:
        """Read API key from an environment variable. Raises
        :class:`ProviderError` if unset or empty."""

        value = os.environ.get(env_var)
        if value is None:
            raise ProviderError(code=0, message=f"env var `{env_var}` not set")
        if not value.strip():
            raise ProviderError(code=0, message=f"env var `{env_var}` is empty")
        return cls(api_key=value, model_id=model_id, **kwargs)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
            self._owns_client = True
        return self._http_client

    async def aclose(self) -> None:
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def _send_with_retry(self, url: str, body: dict[str, Any]) -> httpx.Response:
        client = self._client()
        attempt = 0
        payload = json.dumps(body).encode("utf-8")
        while True:
            try:
                resp = await client.post(
                    url,
                    content=payload,
                    headers=self._headers(),
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as e:
                if attempt < self._max_retries:
                    await asyncio.sleep(_backoff_delay(attempt))
                    attempt += 1
                    continue
                raise _ModelTimeoutError() from e
            except httpx.HTTPError as e:
                raise Transport(f"HTTP transport error: {e}") from e
            status = resp.status_code
            if 200 <= status < 300:
                return resp
            if status in _RETRYABLE_STATUSES and attempt < self._max_retries:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                delay = retry_after if retry_after is not None else _backoff_delay(attempt)
                await resp.aclose()
                await asyncio.sleep(delay)
                attempt += 1
                continue
            retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            body_text = resp.text
            await resp.aclose()
            raise _map_status_error(status, body_text, retry_after)

    # ── ModelInterface protocol ─────────────────────────────────────────

    async def call(self, request: ModelRequest) -> ModelResponse:
        url = f"{self._base_url}/v1/messages"
        body = build_request_body(self._model_id, request, stream=False)
        resp = await self._send_with_retry(url, body)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(code=0, message=f"response decode failed: {e}") from e
        finally:
            await resp.aclose()
        return parse_response_body(data)

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        url = f"{self._base_url}/v1/messages"
        body = build_request_body(self._model_id, request, stream=True)
        payload = json.dumps(body).encode("utf-8")
        headers = {**self._headers(), "accept": "text/event-stream"}
        client = self._client()
        try:
            req = client.build_request(
                "POST", url, content=payload, headers=headers, timeout=self._timeout
            )
            resp = await client.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise _ModelTimeoutError() from e
        except httpx.HTTPError as e:
            raise Transport(f"HTTP transport error: {e}") from e
        if resp.status_code < 200 or resp.status_code >= 300:
            retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            body_text = await resp.aread()
            await resp.aclose()
            raise _map_status_error(
                resp.status_code, body_text.decode("utf-8", "replace"), retry_after
            )
        async for event in _sse_to_events(resp):
            yield event

    async def count_tokens(self, request: ModelRequest) -> int:
        url = f"{self._base_url}/v1/messages/count_tokens"
        body = build_request_body(self._model_id, request, stream=False)
        resp = await self._send_with_retry(url, body)
        try:
            data = resp.json()
        except ValueError as e:
            raise ProviderError(code=0, message=f"count_tokens decode failed: {e}") from e
        finally:
            await resp.aclose()
        try:
            return int(data["input_tokens"])
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(code=0, message=f"count_tokens missing input_tokens: {e}") from e

    def provider(self) -> ProviderInfo:
        return ProviderInfo(
            name="anthropic",
            model_id=self._model_id,
            context_window=context_window(self._model_id),
        )


__all__ = [
    "ANTHROPIC_VERSION",
    "AnthropicModelInterface",
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT_SECONDS",
    "build_request_body",
    "context_window",
    "parse_response_body",
    "parse_stop_reason",
]


# Suppress unused-import lint: Message is re-exported via type hints
_ = Message
