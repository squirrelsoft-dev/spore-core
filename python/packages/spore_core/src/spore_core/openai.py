"""Issue #40 — :class:`OpenAIModelInterface`: real OpenAI Chat Completions
client.

Implements the :class:`spore_core.model.ModelInterface` protocol against
``${base_url}/chat/completions``. Translates :class:`ModelRequest` /
:class:`ModelResponse` to and from OpenAI's wire format, parses the SSE
event stream for ``call_streaming`` (with tool-call delta accumulation),
and maps HTTP errors to typed :class:`ModelError` subclasses with
retry/backoff for transient failures.

Mirrors the Rust reference at ``rust/crates/spore-core/src/openai.rs``
and the analogous TypeScript/Go implementations.

Provider-specific shape:

- System messages stay in the ``messages`` array as ``{role:"system",...}``
  entries (Anthropic extracts them into a top-level field — OpenAI does
  not).
- Assistant tool calls travel as a ``tool_calls`` array on the assistant
  message with ``function.arguments`` as a JSON-encoded string. Tool
  results travel as standalone ``{role:"tool", tool_call_id, content}``
  messages.
- Reasoning models (``o1``, ``o3``, ``o4*``) do not accept ``temperature``
  and replace ``max_tokens`` with ``max_completion_tokens``. Detection is
  by model-id prefix.
- Streaming SSE chunks contain ``delta.content`` (text), optional
  ``delta.reasoning`` (thinking), and ``delta.tool_calls`` (partial tool
  calls indexed and accumulated across chunks). The stream ends with a
  literal ``data: [DONE]`` line. The final usage block only appears when
  the request set ``stream_options: {include_usage: true}``.
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

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3

_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


# ============================================================================
# Helpers
# ============================================================================


def context_window(model_id: str) -> int:
    """Context window for known OpenAI model ids.

    Returns 0 for unknown ids so callers can detect "unknown model"
    rather than silently getting a plausible-but-wrong value.
    """

    if model_id.startswith("gpt-4o"):
        return 128_000
    if model_id.startswith("gpt-4.1"):
        return 1_000_000
    if model_id.startswith("o3") or model_id.startswith("o4"):
        return 200_000
    if model_id.startswith("o1"):
        return 128_000
    return 0


def is_reasoning_model(model_id: str) -> bool:
    """True for o-series reasoning models (``o1``, ``o3``, ``o4*``).

    These have different parameter constraints — no ``temperature``, and
    use ``max_completion_tokens`` instead of ``max_tokens``.
    """

    return model_id.startswith("o1") or model_id.startswith("o3") or model_id.startswith("o4")


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff: 0.5, 1, 2, 4, 8, 16, 30 (cap) seconds."""

    base = 0.5 * (1 << min(attempt, 6))
    return min(base, 30.0)


def _role_to_openai(role: Role) -> str:
    if role is Role.SYSTEM:
        return "system"
    if role is Role.USER:
        return "user"
    if role is Role.ASSISTANT:
        return "assistant"
    if role is Role.TOOL:
        return "tool"
    raise ValueError(f"unexpected role: {role}")


def _message_to_openai(m: Message) -> dict[str, Any]:
    role = _role_to_openai(m.role)
    content = m.content
    if isinstance(content, TextContent):
        return {"role": role, "content": content.text}
    if isinstance(content, ToolCallContent):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": content.id,
                    "type": "function",
                    "function": {
                        "name": content.name,
                        # OpenAI wants arguments as a JSON-encoded STRING.
                        "arguments": json.dumps(content.input, separators=(",", ":")),
                    },
                }
            ],
        }
    if isinstance(content, ToolResultContent):
        return {
            "role": "tool",
            "tool_call_id": content.tool_use_id,
            "content": content.content,
        }
    if isinstance(content, ImageContent):
        # OpenAI chat-completions image input uses a content-parts array
        # (``{type: "image_url", image_url: {url: "data:..."}}``). The
        # harness does not currently emit image content into requests, so
        # we serialize a textual placeholder rather than introduce a
        # heterogeneous shape.
        return {"role": role, "content": f"[image: {content.media_type}]"}
    raise TypeError(f"unsupported Content variant: {type(content).__name__}")


def build_request_body(model_id: str, request: ModelRequest, stream: bool) -> dict[str, Any]:
    """Translate a Spore :class:`ModelRequest` to the OpenAI Chat
    Completions API request body."""

    messages = [_message_to_openai(m) for m in request.messages]

    reasoning = is_reasoning_model(model_id)
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if reasoning:
        if request.params.max_tokens is not None:
            body["max_completion_tokens"] = request.params.max_tokens
        # temperature omitted for reasoning models
    else:
        if request.params.max_tokens is not None:
            body["max_tokens"] = request.params.max_tokens
        if request.params.temperature is not None:
            body["temperature"] = request.params.temperature
    if request.params.top_p is not None:
        body["top_p"] = request.params.top_p
    if request.params.stop_sequences:
        body["stop"] = list(request.params.stop_sequences)
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def parse_stop_reason(raw: str | None) -> StopReason:
    if raw in ("tool_calls", "function_call"):
        return StopReason.TOOL_USE
    if raw == "length":
        return StopReason.MAX_TOKENS
    if raw == "stop":
        return StopReason.END_TURN
    return StopReason.END_TURN


def parse_response_body(body: dict[str, Any]) -> ModelResponse:
    """Translate an OpenAI Chat Completions response body into a Spore
    :class:`ModelResponse`."""

    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content: list[ContentBlock] = []
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        content.append(ThinkingBlock(text=reasoning))
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append(TextBlock(text=text))
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        arg_str = func.get("arguments") or ""
        if arg_str:
            try:
                parsed_input = json.loads(arg_str)
                if not isinstance(parsed_input, dict):
                    parsed_input = {"__raw__": arg_str}
            except ValueError:
                parsed_input = {"__raw__": arg_str}
        else:
            parsed_input = {}
        content.append(
            ToolUseBlock(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                input=parsed_input,
            )
        )

    usage_raw = body.get("usage") or {}
    cache_read: int | None = None
    details = usage_raw.get("prompt_tokens_details")
    if isinstance(details, dict):
        c = details.get("cached_tokens")
        if isinstance(c, int):
            cache_read = c

    usage = TokenUsage(
        input_tokens=int(usage_raw.get("prompt_tokens") or 0),
        output_tokens=int(usage_raw.get("completion_tokens") or 0),
        cache_read_tokens=cache_read,
        # OpenAI does not report cache writes directly.
        cache_write_tokens=None,
    )
    return ModelResponse(
        content=content,
        usage=usage,
        stop_reason=parse_stop_reason(choice.get("finish_reason")),
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
    if status in (408, 504):
        return _ModelTimeoutError()
    return ProviderError(code=status, message=_extract_error_message(body_text))


# ============================================================================
# SSE parsing — OpenAI Chat Completions
# ============================================================================


async def _sse_to_events(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    """Convert an OpenAI SSE stream into a stream of :class:`StreamEvent`s.

    OpenAI streams chat-completion delta chunks. Each ``data:`` line
    carries a JSON object. Tool calls arrive as partial entries in
    ``delta.tool_calls``, indexed; the ``id`` and ``function.name``
    arrive on the first chunk for a given index, and subsequent chunks
    for the same index carry incremental ``function.arguments``
    JSON-fragment strings. The stream ends with ``data: [DONE]``.
    """

    buf = ""
    usage_input = 0
    usage_output = 0
    cache_read: int | None = None
    stop_reason: StopReason = StopReason.END_TURN
    started = False
    tool_indices_seen: set[int] = set()
    content_index_emitted = False
    content_index = 0
    try:
        async for chunk in response.aiter_text():
            buf += chunk
            while True:
                idx = buf.find("\n")
                if idx < 0:
                    break
                raw_line = buf[:idx]
                buf = buf[idx + 1 :]
                line = raw_line.rstrip("\r")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].lstrip(" ")
                if not data:
                    continue
                if data == "[DONE]":
                    yield _MessageStop(
                        usage=TokenUsage(
                            input_tokens=usage_input,
                            output_tokens=usage_output,
                            cache_read_tokens=cache_read,
                        ),
                        stop_reason=stop_reason,
                    )
                    return
                try:
                    value: Any = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(value, dict):
                    continue
                if not started:
                    started = True
                    yield _MessageStart()
                u = value.get("usage")
                if isinstance(u, dict):
                    pt = u.get("prompt_tokens")
                    if isinstance(pt, int):
                        usage_input = pt
                    ct = u.get("completion_tokens")
                    if isinstance(ct, int):
                        usage_output = ct
                    d = u.get("prompt_tokens_details")
                    if isinstance(d, dict):
                        c = d.get("cached_tokens")
                        if isinstance(c, int):
                            cache_read = c
                choices = value.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                fr = choice.get("finish_reason")
                if isinstance(fr, str):
                    stop_reason = parse_stop_reason(fr)
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if isinstance(text, str) and text:
                    if not content_index_emitted:
                        content_index_emitted = True
                    yield _ContentBlockDelta(index=content_index, delta=text)
                reasoning = delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    yield _ThinkingDelta(index=content_index, delta=reasoning)
                tcs = delta.get("tool_calls")
                if isinstance(tcs, list):
                    # Tool call indices are independent of text content
                    # index; shift by 1 to keep them disjoint from index
                    # 0 (which conventionally carries text).
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        i_raw = tc.get("index")
                        i = int(i_raw) if isinstance(i_raw, int) else 0
                        event_index = i + 1
                        func = tc.get("function") or {}
                        if event_index not in tool_indices_seen:
                            tool_indices_seen.add(event_index)
                            if content_index_emitted:
                                yield _ContentBlockStop(index=content_index)
                                content_index_emitted = False
                                content_index = event_index
                            # The id + function.name arrive on this first chunk
                            # for the index; emit ToolUseStart so they aren't
                            # lost when only argument fragments follow.
                            name = ""
                            if isinstance(func, dict):
                                fn = func.get("name")
                                if isinstance(fn, str):
                                    name = fn
                            tc_id = tc.get("id")
                            call_id = (
                                tc_id if isinstance(tc_id, str) and tc_id else f"call_{event_index}"
                            )
                            yield _ToolUseStart(index=event_index, id=call_id, name=name)
                        arg_delta = func.get("arguments") if isinstance(func, dict) else None
                        if isinstance(arg_delta, str) and arg_delta:
                            yield _ToolUseDelta(index=event_index, partial_json=arg_delta)
        # Stream ended without [DONE] — still emit MessageStop.
        yield _MessageStop(
            usage=TokenUsage(
                input_tokens=usage_input,
                output_tokens=usage_output,
                cache_read_tokens=cache_read,
            ),
            stop_reason=stop_reason,
        )
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
# OpenAIModelInterface
# ============================================================================


class OpenAIModelInterface:
    """Reference OpenAI client.

    Construct with an API key and a model id; callers can override
    ``base_url`` (for Azure OpenAI, local proxies, or mocking) and tune
    retry behavior. An optional :class:`httpx.AsyncClient` may be
    injected to share connection pools across multiple interfaces —
    tests pass a transport-mocked client.

    The API key is redacted from :meth:`__repr__` so it never leaks
    into logs.
    """

    DEFAULT_BASE_URL = DEFAULT_BASE_URL
    DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
    DEFAULT_MAX_RETRIES = DEFAULT_MAX_RETRIES

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
            "OpenAIModelInterface("
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
    ) -> OpenAIModelInterface:
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
            "authorization": f"Bearer {self._api_key}",
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
        url = f"{self._base_url}/chat/completions"
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
        url = f"{self._base_url}/chat/completions"
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
        """Estimate token count via the bytes/4 heuristic.

        OpenAI has no dedicated token-counting endpoint. This estimate
        is sufficient for compaction decisions; exact counts come back
        via response ``usage``. A future revision may pull in
        ``tiktoken``.
        """

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
        return ProviderInfo(
            name="openai",
            model_id=self._model_id,
            context_window=context_window(self._model_id),
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "OpenAIModelInterface",
    "build_request_body",
    "context_window",
    "is_reasoning_model",
    "parse_response_body",
    "parse_stop_reason",
]
