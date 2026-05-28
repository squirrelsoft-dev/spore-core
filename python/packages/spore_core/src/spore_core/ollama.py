"""Issue #41 — :class:`OllamaModelInterface`: real Ollama HTTP client.

Implements :class:`spore_core.model.ModelInterface` against a local Ollama
server's ``/api/chat``, ``/api/tags``, and ``/api/embed`` endpoints.
Translates :class:`ModelRequest` / :class:`ModelResponse` to and from the
Ollama wire format, parses Ollama's NDJSON stream (one JSON object per
line — not SSE) for :meth:`call_streaming`, and maps HTTP/transport
errors to typed :class:`ModelError` subclasses.

Unlike the Anthropic and OpenAI clients there is **no retry**: spec says
fail fast on connection errors with a helpful message
("Ollama not running", "Run: ollama pull <model>").

Mirrors the Rust reference at ``rust/crates/spore-core/src/ollama.rs``.

Provider-specific shape:

- No API key; default ``base_url`` is ``http://localhost:11434``.
- Sampling parameters (``num_predict``, ``temperature``, ``top_p``,
  ``stop``) are nested under ``options`` rather than top-level keys.
- ``keep_alive`` (default ``"5m"``) controls how long Ollama keeps the
  model loaded after the call returns.
- Tool-call arguments are a JSON **object** in the wire format, not a
  JSON-encoded string like OpenAI.
- Ollama does not return tool-call ids; we synthesize ``call-{i}`` per
  index so downstream ``ToolResult.tool_use_id`` round-trips work.
- Thinking blocks (response-side) are never emitted by Ollama; thinking
  has no request-side representation in the Spore types either, so we
  do not emit a ``thinking`` key on the wire.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    TextContent,
    TokenUsage,
    ToolCallContent,
    ToolResultContent,
    ToolUseBlock,
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
    TimeoutError as _ModelTimeoutError,
)
from .model import (
    ToolUseDelta as _ToolUseDelta,
)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_KEEP_ALIVE = "5m"


# ============================================================================
# Helpers
# ============================================================================


def context_window(model_id: str) -> int:
    """Context window for known Ollama model id prefixes.

    Returns 0 for unknown ids so callers can detect "unknown model"
    rather than silently getting a plausible-but-wrong value.
    """

    if model_id.startswith("llama3.2"):
        return 128_000
    if model_id.startswith("qwen2.5-coder"):
        return 128_000
    if model_id.startswith("mistral"):
        return 32_000
    if model_id.startswith("gemma"):
        return 8_192
    return 0


@dataclass
class ModelMeta:
    """``/api/show``-discovered metadata for the model.

    Populated once, alongside the ``/api/tags`` availability check. All
    fields are best-effort — ``/api/show`` failures leave them unset
    (empty :class:`ModelMeta`) rather than failing the call.
    """

    #: Discovered context window (``*.context_length`` in ``model_info``).
    context_length: int | None = None
    #: Top-level ``capabilities`` array (may contain ``"tools"``).
    capabilities: list[str] = field(default_factory=list)

    def supports_tools(self) -> bool:
        return "tools" in self.capabilities


def _name_matches(tag: str, requested: str) -> bool:
    """Ollama ``/api/tags`` returns names like ``"llama3.2:latest"`` or
    ``"llama3.2:3b"``. Match if the request id equals the full tag or its
    bare-name prefix (before ``:``)."""

    if tag == requested:
        return True
    bare = tag.split(":", 1)[0]
    return bare == requested


def _role_to_ollama(role: Role) -> str:
    if role is Role.SYSTEM:
        return "system"
    if role is Role.USER:
        return "user"
    if role is Role.ASSISTANT:
        return "assistant"
    if role is Role.TOOL:
        return "tool"
    raise ValueError(f"unexpected role: {role}")


def _message_to_ollama(m: Message) -> dict[str, Any]:
    role = _role_to_ollama(m.role)
    c = m.content
    if isinstance(c, TextContent):
        return {"role": role, "content": c.text}
    if isinstance(c, ToolCallContent):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    # Ollama wants arguments as a JSON object (NOT a
                    # JSON-encoded string like OpenAI).
                    "function": {"name": c.name, "arguments": c.input},
                }
            ],
        }
    if isinstance(c, ToolResultContent):
        return {
            "role": "tool",
            "content": c.content,
            "tool_call_id": c.tool_use_id,
        }
    if isinstance(c, ImageContent):
        # The harness does not currently emit image content into Ollama
        # requests; serialize a textual placeholder rather than introduce
        # a heterogeneous shape.
        return {"role": role, "content": f"[image: {c.media_type}]"}
    raise TypeError(f"unsupported Content variant: {type(c).__name__}")


def build_request_body(
    model_id: str,
    keep_alive: str | None,
    request: ModelRequest,
    stream: bool,
) -> dict[str, Any]:
    """Translate a Spore :class:`ModelRequest` to the Ollama Chat API
    request body."""

    body: dict[str, Any] = {
        "model": model_id,
        "messages": [_message_to_ollama(m) for m in request.messages],
        "stream": stream,
    }
    if keep_alive is not None:
        body["keep_alive"] = keep_alive

    options: dict[str, Any] = {}
    if request.params.max_tokens is not None:
        options["num_predict"] = request.params.max_tokens
    if request.params.temperature is not None:
        options["temperature"] = request.params.temperature
    if request.params.top_p is not None:
        options["top_p"] = request.params.top_p
    if request.params.stop_sequences:
        options["stop"] = list(request.params.stop_sequences)
    if options:
        body["options"] = options

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
    return body


def parse_stop_reason(raw: str | None) -> StopReason:
    if raw == "tool_calls":
        return StopReason.TOOL_USE
    if raw == "length":
        return StopReason.MAX_TOKENS
    if raw == "stop":
        return StopReason.END_TURN
    return StopReason.END_TURN


def parse_response_body(body: dict[str, Any]) -> ModelResponse:
    """Translate an Ollama Chat API response body into a Spore
    :class:`ModelResponse`."""

    message = body.get("message") or {}
    content: list[ContentBlock] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append(TextBlock(text=text))
    tool_calls = message.get("tool_calls") or []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        args = func.get("arguments")
        if not isinstance(args, dict):
            args = {}
        content.append(
            ToolUseBlock(
                # Ollama doesn't return tool-call ids; synthesize so
                # ToolResult.tool_use_id round-trips work.
                id=f"call-{i}",
                name=func.get("name", ""),
                input=args,
            )
        )

    usage = TokenUsage(
        input_tokens=int(body.get("prompt_eval_count") or 0),
        output_tokens=int(body.get("eval_count") or 0),
        # Ollama has no prefix-cache concept; cache fields always None.
        cache_read_tokens=None,
        cache_write_tokens=None,
    )
    return ModelResponse(
        content=content,
        usage=usage,
        stop_reason=parse_stop_reason(body.get("done_reason")),
    )


def _extract_error_message(body_text: str) -> str:
    try:
        data = json.loads(body_text)
    except (ValueError, TypeError):
        return body_text[:500]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, str):
            return err
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg
    return body_text[:500]


def _map_status_error(status: int, body_text: str, model_id: str) -> Exception:
    if status == 404:
        return ProviderError(
            code=404,
            message=f"Model {model_id} not found. Run: ollama pull {model_id}",
        )
    if status in (408, 504):
        return _ModelTimeoutError()
    return ProviderError(code=status, message=_extract_error_message(body_text))


def _concat_request_text(request: ModelRequest) -> str:
    out: list[str] = []
    for m in request.messages:
        c = m.content
        if isinstance(c, TextContent):
            out.append(c.text)
        elif isinstance(c, ToolCallContent):
            out.append(c.name)
            out.append(" ")
            out.append(json.dumps(c.input, separators=(",", ":")))
        elif isinstance(c, ToolResultContent):
            out.append(c.content)
        out.append("\n")
    return "".join(out)


# ============================================================================
# NDJSON parsing — Ollama chat streaming
# ============================================================================


async def _ndjson_to_events(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    """Convert an Ollama NDJSON stream into a stream of :class:`StreamEvent`.

    Ollama streams chat results as newline-delimited JSON (one full JSON
    object per line, NOT SSE). Each line carries an incremental
    ``message.content`` delta; ``tool_calls`` arrive as full argument
    objects per chunk (not partial-fragment strings); the terminator line
    carries ``done: true`` plus ``prompt_eval_count`` and ``eval_count``.
    """

    buf = ""
    started = False
    tool_indices_seen: set[int] = set()
    content_index = 0
    content_open = False
    try:
        async for chunk in response.aiter_text():
            buf += chunk
            while True:
                idx = buf.find("\n")
                if idx < 0:
                    break
                raw_line = buf[:idx]
                buf = buf[idx + 1 :]
                line = raw_line.rstrip("\r").strip()
                if not line:
                    continue
                try:
                    value: Any = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(value, dict):
                    continue
                if not started:
                    started = True
                    yield _MessageStart()
                message = value.get("message") or {}
                if isinstance(message, dict):
                    text = message.get("content")
                    if isinstance(text, str) and text:
                        content_open = True
                        yield _ContentBlockDelta(index=content_index, delta=text)
                    tcs = message.get("tool_calls")
                    if isinstance(tcs, list):
                        for i, tc in enumerate(tcs):
                            if not isinstance(tc, dict):
                                continue
                            event_index = i + 1
                            if event_index not in tool_indices_seen:
                                tool_indices_seen.add(event_index)
                                if content_open:
                                    yield _ContentBlockStop(index=content_index)
                                    content_open = False
                                    content_index = event_index
                            func = tc.get("function")
                            if isinstance(func, dict):
                                args = func.get("arguments")
                                # Ollama emits the FULL arguments object
                                # per chunk; serialize the whole thing as
                                # a single partial_json fragment.
                                partial = json.dumps(
                                    args if args is not None else {},
                                    separators=(",", ":"),
                                )
                                yield _ToolUseDelta(index=event_index, partial_json=partial)
                if value.get("done") is True:
                    yield _MessageStop(
                        usage=TokenUsage(
                            input_tokens=int(value.get("prompt_eval_count") or 0),
                            output_tokens=int(value.get("eval_count") or 0),
                            cache_read_tokens=None,
                            cache_write_tokens=None,
                        ),
                        stop_reason=parse_stop_reason(value.get("done_reason")),
                    )
                    return
        # Stream ended without done:true — still emit MessageStop so
        # consumers see a terminator.
        yield _MessageStop(
            usage=TokenUsage(
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=None,
                cache_write_tokens=None,
            ),
            stop_reason=StopReason.END_TURN,
        )
    finally:
        await response.aclose()


# ============================================================================
# OllamaModelInterface
# ============================================================================


class OllamaModelInterface:
    """Reference Ollama client.

    Construct with a model id; callers may override ``base_url`` (for
    remote Ollama instances or non-default ports), tune ``timeout`` and
    ``keep_alive``, or inject an :class:`httpx.AsyncClient` to share
    connection pools across multiple interfaces — tests pass a
    transport-mocked client.

    Model availability is checked lazily on the first ``call`` /
    ``call_streaming``; the result is cached for the instance's lifetime.
    """

    DEFAULT_BASE_URL = DEFAULT_BASE_URL
    DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
    DEFAULT_KEEP_ALIVE = DEFAULT_KEEP_ALIVE

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        keep_alive: str | None = DEFAULT_KEEP_ALIVE,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._http_client = http_client
        self._owns_client = http_client is None
        self._model_checked = False
        # Set once the availability + discovery probe has run. Holds the
        # ``/api/show``-discovered metadata (empty when discovery failed but
        # availability succeeded). ``None`` until the probe completes.
        self._model_meta: ModelMeta | None = None

    def __repr__(self) -> str:  # pragma: no cover — trivial
        return (
            "OllamaModelInterface("
            f"model_id={self._model_id!r}, "
            f"base_url={self._base_url!r}, "
            f"timeout={self._timeout!r}, "
            f"keep_alive={self._keep_alive!r})"
        )

    @classmethod
    def with_base_url(cls, model_id: str, base_url: str, **kwargs: Any) -> OllamaModelInterface:
        """Convenience constructor mirroring the Rust API."""

        return cls(model_id, base_url=base_url, **kwargs)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def keep_alive(self) -> str | None:
        return self._keep_alive

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
            self._owns_client = True
        return self._http_client

    async def aclose(self) -> None:
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _transport_error(self, exc: Exception) -> Exception:
        if isinstance(exc, httpx.TimeoutException):
            return _ModelTimeoutError()
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return ProviderError(code=0, message=f"Ollama not running at {self._base_url}")
        if isinstance(exc, httpx.HTTPError):
            # NOTE: httpx.RequestError covers DNS / network issues
            # similar to Rust's `is_request` branch; surface them as a
            # helpful "not running" message rather than a generic
            # transport error.
            if isinstance(exc, httpx.RequestError):
                return ProviderError(code=0, message=f"Ollama not running at {self._base_url}")
            return ProviderError(code=0, message=f"HTTP transport error: {exc}")
        return exc

    async def _ensure_model_available(self) -> None:
        if self._model_checked:
            return
        client = self._client()
        url = f"{self._base_url}/api/tags"
        try:
            resp = await client.get(url, timeout=self._timeout)
        except Exception as e:  # noqa: BLE001 — re-raised as typed error
            raise self._transport_error(e) from e
        try:
            if resp.status_code < 200 or resp.status_code >= 300:
                body_text = resp.text
                raise _map_status_error(resp.status_code, body_text, self._model_id)
            try:
                data = resp.json()
            except ValueError as e:
                raise ProviderError(code=0, message=f"tags decode failed: {e}") from e
        finally:
            await resp.aclose()
        models = data.get("models") if isinstance(data, dict) else None
        found = False
        if isinstance(models, list):
            for m in models:
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                if isinstance(name, str) and _name_matches(name, self._model_id):
                    found = True
                    break
        if not found:
            raise ProviderError(
                code=404,
                message=(f"Model {self._model_id} not found. Run: ollama pull {self._model_id}"),
            )
        # Best-effort discovery — never fails the call.
        self._model_meta = await self._discover_meta()
        self._model_checked = True

    async def _discover_meta(self) -> ModelMeta:
        """Best-effort ``POST /api/show`` discovery.

        Returns an empty :class:`ModelMeta` on any failure (404, transport
        error, decode error, missing fields) so discovery being unavailable
        never errors the whole call. Reads the model's ``*.context_length``
        from ``model_info`` and the top-level ``capabilities`` array.
        """

        url = f"{self._base_url}/api/show"
        payload = json.dumps({"model": self._model_id}).encode("utf-8")
        client = self._client()
        try:
            resp = await client.post(
                url,
                content=payload,
                headers={"content-type": "application/json"},
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001 — discovery is best-effort
            return ModelMeta()
        try:
            if resp.status_code < 200 or resp.status_code >= 300:
                return ModelMeta()
            try:
                data = resp.json()
            except ValueError:
                return ModelMeta()
        finally:
            await resp.aclose()
        if not isinstance(data, dict):
            return ModelMeta()

        context_length: int | None = None
        model_info = data.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if key.endswith(".context_length") and isinstance(value, int):
                    context_length = value
                    break

        capabilities_raw = data.get("capabilities")
        capabilities = (
            [c for c in capabilities_raw if isinstance(c, str)]
            if isinstance(capabilities_raw, list)
            else []
        )
        return ModelMeta(context_length=context_length, capabilities=capabilities)

    def _guard_tool_support(self, request: ModelRequest) -> None:
        """Reject tool-bearing requests when the model does not support tools.

        Capability is determined solely by the ``/api/show`` ``capabilities``
        array: the model is tool-capable iff ``capabilities`` contains
        ``"tools"``. Empty or unavailable ``/api/show`` metadata ⟹ NOT
        tool-capable (fail closed). Called after the availability + discovery
        probe.
        """

        if not request.tools:
            return
        meta = self._model_meta
        supported = meta.supports_tools() if meta is not None else False
        if not supported:
            raise ProviderError(
                code=0,
                message=f"Model {self._model_id} does not support tool calling",
            )

    # ── ModelInterface protocol ─────────────────────────────────────────

    async def call(self, request: ModelRequest) -> ModelResponse:
        await self._ensure_model_available()
        self._guard_tool_support(request)
        url = f"{self._base_url}/api/chat"
        body = build_request_body(self._model_id, self._keep_alive, request, stream=False)
        payload = json.dumps(body).encode("utf-8")
        client = self._client()
        try:
            resp = await client.post(
                url,
                content=payload,
                headers={"content-type": "application/json"},
                timeout=self._timeout,
            )
        except Exception as e:  # noqa: BLE001 — re-raised as typed error
            raise self._transport_error(e) from e
        try:
            if resp.status_code < 200 or resp.status_code >= 300:
                body_text = resp.text
                raise _map_status_error(resp.status_code, body_text, self._model_id)
            try:
                data = resp.json()
            except ValueError as e:
                raise ProviderError(code=0, message=f"response decode failed: {e}") from e
        finally:
            await resp.aclose()
        return parse_response_body(data)

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        await self._ensure_model_available()
        self._guard_tool_support(request)
        url = f"{self._base_url}/api/chat"
        body = build_request_body(self._model_id, self._keep_alive, request, stream=True)
        payload = json.dumps(body).encode("utf-8")
        client = self._client()
        try:
            req = client.build_request(
                "POST",
                url,
                content=payload,
                headers={"content-type": "application/json"},
                timeout=self._timeout,
            )
            resp = await client.send(req, stream=True)
        except Exception as e:  # noqa: BLE001 — re-raised as typed error
            raise self._transport_error(e) from e
        if resp.status_code < 200 or resp.status_code >= 300:
            body_text = await resp.aread()
            await resp.aclose()
            raise _map_status_error(
                resp.status_code,
                body_text.decode("utf-8", "replace"),
                self._model_id,
            )
        async for event in _ndjson_to_events(resp):
            yield event

    async def count_tokens(self, request: ModelRequest) -> int:
        """Estimate token count via Ollama's ``/api/embed`` endpoint.

        Falls back to the ``bytes/4`` heuristic if the embed endpoint is
        unavailable. Ollama has no dedicated token-counting endpoint;
        ``prompt_eval_count`` from ``/api/embed`` is the closest proxy.
        """

        text = _concat_request_text(request)
        n = await self._try_embed_count(text)
        if n is not None:
            return n
        return len(text) // 4

    async def _try_embed_count(self, text: str) -> int | None:
        url = f"{self._base_url}/api/embed"
        body = {"model": self._model_id, "input": text}
        payload = json.dumps(body).encode("utf-8")
        client = self._client()
        try:
            resp = await client.post(
                url,
                content=payload,
                headers={"content-type": "application/json"},
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001 — embed is best-effort
            return None
        try:
            if resp.status_code < 200 or resp.status_code >= 300:
                return None
            try:
                data = resp.json()
            except ValueError:
                return None
        finally:
            await resp.aclose()
        if not isinstance(data, dict):
            return None
        v = data.get("prompt_eval_count")
        if isinstance(v, int):
            return v
        return None

    def provider(self) -> ProviderInfo:
        # Prefer the ``/api/show``-discovered context length when the probe has
        # already run and produced one; otherwise fall back to the static
        # table. ``provider()`` is synchronous, so it reads the cached probe
        # result non-blockingly rather than triggering discovery itself.
        discovered = self._model_meta.context_length if self._model_meta is not None else None
        window = discovered if discovered is not None else context_window(self._model_id)
        return ProviderInfo(
            name="ollama",
            model_id=self._model_id,
            context_window=window,
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_KEEP_ALIVE",
    "DEFAULT_TIMEOUT_SECONDS",
    "ModelMeta",
    "OllamaModelInterface",
    "build_request_body",
    "context_window",
    "parse_response_body",
    "parse_stop_reason",
]
