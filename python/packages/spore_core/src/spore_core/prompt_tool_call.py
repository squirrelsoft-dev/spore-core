"""Prompt-based tool calling — an adaptive fallback for models that do not
reliably emit native tool calls.

Mirrors the Rust reference at ``rust/crates/spore-core/src/prompt_tool_call.rs``
(and the prose-detection seam in ``tool_call_repair.rs``). The wire constants —
the injected ``<available_tools>`` block, the ``<tool_call>`` marker grammar, and
the action-intent phrase list — are byte-identical so behaviour matches across
all four language implementations.

Some models (small local ones especially) respond in prose even when a tool call
is the right action. Rather than maintaining a list of known-bad models or asking
callers to wrap them manually, the harness discovers this at runtime: native tool
calling is tried first, and when a turn comes back as prose while tools were
advertised (see :func:`detect_prose_response`), the harness flips a session-scoped
flag that activates :class:`PromptBasedToolCallModelInterface` for the rest of the
run.

Two wrappers
============

- :class:`PromptBasedToolCallModelInterface` — an *always-on* transparent
  wrapper. It injects a tool-definition block into the system prompt and parses
  ``<tool_call>`` markers out of the model's text response into native tool-use
  blocks. Construct it directly for advanced use.
- :class:`AdaptiveToolCallModelInterface` — a *flag-gated* wrapper installed
  automatically by :meth:`HarnessBuilder.conversational`. While its shared flag
  is unset it delegates natively (byte-for-byte); once the harness sets the flag
  it behaves exactly like the always-on wrapper.

Both share the free functions :func:`inject_tool_prompt` and
:func:`parse_prose_response` so injection and parsing can never diverge between
them. Injection is idempotent — double-wrapping never appends the block twice.

Streaming buffers the full inner stream, parses it for markers, then re-emits the
reconstructed response as a stream. Streaming and marker parsing do not compose
cleanly; buffering is the accepted trade-off.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .model import (
    ContentBlock,
    MessageStart,
    MessageStop,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    Role,
    StopReason,
    StreamEvent,
    TextBlock,
    TextContent,
    ThinkingBlock,
    TokenUsage,
    ToolUseBlock,
)
from .model import (
    Message as ModelMessage,
)

# Sentinel that marks an already-injected tool-prompt block. Used to make
# :func:`inject_tool_prompt` idempotent.
_TOOLS_BLOCK_OPEN = "<available_tools>"

# Curated action-intent phrases. Lower-cased substring match — conservative and
# cheap. Each phrase strongly implies "I am about to use a tool". Byte-identical
# to the Rust ``ACTION_PHRASES`` list.
ACTION_PHRASES: tuple[str, ...] = (
    "i'll use",
    "i will use",
    "i'll call",
    "i will call",
    "i'll run",
    "i will run",
    "let me use",
    "let me call",
    "let me run",
    "i need to use",
    "i need to call",
    "i should use",
    "i should call",
    "i'll invoke",
    "i will invoke",
    "using the",
    "i can use the",
    "i'm going to use",
    "i am going to use",
)

# ``<tool_call><name>..</name><input>{json}</input></tool_call>`` — non-greedy,
# dot-matches-newline. Mirrors the Rust ``(?s)`` regex.
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<name>(.*?)</name>\s*<input>(.*?)</input>\s*</tool_call>",
    re.DOTALL,
)


# ============================================================================
# System-prompt injection
# ============================================================================


def build_tool_prompt(tools: list) -> str:
    """Render the tool-definition + response-format block appended to the system
    prompt when prompt-based tool calling is active.

    Schemas are rendered as compact JSON (no insignificant whitespace) so the
    output is byte-identical to the Rust ``build_tool_prompt``.
    """
    parts: list[str] = []
    parts.append(
        "You have access to the following tools. Use them when they would help "
        "complete the task.\n\n"
    )
    parts.append("<available_tools>\n")
    for tool in tools:
        schema_json = json.dumps(tool.input_schema, separators=(",", ":"))
        parts.append("<tool>\n")
        parts.append(f"  <name>{tool.name}</name>\n")
        parts.append(f"  <description>{tool.description}</description>\n")
        parts.append(f"  <input_schema>{schema_json}</input_schema>\n")
        parts.append("</tool>\n")
    parts.append("</available_tools>\n\n")
    parts.append(
        "When you want to use a tool, respond with ONLY the following format and nothing else:\n"
    )
    parts.append(
        '<tool_call>\n  <name>tool_name_here</name>\n  <input>{"key": "value"}</input>\n'
        "</tool_call>\n\n"
    )
    parts.append(
        "When you have a final answer that does not require a tool, respond normally in prose."
    )
    return "".join(parts)


def inject_tool_prompt(request: ModelRequest) -> None:
    """Append the tool-definition block to a request's system prompt, in place.

    - No-op when the request advertises no tools (nothing to describe).
    - Idempotent: if a leading system message already contains the
      ``<available_tools>`` sentinel, nothing is appended (so wrapping a wrapper
      does not double-inject).
    - Appends to an existing leading :attr:`Role.SYSTEM` text message when
      present, otherwise inserts a new one at the front — never clobbering the
      caller's system prompt.
    """
    if not request.tools:
        return
    block = build_tool_prompt(request.tools)

    first = request.messages[0] if request.messages else None
    if first is not None and first.role == Role.SYSTEM and isinstance(first.content, TextContent):
        if _TOOLS_BLOCK_OPEN in first.content.text:
            return  # already injected — idempotent.
        first.content.text = first.content.text + "\n\n" + block
        return

    request.messages.insert(
        0,
        ModelMessage(role=Role.SYSTEM, content=TextContent(text=block)),
    )


# ============================================================================
# Response parsing
# ============================================================================


def parse_prose_response(response: ModelResponse) -> ModelResponse:
    """Rewrite a model response so ``<tool_call>`` markers in its text become
    native :class:`ToolUseBlock` blocks.

    - If the response already carries native tool-use blocks, it is returned
      unchanged (native tool calling succeeded — never second-guess it).
    - Otherwise text blocks are scanned for markers. When at least one parses,
      the response's content becomes any thinking blocks (preserved in order)
      followed by the synthesized tool-use blocks, and ``stop_reason`` becomes
      :attr:`StopReason.TOOL_USE`.
    - When no marker parses, the response is returned unchanged (prose as-is).
    """
    if any(isinstance(b, ToolUseBlock) for b in response.content):
        return response

    text = "".join(b.text for b in response.content if isinstance(b, TextBlock))

    parsed: list[tuple[str, dict]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        raw = m.group(2).strip()
        if not name:
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            # Malformed input JSON: skip this marker. If no marker parses, the
            # whole response falls through as prose (graceful degradation).
            continue
        parsed.append((name, value))

    if not parsed:
        return response  # no tool markers — genuine prose response.

    # Preserve reasoning, replace text with synthesized tool-use blocks.
    content: list[ContentBlock] = [b for b in response.content if isinstance(b, ThinkingBlock)]
    for i, (name, value) in enumerate(parsed):
        content.append(ToolUseBlock(id=f"ptc_call_{i}", name=name, input=value))

    return ModelResponse(
        content=content,
        usage=response.usage,
        stop_reason=StopReason.TOOL_USE,
    )


# ============================================================================
# Stream buffering helpers
# ============================================================================


async def _buffer_stream(stream: AsyncIterator[StreamEvent]) -> ModelResponse:
    """Reassemble a :class:`ModelResponse` from a buffered stream of events.

    Mirrors the agent's accumulator, kept local so this module owns its own
    buffering for the parse-then-re-emit path.
    """
    blocks: list[list] = []  # ordered [index, kind, payload-dict]
    usage = TokenUsage()
    stop_reason: StopReason | None = None

    def entry(index: int, kind: str) -> dict:
        for b in blocks:
            if b[0] == index:
                return b[2]
        payload: dict = {}
        blocks.append([index, kind, payload])
        return payload

    async for event in stream:
        if isinstance(event, MessageStart):
            continue
        from .model import (
            ContentBlockDelta,
            ContentBlockStop,
            ThinkingDelta,
            ToolUseDelta,
            ToolUseStart,
        )

        if isinstance(event, ContentBlockDelta):
            p = entry(event.index, "text")
            p["text"] = p.get("text", "") + event.delta
        elif isinstance(event, ThinkingDelta):
            p = entry(event.index, "thinking")
            p["text"] = p.get("text", "") + event.delta
        elif isinstance(event, ToolUseStart):
            p = entry(event.index, "tool")
            p["id"] = event.id
            p["name"] = event.name
            p.setdefault("json", "")
        elif isinstance(event, ToolUseDelta):
            p = entry(event.index, "tool")
            p["json"] = p.get("json", "") + event.partial_json
        elif isinstance(event, ContentBlockStop):
            continue
        elif isinstance(event, MessageStop):
            usage = event.usage
            stop_reason = event.stop_reason

    content: list[ContentBlock] = []
    for index, kind, payload in blocks:
        if kind == "text":
            content.append(TextBlock(text=payload.get("text", "")))
        elif kind == "thinking":
            content.append(ThinkingBlock(text=payload.get("text", "")))
        elif kind == "tool":
            raw = payload.get("json", "")
            try:
                value = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                value = {}
            if not isinstance(value, dict):
                value = {}
            tid = payload.get("id") or f"call_{index}"
            content.append(ToolUseBlock(id=tid, name=payload.get("name", ""), input=value))

    return ModelResponse(
        content=content,
        usage=usage,
        stop_reason=stop_reason if stop_reason is not None else StopReason.END_TURN,
    )


async def _response_to_stream(response: ModelResponse) -> AsyncIterator[StreamEvent]:
    """Re-emit a :class:`ModelResponse` as a stream the harness accumulator
    understands. The inverse of :func:`_buffer_stream`."""
    from .model import (
        ContentBlockDelta,
        ContentBlockStop,
        ThinkingDelta,
        ToolUseDelta,
        ToolUseStart,
    )

    yield MessageStart()
    for index, block in enumerate(response.content):
        if isinstance(block, TextBlock):
            yield ContentBlockDelta(index=index, delta=block.text)
        elif isinstance(block, ThinkingBlock):
            yield ThinkingDelta(index=index, delta=block.text)
        elif isinstance(block, ToolUseBlock):
            yield ToolUseStart(index=index, id=block.id, name=block.name)
            yield ToolUseDelta(
                index=index,
                partial_json=json.dumps(block.input, separators=(",", ":")),
            )
        yield ContentBlockStop(index=index)
    yield MessageStop(usage=response.usage, stop_reason=response.stop_reason)


async def _streaming_prompt_call(
    inner: ModelInterface, request: ModelRequest
) -> AsyncIterator[StreamEvent]:
    """Shared streaming path: inject, buffer the inner stream, parse, re-emit."""
    inject_tool_prompt(request)
    buffered = await _buffer_stream(inner.call_streaming(request))
    parsed = parse_prose_response(buffered)
    async for event in _response_to_stream(parsed):
        yield event


# ============================================================================
# Prose detection (harness escalation trigger)
# ============================================================================


def detect_prose_response(text: str, tools_advertised: bool) -> str | None:
    """Conservative heuristic: did the model respond in prose when a tool call
    was the expected next step?

    Returns the trimmed prose text only when **both**:

    1. tools were advertised this turn (``tools_advertised``), and
    2. the response text contains an explicit action-intent phrase suggesting the
       model *meant* to act (e.g. "I'll use the … tool", "let me call …").

    The bias is deliberately toward false negatives: a missed prose response
    costs one extra turn, but a false positive activates prompt-based mode for a
    model that was simply giving a final answer. A bare final answer with no
    action-intent language is therefore **not** classified as a prose response.
    """
    if not tools_advertised:
        return None
    trimmed = text.strip()
    if not trimmed:
        return None
    lower = trimmed.lower()
    if any(phrase in lower for phrase in ACTION_PHRASES):
        return trimmed
    return None


# ============================================================================
# PromptBasedToolCallModelInterface — always-on wrapper
# ============================================================================


class PromptBasedToolCallModelInterface:
    """Transparent, *always-on* prompt-based tool-calling wrapper around any
    :class:`ModelInterface`.

    Every call injects the tool-definition block into the system prompt and
    parses ``<tool_call>`` markers from the response into native tool-use blocks.
    ``count_tokens`` and ``provider`` delegate to the inner model unchanged.
    """

    def __init__(self, inner: ModelInterface) -> None:
        self._inner = inner

    @property
    def inner(self) -> ModelInterface:
        return self._inner

    async def call(self, request: ModelRequest) -> ModelResponse:
        inject_tool_prompt(request)
        response = await self._inner.call(request)
        return parse_prose_response(response)

    def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        return _streaming_prompt_call(self._inner, request)

    async def count_tokens(self, request: ModelRequest) -> int:
        return await self._inner.count_tokens(request)

    def provider(self) -> ProviderInfo:
        return self._inner.provider()


# ============================================================================
# AdaptiveToolCallModelInterface — flag-gated wrapper
# ============================================================================


@dataclass
class PromptToolCallFlag:
    """Shared mutable boolean captured by reference by both the harness run loop
    and :class:`AdaptiveToolCallModelInterface`.

    The Python analogue of Rust's ``Arc<AtomicBool>``: a tiny mutable holder so
    the harness can flip ``value`` from the run loop and the wrapper observes it
    on the next call. Single-threaded asyncio means no atomics are needed.
    """

    value: bool = False


class AdaptiveToolCallModelInterface:
    """Flag-gated prompt-based wrapper. While ``flag.value`` is ``False`` it
    delegates to the inner model byte-for-byte (native tool calling). Once the
    harness sets the flag — on detecting a prose response where a tool call was
    expected — it behaves exactly like :class:`PromptBasedToolCallModelInterface`
    for the rest of the run.

    Installed automatically by :meth:`HarnessBuilder.conversational`; the harness
    holds the same :class:`PromptToolCallFlag` and flips it from the run loop.
    Not normally constructed directly.
    """

    def __init__(self, inner: ModelInterface, flag: PromptToolCallFlag) -> None:
        self._inner = inner
        self._flag = flag

    @property
    def is_active(self) -> bool:
        """``True`` once prompt-based mode has been activated for the run."""
        return self._flag.value

    async def call(self, request: ModelRequest) -> ModelResponse:
        if not self._flag.value:
            return await self._inner.call(request)
        inject_tool_prompt(request)
        response = await self._inner.call(request)
        return parse_prose_response(response)

    def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if not self._flag.value:
            return self._inner.call_streaming(request)
        return _streaming_prompt_call(self._inner, request)

    async def count_tokens(self, request: ModelRequest) -> int:
        return await self._inner.count_tokens(request)

    def provider(self) -> ProviderInfo:
        return self._inner.provider()


__all__ = [
    "ACTION_PHRASES",
    "AdaptiveToolCallModelInterface",
    "PromptBasedToolCallModelInterface",
    "PromptToolCallFlag",
    "build_tool_prompt",
    "detect_prose_response",
    "inject_tool_prompt",
    "parse_prose_response",
]
