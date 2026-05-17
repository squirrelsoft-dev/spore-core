"""ContextManager — assemble and maintain the context window (issue #7).

Mirrors the Rust reference at ``rust/crates/spore-core/src/context.rs``.

Builds per-turn context from a pre-computed Block-1 :class:`ComposedPrompt`,
per-session metadata (Block 2), and per-turn ephemera (Block 3). Tracks
token usage, compacts on threshold, offloads large tool results, and
injects just-in-time skill chunks.

See ``docs/harness-engineering-concepts.md`` § "ContextManager" and
§ "Cache Architecture" for the cross-language rules this module enforces.

The :class:`ContextManager` protocol defined here is the canonical
interface for issue #7. The narrower :class:`ContextManager` stub in
:mod:`spore_core.harness` is a placeholder used by the in-tree
``StandardHarness`` while the wider rewrite lands.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, NewType, Protocol, runtime_checkable

from .errors import SporeError

# ``MemoryItem`` is defined in :mod:`spore_core.memory` (issue #8).
# Re-exported here so ``ContextSources.memory`` consumes the canonical type.
from .memory import MemoryItem
from .harness import (
    FileRef,
    HarnessToolResult,
    SandboxProvider,
    SessionId,
    TaskId,
    ToolOutputError,
    ToolOutputSuccess,
    ToolOutputWaitingForHuman,
)
from .model import (
    Message,
    ModelInterface,
    ModelParams,
    ModelRequest,
    Role,
    TextContent,
    ToolSchema,
)
from .tool_registry import TaskPhase

logger = logging.getLogger(__name__)


# ============================================================================
# Forward-declared sibling types (issues #8, #9, #14)
# ============================================================================


GuideId = NewType("GuideId", str)


def new_guide_id(s: str) -> GuideId:
    return GuideId(s)


@dataclass(frozen=True)
class Guide:
    """Forward-declared ``Guide`` (issue #9). Carries the rendered chunk
    and an identifier; full lifecycle metadata lives with
    ``GuideRegistry``. ``content`` is the rendered final form — the spec
    forbids reformatting at assembly time."""

    id: GuideId
    content: str


@dataclass(frozen=True)
class ComposedPrompt:
    """Forward-declared ``ComposedPrompt`` (issue #14 — PromptChunkRegistry).

    Block 1 is computed ONCE at harness startup. ``rendered`` is the final
    byte-for-byte content; ``block_1_hash`` is a stable digest used by the
    :class:`ContextManager` to detect unexpected cache invalidation.
    """

    rendered: str
    block_1_hash: int


@dataclass(frozen=True)
class CacheStats:
    """Forward-declared cache stats parsed by a :class:`CacheProvider`."""

    static_hit: bool | None = None
    session_hit: bool | None = None
    history_hit: bool | None = None


@runtime_checkable
class CacheProvider(Protocol):
    """Forward-declared ``CacheProvider`` protocol (issue #7 dependency).

    The default :class:`NullCacheProvider` is the testing default — it
    never interferes.
    """

    def supports_caching(self) -> bool: ...

    def annotate(self, context: Context) -> None: ...


class NullCacheProvider:
    """Testing default — no-op for all calls. Never interferes with unit
    tests."""

    def supports_caching(self) -> bool:
        return False

    def annotate(self, context: Context) -> None:  # noqa: ARG002
        return None


# ============================================================================
# Spec-defined types
# ============================================================================


class SegmentStability(str, Enum):
    STATIC = "static"
    PER_SESSION = "per_session"
    PER_TURN = "per_turn"


@dataclass
class PromptSegment:
    name: str
    content: str
    stability: SegmentStability
    cache_breakpoint: bool = False


@dataclass
class BreakpointInfo:
    after_segment: str
    token_offset: int


@dataclass
class RenderedSystemPrompt:
    content: str
    breakpoints: list[BreakpointInfo]
    static_block_hash: int
    session_block_hash: int


@dataclass
class CacheBlockStatus:
    static_hit: bool | None = None
    session_hit: bool | None = None
    history_hit: bool | None = None


@dataclass
class ContextMeta:
    session_id: SessionId
    turn_number: int
    active_phase: TaskPhase
    guides_loaded: list[GuideId] = field(default_factory=list)
    skills_injected: list[GuideId] = field(default_factory=list)
    compacted: bool = False
    cache_blocks: CacheBlockStatus = field(default_factory=CacheBlockStatus)


@dataclass
class Context:
    """Assembled per-turn context.

    Distinct from :class:`spore_core.agent.Context`, which is the narrower
    bundle the agent treats as immutable input. :meth:`into_request`
    converts to a :class:`ModelRequest` for the agent layer.
    """

    system_prompt: RenderedSystemPrompt
    messages: list[Message]
    tool_schemas: list[ToolSchema]
    token_count: int
    window_limit: int
    utilization: float
    meta: ContextMeta

    def into_request(self, params: ModelParams | None = None) -> ModelRequest:
        msgs: list[Message] = [
            Message(role=Role.SYSTEM, content=TextContent(text=self.system_prompt.content)),
            *self.messages,
        ]
        return ModelRequest(
            messages=msgs,
            tools=list(self.tool_schemas),
            params=params or ModelParams(),
            stream=False,
        )


@dataclass
class CompactionConfig:
    threshold: float = 0.80
    preserve_recent_n: int = 8
    head_tail_tokens: int = 512
    offload_path: Path = field(default_factory=lambda: Path(".spore/offload"))


@dataclass
class SessionState:
    """Spec-shaped session state for the ContextManager.

    Distinct from :class:`spore_core.harness.SessionState`, which is the
    opaque round-trip envelope owned by the harness. Issue #7 owns this
    richer view; later issues will reconcile.
    """

    session_id: SessionId
    task_id: TaskId
    task_instruction: str
    turn_number: int = 0
    environment: str = ""
    prior_state: str = ""
    operational_instructions: str = ""
    active_phase: TaskPhase = TaskPhase.EXECUTION
    message_history: list[Message] = field(default_factory=list)
    token_budget_used: int = 0
    window_limit: int = 200_000
    guides_loaded: list[GuideId] = field(default_factory=list)
    # Skills pending Block-3 injection on the next assemble. Cleared after
    # each assemble — skills are ephemeral, one turn only.
    pending_skill_injections: list[Guide] = field(default_factory=list)
    budget_warning_active: bool = False


@dataclass
class ContextSources:
    guides: list[Guide]
    memory: list[MemoryItem]
    tool_schemas: list[ToolSchema]
    composed_prompt: ComposedPrompt


@dataclass
class CompactionPreserveHints:
    keep_architectural_decisions: bool = True
    keep_open_problems: bool = True
    keep_current_task_state: bool = True
    keep_recent_file_list: bool = True
    # Defaults to ``True`` — never compact active reasoning blocks.
    keep_thinking_blocks: bool = True


@dataclass
class CompactionRequest:
    messages_to_compact: list[Message]
    preserve_hints: CompactionPreserveHints = field(default_factory=CompactionPreserveHints)


@dataclass
class CompactionResult:
    summary_message: Message
    tokens_reclaimed: int
    messages_removed: int


# ============================================================================
# Errors
# ============================================================================


class ContextError(SporeError):
    """Root of every error raised by a :class:`ContextManager`."""

    kind: ClassVar[str] = "ContextError"


class TokenCountFailed(ContextError):
    kind: ClassVar[str] = "TokenCountFailed"

    def __init__(self, message: str = "token count failed") -> None:
        super().__init__(message)


class CompactionFailed(ContextError):
    kind: ClassVar[str] = "CompactionFailed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"compaction failed: {reason}")


class AssemblyFailed(ContextError):
    kind: ClassVar[str] = "AssemblyFailed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"assembly failed: {reason}")


class CacheHashMismatch(ContextError):
    kind: ClassVar[str] = "CacheHashMismatch"

    def __init__(self, block: str, expected: int, actual: int) -> None:
        self.block = block
        self.expected = expected
        self.actual = actual
        super().__init__(f"cache hash mismatch on block {block}: expected {expected}, got {actual}")


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class ContextManager(Protocol):
    """Canonical issue-#7 ``ContextManager`` interface."""

    async def assemble(self, state: SessionState, sources: ContextSources) -> Context: ...

    async def append_tool_result(
        self,
        state: SessionState,
        result: HarnessToolResult,
        sandbox: SandboxProvider,
    ) -> None: ...

    def append_response(self, state: SessionState, response: str) -> None: ...

    def should_compact(self, state: SessionState) -> bool: ...

    def prepare_compaction(self, state: SessionState) -> CompactionRequest: ...

    def apply_compaction(self, state: SessionState, result: CompactionResult) -> None: ...

    def inject_skill(self, context: Context, skill: Guide) -> None: ...

    def record_cache_result(self, context: Context, cache_stats: CacheStats) -> None: ...


# ============================================================================
# Hashing helpers (stable across processes — uses sha256, not Python's hash)
# ============================================================================


def _hash_strings(parts: list[str]) -> int:
    """Stable 64-bit digest over a sequence of strings.

    Python's built-in ``hash`` is salted per-process and would let cache
    invariants drift across restarts. We use sha256 truncated to 64 bits.
    """

    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        # Length-prefixed-style separator to avoid collisions across
        # concatenations like ["a","bc"] vs ["ab","c"].
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def _segments_hash(segments: list[PromptSegment]) -> int:
    parts: list[str] = []
    for s in segments:
        parts.append(s.name)
        parts.append(s.content)
        parts.append(s.stability.value)
        parts.append("1" if s.cache_breakpoint else "0")
    return _hash_strings(parts)


def _render_segments(
    block_1: str, segments: list[PromptSegment]
) -> tuple[str, list[BreakpointInfo]]:
    """Render Block 1 followed by per-session/per-turn segments.

    Rough token offset proxy: chars/4. Real token offsets are reported by
    the model; this is only useful for cache-marker placement which is
    counted in segments, not tokens.
    """

    parts: list[str] = [block_1]
    breakpoints: list[BreakpointInfo] = [
        BreakpointInfo(after_segment="__block_1__", token_offset=len(block_1) // 4)
    ]
    running = len(block_1)
    for seg in segments:
        if parts[-1] and not parts[-1].endswith("\n"):
            parts.append("\n")
            running += 1
        parts.append(seg.content)
        running += len(seg.content)
        if seg.cache_breakpoint:
            breakpoints.append(BreakpointInfo(after_segment=seg.name, token_offset=running // 4))
    return "".join(parts), breakpoints


def _format_truncated(head_tail: str, full_ref: FileRef | None) -> str:
    if full_ref is not None:
        return (
            f"{head_tail}\n\n[truncated; full output at "
            f"{full_ref.path} ({full_ref.byte_len} bytes)]"
        )
    return f"{head_tail}\n\n[truncated]"


# ============================================================================
# StandardContextManager
# ============================================================================


class StandardContextManager:
    """Reference :class:`ContextManager` implementation.

    Enforces the assembly order from the spec: Block 1 from
    :class:`ComposedPrompt`, Block 2 from :class:`SessionState`, Block 3
    from per-turn ephemera. Tool schemas are sorted by name.

    The Block-1 hash is memoized on first assemble; any subsequent change
    raises :class:`CacheHashMismatch`. Mid-session Block-2 hash changes
    log a warning when ``turn_number > 1``.
    """

    DEFAULT_OFFLOAD_THRESHOLD_BYTES: ClassVar[int] = 32 * 1024

    def __init__(
        self,
        model: ModelInterface,
        cache_provider: CacheProvider | None = None,
        compaction: CompactionConfig | None = None,
        *,
        offload_threshold_bytes: int | None = None,
    ) -> None:
        self._model = model
        self._cache_provider: CacheProvider = cache_provider or NullCacheProvider()
        self._compaction = compaction or CompactionConfig()
        self._offload_threshold_bytes = (
            offload_threshold_bytes
            if offload_threshold_bytes is not None
            else self.DEFAULT_OFFLOAD_THRESHOLD_BYTES
        )
        self._lock = threading.Lock()
        self._static_hash: int | None = None
        self._session_hash: int | None = None

    # ---- assemble ---------------------------------------------------

    def _build_session_segments(self, state: SessionState) -> list[PromptSegment]:
        # Order is load-bearing for prefix-cache stability.
        return [
            PromptSegment(
                name="task",
                content=state.task_instruction,
                stability=SegmentStability.PER_SESSION,
            ),
            PromptSegment(
                name="environment",
                content=state.environment,
                stability=SegmentStability.PER_SESSION,
            ),
            PromptSegment(
                name="prior_state",
                content=state.prior_state,
                stability=SegmentStability.PER_SESSION,
            ),
            PromptSegment(
                name="operational",
                content=state.operational_instructions,
                stability=SegmentStability.PER_SESSION,
                cache_breakpoint=True,
            ),
        ]

    async def assemble(self, state: SessionState, sources: ContextSources) -> Context:
        # ── BLOCK 1 hash check ───────────────────────────────────────
        static_hash = sources.composed_prompt.block_1_hash
        with self._lock:
            if self._static_hash is None:
                self._static_hash = static_hash
            elif self._static_hash != static_hash:
                raise CacheHashMismatch(
                    block="static",
                    expected=self._static_hash,
                    actual=static_hash,
                )

        # ── BLOCK 2 (PerSession) ─────────────────────────────────────
        segments = self._build_session_segments(state)
        session_hash = _segments_hash(segments)
        with self._lock:
            prev = self._session_hash
            if prev is not None and prev != session_hash and state.turn_number > 1:
                logger.warning(
                    "session block hash changed mid-session (%s -> %s)",
                    prev,
                    session_hash,
                )
            self._session_hash = session_hash

        # ── BLOCK 3 (PerTurn, never cached) ──────────────────────────
        if state.budget_warning_active:
            segments.append(
                PromptSegment(
                    name="budget_warning",
                    content=(
                        f"[BUDGET] {state.token_budget_used} of {state.window_limit} tokens used."
                    ),
                    stability=SegmentStability.PER_TURN,
                )
            )
        for skill in state.pending_skill_injections:
            segments.append(
                PromptSegment(
                    name=f"skill:{skill.id}",
                    content=skill.content,
                    stability=SegmentStability.PER_TURN,
                )
            )

        # ── Render ───────────────────────────────────────────────────
        rendered, breakpoints = _render_segments(sources.composed_prompt.rendered, segments)
        system_prompt = RenderedSystemPrompt(
            content=rendered,
            breakpoints=breakpoints,
            static_block_hash=static_hash,
            session_block_hash=session_hash,
        )

        # ── Tool schemas: sort by name (spec: deterministic ordering) ─
        tool_schemas = sorted(sources.tool_schemas, key=lambda s: s.name)

        # ── Message history ──────────────────────────────────────────
        messages = list(state.message_history)

        # ── Token count from ModelInterface (not estimated) ──────────
        req = ModelRequest(
            messages=[
                Message(role=Role.SYSTEM, content=TextContent(text=rendered)),
                *messages,
            ],
            tools=list(tool_schemas),
            params=ModelParams(),
            stream=False,
        )
        try:
            token_count = await self._model.count_tokens(req)
        except Exception as e:  # noqa: BLE001 — convert any failure to typed error
            raise TokenCountFailed(str(e)) from e

        utilization = 0.0 if state.window_limit == 0 else token_count / state.window_limit

        meta = ContextMeta(
            session_id=state.session_id,
            turn_number=state.turn_number,
            active_phase=state.active_phase,
            guides_loaded=list(state.guides_loaded),
            skills_injected=[g.id for g in state.pending_skill_injections],
            compacted=False,
            cache_blocks=CacheBlockStatus(),
        )

        context = Context(
            system_prompt=system_prompt,
            messages=messages,
            tool_schemas=tool_schemas,
            token_count=token_count,
            window_limit=state.window_limit,
            utilization=utilization,
            meta=meta,
        )

        # ── Cache annotation ─────────────────────────────────────────
        self._cache_provider.annotate(context)
        return context

    # ---- append_tool_result -----------------------------------------

    async def append_tool_result(
        self,
        state: SessionState,
        result: HarnessToolResult,
        sandbox: SandboxProvider,
    ) -> None:
        output = result.output
        if isinstance(output, ToolOutputSuccess):
            text = output.content
        elif isinstance(output, ToolOutputError):
            text = f"[error] {output.message}"
        elif isinstance(output, ToolOutputWaitingForHuman):
            text = "[waiting]"
        else:  # pragma: no cover — exhaustive
            text = ""

        # Spec rule: head+tail truncate, offload full to filesystem.
        if (
            sys.getsizeof(text) >= self._offload_threshold_bytes
            or len(text.encode("utf-8")) >= self._offload_threshold_bytes
        ):
            truncated = await sandbox.handle_large_output(
                text,
                result.call_id,
                self._compaction.head_tail_tokens,
                self._compaction.head_tail_tokens,
            )
            final_text = _format_truncated(truncated.content, truncated.full_ref)
        else:
            final_text = text

        state.message_history.append(Message(role=Role.TOOL, content=TextContent(text=final_text)))

    # ---- append_response --------------------------------------------

    def append_response(self, state: SessionState, response: str) -> None:
        state.message_history.append(
            Message(role=Role.ASSISTANT, content=TextContent(text=response))
        )

    # ---- compaction --------------------------------------------------

    def should_compact(self, state: SessionState) -> bool:
        if state.window_limit == 0:
            return False
        util = state.token_budget_used / state.window_limit
        return util >= self._compaction.threshold

    def prepare_compaction(self, state: SessionState) -> CompactionRequest:
        n = len(state.message_history)
        keep = self._compaction.preserve_recent_n
        if n <= keep:
            return CompactionRequest(
                messages_to_compact=[],
                preserve_hints=CompactionPreserveHints(),
            )
        cut = n - keep
        return CompactionRequest(
            messages_to_compact=list(state.message_history[:cut]),
            preserve_hints=CompactionPreserveHints(),
        )

    def apply_compaction(self, state: SessionState, result: CompactionResult) -> None:
        n = len(state.message_history)
        keep = self._compaction.preserve_recent_n
        if n <= keep:
            raise CompactionFailed("history shorter than preserve_recent_n")
        cut = n - keep
        new_history: list[Message] = [result.summary_message]
        new_history.extend(state.message_history[cut:])
        state.message_history = new_history
        state.token_budget_used = max(0, state.token_budget_used - result.tokens_reclaimed)

    # ---- inject_skill ------------------------------------------------

    def inject_skill(self, context: Context, skill: Guide) -> None:
        # Block-3 ephemeral injection: append to system prompt content, do
        # not modify message history, do not invalidate Block 1 or Block 2
        # (their hashes are untouched).
        suffix = f"[SKILL:{skill.id}]\n{skill.content}"
        if context.system_prompt.content and not context.system_prompt.content.endswith("\n"):
            context.system_prompt.content += "\n"
        context.system_prompt.content += suffix
        context.meta.skills_injected.append(skill.id)

    # ---- record_cache_result ----------------------------------------

    def record_cache_result(self, context: Context, cache_stats: CacheStats) -> None:
        context.meta.cache_blocks = CacheBlockStatus(
            static_hit=cache_stats.static_hit,
            session_hit=cache_stats.session_hit,
            history_hit=cache_stats.history_hit,
        )


__all__ = [
    "AssemblyFailed",
    "BreakpointInfo",
    "CacheBlockStatus",
    "CacheHashMismatch",
    "CacheProvider",
    "CacheStats",
    "CompactionConfig",
    "CompactionFailed",
    "CompactionPreserveHints",
    "CompactionRequest",
    "CompactionResult",
    "ComposedPrompt",
    "Context",
    "ContextError",
    "ContextManager",
    "ContextMeta",
    "ContextSources",
    "Guide",
    "GuideId",
    "MemoryItem",
    "NullCacheProvider",
    "PromptSegment",
    "RenderedSystemPrompt",
    "SegmentStability",
    "SessionState",
    "StandardContextManager",
    "TokenCountFailed",
    "new_guide_id",
]
