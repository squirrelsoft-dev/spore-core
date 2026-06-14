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
import re
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, ClassVar, Literal, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

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
from .prompt_chunk_registry import CacheBlock
from .tool_registry import TaskPhase


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
class CacheBlockHits:
    """Per-block cache hit signal recorded into :class:`ContextMeta` after
    each model response. Distinct from
    :class:`spore_core.cache_provider.CacheStats`, which carries token
    counts and costs parsed from the response.
    """

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


# Conservative fallback compaction window when neither the caller's
# ``CompactionConfig.context_length`` nor the model's
# ``provider().context_window`` supplies a usable (``> 0``) value (issue #141).
#
# Deliberately small (8K, gemma-class) rather than the old 200K: when the real
# context length is unknown, assume a tight window so compaction still fires
# rather than silently never running.
DEFAULT_CONTEXT_LENGTH = 8_000


@dataclass
class CompactionConfig:
    threshold: float = 0.80
    preserve_recent_n: int = 8
    head_tail_tokens: int = 512
    offload_path: Path = field(default_factory=lambda: Path(".spore/offload"))
    # Maximum revise-and-retry attempts the harness makes when the
    # post-compaction verifier reports missing items before accepting the
    # summary as-is and logging a warn-level event (issue #29). The verifier
    # itself is NOT held here — the harness owns the verifier instance.
    max_compaction_attempts: int = 2
    # Optional caller override for the resolved compaction window (issue #141).
    # When set and ``> 0``, the resolver
    # (:meth:`StandardContextManager.resolve_context_length`) uses it as the
    # ``window_limit``. ``None`` (the default) and an explicit ``0`` both fall
    # through to the model's ``provider().context_window``, then to
    # :data:`DEFAULT_CONTEXT_LENGTH`. Configured values are NOT clamped to the
    # model's real window.
    #
    # Serialized as ABSENT when ``None`` (see :meth:`to_dict`), so an existing
    # serialized ``CompactionConfig`` stays byte-identical (no new key unset).
    context_length: int | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe dict. ``context_length`` is OMITTED when ``None`` so an
        existing serialized ``CompactionConfig`` stays byte-identical — mirrors
        the Rust ``#[serde(skip_serializing_if = "Option::is_none")]`` on the
        same field (issue #141)."""

        data: dict[str, object] = {
            "threshold": self.threshold,
            "preserve_recent_n": self.preserve_recent_n,
            "head_tail_tokens": self.head_tail_tokens,
            "offload_path": str(self.offload_path),
            "max_compaction_attempts": self.max_compaction_attempts,
        }
        if self.context_length is not None:
            data["context_length"] = self.context_length
        return data


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
    # When the real context length is unknown, default to the conservative
    # :data:`DEFAULT_CONTEXT_LENGTH` (8K) rather than the dangerous old 200K so
    # compaction still fires for small-context local models (issue #141). The
    # manager seam (:meth:`StandardContextManager.seed_session`) overrides this
    # with the resolved window in production.
    window_limit: int = DEFAULT_CONTEXT_LENGTH
    guides_loaded: list[GuideId] = field(default_factory=list)
    # Skills pending Block-3 injection on the next assemble. Cleared after
    # each assemble — skills are ephemeral, one turn only.
    pending_skill_injections: list[Guide] = field(default_factory=list)
    budget_warning_active: bool = False
    # Structured fields feeding the four additional preserve hints (issue
    # #47). All default to empty, so an unset field contributes no terms —
    # identical to today's behavior.
    open_problems: list[str] = field(default_factory=list)
    architectural_decisions: list[str] = field(default_factory=list)
    # Recently touched file paths feeding ``keep_recent_file_list``. Typed as
    # plain strings, not path types — keeps tokenization byte-identical across
    # languages (no per-language path semantics).
    recent_files: list[str] = field(default_factory=list)
    # Reasoning summary feeding ``keep_thinking_blocks``; treated
    # byte-identically to ``task_instruction``.
    reasoning_summary: str = ""


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
# Post-compaction verification (issue #29)
# ============================================================================


@dataclass
class CompactionVerificationResult:
    """Outcome of a :meth:`CompactionVerifier.verify` check.

    ``missing_items`` lists preservation terms absent from the summary, in
    first-occurrence order (already lowercased/normalized).
    """

    passed: bool
    missing_items: list[str]
    detail: str


@runtime_checkable
class CompactionVerifier(Protocol):
    """A lightweight, *synchronous* post-compaction sensor.

    Implementations run after the agent produces a compaction summary and
    before the harness accepts it. They are purely computational and MUST
    NOT call the model.
    """

    def verify(
        self,
        summary: str,
        hints: CompactionPreserveHints,
        session_state: SessionState,
    ) -> CompactionVerificationResult: ...


# Runs of characters that are NOT ASCII lowercase letters or digits — the
# token separator used by :class:`KeyTermVerifier`. Source strings are
# lowercased first, so uppercase letters are folded before splitting.
_TERM_SEPARATOR = re.compile(r"[^a-z0-9]+")

# Tokens shorter than this are discarded during term extraction.
_MIN_TERM_LEN = 4


class KeyTermVerifier:
    """Standard :class:`CompactionVerifier`.

    Extracts key terms from the session state per the enabled hints and
    checks they appear in the summary.

    All five hints contribute source terms, each gated on its hint and pushed
    in this fixed order (issue #47) — this order is the cross-language
    invariant that determines first-occurrence dedup:

    1. ``keep_current_task_state`` → :attr:`SessionState.task_instruction`
    2. ``keep_open_problems`` → each :attr:`SessionState.open_problems`
    3. ``keep_architectural_decisions`` →
       each :attr:`SessionState.architectural_decisions`
    4. ``keep_recent_file_list`` → each :attr:`SessionState.recent_files`
    5. ``keep_thinking_blocks`` → :attr:`SessionState.reasoning_summary`

    Each source string runs through the same :meth:`_extract_terms` rule; an
    empty/unset field contributes no terms.
    """

    @staticmethod
    def _extract_terms(source: str) -> list[str]:
        """Tokenize ``source`` into normalized key terms: lowercase, split on
        runs of any non-``[a-z0-9]`` character, and discard tokens shorter
        than four characters. Empty tokens (from leading/trailing/adjacent
        separators) are dropped."""

        return [tok for tok in _TERM_SEPARATOR.split(source.lower()) if len(tok) >= _MIN_TERM_LEN]

    def verify(
        self,
        summary: str,
        hints: CompactionPreserveHints,
        session_state: SessionState,
    ) -> CompactionVerificationResult:
        # Step 1: collect source strings from enabled hints, each gated on its
        # hint and pushed in this fixed order (issue #47). This order is the
        # cross-language invariant that determines first-occurrence dedup.
        sources: list[str] = []
        if hints.keep_current_task_state:
            sources.append(session_state.task_instruction)
        if hints.keep_open_problems:
            sources.extend(session_state.open_problems)
        if hints.keep_architectural_decisions:
            sources.extend(session_state.architectural_decisions)
        if hints.keep_recent_file_list:
            sources.extend(session_state.recent_files)
        if hints.keep_thinking_blocks:
            sources.append(session_state.reasoning_summary)

        # Step 2: build the term list, deduping preserving first-occurrence
        # order.
        terms: list[str] = []
        for source in sources:
            for term in self._extract_terms(source):
                if term not in terms:
                    terms.append(term)

        # Step 3: a term is present iff the lowercased summary contains it.
        summary_lower = summary.lower()
        missing_items = [term for term in terms if term not in summary_lower]

        # Steps 4 + 5.
        total = len(terms)
        passed = not missing_items
        if passed:
            detail = f"all {total} key term(s) present"
        else:
            detail = (
                f"missing {len(missing_items)} of {total} key term(s): {', '.join(missing_items)}"
            )

        return CompactionVerificationResult(
            passed=passed,
            missing_items=missing_items,
            detail=detail,
        )


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
    """A cache block's content hash changed when it was expected to be stable.

    Both Block 1 (:attr:`CacheBlock.STATIC`) and Block 2
    (:attr:`CacheBlock.PER_SESSION`) halt the run on a mid-session mismatch —
    they are treated consistently (#32). A Block-2 change mid-session means
    session-stable content mutated and every subsequent turn would silently pay
    full input-token cost; rather than warn, the run stops so the caller can fix
    the source.

    ``turn_number`` is the turn on which the mismatch was detected (Block 2 only
    halts when ``turn_number > 1``; the turn-1 assemble records the baseline).
    Estimated cache-cost-delta tracking (``UnexpectedMiss``) is a separate
    observability concern tracked in issue #90.
    """

    kind: ClassVar[str] = "CacheHashMismatch"

    def __init__(self, block: CacheBlock, expected: int, actual: int, turn_number: int) -> None:
        self.block = block
        self.expected = expected
        self.actual = actual
        self.turn_number = turn_number
        super().__init__(
            f"cache hash mismatch on block {block.value} at turn {turn_number}: "
            f"expected {expected}, got {actual}"
        )


# ============================================================================
# Serialized ContextError tagged union (wire format)
# ============================================================================
#
# Mirrors the Rust ``ContextError`` serde-tagged enum
# (``#[serde(tag = "kind", rename_all = "snake_case")]``). The exception
# classes above are what assembly raises in-process; these pydantic models are
# the wire shape carried by ``HaltReason.ContextError`` so a run failure can be
# round-tripped. The agent never re-raises these across the harness boundary —
# they are reported as values, matching the ``AgentError`` wire union in
# :mod:`spore_core.agent`.


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ContextErrorTokenCountFailed(_WireModel):
    """``ContextError::TokenCountFailed`` wire variant."""

    kind: Literal["token_count_failed"] = "token_count_failed"


class ContextErrorCompactionFailed(_WireModel):
    """``ContextError::CompactionFailed`` wire variant."""

    kind: Literal["compaction_failed"] = "compaction_failed"
    reason: str


class ContextErrorAssemblyFailed(_WireModel):
    """``ContextError::AssemblyFailed`` wire variant."""

    kind: Literal["assembly_failed"] = "assembly_failed"
    reason: str


class ContextErrorCacheHashMismatch(_WireModel):
    """``ContextError::CacheHashMismatch`` wire variant.

    ``block`` is the :class:`CacheBlock` that mismatched (``static`` for Block 1,
    ``per_session`` for Block 2); ``turn_number`` is the turn the mismatch was
    detected on. See :class:`CacheHashMismatch` for the halt semantics.
    """

    kind: Literal["cache_hash_mismatch"] = "cache_hash_mismatch"
    block: CacheBlock
    expected: int
    actual: int
    turn_number: int


ContextErrorModel = Annotated[
    ContextErrorTokenCountFailed
    | ContextErrorCompactionFailed
    | ContextErrorAssemblyFailed
    | ContextErrorCacheHashMismatch,
    Field(discriminator="kind"),
]


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

    def record_cache_result(self, context: Context, cache_stats: CacheBlockHits) -> None: ...


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
    raises :class:`CacheHashMismatch`. Mid-session Block-2 hash changes raise
    the same :class:`CacheHashMismatch` when ``turn_number > 1`` — Block 1 and
    Block 2 halt consistently (#32). A turn-1 baseline write never halts.
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

    # ---- configurable compaction window (issue #141) ----------------

    def resolve_context_length(self) -> int:
        """Resolve the compaction window (issue #141). Fallback order:

        1. configured :attr:`CompactionConfig.context_length` when set and
           ``> 0``,
        2. else the model's ``provider().context_window`` when ``> 0``,
        3. else :data:`DEFAULT_CONTEXT_LENGTH`.

        An explicit ``0`` (and ``None``) falls through to model metadata, then
        to the default. The configured value is NOT clamped to the model's real
        window — a larger configured value is used as-is.
        """

        configured = self._compaction.context_length
        if configured is not None and configured > 0:
            return configured
        model_window = self._model.provider().context_window
        if model_window > 0:
            return model_window
        return DEFAULT_CONTEXT_LENGTH

    def seed_session(
        self,
        session_id: SessionId,
        task_id: TaskId,
        task_instruction: str,
    ) -> SessionState:
        """Build the initial rich :class:`SessionState` for a run, seeding its
        ``window_limit`` from :meth:`resolve_context_length` (issue #141).

        The manager owns seeding so the resolved window has a single production
        seam — callers get a ``SessionState`` whose ``window_limit`` already
        reflects the configured/model/default resolution rather than the bare
        :class:`SessionState` constructor default.
        """

        state = SessionState(
            session_id=session_id,
            task_id=task_id,
            task_instruction=task_instruction,
        )
        state.window_limit = self.resolve_context_length()
        return state

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
                    block=CacheBlock.STATIC,
                    expected=self._static_hash,
                    actual=static_hash,
                    turn_number=state.turn_number,
                )

        # ── BLOCK 2 (PerSession) ─────────────────────────────────────
        segments = self._build_session_segments(state)
        session_hash = _segments_hash(segments)
        with self._lock:
            prev = self._session_hash
            if prev is not None and prev != session_hash and state.turn_number > 1:
                # Block 2 (PerSession) is expected to be stable for the life of
                # the session. A mid-session change means cost would silently
                # spike; halt consistently with Block 1 (#32). We raise BEFORE
                # updating the memo — the run is halting, so there is no "rest of
                # the session" to track.
                raise CacheHashMismatch(
                    block=CacheBlock.PER_SESSION,
                    expected=prev,
                    actual=session_hash,
                    turn_number=state.turn_number,
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

    def record_cache_result(self, context: Context, cache_stats: CacheBlockHits) -> None:
        context.meta.cache_blocks = CacheBlockStatus(
            static_hit=cache_stats.static_hit,
            session_hit=cache_stats.session_hit,
            history_hit=cache_stats.history_hit,
        )


__all__ = [
    "AssemblyFailed",
    "BreakpointInfo",
    "CacheBlockHits",
    "CacheBlockStatus",
    "CacheHashMismatch",
    "CacheProvider",
    "CompactionConfig",
    "CompactionFailed",
    "CompactionPreserveHints",
    "CompactionRequest",
    "CompactionResult",
    "CompactionVerificationResult",
    "CompactionVerifier",
    "ComposedPrompt",
    "Context",
    "ContextError",
    "ContextErrorAssemblyFailed",
    "ContextErrorCacheHashMismatch",
    "ContextErrorCompactionFailed",
    "ContextErrorModel",
    "ContextErrorTokenCountFailed",
    "KeyTermVerifier",
    "ContextManager",
    "ContextMeta",
    "ContextSources",
    "DEFAULT_CONTEXT_LENGTH",
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
