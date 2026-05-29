"""Issue #79 — Prompt assembly engine.

Conditional, provider-sourced prompt assembly that *extends* (does not replace)
the #24 :mod:`spore_core.prompt_chunk_registry`.

The shipped #24 ``prompt_chunk_registry`` module composes a static Block-1
:class:`~spore_core.prompt_chunk_registry.ComposedPrompt` once at construction.
This module builds *on top of* it: chunks are loaded from pluggable
:class:`ChunkProvider`\\ s and included conditionally based on mode, active
tools, phase, agent type, trigger words, hook events, or arbitrary
architect-defined predicates. The Static bucket is folded into a #24
``ComposedPrompt`` (Block 1); PerSession / PerTurn chunks flow through the
existing ``ComposedPrompt`` / :class:`~spore_core.context.PromptSegment`
machinery in :mod:`spore_core.context` (decision A4 — no new public segment
vectors on :class:`~spore_core.context.ContextSources`).

This module owns its OWN :class:`PromptChunk` and :class:`ChunkProviderError`,
distinct from the #24 ``prompt_chunk_registry`` types, which are left untouched
(decision A1). It is also the home of the minimal shared :class:`StorageScope`
enum (decision A2).

A3 — ``Custom`` is first-class but unserialized
-----------------------------------------------
:class:`Custom` wraps a predicate ``Callable[[AssemblyContext], bool]``. It is
the PRIMARY escape hatch for conditions that cannot be expressed with the
serializable variants, and it is fully supported in the public API. However it
CANNOT serialize, so:

* ``to_json`` of a bare ``Custom`` yields ``None`` (it is omitted from the wire
  form); ``from_json`` of ``None``/absent yields :class:`Always`.
* ``Custom`` is never equal to anything (including another ``Custom``), since
  closure identity is not comparable.
* When serializing ``All`` / ``Any`` / ``Not``, ``Custom`` children are pruned
  out of the wire form (matching the Rust reference).
* ``Custom`` is excluded from the shared byte-identical fixtures. Architects who
  reach for it knowingly opt that chunk out of the cross-language byte-identical
  contract — a deliberate, supported choice.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .context import ContextSources, PromptSegment, SegmentStability
from .errors import SporeError
from .harness import SessionId, TaskId
from .hooks import HookEvent
from .prompt_chunk_registry import (
    CacheBlock,
    ChunkSlot,
    ComposedPrompt,
    Mode,
)
from .prompt_chunk_registry import (
    PromptChunk as RegistryPromptChunk,
)
from .tool_registry import TaskPhase

__all__ = [
    "AssemblyBuckets",
    "AssemblyContext",
    "ChunkCondition",
    "ChunkProvider",
    "ChunkProviderError",
    "CompositeChunkProvider",
    "ContextSourcesBuilder",
    "CustomPredicate",
    "EmbeddedChunkProvider",
    "InMemoryChunkProvider",
    "PromptChunk",
    "StorageScope",
    "ToolAffinity",
    "breakpoint_ids",
    "chunks_to_segments",
]


# ============================================================================
# StorageScope (A2)
# ============================================================================


class StorageScope(str, Enum):
    """Minimal shared storage scope. This module is its home (decision A2); the
    scope-aware ``FileSystemChunkProvider`` that consumes it is deferred (A6)."""

    USER = "user"
    PROJECT = "project"
    LOCAL = "local"

    @classmethod
    def default(cls) -> StorageScope:
        return cls.PROJECT


# ============================================================================
# ToolAffinity
# ============================================================================


@dataclass(frozen=True)
class ToolAffinity:
    """Binds a chunk to a tool (and optionally a sub-capability). The builder
    includes the chunk only when the tool — and capability, if any — is active."""

    tool_name: str
    capability: str | None = None

    @classmethod
    def tool(cls, tool_name: str) -> ToolAffinity:
        return cls(tool_name=tool_name, capability=None)

    @classmethod
    def with_capability(cls, tool_name: str, capability: str) -> ToolAffinity:
        return cls(tool_name=tool_name, capability=capability)

    def to_json(self) -> dict[str, Any]:
        return {"tool_name": self.tool_name, "capability": self.capability}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ToolAffinity:
        return cls(tool_name=data["tool_name"], capability=data.get("capability"))


# ============================================================================
# ChunkCondition
# ============================================================================

CustomPredicate = Callable[["AssemblyContext"], bool]


class ChunkCondition:
    """The condition primitive tree. Architects compose these; the framework
    evaluates them against an :class:`AssemblyContext` via
    :meth:`ContextSourcesBuilder.evaluate`.

    Implemented as a small closed family of subclasses rather than a tagged
    dataclass union — this keeps ``Custom`` (which carries a non-comparable,
    non-serializable callable) a first-class sibling of the serializable
    variants.

    All variants serialize EXCEPT :class:`Custom` — see the module docstring
    (A3). Equality compares structurally; ``Custom`` is never equal to anything.
    """

    # --- constructors (factory classmethods keep the public surface flat) ----

    @staticmethod
    def always() -> ChunkCondition:
        return Always()

    @staticmethod
    def when_mode(mode: Mode) -> ChunkCondition:
        return WhenMode(mode)

    @staticmethod
    def when_tool_active(tool: str) -> ChunkCondition:
        return WhenToolActive(tool)

    @staticmethod
    def when_tool_capability(tool: str, capability: str) -> ChunkCondition:
        return WhenToolCapability(tool, capability)

    @staticmethod
    def when_phase(phase: TaskPhase) -> ChunkCondition:
        return WhenPhase(phase)

    @staticmethod
    def when_agent_type(agent_type: str) -> ChunkCondition:
        return WhenAgentType(agent_type)

    @staticmethod
    def when_feature(feature: str) -> ChunkCondition:
        return WhenFeature(feature)

    @staticmethod
    def on_trigger(words: Iterable[str]) -> ChunkCondition:
        return OnTrigger(list(words))

    @staticmethod
    def on_event(event: HookEvent) -> ChunkCondition:
        return OnEvent(event)

    @staticmethod
    def all_of(conditions: Iterable[ChunkCondition]) -> ChunkCondition:
        return AllOf(list(conditions))

    @staticmethod
    def any_of(conditions: Iterable[ChunkCondition]) -> ChunkCondition:
        return AnyOf(list(conditions))

    @staticmethod
    def not_(condition: ChunkCondition) -> ChunkCondition:
        return NotCond(condition)

    @staticmethod
    def custom(predicate: CustomPredicate) -> ChunkCondition:
        return Custom(predicate)

    # --- evaluation ----------------------------------------------------------

    def evaluate(self, ctx: AssemblyContext) -> bool:
        """Evaluate this condition against ``ctx``. Rules R1–R9."""
        raise NotImplementedError

    # --- serialization (A3) --------------------------------------------------

    def to_json(self) -> dict[str, Any] | None:
        """Serialize to the internally-tagged wire form, or ``None`` for a
        ``Custom`` node (which is omitted from the wire form). ``Custom``
        children of combinators are pruned (matching the Rust reference)."""
        raise NotImplementedError

    @staticmethod
    def from_json(data: dict[str, Any] | None) -> ChunkCondition:
        """Deserialize a condition. A ``None``/absent node yields
        :class:`Always` (A3 — the wire form a skipped ``Custom`` produces)."""
        if data is None:
            return Always()
        kind = data["type"]
        ctor = _CONDITION_FROM_JSON.get(kind)
        if ctor is None:
            raise ChunkProviderError.parse_error(f"unknown condition type: {kind!r}")
        return ctor(data)


@dataclass(frozen=True)
class Always(ChunkCondition):
    def evaluate(self, ctx: AssemblyContext) -> bool:  # R1
        return True

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "always"}


@dataclass(frozen=True)
class WhenMode(ChunkCondition):
    mode: Mode

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R2
        return ctx.mode == self.mode

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "when_mode", "mode": self.mode.value}


@dataclass(frozen=True)
class WhenToolActive(ChunkCondition):
    tool: str

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R3
        return self.tool in ctx.active_tool_names

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "when_tool_active", "tool": self.tool}


@dataclass(frozen=True)
class WhenToolCapability(ChunkCondition):
    tool: str
    capability: str

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R4
        return (self.tool, self.capability) in ctx.active_capabilities

    def to_json(self) -> dict[str, Any] | None:
        return {
            "type": "when_tool_capability",
            "tool": self.tool,
            "capability": self.capability,
        }


@dataclass(frozen=True)
class WhenPhase(ChunkCondition):
    phase: TaskPhase

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R5
        return ctx.phase == self.phase

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "when_phase", "phase": self.phase.value}


@dataclass(frozen=True)
class WhenAgentType(ChunkCondition):
    agent_type: str

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R5
        return ctx.agent_type == self.agent_type

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "when_agent_type", "agent_type": self.agent_type}


@dataclass(frozen=True)
class WhenFeature(ChunkCondition):
    feature: str

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R5 — present AND true
        return ctx.features.get(self.feature, False)

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "when_feature", "feature": self.feature}


@dataclass(frozen=True)
class OnTrigger(ChunkCondition):
    words: list[str]

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R6 — substring; None -> false
        msg = ctx.incoming_message
        if msg is None:
            return False
        return any(w in msg for w in self.words)

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "on_trigger", "words": list(self.words)}


@dataclass(frozen=True)
class OnEvent(ChunkCondition):
    event: HookEvent

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R7
        return self.event in ctx.pending_events

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "on_event", "event": self.event.value}


@dataclass(frozen=True)
class AllOf(ChunkCondition):
    conditions: list[ChunkCondition]

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R8
        return all(c.evaluate(ctx) for c in self.conditions)

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "all", "conditions": _serialize_children(self.conditions)}


@dataclass(frozen=True)
class AnyOf(ChunkCondition):
    conditions: list[ChunkCondition]

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R8
        return any(c.evaluate(ctx) for c in self.conditions)

    def to_json(self) -> dict[str, Any] | None:
        return {"type": "any", "conditions": _serialize_children(self.conditions)}


@dataclass(frozen=True)
class NotCond(ChunkCondition):
    condition: ChunkCondition

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R8
        return not self.condition.evaluate(ctx)

    def to_json(self) -> dict[str, Any] | None:
        inner = self.condition.to_json()
        # A `Not(Custom)` has no representable inner node; prune the whole node.
        if inner is None:
            return None
        return {"type": "not", "condition": inner}


class Custom(ChunkCondition):
    """Arbitrary predicate (A3). First-class, but not serializable and never
    equal under ``==``. Constructible only programmatically."""

    __slots__ = ("predicate",)

    def __init__(self, predicate: CustomPredicate) -> None:
        self.predicate = predicate

    def evaluate(self, ctx: AssemblyContext) -> bool:  # R9
        return self.predicate(ctx)

    def to_json(self) -> dict[str, Any] | None:
        return None  # omitted from the wire form (A3)

    def __eq__(self, other: object) -> bool:
        # Never equal to anything — closure identity is not comparable (A3).
        return False

    def __hash__(self) -> int:
        return object.__hash__(self)

    def __repr__(self) -> str:
        return "Custom(<predicate>)"


def _serialize_children(conditions: list[ChunkCondition]) -> list[dict[str, Any]]:
    """Serialize combinator children, pruning ``Custom`` (and ``Not(Custom)``)
    nodes that have no wire representation (A3)."""
    out: list[dict[str, Any]] = []
    for c in conditions:
        repr_ = c.to_json()
        if repr_ is not None:
            out.append(repr_)
    return out


_CONDITION_FROM_JSON: dict[str, Callable[[dict[str, Any]], ChunkCondition]] = {
    "always": lambda _d: Always(),
    "when_mode": lambda d: WhenMode(Mode(d["mode"])),
    "when_tool_active": lambda d: WhenToolActive(d["tool"]),
    "when_tool_capability": lambda d: WhenToolCapability(d["tool"], d["capability"]),
    "when_phase": lambda d: WhenPhase(TaskPhase(d["phase"])),
    "when_agent_type": lambda d: WhenAgentType(d["agent_type"]),
    "when_feature": lambda d: WhenFeature(d["feature"]),
    "on_trigger": lambda d: OnTrigger(list(d["words"])),
    "on_event": lambda d: OnEvent(HookEvent(d["event"])),
    "all": lambda d: AllOf([ChunkCondition.from_json(c) for c in d["conditions"]]),
    "any": lambda d: AnyOf([ChunkCondition.from_json(c) for c in d["conditions"]]),
    "not": lambda d: NotCond(ChunkCondition.from_json(d["condition"])),
}


# ============================================================================
# PromptChunk (this module's own — distinct from #24, decision A1)
# ============================================================================


@dataclass
class PromptChunk:
    """The unit of conditional assembly content. Distinct from the #24
    :class:`spore_core.prompt_chunk_registry.PromptChunk`: this carries a
    :class:`ChunkCondition`, triggers, affinities, and a stability bucket
    rather than a slot.

    ``to_json`` omits a ``Custom`` condition from the wire form (A3); the
    condition deserializes back to :class:`Always`.
    """

    id: str
    content: str
    stability: SegmentStability = SegmentStability.STATIC
    condition: ChunkCondition = field(default_factory=Always)
    triggers: list[str] = field(default_factory=list)
    tool_affinity: ToolAffinity | None = None
    agent_affinity: str | None = None
    cache_breakpoint: bool = False

    # --- ergonomic builders (mirror the Rust reference fluent API) -----------

    @classmethod
    def new(cls, chunk_id: str, content: str) -> PromptChunk:
        """Build a ``Static``, ``Always`` chunk — the common case."""
        return cls(id=chunk_id, content=content)

    def with_stability(self, stability: SegmentStability) -> PromptChunk:
        self.stability = stability
        return self

    def with_condition(self, condition: ChunkCondition) -> PromptChunk:
        self.condition = condition
        return self

    def with_triggers(self, triggers: Iterable[str]) -> PromptChunk:
        self.triggers = list(triggers)
        return self

    def with_tool_affinity(self, affinity: ToolAffinity) -> PromptChunk:
        self.tool_affinity = affinity
        return self

    def with_agent_affinity(self, agent_type: str) -> PromptChunk:
        self.agent_affinity = agent_type
        return self

    def with_cache_breakpoint(self, breakpoint_: bool) -> PromptChunk:
        self.cache_breakpoint = breakpoint_
        return self

    # --- serialization -------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "stability": self.stability.value,
            "condition": self.condition.to_json(),
            "triggers": list(self.triggers),
            "tool_affinity": (self.tool_affinity.to_json() if self.tool_affinity else None),
            "agent_affinity": self.agent_affinity,
            "cache_breakpoint": self.cache_breakpoint,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PromptChunk:
        aff = data.get("tool_affinity")
        return cls(
            id=data["id"],
            content=data["content"],
            stability=SegmentStability(data["stability"]),
            condition=ChunkCondition.from_json(data.get("condition")),
            triggers=list(data.get("triggers", [])),
            tool_affinity=ToolAffinity.from_json(aff) if aff is not None else None,
            agent_affinity=data.get("agent_affinity"),
            cache_breakpoint=data.get("cache_breakpoint", False),
        )


# ============================================================================
# AssemblyContext
# ============================================================================


@dataclass
class AssemblyContext:
    """Per-assembly inputs the framework populates before each assembly.
    ``Custom`` conditions read from it; ``features`` is the escape hatch for
    architect flags."""

    session_id: SessionId
    task_id: TaskId
    turn_number: int
    mode: Mode
    phase: TaskPhase
    agent_type: str | None = None
    active_tool_names: set[str] = field(default_factory=set)
    active_capabilities: set[tuple[str, str]] = field(default_factory=set)
    incoming_message: str | None = None
    pending_events: list[HookEvent] = field(default_factory=list)
    features: dict[str, bool] = field(default_factory=dict)
    storage_scope: StorageScope = StorageScope.PROJECT

    @classmethod
    def new(
        cls,
        session_id: SessionId,
        task_id: TaskId,
        turn_number: int,
        mode: Mode,
        phase: TaskPhase,
    ) -> AssemblyContext:
        """Construct a minimal context. Optional collections start empty."""
        return cls(
            session_id=session_id,
            task_id=task_id,
            turn_number=turn_number,
            mode=mode,
            phase=phase,
        )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AssemblyContext:
        return cls(
            session_id=SessionId(data["session_id"]),
            task_id=TaskId(data["task_id"]),
            turn_number=data["turn_number"],
            mode=Mode(data["mode"]),
            phase=TaskPhase(data["phase"]),
            agent_type=data.get("agent_type"),
            active_tool_names=set(data.get("active_tool_names", [])),
            active_capabilities={(t, c) for t, c in data.get("active_capabilities", [])},
            incoming_message=data.get("incoming_message"),
            pending_events=[HookEvent(e) for e in data.get("pending_events", [])],
            features=dict(data.get("features", {})),
            storage_scope=StorageScope(data.get("storage_scope", StorageScope.PROJECT.value)),
        )


# ============================================================================
# ChunkProviderError
# ============================================================================


class ChunkProviderError(SporeError):
    """Errors a :class:`ChunkProvider` can raise while loading chunks. Kept
    minimal because the Remote/FileSystem providers are deferred (A6)."""

    def __init__(self, kind: str, detail: str, provider: str | None = None) -> None:
        self.kind = kind
        self.detail = detail
        self.provider = provider
        if provider is not None:
            super().__init__(f"chunk load failed from {provider}: {detail}")
        else:
            super().__init__(f"chunk parse error: {detail}")

    @classmethod
    def load_failed(cls, provider: str, detail: str) -> ChunkProviderError:
        return cls("load_failed", detail, provider=provider)

    @classmethod
    def parse_error(cls, detail: str) -> ChunkProviderError:
        return cls("parse_error", detail)


# ============================================================================
# ChunkProvider protocol + reference implementations
# ============================================================================


@runtime_checkable
class ChunkProvider(Protocol):
    """The pluggable source of chunks (structural — concrete providers satisfy
    it without inheriting). Injected wherever an architect wires chunk
    sources."""

    async def load(self) -> list[PromptChunk]:
        """Load all chunks this provider is responsible for. Called at harness
        construction (every request in stateless deployments, once at startup
        in long-lived ones)."""
        ...

    def invalidate(self) -> None:
        """Drop any cached state so the next ``load()`` fetches fresh. No-op by
        default; never called mid-session."""
        ...


class _ChunkProviderBase:
    """Optional convenience base supplying a no-op :meth:`invalidate`. Concrete
    providers may use it but are not required to (the contract is structural)."""

    @abstractmethod
    async def load(self) -> list[PromptChunk]: ...

    def invalidate(self) -> None:  # default no-op
        return None


class EmbeddedChunkProvider(_ChunkProviderBase):
    """Compile-time / construction-time chunks. Immutable; ``invalidate`` is a
    no-op and ``load`` always returns the same set."""

    def __init__(self, chunks: Iterable[PromptChunk]) -> None:
        self._chunks = list(chunks)

    async def load(self) -> list[PromptChunk]:
        return list(self._chunks)

    # invalidate: inherited no-op (chunks are immutable constants).


class InMemoryChunkProvider(_ChunkProviderBase):
    """Programmatic provider. :meth:`set` replaces the chunk list; the next
    ``load()`` returns the new set."""

    def __init__(self, chunks: Iterable[PromptChunk] | None = None) -> None:
        self._chunks = list(chunks) if chunks is not None else []

    @classmethod
    def empty(cls) -> InMemoryChunkProvider:
        return cls([])

    def set(self, chunks: Iterable[PromptChunk]) -> None:
        """Replace the chunk list. The next ``load()`` returns the new set."""
        self._chunks = list(chunks)

    async def load(self) -> list[PromptChunk]:
        return list(self._chunks)

    def invalidate(self) -> None:
        # Stateless cache; the architect replaces chunks via `set`. Clearing
        # here would discard programmatic registrations, so this is a no-op.
        return None


class CompositeChunkProvider(_ChunkProviderBase):
    """Merges N providers into one flat list (in add order) and propagates
    ``invalidate`` to every child."""

    def __init__(self, providers: Iterable[ChunkProvider] | None = None) -> None:
        self._providers: list[ChunkProvider] = list(providers) if providers is not None else []

    def add(self, provider: ChunkProvider) -> CompositeChunkProvider:
        """Append a child provider (fluent). Returns ``self``."""
        self._providers.append(provider)
        return self

    async def load(self) -> list[PromptChunk]:
        out: list[PromptChunk] = []
        for p in self._providers:
            out.extend(await p.load())
        return out

    def invalidate(self) -> None:
        for p in self._providers:
            p.invalidate()


# ============================================================================
# ContextSourcesBuilder
# ============================================================================


@dataclass
class AssemblyBuckets:
    """The bucketed outcome of :meth:`ContextSourcesBuilder.assemble`. Buckets
    keep registration order within each stability tier."""

    static_chunks: list[PromptChunk] = field(default_factory=list)
    per_session: list[PromptChunk] = field(default_factory=list)
    per_turn: list[PromptChunk] = field(default_factory=list)


class ContextSourcesBuilder:
    """Evaluates conditions, buckets chunks by stability, derives tool-affinity
    inclusion, scans triggers, injects pending events, and composes a Block-1
    :class:`~spore_core.prompt_chunk_registry.ComposedPrompt` from the Static
    bucket. The result feeds
    :class:`~spore_core.context.ContextSources` (decision A4)."""

    def __init__(self, chunks: Iterable[PromptChunk] | None = None) -> None:
        self._chunks: list[PromptChunk] = list(chunks) if chunks is not None else []

    @classmethod
    def with_chunks(cls, chunks: Iterable[PromptChunk]) -> ContextSourcesBuilder:
        """Seed the builder with chunks (registration order is preserved)."""
        return cls(chunks)

    def register(self, chunk: PromptChunk) -> ContextSourcesBuilder:
        """Append a chunk, preserving registration order (fluent)."""
        self._chunks.append(chunk)
        return self

    # --- evaluation ----------------------------------------------------------

    def evaluate(self, condition: ChunkCondition, ctx: AssemblyContext) -> bool:
        """The load-bearing primitive: recursively evaluate ``condition``
        against ``ctx``. Rules R1–R9."""
        return condition.evaluate(ctx)

    @staticmethod
    def _tool_affinity_ok(chunk: PromptChunk, ctx: AssemblyContext) -> bool:
        """Whether a chunk's ``tool_affinity`` gate passes. A chunk with no
        affinity always passes. Rules R12 / R17."""
        aff = chunk.tool_affinity
        if aff is None:
            return True
        if aff.tool_name not in ctx.active_tool_names:
            return False
        if aff.capability is None:
            return True
        return (aff.tool_name, aff.capability) in ctx.active_capabilities

    @staticmethod
    def _agent_affinity_ok(chunk: PromptChunk, ctx: AssemblyContext) -> bool:
        """Whether a chunk's ``agent_affinity`` gate passes. A chunk with no
        agent_affinity always passes; otherwise it must match ``ctx.agent_type``."""
        if chunk.agent_affinity is None:
            return True
        return ctx.agent_type == chunk.agent_affinity

    @staticmethod
    def _triggers_match(chunk: PromptChunk, ctx: AssemblyContext) -> bool:
        """Whether a chunk's ``triggers`` list matches the incoming message. An
        empty trigger list never forces inclusion. Rule R13."""
        if not chunk.triggers:
            return False
        msg = ctx.incoming_message
        if msg is None:
            return False
        return any(t in msg for t in chunk.triggers)

    # --- assembly ------------------------------------------------------------

    def assemble(self, ctx: AssemblyContext) -> AssemblyBuckets:
        """Run the assembly steps and bucket the included chunks. Registration
        order is preserved within each bucket (R10/R11).

        A chunk is included when its ``condition`` evaluates true AND its
        ``tool_affinity`` AND ``agent_affinity`` gates pass. A chunk whose
        ``triggers`` match the incoming message is forced into the PerTurn bucket
        regardless of its declared stability (R13). Bucket assignment otherwise
        follows the chunk's ``stability``."""
        buckets = AssemblyBuckets()

        for chunk in self._chunks:
            # Gates that apply to EVERY chunk regardless of condition kind.
            if not self._tool_affinity_ok(chunk, ctx):
                continue
            if not self._agent_affinity_ok(chunk, ctx):
                continue

            condition_ok = chunk.condition.evaluate(ctx)
            trigger_forced = self._triggers_match(chunk, ctx)

            if not condition_ok and not trigger_forced:
                continue

            # R13: a trigger match routes the chunk into PerTurn no matter its
            # declared stability. R14 falls out of this too: an OnEvent chunk is
            # only condition_ok when its event is pending, and OnEvent chunks are
            # declared PerTurn by convention.
            if trigger_forced:
                buckets.per_turn.append(chunk)
                continue

            if chunk.stability is SegmentStability.STATIC:
                buckets.static_chunks.append(chunk)
            elif chunk.stability is SegmentStability.PER_SESSION:
                buckets.per_session.append(chunk)
            else:
                buckets.per_turn.append(chunk)

        return buckets

    # --- block-1 composition -------------------------------------------------

    def compose_block_1(self, buckets: AssemblyBuckets) -> ComposedPrompt:
        """Compose the Static bucket into a #24
        :class:`~spore_core.prompt_chunk_registry.ComposedPrompt` (Block 1).
        Each Static chunk maps to a #24 ``PromptChunk`` in
        :attr:`~spore_core.prompt_chunk_registry.ChunkSlot.ENVIRONMENT` (a
        neutral, non-required slot) with
        :attr:`~spore_core.prompt_chunk_registry.CacheBlock.STATIC`, preserving
        order. The block hashes are recomputed so the Block-1 hash is stable
        across identical Static sets (R15)."""
        chunks = [
            RegistryPromptChunk.new(
                c.id,
                c.content,
                ChunkSlot.ENVIRONMENT,
                CacheBlock.STATIC,
            )
            for c in buckets.static_chunks
        ]
        composed = ComposedPrompt(chunks=chunks, block_1_hash=0, block_2_hash=0, rendered=None)
        b1, b2 = composed.recompute_hashes()
        composed.block_1_hash = b1
        composed.block_2_hash = b2
        composed.render()
        return composed

    def build_context_sources(
        self,
        ctx: AssemblyContext,
        guides: list[Any] | None = None,
        memory: list[Any] | None = None,
        tool_schemas: list[Any] | None = None,
    ) -> tuple[ContextSources, AssemblyBuckets]:
        """Full pipeline: assemble buckets, compose Block 1, and produce a
        :class:`~spore_core.context.ContextSources` (decision A4 —
        PerSession/PerTurn fold through the existing composed-prompt/segment
        machinery downstream; this builder supplies the composed Block 1 and the
        buckets the caller threads onward).

        ``guides``, ``memory``, and ``tool_schemas`` are passed through verbatim
        — the builder does not synthesize tool description text (decision A5)."""
        buckets = self.assemble(ctx)
        composed_prompt = self.compose_block_1(buckets)
        sources = ContextSources(
            guides=guides if guides is not None else [],
            memory=memory if memory is not None else [],
            tool_schemas=tool_schemas if tool_schemas is not None else [],
            composed_prompt=composed_prompt,
        )
        return sources, buckets


# ============================================================================
# Free helpers (segment mapping — decision A4)
# ============================================================================


def breakpoint_ids(buckets: AssemblyBuckets) -> list[str]:
    """Ids of chunks that declared ``cache_breakpoint`` (R16), in
    static → per_session → per_turn order. Exposed for callers wiring the
    PerSession/PerTurn segments into the segment machinery."""
    out: list[str] = []
    for chunk in (*buckets.static_chunks, *buckets.per_session, *buckets.per_turn):
        if chunk.cache_breakpoint:
            out.append(chunk.id)
    return out


def chunks_to_segments(chunks: Iterable[PromptChunk]) -> list[PromptSegment]:
    """Map chunks into :class:`~spore_core.context.PromptSegment`\\ s for the #7
    context machinery (decision A4). Preserves order and carries each chunk's
    ``cache_breakpoint`` (R16)."""
    return [
        PromptSegment(
            name=c.id,
            content=c.content,
            stability=c.stability,
            cache_breakpoint=c.cache_breakpoint,
        )
        for c in chunks
    ]


# Re-exported for callers that want to drive a provider synchronously in tests.
def load_chunks_sync(provider: ChunkProvider) -> list[PromptChunk]:
    """Drive ``provider.load()`` to completion synchronously. Convenience for
    non-async call sites; prefer awaiting ``load()`` directly in async code."""
    return asyncio.run(provider.load())
