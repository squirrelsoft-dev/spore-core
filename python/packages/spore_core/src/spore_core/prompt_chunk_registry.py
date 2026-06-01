"""PromptChunkRegistry — named, cacheable prompt chunks (issue #24).

Mirrors the Rust reference at
``rust/crates/spore-core/src/prompt_chunk_registry.rs``.

Manages small, independently cacheable text fragments that compose the
system prompt deterministically at harness construction time. Each chunk
does one thing; the composed prompt for a planning agent differs from
an evaluator agent, and both are fully cached in Block 1.

Rules enforced:

* Each chunk has a unique :class:`ChunkId`; duplicate registration is
  :class:`ChunkError` with kind ``duplicate_id``.
* :attr:`ChunkSlot.BUDGET` and :attr:`ChunkSlot.EPHEMERAL` are always
  :attr:`CacheBlock.PER_TURN`; mismatch is ``conflicting_cache_block``.
* :attr:`ChunkSlot.ROLE` and :attr:`ChunkSlot.MODE` are always
  :attr:`CacheBlock.STATIC`; mismatch is ``conflicting_cache_block``.
* Chunk content must not be empty (``invalid_slot``).
* :meth:`PromptChunkRegistry.compose` requires the named role chunk to
  be registered and sources the mode chunk via :meth:`Mode.prompt_chunk`.
* Composed chunks are ordered by slot (Role → Mode → Capability →
  Skill → Task → Environment → PriorSession → Budget → Ephemeral); within
  a slot, capabilities and skills follow the caller-provided sequence.
* :meth:`PromptChunkRegistry.validate` rejects compositions with a
  PerTurn chunk in Block 1, missing Role or Mode, or more than one Mode.
* Mode is permanent for the life of a harness instance — there is no
  mutation API.

``dangerous`` gate
------------------

``Yolo`` (full autonomy, no approval gates) is a named safety footgun. It is
NOT a member of the default :class:`Mode` enum and is not reachable from a
normal ``import``; ``Mode.YOLO`` and ``Mode("yolo")`` both fail. The gated
mode lives in :mod:`spore_core.dangerous` as ``YoloMode`` and must be opted
into explicitly (``from spore_core.dangerous import YoloMode``). The wire tag
stays ``"yolo"``. See issue #34 and the Rust reference's ``dangerous`` Cargo
feature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, NewType, Protocol, runtime_checkable

from .errors import SporeError
from .tool_registry import TaskPhase

# ============================================================================
# Identity
# ============================================================================

ChunkId = NewType("ChunkId", str)
"""Stable identifier for a registered prompt chunk."""


def new_chunk_id(s: str) -> ChunkId:
    return ChunkId(s)


# ============================================================================
# Enums
# ============================================================================


class ChunkSlot(str, Enum):
    """Where a chunk lands in the composed system prompt.

    Ordering corresponds to render order: Role first, Ephemeral last.
    """

    ROLE = "role"
    MODE = "mode"
    CAPABILITY = "capability"
    SKILL = "skill"
    TASK = "task"
    ENVIRONMENT = "environment"
    PRIOR_SESSION = "prior_session"
    BUDGET = "budget"
    EPHEMERAL = "ephemeral"

    def order(self) -> int:
        return _SLOT_ORDER[self]


_SLOT_ORDER: dict[ChunkSlot, int] = {
    ChunkSlot.ROLE: 0,
    ChunkSlot.MODE: 1,
    ChunkSlot.CAPABILITY: 2,
    ChunkSlot.SKILL: 3,
    ChunkSlot.TASK: 4,
    ChunkSlot.ENVIRONMENT: 5,
    ChunkSlot.PRIOR_SESSION: 6,
    ChunkSlot.BUDGET: 7,
    ChunkSlot.EPHEMERAL: 8,
}


class CacheBlock(str, Enum):
    """Which prompt cache block a chunk belongs to."""

    STATIC = "static"
    PER_SESSION = "per_session"
    PER_TURN = "per_turn"


class ApprovalPolicy(str, Enum):
    """Approval enforcement implied by a :class:`Mode`."""

    ALWAYS_ASK = "always_ask"
    AUTO_EXPLAIN = "auto_explain"
    PLAN_ONLY = "plan_only"
    SAFE_AUTO = "safe_auto"
    NONE = "none"


# Mode behavior tables are keyed by the wire-tag string so the gated ``Yolo``
# mode in :mod:`spore_core.dangerous` can reuse the same source of truth
# without re-importing the ``Mode`` enum (which deliberately omits Yolo).
_MODE_CHUNK_TEXT_BY_TAG: dict[str, tuple[str, str]] = {
    "always_ask": (
        "mode-always-ask",
        "Mode: AlwaysAsk. Describe your plan and wait for explicit approval "
        "before taking any action.",
    ),
    "auto_edit": (
        "mode-auto-edit",
        "Mode: AutoEdit. Edit files freely. Explain the changes you make after they are done.",
    ),
    "plan": (
        "mode-plan",
        "Mode: Plan. Produce a plan only. Do not edit files or execute mutating tools.",
    ),
    "safe_auto": (
        "mode-safe-auto",
        "Mode: SafeAuto. Auto-execute Low and Medium risk actions. High and "
        "Critical actions require approval.",
    ),
    # Gated — reachable only via spore_core.dangerous.YoloMode (issue #34).
    "yolo": (
        "mode-yolo",
        "Mode: Yolo. Full autonomy. No approval gates.",
    ),
}


_MODE_APPROVAL_BY_TAG: dict[str, ApprovalPolicy] = {
    "always_ask": ApprovalPolicy.ALWAYS_ASK,
    "auto_edit": ApprovalPolicy.AUTO_EXPLAIN,
    "plan": ApprovalPolicy.PLAN_ONLY,
    "safe_auto": ApprovalPolicy.SAFE_AUTO,
    # Gated — reachable only via spore_core.dangerous.YoloMode (issue #34).
    "yolo": ApprovalPolicy.NONE,
}


def _mode_prompt_chunk(tag: str) -> PromptChunk:
    """Standard prompt chunk for a mode wire tag. Always Static, slot=Mode."""
    chunk_id, content = _MODE_CHUNK_TEXT_BY_TAG[tag]
    return PromptChunk.new(chunk_id, content, ChunkSlot.MODE, CacheBlock.STATIC)


def _mode_approval_policy(tag: str) -> ApprovalPolicy:
    return _MODE_APPROVAL_BY_TAG[tag]


def _mode_default_tool_phase(tag: str) -> TaskPhase:
    return TaskPhase.PLANNING if tag == "plan" else TaskPhase.EXECUTION


class Mode(str, Enum):
    """Agent behavioral mode — permanent for the life of a harness.

    ``Yolo`` is intentionally absent: it is a named safety footgun gated behind
    the ``dangerous`` opt-in (issue #34). Import it explicitly via
    ``from spore_core.dangerous import YoloMode``. ``Mode("yolo")`` raises.
    """

    ALWAYS_ASK = "always_ask"
    AUTO_EDIT = "auto_edit"
    PLAN = "plan"
    SAFE_AUTO = "safe_auto"

    def prompt_chunk(self) -> PromptChunk:
        """Standard prompt chunk for this mode. Always Static, slot=Mode."""
        return _mode_prompt_chunk(self.value)

    def approval_policy(self) -> ApprovalPolicy:
        return _mode_approval_policy(self.value)

    def default_tool_phase(self) -> TaskPhase:
        return _mode_default_tool_phase(self.value)


# Structural type accepted by :meth:`PromptChunkRegistry.compose`. The default
# :class:`Mode` enum satisfies it; so does the gated ``YoloMode`` from
# :mod:`spore_core.dangerous`. Compose only calls ``prompt_chunk()``.
@runtime_checkable
class ModeLike(Protocol):
    """Anything that can supply a Mode prompt chunk (issue #34)."""

    def prompt_chunk(self) -> PromptChunk: ...

    def approval_policy(self) -> ApprovalPolicy: ...

    def default_tool_phase(self) -> TaskPhase: ...


# ============================================================================
# Records
# ============================================================================


def _hash_content(content: str) -> int:
    """Deterministic content hash. Used to detect cache invalidation.

    BLAKE2b-64 keeps it stable across Python processes; the Rust reference
    uses ``DefaultHasher`` so cross-language hash equality is NOT required
    (and the fixture spec explicitly says so).
    """
    digest = hashlib.blake2b(content.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


@dataclass(frozen=True)
class PromptChunk:
    id: ChunkId
    content: str
    slot: ChunkSlot
    cache_block: CacheBlock
    hash: int

    @classmethod
    def new(
        cls,
        chunk_id: str | ChunkId,
        content: str,
        slot: ChunkSlot,
        cache_block: CacheBlock,
    ) -> PromptChunk:
        """Build a chunk with content hash computed. Prefer this over the
        constructor so ``hash`` stays in sync with ``content``."""
        return cls(
            id=ChunkId(str(chunk_id)),
            content=content,
            slot=slot,
            cache_block=cache_block,
            hash=_hash_content(content),
        )


@dataclass
class ComposedPrompt:
    chunks: list[PromptChunk]
    block_1_hash: int
    block_2_hash: int
    rendered: str | None = None
    """Cached render of all chunks joined by ``\\n\\n``. ``None`` until
    materialized by :meth:`render`. Invalidated when any chunk hash changes."""

    def render(self) -> str:
        """Render chunks deterministically. Caches the result."""
        if self.rendered is None:
            self.rendered = "\n\n".join(c.content for c in self.chunks)
        return self.rendered

    def rendered_str(self) -> str:
        """Cached render or empty string when not yet rendered."""
        return self.rendered if self.rendered is not None else ""

    def recompute_hashes(self) -> tuple[int, int]:
        """Recompute (block_1_hash, block_2_hash) from current chunk state."""
        return _compute_block_hashes(self.chunks)


# ============================================================================
# Errors
# ============================================================================


class ChunkError(SporeError):
    """Root of every error raised by chunk registration."""

    kind: ClassVar[str] = "ChunkError"


class DuplicateChunkId(ChunkError):
    kind: ClassVar[str] = "duplicate_id"

    def __init__(self, chunk_id: ChunkId) -> None:
        self.id = chunk_id
        super().__init__(f"duplicate chunk id: {chunk_id!r}")


class InvalidSlot(ChunkError):
    kind: ClassVar[str] = "invalid_slot"

    def __init__(self, chunk_id: ChunkId, reason: str) -> None:
        self.id = chunk_id
        self.reason = reason
        super().__init__(f"invalid slot for chunk {chunk_id!r}: {reason}")


class ConflictingCacheBlock(ChunkError):
    kind: ClassVar[str] = "conflicting_cache_block"

    def __init__(
        self,
        chunk_id: ChunkId,
        slot: ChunkSlot,
        expected: CacheBlock,
        actual: CacheBlock,
    ) -> None:
        self.id = chunk_id
        self.slot = slot
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"conflicting cache block for chunk {chunk_id!r} in slot "
            f"{slot.value!r}: expected {expected.value!r}, got {actual.value!r}"
        )


class ChunkNotFound(ChunkError):
    kind: ClassVar[str] = "not_found"

    def __init__(self, chunk_id: ChunkId) -> None:
        self.id = chunk_id
        super().__init__(f"chunk not found: {chunk_id!r}")


class ChunkValidationError(SporeError):
    """Root of every composition-validation error."""

    kind: ClassVar[str] = "ChunkValidationError"


class PerTurnChunkInStaticBlock(ChunkValidationError):
    kind: ClassVar[str] = "per_turn_chunk_in_static_block"

    def __init__(self, chunk_id: ChunkId) -> None:
        self.id = chunk_id
        super().__init__(f"per-turn chunk {chunk_id!r} placed in the Static block")


class MissingRequiredSlot(ChunkValidationError):
    kind: ClassVar[str] = "missing_required_slot"

    def __init__(self, slot: ChunkSlot) -> None:
        self.slot = slot
        super().__init__(f"required slot {slot.value!r} is missing from the composition")


class ConflictingModeChunks(ChunkValidationError):
    kind: ClassVar[str] = "conflicting_mode_chunks"

    def __init__(self, ids: list[ChunkId]) -> None:
        self.ids = list(ids)
        super().__init__(f"more than one Mode chunk in the composition: {self.ids!r}")


class ComposeFailed(SporeError):
    """Raised when :meth:`PromptChunkRegistry.compose` fails. Aggregates the
    underlying :class:`ChunkValidationError` instances in ``errors``."""

    kind: ClassVar[str] = "compose_failed"

    def __init__(self, errors: list[ChunkValidationError]) -> None:
        self.errors = list(errors)
        super().__init__(
            "compose failed: " + "; ".join(str(e) for e in errors) if errors else "compose failed"
        )


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class PromptChunkRegistry(Protocol):
    """Canonical issue-#24 ``PromptChunkRegistry`` interface."""

    def register(self, chunk: PromptChunk) -> None: ...

    def compose(
        self,
        role: ChunkId,
        mode: ModeLike,
        capabilities: list[ChunkId],
        skills: list[ChunkId],
    ) -> ComposedPrompt: ...

    def validate(self, composed: ComposedPrompt) -> list[ChunkValidationError]: ...

    def get(self, chunk_id: ChunkId) -> PromptChunk | None: ...


# ============================================================================
# Helpers
# ============================================================================


def _compute_block_hashes(chunks: list[PromptChunk]) -> tuple[int, int]:
    """Compute (Block 1, Block 2) hashes deterministically.

    Group by cache block, then by id — never rely on dict iteration order.
    """
    block_1: dict[str, int] = {}
    block_2: dict[str, int] = {}
    for c in chunks:
        if c.cache_block is CacheBlock.STATIC:
            block_1[str(c.id)] = c.hash
        elif c.cache_block is CacheBlock.PER_SESSION:
            block_2[str(c.id)] = c.hash
        # PerTurn chunks never contribute to block hashes.

    def _digest(entries: dict[str, int]) -> int:
        h = hashlib.blake2b(digest_size=8)
        for k in sorted(entries):
            h.update(k.encode("utf-8"))
            h.update(entries[k].to_bytes(8, "big", signed=False))
        return int.from_bytes(h.digest(), "big", signed=False)

    return _digest(block_1), _digest(block_2)


def _validate_slot_and_cache_block(chunk: PromptChunk) -> None:
    if not chunk.content.strip():
        raise InvalidSlot(chunk.id, "content must not be empty")
    if chunk.slot in (ChunkSlot.BUDGET, ChunkSlot.EPHEMERAL):
        if chunk.cache_block is not CacheBlock.PER_TURN:
            raise ConflictingCacheBlock(
                chunk.id, chunk.slot, CacheBlock.PER_TURN, chunk.cache_block
            )
    elif chunk.slot in (ChunkSlot.ROLE, ChunkSlot.MODE):
        if chunk.cache_block is not CacheBlock.STATIC:
            raise ConflictingCacheBlock(chunk.id, chunk.slot, CacheBlock.STATIC, chunk.cache_block)


# ============================================================================
# StandardPromptChunkRegistry
# ============================================================================


@dataclass
class _Store:
    chunks: dict[ChunkId, PromptChunk] = field(default_factory=dict)
    order: list[ChunkId] = field(default_factory=list)


class StandardPromptChunkRegistry:
    """Reference, in-memory :class:`PromptChunkRegistry`. The harness owns a
    single shared instance; chunks are typically registered at startup and
    never mutated afterwards.
    """

    def __init__(self) -> None:
        self._store = _Store()

    def register_standard_chunks(self) -> None:
        """Register every chunk in :func:`standard_chunks`."""
        for chunk in standard_chunks():
            self.register(chunk)

    # ── PromptChunkRegistry protocol ───────────────────────────────────────

    def register(self, chunk: PromptChunk) -> None:
        _validate_slot_and_cache_block(chunk)
        if chunk.id in self._store.chunks:
            raise DuplicateChunkId(chunk.id)
        self._store.order.append(chunk.id)
        self._store.chunks[chunk.id] = chunk

    def compose(
        self,
        role: ChunkId,
        mode: ModeLike,
        capabilities: list[ChunkId],
        skills: list[ChunkId],
    ) -> ComposedPrompt:
        errors: list[ChunkValidationError] = []
        chosen: list[PromptChunk] = []

        # Role
        role_chunk = self._store.chunks.get(role)
        if role_chunk is not None and role_chunk.slot is ChunkSlot.ROLE:
            chosen.append(role_chunk)
        else:
            errors.append(MissingRequiredSlot(ChunkSlot.ROLE))

        # Mode — always sourced from the enum.
        chosen.append(mode.prompt_chunk())

        # Capabilities
        for cap_id in capabilities:
            c = self._store.chunks.get(cap_id)
            if c is not None and c.slot is ChunkSlot.CAPABILITY:
                chosen.append(c)
            else:
                errors.append(MissingRequiredSlot(ChunkSlot.CAPABILITY))

        # Skills
        for skill_id in skills:
            c = self._store.chunks.get(skill_id)
            if c is not None and c.slot is ChunkSlot.SKILL:
                chosen.append(c)
            else:
                errors.append(MissingRequiredSlot(ChunkSlot.SKILL))

        if errors:
            raise ComposeFailed(errors)

        # Stable sort by slot order; within a slot, preserve insertion order
        # (capabilities and skills follow caller-provided sequence).
        chosen.sort(key=lambda c: c.slot.order())

        block_1_hash, block_2_hash = _compute_block_hashes(chosen)

        composed = ComposedPrompt(
            chunks=chosen,
            block_1_hash=block_1_hash,
            block_2_hash=block_2_hash,
            rendered=None,
        )

        v_errors = self.validate(composed)
        if v_errors:
            raise ComposeFailed(v_errors)

        return composed

    def validate(self, composed: ComposedPrompt) -> list[ChunkValidationError]:
        errors: list[ChunkValidationError] = []

        # Block 1 must not contain PerTurn-slotted chunks marked Static.
        for c in composed.chunks:
            if c.cache_block is CacheBlock.STATIC and c.slot in (
                ChunkSlot.BUDGET,
                ChunkSlot.EPHEMERAL,
            ):
                errors.append(PerTurnChunkInStaticBlock(c.id))

        # Required slots: Role and Mode.
        for required in (ChunkSlot.ROLE, ChunkSlot.MODE):
            if not any(c.slot is required for c in composed.chunks):
                errors.append(MissingRequiredSlot(required))

        # At most one Mode chunk.
        mode_ids = [c.id for c in composed.chunks if c.slot is ChunkSlot.MODE]
        if len(mode_ids) > 1:
            errors.append(ConflictingModeChunks(mode_ids))

        return errors

    def get(self, chunk_id: ChunkId) -> PromptChunk | None:
        return self._store.chunks.get(chunk_id)


# ============================================================================
# Standard chunk library
# ============================================================================


def standard_chunks() -> list[PromptChunk]:
    """Standard chunk library shipped by ``spore_core``. Users register
    additional chunks for their domain."""

    out: list[PromptChunk] = []

    roles: list[tuple[str, str]] = [
        (
            "role-coding-agent",
            "You are an expert software engineer. Read carefully, change "
            "deliberately, and verify your work.",
        ),
        (
            "role-evaluator",
            "You are a fresh evaluator. You did not write the code under review. Default to FAIL.",
        ),
        (
            "role-planner",
            "You are a planning specialist. Decompose tasks into small, verifiable steps.",
        ),
        (
            "role-rag-assistant",
            "You are a retrieval-augmented assistant. Always cite the source "
            "document for any claim.",
        ),
        (
            "role-sql-agent",
            "You are a SQL specialist. Prefer read-only queries; never DROP "
            "without explicit approval.",
        ),
    ]
    for chunk_id, content in roles:
        out.append(PromptChunk.new(chunk_id, content, ChunkSlot.ROLE, CacheBlock.STATIC))

    # Modes — derived from the enum so prompt_chunk() and standard_chunks()
    # are guaranteed to agree. ``Yolo`` is deliberately excluded from the
    # default library (issue #34); the dangerous opt-in supplies it.
    for mode in (Mode.ALWAYS_ASK, Mode.AUTO_EDIT, Mode.PLAN, Mode.SAFE_AUTO):
        out.append(mode.prompt_chunk())

    capabilities: list[tuple[str, str]] = [
        ("capability-bash", "Capability: bash. You can run shell commands inside the sandbox."),
        (
            "capability-filesystem",
            "Capability: filesystem. You can read and write files inside the workspace.",
        ),
        ("capability-git", "Capability: git. You can stage, commit, and inspect history."),
        ("capability-browser", "Capability: browser. You can fetch web pages and follow links."),
        ("capability-subagent", "Capability: subagent. You can delegate work to a child harness."),
        (
            "capability-sql",
            "Capability: sql. You can issue queries against the configured database.",
        ),
    ]
    for chunk_id, content in capabilities:
        out.append(PromptChunk.new(chunk_id, content, ChunkSlot.CAPABILITY, CacheBlock.STATIC))

    skills: list[tuple[str, str]] = [
        (
            "skill-testing",
            "Skill: always run the test suite after changes and report results.",
        ),
        (
            "skill-decomposition",
            "Skill: break large tasks into small, independently verifiable steps.",
        ),
        (
            "skill-security-review",
            "Skill: review changes for injection, auth, and secret-leak issues before commit.",
        ),
        (
            "skill-citation",
            "Skill: cite the source document for every claim drawn from retrieved context.",
        ),
    ]
    for chunk_id, content in skills:
        out.append(PromptChunk.new(chunk_id, content, ChunkSlot.SKILL, CacheBlock.STATIC))

    return out


__all__ = [
    "ApprovalPolicy",
    "CacheBlock",
    "ChunkError",
    "ChunkId",
    "ChunkNotFound",
    "ChunkSlot",
    "ChunkValidationError",
    "ComposeFailed",
    "ComposedPrompt",
    "ConflictingCacheBlock",
    "ConflictingModeChunks",
    "DuplicateChunkId",
    "InvalidSlot",
    "MissingRequiredSlot",
    "Mode",
    "ModeLike",
    "PerTurnChunkInStaticBlock",
    "PromptChunk",
    "PromptChunkRegistry",
    "StandardPromptChunkRegistry",
    "new_chunk_id",
    "standard_chunks",
]
