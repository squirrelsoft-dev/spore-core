"""Issue #73 — :class:`StorageProvider`: a pluggable, per-domain persistence
layer.

Behavioural port of the Rust reference at
``rust/crates/spore-core/src/storage.rs``. NOT a transliteration — idiomatic
Python: ``typing.Protocol`` for the four domain interfaces, ``async`` methods
matching the package's component style, ``pydantic`` for the serialized
:class:`MemoryEntry`, and a :class:`StorageError` exception hierarchy rooted at
:class:`SporeError`.

## Domains

* :class:`SessionStore` — pause/resume lifecycle; stores :class:`PausedState`
  keyed by :class:`SessionId`.
* :class:`MemoryStore` — append-only episodic memory; ``get_memories(limit)``
  returns the **most-recent N, newest-first** (recency semantics).
* :class:`RunStore` — per-run opaque JSON blobs keyed by ``(session_id, key)``;
  the store never knows the schema.
* :class:`ObservabilityStore` — append-only span storage; ``append_span`` /
  ``get_spans`` / ``get_sessions`` / ``flush_session``.

## :class:`StorageProvider`

A struct of four domain stores (``session`` / ``memory`` / ``run`` /
``observability``). Built either from a single backend implementing all four
protocols (cloned into all four slots via :meth:`StorageProvider.single`) or
per-domain via :class:`CompositeStorageProvider`.

## Providers

* :class:`NoOpStorageProvider` — silent discard; reads return ``None`` / ``[]``,
  writes are no-ops. The default for any unconfigured domain.
* :class:`InMemoryStorageProvider` — in-process dicts; used in tests / ephemeral
  runs.
* :class:`FileSystemStorageProvider` — disk-backed under a configurable ``root``.
  Layout (mirrors ``.spore/``):

  - session → ``{root}/sessions/{id}/state.json`` (atomic write-rename)
  - run     → ``{root}/sessions/{id}/run/{key}.json`` (atomic write-rename)
  - memory  → ``{root}/sessions/{id}/memory.jsonl`` (append)
  - obs     → ``{root}/sessions/{id}/trace.jsonl`` (append)

  ``flush_session`` creates a sibling ``.flushed`` marker.

## Rules enforced

* **No-op fallback.** Unconfigured domains fall back to
  :class:`NoOpStorageProvider`; callers never null-check.
* **Single-fills-all-slots.** :meth:`StorageProvider.single` places one backend
  into all four slots.
* **Composite per-domain routing.** :class:`CompositeStorageProvider` holds an
  optional store per domain; :meth:`~CompositeStorageProvider.build` fills each
  unset slot with a :class:`NoOpStorageProvider`.
* **Atomic write-rename.** Non-append writes ensure the parent dir, write full
  bytes to a sibling ``{target}.tmp``, flush + ``fsync``, then ``os.replace``.
  No leftover ``.tmp`` on success. Byte-identical algorithm across all four
  languages.
* **``get_memories`` recency.** Returns the most-recent ``limit`` entries,
  newest-first.
* **Last-writer-wins** for filesystem non-append writes via rename; no per-key
  locking contract — atomic rename is the only durability guarantee.
* **OTLP endpoint parsing.** :func:`parse_otlp_endpoints` is the single most
  important cross-language rule (see ``fixtures/storage/otlp_endpoints_parse``):
  ``split(',')``, trim each segment, drop empty segments.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .errors import SporeError
from .guide_registry import SessionOutcome
from .harness import PausedState, SessionId
from .memory import Timestamp
from .observability import SessionMetrics

if TYPE_CHECKING:
    # Re-export of the canonical :class:`StorageScope` (its home is
    # ``prompt_assembly``, decision A2). It is imported lazily at runtime (see
    # ``__getattr__`` below) to avoid an import cycle: this module is pulled in
    # during ``harness`` initialization via ``observability_outbox``, while
    # ``prompt_assembly`` transitively imports ``harness``. Type-checkers see
    # the real symbol here; runtime resolves it on first attribute access.
    from .prompt_assembly import StorageScope

# A free-form opaque JSON value handled by the storage layer.
JsonValue = Any


def _storage_scope() -> type[StorageScope]:
    """Lazily import the canonical :class:`StorageScope` enum, breaking the
    ``storage`` ↔ ``prompt_assembly`` ↔ ``harness`` import cycle (#78)."""
    from .prompt_assembly import StorageScope as _StorageScope

    return _StorageScope


def __getattr__(name: str) -> Any:
    """Module-level lazy attribute resolution so
    ``spore_core.storage.StorageScope`` resolves to the canonical enum without a
    top-level import (decision A2; cycle-safe — #78)."""
    if name == "StorageScope":
        return _storage_scope()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ============================================================================
# StorageError
# ============================================================================


class StorageError(SporeError):
    """Root of storage-domain errors (issue #73). Mirrors the Rust
    ``StorageError`` enum as an exception hierarchy: :class:`StorageIoError`,
    :class:`StorageSerializationError`, :class:`StorageNotFoundError`,
    :class:`StorageBackendError`."""


class StorageIoError(StorageError):
    """An I/O failure from a filesystem-backed store."""


class StorageSerializationError(StorageError):
    """A (de)serialization failure crossing the storage boundary."""


class StorageNotFoundError(StorageError):
    """A keyed lookup found nothing where the caller required a value."""

    def __init__(self, domain: str, key: str) -> None:
        self.domain = domain
        self.key = key
        super().__init__(f"storage not found: domain={domain} key={key}")


class StorageBackendError(StorageError):
    """A backend-specific failure that does not map to the variants above."""


# ============================================================================
# MemoryEntry
# ============================================================================


class MemoryEntry(BaseModel):
    """One episodic memory entry. Byte-identical cross-language:
    ``{ role, content, timestamp, metadata }`` where ``metadata`` defaults to an
    empty JSON object ``{}``. Matches the Rust serde shape so fixtures replay
    identically."""

    role: str
    content: str
    timestamp: Timestamp
    metadata: JsonValue = Field(default_factory=dict)


# ============================================================================
# WorkspaceId (#78)
# ============================================================================


#: A stable identifier for a workspace, derived purely from its canonical path.
#: Form ``{sanitized_basename}-{8_hex_chars}``, lowercased. A ``NewType`` over
#: ``str`` (Python conventions: IDs are ``NewType`` aliases).
WorkspaceId = NewType("WorkspaceId", str)


def _canonicalize_path_string(path: str) -> str:
    """Steps 1–2 of the :func:`workspace_id_from_canonical_path` derivation:
    produce the canonical path string used for both the hash input and the
    basename. Forward slashes only, no trailing slash. On Windows the
    drive-letter prefix is stripped and ``\\`` → ``/``."""
    # Normalize Windows backslashes.
    s = path.replace("\\", "/")
    # Strip a leading drive-letter prefix like ``C:`` (only at the very start).
    if len(s) >= 2 and s[1] == ":" and s[0].isascii() and s[0].isalpha():
        s = s[2:]
    # Strip a trailing slash, but keep a lone root ``/``.
    while len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    return s


def _sanitize_basename(basename: str) -> str:
    """Step 4 of the derivation: lowercase, replace each non-alphanumeric char
    with ``-``, collapse consecutive ``-``, strip leading/trailing ``-``."""
    lowered = basename.lower()
    out: list[str] = []
    prev_dash = False
    for ch in lowered:
        if ch.isascii() and ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def workspace_id_from_canonical_path(path: str) -> WorkspaceId:
    """Derive a :data:`WorkspaceId` from an already-OS-canonicalized path.

    This is a **pure string function** — it never touches the filesystem, so the
    pinned fixture ``fixtures/storage/workspace_id_derivation.json`` is
    host-independent. The cross-language parity anchor (#78).

    Algorithm (pinned, byte-identical across languages):

    1. Normalize separators to ``/``. On Windows strip the drive-letter prefix
       (e.g. ``C:``) and convert ``\\`` → ``/``. The input is assumed already
       OS-canonicalized; this function does NOT re-canonicalize.
    2. Build the canonical path string: forward slashes only, NO trailing slash,
       UTF-8.
    3. SHA-256 that string; take the first 8 hex chars (lowercase).
    4. Basename of the canonical path, lowercased; replace each non-alphanumeric
       char with ``-``; collapse consecutive ``-``; strip leading/trailing
       ``-``. Empty basename (root ``/``) → ``root``.
    5. Concatenate ``{sanitized_basename}-{8hex}``.
    """
    canonical = _canonicalize_path_string(path)
    hex8 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    basename = canonical.rsplit("/", 1)[-1]
    sanitized = _sanitize_basename(basename) or "root"
    return WorkspaceId(f"{sanitized}-{hex8}")


# ============================================================================
# ProjectId (#142) — project-scoped durable storage namespace
# ============================================================================


#: A **stable** identifier for a project, used as the durable storage namespace
#: (issue #142). Where :data:`~spore_core.harness.SessionId` is regenerated per
#: Ralph context window (``new_session_id()``), a ``ProjectId`` derived from the
#: workspace root stays constant across windows AND across process restarts —
#: that stability is the whole point: the ``task_list``, plan artifact, Ralph
#: checkpoint, and active-run slot persist under it so a window reset re-reads
#: the prior window's work instead of re-planning from scratch.
#:
#: Form ``{sanitized_basename}-{8_hex_chars}``, lowercased — **identical** to
#: :data:`WorkspaceId`. The two share the exact same pure derivation
#: (:func:`workspace_id_from_canonical_path`); a ``ProjectId`` differs only in
#: that it carries FS-touching constructors (:func:`project_id_from_path` /
#: :func:`project_id_from_cwd`) that canonicalize first. A ``NewType`` over
#: ``str`` (Python conventions: IDs are ``NewType`` aliases).
#:
#: ``/a/b`` vs ``/a_b`` collision policy (RESOLVED): a naive "slashes →
#: underscores" slug would map both ``/a/b`` and ``/a_b`` to the same string.
#: This derivation does NOT collide — it slugs ONLY the final basename and
#: appends the first 8 hex of the SHA-256 of the FULL canonical path string.
#: ``/a/b`` and ``/a_b`` have different canonical strings, hence different
#: hashes, hence distinct ids (``b-662b7b62`` vs ``a-b-328ff01f``). Pinned by
#: ``fixtures/storage/project_id_derivation.json``.
ProjectId = NewType("ProjectId", str)


class ProjectIdError(StorageError):
    """Raised while deriving a :data:`ProjectId` from the live filesystem (#142).

    The pure derivation :func:`project_id_from_canonical_path` is infallible —
    only the FS-touching constructors (:func:`project_id_from_path` /
    :func:`project_id_from_cwd`) can fail, and only because canonicalization
    touched the filesystem (the path does not exist, a component is not a
    directory, a permission error, a broken symlink, …). The originating
    :class:`OSError` is chained via ``__cause__``."""


def project_id_from_canonical_path(path: str) -> ProjectId:
    """Derive a :data:`ProjectId` from an already-OS-canonicalized path.

    **PURE and infallible** — never touches the filesystem. This is the
    cross-language fixture anchor (``fixtures/storage/project_id_derivation.json``);
    it reuses the EXACT same algorithm as
    :func:`workspace_id_from_canonical_path` (``{sanitized_basename}-{8hex}``),
    so the two derivations are byte-identical for the same input — the 8-hex
    SHA-256 suffix already resolves the ``/a/b`` vs ``/a_b`` slug collision
    (distinct canonical strings ⇒ distinct hashes ⇒ distinct ids). Mirrors
    Rust's ``ProjectId::from_canonical_path``."""
    # Reuse the WorkspaceId pure algorithm — do NOT duplicate it.
    return ProjectId(str(workspace_id_from_canonical_path(path)))


def project_id_from_path(path: str | Path) -> ProjectId:
    """Derive a :data:`ProjectId` from a path, **canonicalizing the filesystem
    FIRST** (resolves symlinks, relative components, and macOS
    case-insensitivity) before delegating to the pure
    :func:`project_id_from_canonical_path`. Raises :class:`ProjectIdError` when
    the path cannot be canonicalized. Mirrors Rust's ``ProjectId::from_path``."""
    try:
        canonical = Path(path).resolve(strict=True)
    except OSError as exc:
        raise ProjectIdError(f"project id canonicalization failed: {exc}") from exc
    return project_id_from_canonical_path(str(canonical))


def project_id_from_cwd() -> ProjectId:
    """Derive a :data:`ProjectId` from the current working directory,
    canonicalizing FIRST. Convenience wrapper over :func:`project_id_from_path`
    for binaries that want the process cwd; the harness itself derives from
    ``sandbox.workspace_root()``, NOT process cwd (decision 5). Mirrors Rust's
    ``ProjectId::from_cwd``."""
    try:
        cwd = Path.cwd()
    except OSError as exc:
        raise ProjectIdError(f"project id canonicalization failed: {exc}") from exc
    return project_id_from_path(cwd)


def project_namespace(project_id: ProjectId) -> SessionId:
    """Project a :data:`ProjectId` onto the :class:`RunStore`'s ``session_id``
    string axis — the namespace-reuse seam (#142).

    The :class:`RunStore` protocol is keyed by :class:`~spore_core.harness.SessionId`.
    Rather than widening that protocol, a ``ProjectId`` is projected onto the
    same string axis: the returned :class:`SessionId` is NOT a real session —
    its string IS the derived project id — so durable :meth:`RunStore.get` /
    :meth:`RunStore.put` calls key by the stable project namespace without
    changing the protocol. Ephemeral session-keyed state keeps using the real
    per-window :class:`SessionId`. Mirrors Rust's ``ProjectId::namespace``."""
    return SessionId(str(project_id))


# ============================================================================
# Active-run lifecycle (#142, decision 2)
# ============================================================================


#: Reserved :class:`RunStore` key (under the project namespace) holding the
#: :class:`ActiveRun` slot. The caller owns the lifecycle: start-new vs resume
#: is a deterministic match on the caller-supplied ``run_tag``, NOT
#: instruction-diffing and NOT auto-on-success. The harness stays stateless
#: between runs.
ACTIVE_RUN_KEY = "active_run"

#: Reserved :class:`RunStore` key (under the project namespace) holding the
#: Ralph progress checkpoint (#142, decision 3). Previously lived at
#: ``{workspace_root}/.spore/progress.json``; now in the project-id store so it
#: survives Ralph window resets and process restarts.
RALPH_PROGRESS_KEY = "ralph_progress"

#: Reserved :class:`RunStore` key (under the project namespace) holding the
#: Ralph feature-list checkpoint (#142, decision 3). Mirrors the old
#: ``{workspace_root}/.spore/feature_list.json``.
RALPH_FEATURE_LIST_KEY = "ralph_feature_list"


class ActiveRunStatus(str, Enum):
    """Lifecycle status of the project's active run (#142)."""

    #: The run is live: a window reset under the SAME ``run_tag`` resumes it.
    ACTIVE = "active"
    #: The run was explicitly completed via :func:`complete_active_run`; the slot
    #: is archived. A subsequent start under a NEW tag begins fresh.
    COMPLETED = "completed"


class ActiveRun(BaseModel):
    """The active-run slot persisted under :data:`ACTIVE_RUN_KEY` in the project
    store (#142).

    ``run_tag`` is **caller-supplied** and is the sole start-new-vs-resume
    discriminator (decision 2): a :func:`start_or_resume_active_run` call whose
    tag matches a live slot RESUMES; a different tag (or an absent / completed
    slot) starts FRESH. ``started_at`` is an **injected** timestamp (decision 2
    — no wall-clock nondeterminism) so tests are deterministic."""

    run_tag: str
    started_at: Timestamp
    status: ActiveRunStatus


class ActiveRunDecision(str, Enum):
    """Outcome of :func:`start_or_resume_active_run` (#142): did the slot match
    (resume) or did the call mint a fresh active slot (start-new)?"""

    #: No live slot matched the tag — a fresh ``ACTIVE`` slot was written.
    STARTED_NEW = "started_new"
    #: A live slot under the SAME tag was found — the run reattaches.
    RESUMED = "resumed"


async def load_active_run(run_store: RunStore, project: ProjectId) -> ActiveRun | None:
    """Read the active-run slot for ``project``, or ``None`` if absent /
    unparseable (#142). A malformed slot is treated as "no live run" — the next
    start mints a fresh one rather than erroring. Mirrors Rust's
    ``load_active_run``."""
    ns = project_namespace(project)
    value = await run_store.get(ns, ACTIVE_RUN_KEY)
    if value is None:
        return None
    try:
        return ActiveRun.model_validate(value)
    except ValueError:
        return None


async def start_or_resume_active_run(
    run_store: RunStore,
    project: ProjectId,
    run_tag: str,
    started_at: Timestamp,
) -> ActiveRunDecision:
    """Decide start-new vs resume for ``project`` on the caller-supplied
    ``run_tag`` (#142, decision 2).

    Deterministic: a live (``ACTIVE``) slot under the SAME tag ⇒
    :attr:`ActiveRunDecision.RESUMED` (the slot is left intact); otherwise a
    fresh ``ACTIVE`` slot stamped with the injected ``started_at`` is written and
    :attr:`ActiveRunDecision.STARTED_NEW` is returned. ``started_at`` is injected
    (not read from a clock) so the result is deterministic in tests. Mirrors
    Rust's ``start_or_resume_active_run``."""
    existing = await load_active_run(run_store, project)
    if (
        existing is not None
        and existing.status == ActiveRunStatus.ACTIVE
        and existing.run_tag == run_tag
    ):
        return ActiveRunDecision.RESUMED
    fresh = ActiveRun(run_tag=run_tag, started_at=started_at, status=ActiveRunStatus.ACTIVE)
    await run_store.put(project_namespace(project), ACTIVE_RUN_KEY, fresh.model_dump(mode="json"))
    return ActiveRunDecision.STARTED_NEW


async def complete_active_run(run_store: RunStore, project: ProjectId) -> None:
    """Mark the active run for ``project`` complete (#142, decision 2): flips the
    slot's status to :attr:`ActiveRunStatus.COMPLETED` so the next
    :func:`start_or_resume_active_run` (even under the same tag) starts fresh. A
    no-op when there is no slot to complete. Mirrors Rust's
    ``complete_active_run``."""
    existing = await load_active_run(run_store, project)
    if existing is None:
        return
    completed = existing.model_copy(update={"status": ActiveRunStatus.COMPLETED})
    await run_store.put(
        project_namespace(project), ACTIVE_RUN_KEY, completed.model_dump(mode="json")
    )


# ============================================================================
# Domain protocols
# ============================================================================


@runtime_checkable
class SessionStore(Protocol):
    """Pause/resume lifecycle store. Stores :class:`PausedState` keyed by
    :class:`SessionId`."""

    async def get_session(self, session_id: SessionId) -> PausedState | None: ...

    async def put_session(self, session_id: SessionId, state: PausedState) -> None: ...

    async def delete_session(self, session_id: SessionId) -> None: ...

    async def list_sessions(self) -> list[SessionId]: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Episodic memory store. Append-only log per ``(scope, session)`` (#78).

    A leaf backend is **scope-dumb**: it stores under whatever root it was
    given. The ``scope`` argument is carried for symmetry (and so a single
    backend *could* partition by scope), but the v1 wiring routes each scope to
    its own backend via :class:`CompositeStorageProvider`, so leaf backends
    receive a single scope's traffic. The cross-scope merge
    (:meth:`StorageProvider.get_memories_merged`) lives in the routing layer,
    never in a leaf.

    Known v1 limitation: memory addressing stays :class:`SessionId`-keyed. v2
    should address session-independent / cross-session memory keying — do not
    introduce it here.
    """

    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None: ...

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        """Return the **most-recent ``limit`` entries, newest-first** for
        ``scope``."""
        ...

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        """Cross-scope merged read (#78 R6, #82 D2): **User ∪ Project,
        newest-first by ``timestamp``, NO dedup**. ``Local`` is excluded from the
        merge in v1.

        This is the SINGLE source of the merge algorithm: every concrete
        :class:`MemoryStore` delegates to the shared :func:`merge_memories`
        helper, so the :class:`ScopedMemoryRouter`, leaf backends,
        :class:`NoOpStorageProvider`, and any mock inherit identical semantics —
        there is exactly one merge implementation.
        :meth:`StorageProvider.get_memories_merged` and ``MemoryTool``'s merged
        ``read`` both route here.
        """
        ...


@runtime_checkable
class RunStore(Protocol):
    """Per-run structured state keyed by ``(session_id, key)``. Values are
    opaque JSON blobs — the store does not know the schema; callers own
    serialization."""

    async def get(self, session_id: SessionId, key: str) -> JsonValue | None: ...

    async def put(self, session_id: SessionId, key: str, value: JsonValue) -> None: ...

    async def delete(self, session_id: SessionId, key: str) -> None: ...

    async def list_keys(self, session_id: SessionId) -> list[str]: ...


@runtime_checkable
class ObservabilityStore(Protocol):
    """Append-only span storage. Distinct from the other three: no get-by-key,
    queried by session and time range."""

    async def append_span(self, session_id: SessionId, span: JsonValue) -> None: ...

    async def get_spans(self, session_id: SessionId) -> list[JsonValue]: ...

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]: ...

    async def flush_session(self, session_id: SessionId) -> None: ...


# ============================================================================
# NoOpStorageProvider
# ============================================================================


class NoOpStorageProvider:
    """Silent-discard provider. Reads return ``None`` / ``[]``; writes are
    no-ops. The default for any unconfigured domain. Satisfies all four domain
    protocols structurally."""

    # SessionStore
    async def get_session(self, session_id: SessionId) -> PausedState | None:
        return None

    async def put_session(self, session_id: SessionId, state: PausedState) -> None:
        return None

    async def delete_session(self, session_id: SessionId) -> None:
        return None

    async def list_sessions(self) -> list[SessionId]:
        return []

    # MemoryStore
    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None:
        return None

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        return []

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        return await merge_memories(self, session_id, limit)

    # RunStore
    async def get(self, session_id: SessionId, key: str) -> JsonValue | None:
        return None

    async def put(self, session_id: SessionId, key: str, value: JsonValue) -> None:
        return None

    async def delete(self, session_id: SessionId, key: str) -> None:
        return None

    async def list_keys(self, session_id: SessionId) -> list[str]:
        return []

    # ObservabilityStore
    async def append_span(self, session_id: SessionId, span: JsonValue) -> None:
        return None

    async def get_spans(self, session_id: SessionId) -> list[JsonValue]:
        return []

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]:
        return []

    async def flush_session(self, session_id: SessionId) -> None:
        return None


# ============================================================================
# StorageProvider
# ============================================================================


class StorageProvider:
    """A composed persistence layer: four independent domain stores. Built
    either from a single backend (cloned into all four slots via
    :meth:`single`) or per-domain via :class:`CompositeStorageProvider`."""

    def __init__(
        self,
        session: SessionStore,
        memory: MemoryStore,
        run: RunStore,
        observability: ObservabilityStore,
    ) -> None:
        self._session = session
        self._memory = memory
        self._run = run
        self._observability = observability

    @classmethod
    def single(cls, provider: object) -> StorageProvider:
        """Place a single concrete provider implementing all four domain
        protocols into all four slots."""
        return cls(provider, provider, provider, provider)  # type: ignore[arg-type]

    @classmethod
    def no_op(cls) -> StorageProvider:
        """All-no-op provider. The default when ``.storage(...)`` is never set."""
        return cls.single(NoOpStorageProvider())

    def session(self) -> SessionStore:
        return self._session

    def memory(self) -> MemoryStore:
        return self._memory

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        """Merged memory read across scopes (#78 R6): **User ∪ Project,
        newest-first by ``timestamp``, NO dedup**. ``Local`` is excluded from
        the merge in v1.

        Routes through the memory slot — when built via
        :class:`CompositeStorageProvider` that slot is a
        :class:`ScopedMemoryRouter` that fans out to the per-scope backends; for
        ``single``/``__init__`` the one backend serves both scopes (keyed by
        scope) and merges identically. The merge always lives in the routing
        layer, never in a leaf backend.

        Delegates to the single :meth:`MemoryStore.get_memories_merged` (#82 D2)
        so there is exactly ONE merge implementation.
        """
        return await self._memory.get_memories_merged(session_id, limit)

    def run(self) -> RunStore:
        return self._run

    def observability(self) -> ObservabilityStore:
        return self._observability


# ============================================================================
# InMemoryStorageProvider
# ============================================================================


class InMemoryStorageProvider:
    """Lock-guarded in-memory provider. Used in tests and ephemeral runs.
    Satisfies all four domain protocols."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[SessionId, PausedState] = {}
        # Memory is keyed by (scope, session_id) (#78): a single backend can hold
        # multiple scopes, though the v1 composite wiring routes one scope per
        # backend.
        self._memory: dict[tuple[StorageScope, SessionId], list[MemoryEntry]] = {}
        self._run: dict[tuple[SessionId, str], JsonValue] = {}
        self._spans: dict[SessionId, list[JsonValue]] = {}

    # SessionStore
    async def get_session(self, session_id: SessionId) -> PausedState | None:
        with self._lock:
            return self._sessions.get(session_id)

    async def put_session(self, session_id: SessionId, state: PausedState) -> None:
        with self._lock:
            self._sessions[session_id] = state

    async def delete_session(self, session_id: SessionId) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    async def list_sessions(self) -> list[SessionId]:
        with self._lock:
            return sorted(self._sessions.keys())

    # MemoryStore
    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None:
        with self._lock:
            self._memory.setdefault((scope, session_id), []).append(entry)

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        with self._lock:
            entries = list(self._memory.get((scope, session_id), []))
        return _most_recent_newest_first(entries, limit)

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        return await merge_memories(self, session_id, limit)

    # RunStore
    async def get(self, session_id: SessionId, key: str) -> JsonValue | None:
        with self._lock:
            return self._run.get((session_id, key))

    async def put(self, session_id: SessionId, key: str, value: JsonValue) -> None:
        with self._lock:
            self._run[(session_id, key)] = value

    async def delete(self, session_id: SessionId, key: str) -> None:
        with self._lock:
            self._run.pop((session_id, key), None)

    async def list_keys(self, session_id: SessionId) -> list[str]:
        with self._lock:
            keys = [k for (s, k) in self._run if s == session_id]
        return sorted(keys)

    # ObservabilityStore
    async def append_span(self, session_id: SessionId, span: JsonValue) -> None:
        with self._lock:
            self._spans.setdefault(session_id, []).append(span)

    async def get_spans(self, session_id: SessionId) -> list[JsonValue]:
        with self._lock:
            return list(self._spans.get(session_id, []))

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]:
        # The in-memory span store does not roll up SessionMetrics — that is the
        # ObservabilityProvider's job. Storage-only query returns empty.
        return []

    async def flush_session(self, session_id: SessionId) -> None:
        return None


# ============================================================================
# FileSystemStorageProvider
# ============================================================================


def _atomic_write(target: Path, data: bytes) -> None:
    """Atomic write-rename: ensure parent dir, write full bytes to a sibling
    ``{target}.tmp``, flush + ``fsync``, then ``os.replace``. On any failure the
    ``.tmp`` is removed so no partial sidecar is left behind. Byte-identical
    algorithm across all four languages."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        # Best-effort cleanup; leave no leftover .tmp.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise StorageIoError(str(exc)) from exc


def _append_jsonl(path: Path, value: JsonValue) -> None:
    """Append one JSONL line (the value plus a trailing ``\\n``), flushing the
    handle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        line = json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise StorageSerializationError(str(exc)) from exc
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
    except OSError as exc:
        raise StorageIoError(str(exc)) from exc


def _read_jsonl(path: Path) -> list[JsonValue]:
    """Read every non-empty JSONL line from ``path``. Missing file → empty
    list."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise StorageIoError(str(exc)) from exc
    out: list[JsonValue] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise StorageSerializationError(str(exc)) from exc
    return out


class FileSystemStorageProvider:
    """Disk-backed provider rooted at ``root``. Satisfies all four domain
    protocols. Layout mirrors ``.spore/``; non-append writes are
    atomic-write-rename, memory / observability are append-only JSONL."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _session_dir(self, session_id: SessionId) -> Path:
        return self._root / "sessions" / str(session_id)

    def _state_path(self, session_id: SessionId) -> Path:
        return self._session_dir(session_id) / "state.json"

    def _run_dir(self, session_id: SessionId) -> Path:
        return self._session_dir(session_id) / "run"

    def _run_path(self, session_id: SessionId, key: str) -> Path:
        return self._run_dir(session_id) / f"{key}.json"

    def _memory_path(self, session_id: SessionId) -> Path:
        return self._session_dir(session_id) / "memory.jsonl"

    def _trace_path(self, session_id: SessionId) -> Path:
        return self._session_dir(session_id) / "trace.jsonl"

    # SessionStore
    async def get_session(self, session_id: SessionId) -> PausedState | None:
        path = self._state_path(session_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc
        try:
            return PausedState.model_validate_json(raw)
        except ValueError as exc:
            raise StorageSerializationError(str(exc)) from exc

    async def put_session(self, session_id: SessionId, state: PausedState) -> None:
        try:
            data = state.model_dump_json().encode("utf-8")
        except ValueError as exc:
            raise StorageSerializationError(str(exc)) from exc
        _atomic_write(self._state_path(session_id), data)

    async def delete_session(self, session_id: SessionId) -> None:
        try:
            self._state_path(session_id).unlink()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc

    async def list_sessions(self) -> list[SessionId]:
        sessions_dir = self._root / "sessions"
        if not sessions_dir.exists():
            return []
        out: list[SessionId] = []
        try:
            for entry in sessions_dir.iterdir():
                if (entry / "state.json").exists():
                    out.append(SessionId(entry.name))
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc
        return sorted(out)

    # MemoryStore
    #
    # FS is **scope-dumb** (#78): the user-scope backend is pointed at the
    # already-partitioned ``{user_root}/projects/{workspace_id}`` at
    # construction. The provider just writes under whatever root it was given;
    # ``scope`` is ignored at the leaf.
    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None:
        value = entry.model_dump(mode="json")
        _append_jsonl(self._memory_path(session_id), value)

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        values = _read_jsonl(self._memory_path(session_id))
        try:
            entries = [MemoryEntry.model_validate(v) for v in values]
        except ValueError as exc:
            raise StorageSerializationError(str(exc)) from exc
        return _most_recent_newest_first(entries, limit)

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        return await merge_memories(self, session_id, limit)

    # RunStore
    async def get(self, session_id: SessionId, key: str) -> JsonValue | None:
        path = self._run_path(session_id, key)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StorageSerializationError(str(exc)) from exc

    async def put(self, session_id: SessionId, key: str, value: JsonValue) -> None:
        try:
            data = json.dumps(value, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise StorageSerializationError(str(exc)) from exc
        _atomic_write(self._run_path(session_id, key), data)

    async def delete(self, session_id: SessionId, key: str) -> None:
        try:
            self._run_path(session_id, key).unlink()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc

    async def list_keys(self, session_id: SessionId) -> list[str]:
        run_dir = self._run_dir(session_id)
        if not run_dir.exists():
            return []
        out: list[str] = []
        try:
            for entry in run_dir.iterdir():
                if entry.name.endswith(".json"):
                    out.append(entry.name[: -len(".json")])
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc
        return sorted(out)

    # ObservabilityStore
    async def append_span(self, session_id: SessionId, span: JsonValue) -> None:
        _append_jsonl(self._trace_path(session_id), span)

    async def get_spans(self, session_id: SessionId) -> list[JsonValue]:
        return _read_jsonl(self._trace_path(session_id))

    async def get_sessions(
        self,
        since: Timestamp,
        domain: str | None = None,
        outcome: SessionOutcome | None = None,
    ) -> list[SessionMetrics]:
        # SessionMetrics roll-up is owned by the ObservabilityProvider, not the
        # raw on-disk span store. Storage-only query returns empty.
        return []

    async def flush_session(self, session_id: SessionId) -> None:
        try:
            session_dir = self._session_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / ".flushed").touch()
        except OSError as exc:
            raise StorageIoError(str(exc)) from exc


# ============================================================================
# CompositeStorageProvider
# ============================================================================


class CompositeStorageProvider:
    """Builder that routes each domain to its own backend — and, for the memory
    domain, each :class:`StorageScope` to its own backend (#78) — filling any
    unset slot with :class:`NoOpStorageProvider` on :meth:`build`.

    Only the ``memory`` domain varies by scope. ``session``, ``run`` and
    ``observability`` are scope-flat — scope is wiring-only for them.

    Example::

        CompositeStorageProvider()                              \\
            .session(fs(user_root))                             \\
            .run(fs(user_root))                                 \\
            .observability(fs(user_root))                       \\
            .memory(StorageScope.USER, fs(user_workspace_root)) \\
            .memory(StorageScope.PROJECT, fs(project_root))     \\
            .memory(StorageScope.LOCAL, NoOpStorageProvider())  \\
            .build()
    """

    def __init__(self) -> None:
        self._session: SessionStore | None = None
        self._memory: dict[StorageScope, MemoryStore] = {}
        self._run: RunStore | None = None
        self._observability: ObservabilityStore | None = None

    def session(self, store: SessionStore) -> CompositeStorageProvider:
        self._session = store
        return self

    def memory(self, scope: StorageScope, store: MemoryStore) -> CompositeStorageProvider:
        """Configure the memory backend for one :class:`StorageScope`.
        Unconfigured ``(memory, scope)`` pairs fall back to
        :class:`NoOpStorageProvider` on :meth:`build` (#78 R7/R11 — ``Local``
        may be wired to no-op in v1)."""
        self._memory[scope] = store
        return self

    def run(self, store: RunStore) -> CompositeStorageProvider:
        self._run = store
        return self

    def observability(self, store: ObservabilityStore) -> CompositeStorageProvider:
        self._observability = store
        return self

    def build(self) -> StorageProvider:
        """Build a :class:`StorageProvider`, filling each unset domain — and each
        unset ``(memory, scope)`` pair — with a :class:`NoOpStorageProvider`."""
        no_op = NoOpStorageProvider()
        return StorageProvider(
            session=self._session if self._session is not None else no_op,
            memory=ScopedMemoryRouter(dict(self._memory)),
            run=self._run if self._run is not None else no_op,
            observability=(self._observability if self._observability is not None else no_op),
        )


# ============================================================================
# ScopedMemoryRouter (#78) — the (memory, scope) routing layer
# ============================================================================


class ScopedMemoryRouter:
    """Routes :class:`MemoryStore` traffic to a per-:class:`StorageScope`
    backend, filling unconfigured scopes with :class:`NoOpStorageProvider`. Leaf
    backends stay scope-dumb; the cross-scope merge lives one level up in
    :meth:`StorageProvider.get_memories_merged`.

    This is the storage provider's memory slot when built via
    :class:`CompositeStorageProvider`: :meth:`StorageProvider.memory` returns
    this router, so a caller that passes a ``scope`` is routed to the right
    backend. Satisfies the :class:`MemoryStore` protocol structurally.
    """

    def __init__(self, by_scope: dict[StorageScope, MemoryStore]) -> None:
        self._by_scope = by_scope
        self._noop = NoOpStorageProvider()

    def _backend(self, scope: StorageScope) -> MemoryStore:
        """The backend for ``scope``, or the shared no-op if unconfigured."""
        return self._by_scope.get(scope, self._noop)

    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None:
        await self._backend(scope).append_memory(scope, session_id, entry)

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        return await self._backend(scope).get_memories(scope, session_id, limit)

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        # Fans out to the per-scope backends via this router's own
        # ``get_memories`` (each scope routes to its backend), then merges once
        # via the shared helper (#82 D2). The merge lives here in the routing
        # layer, never in a leaf backend.
        return await merge_memories(self, session_id, limit)


# ============================================================================
# Shared helpers
# ============================================================================


def _most_recent_newest_first(items: list[Any], limit: int) -> list[Any]:
    """Return the most-recent ``limit`` items, newest-first, given a list in
    append (oldest-first) order."""
    reversed_items = list(reversed(items))
    if limit < 0:
        limit = 0
    return reversed_items[:limit]


def _merge_newest_first(entries: list[MemoryEntry], limit: int) -> list[MemoryEntry]:
    """Merge step for the cross-scope memory read (#78 R6): sort newest-first by
    ``timestamp`` and truncate to ``limit``. **No dedup** — identical-content
    entries are all retained. A *stable* sort preserves the input order among
    equal timestamps, which keeps the merge deterministic cross-language."""
    ordered = sorted(entries, key=lambda e: str(e.timestamp), reverse=True)
    if limit < 0:
        limit = 0
    return ordered[:limit]


async def merge_memories(
    store: MemoryStore, session_id: SessionId, limit: int
) -> list[MemoryEntry]:
    """The SINGLE cross-scope merge implementation (#78 R6, #82 D2): **User ∪
    Project, newest-first by ``timestamp``, NO dedup**; ``Local`` excluded.

    Reads ``User`` then ``Project`` from ``store`` and merges via
    :func:`_merge_newest_first`. Every :class:`MemoryStore` implementation's
    :meth:`~MemoryStore.get_memories_merged` delegates here, so there are no
    divergent copies of the merge. Defined as a free function (not a Protocol
    default — Python Protocols carry no inherited bodies) so it composes over any
    structural :class:`MemoryStore`."""
    scope = _storage_scope()
    combined = list(await store.get_memories(scope.USER, session_id, limit))
    combined.extend(await store.get_memories(scope.PROJECT, session_id, limit))
    return _merge_newest_first(combined, limit)


# ============================================================================
# OTLP endpoint parsing (cross-language ground truth — see fan-out refactor)
# ============================================================================


def parse_otlp_endpoints(raw: str) -> list[str]:
    """Parse the comma-separated ``SPORE_OTLP_ENDPOINT`` value: ``split(',')``,
    trim each segment, drop empty segments. This is the single most important
    cross-language fixture (``fixtures/storage/otlp_endpoints_parse.json``) and
    MUST be byte-identical in every language."""
    return [segment.strip() for segment in raw.split(",") if segment.strip()]


__all__ = [
    "ACTIVE_RUN_KEY",
    "RALPH_FEATURE_LIST_KEY",
    "RALPH_PROGRESS_KEY",
    "ActiveRun",
    "ActiveRunDecision",
    "ActiveRunStatus",
    "CompositeStorageProvider",
    "FileSystemStorageProvider",
    "InMemoryStorageProvider",
    "JsonValue",
    "MemoryEntry",
    "MemoryStore",
    "NoOpStorageProvider",
    "ObservabilityStore",
    "ProjectId",
    "ProjectIdError",
    "RunStore",
    "ScopedMemoryRouter",
    "SessionStore",
    "StorageBackendError",
    "StorageError",
    "StorageIoError",
    "StorageNotFoundError",
    "StorageProvider",
    "StorageScope",
    "StorageSerializationError",
    "WorkspaceId",
    "complete_active_run",
    "load_active_run",
    "merge_memories",
    "parse_otlp_endpoints",
    "project_id_from_canonical_path",
    "project_id_from_cwd",
    "project_id_from_path",
    "project_namespace",
    "start_or_resume_active_run",
    "workspace_id_from_canonical_path",
]
