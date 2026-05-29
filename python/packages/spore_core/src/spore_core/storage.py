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

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .errors import SporeError
from .guide_registry import SessionOutcome
from .harness import PausedState, SessionId
from .memory import Timestamp
from .observability import SessionMetrics

# A free-form opaque JSON value handled by the storage layer.
JsonValue = Any


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
    """Episodic memory store. Append-only log per session."""

    async def append_memory(self, session_id: SessionId, entry: MemoryEntry) -> None: ...

    async def get_memories(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        """Return the **most-recent ``limit`` entries, newest-first**."""
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
    async def append_memory(self, session_id: SessionId, entry: MemoryEntry) -> None:
        return None

    async def get_memories(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        return []

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
        self._memory: dict[SessionId, list[MemoryEntry]] = {}
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
    async def append_memory(self, session_id: SessionId, entry: MemoryEntry) -> None:
        with self._lock:
            self._memory.setdefault(session_id, []).append(entry)

    async def get_memories(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        with self._lock:
            entries = list(self._memory.get(session_id, []))
        return _most_recent_newest_first(entries, limit)

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
    async def append_memory(self, session_id: SessionId, entry: MemoryEntry) -> None:
        value = entry.model_dump(mode="json")
        _append_jsonl(self._memory_path(session_id), value)

    async def get_memories(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        values = _read_jsonl(self._memory_path(session_id))
        try:
            entries = [MemoryEntry.model_validate(v) for v in values]
        except ValueError as exc:
            raise StorageSerializationError(str(exc)) from exc
        return _most_recent_newest_first(entries, limit)

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
    """Builder that routes each domain to its own backend, filling any unset
    domain with :class:`NoOpStorageProvider` on :meth:`build`."""

    def __init__(self) -> None:
        self._session: SessionStore | None = None
        self._memory: MemoryStore | None = None
        self._run: RunStore | None = None
        self._observability: ObservabilityStore | None = None

    def session(self, store: SessionStore) -> CompositeStorageProvider:
        self._session = store
        return self

    def memory(self, store: MemoryStore) -> CompositeStorageProvider:
        self._memory = store
        return self

    def run(self, store: RunStore) -> CompositeStorageProvider:
        self._run = store
        return self

    def observability(self, store: ObservabilityStore) -> CompositeStorageProvider:
        self._observability = store
        return self

    def build(self) -> StorageProvider:
        """Build a :class:`StorageProvider`, filling each unset domain with a
        :class:`NoOpStorageProvider`."""
        no_op = NoOpStorageProvider()
        return StorageProvider(
            session=self._session if self._session is not None else no_op,
            memory=self._memory if self._memory is not None else no_op,
            run=self._run if self._run is not None else no_op,
            observability=(self._observability if self._observability is not None else no_op),
        )


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
    "CompositeStorageProvider",
    "FileSystemStorageProvider",
    "InMemoryStorageProvider",
    "JsonValue",
    "MemoryEntry",
    "MemoryStore",
    "NoOpStorageProvider",
    "ObservabilityStore",
    "RunStore",
    "SessionStore",
    "StorageBackendError",
    "StorageError",
    "StorageIoError",
    "StorageNotFoundError",
    "StorageProvider",
    "StorageSerializationError",
    "parse_otlp_endpoints",
]
