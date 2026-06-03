"""``MarkdownMemoryProvider`` — a :class:`MemoryStore` that persists agent memory
to a single human-readable ``memory.md`` file on disk.

What this demonstrates
======================
The storage seam. The harness is **stateless** — every byte of durable state
lives behind a :class:`~spore_core.StorageProvider`. Memory is one of its four
domains (:class:`~spore_core.storage.MemoryStore`). This module implements
*only* that domain and composes it with :class:`~spore_core.NoOpStorageProvider`
for the other three (session / run / observability). That composed provider is
what ``main.py`` hands to ``HarnessBuilder.storage(...)``; the harness then
threads ``storage.memory()`` into the built-in ``memory`` tool's context on
every run. No custom harness plumbing — the seam is the whole integration
surface.

The seam
========
:class:`~spore_core.storage.MemoryStore` is the swap point. The built-in
:class:`~spore_core.FileSystemStorageProvider` persists memory as a JSONL log;
this provider persists the *same* :class:`~spore_core.MemoryEntry` values to
readable markdown instead. Same protocol, same agent, same tool — different
on-disk shape. Anything satisfying ``MemoryStore`` slots in here.

On-disk format (round-trips exactly)
====================================
Each :class:`~spore_core.MemoryEntry` is one markdown block. The header line
carries the round-trip fields; the body is the content::

    ## [project] [project-ironwood] 2026-06-02T12:00:00Z — assistant

    Postgres 15 is the system of record.

``append_memory`` writes such a block; ``get_memories`` parses them back,
filters by scope + session, sorts newest-first by timestamp, and takes
``limit``. A hand-edited file (extra prose, blank lines, reordered blocks) is
tolerated: anything that is not a recognized ``## [scope] [session] timestamp —
role`` header is treated as body for the preceding entry, and leading prose
before the first header is ignored.

Pinned-session-id requirement (read this)
=========================================
Memory is keyed by :class:`~spore_core.SessionId`. The ``memory`` tool always
uses the run's session id. For Run 2 (recall) to read Run 1's (store) memories,
**both runs MUST use the SAME ``SessionId``** — see ``main.py``, which pins
``SessionId("project-ironwood")`` rather than ``new_session_id()``. This
provider also stores the session id in each header so a single ``memory.md`` can
hold multiple sessions without cross-talk.
"""

from __future__ import annotations

import threading
from pathlib import Path

from spore_core import (
    MemoryEntry,
    NoOpStorageProvider,
    SessionId,
    StorageProvider,
    StorageScope,
)
from spore_core.memory import Timestamp
from spore_core.storage import merge_memories

_HEADER = (
    "# Agent Memory\n\nHuman-readable working memory for this agent. "
    "Each `##` block below is one remembered entry.\n"
)

#: The em-dash separator between timestamp and role: space, U+2014, space.
_DASH = " — "


def _scope_token(scope: StorageScope) -> str:
    """How a :class:`StorageScope` is spelled in a header line."""
    return scope.value


def _parse_scope_token(token: str) -> StorageScope | None:
    try:
        return StorageScope(token.strip())
    except ValueError:
        return None


def _render_block(scope: StorageScope, session_id: SessionId, entry: MemoryEntry) -> str:
    """Render one entry as a markdown block (header + blank line + body). The
    ``session_id`` is encoded so one file can hold multiple sessions."""
    return (
        f"## [{_scope_token(scope)}] [{session_id}] "
        f"{entry.timestamp}{_DASH}{entry.role}\n\n{entry.content.rstrip()}\n"
    )


class _ParsedBlock:
    """A parsed entry plus the scope + session it was filed under."""

    __slots__ = ("scope", "session", "entry")

    def __init__(self, scope: StorageScope, session: str, entry: MemoryEntry) -> None:
        self.scope = scope
        self.session = session
        self.entry = entry


def _parse_header(line: str) -> tuple[StorageScope, str, str, str] | None:
    """Parse a header line ``## [scope] [session] timestamp — role``. Returns
    ``None`` for any line that is not a recognized header (so prose and
    hand-edits are tolerated)."""
    if not line.startswith("## ["):
        return None
    rest = line[len("## [") :]
    scope_str, sep, rest = rest.partition("] [")
    if not sep:
        return None
    scope = _parse_scope_token(scope_str)
    if scope is None:
        return None
    session, sep, rest = rest.partition("] ")
    if not sep:
        return None
    ts, sep, role = rest.partition(_DASH)
    if not sep:
        return None
    ts = ts.strip()
    role = role.strip()
    if not ts or not role:
        return None
    return scope, session.strip(), ts, role


def _parse_file(contents: str) -> list[_ParsedBlock]:
    """Parse the whole file into blocks. Body lines accumulate under the most
    recent header; text before the first header is discarded. A ``## `` line
    that is not a valid entry header (a human-added subheading) closes the
    current block and is otherwise ignored."""
    blocks: list[_ParsedBlock] = []
    current: tuple[StorageScope, str, str, str] | None = None
    body: list[str] = []

    def _finish() -> None:
        if current is None:
            return
        scope, session, ts, role = current
        content = "\n".join(body).strip()
        blocks.append(
            _ParsedBlock(
                scope, session, MemoryEntry(role=role, content=content, timestamp=Timestamp(ts))
            )
        )

    for line in contents.splitlines():
        header = _parse_header(line)
        if header is not None:
            _finish()
            current = header
            body = []
        elif line.startswith("## "):
            _finish()
            current = None
            body = []
        elif current is not None:
            body.append(line)
        # else: prose before the first header — ignored.
    _finish()
    return blocks


class MarkdownMemoryProvider:
    """A :class:`~spore_core.storage.MemoryStore` backed by a single
    human-readable ``memory.md`` file.

    The lock serializes the read-modify-write of :meth:`append_memory` so
    concurrent appends from the harness (which dispatches the ``memory`` tool
    sequentially anyway) never interleave a partial write. Satisfies the
    ``MemoryStore`` protocol structurally — it does not inherit it.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def into_storage_provider(self) -> StorageProvider:
        """Compose this provider into a full :class:`~spore_core.StorageProvider`:
        the real ``MemoryStore`` for the memory domain,
        :class:`~spore_core.NoOpStorageProvider` for the other three. This is
        exactly what the example hands to the harness."""
        no_op = NoOpStorageProvider()
        return StorageProvider(session=no_op, memory=self, run=no_op, observability=no_op)

    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None:
        """Read-modify-write: load the existing text, append a new block. A
        missing file is seeded with the human-readable header first."""
        with self._lock:
            try:
                existing = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                existing = _HEADER
            if not existing.endswith("\n"):
                existing += "\n"
            existing += "\n" + _render_block(scope, session_id, entry)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(existing, encoding="utf-8")

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        """Parse the file, filter by ``scope`` + ``session``, sort newest-first
        by timestamp, take ``limit``. Missing file → ``[]``."""
        try:
            contents = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        entries = [
            block.entry
            for block in _parse_file(contents)
            if block.scope == scope and block.session == str(session_id)
        ]
        # Newest-first by timestamp. RFC-3339 strings sort lexically; Python's
        # sort is stable, so ties keep insertion (append) order.
        entries.sort(key=lambda e: str(e.timestamp), reverse=True)
        if limit < 0:
            limit = 0
        return entries[:limit]

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        """Cross-scope merged read — delegates to the single shared
        :func:`~spore_core.storage.merge_memories` implementation, never
        reimplemented here."""
        return await merge_memories(self, session_id, limit)
