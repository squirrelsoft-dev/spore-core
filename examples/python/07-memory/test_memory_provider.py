"""Unit tests for ``MarkdownMemoryProvider``.

Each behavior the spec pins has a test: round-trip of a single entry, multiline
content, scope filtering, session filtering, newest-first ordering, the
``limit`` take, a missing file reading as empty, tolerance of a hand-edited
file, and composition into a full :class:`~spore_core.StorageProvider` (memory
slot real, the other three no-op).
"""

from __future__ import annotations

from pathlib import Path

from spore_core import MemoryEntry, SessionId, StorageScope
from spore_core.memory import Timestamp

from memory_provider import MarkdownMemoryProvider

SID = SessionId("s1")


def _entry(role: str, content: str, ts: str) -> MemoryEntry:
    return MemoryEntry(role=role, content=content, timestamp=Timestamp(ts))


async def test_missing_file_reads_empty(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    got = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert got == []


async def test_append_then_get_roundtrips_the_entry(tmp_path: Path) -> None:
    path = tmp_path / "memory.md"
    provider = MarkdownMemoryProvider(path)
    await provider.append_memory(
        StorageScope.PROJECT,
        SID,
        _entry("assistant", "Postgres is the system of record.", "2026-06-02T10:00:00Z"),
    )

    got = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert len(got) == 1
    assert got[0].role == "assistant"
    assert got[0].content == "Postgres is the system of record."
    assert str(got[0].timestamp) == "2026-06-02T10:00:00Z"

    # The artifact is real, readable markdown on disk.
    raw = path.read_text(encoding="utf-8")
    assert "## [project] [s1] 2026-06-02T10:00:00Z — assistant" in raw
    assert "Postgres is the system of record." in raw


async def test_multiline_content_roundtrips(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    await provider.append_memory(
        StorageScope.PROJECT,
        SID,
        _entry("assistant", "line one\nline two\n\nline four", "2026-06-02T10:00:00Z"),
    )
    got = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert got[0].content == "line one\nline two\n\nline four"


async def test_scope_filtering_isolates_scopes(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    await provider.append_memory(
        StorageScope.PROJECT, SID, _entry("user", "proj", "2026-06-02T10:00:00Z")
    )
    await provider.append_memory(
        StorageScope.USER, SID, _entry("user", "usr", "2026-06-02T10:00:01Z")
    )

    proj = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert [e.content for e in proj] == ["proj"]

    usr = await provider.get_memories(StorageScope.USER, SID, 50)
    assert [e.content for e in usr] == ["usr"]


async def test_session_filtering_isolates_sessions(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    a = SessionId("alpha")
    b = SessionId("beta")
    await provider.append_memory(
        StorageScope.PROJECT, a, _entry("user", "from-alpha", "2026-06-02T10:00:00Z")
    )
    await provider.append_memory(
        StorageScope.PROJECT, b, _entry("user", "from-beta", "2026-06-02T10:00:01Z")
    )

    got_a = await provider.get_memories(StorageScope.PROJECT, a, 50)
    assert [e.content for e in got_a] == ["from-alpha"]


async def test_get_returns_newest_first(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    # Appended out of timestamp order on purpose.
    await provider.append_memory(
        StorageScope.PROJECT, SID, _entry("user", "middle", "2026-06-02T11:00:00Z")
    )
    await provider.append_memory(
        StorageScope.PROJECT, SID, _entry("user", "oldest", "2026-06-02T10:00:00Z")
    )
    await provider.append_memory(
        StorageScope.PROJECT, SID, _entry("user", "newest", "2026-06-02T12:00:00Z")
    )

    got = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert [e.content for e in got] == ["newest", "middle", "oldest"]


async def test_limit_takes_most_recent(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    for i in range(5):
        await provider.append_memory(
            StorageScope.PROJECT, SID, _entry("user", f"e{i}", f"2026-06-02T10:00:0{i}Z")
        )
    got = await provider.get_memories(StorageScope.PROJECT, SID, 2)
    assert [e.content for e in got] == ["e4", "e3"]


async def test_tolerates_hand_edited_file(tmp_path: Path) -> None:
    path = tmp_path / "memory.md"
    # A file a human authored/edited: prose before the first header, an extra
    # heading, blank lines, and a normal entry block.
    path.write_text(
        "# My Notes\n\n"
        "Some rambling prose that is not an entry.\n\n"
        "## [project] [s1] 2026-06-02T09:00:00Z — user\n\n"
        "Hand-written fact about Ironwood.\n\n"
        "## A non-entry heading the human added\n\n"
        "more prose\n",
        encoding="utf-8",
    )
    provider = MarkdownMemoryProvider(path)
    got = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert len(got) == 1
    assert got[0].content == "Hand-written fact about Ironwood."

    # And we can still append on top of the hand-edited file.
    await provider.append_memory(
        StorageScope.PROJECT, SID, _entry("assistant", "appended", "2026-06-02T10:00:00Z")
    )
    got = await provider.get_memories(StorageScope.PROJECT, SID, 50)
    assert len(got) == 2
    assert got[0].content == "appended"  # newest-first


async def test_composes_into_storage_provider_memory_slot(tmp_path: Path) -> None:
    provider = MarkdownMemoryProvider(tmp_path / "memory.md")
    storage = provider.into_storage_provider()

    # The memory slot is the markdown provider; round-trips through it.
    await storage.memory().append_memory(
        StorageScope.PROJECT, SID, _entry("user", "via-seam", "2026-06-02T10:00:00Z")
    )
    got = await storage.memory().get_memories(StorageScope.PROJECT, SID, 50)
    assert [e.content for e in got] == ["via-seam"]

    # The other three domains are no-ops: a run read returns nothing.
    assert await storage.run().get(SID, "k") is None
