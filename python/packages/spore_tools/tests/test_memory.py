"""Tests for the ``memory`` tool (#82, storage seam #78).

Mirrors the unit + fixture-replay tests in
``rust/crates/spore-core/src/tools/memory.rs``. Outcomes MUST be byte-identical
across all four languages — the shared fixtures
(``fixtures/tools/memory.json`` and ``fixtures/storage/memory_scoped_merge.json``)
are ground truth.

The merged-read step in ``fixtures/tools/memory.json`` carries an ``unordered``
flag: tool-stamped write timestamps can collide within the same wall-clock
second, so that step compares contents as a MULTISET. Strict newest-first
ordering is verified separately via the explicit-timestamp
``memory_scoped_merge.json`` replay.
"""

from __future__ import annotations

import json
from pathlib import Path

from spore_core.harness import (
    BaseSandboxProvider,
    SandboxViolation,
    SessionId,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import ToolCall
from spore_core.prompt_assembly import StorageScope
from spore_core.storage import (
    CompositeStorageProvider,
    InMemoryStorageProvider,
    MemoryEntry,
    MemoryStore,
    StorageBackendError,
)
from spore_core.tool_registry import ToolContext
from spore_tools.tools.memory import LOCAL_REJECTED_MESSAGE, MemoryTool

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOL_FIXTURES = REPO_ROOT / "fixtures" / "tools"
STORAGE_FIXTURES = REPO_ROOT / "fixtures" / "storage"


# ============================================================================
# helpers
# ============================================================================


class _AllowAllSandbox(BaseSandboxProvider):
    """Permissive sandbox — the tool never touches the filesystem."""

    async def validate(self, call: ToolCall) -> SandboxViolation | None:
        return None


class _FailingMemoryStore:
    """A MemoryStore that always fails, to prove storage errors map to a
    recoverable tool error. Satisfies the :class:`MemoryStore` protocol."""

    async def append_memory(
        self, scope: StorageScope, session_id: SessionId, entry: MemoryEntry
    ) -> None:
        raise StorageBackendError("boom")

    async def get_memories(
        self, scope: StorageScope, session_id: SessionId, limit: int
    ) -> list[MemoryEntry]:
        raise StorageBackendError("boom")

    async def get_memories_merged(self, session_id: SessionId, limit: int) -> list[MemoryEntry]:
        raise StorageBackendError("boom")


def _ctx_with(memory_store: MemoryStore, session: str) -> ToolContext:
    return ToolContext(
        session_id=SessionId(session),
        run_store=InMemoryStorageProvider(),
        memory_store=memory_store,
    )


def _in_memory_ctx() -> ToolContext:
    return _ctx_with(InMemoryStorageProvider(), "test-session")


def _scoped_ctx(session: str = "s") -> tuple[ToolContext, object]:
    """A ctx whose memory store is a scope-routing composite provider seeded with
    User + Project + Local backends. Returns ``(ctx, memory_store)``."""
    provider = (
        CompositeStorageProvider()
        .memory(StorageScope.USER, InMemoryStorageProvider())
        .memory(StorageScope.PROJECT, InMemoryStorageProvider())
        .memory(StorageScope.LOCAL, InMemoryStorageProvider())
        .build()
    )
    memory_store = provider.memory()
    ctx = ToolContext(
        session_id=SessionId(session),
        run_store=InMemoryStorageProvider(),
        memory_store=memory_store,
    )
    return ctx, memory_store


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=MemoryTool.NAME, input=input_)


def _entries(out: ToolOutputSuccess) -> list[dict]:
    return json.loads(out.content)


def _contents(out: ToolOutputSuccess) -> list[str]:
    return [e["content"] for e in _entries(out)]


SB = _AllowAllSandbox()


# ============================================================================
# R1 + R2: write→read roundtrip; write returns the serialized entry
# ============================================================================


async def test_write_then_read_roundtrip() -> None:
    ctx = _in_memory_ctx()
    tool = MemoryTool()

    w = await tool.execute(
        _call({"operation": "write", "scope": "user", "role": "user", "content": "hello"}),
        SB,
        ctx,
    )
    assert isinstance(w, ToolOutputSuccess)
    written = json.loads(w.content)
    assert written["role"] == "user"
    assert written["content"] == "hello"
    assert written["metadata"] == {}  # R4 default {}

    r = await tool.execute(_call({"operation": "read", "scope": "user"}), SB, ctx)
    assert isinstance(r, ToolOutputSuccess)
    entries = _entries(r)
    assert len(entries) == 1
    assert entries[0]["content"] == "hello"


# ============================================================================
# R4: metadata stored verbatim
# ============================================================================


async def test_write_preserves_metadata() -> None:
    ctx = _in_memory_ctx()
    w = await MemoryTool().execute(
        _call(
            {
                "operation": "write",
                "scope": "project",
                "role": "assistant",
                "content": "c",
                "metadata": {"k": "v", "n": 3},
            }
        ),
        SB,
        ctx,
    )
    assert isinstance(w, ToolOutputSuccess)
    assert json.loads(w.content)["metadata"] == {"k": "v", "n": 3}


# ============================================================================
# R5: non-merged scope isolation
# ============================================================================


async def test_scoped_read_does_not_see_other_scope() -> None:
    ctx, _ = _scoped_ctx()
    tool = MemoryTool()
    await tool.execute(
        _call({"operation": "write", "scope": "user", "role": "user", "content": "u1"}), SB, ctx
    )
    await tool.execute(
        _call({"operation": "write", "scope": "project", "role": "assistant", "content": "p1"}),
        SB,
        ctx,
    )

    r_user = await tool.execute(_call({"operation": "read", "scope": "user"}), SB, ctx)
    assert isinstance(r_user, ToolOutputSuccess)
    assert _contents(r_user) == ["u1"]

    r_proj = await tool.execute(_call({"operation": "read", "scope": "project"}), SB, ctx)
    assert isinstance(r_proj, ToolOutputSuccess)
    assert _contents(r_proj) == ["p1"]


# ============================================================================
# R6: merged read drives the shared merge fixture
# ============================================================================


async def test_merged_read_fixture_replay() -> None:
    f = json.loads((STORAGE_FIXTURES / "memory_scoped_merge.json").read_text())
    limit = f["limit"]

    ctx, memory_store = _scoped_ctx()
    for key, scope in [
        ("user", StorageScope.USER),
        ("project", StorageScope.PROJECT),
        ("local", StorageScope.LOCAL),
    ]:
        for entry in f[key]:
            await memory_store.append_memory(
                scope, ctx.session_id, MemoryEntry.model_validate(entry)
            )

    out = await MemoryTool().execute(
        _call({"operation": "read", "scope": "user", "merged": True, "limit": limit}), SB, ctx
    )
    assert isinstance(out, ToolOutputSuccess)
    contents = _contents(out)
    assert contents == f["expected_merged_contents"]
    assert contents.count("dup") == 2  # no dedup
    assert not any("should-not-appear" in c for c in contents)  # local excluded


# ============================================================================
# merged read respects limit (newest-first)
# ============================================================================


async def test_merged_read_respects_limit() -> None:
    ctx, memory_store = _scoped_ctx()
    for i, ts in enumerate(
        ["2026-05-01T00:00:00Z", "2026-05-02T00:00:00Z", "2026-05-03T00:00:00Z"]
    ):
        await memory_store.append_memory(
            StorageScope.USER,
            ctx.session_id,
            MemoryEntry(role="user", content=f"u{i}", timestamp=ts),
        )

    out = await MemoryTool().execute(
        _call({"operation": "read", "scope": "user", "merged": True, "limit": 2}), SB, ctx
    )
    assert isinstance(out, ToolOutputSuccess)
    assert _contents(out) == ["u2", "u1"]  # newest-first, capped at 2


# ============================================================================
# R7: Local rejected on BOTH ops — exact message, nothing written
# ============================================================================


async def test_local_rejected_on_write_writes_nothing() -> None:
    ctx, memory_store = _scoped_ctx()
    out = await MemoryTool().execute(
        _call({"operation": "write", "scope": "local", "role": "user", "content": "x"}), SB, ctx
    )
    assert isinstance(out, ToolOutputError)
    assert out.recoverable
    assert out.message == LOCAL_REJECTED_MESSAGE
    # Nothing written to ANY scope.
    for scope in (StorageScope.USER, StorageScope.PROJECT, StorageScope.LOCAL):
        got = await memory_store.get_memories(scope, ctx.session_id, 50)
        assert got == []


async def test_local_rejected_on_read() -> None:
    ctx = _in_memory_ctx()
    out = await MemoryTool().execute(_call({"operation": "read", "scope": "local"}), SB, ctx)
    assert isinstance(out, ToolOutputError)
    assert out.recoverable
    assert out.message == LOCAL_REJECTED_MESSAGE


# ============================================================================
# Session isolation
# ============================================================================


async def test_memory_is_keyed_by_session_id() -> None:
    store = InMemoryStorageProvider()
    tool = MemoryTool()
    ctx_a = _ctx_with(store, "session-a")
    ctx_b = _ctx_with(store, "session-b")

    await tool.execute(
        _call({"operation": "write", "scope": "user", "role": "user", "content": "a1"}), SB, ctx_a
    )
    await tool.execute(
        _call({"operation": "write", "scope": "user", "role": "user", "content": "b1"}), SB, ctx_b
    )

    a = await store.get_memories(StorageScope.USER, SessionId("session-a"), 50)
    b = await store.get_memories(StorageScope.USER, SessionId("session-b"), 50)
    assert [e.content for e in a] == ["a1"]
    assert [e.content for e in b] == ["b1"]


# ============================================================================
# R8: bad params → recoverable error
# ============================================================================


async def test_bad_params_is_recoverable_error() -> None:
    ctx = _in_memory_ctx()
    # Unknown operation.
    r = await MemoryTool().execute(_call({"operation": "nope"}), SB, ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable
    # Missing required field on write.
    r = await MemoryTool().execute(
        _call({"operation": "write", "scope": "user", "role": "user"}), SB, ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable
    # Unknown scope string.
    r = await MemoryTool().execute(
        _call({"operation": "write", "scope": "bogus", "role": "user", "content": "x"}), SB, ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable


# ============================================================================
# R9: storage failure → recoverable error (both write and read paths)
# ============================================================================


async def test_storage_failure_is_recoverable_error() -> None:
    ctx = _ctx_with(_FailingMemoryStore(), "test-session")
    w = await MemoryTool().execute(
        _call({"operation": "write", "scope": "user", "role": "user", "content": "x"}), SB, ctx
    )
    assert isinstance(w, ToolOutputError)
    assert w.recoverable
    r = await MemoryTool().execute(_call({"operation": "read", "scope": "user"}), SB, ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable
    # Merged read path too.
    rm = await MemoryTool().execute(
        _call({"operation": "read", "scope": "user", "merged": True}), SB, ctx
    )
    assert isinstance(rm, ToolOutputError)
    assert rm.recoverable


# ============================================================================
# R10: read does not write
# ============================================================================


async def test_read_does_not_write() -> None:
    ctx, memory_store = _scoped_ctx()
    r = await MemoryTool().execute(_call({"operation": "read", "scope": "user"}), SB, ctx)
    assert isinstance(r, ToolOutputSuccess)
    assert _entries(r) == []
    got = await memory_store.get_memories(StorageScope.USER, ctx.session_id, 50)
    assert got == []


# ============================================================================
# Schema is NOT read_only (decision E); default limit 50 (decision B)
# ============================================================================


def test_schema_is_not_read_only() -> None:
    s = MemoryTool.schema()
    assert not s.annotations.read_only
    assert not s.annotations.destructive
    assert not s.annotations.open_world
    assert s.name == "memory"
    # Advertised scope enum omits ``local``.
    assert s.parameters["properties"]["scope"]["enum"] == ["project", "user"]
    assert s.parameters["required"] == ["operation", "scope"]


async def test_read_default_limit_is_50() -> None:
    ctx = _in_memory_ctx()
    tool = MemoryTool()
    for i in range(60):
        await tool.execute(
            _call({"operation": "write", "scope": "user", "role": "user", "content": f"m{i}"}),
            SB,
            ctx,
        )
    r = await tool.execute(_call({"operation": "read", "scope": "user"}), SB, ctx)
    assert isinstance(r, ToolOutputSuccess)
    assert len(_entries(r)) == 50


# ============================================================================
# Fixture replay — fixtures/tools/memory.json
# ============================================================================


async def test_fixture_replay_operations() -> None:
    path = TOOL_FIXTURES / "memory.json"
    scenarios = json.loads(path.read_text())
    assert scenarios, "expected >= 1 scenario"
    tool = MemoryTool()

    for sc in scenarios:
        # Fresh isolated scope-routing provider per scenario.
        ctx, _ = _scoped_ctx("fx")
        for i, step in enumerate(sc["steps"]):
            out = await tool.execute(_call(step["input"]), SB, ctx)
            exp = step["expected"]
            if exp["ok"]:
                assert isinstance(out, ToolOutputSuccess), f"{sc['name']} step {i}"
                want = exp.get("contents")
                if want is not None:
                    got = _contents(out)
                    if exp.get("unordered", False):
                        assert sorted(got) == sorted(want), f"{sc['name']} step {i}"
                    else:
                        assert got == want, f"{sc['name']} step {i}"
            else:
                assert isinstance(out, ToolOutputError), f"{sc['name']} step {i}"
                assert out.recoverable, f"{sc['name']} step {i}: errors must be recoverable"
                assert out.message == exp["error"], f"{sc['name']} step {i}"
