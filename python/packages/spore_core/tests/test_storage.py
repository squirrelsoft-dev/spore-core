"""Tests for the :class:`StorageProvider` abstraction — issue #73.

Mirrors the Rust unit + fixture-replay tests in
``rust/crates/spore-core/src/storage/tests.rs``. Covers every pinned rule:
no-op fallback, composite per-domain routing, single-provider-fills-all-slots,
OTLP parse table, atomic write (no leftover .tmp), append ordering + recency,
run-store opaque-json roundtrip, session roundtrip, flush markers, and
cross-language fixture replay. Fully hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

from spore_core.harness import (
    BudgetSnapshot,
    HumanRequestClarification,
    LoopStrategyReAct,
    PausedState,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    TaskId,
)
from spore_core.memory import Timestamp
from spore_core.storage import (
    CompositeStorageProvider,
    FileSystemStorageProvider,
    InMemoryStorageProvider,
    MemoryEntry,
    NoOpStorageProvider,
    StorageProvider,
    parse_otlp_endpoints,
)

FIXTURES = Path(__file__).resolve().parents[4] / "fixtures" / "storage"


# ── helpers ────────────────────────────────────────────────────────────────


def _sid(s: str) -> SessionId:
    return SessionId(s)


def _ts(s: str) -> Timestamp:
    return Timestamp(s)


def _paused(session: str) -> PausedState:
    """Minimal valid PausedState for roundtrip tests."""
    return PausedState(
        session_id=_sid(session),
        task_id=TaskId("task1"),
        turn_number=3,
        session_state=SessionState(),
        pending_tool_calls=[],
        approved_results=[],
        human_request=HumanRequestClarification(question="?"),
        task=Task.new("do the thing", _sid(session), LoopStrategyReAct(max_iterations=1)),
        budget_used=BudgetSnapshot(),
        child_state=None,
    )


def _mem(role: str, content: str, t: str) -> MemoryEntry:
    return MemoryEntry(role=role, content=content, timestamp=_ts(t))


# ── OTLP endpoint parsing (the most important cross-language rule) ───────────


def test_otlp_parse_table() -> None:
    assert parse_otlp_endpoints("a") == ["a"]
    assert parse_otlp_endpoints("a,b,c") == ["a", "b", "c"]
    assert parse_otlp_endpoints(" a , b ") == ["a", "b"]
    assert parse_otlp_endpoints("a,,b,") == ["a", "b"]
    assert parse_otlp_endpoints("") == []
    assert parse_otlp_endpoints("  ") == []


def test_otlp_parse_fixture_replay() -> None:
    cases = json.loads((FIXTURES / "otlp_endpoints_parse.json").read_text())
    for case in cases:
        assert parse_otlp_endpoints(case["input"]) == case["expected"], (
            f"mismatch for input {case['input']!r}"
        )


# ── No-op fallback ───────────────────────────────────────────────────────────


async def test_noop_reads_empty_writes_ok() -> None:
    p = NoOpStorageProvider()
    assert await p.get_session(_sid("s")) is None
    assert await p.list_sessions() == []
    await p.put_session(_sid("s"), _paused("s"))  # no-op, no raise
    assert await p.get_memories(_sid("s"), 10) == []
    await p.append_memory(_sid("s"), _mem("user", "hi", "t"))
    assert await p.get(_sid("s"), "k") is None
    await p.put(_sid("s"), "k", 1)
    assert await p.list_keys(_sid("s")) == []
    assert await p.get_spans(_sid("s")) == []
    await p.append_span(_sid("s"), {})
    assert await p.get_sessions(_ts("t")) == []
    await p.flush_session(_sid("s"))


def test_default_storage_provider_is_noop() -> None:
    p = StorageProvider.no_op()
    assert p.session() is not None
    assert p.memory() is not None
    assert p.run() is not None
    assert p.observability() is not None


# ── Single-provider-fills-all-slots ──────────────────────────────────────────


async def test_single_fills_all_slots() -> None:
    backend = InMemoryStorageProvider()
    p = StorageProvider.single(backend)
    # All four accessors return the same backend object.
    assert p.session() is backend
    assert p.memory() is backend
    assert p.run() is backend
    assert p.observability() is backend
    # Write through each accessor; reads see them — proving slots share backend.
    await p.session().put_session(_sid("s"), _paused("s"))
    await p.memory().append_memory(_sid("s"), _mem("user", "hi", "t1"))
    await p.run().put(_sid("s"), "plan", {"x": 1})
    await p.observability().append_span(_sid("s"), {"kind": "turn"})
    assert await p.session().get_session(_sid("s")) is not None
    assert len(await p.memory().get_memories(_sid("s"), 10)) == 1
    assert await p.run().get(_sid("s"), "plan") == {"x": 1}
    assert len(await p.observability().get_spans(_sid("s"))) == 1


# ── Composite per-domain routing + no-op fallback ────────────────────────────


async def test_composite_routes_per_domain_and_falls_back_to_noop() -> None:
    run_backend = InMemoryStorageProvider()
    # Only the run domain is configured; the other three fall back to no-op.
    p = CompositeStorageProvider().run(run_backend).build()
    assert p.run() is run_backend

    await p.run().put(_sid("s"), "k", "v")
    assert await p.run().get(_sid("s"), "k") == "v"

    # Unconfigured domains silently no-op.
    await p.session().put_session(_sid("s"), _paused("s"))
    assert await p.session().get_session(_sid("s")) is None
    assert await p.memory().get_memories(_sid("s"), 5) == []
    assert await p.observability().get_spans(_sid("s")) == []


# ── In-memory: session-store roundtrip + list + delete ───────────────────────


async def test_in_memory_session_roundtrip_list_delete() -> None:
    p = InMemoryStorageProvider()
    await p.put_session(_sid("b"), _paused("b"))
    await p.put_session(_sid("a"), _paused("a"))
    got = await p.get_session(_sid("a"))
    assert got is not None and got.session_id == _sid("a")
    assert await p.list_sessions() == [_sid("a"), _sid("b")]  # sorted
    await p.delete_session(_sid("a"))
    assert await p.get_session(_sid("a")) is None
    assert await p.list_sessions() == [_sid("b")]


# ── In-memory: run-store opaque-json roundtrip + list_keys + delete ──────────


async def test_in_memory_run_roundtrip_list_delete() -> None:
    p = InMemoryStorageProvider()
    blob = {"nested": {"arr": [1, 2, 3], "s": "x"}, "n": 4.5}
    await p.put(_sid("s"), "plan", blob)
    await p.put(_sid("s"), "tasks", [1, 2])
    assert await p.get(_sid("s"), "plan") == blob
    assert await p.list_keys(_sid("s")) == ["plan", "tasks"]  # sorted, scoped
    await p.delete(_sid("s"), "plan")
    assert await p.get(_sid("s"), "plan") is None
    assert await p.list_keys(_sid("s")) == ["tasks"]


# ── In-memory: memory append ordering + recency limit ────────────────────────


async def test_in_memory_memory_recency_and_limit() -> None:
    p = InMemoryStorageProvider()
    for i, c in enumerate(["m0", "m1", "m2", "m3"]):
        await p.append_memory(_sid("s"), _mem("user", c, f"t{i}"))
    got = await p.get_memories(_sid("s"), 2)
    assert [e.content for e in got] == ["m3", "m2"]  # most-recent 2, newest-first
    all_entries = await p.get_memories(_sid("s"), 99)
    assert [e.content for e in all_entries] == ["m3", "m2", "m1", "m0"]


async def test_in_memory_spans_append_ordering() -> None:
    p = InMemoryStorageProvider()
    await p.append_span(_sid("s"), {"n": 0})
    await p.append_span(_sid("s"), {"n": 1})
    assert await p.get_spans(_sid("s")) == [{"n": 0}, {"n": 1}]


# ── FileSystem: atomic write (no leftover .tmp) ──────────────────────────────


async def test_fs_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    p = FileSystemStorageProvider(tmp_path)
    await p.put_session(_sid("s"), _paused("s"))
    await p.put(_sid("s"), "k", {"a": 1})
    leftovers = [str(x) for x in tmp_path.rglob("*.tmp")]
    assert leftovers == [], f"leftover .tmp files: {leftovers}"
    assert (tmp_path / "sessions/s/state.json").exists()
    assert (tmp_path / "sessions/s/run/k.json").exists()


async def test_fs_session_roundtrip_list_delete(tmp_path: Path) -> None:
    p = FileSystemStorageProvider(tmp_path)
    await p.put_session(_sid("a"), _paused("a"))
    await p.put_session(_sid("b"), _paused("b"))
    got = await p.get_session(_sid("a"))
    assert got is not None and got.turn_number == 3
    assert await p.list_sessions() == [_sid("a"), _sid("b")]
    await p.delete_session(_sid("a"))
    assert await p.get_session(_sid("a")) is None
    # delete of missing is a no-op (no raise).
    await p.delete_session(_sid("missing"))


async def test_fs_run_roundtrip_list_delete(tmp_path: Path) -> None:
    p = FileSystemStorageProvider(tmp_path)
    blob = {"deep": [True, None, "x"]}
    await p.put(_sid("s"), "plan", blob)
    await p.put(_sid("s"), "tasks", 7)
    assert await p.get(_sid("s"), "plan") == blob
    assert await p.list_keys(_sid("s")) == ["plan", "tasks"]
    await p.delete(_sid("s"), "plan")
    assert await p.get(_sid("s"), "plan") is None
    assert await p.get(_sid("missing"), "x") is None


async def test_fs_memory_append_recency_and_jsonl_path(tmp_path: Path) -> None:
    p = FileSystemStorageProvider(tmp_path)
    for i, c in enumerate(["a", "b", "c"]):
        await p.append_memory(_sid("s"), _mem("user", c, f"t{i}"))
    assert (tmp_path / "sessions/s/memory.jsonl").exists()
    got = await p.get_memories(_sid("s"), 2)
    assert [e.content for e in got] == ["c", "b"]
    assert got[0].metadata == {}  # metadata defaults to {}


async def test_fs_spans_append_and_flush_marker(tmp_path: Path) -> None:
    p = FileSystemStorageProvider(tmp_path)
    await p.append_span(_sid("s"), {"n": 0})
    await p.append_span(_sid("s"), {"n": 1})
    assert (tmp_path / "sessions/s/trace.jsonl").exists()
    assert await p.get_spans(_sid("s")) == [{"n": 0}, {"n": 1}]
    await p.flush_session(_sid("s"))
    assert (tmp_path / "sessions/s/.flushed").exists()


# ── MemoryEntry default metadata ─────────────────────────────────────────────


def test_memory_entry_metadata_defaults_to_empty_object() -> None:
    # Deserialize without `metadata` → defaults to {}.
    e = MemoryEntry.model_validate(
        {"role": "user", "content": "hi", "timestamp": "2026-05-28T00:00:00Z"}
    )
    assert e.metadata == {}
    v = e.model_dump(mode="json")
    assert v["role"] == "user"
    assert v["content"] == "hi"
    assert v["metadata"] == {}


# ── Fixture replay: run_store_values + memory_entries ────────────────────────


async def test_run_store_values_fixture_replay(tmp_path: Path) -> None:
    cases = json.loads((FIXTURES / "run_store_values.json").read_text())
    p = InMemoryStorageProvider()
    fsp = FileSystemStorageProvider(tmp_path)
    for case in cases:
        key, value = case["key"], case["value"]
        await p.put(_sid("s"), key, value)
        assert await p.get(_sid("s"), key) == value, f"in-memory mismatch for {key}"
        await fsp.put(_sid("s"), key, value)
        assert await fsp.get(_sid("s"), key) == value, f"fs mismatch for {key}"


async def test_memory_entries_fixture_replay() -> None:
    raw = (FIXTURES / "memory_entries.jsonl").read_text()
    entries = [MemoryEntry.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    assert len(entries) >= 3

    p = InMemoryStorageProvider()
    for e in entries:
        await p.append_memory(_sid("s"), e)
    got = await p.get_memories(_sid("s"), 2)
    assert len(got) == 2
    assert got[0] == entries[-1]
    assert got[1] == entries[-2]
    # Full read is the reverse (newest-first) of the append order.
    all_entries = await p.get_memories(_sid("s"), 999)
    assert all_entries == list(reversed(entries))


# ── Harness default storage no-op + setter round-trips ───────────────────────


async def test_harness_default_storage_is_no_op_and_setter_round_trips() -> None:
    from spore_core.harness import HarnessBuilder

    from .test_harness import _agent, _config  # type: ignore[attr-defined]

    # Default: no .storage() — must be no-op (reads empty).
    h = StandardHarness(_config(_agent()))
    sess = _sid("s")
    # session_store() accessor present and no-op.
    assert await h.session_store().get_session(sess) is None
    assert await h.storage().run().get(sess, "k") is None

    # Setter round-trips a single-backend provider through the builder.
    backend = InMemoryStorageProvider()
    await backend.put(sess, "plan", {"v": 1})
    builder = HarnessBuilder(
        _agent(),
        _config(_agent()).tool_registry,
        _config(_agent()).sandbox,
        _config(_agent()).context_manager,
        _config(_agent()).termination_policy,
    ).storage(StorageProvider.single(backend))
    h2 = builder.build()
    assert await h2.storage().run().get(sess, "plan") == {"v": 1}


# ── MemoryEntry byte-shape parity with the Rust serde output ─────────────────


def test_memory_entry_json_shape_matches_fixture_first_row() -> None:
    e = MemoryEntry(role="user", content="first message", timestamp=_ts("2026-05-28T10:00:00Z"))
    assert json.loads(e.model_dump_json()) == {
        "role": "user",
        "content": "first message",
        "timestamp": "2026-05-28T10:00:00Z",
        "metadata": {},
    }
