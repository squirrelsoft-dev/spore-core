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
import os
import sys
from pathlib import Path

import pytest

from spore_core.harness import (
    BudgetSnapshot,
    HumanRequestClarification,
    ReactConfig,
    PausedState,
    SessionId,
    SessionState,
    StandardHarness,
    Task,
    TaskId,
)
from spore_core.memory import Timestamp
from spore_core.storage import (
    ActiveRunDecision,
    ActiveRunStatus,
    CompositeStorageProvider,
    FileSystemStorageProvider,
    InMemoryStorageProvider,
    MemoryEntry,
    NoOpStorageProvider,
    ProjectId,
    ProjectIdError,
    RunStore,
    StorageProvider,
    StorageScope,
    WorkspaceId,
    complete_active_run,
    load_active_run,
    parse_otlp_endpoints,
    project_id_from_canonical_path,
    project_id_from_cwd,
    project_id_from_path,
    project_namespace,
    start_or_resume_active_run,
    workspace_id_from_canonical_path,
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
        task=Task.new("do the thing", _sid(session), ReactConfig.per_loop(1)),
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
    assert await p.get_memories(StorageScope.PROJECT, _sid("s"), 10) == []
    await p.append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", "hi", "t"))
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
    await p.memory().append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", "hi", "t1"))
    await p.run().put(_sid("s"), "plan", {"x": 1})
    await p.observability().append_span(_sid("s"), {"kind": "turn"})
    assert await p.session().get_session(_sid("s")) is not None
    assert len(await p.memory().get_memories(StorageScope.PROJECT, _sid("s"), 10)) == 1
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
    assert await p.memory().get_memories(StorageScope.PROJECT, _sid("s"), 5) == []
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
        await p.append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", c, f"t{i}"))
    got = await p.get_memories(StorageScope.PROJECT, _sid("s"), 2)
    assert [e.content for e in got] == ["m3", "m2"]  # most-recent 2, newest-first
    all_entries = await p.get_memories(StorageScope.PROJECT, _sid("s"), 99)
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
        await p.append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", c, f"t{i}"))
    assert (tmp_path / "sessions/s/memory.jsonl").exists()
    got = await p.get_memories(StorageScope.PROJECT, _sid("s"), 2)
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
        await p.append_memory(StorageScope.PROJECT, _sid("s"), e)
    got = await p.get_memories(StorageScope.PROJECT, _sid("s"), 2)
    assert len(got) == 2
    assert got[0] == entries[-1]
    assert got[1] == entries[-2]
    # Full read is the reverse (newest-first) of the append order.
    all_entries = await p.get_memories(StorageScope.PROJECT, _sid("s"), 999)
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


# ════════════════════════════════════════════════════════════════════════════
# #78 — scope + workspace-partitioning extension
# ════════════════════════════════════════════════════════════════════════════


# ── R2: WorkspaceId derivation ───────────────────────────────────────────────


def test_workspace_id_is_deterministic_and_pure() -> None:
    a = workspace_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    b = workspace_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    assert a == b
    # Form is `{sanitized_basename}-{8hex}`.
    assert a.startswith("spore-core-")
    assert len(a) == len("spore-core-") + 8


def test_workspace_id_root_uses_literal_root_basename() -> None:
    w = workspace_id_from_canonical_path("/")
    assert w.startswith("root-")


def test_workspace_id_sanitizes_special_chars_and_collapses_dashes() -> None:
    w = workspace_id_from_canonical_path("/Users/me/My Project (v2)!")
    assert w.startswith("my-project-v2-")
    assert "--" not in w


def test_workspace_id_ignores_trailing_slash() -> None:
    a = workspace_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    b = workspace_id_from_canonical_path("/Users/sbeardsley/dev/spore-core/")
    assert a == b


def test_workspace_id_windows_path_strips_drive_and_normalizes_sep() -> None:
    w = workspace_id_from_canonical_path("C:\\Users\\dev\\spore-core")
    assert w.startswith("spore-core-")
    # Distinct from the posix path (drive stripped but the rest differs).
    posix = workspace_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    assert w != posix


def test_workspace_id_returns_a_workspace_id_str() -> None:
    # WorkspaceId is a NewType over str — usable anywhere a str is.
    w: WorkspaceId = workspace_id_from_canonical_path("/a/b/leaf")
    assert isinstance(w, str)


def test_workspace_id_derivation_fixture_replay() -> None:
    cases = json.loads((FIXTURES / "workspace_id_derivation.json").read_text())
    assert len(cases) >= 4, "fixture should carry several rows"
    for case in cases:
        path = case["canonical_path"]
        expected = case["expected_workspace_id"]
        assert workspace_id_from_canonical_path(path) == expected, f"mismatch for path {path!r}"


# ── R5: scope isolation — User and Project land in different backends ─────────


async def test_scoped_writes_isolated_per_scope() -> None:
    user = InMemoryStorageProvider()
    project = InMemoryStorageProvider()
    p = (
        CompositeStorageProvider()
        .memory(StorageScope.USER, user)
        .memory(StorageScope.PROJECT, project)
        .build()
    )

    await p.memory().append_memory(StorageScope.USER, _sid("s"), _mem("user", "U", "t1"))
    await p.memory().append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", "P", "t1"))

    # Each backend physically holds only its own scope's entry.
    u = await user.get_memories(StorageScope.USER, _sid("s"), 10)
    assert [e.content for e in u] == ["U"]
    pr = await project.get_memories(StorageScope.PROJECT, _sid("s"), 10)
    assert [e.content for e in pr] == ["P"]

    # Scoped reads through the router return only own-scope entries.
    ru = await p.memory().get_memories(StorageScope.USER, _sid("s"), 10)
    assert [e.content for e in ru] == ["U"]
    rp = await p.memory().get_memories(StorageScope.PROJECT, _sid("s"), 10)
    assert [e.content for e in rp] == ["P"]


# ── R6: merged read = User ∪ Project, newest-first by timestamp, no dedup ─────


async def test_merged_read_unions_scopes_newest_first_no_dedup() -> None:
    p = (
        CompositeStorageProvider()
        .memory(StorageScope.USER, InMemoryStorageProvider())
        .memory(StorageScope.PROJECT, InMemoryStorageProvider())
        .build()
    )

    # Identical-content "dup" entry in BOTH scopes (same timestamp) to prove no
    # dedup. Local entry must NOT appear in the merge.
    await p.memory().append_memory(
        StorageScope.USER, _sid("s"), _mem("user", "u-old", "2026-05-01T00:00:00Z")
    )
    await p.memory().append_memory(
        StorageScope.USER, _sid("s"), _mem("user", "dup", "2026-05-03T00:00:00Z")
    )
    await p.memory().append_memory(
        StorageScope.USER, _sid("s"), _mem("user", "u-new", "2026-05-05T00:00:00Z")
    )
    await p.memory().append_memory(
        StorageScope.PROJECT, _sid("s"), _mem("a", "p-old", "2026-05-02T00:00:00Z")
    )
    await p.memory().append_memory(
        StorageScope.PROJECT, _sid("s"), _mem("a", "dup", "2026-05-03T00:00:00Z")
    )
    await p.memory().append_memory(
        StorageScope.PROJECT, _sid("s"), _mem("a", "p-new", "2026-05-06T00:00:00Z")
    )

    merged = await p.get_memories_merged(_sid("s"), 10)
    contents = [e.content for e in merged]
    assert contents == ["p-new", "u-new", "dup", "dup", "p-old", "u-old"]
    # No dedup: the identical-content "dup" entry is present twice.
    assert contents.count("dup") == 2


async def test_merged_read_fixture_replay() -> None:
    f = json.loads((FIXTURES / "memory_scoped_merge.json").read_text())
    limit = f["limit"]

    p = (
        CompositeStorageProvider()
        .memory(StorageScope.USER, InMemoryStorageProvider())
        .memory(StorageScope.PROJECT, InMemoryStorageProvider())
        .memory(StorageScope.LOCAL, InMemoryStorageProvider())
        .build()
    )

    for key, scope in [
        ("user", StorageScope.USER),
        ("project", StorageScope.PROJECT),
        ("local", StorageScope.LOCAL),
    ]:
        for entry in f[key]:
            await p.memory().append_memory(scope, _sid("s"), MemoryEntry.model_validate(entry))

    merged = await p.get_memories_merged(_sid("s"), limit)
    contents = [e.content for e in merged]
    assert contents == f["expected_merged_contents"]
    # Local scope entries are excluded from the merge.
    assert not any("should-not-appear" in c for c in contents)


# ── R7: unconfigured (memory, scope) → NoOp returns [] ───────────────────────


async def test_unconfigured_memory_scope_falls_back_to_noop() -> None:
    # Only User wired; Project + Local fall back to no-op.
    p = CompositeStorageProvider().memory(StorageScope.USER, InMemoryStorageProvider()).build()

    # Writes to an unconfigured scope silently no-op.
    await p.memory().append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", "x", "t"))
    # Reads from an unconfigured scope return [].
    assert await p.memory().get_memories(StorageScope.PROJECT, _sid("s"), 10) == []


# ── R8: scoped read newest-first recency (append 4, limit=2 → newest two) ─────


async def test_scoped_read_recency_newest_first() -> None:
    p = CompositeStorageProvider().memory(StorageScope.PROJECT, InMemoryStorageProvider()).build()
    for i, c in enumerate(["m0", "m1", "m2", "m3"]):
        await p.memory().append_memory(StorageScope.PROJECT, _sid("s"), _mem("user", c, f"t{i}"))
    got = await p.memory().get_memories(StorageScope.PROJECT, _sid("s"), 2)
    assert [e.content for e in got] == ["m3", "m2"]


# ── R11: Local falls back to NoOp when not wired ─────────────────────────────


async def test_local_scope_defaults_to_noop() -> None:
    # Local intentionally not wired.
    p = (
        CompositeStorageProvider()
        .memory(StorageScope.USER, InMemoryStorageProvider())
        .memory(StorageScope.PROJECT, InMemoryStorageProvider())
        .build()
    )
    await p.memory().append_memory(StorageScope.LOCAL, _sid("s"), _mem("user", "l", "t"))
    assert await p.memory().get_memories(StorageScope.LOCAL, _sid("s"), 10) == []


# ── R7/R11: an explicit NoOp wired to Local behaves like the fallback ─────────


async def test_local_explicitly_wired_to_noop() -> None:
    p = (
        CompositeStorageProvider()
        .memory(StorageScope.USER, InMemoryStorageProvider())
        .memory(StorageScope.PROJECT, InMemoryStorageProvider())
        .memory(StorageScope.LOCAL, NoOpStorageProvider())
        .build()
    )
    await p.memory().append_memory(StorageScope.LOCAL, _sid("s"), _mem("user", "l", "t"))
    assert await p.memory().get_memories(StorageScope.LOCAL, _sid("s"), 10) == []


# ── R9: ToolContext exposes memory_store threaded by the registry ────────────
#
# The threading-by-registry assertion lives where ``RealToolRegistry`` does
# (``spore_eval``): see ``packages/spore_eval/tests/test_scenarios.py::
# test_real_tool_registry_threads_memory_store``. Here we only prove the
# ``ToolContext`` carries a usable ``memory_store`` seam.


async def test_tool_context_carries_memory_store_seam() -> None:
    from spore_core.tool_registry import ToolContext

    backend = InMemoryStorageProvider()
    ctx = ToolContext(
        session_id=_sid("ctx-test"),
        project_id=project_id_from_canonical_path("/ctx-test-project"),
        run_store=backend,
        memory_store=backend,
    )
    await ctx.memory_store.append_memory(
        StorageScope.PROJECT, ctx.session_id, _mem("user", "via-ctx", "t1")
    )
    got = await backend.get_memories(StorageScope.PROJECT, _sid("ctx-test"), 10)
    assert len(got) == 1
    assert got[0].content == "via-ctx"


# ════════════════════════════════════════════════════════════════════════════
# #142 — ProjectId + project-scoped durable storage
# ════════════════════════════════════════════════════════════════════════════


def _run_store() -> RunStore:
    return InMemoryStorageProvider()


# ── ProjectId pure derivation ────────────────────────────────────────────────


def test_project_id_is_deterministic_and_matches_workspace_id_algorithm() -> None:
    # Same input → same id, repeatedly.
    a = project_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    b = project_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    assert a == b
    # The derivation is byte-identical to WorkspaceId (it delegates to the same
    # pure algorithm) — the cross-language anchor.
    w = workspace_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    assert str(a) == str(w)
    # Form: `{sanitized_basename}-{8hex}`.
    assert a.startswith("spore-core-")
    assert len(a) == len("spore-core-") + 8


def test_project_id_root_and_special_chars() -> None:
    assert project_id_from_canonical_path("/").startswith("root-")
    p = project_id_from_canonical_path("/Users/me/My Project (v2)!")
    assert p.startswith("my-project-v2-")
    assert "--" not in p


def test_project_id_ignores_trailing_slash() -> None:
    a = project_id_from_canonical_path("/Users/sbeardsley/dev/spore-core")
    b = project_id_from_canonical_path("/Users/sbeardsley/dev/spore-core/")
    assert a == b


def test_project_id_returns_a_project_id_str() -> None:
    # ProjectId is a NewType over str — usable anywhere a str is.
    p: ProjectId = project_id_from_canonical_path("/a/b/leaf")
    assert isinstance(p, str)


# ── The `/a/b` vs `/a_b` collision-resolution case (the spec's headline) ─────


def test_project_id_resolves_slash_underscore_collision() -> None:
    # A naive "slashes → underscores" slug would map BOTH of these to `a_b`. The
    # 8-hex SHA-256 suffix of the FULL canonical path keeps them distinct.
    ab = project_id_from_canonical_path("/a/b")
    a_b = project_id_from_canonical_path("/a_b")
    assert ab != a_b, "/a/b and /a_b must derive DISTINCT project ids"
    # And the pinned exact values (cross-language anchor).
    assert ab == "b-662b7b62"
    assert a_b == "a-b-328ff01f"


def test_project_namespace_projects_onto_session_axis() -> None:
    # ``project_namespace`` yields a SessionId whose string IS the derived project
    # id — the namespace-reuse seam over the RunStore's session-id axis.
    p = project_id_from_canonical_path("/work/audit-repo")
    assert str(project_namespace(p)) == str(p)
    assert str(project_namespace(p)) == "audit-repo-9e8ff6f3"


# ── FS-touching constructors: canonicalize FIRST ─────────────────────────────


def test_project_id_from_path_canonicalizes_first(tmp_path: Path) -> None:
    # A real dir + a relative path INTO it must derive the same id as the absolute
    # canonical path — proving from_path canonicalizes before slugging.
    nested = tmp_path / "proj"
    nested.mkdir()
    canonical = nested.resolve(strict=True)
    from_path = project_id_from_path(nested)
    from_canon = project_id_from_canonical_path(str(canonical))
    assert from_path == from_canon


def test_project_id_from_path_resolves_relative_components(tmp_path: Path) -> None:
    # A path with `..` components canonicalizes to the same id as the direct one.
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    # {tmp}/a/../b  ==  {tmp}/b
    via_dotdot = a / ".." / "b"
    direct = project_id_from_path(b)
    dotted = project_id_from_path(via_dotdot)
    assert direct == dotted


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation differs on Windows")
def test_project_id_from_path_resolves_symlink(tmp_path: Path) -> None:
    # A symlink and its target derive the SAME id (symlink resolved by resolve()).
    target = tmp_path / "real-proj"
    target.mkdir()
    link = tmp_path / "link-proj"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted on this filesystem")
    via_target = project_id_from_path(target)
    via_link = project_id_from_path(link)
    assert via_target == via_link, "a symlink must resolve to its target's project id"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific case-insensitive FS behavior")
def test_project_id_from_path_macos_case_insensitive(tmp_path: Path) -> None:
    # macOS's default filesystem is case-insensitive: a dir created lowercase is
    # reachable via a different-cased path. This assertion is FS-behavior- AND
    # stdlib-behavior-dependent and is gated on BOTH:
    #   - the upper-cased path must actually resolve to the same inode (a
    #     case-insensitive volume — tempdirs on a case-sensitive volume skip), and
    #   - ``Path.resolve()`` must case-FOLD to the on-disk casing. Unlike Rust's
    #     ``fs::canonicalize``, CPython's ``realpath`` does NOT case-fold on macOS
    #     (it only resolves symlinks / ``..``), so the differently-cased basenames
    #     hash differently. When that is the case there is nothing to assert here —
    #     the 8-hex suffix already keeps DISTINCT path strings distinct, which is
    #     the documented behavior. The cross-language case-fold parity is enforced
    #     at the Rust reference level; Python honours its own stdlib semantics.
    lower = tmp_path / "caseproj"
    lower.mkdir()
    upper = tmp_path / "CASEPROJ"
    if not os.path.exists(upper):  # case-sensitive volume ⇒ nothing to assert.
        pytest.skip("filesystem is case-sensitive")
    if lower.resolve().name == upper.resolve().name:
        # ``resolve()`` case-folded to one on-disk casing ⇒ same id.
        assert project_id_from_path(lower) == project_id_from_path(upper)
    else:
        # CPython's realpath preserved the as-written casing (the common macOS
        # case): distinct canonical strings ⇒ distinct (but deterministic) ids.
        assert project_id_from_path(lower) != project_id_from_path(upper)


def test_project_id_from_path_errors_on_missing_path(tmp_path: Path) -> None:
    # A non-existent path cannot be canonicalized ⇒ ProjectIdError.
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ProjectIdError):
        project_id_from_path(missing)


def test_project_id_from_cwd_succeeds() -> None:
    # The process cwd exists, so from_cwd derives an id without error.
    p = project_id_from_cwd()
    assert p != ""


# ── ProjectId derivation fixture replay (cross-language anchor) ───────────────


def test_project_id_derivation_fixture_replay() -> None:
    cases = json.loads((FIXTURES / "project_id_derivation.json").read_text())
    assert len(cases) >= 6, "fixture should carry the collision rows"
    saw_collision_a = False
    saw_collision_b = False
    for case in cases:
        path = case["canonical_path"]
        expected = case["expected_project_id"]
        assert project_id_from_canonical_path(path) == expected, f"mismatch for path {path!r}"
        if path == "/a/b":
            saw_collision_a = True
        if path == "/a_b":
            saw_collision_b = True
    assert saw_collision_a and saw_collision_b, "fixture must pin both /a/b and /a_b"


# ── task_list visible across DIFFERENT sessions with SAME project_id ──────────


async def test_run_store_value_visible_across_sessions_same_project() -> None:
    # The durable seam: write keyed by the project namespace; a DIFFERENT session
    # reading the SAME project namespace sees it.
    store = _run_store()
    project = project_id_from_canonical_path("/work/repo")

    await store.put(project_namespace(project), "task_list", {"tasks": [1, 2]})

    # A fresh session id (mirrors per-Ralph-window regeneration) does NOT change
    # what the project namespace returns.
    _fresh_session = SessionId("fresh-window-session")
    got = await store.get(project_namespace(project), "task_list")
    assert got == {"tasks": [1, 2]}


# ── AC5: cross-window AND cross-process durability via FileSystemStorageProvider


async def test_project_durable_survives_window_reset_and_process_restart(tmp_path: Path) -> None:
    root = tmp_path
    project = project_id_from_canonical_path("/work/audit-repo")
    task_list = {
        "tasks": [{"id": 1, "description": "discover", "status": "completed", "blockers": []}],
        "next_id": 2,
    }

    # Window 1 (session A): write the durable task_list under the project ns.
    provider1: RunStore = FileSystemStorageProvider(root)
    await provider1.put(project_namespace(project), "task_list", task_list)

    # Window 2 (a DIFFERENT, freshly generated session) over the SAME provider
    # root reads window 1's list — cross-window survival.
    provider2: RunStore = FileSystemStorageProvider(root)
    _window_2_session = SessionId("window-2-session")
    got = await provider2.get(project_namespace(project), "task_list")
    assert got == task_list, "window 2 must see window 1's list"

    # A BRAND-NEW FileSystemStorageProvider over the same on-disk root (a fresh
    # process) reads the same bytes — cross-process durability.
    fresh: RunStore = FileSystemStorageProvider(root)
    got2 = await fresh.get(project_namespace(project), "task_list")
    assert got2 == task_list, "a fresh provider must read the durable list"


async def test_project_durable_survival_fixture_replay(tmp_path: Path) -> None:
    fx = json.loads((FIXTURES / "project_durable_survival.json").read_text())
    project = project_id_from_canonical_path(fx["project_canonical_path"])
    # The pinned project id must match.
    assert project == fx["expected_project_id"]
    run_key = fx["run_key"]
    root = tmp_path

    # Window 1 writes the fixture's task_list (under a DISTINCT session id).
    w1_list = fx["window_1"]["task_list"]
    provider1: RunStore = FileSystemStorageProvider(root)
    _session_a = SessionId(fx["window_1"]["session_id"])
    await provider1.put(project_namespace(project), run_key, w1_list)

    # Window 2 (a different session id) reads the expected list cross-window.
    provider2: RunStore = FileSystemStorageProvider(root)
    _session_b = SessionId(fx["window_2"]["session_id"])
    got = await provider2.get(project_namespace(project), run_key)
    assert got == fx["window_2"]["expected_task_list"]

    # A fresh provider (cross-process) reads the expected list.
    fresh: RunStore = FileSystemStorageProvider(root)
    got2 = await fresh.get(project_namespace(project), run_key)
    assert got2 == fx["cross_process"]["expected_task_list"]


# ── Active-run lifecycle: new / resume / complete ────────────────────────────


async def test_active_run_starts_new_when_no_slot() -> None:
    store = _run_store()
    project = project_id_from_canonical_path("/work/p")
    assert await load_active_run(store, project) is None

    decision = await start_or_resume_active_run(
        store, project, "tag-1", _ts("2026-06-12T00:00:00Z")
    )
    assert decision == ActiveRunDecision.STARTED_NEW

    slot = await load_active_run(store, project)
    assert slot is not None
    assert slot.run_tag == "tag-1"
    assert slot.status == ActiveRunStatus.ACTIVE
    # started_at is the INJECTED timestamp (deterministic, no clock).
    assert slot.started_at == _ts("2026-06-12T00:00:00Z")


async def test_active_run_resumes_on_matching_tag() -> None:
    store = _run_store()
    project = project_id_from_canonical_path("/work/p")
    await start_or_resume_active_run(store, project, "tag-1", _ts("t1"))

    # Same tag ⇒ resume; the original started_at is preserved (slot untouched).
    decision = await start_or_resume_active_run(store, project, "tag-1", _ts("t2"))
    assert decision == ActiveRunDecision.RESUMED
    slot = await load_active_run(store, project)
    assert slot is not None
    assert slot.started_at == _ts("t1"), "resume must not restamp"


async def test_active_run_starts_fresh_on_different_tag() -> None:
    store = _run_store()
    project = project_id_from_canonical_path("/work/p")
    await start_or_resume_active_run(store, project, "tag-1", _ts("t1"))

    # A DIFFERENT tag is a genuinely new job in the same repo ⇒ start fresh.
    decision = await start_or_resume_active_run(store, project, "tag-2", _ts("t2"))
    assert decision == ActiveRunDecision.STARTED_NEW
    slot = await load_active_run(store, project)
    assert slot is not None
    assert slot.run_tag == "tag-2"
    assert slot.started_at == _ts("t2")


async def test_active_run_complete_then_restart_is_fresh() -> None:
    store = _run_store()
    project = project_id_from_canonical_path("/work/p")
    await start_or_resume_active_run(store, project, "tag-1", _ts("t1"))

    await complete_active_run(store, project)
    slot = await load_active_run(store, project)
    assert slot is not None
    assert slot.status == ActiveRunStatus.COMPLETED

    # After completion, even the SAME tag starts a fresh Active slot.
    decision = await start_or_resume_active_run(store, project, "tag-1", _ts("t3"))
    assert decision == ActiveRunDecision.STARTED_NEW
    slot = await load_active_run(store, project)
    assert slot is not None
    assert slot.status == ActiveRunStatus.ACTIVE
    assert slot.started_at == _ts("t3")


async def test_complete_active_run_is_noop_without_slot() -> None:
    store = _run_store()
    project = project_id_from_canonical_path("/work/empty")
    # No active slot ⇒ completing is a silent no-op (no error).
    await complete_active_run(store, project)
    assert await load_active_run(store, project) is None


async def test_active_run_isolated_per_project() -> None:
    # Two projects over the SAME store keep independent active slots.
    store = _run_store()
    p1 = project_id_from_canonical_path("/work/one")
    p2 = project_id_from_canonical_path("/work/two")
    await start_or_resume_active_run(store, p1, "a", _ts("t1"))
    await start_or_resume_active_run(store, p2, "b", _ts("t1"))
    slot1 = await load_active_run(store, p1)
    slot2 = await load_active_run(store, p2)
    assert slot1 is not None and slot1.run_tag == "a"
    assert slot2 is not None and slot2.run_tag == "b"


async def test_active_run_malformed_slot_is_treated_as_absent() -> None:
    # A malformed slot is treated as "no live run" — the next start mints a fresh
    # one rather than erroring.
    store = _run_store()
    project = project_id_from_canonical_path("/work/garbage")
    await store.put(project_namespace(project), "active_run", {"not": "an active run"})
    assert await load_active_run(store, project) is None
    decision = await start_or_resume_active_run(store, project, "tag-1", _ts("t1"))
    assert decision == ActiveRunDecision.STARTED_NEW
