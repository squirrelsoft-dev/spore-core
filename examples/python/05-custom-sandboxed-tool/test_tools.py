"""Unit tests for the example's two custom tools (``remember`` / ``recall``).

Each spec rule has a test, mirroring the Rust suite: ``remember`` stores under
the ``fact:`` prefix, ``recall`` returns the stored value, a ``recall`` miss is a
recoverable error, missing/wrong args are recoverable ``invalid parameters``
errors (so tool-call repair can retry), a store failure is recoverable (via a
``FailingRunStore``), and the annotations (read-only + idempotent on ``recall``,
neither on ``remember``) and the derived schema are correct.
"""

from __future__ import annotations

from spore_core.harness import SessionId, ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.storage import (
    InMemoryStorageProvider,
    JsonValue,
    StorageBackendError,
)
from spore_core.tool_registry import AllowAllSandbox, ToolContext

from tools.recall import recall_tool
from tools.remember import FACT_PREFIX, remember_tool


def _ctx() -> ToolContext:
    backend = InMemoryStorageProvider()
    return ToolContext(
        session_id=SessionId("fact-session"),
        run_store=backend,
        memory_store=backend,
    )


class _FailingRunStore:
    """A RunStore whose every operation fails — proves storage errors map to a
    recoverable tool error (mirrors the Rust/TS/Go ``FailingRunStore``)."""

    async def get(self, session_id: SessionId, key: str) -> JsonValue | None:
        raise StorageBackendError("boom")

    async def put(self, session_id: SessionId, key: str, value: JsonValue) -> None:
        raise StorageBackendError("boom")

    async def delete(self, session_id: SessionId, key: str) -> None:
        return None

    async def list_keys(self, session_id: SessionId) -> list[str]:
        return []


def _failing_ctx() -> ToolContext:
    return ToolContext(
        session_id=SessionId("fact-session"),
        run_store=_FailingRunStore(),  # type: ignore[arg-type]
        memory_store=InMemoryStorageProvider(),
    )


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


# ---- remember -------------------------------------------------------------


async def test_remember_stores_under_fact_prefix() -> None:
    ctx = _ctx()
    r = await remember_tool().implementation.execute(
        _call("remember", {"key": "habitat", "value": "coastal ocean waters"}),
        AllowAllSandbox(),
        ctx,
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "remembered habitat"
    # Persisted under the namespaced key, not the bare key.
    stored = await ctx.run_store.get(ctx.session_id, f"{FACT_PREFIX}habitat")
    assert stored == "coastal ocean waters"
    assert await ctx.run_store.get(ctx.session_id, "habitat") is None


async def test_remember_missing_value_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await remember_tool().implementation.execute(
        _call("remember", {"key": "habitat"}), AllowAllSandbox(), ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "invalid parameters" in r.message
    assert "value" in r.message


async def test_remember_non_string_key_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await remember_tool().implementation.execute(
        _call("remember", {"key": 7, "value": "x"}), AllowAllSandbox(), ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "invalid parameters" in r.message


async def test_remember_store_failure_is_recoverable_error() -> None:
    r = await remember_tool().implementation.execute(
        _call("remember", {"key": "k", "value": "v"}), AllowAllSandbox(), _failing_ctx()
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


def test_remember_schema_not_read_only() -> None:
    s = remember_tool().schema
    assert s.name == "remember"
    assert s.annotations.read_only is False
    assert s.annotations.destructive is False
    assert s.annotations.idempotent is False
    # Derived from RememberInput.
    props = s.parameters["properties"]
    assert "key" in props
    assert "value" in props


# ---- recall ---------------------------------------------------------------


async def test_recall_returns_stored_value() -> None:
    ctx = _ctx()
    await remember_tool().implementation.execute(
        _call("remember", {"key": "diet", "value": "crabs and shrimp"}),
        AllowAllSandbox(),
        ctx,
    )
    r = await recall_tool().implementation.execute(
        _call("recall", {"key": "diet"}), AllowAllSandbox(), ctx
    )
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "crabs and shrimp"


async def test_recall_miss_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await recall_tool().implementation.execute(
        _call("recall", {"key": "unknown"}), AllowAllSandbox(), ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert r.message == "no fact stored under 'unknown'"


async def test_recall_missing_key_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await recall_tool().implementation.execute(_call("recall", {}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "invalid parameters" in r.message


async def test_recall_non_string_key_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await recall_tool().implementation.execute(
        _call("recall", {"key": 123}), AllowAllSandbox(), ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


async def test_recall_read_does_not_write() -> None:
    ctx = _ctx()
    await recall_tool().implementation.execute(
        _call("recall", {"key": "k"}), AllowAllSandbox(), ctx
    )
    keys = await ctx.run_store.list_keys(ctx.session_id)
    assert keys == []


def test_recall_schema_read_only_and_idempotent() -> None:
    s = recall_tool().schema
    assert s.name == "recall"
    assert s.annotations.read_only is True
    assert s.annotations.idempotent is True
    assert s.annotations.destructive is False
    # Derived from RecallInput.
    assert "key" in s.parameters["properties"]
