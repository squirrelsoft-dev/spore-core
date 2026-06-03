"""Unit tests for the example's two custom tools (``remember`` / ``recall``).

Each spec rule has a test: ``remember`` stores under the ``fact:`` prefix,
``recall`` returns the stored value, a ``recall`` miss is a recoverable error,
missing/wrong args are recoverable errors, and the annotations (read-only +
idempotent on ``recall``, neither on ``remember``) are correct.
"""

from __future__ import annotations

from spore_core.harness import SessionId, ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.storage import InMemoryStorageProvider
from spore_core.tool_registry import AllowAllSandbox, ToolContext

from tools.recall import RecallTool
from tools.remember import FACT_PREFIX, RememberTool


def _ctx() -> ToolContext:
    backend = InMemoryStorageProvider()
    return ToolContext(
        session_id=SessionId("fact-session"),
        run_store=backend,
        memory_store=backend,
    )


def _call(name: str, input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=name, input=input_)


# ---- remember -------------------------------------------------------------


async def test_remember_stores_under_fact_prefix() -> None:
    ctx = _ctx()
    r = await RememberTool().execute(
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
    r = await RememberTool().execute(_call("remember", {"key": "habitat"}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "value" in r.message


async def test_remember_non_string_key_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await RememberTool().execute(
        _call("remember", {"key": 7, "value": "x"}), AllowAllSandbox(), ctx
    )
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "key" in r.message


def test_remember_schema_not_read_only() -> None:
    s = RememberTool.schema()
    assert s.name == RememberTool.NAME
    assert s.annotations.read_only is False
    assert s.annotations.destructive is False


# ---- recall ---------------------------------------------------------------


async def test_recall_returns_stored_value() -> None:
    ctx = _ctx()
    await RememberTool().execute(
        _call("remember", {"key": "diet", "value": "crabs and shrimp"}),
        AllowAllSandbox(),
        ctx,
    )
    r = await RecallTool().execute(_call("recall", {"key": "diet"}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputSuccess)
    assert r.content == "crabs and shrimp"


async def test_recall_miss_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await RecallTool().execute(_call("recall", {"key": "unknown"}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert r.message == "no fact stored under 'unknown'"


async def test_recall_missing_key_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await RecallTool().execute(_call("recall", {}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True
    assert "key" in r.message


def test_recall_schema_read_only_and_idempotent() -> None:
    s = RecallTool.schema()
    assert s.name == RecallTool.NAME
    assert s.annotations.read_only is True
    assert s.annotations.idempotent is True
