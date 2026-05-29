"""Tests for the net-new ``todo_write`` tool (#81)."""

from __future__ import annotations

import json

from spore_core.harness import SessionId, ToolOutputError, ToolOutputSuccess
from spore_core.model import ToolCall
from spore_core.storage import InMemoryStorageProvider
from spore_core.tool_registry import AllowAllSandbox, ToolContext
from spore_tools.tools.todo import TODO_STORE_KEY, TodoWriteTool


def _ctx() -> ToolContext:
    backend = InMemoryStorageProvider()
    return ToolContext(
        session_id=SessionId("todo-session"),
        run_store=backend,
        memory_store=backend,
    )


def _call(input_: dict) -> ToolCall:
    return ToolCall(id="c1", name=TodoWriteTool.NAME, input=input_)


async def test_writes_and_persists_under_todo_key() -> None:
    ctx = _ctx()
    r = await TodoWriteTool().execute(
        _call(
            {
                "todos": [
                    {"content": "a", "status": "pending"},
                    {"content": "b", "status": "in_progress"},
                ]
            }
        ),
        AllowAllSandbox(),
        ctx,
    )
    assert isinstance(r, ToolOutputSuccess)
    got = json.loads(r.content)
    assert got == [
        {"content": "a", "status": "pending"},
        {"content": "b", "status": "in_progress"},
    ]
    # Persisted under the "todo" key.
    blob = await ctx.run_store.get(ctx.session_id, TODO_STORE_KEY)
    assert blob == got


async def test_replaces_list_wholesale() -> None:
    ctx = _ctx()
    tool = TodoWriteTool()
    await tool.execute(
        _call(
            {
                "todos": [
                    {"content": "old1", "status": "pending"},
                    {"content": "old2", "status": "pending"},
                ]
            }
        ),
        AllowAllSandbox(),
        ctx,
    )
    r = await tool.execute(
        _call({"todos": [{"content": "new", "status": "completed"}]}),
        AllowAllSandbox(),
        ctx,
    )
    assert isinstance(r, ToolOutputSuccess)
    got = json.loads(r.content)
    assert got == [{"content": "new", "status": "completed"}]


async def test_empty_list() -> None:
    ctx = _ctx()
    r = await TodoWriteTool().execute(_call({"todos": []}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputSuccess)
    assert json.loads(r.content) == []


async def test_bad_params_is_recoverable_error() -> None:
    ctx = _ctx()
    r = await TodoWriteTool().execute(_call({"todos": "not-an-array"}), AllowAllSandbox(), ctx)
    assert isinstance(r, ToolOutputError)
    assert r.recoverable is True


def test_schema_not_read_only() -> None:
    s = TodoWriteTool.schema()
    assert s.annotations.read_only is False
    assert s.annotations.destructive is False
