"""Tests for the ToolSearchRegistry deferred-tool pattern (parity plan §5.11).

Mirrors ``rust/crates/spore-core/src/tool_registry.rs`` ToolSearchRegistry tests.
"""

from __future__ import annotations

import asyncio

from spore_core import ToolOutputSuccess
from spore_core.model import ToolCall, ToolSchema
from spore_core.tool_registry import ToolNamespace, ToolSearchConfig, ToolSearchRegistry


class _MockLoopReg:
    """A minimal harness-loop ToolRegistry over a fixed catalogue."""

    def __init__(self, schemas: list[ToolSchema]) -> None:
        self._s = schemas
        self.dispatched: list[str] = []

    def schemas(self) -> list[ToolSchema]:
        return [s.model_copy(deep=True) for s in self._s]

    def is_always_halt(self, name: str) -> bool:
        return False

    async def dispatch(self, call: ToolCall) -> ToolOutputSuccess:
        self.dispatched.append(call.name)
        return ToolOutputSuccess(content=call.name)


def _catalogue() -> list[ToolSchema]:
    return [
        ToolSchema(name=n, description=d, input_schema={"type": "object", "properties": {}})
        for n, d in [
            ("read_file", "read a file"),
            ("write_file", "write a file"),
            ("bash", "run a shell command"),
            ("grep", "search files for a pattern"),
            ("web_search", "search the web"),
        ]
    ]


def test_core_and_search_advertised_deferred_hidden() -> None:
    tsr = ToolSearchRegistry(_MockLoopReg(_catalogue()), ToolSearchConfig.new(["read_file"]))
    names = {s.name for s in tsr.schemas()}
    assert "read_file" in names
    assert "tool_search" in names
    # Deferred tools are NOT advertised until searched/opened.
    assert "bash" not in names
    assert "grep" not in names


async def test_tool_search_loads_matching_tools() -> None:
    tsr = ToolSearchRegistry(
        _MockLoopReg(_catalogue()),
        ToolSearchConfig.new(["read_file"]).with_description_budget(1000),
    )
    out = await tsr.dispatch(
        ToolCall(id="c1", name="tool_search", input={"query": "shell command"})
    )
    assert "bash" in out.content
    # Now advertised.
    assert "bash" in {s.name for s in tsr.schemas()}


async def test_tool_search_empty_query_is_recoverable_error() -> None:
    from spore_core import ToolOutputError

    tsr = ToolSearchRegistry(_MockLoopReg(_catalogue()), ToolSearchConfig.new(["read_file"]))
    out = await tsr.dispatch(ToolCall(id="c1", name="tool_search", input={"query": ""}))
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True


async def test_tool_search_no_matches() -> None:
    tsr = ToolSearchRegistry(_MockLoopReg(_catalogue()), ToolSearchConfig.new(["read_file"]))
    out = await tsr.dispatch(ToolCall(id="c1", name="tool_search", input={"query": "zzzzz"}))
    assert "No tools matched" in out.content


async def test_deferred_tool_dispatch_succeeds_even_when_not_advertised() -> None:
    inner = _MockLoopReg(_catalogue())
    tsr = ToolSearchRegistry(inner, ToolSearchConfig.new(["read_file"]))
    # grep is not advertised, but dispatch still reaches the inner registry.
    out = await tsr.dispatch(ToolCall(id="c2", name="grep", input={}))
    assert out.content == "grep"
    assert inner.dispatched == ["grep"]
    # And it is now touched (kept warm).
    assert "grep" in {s.name for s in tsr.schemas()}


async def test_namespace_header_loads_group() -> None:
    cfg = (
        ToolSearchConfig.new(["read_file"])
        .with_description_budget(1000)
        .namespace(ToolNamespace.of("fs_tools", "filesystem tools", ["write_file", "grep"]))
    )
    tsr = ToolSearchRegistry(_MockLoopReg(_catalogue()), cfg)
    names = {s.name for s in tsr.schemas()}
    assert "fs_tools" in names
    # Members are not advertised until the namespace is opened.
    assert "write_file" not in names
    out = await tsr.dispatch(ToolCall(id="c3", name="fs_tools", input={}))
    assert "write_file" in out.content
    assert "grep" in out.content
    assert "write_file" in {s.name for s in tsr.schemas()}


def test_namespace_name_colliding_with_tool_is_dropped() -> None:
    # "bash" collides with a catalogue tool → the namespace is dropped at
    # construction (it would shadow the real tool on dispatch). The catalogue
    # "bash" tool remains a standalone deferred tool (NOT a namespace header).
    cfg = ToolSearchConfig.new(["read_file"]).namespace(ToolNamespace.of("bash", "group", ["grep"]))
    tsr = ToolSearchRegistry(_MockLoopReg(_catalogue()), cfg)
    names = {s.name for s in tsr.schemas()}
    # No namespace header named "bash" is advertised (only core + search).
    assert "bash" not in names
    assert "read_file" in names and "tool_search" in names


def test_description_budget_evicts_loaded_tools() -> None:
    # Tiny budget: core + search schema alone fit, but loaded tools get evicted.
    tsr = ToolSearchRegistry(
        _MockLoopReg(_catalogue()), ToolSearchConfig.new(["read_file"]).with_description_budget(50)
    )
    asyncio.run(tsr.dispatch(ToolCall(id="c", name="tool_search", input={"query": "shell"})))
    # bash's description (20 chars) does not fit under the 50-byte budget alongside
    # the always-on baseline → evicted from the advertised set.
    assert "bash" not in {s.name for s in tsr.schemas()}


def test_advertised_description_bytes_respects_budget() -> None:
    tsr = ToolSearchRegistry(
        _MockLoopReg(_catalogue()),
        ToolSearchConfig.new(["read_file"]).with_description_budget(1000),
    )
    asyncio.run(tsr.dispatch(ToolCall(id="c", name="tool_search", input={"query": "shell"})))
    assert tsr.advertised_description_bytes() <= 1000 or {s.name for s in tsr.schemas()} == {
        "read_file",
        "tool_search",
    }


async def test_is_always_halt_false_for_search_and_namespaces() -> None:
    cfg = ToolSearchConfig.new(["read_file"]).namespace(
        ToolNamespace.of("fs_tools", "fs", ["grep"])
    )
    tsr = ToolSearchRegistry(_MockLoopReg(_catalogue()), cfg)
    assert tsr.is_always_halt("tool_search") is False
    assert tsr.is_always_halt("fs_tools") is False
