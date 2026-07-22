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


async def test_tool_search_rejects_non_string_query() -> None:
    # A non-string `query` (e.g. a list) is treated as empty → recoverable error,
    # and NO tools are loaded (mirrors Rust's string-only extraction).
    from spore_core import ToolOutputError

    tsr = ToolSearchRegistry(
        _MockLoopReg(_catalogue()),
        ToolSearchConfig.new(["read_file"]).with_description_budget(1000),
    )
    out = await tsr.dispatch(ToolCall(id="c1", name="tool_search", input={"query": ["shell"]}))
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True
    # Nothing was loaded — advertised set unchanged (core + search only).
    assert {s.name for s in tsr.schemas()} == {"read_file", "tool_search"}


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


# ---------------------------------------------------------------------------
# Harness integration (§review Medium #3): tool_search wired through the
# builder advertises only core + tool_search before a search runs.
# ---------------------------------------------------------------------------


async def test_harness_tool_search_advertises_only_core_before_search() -> None:
    from dataclasses import dataclass
    from typing import Any

    from spore_core import (
        AllowAllSandbox,
        AlwaysContinuePolicy,
        HarnessBuilder,
        HarnessRunOptions,
        NoopContextManager,
        ReactConfig,
        RunResultSuccess,
        ScriptedToolRegistry,
        SessionId,
        Task,
    )
    from spore_core.agent import FinalResponse
    from spore_core.model import TokenUsage

    @dataclass
    class _Bundle:
        implementation: Any
        schema: Any

    class _RecTool:
        def __init__(self, n: str) -> None:
            self._n = n

        def name(self) -> str:
            return self._n

        def is_subagent_tool(self) -> bool:
            return False

        def may_produce_large_output(self) -> bool:
            return False

        async def execute(self, call: Any, sandbox: Any, ctx: Any) -> Any:
            from spore_core import ToolOutputSuccess

            return ToolOutputSuccess(content=self._n)

    from spore_core import RegistryToolSchema

    bundles = [
        _Bundle(
            implementation=_RecTool(n),
            schema=RegistryToolSchema(
                name=n, description=d, parameters={"type": "object", "properties": {}}
            ),
        )
        for n, d in [
            ("read_file", "read a file"),
            ("bash", "run a shell command"),
            ("grep", "search files for a pattern"),
        ]
    ]

    seen_tools: list[list[str]] = []

    class _RecAgent:
        async def turn(self, context: Any) -> Any:
            seen_tools.append([t.name for t in context.tools])
            return FinalResponse(content="done", usage=TokenUsage())

        def id(self) -> Any:
            from spore_core import AgentId

            return AgentId("rec")

    builder = HarnessBuilder(
        agent=_RecAgent(),
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=NoopContextManager(),
        termination_policy=AlwaysContinuePolicy(),
    )
    for b in bundles:
        builder = builder.tool(b)
    builder = builder.tool_search(ToolSearchConfig.new(["read_file"]).with_description_budget(1000))
    h = builder.build()
    r = await h.run(HarnessRunOptions(Task.new("do", SessionId("s1"), ReactConfig.per_loop(3))))
    assert isinstance(r, RunResultSuccess)
    # First turn advertises ONLY core + tool_search (deferred tools hidden).
    assert set(seen_tools[0]) == {"read_file", "tool_search"}, seen_tools[0]


async def test_harness_tool_search_leaves_custom_registry_unchanged() -> None:
    """Rust returns the injected custom ``tool_registry`` unchanged when no
    catalogue registry exists; ``tool_search`` only wraps catalogue-backed
    registries. A custom registry therefore advertises its full schema list and
    does NOT grow a synthetic ``tool_search`` schema."""
    from typing import Any

    from spore_core import (
        AllowAllSandbox,
        AlwaysContinuePolicy,
        HarnessBuilder,
        HarnessRunOptions,
        NoopContextManager,
        ReactConfig,
        RunResultSuccess,
        SessionId,
        Task,
    )
    from spore_core.agent import FinalResponse
    from spore_core.model import TokenUsage

    seen_tools: list[list[str]] = []

    class _RecAgent:
        async def turn(self, context: Any) -> Any:
            seen_tools.append([t.name for t in context.tools])
            return FinalResponse(content="done", usage=TokenUsage())

        def id(self) -> Any:
            from spore_core import AgentId

            return AgentId("rec")

    custom = _MockLoopReg(_catalogue())
    h = (
        HarnessBuilder(
            agent=_RecAgent(),
            tool_registry=custom,
            sandbox=AllowAllSandbox(),
            context_manager=NoopContextManager(),
            termination_policy=AlwaysContinuePolicy(),
        )
        .tool_search(ToolSearchConfig.new(["read_file"]).with_description_budget(1000))
        .build()
    )

    r = await h.run(
        HarnessRunOptions(Task.new("do", SessionId("s-custom"), ReactConfig.per_loop(3)))
    )

    assert isinstance(r, RunResultSuccess)
    assert set(seen_tools[0]) == {s.name for s in _catalogue()}
    assert "tool_search" not in seen_tools[0]
