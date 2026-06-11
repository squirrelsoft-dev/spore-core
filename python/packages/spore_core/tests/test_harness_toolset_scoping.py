"""Issue 2 — per-node toolset scoping wired end-to-end.

Mirrors the Rust reference tests in ``rust/crates/spore-core/src/harness.rs``
(``per_node_toolset_scoping_closes_cross_node_leaks`` et al.):

* A leaf carrying a NON-EMPTY toolset handle dispatches ONLY that toolset's
  catalogue — the planner (``plan-tools``) cannot reach an exec-only tool
  (``read_file``) and the executor (``exec-tools``) cannot reach plan-only tools
  (``task_list``/``list_dir``). Each node sees only its own schemas.
* An unknown tool called by a scoped node still yields the existing recoverable
  unknown-tool error.
* An EMPTY toolset handle falls back to the GLOBAL catalogue wired via
  ``.tools()`` (back-compat with examples 01–11).
* A non-empty handle with NO registered per-key catalogue falls back to the
  global catalogue / seam.
* ``.toolset_tools`` auto-fills a registry presence entry so the tree's handle
  passes ``ExecutionRegistry.validate``.

Catalogue tools live in ``spore_tools`` (which depends on ``spore_core``), so to
keep this an in-package test we register tiny local :class:`Tool`-shaped doubles
rather than importing the real catalogue — exactly as ``test_harness_catalogue``.
"""

from __future__ import annotations

from dataclasses import dataclass

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    HarnessBuilder,
    MockAgent,
    NoopContextManager,
    RealToolRegistry,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    ToolCall,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.tool_registry import (
    SandboxProvider,
    Tool,
    ToolAnnotations,
    ToolContext,
    ToolOutput,
    ToolSchema,
)


# ---------------------------------------------------------------------------
# Test doubles (mirror test_harness_catalogue)
# ---------------------------------------------------------------------------


class _NoopTool:
    """A minimal :class:`Tool` — tests only assert on registration, schema
    advertisement, and dispatch routing."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def is_subagent_tool(self) -> bool:
        return False

    def may_produce_large_output(self) -> bool:
        return False

    async def execute(
        self, call: ToolCall, sandbox: SandboxProvider, ctx: ToolContext
    ) -> ToolOutput:
        return ToolOutputSuccess.success(f"ran {self._name}")


@dataclass
class _StandardTool:
    """``StandardTool``-shaped bundle the builder destructures (issue #81)."""

    implementation: Tool
    schema: ToolSchema


def _catalogue_tool(name: str) -> _StandardTool:
    return _StandardTool(
        implementation=_NoopTool(name),
        schema=ToolSchema(
            name=name,
            description=f"the {name} tool",
            parameters={"type": "object", "properties": {}},
            annotations=ToolAnnotations(read_only=True),
        ),
    )


def _builder() -> HarnessBuilder:
    return HarnessBuilder(
        MockAgent(AgentId("test")),
        ScriptedToolRegistry(),
        AllowAllSandbox(),
        NoopContextManager(),
        AlwaysContinuePolicy(),
    )


async def _dispatched_ok(reg: object, name: str) -> bool:
    out = await reg.dispatch(ToolCall(id="c", name=name, input={}))  # type: ignore[attr-defined]
    return not isinstance(out, ToolOutputError)


def _names(reg: object) -> list[str]:
    return [s.name for s in reg.schemas()]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Strict per-node scoping closes cross-node leaks
# ---------------------------------------------------------------------------


async def test_per_node_toolset_scoping_closes_cross_node_leaks() -> None:
    cfg = (
        _builder()
        .toolset_tools("plan-tools", [_catalogue_tool("list_dir"), _catalogue_tool("task_list")])
        .toolset_tools("exec-tools", [_catalogue_tool("read_file")])
        .build_config()
    )
    h = StandardHarness(cfg)
    sid = SessionId("s1")

    # Planner node: plan-tools only. Its OWN tools advertise; exec-only do not.
    plan = h._effective_tool_registry(sid, "plan-tools")
    assert "list_dir" in _names(plan)
    assert "task_list" in _names(plan)
    assert "read_file" not in _names(plan)
    # The leak the live run exhibited: planner calling an exec-only tool now
    # resolves to unknown-tool / not-available, NOT success.
    assert not await _dispatched_ok(plan, "read_file")

    # Executor node: exec-tools only. It cannot reach the plan-only tools.
    exec_reg = h._effective_tool_registry(sid, "exec-tools")
    assert "read_file" in _names(exec_reg)
    assert "task_list" not in _names(exec_reg)
    assert "list_dir" not in _names(exec_reg)
    assert not await _dispatched_ok(exec_reg, "task_list")
    assert not await _dispatched_ok(exec_reg, "list_dir")


async def test_per_node_toolset_unknown_tool_is_recoverable_error() -> None:
    cfg = _builder().toolset_tools("plan-tools", [_catalogue_tool("list_dir")]).build_config()
    h = StandardHarness(cfg)
    reg = h._effective_tool_registry(SessionId("s1"), "plan-tools")
    out = await reg.dispatch(ToolCall(id="c", name="does_not_exist", input={}))
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True


# ---------------------------------------------------------------------------
# Empty / unknown handle fallback to the global catalogue
# ---------------------------------------------------------------------------


async def test_empty_toolset_handle_falls_back_to_global_catalogue() -> None:
    cfg = (
        _builder()
        .tools([_catalogue_tool("read_file")])  # global catalogue
        .toolset_tools("plan-tools", [_catalogue_tool("list_dir")])  # scoped
        .build_config()
    )
    h = StandardHarness(cfg)
    # Empty handle ⇒ global catalogue (read_file), NOT the scoped plan-tools.
    global_reg = h._effective_tool_registry(SessionId("s1"), "")
    assert isinstance(global_reg, RealToolRegistry)
    assert "read_file" in _names(global_reg)
    assert "list_dir" not in _names(global_reg)


async def test_unknown_toolset_handle_falls_back_to_global_catalogue() -> None:
    cfg = _builder().tools([_catalogue_tool("read_file")]).build_config()
    h = StandardHarness(cfg)
    reg = h._effective_tool_registry(SessionId("s1"), "not-wired")
    assert "read_file" in _names(reg)


def test_toolset_tools_autofill_registry_presence_for_validate() -> None:
    cfg = _builder().toolset_tools("plan-tools", [_catalogue_tool("list_dir")]).build_config()
    # Registry presence entry exists so a tree referencing the handle validates.
    assert cfg.registry.resolve_toolset("plan-tools") is not None
    # And the dispatchable catalogue is present.
    assert "plan-tools" in cfg.toolset_catalogues
