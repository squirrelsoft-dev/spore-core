"""Issue #91 — catalogue tool path wired end-to-end + system_prompt seam.

Mirrors the Rust reference tests in ``rust/crates/spore-core/src/harness.rs``:

* ``.tool()`` / ``.tools()`` accumulate, and ``build_config`` folds them into a
  populated :class:`StandardToolRegistry`, defaulting storage to in-memory.
* The per-run :class:`RealToolRegistry` bridge advertises the catalogue schemas
  and maps an unknown-tool dispatch to a *recoverable* error.
* ``.system_prompt()`` prepends a leading System message each turn, with the
  no-duplicate and none-set cases preserved.

Catalogue tools live in ``spore_tools`` (which depends on ``spore_core``), so to
keep this an in-package test we register a tiny local :class:`Tool`-shaped
double rather than importing the real catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    FinalResponse,
    HarnessBuilder,
    HarnessRunOptions,
    ReactConfig,
    MockAgent,
    NoopContextManager,
    RealToolRegistry,
    ScriptedToolRegistry,
    SessionId,
    StandardHarness,
    Task,
    TokenUsage,
    ToolCall,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.agent import Context, TurnResult
from spore_core.tool_registry import (
    SandboxProvider,
    Tool,
    ToolAnnotations,
    ToolContext,
    ToolOutput,
    ToolSchema,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NoopTool:
    """A minimal :class:`Tool` that never runs in these tests — they only assert
    on registration, schema advertisement, and dispatch routing."""

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
    """``StandardTool``-shaped bundle the builder destructures: an
    ``implementation`` + a ``schema`` (issue #81)."""

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


class _CapturingAgent:
    """Records the messages of the context it was handed, then replies with a
    final response — lets a test assert what the model saw."""

    def __init__(self, agent_id: AgentId) -> None:
        self._id = agent_id
        self.seen: list = []

    def id(self) -> AgentId:
        return self._id

    async def turn(self, context: Context) -> TurnResult:
        self.seen = list(context.messages)
        return FinalResponse(content="done", usage=TokenUsage(input_tokens=1, output_tokens=1))


def _builder(agent: object) -> HarnessBuilder:
    return HarnessBuilder(
        agent,  # type: ignore[arg-type]
        ScriptedToolRegistry(),
        AllowAllSandbox(),
        NoopContextManager(),
        AlwaysContinuePolicy(),
    )


def _react_task() -> Task:
    return Task.new("do something", SessionId("s1"), ReactConfig.per_loop(2))


# ---------------------------------------------------------------------------
# Catalogue fold + in-memory storage default
# ---------------------------------------------------------------------------


async def test_catalogue_tools_fold_into_registry_with_inmemory_storage() -> None:
    cfg = (
        _builder(MockAgent(AgentId("test")))
        .tool(_catalogue_tool("read_file"))
        .tool(_catalogue_tool("write_file"))
        .build_config()
    )

    # Accumulated catalogue tools were folded into a registry.
    assert cfg.catalogue_registry is not None
    names = [s.name for s in cfg.catalogue_registry.active_schemas(None)]
    assert "read_file" in names
    assert "write_file" in names

    # Storage defaulted to in-memory (not all-no-op) because catalogue tools are
    # present: a put/get round-trips on the run store.
    sid = SessionId("s1")
    run = cfg.storage.run()
    await run.put(sid, "k", {"v": 1})
    assert await run.get(sid, "k") is not None


def test_no_catalogue_tools_keeps_tool_registry_seam() -> None:
    cfg = _builder(MockAgent(AgentId("test"))).build_config()
    assert cfg.catalogue_registry is None


def test_sandbox_setter_overrides_the_configured_sandbox() -> None:
    sb = AllowAllSandbox()
    cfg = _builder(MockAgent(AgentId("test"))).sandbox(sb).build_config()
    assert cfg.sandbox is sb


# ---------------------------------------------------------------------------
# Per-run bridge: schemas advertised + recoverable error mapping
# ---------------------------------------------------------------------------


async def test_effective_registry_bridges_catalogue_tools() -> None:
    cfg = _builder(MockAgent(AgentId("test"))).tool(_catalogue_tool("read_file")).build_config()
    h = StandardHarness(cfg)
    reg = h._effective_tool_registry(SessionId("s1"))

    # The bridge is the production wiring, not the slim seam.
    assert isinstance(reg, RealToolRegistry)
    # It advertises the catalogue schemas to the model.
    assert any(s.name == "read_file" for s in reg.schemas())
    # And maps an inner dispatch failure (unknown tool) to a *recoverable* error
    # so the loop appends it and the agent can adapt.
    out = await reg.dispatch(ToolCall(id="c", name="does_not_exist", input={}))
    assert isinstance(out, ToolOutputError)
    assert out.recoverable is True


def test_no_catalogue_returns_injected_seam() -> None:
    seam = ScriptedToolRegistry()
    cfg = HarnessBuilder(
        MockAgent(AgentId("test")),
        seam,
        AllowAllSandbox(),
        NoopContextManager(),
        AlwaysContinuePolicy(),
    ).build_config()
    h = StandardHarness(cfg)
    assert h._effective_tool_registry(SessionId("s1")) is seam


# ---------------------------------------------------------------------------
# system_prompt seam
# ---------------------------------------------------------------------------


async def test_system_prompt_is_prepended_to_context() -> None:
    agent = _CapturingAgent(AgentId("cap"))
    cfg = _builder(agent).system_prompt("OPERATING RULES").build_config()
    h = StandardHarness(cfg)
    await h.run(HarnessRunOptions(_react_task()))

    first = agent.seen[0]
    assert first.role.value == "system"
    assert first.content.text == "OPERATING RULES"


async def test_no_system_prompt_leaves_context_without_a_system_message() -> None:
    agent = _CapturingAgent(AgentId("cap"))
    cfg = _builder(agent).build_config()  # system_prompt stays None
    h = StandardHarness(cfg)
    await h.run(HarnessRunOptions(_react_task()))

    assert all(m.role.value != "system" for m in agent.seen)


async def test_system_prompt_not_duplicated_when_context_already_has_one() -> None:
    # A context manager that already leads with a System message must not get a
    # second one. We seed one via a tiny custom manager.
    from spore_core.model import Message, ModelParams, Role, TextContent

    class _SystemFirstContextManager:
        async def assemble(self, session: object, task: object) -> Context:
            return Context(
                messages=[Message(role=Role.SYSTEM, content=TextContent(text="MANAGER PROMPT"))],
                tools=[],
                params=ModelParams(),
            )

        async def append_tool_result(self, session: object, result: object) -> None:
            return None

        async def append_user_message(self, session: object, text: str) -> None:
            return None

        def should_compact(self, session: object) -> bool:
            return False

    agent = _CapturingAgent(AgentId("cap"))
    cfg = (
        HarnessBuilder(
            agent,  # type: ignore[arg-type]
            ScriptedToolRegistry(),
            AllowAllSandbox(),
            _SystemFirstContextManager(),  # type: ignore[arg-type]
            AlwaysContinuePolicy(),
        )
        .system_prompt("OPERATING RULES")
        .build_config()
    )
    h = StandardHarness(cfg)
    await h.run(HarnessRunOptions(_react_task()))

    system_messages = [m for m in agent.seen if m.role == Role.SYSTEM]
    assert len(system_messages) == 1
    assert system_messages[0].content.text == "MANAGER PROMPT"


# ---------------------------------------------------------------------------
# ToolOutput constructors
# ---------------------------------------------------------------------------


def test_tool_output_constructors() -> None:
    ok = ToolOutputSuccess.success("hi")
    assert ok.content == "hi"
    assert ok.truncated is False

    err = ToolOutputError.error("bad arg")
    assert err.message == "bad arg"
    assert err.recoverable is True

    fatal = ToolOutputError.fatal("unrecoverable")
    assert fatal.message == "unrecoverable"
    assert fatal.recoverable is False
