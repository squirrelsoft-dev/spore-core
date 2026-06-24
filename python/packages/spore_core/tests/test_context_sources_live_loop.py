"""SC-26 / #115: structural ``ContextSources`` reach the model live-loop.

Covers the slice-2 acceptance (a guide registered on the harness reaches the
model as a LEADING System block with the configured system prompt merged in
front of it — NOT a User message), the ``StandardCompactionAdapter`` render
block, and ``StandardHarness._build_context_sources`` appending a catalog's
``active_guides``.
"""

from __future__ import annotations

from spore_core import (
    AgentId,
    AllowAllSandbox,
    AlwaysContinuePolicy,
    ComposedPrompt,
    ContextSources,
    FinalResponse,
    HarnessConfig,
    HarnessRunOptions,
    ReactConfig,
    ScriptedToolRegistry,
    SessionId,
    SessionState,
    SkillCatalog,
    SkillEntry,
    StandardHarness,
    Task,
    TokenUsage,
)
from spore_core.agent import Context as AgentContext
from spore_core.compaction_adapter import _render_context_block
from spore_core.context import Guide, GuideId
from spore_core.model import Message, ModelParams, Role, TextContent


# ---------------------------------------------------------------------------
# render_context_block (compaction adapter)
# ---------------------------------------------------------------------------


def _empty_sources() -> ContextSources:
    return ContextSources(
        guides=[],
        memory=[],
        tool_schemas=[],
        composed_prompt=ComposedPrompt(rendered="", block_1_hash=0),
    )


def test_render_context_block_formats_guides_and_empty_when_no_sources() -> None:
    # Empty sources → empty block → the adapter adds no System message, so the
    # no-source path is byte-identical to the pre-#115 pass-through.
    assert _render_context_block(_empty_sources()) == ""

    sources = ContextSources(
        guides=[
            Guide(id=GuideId("audit"), content="AUDIT BODY"),
            Guide(id=GuideId("style"), content="STYLE BODY"),
        ],
        memory=[],
        tool_schemas=[],
        composed_prompt=ComposedPrompt(rendered="", block_1_hash=0),
    )
    block = _render_context_block(sources)
    # Both guides, id-headed, in registration order, joined by blank lines.
    assert block == "# audit\nAUDIT BODY\n\n# style\nSTYLE BODY"


# ---------------------------------------------------------------------------
# Guide reaches the model as a LEADING System block (end-to-end, through loop)
# ---------------------------------------------------------------------------


class _RecordingAgent:
    """Agent double that records the :class:`agent.Context` it is handed each
    turn, so a test can assert what actually reached the model."""

    def __init__(self) -> None:
        self.seen: list[AgentContext] = []

    async def turn(self, context: AgentContext):  # type: ignore[no-untyped-def]
        self.seen.append(context)
        return FinalResponse(content="done", usage=TokenUsage(input_tokens=1, output_tokens=1))

    def id(self) -> AgentId:
        return AgentId("recording")


class _GuideRenderingCm:
    """Minimal context manager that renders ``sources.guides`` into a leading
    System block — mirroring the production ``StandardCompactionAdapter`` so the
    loop's sources-building + system-prompt merge can be asserted without the
    adapter's model machinery."""

    async def assemble(
        self, session: SessionState, task: object, sources: ContextSources
    ) -> AgentContext:
        _ = task
        messages = list(session.messages)
        block = "\n\n".join(f"# {g.id}\n{g.content}" for g in sources.guides)
        if block:
            messages.insert(0, Message(role=Role.SYSTEM, content=TextContent(text=block)))
        return AgentContext(messages=messages, tools=[], params=ModelParams())

    async def append_tool_result(self, session: SessionState, result: object) -> None:
        _ = (session, result)

    async def append_user_message(self, session: SessionState, text: str) -> None:
        session.messages.append(Message(role=Role.USER, content=TextContent(text=text)))

    def should_compact(self, session: SessionState) -> bool:
        _ = session
        return False


def _react_task() -> Task:
    return Task.new("do something", SessionId("s1"), ReactConfig.per_loop(5))


async def test_guide_reaches_model_via_assemble_seam() -> None:
    agent = _RecordingAgent()
    cm = _GuideRenderingCm()
    config = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=cm,
        termination_policy=AlwaysContinuePolicy(),
        system_prompt="SYSTEM PROMPT",
        guides=[Guide(id=GuideId("audit"), content="AUDIT PLAYBOOK BODY")],
    )
    h = StandardHarness(config)
    await h.run(HarnessRunOptions(_react_task()))

    assert agent.seen, "the agent must have been called"
    first = agent.seen[0]
    head = first.messages[0]
    assert head.role == Role.SYSTEM
    assert isinstance(head.content, TextContent)
    # System prompt leads the merged System block.
    assert head.content.text.startswith("SYSTEM PROMPT")
    # The guide reached the model structurally.
    assert "# audit" in head.content.text
    assert "AUDIT PLAYBOOK BODY" in head.content.text
    # The guide is NOT delivered as a stray User message.
    assert not any(
        m.role == Role.USER
        and isinstance(m.content, TextContent)
        and "AUDIT PLAYBOOK BODY" in m.content.text
        for m in first.messages
    )


async def test_no_sources_no_leading_system_block_when_no_prompt() -> None:
    # With no system prompt and no guides, the manager produces no System block,
    # so the byte-identical no-source path holds (no stray System message).
    agent = _RecordingAgent()
    cm = _GuideRenderingCm()
    config = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=cm,
        termination_policy=AlwaysContinuePolicy(),
    )
    h = StandardHarness(config)
    await h.run(HarnessRunOptions(_react_task()))

    assert agent.seen
    first = agent.seen[0]
    assert all(m.role != Role.SYSTEM for m in first.messages)


# ---------------------------------------------------------------------------
# _build_context_sources appends a catalog's active_guides
# ---------------------------------------------------------------------------


def test_build_context_sources_appends_active_guides() -> None:
    cat = SkillCatalog.from_entries(
        [SkillEntry(name="audit", description="Audit a module.", body="AUDIT BODY")]
    )
    cat.activate("audit")
    config = HarnessConfig(
        agent=_RecordingAgent(),
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=_GuideRenderingCm(),
        termination_policy=AlwaysContinuePolicy(),
        guides=[Guide(id=GuideId("playbook"), content="PLAYBOOK BODY")],
        skills=cat,
    )
    h = StandardHarness(config)
    sources = h._build_context_sources(config, [])
    ids = [str(g.id) for g in sources.guides]
    # Configured guide first, then the catalog manifest + active body.
    assert ids == ["playbook", "AVAILABLE SKILLS", "ACTIVE SKILL — audit"]
    assert sources.memory == []
    assert sources.composed_prompt.rendered == ""
