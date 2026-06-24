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
    MemoryConfig,
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
from spore_core.memory import (
    MemoryId,
    MemoryItem,
    MemoryProvider,
    MemorySourceManual,
    MemoryStatusActive,
    MergeStrategy,
    SemanticMemory,
    StandardMemoryProvider,
    Timestamp,
)
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


def _semantic(id_: str, content: str) -> SemanticMemory:
    """A single active, manual semantic memory."""
    ts = Timestamp("2026-06-24T00:00:00Z")
    return SemanticMemory(
        id=MemoryId(id_),
        content=content,
        source=MemorySourceManual(),
        domain=None,
        version=1,
        previous_versions=[],
        created_at=ts,
        updated_at=ts,
        status=MemoryStatusActive(),
    )


def test_render_context_block_appends_memory_after_guides() -> None:
    # #160: memory items render into the SAME structural block, after the guides,
    # as plain content joined by blank lines.
    sources = ContextSources(
        guides=[Guide(id=GuideId("audit"), content="AUDIT BODY")],
        memory=[MemoryItem(memory=_semantic("m1", "MEMORY CONTENT"), relevance_score=0.9)],
        tool_schemas=[],
        composed_prompt=ComposedPrompt(rendered="", block_1_hash=0),
    )
    block = _render_context_block(sources)
    assert block == "# audit\nAUDIT BODY\n\nMEMORY CONTENT"


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
    """Minimal context manager that renders ``sources.guides`` then
    ``sources.memory`` into a leading System block — mirroring the production
    ``StandardCompactionAdapter``'s ``_render_context_block`` so the loop's
    sources-building + system-prompt merge can be asserted without the
    adapter's model machinery."""

    async def assemble(
        self, session: SessionState, task: object, sources: ContextSources
    ) -> AgentContext:
        _ = task
        messages = list(session.messages)
        parts = [f"# {g.id}\n{g.content}" for g in sources.guides]
        parts.extend(m.memory.content for m in sources.memory)
        block = "\n\n".join(parts)
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


async def test_build_context_sources_appends_active_guides() -> None:
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
    sources = await h._build_context_sources(config, [], "do something")
    ids = [str(g.id) for g in sources.guides]
    # Configured guide first, then the catalog manifest + active body.
    assert ids == ["playbook", "AVAILABLE SKILLS", "ACTIVE SKILL — audit"]
    assert sources.memory == []
    assert sources.composed_prompt.rendered == ""


# ---------------------------------------------------------------------------
# Memory reaches the model structurally (#160 / SC-26 follow-up)
# ---------------------------------------------------------------------------


async def _memory_with(id_: str, content: str) -> StandardMemoryProvider:
    """A :class:`StandardMemoryProvider` holding a single active semantic memory."""
    provider = StandardMemoryProvider()
    await provider.store_semantic(_semantic(id_, content), MergeStrategy.REJECT)
    return provider


def _base_config(**kwargs: object) -> HarnessConfig:
    cfg_kwargs: dict[str, object] = {
        "agent": _RecordingAgent(),
        "tool_registry": ScriptedToolRegistry(),
        "sandbox": AllowAllSandbox(),
        "context_manager": _GuideRenderingCm(),
        "termination_policy": AlwaysContinuePolicy(),
    }
    cfg_kwargs.update(kwargs)
    return HarnessConfig(**cfg_kwargs)  # type: ignore[arg-type]


async def test_memory_reaches_model_via_assemble_seam() -> None:
    # #160 / SC-26 acceptance (memory half): a memory provider registered on the
    # harness has its relevant items reach the model through the SAME structural
    # assemble seam as guides — a leading System block, NOT an ad-hoc User
    # message. The provider is held by Protocol type (object-safety is a no-op in
    # Python — structural / dynamic dispatch already).
    provider: MemoryProvider = await _memory_with(
        "m1", "REFUND IDEMPOTENCY: audit the payments refund path"
    )
    agent = _RecordingAgent()
    config = HarnessConfig(
        agent=agent,
        tool_registry=ScriptedToolRegistry(),
        sandbox=AllowAllSandbox(),
        context_manager=_GuideRenderingCm(),
        termination_policy=AlwaysContinuePolicy(),
        system_prompt="SYSTEM PROMPT",
        # Low threshold so the related task surfaces the memory.
        memory=MemoryConfig(provider=provider, min_relevance=0.1),
    )
    h = StandardHarness(config)
    # The query text defaults to the task instruction, so a related task surfaces
    # the memory.
    task = Task.new("audit the payments refund path", SessionId("s1"), ReactConfig.per_loop(5))
    await h.run(HarnessRunOptions(task))

    assert agent.seen, "the agent must have been called"
    first = agent.seen[0]
    head = first.messages[0]
    assert head.role == Role.SYSTEM
    assert isinstance(head.content, TextContent)
    # System prompt leads the merged System block.
    assert head.content.text.startswith("SYSTEM PROMPT")
    # The memory reached the model structurally.
    assert "REFUND IDEMPOTENCY" in head.content.text
    # The memory is NOT delivered as a stray User message.
    assert not any(
        m.role == Role.USER
        and isinstance(m.content, TextContent)
        and "REFUND IDEMPOTENCY" in m.content.text
        for m in first.messages
    )


async def test_build_context_sources_query_defaults_to_task_instruction() -> None:
    # #160: query defaults to the task instruction; an unrelated instruction
    # surfaces nothing at the same threshold.
    provider = await _memory_with("m1", "REFUND IDEMPOTENCY: audit the payments refund path")
    # 0.3 cleanly separates the related instruction from the unrelated one below.
    config = _base_config(memory=MemoryConfig(provider=provider, min_relevance=0.3))
    h = StandardHarness(config)

    by_task = await h._build_context_sources(config, [], "audit the payments refund path")
    assert len(by_task.memory) == 1, "the relevant memory must surface"
    assert "REFUND IDEMPOTENCY" in by_task.memory[0].memory.content

    unrelated = await h._build_context_sources(config, [], "compile the rust workspace")
    assert unrelated.memory == [], "an unrelated instruction must not surface the memory"


async def test_build_context_sources_configured_query_overrides() -> None:
    # #160: a configured ``query`` overrides the task instruction — so even an
    # unrelated instruction surfaces the memory when the fixed query matches.
    provider = await _memory_with("m1", "REFUND IDEMPOTENCY: audit the payments refund path")
    config = _base_config(
        memory=MemoryConfig(
            provider=provider,
            query="audit the payments refund path",
            min_relevance=0.1,
        ),
    )
    h = StandardHarness(config)
    by_override = await h._build_context_sources(config, [], "compile the rust workspace")
    assert len(by_override.memory) == 1, "the configured query must override the task instruction"


async def test_build_context_sources_no_provider_leaves_memory_empty() -> None:
    # #160: no provider configured leaves memory empty, regardless of
    # instruction — byte-identical to the pre-#160 path.
    config = _base_config()
    h = StandardHarness(config)
    sources = await h._build_context_sources(config, [], "audit the payments refund path")
    assert sources.memory == [], "no provider configured must leave memory empty"
