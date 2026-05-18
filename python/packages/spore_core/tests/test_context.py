"""Tests for :class:`StandardContextManager` — issue #7."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import pytest

from spore_core.context import (
    CacheBlockHits,
    CacheHashMismatch,
    CacheProvider,
    CompactionConfig,
    CompactionFailed,
    CompactionPreserveHints,
    CompactionResult,
    ComposedPrompt,
    Context,
    ContextSources,
    Guide,
    GuideId,
    NullCacheProvider,
    SessionState,
    StandardContextManager,
    TokenCountFailed,
)
from spore_core.harness import (
    BaseSandboxProvider,
    HarnessToolResult,
    SessionId,
    TaskId,
    ToolOutputError,
    ToolOutputSuccess,
)
from spore_core.model import (
    Message,
    ModelRequest,
    ModelResponse,
    ProviderInfo,
    Role,
    StopReason,
    StreamEvent,
    TextContent,
    TimeoutError as ModelTimeoutError,
    TokenUsage,
    ToolSchema,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeModel:
    """ModelInterface returning a constant token count."""

    def __init__(self, count: int = 100) -> None:
        self._count = count

    async def call(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content=[], usage=TokenUsage(), stop_reason=StopReason.END_TURN)

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]
        raise NotImplementedError

    async def count_tokens(self, request: ModelRequest) -> int:
        return self._count

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="fake", model_id="fake", context_window=200_000)


class FailingModel:
    async def call(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    async def call_streaming(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]
        raise NotImplementedError

    async def count_tokens(self, request: ModelRequest) -> int:
        raise ModelTimeoutError()

    def provider(self) -> ProviderInfo:
        return ProviderInfo(name="f", model_id="f", context_window=1)


class CountingCache:
    def __init__(self) -> None:
        self.calls: int = 0

    def supports_caching(self) -> bool:
        return True

    def annotate(self, context: Context) -> None:
        self.calls += 1


class PassthroughSandbox(BaseSandboxProvider):
    """Truncates inline; never offloads. Inherits defaults from
    ``BaseSandboxProvider`` for ``handle_large_output``."""

    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sources(
    rendered: str = "BLOCK1", h: int = 0xAB, schemas: list[ToolSchema] | None = None
) -> ContextSources:
    return ContextSources(
        guides=[],
        memory=[],
        tool_schemas=schemas or [],
        composed_prompt=ComposedPrompt(rendered=rendered, block_1_hash=h),
    )


def _state() -> SessionState:
    return SessionState(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        task_instruction="do the thing",
        window_limit=1000,
        token_budget_used=100,
    )


def _mk(
    model: FakeModel | FailingModel | None = None, cache: CacheProvider | None = None
) -> StandardContextManager:
    return StandardContextManager(
        model=model or FakeModel(),
        cache_provider=cache,
        compaction=CompactionConfig(),
    )


# ---------------------------------------------------------------------------
# Rule: assemble returns context with model-reported token count
# ---------------------------------------------------------------------------


async def test_assemble_returns_context_with_token_count_from_model() -> None:
    mgr = _mk()
    ctx = await mgr.assemble(_state(), _sources())
    assert ctx.token_count == 100
    assert ctx.window_limit == 1000
    assert ctx.utilization == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Rule: Block-1 hash invariance
# ---------------------------------------------------------------------------


async def test_block_1_hash_mismatch_is_an_error() -> None:
    mgr = _mk()
    await mgr.assemble(_state(), _sources(h=0xAB))
    with pytest.raises(CacheHashMismatch) as ei:
        await mgr.assemble(_state(), _sources(h=0xCD))
    assert ei.value.block == "static"


async def test_block_1_hash_match_succeeds_across_calls() -> None:
    mgr = _mk()
    await mgr.assemble(_state(), _sources(h=0xAB))
    # Same hash a second time: no error.
    await mgr.assemble(_state(), _sources(h=0xAB))


# ---------------------------------------------------------------------------
# Rule: mid-session Block-2 hash change logs a warning when turn > 1
# ---------------------------------------------------------------------------


async def test_session_hash_change_mid_session_warns(caplog: pytest.LogCaptureFixture) -> None:
    mgr = _mk()
    st = _state()
    await mgr.assemble(st, _sources())
    st.turn_number = 2
    st.environment = "changed"
    with caplog.at_level(logging.WARNING, logger="spore_core.context"):
        await mgr.assemble(st, _sources())
    assert any("session block hash changed" in r.message for r in caplog.records)


async def test_session_hash_change_first_turn_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mgr = _mk()
    st = _state()
    await mgr.assemble(st, _sources())
    st.environment = "changed"
    st.turn_number = 1
    with caplog.at_level(logging.WARNING, logger="spore_core.context"):
        await mgr.assemble(st, _sources())
    assert not any("session block hash changed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Rule: tool schemas sorted by name
# ---------------------------------------------------------------------------


async def test_tool_schemas_are_sorted_by_name() -> None:
    mgr = _mk()
    schemas = [
        ToolSchema(name="zebra", description="", input_schema={}),
        ToolSchema(name="apple", description="", input_schema={}),
        ToolSchema(name="mango", description="", input_schema={}),
    ]
    ctx = await mgr.assemble(_state(), _sources(schemas=schemas))
    assert [s.name for s in ctx.tool_schemas] == ["apple", "mango", "zebra"]


# ---------------------------------------------------------------------------
# Rule: should_compact at threshold (default 80%)
# ---------------------------------------------------------------------------


def test_should_compact_at_threshold() -> None:
    mgr = _mk()
    st = _state()
    st.window_limit = 1000
    st.token_budget_used = 799
    assert not mgr.should_compact(st)
    st.token_budget_used = 800
    assert mgr.should_compact(st)
    st.token_budget_used = 900
    assert mgr.should_compact(st)


def test_should_compact_handles_zero_window() -> None:
    mgr = _mk()
    st = _state()
    st.window_limit = 0
    assert not mgr.should_compact(st)


# ---------------------------------------------------------------------------
# Rule: compaction preserves recent N + uses default hints
# ---------------------------------------------------------------------------


def test_prepare_compaction_keeps_recent_n_and_uses_default_hints() -> None:
    mgr = _mk()
    st = _state()
    for i in range(20):
        st.message_history.append(Message(role=Role.ASSISTANT, content=TextContent(text=f"m{i}")))
    req = mgr.prepare_compaction(st)
    assert len(req.messages_to_compact) == 12
    assert req.preserve_hints.keep_thinking_blocks is True
    assert req.preserve_hints.keep_architectural_decisions is True
    assert req.preserve_hints.keep_open_problems is True


def test_compaction_preserve_hints_default_keep_thinking_true() -> None:
    h = CompactionPreserveHints()
    assert h.keep_thinking_blocks is True


def test_apply_compaction_replaces_old_with_summary() -> None:
    mgr = _mk()
    st = _state()
    for i in range(20):
        st.message_history.append(Message(role=Role.ASSISTANT, content=TextContent(text=f"m{i}")))
    st.token_budget_used = 800
    summary = Message(role=Role.ASSISTANT, content=TextContent(text="summary"))
    mgr.apply_compaction(
        st,
        CompactionResult(summary_message=summary, tokens_reclaimed=500, messages_removed=12),
    )
    # 1 summary + 8 preserved recents
    assert len(st.message_history) == 9
    assert st.token_budget_used == 300
    head = st.message_history[0].content
    assert isinstance(head, TextContent) and head.text == "summary"


def test_apply_compaction_fails_when_history_too_short() -> None:
    mgr = _mk()
    st = _state()
    for i in range(4):
        st.message_history.append(Message(role=Role.ASSISTANT, content=TextContent(text=f"m{i}")))
    with pytest.raises(CompactionFailed):
        mgr.apply_compaction(
            st,
            CompactionResult(
                summary_message=Message(role=Role.ASSISTANT, content=TextContent(text="x")),
                tokens_reclaimed=0,
                messages_removed=0,
            ),
        )


# ---------------------------------------------------------------------------
# Rule: append_tool_result head+tail truncates large content
# ---------------------------------------------------------------------------


async def test_append_tool_result_truncates_large_output() -> None:
    mgr = StandardContextManager(
        model=FakeModel(), compaction=CompactionConfig(), offload_threshold_bytes=64
    )
    st = _state()
    sb = PassthroughSandbox()
    big = "x" * (8 * 1024)
    result = HarnessToolResult(
        call_id="c1",
        output=ToolOutputSuccess(content=big, truncated=False),
    )
    await mgr.append_tool_result(st, result, sb)
    assert len(st.message_history) == 1
    msg = st.message_history[0]
    assert isinstance(msg.content, TextContent)
    text = msg.content.text
    # Pipeline marker is added when truncation occurred (head_tail formatter
    # OR the sandbox's own elided marker).
    assert "[truncated" in text or "elided" in text
    assert len(text) < len(big)


async def test_append_tool_result_small_output_passes_through() -> None:
    mgr = _mk()
    st = _state()
    sb = PassthroughSandbox()
    result = HarnessToolResult(
        call_id="c1",
        output=ToolOutputSuccess(content="hello", truncated=False),
    )
    await mgr.append_tool_result(st, result, sb)
    msg = st.message_history[0]
    assert isinstance(msg.content, TextContent) and msg.content.text == "hello"


async def test_append_tool_result_error_output_is_rendered() -> None:
    mgr = _mk()
    st = _state()
    sb = PassthroughSandbox()
    result = HarnessToolResult(
        call_id="c1",
        output=ToolOutputError(message="boom", recoverable=True),
    )
    await mgr.append_tool_result(st, result, sb)
    msg = st.message_history[0]
    assert isinstance(msg.content, TextContent) and msg.content.text == "[error] boom"


# ---------------------------------------------------------------------------
# Rule: append_response appends an Assistant message
# ---------------------------------------------------------------------------


def test_append_response_pushes_assistant_message() -> None:
    mgr = _mk()
    st = _state()
    mgr.append_response(st, "ack")
    assert len(st.message_history) == 1
    assert st.message_history[0].role == Role.ASSISTANT


# ---------------------------------------------------------------------------
# Rule: inject_skill is ephemeral — no history mutation, no hash drift
# ---------------------------------------------------------------------------


async def test_inject_skill_does_not_touch_history_or_static_hashes() -> None:
    mgr = _mk()
    st = _state()
    ctx = await mgr.assemble(st, _sources())
    before_static = ctx.system_prompt.static_block_hash
    before_session = ctx.system_prompt.session_block_hash
    before_messages = len(ctx.messages)
    mgr.inject_skill(
        ctx,
        Guide(id=GuideId("rust-style"), content="prefer iterators"),
    )
    assert ctx.system_prompt.static_block_hash == before_static
    assert ctx.system_prompt.session_block_hash == before_session
    assert len(ctx.messages) == before_messages
    assert "[SKILL:rust-style]" in ctx.system_prompt.content
    assert ctx.meta.skills_injected == [GuideId("rust-style")]


# ---------------------------------------------------------------------------
# Rule: record_cache_result updates ContextMeta.cache_blocks
# ---------------------------------------------------------------------------


async def test_record_cache_result_updates_meta() -> None:
    mgr = _mk()
    ctx = await mgr.assemble(_state(), _sources())
    mgr.record_cache_result(
        ctx, CacheBlockHits(static_hit=True, session_hit=False, history_hit=True)
    )
    assert ctx.meta.cache_blocks.static_hit is True
    assert ctx.meta.cache_blocks.session_hit is False
    assert ctx.meta.cache_blocks.history_hit is True


# ---------------------------------------------------------------------------
# Rule: CacheProvider.annotate invoked once per assemble
# ---------------------------------------------------------------------------


async def test_cache_provider_annotate_is_called_each_assemble() -> None:
    cache = CountingCache()
    mgr = StandardContextManager(model=FakeModel(), cache_provider=cache)
    srcs = _sources()
    await mgr.assemble(_state(), srcs)
    await mgr.assemble(_state(), srcs)
    assert cache.calls == 2


# ---------------------------------------------------------------------------
# Rule: pending skill injections become Block-3 segments + appear in meta
# ---------------------------------------------------------------------------


async def test_pending_skill_injections_appear_in_meta() -> None:
    mgr = _mk()
    st = _state()
    st.pending_skill_injections.append(Guide(id=GuideId("g1"), content="do x"))
    ctx = await mgr.assemble(st, _sources())
    assert ctx.meta.skills_injected == [GuideId("g1")]
    assert "do x" in ctx.system_prompt.content


# ---------------------------------------------------------------------------
# Rule: Block-3 budget warning only when active
# ---------------------------------------------------------------------------


async def test_budget_warning_only_when_active() -> None:
    mgr = _mk()
    st = _state()
    ctx_off = await mgr.assemble(st, _sources())
    assert "[BUDGET]" not in ctx_off.system_prompt.content
    st.budget_warning_active = True
    ctx_on = await mgr.assemble(st, _sources())
    assert "[BUDGET]" in ctx_on.system_prompt.content


# ---------------------------------------------------------------------------
# Rule: TokenCountFailed surfaces when ModelInterface fails
# ---------------------------------------------------------------------------


async def test_token_count_failure_returns_token_count_failed() -> None:
    mgr = StandardContextManager(model=FailingModel(), cache_provider=NullCacheProvider())
    with pytest.raises(TokenCountFailed):
        await mgr.assemble(_state(), _sources())


# ---------------------------------------------------------------------------
# Cache-stability invariant: same inputs ⇒ identical prefix bytes + hashes
# ---------------------------------------------------------------------------


async def test_deterministic_prefix_across_calls() -> None:
    mgr = _mk()
    srcs = _sources(rendered="BLOCK1-content", h=0x11)
    a = await mgr.assemble(_state(), srcs)
    b = await mgr.assemble(_state(), srcs)
    assert a.system_prompt.content == b.system_prompt.content
    assert a.system_prompt.static_block_hash == b.system_prompt.static_block_hash
    assert a.system_prompt.session_block_hash == b.system_prompt.session_block_hash


# ---------------------------------------------------------------------------
# Rule: into_request prepends system prompt as a Role.SYSTEM message
# ---------------------------------------------------------------------------


async def test_context_into_request_prepends_system_message() -> None:
    mgr = _mk()
    st = _state()
    st.message_history.append(Message(role=Role.USER, content=TextContent(text="hi")))
    ctx = await mgr.assemble(st, _sources())
    req = ctx.into_request()
    assert req.messages[0].role == Role.SYSTEM
    assert isinstance(req.messages[0].content, TextContent)
    assert req.messages[1].role == Role.USER


# ---------------------------------------------------------------------------
# Rule: tool schemas are duplicated, not aliased (no mutation leak)
# ---------------------------------------------------------------------------


async def test_assembled_tool_schemas_are_independent_copy() -> None:
    mgr = _mk()
    schemas = [ToolSchema(name="b", description="", input_schema={})]
    srcs = _sources(schemas=schemas)
    ctx = await mgr.assemble(_state(), srcs)
    schemas.append(ToolSchema(name="a", description="", input_schema={}))
    assert [s.name for s in ctx.tool_schemas] == ["b"]
