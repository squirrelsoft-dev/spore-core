"""Tests for :class:`StandardContextManager` — issue #7."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from spore_core.context import (
    CacheBlockHits,
    CacheHashMismatch,
    CacheProvider,
    CompactionConfig,
    CompactionFailed,
    CompactionPreserveHints,
    CompactionResult,
    CompactionVerificationResult,
    CompactionVerifier,
    ComposedPrompt,
    Context,
    ContextSources,
    Guide,
    GuideId,
    KeyTermVerifier,
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
from spore_core.prompt_chunk_registry import CacheBlock
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
    assert ei.value.block is CacheBlock.STATIC
    assert ei.value.expected == 0xAB
    assert ei.value.actual == 0xCD


async def test_block_1_hash_match_succeeds_across_calls() -> None:
    mgr = _mk()
    await mgr.assemble(_state(), _sources(h=0xAB))
    # Same hash a second time: no error.
    await mgr.assemble(_state(), _sources(h=0xAB))


# ---------------------------------------------------------------------------
# Rule: mid-session Block-2 hash change HALTS when turn > 1 (#32) — consistent
# with Block 1. A turn-1 baseline write never halts.
# ---------------------------------------------------------------------------


async def test_session_hash_change_mid_session_halts() -> None:
    mgr = _mk()
    st = _state()
    # Turn-1 baseline assemble records the session hash.
    st.turn_number = 1
    await mgr.assemble(st, _sources())
    # Mid-session (turn 2) the session content changes: must halt.
    st.turn_number = 2
    st.environment = "changed"
    with pytest.raises(CacheHashMismatch) as ei:
        await mgr.assemble(st, _sources())
    assert ei.value.block is CacheBlock.PER_SESSION
    assert ei.value.turn_number == 2


async def test_stable_session_across_turns_does_not_halt() -> None:
    mgr = _mk()
    st = _state()
    st.turn_number = 1
    await mgr.assemble(st, _sources())
    # Identical session content on a later turn: no halt.
    st.turn_number = 2
    await mgr.assemble(st, _sources())
    st.turn_number = 3
    await mgr.assemble(st, _sources())


async def test_session_hash_change_first_turn_does_not_halt() -> None:
    mgr = _mk()
    st = _state()
    # First assemble records the baseline at turn 1.
    st.turn_number = 1
    await mgr.assemble(st, _sources())
    # A change still at turn 1 must NOT halt (baseline guard: turn_number > 1).
    st.environment = "changed"
    st.turn_number = 1
    await mgr.assemble(st, _sources())


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


# ===========================================================================
# Issue #29: CompactionVerifier / KeyTermVerifier
# ===========================================================================


def _verifier_state(
    task_instruction: str,
    *,
    open_problems: list[str] | None = None,
    architectural_decisions: list[str] | None = None,
    recent_files: list[str] | None = None,
    reasoning_summary: str = "",
) -> SessionState:
    return SessionState(
        session_id=SessionId("s1"),
        task_id=TaskId("t1"),
        task_instruction=task_instruction,
        open_problems=open_problems or [],
        architectural_decisions=architectural_decisions or [],
        recent_files=recent_files or [],
        reasoning_summary=reasoning_summary,
    )


def test_config_default_max_compaction_attempts_is_two() -> None:
    assert CompactionConfig().max_compaction_attempts == 2


def test_key_term_verifier_satisfies_protocol() -> None:
    assert isinstance(KeyTermVerifier(), CompactionVerifier)


def test_all_terms_present_passes() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "We will refactor the parser module to be faster.",
        CompactionPreserveHints(keep_current_task_state=True),
        _verifier_state("Refactor the parser module"),
    )
    assert res.passed is True
    assert res.missing_items == []
    assert res.detail == "all 3 key term(s) present"


def test_missing_term_is_listed() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "We will refactor the parser.",
        CompactionPreserveHints(keep_current_task_state=True),
        _verifier_state("Refactor the parser module"),
    )
    assert res.passed is False
    assert res.missing_items == ["module"]
    assert res.detail == "missing 1 of 3 key term(s): module"


def test_current_task_state_off_yields_zero_terms_and_passes() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "Nothing in particular here.",
        CompactionPreserveHints(keep_current_task_state=False),
        _verifier_state("Refactor the parser module"),
    )
    assert res.passed is True
    assert res.missing_items == []
    assert res.detail == "all 0 key term(s) present"


def test_tokens_under_length_four_are_ignored() -> None:
    v = KeyTermVerifier()
    # "the", "api" (len 3), and the 1-char/3-char tokens drop out; only
    # "test" and "endpoint" remain — both present.
    res = v.verify(
        "Wrote a test for the endpoint.",
        CompactionPreserveHints(keep_current_task_state=True),
        _verifier_state("Test the api endpoint"),
    )
    assert res.passed is True
    assert res.missing_items == []


def test_case_insensitive_matching() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "REFACTOR THE PARSER MODULE",
        CompactionPreserveHints(keep_current_task_state=True),
        _verifier_state("refactor the parser module"),
    )
    assert res.passed is True
    assert res.missing_items == []


def test_terms_are_deduped_first_occurrence_order() -> None:
    v = KeyTermVerifier()
    # "deploy" repeats in the source; it must appear once in missing_items.
    res = v.verify(
        "An unrelated note.",
        CompactionPreserveHints(keep_current_task_state=True),
        _verifier_state("Deploy deploy the service"),
    )
    assert res.passed is False
    assert res.missing_items == ["deploy", "service"]


def test_non_task_hints_contribute_no_terms() -> None:
    v = KeyTermVerifier()
    # All four non-task hints True, keep_current_task_state False → no source
    # terms regardless of task_instruction content.
    res = v.verify(
        "Totally unrelated summary.",
        CompactionPreserveHints(
            keep_architectural_decisions=True,
            keep_open_problems=True,
            keep_current_task_state=False,
            keep_recent_file_list=True,
            keep_thinking_blocks=True,
        ),
        _verifier_state("Refactor the parser module deployment pipeline"),
    )
    assert res.passed is True
    assert res.missing_items == []
    assert res.detail == "all 0 key term(s) present"


# ---------------------------------------------------------------------------
# Issue #47: structured fields feed the four additional hints
# ---------------------------------------------------------------------------


def _only(**overrides: bool) -> CompactionPreserveHints:
    base = dict(
        keep_architectural_decisions=False,
        keep_open_problems=False,
        keep_current_task_state=False,
        keep_recent_file_list=False,
        keep_thinking_blocks=False,
    )
    base.update(overrides)
    return CompactionPreserveHints(**base)


def test_open_problems_isolated() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "we noted the deadlock",
        _only(keep_open_problems=True),
        _verifier_state("ignored task", open_problems=["Resolve the deadlock issue"]),
    )
    assert res.passed is False
    assert res.missing_items == ["resolve", "issue"]


def test_architectural_decisions_isolated() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "we will adopt hexagonal architecture",
        _only(keep_architectural_decisions=True),
        _verifier_state("ignored task", architectural_decisions=["Adopt hexagonal architecture"]),
    )
    assert res.passed is True
    assert res.missing_items == []


def test_recent_files_path_tokenization() -> None:
    v = KeyTermVerifier()
    # src, mod, rs are <4 chars and dropped; only "parser" survives.
    res = v.verify(
        "touched the lexer",
        _only(keep_recent_file_list=True),
        _verifier_state("ignored task", recent_files=["src/parser/mod.rs"]),
    )
    assert res.passed is False
    assert res.missing_items == ["parser"]


def test_reasoning_summary_isolated() -> None:
    v = KeyTermVerifier()
    res = v.verify(
        "nothing relevant",
        _only(keep_thinking_blocks=True),
        _verifier_state("ignored task", reasoning_summary="Considered caching strategy"),
    )
    assert res.passed is False
    assert res.missing_items == ["considered", "caching", "strategy"]


def test_multi_hint_dedup_ordering() -> None:
    v = KeyTermVerifier()
    # "parser" reachable via both task_instruction and open_problems;
    # first-occurrence is the task position (pushed first). "bug" <4 dropped.
    res = v.verify(
        "nothing matched",
        _only(keep_open_problems=True, keep_current_task_state=True),
        _verifier_state("Refactor parser", open_problems=["parser bug remains"]),
    )
    assert res.passed is False
    assert res.missing_items == ["refactor", "parser", "remains"]


def test_empty_list_with_hint_on_passes() -> None:
    v = KeyTermVerifier()
    # open_problems empty but its hint on ⇒ contributes nothing ⇒ passes.
    res = v.verify(
        "anything",
        _only(keep_open_problems=True),
        _verifier_state("ignored task", open_problems=[]),
    )
    assert res.passed is True
    assert res.missing_items == []


# ---------------------------------------------------------------------------
# Cross-language fixture replay (fixtures/compaction_verifier/cases.json)
# ---------------------------------------------------------------------------


def _compaction_verifier_cases() -> list[dict]:
    # tests/ -> spore_core/ -> packages/ -> python/ -> repo_root/fixtures/...
    path = Path(__file__).resolve().parents[4] / "fixtures" / "compaction_verifier" / "cases.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _compaction_verifier_cases(), ids=lambda c: c["name"])
def test_key_term_verifier_fixture_replay(case: dict) -> None:
    v = KeyTermVerifier()
    hints = CompactionPreserveHints(**case["hints"])
    state = _verifier_state(
        case["task_instruction"],
        open_problems=case.get("open_problems", []),
        architectural_decisions=case.get("architectural_decisions", []),
        recent_files=case.get("recent_files", []),
        reasoning_summary=case.get("reasoning_summary", ""),
    )
    res: CompactionVerificationResult = v.verify(case["summary"], hints, state)
    assert res.passed == case["expected"]["passed"]
    assert res.missing_items == case["expected"]["missing_items"]
