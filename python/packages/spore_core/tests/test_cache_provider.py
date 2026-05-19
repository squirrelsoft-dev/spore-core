"""Tests for :mod:`spore_core.cache_provider` — issue #25."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spore_core.cache_provider import (
    AnthropicCacheProvider,
    CacheAnnotationResult,
    CacheProvider,
    CacheStats,
    NullCacheProvider,
    OllamaCacheProvider,
    OpenAICacheProvider,
    auto_detect_cache_provider,
)
from spore_core.context import (
    BreakpointInfo,
    CacheBlockStatus,
    Context,
    ContextMeta,
    RenderedSystemPrompt,
)
from spore_core.harness import SessionId
from spore_core.model import (
    ContentBlock,
    Message,
    ModelResponse,
    Role,
    StopReason,
    TextBlock,
    TextContent,
    TokenUsage,
)
from spore_core.tool_registry import TaskPhase


def _ctx(tokens: int, breakpoints: list[BreakpointInfo], msgs: int) -> Context:
    return Context(
        system_prompt=RenderedSystemPrompt(
            content="system",
            breakpoints=breakpoints,
            static_block_hash=0,
            session_block_hash=0,
        ),
        messages=[Message(role=Role.USER, content=TextContent(text="m")) for _ in range(msgs)],
        tool_schemas=[],
        token_count=tokens,
        window_limit=200_000,
        utilization=0.0,
        meta=ContextMeta(
            session_id=SessionId("s"),
            turn_number=0,
            active_phase=TaskPhase.EXECUTION,
            guides_loaded=[],
            skills_injected=[],
            compacted=False,
            cache_blocks=CacheBlockStatus(),
        ),
    )


def _response(read: int | None, write: int | None) -> ModelResponse:
    content_block: ContentBlock = TextBlock(text="hi")
    return ModelResponse(
        content=[content_block],
        usage=TokenUsage(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=read,
            cache_write_tokens=write,
        ),
        stop_reason=StopReason.END_TURN,
    )


# ── Rule: All standard impls satisfy the CacheProvider Protocol ─────────


def test_protocol_satisfied() -> None:
    assert isinstance(NullCacheProvider(), CacheProvider)
    assert isinstance(AnthropicCacheProvider(), CacheProvider)
    assert isinstance(OpenAICacheProvider(), CacheProvider)
    assert isinstance(OllamaCacheProvider(), CacheProvider)


# ── Rule: Null provider is no-op ────────────────────────────────────────


def test_null_provider_does_nothing() -> None:
    p = NullCacheProvider()
    assert p.supports_caching() is False
    assert p.provider_name() == "null"
    c = _ctx(100, [], 0)
    r = p.annotate(c)
    assert r == CacheAnnotationResult()
    assert p.parse_cache_stats(_response(5, None)) is None
    assert p.parse_cache_stats(_response(None, None)) is None


# ── Rule: Anthropic supports caching and reports its name ───────────────


def test_anthropic_identity() -> None:
    p = AnthropicCacheProvider()
    assert p.supports_caching() is True
    assert p.provider_name() == "anthropic"
    assert p.max_cache_anchors == 4


# ── Rule: Anthropic.annotate caps at max_cache_anchors ─────────────────


def test_anthropic_annotate_caps_at_max_anchors() -> None:
    p = AnthropicCacheProvider(max_cache_anchors=2)
    bps = [
        BreakpointInfo(after_segment="block_1_static", token_offset=10),
        BreakpointInfo(after_segment="block_2_per_session", token_offset=20),
    ]
    c = _ctx(50, bps, 3)
    r = p.annotate(c)
    assert r.markers_inserted == 2
    assert r.estimated_cacheable_tokens > 0
    # No history anchor appended — already at the cap.
    assert len(c.system_prompt.breakpoints) == 2


# ── Rule: Anthropic.annotate adds history anchor when room and history
#   present ─────────────────────────────────────────────────────────────


def test_anthropic_annotate_adds_history_anchor() -> None:
    p = AnthropicCacheProvider()
    bps = [BreakpointInfo(after_segment="block_1_static", token_offset=10)]
    c = _ctx(75, bps, 4)
    r = p.annotate(c)
    assert r.markers_inserted == 2
    assert len(c.system_prompt.breakpoints) == 2
    assert c.system_prompt.breakpoints[1].after_segment == "__history_tail__"
    assert c.system_prompt.breakpoints[1].token_offset == 75


# ── Rule: Anthropic.annotate returns 0 markers when no history and no
#   existing breakpoints ──────────────────────────────────────────────


def test_anthropic_annotate_zero_when_empty() -> None:
    p = AnthropicCacheProvider()
    c = _ctx(50, [], 0)
    r = p.annotate(c)
    assert r.markers_inserted == 0
    assert r.estimated_cacheable_tokens == 0


# ── Rule: Anthropic.parse_cache_stats returns None without metadata ─────


def test_anthropic_parse_returns_none_without_metadata() -> None:
    p = AnthropicCacheProvider()
    assert p.parse_cache_stats(_response(None, None)) is None


# ── Rule: Anthropic.parse_cache_stats reads read/write tokens ───────────


def test_anthropic_parse_reads_tokens() -> None:
    p = AnthropicCacheProvider()
    s = p.parse_cache_stats(_response(900, 120))
    assert s is not None
    assert s.cache_read_tokens == 900
    assert s.cache_write_tokens == 120
    # Default (sonnet) pricing: 0.30 read / 3.75 write per 1M.
    assert s.cache_read_cost_usd == pytest.approx(900 / 1_000_000 * 0.30)
    assert s.cache_write_cost_usd == pytest.approx(120 / 1_000_000 * 3.75)


# ── Rule: Anthropic.parse_cache_stats treats one-sided metadata as Some
#   (attempted but only one direction present) ──────────────────────


def test_anthropic_parse_one_sided_is_some() -> None:
    p = AnthropicCacheProvider()
    s = p.parse_cache_stats(_response(0, None))
    assert s is not None
    assert s.cache_read_tokens == 0
    assert s.cache_write_tokens == 0

    s2 = p.parse_cache_stats(_response(None, 0))
    assert s2 is not None
    assert s2.cache_read_tokens == 0
    assert s2.cache_write_tokens == 0


# ── Rule: Anthropic.parse_cache_stats computes USD cost from per-model
#   pricing (#39) ────────────────────────────────────────────────────────


def test_anthropic_parse_computes_cost_default_sonnet() -> None:
    p = AnthropicCacheProvider()
    s = p.parse_cache_stats(_response(1_000_000, 1_000_000))
    assert s is not None
    # Sonnet pricing: 0.30 read / 3.75 write per 1M.
    assert s.cache_read_cost_usd == pytest.approx(0.30)
    assert s.cache_write_cost_usd == pytest.approx(3.75)


def test_anthropic_parse_with_opus_pricing() -> None:
    p = AnthropicCacheProvider().with_model_pricing("claude-opus-4-7")
    s = p.parse_cache_stats(_response(1_000_000, 1_000_000))
    assert s is not None
    # Opus pricing: 1.50 read / 18.75 write per 1M.
    assert s.cache_read_cost_usd == pytest.approx(1.50)
    assert s.cache_write_cost_usd == pytest.approx(18.75)


def test_anthropic_parse_with_haiku_pricing() -> None:
    p = AnthropicCacheProvider().with_model_pricing("claude-haiku-4-5")
    s = p.parse_cache_stats(_response(1_000_000, 1_000_000))
    assert s is not None
    # Haiku pricing: 0.08 read / 1.00 write per 1M.
    assert s.cache_read_cost_usd == pytest.approx(0.08)
    assert s.cache_write_cost_usd == pytest.approx(1.00)


def test_anthropic_with_model_pricing_substring_match() -> None:
    # Substring match: any id containing "opus" gets opus pricing.
    p = AnthropicCacheProvider().with_model_pricing("anthropic.claude-opus-future")
    assert p.cache_read_usd_per_million == pytest.approx(1.50)
    p = AnthropicCacheProvider().with_model_pricing("unknown-model")
    # Unknown falls back to sonnet pricing.
    assert p.cache_read_usd_per_million == pytest.approx(0.30)


# ── Rule: OpenAI.annotate is a no-op and counts cacheable tokens only
#   above the threshold ──────────────────────────────────────────────


def test_openai_annotate_threshold() -> None:
    p = OpenAICacheProvider()
    below = _ctx(1023, [], 0)
    r = p.annotate(below)
    assert r.markers_inserted == 0
    assert r.estimated_cacheable_tokens == 0

    above = _ctx(2048, [], 0)
    r = p.annotate(above)
    assert r.markers_inserted == 0
    assert r.estimated_cacheable_tokens == 2048


def test_openai_identity() -> None:
    p = OpenAICacheProvider()
    assert p.supports_caching() is True
    assert p.provider_name() == "openai"
    assert p.min_cacheable_tokens == 1024


# ── Rule: OpenAI.parse_cache_stats reads cached_tokens; write is zero ──


def test_openai_parse_reads_only_reads() -> None:
    p = OpenAICacheProvider()
    s = p.parse_cache_stats(_response(512, 99))
    assert s is not None
    assert s.cache_read_tokens == 512
    assert s.cache_write_tokens == 0
    assert p.parse_cache_stats(_response(None, None)) is None
    assert p.parse_cache_stats(_response(None, 50)) is None


# ── Rule: Ollama supports_caching is false; all ops are no-ops ─────────


def test_ollama_no_op() -> None:
    p = OllamaCacheProvider()
    assert p.supports_caching() is False
    assert p.provider_name() == "ollama"
    c = _ctx(99, [], 0)
    assert p.annotate(c) == CacheAnnotationResult()
    assert p.parse_cache_stats(_response(5, 5)) is None
    assert p.parse_cache_stats(_response(None, None)) is None


# ── Rule: auto_detect maps provider names case-insensitively ───────────


def test_auto_detect_maps_known_providers() -> None:
    p_anthropic = auto_detect_cache_provider("anthropic")
    assert p_anthropic is not None
    assert p_anthropic.provider_name() == "anthropic"

    p_openai = auto_detect_cache_provider("OpenAI")
    assert p_openai is not None
    assert p_openai.provider_name() == "openai"

    p_ollama = auto_detect_cache_provider("ollama")
    assert p_ollama is not None
    assert p_ollama.provider_name() == "ollama"

    assert auto_detect_cache_provider("mystery") is None


# ── Rule: CacheStats default is all zeros ──────────────────────────────


def test_cache_stats_default() -> None:
    s = CacheStats()
    assert s.cache_read_tokens == 0
    assert s.cache_write_tokens == 0
    assert s.cache_read_cost_usd == 0.0
    assert s.cache_write_cost_usd == 0.0


# ── Fixture-replay test ────────────────────────────────────────────────


def test_fixture_parse_cache_stats() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[4]
        / "fixtures"
        / "cache_provider"
        / "parse_cache_stats.json"
    )
    data = json.loads(fixture_path.read_text())

    providers: dict[str, CacheProvider] = {
        "anthropic": AnthropicCacheProvider(),
        "openai": OpenAICacheProvider(),
        "ollama": OllamaCacheProvider(),
        "null": NullCacheProvider(),
    }

    for case in data["cases"]:
        name = case["name"]
        provider = providers[case["provider"]]
        usage = case["usage"]
        resp = _response(usage.get("cache_read_tokens"), usage.get("cache_write_tokens"))
        stats = provider.parse_cache_stats(resp)
        expected = case["expected"]
        if expected["is_some"]:
            assert stats is not None, f"case {name}: expected Some, got None"
            assert stats.cache_read_tokens == expected["cache_read_tokens"], (
                f"case {name}: read tokens"
            )
            assert stats.cache_write_tokens == expected["cache_write_tokens"], (
                f"case {name}: write tokens"
            )
        else:
            assert stats is None, f"case {name}: expected None, got {stats!r}"
