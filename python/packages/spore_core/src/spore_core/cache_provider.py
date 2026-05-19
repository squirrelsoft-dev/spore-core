"""CacheProvider — provider-specific cache annotation and stats (issue #25).

Mirrors the Rust reference at
``rust/crates/spore-core/src/cache_provider.rs``.

Cache control is provider-specific at the API level: Anthropic uses
explicit ``cache_control`` markers, OpenAI caches automatically above a
token threshold, Ollama has no caching. :class:`CacheProvider` is the
abstraction that keeps these concerns out of provider-agnostic
:class:`spore_core.context.ContextManager`.

Flow (see ``docs/harness-engineering-concepts.md`` § "Cache Architecture")::

    ContextManager.assemble():
      ... build and render segments ...
      if cache_provider.supports_caching():
        cache_provider.annotate(context)
      return context

    # After each model response:
    stats = cache_provider.parse_cache_stats(response)
    observability.emit_cache_stats(session_id, stats)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .context import BreakpointInfo, Context
from .model import ModelResponse

# ============================================================================
# Spec-defined types
# ============================================================================


@dataclass(frozen=True)
class CacheAnnotationResult:
    """Result of annotating a context with provider-specific cache markers."""

    markers_inserted: int = 0
    estimated_cacheable_tokens: int = 0


@dataclass(frozen=True)
class CacheStats:
    """Cache token usage parsed from a single model response.

    ``None`` from :meth:`CacheProvider.parse_cache_stats` means the
    response had no cache metadata at all (caching wasn't attempted). A
    :class:`CacheStats` with all-zero fields means caching was attempted
    and missed — the distinction matters for observability.
    """

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class CacheProvider(Protocol):
    """Provider-specific cache annotation and response parsing."""

    def supports_caching(self) -> bool:
        """Whether this provider supports prefix caching."""
        ...

    def annotate(self, context: Context) -> CacheAnnotationResult:
        """Annotate a fully assembled context with provider-specific cache
        markers. No-op when :meth:`supports_caching` is ``False``."""
        ...

    def parse_cache_stats(self, response: ModelResponse) -> CacheStats | None:
        """Parse cache usage from a model response. Returns ``None`` when
        the response has no cache metadata at all."""
        ...

    def provider_name(self) -> str:
        """Provider identity — used for observability and auto-detection."""
        ...


# ============================================================================
# Standard implementations
# ============================================================================


@dataclass(frozen=True)
class NullCacheProvider:
    """Testing default. All operations are no-ops; :meth:`supports_caching`
    is ``False``.

    Always use :class:`NullCacheProvider` in unit tests so cache logic
    never interferes with assertions.
    """

    def supports_caching(self) -> bool:
        return False

    def annotate(self, context: Context) -> CacheAnnotationResult:  # noqa: ARG002
        return CacheAnnotationResult()

    def parse_cache_stats(self, response: ModelResponse) -> CacheStats | None:  # noqa: ARG002
        return None

    def provider_name(self) -> str:
        return "null"


def _anthropic_cache_pricing(model_id: str) -> tuple[float, float]:
    """Return ``(cache_read_usd_per_million, cache_write_usd_per_million)``.

    Anthropic cache pricing (USD per 1M tokens, 5-minute TTL):

    - opus-4.x:   1.50 read / 18.75 write
    - sonnet-4.x: 0.30 read /  3.75 write
    - haiku-4.x:  0.08 read /  1.00 write

    Substring matching: any ``model_id`` containing "opus" gets opus
    pricing, "haiku" gets haiku pricing, everything else (sonnet, unknown)
    gets sonnet pricing.
    """

    lower = model_id.lower()
    if "opus" in lower:
        return (1.50, 18.75)
    if "haiku" in lower:
        return (0.08, 1.00)
    # sonnet, unknown, default → sonnet pricing
    return (0.30, 3.75)


@dataclass(frozen=True)
class AnthropicCacheProvider:
    """Anthropic prefix caching.

    Inserts logical ``cache_control: ephemeral`` breakpoints after each
    stable block boundary (Block 1: Static, Block 2: PerSession, plus
    history and optional tool-schema anchors). Reads
    ``cache_read_tokens`` and ``cache_write_tokens`` from response usage.
    """

    #: Anthropic supports up to 4 breakpoints per request.
    max_cache_anchors: int = 4
    #: USD per 1M tokens for cache reads. Default matches Sonnet 4.x
    #: published pricing (0.30 USD / 1M cache-read tokens).
    cache_read_usd_per_million: float = 0.30
    #: USD per 1M tokens for cache writes (5-minute TTL). Default matches
    #: Sonnet 4.x (3.75 USD / 1M cache-write tokens).
    cache_write_usd_per_million: float = 3.75

    def with_model_pricing(self, model_id: str) -> AnthropicCacheProvider:
        """Return a copy with cache pricing overridden for ``model_id``.

        Pricing data lives in the implementation so callers don't have to
        import a table — pass the model id and we look it up. Unknown ids
        return Sonnet pricing. See :func:`_anthropic_cache_pricing`.
        """

        read, write = _anthropic_cache_pricing(model_id)
        return AnthropicCacheProvider(
            max_cache_anchors=self.max_cache_anchors,
            cache_read_usd_per_million=read,
            cache_write_usd_per_million=write,
        )

    def supports_caching(self) -> bool:
        return True

    def annotate(self, context: Context) -> CacheAnnotationResult:
        # Anchors are derived from rendered system-prompt breakpoints
        # (Block-1 / Block-2 boundaries) plus an optional history anchor
        # if there are any prior messages. Cap at ``max_cache_anchors``.
        existing = list(context.system_prompt.breakpoints)
        anchors = len(existing)

        history_anchor_eligible = len(context.messages) > 0
        if history_anchor_eligible and anchors < self.max_cache_anchors:
            context.system_prompt.breakpoints.append(
                BreakpointInfo(
                    after_segment="__history_tail__",
                    token_offset=context.token_count,
                )
            )
            anchors += 1

        markers = min(anchors, self.max_cache_anchors)
        estimated = 0 if markers == 0 else context.token_count
        return CacheAnnotationResult(
            markers_inserted=markers,
            estimated_cacheable_tokens=estimated,
        )

    def parse_cache_stats(self, response: ModelResponse) -> CacheStats | None:
        read = response.usage.cache_read_tokens
        write = response.usage.cache_write_tokens
        if read is None and write is None:
            return None
        read_tokens = read or 0
        write_tokens = write or 0
        return CacheStats(
            cache_read_tokens=read_tokens,
            cache_write_tokens=write_tokens,
            cache_read_cost_usd=read_tokens / 1_000_000.0 * self.cache_read_usd_per_million,
            cache_write_cost_usd=write_tokens / 1_000_000.0 * self.cache_write_usd_per_million,
        )

    def provider_name(self) -> str:
        return "anthropic"


@dataclass(frozen=True)
class OpenAICacheProvider:
    """OpenAI prefix caching.

    OpenAI caches automatically on prompts above ``min_cacheable_tokens``
    (1024 by default) — no explicit markers required. :meth:`annotate` is
    a no-op (returning zeros). :meth:`parse_cache_stats` reads
    ``cache_read_tokens`` from the response. OpenAI does not return a
    "cache write" count, so writes remain zero.
    """

    #: Below this token count OpenAI will not cache.
    min_cacheable_tokens: int = 1024

    def supports_caching(self) -> bool:
        return True

    def annotate(self, context: Context) -> CacheAnnotationResult:
        # No markers needed — OpenAI caches automatically.
        cacheable = context.token_count if context.token_count >= self.min_cacheable_tokens else 0
        return CacheAnnotationResult(
            markers_inserted=0,
            estimated_cacheable_tokens=cacheable,
        )

    def parse_cache_stats(self, response: ModelResponse) -> CacheStats | None:
        read = response.usage.cache_read_tokens
        if read is None:
            return None
        return CacheStats(
            cache_read_tokens=read,
            cache_write_tokens=0,
            cache_read_cost_usd=0.0,
            cache_write_cost_usd=0.0,
        )

    def provider_name(self) -> str:
        return "openai"


@dataclass(frozen=True)
class OllamaCacheProvider:
    """Ollama has no prefix caching. Every method is a no-op."""

    def supports_caching(self) -> bool:
        return False

    def annotate(self, context: Context) -> CacheAnnotationResult:  # noqa: ARG002
        return CacheAnnotationResult()

    def parse_cache_stats(self, response: ModelResponse) -> CacheStats | None:  # noqa: ARG002
        return None

    def provider_name(self) -> str:
        return "ollama"


# ============================================================================
# Auto-detection from model provider name
# ============================================================================


def auto_detect_cache_provider(provider_name: str) -> CacheProvider | None:
    """Map a model provider name to the appropriate :class:`CacheProvider`.

    Returns ``None`` when the provider is unknown — the caller (typically
    ``HarnessBuilder``) should emit a ``CacheProviderNotDetected`` warning
    and fall back to :class:`NullCacheProvider`. Matching is
    case-insensitive.
    """

    key = provider_name.lower()
    if key == "anthropic":
        return AnthropicCacheProvider()
    if key == "openai":
        return OpenAICacheProvider()
    if key == "ollama":
        return OllamaCacheProvider()
    return None


__all__ = [
    "AnthropicCacheProvider",
    "CacheAnnotationResult",
    "CacheProvider",
    "CacheStats",
    "NullCacheProvider",
    "OllamaCacheProvider",
    "OpenAICacheProvider",
    "auto_detect_cache_provider",
]
