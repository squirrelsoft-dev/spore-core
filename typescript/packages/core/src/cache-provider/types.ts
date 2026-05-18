/**
 * CacheProvider — canonical types (spore-core issue #25).
 *
 * Mirrors `rust/crates/spore-core/src/cache_provider.rs`. Provider-specific
 * cache annotation and stats parsing. Cache control is provider-specific at
 * the API level — Anthropic uses explicit `cache_control` markers, OpenAI
 * caches automatically above a token threshold, Ollama has no caching.
 *
 * Flow (see `docs/harness-engineering-concepts.md` §"Cache Architecture"):
 *
 * ```text
 * ContextManager.assemble():
 *   ... build and render segments ...
 *   if cache_provider.supportsCaching():
 *     cache_provider.annotate(context)
 *   return context
 *
 * // After each model response:
 * stats = cache_provider.parseCacheStats(response)
 * observability.emitCacheStats(session_id, stats)
 * ```
 */

import { z } from "zod";

import type { Context, BreakpointInfo } from "../context/types.js";
import type { ModelResponse } from "../model/schemas.js";

// ============================================================================
// Spec-defined types
// ============================================================================

/** Result of annotating a context with provider-specific cache markers. */
export const CacheAnnotationResultSchema = z.object({
  markers_inserted: z.number().int().nonnegative(),
  estimated_cacheable_tokens: z.number().int().nonnegative(),
});
export type CacheAnnotationResult = z.infer<typeof CacheAnnotationResultSchema>;

export function emptyCacheAnnotationResult(): CacheAnnotationResult {
  return { markers_inserted: 0, estimated_cacheable_tokens: 0 };
}

/**
 * Cache token usage parsed from a single model response.
 *
 * `null` from {@link CacheProvider.parseCacheStats} means the response had no
 * cache metadata at all (caching wasn't attempted). A `CacheStats` with
 * all-zero fields means caching was attempted and missed — the distinction
 * matters for observability.
 */
export const CacheStatsSchema = z.object({
  cache_read_tokens: z.number().int().nonnegative(),
  cache_write_tokens: z.number().int().nonnegative(),
  cache_read_cost_usd: z.number(),
  cache_write_cost_usd: z.number(),
});
export type CacheStats = z.infer<typeof CacheStatsSchema>;

export function emptyCacheStats(): CacheStats {
  return {
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cache_read_cost_usd: 0,
    cache_write_cost_usd: 0,
  };
}

// ============================================================================
// Trait
// ============================================================================

export interface CacheProvider {
  /** Whether this provider supports prefix caching at all. */
  supportsCaching(): boolean;

  /**
   * Annotate a fully assembled context with provider-specific cache markers.
   * Called by {@link "../context/types.js".ContextManager} after assembly,
   * before sending to {@link "../model/index.js".ModelInterface}. No-op when
   * {@link supportsCaching} returns false.
   */
  annotate(context: Context): CacheAnnotationResult;

  /**
   * Parse cache usage from a model response. Returns `null` when the response
   * has no cache metadata at all.
   */
  parseCacheStats(response: ModelResponse): CacheStats | null;

  /** Provider identity — used for observability and auto-detection. */
  providerName(): string;
}

// ============================================================================
// Standard implementations
// ============================================================================

/**
 * Testing default. All operations are no-ops; {@link supportsCaching} is false.
 *
 * Always use `NullCacheProvider` in unit tests so cache logic never interferes
 * with assertions.
 */
export class NullCacheProvider implements CacheProvider {
  supportsCaching(): boolean {
    return false;
  }
  annotate(_context: Context): CacheAnnotationResult {
    void _context;
    return emptyCacheAnnotationResult();
  }
  parseCacheStats(_response: ModelResponse): CacheStats | null {
    void _response;
    return null;
  }
  providerName(): string {
    return "null";
  }
}

export interface AnthropicCacheProviderOptions {
  /** Anthropic supports up to 4 breakpoints per request. Defaults to 4. */
  max_cache_anchors?: number;
}

/**
 * Anthropic prefix caching.
 *
 * Inserts logical `cache_control: ephemeral` breakpoints after each stable
 * block boundary (Block 1: Static, Block 2: PerSession, plus history and
 * optional tool-schema anchors). Reads `cache_read_tokens` and
 * `cache_write_tokens` from response usage.
 */
export class AnthropicCacheProvider implements CacheProvider {
  readonly max_cache_anchors: number;

  constructor(options: AnthropicCacheProviderOptions = {}) {
    this.max_cache_anchors = options.max_cache_anchors ?? 4;
  }

  supportsCaching(): boolean {
    return true;
  }

  annotate(context: Context): CacheAnnotationResult {
    // Anchors are derived from rendered system-prompt breakpoints
    // (Block-1 / Block-2 boundaries) plus an optional history anchor
    // if there are any prior messages. Cap at max_cache_anchors.
    const existing: BreakpointInfo[] = context.system_prompt.breakpoints;
    let anchors = existing.length;

    const historyAnchorEligible = context.messages.length > 0;
    if (historyAnchorEligible && anchors < this.max_cache_anchors) {
      context.system_prompt.breakpoints.push({
        after_segment: "__history_tail__",
        token_offset: context.token_count,
      });
      anchors += 1;
    }

    const markers = Math.min(anchors, this.max_cache_anchors);
    const estimated = markers === 0 ? 0 : context.token_count;
    return {
      markers_inserted: markers,
      estimated_cacheable_tokens: estimated,
    };
  }

  parseCacheStats(response: ModelResponse): CacheStats | null {
    const read = response.usage.cache_read_tokens;
    const write = response.usage.cache_write_tokens;
    if ((read === null || read === undefined) && (write === null || write === undefined)) {
      return null;
    }
    return {
      cache_read_tokens: read ?? 0,
      cache_write_tokens: write ?? 0,
      cache_read_cost_usd: 0,
      cache_write_cost_usd: 0,
    };
  }

  providerName(): string {
    return "anthropic";
  }
}

export interface OpenAICacheProviderOptions {
  /** Below this token count OpenAI will not cache. Defaults to 1024. */
  min_cacheable_tokens?: number;
}

/**
 * OpenAI prefix caching.
 *
 * OpenAI caches automatically on prompts above `min_cacheable_tokens` (1024
 * by default) — no explicit markers required. {@link annotate} is a no-op
 * (returning zeros for markers). {@link parseCacheStats} reads
 * `cache_read_tokens` from the response. OpenAI does not return a "cache
 * write" count, so writes remain zero.
 */
export class OpenAICacheProvider implements CacheProvider {
  readonly min_cacheable_tokens: number;

  constructor(options: OpenAICacheProviderOptions = {}) {
    this.min_cacheable_tokens = options.min_cacheable_tokens ?? 1024;
  }

  supportsCaching(): boolean {
    return true;
  }

  annotate(context: Context): CacheAnnotationResult {
    const cacheable = context.token_count >= this.min_cacheable_tokens ? context.token_count : 0;
    return {
      markers_inserted: 0,
      estimated_cacheable_tokens: cacheable,
    };
  }

  parseCacheStats(response: ModelResponse): CacheStats | null {
    const read = response.usage.cache_read_tokens;
    if (read === null || read === undefined) {
      return null;
    }
    return {
      cache_read_tokens: read,
      cache_write_tokens: 0,
      cache_read_cost_usd: 0,
      cache_write_cost_usd: 0,
    };
  }

  providerName(): string {
    return "openai";
  }
}

/** Ollama has no prefix caching. Every method is a no-op. */
export class OllamaCacheProvider implements CacheProvider {
  supportsCaching(): boolean {
    return false;
  }
  annotate(_context: Context): CacheAnnotationResult {
    void _context;
    return emptyCacheAnnotationResult();
  }
  parseCacheStats(_response: ModelResponse): CacheStats | null {
    void _response;
    return null;
  }
  providerName(): string {
    return "ollama";
  }
}

// ============================================================================
// Auto-detection from model provider name
// ============================================================================

/**
 * Map a {@link "../model/index.js".ModelInterface.provider} name to the
 * appropriate {@link CacheProvider}. Returns `null` when the provider is
 * unknown — the caller (typically `HarnessBuilder`) should emit a
 * `CacheProviderNotDetected` warning and fall back to {@link NullCacheProvider}.
 */
export function autoDetectCacheProvider(providerName: string): CacheProvider | null {
  switch (providerName.toLowerCase()) {
    case "anthropic":
      return new AnthropicCacheProvider();
    case "openai":
      return new OpenAICacheProvider();
    case "ollama":
      return new OllamaCacheProvider();
    default:
      return null;
  }
}
