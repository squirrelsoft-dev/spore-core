/**
 * Unit tests for {@link CacheProvider} (spore-core issue #25).
 *
 * Mirrors `rust/crates/spore-core/src/cache_provider.rs#tests` — same rules,
 * same verdicts, parallel structure.
 */

import { describe, expect, it } from "vitest";

import { cacheProvider, context, SessionId } from "../src/index.js";
import type { ModelResponse } from "../src/index.js";

const {
  AnthropicCacheProvider,
  NullCacheProvider,
  OllamaCacheProvider,
  OpenAICacheProvider,
  autoDetectCacheProvider,
} = cacheProvider;

type Context = context.Context;
type BreakpointInfo = context.BreakpointInfo;

function ctx(tokens: number, breakpoints: BreakpointInfo[], msgs: number): Context {
  const messages = Array.from({ length: msgs }, () => ({
    role: "user" as const,
    content: { type: "text" as const, text: "m" },
  }));
  return {
    system_prompt: {
      content: "system",
      breakpoints: [...breakpoints],
      static_block_hash: 0,
      session_block_hash: 0,
    },
    messages,
    tool_schemas: [],
    token_count: tokens,
    window_limit: 200_000,
    utilization: 0,
    meta: {
      session_id: SessionId.of("s"),
      turn_number: 0,
      active_phase: "execution",
      guides_loaded: [],
      skills_injected: [],
      compacted: false,
      cache_blocks: { static_hit: null, session_hit: null, history_hit: null },
    },
  };
}

function response(read: number | null, write: number | null): ModelResponse {
  return {
    content: [{ type: "text", text: "hi" }],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: read,
      cache_write_tokens: write,
    },
    stop_reason: "end_turn",
  };
}

describe("NullCacheProvider", () => {
  // Rule: Null provider is a no-op.
  it("does nothing", () => {
    const p = new NullCacheProvider();
    expect(p.supportsCaching()).toBe(false);
    expect(p.providerName()).toBe("null");
    const c = ctx(100, [], 0);
    const r = p.annotate(c);
    expect(r).toEqual({ markers_inserted: 0, estimated_cacheable_tokens: 0 });
    expect(p.parseCacheStats(response(5, null))).toBeNull();
  });
});

describe("AnthropicCacheProvider", () => {
  // Rule: supports caching and reports its name; default max_cache_anchors = 4.
  it("identity", () => {
    const p = new AnthropicCacheProvider();
    expect(p.supportsCaching()).toBe(true);
    expect(p.providerName()).toBe("anthropic");
    expect(p.max_cache_anchors).toBe(4);
  });

  // Rule: annotate caps at max_cache_anchors (no history anchor when at cap).
  it("annotate caps at max_cache_anchors", () => {
    const p = new AnthropicCacheProvider({ max_cache_anchors: 2 });
    const bps: BreakpointInfo[] = [
      { after_segment: "block_1_static", token_offset: 10 },
      { after_segment: "block_2_per_session", token_offset: 20 },
    ];
    const c = ctx(50, bps, 3);
    const r = p.annotate(c);
    expect(r.markers_inserted).toBe(2);
    expect(r.estimated_cacheable_tokens).toBeGreaterThan(0);
    expect(c.system_prompt.breakpoints.length).toBe(2);
  });

  // Rule: annotate adds a history anchor when room remains and history present.
  it("annotate adds history anchor when room and history present", () => {
    const p = new AnthropicCacheProvider();
    const bps: BreakpointInfo[] = [{ after_segment: "block_1_static", token_offset: 10 }];
    const c = ctx(75, bps, 4);
    const r = p.annotate(c);
    expect(r.markers_inserted).toBe(2);
    expect(c.system_prompt.breakpoints.length).toBe(2);
    expect(c.system_prompt.breakpoints[1]?.after_segment).toBe("__history_tail__");
  });

  // Rule: annotate returns 0 markers when no history and no existing breakpoints.
  it("annotate zero when empty", () => {
    const p = new AnthropicCacheProvider();
    const c = ctx(50, [], 0);
    const r = p.annotate(c);
    expect(r.markers_inserted).toBe(0);
    expect(r.estimated_cacheable_tokens).toBe(0);
  });

  // Rule: parseCacheStats returns null when no metadata.
  it("parseCacheStats null without metadata", () => {
    const p = new AnthropicCacheProvider();
    expect(p.parseCacheStats(response(null, null))).toBeNull();
  });

  // Rule: parseCacheStats reads read/write tokens.
  it("parseCacheStats reads tokens", () => {
    const p = new AnthropicCacheProvider();
    const s = p.parseCacheStats(response(900, 120));
    expect(s).not.toBeNull();
    expect(s!.cache_read_tokens).toBe(900);
    expect(s!.cache_write_tokens).toBe(120);
  });

  // Rule: one-sided metadata is non-null (attempted but one direction).
  it("parseCacheStats one-sided is non-null", () => {
    const p = new AnthropicCacheProvider();
    const s = p.parseCacheStats(response(0, null));
    expect(s).not.toBeNull();
    expect(s!.cache_read_tokens).toBe(0);
    expect(s!.cache_write_tokens).toBe(0);
  });
});

describe("OpenAICacheProvider", () => {
  // Rule: annotate is a no-op and counts cacheable tokens only above the threshold.
  it("annotate threshold", () => {
    const p = new OpenAICacheProvider();
    const below = ctx(1023, [], 0);
    const r1 = p.annotate(below);
    expect(r1.markers_inserted).toBe(0);
    expect(r1.estimated_cacheable_tokens).toBe(0);

    const above = ctx(2048, [], 0);
    const r2 = p.annotate(above);
    expect(r2.markers_inserted).toBe(0);
    expect(r2.estimated_cacheable_tokens).toBe(2048);
  });

  it("supports caching and default min_cacheable_tokens is 1024", () => {
    const p = new OpenAICacheProvider();
    expect(p.supportsCaching()).toBe(true);
    expect(p.providerName()).toBe("openai");
    expect(p.min_cacheable_tokens).toBe(1024);
  });

  // Rule: parseCacheStats reads cached_tokens; write is forced to zero.
  it("parseCacheStats reads only reads", () => {
    const p = new OpenAICacheProvider();
    const s = p.parseCacheStats(response(512, 99));
    expect(s).not.toBeNull();
    expect(s!.cache_read_tokens).toBe(512);
    expect(s!.cache_write_tokens).toBe(0);
    expect(p.parseCacheStats(response(null, null))).toBeNull();
  });
});

describe("OllamaCacheProvider", () => {
  // Rule: supportsCaching is false; all ops are no-ops.
  it("is a complete no-op", () => {
    const p = new OllamaCacheProvider();
    expect(p.supportsCaching()).toBe(false);
    expect(p.providerName()).toBe("ollama");
    const c = ctx(99, [], 0);
    expect(p.annotate(c)).toEqual({
      markers_inserted: 0,
      estimated_cacheable_tokens: 0,
    });
    expect(p.parseCacheStats(response(5, 5))).toBeNull();
  });
});

describe("autoDetectCacheProvider", () => {
  // Rule: maps provider names case-insensitively; unknown returns null.
  it("maps known providers", () => {
    expect(autoDetectCacheProvider("anthropic")?.providerName()).toBe("anthropic");
    expect(autoDetectCacheProvider("OpenAI")?.providerName()).toBe("openai");
    expect(autoDetectCacheProvider("ollama")?.providerName()).toBe("ollama");
    expect(autoDetectCacheProvider("mystery")).toBeNull();
  });
});
