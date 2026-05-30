/**
 * StandardContextManager — canonical {@link ContextManager} (spore-core
 * issue #7). Mirrors `rust/crates/spore-core/src/context.rs` — same
 * assembly order, same rules, same fixture outcomes.
 *
 * Block 1 comes from a pre-computed {@link ComposedPrompt}. Block 2
 * (per-session) is built from {@link SessionState}. Block 3 (per-turn)
 * holds the budget warning and pending skill injections. Tool schemas
 * are sorted by name. The {@link CacheProvider} is invoked at the end
 * of `assemble` to annotate provider-specific cache markers.
 */

import type { Message, ModelInterface, ModelRequest, ToolSchema } from "../model/index.js";
import type { SandboxProvider } from "../harness/types.js";

import {
  type BreakpointInfo,
  type CacheBlockHits,
  type CacheProvider,
  type CompactionConfig,
  type CompactionRequest,
  type CompactionResult,
  type Context,
  type ContextError,
  ContextErrorException,
  type ContextManager,
  type ContextSources,
  type Guide,
  type PromptSegment,
  type RenderedSystemPrompt,
  type SessionState,
  type ToolResult,
  defaultCompactionConfig,
  defaultCompactionPreserveHints,
  emptyCacheBlockStatus,
} from "./types.js";

// ============================================================================
// Hash helpers — FNV-1a 64-bit (deterministic across runs and processes).
// ============================================================================

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const U64_MASK = 0xffffffffffffffffn;

function fnv1aMix(h: bigint, byte: number): bigint {
  return ((h ^ BigInt(byte & 0xff)) * FNV_PRIME) & U64_MASK;
}

function fnv1aHashString(h: bigint, s: string): bigint {
  // Hash the UTF-8 bytes so output matches what other languages produce
  // for the same logical string.
  const encoded = new TextEncoder().encode(s);
  for (let i = 0; i < encoded.length; i += 1) {
    h = fnv1aMix(h, encoded[i]!);
  }
  // Length-prefixed-style separator to avoid collisions across
  // concatenations like ["a","bc"] vs ["ab","c"].
  return fnv1aMix(h, 0);
}

function segmentsHash(segments: readonly PromptSegment[]): number {
  let h = FNV_OFFSET;
  const stabilityByte: Record<string, number> = {
    static: 0,
    per_session: 1,
    per_turn: 2,
  };
  for (const s of segments) {
    h = fnv1aHashString(h, s.name);
    h = fnv1aHashString(h, s.content);
    h = fnv1aMix(h, stabilityByte[s.stability] ?? 0);
    h = fnv1aMix(h, s.cache_breakpoint ? 1 : 0);
  }
  // Truncate to a JS-safe integer for downstream comparison. We only need
  // change-detection, not cryptographic strength.
  return Number(h & 0x1fffffffffffffn);
}

// ============================================================================
// Rendering
// ============================================================================

function renderSegments(
  block_1: string,
  segments: readonly PromptSegment[],
): { content: string; breakpoints: BreakpointInfo[] } {
  let content = block_1;
  const breakpoints: BreakpointInfo[] = [
    // Block 1 always ends with an implicit breakpoint (spec: cache_provider
    // inserts a breakpoint after Block 1).
    { after_segment: "__block_1__", token_offset: Math.floor(content.length / 4) },
  ];
  for (const seg of segments) {
    if (!content.endsWith("\n")) content += "\n";
    content += seg.content;
    if (seg.cache_breakpoint) {
      breakpoints.push({
        after_segment: seg.name,
        token_offset: Math.floor(content.length / 4),
      });
    }
  }
  return { content, breakpoints };
}

// ============================================================================
// StandardContextManager
// ============================================================================

interface CacheHashMemo {
  static_hash: number | null;
  session_hash: number | null;
}

export interface StandardContextManagerOptions {
  /**
   * Sandbox-side threshold (bytes) above which `appendToolResult` head+tail
   * truncates via {@link SandboxProvider.handleLargeOutput}. Defaults to
   * 32 KiB.
   */
  offloadThresholdBytes?: number;
}

export class StandardContextManager implements ContextManager {
  private readonly model: ModelInterface;
  private readonly cacheProvider: CacheProvider;
  private readonly compaction: CompactionConfig;
  private readonly offloadThresholdBytes: number;
  private readonly memo: CacheHashMemo = { static_hash: null, session_hash: null };

  constructor(
    model: ModelInterface,
    cacheProvider: CacheProvider,
    compaction: CompactionConfig = defaultCompactionConfig(),
    options: StandardContextManagerOptions = {},
  ) {
    this.model = model;
    this.cacheProvider = cacheProvider;
    this.compaction = compaction;
    this.offloadThresholdBytes = options.offloadThresholdBytes ?? 32 * 1024;
  }

  async assemble(state: SessionState, sources: ContextSources): Promise<Context> {
    // ── BLOCK 1 hash check ─────────────────────────────────────────────
    const staticHash = sources.composed_prompt.block_1_hash;
    if (this.memo.static_hash !== null) {
      if (this.memo.static_hash !== staticHash) {
        throw new ContextErrorException({
          kind: "CacheHashMismatch",
          block: "static",
          expected: this.memo.static_hash,
          actual: staticHash,
          turn_number: state.turn_number,
        });
      }
    } else {
      this.memo.static_hash = staticHash;
    }

    // ── BLOCK 2 (PerSession) ───────────────────────────────────────────
    const segments = buildSessionSegments(state);
    const sessionHash = segmentsHash(segments);
    if (this.memo.session_hash !== null) {
      if (this.memo.session_hash !== sessionHash && state.turn_number > 1) {
        // Block 2 (PerSession) is expected to be stable for the life of the
        // session. A mid-session change means cost would silently spike; halt
        // consistently with Block 1 (#32). We throw BEFORE updating the memo —
        // the run is halting, so there is no "rest of the session" to track.
        throw new ContextErrorException({
          kind: "CacheHashMismatch",
          block: "per_session",
          expected: this.memo.session_hash,
          actual: sessionHash,
          turn_number: state.turn_number,
        });
      }
    }
    this.memo.session_hash = sessionHash;

    // ── BLOCK 3 (PerTurn, never cached) ────────────────────────────────
    if (state.budget_warning_active) {
      segments.push({
        name: "budget_warning",
        content: `[BUDGET] ${state.token_budget_used} of ${state.window_limit} tokens used.`,
        stability: "per_turn",
        cache_breakpoint: false,
      });
    }
    for (const skill of state.pending_skill_injections) {
      segments.push({
        name: `skill:${skill.id.asString()}`,
        content: skill.content,
        stability: "per_turn",
        cache_breakpoint: false,
      });
    }

    // ── Render ─────────────────────────────────────────────────────────
    const rendered = renderSegments(sources.composed_prompt.rendered, segments);
    const system_prompt: RenderedSystemPrompt = {
      content: rendered.content,
      breakpoints: rendered.breakpoints,
      static_block_hash: staticHash,
      session_block_hash: sessionHash,
    };

    // ── Tool schemas: sort by name (spec rule) ─────────────────────────
    const tool_schemas: ToolSchema[] = [...sources.tool_schemas].sort((a, b) =>
      a.name < b.name ? -1 : a.name > b.name ? 1 : 0,
    );

    // ── Message history ────────────────────────────────────────────────
    const messages: Message[] = [...state.message_history];

    // ── Token count (from ModelInterface, not estimated) ───────────────
    const req: ModelRequest = {
      messages: [
        { role: "system", content: { type: "text", text: system_prompt.content } },
        ...messages,
      ],
      tools: tool_schemas,
      params: { stop_sequences: [] },
      stream: false,
    };
    let token_count: number;
    try {
      token_count = await this.model.countTokens(req);
    } catch {
      throw new ContextErrorException({ kind: "TokenCountFailed" });
    }
    const utilization = state.window_limit === 0 ? 0 : token_count / state.window_limit;

    const context: Context = {
      system_prompt,
      messages,
      tool_schemas,
      token_count,
      window_limit: state.window_limit,
      utilization,
      meta: {
        session_id: state.session_id,
        turn_number: state.turn_number,
        active_phase: state.active_phase,
        guides_loaded: [...state.guides_loaded],
        skills_injected: state.pending_skill_injections.map((g) => g.id),
        compacted: false,
        cache_blocks: emptyCacheBlockStatus(),
      },
    };

    this.cacheProvider.annotate(context);
    return context;
  }

  async appendToolResult(
    state: SessionState,
    result: ToolResult,
    sandbox: SandboxProvider,
  ): Promise<void> {
    const text = renderToolOutput(result);

    // Spec rule: head+tail truncate large outputs, offload full to filesystem.
    let finalText: string;
    if (text.length > this.offloadThresholdBytes && sandbox.handleLargeOutput) {
      const truncated = await sandbox.handleLargeOutput(
        text,
        result.call_id,
        this.compaction.head_tail_tokens,
        this.compaction.head_tail_tokens,
      );
      finalText = formatTruncated(truncated.content, truncated.full_ref);
    } else {
      finalText = text;
    }

    state.message_history.push({
      role: "tool",
      content: { type: "text", text: finalText },
    });
  }

  appendResponse(state: SessionState, response: string): void {
    state.message_history.push({
      role: "assistant",
      content: { type: "text", text: response },
    });
  }

  shouldCompact(state: SessionState): boolean {
    if (state.window_limit === 0) return false;
    return state.token_budget_used / state.window_limit >= this.compaction.threshold;
  }

  prepareCompaction(state: SessionState): CompactionRequest {
    const n = state.message_history.length;
    const keep = this.compaction.preserve_recent_n;
    if (n <= keep) {
      return {
        messages_to_compact: [],
        preserve_hints: defaultCompactionPreserveHints(),
      };
    }
    const cut = n - keep;
    return {
      messages_to_compact: state.message_history.slice(0, cut),
      preserve_hints: defaultCompactionPreserveHints(),
    };
  }

  applyCompaction(state: SessionState, result: CompactionResult): void {
    const n = state.message_history.length;
    const keep = this.compaction.preserve_recent_n;
    if (n <= keep) {
      const err: ContextError = {
        kind: "CompactionFailed",
        reason: "history shorter than preserve_recent_n",
      };
      throw new ContextErrorException(err);
    }
    const cut = n - keep;
    const newHistory: Message[] = [result.summary_message, ...state.message_history.slice(cut)];
    state.message_history = newHistory;
    state.token_budget_used = Math.max(0, state.token_budget_used - result.tokens_reclaimed);
  }

  injectSkill(context: Context, skill: Guide): void {
    // Block-3 ephemeral injection: append to system prompt content, do
    // not modify message history, do not invalidate Block 1 or Block 2
    // (their hashes are untouched).
    if (!context.system_prompt.content.endsWith("\n")) {
      context.system_prompt.content += "\n";
    }
    context.system_prompt.content += `[SKILL:${skill.id.asString()}]\n${skill.content}`;
    context.meta.skills_injected.push(skill.id);
  }

  recordCacheResult(context: Context, cacheStats: CacheBlockHits): void {
    context.meta.cache_blocks = {
      static_hit: cacheStats.static_hit,
      session_hit: cacheStats.session_hit,
      history_hit: cacheStats.history_hit,
    };
  }
}

// ============================================================================
// Helpers
// ============================================================================

function buildSessionSegments(state: SessionState): PromptSegment[] {
  // Order is load-bearing for prefix-cache stability.
  return [
    {
      name: "task",
      content: state.task_instruction,
      stability: "per_session",
      cache_breakpoint: false,
    },
    {
      name: "environment",
      content: state.environment,
      stability: "per_session",
      cache_breakpoint: false,
    },
    {
      name: "prior_state",
      content: state.prior_state,
      stability: "per_session",
      cache_breakpoint: false,
    },
    {
      name: "operational",
      content: state.operational_instructions,
      stability: "per_session",
      cache_breakpoint: true,
    },
  ];
}

function renderToolOutput(result: ToolResult): string {
  switch (result.output.kind) {
    case "success":
      return result.output.content;
    case "error":
      return `[error] ${result.output.message}`;
    case "waiting_for_human":
      return "[waiting]";
    case "escalate":
      return "[escalate]";
    case "awaiting_clarification":
      return "[awaiting clarification]";
  }
}

function formatTruncated(headTail: string, fullRef: { path: string; size: number } | null): string {
  if (fullRef) {
    return `${headTail}\n\n[truncated; full output at ${fullRef.path} (${fullRef.size} bytes)]`;
  }
  return `${headTail}\n\n[truncated]`;
}
