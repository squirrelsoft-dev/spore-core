/**
 * ContextManager — canonical types (spore-core issue #7).
 *
 * Mirrors `rust/crates/spore-core/src/context.rs`. Same field names
 * (snake_case on the wire), same enum variants, same assembly rules —
 * shared fixtures must produce identical outcomes across languages.
 *
 * The harness module ships a narrower forward-declared
 * {@link "../harness/types.js".ContextManager} used by the existing
 * runtime loop. The canonical interface defined here will replace it
 * once downstream consumers migrate.
 */

import { z } from "zod";

import type { Message, ToolSchema } from "../model/schemas.js";
import type { SandboxProvider } from "../harness/types.js";
import { SessionId, TaskId, SessionIdSchema, TaskIdSchema } from "../harness/types.js";
import { TaskPhaseSchema, type TaskPhase } from "../tool-registry/types.js";
import { MessageSchema } from "../model/schemas.js";
import type { CacheBlock } from "../prompt-chunk-registry/types.js";

// ============================================================================
// Forward-declared sibling types (issues #8, #9, #14)
// ============================================================================

/** Stable identifier for a guide or skill (issue #9 — GuideRegistry). */
export class GuideId {
  constructor(readonly value: string) {}
  static of(value: string): GuideId {
    return new GuideId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: GuideId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export const GuideIdSchema = z.string().transform((s) => new GuideId(s));

/**
 * Forward-declared `Guide` (issue #9). Carries the rendered chunk and an
 * identifier; full lifecycle metadata lives with `GuideRegistry`.
 *
 * `content` must be the final rendered form — the spec forbids reformatting
 * at assembly time.
 */
export interface Guide {
  id: GuideId;
  content: string;
}

/** Forward-declared `MemoryItem` (issue #8 — MemoryProvider). */
export interface MemoryItem {
  key: string;
  content: string;
}

/**
 * Forward-declared `ComposedPrompt` (issue #14 — PromptChunkRegistry).
 *
 * Block 1 is computed ONCE at harness startup. `rendered` is the final
 * byte-for-byte content; `block_1_hash` is a stable digest used by the
 * {@link ContextManager} to detect unexpected cache invalidation.
 */
export interface ComposedPrompt {
  rendered: string;
  block_1_hash: number;
}

/**
 * Per-block cache hit signal recorded into {@link ContextMeta} after each
 * model response. Distinct from {@link "../cache-provider/types.js".CacheStats},
 * which carries token counts and costs parsed from the response.
 *
 * `null` means the provider has no signal for that block (e.g. the provider
 * does not support caching at all).
 */
export interface CacheBlockHits {
  static_hit: boolean | null;
  session_hit: boolean | null;
  history_hit: boolean | null;
}

export function emptyCacheBlockHits(): CacheBlockHits {
  return { static_hit: null, session_hit: null, history_hit: null };
}

// `CacheProvider` and `NullCacheProvider` are owned by the canonical
// cache-provider module (issue #25). Re-exported here so existing context
// callers keep working.
export {
  type CacheProvider,
  NullCacheProvider,
  type CacheStats,
  type CacheAnnotationResult,
} from "../cache-provider/types.js";

// ============================================================================
// Spec-defined types
// ============================================================================

export const SegmentStabilitySchema = z.enum(["static", "per_session", "per_turn"]);
export type SegmentStability = z.infer<typeof SegmentStabilitySchema>;

export const PromptSegmentSchema = z.object({
  name: z.string(),
  content: z.string(),
  stability: SegmentStabilitySchema,
  cache_breakpoint: z.boolean().default(false),
});
export type PromptSegment = z.infer<typeof PromptSegmentSchema>;

export const BreakpointInfoSchema = z.object({
  after_segment: z.string(),
  token_offset: z.number().int().nonnegative(),
});
export type BreakpointInfo = z.infer<typeof BreakpointInfoSchema>;

export const RenderedSystemPromptSchema = z.object({
  content: z.string(),
  breakpoints: z.array(BreakpointInfoSchema).default([]),
  static_block_hash: z.number(),
  session_block_hash: z.number(),
});
export type RenderedSystemPrompt = z.infer<typeof RenderedSystemPromptSchema>;

export const CacheBlockStatusSchema = z.object({
  static_hit: z.boolean().nullable().default(null),
  session_hit: z.boolean().nullable().default(null),
  history_hit: z.boolean().nullable().default(null),
});
export type CacheBlockStatus = z.infer<typeof CacheBlockStatusSchema>;

export function emptyCacheBlockStatus(): CacheBlockStatus {
  return { static_hit: null, session_hit: null, history_hit: null };
}

export const ContextMetaSchema = z.object({
  session_id: SessionIdSchema,
  turn_number: z.number().int().nonnegative(),
  active_phase: TaskPhaseSchema,
  guides_loaded: z.array(GuideIdSchema).default([]),
  skills_injected: z.array(GuideIdSchema).default([]),
  compacted: z.boolean().default(false),
  cache_blocks: CacheBlockStatusSchema.default(emptyCacheBlockStatus()),
});
export type ContextMeta = {
  session_id: SessionId;
  turn_number: number;
  active_phase: TaskPhase;
  guides_loaded: GuideId[];
  skills_injected: GuideId[];
  compacted: boolean;
  cache_blocks: CacheBlockStatus;
};

/**
 * Assembled per-turn context. Distinct from the agent-side `Context` in
 * `../agent/types.js`, which is the narrower bundle the agent treats as
 * immutable input.
 */
export interface Context {
  system_prompt: RenderedSystemPrompt;
  messages: Message[];
  tool_schemas: ToolSchema[];
  token_count: number;
  window_limit: number;
  utilization: number;
  meta: ContextMeta;
}

/**
 * Conservative fallback compaction window when neither the caller's
 * {@link CompactionConfig.context_length} nor the model's
 * `provider().context_window` supplies a usable (`> 0`) value (issue #141).
 *
 * Deliberately small (8K, gemma-class) rather than the old 200K: when the
 * real context length is unknown, assume a tight window so compaction still
 * fires rather than silently never running.
 */
export const DEFAULT_CONTEXT_LENGTH = 8_000;

export const CompactionConfigSchema = z.object({
  threshold: z.number().default(0.8),
  preserve_recent_n: z.number().int().nonnegative().default(8),
  head_tail_tokens: z.number().int().nonnegative().default(512),
  offload_path: z.string().default(".spore/offload"),
  /**
   * Max times the harness re-requests a revised compaction summary when the
   * {@link CompactionVerifier} fails before accepting as-is and logging a
   * warn (issue #29). Mapped to/from `max_compaction_attempts` on the wire.
   */
  max_compaction_attempts: z.number().int().nonnegative().default(2),
  /**
   * Optional caller override for the resolved compaction window (issue #141).
   * When set to a value `> 0`, the resolver
   * ({@link "./standard.js".StandardContextManager.resolveContextLength})
   * uses it as the `window_limit`. `null`/absent (the default) and an explicit
   * `0` all fall through to the model's `provider().context_window`, then to
   * {@link DEFAULT_CONTEXT_LENGTH}. Configured values are NOT clamped to the
   * model's real window.
   *
   * Serialized as ABSENT when unset — `JSON.stringify` emits no
   * `context_length` key, so an existing serialized `CompactionConfig` stays
   * byte-identical (no new key when omitted).
   *
   * `nonnegative` (not `positive`): the cross-language domain is unsigned
   * (Rust `u32` / Go `uint32` / Python `int`), so an explicit `0` is a VALID
   * input that parses and then falls through in the resolver (honored only
   * when `> 0`). Rejecting `0` here would throw in TS while three sibling
   * languages accept it — a real interface divergence (#141).
   */
  context_length: z.number().int().nonnegative().nullable().optional(),
});
export type CompactionConfig = z.infer<typeof CompactionConfigSchema>;

export function defaultCompactionConfig(): CompactionConfig {
  // `context_length` is intentionally omitted so it serializes as ABSENT,
  // keeping existing serialized configs byte-identical (issue #141).
  return {
    threshold: 0.8,
    preserve_recent_n: 8,
    head_tail_tokens: 512,
    offload_path: ".spore/offload",
    max_compaction_attempts: 2,
  };
}

/**
 * Session state owned by the ContextManager — distinct from the harness's
 * opaque {@link "../harness/types.js".SessionState} which only carries the
 * loop's message log and arbitrary extras. The canonical assembly inputs
 * live here.
 */
export const SessionStateSchema = z.object({
  session_id: SessionIdSchema,
  task_id: TaskIdSchema,
  turn_number: z.number().int().nonnegative().default(0),
  task_instruction: z.string(),
  environment: z.string().default(""),
  prior_state: z.string().default(""),
  operational_instructions: z.string().default(""),
  active_phase: TaskPhaseSchema.default("execution"),
  message_history: z.array(MessageSchema).default([]),
  token_budget_used: z.number().int().nonnegative().default(0),
  // When the real context length is unknown, default to the conservative
  // DEFAULT_CONTEXT_LENGTH (8_000) rather than the dangerous old 200_000, so
  // compaction still fires for small-context models (issue #141). The manager's
  // `seedSession` overrides this with the resolved window.
  window_limit: z.number().int().nonnegative().default(DEFAULT_CONTEXT_LENGTH),
  guides_loaded: z.array(GuideIdSchema).default([]),
  /**
   * Skills pending Block-3 injection on the next assemble. Skills are
   * ephemeral; callers may clear after each assemble.
   */
  pending_skill_injections: z
    .array(
      z.object({
        id: GuideIdSchema,
        content: z.string(),
      }),
    )
    .default([]),
  budget_warning_active: z.boolean().default(false),
  /** Open problems feeding the `keep_open_problems` hint (issue #47). */
  open_problems: z.array(z.string()).default([]),
  /** Architectural decisions feeding `keep_architectural_decisions` (#47). */
  architectural_decisions: z.array(z.string()).default([]),
  /**
   * Recently touched file paths feeding `keep_recent_file_list` (#47).
   * Typed as strings, not path types — keeps tokenization byte-identical
   * across languages (no per-language path semantics).
   */
  recent_files: z.array(z.string()).default([]),
  /** Reasoning summary feeding the `keep_thinking_blocks` hint (issue #47). */
  reasoning_summary: z.string().default(""),
});
export type SessionState = {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  task_instruction: string;
  environment: string;
  prior_state: string;
  operational_instructions: string;
  active_phase: TaskPhase;
  message_history: Message[];
  token_budget_used: number;
  window_limit: number;
  guides_loaded: GuideId[];
  pending_skill_injections: Guide[];
  budget_warning_active: boolean;
  open_problems: string[];
  architectural_decisions: string[];
  recent_files: string[];
  reasoning_summary: string;
};

export function newSessionState(
  session_id: SessionId,
  task_id: TaskId,
  task_instruction: string,
): SessionState {
  return {
    session_id,
    task_id,
    turn_number: 0,
    task_instruction,
    environment: "",
    prior_state: "",
    operational_instructions: "",
    active_phase: "execution",
    message_history: [],
    token_budget_used: 0,
    window_limit: DEFAULT_CONTEXT_LENGTH,
    guides_loaded: [],
    pending_skill_injections: [],
    budget_warning_active: false,
    open_problems: [],
    architectural_decisions: [],
    recent_files: [],
    reasoning_summary: "",
  };
}

export interface ContextSources {
  guides: Guide[];
  memory: MemoryItem[];
  tool_schemas: ToolSchema[];
  composed_prompt: ComposedPrompt;
}

/**
 * An empty {@link ContextSources} (issue #115 / SC-26) — no guides, no memory,
 * no tool schemas, and an empty composed prompt. Renders to nothing through
 * {@link "./compaction-adapter.js".renderContextBlock}, so a harness that
 * supplies no structural sources stays byte-identical to the pre-#115
 * pass-through. The harness-loop `assemble` seam takes a `ContextSources`
 * argument; callers with nothing to inject pass this.
 */
export function emptyContextSources(): ContextSources {
  return {
    guides: [],
    memory: [],
    tool_schemas: [],
    composed_prompt: { rendered: "", block_1_hash: 0 },
  };
}

export interface CompactionPreserveHints {
  keep_architectural_decisions: boolean;
  keep_open_problems: boolean;
  keep_current_task_state: boolean;
  keep_recent_file_list: boolean;
  /** Defaults to `true` — never compact active reasoning blocks. */
  keep_thinking_blocks: boolean;
}

export function defaultCompactionPreserveHints(): CompactionPreserveHints {
  return {
    keep_architectural_decisions: true,
    keep_open_problems: true,
    keep_current_task_state: true,
    keep_recent_file_list: true,
    keep_thinking_blocks: true,
  };
}

export interface CompactionRequest {
  messages_to_compact: Message[];
  preserve_hints: CompactionPreserveHints;
}

export interface CompactionResult {
  summary_message: Message;
  tokens_reclaimed: number;
  messages_removed: number;
}

// ============================================================================
// Post-compaction verification (issue #29)
// ============================================================================

/** Outcome of a {@link CompactionVerifier.verify} check. */
export interface CompactionVerificationResult {
  passed: boolean;
  /**
   * Items from the preservation list not found in the summary, in
   * first-occurrence order (already lowercased/normalized).
   */
  missingItems: string[];
  detail: string;
}

/**
 * A lightweight, SYNCHRONOUS post-compaction sensor. Implementations run
 * after the agent produces a summary and before the harness accepts it.
 * They are purely computational and MUST NOT call the model — hence the
 * direct (non-`Promise`) return.
 */
export interface CompactionVerifier {
  verify(
    summary: string,
    hints: CompactionPreserveHints,
    sessionState: SessionState,
  ): CompactionVerificationResult;
}

/**
 * Standard {@link CompactionVerifier}: extracts key terms from the session
 * state per the enabled hints and checks they appear in the summary.
 *
 * All five hints contribute source terms, each gated on its hint and pushed
 * in a fixed order (issue #47) — this order is the cross-language invariant
 * that determines first-occurrence dedup:
 *
 * 1. `keep_current_task_state` → {@link SessionState.task_instruction}
 * 2. `keep_open_problems` → each of {@link SessionState.open_problems}
 * 3. `keep_architectural_decisions` → each of {@link SessionState.architectural_decisions}
 * 4. `keep_recent_file_list` → each of {@link SessionState.recent_files}
 * 5. `keep_thinking_blocks` → {@link SessionState.reasoning_summary}
 *
 * Each source string runs through the same `extractTerms` rule; an
 * empty/unset field contributes no terms.
 */
export class KeyTermVerifier implements CompactionVerifier {
  /**
   * Tokenize a source string into normalized key terms: lowercase, split on
   * runs of any char that is NOT `a-z` or `0-9`, drop empty tokens and tokens
   * shorter than 4 characters.
   */
  private static extractTerms(source: string): string[] {
    return source
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((tok) => tok.length >= 4);
  }

  verify(
    summary: string,
    hints: CompactionPreserveHints,
    sessionState: SessionState,
  ): CompactionVerificationResult {
    // Step 1: collect source strings from enabled hints, each gated on its
    // hint and pushed in this fixed order (issue #47). This order is the
    // cross-language invariant that determines first-occurrence dedup.
    const sources: string[] = [];
    if (hints.keep_current_task_state) {
      sources.push(sessionState.task_instruction);
    }
    if (hints.keep_open_problems) {
      sources.push(...sessionState.open_problems);
    }
    if (hints.keep_architectural_decisions) {
      sources.push(...sessionState.architectural_decisions);
    }
    if (hints.keep_recent_file_list) {
      sources.push(...sessionState.recent_files);
    }
    if (hints.keep_thinking_blocks) {
      sources.push(sessionState.reasoning_summary);
    }

    // Step 2: build the term list, deduped, first-occurrence order.
    const terms: string[] = [];
    for (const source of sources) {
      for (const term of KeyTermVerifier.extractTerms(source)) {
        if (!terms.includes(term)) {
          terms.push(term);
        }
      }
    }

    // Step 3: a term is present iff the lowercased summary contains it.
    const summaryLower = summary.toLowerCase();
    const missingItems = terms.filter((term) => !summaryLower.includes(term));

    // Steps 4 + 5.
    const total = terms.length;
    const passed = missingItems.length === 0;
    const detail = passed
      ? `all ${total} key term(s) present`
      : `missing ${missingItems.length} of ${total} key term(s): ${missingItems.join(", ")}`;

    return { passed, missingItems, detail };
  }
}

// ============================================================================
// Errors
// ============================================================================

export type ContextError =
  | { kind: "TokenCountFailed" }
  | { kind: "CompactionFailed"; reason: string }
  | { kind: "AssemblyFailed"; reason: string }
  /**
   * A cache block's content hash changed when it was expected to be stable.
   *
   * Both Block 1 ({@link CacheBlock} `"static"`) and Block 2 (`"per_session"`)
   * halt the run on a mid-session mismatch — they are treated consistently
   * (issue #32). A Block-2 change mid-session means session-stable content
   * mutated and every subsequent turn would silently pay full input-token
   * cost; rather than warn, the run stops so the caller can fix the source.
   *
   * `turn_number` is the turn on which the mismatch was detected (Block 2 only
   * halts when `turn_number > 1`; the turn-1 assemble records the baseline).
   * Estimated cache-cost-delta tracking (`UnexpectedMiss`) is a separate
   * observability concern tracked in issue #90.
   */
  | {
      kind: "CacheHashMismatch";
      block: CacheBlock;
      expected: number;
      actual: number;
      turn_number: number;
    };

export function contextErrorMessage(e: ContextError): string {
  switch (e.kind) {
    case "TokenCountFailed":
      return "token count failed";
    case "CompactionFailed":
      return `compaction failed: ${e.reason}`;
    case "AssemblyFailed":
      return `assembly failed: ${e.reason}`;
    case "CacheHashMismatch":
      return `cache hash mismatch on block ${e.block} at turn ${e.turn_number}: expected ${e.expected}, got ${e.actual}`;
  }
}

/** Marker class so domain errors can be `throw`n where appropriate. */
export class ContextErrorException extends Error {
  override readonly name = "ContextErrorException";
  constructor(readonly error: ContextError) {
    super(contextErrorMessage(error));
  }
}

// ============================================================================
// Tool result shape consumed by appendToolResult
// ============================================================================

/**
 * Same shape as `toolRegistry.ToolResult` / `harness.ToolResultRecord`,
 * re-declared here so this module does not depend on either.
 */
export interface ToolResult {
  call_id: string;
  output:
    | { kind: "success"; content: string; truncated?: boolean }
    | { kind: "error"; message: string; recoverable?: boolean }
    | { kind: "sandbox_violation"; violation: { kind: string; [k: string]: unknown } }
    | { kind: "waiting_for_human"; [k: string]: unknown }
    | { kind: "escalate"; [k: string]: unknown }
    | { kind: "awaiting_clarification"; [k: string]: unknown };
}

// ============================================================================
// ContextManager interface
// ============================================================================

/**
 * Canonical {@link ContextManager} interface (spore-core issue #7).
 *
 * Distinct from {@link "../harness/types.js".ContextManager}, which is the
 * narrower loop-only shape forward-declared by the harness. The standard
 * implementation lives in {@link "./standard.js".StandardContextManager}.
 */
export interface ContextManager {
  assemble(state: SessionState, sources: ContextSources): Promise<Context>;

  appendToolResult(
    state: SessionState,
    result: ToolResult,
    sandbox: SandboxProvider,
  ): Promise<void>;

  appendResponse(state: SessionState, response: string): void;

  shouldCompact(state: SessionState): boolean;

  prepareCompaction(state: SessionState): CompactionRequest;

  applyCompaction(state: SessionState, result: CompactionResult): void;

  injectSkill(context: Context, skill: Guide): void;

  recordCacheResult(context: Context, cacheStats: CacheBlockHits): void;
}

// Re-export the model-side ToolSchema for callers building ContextSources.
export type { ToolSchema };
