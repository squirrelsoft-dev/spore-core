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
 * Cache hit/miss stats parsed by a {@link CacheProvider}. `null` means the
 * provider has no signal for that block (e.g. the provider does not support
 * caching at all).
 */
export interface CacheStats {
  static_hit: boolean | null;
  session_hit: boolean | null;
  history_hit: boolean | null;
}

export function emptyCacheStats(): CacheStats {
  return { static_hit: null, session_hit: null, history_hit: null };
}

/**
 * Forward-declared `CacheProvider`. The default {@link NullCacheProvider}
 * is the testing default — it never interferes.
 */
export interface CacheProvider {
  supportsCaching(): boolean;
  /** No-op when `supportsCaching()` is false. */
  annotate(context: Context): void;
}

/** Testing default — no-op for all calls. */
export class NullCacheProvider implements CacheProvider {
  supportsCaching(): boolean {
    return false;
  }
  annotate(_context: Context): void {
    void _context;
  }
}

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

export const CompactionConfigSchema = z.object({
  threshold: z.number().default(0.8),
  preserve_recent_n: z.number().int().nonnegative().default(8),
  head_tail_tokens: z.number().int().nonnegative().default(512),
  offload_path: z.string().default(".spore/offload"),
});
export type CompactionConfig = z.infer<typeof CompactionConfigSchema>;

export function defaultCompactionConfig(): CompactionConfig {
  return {
    threshold: 0.8,
    preserve_recent_n: 8,
    head_tail_tokens: 512,
    offload_path: ".spore/offload",
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
  window_limit: z.number().int().nonnegative().default(200_000),
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
    window_limit: 200_000,
    guides_loaded: [],
    pending_skill_injections: [],
    budget_warning_active: false,
  };
}

export interface ContextSources {
  guides: Guide[];
  memory: MemoryItem[];
  tool_schemas: ToolSchema[];
  composed_prompt: ComposedPrompt;
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
// Errors
// ============================================================================

export type ContextError =
  | { kind: "TokenCountFailed" }
  | { kind: "CompactionFailed"; reason: string }
  | { kind: "AssemblyFailed"; reason: string }
  | { kind: "CacheHashMismatch"; block: string; expected: number; actual: number };

export function contextErrorMessage(e: ContextError): string {
  switch (e.kind) {
    case "TokenCountFailed":
      return "token count failed";
    case "CompactionFailed":
      return `compaction failed: ${e.reason}`;
    case "AssemblyFailed":
      return `assembly failed: ${e.reason}`;
    case "CacheHashMismatch":
      return `cache hash mismatch on block ${e.block}: expected ${e.expected}, got ${e.actual}`;
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
    | { kind: "waiting_for_human"; [k: string]: unknown };
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

  recordCacheResult(context: Context, cacheStats: CacheStats): void;
}

// Re-export the model-side ToolSchema for callers building ContextSources.
export type { ToolSchema };
