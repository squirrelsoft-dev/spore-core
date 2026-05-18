/**
 * PromptChunkRegistry — canonical types (spore-core issue #24).
 *
 * Mirrors `rust/crates/spore-core/src/prompt_chunk_registry.rs` byte-for-byte
 * on the wire: tagged unions use a `kind` discriminator in `snake_case`; enum
 * variants are `snake_case`. Same fixture, same outcome.
 *
 * Manages named, cacheable prompt chunks that compose the system prompt
 * deterministically at harness construction time. Each chunk is independently
 * cacheable; the resulting {@link ComposedPrompt} is stored on the harness and
 * its rendered text is reused for the life of the instance.
 *
 * ## Rules enforced
 *   - Unique `ChunkId` — duplicate registration is `duplicate_id`.
 *   - {@link ChunkSlot} `budget`/`ephemeral` must be {@link CacheBlock}
 *     `per_turn`; `role`/`mode` must be `static`. Otherwise
 *     `conflicting_cache_block`.
 *   - Chunk `content` must not be empty (after trim).
 *   - `compose()` requires the named role chunk to be registered and resolves
 *     the mode chunk via {@link Mode.promptChunk}. Capability/skill chunks
 *     must already be registered.
 *   - The composed chunk list is ordered by slot (Role, Mode, Capability,
 *     Skill, Task, Environment, PriorSession, Budget, Ephemeral) and, within a
 *     slot, by caller-provided / registration order.
 *   - `validate()` flags `per_turn` chunks in Block 1, missing Role/Mode
 *     slots, and more than one Mode chunk.
 */

import { TaskPhaseSchema, type TaskPhase } from "../tool-registry/types.js";

// Reference for tree-shaken import retention.
void TaskPhaseSchema;

// ============================================================================
// Identity
// ============================================================================

export class ChunkId {
  constructor(readonly value: string) {}
  static of(value: string): ChunkId {
    return new ChunkId(value);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: ChunkId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

// ============================================================================
// Enums
// ============================================================================

export type ChunkSlot =
  | "role"
  | "mode"
  | "capability"
  | "skill"
  | "task"
  | "environment"
  | "prior_session"
  | "budget"
  | "ephemeral";

export const CHUNK_SLOTS: readonly ChunkSlot[] = [
  "role",
  "mode",
  "capability",
  "skill",
  "task",
  "environment",
  "prior_session",
  "budget",
  "ephemeral",
];

/** Render ordering used by {@link PromptChunkRegistry.compose}. */
export function chunkSlotOrder(slot: ChunkSlot): number {
  return CHUNK_SLOTS.indexOf(slot);
}

export type CacheBlock = "static" | "per_session" | "per_turn";

export type ApprovalPolicy =
  /** Every action requires approval before execution. */
  | "always_ask"
  /** Actions proceed automatically; the agent narrates afterwards. */
  | "auto_explain"
  /** Planning only — file edits are blocked until the user confirms. */
  | "plan_only"
  /** Auto for Low/Medium; High/Critical require approval (middleware). */
  | "safe_auto"
  /** Full autonomy — no approval gates. */
  | "none";

export type Mode = "always_ask" | "auto_edit" | "plan" | "safe_auto" | "yolo";

// ============================================================================
// Mode helpers — analogue of Rust `impl Mode`.
// ============================================================================

/** The standard prompt chunk for a mode. Always in slot `mode`, cache `static`. */
export function modePromptChunk(mode: Mode): PromptChunk {
  const spec = modeChunkSpec(mode);
  return promptChunk(spec.id, spec.content, "mode", "static");
}

function modeChunkSpec(mode: Mode): { id: string; content: string } {
  switch (mode) {
    case "always_ask":
      return {
        id: "mode-always-ask",
        content:
          "Mode: AlwaysAsk. Describe your plan and wait for explicit approval before taking any action.",
      };
    case "auto_edit":
      return {
        id: "mode-auto-edit",
        content:
          "Mode: AutoEdit. Edit files freely. Explain the changes you make after they are done.",
      };
    case "plan":
      return {
        id: "mode-plan",
        content: "Mode: Plan. Produce a plan only. Do not edit files or execute mutating tools.",
      };
    case "safe_auto":
      return {
        id: "mode-safe-auto",
        content:
          "Mode: SafeAuto. Auto-execute Low and Medium risk actions. High and Critical actions require approval.",
      };
    case "yolo":
      return { id: "mode-yolo", content: "Mode: Yolo. Full autonomy. No approval gates." };
  }
}

/** Enforcement policy implied by a mode. */
export function modeApprovalPolicy(mode: Mode): ApprovalPolicy {
  switch (mode) {
    case "always_ask":
      return "always_ask";
    case "auto_edit":
      return "auto_explain";
    case "plan":
      return "plan_only";
    case "safe_auto":
      return "safe_auto";
    case "yolo":
      return "none";
  }
}

/** Initial task phase implied by a mode. */
export function modeDefaultToolPhase(mode: Mode): TaskPhase {
  return mode === "plan" ? "planning" : "execution";
}

// ============================================================================
// Records
// ============================================================================

export interface PromptChunk {
  id: ChunkId;
  content: string;
  slot: ChunkSlot;
  cache_block: CacheBlock;
  /** Stable content digest. Recomputed via {@link promptChunk}. */
  hash: number;
}

/**
 * Build a {@link PromptChunk}, computing the content hash so it stays in sync
 * with `content`. Prefer this over a struct literal.
 */
export function promptChunk(
  id: string | ChunkId,
  content: string,
  slot: ChunkSlot,
  cache_block: CacheBlock,
): PromptChunk {
  return {
    id: typeof id === "string" ? new ChunkId(id) : id,
    content,
    slot,
    cache_block,
    hash: hashContent(content),
  };
}

export interface ComposedPrompt {
  chunks: PromptChunk[];
  block_1_hash: number;
  block_2_hash: number;
  /**
   * Cached render — `null` until materialized by `ContextManager.assemble()`.
   * Invalidated when any chunk hash changes.
   */
  rendered: string | null;
}

/** Render a {@link ComposedPrompt} deterministically, caching the result. */
export function renderComposed(composed: ComposedPrompt): string {
  if (composed.rendered === null) {
    composed.rendered = composed.chunks.map((c) => c.content).join("\n\n");
  }
  return composed.rendered;
}

/** Cached render or empty string when not yet rendered. Read-only friendly. */
export function renderedStr(composed: ComposedPrompt): string {
  return composed.rendered ?? "";
}

/** Recompute (block_1_hash, block_2_hash) from the current chunk state. */
export function recomputeBlockHashes(composed: ComposedPrompt): [number, number] {
  return computeBlockHashes(composed.chunks);
}

// ============================================================================
// Errors
// ============================================================================

export type ChunkError =
  | { kind: "duplicate_id"; id: ChunkId }
  | { kind: "invalid_slot"; id: ChunkId; reason: string }
  | {
      kind: "conflicting_cache_block";
      id: ChunkId;
      slot: ChunkSlot;
      expected: CacheBlock;
      actual: CacheBlock;
    }
  | { kind: "not_found"; id: ChunkId };

export function chunkErrorMessage(e: ChunkError): string {
  switch (e.kind) {
    case "duplicate_id":
      return `duplicate chunk id: ${e.id.value}`;
    case "invalid_slot":
      return `invalid slot for chunk ${e.id.value}: ${e.reason}`;
    case "conflicting_cache_block":
      return `conflicting cache block for chunk ${e.id.value} in slot ${e.slot}: expected ${e.expected}, got ${e.actual}`;
    case "not_found":
      return `chunk not found: ${e.id.value}`;
  }
}

export class ChunkErrorException extends Error {
  override readonly name = "ChunkErrorException";
  readonly kind: ChunkError["kind"];
  constructor(readonly error: ChunkError) {
    super(chunkErrorMessage(error));
    this.kind = error.kind;
  }
}

export type ChunkValidationError =
  | { kind: "per_turn_chunk_in_static_block"; id: ChunkId }
  | { kind: "missing_required_slot"; slot: ChunkSlot }
  | { kind: "conflicting_mode_chunks"; ids: ChunkId[] };

export function chunkValidationErrorMessage(e: ChunkValidationError): string {
  switch (e.kind) {
    case "per_turn_chunk_in_static_block":
      return `per-turn chunk ${e.id.value} placed in the Static block`;
    case "missing_required_slot":
      return `required slot ${e.slot} is missing from the composition`;
    case "conflicting_mode_chunks":
      return `more than one Mode chunk in the composition: ${e.ids.map((i) => i.value).join(", ")}`;
  }
}

export class ChunkValidationErrorException extends Error {
  override readonly name = "ChunkValidationErrorException";
  constructor(readonly errors: ChunkValidationError[]) {
    super(errors.map(chunkValidationErrorMessage).join("; "));
  }
}

// ============================================================================
// Trait
// ============================================================================

/**
 * Canonical {@link PromptChunkRegistry} interface. The concrete reference
 * implementation is {@link StandardPromptChunkRegistry}.
 */
export interface PromptChunkRegistry {
  /** Register a chunk — validates slot/cache_block compatibility. */
  register(chunk: PromptChunk): ChunkError | null;

  /**
   * Compose chunks for a given agent configuration. Called once at harness
   * construction; the result is cached on the harness.
   */
  compose(
    role: ChunkId,
    mode: Mode,
    capabilities: ChunkId[],
    skills: ChunkId[],
  ): { ok: true; composed: ComposedPrompt } | { ok: false; errors: ChunkValidationError[] };

  /** Validate a composition. Returns an empty array when valid. */
  validate(composed: ComposedPrompt): ChunkValidationError[];

  /** Look up a chunk by id. Returns `undefined` if absent. */
  get(id: ChunkId): PromptChunk | undefined;
}

// ============================================================================
// Hashing — FNV-1a 64-bit (deterministic; matches no other language byte-for-
// byte, but cross-language fixture parity does not require equal hashes; the
// shared fixture asserts (slot, id) sequence and rendered text, per the Rust
// suite's own contract).
// ============================================================================

// We work in BigInt-space for stability, then narrow to a Number for the
// public field. The wire hash is documented as a `u64` analogue in spec; we
// expose it as a JS number for ergonomics — collisions in the upper 11 bits
// of a 64-bit space are not a concern for cache-invalidation use.

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK_64 = (1n << 64n) - 1n;

function fnv1a64(bytes: Uint8Array): bigint {
  let h = FNV_OFFSET;
  for (let i = 0; i < bytes.length; i++) {
    h ^= BigInt(bytes[i] ?? 0);
    h = (h * FNV_PRIME) & MASK_64;
  }
  return h;
}

const TEXT_ENCODER = new TextEncoder();

export function hashContent(content: string): number {
  // Project a 64-bit FNV-1a digest into a JS-safe integer space. We fold the
  // high 32 bits into the low 32 with XOR — preserves all input bits while
  // remaining a Number. Cache invalidation cares only about content stability,
  // not about hash collisions across languages.
  const h = fnv1a64(TEXT_ENCODER.encode(content));
  const lo = Number(h & 0xffffffffn);
  const hi = Number((h >> 32n) & 0xffffffffn);
  return (lo ^ hi) >>> 0;
}

export function computeBlockHashes(chunks: readonly PromptChunk[]): [number, number] {
  // Group by cache block, then sort by id for deterministic digest input.
  const block1: Array<[string, number]> = [];
  const block2: Array<[string, number]> = [];
  for (const c of chunks) {
    if (c.cache_block === "static") block1.push([c.id.value, c.hash]);
    else if (c.cache_block === "per_session") block2.push([c.id.value, c.hash]);
  }
  block1.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  block2.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return [digestPairs(block1), digestPairs(block2)];
}

function digestPairs(pairs: Array<[string, number]>): number {
  // Concatenate id + ":" + hash + ";" and hash the result.
  const parts: string[] = [];
  for (const [id, h] of pairs) parts.push(`${id}:${h};`);
  return hashContent(parts.join(""));
}
