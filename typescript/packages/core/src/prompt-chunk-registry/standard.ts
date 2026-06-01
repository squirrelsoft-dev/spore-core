/**
 * StandardPromptChunkRegistry — in-memory reference implementation of
 * {@link PromptChunkRegistry} (spore-core issue #24).
 *
 * Mirrors `rust/crates/spore-core/src/prompt_chunk_registry.rs#StandardPromptChunkRegistry`
 * — same rules, same fixture outcomes. The harness owns a single shared
 * instance; chunks are typically registered at startup and never mutated.
 */

import {
  type AnyMode,
  type CacheBlock,
  ChunkId,
  type ChunkError,
  type ChunkSlot,
  type ChunkValidationError,
  type ComposedPrompt,
  type Mode,
  type PromptChunk,
  type PromptChunkRegistry,
  anyModePromptChunk,
  chunkSlotOrder,
  computeBlockHashes,
  promptChunk,
} from "./types.js";

// ============================================================================
// StandardPromptChunkRegistry
// ============================================================================

export class StandardPromptChunkRegistry implements PromptChunkRegistry {
  private readonly chunks = new Map<string, PromptChunk>();
  /** Registration order for ids — drives intra-slot ordering during compose(). */
  private readonly order: string[] = [];

  register(chunk: PromptChunk): ChunkError | null {
    const slotErr = validateSlotAndCacheBlock(chunk);
    if (slotErr) return slotErr;
    if (this.chunks.has(chunk.id.value)) {
      return { kind: "duplicate_id", id: new ChunkId(chunk.id.value) };
    }
    this.chunks.set(chunk.id.value, cloneChunk(chunk));
    this.order.push(chunk.id.value);
    return null;
  }

  /**
   * Register every chunk in the {@link standardChunks} library. Returns the
   * first registration error if any chunk fails validation (e.g. a colliding
   * id is already present).
   */
  registerStandardChunks(): ChunkError | null {
    for (const c of standardChunks()) {
      const err = this.register(c);
      if (err) return err;
    }
    return null;
  }

  compose(
    role: ChunkId,
    mode: Mode,
    capabilities: ChunkId[],
    skills: ChunkId[],
  ): { ok: true; composed: ComposedPrompt } | { ok: false; errors: ChunkValidationError[] } {
    return this.composeAny(role, mode, capabilities, skills);
  }

  /**
   * Internal compose that also accepts the dangerous `"yolo"` mode. The public
   * {@link compose} narrows `mode` to {@link Mode}, so a default-build caller
   * cannot compose a Yolo prompt. The dangerous entry point
   * (`@spore/core/dangerous`) exposes a wrapper that reaches this method with
   * `"yolo"` (issue #34).
   */
  composeAny(
    role: ChunkId,
    mode: AnyMode,
    capabilities: ChunkId[],
    skills: ChunkId[],
  ): { ok: true; composed: ComposedPrompt } | { ok: false; errors: ChunkValidationError[] } {
    const errors: ChunkValidationError[] = [];
    const chosen: PromptChunk[] = [];

    // Role
    const roleChunk = this.chunks.get(role.value);
    if (roleChunk && roleChunk.slot === "role") {
      chosen.push(cloneChunk(roleChunk));
    } else {
      errors.push({ kind: "missing_required_slot", slot: "role" });
    }

    // Mode — always sourced from the enum, never from the registry.
    chosen.push(anyModePromptChunk(mode));

    // Capabilities
    for (const id of capabilities) {
      const c = this.chunks.get(id.value);
      if (c && c.slot === "capability") {
        chosen.push(cloneChunk(c));
      } else {
        errors.push({ kind: "missing_required_slot", slot: "capability" });
      }
    }

    // Skills
    for (const id of skills) {
      const c = this.chunks.get(id.value);
      if (c && c.slot === "skill") {
        chosen.push(cloneChunk(c));
      } else {
        errors.push({ kind: "missing_required_slot", slot: "skill" });
      }
    }

    if (errors.length > 0) return { ok: false, errors };

    // Stable sort by slot order; within a slot preserve insertion order
    // (capabilities/skills follow the caller-provided sequence).
    stableSortBySlot(chosen);

    const [b1, b2] = computeBlockHashes(chosen);
    const composed: ComposedPrompt = {
      chunks: chosen,
      block_1_hash: b1,
      block_2_hash: b2,
      rendered: null,
    };

    const vErrors = this.validate(composed);
    if (vErrors.length > 0) return { ok: false, errors: vErrors };

    return { ok: true, composed };
  }

  validate(composed: ComposedPrompt): ChunkValidationError[] {
    const errors: ChunkValidationError[] = [];

    // Block 1 must not contain Budget/Ephemeral chunks (those are PerTurn).
    for (const c of composed.chunks) {
      if (c.cache_block === "static" && (c.slot === "budget" || c.slot === "ephemeral")) {
        errors.push({
          kind: "per_turn_chunk_in_static_block",
          id: new ChunkId(c.id.value),
        });
      }
    }

    // Required slots: role and mode.
    for (const required of ["role", "mode"] as const) {
      if (!composed.chunks.some((c) => c.slot === required)) {
        errors.push({ kind: "missing_required_slot", slot: required });
      }
    }

    // Exactly one Mode chunk.
    const modeIds = composed.chunks
      .filter((c) => c.slot === "mode")
      .map((c) => new ChunkId(c.id.value));
    if (modeIds.length > 1) {
      errors.push({ kind: "conflicting_mode_chunks", ids: modeIds });
    }

    return errors;
  }

  get(id: ChunkId): PromptChunk | undefined {
    const c = this.chunks.get(id.value);
    return c ? cloneChunk(c) : undefined;
  }
}

// ============================================================================
// Helpers
// ============================================================================

function cloneChunk(c: PromptChunk): PromptChunk {
  return {
    id: new ChunkId(c.id.value),
    content: c.content,
    slot: c.slot,
    cache_block: c.cache_block,
    hash: c.hash,
  };
}

function validateSlotAndCacheBlock(chunk: PromptChunk): ChunkError | null {
  if (chunk.content.trim().length === 0) {
    return {
      kind: "invalid_slot",
      id: new ChunkId(chunk.id.value),
      reason: "content must not be empty",
    };
  }
  const required = requiredCacheBlock(chunk.slot);
  if (required && chunk.cache_block !== required) {
    return {
      kind: "conflicting_cache_block",
      id: new ChunkId(chunk.id.value),
      slot: chunk.slot,
      expected: required,
      actual: chunk.cache_block,
    };
  }
  return null;
}

function requiredCacheBlock(slot: ChunkSlot): CacheBlock | null {
  switch (slot) {
    case "budget":
    case "ephemeral":
      return "per_turn";
    case "role":
    case "mode":
      return "static";
    default:
      return null;
  }
}

function stableSortBySlot(chunks: PromptChunk[]): void {
  // Decorate–sort–undecorate to preserve insertion order within a slot.
  const decorated = chunks.map((c, i) => ({ c, i }));
  decorated.sort((a, b) => {
    const so = chunkSlotOrder(a.c.slot) - chunkSlotOrder(b.c.slot);
    return so !== 0 ? so : a.i - b.i;
  });
  for (let i = 0; i < decorated.length; i++) {
    chunks[i] = decorated[i]!.c;
  }
}

// ============================================================================
// Standard chunk library
// ============================================================================

/**
 * The standard chunk library shipped by `@spore/core`. Users register
 * additional chunks for their domain. Matches the Rust reference exactly.
 */
export function standardChunks(): PromptChunk[] {
  const out: PromptChunk[] = [];

  // Roles
  const roles: Array<[string, string]> = [
    [
      "role-coding-agent",
      "You are an expert software engineer. Read carefully, change deliberately, and verify your work.",
    ],
    [
      "role-evaluator",
      "You are a fresh evaluator. You did not write the code under review. Default to FAIL.",
    ],
    [
      "role-planner",
      "You are a planning specialist. Decompose tasks into small, verifiable steps.",
    ],
    [
      "role-rag-assistant",
      "You are a retrieval-augmented assistant. Always cite the source document for any claim.",
    ],
    [
      "role-sql-agent",
      "You are a SQL specialist. Prefer read-only queries; never DROP without explicit approval.",
    ],
  ];
  for (const [id, content] of roles) out.push(promptChunk(id, content, "role", "static"));

  // Modes — derived from the enum so promptChunk() and standardChunks() agree.
  // `"yolo"` is intentionally absent here: it is a dangerous-only mode (issue
  // #34) and is added by `dangerousStandardChunks()` in `@spore/core/dangerous`.
  const modes: Mode[] = ["always_ask", "auto_edit", "plan", "safe_auto"];
  for (const m of modes) out.push(anyModePromptChunk(m));

  // Capabilities
  const caps: Array<[string, string]> = [
    ["capability-bash", "Capability: bash. You can run shell commands inside the sandbox."],
    [
      "capability-filesystem",
      "Capability: filesystem. You can read and write files inside the workspace.",
    ],
    ["capability-git", "Capability: git. You can stage, commit, and inspect history."],
    ["capability-browser", "Capability: browser. You can fetch web pages and follow links."],
    ["capability-subagent", "Capability: subagent. You can delegate work to a child harness."],
    ["capability-sql", "Capability: sql. You can issue queries against the configured database."],
  ];
  for (const [id, content] of caps) out.push(promptChunk(id, content, "capability", "static"));

  // Skills
  const skills: Array<[string, string]> = [
    ["skill-testing", "Skill: always run the test suite after changes and report results."],
    ["skill-decomposition", "Skill: break large tasks into small, independently verifiable steps."],
    [
      "skill-security-review",
      "Skill: review changes for injection, auth, and secret-leak issues before commit.",
    ],
    [
      "skill-citation",
      "Skill: cite the source document for every claim drawn from retrieved context.",
    ],
  ];
  for (const [id, content] of skills) out.push(promptChunk(id, content, "skill", "static"));

  return out;
}
