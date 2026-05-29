/**
 * Issue #79 — Prompt assembly engine: conditional, provider-sourced prompt
 * assembly that EXTENDS (does not replace) the #24
 * {@link "../prompt-chunk-registry/types.js".PromptChunkRegistry}.
 *
 * The shipped #24 `prompt-chunk-registry` module composes a static Block-1
 * {@link "../prompt-chunk-registry/types.js".ComposedPrompt} once at
 * construction. This module builds *on top of* it: chunks are loaded from
 * pluggable {@link ChunkProvider}s and included conditionally based on mode,
 * active tools, phase, agent type, trigger words, hook events, or arbitrary
 * architect-defined predicates. The Static bucket is folded into a #24
 * `ComposedPrompt` (Block 1); PerSession / PerTurn chunks flow through the
 * existing `PromptSegment` machinery in {@link "../context/types.js"} (decision
 * A4 — no new public segment vectors on `ContextSources`).
 *
 * This module owns its OWN {@link PromptChunk} and {@link ChunkProviderError},
 * distinct from the #24 `prompt-chunk-registry` types, which are left untouched
 * (decision A1). It is also the home of the minimal shared {@link StorageScope}
 * enum (decision A2).
 *
 * The wire form is byte-identical with the Rust/Python/Go reference and the
 * shared fixtures in `fixtures/prompt_assembly/`: {@link ChunkCondition}
 * serializes to an internally-tagged object keyed on `type` in `snake_case`;
 * `active_capabilities` is an array of `[tool, capability]` pairs.
 *
 * ## A3 — `Custom` is first-class but unserialized
 *
 * {@link ChunkCondition} with `kind: "custom"` wraps a predicate
 * `(ctx: AssemblyContext) => boolean`. It is the PRIMARY escape hatch for
 * conditions that cannot be expressed with the serializable variants, and it is
 * fully supported in the public API. However it CANNOT serialize:
 *   - {@link serializeCondition} omits a `custom` node (it is pruned out of the
 *     `all`/`any`/`not` combinators and yields `null` at the top level);
 *   - {@link deserializeCondition} can therefore never produce a `custom`;
 *   - a `null`/absent condition deserializes to the `always` default;
 *   - {@link conditionsEqual} treats `custom` as NEVER equal to anything
 *     (including another `custom`), since closure identity is not comparable.
 *   - `custom` is excluded from the shared byte-identical fixtures.
 */

import {
  SegmentStabilitySchema,
  type Guide,
  type MemoryItem,
  type PromptSegment,
  type SegmentStability,
  type ContextSources,
  type ComposedPrompt as ContextComposedPrompt,
} from "../context/types.js";
import { SessionId, TaskId } from "../harness/types.js";
import { ALL_HOOK_EVENTS, type HookEvent } from "../hooks/types.js";
import type { ToolSchema } from "../model/index.js";
import {
  CacheBlock,
  ChunkSlot,
  promptChunk as registryPromptChunk,
  computeBlockHashes,
  renderComposed,
  type ComposedPrompt as RegistryComposedPrompt,
  type Mode,
} from "../prompt-chunk-registry/types.js";
import { TaskPhaseSchema, type TaskPhase } from "../tool-registry/types.js";

// ============================================================================
// StorageScope (A2)
// ============================================================================

/**
 * Minimal shared storage scope. This module is its home (decision A2); the
 * scope-aware `FileSystemChunkProvider` that consumes it is deferred (A6).
 * Wire form is the `snake_case` variant string.
 */
export type StorageScope = "user" | "project" | "local";

export const STORAGE_SCOPES: readonly StorageScope[] = ["user", "project", "local"];

export const DEFAULT_STORAGE_SCOPE: StorageScope = "project";

// ============================================================================
// ToolAffinity
// ============================================================================

/**
 * Binds a chunk to a tool (and optionally a sub-capability). The builder
 * includes the chunk only when the tool — and capability, if any — is active.
 */
export interface ToolAffinity {
  tool_name: string;
  capability?: string | undefined;
}

export function toolAffinity(toolName: string): ToolAffinity {
  return { tool_name: toolName };
}

export function toolAffinityWithCapability(toolName: string, capability: string): ToolAffinity {
  return { tool_name: toolName, capability };
}

// ============================================================================
// ChunkCondition
// ============================================================================

/** Closure type behind a `custom` {@link ChunkCondition} (A3). */
export type CustomCondition = (ctx: AssemblyContext) => boolean;

/**
 * The condition primitive tree. Architects compose these; the framework
 * evaluates them against an {@link AssemblyContext} via
 * {@link ContextSourcesBuilder.evaluate}.
 *
 * All variants serialize EXCEPT `custom` — see the module-level A3 note.
 */
export type ChunkCondition =
  | { kind: "always" }
  | { kind: "when_mode"; mode: Mode }
  | { kind: "when_tool_active"; tool: string }
  | { kind: "when_tool_capability"; tool: string; capability: string }
  | { kind: "when_phase"; phase: TaskPhase }
  | { kind: "when_agent_type"; agent_type: string }
  | { kind: "when_feature"; feature: string }
  | { kind: "on_trigger"; words: string[] }
  | { kind: "on_event"; event: HookEvent }
  | { kind: "all"; conditions: ChunkCondition[] }
  | { kind: "any"; conditions: ChunkCondition[] }
  | { kind: "not"; condition: ChunkCondition }
  | { kind: "custom"; predicate: CustomCondition };

// ── Constructors (ergonomic, idiomatic) ─────────────────────────────────────

export const ChunkConditions = {
  always(): ChunkCondition {
    return { kind: "always" };
  },
  whenMode(mode: Mode): ChunkCondition {
    return { kind: "when_mode", mode };
  },
  whenToolActive(tool: string): ChunkCondition {
    return { kind: "when_tool_active", tool };
  },
  whenToolCapability(tool: string, capability: string): ChunkCondition {
    return { kind: "when_tool_capability", tool, capability };
  },
  whenPhase(phase: TaskPhase): ChunkCondition {
    return { kind: "when_phase", phase };
  },
  whenAgentType(agentType: string): ChunkCondition {
    return { kind: "when_agent_type", agent_type: agentType };
  },
  whenFeature(feature: string): ChunkCondition {
    return { kind: "when_feature", feature };
  },
  onTrigger(words: string[]): ChunkCondition {
    return { kind: "on_trigger", words };
  },
  onEvent(event: HookEvent): ChunkCondition {
    return { kind: "on_event", event };
  },
  all(conditions: ChunkCondition[]): ChunkCondition {
    return { kind: "all", conditions };
  },
  any(conditions: ChunkCondition[]): ChunkCondition {
    return { kind: "any", conditions };
  },
  not(condition: ChunkCondition): ChunkCondition {
    return { kind: "not", condition };
  },
  /** The A3 escape hatch. Constructible only programmatically; never serialized. */
  custom(predicate: CustomCondition): ChunkCondition {
    return { kind: "custom", predicate };
  },
};

/**
 * Structural equality for {@link ChunkCondition}. A `custom` node is NEVER equal
 * to anything (including another `custom`) — closure identity is not comparable
 * (A3). Mirrors the Rust manual `PartialEq`.
 */
export function conditionsEqual(a: ChunkCondition, b: ChunkCondition): boolean {
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case "always":
      return true;
    case "when_mode":
      return b.kind === "when_mode" && a.mode === b.mode;
    case "when_tool_active":
      return b.kind === "when_tool_active" && a.tool === b.tool;
    case "when_tool_capability":
      return (
        b.kind === "when_tool_capability" && a.tool === b.tool && a.capability === b.capability
      );
    case "when_phase":
      return b.kind === "when_phase" && a.phase === b.phase;
    case "when_agent_type":
      return b.kind === "when_agent_type" && a.agent_type === b.agent_type;
    case "when_feature":
      return b.kind === "when_feature" && a.feature === b.feature;
    case "on_trigger":
      return (
        b.kind === "on_trigger" &&
        a.words.length === b.words.length &&
        a.words.every((w, i) => w === b.words[i])
      );
    case "on_event":
      return b.kind === "on_event" && a.event === b.event;
    case "all":
      return (
        b.kind === "all" &&
        a.conditions.length === b.conditions.length &&
        a.conditions.every((c, i) => conditionsEqual(c, b.conditions[i] as ChunkCondition))
      );
    case "any":
      return (
        b.kind === "any" &&
        a.conditions.length === b.conditions.length &&
        a.conditions.every((c, i) => conditionsEqual(c, b.conditions[i] as ChunkCondition))
      );
    case "not":
      return b.kind === "not" && conditionsEqual(a.condition, b.condition);
    case "custom":
      // Never equal (A3).
      return false;
  }
}

// ── Serialization (A3): all variants except custom ──────────────────────────
//
// Wire form is an internally-tagged object keyed on `type` (snake_case). A
// `custom` node is pruned: at the top level it serializes to `null`; inside an
// `all`/`any` it is dropped; a `not` over `custom` collapses to `null`.

/** The serialized wire form of a {@link ChunkCondition} (everything but custom). */
export type SerializedCondition =
  | { type: "always" }
  | { type: "when_mode"; mode: Mode }
  | { type: "when_tool_active"; tool: string }
  | { type: "when_tool_capability"; tool: string; capability: string }
  | { type: "when_phase"; phase: TaskPhase }
  | { type: "when_agent_type"; agent_type: string }
  | { type: "when_feature"; feature: string }
  | { type: "on_trigger"; words: string[] }
  | { type: "on_event"; event: HookEvent }
  | { type: "all"; conditions: SerializedCondition[] }
  | { type: "any"; conditions: SerializedCondition[] }
  | { type: "not"; condition: SerializedCondition };

/**
 * Convert a {@link ChunkCondition} into its wire form, or `null` for a node
 * that cannot be represented (a top-level `custom`, or a `not` whose inner
 * condition prunes away). `custom` children of `all`/`any` are dropped.
 */
export function serializeCondition(c: ChunkCondition): SerializedCondition | null {
  switch (c.kind) {
    case "always":
      return { type: "always" };
    case "when_mode":
      return { type: "when_mode", mode: c.mode };
    case "when_tool_active":
      return { type: "when_tool_active", tool: c.tool };
    case "when_tool_capability":
      return { type: "when_tool_capability", tool: c.tool, capability: c.capability };
    case "when_phase":
      return { type: "when_phase", phase: c.phase };
    case "when_agent_type":
      return { type: "when_agent_type", agent_type: c.agent_type };
    case "when_feature":
      return { type: "when_feature", feature: c.feature };
    case "on_trigger":
      return { type: "on_trigger", words: [...c.words] };
    case "on_event":
      return { type: "on_event", event: c.event };
    case "all": {
      const conditions = c.conditions
        .map(serializeCondition)
        .filter((x): x is SerializedCondition => x !== null);
      return { type: "all", conditions };
    }
    case "any": {
      const conditions = c.conditions
        .map(serializeCondition)
        .filter((x): x is SerializedCondition => x !== null);
      return { type: "any", conditions };
    }
    case "not": {
      const inner = serializeCondition(c.condition);
      return inner === null ? null : { type: "not", condition: inner };
    }
    case "custom":
      return null;
  }
}

/**
 * Reconstruct a {@link ChunkCondition} from its wire form. A `null`/`undefined`
 * node has no representable condition and yields the `always` default (A3).
 * Throws on a malformed (non-null) node.
 */
export function deserializeCondition(value: unknown): ChunkCondition {
  if (value === null || value === undefined) return { kind: "always" };
  if (typeof value !== "object") {
    throw new Error(`invalid ChunkCondition: expected object, got ${typeof value}`);
  }
  const node = value as Record<string, unknown>;
  const type = node["type"];
  switch (type) {
    case "always":
      return { kind: "always" };
    case "when_mode":
      return { kind: "when_mode", mode: node["mode"] as Mode };
    case "when_tool_active":
      return { kind: "when_tool_active", tool: String(node["tool"]) };
    case "when_tool_capability":
      return {
        kind: "when_tool_capability",
        tool: String(node["tool"]),
        capability: String(node["capability"]),
      };
    case "when_phase":
      return { kind: "when_phase", phase: TaskPhaseSchema.parse(node["phase"]) };
    case "when_agent_type":
      return { kind: "when_agent_type", agent_type: String(node["agent_type"]) };
    case "when_feature":
      return { kind: "when_feature", feature: String(node["feature"]) };
    case "on_trigger":
      return { kind: "on_trigger", words: (node["words"] as unknown[]).map(String) };
    case "on_event":
      return { kind: "on_event", event: parseHookEvent(node["event"]) };
    case "all":
      return {
        kind: "all",
        conditions: (node["conditions"] as unknown[]).map(deserializeCondition),
      };
    case "any":
      return {
        kind: "any",
        conditions: (node["conditions"] as unknown[]).map(deserializeCondition),
      };
    case "not":
      return { kind: "not", condition: deserializeCondition(node["condition"]) };
    default:
      throw new Error(`unknown ChunkCondition type: ${String(type)}`);
  }
}

function parseHookEvent(value: unknown): HookEvent {
  if (typeof value === "string" && (ALL_HOOK_EVENTS as readonly string[]).includes(value)) {
    return value as HookEvent;
  }
  throw new Error(`invalid HookEvent: ${String(value)}`);
}

// ============================================================================
// PromptChunk (this module's own — distinct from #24, decision A1)
// ============================================================================

/**
 * The unit of conditional assembly content. Distinct from the #24
 * {@link "../prompt-chunk-registry/types.js".PromptChunk}: this carries a
 * {@link ChunkCondition}, triggers, affinities, and a stability bucket rather
 * than a slot.
 */
export interface PromptChunk {
  id: string;
  content: string;
  stability: SegmentStability;
  condition: ChunkCondition;
  triggers: string[];
  tool_affinity?: ToolAffinity | undefined;
  agent_affinity?: string | undefined;
  cache_breakpoint: boolean;
}

/** Build a `static`, `always` chunk — the common case. */
export function promptChunk(id: string, content: string): PromptChunk {
  return {
    id,
    content,
    stability: "static",
    condition: { kind: "always" },
    triggers: [],
    tool_affinity: undefined,
    agent_affinity: undefined,
    cache_breakpoint: false,
  };
}

/** Serialize a {@link PromptChunk} to its wire form (condition pruned per A3). */
export function serializePromptChunk(chunk: PromptChunk): Record<string, unknown> {
  const out: Record<string, unknown> = {
    id: chunk.id,
    content: chunk.content,
    stability: chunk.stability,
    condition: serializeCondition(chunk.condition),
    triggers: [...chunk.triggers],
    tool_affinity: chunk.tool_affinity ?? null,
    agent_affinity: chunk.agent_affinity ?? null,
    cache_breakpoint: chunk.cache_breakpoint,
  };
  return out;
}

/**
 * Parse a {@link PromptChunk} from its wire form. `condition` defaults to
 * `always` when null/absent; `triggers`/`tool_affinity`/`agent_affinity`/
 * `cache_breakpoint` default to empty/undefined/false.
 */
export function parsePromptChunk(value: unknown): PromptChunk {
  const node = value as Record<string, unknown>;
  const affinityRaw = node["tool_affinity"];
  let affinity: ToolAffinity | undefined;
  if (affinityRaw !== null && affinityRaw !== undefined) {
    const a = affinityRaw as Record<string, unknown>;
    const cap = a["capability"];
    affinity = {
      tool_name: String(a["tool_name"]),
      capability: cap === null || cap === undefined ? undefined : String(cap),
    };
  }
  const agentAffinity = node["agent_affinity"];
  return {
    id: String(node["id"]),
    content: String(node["content"]),
    stability: SegmentStabilitySchema.parse(node["stability"]),
    condition: deserializeCondition(node["condition"]),
    triggers: Array.isArray(node["triggers"]) ? (node["triggers"] as unknown[]).map(String) : [],
    tool_affinity: affinity,
    agent_affinity:
      agentAffinity === null || agentAffinity === undefined ? undefined : String(agentAffinity),
    cache_breakpoint: node["cache_breakpoint"] === true,
  };
}

// ============================================================================
// AssemblyContext
// ============================================================================

/**
 * The `(tool, capability)` set on an {@link AssemblyContext}. Modeled as a
 * `Set<string>` of join keys (`{@link capabilityKey}`) so value-equality dedupes
 * — JS `Set<[string, string]>` would key on tuple reference identity.
 */
export type CapabilitySet = Set<string>;

const CAP_SEP = " ";

/** The canonical join key for a `(tool, capability)` pair in a {@link CapabilitySet}. */
export function capabilityKey(tool: string, capability: string): string {
  return `${tool}${CAP_SEP}${capability}`;
}

/**
 * Per-assembly inputs the framework populates before each assembly. `custom`
 * conditions read from it; `features` is the escape hatch for architect flags.
 */
export interface AssemblyContext {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  mode: Mode;
  phase: TaskPhase;
  agent_type?: string | undefined;
  active_tool_names: Set<string>;
  active_capabilities: CapabilitySet;
  incoming_message?: string | undefined;
  pending_events: HookEvent[];
  features: Map<string, boolean>;
  storage_scope: StorageScope;
}

/** Construct a minimal context. Optional collections start empty. */
export function assemblyContext(
  sessionId: SessionId,
  taskId: TaskId,
  turnNumber: number,
  mode: Mode,
  phase: TaskPhase,
): AssemblyContext {
  return {
    session_id: sessionId,
    task_id: taskId,
    turn_number: turnNumber,
    mode,
    phase,
    agent_type: undefined,
    active_tool_names: new Set(),
    active_capabilities: new Set(),
    incoming_message: undefined,
    pending_events: [],
    features: new Map(),
    storage_scope: DEFAULT_STORAGE_SCOPE,
  };
}

/**
 * Parse an {@link AssemblyContext} from its wire form (the fixture shape).
 * `active_capabilities` is an array of `[tool, capability]` pairs; `features`
 * is an object map; optional fields default to empty/absent.
 */
export function parseAssemblyContext(value: unknown): AssemblyContext {
  const node = value as Record<string, unknown>;
  const toolNames = Array.isArray(node["active_tool_names"])
    ? (node["active_tool_names"] as unknown[]).map(String)
    : [];
  const capsRaw = Array.isArray(node["active_capabilities"])
    ? (node["active_capabilities"] as unknown[])
    : [];
  const caps: CapabilitySet = new Set();
  for (const pair of capsRaw) {
    const [tool, cap] = pair as [unknown, unknown];
    caps.add(capabilityKey(String(tool), String(cap)));
  }
  const featuresRaw = (node["features"] ?? {}) as Record<string, unknown>;
  const features = new Map<string, boolean>();
  for (const [k, v] of Object.entries(featuresRaw)) features.set(k, v === true);
  const events = Array.isArray(node["pending_events"])
    ? (node["pending_events"] as unknown[]).map(parseHookEvent)
    : [];
  const agentType = node["agent_type"];
  const incoming = node["incoming_message"];
  return {
    session_id: SessionId.of(String(node["session_id"])),
    task_id: TaskId.of(String(node["task_id"])),
    turn_number: Number(node["turn_number"]),
    mode: node["mode"] as Mode,
    phase: TaskPhaseSchema.parse(node["phase"]),
    agent_type: agentType === null || agentType === undefined ? undefined : String(agentType),
    active_tool_names: new Set(toolNames),
    active_capabilities: caps,
    incoming_message: incoming === null || incoming === undefined ? undefined : String(incoming),
    pending_events: events,
    features,
    storage_scope: (node["storage_scope"] as StorageScope | undefined) ?? DEFAULT_STORAGE_SCOPE,
  };
}

// ============================================================================
// ChunkProviderError
// ============================================================================

export type ChunkProviderErrorKind =
  | { kind: "load_failed"; provider: string; detail: string }
  | { kind: "parse_error"; detail: string };

export function chunkProviderErrorMessage(e: ChunkProviderErrorKind): string {
  switch (e.kind) {
    case "load_failed":
      return `chunk load failed from ${e.provider}: ${e.detail}`;
    case "parse_error":
      return `chunk parse error: ${e.detail}`;
  }
}

/**
 * Errors a {@link ChunkProvider} can raise while loading chunks. Kept minimal
 * because the Remote/FileSystem providers are deferred (A6). Carries a
 * discriminant `kind` for `switch` exhaustiveness.
 */
export class ChunkProviderError extends Error {
  override readonly name = "ChunkProviderError";
  readonly kind: ChunkProviderErrorKind["kind"];
  constructor(readonly error: ChunkProviderErrorKind) {
    super(chunkProviderErrorMessage(error));
    this.kind = error.kind;
  }

  static loadFailed(provider: string, detail: string): ChunkProviderError {
    return new ChunkProviderError({ kind: "load_failed", provider, detail });
  }

  static parseError(detail: string): ChunkProviderError {
    return new ChunkProviderError({ kind: "parse_error", detail });
  }
}

// ============================================================================
// ChunkProvider interface + reference providers
// ============================================================================

/**
 * The pluggable source of chunks. Injected as `ChunkProvider`. `load()` is
 * called at harness construction (every request in stateless deployments, once
 * at startup in long-lived ones). `invalidate()` drops cached state so the next
 * `load()` fetches fresh; never called mid-session.
 */
export interface ChunkProvider {
  load(signal?: AbortSignal): Promise<PromptChunk[]>;
  invalidate?(): void;
}

/**
 * Compile-time / construction-time chunks. Immutable; `invalidate` is a no-op
 * and `load` always returns the same set (cloned).
 */
export class EmbeddedChunkProvider implements ChunkProvider {
  constructor(private readonly chunks: PromptChunk[]) {}

  load(): Promise<PromptChunk[]> {
    return Promise.resolve(this.chunks.map(cloneChunk));
  }
  // invalidate: no-op (chunks are immutable constants).
}

/**
 * Programmatic provider. {@link set} replaces the chunk list; the next `load()`
 * returns the new set. Primary path for testing and full architect control.
 */
export class InMemoryChunkProvider implements ChunkProvider {
  private chunks: PromptChunk[];
  constructor(chunks: PromptChunk[] = []) {
    this.chunks = chunks;
  }

  static empty(): InMemoryChunkProvider {
    return new InMemoryChunkProvider([]);
  }

  /** Replace the chunk list. The next `load()` returns the new set. */
  set(chunks: PromptChunk[]): void {
    this.chunks = chunks;
  }

  load(): Promise<PromptChunk[]> {
    return Promise.resolve(this.chunks.map(cloneChunk));
  }

  invalidate(): void {
    // Stateless cache; the architect replaces chunks via `set`. Clearing here
    // would discard programmatic registrations, so this is a no-op.
  }
}

/**
 * Merges N providers into one flat list (in add order) and propagates
 * `invalidate` to every child.
 */
export class CompositeChunkProvider implements ChunkProvider {
  private readonly providers: ChunkProvider[] = [];

  add(provider: ChunkProvider): this {
    this.providers.push(provider);
    return this;
  }

  async load(signal?: AbortSignal): Promise<PromptChunk[]> {
    const out: PromptChunk[] = [];
    for (const p of this.providers) {
      out.push(...(await p.load(signal)));
    }
    return out;
  }

  invalidate(): void {
    for (const p of this.providers) p.invalidate?.();
  }
}

function cloneChunk(chunk: PromptChunk): PromptChunk {
  return {
    ...chunk,
    triggers: [...chunk.triggers],
    tool_affinity: chunk.tool_affinity ? { ...chunk.tool_affinity } : undefined,
    condition: chunk.condition,
  };
}

// ============================================================================
// ContextSourcesBuilder
// ============================================================================

/**
 * The bucketed outcome of {@link ContextSourcesBuilder.assemble}. Buckets keep
 * registration order within each stability tier.
 */
export interface AssemblyBuckets {
  static_chunks: PromptChunk[];
  per_session: PromptChunk[];
  per_turn: PromptChunk[];
}

/**
 * Evaluates conditions, buckets chunks by stability, derives tool-affinity
 * inclusion, scans triggers, injects pending events, and composes a Block-1
 * {@link RegistryComposedPrompt} from the Static bucket. The result feeds
 * {@link ContextSources} (decision A4).
 */
export class ContextSourcesBuilder {
  private chunks: PromptChunk[];

  constructor(chunks: PromptChunk[] = []) {
    this.chunks = chunks;
  }

  /** Seed the builder with chunks (registration order is preserved). */
  static withChunks(chunks: PromptChunk[]): ContextSourcesBuilder {
    return new ContextSourcesBuilder(chunks);
  }

  /** Append a chunk, preserving registration order. */
  register(chunk: PromptChunk): this {
    this.chunks.push(chunk);
    return this;
  }

  /**
   * The load-bearing primitive: recursively evaluate `condition` against `ctx`.
   * Rules R1–R9.
   */
  evaluate(condition: ChunkCondition, ctx: AssemblyContext): boolean {
    switch (condition.kind) {
      // R1
      case "always":
        return true;
      // R2
      case "when_mode":
        return ctx.mode === condition.mode;
      // R3
      case "when_tool_active":
        return ctx.active_tool_names.has(condition.tool);
      // R4
      case "when_tool_capability":
        return ctx.active_capabilities.has(capabilityKey(condition.tool, condition.capability));
      // R5
      case "when_phase":
        return ctx.phase === condition.phase;
      case "when_agent_type":
        return ctx.agent_type === condition.agent_type;
      case "when_feature":
        return ctx.features.get(condition.feature) === true;
      // R6 — substring match; absent message never matches.
      case "on_trigger":
        return ctx.incoming_message === undefined
          ? false
          : condition.words.some((w) => ctx.incoming_message!.includes(w));
      // R7
      case "on_event":
        return ctx.pending_events.includes(condition.event);
      // R8
      case "all":
        return condition.conditions.every((c) => this.evaluate(c, ctx));
      case "any":
        return condition.conditions.some((c) => this.evaluate(c, ctx));
      case "not":
        return !this.evaluate(condition.condition, ctx);
      // R9
      case "custom":
        return condition.predicate(ctx);
    }
  }

  /**
   * Run the assembly steps and bucket the included chunks. Registration order is
   * preserved within each bucket (R10/R11).
   *
   * A chunk is included when its `condition` evaluates true AND its
   * `tool_affinity` AND `agent_affinity` gates pass. A chunk whose `triggers`
   * match the incoming message is forced into the PerTurn bucket regardless of
   * its declared stability (R13).
   */
  assemble(ctx: AssemblyContext): AssemblyBuckets {
    const staticChunks: PromptChunk[] = [];
    const perSession: PromptChunk[] = [];
    const perTurn: PromptChunk[] = [];

    for (const chunk of this.chunks) {
      // Gates that apply to EVERY chunk regardless of condition kind.
      if (!toolAffinityOk(chunk, ctx)) continue;
      if (!agentAffinityOk(chunk, ctx)) continue;

      const conditionOk = this.evaluate(chunk.condition, ctx);
      const triggerForced = triggersMatch(chunk, ctx);

      if (!conditionOk && !triggerForced) continue;

      // R13: a trigger match routes the chunk into PerTurn no matter its
      // declared stability. R14 falls out of this too: an `on_event` chunk is
      // only condition-ok when its event is pending.
      if (triggerForced) {
        perTurn.push(chunk);
        continue;
      }

      switch (chunk.stability) {
        case "static":
          staticChunks.push(chunk);
          break;
        case "per_session":
          perSession.push(chunk);
          break;
        case "per_turn":
          perTurn.push(chunk);
          break;
      }
    }

    return { static_chunks: staticChunks, per_session: perSession, per_turn: perTurn };
  }

  /**
   * Compose the Static bucket into a #24 {@link RegistryComposedPrompt}
   * (Block 1). Each Static chunk maps to a #24 `PromptChunk` in the
   * `environment` slot (a neutral, non-required slot) with the `static` cache
   * block, preserving order. Block hashes are recomputed so the Block-1 hash is
   * stable across identical Static sets (R15).
   */
  composeBlock1(buckets: AssemblyBuckets): RegistryComposedPrompt {
    const chunks = buckets.static_chunks.map((c) =>
      registryPromptChunk(
        c.id,
        c.content,
        "environment" satisfies ChunkSlot,
        "static" satisfies CacheBlock,
      ),
    );
    const [b1, b2] = computeBlockHashes(chunks);
    const composed: RegistryComposedPrompt = {
      chunks,
      block_1_hash: b1,
      block_2_hash: b2,
      rendered: null,
    };
    renderComposed(composed);
    return composed;
  }

  /**
   * Full pipeline: assemble buckets, compose Block 1, and produce a
   * {@link ContextSources} (decision A4). `guides`, `memory`, and `toolSchemas`
   * are passed through verbatim — the builder does NOT synthesize tool
   * description text (decision A5).
   */
  buildContextSources(
    ctx: AssemblyContext,
    guides: Guide[] = [],
    memory: MemoryItem[] = [],
    toolSchemas: ToolSchema[] = [],
  ): { sources: ContextSources; buckets: AssemblyBuckets } {
    const buckets = this.assemble(ctx);
    const registryComposed = this.composeBlock1(buckets);
    const composed_prompt: ContextComposedPrompt = {
      rendered: renderComposed(registryComposed),
      block_1_hash: registryComposed.block_1_hash,
    };
    const sources: ContextSources = {
      guides,
      memory,
      tool_schemas: toolSchemas,
      composed_prompt,
    };
    return { sources, buckets };
  }
}

/** Whether a chunk's `tool_affinity` gate passes for `ctx` (R12 / R17). */
function toolAffinityOk(chunk: PromptChunk, ctx: AssemblyContext): boolean {
  const aff = chunk.tool_affinity;
  if (!aff) return true;
  if (!ctx.active_tool_names.has(aff.tool_name)) return false;
  if (aff.capability === undefined) return true;
  return ctx.active_capabilities.has(capabilityKey(aff.tool_name, aff.capability));
}

/** Whether a chunk's `agent_affinity` gate passes. */
function agentAffinityOk(chunk: PromptChunk, ctx: AssemblyContext): boolean {
  if (chunk.agent_affinity === undefined) return true;
  return ctx.agent_type === chunk.agent_affinity;
}

/** Whether a chunk's `triggers` match the incoming message (R13). */
function triggersMatch(chunk: PromptChunk, ctx: AssemblyContext): boolean {
  if (chunk.triggers.length === 0) return false;
  if (ctx.incoming_message === undefined) return false;
  const msg = ctx.incoming_message;
  return chunk.triggers.some((t) => msg.includes(t));
}

/**
 * Breakpoint ids derived from a bucket: an entry per chunk that declared
 * `cache_breakpoint` (R16), in order (static → per_session → per_turn).
 */
export function breakpointIds(buckets: AssemblyBuckets): string[] {
  const out: string[] = [];
  for (const chunk of [...buckets.static_chunks, ...buckets.per_session, ...buckets.per_turn]) {
    if (chunk.cache_breakpoint) out.push(chunk.id);
  }
  return out;
}

/**
 * Map a bucket of chunks into {@link PromptSegment}s for the #7 context
 * machinery (decision A4). Preserves order and carries each chunk's
 * `cache_breakpoint` (R16).
 */
export function chunksToSegments(chunks: PromptChunk[]): PromptSegment[] {
  return chunks.map((c) => ({
    name: c.id,
    content: c.content,
    stability: c.stability,
    cache_breakpoint: c.cache_breakpoint,
  }));
}
