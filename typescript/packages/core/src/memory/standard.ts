/**
 * StandardMemoryProvider — in-memory reference implementation of
 * {@link MemoryProvider} (spore-core issue #8).
 *
 * Mirrors `rust/crates/spore-core/src/memory.rs#StandardMemoryProvider` —
 * same rules, same fixture outcomes. Suitable for tests and short-lived
 * processes; production deployments use a durable backing store.
 *
 * Relevance scoring is Jaccard token-overlap on lowercased alphanumeric
 * tokens. The spec calls for embedding similarity in production; the trait
 * only mandates that items below `min_relevance` are not returned.
 */

import type { SessionId } from "../harness/types.js";

import {
  type EpisodicMemory,
  MemoryError,
  MemoryId,
  type MemoryItem,
  type MemoryProvider,
  type MemoryQuery,
  type MergeStrategy,
  type SemanticMemory,
} from "./types.js";

// ============================================================================
// Helpers
// ============================================================================

function validateContent(s: string): void {
  if (s.trim().length === 0) {
    throw MemoryError.validationFailed("content must not be empty");
  }
}

/**
 * Force `meta_agent_proposed` memories into `pending_review` regardless of
 * caller-supplied status. The `proposed_at` falls back to `created_at`.
 */
function enforceMetaAgentReview(memory: SemanticMemory): void {
  if (memory.source.kind !== "meta_agent_proposed") return;
  if (memory.status.kind === "pending_review") return;
  memory.status = {
    kind: "pending_review",
    proposed_at: memory.created_at,
  };
}

function tokenize(s: string): string[] {
  return s
    .toLowerCase()
    .split(/[^a-z0-9]+/u)
    .filter((t) => t.length > 0);
}

function jaccard(a: string, b: string): number {
  const ta = new Set(tokenize(a));
  const tb = new Set(tokenize(b));
  if (ta.size === 0 && tb.size === 0) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter += 1;
  const union = ta.size + tb.size - inter;
  return union === 0 ? 0 : inter / union;
}

function cloneMemoryId(id: MemoryId): MemoryId {
  return new MemoryId(id.value);
}

function cloneSemantic(m: SemanticMemory): SemanticMemory {
  return {
    id: cloneMemoryId(m.id),
    content: m.content,
    source: cloneSource(m.source),
    domain: m.domain ?? null,
    version: m.version,
    previous_versions: m.previous_versions.map(cloneMemoryId),
    created_at: m.created_at,
    updated_at: m.updated_at,
    status: cloneStatus(m.status),
  };
}

function cloneSource(s: SemanticMemory["source"]): SemanticMemory["source"] {
  switch (s.kind) {
    case "manual":
      return { kind: "manual" };
    case "session_generated":
      return { kind: "session_generated", session_id: s.session_id };
    case "trace_distilled":
      return { kind: "trace_distilled", session_ids: [...s.session_ids] };
    case "meta_agent_proposed":
      return { kind: "meta_agent_proposed", approved_by: s.approved_by ?? null };
  }
}

function cloneStatus(s: SemanticMemory["status"]): SemanticMemory["status"] {
  switch (s.kind) {
    case "active":
      return { kind: "active" };
    case "deprecated":
      return { kind: "deprecated", reason: s.reason, at: s.at };
    case "pending_review":
      return { kind: "pending_review", proposed_at: s.proposed_at };
  }
}

function cloneEpisodic(m: EpisodicMemory): EpisodicMemory {
  return {
    id: cloneMemoryId(m.id),
    session_id: m.session_id,
    content: m.content,
    created_at: m.created_at,
    tags: [...m.tags],
  };
}

// ============================================================================
// StandardMemoryProvider
// ============================================================================

export class StandardMemoryProvider implements MemoryProvider {
  /** Episodic memories keyed by session id string, preserving insertion order. */
  private readonly episodic = new Map<string, EpisodicMemory[]>();
  /** All semantic memories ever stored, keyed by id string (current + archived). */
  private readonly semantic = new Map<string, SemanticMemory>();
  /** Monotonic counter for generating archival ids on Replace. */
  private archiveSeq = 0;

  private nextArchiveId(original: MemoryId): MemoryId {
    this.archiveSeq += 1;
    return new MemoryId(`${original.value}@v${this.archiveSeq}`);
  }

  async storeEpisodic(memory: EpisodicMemory): Promise<MemoryId> {
    validateContent(memory.content);
    const key = memory.session_id.asString();
    const bucket = this.episodic.get(key) ?? [];
    bucket.push(cloneEpisodic(memory));
    this.episodic.set(key, bucket);
    return cloneMemoryId(memory.id);
  }

  async getEpisodic(session_id: SessionId): Promise<EpisodicMemory[]> {
    const bucket = this.episodic.get(session_id.asString());
    if (!bucket) return [];
    return bucket.map(cloneEpisodic);
  }

  async storeSemantic(memory: SemanticMemory, on_conflict: MergeStrategy): Promise<MemoryId> {
    validateContent(memory.content);
    const incoming = cloneSemantic(memory);
    enforceMetaAgentReview(incoming);

    const id = incoming.id;
    const key = id.value;
    const existing = this.semantic.get(key);

    if (existing) {
      switch (on_conflict) {
        case "reject":
          throw MemoryError.mergeConflict(
            cloneMemoryId(existing.id),
            "memory exists; on_conflict=Reject",
          );
        case "append": {
          const merged = cloneSemantic(existing);
          merged.content = merged.content + incoming.content;
          merged.updated_at = incoming.updated_at;
          this.semantic.set(key, merged);
          return cloneMemoryId(id);
        }
        case "replace": {
          // Archive the existing record under a fresh archival id, link it
          // into the new memory's previous_versions, bump version, and write
          // the new record under the original id.
          const archiveId = this.nextArchiveId(existing.id);
          const archived = cloneSemantic(existing);
          archived.id = archiveId;
          this.semantic.set(archiveId.value, archived);

          incoming.version = existing.version + 1;
          incoming.previous_versions = [
            ...existing.previous_versions.map(cloneMemoryId),
            cloneMemoryId(archiveId),
          ];
          break;
        }
      }
    }

    this.semantic.set(key, incoming);
    return cloneMemoryId(id);
  }

  async getSemantic(id: MemoryId): Promise<SemanticMemory> {
    const found = this.semantic.get(id.value);
    if (!found) throw MemoryError.notFound(cloneMemoryId(id));
    return cloneSemantic(found);
  }

  async query(query: MemoryQuery): Promise<MemoryItem[]> {
    const wantDomain = query.domain ?? null;
    const scored: MemoryItem[] = [];
    for (const m of this.semantic.values()) {
      if (m.status.kind !== "active") continue;
      const haveDomain = m.domain ?? null;
      if (wantDomain !== null) {
        if (haveDomain === null) continue;
        if (haveDomain !== wantDomain) continue;
      }
      const score = jaccard(query.task_instruction, m.content);
      if (score < query.min_relevance) continue;
      scored.push({ memory: cloneSemantic(m), relevance_score: score });
    }
    scored.sort((a, b) => b.relevance_score - a.relevance_score);
    return scored.slice(0, query.max_items);
  }

  async deprecate(id: MemoryId, reason: string): Promise<void> {
    const found = this.semantic.get(id.value);
    if (!found) throw MemoryError.notFound(cloneMemoryId(id));
    found.status = {
      kind: "deprecated",
      reason,
      at: found.updated_at,
    };
  }

  async getVersionHistory(id: MemoryId): Promise<SemanticMemory[]> {
    const head = this.semantic.get(id.value);
    if (!head) throw MemoryError.notFound(cloneMemoryId(id));
    const chain: SemanticMemory[] = [cloneSemantic(head)];
    for (const prevId of head.previous_versions) {
      const prev = this.semantic.get(prevId.value);
      if (prev) chain.push(cloneSemantic(prev));
    }
    return chain;
  }

  async markPendingReview(id: MemoryId): Promise<void> {
    const found = this.semantic.get(id.value);
    if (!found) throw MemoryError.notFound(cloneMemoryId(id));
    found.status = {
      kind: "pending_review",
      proposed_at: found.updated_at,
    };
  }
}
