/**
 * `VectorStore` — store and similarity-search embedding vectors.
 *
 * Mirrors `rust/crates/spore-core/src/vector_store.rs`. The trait is
 * deliberately vector-only — "embed this text, then store/search it" is a
 * composition the caller performs, keeping the storage backend independent of
 * any particular embedding provider.
 */

// ============================================================================
// Records, hits, filters
// ============================================================================

/**
 * A vector to store: a caller-chosen `id`, the `vector` itself, and arbitrary
 * JSON `metadata` (a JSON object by convention — e.g. the source text, a stack
 * id, a `kind` discriminator). Re-adding an existing `id` is an upsert.
 */
export interface VectorRecord {
  readonly id: string;
  readonly vector: number[];
  readonly metadata: unknown;
}

/** Construct a {@link VectorRecord}. */
export function vectorRecord(id: string, vector: number[], metadata: unknown): VectorRecord {
  return { id, vector, metadata };
}

/**
 * One search result: the stored `id`, its `score` (HIGHER = more similar —
 * backends normalise their native distance into this convention), and the
 * record's `metadata`.
 */
export interface VectorHit {
  readonly id: string;
  readonly score: number;
  readonly metadata: unknown;
}

/**
 * An AND-ed set of equality predicates over top-level metadata keys. Kept
 * intentionally small: it expresses the common "scope this search to
 * `kind = "chunk"` AND `stack_id = "x"`" need. Disjunctions are expressed by
 * running multiple searches and merging.
 */
export class MetadataFilter {
  readonly equals: Array<[string, unknown]>;

  constructor(equals: Array<[string, unknown]> = []) {
    this.equals = equals;
  }

  /** Empty filter (matches everything). */
  static new(): MetadataFilter {
    return new MetadataFilter();
  }

  /** Add an equality predicate (builder style). */
  eq(key: string, value: unknown): MetadataFilter {
    return new MetadataFilter([...this.equals, [key, value]]);
  }

  isEmpty(): boolean {
    return this.equals.length === 0;
  }

  /** Whether a record's `metadata` satisfies every predicate. */
  matches(metadata: unknown): boolean {
    if (this.equals.length === 0) return true;
    const obj = metadata as Record<string, unknown> | null | undefined;
    if (obj == null || typeof obj !== "object") return false;
    return this.equals.every(([k, v]) => obj[k] === v);
  }
}

// ============================================================================
// Errors
// ============================================================================

export type VectorStoreError =
  | { readonly kind: "dimension_mismatch"; readonly expected: number; readonly actual: number }
  | { readonly kind: "backend"; readonly message: string };

export function dimensionMismatch(expected: number, actual: number): VectorStoreError {
  return { kind: "dimension_mismatch", expected, actual };
}

export function backendError(message: string): VectorStoreError {
  return { kind: "backend", message };
}

// ============================================================================
// Trait
// ============================================================================

/**
 * Store and similarity-search embedding vectors.
 *
 * `add` rejects a record whose vector length differs from the store's
 * dimensionality with {@link dimensionMismatch}. `search` returns up to
 * `topK` nearest records to `query`, highest `score` first; when `filter` is
 * given, only records whose metadata satisfies it are considered (the filter
 * is applied as part of the search, not as a post-truncation drop). `delete`
 * removes every record whose metadata matches `filter` (an empty filter
 * matches all — a full clear) and returns the count removed.
 */
export interface VectorStore {
  add(records: readonly VectorRecord[], signal?: AbortSignal): Promise<void>;
  search(
    query: number[],
    topK: number,
    filter?: MetadataFilter,
    signal?: AbortSignal,
  ): Promise<VectorHit[]>;
  delete(filter: MetadataFilter, signal?: AbortSignal): Promise<number>;
  len(signal?: AbortSignal): Promise<number>;
  isEmpty(signal?: AbortSignal): Promise<boolean>;
}