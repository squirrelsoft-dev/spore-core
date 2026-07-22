/**
 * Reference in-memory `VectorStore`: the trait's executable contract.
 * Mirrors the `InMemoryVectorStore` in `rust/.../vector_store.rs` tests (the
 * trait's reference backend; `spore-sqlvec` is the durable one, out of scope).
 *
 * Uses cosine similarity over a `VectorRecord[]`. Intentionally SHARED across
 * all callers of the instance — scoping is the caller's responsibility via
 * {@link MetadataFilter}. Concurrency: operations are serialized (JS is
 * single-threaded); a `search` sees all prior `add`s.
 */

import { cosineSimilarity } from "../embedding/types.js";
import {
  backendError,
  dimensionMismatch,
  MetadataFilter,
  type VectorHit,
  type VectorRecord,
  type VectorStore,
  type VectorStoreError,
} from "./types.js";

export class InMemoryVectorStore implements VectorStore {
  private readonly dimensions: number;
  private readonly rows: VectorRecord[] = [];

  constructor(dimensions: number) {
    this.dimensions = dimensions;
  }

  async add(records: readonly VectorRecord[]): Promise<void> {
    for (const record of records) {
      if (record.vector.length !== this.dimensions) {
        throw dimensionMismatch(this.dimensions, record.vector.length);
      }
      // Upsert by id.
      const existing = this.rows.findIndex((r) => r.id === record.id);
      if (existing >= 0) {
        this.rows[existing] = record;
      } else {
        this.rows.push(record);
      }
    }
  }

  async search(
    query: number[],
    topK: number,
    filter?: MetadataFilter,
  ): Promise<VectorHit[]> {
    const f = filter ?? MetadataFilter.new();
    const hits: VectorHit[] = this.rows
      .filter((r) => f.matches(r.metadata))
      .map((r) => ({
        id: r.id,
        score: cosineSimilarity(query, r.vector),
        metadata: r.metadata,
      }));
    hits.sort((a, b) => b.score - a.score);
    hits.length = Math.min(topK, hits.length);
    return hits;
  }

  async delete(filter: MetadataFilter): Promise<number> {
    const before = this.rows.length;
    // retain records that do NOT match (remove the ones that do).
    let removed = 0;
    for (let i = this.rows.length - 1; i >= 0; i -= 1) {
      if (filter.matches(this.rows[i]!.metadata)) {
        this.rows.splice(i, 1);
        removed += 1;
      }
    }
    void before;
    return removed;
  }

  async len(): Promise<number> {
    return this.rows.length;
  }

  async isEmpty(): Promise<boolean> {
    return this.rows.length === 0;
  }
}

export type { MetadataFilter, VectorHit, VectorRecord, VectorStore, VectorStoreError };
export { backendError, dimensionMismatch };