/**
 * `VectorStore` — store and similarity-search embedding vectors.
 * Mirrors `rust/crates/spore-core/src/vector_store.rs`.
 */

export type {
  VectorRecord,
  VectorHit,
  MetadataFilter,
  VectorStore,
  VectorStoreError,
} from "./types.js";
export { vectorRecord, dimensionMismatch, backendError } from "./types.js";
export { InMemoryVectorStore } from "./standard.js";