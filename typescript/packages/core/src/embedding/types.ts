/**
 * `EmbeddingModel` — boundary between the RAG layer and an embedding provider.
 *
 * Mirrors `rust/crates/spore-core/src/embedding.rs`: the trait deals ONLY in
 * text → vector; storing and searching the resulting vectors is
 * {@link "../vector-store/index.js".VectorStore}. Two providers ship in-core
 * because they add no new dependency — they reuse the existing HTTP plumbing:
 *   - {@link OllamaEmbeddingModel} — POST `/api/embed`.
 *   - {@link OpenAIEmbeddingModel} — POST `${base_url}/embeddings`.
 *
 * Both report a fixed {@link EmbeddingModel.dimensions}. Ollama does not
 * advertise vector length out-of-band, so a caller that does not know it
 * statically can discover it once via
 * {@link OllamaEmbeddingModel.probeDimensions} and pin it with
 * {@link OllamaEmbeddingModel.withDimensions} before sizing a vector store.
 */

import type { ModelError } from "../model/errors.js";

// ============================================================================
// Identity
// ============================================================================

/**
 * Identity of an embedding provider/model, for routing and diagnostics.
 * Mirrors Rust `EmbeddingInfo`.
 */
export interface EmbeddingInfo {
  /** Provider name, e.g. `"ollama"` / `"openai"`. */
  readonly provider: string;
  /** Provider model id, e.g. `"nomic-embed-text"` / `"text-embedding-3-small"`. */
  readonly model_id: string;
  /** Length of the vectors this model produces (`0` when not yet known). */
  readonly dimensions: number;
}

// ============================================================================
// Trait
// ============================================================================

/**
 * Turns text into embedding vectors. A `Box<dyn EmbeddingModel>` /
 * `Arc<dyn EmbeddingModel>` in Rust is, in TS, just a value implementing this
 * interface (structural typing).
 *
 * `embed` returns one vector per input IN INPUT ORDER. An empty `inputs`
 * returns an empty array without contacting the provider.
 */
export interface EmbeddingModel {
  /** Embed a batch of inputs, returning one vector per input IN INPUT ORDER. */
  embed(inputs: string[], signal?: AbortSignal): Promise<number[][]>;

  /**
   * Convenience: embed a single input. Defaults to a one-element `embed` call.
   * Throws `ProviderError` if the provider returns no vectors.
   */
  embedOne(input: string, signal?: AbortSignal): Promise<number[]>;

  /** Vector length this model produces. `0` means "not yet known". */
  dimensions(): number;

  /** Provider identity for tracing/routing. */
  info(): EmbeddingInfo;
}

// ============================================================================
// cosine similarity
// ============================================================================

/**
 * Cosine similarity of two equal-length vectors in `[-1, 1]` (higher = closer).
 * Returns `0.0` when either vector has zero magnitude or lengths differ — a
 * degenerate input is treated as "no similarity" rather than `NaN`/throwing.
 * Used by in-memory vector stores and as the metric most providers' vectors
 * are tuned for. Mirrors Rust `cosine_similarity`.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export type { ModelError };