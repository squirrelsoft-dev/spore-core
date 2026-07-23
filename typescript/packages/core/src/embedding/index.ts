/**
 * `EmbeddingModel` — text → embedding vectors. Mirrors
 * `rust/crates/spore-core/src/embedding.rs`.
 */

export type { EmbeddingInfo, EmbeddingModel, ModelError } from "./types.js";
export { cosineSimilarity } from "./types.js";
export { OllamaEmbeddingModel, OpenAIEmbeddingModel } from "./standard.js";