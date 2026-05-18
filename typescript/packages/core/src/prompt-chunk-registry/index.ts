/**
 * Public re-exports for the canonical {@link PromptChunkRegistry} (spore-core
 * issue #24). Re-exported under the `promptChunkRegistry` namespace at the
 * package root for symmetry with `guideRegistry`, `memory`, etc.
 */

export * from "./types.js";
export { StandardPromptChunkRegistry, standardChunks } from "./standard.js";
