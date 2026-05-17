/**
 * Public re-exports for the canonical {@link MemoryProvider} (spore-core
 * issue #8).
 *
 * Re-exported under the `memory` namespace at the package root to avoid
 * name collisions with the harness's forward-declared `MemoryItem` shim
 * in `../context/types.ts`.
 */

export * from "./types.js";
export { StandardMemoryProvider } from "./standard.js";
