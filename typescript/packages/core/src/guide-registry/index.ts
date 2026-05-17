/**
 * Public re-exports for the canonical {@link GuideRegistry} (spore-core
 * issue #9).
 *
 * Re-exported under the `guideRegistry` namespace at the package root for
 * symmetry with `memory` and `context` and to avoid name collisions with
 * forward-declared shims elsewhere in the package.
 */

export * from "./types.js";
export { StandardGuideRegistry } from "./standard.js";
