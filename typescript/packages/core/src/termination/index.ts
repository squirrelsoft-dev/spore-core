/**
 * Public re-exports for the canonical {@link TerminationPolicy} (spore-core
 * issue #13). Re-exported under the `termination` namespace at the package
 * root for symmetry with `memory`, `context`, `guideRegistry`, `sensor`,
 * `middleware`, and `observability`.
 */

export * from "./types.js";
export { StandardTerminationPolicy } from "./standard.js";
