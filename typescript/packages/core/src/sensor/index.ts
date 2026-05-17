/**
 * Public re-exports for the canonical {@link SensorChain} (spore-core
 * issue #10). Re-exported under the `sensor` namespace at the package root
 * for symmetry with `memory`, `context`, and `guideRegistry`.
 */

export * from "./types.js";
export { StandardSensorChain } from "./standard.js";
