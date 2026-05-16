/**
 * Public re-exports for the `harness` component (spore-core issue #3).
 */

export * from "./types.js";
export * from "./interface.js";
export { StandardHarness, type HarnessConfig, HOOK_POINTS, isReact } from "./standard.js";
export * as harnessTesting from "./testing.js";
