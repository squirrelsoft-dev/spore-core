/**
 * Public re-exports for the canonical {@link ContextManager} (spore-core
 * issue #7).
 *
 * Re-exported under the `context` namespace at the package root to avoid
 * name collisions with the harness's narrower forward-declared
 * `ContextManager` / `SessionState` types in `../harness/types.ts`.
 */

export * from "./types.js";
export { StandardContextManager } from "./standard.js";
export type { StandardContextManagerOptions } from "./standard.js";
export {
  StandardCompactionAdapter,
  intoHarnessAdapter,
  seedRichState,
  RICH_STATE_KEY,
} from "./compaction-adapter.js";
