/**
 * Public re-exports for the canonical {@link ToolRegistry} (spore-core issue #4).
 *
 * The package root re-exports this module under the `toolRegistry`
 * namespace to avoid name collisions with the model-side `ToolSchema`
 * and the harness's forward-declared `ToolRegistry`.
 */

export * from "./types.js";
export {
  defineTool,
  type DefineToolExecute,
  type DefineToolOptions,
} from "./define-tool.js";
export { StandardToolRegistry } from "./standard.js";
export { EmptyToolRegistry } from "./empty.js";
export { RealToolRegistry, toModelSchema } from "./real.js";
export * as toolRegistryMock from "./mock.js";
