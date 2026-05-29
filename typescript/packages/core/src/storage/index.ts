/**
 * Public re-exports for the {@link StorageProvider} abstraction (spore-core
 * issue #73). Re-exported under the `storage` namespace at the package root for
 * symmetry with `memory`, `observability`, etc.
 */

export * from "./types.js";
export * from "./errors.js";
export {
  NoOpStorageProvider,
  InMemoryStorageProvider,
  FileSystemStorageProvider,
  StorageProvider,
  CompositeStorageProvider,
  ScopedMemoryRouter,
  parseOtlpEndpoints,
} from "./providers.js";
