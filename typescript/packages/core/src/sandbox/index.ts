/**
 * `@spore/core` sandbox subpackage — issue #6.
 *
 * Re-exports `WorkspaceScopedSandbox` and its construction error type. All
 * other sandbox types (`SandboxViolation`, `IsolationMode`, `Operation`,
 * `WorkspaceConfig`, `CommandOutput`, `TruncatedOutput`, `FileRef`,
 * `NetworkPolicy`, `BwrapProfile`) live in `harness/types.ts` so the
 * `SandboxProvider` interface can reference them.
 */

export { WorkspaceScopedSandbox, BuildError } from "./workspace-sandbox.js";
export { ReadOnlySandbox, DEFAULT_WRITE_TOOLS } from "./read-only-sandbox.js";
