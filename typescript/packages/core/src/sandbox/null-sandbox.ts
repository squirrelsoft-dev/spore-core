/**
 * `NullSandbox` — a {@link SandboxProvider} with no filesystem.
 *
 * `validate` allows everything because nothing is meant to run against it: it
 * is the sandbox for tool-less agents (e.g.
 * {@link "../harness/standard.js".HarnessBuilder.conversational}), where no tool
 * is ever dispatched and the boundary is never exercised. Agents that actually
 * touch the filesystem or shell must use a real sandbox such as
 * {@link "./workspace-sandbox.js".WorkspaceScopedSandbox}.
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs` `NullSandbox`.
 */

import type { ToolCall } from "../model/schemas.js";
import type { AnyIsolationMode, SandboxProvider, SandboxViolation } from "../harness/types.js";

export class NullSandbox implements SandboxProvider {
  /** Allows everything — the boundary is never exercised by a tool-less agent. */
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }

  /**
   * Safe-by-default isolation mode (issue #34) — never `none`, which is a
   * dangerous-only opt-in. Mirrors Rust's `IsolationMode::WorkspaceScoped`.
   */
  isolationMode(): AnyIsolationMode {
    return { kind: "workspace_scoped" };
  }
}
