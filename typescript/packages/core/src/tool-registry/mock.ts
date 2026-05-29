/**
 * Mock {@link Tool} implementations and sandbox stubs for unit tests and
 * fixture replay (spore-core issue #4).
 */

import { SessionId } from "../harness/types.js";
import type { ToolCall } from "../model/schemas.js";
import type { SandboxProvider, SandboxViolation, ToolOutput } from "../harness/types.js";
import { InMemoryStorageProvider } from "../storage/providers.js";
import { ToolContext, type Tool } from "./types.js";

/** Echo tool — returns its input serialised as JSON. Intended `read_only: true`. */
export class EchoTool implements Tool {
  callCount = 0;
  constructor(readonly name: string) {}
  async execute(call: ToolCall, _sandbox: SandboxProvider, _ctx: ToolContext): Promise<ToolOutput> {
    this.callCount += 1;
    return {
      kind: "success",
      content: JSON.stringify(call.input),
      truncated: false,
    };
  }
}

/** Failing tool — returns a recoverable error. */
export class FailingTool implements Tool {
  constructor(readonly name: string) {}
  async execute(
    _call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    return { kind: "error", message: "boom", recoverable: true };
  }
}

/** Subagent-flagged tool. */
export class SubagentMockTool implements Tool {
  readonly isSubagentTool = true;
  constructor(readonly name: string) {}
  async execute(
    _call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
  ): Promise<ToolOutput> {
    return { kind: "success", content: "subagent done", truncated: false };
  }
}

/**
 * Build a throwaway {@link ToolContext} for tests: a fresh in-memory backend
 * (serving both the run-store and the scope-aware memory-store seams) and a
 * fixed test session id. Mirrors Rust's `mock::test_ctx`.
 */
export function testCtx(): ToolContext {
  const backend = new InMemoryStorageProvider();
  return new ToolContext(SessionId.of("test-session"), backend, backend);
}

/** Permissive sandbox stub — accepts everything. */
export class AllowAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
}

/** Denying sandbox stub — rejects everything with PathEscape. */
export class DenyAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return { kind: "path_escape", path: "denied" };
  }
}
