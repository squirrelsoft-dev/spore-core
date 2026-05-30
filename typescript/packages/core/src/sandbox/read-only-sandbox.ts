/**
 * `ReadOnlySandbox` — a {@link SandboxProvider} decorator that blocks mutating
 * tools (spore-core issue #61).
 *
 * Used by the `self_verifying` loop strategy to run the evaluate phase over a
 * read-only view of the build workspace: the evaluator may READ the work it is
 * judging but must not be able to alter it. A blocked write surfaces as a
 * `read_only_violation` — a Layer-2 (recoverable) {@link SandboxViolation}, so
 * inside the harness loop it becomes a recoverable tool error rather than a
 * halt; reads (and every non-mutating call) delegate to the wrapped sandbox.
 *
 * Mirrors `rust/crates/spore-core/src/sandbox.rs` `ReadOnlySandbox`. The default
 * blocked set is the standard catalogue's mutating tools.
 */

import type { ToolCall } from "../model/schemas.js";
import type {
  CommandOutput,
  IsolationMode,
  Operation,
  SandboxProvider,
  SandboxViolation,
  TruncatedOutput,
} from "../harness/types.js";

/**
 * Standard-catalogue tool names that MUTATE the workspace and are therefore
 * blocked by a read-only sandbox. Mirrors Rust's `DEFAULT_WRITE_TOOLS`.
 */
export const DEFAULT_WRITE_TOOLS: readonly string[] = [
  "write_file",
  "edit_file",
  "delete_file",
  "move_file",
  "exec",
  "bash_command",
  "run_tests",
];

export class ReadOnlySandbox implements SandboxProvider {
  private readonly writeTools: Set<string>;

  /**
   * Wrap `inner`, blocking `writeTools` (defaults to {@link DEFAULT_WRITE_TOOLS}).
   */
  constructor(
    private readonly inner: SandboxProvider,
    writeTools: Iterable<string> = DEFAULT_WRITE_TOOLS,
  ) {
    this.writeTools = new Set(writeTools);
  }

  private isWrite(toolName: string): boolean {
    return this.writeTools.has(toolName);
  }

  async validate(call: ToolCall, signal?: AbortSignal): Promise<SandboxViolation | null> {
    if (this.isWrite(call.name)) {
      return { kind: "read_only_violation", path: call.name };
    }
    return this.inner.validate(call, signal);
  }

  async executeCommand(
    command: string,
    _args: readonly string[],
    _cwd?: string | null,
    _timeoutMs?: number | null,
    _signal?: AbortSignal,
  ): Promise<CommandOutput | SandboxViolation> {
    // A read-only sandbox forbids subprocess execution outright (commands may
    // have arbitrary write side effects).
    return { kind: "read_only_violation", path: command };
  }

  async handleLargeOutput(
    content: string,
    callId: string,
    headTokens: number,
    tailTokens: number,
  ): Promise<TruncatedOutput> {
    if (this.inner.handleLargeOutput) {
      return this.inner.handleLargeOutput(content, callId, headTokens, tailTokens);
    }
    return { content, truncated: false, full_ref: null, original_size: content.length };
  }

  async resolvePath(path: string, operation: Operation): Promise<string | SandboxViolation> {
    if (operation === "write" || operation === "execute") {
      return { kind: "read_only_violation", path };
    }
    if (this.inner.resolvePath) {
      return this.inner.resolvePath(path, operation);
    }
    return path;
  }

  isolationMode(): IsolationMode {
    return this.inner.isolationMode?.() ?? { kind: "none" };
  }

  workspaceRoot(): string {
    return this.inner.workspaceRoot?.() ?? "";
  }
}
