/**
 * `ReadOnlyToolView` — a read-only VIEW over an inner harness-loop
 * {@link ToolRegistry} (spore-core SC-30, issue #153).
 *
 * It advertises ({@link ReadOnlyToolView.schemas}) and dispatches ONLY tools
 * whose name is in `allow` — the INTERSECTION of the wrapped catalogue with a
 * read-only allow-list ({@link READONLY_EVAL_TOOL_NAMES}). Used internally for
 * the SelfVerifying evaluate phase so a reviewer cannot reach
 * write / exec / side-effecting tools (web/MCP the read-only sandbox does not
 * gate) even though the work phase could — WITHOUT the consumer registering a
 * scoped read-only toolset. A non-allow-listed dispatch (which the model should
 * never request, since it is never advertised) returns a recoverable
 * {@link toolOutput.error}. A consumer that wants a different reviewer toolset
 * sets an explicit `evalToolset` handle, which bypasses this view entirely.
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs` `ReadOnlyToolView`.
 */

import type { ToolCall, ToolSchema } from "../model/schemas.js";
import type { ToolOutput, ToolRegistry } from "../harness/types.js";
import { toolOutput } from "../harness/types.js";

/**
 * The read-only allow-list for the auto-derived SelfVerifying evaluate-phase
 * toolset (SC-30). These are exactly the tool names in
 * `@spore/tools`'s `StandardTools.readonlySet()`.
 *
 * It is DUPLICATED here (rather than imported) because `@spore/tools` depends on
 * `@spore/core`, so `core` cannot import the catalogue without a dependency
 * cycle. A drift-guard test in `@spore/tools` (which can import both) asserts the
 * two stay in lockstep; if `readonlySet()` ever changes, that test fails loudly.
 */
export const READONLY_EVAL_TOOL_NAMES: readonly string[] = [
  "read_file",
  "list_dir",
  "grep_files",
  "grep",
  "find_files",
  "web_fetch",
  "web_search",
] as const;

/**
 * A read-only VIEW over an inner harness-loop {@link ToolRegistry}: advertises
 * and dispatches ONLY the tools whose name is in `allow`. See the module docs
 * for the full rationale (SC-30 / issue #153).
 */
export class ReadOnlyToolView implements ToolRegistry {
  private readonly allow: Set<string>;

  constructor(
    private readonly inner: ToolRegistry,
    allow: Iterable<string>,
  ) {
    this.allow = new Set(allow);
  }

  /**
   * Dispatch through to the inner registry when the tool is allow-listed;
   * otherwise return a RECOVERABLE error (the model should never request a
   * non-advertised tool, so this is a guard, not the happy path).
   */
  async dispatch(call: ToolCall, signal?: AbortSignal): Promise<ToolOutput> {
    if (this.allow.has(call.name)) {
      return this.inner.dispatch(call, signal);
    }
    return toolOutput.error(
      `tool \`${call.name}\` is not available in the read-only evaluate phase`,
    );
  }

  /** Delegates to the inner registry — the allow-list does not change always-halt. */
  isAlwaysHalt(toolName: string): boolean {
    return this.inner.isAlwaysHalt(toolName);
  }

  /** Only the allow-listed (intersection) schemas are advertised to the model. */
  schemas(): ToolSchema[] {
    return this.inner.schemas().filter((s) => this.allow.has(s.name));
  }
}
