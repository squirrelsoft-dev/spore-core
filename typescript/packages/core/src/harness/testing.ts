/**
 * Test-only stubs of the sibling traits. Covers ReAct paths in unit tests
 * before the canonical impls land in #4–#13.
 */

import type { Context } from "../agent/types.js";
import type { ToolCall, ToolSchema } from "../model/schemas.js";

import type { VcsLogArgs, VcsProvider } from "./vcs.js";
import type {
  ContextManager,
  HookPoint,
  MiddlewareChain,
  MiddlewareDecision,
  SandboxProvider,
  SandboxViolation,
  SessionState,
  Task,
  TerminationDecision,
  TerminationPolicy,
  ToolOutput,
  ToolRegistry,
  ToolResultRecord,
} from "./types.js";

export class NoopContextManager implements ContextManager {
  async assemble(session: SessionState, _task: Task): Promise<Context> {
    return {
      messages: session.messages.slice(),
      tools: [],
      params: { stop_sequences: [] },
    };
  }
  async appendToolResult(session: SessionState, result: ToolResultRecord): Promise<void> {
    let text: string;
    switch (result.output.kind) {
      case "success":
        text = result.output.content;
        break;
      case "error":
        text = `[error] ${result.output.message}`;
        break;
      case "waiting_for_human":
        text = "[waiting]";
        break;
      case "escalate":
        // Never reached at runtime: the harness returns RunResult.escalate
        // before appending an escalation to history (#80). Present for
        // exhaustiveness.
        text = "[escalate]";
        break;
      case "awaiting_clarification":
        // Never reached at runtime: the harness pauses (waiting_for_human)
        // before appending; present for exhaustiveness (#81).
        text = "[awaiting clarification]";
        break;
    }
    session.messages.push({ role: "tool", content: { type: "text", text } });
  }
  async appendUserMessage(session: SessionState, text: string): Promise<void> {
    session.messages.push({ role: "user", content: { type: "text", text } });
  }
  shouldCompact(_session: SessionState): boolean {
    return false;
  }
}

/**
 * Deterministic {@link VcsProvider} double for tests and fixture replay (issue
 * #58 v2). It returns pre-loaded strings VERBATIM with no process spawning, so
 * multi-context-window Ralph continuation can be exercised hermetically.
 * {@link log} ignores its {@link VcsLogArgs} and yields `logOutput`;
 * {@link status} yields `statusOutput`.
 */
export class FixtureVcsProvider implements VcsProvider {
  constructor(
    private readonly logOutput: string,
    private readonly statusOutput: string = "",
  ) {}
  async log(_args: VcsLogArgs): Promise<string> {
    return this.logOutput;
  }
  async status(): Promise<string> {
    return this.statusOutput;
  }
}

export class AllowAllSandbox implements SandboxProvider {
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    return null;
  }
}

export class ScriptedSandbox implements SandboxProvider {
  private readonly outcomes: (SandboxViolation | null)[] = [];
  push(outcome: SandboxViolation | null): this {
    this.outcomes.push(outcome);
    return this;
  }
  async validate(_call: ToolCall): Promise<SandboxViolation | null> {
    if (this.outcomes.length === 0) return null;
    return this.outcomes.shift() ?? null;
  }
}

export class ScriptedToolRegistry implements ToolRegistry {
  private readonly outputs: ToolOutput[] = [];
  private readonly haltNames = new Set<string>();
  callCount = 0;

  push(output: ToolOutput): this {
    this.outputs.push(output);
    return this;
  }
  markAlwaysHalt(name: string): this {
    this.haltNames.add(name);
    return this;
  }
  isAlwaysHalt(toolName: string): boolean {
    return this.haltNames.has(toolName);
  }
  schemas(): ToolSchema[] {
    return [];
  }
  async dispatch(_call: ToolCall): Promise<ToolOutput> {
    this.callCount += 1;
    const next = this.outputs.shift();
    return next ?? { kind: "success", content: "ok" };
  }
}

export class AlwaysContinuePolicy implements TerminationPolicy {
  async evaluate(): Promise<TerminationDecision> {
    return { kind: "continue" };
  }
}

export class ScriptedTerminationPolicy implements TerminationPolicy {
  private readonly decisions: TerminationDecision[] = [];
  push(d: TerminationDecision): this {
    this.decisions.push(d);
    return this;
  }
  async evaluate(): Promise<TerminationDecision> {
    return this.decisions.shift() ?? { kind: "continue" };
  }
}

export class ScriptedMiddleware implements MiddlewareChain {
  private readonly decisions: [HookPoint, MiddlewareDecision][] = [];
  push(hook: HookPoint, d: MiddlewareDecision): this {
    this.decisions.push([hook, d]);
    return this;
  }
  async fire(hook: HookPoint, _session: SessionState): Promise<MiddlewareDecision> {
    const front = this.decisions[0];
    if (front && front[0] === hook) {
      this.decisions.shift();
      return front[1];
    }
    return { kind: "continue" };
  }
}
