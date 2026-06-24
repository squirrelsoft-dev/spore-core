/**
 * Test-only stubs of the sibling traits. Covers ReAct paths in unit tests
 * before the canonical impls land in #4–#13.
 */

import type { Context } from "../agent/types.js";
import type { ContextSources } from "../context/types.js";
import type { Agent } from "../agent/interface.js";
import type { ToolCall, ToolResult, ToolSchema } from "../model/schemas.js";
import type { Verifier } from "../verifier/types.js";
import type { MetricEvaluator } from "../metric/types.js";
import type { Middleware } from "../middleware/types.js";

import type { VcsLogArgs, VcsProvider } from "./vcs.js";
import { ExecutionRegistry } from "./execution-registry.js";
import { EmptyToolRegistry } from "../tool-registry/empty.js";
import type {
  ContextManager,
  HookPoint,
  MiddlewareChain,
  MiddlewareDecision,
  RunResult,
  SandboxProvider,
  SandboxViolation,
  SessionId,
  SessionState,
  Task,
  TerminationDecision,
  TerminationPolicy,
  ToolOutput,
  ToolRegistry,
  ToolResultRecord,
} from "./types.js";

/**
 * #124 test helper: build an {@link ExecutionRegistry} that registers the given
 * collaborators under the DEFAULT empty-string key — exactly what
 * `HarnessBuilder.buildConfig` folds in. Tests that construct a `HarnessConfig`
 * literally (no builder) use this to supply the default agent / toolset /
 * verifier / metric evaluator that bare `reactPerLoop` leaves and the default
 * SelfVerifying / HillClimbing evaluator handles resolve to.
 */
export function registryWith(opts: {
  agent?: Agent;
  toolset?: ToolRegistry;
  verifier?: Verifier;
  metricEvaluator?: MetricEvaluator;
}): ExecutionRegistry {
  let b = ExecutionRegistry.builder();
  if (opts.agent) b = b.agent("", opts.agent);
  // Bare `reactPerLoop` leaves carry an empty `ToolsetRef`; validation requires
  // it to resolve. Register a default empty toolset under "" unless overridden.
  b = b.toolset("", opts.toolset ?? new EmptyToolRegistry());
  // A.5 structured-slot leaves (`plan` / `worker` / `propose`) declare an output
  // schema; register a default schema under "" so a leaf with `output: ""`
  // resolves under validation.
  b = b.schema("", {});
  if (opts.verifier) b = b.verifier("", opts.verifier);
  if (opts.metricEvaluator) b = b.metricEvaluator("", opts.metricEvaluator);
  return b.build();
}

export class NoopContextManager implements ContextManager {
  async assemble(session: SessionState, _task: Task, _sources: ContextSources): Promise<Context> {
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
      case "consult":
        // Never reached at runtime: the harness pauses (RunResult.consult)
        // before appending; present for exhaustiveness (#114).
        text = "[consult]";
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
  private readonly toolSchemas: ToolSchema[] = [];
  /** When set, EVERY dispatch returns this recoverable error (#137). */
  private alwaysError?: string;
  callCount = 0;

  push(output: ToolOutput): this {
    this.outputs.push(output);
    return this;
  }
  markAlwaysHalt(name: string): this {
    this.haltNames.add(name);
    return this;
  }
  /**
   * Make EVERY dispatch return the same recoverable {@link ToolOutput} `error`
   * with `message`, regardless of args (#137 — the gemma
   * `add_task`-without-`description` grinding scenario). Mirrors Rust's
   * `ScriptedToolRegistry::always_recoverable_error`.
   */
  alwaysRecoverableError(message: string): this {
    this.alwaysError = message;
    return this;
  }
  /** Advertise a tool schema so `enrichToolError` can render it (#137). Mirrors
   *  Rust's `ScriptedToolRegistry::with_schema`. */
  withSchema(schema: ToolSchema): this {
    this.toolSchemas.push(schema);
    return this;
  }
  isAlwaysHalt(toolName: string): boolean {
    return this.haltNames.has(toolName);
  }
  schemas(): ToolSchema[] {
    return this.toolSchemas;
  }
  async dispatch(_call: ToolCall): Promise<ToolOutput> {
    this.callCount += 1;
    if (this.alwaysError !== undefined) {
      return { kind: "error", message: this.alwaysError, recoverable: true };
    }
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

/**
 * Scripted {@link MiddlewareChain} test double (rich surface, issue #11).
 * Decisions are queued per hook via {@link ScriptedMiddleware.push}; each
 * `fire*` method pops the front entry if it targets that hook, else returns
 * `continue`. Unlike {@link StandardMiddlewareChain}, scripted decisions are
 * returned RAW (no `validateDecision`), so a test can exercise the harness's
 * defensive handling of an out-of-place decision. The `.push(hook, decision)`
 * API is stable across the stub→rich migration.
 */
export class ScriptedMiddleware implements MiddlewareChain {
  private readonly decisions: [HookPoint, MiddlewareDecision][] = [];

  push(hook: HookPoint, d: MiddlewareDecision): this {
    this.decisions.push([hook, d]);
    return this;
  }

  /** Pop and return the scripted decision for `hook` if it is at the front of
   *  the queue, else `continue`. */
  private nextFor(hook: HookPoint): MiddlewareDecision {
    const front = this.decisions[0];
    if (front && front[0] === hook) {
      this.decisions.shift();
      return front[1];
    }
    return { kind: "continue" };
  }

  // The double scripts decisions directly; registration is a no-op.
  register(_middleware: Middleware): void {
    void _middleware;
  }

  async fireBeforeSession(_task: Task, _sessionId: SessionId): Promise<MiddlewareDecision> {
    return this.nextFor("before_session");
  }

  async fireBeforeTurn(_session: SessionState, _turnNumber: number): Promise<MiddlewareDecision> {
    return this.nextFor("before_turn");
  }

  async fireBeforeTool(_calls: ToolCall[], _turnNumber: number): Promise<MiddlewareDecision> {
    return this.nextFor("before_tool");
  }

  async fireAfterTool(_calls: ToolCall[], _results: ToolResult[]): Promise<MiddlewareDecision> {
    return this.nextFor("after_tool");
  }

  async fireBeforeCompletion(
    _response: string,
    _turnNumber: number,
    _state: SessionState,
  ): Promise<MiddlewareDecision> {
    return this.nextFor("before_completion");
  }

  async fireAfterSession(_result: RunResult, _sessionId: SessionId): Promise<void> {
    // After hooks ignore the decision; drain a scripted after_session entry.
    void this.nextFor("after_session");
  }
}
