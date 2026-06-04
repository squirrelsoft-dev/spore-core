/**
 * SubagentTool — wraps a child {@link Harness} and exposes it as a {@link Tool}.
 *
 * Subagents cannot spawn their own subagents — enforced at construction time
 * by inspecting the child's {@link ToolRegistry} via `hasSubagentTools()`.
 *
 * ## Mid-loop consult mediation — seam A1 (issue #114)
 *
 * This is the ORCHESTRATOR side of the consult primitive (the worker side and
 * the type/resume seams live in `@spore/core`). {@link SubagentTool.execute}
 * drives the FULL consult cycle internally:
 *
 *   1. It runs the child worker.
 *   2. On a child {@link RunResult} `consult` it does the mediation ITSELF — it
 *      never bubbles the consult to the parent orchestrator's model. It routes
 *      by `request.kind` to the matching {@link ConsultHandlerEntry} in its
 *      `consultHandlers` map, checks the per-kind budget, runs the handler
 *      harness on the request, builds a {@link ConsultResponse} `answer` from the
 *      handler's output, and calls `child.resumeConsult(..)` to continue the
 *      worker.
 *   3. It repeats until the child reaches a terminal result, then returns the
 *      appropriate terminal {@link ToolOutput} to the parent.
 *
 * Rules enforced here: R2 (mediate, do not bubble), R3 (route by kind, no parent
 * model, parent sees success), R4 (per-kind budget), R5a (`soft_fail` overflow →
 * `budget_exhausted` resume), R5b (`escalate_to_human` overflow →
 * `waiting_for_human`), R6 (no matching kind → `escalate`), R7 (depth-1: the
 * handler is the orchestrator's direct child, run via `handler.run(..)`).
 *
 * The handlers reach this tool through {@link SubagentToolConfig.consultHandlers}
 * (the orchestrator builds them from its `HarnessConfig.consultHandlers`).
 */

import type {
  ChildPausedState,
  ConsultHandlerMap,
  ConsultRequest,
  ConsultResponse,
  Harness,
  HumanRequest,
  PausedState,
  RunResult,
  SandboxProvider,
  SessionState,
  StreamSink,
  Task,
  ToolCall,
  ToolOutput,
} from "@spore/core";
import { newTask, SessionId, type toolRegistry } from "@spore/core";

type Tool = toolRegistry.Tool;
type ToolContext = toolRegistry.ToolContext;
type ToolRegistry = toolRegistry.ToolRegistry;

/** How the subagent inherits / does not inherit context from its parent. */
export type ContextSharing =
  | { kind: "isolated" }
  | { kind: "shared_session"; session_id: SessionId }
  | { kind: "summary_handoff"; summary: string };

/** Build-time errors from {@link SubagentTool}. */
export type BuildError = { kind: "invalid_configuration"; reason: string };

export class SubagentToolBuildError extends Error {
  override readonly name = "SubagentToolBuildError";
  constructor(readonly error: BuildError) {
    super(`subagent build failed: ${error.reason}`);
  }
}

export interface SubagentToolConfig {
  name: string;
  description: string;
  /** JSON schema describing the tool input — typically `{ instruction: string }`. */
  inputSchema: unknown;
  /** Timeout in milliseconds for the wrapped run. */
  timeoutMs: number;
  contextSharing: ContextSharing;
  harness: Harness;
  /** Child harness's tool registry — inspected for depth-1 enforcement. */
  childRegistry: ToolRegistry;
  /**
   * Per-kind consult handlers (issue #114, seam A1). Absent / empty (the
   * default) means consults are NOT mediated here — a child {@link RunResult}
   * `consult` degrades gracefully per R6 (no matching kind → `escalate`).
   * Typically the orchestrator passes a clone of its
   * `HarnessConfig.consultHandlers`.
   */
  consultHandlers?: ConsultHandlerMap;
}

export class SubagentTool implements Tool {
  readonly name: string;
  readonly isSubagentTool = true;
  readonly description: string;
  readonly inputSchema: unknown;
  readonly timeoutMs: number;
  readonly contextSharing: ContextSharing;
  private readonly harness: Harness;
  private readonly consultHandlers: ConsultHandlerMap;

  private constructor(cfg: SubagentToolConfig) {
    this.name = cfg.name;
    this.description = cfg.description;
    this.inputSchema = cfg.inputSchema;
    this.timeoutMs = cfg.timeoutMs;
    this.contextSharing = cfg.contextSharing;
    this.harness = cfg.harness;
    this.consultHandlers = cfg.consultHandlers ?? new Map();
  }

  /**
   * Construct a {@link SubagentTool}. Returns a `BuildError` if the child
   * harness's registry already contains a subagent tool (depth-1 rule).
   */
  static build(
    cfg: SubagentToolConfig,
  ): { ok: true; tool: SubagentTool } | { ok: false; error: BuildError } {
    if (cfg.childRegistry.hasSubagentTools()) {
      return {
        ok: false,
        error: {
          kind: "invalid_configuration",
          reason: "child harness must not contain SubagentTool (depth-1 rule)",
        },
      };
    }
    return { ok: true, tool: new SubagentTool(cfg) };
  }

  /** Throwing variant for ergonomic constructor-style usage. */
  static buildOrThrow(cfg: SubagentToolConfig): SubagentTool {
    const r = SubagentTool.build(cfg);
    if (!r.ok) throw new SubagentToolBuildError(r.error);
    return r.tool;
  }

  async execute(
    call: ToolCall,
    _sandbox: SandboxProvider,
    _ctx: ToolContext,
    signal?: AbortSignal,
  ): Promise<ToolOutput> {
    const input = call.input;
    const instruction =
      typeof input === "object" &&
      input !== null &&
      typeof (input as Record<string, unknown>).instruction === "string"
        ? ((input as Record<string, unknown>).instruction as string)
        : null;
    if (instruction == null) {
      return {
        kind: "error",
        message: "invalid parameters: missing `instruction`",
        recoverable: true,
      };
    }

    const { sessionId, seededSession } = resolveSession(this.contextSharing);
    const task: Task = newTask(instruction, sessionId, {
      kind: "re_act",
      max_iterations: 16,
    });

    const runOpts: Parameters<Harness["run"]>[0] = { task };
    if (seededSession) runOpts.session_state = seededSession;
    if (signal) runOpts.signal = signal;
    const fut = this.harness.run(runOpts);

    let timer: NodeJS.Timeout | null = null;
    const timeoutPromise = new Promise<"timeout">((resolve) => {
      timer = setTimeout(() => resolve("timeout"), this.timeoutMs);
    });
    const raced = await Promise.race([fut, timeoutPromise]);
    if (timer) clearTimeout(timer);

    if (raced === "timeout") {
      return {
        kind: "error",
        message: `subagent timed out after ${Math.round(this.timeoutMs / 1000)}s`,
        recoverable: true,
      };
    }

    // Per-kind consult counters (issue #114, R4). Each consult of a given kind
    // decrements its remaining budget; the (budget+1)th triggers the overflow.
    const consultCounts = new Map<string, number>();

    // A1 mediation loop: drive the full consult cycle internally. On a child
    // `consult`, mediate (route → run handler → resume) and continue until the
    // child reaches a terminal result.
    let result = raced;
    for (;;) {
      switch (result.kind) {
        case "success":
          return { kind: "success", content: result.output, truncated: false };
        case "failure":
          return {
            kind: "error",
            message: `subagent failed: ${result.reason.kind}`,
            recoverable: true,
          };
        case "waiting_for_human": {
          const child = childStateFromPaused(result.state, call.id);
          return {
            kind: "waiting_for_human",
            child_state: child,
            request: result.request,
          };
        }
        // A child harness escalation (#80) re-escalates through the parent's
        // dispatch: the parent loop recognizes `escalate` and terminates cleanly,
        // propagating the same signal up. Mirrors the Rust subagent tool.
        case "escalate":
          return { kind: "escalate", signal: result.signal };
        // Mid-loop consult (issue #114, R2): mediate it here — never bubble it
        // to the parent orchestrator's model.
        case "consult": {
          const outcome = await this.mediateConsult(
            result.state,
            result.request,
            consultCounts,
            call.id,
            signal,
          );
          if (outcome.kind === "resume") {
            // The handler answered (or soft-failed): loop again on the new
            // result (R3/R5a).
            result = outcome.next;
            continue;
          }
          // Terminal mapping surfaced to the parent (R5b/R6).
          return outcome.output;
        }
      }
    }
  }

  /**
   * Mediate one child consult (issue #114, seam A1). Routes by `kind`, enforces
   * the per-kind budget, runs the handler as the ORCHESTRATOR's direct child
   * (R7), and resumes the worker — OR applies the overflow policy / graceful
   * degradation.
   */
  private async mediateConsult(
    state: PausedState,
    request: ConsultRequest,
    counts: Map<string, number>,
    parentCallId: string,
    signal?: AbortSignal,
  ): Promise<MediateOutcome> {
    // R6: no matching handler (empty map or unknown kind) → escalate. Loud, not
    // silent. The parent harness terminates cleanly.
    const entry = this.consultHandlers.get(request.kind);
    if (entry == null) {
      return {
        kind: "terminal",
        output: {
          kind: "escalate",
          signal: {
            kind: "abort",
            reason: `no consult handler registered for kind "${request.kind}"`,
          },
        },
      };
    }

    // R4: per-kind budget. `used` is the number of consults of this kind ALREADY
    // mediated. The handler runs while `used < budget`; the (budget+1)th
    // consult overflows.
    const used = counts.get(request.kind) ?? 0;
    if (used >= entry.budget) {
      // R5: overflow policy.
      if (entry.overflow.kind === "soft_fail") {
        // R5a: resume the worker with a budget_exhausted response so it finishes
        // with what it has.
        const response: ConsultResponse = {
          kind: "budget_exhausted",
          message: `consult budget for kind "${request.kind}" exhausted; proceed without further help`,
        };
        return {
          kind: "resume",
          next: await this.resumeChild(state, response, signal),
        };
      }
      // R5b: convert the over-budget consult into a human pause so the host
      // decides. The parent sees waiting_for_human.
      const child = childStateFromPaused(state, parentCallId);
      const humanRequest: HumanRequest = {
        kind: "review",
        content: `consult budget for kind "${request.kind}" exhausted. situation: ${request.situation} | question: ${request.question}`,
      };
      return {
        kind: "terminal",
        output: {
          kind: "waiting_for_human",
          child_state: child,
          request: humanRequest,
        },
      };
    }

    // R3/R7: run the handler harness as the orchestrator's direct child
    // (depth-1), WITHOUT the orchestrator model. The handler's instruction is
    // the consult request rendered to text.
    counts.set(request.kind, used + 1);
    const instruction = renderConsultInstruction(request);
    const task: Task = newTask(instruction, SessionId.generate(), {
      kind: "re_act",
      max_iterations: 16,
    });
    const runOpts: Parameters<Harness["run"]>[0] = { task };
    if (signal) runOpts.signal = signal;
    const handlerResult = await entry.handler.run(runOpts);
    // A handler that does not cleanly complete still must not stall the worker —
    // feed its outcome back as the consult answer so the worker can adapt. (The
    // orchestrator model is never involved.)
    const answer =
      handlerResult.kind === "success"
        ? handlerResult.output
        : `consult handler did not complete cleanly: ${handlerResult.kind}`;
    const response: ConsultResponse = { kind: "answer", text: answer };
    return {
      kind: "resume",
      next: await this.resumeChild(state, response, signal),
    };
  }

  /**
   * Resume the paused worker with a {@link ConsultResponse} (issue #114). The
   * child harness's `resumeConsult` is optional on the {@link Harness} interface;
   * a child that does not implement it cannot be resumed mid-consult, so this
   * treats absence as a recoverable failure surfaced as a tool error on the next
   * loop turn.
   */
  private async resumeChild(
    state: PausedState,
    response: ConsultResponse,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    if (this.harness.resumeConsult == null) {
      return {
        kind: "failure",
        reason: { kind: "human_halted" },
        session_id: state.session_id,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_read_tokens: 0,
          cache_write_tokens: 0,
          cost_usd: 0,
        },
        turns: state.turn_number,
        session_state: state.session_state,
      };
    }
    const onStream: StreamSink | undefined = undefined;
    return this.harness.resumeConsult(state, response, onStream, signal);
  }
}

/** Outcome of one mediation step (issue #114). */
type MediateOutcome =
  | { kind: "resume"; next: RunResult }
  | { kind: "terminal"; output: ToolOutput };

/** Render a {@link ConsultRequest} to a handler instruction string (issue #114). */
function renderConsultInstruction(request: ConsultRequest): string {
  return (
    `A worker agent is requesting help (kind: ${request.kind}).\n\n` +
    `Situation: ${request.situation}\n\n` +
    `Attempts so far: ${request.attempts}\n\n` +
    `Question: ${request.question}`
  );
}

function resolveSession(c: ContextSharing): {
  sessionId: SessionId;
  seededSession: SessionState | null;
} {
  switch (c.kind) {
    case "isolated":
      return { sessionId: SessionId.generate(), seededSession: null };
    case "shared_session":
      return { sessionId: c.session_id, seededSession: null };
    case "summary_handoff": {
      const state: SessionState = {
        messages: [],
        extras: { subagent_handoff_summary: c.summary },
      };
      return { sessionId: SessionId.generate(), seededSession: state };
    }
  }
}

function childStateFromPaused(
  state: PausedState,
  parentToolCallId: string,
): ChildPausedState {
  return {
    session_id: state.session_id,
    task_id: state.task_id,
    turn_number: state.turn_number,
    session_state: state.session_state,
    pending_tool_calls: state.pending_tool_calls,
    approved_results: state.approved_results,
    human_request: state.human_request,
    task: state.task,
    budget_used: state.budget_used,
    parent_tool_call_id: parentToolCallId,
  };
}
