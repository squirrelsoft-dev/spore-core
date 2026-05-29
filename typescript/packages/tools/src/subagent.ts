/**
 * SubagentTool — wraps a child {@link Harness} and exposes it as a {@link Tool}.
 *
 * Subagents cannot spawn their own subagents — enforced at construction time
 * by inspecting the child's {@link ToolRegistry} via `hasSubagentTools()`.
 */

import type {
  ChildPausedState,
  Harness,
  PausedState,
  SandboxProvider,
  SessionState,
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
}

export class SubagentTool implements Tool {
  readonly name: string;
  readonly isSubagentTool = true;
  readonly description: string;
  readonly inputSchema: unknown;
  readonly timeoutMs: number;
  readonly contextSharing: ContextSharing;
  private readonly harness: Harness;

  private constructor(cfg: SubagentToolConfig) {
    this.name = cfg.name;
    this.description = cfg.description;
    this.inputSchema = cfg.inputSchema;
    this.timeoutMs = cfg.timeoutMs;
    this.contextSharing = cfg.contextSharing;
    this.harness = cfg.harness;
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

    const result = raced;
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
    }
  }
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
