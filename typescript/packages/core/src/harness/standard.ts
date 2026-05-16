/**
 * `StandardHarness` — canonical implementation of the harness runtime loop.
 *
 * ## Rules enforced here
 *
 * 1. Harness owns the loop — the agent executes one turn at a time.
 * 2. Termination is evaluated against external state via TerminationPolicy.
 * 3. Any budget overrun terminates the loop with an explicit HaltReason.
 * 4. A turn that yields neither tool call nor final response is an error.
 * 5. All components are injected at construction.
 * 6. Stateless between pause and resume — caller owns PausedState.
 * 7. WaitingForHuman returns immediately; no internal timeout.
 * 8. `approved_results` prevents double-execution on resume.
 * 9. Subagents cannot spawn subagents (depth-1 enforcement in types).
 */

import type { Agent } from "../agent/interface.js";

import type { Harness } from "./interface.js";
import {
  addTurnUsage,
  emptyAggregateUsage,
  emptyBudgetSnapshot,
  emptySessionState,
  type AggregateUsage,
  type BudgetLimitType,
  type BudgetLimits,
  type BudgetSnapshot,
  type ChildPausedState,
  type ContextManager,
  type HaltReason,
  type HarnessRunOptions,
  type HookPoint,
  type HumanResponse,
  type LoopStrategy,
  type MiddlewareChain,
  type ObservabilityProvider,
  type PausedState,
  type RunResult,
  type SandboxProvider,
  type SessionId,
  type SessionState,
  type StreamSink,
  type Task,
  type TerminationPolicy,
  type ToolOutput,
  type ToolRegistry,
  type ToolResultRecord,
} from "./types.js";

/** Components injected at construction. Mirrors `HarnessConfig` in the spec. */
export interface HarnessConfig {
  agent: Agent;
  toolRegistry: ToolRegistry;
  sandbox: SandboxProvider;
  contextManager: ContextManager;
  terminationPolicy: TerminationPolicy;
  middleware?: MiddlewareChain;
  observability?: ObservabilityProvider;
}

export class StandardHarness implements Harness {
  constructor(private readonly config: HarnessConfig) {}

  async run(options: HarnessRunOptions): Promise<RunResult> {
    const sessionState = options.session_state ?? emptySessionState();
    const budgetUsed = emptyBudgetSnapshot();
    const task = options.task;

    switch (task.loop_strategy.kind) {
      case "re_act":
        return this.runReact(
          task,
          task.loop_strategy.max_iterations,
          sessionState,
          budgetUsed,
          options.on_stream,
          options.signal,
        );
      case "plan_execute":
        return notYetImplemented("plan_execute", task.session_id);
      case "ralph":
        return notYetImplemented("ralph", task.session_id);
      case "self_verifying":
        return notYetImplemented("self_verifying", task.session_id);
      case "hill_climbing":
        return notYetImplemented("hill_climbing", task.session_id);
      default: {
        const _exhaustive: never = task.loop_strategy;
        return _exhaustive;
      }
    }
  }

  async resume(
    state: PausedState,
    response: HumanResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    const sessionState = state.session_state;
    const pendingCalls = state.pending_tool_calls;

    // Subagent depth: if there's a child, the caller-installed SubagentTool
    // owns the dispatch back into the child harness; without #4/#5 wired up
    // the harness round-trips the child state but continues the parent loop.
    // This matches the Rust reference (placeholder until #4/#5 land).
    if (state.child_state != null) {
      // Intentional no-op; the full child.resume() dispatch lives in #4/#5.
    }

    switch (response.kind) {
      case "halt":
        return {
          kind: "failure",
          reason: { kind: "human_halted" },
          session_id: state.session_id,
          usage: emptyAggregateUsage(),
          turns: state.turn_number,
        };

      case "deny": {
        const reason = response.reason;
        for (const call of pendingCalls) {
          const tr: ToolResultRecord = {
            call_id: call.id,
            output: { kind: "error", message: reason, recoverable: true },
          };
          await this.config.contextManager.appendToolResult(sessionState, tr);
        }
        break;
      }

      case "reject":
        await this.config.contextManager.appendUserMessage(sessionState, response.reason);
        break;

      case "answer":
        await this.config.contextManager.appendUserMessage(sessionState, response.text);
        break;

      case "approve_with_feedback":
        await this.config.contextManager.appendUserMessage(sessionState, response.feedback);
        break;

      case "allow": {
        for (const call of pendingCalls) {
          const output = await this.config.toolRegistry.dispatch(call, signal);
          const tr: ToolResultRecord = { call_id: call.id, output };
          await this.config.contextManager.appendToolResult(sessionState, tr);
        }
        break;
      }

      case "allow_with_modification": {
        for (const call of response.calls) {
          const output = await this.config.toolRegistry.dispatch(call, signal);
          const tr: ToolResultRecord = { call_id: call.id, output };
          await this.config.contextManager.appendToolResult(sessionState, tr);
        }
        break;
      }

      default: {
        const _exhaustive: never = response;
        return _exhaustive;
      }
    }

    const max =
      state.task.loop_strategy.kind === "re_act"
        ? state.task.loop_strategy.max_iterations
        : Number.MAX_SAFE_INTEGER;
    return this.runReact(state.task, max, sessionState, state.budget_used, onStream, signal);
  }

  // --------------------------------------------------------------------------
  // ReAct loop
  // --------------------------------------------------------------------------

  private async runReact(
    task: Task,
    maxIterations: number,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    const startedAt = Date.now();
    const usage: AggregateUsage = emptyAggregateUsage();
    const taskMaxTurns = task.budget.max_turns ?? undefined;
    const effectiveTurnCap =
      taskMaxTurns != null ? Math.min(taskMaxTurns, maxIterations) : maxIterations;

    // Outer loop
    for (;;) {
      // Layer-1 budget gates before the turn.
      if (budgetUsed.turns >= effectiveTurnCap) {
        return failure(
          { kind: "budget_exceeded", limit_type: "turns" },
          sessionId,
          usage,
          budgetUsed.turns,
        );
      }
      const overrun = budgetExceeded(task.budget, budgetUsed, startedAt);
      if (overrun != null) {
        return failure(
          { kind: "budget_exceeded", limit_type: overrun },
          sessionId,
          usage,
          budgetUsed.turns,
        );
      }

      // Middleware: BeforeTurn
      if (this.config.middleware) {
        const decision = await this.config.middleware.fire("before_turn", sessionState);
        switch (decision.kind) {
          case "continue":
          case "continue_with_modification":
            break;
          case "halt":
            return failure(
              { kind: "middleware_halt", hook: "before_turn", reason: decision.reason },
              sessionId,
              usage,
              budgetUsed.turns,
            );
          case "surface_to_human": {
            const ps: PausedState = {
              session_id: sessionId,
              task_id: task.id,
              turn_number: budgetUsed.turns,
              session_state: sessionState,
              pending_tool_calls: [],
              approved_results: [],
              human_request: decision.request,
              task,
              budget_used: budgetUsed,
              child_state: null,
            };
            return { kind: "waiting_for_human", state: ps, request: decision.request };
          }
          default: {
            const _exhaustive: never = decision;
            return _exhaustive;
          }
        }
      }

      // Assemble + invoke agent for one turn.
      const context = await this.config.contextManager.assemble(sessionState, task, signal);
      emit(onStream, { kind: "turn_start", turn: budgetUsed.turns + 1 });
      const result = await this.config.agent.turn(context, signal);
      budgetUsed.turns += 1;

      // Record turn usage in observability (zero usage for typed error w/o usage).
      if (this.config.observability) {
        const zero = { input_tokens: 0, output_tokens: 0 };
        const u =
          result.kind === "tool_call_requested" || result.kind === "final_response"
            ? result.usage
            : (result.usage ?? zero);
        await this.config.observability.recordTurn(budgetUsed.turns, u);
      }
      emit(onStream, { kind: "turn_end", turn: budgetUsed.turns });

      switch (result.kind) {
        case "final_response": {
          addTurnUsage(usage, result.usage);
          budgetUsed.input_tokens += result.usage.input_tokens;
          budgetUsed.output_tokens += result.usage.output_tokens;

          // Middleware: BeforeCompletion
          if (this.config.middleware) {
            const d = await this.config.middleware.fire("before_completion", sessionState);
            switch (d.kind) {
              case "continue":
              case "continue_with_modification":
                break;
              case "halt":
                return failure(
                  { kind: "middleware_halt", hook: "before_completion", reason: d.reason },
                  sessionId,
                  usage,
                  budgetUsed.turns,
                );
              case "surface_to_human": {
                const ps: PausedState = {
                  session_id: sessionId,
                  task_id: task.id,
                  turn_number: budgetUsed.turns,
                  session_state: sessionState,
                  pending_tool_calls: [],
                  approved_results: [],
                  human_request: d.request,
                  task,
                  budget_used: budgetUsed,
                  child_state: null,
                };
                return { kind: "waiting_for_human", state: ps, request: d.request };
              }
              default: {
                const _exhaustive: never = d;
                return _exhaustive;
              }
            }
          }

          // Termination policy
          const decision = await this.config.terminationPolicy.evaluate(sessionState, budgetUsed);
          if (decision.kind === "halt") {
            return failure(
              { kind: "termination_policy_halt", reason: decision.reason },
              sessionId,
              usage,
              budgetUsed.turns,
            );
          }

          emit(onStream, { kind: "final_response", content: result.content });
          return {
            kind: "success",
            output: result.content,
            session_id: sessionId,
            usage,
            turns: budgetUsed.turns,
          };
        }

        case "tool_call_requested": {
          addTurnUsage(usage, result.usage);
          budgetUsed.input_tokens += result.usage.input_tokens;
          budgetUsed.output_tokens += result.usage.output_tokens;

          // Always-halt short-circuit (Layer 1).
          const haltingTool = result.calls.find((c) =>
            this.config.toolRegistry.isAlwaysHalt(c.name),
          );
          if (haltingTool) {
            return failure(
              {
                kind: "unrecoverable_tool_error",
                tool: haltingTool.name,
                error: "tool is annotated always_halt",
              },
              sessionId,
              usage,
              budgetUsed.turns,
            );
          }

          // Middleware: BeforeTool
          let calls = result.calls;
          if (this.config.middleware) {
            const d = await this.config.middleware.fire("before_tool", sessionState);
            switch (d.kind) {
              case "continue":
                break;
              case "continue_with_modification":
                calls = d.calls;
                break;
              case "halt":
                return failure(
                  { kind: "middleware_halt", hook: "before_tool", reason: d.reason },
                  sessionId,
                  usage,
                  budgetUsed.turns,
                );
              case "surface_to_human": {
                const ps: PausedState = {
                  session_id: sessionId,
                  task_id: task.id,
                  turn_number: budgetUsed.turns,
                  session_state: sessionState,
                  pending_tool_calls: calls,
                  approved_results: [],
                  human_request: d.request,
                  task,
                  budget_used: budgetUsed,
                  child_state: null,
                };
                return { kind: "waiting_for_human", state: ps, request: d.request };
              }
              default: {
                const _exhaustive: never = d;
                return _exhaustive;
              }
            }
          }

          const approvedResults: ToolResultRecord[] = [];
          for (let i = 0; i < calls.length; i++) {
            const call = calls[i]!;
            // Sandbox validation
            const violation = await this.config.sandbox.validate(call, signal);
            if (violation != null) {
              if (violation.kind === "path_escape" || violation.kind === "network_violation") {
                return failure(
                  { kind: "sandbox_violation", violation },
                  sessionId,
                  usage,
                  budgetUsed.turns,
                );
              }
              // Layer-2 default: recoverable — append as tool error.
              const tr: ToolResultRecord = {
                call_id: call.id,
                output: {
                  kind: "error",
                  message: `sandbox: ${violation.kind}`,
                  recoverable: true,
                },
              };
              emit(onStream, { kind: "tool_result", call_id: call.id, is_error: true });
              await this.config.contextManager.appendToolResult(sessionState, tr);
              approvedResults.push(tr);
              continue;
            }

            emit(onStream, { kind: "tool_call", call_id: call.id, name: call.name });
            const output: ToolOutput = await this.config.toolRegistry.dispatch(call, signal);

            // WaitingForHuman from subagent tool
            if (output.kind === "waiting_for_human") {
              const remaining = calls.slice(i + 1);
              const child: ChildPausedState = output.child_state;
              const ps: PausedState = {
                session_id: sessionId,
                task_id: task.id,
                turn_number: budgetUsed.turns,
                session_state: sessionState,
                pending_tool_calls: remaining,
                approved_results: approvedResults,
                human_request: output.request,
                task,
                budget_used: budgetUsed,
                child_state: child,
              };
              return { kind: "waiting_for_human", state: ps, request: output.request };
            }

            // Layer-2: unrecoverable tool error halts immediately.
            if (output.kind === "error" && !output.recoverable) {
              return failure(
                {
                  kind: "unrecoverable_tool_error",
                  tool: call.name,
                  error: output.message,
                },
                sessionId,
                usage,
                budgetUsed.turns,
              );
            }

            const isError = output.kind === "error";
            const tr: ToolResultRecord = { call_id: call.id, output };
            emit(onStream, { kind: "tool_result", call_id: call.id, is_error: isError });
            await this.config.contextManager.appendToolResult(sessionState, tr);
            approvedResults.push(tr);
          }

          // Middleware: AfterTool
          if (this.config.middleware) {
            const d = await this.config.middleware.fire("after_tool", sessionState);
            if (d.kind === "halt") {
              return failure(
                { kind: "middleware_halt", hook: "after_tool", reason: d.reason },
                sessionId,
                usage,
                budgetUsed.turns,
              );
            }
          }

          continue;
        }

        case "error": {
          if (result.usage != null) {
            addTurnUsage(usage, result.usage);
            budgetUsed.input_tokens += result.usage.input_tokens;
            budgetUsed.output_tokens += result.usage.output_tokens;
          }
          return failure(
            { kind: "agent_error", error: result.error },
            sessionId,
            usage,
            budgetUsed.turns,
          );
        }

        default: {
          const _exhaustive: never = result;
          return _exhaustive;
        }
      }
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

function emit(sink: StreamSink | undefined, event: Parameters<StreamSink>[0]): void {
  if (sink) sink(event);
}

function failure(
  reason: HaltReason,
  sessionId: SessionId,
  usage: AggregateUsage,
  turns: number,
): RunResult {
  return { kind: "failure", reason, session_id: sessionId, usage, turns };
}

function notYetImplemented(strategy: string, sessionId: SessionId): RunResult {
  return {
    kind: "failure",
    reason: { kind: "strategy_not_yet_implemented", strategy },
    session_id: sessionId,
    usage: emptyAggregateUsage(),
    turns: 0,
  };
}

function budgetExceeded(
  budget: BudgetLimits,
  used: BudgetSnapshot,
  startedAt: number,
): BudgetLimitType | null {
  if (budget.max_turns != null && used.turns >= budget.max_turns) return "turns";
  if (budget.max_input_tokens != null && used.input_tokens > budget.max_input_tokens) {
    return "input_tokens";
  }
  if (budget.max_output_tokens != null && used.output_tokens > budget.max_output_tokens) {
    return "output_tokens";
  }
  if (budget.max_wall_time != null) {
    const elapsedSec = (Date.now() - startedAt) / 1000;
    if (elapsedSec >= budget.max_wall_time) return "wall_time";
  }
  if (budget.max_cost_usd != null && used.cost_usd > budget.max_cost_usd) return "cost_usd";
  return null;
}

// Hook point set (helper for completeness, not required by the loop).
export const HOOK_POINTS: readonly HookPoint[] = [
  "before_turn",
  "before_tool",
  "after_tool",
  "before_completion",
];

/** Type marker used by `LoopStrategy` consumers to detect non-ReAct strategies. */
export function isReact(strategy: LoopStrategy): boolean {
  return strategy.kind === "re_act";
}
