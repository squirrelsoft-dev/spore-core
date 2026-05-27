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
import type { Context } from "../agent/types.js";
import type { SessionOutcome } from "../guide-registry/types.js";
import { Timestamp } from "../memory/types.js";
import type { StopReason, TokenUsage } from "../model/schemas.js";
import {
  ContentCaptureConfig,
  PricingTable,
  SpanId,
  finishSpanBase,
  newRootSpanBase,
  newChildSpanBase,
  newWarnSpan,
  truncateField,
  type ContextSpan,
  type GenAiMessage,
  type SpanBase,
  type SpanStatus,
  type ToolCallContent,
  type ToolCallSpan,
  type ToolResultContent,
  type TurnSpan,
  type WarnEvent,
} from "../observability/types.js";
import type { Message, ToolCall } from "../model/schemas.js";
import type { GenAiRole } from "../observability/types.js";
import { OutboxObservabilityProvider, outboxConfig } from "../observability/outbox.js";
import { KeyTermVerifier, type CompactionVerifier } from "../context/types.js";

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
  type CompactionTurn,
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
  type TaskId,
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
  /**
   * Post-compaction verifier (issue #29/#46). The harness runs it after each
   * compaction turn and retries up to `maxCompactionAttempts` before accepting
   * a failing summary. Optional; defaults to {@link KeyTermVerifier}.
   */
  compactionVerifier?: CompactionVerifier;
  /**
   * Maximum compaction-summary attempts before accepting a failing summary
   * anyway (issue #46). Optional; defaults to `2`. Clamped to a minimum of `1`.
   */
  maxCompactionAttempts?: number;
  /**
   * Token → USD pricing used to stamp `cost_usd` on emitted {@link TurnSpan}s.
   * Optional; the loop falls back to {@link PricingTable.DEFAULT} (zero cost)
   * when unset, mirroring Rust's `pricing: PricingTable::DEFAULT`.
   */
  pricing?: PricingTable;
  /**
   * LLM-native content capture (issue #64). Gates whether the turn/tool-call
   * spans carry `gen_ai.*` conversation/tool content. Optional; defaults to
   * {@link ContentCaptureConfig.default} (OFF) so the durable JSONL stays
   * byte-identical to the pre-#64 output. Use {@link ContentCaptureConfig.fromEnv}
   * to honor `SPORE_TRACE_CONTENT` / `SPORE_TRACE_CONTENT_MAX_LEN`.
   */
  contentCapture?: ContentCaptureConfig;
}

export class StandardHarness implements Harness {
  constructor(private readonly config: HarnessConfig) {}

  /**
   * Capture a requested tool call's arguments (issue #64). When the serialized
   * arguments exceed the byte budget, the clipped marker-bearing string is
   * stored as the `arguments` value; otherwise the raw input is preserved.
   * Mirrors Rust's `capture_tool_call_args`.
   */
  private static captureToolCallArgs(call: ToolCall, max: number): ToolCallContent {
    const serialized = JSON.stringify(call.input ?? null);
    const [clipped, truncated] = truncateField(serialized, max);
    return {
      name: call.name,
      arguments: truncated ? clipped : call.input,
      arguments_truncated: truncated,
    };
  }

  /**
   * Snapshot the assembled INPUT messages (the full prompt the model saw) into
   * {@link GenAiMessage}s for LLM-native tracing (issue #64). Each message's
   * role maps to the conventional {@link GenAiRole}; the content is rendered to
   * a plain string and truncated to `max` bytes:
   *   - text        → the text verbatim
   *   - tool_result → its result body (role stays `tool`)
   *   - tool_call   → `"<name> <compact-json-args>"` (assistant)
   *   - image       → `"[image <media_type>]"` — NEVER the base64 data
   *
   * System-first, then history order is preserved because the assembled
   * `messages` already lead with the `system` prompt. Mirrors Rust's
   * `capture_input_messages`.
   */
  private static captureInputMessages(messages: Message[], max: number): GenAiMessage[] {
    return messages.map((m): GenAiMessage => {
      const role: GenAiRole = m.role;
      let rendered: string;
      switch (m.content.type) {
        case "text":
          rendered = m.content.text;
          break;
        case "tool_result":
          rendered = m.content.content;
          break;
        case "tool_call":
          rendered = `${m.content.name} ${JSON.stringify(m.content.input ?? null)}`;
          break;
        case "image":
          // NEVER dump the base64 `data` — placeholder only.
          rendered = `[image ${m.content.media_type}]`;
          break;
        default: {
          const _exhaustive: never = m.content;
          rendered = String(_exhaustive);
          break;
        }
      }
      const [content, truncated] = truncateField(rendered, max);
      return { role, content, truncated };
    });
  }

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
          true,
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
    return this.runReact(state.task, max, sessionState, state.budget_used, onStream, signal, false);
  }

  // --------------------------------------------------------------------------
  // ReAct loop
  // --------------------------------------------------------------------------

  /**
   * Record the terminal outcome and flush the observability session. Called at
   * every terminal `runReact` outcome (success or any halt) — never on a
   * `WaitingForHuman` pause, which is not terminal. No-op when no provider is
   * configured. Mirrors Rust's `finalize_observability`.
   */
  private async finalizeObservability(
    sessionId: SessionId,
    outcome: SessionOutcome,
  ): Promise<void> {
    const obs = this.config.observability;
    if (obs) {
      obs.setSessionOutcome(sessionId, outcome);
      await obs.flushSession(sessionId);
    }
  }

  /**
   * Drive the ReAct loop, then finalize observability for terminal outcomes. A
   * `WaitingForHuman` pause is not terminal, so it is never flushed here — the
   * eventual `resume` path reaches a terminal outcome and flushes then. Mirrors
   * Rust's `run_react` / `run_react_inner` split.
   */
  private async runReact(
    task: Task,
    maxIterations: number,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
    seedInstruction: boolean,
  ): Promise<RunResult> {
    const result = await this.runReactInner(
      task,
      maxIterations,
      sessionState,
      budgetUsed,
      onStream,
      signal,
      seedInstruction,
    );
    switch (result.kind) {
      case "success":
        await this.finalizeObservability(result.session_id, { kind: "success" });
        break;
      case "failure":
        await this.finalizeObservability(result.session_id, {
          kind: "failure",
          reason: haltReasonToString(result.reason),
        });
        break;
      case "waiting_for_human":
        // Not terminal — do not finalize.
        break;
    }
    return result;
  }

  private async runReactInner(
    task: Task,
    maxIterations: number,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
    seedInstruction: boolean,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    const startedAt = Date.now();
    const usage: AggregateUsage = emptyAggregateUsage();
    const pricing = this.config.pricing ?? PricingTable.DEFAULT;
    // Monotonic per-run span counter for turn / tool-call span ids, and the most
    // recent turn span base — parent for the tool-call spans of that turn.
    let spanSeq = 0;
    let currentTurnBase: SpanBase | undefined;
    const taskMaxTurns = task.budget.max_turns ?? undefined;
    const effectiveTurnCap =
      taskMaxTurns != null ? Math.min(taskMaxTurns, maxIterations) : maxIterations;

    // Seed the task instruction as the initial user message of this run.
    // The compaction adapter intentionally mirrors `session.messages` and
    // ignores `task` on `assemble`, so the harness must own delivering the
    // prompt. On a fresh run this turns an otherwise-empty conversation into a
    // real user turn; on multi-turn runs over a carried `session_state` each
    // `run()` call appends its own follow-up instruction. The resume path does
    // NOT seed (`seedInstruction === false`) — its conversation already exists
    // and `resume` has already appended the human response.
    if (seedInstruction) {
      await this.config.contextManager.appendUserMessage(sessionState, task.instruction);
    }

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
      const turnStartedAt = Timestamp.now();
      const turnClock = Date.now();
      // LLM-native content capture (issue #64): snapshot the assembled INPUT
      // messages (the full prompt the model saw) BEFORE the agent turn. Guard
      // off → no work (and no `input_messages` on the span).
      const ccTurn = this.config.contentCapture ?? ContentCaptureConfig.default();
      const inputMessages: GenAiMessage[] | null = ccTurn.enabled
        ? StandardHarness.captureInputMessages(context.messages, ccTurn.maxFieldLen)
        : null;
      const result = await this.config.agent.turn(context, signal);
      budgetUsed.turns += 1;

      // Emit a turn span for every model call (issue #12). Fire-and-forget; it
      // never affects control flow. The span base is retained as the parent for
      // any tool-call spans dispatched this turn.
      {
        const zero: TokenUsage = { input_tokens: 0, output_tokens: 0 };
        const u =
          result.kind === "tool_call_requested" || result.kind === "final_response"
            ? result.usage
            : (result.usage ?? zero);
        let stopReason: StopReason;
        let toolCallsRequested: number;
        switch (result.kind) {
          case "final_response":
            stopReason = "end_turn";
            toolCallsRequested = 0;
            break;
          case "tool_call_requested":
            stopReason = "tool_use";
            toolCallsRequested = result.calls.length;
            break;
          default:
            stopReason = "end_turn";
            toolCallsRequested = 0;
            break;
        }
        const status: SpanStatus =
          result.kind === "error"
            ? { kind: "error", message: JSON.stringify(result.error) }
            : { kind: "ok" };
        const base = finishSpanBase(
          newRootSpanBase(
            SpanId.of(`${sessionId.asString()}-turn-${budgetUsed.turns}`),
            sessionId,
            task.id,
            "turn",
            turnStartedAt,
          ),
          Timestamp.now(),
          status,
          Date.now() - turnClock,
        );
        if (this.config.observability) {
          // LLM-native content capture (issue #64): output text + requested
          // tool calls, ONLY when the guard is enabled. Decision 4: the turn
          // span carries output + tool calls; no assembled input-message history.
          const cc = this.config.contentCapture ?? ContentCaptureConfig.default();
          let outputText: GenAiMessage | null = null;
          let toolCalls: ToolCallContent[] | null = null;
          if (cc.enabled) {
            if (result.kind === "final_response") {
              const [content, truncated] = truncateField(result.content, cc.maxFieldLen);
              outputText = { role: "assistant", content, truncated };
            } else if (result.kind === "tool_call_requested") {
              toolCalls = result.calls.map((c) =>
                StandardHarness.captureToolCallArgs(c, cc.maxFieldLen),
              );
            }
          }
          const turnSpan: TurnSpan = {
            base,
            turn_number: budgetUsed.turns,
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.cache_read_tokens ?? null,
            cache_write_tokens: u.cache_write_tokens ?? null,
            cost_usd: PricingTable.costFor(
              pricing,
              u.input_tokens,
              u.output_tokens,
              u.cache_read_tokens,
              u.cache_write_tokens,
            ),
            stop_reason: stopReason,
            tool_calls_requested: toolCallsRequested,
            output_text: outputText,
            tool_calls: toolCalls,
            input_messages: inputMessages,
          };
          this.config.observability.emitTurn(turnSpan);
        }
        spanSeq += 1;
        currentTurnBase = base;
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
            const toolStartedAt = Timestamp.now();
            const toolClock = Date.now();
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

            // Tool-call span (issue #12), child of the current turn. Fire-and-forget.
            if (this.config.observability) {
              let outputSizeBytes = 0;
              let truncated = false;
              if (output.kind === "success") {
                outputSizeBytes = output.content.length;
                truncated = output.truncated ?? false;
              } else if (output.kind === "error") {
                outputSizeBytes = output.message.length;
              }
              const spanId = SpanId.of(`${sessionId.asString()}-tool-${spanSeq}`);
              const childBase =
                currentTurnBase != null
                  ? newChildSpanBase(spanId, currentTurnBase, "tool_call", toolStartedAt)
                  : newRootSpanBase(spanId, sessionId, task.id, "tool_call", toolStartedAt);
              const status: SpanStatus = isError
                ? { kind: "error", message: "tool returned a recoverable error" }
                : { kind: "ok" };
              // LLM-native content capture (issue #64): tool args + tool result,
              // ONLY when the guard is enabled.
              const cc = this.config.contentCapture ?? ContentCaptureConfig.default();
              let argsContent: ToolCallContent | null = null;
              let resultContent: ToolResultContent | null = null;
              if (cc.enabled) {
                argsContent = StandardHarness.captureToolCallArgs(call, cc.maxFieldLen);
                if (output.kind === "success") {
                  const [content, t] = truncateField(output.content, cc.maxFieldLen);
                  resultContent = { content, truncated: t };
                } else if (output.kind === "error") {
                  const [content, t] = truncateField(output.message, cc.maxFieldLen);
                  resultContent = { content, truncated: t };
                }
              }
              const toolSpan: ToolCallSpan = {
                base: finishSpanBase(childBase, Timestamp.now(), status, Date.now() - toolClock),
                tool_name: call.name,
                call_id: call.id,
                parameters_size_bytes: JSON.stringify(call.input).length,
                output_size_bytes: outputSizeBytes,
                truncated,
                sandbox_mode: "",
                sandbox_violations: [],
                arguments: argsContent,
                result: resultContent,
              };
              this.config.observability.emitToolCall(toolSpan);
              spanSeq += 1;
            }

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

          // Compaction (issue #46): after tool results are appended and the
          // AfterTool middleware fires, before the loop restarts. Runs the
          // verify→retry→warn loop; never halts the run.
          if (this.config.contextManager.shouldCompact(sessionState)) {
            spanSeq = await this.runCompaction(
              sessionState,
              sessionId,
              task.id,
              spanSeq,
              usage,
              signal,
            );
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

  /**
   * Run the post-compaction verify→retry→warn loop (issue #46/#29).
   *
   * Drives one compaction turn through the agent, verifies the summary, and
   * either accepts it, retries with the missing items injected, or — after
   * `maxCompactionAttempts` — emits a warn event and accepts the summary
   * anyway. A blocked compaction is worse than an imperfect one, so this method
   * NEVER throws or halts the run; the worst case is an accepted-anyway summary
   * plus one warn span.
   *
   * Token usage from compaction turns folds into the run-level
   * {@link AggregateUsage}; each compaction turn that produces a summary is
   * surfaced as a `compaction` {@link ContextSpan}. The
   * `compaction_verification_failures` metric is derived from the emitted
   * {@link WarnSpan}. Returns the advanced `spanSeq`.
   */
  private async runCompaction(
    sessionState: SessionState,
    sessionId: SessionId,
    taskId: TaskId,
    spanSeq: number,
    usage: AggregateUsage,
    signal?: AbortSignal,
  ): Promise<number> {
    const cm = this.config.contextManager;
    // Compaction is opt-in: managers that never compact do not implement
    // prepareCompactionTurn (default `undefined` = skip).
    const turn: CompactionTurn | undefined = cm.prepareCompactionTurn?.(sessionState);
    if (!turn) {
      // Nothing to compact (e.g. history shorter than the preserve window).
      return spanSeq;
    }

    const tokensBefore = turn.verificationState.token_budget_used;
    const verifier: CompactionVerifier = this.config.compactionVerifier ?? new KeyTermVerifier();
    const maxAttempts = Math.max(1, this.config.maxCompactionAttempts ?? 2);
    let attempt = 0;

    for (;;) {
      attempt += 1;
      // Run one compaction turn through the agent to produce a summary.
      const result = await this.config.agent.turn(turn.context, signal);
      let summary: string;
      switch (result.kind) {
        case "final_response":
          addTurnUsage(usage, result.usage);
          summary = result.content;
          break;
        case "tool_call_requested":
          // A compaction turn is expected to yield a summary, not a tool call.
          // Treat the (empty) response as the summary so verification can run
          // and the loop terminates predictably.
          addTurnUsage(usage, result.usage);
          summary = "";
          break;
        default:
          if (result.usage != null) addTurnUsage(usage, result.usage);
          summary = "";
          break;
      }

      const verification = verifier.verify(summary, turn.preserveHints, turn.verificationState);

      if (verification.passed) {
        return this.acceptCompaction(
          sessionState,
          summary,
          turn.messagesRemoved,
          tokensBefore,
          sessionId,
          taskId,
          spanSeq,
        );
      }

      if (attempt < maxAttempts) {
        // Inject the missing items and retry.
        this.injectMissingItems(turn.context, verification.missingItems);
        continue;
      }

      // Exhausted attempts: warn, then accept anyway.
      const obs = this.config.observability;
      if (obs?.emitWarn) {
        const base = newRootSpanBase(
          SpanId.of(`${sessionId.asString()}-warn-${spanSeq}`),
          sessionId,
          taskId,
          "warn",
          Timestamp.now(),
        );
        const event: WarnEvent = {
          warn: "compaction_verification_failed",
          missing_items: verification.missingItems.slice(),
          accepted_anyway: true,
        };
        obs.emitWarn(newWarnSpan(base, event));
        spanSeq += 1;
      }
      return this.acceptCompaction(
        sessionState,
        summary,
        turn.messagesRemoved,
        tokensBefore,
        sessionId,
        taskId,
        spanSeq,
      );
    }
  }

  /** Apply the spec's default missing-items retry message when the manager
   *  does not override {@link ContextManager.injectMissingItems}. */
  private injectMissingItems(context: Context, missing: string[]): void {
    const cm = this.config.contextManager;
    if (cm.injectMissingItems) {
      cm.injectMissingItems(context, missing);
      return;
    }
    context.messages.push({
      role: "user",
      content: {
        type: "text",
        text: `Your summary is missing these items: ${missing.join(", ")}. Please revise.`,
      },
    });
  }

  /** Apply an accepted summary and emit the `compaction` context span. Returns
   *  the advanced `spanSeq`. */
  private acceptCompaction(
    sessionState: SessionState,
    summary: string,
    messagesRemoved: number,
    tokensBefore: number,
    sessionId: SessionId,
    taskId: TaskId,
    spanSeq: number,
  ): number {
    // Default applyCompaction is a no-op (only compaction-capable managers
    // implement it).
    const cm = this.config.contextManager;
    cm.applyCompaction?.(sessionState, summary);

    const obs = this.config.observability;
    if (obs) {
      // Stamp real post-compaction budget when the manager exposes it (issue
      // #57 token-accounting fix); otherwise fall back to the pre-compaction
      // budget (no reclamation surfaced).
      const tokensAfter = cm.tokenBudgetUsed?.(sessionState) ?? tokensBefore;
      const tokensReclaimed = Math.max(0, tokensBefore - tokensAfter);
      const base = newRootSpanBase(
        SpanId.of(`${sessionId.asString()}-compaction-${spanSeq}`),
        sessionId,
        taskId,
        "compaction",
        Timestamp.now(),
      );
      const span: ContextSpan = {
        base,
        operation: {
          kind: "compaction",
          messages_removed: messagesRemoved,
          tokens_reclaimed: tokensReclaimed,
        },
        tokens_before: tokensBefore,
        tokens_after: tokensAfter,
        utilization_before: 0,
        utilization_after: 0,
      };
      obs.emitContext(span);
      spanSeq += 1;
    }
    return spanSeq;
  }
}

// ============================================================================
// HarnessBuilder
// ============================================================================

/**
 * Fluent assembler for a {@link HarnessConfig} / {@link StandardHarness}.
 *
 * The harness follows strict inversion of control: every component is injected.
 * `HarnessBuilder` takes the five required components up front and exposes
 * fluent setters for the optional ones (middleware, observability, pricing).
 * It is the intended caller that wires the {@link ObservabilityProvider} into
 * the loop, including the durable outbox via {@link withObservabilityOutbox}.
 * Mirrors Rust's `HarnessBuilder`.
 *
 * Note (issue #12): the harness deliberately does NOT emit `emitMiddleware` /
 * `emitSensor` / `emitContext` from the loop — middleware uses a separate
 * forward-declared surface and there are no sensor/context call sites here.
 * `emitPatch` is emitted by the patch middleware separately.
 */
export class HarnessBuilder {
  private _middleware?: MiddlewareChain;
  private _observability?: ObservabilityProvider;
  private _compactionVerifier: CompactionVerifier = new KeyTermVerifier();
  private _maxCompactionAttempts = 2;
  private _pricing: PricingTable = PricingTable.DEFAULT;
  private _contentCapture: ContentCaptureConfig = ContentCaptureConfig.default();

  constructor(
    private readonly agent: Agent,
    private readonly toolRegistry: ToolRegistry,
    private readonly sandbox: SandboxProvider,
    private readonly contextManager: ContextManager,
    private readonly terminationPolicy: TerminationPolicy,
  ) {}

  /** Inject a middleware chain. */
  middleware(middleware: MiddlewareChain): this {
    this._middleware = middleware;
    return this;
  }

  /** Inject an observability provider. The harness loop emits real spans through
   *  it (turn spans, tool-call spans) and flushes on terminal outcomes. */
  observability(provider: ObservabilityProvider): this {
    this._observability = provider;
    return this;
  }

  /** Convenience: construct and inject a durable-outbox observability provider
   *  rooted at `root` (typically the `.spore` directory). */
  withObservabilityOutbox(root: string): this {
    this._observability = new OutboxObservabilityProvider(
      outboxConfig(root, { flushOnSessionEnd: true }),
    );
    return this;
  }

  /** Inject a post-compaction verifier (issue #46). Defaults to
   *  {@link KeyTermVerifier}. */
  compactionVerifier(verifier: CompactionVerifier): this {
    this._compactionVerifier = verifier;
    return this;
  }

  /** Set the maximum number of compaction-summary attempts before accepting a
   *  failing summary anyway (issue #46). Defaults to `2`; clamped to `1`. */
  maxCompactionAttempts(attempts: number): this {
    this._maxCompactionAttempts = attempts;
    return this;
  }

  /** Set the token → USD pricing table used to stamp `cost_usd` on turn spans. */
  pricing(table: PricingTable): this {
    this._pricing = table;
    return this;
  }

  /** Configure LLM-native content capture (issue #64). Defaults to OFF. Use
   *  {@link ContentCaptureConfig.fromEnv} to honor the env guard. */
  contentCapture(config: ContentCaptureConfig): this {
    this._contentCapture = config;
    return this;
  }

  /** Assemble the {@link HarnessConfig} without wrapping it in a harness. */
  buildConfig(): HarnessConfig {
    return {
      agent: this.agent,
      toolRegistry: this.toolRegistry,
      sandbox: this.sandbox,
      contextManager: this.contextManager,
      terminationPolicy: this.terminationPolicy,
      middleware: this._middleware,
      observability: this._observability,
      compactionVerifier: this._compactionVerifier,
      maxCompactionAttempts: this._maxCompactionAttempts,
      pricing: this._pricing,
      contentCapture: this._contentCapture,
    };
  }

  /** Assemble a ready-to-run {@link StandardHarness}. */
  build(): StandardHarness {
    return new StandardHarness(this.buildConfig());
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

/** Derive the `SessionOutcome.failure.reason` string from a {@link HaltReason}.
 *  Mirrors Rust's `format!("{reason:?}")` for the failure outcome. */
function haltReasonToString(reason: HaltReason): string {
  return JSON.stringify(reason);
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
