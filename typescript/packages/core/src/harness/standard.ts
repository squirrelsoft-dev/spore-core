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
import { StorageProvider } from "../storage/index.js";
import { KeyTermVerifier, type CompactionVerifier } from "../context/types.js";
import { newSessionState as newContextSessionState } from "../context/types.js";
import {
  emptyTurnOutput,
  type FireOutcome,
  type HookChain,
  type HookContext,
} from "../hooks/index.js";
import {
  PLAN_EXECUTE_EXTRAS_KEY,
  PlanPhaseError,
  capturePlanArtifact,
  type PlanArtifact,
} from "../plan/index.js";

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
   * Pluggable per-domain persistence layer (issue #73). Optional; defaults to an
   * all-no-op {@link StorageProvider} so existing callers compile and behave
   * unchanged. v1 is expose-only: the harness does NOT read/write sessions
   * internally on pause/resume — it only carries the provider for callers and
   * for the observability fan-out.
   */
  storage?: StorageProvider;
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
  /**
   * Lifecycle hook chain (issue #69). When set, the harness fires registered
   * `stop` hooks synchronously when the loop strategy believes the task is
   * complete. A `block` decision injects its reason into the next turn and the
   * loop continues, up to {@link maxStopBlocks} times per run. Optional;
   * absent means no hooks fire and the loop terminates normally.
   */
  hooks?: HookChain;
  /**
   * Maximum consecutive Stop-hook blocks honored within a single run before the
   * loop terminates anyway (issue #69, R14). The counter is PER-RUN and resets
   * each `run()`/`resume()` call. Optional; defaults to `8`.
   */
  maxStopBlocks?: number;
  /**
   * Optional alternate agent used for the `plan_execute` plan phase (issue #70,
   * Q1). When the loop strategy is `plan_execute` and this is set, the one-shot
   * plan turn runs on this agent; otherwise it runs on the default {@link agent}.
   * `plan_model` on the strategy stays DESCRIPTIVE metadata only.
   */
  plannerAgent?: Agent;
}

const DEFAULT_MAX_STOP_BLOCKS = 8;

export class StandardHarness implements Harness {
  constructor(private readonly config: HarnessConfig) {}

  /**
   * The configured {@link StorageProvider} (issue #73). Defaults to an all-no-op
   * provider when `.storage(...)` was never set, so callers never null-check —
   * they always get a usable provider and the store decides what to do.
   */
  storage(): StorageProvider {
    return this.config.storage ?? StorageProvider.noOp();
  }

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

  /**
   * Fire registered `stop` hooks (issue #69, R12–R14). The strategy believes
   * the task is done; fire the chain synchronously. Returns the reason string
   * to inject + continue the loop when a hook blocked AND the per-run
   * `maxStopBlocks` cap has not yet been hit (incrementing `stopBlocks`).
   * Returns `null` to allow normal termination — no chain configured, no hook
   * blocked, the cap was reached, or a hook errored.
   *
   * A Stop-hook error (e.g. a failing command handler) is treated as a
   * non-blocking outcome: the loop terminates normally rather than looping
   * forever on a broken hook.
   */
  private async fireStopHooks(
    sessionId: SessionId,
    task: Task,
    turnNumber: number,
    lastOutputText: string,
    stopBlocks: { value: number },
    signal?: AbortSignal,
  ): Promise<string | null> {
    const chain = this.config.hooks;
    if (!chain) return null;

    const richState = newContextSessionState(sessionId, task.id, task.instruction);
    const lastOutput = { ...emptyTurnOutput(), text: lastOutputText };
    const ctx: HookContext = {
      event: "stop",
      session_id: sessionId,
      turn_number: turnNumber,
      last_output: lastOutput,
      task_instruction: task.instruction,
      session_state: richState,
    };

    let outcome: FireOutcome;
    try {
      outcome = await chain.fire(ctx, signal);
    } catch {
      // Broken hook → allow normal termination.
      return null;
    }

    if (outcome.kind === "block") {
      const cap = this.config.maxStopBlocks ?? DEFAULT_MAX_STOP_BLOCKS;
      if (stopBlocks.value >= cap) return null; // R14: cap reached.
      stopBlocks.value += 1;
      return outcome.reason;
    }
    // continue / inject / deny → allow normal termination.
    return null;
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
        return this.runPlanExecute(
          task,
          sessionState,
          budgetUsed,
          options.on_stream,
          options.signal,
        );
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

  // --------------------------------------------------------------------------
  // PlanExecute plan phase (issue #70)
  // --------------------------------------------------------------------------

  /**
   * Drive the `plan_execute` strategy (issue #70).
   *
   * Runs the one-shot plan phase ({@link runPlanPhase}), then — per Q4 — HALTS
   * with the distinct {@link HaltReason} `execute_phase_not_implemented` once an
   * artifact has been produced, had `on_plan_created` fired on it, and been
   * stored. The execute loop itself ships with #59/#72. On any plan-phase
   * failure the underlying `failure` `RunResult` is returned unchanged (no
   * artifact stored). Like `runReact`, finalizes observability for the terminal
   * outcome.
   */
  private async runPlanExecute(
    task: Task,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    const outcome = await this.runPlanPhase(task, sessionState, budgetUsed, onStream, signal);

    let result: RunResult;
    if (outcome.ok) {
      // Plan produced + stored. Q4: halt with the distinct reason.
      result = {
        kind: "failure",
        reason: { kind: "execute_phase_not_implemented" },
        session_id: sessionId,
        usage: outcome.usage,
        turns: outcome.turns,
      };
    } else {
      result = outcome.failure;
    }

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
        break;
    }
    return result;
  }

  /**
   * Run the one-shot `plan_execute` plan phase (issue #70).
   *
   * Selects the planner agent (Q1: {@link HarnessConfig.plannerAgent} if set,
   * else the default agent), seeds a planning directive as a user message, runs
   * EXACTLY ONE constrained turn (R1), expects a `final_response` (a tool call
   * is a planning failure — R2 — never a dispatch loop), captures the response
   * via {@link capturePlanArtifact} (R3), fires `on_plan_created` (which may
   * rewrite the artifact — R11), stores the result in `extras["plan_execute"]`
   * (R4), emits the turn span (R8), and counts the turn against the shared
   * budget (R7). A budget exhausted before the turn returns a budget-exceeded
   * `failure` with no artifact stored (R10).
   *
   * On success resolves `{ ok: true, artifact, usage, turns }`. On any failure
   * resolves `{ ok: false, failure }` with the terminal `failure` `RunResult`.
   */
  private async runPlanPhase(
    task: Task,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<
    | { ok: true; artifact: PlanArtifact; usage: AggregateUsage; turns: number }
    | { ok: false; failure: RunResult }
  > {
    const sessionId = task.session_id;
    const startedAt = Date.now();
    const usage: AggregateUsage = emptyAggregateUsage();
    const pricing = this.config.pricing ?? PricingTable.DEFAULT;

    // R10: Layer-1 budget gate BEFORE the plan turn. Mirrors runReactInner.
    const taskMaxTurns = task.budget.max_turns ?? undefined;
    const effectiveTurnCap =
      taskMaxTurns != null ? Math.max(taskMaxTurns, 1) : Number.MAX_SAFE_INTEGER;
    if (budgetUsed.turns >= effectiveTurnCap) {
      return {
        ok: false,
        failure: failure(
          { kind: "budget_exceeded", limit_type: "turns" },
          sessionId,
          usage,
          budgetUsed.turns,
        ),
      };
    }
    const overrun = budgetExceeded(task.budget, budgetUsed, startedAt);
    if (overrun != null) {
      return {
        ok: false,
        failure: failure(
          { kind: "budget_exceeded", limit_type: overrun },
          sessionId,
          usage,
          budgetUsed.turns,
        ),
      };
    }

    // Q1: select the planner agent (alternate if configured, else default).
    const planner = this.config.plannerAgent ?? this.config.agent;

    // Seed the planning directive as a user message (reuse ContextManager).
    const directive =
      "Produce a step-by-step plan for the following task. Respond with a " +
      'single JSON object: {"tasks": [<ordered step strings>], ' +
      '"rationale": <string>}.\n\nTask:\n' +
      task.instruction;
    await this.config.contextManager.appendUserMessage(sessionState, directive);

    // Assemble + invoke the planner for exactly ONE turn (R1).
    const context = await this.config.contextManager.assemble(sessionState, task, signal);
    emit(onStream, { kind: "turn_start", turn: budgetUsed.turns + 1 });
    const turnStartedAt = Timestamp.now();
    const turnClock = Date.now();
    const result = await planner.turn(context, signal);
    budgetUsed.turns += 1; // R7: the plan turn counts against the budget.

    // R8: emit exactly one turn span for the plan turn. Mirrors the metrics path
    // of runReactInner; content capture intentionally omitted (the plan turn
    // carries no tool calls and #64 content capture is wired in the ReAct loop).
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
          output_text: null,
          tool_calls: null,
          input_messages: null,
        };
        this.config.observability.emitTurn(turnSpan);
      }
    }
    emit(onStream, { kind: "turn_end", turn: budgetUsed.turns });

    // Classify the one-shot turn. R2: a tool call is a planning failure, NOT a
    // dispatch loop.
    let finalText: string;
    switch (result.kind) {
      case "final_response":
        addTurnUsage(usage, result.usage);
        budgetUsed.input_tokens += result.usage.input_tokens;
        budgetUsed.output_tokens += result.usage.output_tokens;
        finalText = result.content;
        break;
      case "tool_call_requested": {
        addTurnUsage(usage, result.usage);
        const error = PlanPhaseError.planningTurnFailed(
          "planner requested a tool call in the one-shot plan turn",
        );
        return {
          ok: false,
          failure: failure(
            { kind: "plan_phase_failed", error: error.detail },
            sessionId,
            usage,
            budgetUsed.turns,
          ),
        };
      }
      case "error": {
        if (result.usage != null) addTurnUsage(usage, result.usage);
        return {
          ok: false,
          failure: failure(
            { kind: "agent_error", error: result.error },
            sessionId,
            usage,
            budgetUsed.turns,
          ),
        };
      }
      default: {
        const _exhaustive: never = result;
        return _exhaustive;
      }
    }

    // R3: capture the artifact from the response text.
    const captured = capturePlanArtifact(finalText);
    if (!captured.ok) {
      return {
        ok: false,
        failure: failure(
          { kind: "plan_phase_failed", error: captured.error.detail },
          sessionId,
          usage,
          budgetUsed.turns,
        ),
      };
    }
    // R11: fire on_plan_created synchronously; the hook may rewrite the artifact
    // — either by mutating it in place OR by returning a `mutate` decision that
    // reassigns `ctx.plan` to a new object. Read the final value back off `ctx`
    // so either path is honored. Errors are non-fatal: a successfully-captured
    // plan is not lost to a handler error.
    const ctx: HookContext = {
      event: "on_plan_created",
      session_id: sessionId,
      plan: captured.artifact,
    };
    if (this.config.hooks) {
      try {
        await this.config.hooks.fire(ctx, signal);
      } catch {
        // Swallow — the (possibly mutated) artifact is still stored.
      }
    }
    const artifact: PlanArtifact = ctx.plan;

    // R4: store the produced artifact in extras["plan_execute"] as a JSON value
    // (the stable cross-language shape: `{ tasks, rationale }`).
    sessionState.extras[PLAN_EXECUTE_EXTRAS_KEY] = {
      tasks: artifact.tasks,
      rationale: artifact.rationale,
    };

    return { ok: true, artifact, usage, turns: budgetUsed.turns };
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
    // Per-run Stop-hook block counter (issue #69, R14). Resets on every
    // run()/resume() — a resumed loop starts fresh. After `maxStopBlocks`
    // consecutive blocks the loop terminates anyway. Boxed so `fireStopHooks`
    // can mutate it.
    const stopBlocks = { value: 0 };
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

          // Record the assistant's final text in history so a continued
          // session reflects what the agent said (multi-turn / S2 correctness).
          {
            const msg: Message = {
              role: "assistant",
              content: { type: "text", text: result.content },
            };
            await this.config.contextManager.appendAssistantMessage?.(sessionState, msg);
          }

          // Stop hook (issue #69, R12). The strategy believes the task is done;
          // fire registered `stop` hooks synchronously. If any blocks (and we
          // are under `maxStopBlocks`), inject the reason as a user message —
          // the same path `force_another_turn` injects through — and continue
          // the loop instead of terminating.
          {
            const reason = await this.fireStopHooks(
              sessionId,
              task,
              budgetUsed.turns,
              result.content,
              stopBlocks,
              signal,
            );
            if (reason != null) {
              await this.config.contextManager.appendUserMessage(sessionState, reason);
              continue;
            }
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

          // Record the assistant's turn (the tool calls the model requested) as
          // soon as the calls are known — BEFORE the BeforeTool middleware
          // (which may pause via SurfaceToHuman) and before any tool result.
          // This keeps the conversation well-formed (assistant tool_use precedes
          // its tool result) on every path, including human-in-the-loop resume,
          // so the resume path never has to append it (and never double-records
          // it). The recorded turn reflects the model's original request; a
          // middleware or human modification changes only what is dispatched.
          for (const call of result.calls) {
            const msg: Message = {
              role: "assistant",
              content: { type: "tool_call", id: call.id, name: call.name, input: call.input },
            };
            await this.config.contextManager.appendAssistantMessage?.(sessionState, msg);
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
  private _storage?: StorageProvider;
  private _compactionVerifier: CompactionVerifier = new KeyTermVerifier();
  private _maxCompactionAttempts = 2;
  private _pricing: PricingTable = PricingTable.DEFAULT;
  private _contentCapture: ContentCaptureConfig = ContentCaptureConfig.default();
  private _hooks?: HookChain;
  private _maxStopBlocks = DEFAULT_MAX_STOP_BLOCKS;
  private _plannerAgent?: Agent;

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

  /** Inject a {@link StorageProvider} (issue #73). Defaults to an all-no-op
   *  provider when unset, so existing callers compile and behave unchanged. */
  storage(provider: StorageProvider): this {
    this._storage = provider;
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

  /** Inject a lifecycle hook chain (issue #69). The harness fires registered
   *  `stop` hooks at the loop's completion gate. */
  hooks(hooks: HookChain): this {
    this._hooks = hooks;
    return this;
  }

  /** Set the per-run cap on honored Stop-hook blocks (issue #69, R14).
   *  Defaults to `8`. */
  maxStopBlocks(max: number): this {
    this._maxStopBlocks = max;
    return this;
  }

  /** Inject an alternate agent for the `plan_execute` plan phase (issue #70,
   *  Q1). When set and the loop strategy is `plan_execute`, the one-shot plan
   *  turn runs on this agent; otherwise it runs on the default agent. */
  plannerAgent(agent: Agent): this {
    this._plannerAgent = agent;
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
      storage: this._storage ?? StorageProvider.noOp(),
      compactionVerifier: this._compactionVerifier,
      maxCompactionAttempts: this._maxCompactionAttempts,
      pricing: this._pricing,
      contentCapture: this._contentCapture,
      hooks: this._hooks,
      maxStopBlocks: this._maxStopBlocks,
      plannerAgent: this._plannerAgent,
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
