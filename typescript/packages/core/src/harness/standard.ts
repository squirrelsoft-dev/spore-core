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

import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { Agent, AgentStreamSink } from "../agent/interface.js";
import { turnStreaming } from "../agent/interface.js";
import type { Context } from "../agent/types.js";
import { AgentId, ModelAgent } from "../agent/index.js";
import type { ModelInterface } from "../model/interface.js";
import {
  AdaptiveToolCallModelInterface,
  detectProseResponse,
  newSharedFlag,
  PROMPT_TOOL_CALL_NUDGE,
  type SharedFlag,
} from "../model/prompt-tool-call.js";
import { StandardContextManager } from "../context/standard.js";
import { intoHarnessAdapter } from "../context/compaction-adapter.js";
import { defaultCompactionConfig } from "../context/types.js";
import { NullCacheProvider } from "../cache-provider/types.js";
import { NullSandbox } from "../sandbox/null-sandbox.js";
import { EmptyToolRegistry } from "../tool-registry/empty.js";
import { CompleteOnFinalResponse } from "./complete-on-final-response.js";
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
import type { Message, ModelParams, ToolCall } from "../model/schemas.js";
import { ModelParamsSchema } from "../model/schemas.js";
import type { GenAiRole } from "../observability/types.js";
import { OutboxObservabilityProvider, outboxConfig } from "../observability/outbox.js";
import { InMemoryStorageProvider, StorageProvider } from "../storage/index.js";
import {
  RealToolRegistry,
  RegistrationErrorException,
  StandardToolRegistry,
  type StandardTool,
} from "../tool-registry/index.js";
import {
  InMemoryChunkProvider,
  type ChunkProvider,
  type PromptChunk as AssemblyPromptChunk,
} from "../prompt-assembly/index.js";
import { KeyTermVerifier, type CompactionVerifier } from "../context/types.js";
import type { Verifier } from "../verifier/types.js";
import {
  shouldKeep,
  iterationStatusFromError,
  metricErrorMessage,
  type IterationStatus,
  type MetricEvaluator,
  type ResultsEntry,
} from "../metric/types.js";
import { newSessionStateSnapshot } from "../termination/types.js";
import { ReadOnlySandbox } from "../sandbox/read-only-sandbox.js";
import { newSessionState as newContextSessionState } from "../context/types.js";
import {
  FunctionHook,
  StandardHookChain,
  emptyTurnOutput,
  type FireOutcome,
  type HookChain,
  type HookContext,
} from "../hooks/index.js";
import { PLAN_EXECUTE_EXTRAS_KEY, capturePlanArtifact, type PlanArtifact } from "../plan/index.js";
import {
  TASK_LIST_EXTRAS_KEY,
  TaskListSchema,
  completeTask,
  serializeTaskList,
  updateTask,
  type TaskList,
} from "../tasklist/index.js";

import type { Harness } from "./interface.js";
import type { VcsLogArgs, VcsProvider } from "./vcs.js";
import { SessionId, TaskId, TurnStreamState, mapModelStreamEvent } from "./types.js";
import {
  addTurnUsage,
  emptyAggregateUsage,
  emptyBudgetSnapshot,
  emptySessionState,
  haltReasonToString,
  newExecutionContext,
  newTask,
  runResultSessionState,
  type ExecutionContext,
  type PlanPhaseOutcome,
  type StrategyExecutor,
  type AggregateUsage,
  type BudgetLimitType,
  type BudgetLimits,
  type BudgetSnapshot,
  type ChildPausedState,
  type CompactionTurn,
  type ConsultHandlerEntry,
  type ConsultHandlerMap,
  type ConsultOverflowPolicy,
  type ConsultResponse,
  type ContextManager,
  type EscalationAction,
  type EscalationMode,
  type HaltReason,
  HarnessError,
  InvalidConfiguration,
  UnresolvedHandle,
  type AgentRef,
  autonomous,
  type BudgetExhausted,
  type BudgetExhaustedBehavior,
  BudgetContext,
  budgetPolicyAllowanceValue,
  grantTaskBudget,
  leafEscalationActions,
  promoteBudgetExhaustedToHuman,
  type HarnessRunOptions,
  surfaceToHuman,
  type HookPoint,
  type HumanRequest,
  type HumanResponse,
  type LoopStrategy,
  type MiddlewareChain,
  type ObservabilityProvider,
  type HillClimbingDirection,
  reactMaxIterations,
  reactPerLoop,
  runStrategy,
  type PausedState,
  type RunResult,
  type RunStrategy,
  type SandboxProvider,
  type SessionState,
  type StreamSink,
  type Task,
  type TerminationPolicy,
  type ToolOutput,
  type ToolRegistry,
  type ToolResultRecord,
} from "./types.js";
import { ExecutionRegistry } from "./execution-registry.js";

/** Components injected at construction. Mirrors `HarnessConfig` in the spec. */
export interface HarnessConfig {
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
   * Pluggable source of conditional prompt chunks (issue #79). Loaded at harness
   * construction and fed through {@link "../prompt-assembly/index.js".ContextSourcesBuilder}.
   * Optional; defaults to an empty {@link InMemoryChunkProvider}.
   */
  chunkProvider?: ChunkProvider;
  /**
   * Outer-loop reset cap for the `ralph` loop strategy (issue #58, B3). The
   * maximum number of context-window RESETS the multi-context-window
   * continuation loop runs before halting with {@link HaltReason}
   * `ralph_completion_unmet` when tasks are still incomplete. Independent of
   * `max_turns` (the per-window ReAct turn budget) and {@link maxStopBlocks}.
   * Optional; defaults to `3`. Clamped to a minimum of `1`. Ignored by every
   * other strategy.
   */
  maxResets?: number;
  /**
   * Optional VCS read seam for the `ralph` loop strategy (issue #58 v2). When
   * set, Ralph's per-window reload phase ALSO calls {@link VcsProvider.log} and
   * injects the output into the fresh context window as a delimited
   * "Recent VCS history:" section — alongside the reloaded `.spore/progress.json`
   * + `.spore/feature_list.json` content. When unset (the default) the git-log
   * section is OMITTED and Ralph behaves byte-for-byte like v1 (the B4→none
   * decision). Ignored by every other strategy.
   */
  vcsProvider?: VcsProvider;
  /**
   * Catalogue tools accumulated via {@link HarnessBuilder.tool} / `tools` (issue
   * #81), drained into a populated {@link StandardToolRegistry} at
   * {@link HarnessBuilder.buildConfig}. When set, the run loop bridges it per-run
   * via {@link RealToolRegistry} — threading the run's {@link SessionId}, sandbox,
   * and storage into every tool dispatch — and uses that instead of
   * {@link toolRegistry} (which stays the harness-loop seam for custom slim
   * registries). Absent (the default) preserves the `toolRegistry`-only path
   * unchanged.
   */
  catalogueRegistry?: StandardToolRegistry;
  /**
   * Operating system prompt prepended to each turn's assembled context when the
   * context manager renders none (issue #91). See {@link HarnessBuilder.systemPrompt}.
   * Absent (the default) preserves today's behaviour.
   */
  systemPrompt?: string;
  /**
   * Authoritative per-run model sampling/decoding parameters (issue #93). The
   * harness replaces each turn's {@link Context.params} with this value
   * UNCONDITIONALLY (builder params win) right before the request is built, so
   * the configured params reach every agent turn that requests tools. See
   * {@link HarnessBuilder.modelParams}. Defaults to the schema's default
   * {@link ModelParams} (`{ stop_sequences: [] }`, `structured_tool_calls`
   * absent ⇒ `false`).
   */
  modelParams: ModelParams;
  /**
   * Opt-in conversation-history threading through the {@link StorageProvider}'s
   * {@link SessionStore} (issue #102). When `true`, the harness:
   *   - **auto-loads** the prior {@link SessionState} for the run's `session_id`
   *     at the start of `run()` (ReAct / SelfVerifying only — Ralph/HillClimbing
   *     discard incoming state by design). An explicit
   *     {@link HarnessRunOptions.session_state} always wins (no load).
   *   - **auto-persists** the post-run {@link SessionState} back to the store at
   *     the terminal seam — one write per `run()`/`resume()`.
   *
   * Absent / `false` (the default) is the off-by-default zero-I/O contract: the
   * harness performs NO session-store reads or writes and behaves byte-for-byte
   * like today. Storage errors are swallowed-and-logged, never surfaced as a
   * {@link HaltReason}. See {@link HarnessBuilder.autoPersistSessions}.
   */
  autoPersistSessions?: boolean;
  /**
   * Shared escalation flag for adaptive prompt-based tool calling (#111). Set
   * ONLY by {@link HarnessBuilder.conversational}, which wraps the agent's model
   * in an {@link AdaptiveToolCallModelInterface} sharing this SAME holder. While
   * `flag.value` is `false` the wrapper delegates natively (byte-for-byte); the
   * run loop flips it to `true` on detecting a prose response where a tool call
   * was expected, switching the model to prompt-based tool calling for the rest
   * of the run. Reset to `false` at the start of each turn-loop window so
   * detection is scoped to the window. Absent (the default) ⇒ no adaptive
   * wrapper is installed and the escalation path is inert.
   */
  prompt_tool_call_flag?: SharedFlag;
  /**
   * Per-kind consult handlers (issue #114), keyed by {@link ConsultRequest.kind}.
   * Absent / empty (the default) means consults are NOT mediated: a worker that
   * pauses with {@link RunResult} `consult` surfaces it unchanged to its caller
   * (R6 graceful degradation), and existing callers are unaffected (R9). When
   * populated, the orchestrator typically passes a clone of this map to its
   * {@link "@spore/tools".SubagentTool}, which runs the matching handler
   * deterministically (the A1 mediation seam) — the orchestrator model is never
   * involved.
   */
  consultHandlers?: ConsultHandlerMap;
  /**
   * Runtime resolver for the serializable strategy handles
   * ({@link AgentRef}/{@link ToolsetRef}/{@link SchemaRef}) and `StrategyRef`
   * custom keys held by a task's strategy tree (issue #120). {@link run} calls
   * {@link ExecutionRegistry.validate} at entry, so an unresolved handle is a
   * STARTUP error before the first turn — but only when the registry is
   * populated, so legacy callers (Option B) stay byte-identical. This slice the
   * registry coexists with the deprecated single-collaborator fields and is not
   * yet read by the run bodies (#123/#124). Optional; defaults to an empty
   * registry.
   */
  registry?: ExecutionRegistry;
  /**
   * HITL-vs-AFK escalation knob (issue #120, PRD goal #7). Selects whether
   * budget escalation surfaces to a human or proceeds autonomously. STORED only
   * this slice; consumed in #130. Optional; the {@link HarnessBuilder} defaults
   * it to {@link surfaceToHuman}.
   */
  escalationMode?: EscalationMode;
}

const DEFAULT_MAX_STOP_BLOCKS = 8;
const DEFAULT_MAX_RESETS = 3;

export class StandardHarness implements Harness, StrategyExecutor {
  /**
   * #126 (AC2): the harness-OBSERVED write/edit tool-call paths for the CURRENT
   * execute step. The dispatch seam records the `path` of every `write_file` /
   * `edit_file` tool call here as it dispatches ({@link observeWriteCall}); the
   * PlanExecute DAG executor drains it ({@link takeObservedWrites}) on task
   * completion to build the task's `StepLedgerEntry.files_touched`, and clears it
   * ({@link clearObservedWrites}) before each step. The path always comes from
   * the real tool call — never a model-self-reported field.
   */
  private observedWrites: string[] = [];

  constructor(private readonly config: HarnessConfig) {
    // Issue #58, B1: drive Ralph off the Stop hook. Register a `stop` hook at
    // construction that reads `.spore/progress.json` under the sandbox's
    // workspace root: while tasks remain incomplete it blocks (the loop
    // continues into a new context window); all complete ⇒ continue (the loop
    // terminates). Absent progress file ⇒ continue, so the hook is INERT for a
    // non-Ralph run over the same workspace (matching the Rust reference, which
    // registers the hook at construction).
    //
    // Registered ONLY when the sandbox exposes a concrete workspace root —
    // without one, `.spore/progress.json` cannot be resolved deterministically,
    // and registering would perturb the `config.hooks`-absent contract every
    // non-Ralph caller relies on. Goes into the configured chain, or a fresh
    // `StandardHookChain` when none was supplied.
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    if (workspaceRoot.length > 0) {
      const chain = this.config.hooks ?? new StandardHookChain();
      chain.register(
        new FunctionHook("ralph-stop", ["stop"], (ctx) => {
          if (ctx.event !== "stop") return { decision: "continue" };
          // Absent progress file ⇒ do not interfere with non-Ralph runs over
          // this workspace (the completion mechanism only engages once a
          // `.spore/progress.json` is present). Mirrors the Rust RalphStopHook.
          if (!StandardHarness.ralphProgressFilePresent(workspaceRoot)) {
            return { decision: "continue" };
          }
          const reason = StandardHarness.ralphCompletionStatus(workspaceRoot);
          return reason == null ? { decision: "continue" } : { decision: "block", reason };
        }),
      );
      this.config = { ...this.config, hooks: chain };
    }
  }

  /**
   * The configured {@link StorageProvider} (issue #73). Defaults to an all-no-op
   * provider when `.storage(...)` was never set, so callers never null-check —
   * they always get a usable provider and the store decides what to do.
   */
  storage(): StorageProvider {
    return this.config.storage ?? StorageProvider.noOp();
  }

  /**
   * The harness-loop {@link ToolRegistry} to use for a run keyed by `sessionId`
   * (issue #91). When catalogue tools were added via {@link HarnessBuilder.tool} /
   * `tools`, this bridges the folded {@link StandardToolRegistry} through a
   * {@link RealToolRegistry} — built fresh per run so the run's {@link SessionId}
   * + sandbox + storage (run store + memory store) thread into every tool
   * dispatch. Otherwise it returns the injected {@link HarnessConfig.toolRegistry}
   * seam unchanged.
   */
  private effectiveToolRegistry(sessionId: SessionId): ToolRegistry {
    const catalogue = this.config.catalogueRegistry;
    if (catalogue == null) return this.config.toolRegistry;
    const store = this.storage();
    return new RealToolRegistry(
      catalogue,
      this.config.sandbox,
      sessionId,
      store.run(),
      store.memory(),
    );
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
    const result = await this.runInner(options);
    await this.autoPersistTerminal(result);
    return result;
  }

  /**
   * Issue #102 auto-persist seam: write the terminal run state to the
   * {@link SessionStore} when {@link HarnessConfig.autoPersistSessions} is
   * enabled. One write per `run()`/`resume()`, at the same terminal seam as the
   * observability flush.
   *
   * For `success`/`failure` a {@link PausedState} is synthesized carrying the
   * final {@link SessionState} with empty pending fields (D4); for
   * `waiting_for_human`/`escalate` the carried {@link PausedState} is persisted
   * (D6 — the cross-process pause case). Storage errors are swallowed-and-logged
   * (D8): a put failure must never lose the run nor surface as a {@link HaltReason}.
   *
   * When disabled (the default) this returns immediately WITHOUT touching the
   * store — the off-by-default zero-I/O contract (#102).
   */
  private async autoPersistTerminal(result: RunResult): Promise<void> {
    if (!this.config.autoPersistSessions) return;

    let sessionId: SessionId;
    let state: PausedState;
    switch (result.kind) {
      case "success":
      case "failure": {
        sessionId = result.session_id;
        // Synthesize a completed-run PausedState: empty pending fields, no human
        // request, no child — it carries only the final history so a later
        // getSession(..).session_state resumes losslessly.
        state = {
          session_id: sessionId,
          task_id: new TaskId(sessionId.asString()),
          turn_number: result.turns,
          session_state: runResultSessionState(result),
          pending_tool_calls: [],
          approved_results: [],
          task: newTask("", sessionId, reactPerLoop(0)),
          budget_used: emptyBudgetSnapshot(),
          child_state: null,
        };
        break;
      }
      // Persist the carried pause state directly (D6).
      case "waiting_for_human":
        sessionId = result.state.session_id;
        state = result.state;
        break;
      case "escalate":
        sessionId = result.session_id;
        state = result.state;
        break;
      // Consult (issue #114) is non-terminal — persist the carried pause state
      // directly (same as waiting_for_human) so a cross-process host can later
      // resumeConsult it.
      case "consult":
        sessionId = result.session_id;
        state = result.state;
        break;
    }

    // Swallow-and-log on rejection (D8): a storage hiccup must not lose the run.
    try {
      await this.storage().session().putSession(sessionId, state);
    } catch {
      // Intentionally dropped — never surfaced as a HaltReason. (No logging
      // facade is wired into @spore/core; the error is dropped, not propagated.)
    }
  }

  private async runInner(options: HarnessRunOptions): Promise<RunResult> {
    const budgetUsed = emptyBudgetSnapshot();
    const task = options.task;

    // #124 startup validation: every serializable handle in the task's strategy
    // tree must resolve against the configured ExecutionRegistry, BEFORE the
    // first turn. The legacy single-collaborator fields are gone — resolution is
    // now the SINGLE path, so validation ALWAYS runs (the `!isEmpty()` gate is
    // dropped). An unresolved handle is a startup error.
    const registry = this.config.registry ?? ExecutionRegistry.empty();
    try {
      registry.validate(task);
    } catch (e) {
      if (e instanceof HarnessError) {
        return {
          kind: "failure",
          reason: { kind: "configuration_error", error: e },
          session_id: task.session_id,
          usage: emptyAggregateUsage(),
          turns: 0,
          session_state: emptySessionState(),
        };
      }
      throw e;
    }

    // Issue #102 auto-load: when enabled AND no explicit session_state was
    // provided AND the strategy seeds incoming state (ReAct / SelfVerifying —
    // Ralph/HillClimbing discard it by design, D7), load the prior session for
    // this `session_id` from the SessionStore so a caller can resume by id.
    // Explicit `session_state` always wins (D5). Errors are swallowed-and-logged:
    // a load failure starts fresh (D8).
    let sessionState = options.session_state;
    if (
      sessionState == null &&
      this.config.autoPersistSessions === true &&
      (task.loop_strategy.kind === "react" || task.loop_strategy.kind === "self_verifying")
    ) {
      try {
        const prior = await this.storage().session().getSession(task.session_id);
        if (prior != null) sessionState = prior.session_state;
      } catch {
        // Swallow-and-log: start fresh on a load failure (D8).
      }
    }
    const resolvedState = sessionState ?? emptySessionState();

    // #124: the central dispatch `match` is GONE — the only `switch` left is the
    // enum→config delegation inside {@link runStrategy}. The harness entry
    // collapses to `task.loop_strategy.run(cx)` via the recursive executor.
    // Instruction seeding stays OWNED by the leaf {@link reactWindow} / the
    // combinators' own build sub-loops (byte-for-byte parity with the ported
    // bodies, AC6) rather than being lifted to the entry.
    return this.driveStrategy(task, resolvedState, budgetUsed, options.on_stream, options.signal);
  }

  /**
   * The recursive-executor entry (#124): build the shared {@link ExecutionContext},
   * seed the per-run scratch (task / session / budget), drive
   * `task.loop_strategy.run(cx)` via {@link runStrategy}, and translate the
   * outcome back into a terminal {@link RunResult} (Q5). A non-terminal pause /
   * escalate stashed in `scratch.terminalOverride` propagates VERBATIM (it never
   * collapses into a {@link StrategyOutcome}).
   */
  private async driveStrategy(
    task: Task,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    return this.driveStrategyWithResumeSeed(
      task,
      sessionState,
      budgetUsed,
      onStream,
      signal,
      undefined,
      undefined,
    );
  }

  /**
   * `driveStrategy` with an optional cross-process Continue checkpoint seed
   * (#129, AC2): `resumeContinues = [phase, continuesUsed]` seeds the FIRST
   * matching budget scope's `continuesUsed` (via {@link BudgetContext.resumed})
   * so a `continue` that spanned a process pause resumes with the correct
   * continue count instead of a zeroed one. `undefined` is the fresh-run path.
   *
   * #129: the BARE-LEAF resolution site is HERE (a bare leaf never self-resolves
   * inside its own body — rule 6 — it PROPAGATES a typed `budget_exhausted`, the
   * single recovery site for a top-level leaf). When the leaf's CONFIGURED
   * `behavior` resolves to `continue`, the leaf is granted one more round
   * IN-PROCESS (bump `continuesUsed`, refresh the step cap) and re-driven WITHOUT
   * any serialization (AC3) — looping until the behavior resolves to
   * `fail`/`escalate` or the strategy completes. `behaviorForResolution` carries
   * the resolution chain's mutated state across in-process continues so
   * `max_continues` is honored.
   *
   * `consultResume = session` (#131) seeds the FIRST PlanExecute walk's
   * in-progress task with `session` (a worker conversation whose consult answer
   * is already injected) so a resumed consult continues mid-loop and walks the
   * remaining ready-set. `undefined` on every non-consult path.
   */
  private async driveStrategyWithResumeSeed(
    task: Task,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
    resumeContinues: [string, number] | undefined,
    consultResume: SessionState | undefined,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    const registry = this.config.registry ?? ExecutionRegistry.empty();

    let currentTask = task;
    let currentSession = sessionState;
    let currentBudget = budgetUsed;
    let currentStream = onStream;
    // The Continue resolution state threaded across in-process rounds: the
    // (possibly fallen-through) behavior + how many continues have been spent.
    let behaviorForResolution: [BudgetExhaustedBehavior, number] | undefined;
    // #129 (AC2): the cross-process checkpoint seed is consumed by the FIRST
    // round only (the resumed node's scope); later in-process rounds carry their
    // continue count via `behaviorForResolution`.
    let seed = resumeContinues;
    // #131: the consult re-drive seed is consumed by the FIRST round's
    // PlanExecute walk only; later in-process rounds run normally.
    let consultSeed = consultResume;

    for (;;) {
      const cx: ExecutionContext = newExecutionContext(registry);
      cx.executor = this;
      cx.stream = currentStream;
      cx.signal = signal;
      cx.scratch.runSession = currentSession;
      cx.scratch.runBudget = currentBudget;
      cx.scratch.task = currentTask;
      cx.scratch.resumeContinues = seed;
      cx.scratch.consultResume = consultSeed;
      seed = undefined;
      consultSeed = undefined;

      const outcome = await runStrategy(currentTask.loop_strategy, cx);

      // A pause / escalate (or any fully-formed terminal) propagates verbatim —
      // preserves the HITL / consult / escalation contract and the strategy's
      // typed HaltReason + accounting through the recursive executor.
      if (cx.scratch.terminalOverride !== undefined) {
        return cx.scratch.terminalOverride;
      }
      switch (outcome.kind) {
        case "complete":
          return {
            kind: "success",
            output: outcome.output,
            session_id: sessionId,
            usage: cx.usage,
            turns: cx.scratch.runBudget.turns,
            session_state: cx.scratch.runSession,
          };
        case "failed":
          return {
            kind: "failure",
            reason: { kind: "configuration_error", error: outcome.error },
            session_id: sessionId,
            usage: cx.usage,
            turns: cx.scratch.runBudget.turns,
            session_state: cx.scratch.runSession,
          };
        case "budget_exhausted": {
          // #129: a BARE LEAF exhaustion propagates here carrying its CONFIGURED
          // `behavior` (Q1 — the bare-leaf resolution site honors it; the leaf
          // body never did). Resolve it ONCE: a `continue` with continues left
          // re-drives in-process; a spent `continue` falls through to
          // `fail`/`escalate`. Carry the resolution chain's state across rounds so
          // `max_continues` is respected. (A combinator under `surface_to_human`
          // never reaches this arm — it sets `terminalOverride`, returned above.)
          // #125: the exhausted node's own `stepsTaken` is the turn count it
          // reached (the scratch budget is not written back on the propagate
          // path). Fall back to the scratch turns if it is somehow 0.
          const turns = outcome.stepsTaken > 0 ? outcome.stepsTaken : cx.scratch.runBudget.turns;
          // Reconstruct the resolution scope: the FIRST round uses the leaf's
          // propagated behavior + continuesUsed; later in-process rounds reuse the
          // threaded (possibly fallen-through) state.
          const [resolveBehavior, resolveContinues] = behaviorForResolution ?? [
            outcome.behavior,
            outcome.continuesUsed,
          ];
          const scope = BudgetContext.resumed(
            outcome.policy,
            resolveBehavior,
            outcome.phase,
            resolveContinues,
          );
          const resolution = scope.resolveExhausted();
          if (resolution === "continue") {
            // In-process continue (AC3: NO serialization). Refresh the leaf's
            // step cap and re-enter the loop carrying the post-run session so the
            // conversation context survives (Continue PRESERVES context — AC4).
            // The granted cap is `stepsTaken + policy.value` so the leaf gets a
            // fresh window after the checkpoint.
            const granted = outcome.stepsTaken + (budgetPolicyAllowanceValue(outcome.policy) ?? 1);
            currentTask = grantTaskBudget(currentTask, granted);
            currentSession = cx.scratch.runSession;
            currentBudget = { ...cx.scratch.runBudget };
            currentStream = cx.stream;
            // Thread the resolution chain's post-continue state so a subsequent
            // exhaustion sees the bumped `continuesUsed`.
            behaviorForResolution = [scope.behavior, scope.continuesUsed];
            continue;
          }
          if (resolution === "escalate") {
            // #130: a BARE LEAF exhaustion under `surface_to_human` PAUSES with a
            // `budget_exhausted` request. A bare leaf offers
            // `[continue_with_budget, fail]` (fork C — no sibling to `skip` to).
            if (this.escalationMode().kind === "surface_to_human") {
              const err: BudgetExhausted = {
                policy: outcome.policy,
                behavior: scope.behavior,
                stepsTaken: outcome.stepsTaken,
                continuesUsed: scope.continuesUsed,
                phase: outcome.phase,
              };
              return promoteBudgetExhaustedToHuman(
                err,
                outcome.partialOutput,
                leafEscalationActions(err),
                sessionId,
                currentTask,
                cx.scratch.runBudget,
                turns,
              );
            }
            return {
              kind: "failure",
              reason: { kind: "budget_exceeded", limit_type: "turns" },
              session_id: sessionId,
              usage: cx.usage,
              turns,
              // #125: carry the node-concrete partial as an assistant text message
              // so a parent / caller can inspect what was produced before
              // exhaustion.
              session_state:
                outcome.partialOutput != null
                  ? {
                      messages: [
                        {
                          role: "assistant",
                          content: { type: "text", text: outcome.partialOutput },
                        },
                      ],
                      extras: {},
                    }
                  : emptySessionState(),
            };
          }
          // Fail contract: the partial is DISCARDED.
          return {
            kind: "failure",
            reason: { kind: "budget_exceeded", limit_type: "turns" },
            session_id: sessionId,
            usage: cx.usage,
            turns,
            session_state: emptySessionState(),
          };
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // StrategyExecutor (#124): the harness-side primitives the recursive
  // per-variant RunStrategy.run bodies delegate to. Each wraps an existing,
  // tested orchestration method so behavior stays at parity (AC6) — the only
  // structural change is that the per-variant bodies now own their loops and
  // the central dispatch switch is gone.
  // --------------------------------------------------------------------------

  /** {@inheritDoc StrategyExecutor.reactWindow} */
  reactWindow(
    task: Task,
    maxIterations: number,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
    agent: Agent,
  ): Promise<RunResult> {
    // The leaf seeds the task instruction (parity with the old top-level ReAct
    // entry, which called runReact(.., seedInstruction: true)). The resolved
    // worker `agent` (#124) drives every turn of this window.
    return this.runReactInner(
      task,
      maxIterations,
      sessionState,
      budgetUsed,
      onStream,
      signal,
      true,
      agent,
    );
  }

  /** {@inheritDoc StrategyExecutor.resolveAgentRef} */
  resolveAgentRef(ref: AgentRef, sessionId: SessionId): Agent | RunResult {
    const registry = this.config.registry ?? ExecutionRegistry.empty();
    const agent = registry.resolveAgent(ref);
    if (agent !== undefined) return agent;
    return {
      kind: "failure",
      reason: { kind: "configuration_error", error: new UnresolvedHandle("agent", ref) },
      session_id: sessionId,
      usage: emptyAggregateUsage(),
      turns: 0,
      session_state: emptySessionState(),
    };
  }

  /** {@inheritDoc StrategyExecutor.evaluatePhase} */
  evaluatePhase(
    task: Task,
    evalAgent: Agent,
    carried: BudgetSnapshot,
    totalUsage: AggregateUsage,
  ): Promise<RunResult> {
    return this.runEvaluatePhase(task, evalAgent, carried, totalUsage);
  }

  /** {@inheritDoc StrategyExecutor.appendUserMessage} */
  async appendUserMessage(sessionState: SessionState, text: string): Promise<void> {
    await this.config.contextManager.appendUserMessage(sessionState, text);
  }

  /** {@inheritDoc StrategyExecutor.workspaceRoot} */
  workspaceRoot(): string {
    return this.config.sandbox.workspaceRoot?.() ?? "";
  }

  /**
   * Resolve the worker agent for a {@link LoopStrategy} tree from the
   * {@link ExecutionRegistry} (#124). The worker is the agent on the LEAF
   * reached by descending the recursion: a `react` leaf's `agent`; a combinator
   * descends into its primary worker child (`inner` / `execute`). A `ralph` with
   * a non-empty `agent` override resolves THAT (Q3). Returns the resolved agent,
   * or a typed `configuration_error` failure {@link RunResult}. Used by the
   * resume paths, which run a worker ReAct window directly.
   */
  private resolveWorkerAgent(ls: LoopStrategy, sessionId: SessionId): Agent | RunResult {
    return this.resolveAgentRef(StandardHarness.workerAgentKey(ls), sessionId);
  }

  /** The agent key the worker leaf of `ls` references
   *  (see {@link resolveWorkerAgent}). */
  private static workerAgentKey(ls: LoopStrategy): string {
    switch (ls.kind) {
      case "react":
        return ls.agent;
      case "plan_execute":
        return StandardHarness.workerAgentKey(ls.execute);
      case "self_verifying":
        return StandardHarness.workerAgentKey(ls.inner);
      case "ralph":
        return ls.agent !== "" ? ls.agent : StandardHarness.workerAgentKey(ls.inner);
      case "hill_climbing":
        return StandardHarness.workerAgentKey(ls.inner);
    }
  }

  /** {@inheritDoc StrategyExecutor.planDirective} */
  planDirective(instruction: string): string {
    return (
      "Produce a step-by-step plan for the following task. Respond with a " +
      'single JSON object: {"tasks": [<ordered step strings>], ' +
      '"rationale": <string>}.\n\nTask:\n' +
      instruction
    );
  }

  /** {@inheritDoc StrategyExecutor.seedUserMessage} */
  async seedUserMessage(sessionState: SessionState, text: string): Promise<void> {
    await this.config.contextManager.appendUserMessage(sessionState, text);
  }

  /** {@inheritDoc StrategyExecutor.runPlanSubtree} */
  async runPlanSubtree(
    plan: LoopStrategy,
    planTask: Task,
    planSession: SessionState,
    budgetUsed: BudgetSnapshot,
    signal: AbortSignal | undefined,
  ): Promise<RunResult | undefined> {
    // #124 Q1: the planner concept is DROPPED — the plan child's leaf
    // `ReactConfig.agent` is authoritative and resolved by the recursing leaf
    // itself. The child's `.run(cx)` drives the WHOLE plan loop (genuine
    // recursion).
    const registry = this.config.registry ?? ExecutionRegistry.empty();
    const cx: ExecutionContext = newExecutionContext(registry);
    cx.executor = this;
    cx.signal = signal;
    cx.scratch.runSession = planSession;
    cx.scratch.runBudget = budgetUsed;
    cx.scratch.task = planTask;
    await runStrategy(plan, cx);
    return cx.scratch.terminalOverride;
  }

  /** {@inheritDoc StrategyExecutor.capturePlanArtifact} */
  capturePlanArtifact(
    sessionId: SessionId,
    planOutput: string,
    usage: AggregateUsage,
    turns: number,
    signal: AbortSignal | undefined,
  ): Promise<PlanPhaseOutcome> {
    return this.captureAndPersistPlan(sessionId, planOutput, usage, turns, signal);
  }

  /** {@inheritDoc StrategyExecutor.reconcileCompletedTasks} */
  reconcileCompletedTasks(sessionId: SessionId, taskList: TaskList): Promise<void> {
    return this.reconcileDeepResume(sessionId, taskList);
  }

  /** {@inheritDoc StrategyExecutor.fireTaskAdvance} */
  async fireTaskAdvance(
    sessionId: SessionId,
    stepTask: Task,
    taskIndex: number,
    totalTasks: number,
    signal: AbortSignal | undefined,
  ): Promise<void> {
    if (!this.config.hooks) return;
    const ctx: HookContext = {
      event: "on_task_advance",
      session_id: sessionId,
      task: stepTask,
      task_index: taskIndex,
      total_tasks: totalTasks,
    };
    try {
      await this.config.hooks.fire(ctx, signal);
    } catch {
      // Hook errors are non-fatal; the step proceeds with the current task.
    }
  }

  /**
   * Test-facing driver (#124) for the GENUINE recursive execute phase: drains
   * `taskList` by dispatching the PlanExecute strategy's `execute` child via
   * `runStrategy(execute, cx)` per task. Reproduces the execute half of the
   * combinator {@link runPlanExecuteConfig} so the granular execute regression
   * tests exercise the real per-task `execute.run(cx)` dispatch (the phase logic
   * is NOT duplicated for production — production runs through the combinator).
   * `task.loop_strategy` MUST be a `plan_execute`.
   */
  async executePhase(
    task: Task,
    sessionState: SessionState,
    taskList: TaskList,
    carried: BudgetSnapshot,
    planUsage: AggregateUsage,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    if (task.loop_strategy.kind !== "plan_execute") {
      throw new InvalidConfiguration("executePhase requires a plan_execute strategy");
    }
    const executeChild = task.loop_strategy.execute;
    const registry = this.config.registry ?? ExecutionRegistry.empty();
    const sessionId = task.session_id;

    await this.reconcileDeepResume(sessionId, taskList);

    const totalTasks = taskList.tasks.length;
    const totalUsage: AggregateUsage = { ...planUsage };
    const sharedCarried: BudgetSnapshot = { ...carried };
    let shared = sessionState;
    let lastOutput = "";
    let lastState: SessionState = emptySessionState();

    for (let index = 0; index < totalTasks; index += 1) {
      const taskId = taskList.tasks[index]!.id;
      const instruction = taskList.tasks[index]!.description;

      if (taskList.tasks[index]!.status === "completed") {
        lastOutput = instruction;
        continue;
      }

      // #125: the per-task `remainingTurns / remainingTasks / stepCap` derivation
      // is REMOVED (dead) — enforcement is now charge-based on the PlanExecute
      // scope (see `runPlanExecuteConfig`). This helper mirrors the live body's
      // structure and passes the task budget through verbatim.
      updateTask(taskList, taskId, "in_progress");
      await this.persistTaskList(sessionId, taskList);

      const stepTask: Task = {
        id: task.id,
        instruction,
        session_id: sessionId,
        budget: { ...task.budget },
        loop_strategy: executeChild,
      };
      await this.fireTaskAdvance(sessionId, stepTask, index, totalTasks, signal);

      await this.seedUserMessage(shared, stepTask.instruction);

      const cx: ExecutionContext = newExecutionContext(registry);
      cx.executor = this;
      cx.signal = signal;
      cx.scratch.runSession = shared;
      cx.scratch.runBudget = { ...sharedCarried };
      cx.scratch.task = stepTask;
      await runStrategy(executeChild, cx);
      const subResult = cx.scratch.terminalOverride;

      if (subResult != null && subResult.kind === "success") {
        sharedCarried.turns = subResult.turns;
        shared = runResultSessionState(subResult);
        lastState = { messages: [...shared.messages], extras: { ...shared.extras } };
        sharedCarried.input_tokens += subResult.usage.input_tokens;
        sharedCarried.output_tokens += subResult.usage.output_tokens;
        totalUsage.input_tokens += subResult.usage.input_tokens;
        totalUsage.output_tokens += subResult.usage.output_tokens;
        totalUsage.cache_read_tokens += subResult.usage.cache_read_tokens;
        totalUsage.cache_write_tokens += subResult.usage.cache_write_tokens;
        totalUsage.cost_usd += subResult.usage.cost_usd;
        lastOutput = subResult.output;
        completeTask(taskList, taskId);
        await this.persistTaskList(sessionId, taskList);
        emit(onStream, { kind: "final_response", content: lastOutput });
      } else if (subResult != null && subResult.kind === "failure") {
        totalUsage.input_tokens += subResult.usage.input_tokens;
        totalUsage.output_tokens += subResult.usage.output_tokens;
        totalUsage.cache_read_tokens += subResult.usage.cache_read_tokens;
        totalUsage.cache_write_tokens += subResult.usage.cache_write_tokens;
        totalUsage.cost_usd += subResult.usage.cost_usd;
        updateTask(taskList, taskId, "blocked");
        await this.persistTaskList(sessionId, taskList);
        const terminalReason: HaltReason =
          subResult.reason.kind === "budget_exceeded"
            ? subResult.reason
            : {
                kind: "step_failed",
                task_index: index,
                task: taskList.tasks[index]!.description,
                reason: haltReasonToString(subResult.reason),
              };
        return {
          kind: "failure",
          reason: terminalReason,
          session_id: sessionId,
          usage: totalUsage,
          turns: subResult.turns,
          session_state: lastState,
        };
      } else if (subResult != null) {
        return subResult;
      } else {
        return {
          kind: "failure",
          reason: {
            kind: "step_failed",
            task_index: index,
            task: taskList.tasks[index]!.description,
            reason: "execute sub-strategy produced no terminal",
          },
          session_id: sessionId,
          usage: totalUsage,
          turns: sharedCarried.turns,
          session_state: lastState,
        };
      }
    }

    return {
      kind: "success",
      output: lastOutput,
      session_id: sessionId,
      usage: totalUsage,
      turns: sharedCarried.turns,
      session_state: lastState,
    };
  }

  /** {@inheritDoc StrategyExecutor.ralphSeedSession} */
  async ralphSeedSession(instruction: string): Promise<SessionState> {
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    const sessionState = emptySessionState();
    await this.config.contextManager.appendUserMessage(sessionState, instruction);
    const reload = StandardHarness.ralphReloadContext(workspaceRoot);
    if (reload != null) {
      await this.config.contextManager.appendUserMessage(sessionState, reload);
    }
    // R3 (issue #58 v2): inject git history when a VcsProvider is wired.
    if (this.config.vcsProvider != null) {
      const args: VcsLogArgs = { maxEntries: 20 };
      try {
        const trimmed = (await this.config.vcsProvider.log(args)).trim();
        if (trimmed.length > 0) {
          await this.config.contextManager.appendUserMessage(
            sessionState,
            `Recent VCS history:\n${trimmed}`,
          );
        }
      } catch {
        // A VCS read failure is non-fatal: skip the git section and continue.
      }
    }
    return sessionState;
  }

  /** {@inheritDoc StrategyExecutor.ralphCompletionStatus} */
  ralphCompletionStatus(): string | null {
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    return StandardHarness.ralphCompletionStatus(workspaceRoot);
  }

  /** {@inheritDoc StrategyExecutor.ralphMaxResets} */
  ralphMaxResets(): number {
    return this.config.maxResets ?? DEFAULT_MAX_RESETS;
  }

  /** {@inheritDoc StrategyExecutor.resolveMetricEvaluator} */
  resolveMetricEvaluator(key: string, sessionId: SessionId): MetricEvaluator | RunResult {
    const registry = this.config.registry ?? ExecutionRegistry.empty();
    const evaluator = registry.resolveMetricEvaluator(key);
    if (evaluator !== undefined) return evaluator;
    return {
      kind: "failure",
      reason: {
        kind: "hill_climbing_misconfigured",
        reason: `hill_climbing requires a metric evaluator registered under key ${JSON.stringify(key)}`,
      },
      session_id: sessionId,
      usage: emptyAggregateUsage(),
      turns: 0,
      session_state: emptySessionState(),
    };
  }

  /** {@inheritDoc StrategyExecutor.hillBaseline} */
  async hillBaseline(
    evaluator: MetricEvaluator,
    sessionId: SessionId,
    taskId: TaskId,
    direction: HillClimbingDirection,
    rows: ResultsEntry[],
    spanSeq: { value: number },
    totalUsage: AggregateUsage,
    turns: number,
    signal: AbortSignal | undefined,
  ): Promise<{ ok: true; value: number } | { ok: false; failure: RunResult }> {
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    const description = evaluator.description();
    const snapshot = newSessionStateSnapshot(sessionId, taskId, emptySessionState(), workspaceRoot);
    const baseline = await evaluator.evaluate(this.config.sandbox, snapshot, signal);
    if (baseline.kind === "ok") {
      const value = baseline.result.value;
      rows.push({
        iteration: 0,
        commit_hash: await this.hillClimbingCommitHash(),
        metric_value: value,
        direction,
        status: "kept",
        duration: baseline.result.duration,
        description,
        metadata: {},
      });
      this.emitHillClimbingSpan(sessionId, taskId, spanSeq, 0, value, null, "kept", false);
      return { ok: true, value };
    }
    // D7: a baseline that cannot even be measured is a misconfiguration.
    const status = iterationStatusFromError(baseline.error);
    rows.push({
      iteration: 0,
      commit_hash: await this.hillClimbingCommitHash(),
      metric_value: NaN,
      direction,
      status,
      duration: 0,
      description,
      metadata: {},
    });
    this.emitHillClimbingSpan(sessionId, taskId, spanSeq, 0, null, null, status, false);
    await this.writeHillClimbingTsv(workspaceRoot, taskId, rows);
    return {
      ok: false,
      failure: {
        kind: "failure",
        reason: {
          kind: "hill_climbing_misconfigured",
          reason: `baseline evaluation failed: ${metricErrorMessage(baseline.error)}`,
        },
        session_id: sessionId,
        usage: totalUsage,
        turns,
        session_state: emptySessionState(),
      },
    };
  }

  /** {@inheritDoc StrategyExecutor.hillIteration} */
  async hillIteration(
    evaluator: MetricEvaluator,
    sessionId: SessionId,
    taskId: TaskId,
    iteration: number,
    direction: HillClimbingDirection,
    revertOnNoImprovement: boolean,
    minImprovementDelta: number | undefined,
    currentBest: number,
    rows: ResultsEntry[],
    spanSeq: { value: number },
    signal: AbortSignal | undefined,
  ): Promise<{ currentBest: number; nonImprovement: boolean }> {
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    const description = evaluator.description();
    const snapshot = newSessionStateSnapshot(sessionId, taskId, emptySessionState(), workspaceRoot);
    const evalOutcome = await evaluator.evaluate(this.config.sandbox, snapshot, signal);
    if (evalOutcome.kind === "ok") {
      const value = evalOutcome.result.value;
      const kept = shouldKeep(value, currentBest, direction, minImprovementDelta ?? null);
      const delta = direction === "minimize" ? currentBest - value : value - currentBest;
      if (kept) {
        rows.push({
          iteration,
          commit_hash: await this.hillClimbingCommitHash(),
          metric_value: value,
          direction,
          status: "kept",
          duration: evalOutcome.result.duration,
          description,
          metadata: {},
        });
        this.emitHillClimbingSpan(
          sessionId,
          taskId,
          spanSeq,
          iteration,
          value,
          delta,
          "kept",
          false,
        );
        return { currentBest: value, nonImprovement: false };
      }
      const reverted = revertOnNoImprovement;
      if (reverted) await this.hillClimbingRevert(signal);
      rows.push({
        iteration,
        commit_hash: await this.hillClimbingCommitHash(),
        metric_value: value,
        direction,
        status: "discarded",
        duration: evalOutcome.result.duration,
        description,
        metadata: {},
      });
      this.emitHillClimbingSpan(
        sessionId,
        taskId,
        spanSeq,
        iteration,
        value,
        delta,
        "discarded",
        reverted,
      );
      return { currentBest, nonImprovement: true };
    }
    // Crash/timeout/etc.: counts as a non-improvement.
    const status = iterationStatusFromError(evalOutcome.error);
    const reverted = revertOnNoImprovement;
    if (reverted) await this.hillClimbingRevert(signal);
    rows.push({
      iteration,
      commit_hash: await this.hillClimbingCommitHash(),
      metric_value: NaN,
      direction,
      status,
      duration: 0,
      description,
      metadata: {},
    });
    this.emitHillClimbingSpan(sessionId, taskId, spanSeq, iteration, null, null, status, reverted);
    return { currentBest, nonImprovement: true };
  }

  /** {@inheritDoc StrategyExecutor.hillWriteTsv} */
  async hillWriteTsv(taskId: TaskId, rows: ResultsEntry[]): Promise<void> {
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    await this.writeHillClimbingTsv(workspaceRoot, taskId, rows);
  }

  /** {@inheritDoc StrategyExecutor.budgetExceeded} */
  budgetExceeded(budget: BudgetLimits, used: BudgetSnapshot): BudgetLimitType | null {
    return budgetExceeded(budget, used, Date.now());
  }

  /** {@inheritDoc StrategyExecutor.finalize} */
  async finalize(result: RunResult): Promise<void> {
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
      case "escalate":
        await this.finalizeObservability(result.session_id, { kind: "escalated" });
        break;
      // Non-terminal pauses are never finalized.
      case "waiting_for_human":
      case "consult":
        break;
    }
  }

  /** {@inheritDoc StrategyExecutor.escalationMode} */
  escalationMode(): EscalationMode {
    // Mirrors Rust `StandardHarness::escalation_mode` (returns
    // `self.config.escalation_mode`). In Rust the field is non-optional; here it
    // is optional, so a raw {@link HarnessConfig} that omits the knob falls back
    // to {@link autonomous} — preserving the pre-#130 propagate behavior (no
    // pause) for legacy callers that never set it. The {@link HarnessBuilder}
    // explicitly defaults the knob to {@link surfaceToHuman}, so builder-based
    // callers opt into HITL.
    return this.config.escalationMode ?? autonomous;
  }

  async resume(
    state: PausedState,
    response: HumanResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    const result = await this.resumeInner(state, response, onStream, signal);
    await this.autoPersistTerminal(result);
    return result;
  }

  private async resumeInner(
    state: PausedState,
    response: HumanResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    const sessionState = state.session_state;
    const pendingCalls = state.pending_tool_calls;

    // Budget-escalation resume (#130/#129): if this pause came from
    // `budget_exhausted`, map the operator's `EscalationAction` BEFORE the
    // generic `switch (response)` (mirrors the clarification branch). The node's
    // budget context is reconstructed from the request fields: `steps_taken`
    // raises the granted cap, and `continues_used` (#129, AC2) is the SOLE budget
    // field that rides a process pause — it is seeded back into the rebuilt scope
    // so a `continue` spanning the pause resumes with the correct continue count
    // (it cannot exceed `max_continues`). Q3: `continues_used` rides the REQUEST
    // payload, NOT a new serialized budget / PausedState field.
    if (state.human_request?.kind === "budget_exhausted" && response.kind === "escalate") {
      const stepsTaken = state.human_request.steps_taken;
      const resumeSeed: [string, number] = [
        state.human_request.phase,
        state.human_request.continues_used,
      ];
      const action: EscalationAction = response.action;
      switch (action.kind) {
        // Grant `steps` ADDITIONAL allowance and re-enter the loop from the
        // restored checkpoint. The strategy tree is rebuilt with the node's
        // budget policy raised to `stepsTaken + steps` so the restored scope has
        // room for `steps` more steps, and the resumed scope's `continuesUsed` is
        // seeded from the request (#129, AC2).
        case "continue_with_budget": {
          const granted = stepsTaken + action.steps;
          const resumedTask = grantTaskBudget(state.task, granted);
          return this.driveStrategyWithResumeSeed(
            resumedTask,
            sessionState,
            state.budget_used,
            onStream,
            signal,
            resumeSeed,
            undefined,
          );
        }
        // Skip: the node is marked skipped and the outer loop advances. For a
        // combinator (PlanExecute) re-entering the loop from the checkpoint
        // advances to the remaining ready tasks. For a leaf there is no sibling,
        // so a skip resolves to a clean (empty) Success carrying whatever partial
        // history was captured.
        case "skip": {
          if (state.task.loop_strategy.kind === "plan_execute") {
            return this.driveStrategy(
              state.task,
              sessionState,
              state.budget_used,
              onStream,
              signal,
            );
          }
          return {
            kind: "success",
            output: "",
            session_id: state.session_id,
            usage: emptyAggregateUsage(),
            turns: state.turn_number,
            session_state: sessionState,
          };
        }
        // Fail: abort the node and propagate `budget_exceeded`; the partial is
        // discarded (the `fail` resolution contract).
        case "fail":
          return {
            kind: "failure",
            reason: { kind: "budget_exceeded", limit_type: "turns" },
            session_id: state.session_id,
            usage: emptyAggregateUsage(),
            turns: state.turn_number,
            session_state: emptySessionState(),
          };
      }
    }

    // Resolve the effective tool registry for this resumed session — bridges
    // catalogue tools the same way the turn loop does (issue #91), so pending
    // tool calls dispatched during resume thread the run's storage + sandbox.
    const toolRegistry = this.effectiveToolRegistry(state.session_id);

    // Subagent depth: if there's a child, the caller-installed SubagentTool
    // owns the dispatch back into the child harness; without #4/#5 wired up
    // the harness round-trips the child state but continues the parent loop.
    // This matches the Rust reference (placeholder until #4/#5 land).
    if (state.child_state != null) {
      // Intentional no-op; the full child.resume() dispatch lives in #4/#5.
    }

    // Clarification resume (issue #81, Q4b): if this pause came from
    // `awaiting_clarification`, the human's answer is injected as the tool
    // RESULT for the clarifying call (the head of `pending_tool_calls`) — NOT
    // appended as a free-standing user message. Any remaining pending calls
    // from the same batch are then dispatched normally before the loop resumes.
    if (
      state.human_request?.kind === "clarification" &&
      (response.kind === "answer" || response.kind === "approve_with_feedback")
    ) {
      const text = response.kind === "answer" ? response.text : response.feedback;
      const [clarifyingCall, ...remaining] = pendingCalls;
      if (clarifyingCall) {
        const tr: ToolResultRecord = {
          call_id: clarifyingCall.id,
          output: { kind: "success", content: text, truncated: false },
        };
        await this.config.contextManager.appendToolResult(sessionState, tr);
      }
      for (const call of remaining) {
        const output = await toolRegistry.dispatch(call, signal);
        const tr: ToolResultRecord = { call_id: call.id, output };
        await this.config.contextManager.appendToolResult(sessionState, tr);
      }
      const maxIter =
        state.task.loop_strategy.kind === "react"
          ? reactMaxIterations(state.task.loop_strategy)
          : Number.MAX_SAFE_INTEGER;
      const agent = this.resolveWorkerAgent(state.task.loop_strategy, state.session_id);
      if (!("turn" in agent)) return agent;
      return this.runReact(
        state.task,
        maxIter,
        sessionState,
        state.budget_used,
        onStream,
        signal,
        false,
        agent,
      );
    }

    switch (response.kind) {
      case "halt":
        return {
          kind: "failure",
          reason: { kind: "human_halted" },
          session_id: state.session_id,
          usage: emptyAggregateUsage(),
          turns: state.turn_number,
          session_state: sessionState,
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
          const output = await toolRegistry.dispatch(call, signal);
          const tr: ToolResultRecord = { call_id: call.id, output };
          await this.config.contextManager.appendToolResult(sessionState, tr);
        }
        break;
      }

      case "allow_with_modification": {
        for (const call of response.calls) {
          const output = await toolRegistry.dispatch(call, signal);
          const tr: ToolResultRecord = { call_id: call.id, output };
          await this.config.contextManager.appendToolResult(sessionState, tr);
        }
        break;
      }

      // #130: an `escalate` response paired with a `budget_exhausted` request is
      // handled by the dedicated branch ABOVE, which returns before this switch.
      // An `escalate` reaching here is therefore out of contract — the operator
      // supplied an `EscalationAction` for a `tool_approval` / `review` / `none`
      // pause — so treat it conservatively as a budget-exceeded failure rather
      // than silently re-entering the loop.
      case "escalate":
        return {
          kind: "failure",
          reason: { kind: "budget_exceeded", limit_type: "turns" },
          session_id: state.session_id,
          usage: emptyAggregateUsage(),
          turns: state.turn_number,
          session_state: emptySessionState(),
        };

      default: {
        const _exhaustive: never = response;
        return _exhaustive;
      }
    }

    const max =
      state.task.loop_strategy.kind === "react"
        ? reactMaxIterations(state.task.loop_strategy)
        : Number.MAX_SAFE_INTEGER;
    const agent = this.resolveWorkerAgent(state.task.loop_strategy, state.session_id);
    if (!("turn" in agent)) return agent;
    return this.runReact(
      state.task,
      max,
      sessionState,
      state.budget_used,
      onStream,
      signal,
      false,
      agent,
    );
  }

  /**
   * Resume a worker paused by {@link RunResult} `consult` (issue #114). The
   * resume seam parallel to {@link resume}: it injects the {@link ConsultResponse}
   * text as the tool RESULT for the head pending (consult) call — NOT appended as
   * a free-standing user message (R10) — then dispatches any remaining pending
   * calls and resumes the ReAct loop.
   */
  async resumeConsult(
    state: PausedState,
    response: ConsultResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    const result = await this.resumeConsultInner(state, response, onStream, signal);
    await this.autoPersistTerminal(result);
    return result;
  }

  private async resumeConsultInner(
    state: PausedState,
    response: ConsultResponse,
    onStream?: StreamSink,
    signal?: AbortSignal,
  ): Promise<RunResult> {
    const sessionState = state.session_state;
    const [text, answered] =
      response.kind === "answer"
        ? ([response.text, true] as const)
        : ([response.message, false] as const);

    // Observability: lightweight consult-resume event.
    if (this.config.observability) {
      // Recover the consult `kind` from the head pending call's args, if present.
      const head = state.pending_tool_calls[0];
      const input = head?.input as Record<string, unknown> | undefined;
      const kind = typeof input?.kind === "string" ? input.kind : "";
      const base = newRootSpanBase(
        SpanId.of(`${state.session_id.asString()}-consult-resume`),
        state.session_id,
        state.task.id,
        "context_assembly",
        Timestamp.now(),
      );
      const span: ContextSpan = {
        base,
        operation: { kind: "consult_resumed", consult_kind: kind, answered },
        tokens_before: 0,
        tokens_after: 0,
        utilization_before: 0,
        utilization_after: 0,
      };
      this.config.observability.emitContext(span);
    }

    const toolRegistry = this.effectiveToolRegistry(state.session_id);

    // Inject the consult answer as the RESULT of the head pending (consult) call,
    // then dispatch the remaining pending calls in the same batch.
    const [consultCall, ...remaining] = state.pending_tool_calls;
    if (consultCall) {
      const tr: ToolResultRecord = {
        call_id: consultCall.id,
        output: { kind: "success", content: text, truncated: false },
      };
      await this.config.contextManager.appendToolResult(sessionState, tr);
    }
    for (const call of remaining) {
      const output = await toolRegistry.dispatch(call, signal);
      const tr: ToolResultRecord = { call_id: call.id, output };
      await this.config.contextManager.appendToolResult(sessionState, tr);
    }

    // #131: a consult that surfaced from inside a composed tree carries the FULL
    // strategy in `task.loop_strategy` (each combinator's `finish` rewrote the
    // pause's task on the way up). Re-DRIVE that strategy rather than resuming
    // only the worker leaf: the PlanExecute walk resumes its in-progress task
    // from the injected worker session (`consultResume` seed), so the worker
    // finishes mid-loop, its SelfVerifying evaluator runs, the task is marked
    // completed, and the remaining ready-set is walked. A BARE worker leaf
    // (depth-1, e.g. a SubagentTool-mediated consult) has no surrounding walk, so
    // it keeps the original leaf-only resume (back-compat).
    if (state.task.loop_strategy.kind !== "react") {
      return this.driveStrategyWithResumeSeed(
        state.task,
        // Top-level session starts fresh; the worker conversation is threaded
        // into the in-progress task via the consult seed.
        emptySessionState(),
        state.budget_used,
        onStream,
        signal,
        undefined,
        sessionState,
      );
    }

    const max =
      state.task.loop_strategy.kind === "react"
        ? reactMaxIterations(state.task.loop_strategy)
        : Number.MAX_SAFE_INTEGER;
    const agent = this.resolveWorkerAgent(state.task.loop_strategy, state.session_id);
    if (!("turn" in agent)) return agent;
    return this.runReact(
      state.task,
      max,
      sessionState,
      state.budget_used,
      onStream,
      signal,
      false,
      agent,
    );
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
  // PlanExecute (issues #70 plan phase + #59 execute phase)
  // --------------------------------------------------------------------------

  /**
   * Persist the parsed {@link TaskList} for the run (Q4). The DURABLE write goes
   * through the `RunStore` seam under `TASK_LIST_EXTRAS_KEY`; the #71
   * sandbox-filesystem path (`.spore/task_list.json`) is intentionally NOT used
   * — the `RunStore` write is the single source of truth (#76 removed the
   * redundant `SessionState.extras` mirror). Failures are swallowed: a
   * successful plan must not be lost to a storage hiccup (the default no-op /
   * in-memory provider never fails).
   */
  async persistTaskList(sessionId: SessionId, taskList: TaskList): Promise<void> {
    // Serialize via the canonical task-list form, then re-parse to a plain JSON
    // value so the durable blob is byte-identical to the cross-language
    // `{ tasks, next_id }` shape.
    let value: unknown;
    try {
      value = JSON.parse(serializeTaskList(taskList));
    } catch {
      return; // Serialization hiccup — never lose a successful plan to it.
    }
    try {
      await this.storage()
        .run()
        .put(sessionId, TASK_LIST_EXTRAS_KEY, value as never);
    } catch {
      // Durable write failure is non-fatal.
    }
  }

  /**
   * Load the persisted {@link TaskList} from the RunStore under
   * `TASK_LIST_EXTRAS_KEY` (#126, decision C): the ONE authoring path that can
   * carry real `blockers`. A storage miss / deserialize failure yields
   * `undefined` (the DAG executor then falls back to the linear plan-artifact
   * bridge). Mirrors Rust's `load_task_list`.
   */
  async loadTaskList(sessionId: SessionId): Promise<TaskList | undefined> {
    try {
      const saved = await this.storage().run().get(sessionId, TASK_LIST_EXTRAS_KEY);
      if (saved == null) return undefined;
      const parsed = TaskListSchema.safeParse(saved);
      return parsed.success ? parsed.data : undefined;
    } catch {
      return undefined;
    }
  }

  /**
   * #126 (AC2): record a `write_file` / `edit_file` tool call's `path` into the
   * observed-write accumulator for the current step. De-duplicated against what
   * is already accumulated. Called from the ReAct dispatch seam for the call
   * ACTUALLY dispatched, so the path comes from the real tool call — never a
   * model-self-reported field. Mirrors Rust's `observe_write_call`.
   */
  observeWriteCall(call: ToolCall): void {
    if (call.name !== "write_file" && call.name !== "edit_file") return;
    const input = call.input as Record<string, unknown> | undefined;
    const path = input?.["path"];
    if (typeof path !== "string") return;
    if (!this.observedWrites.includes(path)) this.observedWrites.push(path);
  }

  /** {@inheritDoc StrategyExecutor.takeObservedWrites} */
  takeObservedWrites(): string[] {
    const acc = this.observedWrites;
    this.observedWrites = [];
    return acc;
  }

  /** {@inheritDoc StrategyExecutor.clearObservedWrites} */
  clearObservedWrites(): void {
    this.observedWrites = [];
  }

  /**
   * A.6 deep-resume reconcile (#124, Q2): mark every task already `completed` on
   * the DURABLE RunStore checkpoint as `completed` in the freshly-parsed
   * `taskList` so it is NOT re-run. Tasks are matched by `id` (the list is
   * regenerated deterministically from the same plan artifact). A checkpoint read
   * hiccup must not block a fresh execute run.
   */
  private async reconcileDeepResume(sessionId: SessionId, taskList: TaskList): Promise<void> {
    try {
      const saved = await this.storage().run().get(sessionId, TASK_LIST_EXTRAS_KEY);
      if (saved == null) return;
      const parsed = TaskListSchema.safeParse(saved);
      if (!parsed.success) return;
      const completed = new Set(
        parsed.data.tasks.filter((s) => s.status === "completed").map((s) => s.id),
      );
      for (const t of taskList.tasks) {
        if (completed.has(t.id)) {
          t.status = "completed";
        }
      }
    } catch {
      // A checkpoint read hiccup must not block a fresh execute run.
    }
  }

  /**
   * Capture + persist a {@link PlanArtifact} from the plan child's output text
   * (#124): R3 (parse), R11 (fire `on_plan_created`, mutable), R4 (persist to the
   * RunStore under `PLAN_EXECUTE_EXTRAS_KEY`). The model turn that produced
   * `planOutput` ran elsewhere — the recursive `plan.run(cx)` child — so this
   * carries no agent call. Returns the captured artifact + accounting on success,
   * or a terminal `failure` `RunResult` to propagate.
   */
  private async captureAndPersistPlan(
    sessionId: SessionId,
    planOutput: string,
    usage: AggregateUsage,
    turns: number,
    signal: AbortSignal | undefined,
  ): Promise<PlanPhaseOutcome> {
    // R3: capture the artifact from the response text.
    const captured = capturePlanArtifact(planOutput);
    if (!captured.ok) {
      return {
        ok: false,
        failure: failure(
          { kind: "plan_phase_failed", error: captured.error.detail },
          sessionId,
          usage,
          turns,
        ),
      };
    }

    // R11: fire on_plan_created synchronously; the hook may rewrite the artifact
    // — either in place OR by reassigning `ctx.plan`. Read the final value back
    // off `ctx` so either path is honored. Errors are non-fatal: a successfully-
    // captured plan is not lost to a handler error.
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

    // R4: persist the produced artifact to the RunStore seam under
    // PLAN_EXECUTE_EXTRAS_KEY (the stable cross-language `{ tasks, rationale }`
    // shape). #76 — the durable single source of truth, no longer mirrored into
    // SessionState.extras. The put failure is swallowed (matching persistTaskList):
    // a successfully-captured plan must not be lost to a storage hiccup.
    try {
      await this.storage().run().put(sessionId, PLAN_EXECUTE_EXTRAS_KEY, {
        tasks: artifact.tasks,
        rationale: artifact.rationale,
      });
    } catch {
      // Durable write failure is non-fatal.
    }

    return { ok: true, artifact, usage, turns };
  }

  /**
   * Run the `self_verifying` evaluate phase (issue #61): a fresh evaluator RUN
   * over a read-only sandbox in a never-shared session.
   *
   * Builds a child {@link StandardHarness} from a clone of `this.config` with the
   * `agent` swapped to the evaluator agent (D2 defaulting) and the `sandbox`
   * wrapped in a {@link ReadOnlySandbox} (R3). The evaluator runs a fresh ReAct
   * loop seeded with the `role-evaluator` chunk (R4, presence-only) plus a review
   * directive, in a freshly generated session (R2/R9). Folds the evaluate run's
   * usage into `totalUsage` / `carried` (R8) and returns its terminal
   * {@link RunResult}.
   */
  private async runEvaluatePhase(
    task: Task,
    evaluator: Agent,
    carried: BudgetSnapshot,
    totalUsage: AggregateUsage,
  ): Promise<RunResult> {
    // R3: derive a read-only sandbox internally from the build sandbox.
    const readOnlySandbox: SandboxProvider = new ReadOnlySandbox(this.config.sandbox);

    // R2/R9: fresh, never-shared session id for the evaluate run.
    const evalSessionId = SessionId.generate();

    // R4 (presence-only): prepend the `role-evaluator` chunk content (if the
    // configured provider supplies it) to the review directive.
    const roleChunk = await this.roleEvaluatorChunk();
    const reviewBody =
      "Review the work produced for the following task and report whether it is " +
      "correct. You did NOT write this code; default to FAIL unless you can " +
      `confirm it is right.\n\nTask:\n${task.instruction}`;
    const directive = roleChunk != null ? `${roleChunk}\n\n${reviewBody}` : reviewBody;

    const evalTask: Task = {
      id: TaskId.generate(),
      instruction: directive,
      session_id: evalSessionId,
      budget: task.budget,
      loop_strategy: reactPerLoop(task.budget.max_turns ?? Number.MAX_SAFE_INTEGER),
    };

    // Child harness: clone the config, swap the sandbox to read-only. Cloning
    // shares the same observability/storage seams so the evaluate run's spans
    // land in the SAME trace stream (distinguished by its distinct session id).
    // The evaluate agent (#124 — no `config.agent`) is passed to `runReact`.
    const evalConfig: HarnessConfig = {
      ...this.config,
      sandbox: readOnlySandbox,
    };
    const evalHarness = new StandardHarness(evalConfig);

    const evalState = emptySessionState();
    const cap = task.budget.max_turns ?? Number.MAX_SAFE_INTEGER;
    const evalResult = await evalHarness.runReact(
      evalTask,
      cap,
      evalState,
      emptyBudgetSnapshot(),
      undefined,
      undefined,
      true,
      evaluator,
    );

    foldUsage(totalUsage, carried, evalResult);
    return evalResult;
  }

  /**
   * Look up the `role-evaluator` chunk content from the configured
   * {@link ChunkProvider} (R4, presence-only). Returns `undefined` if no provider
   * is configured, it has no such chunk, or it fails to load.
   */
  private async roleEvaluatorChunk(): Promise<string | undefined> {
    const provider = this.config.chunkProvider;
    if (provider == null) return undefined;
    try {
      const chunks = await provider.load();
      return chunks.find((c) => c.id === "role-evaluator")?.content;
    } catch {
      return undefined;
    }
  }

  // --------------------------------------------------------------------------
  // HillClimbing (issue #60) — iterative optimization loop
  // --------------------------------------------------------------------------

  /**
   * Revert the working tree to current HEAD for a no-improvement iteration
   * (issue #60, D1). Runs `git reset --hard HEAD` THROUGH the sandbox; the
   * harness NEVER spawns git directly. A sandbox rejection / non-zero exit is
   * best-effort: the loop continues (the next agent turn re-derives state).
   */
  private async hillClimbingRevert(signal?: AbortSignal): Promise<void> {
    const exec = this.config.sandbox.executeCommand;
    if (exec == null) return;
    try {
      await exec.call(this.config.sandbox, "git", ["reset", "--hard", "HEAD"], null, null, signal);
    } catch {
      // Best-effort — swallow exactly like the Rust impl.
    }
  }

  /**
   * Resolve the `commit_hash` recorded on a HillClimbing TSV row (issue #60,
   * D1). The harness never commits, so this is `null` (serialized as the empty
   * string in the TSV) unless a {@link VcsProvider} is wired to supply a hash.
   * v1 has no per-keep commit, so we always return `null`; the VcsProvider seam
   * is reserved for a later revision.
   */
  private async hillClimbingCommitHash(): Promise<string | null> {
    return null;
  }

  /**
   * Emit one fire-and-forget per-iteration observability span for a HillClimbing
   * run (issue #60). No-op when no provider is configured.
   */
  private emitHillClimbingSpan(
    sessionId: SessionId,
    taskId: TaskId,
    spanSeq: { value: number },
    iteration: number,
    metricValue: number | null,
    delta: number | null,
    status: IterationStatus,
    reverted: boolean,
  ): void {
    const obs = this.config.observability;
    if (obs?.emitWarn) {
      const base = newRootSpanBase(
        SpanId.of(`${sessionId.asString()}-hill-${spanSeq.value}`),
        sessionId,
        taskId,
        "warn",
        Timestamp.now(),
      );
      const event: WarnEvent = {
        warn: "hill_climbing_iteration",
        iteration,
        metric_value: metricValue,
        delta,
        status,
        reverted,
      };
      obs.emitWarn(newWarnSpan(base, event));
      spanSeq.value += 1;
    }
  }

  /**
   * Serialize the HillClimbing results log and write it to
   * `{workspace_root}/.spore/results/{task_id}.tsv` (issue #60, D2/D3).
   * Best-effort: a filesystem error is swallowed (the run outcome is
   * authoritative, the TSV is a diagnostic artifact). Skipped entirely when no
   * workspace root is known.
   */
  private async writeHillClimbingTsv(
    workspaceRoot: string,
    taskId: TaskId,
    rows: ResultsEntry[],
  ): Promise<void> {
    if (workspaceRoot.length === 0) return;
    const body = StandardHarness.renderHillClimbingTsv(rows);
    const dir = join(workspaceRoot, ".spore", "results");
    try {
      await mkdir(dir, { recursive: true });
      await writeFile(join(dir, `${taskId.asString()}.tsv`), body);
    } catch {
      // Best-effort — swallow exactly like the Rust impl.
    }
  }

  /**
   * Render the HillClimbing results-log TSV body (issue #60, D2/D3). Pure
   * function over the rows so the exact byte content is unit-testable and
   * cross-language-comparable. Trailing newline after every row (including the
   * last) so appends and diffs stay line-oriented. Floats use exactly 6 decimal
   * places; `metric_value` is the empty string on crashed/timeout rows; the
   * empty commit hash and snake_case direction/status mirror Rust byte-for-byte.
   */
  static renderHillClimbingTsv(rows: ResultsEntry[]): string {
    let out =
      "iteration\tcommit_hash\tmetric_value\tdirection\tstatus\tduration_secs\tdescription\n";
    for (const r of rows) {
      // D3: metric_value is EMPTY on crashed/timeout rows.
      const metricValue =
        r.status === "crashed" || r.status === "timeout" ? "" : r.metric_value.toFixed(6);
      const commitHash = r.commit_hash ?? "";
      const durationSecs = r.duration.toFixed(6);
      out += `${r.iteration}\t${commitHash}\t${metricValue}\t${r.direction}\t${r.status}\t${durationSecs}\t${r.description}\n`;
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Ralph (issue #58) — multi-context-window continuation loop
  // --------------------------------------------------------------------------

  /**
   * Ralph external completion check (issue #58, B1). Reads the deterministic
   * `.spore/` files under `workspaceRoot` and reports whether the task is
   * complete: `null` when complete, a reason string when tasks remain. This is
   * the SAME logic the registered `ralph-stop` hook applies — one source of
   * truth for the completion mechanism.
   *
   * Contract (B4 — no git):
   *   - `.spore/progress.json`: `{ "complete": boolean, "remaining": string[] }`.
   *     `complete: true` with empty `remaining` ⇒ progress satisfied.
   *     Missing/unreadable/invalid ⇒ incomplete (so the agent learns to write it).
   *   - `.spore/feature_list.json`: a JSON array of `{ "name", "passes" }`. Any
   *     `passes: false` ⇒ incomplete. A MISSING feature list is tolerated here
   *     (progress.json is the primary signal); an invalid one is not.
   */
  /**
   * Whether `.spore/progress.json` exists under `workspaceRoot` (issue #58).
   * The registered `ralph-stop` hook uses this to stay INERT for non-Ralph runs
   * over the same workspace: with no progress file the completion mechanism does
   * not engage. Mirrors the Rust RalphStopHook's `progress_path.exists()` guard.
   */
  static ralphProgressFilePresent(workspaceRoot: string): boolean {
    return existsSync(join(workspaceRoot, ".spore", "progress.json"));
  }

  static ralphCompletionStatus(workspaceRoot: string): string | null {
    const progressPath = join(workspaceRoot, ".spore", "progress.json");
    let raw: string;
    try {
      raw = readFileSync(progressPath, "utf-8");
    } catch {
      return ".spore/progress.json missing";
    }
    let progress: { complete?: unknown; remaining?: unknown };
    try {
      progress = JSON.parse(raw) as { complete?: unknown; remaining?: unknown };
    } catch (e) {
      return `.spore/progress.json invalid JSON: ${(e as Error).message}`;
    }
    const remaining = Array.isArray(progress.remaining)
      ? (progress.remaining as unknown[]).map(String)
      : [];
    if (progress.complete !== true) {
      return remaining.length === 0
        ? "task not marked complete"
        : `remaining: ${remaining.join(", ")}`;
    }
    if (remaining.length > 0) {
      return `remaining: ${remaining.join(", ")}`;
    }

    // Progress says done — corroborate against the feature list when present.
    const featurePath = join(workspaceRoot, ".spore", "feature_list.json");
    let featureRaw: string;
    try {
      featureRaw = readFileSync(featurePath, "utf-8");
    } catch {
      return null; // A missing feature list is tolerated.
    }
    let entries: { name?: unknown; passes?: unknown }[];
    try {
      entries = JSON.parse(featureRaw) as { name?: unknown; passes?: unknown }[];
    } catch (e) {
      return `.spore/feature_list.json invalid JSON: ${(e as Error).message}`;
    }
    const incomplete = entries.filter((x) => x.passes !== true).map((x) => String(x.name));
    if (incomplete.length > 0) {
      return `incomplete features: ${incomplete.join(", ")}`;
    }
    return null;
  }

  /**
   * Build the filesystem-reload context block injected into each fresh context
   * window (issue #58, R3). Returns the verbatim `.spore/progress.json` and
   * `.spore/feature_list.json` contents (when present) so the re-seeded window
   * knows what is already done and what remains. Returns `null` when neither
   * file exists (nothing to reload).
   */
  static ralphReloadContext(workspaceRoot: string): string | null {
    const parts: string[] = [];
    try {
      const raw = readFileSync(join(workspaceRoot, ".spore", "progress.json"), "utf-8");
      parts.push(`Reloaded .spore/progress.json:\n${raw.trim()}`);
    } catch {
      // Absent — nothing to reload from this file.
    }
    try {
      const raw = readFileSync(join(workspaceRoot, ".spore", "feature_list.json"), "utf-8");
      parts.push(`Reloaded .spore/feature_list.json:\n${raw.trim()}`);
    } catch {
      // Absent — nothing to reload from this file.
    }
    return parts.length === 0 ? null : parts.join("\n\n");
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
    agent: Agent,
  ): Promise<RunResult> {
    const result = await this.runReactInner(
      task,
      maxIterations,
      sessionState,
      budgetUsed,
      onStream,
      signal,
      seedInstruction,
      agent,
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
      case "consult":
        // Consult (issue #114) is NOT terminal — like waiting_for_human, the
        // worker is paused awaiting a resume. Do not finalize observability.
        break;
      case "escalate":
        // Escalation is a clean terminal outcome (#80). Finalize observability
        // with the dedicated `escalated` outcome (NOT `partial`).
        await this.finalizeObservability(result.session_id, { kind: "escalated" });
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
    agent: Agent,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    // Resolve the effective tool registry once per turn-loop window (issue #91).
    // Bridges catalogue tools per-run via RealToolRegistry, else the slim seam.
    const toolRegistry = this.effectiveToolRegistry(sessionId);
    // Reset the adaptive prompt-based-tool-calling escalation flag at the start
    // of this turn-loop window so detection is scoped to the window and does not
    // leak across run() calls (the flag is shared with the model wrapper for the
    // harness's lifetime). No-op unless a `conversational` harness installed the
    // adaptive wrapper (#111).
    if (this.config.prompt_tool_call_flag != null) {
      this.config.prompt_tool_call_flag.value = false;
    }
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
          sessionState,
        );
      }
      const overrun = budgetExceeded(task.budget, budgetUsed, startedAt);
      if (overrun != null) {
        return failure(
          { kind: "budget_exceeded", limit_type: overrun },
          sessionId,
          usage,
          budgetUsed.turns,
          sessionState,
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
              sessionState,
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
      // Fold the registry's tool schemas in only when the context manager left
      // the tool list empty (issue #91). A manager that deliberately sets a
      // phase-specific subset is preserved.
      if (context.tools.length === 0) {
        context.tools = toolRegistry.schemas();
      }
      // Prepend the configured operating system prompt (issue #91). The standard
      // compaction adapter renders none, so without this the model gets only the
      // task and no guidance. Guard against duplicates so a context manager that
      // already leads with a system message (or a resumed/seeded session) isn't
      // given two.
      if (this.config.systemPrompt != null) {
        const hasSystem = context.messages[0]?.role === "system";
        if (!hasSystem) {
          context.messages.unshift({
            role: "system",
            content: { type: "text", text: this.config.systemPrompt },
          });
        }
      }
      // Per-run model params win unconditionally (issue #93). The agent copies
      // `Context.params` verbatim into the `ModelRequest`, so this is the single
      // seam that delivers configured params (e.g. structured tool calls) to
      // every tool-requesting ReAct/execute/streaming turn.
      context.params = this.config.modelParams;
      // Whether tools were advertised to the model this turn — a precondition for
      // classifying a prose final response as a missed tool call (adaptive
      // prompt-based escalation, #111). Captured before `context` is consumed by
      // the turn.
      const toolsAdvertised = context.tools.length > 0;
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
      // Issue #103: when a stream sink is attached, drive the turn through
      // `turnStreaming` and forward each raw model `StreamEvent` mapped to
      // harness `StreamEvent`s, preserving order: turn_start → deltas →
      // turn_end → coarse events. When no sink is attached we keep the plain
      // `turn` path so the baseline RunResult is byte-identical (back-compat).
      const result = onStream
        ? await turnStreaming(
            agent,
            context,
            ((): AgentStreamSink => {
              const streamState = new TurnStreamState();
              return (ev) => {
                for (const mapped of mapModelStreamEvent(ev, streamState)) {
                  emit(onStream, mapped);
                }
              };
            })(),
            signal,
          )
        : await agent.turn(context, signal);
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

          // Adaptive prompt-based tool-calling escalation (#111). When tools were
          // advertised but the model answered in prose with action-intent
          // language (it *meant* to act), classify the turn as a prose response
          // and escalate: set the session flag so the wrapped model switches to
          // prompt-based tool calling, record the prose + a corrective nudge, and
          // force another turn instead of completing. Guarded on the flag being
          // unset so it fires at most once per window (bounded — one extra turn)
          // and only on the `conversational` adaptive path.
          {
            const flag = this.config.prompt_tool_call_flag;
            if (
              flag != null &&
              flag.value === false &&
              detectProseResponse(result.content, toolsAdvertised) != null
            ) {
              flag.value = true;
              // Record the model's prose, then a corrective nudge, so the next
              // turn has coherent context.
              const assistant: Message = {
                role: "assistant",
                content: { type: "text", text: result.content },
              };
              await this.config.contextManager.appendAssistantMessage?.(sessionState, assistant);
              await this.config.contextManager.appendUserMessage(
                sessionState,
                PROMPT_TOOL_CALL_NUDGE,
              );
              continue;
            }
          }

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
                  sessionState,
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
              sessionState,
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
            session_state: sessionState,
          };
        }

        case "tool_call_requested": {
          addTurnUsage(usage, result.usage);
          budgetUsed.input_tokens += result.usage.input_tokens;
          budgetUsed.output_tokens += result.usage.output_tokens;

          // Always-halt short-circuit (Layer 1).
          const haltingTool = result.calls.find((c) => toolRegistry.isAlwaysHalt(c.name));
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
              sessionState,
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
                  sessionState,
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
                  sessionState,
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
              emit(onStream, {
                kind: "tool_result",
                call_id: call.id,
                is_error: true,
                // Q5: carry the result content.
                content: `sandbox: ${violation.kind}`,
              });
              await this.config.contextManager.appendToolResult(sessionState, tr);
              approvedResults.push(tr);
              continue;
            }

            emit(onStream, {
              kind: "tool_call",
              call_id: call.id,
              name: call.name,
              // Q5: carry the final tool-call arguments.
              args: call.input,
            });
            const toolStartedAt = Timestamp.now();
            const toolClock = Date.now();
            // #126 (AC2): observe write/edit calls as they dispatch so the
            // PlanExecute DAG executor can attach HARNESS-OBSERVED `files_touched`
            // to a task's ledger entry — never model-self-reported.
            this.observeWriteCall(call);
            const output: ToolOutput = await toolRegistry.dispatch(call, signal);

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

            // Escalate (#80): the tool requests a structural state change from
            // the harness's parent. The harness is a pure intermediary — it
            // does NOT append the escalation to message history (it is a control
            // signal, not a conversation turn), preserves the remaining batch
            // tool calls into `pending_tool_calls` for a possible resume, and
            // returns the `escalate` RunResult carrying the signal + full
            // PausedState. The signal is NOT stored in PausedState, so it is
            // discarded on resume — the harness never re-acts on it.
            if (output.kind === "escalate") {
              const remaining = calls.slice(i + 1);
              const ps: PausedState = {
                session_id: sessionId,
                task_id: task.id,
                turn_number: budgetUsed.turns,
                session_state: sessionState,
                pending_tool_calls: remaining,
                approved_results: approvedResults,
                // No human_request on an escalation pause (#80, optional field).
                task,
                budget_used: budgetUsed,
                child_state: null,
              };
              return {
                kind: "escalate",
                signal: output.signal,
                state: ps,
                session_id: sessionId,
                usage,
                turns: budgetUsed.turns,
              };
            }

            // Clarification pause (issue #81, Q4b): a tool (e.g.
            // `ask_user_question`) needs a human answer before it can produce a
            // result. UNLIKE the subagent `waiting_for_human` path there is NO
            // ChildPausedState: build a PausedState directly with `human_request`
            // set to `clarification`. The CLARIFYING call itself is preserved as
            // the head of `pending_tool_calls` (followed by the remaining batch)
            // so that, on resume, the human's answer is injected as the tool
            // RESULT for that pending call.
            if (output.kind === "awaiting_clarification") {
              const pending = [call, ...calls.slice(i + 1)];
              const request: HumanRequest = {
                kind: "clarification",
                question: output.question,
                options: output.options,
              };
              const ps: PausedState = {
                session_id: sessionId,
                task_id: task.id,
                turn_number: budgetUsed.turns,
                session_state: sessionState,
                pending_tool_calls: pending,
                approved_results: approvedResults,
                human_request: request,
                task,
                budget_used: budgetUsed,
                child_state: null,
              };
              return { kind: "waiting_for_human", state: ps, request };
            }

            // Consult pause (issue #114, R1/R10): a worker-side tool returns
            // `consult` (with `child_state` absent) to ask for mid-loop help.
            // Like the `awaiting_clarification` arm there is NO ChildPausedState:
            // build a PausedState directly with `human_request` absent, and
            // preserve the CONSULTING call as the head of `pending_tool_calls`
            // (followed by the remaining batch) so that on `resumeConsult` the
            // helper's answer is injected as the tool RESULT for that pending
            // call. The consult is a control signal, NOT a conversation turn — it
            // is never appended to message history here (R10).
            if (output.kind === "consult") {
              // Observability: lightweight consult-spawn event alongside
              // `skill_injected`. Emitted before returning the pause.
              if (this.config.observability) {
                const base = newRootSpanBase(
                  SpanId.of(`${sessionId.asString()}-consult-spawn-${spanSeq}`),
                  sessionId,
                  task.id,
                  "context_assembly",
                  Timestamp.now(),
                );
                const span: ContextSpan = {
                  base,
                  operation: { kind: "consult_spawned", consult_kind: output.request.kind },
                  tokens_before: 0,
                  tokens_after: 0,
                  utilization_before: 0,
                  utilization_after: 0,
                };
                this.config.observability.emitContext(span);
                // No spanSeq increment: the pause returns immediately below.
              }
              const pending = [call, ...calls.slice(i + 1)];
              const ps: PausedState = {
                session_id: sessionId,
                task_id: task.id,
                turn_number: budgetUsed.turns,
                session_state: sessionState,
                pending_tool_calls: pending,
                approved_results: approvedResults,
                // No human_request on a consult pause (#114, optional field).
                task,
                budget_used: budgetUsed,
                child_state: null,
              };
              return {
                kind: "consult",
                request: output.request,
                state: ps,
                session_id: sessionId,
                usage,
                turns: budgetUsed.turns,
              };
            }

            // SendMessage (issue #81): the `send_message` tool surfaces an
            // out-of-band message to the user. Emit a `user_message` stream
            // event rather than collapsing the content into a normal tool
            // result, then record a minimal success result so the loop
            // continues. (Bad params produce an `error` output, not a success,
            // and so no event is emitted.)
            if (call.name === "send_message" && output.kind === "success") {
              emit(onStream, { kind: "user_message", content: output.content });
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
                sessionState,
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
            // Q5: surface the result content on the coarse event.
            const resultContentForStream =
              output.kind === "success"
                ? output.content
                : output.kind === "error"
                  ? output.message
                  : "";
            emit(onStream, {
              kind: "tool_result",
              call_id: call.id,
              is_error: isError,
              content: resultContentForStream,
            });
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
                sessionState,
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
              agent,
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
            sessionState,
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
    agent: Agent,
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
      const result = await agent.turn(turn.context, signal);
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
  private _chunkProvider: ChunkProvider = new InMemoryChunkProvider();
  private _verifier?: Verifier;
  private _metricEvaluator?: MetricEvaluator;
  private _vcsProvider?: VcsProvider;
  private readonly _standardTools: StandardTool[] = [];
  private _systemPrompt?: string;
  private _modelParams: ModelParams = ModelParamsSchema.parse({});
  private _autoPersistSessions = false;
  private _promptToolCallFlag?: SharedFlag;
  private readonly _consultHandlers: ConsultHandlerMap = new Map();
  private _registry: ExecutionRegistry = ExecutionRegistry.empty();
  private _escalationMode: EscalationMode = surfaceToHuman;

  constructor(
    private readonly agent: Agent,
    private _toolRegistry: ToolRegistry,
    private _sandbox: SandboxProvider,
    private _contextManager: ContextManager,
    private readonly terminationPolicy: TerminationPolicy,
  ) {}

  /**
   * Assemble a minimal conversational harness builder from a model — no tools,
   * no filesystem.
   *
   * This is the few-lines path: it defaults every required component so you can
   * go from a model to a running harness in one call. The defaults are a
   * {@link ModelAgent} over `model` (agent id `"agent"`), an
   * {@link EmptyToolRegistry}, a {@link NullSandbox}, a
   * {@link StandardContextManager} with a {@link NullCacheProvider} and default
   * compaction (wrapped through {@link intoHarnessAdapter}), and
   * {@link CompleteOnFinalResponse} termination (the model's first final
   * response is the result).
   *
   * Every default is overridable through the fluent setters (e.g.
   * {@link tool} / {@link tools} for catalogue tools, or {@link sandbox} to
   * swap in a workspace-scoped sandbox).
   *
   * Mirrors `HarnessBuilder::conversational` in
   * `rust/crates/spore-core/src/harness.rs`.
   *
   * ```ts
   * const harness = HarnessBuilder.conversational(
   *   new OllamaModelInterface("llama3.2"),
   * ).build();
   * const result = await harness.run({ task: simpleTask("Say hello.") });
   * ```
   */
  static conversational(model: ModelInterface): HarnessBuilder {
    // Install the adaptive prompt-based tool-calling wrapper around the agent's
    // model (#111). While its shared flag is unset it delegates natively
    // (byte-for-byte); the run loop flips the flag on detecting a prose response
    // so the model switches to prompt-based tool calling for the rest of the
    // run. The context manager keeps the *raw* model (its only model use is
    // compaction summarization, which advertises no tools).
    const promptToolCallFlag = newSharedFlag();
    const adaptiveModel = new AdaptiveToolCallModelInterface(model, promptToolCallFlag);
    const agent = new ModelAgent(AgentId.of("agent"), adaptiveModel);
    const toolRegistry = new EmptyToolRegistry();
    const sandbox = new NullSandbox();
    const contextManager = intoHarnessAdapter(
      new StandardContextManager(model, new NullCacheProvider(), defaultCompactionConfig()),
    );
    const terminationPolicy = new CompleteOnFinalResponse();
    const builder = new HarnessBuilder(
      agent,
      toolRegistry,
      sandbox,
      contextManager,
      terminationPolicy,
    );
    builder._promptToolCallFlag = promptToolCallFlag;
    return builder;
  }

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

  /**
   * Register a per-kind consult handler (issue #114). When a worker (run via {@link "@spore/tools".SubagentTool})
   * pauses with a {@link ConsultRequest} whose `kind` matches `kind`, the
   * orchestrator runs `handler` on the request deterministically (no orchestrator
   * model turn), up to `budget` consults of this kind before applying `overflow`.
   * Without any registered handler, consults degrade gracefully (R6): a standalone
   * worker surfaces {@link RunResult} `consult` unchanged. Empty by default.
   */
  consultHandler(
    kind: string,
    handler: Harness,
    budget: number,
    overflow: ConsultOverflowPolicy,
  ): this {
    const entry: ConsultHandlerEntry = { handler, budget, overflow };
    this._consultHandlers.set(kind, entry);
    return this;
  }

  /** Set the conditional-chunk provider (issue #79). Defaults to an empty
   *  {@link InMemoryChunkProvider}. */
  chunkProvider(provider: ChunkProvider): this {
    this._chunkProvider = provider;
    return this;
  }

  /** Convenience: register chunks inline without constructing a provider
   *  (issue #79). Resolves to an {@link InMemoryChunkProvider} internally. */
  chunks(chunks: AssemblyPromptChunk[]): this {
    this._chunkProvider = new InMemoryChunkProvider(chunks);
    return this;
  }

  /** Inject the verdict oracle for the `self_verifying` strategy (issue #61).
   *  REQUIRED for that strategy — absent it, a `self_verifying` run halts with
   *  `self_verify_misconfigured`. Ignored by every other strategy. */
  verifier(verifier: Verifier): this {
    this._verifier = verifier;
    return this;
  }

  /** Inject the metric oracle for the `hill_climbing` strategy (issue #60).
   *  REQUIRED for that strategy — absent it, a `hill_climbing` run halts with
   *  `hill_climbing_misconfigured`. Ignored by every other strategy. Folded into
   *  the {@link ExecutionRegistry} under the default key (#124, Q2). */
  metricEvaluator(evaluator: MetricEvaluator): this {
    this._metricEvaluator = evaluator;
    return this;
  }

  /** Inject a {@link VcsProvider} for the `ralph` loop strategy (issue #58 v2).
   *  When set, Ralph's per-window reload phase also calls {@link VcsProvider.log}
   *  and injects a delimited "Recent VCS history:" section into the fresh context
   *  window. Defaults to unset, which omits the git-log section and preserves v1
   *  Ralph behavior byte-for-byte (the B4→none decision). */
  vcsProvider(provider: VcsProvider): this {
    this._vcsProvider = provider;
    return this;
  }

  /**
   * Add a single catalogue {@link StandardTool} to this harness (issue #81,
   * Q1/Q2). At {@link buildConfig} the accumulated tools are folded into a
   * populated {@link StandardToolRegistry} and the run loop bridges them per-run
   * through a {@link RealToolRegistry}, so they run with sandbox + storage wired
   * in — no manual bridging required. Registration applies LAST-WINS upsert: a
   * later `.tool()` with the same name overrides an earlier one (e.g. a custom
   * tool registered after a preset).
   *
   * To author a *custom* sandboxed tool, build a {@link StandardTool} over your
   * own {@link "../tool-registry/index.js".Tool} implementation and pass it here;
   * catalogue tools take precedence over a registry supplied via the constructor.
   */
  tool(tool: StandardTool): this {
    this._standardTools.push(tool);
    return this;
  }

  /**
   * Add many catalogue {@link StandardTool}s at once (e.g. a preset like
   * `StandardTools.codingSet()`). Order is preserved, so last-wins upsert still
   * applies across the batch.
   */
  tools(tools: Iterable<StandardTool>): this {
    for (const t of tools) this._standardTools.push(t);
    return this;
  }

  /**
   * Set an operating system prompt prepended to each turn's assembled context
   * (issue #91).
   *
   * The standard compaction context manager renders no system prompt, so without
   * this the model receives only the task as a user message and no guidance on
   * how to behave. When set, the run loop inserts this text as a leading
   * `system` message each turn — but only when the assembled context does not
   * already start with one, so a context manager that renders its own system
   * prompt is preserved. Absent (the default) preserves today's behaviour.
   */
  systemPrompt(systemPrompt: string): this {
    this._systemPrompt = systemPrompt;
    return this;
  }

  /**
   * Set the authoritative model sampling/decoding parameters for the whole run
   * (issue #93).
   *
   * These params are authoritative: the harness replaces each turn's
   * {@link Context.params} with this value UNCONDITIONALLY (builder params win)
   * right before the request is built, so the configured params reach every
   * agent turn that requests tools — the ReAct loop, the PlanExecute plan
   * phase, the execute sub-loop, and the streaming path alike. (The internal
   * compaction/summarization turn is intentionally left on defaults; it
   * requests no tools, so decoding params are a no-op there.)
   *
   * Enabling {@link ModelParams.structured_tool_calls} trades interleaved
   * reasoning for one schema-constrained tool call per turn — useful for small
   * local models that otherwise emit malformed tool calls. Defaults to the
   * schema's default {@link ModelParams}.
   */
  modelParams(p: ModelParams): this {
    this._modelParams = p;
    return this;
  }

  /**
   * Opt into automatic conversation-history threading through the
   * {@link StorageProvider}'s {@link SessionStore} (issue #102). Defaults to
   * `false` — the off-by-default zero-I/O contract.
   *
   * When `true`, `run()` auto-loads the prior session by `session_id` (ReAct /
   * SelfVerifying; explicit {@link HarnessRunOptions.session_state} still wins)
   * and auto-persists the post-run state at the terminal seam. For cross-process
   * continuity, pair this with a durable {@link StorageProvider}; without one the
   * default no-op store makes this an inert flag. See
   * {@link HarnessConfig.autoPersistSessions} for the full contract.
   */
  autoPersistSessions(enabled: boolean): this {
    this._autoPersistSessions = enabled;
    return this;
  }

  /**
   * Override the {@link SandboxProvider} supplied at construction — the only
   * path tools have to the environment (filesystem, process exec).
   *
   * Catalogue file tools (`read_file` / `write_file` / `list_dir`) operate
   * *through* the sandbox, so an agent that touches a real directory needs a
   * workspace-scoped sandbox here. This lets `.sandbox(workspace).tools(...)`
   * reach a real workspace without re-threading every other component through
   * the constructor:
   *
   * ```ts
   * const harness = builder
   *   .sandbox(workspace)
   *   .tools(StandardTools.codingSet())
   *   .build();
   * ```
   */
  sandbox(sandbox: SandboxProvider): this {
    this._sandbox = sandbox;
    return this;
  }

  /**
   * Override the harness-loop {@link ToolRegistry} supplied at construction.
   *
   * Use this to supply your own registry — e.g. a custom set of tools — on top of
   * a preset like {@link conversational}. The registry's {@link ToolRegistry.schemas}
   * are delivered to the model automatically each turn, and
   * {@link ToolRegistry.dispatch} is called when the model requests a tool.
   *
   * Mirrors `HarnessBuilder::tool_registry` in
   * `rust/crates/spore-core/src/harness.rs`.
   *
   * ```ts
   * const harness = HarnessBuilder.conversational(model)
   *   .toolRegistry(new MyTools())
   *   .build();
   * ```
   */
  toolRegistry(toolRegistry: ToolRegistry): this {
    this._toolRegistry = toolRegistry;
    return this;
  }

  /**
   * Override the {@link ContextManager} that assembles per-turn context and
   * drives compaction.
   *
   * {@link conversational} installs a {@link StandardContextManager} with
   * default compaction (compaction at 80% of a 200K window) wrapped through
   * {@link intoHarnessAdapter}; supply your own (e.g. a lower compaction
   * `threshold`) to make compaction fire earlier for models with a smaller
   * context window. Wrap a {@link StandardContextManager} with
   * {@link intoHarnessAdapter} to obtain the harness-seam type.
   *
   * Mirrors `HarnessBuilder::context_manager` in
   * `rust/crates/spore-core/src/harness.rs`.
   *
   * ```ts
   * const cm = intoHarnessAdapter(
   *   new StandardContextManager(model, new NullCacheProvider(), {
   *     ...defaultCompactionConfig(),
   *     threshold: 0.45,
   *   }),
   * );
   * const harness = HarnessBuilder.conversational(model).contextManager(cm).build();
   * ```
   */
  contextManager(contextManager: ContextManager): this {
    this._contextManager = contextManager;
    return this;
  }

  /**
   * Inject a fully-assembled {@link ExecutionRegistry} (issue #120). REPLACES
   * any registry accumulated via the per-key convenience setters
   * ({@link agentRef}/{@link toolsetRef}/{@link schemaRef}/{@link verifierRef}/
   * {@link registerStrategy}). Empty by default (Option B — legacy callers stay
   * byte-identical and skip startup validation).
   */
  registry(registry: ExecutionRegistry): this {
    this._registry = registry;
    return this;
  }

  /** Convenience: register an agent in the {@link ExecutionRegistry} under
   *  `key` (issue #120). */
  agentRef(key: string, agent: Agent): this {
    this._registry = this._registry.toBuilder().agent(key, agent).build();
    return this;
  }

  /** Convenience: register a toolset in the {@link ExecutionRegistry} under
   *  `key` (issue #120). */
  toolsetRef(key: string, toolset: ToolRegistry): this {
    this._registry = this._registry.toBuilder().toolset(key, toolset).build();
    return this;
  }

  /** Convenience: register a JSON schema in the {@link ExecutionRegistry} under
   *  `key` (issue #120). */
  schemaRef(key: string, schema: unknown): this {
    this._registry = this._registry.toBuilder().schema(key, schema).build();
    return this;
  }

  /** Convenience: register a verifier in the {@link ExecutionRegistry} under
   *  `key` (issue #120). */
  verifierRef(key: string, verifier: Verifier): this {
    this._registry = this._registry.toBuilder().verifier(key, verifier).build();
    return this;
  }

  /** Convenience: register a custom strategy in the {@link ExecutionRegistry}
   *  under `key` (issue #120). */
  registerStrategy(key: string, strategy: RunStrategy): this {
    this._registry = this._registry.toBuilder().registerStrategy(key, strategy).build();
    return this;
  }

  /** Select the HITL-vs-AFK {@link EscalationMode} (issue #120). Defaults to
   *  {@link surfaceToHuman}. STORED only this slice; consumed in #130. */
  escalationMode(mode: EscalationMode): this {
    this._escalationMode = mode;
    return this;
  }

  /** Assemble the {@link HarnessConfig} without wrapping it in a harness. */
  buildConfig(): HarnessConfig {
    // Fold catalogue tools accumulated via `.tool()` / `.tools()` into a
    // populated StandardToolRegistry (last-wins upsert). The run loop bridges it
    // per-run — `build()` can't, because the ToolContext is keyed by the run's
    // SessionId, unknown until `run()`.
    let catalogueRegistry: StandardToolRegistry | undefined;
    if (this._standardTools.length > 0) {
      const registry = new StandardToolRegistry();
      const err = registry.tools(this._standardTools);
      if (err) {
        throw new RegistrationErrorException(err);
      }
      catalogueRegistry = registry;
    }
    // When catalogue tools are present and the caller wired no storage, default
    // to an in-memory provider (not the all-no-op default) so that session-aware
    // tools (todo_write, memory, task_list) actually persist within the run.
    // Pure tools (read_file/write_file via the sandbox) are unaffected.
    const storage =
      this._storage ??
      (catalogueRegistry != null
        ? StorageProvider.single(new InMemoryStorageProvider())
        : StorageProvider.noOp());

    // #124: the legacy single-collaborator fields are gone — fold the builder's
    // collaborators into the ExecutionRegistry under the DEFAULT empty-string
    // handle (`reactPerLoop` leaves use an empty AgentRef/ToolsetRef; the default
    // SelfVerifying / HillClimbing evaluator likewise uses the empty key).
    // Explicitly registered handles (via `agentRef` / `verifierRef` / …) take
    // precedence: `fillDefault*` only fills a key the caller did not already wire.
    let regBuilder = this._registry
      .toBuilder()
      .fillDefaultAgent(this.agent)
      .fillDefaultToolset(this._toolRegistry);
    if (this._verifier != null) regBuilder = regBuilder.fillDefaultVerifier(this._verifier);
    if (this._metricEvaluator != null) {
      regBuilder = regBuilder.fillDefaultMetricEvaluator(this._metricEvaluator);
    }
    const registry = regBuilder.build();

    return {
      toolRegistry: this._toolRegistry,
      sandbox: this._sandbox,
      contextManager: this._contextManager,
      terminationPolicy: this.terminationPolicy,
      middleware: this._middleware,
      observability: this._observability,
      storage,
      compactionVerifier: this._compactionVerifier,
      maxCompactionAttempts: this._maxCompactionAttempts,
      pricing: this._pricing,
      contentCapture: this._contentCapture,
      hooks: this._hooks,
      maxStopBlocks: this._maxStopBlocks,
      chunkProvider: this._chunkProvider,
      vcsProvider: this._vcsProvider,
      catalogueRegistry,
      systemPrompt: this._systemPrompt,
      modelParams: this._modelParams,
      autoPersistSessions: this._autoPersistSessions,
      prompt_tool_call_flag: this._promptToolCallFlag,
      // Only attach when populated so the default config stays byte-for-byte
      // unchanged for callers that never register a consult handler (R9).
      consultHandlers: this._consultHandlers.size > 0 ? this._consultHandlers : undefined,
      registry,
      escalationMode: this._escalationMode,
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
  sessionState: SessionState = emptySessionState(),
): RunResult {
  return {
    kind: "failure",
    reason,
    session_id: sessionId,
    usage,
    turns,
    session_state: sessionState,
  };
}

/**
 * Fold a sub-run's token usage / turn count into the cumulative `totalUsage`
 * and the shared `carried` budget snapshot (issue #61, R8). Mirrors the
 * PlanExecute / Rust `fold_usage`. `carried.turns` becomes the running MAX of
 * the absolute turn counts (the build sub-loop already gates on cumulative
 * turns; the fresh-session evaluate run reports its own turns). `waiting_for_human`
 * carries no usage and is skipped.
 */
function foldUsage(totalUsage: AggregateUsage, carried: BudgetSnapshot, r: RunResult): void {
  if (r.kind === "waiting_for_human") return;
  // A consult pause carries usage/turns but is non-terminal; like
  // waiting_for_human, do not fold it into a sub-run total (issue #114).
  if (r.kind === "consult") return;
  const usage = r.usage;
  const turns = r.turns;
  totalUsage.input_tokens += usage.input_tokens;
  totalUsage.output_tokens += usage.output_tokens;
  totalUsage.cache_read_tokens += usage.cache_read_tokens;
  totalUsage.cache_write_tokens += usage.cache_write_tokens;
  totalUsage.cost_usd += usage.cost_usd;
  carried.input_tokens += usage.input_tokens;
  carried.output_tokens += usage.output_tokens;
  carried.turns = Math.max(carried.turns, turns);
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
  return strategy.kind === "react";
}
