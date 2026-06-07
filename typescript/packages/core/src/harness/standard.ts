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
import type { Verifier, VerifierInput } from "../verifier/types.js";
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
import {
  PLAN_EXECUTE_EXTRAS_KEY,
  PlanPhaseError,
  capturePlanArtifact,
  type PlanArtifact,
} from "../plan/index.js";
import {
  TASK_LIST_EXTRAS_KEY,
  completeTask,
  planArtifactToTaskList,
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
  newTask,
  runResultSessionState,
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
  type EscalationMode,
  type HaltReason,
  HarnessError,
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
  /**
   * @deprecated Superseded by {@link ExecutionRegistry} (issue #120): agents are
   * resolved per-node via a strategy's {@link AgentRef}. Physical removal +
   * executor migration to registry resolution lands in #124. Still the live
   * default-agent field this slice (Option B additive scope).
   */
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
   *
   * @deprecated Superseded by {@link ExecutionRegistry} (issue #120): resolved
   * per-node via a strategy's {@link AgentRef}. Physical removal + executor
   * migration to registry resolution lands in #124.
   */
  plannerAgent?: Agent;
  /**
   * Pluggable source of conditional prompt chunks (issue #79). Loaded at harness
   * construction and fed through {@link "../prompt-assembly/index.js".ContextSourcesBuilder}.
   * Optional; defaults to an empty {@link InMemoryChunkProvider}.
   */
  chunkProvider?: ChunkProvider;
  /**
   * Verdict oracle for the `self_verifying` loop strategy (issue #61, D2/D4).
   * REQUIRED for that strategy: when absent, a `self_verifying` run halts with
   * `self_verify_misconfigured` (a typed halt, NOT a throw). Its
   * {@link "../verifier/types.js".Verifier.maxIterations} (default 3) caps the
   * build↔evaluate round-trips (D3). Ignored by every other strategy.
   *
   * @deprecated Superseded by {@link ExecutionRegistry} (issue #120): verifiers
   * are resolved by key from the registry's `verifiers` map. Physical removal +
   * executor migration to registry resolution lands in #124.
   */
  verifier?: Verifier;
  /**
   * Optional alternate agent used for the `self_verifying` evaluate phase
   * (issue #61, D2). Defaulting contract — IDENTICAL to {@link plannerAgent}
   * (#70): when this is absent, the evaluate phase runs on the default
   * {@link agent}. The read-only sandbox and the fresh, never-shared session id
   * for the evaluate run are derived INTERNALLY by the strategy — callers do NOT
   * supply an evaluator sandbox or chunk provider here.
   *
   * @deprecated Superseded by {@link ExecutionRegistry} (issue #120): resolved
   * per-node via a strategy's {@link AgentRef}. Physical removal + executor
   * migration to registry resolution lands in #124.
   */
  evaluatorAgent?: Agent;
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
   * Metric oracle for the `hill_climbing` loop strategy (issue #60). REQUIRED
   * for that strategy: when absent, a `hill_climbing` run halts with
   * {@link HaltReason} `hill_climbing_misconfigured` (a typed halt, NOT a throw —
   * Decision 6). The evaluator is called once at iteration 0 to establish the
   * baseline (no agent turn) and again after each agent turn to score the change.
   * Ignored by every other strategy.
   */
  metricEvaluator?: MetricEvaluator;
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

export class StandardHarness implements Harness {
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

    // Issue #120 startup validation: every serializable handle in the task's
    // strategy tree must resolve against the configured ExecutionRegistry,
    // BEFORE the first turn. Validation runs only when the registry is populated,
    // so existing callers that never wire a registry (and instead use the
    // deprecated single-collaborator fields, Option B) are unaffected
    // byte-for-byte. An unresolved handle is a startup error.
    const registry = this.config.registry;
    if (registry != null && !registry.isEmpty()) {
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

    switch (task.loop_strategy.kind) {
      case "react":
        return this.runReact(
          task,
          reactMaxIterations(task.loop_strategy),
          resolvedState,
          budgetUsed,
          options.on_stream,
          options.signal,
          true,
        );
      case "plan_execute":
        return this.runPlanExecute(
          task,
          resolvedState,
          budgetUsed,
          options.on_stream,
          options.signal,
        );
      case "ralph":
        return this.runRalph(task, budgetUsed, options.on_stream);
      case "self_verifying":
        return this.runSelfVerifying(task, resolvedState, budgetUsed, options.on_stream);
      case "hill_climbing":
        return this.runHillClimbing(
          task,
          task.loop_strategy.direction,
          task.loop_strategy.max_stagnation,
          task.loop_strategy.revert_on_no_improvement,
          task.loop_strategy.min_improvement_delta,
          budgetUsed,
          options.on_stream,
          options.signal,
        );
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
      return this.runReact(
        state.task,
        maxIter,
        sessionState,
        state.budget_used,
        onStream,
        signal,
        false,
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

      default: {
        const _exhaustive: never = response;
        return _exhaustive;
      }
    }

    const max =
      state.task.loop_strategy.kind === "react"
        ? reactMaxIterations(state.task.loop_strategy)
        : Number.MAX_SAFE_INTEGER;
    return this.runReact(state.task, max, sessionState, state.budget_used, onStream, signal, false);
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

    const max =
      state.task.loop_strategy.kind === "react"
        ? reactMaxIterations(state.task.loop_strategy)
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
  // PlanExecute (issues #70 plan phase + #59 execute phase)
  // --------------------------------------------------------------------------

  /**
   * Drive the `plan_execute` strategy (issue #59) — the two-phase loop.
   *
   * ## Phases
   * 1. **Plan phase** (issue #70, runs EXACTLY ONCE): {@link runPlanPhase} seeds
   *    a planning directive, runs one constrained planner turn, captures a
   *    {@link PlanArtifact}, fires `on_plan_created`, and counts the turn against
   *    the shared budget.
   * 2. **Execute phase** (issue #59, loops): {@link runExecutePhase} drains the
   *    task list, running each task in a bounded ReAct sub-loop that BUILDS ON
   *    the accumulated execute-phase context (prior steps' tool results and
   *    assistant outputs carry forward).
   *
   * Between the phases the artifact is parsed into a {@link TaskList} via
   * `planArtifactToTaskList` and persisted through the storage seam (Q4) plus
   * the `extras` mirror.
   *
   * ## Resolved spec decisions (issue #59 — all FINAL)
   * - **Q1:** each task runs a bounded, SEQUENTIAL ReAct sub-loop that BUILDS ON
   *   the accumulated execute-phase context — after each successful step its
   *   conversation (instruction + tool calls + tool results + assistant output)
   *   is folded back into the shared session, so the next step sees all prior
   *   steps' results. Steps stay SEQUENTIAL (task N completes before N+1). The
   *   shared budget still gates: the per-task turn cap is derived at the START of
   *   each step: `per_task_turns = floor(remaining_turns / remaining_tasks)`,
   *   floored at 1 (`remaining_tasks` counts the not-yet-started tasks including
   *   the current one). The shared budget — turns, tokens, observability spans,
   *   compaction — is carried across EVERY step and the global budget is the hard
   *   stop.
   * - **Q2:** on success `output` is the LAST completed step's `final_response`
   *   text — not a concatenation, not the plan rationale.
   * - **Q3:** an empty task list ⇒ {@link HaltReason} `empty_plan`.
   * - **Q4:** the task list / plan are persisted through the {@link StorageProvider}
   *   / `RunStore` seam ONLY; the #71 sandbox path (`.spore/task_list.json`) is
   *   NOT used by the execute loop. The `extras` mirror is kept.
   * - **Q5:** a step's sub-loop erroring or returning a blocked/failed outcome
   *   ABORTS the whole run with {@link HaltReason} `step_failed`; execution does
   *   NOT continue to the next task.
   *
   * The removed `execute_phase_not_implemented` halt no longer exists. On any
   * plan-phase failure the underlying `failure` `RunResult` is returned
   * unchanged (no task list persisted). Like `runReact`, finalizes
   * observability for the terminal outcome.
   */
  private async runPlanExecute(
    task: Task,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    const sessionId = task.session_id;

    // ── Phase 1: plan (runs exactly once) ───────────────────────────────────
    const outcome = await this.runPlanPhase(task, sessionState, budgetUsed, onStream, signal);
    if (!outcome.ok) {
      // Plan-phase failure: propagate unchanged (no task list persisted).
      await this.finalizePlanExecute(outcome.failure);
      return outcome.failure;
    }

    // Bridge: parse the accepted plan into a TaskList (#72).
    const taskList = planArtifactToTaskList(outcome.artifact);

    // Q3: an empty plan is a failure, not a silent success.
    if (taskList.tasks.length === 0) {
      const result: RunResult = {
        kind: "failure",
        reason: { kind: "empty_plan" },
        session_id: sessionId,
        usage: outcome.usage,
        turns: outcome.turns,
      };
      await this.finalizePlanExecute(result);
      return result;
    }

    // Q4: persist the task list through the storage seam (RunStore) ONLY, plus
    // the extras mirror. The #71 sandbox path is intentionally unused.
    await this.persistTaskList(sessionId, taskList);

    // Carry the shared budget forward: the plan turn already consumed
    // `outcome.turns` turns and `outcome.usage` tokens (Q1 — shared budget).
    const carried: BudgetSnapshot = { ...budgetUsed };
    carried.turns = outcome.turns;
    carried.input_tokens += outcome.usage.input_tokens;
    carried.output_tokens += outcome.usage.output_tokens;

    // ── Phase 2: execute (loops over the task list) ─────────────────────────
    const result = await this.runExecutePhase(
      task,
      sessionState,
      taskList,
      carried,
      outcome.usage,
      onStream,
      signal,
    );
    await this.finalizePlanExecute(result);
    return result;
  }

  /**
   * Finalize observability for a terminal PlanExecute outcome. Mirrors the tail
   * of {@link runReact}: `waiting_for_human` is not terminal and is never
   * flushed here.
   */
  private async finalizePlanExecute(result: RunResult): Promise<void> {
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
      // Consult (issue #114) is non-terminal; do not finalize.
      case "consult":
        break;
      case "escalate":
        // Escalation propagated from an execute step (#80) is a clean terminal
        // outcome; finalize with the dedicated `escalated` outcome.
        await this.finalizeObservability(result.session_id, { kind: "escalated" });
        break;
    }
  }

  /**
   * Persist the parsed {@link TaskList} for the run (Q4). The DURABLE write goes
   * through the `RunStore` seam under `TASK_LIST_EXTRAS_KEY`; the #71
   * sandbox-filesystem path (`.spore/task_list.json`) is intentionally NOT used
   * — the `RunStore` write is the single source of truth (#76 removed the
   * redundant `SessionState.extras` mirror). Failures are swallowed: a
   * successful plan must not be lost to a storage hiccup (the default no-op /
   * in-memory provider never fails).
   */
  private async persistTaskList(sessionId: SessionId, taskList: TaskList): Promise<void> {
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
   * Drive the PlanExecute execute phase (issue #59), draining `taskList`.
   *
   * Per Q1 each task runs a bounded, SEQUENTIAL ReAct sub-loop that BUILDS ON the
   * accumulated execute-phase context: after each successful step its resulting
   * conversation (instruction + tool calls + tool results + assistant output) is
   * folded back into the shared `sessionState`, so the next step's sub-loop
   * (which clones `sessionState`) sees every prior step's results. The per-task
   * turn cap is derived at the START of each step from the shared budget:
   * `per_task_turns = floor(remaining_turns / remaining_tasks)`, floored at 1
   * (`remaining_tasks` counts not-yet-started tasks including the current one).
   * The shared budget snapshot (`carried`) is threaded through every step so
   * early tasks cannot starve later ones and the global budget stays the hard
   * stop.
   *
   * Before each step the task is marked `in_progress` (and `completed` after),
   * the list is re-persisted (Q4), and `on_task_advance` fires with the correct
   * `task_index` / `total_tasks` (the hook may rewrite the step instruction).
   *
   * Q2: on success `output` is the LAST completed step's `final_response`.
   * Q5: a step that errors / blocks aborts the run with `step_failed` — no
   * further tasks run.
   *
   * `planUsage` seeds the cumulative {@link AggregateUsage} with the plan turn's
   * usage so the terminal `RunResult` reflects the WHOLE run.
   */
  private async runExecutePhase(
    task: Task,
    sessionState: SessionState,
    taskList: TaskList,
    carried: BudgetSnapshot,
    planUsage: AggregateUsage,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    const totalTasks = taskList.tasks.length;
    // Cumulative usage across the plan turn + every execute step (Q1).
    const totalUsage: AggregateUsage = { ...planUsage };
    // Q2: the success handle is the LAST completed step's final text.
    let lastOutput = "";
    // Issue #102: the post-run conversation history threaded into the terminal
    // result is the LAST step's sub-loop history (each step runs on a clone),
    // seeded with the plan-phase session so an empty/all-skipped plan still
    // carries the planning turns forward. Mirrors Rust's `last_state`.
    let lastState: SessionState = sessionState;
    // Global turn cap (the hard stop). `null` ⇒ no global turn ceiling.
    const globalMaxTurns = task.budget.max_turns ?? null;

    for (let index = 0; index < totalTasks; index += 1) {
      const taskId = taskList.tasks[index]!.id;
      const instruction = taskList.tasks[index]!.description;

      // Q1: per-task turn allocation, derived at the START of this step.
      // remaining_tasks = not-yet-started tasks including this one.
      const remainingTasks = totalTasks - index;
      let perTaskTurns: number;
      if (globalMaxTurns != null) {
        const remainingTurns = Math.max(globalMaxTurns - carried.turns, 0);
        perTaskTurns = Math.max(Math.floor(remainingTurns / remainingTasks), 1);
      } else {
        // No global turn cap: the sub-loop is bounded only by the other
        // (token / wall / cost) budget gates.
        perTaskTurns = Number.MAX_SAFE_INTEGER;
      }
      // The sub-loop's effective cap is RELATIVE to the carried turns:
      // runReactInner gates on the cumulative `budgetUsed.turns`, so a per-task
      // cap of K means "stop K turns from now" while the global budget (carried
      // forward) remains the hard stop.
      const subLoopCap =
        perTaskTurns === Number.MAX_SAFE_INTEGER
          ? Number.MAX_SAFE_INTEGER
          : carried.turns + perTaskTurns;

      // Mark in_progress (pending -> in_progress) and re-persist (Q4).
      updateTask(taskList, taskId, "in_progress");
      await this.persistTaskList(sessionId, taskList);

      // Fire on_task_advance (pre, mutable). The hook may rewrite the step's
      // instruction via the carried Task; the (possibly mutated) instruction
      // seeds the sub-loop.
      const stepTask: Task = {
        id: task.id,
        instruction,
        session_id: sessionId,
        budget: task.budget,
        loop_strategy: task.loop_strategy,
      };
      if (this.config.hooks) {
        const ctx: HookContext = {
          event: "on_task_advance",
          session_id: sessionId,
          task: stepTask,
          task_index: index,
          total_tasks: totalTasks,
        };
        try {
          await this.config.hooks.fire(ctx, signal);
        } catch {
          // Hook errors are non-fatal; the step proceeds with the current task.
        }
      }

      // Seed the (possibly mutated) step instruction as a user message, then run
      // the bounded ReAct sub-loop carrying the shared budget (Q1). The sub-loop
      // works on a CLONE of the accumulated session; on success the clone's
      // resulting state is folded back into `sessionState` so the NEXT step
      // builds on this step's tool results and assistant output.
      await this.config.contextManager.appendUserMessage(sessionState, stepTask.instruction);
      const subState: SessionState = {
        messages: [...sessionState.messages],
        extras: { ...sessionState.extras },
      };
      const subBudget: BudgetSnapshot = { ...carried };

      const subResult = await this.runReactInner(
        stepTask,
        subLoopCap,
        subState,
        subBudget,
        onStream,
        signal,
        false,
      );

      if (subResult.kind === "success") {
        // Carry the shared budget forward (Q1): cumulative turns are the
        // sub-loop's absolute count; fold in its token usage.
        carried.turns = subResult.turns;
        carried.input_tokens += subResult.usage.input_tokens;
        carried.output_tokens += subResult.usage.output_tokens;
        totalUsage.input_tokens += subResult.usage.input_tokens;
        totalUsage.output_tokens += subResult.usage.output_tokens;
        totalUsage.cache_read_tokens += subResult.usage.cache_read_tokens;
        totalUsage.cache_write_tokens += subResult.usage.cache_write_tokens;
        totalUsage.cost_usd += subResult.usage.cost_usd;
        lastOutput = subResult.output;
        // Fold this step's resulting conversation back into the SHARED execute
        // context so the NEXT step's sub-loop (which clones `sessionState`)
        // builds on this step's tool results and assistant output, not just its
        // instruction. The returned step state already contains this step's
        // instruction + tool calls + tool results + final output.
        const stepState = runResultSessionState(subResult);
        sessionState.messages = stepState.messages;
        sessionState.extras = stepState.extras;
        // The terminal RunResult carries the LAST completed step's full state
        // (now the accumulated context, Q2/#102). Snapshot it so a later step's
        // seeded user message cannot retroactively mutate this result.
        lastState = {
          messages: [...sessionState.messages],
          extras: { ...sessionState.extras },
        };

        // Mark completed and re-persist (Q4).
        completeTask(taskList, taskId);
        await this.persistTaskList(sessionId, taskList);
        // Surface the completed step's final text to the caller's sink — the
        // parent-visible step boundary.
        emit(onStream, { kind: "final_response", content: lastOutput });
      } else if (subResult.kind === "failure") {
        // Q5: any non-success step aborts the whole run. A budget halt surfaces
        // as budget_exceeded (mid-execute exhaustion); other failures surface as
        // step_failed carrying the step context.
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
      } else {
        // A step surfacing to a human pauses the whole run; propagate.
        return subResult;
      }
    }

    // Q2: success output is the LAST completed step's final_response text.
    return {
      kind: "success",
      output: lastOutput,
      session_id: sessionId,
      usage: totalUsage,
      turns: carried.turns,
      session_state: lastState,
    };
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
    // CRITICAL (#93): the directive must NOT mutate the SHARED `sessionState`.
    // That same state is threaded into the execute phase, where each subtask
    // sub-loop assembles its context from it. If the directive leaked in,
    // every execute step would still see "respond with {tasks, rationale}"
    // and an instruction-following model would re-emit a plan instead of
    // calling tools. Append to a throwaway CLONE so the plan turn sees the
    // directive while the shared state stays `[user: task.instruction]`.
    const directive =
      "Produce a step-by-step plan for the following task. Respond with a " +
      'single JSON object: {"tasks": [<ordered step strings>], ' +
      '"rationale": <string>}.\n\nTask:\n' +
      task.instruction;
    const planState = structuredClone(sessionState);
    await this.config.contextManager.appendUserMessage(planState, directive);

    // Assemble + invoke the planner for exactly ONE turn (R1).
    const context = await this.config.contextManager.assemble(planState, task, signal);
    // Per-run model params win unconditionally (issue #93) — same seam as
    // runReactInner, before the plan turn is dispatched.
    context.params = this.config.modelParams;
    emit(onStream, { kind: "turn_start", turn: budgetUsed.turns + 1 });
    const turnStartedAt = Timestamp.now();
    const turnClock = Date.now();
    // Issue #103: stream the plan turn's deltas when a sink is attached.
    const result = onStream
      ? await turnStreaming(
          planner,
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
      : await planner.turn(context, signal);
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

    // R4: persist the produced artifact to the RunStore seam under
    // PLAN_EXECUTE_EXTRAS_KEY as a JSON value (the stable cross-language shape:
    // `{ tasks, rationale }`). #76 — the durable single source of truth; no
    // longer mirrored into SessionState.extras. The put failure is swallowed
    // (matching persistTaskList): a successfully-captured plan must not be lost
    // to a storage hiccup (the default no-op / in-memory provider never fails).
    try {
      await this.storage().run().put(sessionId, PLAN_EXECUTE_EXTRAS_KEY, {
        tasks: artifact.tasks,
        rationale: artifact.rationale,
      });
    } catch {
      // Durable write failure is non-fatal.
    }

    return { ok: true, artifact, usage, turns: budgetUsed.turns };
  }

  /**
   * Drive the `self_verifying` loop strategy (issue #61).
   *
   * A loop-within-a-loop. Each iteration: (1) a bounded ReAct BUILD sub-loop runs
   * until the agent claims done, carrying the shared budget; (2) a fresh EVALUATE
   * run executes over a read-only sandbox in a never-shared session with the
   * evaluator agent and the `role-evaluator` chunk; (3) the {@link Verifier}
   * renders a verdict. `passed` ⇒ `success`; `failed` ⇒ the reason is injected
   * into the build context (the SAME user-message path the Stop-block / reject
   * resume uses — D1) and the loop continues. Running out of the verifier's
   * `maxIterations` round-trips without a pass yields
   * `failure { self_verify_exhausted }` (D4 — the stagnation guard / Default-FAIL
   * contract). An absent `config.verifier` yields `failure
   * { self_verify_misconfigured }` (D4 — a typed halt, NOT a throw).
   *
   * Budgets fold BOTH phases across ALL iterations (R8). Build and evaluate are
   * distinguishable in traces via distinct session ids: the build keeps the
   * task's session, each evaluate gets a freshly generated one (R2/R9).
   */
  private async runSelfVerifying(
    task: Task,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    _onStream: StreamSink | undefined,
  ): Promise<RunResult> {
    const buildSessionId = task.session_id;

    // D4/R11: a missing verifier is a typed halt, not a throw.
    const verifier = this.config.verifier;
    if (verifier == null) {
      const result: RunResult = {
        kind: "failure",
        reason: {
          kind: "self_verify_misconfigured",
          reason: "self_verifying requires `config.verifier`, but it is absent",
        },
        session_id: buildSessionId,
        usage: emptyAggregateUsage(),
        turns: 0,
      };
      await this.finalizeSelfVerifying(result);
      return result;
    }

    const maxIterations = verifier.maxIterations();
    // Shared budget threaded across every build + evaluate sub-run (R8).
    const carried: BudgetSnapshot = { ...budgetUsed };
    // Cumulative usage across ALL build + evaluate runs of ALL iterations (R8).
    const totalUsage: AggregateUsage = emptyAggregateUsage();
    // The most recent verifier failure reason (for self_verify_exhausted).
    let lastReason = "";

    for (let iteration = 0; iteration < maxIterations; iteration += 1) {
      // ── Build phase (R1): a bounded ReAct sub-loop carrying the shared
      //    budget. Iteration 0's seed instruction is delivered by the sub-loop;
      //    later iterations already have the prior verdict reason injected as a
      //    user message (R6), so they do NOT re-seed.
      const buildTask: Task = {
        id: task.id,
        instruction: task.instruction,
        session_id: buildSessionId,
        budget: task.budget,
        loop_strategy: task.loop_strategy,
      };
      const buildCap = task.budget.max_turns ?? Number.MAX_SAFE_INTEGER;
      const buildResult = await this.runReactInner(
        buildTask,
        buildCap,
        sessionState,
        { ...carried },
        // Sub-loops run with a suppressed sink (mirrors PlanExecute); terminal
        // observability is finalized by this strategy.
        undefined,
        undefined,
        iteration === 0,
      );
      foldUsage(totalUsage, carried, buildResult);

      // A build run that paused / escalated is propagated up unchanged — the
      // caller must handle it before verification can resume.
      if (buildResult.kind === "waiting_for_human") {
        return buildResult;
      }
      // A build run pausing to consult (issue #114) pauses the whole run;
      // propagate so the caller (or SubagentTool) mediates.
      if (buildResult.kind === "consult") {
        return buildResult;
      }
      if (buildResult.kind === "escalate") {
        await this.finalizeSelfVerifying(buildResult);
        return buildResult;
      }

      // ── Evaluate phase (R2/R3/R4): a fresh evaluator RUN. Distinct generated
      //    session id (never shared with build — R2/R9), a read-only sandbox
      //    derived internally (R3), the evaluator agent (D2 defaulting), and the
      //    `role-evaluator` chunk (R4).
      const evalResult = await this.runEvaluatePhase(task, carried, totalUsage);

      // ── Verdict.
      const input: VerifierInput = {
        build_result: buildResult,
        eval_result: evalResult,
        workspace: this.config.sandbox.workspaceRoot?.() ?? "",
        iteration,
      };
      const verdict = await verifier.verify(input);
      if (verdict.kind === "passed") {
        // Reuse the build run's output as the run's output. Carry the build
        // run's final conversation history into the terminal result (issue
        // #102) so a SelfVerifying caller resumes losslessly; fall back to the
        // running build context for a non-Success build a bespoke verifier
        // accepted.
        const output = buildResult.kind === "success" ? buildResult.output : "";
        const turns = buildResult.kind === "success" ? buildResult.turns : carried.turns;
        const finalState =
          buildResult.kind === "success" ? runResultSessionState(buildResult) : sessionState;
        const result: RunResult = {
          kind: "success",
          output,
          session_id: buildSessionId,
          usage: totalUsage,
          turns,
          session_state: finalState,
        };
        await this.finalizeSelfVerifying(result);
        return result;
      }
      // R5/R6: Default-FAIL keeps looping; inject the reason into the build
      // context via the SAME path the Stop-block uses (append a user message) so
      // the next build iteration sees it.
      lastReason = verdict.reason;
      await this.config.contextManager.appendUserMessage(sessionState, verdict.reason);
    }

    // R7: ran out of round-trips without a pass — clean exhaustion. Carry the
    // live build session history (issue #102).
    const result: RunResult = {
      kind: "failure",
      reason: { kind: "self_verify_exhausted", iterations: maxIterations, last_reason: lastReason },
      session_id: buildSessionId,
      usage: totalUsage,
      turns: carried.turns,
      session_state: sessionState,
    };
    await this.finalizeSelfVerifying(result);
    return result;
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
    carried: BudgetSnapshot,
    totalUsage: AggregateUsage,
  ): Promise<RunResult> {
    // D2: evaluator-agent defaulting — identical contract to plannerAgent.
    const evaluator = this.config.evaluatorAgent ?? this.config.agent;

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

    // Child harness: clone the config, swap agent + sandbox. Cloning shares the
    // same observability/storage seams so the evaluate run's spans land in the
    // SAME trace stream (distinguished by its distinct session id).
    const evalConfig: HarnessConfig = {
      ...this.config,
      agent: evaluator,
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

  /**
   * Finalize observability for a terminal `self_verifying` outcome. Mirrors the
   * tail of {@link runReact} / {@link finalizePlanExecute}. `waiting_for_human`
   * is not terminal and is never flushed here.
   */
  private async finalizeSelfVerifying(result: RunResult): Promise<void> {
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
      // Consult (issue #114) is non-terminal; do not finalize.
      case "consult":
        break;
      case "escalate":
        await this.finalizeObservability(result.session_id, { kind: "escalated" });
        break;
    }
  }

  // --------------------------------------------------------------------------
  // HillClimbing (issue #60) — iterative optimization loop
  // --------------------------------------------------------------------------

  /**
   * Drive the `hill_climbing` loop strategy (issue #60) — the iterative
   * optimization loop. Establish a baseline metric (iteration 0, NO agent turn),
   * then for each subsequent iteration: run one bounded ReAct agent turn to
   * propose a change, re-evaluate the metric, and keep or revert based on
   * {@link shouldKeep}. The harness — never the agent — writes the results log
   * to `{workspace_root}/.spore/results/{task_id}.tsv`.
   *
   * ## Resolved spec decisions (issue #60 — all FINAL, mirror Rust exactly)
   * - **D1 (revert):** `revert_on_no_improvement` ⇒ `git reset --hard HEAD` runs
   *   through the sandbox's `executeCommand` seam directly from the harness. The
   *   harness NEVER commits; `commit_hash` is the empty string (no VcsProvider).
   * - **D2/D3 (TSV):** tab-separated, REQUIRED header, one row per iteration in
   *   ascending order. Floats (`metric_value`, `duration_secs`) use EXACTLY 6
   *   decimals; `metric_value` is the empty string on crashed/timeout rows;
   *   `metadata` is excluded. `direction`/`status` are snake_case.
   * - **D4 (direction):** the payload `direction` is authoritative for the
   *   keep/revert decision and is the value recorded in the TSV.
   * - **D5 (baseline):** iteration 0 is a pure baseline measurement — no agent
   *   turn, `status: kept`, `current_best` set. Agent turns start at iteration 1.
   * - **D6 (misconfiguration):** an absent `config.metricEvaluator` ⇒ a typed
   *   halt `hill_climbing_misconfigured`, NOT a throw.
   * - **D7 (baseline error):** if the iteration-0 baseline evaluation itself
   *   errors, it is treated as `hill_climbing_misconfigured` (no `current_best`
   *   to climb from), NOT a stagnation increment.
   *
   * ## Loop semantics
   * Iterations 1+: agent turn → evaluate → `shouldKeep(new, currentBest,
   * payloadDirection, minImprovementDelta)`. Keep ⇒ `status: kept`, update best,
   * reset the stagnation counter. No keep ⇒ `status: discarded`, optional revert,
   * increment the counter. An evaluator error (crashed/timeout) records an
   * empty-metric row, counts as a non-improvement, and the loop continues.
   * `max_stagnation = N` ⇒ halt `stagnation_limit_reached { iterations,
   * best_metric }` after N consecutive non-improvements (an improvement resets
   * the counter); `null` ⇒ run until the budget / turn cap is hit.
   */
  private async runHillClimbing(
    task: Task,
    direction: HillClimbingDirection,
    maxStagnation: number | null,
    revertOnNoImprovement: boolean,
    minImprovementDelta: number | null,
    budgetUsed: BudgetSnapshot,
    _onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
  ): Promise<RunResult> {
    const sessionId = task.session_id;
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";

    // D6: a missing evaluator is a typed halt, not a throw.
    const evaluator = this.config.metricEvaluator;
    if (evaluator == null) {
      const result: RunResult = {
        kind: "failure",
        reason: {
          kind: "hill_climbing_misconfigured",
          reason: "hill_climbing requires `config.metricEvaluator`, but it is absent",
        },
        session_id: sessionId,
        usage: emptyAggregateUsage(),
        turns: 0,
      };
      await this.finalizeHillClimbing(result);
      return result;
    }

    const description = evaluator.description();
    // Per-iteration observability span counter.
    const spanSeq = { value: 0 };
    // Cumulative usage + turns across ALL agent-turn iterations.
    const totalUsage: AggregateUsage = emptyAggregateUsage();
    // Shared budget threaded across every agent sub-run.
    const carried: BudgetSnapshot = { ...budgetUsed };
    // The TSV rows, in iteration order.
    const rows: ResultsEntry[] = [];

    // A snapshot for the evaluator. HillClimbing keeps no carried message state
    // of its own (each iteration is a fresh sub-run), so a default SessionState
    // is the right snapshot to hand the evaluator.
    const snapshot = newSessionStateSnapshot(
      sessionId,
      task.id,
      emptySessionState(),
      workspaceRoot,
    );

    // ── Iteration 0: pure baseline. No agent turn (D5).
    const baseline = await evaluator.evaluate(this.config.sandbox, snapshot, signal);
    let currentBest: number;
    if (baseline.kind === "ok") {
      currentBest = baseline.result.value;
      rows.push({
        iteration: 0,
        commit_hash: await this.hillClimbingCommitHash(),
        metric_value: currentBest,
        direction,
        status: "kept",
        duration: baseline.result.duration,
        description,
        metadata: {},
      });
      this.emitHillClimbingSpan(sessionId, task.id, spanSeq, 0, currentBest, null, "kept", false);
    } else {
      // D7: a baseline that cannot even be measured is a misconfiguration of the
      // experiment, not a non-improvement to climb away from — there is no
      // `current_best` to compare against. Record the failed row, write the TSV,
      // and halt.
      const status = iterationStatusFromError(baseline.error);
      rows.push({
        iteration: 0,
        commit_hash: await this.hillClimbingCommitHash(),
        // Sentinel; excluded from the TSV (crashed/timeout ⇒ empty).
        metric_value: NaN,
        direction,
        status,
        duration: 0,
        description,
        metadata: {},
      });
      this.emitHillClimbingSpan(sessionId, task.id, spanSeq, 0, null, null, status, false);
      await this.writeHillClimbingTsv(workspaceRoot, task.id, rows);
      const result: RunResult = {
        kind: "failure",
        reason: {
          kind: "hill_climbing_misconfigured",
          reason: `baseline evaluation failed: ${metricErrorMessage(baseline.error)}`,
        },
        session_id: sessionId,
        usage: totalUsage,
        turns: carried.turns,
      };
      await this.finalizeHillClimbing(result);
      return result;
    }

    // Consecutive non-improvement counter (the stagnation halt).
    let stagnation = 0;
    // The 0-based iteration index; agent turns begin at 1.
    let iteration = 1;

    for (;;) {
      // Budget gate before the iteration's agent turn (mirrors runReactInner).
      const turnCap = task.budget.max_turns ?? Number.MAX_SAFE_INTEGER;
      if (carried.turns >= turnCap) {
        break;
      }
      const overrun = budgetExceeded(task.budget, carried, Date.now());
      if (overrun != null) {
        // A wall-time/cost/token cap reached BEFORE any iteration work is a clean
        // budget halt. Mirrors runReactInner's contract.
        await this.writeHillClimbingTsv(workspaceRoot, task.id, rows);
        const result: RunResult = {
          kind: "failure",
          reason: { kind: "budget_exceeded", limit_type: overrun },
          session_id: sessionId,
          usage: totalUsage,
          turns: carried.turns,
        };
        await this.finalizeHillClimbing(result);
        return result;
      }

      // ── One bounded agent turn proposes a change. The sub-run carries the
      //    shared budget so per-iteration turns count toward the cap. Each
      //    iteration is a fresh, isolated ReAct sub-loop.
      const iterTask: Task = {
        id: task.id,
        instruction: task.instruction,
        session_id: sessionId,
        budget: task.budget,
        loop_strategy: task.loop_strategy,
      };
      const iterState = emptySessionState();
      const turnResult = await this.runReactInner(
        iterTask,
        turnCap,
        iterState,
        { ...carried },
        undefined,
        signal,
        true,
      );
      foldUsage(totalUsage, carried, turnResult);

      // A turn that paused / escalated is propagated up unchanged.
      if (turnResult.kind === "waiting_for_human") {
        return turnResult;
      }
      // A turn pausing to consult (issue #114) pauses the whole run; propagate.
      if (turnResult.kind === "consult") {
        return turnResult;
      }
      if (turnResult.kind === "escalate") {
        await this.writeHillClimbingTsv(workspaceRoot, task.id, rows);
        await this.finalizeHillClimbing(turnResult);
        return turnResult;
      }

      // ── Evaluate the metric after the change.
      const evalOutcome = await evaluator.evaluate(this.config.sandbox, snapshot, signal);
      if (evalOutcome.kind === "ok") {
        const value = evalOutcome.result.value;
        const kept = shouldKeep(value, currentBest, direction, minImprovementDelta);
        const delta = direction === "minimize" ? currentBest - value : value - currentBest;
        if (kept) {
          currentBest = value;
          stagnation = 0;
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
            task.id,
            spanSeq,
            iteration,
            value,
            delta,
            "kept",
            false,
          );
        } else {
          // No improvement (D1: optionally revert).
          const reverted = revertOnNoImprovement;
          if (reverted) {
            await this.hillClimbingRevert(signal);
          }
          stagnation += 1;
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
            task.id,
            spanSeq,
            iteration,
            value,
            delta,
            "discarded",
            reverted,
          );
        }
      } else {
        // Crash/timeout/etc.: counts as a non-improvement. Optionally revert,
        // increment stagnation, record an empty-metric row.
        const status = iterationStatusFromError(evalOutcome.error);
        const reverted = revertOnNoImprovement;
        if (reverted) {
          await this.hillClimbingRevert(signal);
        }
        stagnation += 1;
        rows.push({
          iteration,
          commit_hash: await this.hillClimbingCommitHash(),
          // Sentinel; excluded from the TSV (crashed/timeout ⇒ empty).
          metric_value: NaN,
          direction,
          status,
          duration: 0,
          description,
          metadata: {},
        });
        this.emitHillClimbingSpan(
          sessionId,
          task.id,
          spanSeq,
          iteration,
          null,
          null,
          status,
          reverted,
        );
      }

      // ── Stagnation halt (only when a cap is configured).
      if (maxStagnation != null && stagnation >= maxStagnation) {
        await this.writeHillClimbingTsv(workspaceRoot, task.id, rows);
        const result: RunResult = {
          kind: "failure",
          reason: {
            kind: "stagnation_limit_reached",
            iterations: stagnation,
            best_metric: currentBest,
          },
          session_id: sessionId,
          usage: totalUsage,
          turns: carried.turns,
        };
        await this.finalizeHillClimbing(result);
        return result;
      }

      iteration += 1;
    }

    // Budget/turn cap reached without a stagnation halt — clean budget halt.
    await this.writeHillClimbingTsv(workspaceRoot, task.id, rows);
    const result: RunResult = {
      kind: "failure",
      reason: { kind: "budget_exceeded", limit_type: "turns" },
      session_id: sessionId,
      usage: totalUsage,
      turns: carried.turns,
    };
    await this.finalizeHillClimbing(result);
    return result;
  }

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

  /**
   * Finalize observability for a terminal `hill_climbing` outcome. Mirrors the
   * tail of {@link runReact} / {@link finalizeSelfVerifying}. `waiting_for_human`
   * is not terminal and is never flushed here.
   */
  private async finalizeHillClimbing(result: RunResult): Promise<void> {
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
      // Consult (issue #114) is non-terminal; do not finalize.
      case "consult":
        break;
      case "escalate":
        await this.finalizeObservability(result.session_id, { kind: "escalated" });
        break;
    }
  }

  // --------------------------------------------------------------------------
  // Ralph (issue #58) — multi-context-window continuation loop
  // --------------------------------------------------------------------------

  /**
   * Drive the `ralph` loop strategy (issue #58) — the multi-context-window
   * continuation loop. Each OUTER iteration is ONE context window: a FRESH
   * {@link SessionState} (no message carryover) re-seeded with the instruction
   * plus the reloaded `.spore/` state, then a bounded inner ReAct sub-loop. The
   * external completion check (B1) reads `.spore/progress.json` +
   * `.spore/feature_list.json` — the SAME files the registered `ralph-stop` hook
   * reads. Incomplete ⇒ reset into a new window; all complete ⇒ `success`.
   *
   * ## Resolved spec decisions (issue #58 — all FINAL)
   * - **B1:** completion is driven off the Stop hook; the OUTER loop consults the
   *   SAME filesystem check ({@link ralphCompletionStatus}) to decide reset vs
   *   success. No `completion_check` config field; the deprecated CompletionCheck
   *   trait is NOT reused.
   * - **B2:** canonical paths `.spore/progress.json` + `.spore/feature_list.json`.
   * - **B3:** {@link HarnessConfig.maxResets} (default 3) caps the OUTER loop,
   *   independent of `max_turns`. Exhausting it with tasks incomplete yields
   *   {@link HaltReason} `ralph_completion_unmet { iterations, last_reason }`.
   * - **B4:** v1 reloads ONLY progress + feature_list (no git). Hermetic.
   *
   * ## Rules enforced (each maps to a test):
   * - R1 the model's exit attempt RESETS the context window instead of
   *   terminating, while tasks remain incomplete.
   * - R2 each reset builds a FRESH {@link SessionState} — no message carryover.
   * - R3 the filesystem reload injects the reloaded `.spore/` content into the
   *   fresh seed.
   * - R4 `incomplete,incomplete,complete` ⇒ `success` at iteration 3.
   * - R5 always-incomplete ⇒ exactly `maxResets` iterations ⇒
   *   `failure { ralph_completion_unmet }`.
   * - R6 budgets fold across ALL context windows.
   * - R7 each reset is traceable (a distinct generated session id per window).
   */
  private async runRalph(
    task: Task,
    _budgetUsed: BudgetSnapshot,
    _onStream: StreamSink | undefined,
  ): Promise<RunResult> {
    const workspaceRoot = this.config.sandbox.workspaceRoot?.() ?? "";
    const maxResets = Math.max(this.config.maxResets ?? DEFAULT_MAX_RESETS, 1);

    // Cumulative usage + turns across ALL context windows (R6). Each window is a
    // fresh start with its own per-window turn budget; token/turn accounting is
    // accumulated separately for terminal reporting.
    const totalUsage: AggregateUsage = emptyAggregateUsage();
    const carriedAccounting: BudgetSnapshot = emptyBudgetSnapshot();
    // The most recent incompletion reason (for ralph_completion_unmet).
    let lastReason = ".spore/progress.json missing";
    // Session id of the most recent context window (terminal accounting).
    let lastSessionId = task.session_id;

    // The OUTER loop: each iteration is ONE context window (R1). `maxResets`
    // caps the number of windows (B3).
    for (let iteration = 0; iteration < maxResets; iteration += 1) {
      // R7: a fresh, distinct session id per context window so each reset is
      // independently traceable. Window 0 keeps the task's session id.
      const windowSessionId = iteration === 0 ? task.session_id : SessionId.generate();
      lastSessionId = windowSessionId;

      // R2: a FRESH SessionState per window — no message carryover.
      const sessionState = emptySessionState();

      // R2: seed the instruction, then R3: reload the deterministic `.spore/`
      // state from the filesystem and inject it as context so the fresh window
      // knows what is already done / still outstanding.
      await this.config.contextManager.appendUserMessage(sessionState, task.instruction);
      const reload = StandardHarness.ralphReloadContext(workspaceRoot);
      if (reload != null) {
        await this.config.contextManager.appendUserMessage(sessionState, reload);
      }
      // R3 (issue #58 v2): when a VcsProvider is wired, ALSO reload git history
      // and inject it as a delimited "Recent VCS history:" section, exactly as
      // the `.spore/` reload content is injected. When the provider is unset
      // (the default), this section is omitted entirely — Ralph's reloaded
      // context is then byte-for-byte the v1 behavior (the B4→none decision).
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

      // The per-window bounded ReAct sub-loop. The registered `ralph-stop` hook
      // (B1) fires inside it on each final response; this strategy's OUTER loop
      // then decides reset vs success. FRESH per-window budget (the reset
      // discards the turn budget); token fold is accumulated via `totalUsage`.
      const windowTask: Task = {
        id: task.id,
        instruction: task.instruction,
        session_id: windowSessionId,
        budget: task.budget,
        loop_strategy: task.loop_strategy,
      };
      const windowCap = task.budget.max_turns ?? Number.MAX_SAFE_INTEGER;
      const windowResult = await this.runReactInner(
        windowTask,
        windowCap,
        sessionState,
        emptyBudgetSnapshot(),
        undefined,
        undefined,
        // The instruction is seeded above; the sub-loop must NOT re-seed it.
        false,
      );
      foldUsage(totalUsage, carriedAccounting, windowResult);

      // A window that paused / escalated is propagated up unchanged.
      if (windowResult.kind === "waiting_for_human") {
        return windowResult;
      }
      // A window pausing to consult (issue #114) pauses the whole run; propagate.
      if (windowResult.kind === "consult") {
        return windowResult;
      }
      if (windowResult.kind === "escalate") {
        await this.finalizeSelfVerifying(windowResult);
        return windowResult;
      }

      // External completion check (B1): consult the SAME filesystem state the
      // Stop hook reads. `null` ⇒ done ⇒ success; otherwise tasks remain ⇒ reset
      // into the next window (R1) unless the cap is reached (R5).
      const status = StandardHarness.ralphCompletionStatus(workspaceRoot);
      if (status == null) {
        const output = windowResult.kind === "success" ? windowResult.output : "";
        // Carry the final context window's history into the terminal result
        // (issue #102).
        const finalState =
          windowResult.kind === "success"
            ? runResultSessionState(windowResult)
            : emptySessionState();
        const result: RunResult = {
          kind: "success",
          output,
          session_id: windowSessionId,
          usage: totalUsage,
          turns: carriedAccounting.turns,
          session_state: finalState,
        };
        await this.finalizeSelfVerifying(result);
        return result;
      }
      lastReason = status;
    }

    // R5: ran out of context-window resets without completion.
    const result: RunResult = {
      kind: "failure",
      reason: { kind: "ralph_completion_unmet", iterations: maxResets, last_reason: lastReason },
      session_id: lastSessionId,
      usage: totalUsage,
      turns: carriedAccounting.turns,
    };
    await this.finalizeSelfVerifying(result);
    return result;
  }

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
            this.config.agent,
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
        : await this.config.agent.turn(context, signal);
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
  private _chunkProvider: ChunkProvider = new InMemoryChunkProvider();
  private _verifier?: Verifier;
  private _evaluatorAgent?: Agent;
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

  /** Inject an alternate agent for the `plan_execute` plan phase (issue #70,
   *  Q1). When set and the loop strategy is `plan_execute`, the one-shot plan
   *  turn runs on this agent; otherwise it runs on the default agent. */
  plannerAgent(agent: Agent): this {
    this._plannerAgent = agent;
    return this;
  }

  /**
   * Register a per-kind consult handler (issue #114). Analogous to
   * {@link plannerAgent}. When a worker (run via {@link "@spore/tools".SubagentTool})
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

  /** Inject an alternate agent for the `self_verifying` evaluate phase (issue
   *  #61, D2). Mirrors {@link plannerAgent}: when set and the strategy is
   *  `self_verifying`, the evaluate run uses this agent; otherwise it runs on
   *  the default agent. */
  evaluatorAgent(agent: Agent): this {
    this._evaluatorAgent = agent;
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

    return {
      agent: this.agent,
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
      plannerAgent: this._plannerAgent,
      chunkProvider: this._chunkProvider,
      verifier: this._verifier,
      evaluatorAgent: this._evaluatorAgent,
      vcsProvider: this._vcsProvider,
      catalogueRegistry,
      systemPrompt: this._systemPrompt,
      modelParams: this._modelParams,
      autoPersistSessions: this._autoPersistSessions,
      prompt_tool_call_flag: this._promptToolCallFlag,
      // Only attach when populated so the default config stays byte-for-byte
      // unchanged for callers that never register a consult handler (R9).
      consultHandlers: this._consultHandlers.size > 0 ? this._consultHandlers : undefined,
      registry: this._registry,
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

/** Derive the `SessionOutcome.failure.reason` string from a {@link HaltReason}.
 *  Mirrors Rust's `format!("{reason:?}")` for the failure outcome. */
function haltReasonToString(reason: HaltReason): string {
  return JSON.stringify(reason);
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
