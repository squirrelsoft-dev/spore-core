/**
 * Public types for the Harness runtime loop (spore-core issue #3).
 *
 * The wire shape mirrors the Rust reference implementation byte-for-byte:
 * tagged unions use a `kind` discriminator in `snake_case`. Static types
 * are derived from zod schemas for safe (de)serialization of `PausedState`
 * and `RunResult` across pause/resume boundaries.
 *
 * Component dependencies (#4–#13) ship in their own issues. Until those
 * land, this module defines minimal forward declarations of the trait
 * surface the loop consumes — each tagged with the owning issue.
 */

import { z } from "zod";

import type { Context } from "../agent/types.js";
import type { AgentError } from "../agent/errors.js";
import type {
  CompactionPreserveHints,
  SessionState as ContextSessionState,
} from "../context/types.js";
import {
  MessageSchema,
  ToolCallSchema,
  type Message,
  type TokenUsage,
  type ToolCall,
  type ToolSchema,
} from "../model/schemas.js";

// ============================================================================
// Identity newtypes
// ============================================================================

let __idCounter = 0;
function randomId(): string {
  __idCounter += 1;
  return __idCounter.toString(16).padStart(16, "0");
}

export class SessionId {
  constructor(readonly value: string) {}
  static of(value: string): SessionId {
    return new SessionId(value);
  }
  /** Fresh, opaque session id (used e.g. by SelfVerifying evaluator). */
  static generate(): SessionId {
    return new SessionId(`sess-${randomId()}`);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: SessionId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

export class TaskId {
  constructor(readonly value: string) {}
  static of(value: string): TaskId {
    return new TaskId(value);
  }
  static generate(): TaskId {
    return new TaskId(`task-${randomId()}`);
  }
  asString(): string {
    return this.value;
  }
  toString(): string {
    return this.value;
  }
  equals(other: TaskId): boolean {
    return this.value === other.value;
  }
  toJSON(): string {
    return this.value;
  }
}

// Schemas (string) for SessionId / TaskId on the wire.
export const SessionIdSchema = z.string().transform((s) => new SessionId(s));
export const TaskIdSchema = z.string().transform((s) => new TaskId(s));

// ============================================================================
// Budget tracking
// ============================================================================

export const BudgetLimitsSchema = z.object({
  max_turns: z.number().int().nonnegative().nullable().optional(),
  max_input_tokens: z.number().int().nonnegative().nullable().optional(),
  max_output_tokens: z.number().int().nonnegative().nullable().optional(),
  /** Wall time limit, seconds. */
  max_wall_time: z.number().nonnegative().nullable().optional(),
  max_cost_usd: z.number().nonnegative().nullable().optional(),
});
export type BudgetLimits = z.infer<typeof BudgetLimitsSchema>;

export const BudgetLimitTypeSchema = z.enum([
  "turns",
  "input_tokens",
  "output_tokens",
  "wall_time",
  "cost_usd",
]);
export type BudgetLimitType = z.infer<typeof BudgetLimitTypeSchema>;

export const BudgetSnapshotSchema = z.object({
  turns: z.number().int().nonnegative().default(0),
  input_tokens: z.number().int().nonnegative().default(0),
  output_tokens: z.number().int().nonnegative().default(0),
  wall_time: z.number().nonnegative().nullable().optional(),
  cost_usd: z.number().nonnegative().default(0),
});
export type BudgetSnapshot = z.infer<typeof BudgetSnapshotSchema>;

export function emptyBudgetSnapshot(): BudgetSnapshot {
  return { turns: 0, input_tokens: 0, output_tokens: 0, cost_usd: 0 };
}

export const AggregateUsageSchema = z.object({
  input_tokens: z.number().int().nonnegative().default(0),
  output_tokens: z.number().int().nonnegative().default(0),
  cache_read_tokens: z.number().int().nonnegative().default(0),
  cache_write_tokens: z.number().int().nonnegative().default(0),
  cost_usd: z.number().nonnegative().default(0),
});
export type AggregateUsage = z.infer<typeof AggregateUsageSchema>;

export function emptyAggregateUsage(): AggregateUsage {
  return {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cost_usd: 0,
  };
}

export function addTurnUsage(agg: AggregateUsage, u: TokenUsage): void {
  agg.input_tokens += u.input_tokens;
  agg.output_tokens += u.output_tokens;
  agg.cache_read_tokens += u.cache_read_tokens ?? 0;
  agg.cache_write_tokens += u.cache_write_tokens ?? 0;
}

// ============================================================================
// Task + loop strategy
// ============================================================================

export const OptimizationDirectionSchema = z.enum(["minimize", "maximize"]);
export type OptimizationDirection = z.infer<typeof OptimizationDirectionSchema>;

export const ModelConfigSchema = z.object({
  provider: z.string(),
  model_id: z.string(),
});
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

/**
 * Loop strategy. Data shapes are canonical; only `react` is fully executable
 * in {@link StandardHarness}. Other variants return {@link HaltReason} of
 * `strategy_not_yet_implemented` until the trait dependencies (CompletionCheck,
 * Verifier, MetricEvaluator) ship in later component issues.
 */
export const LoopStrategySchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("re_act"), max_iterations: z.number().int().nonnegative() }),
  z.object({
    kind: z.literal("plan_execute"),
    plan_model: ModelConfigSchema.nullable().optional(),
  }),
  z.object({ kind: z.literal("ralph") }),
  z.object({ kind: z.literal("self_verifying") }),
  z.object({
    kind: z.literal("hill_climbing"),
    direction: OptimizationDirectionSchema,
    max_stagnation: z.number().int().nonnegative().nullable().optional(),
    revert_on_no_improvement: z.boolean(),
    min_improvement_delta: z.number().nullable().optional(),
  }),
]);
export type LoopStrategy = z.infer<typeof LoopStrategySchema>;

export const TaskSchema = z.object({
  id: z.string().transform((s) => new TaskId(s)),
  instruction: z.string(),
  session_id: z.string().transform((s) => new SessionId(s)),
  budget: BudgetLimitsSchema,
  loop_strategy: LoopStrategySchema,
});
export type Task = {
  id: TaskId;
  instruction: string;
  session_id: SessionId;
  budget: BudgetLimits;
  loop_strategy: LoopStrategy;
};

export function newTask(
  instruction: string,
  session_id: SessionId,
  loop_strategy: LoopStrategy,
  budget: BudgetLimits = {},
): Task {
  return {
    id: TaskId.generate(),
    instruction,
    session_id,
    budget,
    loop_strategy,
  };
}

// ============================================================================
// Streaming events emitted by the harness
// ============================================================================

export type HarnessStreamEvent =
  | { kind: "turn_start"; turn: number }
  | { kind: "turn_end"; turn: number }
  | { kind: "tool_call"; call_id: string; name: string }
  | { kind: "tool_result"; call_id: string; is_error: boolean }
  | { kind: "final_response"; content: string }
  | { kind: "budget_warning"; limit_type: BudgetLimitType };

export type StreamSink = (event: HarnessStreamEvent) => void;

// ============================================================================
// Forward-declared sibling component types
// ============================================================================

/** Tool dispatch output. Full type lives in issue #4/#5; this covers loop routing. */
export type ToolOutput =
  | { kind: "success"; content: string; truncated?: boolean }
  | { kind: "error"; message: string; recoverable: boolean }
  | { kind: "waiting_for_human"; child_state: ChildPausedState; request: HumanRequest };

export interface ToolResultRecord {
  call_id: string;
  output: ToolOutput;
}

/** Sandbox violation — issue #6.
 *
 * Discriminated union, `kind` tag in `snake_case`. Wire-compatible with the
 * Rust `SandboxViolation` enum (which uses `#[serde(tag = "kind")]`).
 */
export type SandboxViolation =
  | { kind: "path_escape"; path: string }
  | { kind: "path_denied"; path: string; matched_rule: string }
  | { kind: "extension_denied"; path: string; extension: string }
  | { kind: "read_only_violation"; path: string }
  | { kind: "file_size_exceeded"; path: string; size: number; limit: number }
  | { kind: "disallowed_command"; command: string }
  | { kind: "network_violation"; host: string };

export function sandboxViolationIsAlwaysHalt(v: SandboxViolation): boolean {
  return v.kind === "path_escape" || v.kind === "network_violation";
}

// ============================================================================
// Sandbox isolation modes — issue #6
// ============================================================================

/** Read/write/execute operation tag — passed to `resolvePath`. */
export type Operation = "read" | "write" | "execute";

/** Bubblewrap profile — placeholder; backend not wired in v1. */
export interface BwrapProfile {
  /** Free-form profile name; semantics deferred to the backend. */
  name?: string;
}

/** Docker network policy. */
export type NetworkPolicy =
  | { kind: "none" }
  | { kind: "allowlist"; hosts: string[] }
  | { kind: "full" };

/** Discriminated isolation mode. */
export type IsolationMode =
  | { kind: "none" }
  | { kind: "workspace_scoped" }
  | { kind: "bubblewrap"; profile: BwrapProfile }
  | { kind: "docker"; image: string; network: NetworkPolicy };

/** Configuration consumed by `WorkspaceScopedSandbox`. */
export interface WorkspaceConfig {
  /** Workspace root. Canonicalized at construction. */
  root: string;
  /** Explicit allowlist (relative to root, or absolute under root). */
  allowed_paths?: string[];
  /** Explicit denylist; evaluated after the allowlist. */
  denied_paths?: string[];
  /** Allowed extensions. `undefined` = allow all. Advisory in v1. */
  allowed_extensions?: string[];
  /** Denied extensions (e.g. `["env", "pem", "key"]`). Leading dot tolerated. */
  denied_extensions?: string[];
  /** If true, Write/Execute resolve to ReadOnlyViolation. */
  read_only?: boolean;
  /** Max file size (bytes) for reads. `0` disables. */
  max_file_size?: number;
}

export type HookPoint = "before_turn" | "before_tool" | "after_tool" | "before_completion";

export type TerminationDecision = { kind: "continue" } | { kind: "halt"; reason: string };

/**
 * Session state handed back and forth across pause/resume. The harness does
 * not interpret its contents; it round-trips opaquely so that #7
 * (ContextManager) and #8 (MemoryProvider) own the schema.
 */
export const SessionStateSchema = z.object({
  messages: z.array(MessageSchema).default([]),
  extras: z.record(z.unknown()).default({}),
});
export type SessionState = z.infer<typeof SessionStateSchema>;

export function emptySessionState(): SessionState {
  return { messages: [], extras: {} };
}

// ============================================================================
// Forward-declared sibling component interfaces
// ============================================================================

/** Issue #4 — ToolRegistry. */
export interface ToolRegistry {
  dispatch(call: ToolCall, signal?: AbortSignal): Promise<ToolOutput>;
  isAlwaysHalt(toolName: string): boolean;
  schemas(): ToolSchema[];
}

/** Output of a sandboxed command. */
export interface CommandOutput {
  stdout: string;
  stderr: string;
  exit_code: number;
  timed_out: boolean;
  /** Additive field (#6): true when stdout/stderr were truncated upstream. */
  truncated: boolean;
}

/** Reference to a file containing offloaded full content. */
export interface FileRef {
  path: string;
  size: number;
}

/** Result of routing oversized tool output through the sandbox (#6 shape). */
export interface TruncatedOutput {
  /** head + separator + tail (or original content when below threshold). */
  content: string;
  truncated: boolean;
  /** Path + size of the offloaded full content, when one was written. */
  full_ref: FileRef | null;
  /** Original (untruncated) size in characters. */
  original_size: number;
}

/** Issue #6 — SandboxProvider.
 *
 * `executeCommand`, `handleLargeOutput`, and `resolvePath` mirror the
 * Rust trait's defaulted methods (issue #5). They are optional here so
 * lightweight test stubs only need `validate`; tools fall back to
 * Node-based defaults when an implementation does not provide them.
 *
 * `isolationMode` and `workspaceRoot` are likewise optional for the same
 * reason — production sandboxes implement both.
 */
export interface SandboxProvider {
  /** Resolves with `null` on success, or a SandboxViolation. */
  validate(call: ToolCall, signal?: AbortSignal): Promise<SandboxViolation | null>;

  /** Execute a command (no shell). Default impl spawns via `child_process`. */
  executeCommand?(
    command: string,
    args: readonly string[],
    cwd?: string | null,
    timeoutMs?: number | null,
    signal?: AbortSignal,
  ): Promise<CommandOutput | SandboxViolation>;

  /** Truncate a large body head+tail. Default impl returns head+marker+tail. */
  handleLargeOutput?(
    content: string,
    callId: string,
    headTokens: number,
    tailTokens: number,
  ): Promise<TruncatedOutput>;

  /** Resolve a path against the workspace root. */
  resolvePath?(path: string, operation: Operation): Promise<string | SandboxViolation>;

  /** Current isolation mode. */
  isolationMode?(): IsolationMode;

  /** Canonical workspace root. */
  workspaceRoot?(): string;
}

/**
 * Inputs the harness compaction loop (issue #46) needs to run one compaction
 * turn and verify its result.
 *
 * The harness loop operates on the opaque {@link SessionState} above; the rich
 * compaction/verification API ({@link "../context/types.js".ContextManager},
 * {@link "../context/types.js".CompactionVerifier}) operates on
 * {@link "../context/types.js".SessionState}. This struct is the bridge: a
 * compaction-capable {@link ContextManager} projects everything the loop needs
 * into one value, so the loop never has to know which concrete state type its
 * manager uses internally.
 *
 * `context` is fed straight to `Agent.turn` to produce the summary;
 * `preserveHints` and `verificationState` are passed to
 * {@link "../context/types.js".CompactionVerifier.verify}. On a verification
 * failure the loop re-runs the turn with {@link ContextManager.injectMissingItems}
 * applied to `context`.
 */
export interface CompactionTurn {
  /** Context to feed `Agent.turn` to elicit the summary. */
  context: Context;
  /** Preservation hints to hand the verifier. */
  preserveHints: CompactionPreserveHints;
  /** Verifier-facing session state (rich `context` `SessionState`). */
  verificationState: ContextSessionState;
  /** Messages about to be removed — used to stamp the compaction span. */
  messagesRemoved: number;
}

/**
 * Issue #7 — ContextManager.
 *
 * Issue #46 adds the optional compaction-loop surface
 * ({@link prepareCompactionTurn}, {@link injectMissingItems},
 * {@link applyCompaction}). All three are OPTIONAL so managers that do not
 * compact (the default `shouldCompact` returns `false`) need not implement
 * them; the harness applies the spec defaults when a method is absent.
 */
export interface ContextManager {
  assemble(session: SessionState, task: Task, signal?: AbortSignal): Promise<Context>;
  appendToolResult(session: SessionState, result: ToolResultRecord): Promise<void>;
  appendUserMessage(session: SessionState, text: string): Promise<void>;
  shouldCompact(session: SessionState): boolean;

  /** Append the assistant's turn (model output: text and/or the tool calls it
   *  requested) to the conversation so the next {@link assemble} reflects what
   *  the agent already did. Without this the model loses track of its own
   *  actions and the conversation is malformed (a tool result with no preceding
   *  assistant tool_use). OPTIONAL so test-double / non-recording managers need
   *  not implement it; the harness calls it via `?.` and treats absence as a
   *  no-op (mirrors the Rust trait's default-no-op method). */
  appendAssistantMessage?(session: SessionState, message: Message): Promise<void>;

  /** Build the inputs for one compaction turn (issue #46). Returns `undefined`
   *  when there is nothing to compact (e.g. history shorter than the preserve
   *  window), in which case the harness skips compaction entirely. Default
   *  (when absent): `undefined` — managers that never compact need not
   *  implement this. */
  prepareCompactionTurn?(session: SessionState): CompactionTurn | undefined;

  /** Mutate a compaction {@link Context} in place to request a revised summary
   *  on retry (issue #46). The harness calls this with the items the prior
   *  summary failed to preserve. Default (when absent): append the standard
   *  "Your summary is missing these items: {missing}. Please revise." user
   *  message. */
  injectMissingItems?(context: Context, missing: string[]): void;

  /** Accept a verified (or accepted-anyway) summary into the session, replacing
   *  the compacted span (issue #46). Default (when absent): no-op — only
   *  compaction-capable managers implement it. */
  applyCompaction?(session: SessionState, summary: string): void;

  /** Report the manager's current token-budget usage for the session, so the
   *  harness can stamp the post-compaction `tokens_after`/`tokens_reclaimed`
   *  on the compaction span with real values (issue #57 token-accounting fix).
   *  Default (when absent): `undefined` — the harness falls back to the
   *  pre-compaction budget. */
  tokenBudgetUsed?(session: SessionState): number | undefined;
}

/** Issue #13 — TerminationPolicy. */
export interface TerminationPolicy {
  evaluate(session: SessionState, budgetUsed: BudgetSnapshot): Promise<TerminationDecision>;
}

/** Issue #11 — Middleware decision. */
export type MiddlewareDecision =
  | { kind: "continue" }
  | { kind: "continue_with_modification"; calls: ToolCall[] }
  | { kind: "halt"; reason: string }
  | { kind: "surface_to_human"; request: HumanRequest };

export interface MiddlewareChain {
  fire(hook: HookPoint, session: SessionState): Promise<MiddlewareDecision>;
}

/** Issue #12 — ObservabilityProvider. Re-exported from the canonical
 *  definition in {@link ../observability/types.js} so the harness loop and the
 *  observability backends share one interface (emitTurn / emitToolCall / … /
 *  setSessionOutcome / flushSession). */
export type { ObservabilityProvider } from "../observability/types.js";

// ============================================================================
// Human-in-the-loop
// ============================================================================

export const RiskLevelSchema = z.enum(["low", "medium", "high", "critical"]);
export type RiskLevel = z.infer<typeof RiskLevelSchema>;

export const HumanRequestSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("tool_approval"),
    calls: z.array(ToolCallSchema),
    risk_level: RiskLevelSchema,
  }),
  z.object({ kind: z.literal("clarification"), question: z.string() }),
  z.object({ kind: z.literal("review"), content: z.string() }),
]);
export type HumanRequest = z.infer<typeof HumanRequestSchema>;

export type HumanResponse =
  | { kind: "allow" }
  | { kind: "allow_with_modification"; calls: ToolCall[] }
  | { kind: "deny"; reason: string }
  | { kind: "halt" }
  | { kind: "answer"; text: string }
  | { kind: "approve_with_feedback"; feedback: string }
  | { kind: "reject"; reason: string };

// ============================================================================
// PausedState / ChildPausedState
// ============================================================================

/**
 * Child paused state. **Deliberately has no `child_state` field** — the
 * type system enforces one-level subagent depth (spec rule #9).
 */
export interface ChildPausedState {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  session_state: SessionState;
  pending_tool_calls: ToolCall[];
  approved_results: ToolResultRecord[];
  human_request: HumanRequest;
  task: Task;
  budget_used: BudgetSnapshot;
  parent_tool_call_id: string;
}

export interface PausedState {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  session_state: SessionState;
  pending_tool_calls: ToolCall[];
  approved_results: ToolResultRecord[];
  human_request: HumanRequest;
  task: Task;
  budget_used: BudgetSnapshot;
  child_state: ChildPausedState | null;
}

// ============================================================================
// Halt reasons / RunResult
// ============================================================================

export type HaltReason =
  | { kind: "budget_exceeded"; limit_type: BudgetLimitType }
  | { kind: "termination_policy_halt"; reason: string }
  | { kind: "middleware_halt"; hook: HookPoint; reason: string }
  | { kind: "agent_error"; error: AgentError }
  | { kind: "sandbox_violation"; violation: SandboxViolation }
  | { kind: "unrecoverable_tool_error"; tool: string; error: string }
  | { kind: "human_halted" }
  | { kind: "stagnation_limit_reached"; iterations: number; best_metric: number }
  | { kind: "strategy_not_yet_implemented"; strategy: string };

export type RunResult =
  | {
      kind: "success";
      output: string;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
    }
  | {
      kind: "failure";
      reason: HaltReason;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
    }
  | { kind: "waiting_for_human"; state: PausedState; request: HumanRequest };

// ============================================================================
// HarnessRunOptions
// ============================================================================

export interface HarnessRunOptions {
  task: Task;
  on_stream?: StreamSink;
  /** Optional starting session state (resumed history). */
  session_state?: SessionState;
  signal?: AbortSignal;
}

// ============================================================================
// Re-export of underlying schema types referenced in this file (for callers).
// ============================================================================
export type { Message, ToolCall, ToolSchema };
