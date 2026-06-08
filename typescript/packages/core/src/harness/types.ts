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

import type { Harness } from "./interface.js";
import type { ExecutionRegistry } from "./execution-registry.js";
import {
  completeTask,
  planArtifactToTaskList,
  updateTask,
  nextReady,
  transitiveBlockers,
  transitiveDependents,
  hasCycle,
  pushStepLedger,
  renderStepLedger,
  type StepLedgerEntry,
  type TaskList,
} from "../tasklist/index.js";
import type { SpanId } from "../observability/types.js";
import type { Context } from "../agent/types.js";
import type { Agent } from "../agent/interface.js";
import type { AgentError } from "../agent/errors.js";
import type { MetricEvaluator, ResultsEntry } from "../metric/types.js";
import type { VerifierInput } from "../verifier/types.js";
import type { PlanPhaseErrorKind } from "../plan/types.js";
import type { PlanArtifact } from "../plan/types.js";
import type { Mode } from "../prompt-chunk-registry/types.js";
import type {
  CompactionPreserveHints,
  ContextError,
  SessionState as ContextSessionState,
} from "../context/types.js";
import {
  MessageSchema,
  ToolCallSchema,
  type Message,
  type StreamEvent as ModelStreamEvent,
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

// ============================================================================
// BudgetPolicy + BudgetExhaustedBehavior (issue #117)
// ============================================================================
//
// Composable-execution budget vocabulary (PRD Part B). These are pure,
// serializable value types — no executor wiring. Later slices thread them
// through the strategy tree. They layer *on top of* {@link BudgetLimits} (the
// global turns/tokens/wall/cost backstop), which is unchanged.
//
// Wire format: internally tagged on `kind`, snake_case tag values. `value` and
// `max_continues` are u32 integers. `on_exhausted` is a recursively nested
// BudgetExhaustedBehavior. No node silently defaults to `continue`.

/** Non-negative 32-bit integer (`u32`) — a step is one model turn. */
const u32 = z.number().int().nonnegative().max(0xffffffff);

/**
 * Per-scope step allowance. A **step is one model turn** (matches
 * {@link BudgetSnapshot} turns). `per_goal` is intentionally excluded in v1.
 *
 *   - `{"kind":"unlimited"}` — no per-scope cap.
 *   - `{"kind":"total_steps","value":N}` — cap across the whole run.
 *   - `{"kind":"per_loop","value":N}` — cap per loop iteration.
 *   - `{"kind":"per_attempt","value":N}` — cap per attempt.
 */
export const BudgetPolicySchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("unlimited") }),
  z.object({ kind: z.literal("total_steps"), value: u32 }),
  z.object({ kind: z.literal("per_loop"), value: u32 }),
  z.object({ kind: z.literal("per_attempt"), value: u32 }),
]);
export type BudgetPolicy = z.infer<typeof BudgetPolicySchema>;

/**
 * What to do when a policy's allowance is spent. There is deliberately no
 * default: an unknown or missing `kind` is rejected rather than silently
 * treated as `continue`.
 *
 *   - `{"kind":"continue","max_continues":N,"on_exhausted":{...nested...}}` —
 *     grant up to `max_continues` extra rounds, then fall through to the nested
 *     `on_exhausted` behavior. `max_continues === 0` means immediate
 *     fall-through. `max_continues` is required (no default).
 *   - `{"kind":"escalate"}` — hand off to a parent/escalation path.
 *   - `{"kind":"fail"}` — terminate with failure.
 */
export type BudgetExhaustedBehavior =
  | { kind: "continue"; max_continues: number; on_exhausted: BudgetExhaustedBehavior }
  | { kind: "escalate" }
  | { kind: "fail" };

export const BudgetExhaustedBehaviorSchema: z.ZodType<BudgetExhaustedBehavior> = z.lazy(() =>
  z.discriminatedUnion("kind", [
    z.object({
      kind: z.literal("continue"),
      max_continues: u32,
      on_exhausted: BudgetExhaustedBehaviorSchema,
    }),
    z.object({ kind: z.literal("escalate") }),
    z.object({ kind: z.literal("fail") }),
  ]),
);

/**
 * The per-scope step allowance carried by a {@link BudgetPolicy} (`undefined`
 * for `unlimited`). Shared by {@link BudgetContext.charge}'s allowance and the
 * leaf cap-binding check in {@link runReactConfig} (#125). Mirrors Rust
 * `BudgetPolicy::allowance_value`.
 */
export function budgetPolicyAllowanceValue(policy: BudgetPolicy): number | undefined {
  return policy.kind === "unlimited" ? undefined : policy.value;
}

/** Parse a JSON value into a {@link BudgetPolicy}, rejecting unknown variants. */
export function budgetPolicyFromJson(value: unknown): BudgetPolicy {
  return BudgetPolicySchema.parse(value);
}

/**
 * Serialize a {@link BudgetPolicy} to a plain object with canonical field
 * order (`kind` first, then `value`) for byte-identical cross-language output.
 */
export function budgetPolicyToJson(policy: BudgetPolicy): Record<string, unknown> {
  switch (policy.kind) {
    case "unlimited":
      return { kind: "unlimited" };
    case "total_steps":
    case "per_loop":
    case "per_attempt":
      return { kind: policy.kind, value: policy.value };
  }
}

/** Parse a JSON value into a {@link BudgetExhaustedBehavior}, rejecting unknown variants. */
export function budgetExhaustedBehaviorFromJson(value: unknown): BudgetExhaustedBehavior {
  return BudgetExhaustedBehaviorSchema.parse(value);
}

/**
 * The default {@link BudgetExhaustedBehavior} for a config node's serialized
 * `behavior` field (#129): `escalate`. Two roles (mirrors Rust
 * `default_budget_behavior`):
 *   - the zod schema `.default(...)` so a strategy tree serialized BEFORE #129
 *     (no `behavior` key) still deserializes to the historical placeholder,
 *     preserving backward-compat reads;
 *   - the value {@link loopStrategyToJson} stamps when a config omits `behavior`,
 *     so a bare leaf keeps its pre-#129 propagate-to-parent contract by default.
 *
 * The field is NEVER omitted on serialize: it ALWAYS serializes (uniform wire
 * shape across all five config structs, Q1), so the cross-language fixtures
 * carry an explicit `"behavior":{"kind":"escalate"}` on every node.
 */
export function defaultBudgetBehavior(): BudgetExhaustedBehavior {
  return { kind: "escalate" };
}

/**
 * Serialize a {@link BudgetExhaustedBehavior} to a plain object with canonical
 * field order (`kind`, `max_continues`, `on_exhausted`), recursing into the
 * nested behavior, for byte-identical cross-language output.
 */
export function budgetExhaustedBehaviorToJson(
  behavior: BudgetExhaustedBehavior,
): Record<string, unknown> {
  switch (behavior.kind) {
    case "continue":
      return {
        kind: "continue",
        max_continues: behavior.max_continues,
        on_exhausted: budgetExhaustedBehaviorToJson(behavior.on_exhausted),
      };
    case "escalate":
      return { kind: "escalate" };
    case "fail":
      return { kind: "fail" };
  }
}

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

/**
 * HillClimbing optimization direction (Rust `HillClimbingDirection`). Wire
 * format is snake_case `"minimize"` / `"maximize"` — identical to
 * {@link OptimizationDirection}; this alias keeps the strategy-config naming in
 * lockstep with the Rust reference (#119).
 */
export const HillClimbingDirectionSchema = OptimizationDirectionSchema;
export type HillClimbingDirection = OptimizationDirection;

export const ModelConfigSchema = z.object({
  provider: z.string(),
  model_id: z.string(),
});
export type ModelConfig = z.infer<typeof ModelConfigSchema>;

// ============================================================================
// Composable Execution Part A (issue #119): recursive LoopStrategy config
// newtypes + per-node collaborator handles + StrategyRef + RunStrategy.
// ============================================================================
//
// `LoopStrategy` is a recursive, closed discriminated union of config shapes:
// `react` (the leaf) plus the combinators `plan_execute`, `self_verifying`,
// `ralph`, `hill_climbing` (each holding nested `LoopStrategy` children). The
// wire form is internally tagged on `kind` (snake_case), byte-identical across
// Rust / TS / Python / Go. `react` (NOT `re_act`) is the leaf tag, and its
// `budget` field is the renamed `max_iterations` (semantically PerLoop(n)).
//
// Per-node collaborator handles — `AgentRef`, `ToolsetRef`, `SchemaRef` — are
// transparent strings on the wire (idiomatic TS: bare `string`). Resolution to
// concrete collaborators lands with the registry slice (#120).
//
// `StrategyRef` is the serializable identity of a strategy: a closed built-in
// `LoopStrategy` tree, or an opaque `Custom` string key resolved at runtime
// (registry: #120). Adjacently tagged on `kind`/`value` to avoid a tag
// collision with the nested `LoopStrategy`'s own `kind`.
//
// `RunStrategy` is the runtime composition seam: every strategy node knows how
// to run itself given an {@link ExecutionContext}. The single dispatch is one
// `switch (strategy.kind)` (see {@link runStrategy}); per-variant bodies are
// STUBS returning {@link StrategyOutcome} `pending` (they do NOT throw). Real
// bodies land in #124. {@link ExecutionContext} / {@link StrategyOutcome} are
// minimal placeholders whose full shapes are owned by #123.
//
// JSON field order follows the declaration order below (cross-language
// byte-identity target). The {@link loopStrategyToJson} / {@link strategyRefToJson}
// serializers emit keys in that order so output is byte-identical to the
// `fixtures/strategy/` ground truth.

/**
 * Per-node handle to a named agent definition. Serializes as a bare JSON
 * string. Resolution lands with the registry slice (#120).
 */
export type AgentRef = string;
/**
 * Per-node handle to a named toolset. Serializes as a bare JSON string.
 * Resolution lands with the registry slice (#120).
 */
export type ToolsetRef = string;
/**
 * Per-node handle to a named output/evaluator schema. Serializes as a bare JSON
 * string. Resolution lands with the registry slice (#120).
 */
export type SchemaRef = string;

const AgentRefSchema = z.string();
const ToolsetRefSchema = z.string();
const SchemaRefSchema = z.string();

/**
 * Leaf ReAct node config. `budget` is the renamed `max_iterations`
 * (semantically `PerLoop(n)`). `output` is OMITTED from JSON when absent.
 */
export interface ReactConfig {
  kind: "react";
  budget: BudgetPolicy;
  /**
   * What this node does when its `budget` is spent (#129). CANONICAL POSITION:
   * IMMEDIATELY after `budget`. A leaf honors its `behavior` ONLY at the
   * top-level/bare-leaf resolution site ({@link StandardHarness} `driveStrategy`);
   * in the normal NESTED case the leaf still PROPAGATES exhaustion to its parent
   * (#125 rule 6 — a nested leaf never self-resolves). Optional in TS for
   * backward-compat / construction ergonomics, but ALWAYS serialized (Q1):
   * {@link loopStrategyToJson} emits `{"kind":"escalate"}` when absent.
   */
  behavior?: BudgetExhaustedBehavior;
  agent: AgentRef;
  toolset: ToolsetRef;
  output?: SchemaRef;
}

/**
 * PlanExecute combinator: a `plan` sub-strategy feeds an `execute`
 * sub-strategy. `plan_model` stays optional/omittable.
 */
export interface PlanExecuteConfig {
  kind: "plan_execute";
  plan: LoopStrategy;
  execute: LoopStrategy;
  plan_model?: ModelConfig;
  /**
   * What this combinator does when its execute-phase budget is spent (#129).
   * Canonical position on a combinator: the LAST field. Always serialized (Q1) —
   * {@link loopStrategyToJson} emits `{"kind":"escalate"}` when absent.
   */
  behavior?: BudgetExhaustedBehavior;
}

/** SelfVerifying combinator: run `inner`, then judge it against `evaluator`. */
export interface SelfVerifyingConfig {
  kind: "self_verifying";
  inner: LoopStrategy;
  evaluator: SchemaRef;
  /**
   * What this combinator does when its build↔evaluate budget is spent (#129).
   * Canonical position on a combinator: the LAST field. Always serialized (Q1) —
   * {@link loopStrategyToJson} emits `{"kind":"escalate"}` when absent.
   */
  behavior?: BudgetExhaustedBehavior;
}

/**
 * Ralph combinator: re-run `inner` under a fixed `agent` across context-window
 * resets.
 */
export interface RalphConfig {
  kind: "ralph";
  inner: LoopStrategy;
  agent: AgentRef;
  /**
   * What Ralph does when its OWN scope is spent (#129). Canonical position on a
   * combinator: the LAST field. Always serialized (Q1) — {@link loopStrategyToJson}
   * emits `{"kind":"escalate"}` when absent. NOTE: Ralph's window recovery
   * (reset + retry) is independent of this field — it governs Ralph's own budget
   * scope, not the per-window child exhaustion (which Ralph already absorbs as
   * "window incomplete" and retries).
   */
  behavior?: BudgetExhaustedBehavior;
}

/**
 * HillClimbing combinator: iterate `inner`, keeping/reverting per the metric
 * `evaluator` and `direction`. `max_stagnation` and `min_improvement_delta` are
 * required (#119).
 */
export interface HillClimbingConfig {
  kind: "hill_climbing";
  inner: LoopStrategy;
  direction: HillClimbingDirection;
  max_stagnation: number;
  revert_on_no_improvement: boolean;
  min_improvement_delta: number;
  evaluator: AgentRef;
  /**
   * What this combinator does when its optimization-loop budget is spent (#129).
   * Canonical position on a combinator: the LAST field. Always serialized (Q1) —
   * {@link loopStrategyToJson} emits `{"kind":"escalate"}` when absent.
   */
  behavior?: BudgetExhaustedBehavior;
}

/**
 * Loop strategy — a closed, recursive discriminated union of config shapes. The
 * `react` variant is the leaf; the rest are combinators holding nested
 * `LoopStrategy` children. Internally tagged on `kind` (snake_case), the
 * `react` tag overriding the would-be `re_act`. Wire form is byte-identical
 * across all four language targets (see `fixtures/strategy/`).
 */
export type LoopStrategy =
  | ReactConfig
  | PlanExecuteConfig
  | SelfVerifyingConfig
  | RalphConfig
  | HillClimbingConfig;

export const LoopStrategySchema: z.ZodType<LoopStrategy> = z.lazy(() =>
  z.discriminatedUnion("kind", [
    z.object({
      kind: z.literal("react"),
      budget: BudgetPolicySchema,
      // #129: default `escalate` so a pre-#129 tree (no `behavior` key) still
      // deserializes; always serialized via `loopStrategyToJson`.
      behavior: BudgetExhaustedBehaviorSchema.default(defaultBudgetBehavior()),
      agent: AgentRefSchema,
      toolset: ToolsetRefSchema,
      output: SchemaRefSchema.optional(),
    }),
    z.object({
      kind: z.literal("plan_execute"),
      plan: LoopStrategySchema,
      execute: LoopStrategySchema,
      plan_model: ModelConfigSchema.optional(),
      behavior: BudgetExhaustedBehaviorSchema.default(defaultBudgetBehavior()),
    }),
    z.object({
      kind: z.literal("self_verifying"),
      inner: LoopStrategySchema,
      evaluator: SchemaRefSchema,
      behavior: BudgetExhaustedBehaviorSchema.default(defaultBudgetBehavior()),
    }),
    z.object({
      kind: z.literal("ralph"),
      inner: LoopStrategySchema,
      agent: AgentRefSchema,
      behavior: BudgetExhaustedBehaviorSchema.default(defaultBudgetBehavior()),
    }),
    z.object({
      kind: z.literal("hill_climbing"),
      inner: LoopStrategySchema,
      direction: HillClimbingDirectionSchema,
      max_stagnation: u32,
      revert_on_no_improvement: z.boolean(),
      min_improvement_delta: z.number(),
      evaluator: AgentRefSchema,
      behavior: BudgetExhaustedBehaviorSchema.default(defaultBudgetBehavior()),
    }),
  ]),
);

/** Parse a JSON value into a {@link LoopStrategy}, rejecting unknown variants. */
export function loopStrategyFromJson(value: unknown): LoopStrategy {
  return LoopStrategySchema.parse(value);
}

/**
 * Serialize a {@link LoopStrategy} to a plain object with canonical field order
 * (declaration order, matching the Rust serializer / `fixtures/strategy/`), for
 * byte-identical cross-language output. `output` / `plan_model` are OMITTED when
 * absent (never `null`). Recurses into nested strategy children.
 */
export function loopStrategyToJson(strategy: LoopStrategy): Record<string, unknown> {
  switch (strategy.kind) {
    case "react": {
      // #129: `behavior` is ALWAYS emitted (Q1), immediately after `budget`,
      // defaulting to `escalate` when the config omits it.
      const out: Record<string, unknown> = {
        kind: "react",
        budget: budgetPolicyToJson(strategy.budget),
        behavior: budgetExhaustedBehaviorToJson(strategy.behavior ?? defaultBudgetBehavior()),
        agent: strategy.agent,
        toolset: strategy.toolset,
      };
      if (strategy.output !== undefined) out.output = strategy.output;
      return out;
    }
    case "plan_execute": {
      // #129: `behavior` is the LAST field on a combinator, always emitted (Q1).
      const out: Record<string, unknown> = {
        kind: "plan_execute",
        plan: loopStrategyToJson(strategy.plan),
        execute: loopStrategyToJson(strategy.execute),
      };
      if (strategy.plan_model !== undefined) {
        out.plan_model = {
          provider: strategy.plan_model.provider,
          model_id: strategy.plan_model.model_id,
        };
      }
      out.behavior = budgetExhaustedBehaviorToJson(strategy.behavior ?? defaultBudgetBehavior());
      return out;
    }
    case "self_verifying":
      return {
        kind: "self_verifying",
        inner: loopStrategyToJson(strategy.inner),
        evaluator: strategy.evaluator,
        behavior: budgetExhaustedBehaviorToJson(strategy.behavior ?? defaultBudgetBehavior()),
      };
    case "ralph":
      return {
        kind: "ralph",
        inner: loopStrategyToJson(strategy.inner),
        agent: strategy.agent,
        behavior: budgetExhaustedBehaviorToJson(strategy.behavior ?? defaultBudgetBehavior()),
      };
    case "hill_climbing":
      return {
        kind: "hill_climbing",
        inner: loopStrategyToJson(strategy.inner),
        direction: strategy.direction,
        max_stagnation: strategy.max_stagnation,
        revert_on_no_improvement: strategy.revert_on_no_improvement,
        min_improvement_delta: strategy.min_improvement_delta,
        evaluator: strategy.evaluator,
        behavior: budgetExhaustedBehaviorToJson(strategy.behavior ?? defaultBudgetBehavior()),
      };
  }
}

/**
 * The `max_iterations` value extracted from a `react` node's `per_loop` budget;
 * any other budget shape yields `Number.MAX_SAFE_INTEGER` (matching the legacy
 * executor's "unbounded" fall-through). Mirrors `ReactConfig::max_iterations`.
 */
export function reactMaxIterations(config: ReactConfig): number {
  return config.budget.kind === "per_loop" ? config.budget.value : Number.MAX_SAFE_INTEGER;
}

/**
 * A bare `react` leaf with a `per_loop` budget and empty agent/toolset handles
 * (resolution lands with the registry slice, #120). Migration shim for the old
 * `{ kind: "re_act", max_iterations }` shape. Mirrors `ReactConfig::per_loop`.
 */
export function reactPerLoop(value: number): ReactConfig {
  return { kind: "react", budget: { kind: "per_loop", value }, agent: "", toolset: "" };
}

/**
 * Serializable identity of a strategy: either a closed built-in
 * {@link LoopStrategy} tree or an opaque `custom` string key resolved at runtime
 * (registry: #120). Adjacently tagged on `kind`/`value` to avoid a tag
 * collision with the nested {@link LoopStrategy}'s own `kind`:
 *   - `{"kind":"built_in","value":{"kind":"react",...}}`
 *   - `{"kind":"custom","value":"my-harness::DoubleVerify"}`
 */
export type StrategyRef =
  | { kind: "built_in"; value: LoopStrategy }
  | { kind: "custom"; value: string };

export const StrategyRefSchema: z.ZodType<StrategyRef> = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("built_in"), value: LoopStrategySchema }),
  z.object({ kind: z.literal("custom"), value: z.string() }),
]);

/** Parse a JSON value into a {@link StrategyRef}, rejecting unknown variants. */
export function strategyRefFromJson(value: unknown): StrategyRef {
  return StrategyRefSchema.parse(value);
}

/**
 * Serialize a {@link StrategyRef} to a plain object with canonical field order
 * (`kind`, `value`) for byte-identical cross-language output. Recurses into the
 * nested {@link LoopStrategy} for the `built_in` arm.
 */
export function strategyRefToJson(ref: StrategyRef): Record<string, unknown> {
  switch (ref.kind) {
    case "built_in":
      return { kind: "built_in", value: loopStrategyToJson(ref.value) };
    case "custom":
      return { kind: "custom", value: ref.value };
  }
}

// ── Composable Execution runtime scaffold (#123) ────────────────────────────
//
// StrategyOutcome + ExecutionContext / BudgetContext / BudgetStack / SpanStack.
// These are the typed result a strategy node returns and the one shared,
// mutable runtime context threaded through a whole nested strategy tree. This
// is the SCAFFOLD slice: it establishes the types and the threading. The
// enforcement semantics (walking the behavior chain, consuming continues,
// promoting a charge error to a {@link StrategyOutcome}) land in a later slice.
//
// Resolved spec decisions (mirrors the Rust reference):
//   1. {@link ExecutionContext} holds the {@link ExecutionRegistry} object — no
//      lifetime concerns in TS. {@link RunStrategy.run} threads the context.
//   2. {@link BudgetContext.charge} is PURE ARITHMETIC: it debits `turns`
//      against the per-scope allowance; on success increments `stepsTaken`; on
//      overflow returns the error WITHOUT mutating. It does NOT walk the
//      behavior chain or consume continues. `unlimited` never exhausts.
//   3. `continuesUsed` is an in-memory field ONLY this slice; checkpoint
//      persistence is deferred.
//   4. {@link SpanStack} is a runtime-only push/pop stack of {@link SpanId}.
//
// All of these types are RUNTIME-ONLY and are NEVER serialized (there is
// deliberately no zod schema and no `*ToJson` for them — that absence is the
// "not serialized" guarantee).

/**
 * The error {@link BudgetContext.charge} returns when a debit would exceed the
 * scope's step allowance. Captures the budget state at the moment of
 * exhaustion. Runtime-only (NOT serialized).
 *
 * Promotes to a {@link StrategyOutcome} `budget_exhausted` (which adds
 * `partialOutput`) at the strategy boundary in the enforcement slice.
 */
export interface BudgetExhausted {
  policy: BudgetPolicy;
  behavior: BudgetExhaustedBehavior;
  stepsTaken: number;
  continuesUsed: number;
  phase: string;
}

/**
 * The result of charging a {@link BudgetContext}. A Result-style discriminated
 * union: `{ ok: true }` on success, `{ ok: false, error }` carrying the
 * {@link BudgetExhausted} state on overflow — non-throwing, matching how the
 * codebase models fallible, recoverable operations.
 */
export type ChargeResult = { ok: true } | { ok: false; error: BudgetExhausted };

/**
 * The typed outcome a strategy node returns. Internally tagged on `kind`
 * (snake_case for symmetry with the other harness unions). Runtime-only — NEVER
 * serialized.
 *
 *   - `complete` — the strategy finished and produced its final `output`
 *     (`Output` maps to `string`, mirroring {@link RunResult}).
 *   - `budget_exhausted` — the strategy's budget scope ran out of allowance.
 *     Mirrors the {@link BudgetExhausted} charge-error fields and adds
 *     `partialOutput` (any output produced before exhaustion). A child's
 *     `budget_exhausted` is an INSPECTABLE value the parent reads (e.g. to grant
 *     a continue or escalate); it does NOT auto-propagate as a failure.
 *   - `failed` — the strategy halted with a {@link HarnessError}. Callers can
 *     distinguish this from `budget_exhausted`.
 */
export type StrategyOutcome =
  | { kind: "complete"; output: string }
  | {
      kind: "budget_exhausted";
      policy: BudgetPolicy;
      behavior: BudgetExhaustedBehavior;
      stepsTaken: number;
      continuesUsed: number;
      phase: string;
      partialOutput?: string;
    }
  | { kind: "failed"; error: HarnessError };

/**
 * One budget scope in the strategy tree. Each recursion node gets its OWN
 * `BudgetContext`; siblings do NOT share. Runtime-only (NOT serialized).
 *
 * The per-scope step allowance is the policy's own `value`: `total_steps` /
 * `per_loop` / `per_attempt` all expose `value` as the cap for this scope;
 * `unlimited` is uncapped ({@link remaining} → `undefined`).
 *
 * `continuesUsed` is an in-memory field ONLY in this slice; its checkpoint
 * persistence is deferred to the enforcement slice (resolution #3).
 */
export class BudgetContext {
  stepsTaken = 0;
  continuesUsed = 0;
  /**
   * Mutable on the {@link resolveExhausted} fall-through path: when a `continue`
   * scope's continues are spent it ADOPTS the boxed `on_exhausted` behavior so
   * subsequent resolutions see the post-fall-through behavior (#125).
   */
  behavior: BudgetExhaustedBehavior;

  constructor(
    readonly policy: BudgetPolicy,
    behavior: BudgetExhaustedBehavior,
    readonly phase: string,
  ) {
    this.behavior = behavior;
  }

  /**
   * Reconstruct a RESUMED scope (#129) whose `continuesUsed` is seeded from a
   * cross-process checkpoint — the sole field of {@link BudgetContext} that must
   * survive a process pause. `stepsTaken` starts at 0 (the resumed run re-enters
   * the loop with a fresh per-round step budget; the checkpoint only carries how
   * many continues were ALREADY spent so a `continue` spanning the pause cannot
   * exceed `max_continues`). Runtime-only — `continuesUsed` is read off the
   * `HumanRequest::BudgetExhausted` payload (Q3: NOT a new serialized
   * {@link BudgetContext}/{@link PausedState} field). Mirrors Rust
   * `BudgetContext::resumed`.
   */
  static resumed(
    policy: BudgetPolicy,
    behavior: BudgetExhaustedBehavior,
    phase: string,
    continuesUsed: number,
  ): BudgetContext {
    const cx = new BudgetContext(policy, behavior, phase);
    cx.continuesUsed = continuesUsed;
    return cx;
  }

  /** The per-scope step allowance (`undefined` for `unlimited`). */
  private allowance(): number | undefined {
    return budgetPolicyAllowanceValue(this.policy);
  }

  /**
   * Debit `turns` steps against the scope allowance (pure arithmetic —
   * resolution #2). On success increments `stepsTaken` and returns
   * `{ ok: true }`. If the debit would exceed the allowance, returns
   * `{ ok: false, error }` capturing current state WITHOUT mutating. Does NOT
   * walk the behavior chain or consume continues. `unlimited` never exhausts.
   */
  charge(turns: number): ChargeResult {
    const allowance = this.allowance();
    if (allowance !== undefined && this.stepsTaken + turns > allowance) {
      return {
        ok: false,
        error: {
          policy: this.policy,
          behavior: this.behavior,
          stepsTaken: this.stepsTaken,
          continuesUsed: this.continuesUsed,
          phase: this.phase,
        },
      };
    }
    this.stepsTaken += turns;
    return { ok: true };
  }

  /**
   * Steps left in this scope (`allowance - stepsTaken`, floored at 0).
   * `undefined` for `unlimited` (no cap).
   */
  remaining(): number | undefined {
    const allowance = this.allowance();
    return allowance === undefined ? undefined : Math.max(0, allowance - this.stepsTaken);
  }

  /**
   * Continues left before fall-through. For a `continue` behavior this is
   * `maxContinues - continuesUsed` (floored at 0); for `escalate` / `fail`
   * there are no continues, so `0`.
   */
  continuesRemaining(): number {
    return this.behavior.kind === "continue"
      ? Math.max(0, this.behavior.max_continues - this.continuesUsed)
      : 0;
  }

  /**
   * Grant one in-process continue (#125): bump `continuesUsed` and RESET
   * `stepsTaken` to 0 so the scope's step allowance refreshes for the next
   * round. A purely in-memory reset — the session / messages are untouched (the
   * loop keeps the same conversation; only the per-scope step counter rewinds).
   * `continuesUsed` persistence across a serialized checkpoint is DEFERRED to
   * #129. Mirrors Rust `BudgetContext::consume_continue`.
   */
  consumeContinue(): void {
    this.continuesUsed += 1;
    this.stepsTaken = 0;
  }

  /**
   * Resolve this scope's {@link BudgetExhaustedBehavior} at the moment of
   * exhaustion (#125), walking the on-exhausted fall-through chain:
   *
   *   - `fail`     → {@link ExhaustedResolution} `fail`.
   *   - `escalate` → {@link ExhaustedResolution} `escalate`.
   *   - `continue { max_continues, on_exhausted }`:
   *       - if {@link continuesRemaining} `> 0`: {@link consumeContinue} (reset
   *         counter, bump `continuesUsed`) and return `continue`;
   *       - otherwise the continues are spent: ADOPT the nested `on_exhausted`
   *         behavior as this scope's behavior and recurse into it (the
   *         fall-through), so a `continue { on_exhausted: escalate }` whose
   *         continues are spent resolves to `escalate`.
   *
   * Mutates `this`: on a granted continue the counter resets; on fall-through
   * `this.behavior` is replaced by the nested behavior so subsequent resolutions
   * see the post-fall-through behavior. Mirrors Rust
   * `BudgetContext::resolve_exhausted`.
   */
  resolveExhausted(): ExhaustedResolution {
    switch (this.behavior.kind) {
      case "fail":
        return "fail";
      case "escalate":
        return "escalate";
      case "continue":
        if (this.continuesRemaining() > 0) {
          this.consumeContinue();
          return "continue";
        }
        // Continues spent — fall through to the nested behavior.
        this.behavior = this.behavior.on_exhausted;
        return this.resolveExhausted();
    }
  }
}

/**
 * The runtime-only resolution of a {@link BudgetExhaustedBehavior} chain at the
 * moment of exhaustion (#125). NOT serialized — purely a control-flow signal
 * returned by {@link BudgetContext.resolveExhausted}.
 *
 *   - `continue` — the scope was granted an in-process continue (counter reset,
 *     `continuesUsed` bumped); the caller loops again.
 *   - `fail`     — terminate; `partialOutput = undefined` (discarded by contract).
 *   - `escalate` — hand off to the parent; `partialOutput = <node JSON>` carries
 *     the node-concrete partial.
 */
export type ExhaustedResolution = "continue" | "fail" | "escalate";

/**
 * Runtime push/pop stack of {@link BudgetContext} scopes — one node per
 * recursion frame, pushed on descent and popped on ascent. Runtime-only (NOT
 * serialized). Siblings get DISTINCT contexts and do not share state.
 */
export class BudgetStack {
  readonly stack: BudgetContext[] = [];

  push(cx: BudgetContext): void {
    this.stack.push(cx);
  }

  /** Pop the current scope, returning it (or `undefined` when empty). */
  pop(): BudgetContext | undefined {
    return this.stack.pop();
  }

  /** The current (innermost) scope, or `undefined` when empty. */
  current(): BudgetContext | undefined {
    return this.stack[this.stack.length - 1];
  }

  /** The current stack depth (recursion frames active). */
  depth(): number {
    return this.stack.length;
  }
}

/**
 * Runtime push/pop stack of {@link SpanId} (resolution #4). Runtime-only (NOT
 * serialized).
 */
export class SpanStack {
  readonly stack: SpanId[] = [];

  push(id: SpanId): void {
    this.stack.push(id);
  }

  /** Pop the current span id, returning it (or `undefined` when empty). */
  pop(): SpanId | undefined {
    return this.stack.pop();
  }

  /** The current stack depth. */
  depth(): number {
    return this.stack.length;
  }
}

/**
 * The PlanExecute plan-phase result the {@link StrategyExecutor} returns
 * (#124, Q4). The plan phase is a real loop that drives the constrained planner
 * and captures a {@link PlanArtifact} + accounting, or a terminal failure
 * {@link RunResult} the combinator must propagate.
 */
export type PlanPhaseOutcome =
  | { ok: true; artifact: PlanArtifact; usage: AggregateUsage; turns: number }
  | { ok: false; failure: RunResult };

/**
 * The harness-side primitives the per-variant {@link RunStrategy.run} bodies
 * delegate to (#124). Implemented by `StandardHarness`. This is the seam that
 * lets the recursive config bodies OWN their loops while the LEAF model-touching
 * machinery (the ReAct turn-loop window, the SelfVerifying evaluate phase, the
 * HillClimbing metric machinery, the Ralph `.spore/` checks, and the PlanExecute
 * artifact-capture / deep-resume / hook helpers) stays where it is tested.
 *
 * For PlanExecute the recursion is GENUINE: {@link runPlanSubtree} dispatches
 * the plan child's `.run(cx)` and the per-task ORCHESTRATION loop lives in
 * {@link runPlanExecuteConfig}, where it dispatches the execute child's
 * `.run(cx)` once per task — so a non-ReAct execute child really executes its
 * loop per task. The executor keeps only the harness-side helpers that DON'T
 * touch the per-task model loop (directive, plan dispatch, artifact
 * capture/persist, deep-resume reconcile, `on_task_advance` fire).
 *
 * Most primitives return a terminal {@link RunResult} for their phase; the
 * config bodies translate the terminal into a {@link StrategyOutcome} (or
 * recurse). Runtime-only — NEVER serialized. Mirrors the Rust `StrategyExecutor`
 * trait.
 */
export interface StrategyExecutor {
  /**
   * Run ONE bounded ReAct turn-loop window over `sessionState`, carrying the
   * shared `budgetUsed`. The leaf primitive (the body of the legacy
   * `runReactInner`). Does NOT finalize observability — the caller does.
   */
  reactWindow(
    task: Task,
    maxIterations: number,
    sessionState: SessionState,
    budgetUsed: BudgetSnapshot,
    onStream: StreamSink | undefined,
    signal: AbortSignal | undefined,
    agent: Agent,
  ): Promise<RunResult>;

  /**
   * Resolve an {@link AgentRef} to its registered agent (#124). The leaf and the
   * combinators resolve their worker agent through this so a missing handle is a
   * typed terminal {@link RunResult} `failure` rather than a throw. Returns the
   * failure `RunResult` (NOT the agent) when the key is absent.
   */
  resolveAgentRef(ref: AgentRef, sessionId: SessionId): Agent | RunResult;

  /**
   * Run the SelfVerifying evaluate phase (#124): a fresh evaluator RUN over a
   * read-only sandbox in a never-shared session, on `evalAgent`. Folds the run's
   * usage into `totalUsage` / `carried`; returns its terminal {@link RunResult}.
   */
  evaluatePhase(
    task: Task,
    evalAgent: Agent,
    carried: BudgetSnapshot,
    totalUsage: AggregateUsage,
  ): Promise<RunResult>;

  /** Append `text` as a user message on `sessionState` (Default-FAIL re-seed). */
  appendUserMessage(sessionState: SessionState, text: string): Promise<void>;

  /** The configured sandbox workspace root (#124, for `VerifierInput`). */
  workspaceRoot(): string;

  /**
   * The planning directive seeded before the plan sub-strategy runs (#124, R1):
   * the "respond with a single JSON plan" instruction wrapped around the task.
   */
  planDirective(instruction: string): string;

  /**
   * Append `text` as a user message to `sessionState` through the configured
   * {@link ContextManager} (#124). Leaf seam used to seed the plan directive and
   * each execute step instruction onto the shared / plan session.
   */
  seedUserMessage(sessionState: SessionState, text: string): Promise<void>;

  /**
   * Dispatch the plan sub-strategy GENUINELY (#124): drive `plan.run(cx)` over a
   * child {@link ExecutionContext} seeded with `planSession` / `budgetUsed` /
   * `planTask`, returning its terminal {@link RunResult}. Routes the configured
   * planner agent (R5/R6) by running the child against an agent-swapped child
   * harness when one is set; otherwise the default agent runs the plan turn.
   * Returns `undefined` if the child produced no terminal.
   */
  runPlanSubtree(
    plan: LoopStrategy,
    planTask: Task,
    planSession: SessionState,
    budgetUsed: BudgetSnapshot,
    signal: AbortSignal | undefined,
  ): Promise<RunResult | undefined>;

  /**
   * Capture + persist a {@link PlanArtifact} from the plan child's final output
   * text (#124, R3/R4/R11): parse the response, fire `on_plan_created`, and
   * persist to the RunStore under `PLAN_EXECUTE_EXTRAS_KEY`. The model turn that
   * produced `planOutput` ran elsewhere (the recursive `plan.run(cx)` child), so
   * this carries no agent call.
   */
  capturePlanArtifact(
    sessionId: SessionId,
    planOutput: string,
    usage: AggregateUsage,
    turns: number,
    signal: AbortSignal | undefined,
  ): Promise<PlanPhaseOutcome>;

  /**
   * Reconcile a freshly-parsed task list against the DURABLE RunStore checkpoint
   * (#124, A.6 deep-resume): any task already `completed` on the checkpoint is
   * marked `completed` in `taskList` so it is NOT re-run.
   */
  reconcileCompletedTasks(sessionId: SessionId, taskList: TaskList): Promise<void>;

  /**
   * Fire the `on_task_advance` hook (#124, pre, mutable) for an execute step. The
   * hook may rewrite `stepTask.instruction`; the (possibly mutated) instruction
   * is what the execute sub-strategy then runs.
   */
  fireTaskAdvance(
    sessionId: SessionId,
    stepTask: Task,
    taskIndex: number,
    totalTasks: number,
    signal: AbortSignal | undefined,
  ): Promise<void>;

  /** Persist a parsed task list through the RunStore seam (legacy `persistTaskList`). */
  persistTaskList(sessionId: SessionId, taskList: TaskList): Promise<void>;

  /**
   * Load the persisted {@link TaskList} from the RunStore under
   * `TASK_LIST_EXTRAS_KEY` (#126, decision C): the ONE authoring path that can
   * carry real `blockers`. A storage miss / deserialize failure yields
   * `undefined` (the DAG executor then falls back to the linear plan-artifact
   * bridge).
   */
  loadTaskList(sessionId: SessionId): Promise<TaskList | undefined>;

  /**
   * Drain and return the harness-OBSERVED write/edit tool-call paths recorded
   * since the last {@link clearObservedWrites} (#126, AC2). The dispatch seam
   * records the `path` of every `write_file` / `edit_file` tool call as it
   * happens; the DAG executor calls this on task completion to attach the
   * OBSERVED `files_touched` to the task's {@link StepLedgerEntry}. Paths are
   * never model-self-reported. Draining resets the accumulator.
   */
  takeObservedWrites(): string[];

  /**
   * Clear the observed-write accumulator (#126, AC2). The DAG executor calls
   * this before each execute step so a step's `files_touched` reflects ONLY the
   * writes that step issues.
   */
  clearObservedWrites(): void;

  /**
   * Build the per-window Ralph seed session (#124): a fresh session seeded with
   * `instruction`, the `.spore/` reload context (R3), and the optional VCS
   * history block — exactly the legacy `runRalph` window setup, minus the model
   * loop (which now recurses).
   */
  ralphSeedSession(instruction: string): Promise<SessionState>;

  /** Ralph external completion check (#124): `null` ⇒ complete; a string ⇒
   *  tasks remain. */
  ralphCompletionStatus(): string | null;

  /** The Ralph outer-loop reset cap (`config.maxResets`, #124). */
  ralphMaxResets(): number;

  /**
   * Resolve the HillClimbing metric evaluator for `key` (#124, Q2), or the
   * misconfiguration failure {@link RunResult} when absent.
   */
  resolveMetricEvaluator(key: string, sessionId: SessionId): MetricEvaluator | RunResult;

  /**
   * HillClimbing iteration-0 baseline (#124): evaluate the metric (no agent
   * turn), record the row + span, and return the baseline value — or the
   * baseline-evaluation failure {@link RunResult} (already records the failed row
   * + writes the TSV). Leaf primitive of the legacy `runHillClimbing`.
   */
  hillBaseline(
    evaluator: MetricEvaluator,
    sessionId: SessionId,
    taskId: TaskId,
    direction: HillClimbingDirection,
    rows: ResultsEntry[],
    spanSeq: { value: number },
    totalUsage: AggregateUsage,
    turns: number,
    signal: AbortSignal | undefined,
  ): Promise<{ ok: true; value: number } | { ok: false; failure: RunResult }>;

  /**
   * HillClimbing per-iteration metric eval + keep/revert decision (#124): the
   * agent turn already ran (recursively); this evaluates the metric, applies
   * `shouldKeep`, optionally reverts, records the row + span, and returns the
   * updated `(currentBest, nonImprovement)`. Leaf primitive of the legacy
   * `runHillClimbing`.
   */
  hillIteration(
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
  ): Promise<{ currentBest: number; nonImprovement: boolean }>;

  /** Write the HillClimbing results TSV (#124, leaf primitive). */
  hillWriteTsv(taskId: TaskId, rows: ResultsEntry[]): Promise<void>;

  /**
   * Wall-time / cost / token budget gate (#124): the {@link RunResult}-agnostic
   * check the HillClimbing loop runs before each iteration's agent turn. Returns
   * the breached {@link BudgetLimitType} or `null`. Mirrors the static
   * `StandardHarness.budgetExceeded` used by the legacy loop.
   */
  budgetExceeded(budget: BudgetLimits, used: BudgetSnapshot): BudgetLimitType | null;

  /** Finalize observability for a terminal outcome. No-op for non-terminal. */
  finalize(result: RunResult): Promise<void>;

  /**
   * The configured HITL-vs-AFK {@link EscalationMode} (#130). Consulted at each
   * `escalate` budget-resolution site: `surface_to_human` PAUSES with a
   * {@link HumanRequest} `budget_exhausted`; `autonomous` keeps the existing
   * propagate behavior. Mirrors Rust `StrategyExecutor::escalation_mode`.
   */
  escalationMode(): EscalationMode;
}

/**
 * Per-run mutable orchestration state threaded through the recursive strategy
 * tree (#124). Runtime-only (NOT serialized). The combinator bodies set up the
 * sub-phase `task` here before recursing, and the leaf ({@link ReactConfig})
 * reads it to drive the ReAct window. `onStream` lives here (it is taken at the
 * leaf so combinators that recurse per-phase suppress it).
 *
 *   - `task` — the task whose strategy is currently executing.
 *   - `runSession` — the conversation/session state the current leaf builds on.
 *   - `runBudget` — the shared budget snapshot threaded across every sub-loop.
 *   - `terminalOverride` — a non-terminal pause (`waiting_for_human` / `consult`
 *     / `escalate`) or a fully-formed terminal that must propagate VERBATIM as a
 *     {@link RunResult} (preserving the strategy's typed {@link HaltReason} and
 *     accounting) rather than being collapsed into a {@link StrategyOutcome}.
 */
export interface RunScratch {
  task?: Task;
  runSession: SessionState;
  runBudget: BudgetSnapshot;
  terminalOverride?: RunResult;
  /**
   * Cross-process Continue checkpoint seed (#129): `[phase, continuesUsed]`
   * carried from a resumed `HumanRequest::BudgetExhausted`. The FIRST
   * {@link pushBudget} whose `phase` matches seeds the reconstructed scope's
   * `continuesUsed` (via {@link BudgetContext.resumed}) and CLEARS this seed — so
   * a `continue` spanning a process pause resumes with the correct continue count
   * (AC2). Runtime-only; the value rides the request payload, NOT a serialized
   * {@link BudgetContext}/{@link PausedState} field (Q3). `undefined` on a fresh
   * run and after the seed is consumed (an in-process continue never sets it →
   * AC3: no serialization on the in-process path).
   */
  resumeContinues?: [string, number];
}

function newRunScratch(): RunScratch {
  return { runSession: emptySessionState(), runBudget: emptyBudgetSnapshot() };
}

/**
 * The one shared, mutable runtime context threaded through a whole nested
 * strategy tree. Holds the {@link ExecutionRegistry} for the duration of the
 * run (resolution #1 — no lifetime concerns in TS). Runtime-only — NEVER
 * serialized.
 */
export interface ExecutionContext {
  /** Handle registry (agents/toolsets/schemas/custom strategies). */
  registry: ExecutionRegistry;
  /** Per-scope budget stack, pushed/popped through recursion. */
  budgets: BudgetStack;
  /** Aggregated token/cost usage across the whole tree. */
  usage: AggregateUsage;
  /** Conversation/session state round-tripped across the tree. */
  session: SessionState;
  /** Span stack for observability nesting. */
  spans: SpanStack;
  /** Optional streaming sink for emitted events. */
  stream?: StreamSink;
  /**
   * Optional cancellation signal threaded into every model/tool call across the
   * recursive tree (TS-only — Rust uses a different cancellation model).
   * Runtime-only; NEVER serialized.
   */
  signal?: AbortSignal;
  /**
   * The harness primitives the per-variant run bodies delegate to (#124).
   * Absent only for the scaffold/unit fixtures that exercise the runtime
   * context without a real harness (the recursion stub tests); without one a
   * config body returns a TYPED {@link StrategyOutcome} `failed` (never throws).
   */
  executor?: StrategyExecutor;
  /** Per-run orchestration scratch threaded across recursion (#124). */
  scratch: RunScratch;
}

/**
 * A fresh {@link ExecutionContext} bound to `registry`, with empty stacks and
 * default usage/session and no stream sink.
 */
export function newExecutionContext(registry: ExecutionRegistry): ExecutionContext {
  return {
    registry,
    budgets: new BudgetStack(),
    usage: emptyAggregateUsage(),
    session: emptySessionState(),
    spans: new SpanStack(),
    scratch: newRunScratch(),
  };
}

// ── ExecutionContext budget-scope helpers (#125) ────────────────────────────
//
// `ExecutionContext` is an interface in TS, so the Rust `impl ExecutionContext`
// budget methods become standalone functions over `cx.budgets`. Each node — even
// a sibling — pushes its OWN scope (`stepsTaken = 0`), so a node capped at N
// never spends a sibling's allowance (rule 1) and a child's exhaustion never
// touches the parent scope (rule 4/7). `chargeCurrent` is the real enforcement
// point; `resolveCurrent` walks the behavior chain at exhaustion.

/**
 * Push a fresh per-node {@link BudgetContext} scope for `policy`/`behavior`/
 * `phase` onto `cx.budgets` (#125). Returns the depth AFTER the push (symmetry
 * debugging). Mirrors Rust `ExecutionContext::push_budget`.
 */
export function pushBudget(
  cx: ExecutionContext,
  policy: BudgetPolicy,
  behavior: BudgetExhaustedBehavior,
  phase: string,
): number {
  // #129 (AC2): if a resumed `continue` checkpoint seed is waiting for THIS
  // phase, reconstruct the scope with its prior `continuesUsed` (consuming the
  // seed once) instead of zeroing it. The root resumed node pushes first, and
  // the request's `phase` names that node, so the FIRST matching push restores
  // the count. Any other push (or a fresh run) is unaffected.
  const seed = cx.scratch.resumeContinues;
  if (seed !== undefined && seed[0] === phase) {
    cx.scratch.resumeContinues = undefined;
    cx.budgets.push(BudgetContext.resumed(policy, behavior, phase, seed[1]));
  } else {
    cx.budgets.push(new BudgetContext(policy, behavior, phase));
  }
  return cx.budgets.depth();
}

/**
 * Pop the current per-node budget scope (#125). Always paired with
 * {@link pushBudget}. Mirrors Rust `ExecutionContext::pop_budget`.
 */
export function popBudget(cx: ExecutionContext): BudgetContext | undefined {
  return cx.budgets.pop();
}

/**
 * Charge `turns` steps against the CURRENT (innermost) budget scope (#125): the
 * real enforcement point. `{ ok: true }` when within allowance; `{ ok: false,
 * error }` carries the budget state at exhaustion. A node with no pushed scope
 * (scaffold contexts) never exhausts — charging is a no-op `{ ok: true }`.
 * Mirrors Rust `ExecutionContext::charge_current`.
 */
export function chargeCurrent(cx: ExecutionContext, turns: number): ChargeResult {
  const scope = cx.budgets.current();
  return scope === undefined ? { ok: true } : scope.charge(turns);
}

/**
 * Resolve the current scope's exhaustion behavior (#125). Walks the chain
 * (continue grants a reset; spent continues fall through). A node with no pushed
 * scope resolves to `fail` (defensive — should not happen in a wired run).
 * Mirrors Rust `ExecutionContext::resolve_current`.
 */
export function resolveCurrent(cx: ExecutionContext): ExhaustedResolution {
  const scope = cx.budgets.current();
  return scope === undefined ? "fail" : scope.resolveExhausted();
}

/**
 * The runtime composition seam: every strategy node knows how to run itself
 * given an {@link ExecutionContext}. The TS runtime-polymorphism idiom is one
 * `run(cx)` method whose single dispatch is {@link runStrategy}.
 */
export interface RunStrategy {
  run(cx: ExecutionContext): Promise<StrategyOutcome>;
}

/**
 * The ONLY dispatch site for the strategy tree (AC1): one `switch` over the
 * `kind` discriminant — the enum→config delegation. There is no central
 * dispatch `match` anymore; each per-config body OWNS its loop and recurses via
 * `runStrategy(child, cx)`.
 */
export async function runStrategy(
  strategy: LoopStrategy,
  cx: ExecutionContext,
): Promise<StrategyOutcome> {
  switch (strategy.kind) {
    case "react":
      return runReactConfig(strategy, cx);
    case "plan_execute":
      return runPlanExecuteConfig(strategy, cx);
    case "self_verifying":
      return runSelfVerifyingConfig(strategy, cx);
    case "ralph":
      return runRalphConfig(strategy, cx);
    case "hill_climbing":
      return runHillClimbingConfig(strategy, cx);
  }
}

/** Wrap a {@link LoopStrategy} as a {@link RunStrategy} (delegates to {@link runStrategy}). */
export function asRunStrategy(strategy: LoopStrategy): RunStrategy {
  return { run: (cx) => runStrategy(strategy, cx) };
}

// ── Per-config run bodies (#124) ─────────────────────────────────────────────
//
// Each body OWNS its loop and recurses into children via `runStrategy`. The
// model-touching machinery stays on the harness behind {@link StrategyExecutor},
// reachable through `cx.executor`. Without a wired executor (scaffold-only
// contexts) every body returns a TYPED {@link StrategyOutcome} `failed` — never
// a throw (CONVENTIONS).

/** The current per-run task, or `undefined` when scratch is unset (misuse). */
function currentTask(cx: ExecutionContext): Task | undefined {
  return cx.scratch.task;
}

/** The executor primitives, or a typed failure outcome when absent. */
function requireExecutor(cx: ExecutionContext): StrategyExecutor | StrategyOutcome {
  if (cx.executor === undefined) {
    return {
      kind: "failed",
      error: new InvalidConfiguration("ExecutionContext has no StrategyExecutor wired"),
    };
  }
  return cx.executor;
}

/** Derive the `SessionOutcome.failure.reason` string from a {@link HaltReason}.
 *  Mirrors Rust's `format!("{reason:?}")` for the failure outcome. */
export function haltReasonToString(reason: HaltReason): string {
  return JSON.stringify(reason);
}

/**
 * Translate a terminal {@link RunResult} into a {@link StrategyOutcome} (#124,
 * Q5): `success → complete(output)`; every non-success terminal → `failed`. A
 * `budget_exceeded` failure maps to `failed` here (the budget-enforcement
 * `budget_exhausted` value is produced by {@link BudgetContext.charge} at the
 * boundary; full HITL-through-recursion is #130). The pause variants are handled
 * separately (the override path) and degrade to a typed failure only if they
 * ever reach this mapping.
 */
function outcomeFromRunResult(result: RunResult): StrategyOutcome {
  switch (result.kind) {
    case "success":
      return { kind: "complete", output: result.output };
    case "failure":
      return {
        kind: "failed",
        error: new InvalidConfiguration(haltReasonToString(result.reason)),
      };
    case "waiting_for_human":
    case "consult":
    case "escalate":
      return {
        kind: "failed",
        error: new InvalidConfiguration("non-terminal outcome reached strategy boundary"),
      };
  }
}

/**
 * Record a terminal/pause {@link RunResult} from a whole-loop primitive
 * (ReAct / SelfVerifying / Ralph / HillClimbing): carry the post-run session
 * into the scratch (so a parent resumes losslessly) and stash the FULL result
 * in `terminalOverride` so the harness entry returns it VERBATIM — preserving
 * the strategy's typed {@link HaltReason} and accounting. Returns the matchable
 * {@link StrategyOutcome} for any combinator that recurses into this node.
 */
function recordTerminal(cx: ExecutionContext, result: RunResult): StrategyOutcome {
  if (result.kind === "success" || result.kind === "failure") {
    cx.scratch.runSession = result.session_state ?? emptySessionState();
  }
  const outcome = outcomeFromRunResult(result);
  cx.scratch.terminalOverride = result;
  return outcome;
}

/**
 * Take (and CLEAR) the full terminal {@link RunResult} a child strategy stashed
 * into `terminalOverride` when it returned from `runStrategy(child, cx)` (#124).
 * A combinator that recurses per-phase / per-task calls this immediately after
 * each child dispatch to fold the child's usage / turns / session back into the
 * shared execute context. Clearing the override is REQUIRED: the combinator
 * builds its OWN terminal once the loop finishes (via {@link finishCombinator}),
 * and a stale child override would otherwise propagate verbatim and mask it.
 */
function takeChildOverride(cx: ExecutionContext): RunResult | undefined {
  const result = cx.scratch.terminalOverride;
  cx.scratch.terminalOverride = undefined;
  return result;
}

/**
 * A combinator's terminal seam: finalize observability for `result`, restore the
 * parent `task` into scratch, stash `result` as the override so the harness entry
 * returns it VERBATIM, and return the matching outcome.
 */
async function finishCombinator(
  cx: ExecutionContext,
  executor: StrategyExecutor,
  parentTask: Task,
  result: RunResult,
): Promise<StrategyOutcome> {
  await executor.finalize(result);
  cx.scratch.task = parentTask;
  if (result.kind === "success" || result.kind === "failure") {
    cx.scratch.runSession = result.session_state ?? emptySessionState();
  }
  const outcome = outcomeFromRunResult(result);
  cx.scratch.terminalOverride = result;
  return outcome;
}

// ============================================================================
// Per-node budget enforcement + failure isolation helpers (issue #125)
// ============================================================================
//
// Makes {@link BudgetContext.charge} the REAL per-node enforcement point and a
// {@link StrategyOutcome} `budget_exhausted` a real, isolated, parent-inspectable
// value. The per-node partial shapes (rule 5) and the promotion boundary live
// here. No new fixtures (fork #3): every type here is runtime-only.
//
// Resolved spec forks (DECIDED — do NOT re-litigate):
//   - `escalate` carries `partialOutput = <node JSON>`; `fail` carries `undefined`.
//   - `partialOutput` is a JSON string of the structured per-node partial.
//   - `continuesUsed` persistence is DEFERRED to #129 — in-process continue ONLY.

/**
 * The last FinalResponse text from a ReAct window terminal (#125, fork #2): the
 * `success.output`, or for a `failure` the last assistant text message on its
 * post-run session state (the partial captured before exhaustion). `undefined`
 * for non-terminal pauses. Mirrors Rust `last_final_response_text`.
 */
export function lastFinalResponseText(result: RunResult): string | undefined {
  if (result.kind === "success") return result.output;
  if (result.kind === "failure") {
    const messages = result.session_state?.messages ?? [];
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const m = messages[i]!;
      if (m.role === "assistant" && m.content.type === "text") return m.content.text;
    }
    return undefined;
  }
  return undefined;
}

/**
 * ReAct partial: the window's last FinalResponse text (#125, fork #2). Each node
 * serializes its own shape; `fail` discards the partial entirely. Mirrors Rust
 * `react_partial_json`.
 */
export function reactPartialJson(lastFinalResponse: string): string {
  return JSON.stringify({ node: "react", last_final_response: lastFinalResponse });
}

/**
 * PlanExecute partial: the task list + per-task statuses + ledger (#125, fork
 * #2). `ledger` is the per-task `(id, description, status)` rows. Mirrors Rust
 * `plan_execute_partial_json`.
 */
export function planExecutePartialJson(taskList: TaskList): string {
  const ledger = taskList.tasks.map((t) => ({
    id: String(t.id),
    description: t.description,
    status: t.status,
  }));
  return JSON.stringify({
    node: "plan_execute",
    tasks: taskList.tasks.length,
    ledger,
  });
}

/**
 * SelfVerifying partial: the last worker result summary + the last verdict
 * reason (#125, fork #2). Mirrors Rust `self_verifying_partial_json`.
 */
export function selfVerifyingPartialJson(lastWorkerOutput: string, lastVerdict: string): string {
  return JSON.stringify({
    node: "self_verifying",
    last_worker_result: lastWorkerOutput,
    last_verdict: lastVerdict,
  });
}

/**
 * HillClimbing partial: the best candidate value + its score (#125, fork #2).
 * Mirrors Rust `hill_climbing_partial_json`.
 */
export function hillClimbingPartialJson(bestScore: number): string {
  return JSON.stringify({ node: "hill_climbing", best_candidate: bestScore, score: bestScore });
}

/**
 * Promote a charge-time {@link BudgetExhausted} to a {@link StrategyOutcome}
 * `budget_exhausted` (#125 promotion boundary), attaching `partialOutput`. Per
 * fork #1: an `escalate`-resolved exhaustion carries `<node JSON>`; a
 * `fail`-resolved one carries `undefined`. Mirrors Rust `promote_budget_exhausted`.
 */
export function promoteBudgetExhausted(
  err: BudgetExhausted,
  partialOutput: string | undefined,
): StrategyOutcome {
  return {
    kind: "budget_exhausted",
    policy: err.policy,
    behavior: err.behavior,
    stepsTaken: err.stepsTaken,
    continuesUsed: err.continuesUsed,
    phase: err.phase,
    partialOutput,
  };
}

// ============================================================================
// #130 — budget-exhaustion HITL pause helpers
// ============================================================================

/**
 * The advisory `available_actions` a COMBINATOR offers on a budget-exhaustion
 * pause (#130, fork C): `[continue_with_budget, skip, fail]`. The suggested
 * `steps` defaults to the scope's own allowance (or 1 for an uncapped scope).
 * Mirrors Rust `combinator_escalation_actions`.
 */
export function combinatorEscalationActions(err: BudgetExhausted): EscalationAction[] {
  const steps = budgetPolicyAllowanceValue(err.policy) ?? 1;
  return [{ kind: "continue_with_budget", steps }, { kind: "skip" }, { kind: "fail" }];
}

/**
 * The advisory `available_actions` a BARE LEAF offers on a budget-exhaustion
 * pause (#130, fork C): `[continue_with_budget, fail]` — a leaf has no sibling
 * tasks to advance to, so `skip` is OMITTED. Mirrors Rust
 * `leaf_escalation_actions`.
 */
export function leafEscalationActions(err: BudgetExhausted): EscalationAction[] {
  const steps = budgetPolicyAllowanceValue(err.policy) ?? 1;
  return [{ kind: "continue_with_budget", steps }, { kind: "fail" }];
}

/**
 * Promote a charge-time {@link BudgetExhausted} to a {@link RunResult}
 * `waiting_for_human` (#130 HITL pause boundary). Built ONLY when a node's
 * `escalate` resolution is consulted under `surface_to_human`; it carries the
 * node's `partialOutput` and the advisory `available_actions` (combinators pass
 * {@link combinatorEscalationActions}; a bare leaf passes
 * {@link leafEscalationActions} — fork C). The {@link PausedState} records the
 * node's `steps_taken` / `continues_used` (on the request) so `resumeInner` can
 * reconstruct the node's budget context from the request alone (fork E).
 *
 * The `partialOutput` is preserved both on the request (for the operator to
 * inspect) AND as a single assistant text message on the paused `session_state`
 * (so a resume re-enters the loop with that context). Mirrors Rust
 * `promote_budget_exhausted_to_human`.
 */
export function promoteBudgetExhaustedToHuman(
  err: BudgetExhausted,
  partialOutput: string | undefined,
  availableActions: EscalationAction[],
  sessionId: SessionId,
  task: Task,
  budgetUsed: BudgetSnapshot,
  turnNumber: number,
): RunResult {
  const sessionState: SessionState =
    partialOutput != null
      ? {
          messages: [{ role: "assistant", content: { type: "text", text: partialOutput } }],
          extras: {},
        }
      : emptySessionState();
  const request: HumanRequest = {
    kind: "budget_exhausted",
    phase: err.phase,
    policy: err.policy,
    steps_taken: err.stepsTaken,
    continues_used: err.continuesUsed,
    partial_output: partialOutput,
    available_actions: availableActions,
  };
  const state: PausedState = {
    session_id: sessionId,
    task_id: task.id,
    turn_number: turnNumber,
    session_state: sessionState,
    pending_tool_calls: [],
    approved_results: [],
    human_request: request,
    task,
    budget_used: budgetUsed,
    child_state: null,
  };
  return { kind: "waiting_for_human", state, request };
}

/**
 * Raise a {@link BudgetPolicy}'s per-scope cap to at least `granted` (#130
 * `continue_with_budget` grant). `unlimited` is left untouched (already
 * uncapped); a `total_steps` / `per_loop` / `per_attempt` value below `granted`
 * is raised to `granted`. Lower grants are no-ops (never SHRINKS an allowance).
 * Returns a fresh policy — does not mutate. Mirrors Rust `grant_budget_policy`.
 */
function grantBudgetPolicy(policy: BudgetPolicy, granted: number): BudgetPolicy {
  if (policy.kind === "unlimited") return policy;
  return policy.value < granted ? { kind: policy.kind, value: granted } : policy;
}

/**
 * Recurse a {@link LoopStrategy} tree raising every ReAct leaf's `budget` cap to
 * at least `granted` (#130). The combinator nodes carry no inline policy (they
 * derive it from `task.budget.max_turns`, raised by {@link grantTaskBudget}), so
 * this only touches the leaves. Returns a fresh strategy. Mirrors Rust
 * `grant_strategy_budget`.
 */
function grantStrategyBudget(ls: LoopStrategy, granted: number): LoopStrategy {
  switch (ls.kind) {
    case "react":
      return { ...ls, budget: grantBudgetPolicy(ls.budget, granted) };
    case "plan_execute":
      return {
        ...ls,
        plan: grantStrategyBudget(ls.plan, granted),
        execute: grantStrategyBudget(ls.execute, granted),
      };
    case "self_verifying":
      return { ...ls, inner: grantStrategyBudget(ls.inner, granted) };
    case "ralph":
      return { ...ls, inner: grantStrategyBudget(ls.inner, granted) };
    case "hill_climbing":
      return { ...ls, inner: grantStrategyBudget(ls.inner, granted) };
  }
}

/**
 * Reconstruct a resumed task's strategy tree with its budget caps raised to
 * `granted` (#130 `continue_with_budget`). The ReAct leaf caps live on each
 * node's own `budget` {@link BudgetPolicy}; the combinator nodes derive their cap
 * from `task.budget.max_turns`, so BOTH are raised. Fork E: `granted` is
 * `request.steps_taken + steps`, so the restored scope has room for `steps` more
 * steps after the checkpoint. Returns a fresh task — does not mutate. Mirrors
 * Rust `grant_task_budget`.
 */
export function grantTaskBudget(task: Task, granted: number): Task {
  const maxTurns = task.budget.max_turns;
  const raised = maxTurns == null || maxTurns < granted ? granted : maxTurns;
  return {
    ...task,
    budget: { ...task.budget, max_turns: raised },
    loop_strategy: grantStrategyBudget(task.loop_strategy, granted),
  };
}

/**
 * Snapshot the current budget scope as a {@link BudgetExhausted} (#125). Used by
 * the combinators to surface their OWN typed exhaustion when a CHILD returned a
 * `budget_exhausted` outcome (rule 4/7): the parent reads its own scope state
 * rather than charging the child's exhaustion against itself. `fallback` is the
 * defensive value when no scope is pushed (should not happen in a wired run).
 */
function currentBudgetExhausted(cx: ExecutionContext, fallback: BudgetExhausted): BudgetExhausted {
  const scope = cx.budgets.current();
  if (scope === undefined) return fallback;
  return {
    policy: scope.policy,
    behavior: scope.behavior,
    stepsTaken: scope.stepsTaken,
    continuesUsed: scope.continuesUsed,
    phase: scope.phase,
  };
}

/** Derive the {@link BudgetPolicy} for a combinator scope from a task's global
 *  `max_turns` ceiling: `total_steps` when capped, `unlimited` otherwise (#125). */
function combinatorBudgetPolicy(maxTurns: number | null | undefined): BudgetPolicy {
  return maxTurns != null ? { kind: "total_steps", value: maxTurns } : { kind: "unlimited" };
}

/**
 * The leaf: a bounded ReAct turn-loop window. Reads the per-run scratch
 * (`task`, `runSession`, `runBudget`) and drives one ReAct window through the
 * executor primitive. The leaf takes the run's stream sink for the window
 * (combinators that recurse per-phase suppress it by taking it first).
 *
 * ## Budget enforcement (#125)
 *
 * The leaf pushes its OWN {@link BudgetContext} scope for `c.budget` and charges
 * the turns the window consumed against it. Per Open Q A-2 (rule 6) the leaf
 * does NOT carry its own behavior and NEVER self-resolves a continue/fail/escalate
 * at the leaf — on charge exhaustion it PROPAGATES a {@link StrategyOutcome}
 * `budget_exhausted` to its parent (which owns the single recovery site). The
 * propagated `partialOutput` is the window's last FinalResponse text as JSON
 * (fork #2). A clean (non-budget) terminal is recorded verbatim through
 * {@link recordTerminal}.
 */
async function runReactConfig(c: ReactConfig, cx: ExecutionContext): Promise<StrategyOutcome> {
  const executor = requireExecutor(cx);
  if (!("reactWindow" in executor)) return executor;
  const task = currentTask(cx);
  if (task === undefined) {
    return {
      kind: "failed",
      error: new InvalidConfiguration("no task in ExecutionContext scratch"),
    };
  }
  const maxIterations = reactMaxIterations(c);
  const sessionState = cx.scratch.runSession;
  cx.scratch.runSession = emptySessionState();
  const budgetUsed = { ...cx.scratch.runBudget };
  // #124: resolve the worker agent from the registry by THIS leaf's handle
  // (genuine recursion — no `config.agent`). A missing handle is a typed
  // terminal failure.
  const agent = executor.resolveAgentRef(c.agent, task.session_id);
  if (isRunResult(agent)) {
    await executor.finalize(agent);
    return recordTerminal(cx, agent);
  }
  // #125/#129: push this leaf's OWN budget scope carrying its CONFIGURED
  // `behavior` (default `escalate`). The leaf still never RESOLVES it in the
  // nested case (rule 6: it PROPAGATES a `budget_exhausted` to its parent, which
  // owns the single recovery site). Carrying the real `behavior` only means the
  // propagated error reports it, so the TOP-LEVEL/bare-leaf resolution site
  // (`driveStrategy`) can honor it (Q1 — a bare leaf self-resolves, a nested leaf
  // does not).
  const leafBehavior = c.behavior ?? defaultBudgetBehavior();
  pushBudget(cx, c.budget, leafBehavior, "react");
  const onStream = cx.stream;
  cx.stream = undefined;
  const result = await executor.reactWindow(
    task,
    maxIterations,
    sessionState,
    budgetUsed,
    onStream,
    cx.signal,
    agent,
  );
  await executor.finalize(result);

  // #125: charge the window's turns against this leaf's OWN scope. The leaf
  // POLICY (`c.budget`) — not the global BudgetLimits backstop — is the per-node
  // enforcement point. When the LEAF cap is the binding constraint (the window
  // consumed >= the leaf policy value), the leaf is exhausted and PROPAGATES a
  // typed `budget_exhausted` to its parent (rule 6 — the leaf never self-resolves).
  // When the smaller GLOBAL backstop trips first, the legacy `budget_exceeded`
  // terminal is recorded VERBATIM (the global cap is unchanged, #117 backstop).
  const windowTurns = result.kind === "success" || result.kind === "failure" ? result.turns : 0;
  const windowHitBudget = result.kind === "failure" && result.reason.kind === "budget_exceeded";
  const leafAllowance = budgetPolicyAllowanceValue(c.budget);
  const leafCapBinding =
    windowHitBudget && leafAllowance !== undefined && windowTurns >= leafAllowance;
  const charge = chargeCurrent(cx, windowTurns);
  if (leafCapBinding || !charge.ok) {
    // The leaf partial is the window's last FinalResponse text.
    const lastFinal = lastFinalResponseText(result) ?? "";
    // Carry the post-run session so a parent resumes losslessly.
    if (result.kind === "success" || result.kind === "failure") {
      cx.scratch.runSession = result.session_state ?? emptySessionState();
    }
    const err: BudgetExhausted = !charge.ok
      ? charge.error
      : currentBudgetExhausted(cx, {
          policy: c.budget,
          behavior: leafBehavior,
          stepsTaken: windowTurns,
          continuesUsed: 0,
          phase: "react",
        });
    popBudget(cx);
    // Rule 6: the leaf PROPAGATES — partial carries the last FinalResponse
    // (escalate semantics, fork #1/#2).
    return promoteBudgetExhausted(err, reactPartialJson(lastFinal));
  }
  popBudget(cx);
  return recordTerminal(cx, result);
}

/** Narrow an executor resolution result to a {@link RunResult} (the failure
 *  branch of `resolveAgentRef` / `resolveMetricEvaluator`). */
function isRunResult(value: unknown): value is RunResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "kind" in value &&
    ((value as { kind: unknown }).kind === "success" ||
      (value as { kind: unknown }).kind === "failure" ||
      (value as { kind: unknown }).kind === "waiting_for_human" ||
      (value as { kind: unknown }).kind === "consult" ||
      (value as { kind: unknown }).kind === "escalate") &&
    "session_id" in value
  );
}

/**
 * Descend a {@link LoopStrategy} tree to the worker leaf's agent key (#124).
 * Mirrors Rust `worker_agent_key_of` — used by SelfVerifying (Q1c: the
 * evaluate-phase agent defaults to the inner worker's agent) and as the descent
 * point for Ralph's per-window override.
 */
function workerAgentKeyOf(ls: LoopStrategy): string {
  switch (ls.kind) {
    case "react":
      return ls.agent;
    case "plan_execute":
      return workerAgentKeyOf(ls.execute);
    case "self_verifying":
      return workerAgentKeyOf(ls.inner);
    case "ralph":
      return ls.agent !== "" ? ls.agent : workerAgentKeyOf(ls.inner);
    case "hill_climbing":
      return workerAgentKeyOf(ls.inner);
  }
}

/**
 * Rewrite the worker leaf's agent handle of `ls` to `agent` (#124 Q3 — Ralph's
 * per-window agent override). Returns a structurally-cloned tree with the leaf
 * reached by descending the worker child chain swapped.
 */
function overrideWorkerAgent(ls: LoopStrategy, agent: AgentRef): LoopStrategy {
  switch (ls.kind) {
    case "react":
      return { ...ls, agent };
    case "plan_execute":
      return { ...ls, execute: overrideWorkerAgent(ls.execute, agent) };
    case "self_verifying":
      return { ...ls, inner: overrideWorkerAgent(ls.inner, agent) };
    case "ralph":
      return { ...ls, inner: overrideWorkerAgent(ls.inner, agent) };
    case "hill_climbing":
      return { ...ls, inner: overrideWorkerAgent(ls.inner, agent) };
  }
}

/**
 * Fold a child terminal's usage + turns into the cumulative `totalUsage` and
 * the shared `carried` budget snapshot (#124). Mirrors Rust `fold_usage`:
 * `waiting_for_human` / `consult` are non-terminal pauses and carry nothing;
 * `success` / `failure` / `escalate` fold usage and advance `carried.turns` to
 * the max.
 */
function foldUsageInto(
  totalUsage: AggregateUsage,
  carried: BudgetSnapshot,
  result: RunResult,
): void {
  if (result.kind === "waiting_for_human" || result.kind === "consult") return;
  const usage = result.usage;
  totalUsage.input_tokens += usage.input_tokens;
  totalUsage.output_tokens += usage.output_tokens;
  totalUsage.cache_read_tokens += usage.cache_read_tokens;
  totalUsage.cache_write_tokens += usage.cache_write_tokens;
  totalUsage.cost_usd += usage.cost_usd;
  carried.input_tokens += usage.input_tokens;
  carried.output_tokens += usage.output_tokens;
  carried.turns = Math.max(carried.turns, result.turns);
}

/**
 * Plan→execute (#124). GENUINELY recursive: the plan phase dispatches
 * `runStrategy(c.plan, cx)` (seeding the planning directive + a one-turn budget
 * on the scratch first) and the execute phase dispatches `runStrategy(c.execute,
 * cx)` ONCE PER TASK. The child strategy's full loop runs for each phase — a
 * non-ReAct execute child (SelfVerifying / HillClimbing) genuinely executes its
 * loop per task, not a hardcoded flat ReAct (the bug this fixes: the old body
 * called `executePhase`, which hardcoded a ReAct sub-loop and silently dropped
 * the configured execute child).
 *
 * This combinator OWNS the orchestration: per-task turn/budget allocation (Q1),
 * the `on_task_advance` hook (pre, mutable), seeding each step instruction as a
 * user message, A.6 deep-resume against the durable RunStore checkpoint,
 * task-list persistence after each transition (Q4), and cumulative usage /
 * last-output / last-state carry. The executor keeps only LEAF primitives
 * (directive, plan dispatch, artifact capture/persist, deep-resume reconcile,
 * `on_task_advance` fire) — none of which touch the per-task model loop. The
 * ready-set walk lands in #126 (execute runs per task sequentially for now).
 */
async function runPlanExecuteConfig(
  c: PlanExecuteConfig,
  cx: ExecutionContext,
): Promise<StrategyOutcome> {
  const executor = requireExecutor(cx);
  if (!("runPlanSubtree" in executor)) return executor;
  const task = currentTask(cx);
  if (task === undefined) {
    return {
      kind: "failed",
      error: new InvalidConfiguration("no task in ExecutionContext scratch"),
    };
  }
  const sessionId = task.session_id;
  // The incoming shared execute session ( `[user: task.instruction]` ).
  const baseSession = cx.scratch.runSession;
  cx.scratch.runSession = emptySessionState();
  const budgetUsed: BudgetSnapshot = { ...cx.scratch.runBudget };
  // PlanExecute suppresses the run's stream sink for its phases (parent-visible
  // step boundaries are re-emitted on `cx.stream`). Take it now and keep it OUT
  // of `cx.stream` so the recursive children run with a suppressed sink; restore
  // it before returning.
  const onStream = cx.stream;
  cx.stream = undefined;

  // ── Phase 1: plan (dispatch through `c.plan`). ──────────────────────────────
  //
  // Seed the planning directive onto a CLONE of the base session so the shared
  // execute context stays `[user: task.instruction]` (#93 — a leaked directive
  // would make every execute step re-emit a plan). Cap the plan child at ONE
  // turn (R1) but never beyond the task's global turn ceiling (so an already-
  // exhausted budget fails the plan turn before it runs — R10).
  const directive = executor.planDirective(task.instruction);
  const planSession = structuredClone(baseSession);
  await executor.seedUserMessage(planSession, directive);
  const planCap =
    task.budget.max_turns != null
      ? Math.min(task.budget.max_turns, budgetUsed.turns + 1)
      : budgetUsed.turns + 1;
  const planTask: Task = {
    id: task.id,
    instruction: directive,
    session_id: sessionId,
    budget: { ...task.budget, max_turns: planCap },
    loop_strategy: c.plan,
  };
  const planResult = await executor.runPlanSubtree(
    c.plan,
    planTask,
    planSession,
    { ...budgetUsed },
    cx.signal,
  );

  let planOutput: string;
  let planUsage: AggregateUsage;
  let planTurns: number;
  if (planResult == null) {
    cx.stream = onStream;
    const result: RunResult = {
      kind: "failure",
      reason: {
        kind: "plan_phase_failed",
        error: { kind: "planning_turn_failed", message: "plan sub-strategy produced no terminal" },
      },
      session_id: sessionId,
      usage: emptyAggregateUsage(),
      turns: budgetUsed.turns,
      session_state: emptySessionState(),
    };
    return finishCombinator(cx, executor, task, result);
  } else if (planResult.kind === "success") {
    planOutput = planResult.output;
    planUsage = planResult.usage;
    planTurns = planResult.turns;
  } else {
    // A non-success plan terminal (budget / agent error / pause) propagates
    // verbatim — the run never reaches execute.
    cx.stream = onStream;
    return finishCombinator(cx, executor, task, planResult);
  }

  // Capture + persist the artifact from the plan child's output (R3/R4/R11) —
  // the harness-side machinery, no model turn.
  const outcome = await executor.capturePlanArtifact(
    sessionId,
    planOutput,
    planUsage,
    planTurns,
    cx.signal,
  );
  if (!outcome.ok) {
    cx.stream = onStream;
    return finishCombinator(cx, executor, task, outcome.failure);
  }

  // #126 decision C: the runnable task list comes from the persisted `task_list`
  // tool store (the ONE authoring path — it can carry real blockers). Fall back
  // to the linear plan-artifact bridge only when nothing was authored via the
  // tool (back-compat with the #59/#124 plan-only path and its replay fixtures).
  // The plan artifact is still captured + persisted above so the plan-phase
  // replay tests stay green.
  const persisted = await executor.loadTaskList(sessionId);
  const taskList: TaskList =
    persisted != null && persisted.tasks.length > 0
      ? persisted
      : planArtifactToTaskList(outcome.artifact);
  if (taskList.tasks.length === 0) {
    cx.stream = onStream;
    const result: RunResult = {
      kind: "failure",
      reason: { kind: "empty_plan" },
      session_id: sessionId,
      usage: outcome.usage,
      turns: outcome.turns,
    };
    return finishCombinator(cx, executor, task, result);
  }

  // #126 AC5: re-check the WHOLE graph for cycles at execute entry (defense in
  // depth — `add_task` already rejects cycles, but the persisted store could be
  // cyclic out of band). No task runs.
  if (hasCycle(taskList)) {
    cx.stream = onStream;
    const result: RunResult = {
      kind: "failure",
      reason: {
        kind: "task_graph_cycle",
        reason: "persisted task graph contains a directed cycle",
      },
      session_id: sessionId,
      usage: outcome.usage,
      turns: outcome.turns,
    };
    return finishCombinator(cx, executor, task, result);
  }

  await executor.persistTaskList(sessionId, taskList);

  // Carry the shared budget past the plan turn.
  const carried: BudgetSnapshot = { ...budgetUsed };
  carried.turns = outcome.turns;
  carried.input_tokens += outcome.usage.input_tokens;
  carried.output_tokens += outcome.usage.output_tokens;

  // ── Phase 2: ready-set DAG walk (#126). ─────────────────────────────────────
  //
  // Repeatedly pick the lowest-id `pending` task whose blockers are all
  // `completed`. A task whose blocker FAILED is cascade-`blocked` (so it is no
  // longer `pending` and never becomes ready). When no `pending` task is ready,
  // the walk drains.

  // A.6 deep-resume (Q2): reconcile against the durable checkpoint so already-
  // Completed tasks are not re-run.
  await executor.reconcileCompletedTasks(sessionId, taskList);

  // Total positional count for the on_task_advance hook (stable).
  const totalTasks = taskList.tasks.length;
  const totalUsage: AggregateUsage = { ...outcome.usage };
  let lastOutput = "";
  let lastState: SessionState = emptySessionState();

  // #126 Tier-1/Tier-2 + cascade run-local state.
  // - `finalOutputs`: each completed task's `success.output` (Tier-1).
  // - `ledger`: the Tier-2 global running ledger (bounded, drop-oldest).
  // - `ledgerElided`: sticky flag set the first time entries are dropped.
  // - `blockedByFailure`: tasks cascade-`blocked` by a terminal failure
  //   (decision E — run scratch, NOT a TaskStatus variant).
  const finalOutputs = new Map<number, string>();
  const ledger: StepLedgerEntry[] = [];
  let ledgerElided = false;
  const blockedByFailure = new Set<number>();
  // The first terminal failure that triggered a cascade (decision A).
  let firstFailure: { failedTask: number; reason: string } | undefined;

  // #125/#129: PlanExecute owns a budget scope for its execute phase carrying its
  // CONFIGURED `behavior` (default `escalate`). Enforcement is `charge`-based per
  // node.
  const peBehavior = c.behavior ?? defaultBudgetBehavior();
  pushBudget(cx, combinatorBudgetPolicy(task.budget.max_turns), peBehavior, "plan_execute");

  for (;;) {
    const taskId = nextReady(taskList);
    if (taskId === undefined) break;
    const index = taskList.tasks.findIndex((t) => t.id === taskId);
    const instruction = taskList.tasks[index]!.description;

    // Mark in_progress and re-persist (Q4).
    updateTask(taskList, taskId, "in_progress");
    await executor.persistTaskList(sessionId, taskList);

    // Fire on_task_advance (pre, mutable). The hook may rewrite the step
    // instruction; the (possibly mutated) instruction seeds the execute child.
    const stepTask: Task = {
      id: task.id,
      instruction,
      session_id: sessionId,
      budget: { ...task.budget },
      loop_strategy: c.execute,
    };
    await executor.fireTaskAdvance(sessionId, stepTask, index, totalTasks, cx.signal);

    // #126 Tier-1 scoped context: seed this step from a FRESH copy of the base
    // session (NOT a forward-folded shared transcript — that breaks on a DAG)
    // plus, for THIS task's transitive blockers ONLY, their final outputs + their
    // ledger rows. Independent branches never appear (AC1 isolation).
    const stepSession = structuredClone(baseSession);
    const blockers = transitiveBlockers(taskList, taskId);
    if (blockers.length > 0) {
      const blockerSet = new Set(blockers);
      // Tier-1: transitive blockers' final outputs (ascending id).
      const tier1Lines: string[] = [];
      for (const b of blockers) {
        const out = finalOutputs.get(b);
        if (out !== undefined) tier1Lines.push(`#${b} result: ${out}`);
      }
      if (tier1Lines.length > 0) {
        await executor.seedUserMessage(
          stepSession,
          `Results from upstream tasks:\n${tier1Lines.join("\n")}`,
        );
      }
      // Tier-1 ledger: the Tier-2 ledger rows for this transitive set.
      const scoped = ledger.filter((e) => blockerSet.has(e.task_id));
      const scopedBlock = renderStepLedger(scoped, false);
      if (scopedBlock !== undefined) await executor.seedUserMessage(stepSession, scopedBlock);
    }

    // #126 Tier-2: inject the FULL global running ledger into EVERY step (with
    // the static elision marker once entries were dropped).
    const tier2Block = renderStepLedger(ledger, ledgerElided);
    if (tier2Block !== undefined) await executor.seedUserMessage(stepSession, tier2Block);

    // Finally seed this step's own instruction.
    await executor.seedUserMessage(stepSession, stepTask.instruction);

    // #126 AC2: clear the observed-write accumulator so this task's
    // files_touched reflect ONLY the writes this step issues.
    executor.clearObservedWrites();

    // #125: absolute turn count BEFORE this step, so the success path charges
    // only the DELTA against the PlanExecute scope.
    const carriedBefore = carried.turns;
    cx.scratch.task = stepTask;
    cx.scratch.runSession = stepSession;
    cx.scratch.runBudget = { ...carried };
    const stepOutcome = await runStrategy(c.execute, cx);

    // ── BudgetExhausted (#125 rule 4/7) — resolve THIS scope. ──────────────
    if (stepOutcome.kind === "budget_exhausted") {
      const err = currentBudgetExhausted(cx, {
        policy: combinatorBudgetPolicy(task.budget.max_turns),
        behavior: peBehavior,
        stepsTaken: carried.turns,
        continuesUsed: 0,
        phase: "plan_execute",
      });
      const resolution = resolveCurrent(cx);
      // #129: a granted `continue` loops IN-PROCESS — the scope's `resolveCurrent`
      // already reset `stepsTaken` and bumped `continuesUsed`. Reset this task to
      // `pending` and re-enter the ready-set walk so it runs again under the
      // refreshed scope allowance (NO serialization — AC3). `max_continues` bounds
      // the loop: once continues are spent, the chain falls through to
      // `fail`/`escalate`.
      if (resolution === "continue") {
        updateTask(taskList, taskId, "pending");
        await executor.persistTaskList(sessionId, taskList);
        continue;
      }
      if (resolution === "fail") {
        // #126 AC4: a budget-`fail` task cascades IDENTICALLY to an error-failed
        // one — block its transitive dependents and keep scheduling unrelated
        // tasks.
        updateTask(taskList, taskId, "blocked");
        for (const dep of transitiveDependents(taskList, taskId)) {
          updateTask(taskList, dep, "blocked");
          blockedByFailure.add(dep);
        }
        blockedByFailure.add(taskId);
        await executor.persistTaskList(sessionId, taskList);
        if (firstFailure === undefined) {
          firstFailure = {
            failedTask: taskId,
            reason: `budget exhausted (fail): ${JSON.stringify(err.policy)}`,
          };
        }
        continue;
      }
      // Escalate: under `autonomous`, surface the partial and abort the run.
      // Under `surface_to_human` (#130) the node PAUSES with a `budget_exhausted`
      // request via the `terminalOverride` seam instead of propagating up.
      // Combinators offer [continue_with_budget, skip, fail]. (#129: `continue` is
      // handled above as an in-process re-schedule.)
      updateTask(taskList, taskId, "blocked");
      await executor.persistTaskList(sessionId, taskList);
      const partial = planExecutePartialJson(taskList);
      popBudget(cx);
      cx.scratch.task = task;
      cx.stream = onStream;
      if (executor.escalationMode().kind === "surface_to_human") {
        const waiting = promoteBudgetExhaustedToHuman(
          err,
          partial,
          combinatorEscalationActions(err),
          sessionId,
          task,
          { ...carried },
          carried.turns,
        );
        return recordTerminal(cx, waiting);
      }
      return promoteBudgetExhausted(err, partial);
    }

    const subResult = takeChildOverride(cx);

    if (subResult != null && subResult.kind === "success") {
      carried.turns = subResult.turns;
      lastState = runResultSessionState(subResult);
      carried.input_tokens += subResult.usage.input_tokens;
      carried.output_tokens += subResult.usage.output_tokens;
      totalUsage.input_tokens += subResult.usage.input_tokens;
      totalUsage.output_tokens += subResult.usage.output_tokens;
      totalUsage.cache_read_tokens += subResult.usage.cache_read_tokens;
      totalUsage.cache_write_tokens += subResult.usage.cache_write_tokens;
      totalUsage.cost_usd += subResult.usage.cost_usd;
      lastOutput = subResult.output;

      completeTask(taskList, taskId);
      await executor.persistTaskList(sessionId, taskList);

      // #126: record this task's final output (Tier-1) and append a ledger entry
      // whose files_touched is HARNESS-OBSERVED (AC2) — never self-reported.
      finalOutputs.set(taskId, subResult.output);
      const filesTouched = executor.takeObservedWrites();
      const entry: StepLedgerEntry = {
        task_id: taskId,
        summary: subResult.output,
        files_touched: filesTouched,
      };
      if (pushStepLedger(ledger, entry)) ledgerElided = true;

      // Surface the completed step's final text to the caller's sink — the
      // parent-visible step boundary.
      if (onStream != null) onStream({ kind: "final_response", content: lastOutput });

      // #125: charge this step's turns against the PlanExecute scope.
      const stepCharge = chargeCurrent(cx, Math.max(0, subResult.turns - carriedBefore));
      if (!stepCharge.ok) {
        const resolution = resolveCurrent(cx);
        // #129: a granted `continue` refreshes the scope and keeps scheduling the
        // remaining ready tasks IN-PROCESS (this step already completed). NO
        // serialization (AC3); do NOT pop the scope.
        if (resolution === "continue") {
          continue;
        }
        const partial = planExecutePartialJson(taskList);
        popBudget(cx);
        cx.scratch.task = task;
        cx.stream = onStream;
        switch (resolution) {
          case "fail":
            return promoteBudgetExhausted(stepCharge.error, undefined);
          // #130: under `surface_to_human`, PAUSE with a `budget_exhausted`
          // request instead of propagating up. Combinators offer
          // [continue_with_budget, skip, fail]. (#129: `continue` handled above.)
          case "escalate":
            if (executor.escalationMode().kind === "surface_to_human") {
              const waiting = promoteBudgetExhaustedToHuman(
                stepCharge.error,
                partial,
                combinatorEscalationActions(stepCharge.error),
                sessionId,
                task,
                { ...carried },
                carried.turns,
              );
              return recordTerminal(cx, waiting);
            }
            return promoteBudgetExhausted(stepCharge.error, partial);
        }
      }
    } else if (
      subResult != null &&
      subResult.kind === "failure" &&
      subResult.reason.kind === "budget_exceeded"
    ) {
      // A GLOBAL turn-budget hard stop (#117 backstop) surfaces as a
      // `budget_exceeded` failure from the leaf. That is a WHOLE-RUN hard stop,
      // NOT a single-task terminal failure — it aborts the run verbatim
      // (preserving the pre-#126 mid-execute budget behavior). Distinct from a
      // per-NODE `budget_exhausted` resolving to `fail`, which DOES cascade (AC4).
      totalUsage.input_tokens += subResult.usage.input_tokens;
      totalUsage.output_tokens += subResult.usage.output_tokens;
      totalUsage.cache_read_tokens += subResult.usage.cache_read_tokens;
      totalUsage.cache_write_tokens += subResult.usage.cache_write_tokens;
      totalUsage.cost_usd += subResult.usage.cost_usd;
      updateTask(taskList, taskId, "blocked");
      await executor.persistTaskList(sessionId, taskList);
      popBudget(cx);
      cx.stream = onStream;
      const result: RunResult = {
        kind: "failure",
        reason: subResult.reason,
        session_id: sessionId,
        usage: totalUsage,
        turns: subResult.turns,
        session_state: lastState,
      };
      return finishCombinator(cx, executor, task, result);
    } else if (subResult != null && subResult.kind === "failure") {
      // #126 AC3: a terminal task FAILURE cascade-blocks its transitive
      // dependents and KEEPS scheduling unrelated tasks (replaces the Q5 blanket
      // abort).
      carried.turns = subResult.turns;
      totalUsage.input_tokens += subResult.usage.input_tokens;
      totalUsage.output_tokens += subResult.usage.output_tokens;
      totalUsage.cache_read_tokens += subResult.usage.cache_read_tokens;
      totalUsage.cache_write_tokens += subResult.usage.cache_write_tokens;
      totalUsage.cost_usd += subResult.usage.cost_usd;

      updateTask(taskList, taskId, "blocked");
      for (const dep of transitiveDependents(taskList, taskId)) {
        updateTask(taskList, dep, "blocked");
        blockedByFailure.add(dep);
      }
      blockedByFailure.add(taskId);
      await executor.persistTaskList(sessionId, taskList);
      if (firstFailure === undefined) {
        firstFailure = { failedTask: taskId, reason: haltReasonToString(subResult.reason) };
      }
      continue;
    } else if (subResult != null) {
      // A pause / consult / escalate propagates the whole run verbatim.
      popBudget(cx);
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, subResult);
    } else {
      // No terminal from the child: treat as a terminal failure of this task and
      // cascade (same as a Failure).
      updateTask(taskList, taskId, "blocked");
      for (const dep of transitiveDependents(taskList, taskId)) {
        updateTask(taskList, dep, "blocked");
        blockedByFailure.add(dep);
      }
      blockedByFailure.add(taskId);
      await executor.persistTaskList(sessionId, taskList);
      if (firstFailure === undefined) {
        firstFailure = {
          failedTask: taskId,
          reason: "execute sub-strategy produced no terminal",
        };
      }
      continue;
    }
  }

  popBudget(cx);
  cx.stream = onStream;

  // ── Drain (#126, decision A). ───────────────────────────────────────────────
  //
  // A run where a terminal failure cascade-blocked any task returns a PARTIAL
  // terminal `failure` reporting the full partition. A run where every task
  // completed returns `success` (output = last step's text).
  if (firstFailure !== undefined) {
    const completed = taskList.tasks
      .filter((t) => t.status === "completed")
      .map((t) => t.id)
      .sort((a, b) => a - b);
    const blocked = [...blockedByFailure].sort((a, b) => a - b);
    const result: RunResult = {
      kind: "failure",
      reason: {
        kind: "tasks_blocked_by_failure",
        completed,
        blocked,
        failed_task: firstFailure.failedTask,
        reason: firstFailure.reason,
      },
      session_id: sessionId,
      usage: totalUsage,
      turns: carried.turns,
      session_state: lastState,
    };
    return finishCombinator(cx, executor, task, result);
  }

  const result: RunResult = {
    kind: "success",
    output: lastOutput,
    session_id: sessionId,
    usage: totalUsage,
    turns: carried.turns,
    session_state: lastState,
  };
  return finishCombinator(cx, executor, task, result);
}

/**
 * SelfVerifying (#124): GENUINELY recursive build↔evaluate loop. Each iteration
 * dispatches `runStrategy(c.inner, cx)` for the build phase (a non-ReAct inner —
 * e.g. PlanExecute — really runs its whole loop per iteration), then runs a fresh
 * evaluate phase on the inner worker's resolved agent (Q1c) and consults the
 * verifier resolved from `c.evaluator`'s key (Q1a). Passed ⇒ Success; Failed ⇒
 * append the reason (Default-FAIL) and loop; exhausted ⇒ `self_verify_exhausted`.
 */
async function runSelfVerifyingConfig(
  c: SelfVerifyingConfig,
  cx: ExecutionContext,
): Promise<StrategyOutcome> {
  const executor = requireExecutor(cx);
  if (!("evaluatePhase" in executor)) return executor;
  const task = currentTask(cx);
  if (task === undefined) {
    return {
      kind: "failed",
      error: new InvalidConfiguration("no task in ExecutionContext scratch"),
    };
  }
  const buildSessionId = task.session_id;
  let sessionState = cx.scratch.runSession;
  cx.scratch.runSession = emptySessionState();
  const carried: BudgetSnapshot = { ...cx.scratch.runBudget };
  // Suppress the run's stream sink for the recursive child phases.
  const onStream = cx.stream;
  cx.stream = undefined;

  // Q1a: resolve the verifier from `evaluator`'s key (NO wire change).
  const verifier = cx.registry.resolveVerifier(c.evaluator);
  if (verifier == null) {
    const result: RunResult = {
      kind: "failure",
      reason: {
        kind: "self_verify_misconfigured",
        reason: `self_verifying requires a verifier registered under key ${JSON.stringify(c.evaluator)}`,
      },
      session_id: buildSessionId,
      usage: emptyAggregateUsage(),
      turns: 0,
      session_state: emptySessionState(),
    };
    cx.stream = onStream;
    return finishCombinator(cx, executor, task, result);
  }
  // Q1c: the evaluate-phase agent defaults to the inner worker's agent.
  const evalAgent = executor.resolveAgentRef(workerAgentKeyOf(c.inner), buildSessionId);
  if (isRunResult(evalAgent)) {
    cx.stream = onStream;
    return finishCombinator(cx, executor, task, evalAgent);
  }

  const maxIterations = verifier.maxIterations();
  const totalUsage: AggregateUsage = emptyAggregateUsage();
  let lastReason = "";
  let lastWorkerOutput = "";

  // #125/#129: SelfVerifying owns a budget scope for its build↔evaluate loop.
  // POLICY is the task's global turn ceiling (`total_steps`); behavior is its
  // CONFIGURED `behavior` (default `escalate`).
  const svBehavior = c.behavior ?? defaultBudgetBehavior();
  pushBudget(cx, combinatorBudgetPolicy(task.budget.max_turns), svBehavior, "self_verifying");

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    // ── Build phase: recurse `runStrategy(c.inner, cx)`.
    const buildTask: Task = {
      id: task.id,
      instruction: task.instruction,
      session_id: buildSessionId,
      budget: task.budget,
      loop_strategy: c.inner,
    };
    cx.scratch.task = buildTask;
    cx.scratch.runSession = {
      messages: [...sessionState.messages],
      extras: { ...sessionState.extras },
    };
    cx.scratch.runBudget = { ...carried };
    const carriedBefore = carried.turns;
    const buildOutcome = await runStrategy(c.inner, cx);
    // #125 rule 4/7: a child's `budget_exhausted` reaches THIS parent as a
    // `StrategyOutcome`, never auto-cascaded. SelfVerifying surfaces its own
    // typed `budget_exhausted` (partial = last worker result + last verdict)
    // without charging the child's exhaustion against its own scope.
    if (buildOutcome.kind === "budget_exhausted") {
      const partial = selfVerifyingPartialJson(lastWorkerOutput, lastReason);
      const err = currentBudgetExhausted(cx, {
        policy: combinatorBudgetPolicy(task.budget.max_turns),
        behavior: svBehavior,
        stepsTaken: carried.turns,
        continuesUsed: 0,
        phase: "self_verifying",
      });
      popBudget(cx);
      cx.scratch.task = task;
      cx.stream = onStream;
      return promoteBudgetExhausted(err, partial);
    }
    const buildResult: RunResult = takeChildOverride(cx) ?? {
      kind: "failure",
      reason: {
        kind: "self_verify_misconfigured",
        reason: "build sub-strategy produced no terminal",
      },
      session_id: buildSessionId,
      usage: emptyAggregateUsage(),
      turns: carried.turns,
      session_state: {
        messages: [...sessionState.messages],
        extras: { ...sessionState.extras },
      },
    };
    foldUsageInto(totalUsage, carried, buildResult);

    // A paused / escalated build propagates verbatim.
    if (
      buildResult.kind === "waiting_for_human" ||
      buildResult.kind === "consult" ||
      buildResult.kind === "escalate"
    ) {
      popBudget(cx);
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, buildResult);
    }
    // Capture the build's output for the partial (last worker result).
    if (buildResult.kind === "success") {
      lastWorkerOutput = buildResult.output;
    }
    // Carry the build's post-run session forward for the next round.
    if (buildResult.kind === "success" || buildResult.kind === "failure") {
      sessionState = runResultSessionState(buildResult);
    }

    // #125: charge this iteration's build turns against the SelfVerifying scope.
    // If the global cap is spent, the node surfaces its OWN typed
    // `budget_exhausted` (partial = last worker result + last verdict).
    const buildCharge = chargeCurrent(cx, Math.max(0, carried.turns - carriedBefore));
    if (!buildCharge.ok) {
      const resolution = resolveCurrent(cx);
      // #129: a granted `continue` resets the scope and RE-RUNS the build↔evaluate
      // iteration IN-PROCESS (do NOT pop the scope; the loop continues under the
      // refreshed allowance). NO serialization (AC3). `max_continues` bounds the
      // loop.
      if (resolution === "continue") {
        continue;
      }
      const partial = selfVerifyingPartialJson(lastWorkerOutput, lastReason);
      popBudget(cx);
      cx.scratch.task = task;
      cx.stream = onStream;
      switch (resolution) {
        case "fail":
          return promoteBudgetExhausted(buildCharge.error, undefined);
        // #130: under `surface_to_human`, PAUSE with a `budget_exhausted` request
        // instead of propagating up. Combinators offer
        // [continue_with_budget, skip, fail]. (#129: `continue` handled above as
        // an in-process re-run.)
        case "escalate":
          if (executor.escalationMode().kind === "surface_to_human") {
            const waiting = promoteBudgetExhaustedToHuman(
              buildCharge.error,
              partial,
              combinatorEscalationActions(buildCharge.error),
              buildSessionId,
              task,
              { ...carried },
              carried.turns,
            );
            return recordTerminal(cx, waiting);
          }
          return promoteBudgetExhausted(buildCharge.error, partial);
      }
    }

    // ── Evaluate phase: a fresh evaluator run on `evalAgent`.
    const evalResult = await executor.evaluatePhase(task, evalAgent, carried, totalUsage);

    const input: VerifierInput = {
      build_result: buildResult,
      eval_result: evalResult,
      workspace: executor.workspaceRoot(),
      iteration,
    };
    const verdict = await verifier.verify(input);
    if (verdict.kind === "passed") {
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
      popBudget(cx);
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, result);
    }
    // Default-FAIL: inject the reason into the build context and loop.
    lastReason = verdict.reason;
    await executor.appendUserMessage(sessionState, verdict.reason);
  }

  popBudget(cx);
  const result: RunResult = {
    kind: "failure",
    reason: { kind: "self_verify_exhausted", iterations: maxIterations, last_reason: lastReason },
    session_id: buildSessionId,
    usage: totalUsage,
    turns: carried.turns,
    session_state: sessionState,
  };
  cx.stream = onStream;
  return finishCombinator(cx, executor, task, result);
}

/**
 * Ralph (#124): GENUINELY recursive continuation wrapper. Each context window
 * seeds a FRESH session from the `.spore/` checkpoint, then recurses
 * `runStrategy(innerForWindow, cx)` (a non-ReAct inner — e.g. SelfVerifying —
 * really runs its whole loop per window). Q3: when `c.agent` is set it OVERRIDES
 * the inner leaf's agent per window; when unset the worker resolves via the inner
 * leaf. `ralphCompletionStatus` drives the OUTER reset loop; exhaustion ⇒
 * `ralph_completion_unmet`.
 */
async function runRalphConfig(c: RalphConfig, cx: ExecutionContext): Promise<StrategyOutcome> {
  const executor = requireExecutor(cx);
  if (!("ralphSeedSession" in executor)) return executor;
  const task = currentTask(cx);
  if (task === undefined) {
    return {
      kind: "failed",
      error: new InvalidConfiguration("no task in ExecutionContext scratch"),
    };
  }
  const onStream = cx.stream;
  cx.stream = undefined;
  // Ralph discards the incoming session state by design (each window is a fresh
  // start re-seeded from the filesystem checkpoint).
  cx.scratch.runSession = emptySessionState();
  const maxResets = Math.max(executor.ralphMaxResets(), 1);

  // Q3: when `c.agent` is set, override the inner leaf's agent for every window
  // by rewriting the inner tree's worker leaf handle.
  const innerForWindow: LoopStrategy =
    c.agent === "" ? c.inner : overrideWorkerAgent(c.inner, c.agent);

  const totalUsage: AggregateUsage = emptyAggregateUsage();
  let cumulativeTurns = 0;
  let lastReason = ".spore/progress.json missing";
  let lastSessionId = task.session_id;

  for (let iteration = 0; iteration < maxResets; iteration += 1) {
    const windowSessionId = iteration === 0 ? task.session_id : SessionId.generate();
    lastSessionId = windowSessionId;

    // R2/R3: a FRESH session seeded from the `.spore/` checkpoint.
    const sessionState = await executor.ralphSeedSession(task.instruction);

    const windowTask: Task = {
      id: task.id,
      instruction: task.instruction,
      session_id: windowSessionId,
      budget: task.budget,
      loop_strategy: innerForWindow,
    };
    cx.scratch.task = windowTask;
    cx.scratch.runSession = sessionState;
    // FRESH per-window budget (the reset discards the turn budget).
    cx.scratch.runBudget = emptyBudgetSnapshot();
    const windowOutcome = await runStrategy(innerForWindow, cx);
    // #125 rule 4/7: a window child's `budget_exhausted` reaches Ralph as a
    // `StrategyOutcome`, never auto-cascaded. Ralph's recovery semantics: a
    // budget-exhausted window is treated as "window incomplete" — RESET the
    // context window and retry (next outer iteration). After `maxResets` this
    // falls through to `ralph_completion_unmet`. Ralph's own scope is unaffected.
    if (windowOutcome.kind === "budget_exhausted") {
      lastReason = `window ${iteration + 1} budget-exhausted: ${
        windowOutcome.partialOutput ?? "<no partial>"
      }`;
      continue;
    }
    const windowResult: RunResult = takeChildOverride(cx) ?? {
      kind: "failure",
      reason: {
        kind: "ralph_completion_unmet",
        iterations: iteration + 1,
        last_reason: "window sub-strategy produced no terminal",
      },
      session_id: windowSessionId,
      usage: emptyAggregateUsage(),
      turns: 0,
      session_state: emptySessionState(),
    };
    const windowBudget = emptyBudgetSnapshot();
    foldUsageInto(totalUsage, windowBudget, windowResult);
    cumulativeTurns += windowBudget.turns;

    // A paused / escalated window propagates verbatim.
    if (
      windowResult.kind === "waiting_for_human" ||
      windowResult.kind === "consult" ||
      windowResult.kind === "escalate"
    ) {
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, windowResult);
    }

    const status = executor.ralphCompletionStatus();
    if (status == null) {
      const output = windowResult.kind === "success" ? windowResult.output : "";
      const finalState =
        windowResult.kind === "success" ? runResultSessionState(windowResult) : emptySessionState();
      const result: RunResult = {
        kind: "success",
        output,
        session_id: windowSessionId,
        usage: totalUsage,
        turns: cumulativeTurns,
        session_state: finalState,
      };
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, result);
    }
    lastReason = status;
  }

  const result: RunResult = {
    kind: "failure",
    reason: { kind: "ralph_completion_unmet", iterations: maxResets, last_reason: lastReason },
    session_id: lastSessionId,
    usage: totalUsage,
    turns: cumulativeTurns,
    session_state: emptySessionState(),
  };
  cx.stream = onStream;
  return finishCombinator(cx, executor, task, result);
}

/**
 * HillClimbing (#124): GENUINELY recursive optimization loop. Iteration 0 is a
 * pure baseline (no agent turn). Iterations 1.. recurse `runStrategy(c.inner,
 * cx)` to propose a change (a non-ReAct inner — e.g. PlanExecute — really runs
 * its whole loop per iteration), then evaluate the metric (resolved via
 * `resolveMetricEvaluator`, Q2) and keep/revert. Bounded by `max_stagnation` and
 * the turn budget. A `Number.MAX_SAFE_INTEGER` sentinel ⇒ no stagnation cap.
 */
async function runHillClimbingConfig(
  c: HillClimbingConfig,
  cx: ExecutionContext,
): Promise<StrategyOutcome> {
  const executor = requireExecutor(cx);
  if (!("hillBaseline" in executor)) return executor;
  const task = currentTask(cx);
  if (task === undefined) {
    return {
      kind: "failed",
      error: new InvalidConfiguration("no task in ExecutionContext scratch"),
    };
  }
  const sessionId = task.session_id;
  const taskId = task.id;
  const onStream = cx.stream;
  cx.stream = undefined;
  const carried: BudgetSnapshot = { ...cx.scratch.runBudget };
  cx.scratch.runSession = emptySessionState();
  const direction = c.direction;
  const revert = c.revert_on_no_improvement;
  const minDelta = c.min_improvement_delta;
  // `Number.MAX_SAFE_INTEGER` sentinel ⇒ no stagnation cap.
  const maxStagnation = c.max_stagnation !== Number.MAX_SAFE_INTEGER ? c.max_stagnation : undefined;

  // Q2: resolve the metric evaluator from `evaluator`'s key.
  const evaluator = executor.resolveMetricEvaluator(c.evaluator, sessionId);
  if (isRunResult(evaluator)) {
    cx.stream = onStream;
    return finishCombinator(cx, executor, task, evaluator);
  }

  const totalUsage: AggregateUsage = emptyAggregateUsage();
  const rows: ResultsEntry[] = [];
  const spanSeq = { value: 0 };

  // ── Iteration 0: pure baseline (no agent turn).
  const baseline = await executor.hillBaseline(
    evaluator,
    sessionId,
    taskId,
    direction,
    rows,
    spanSeq,
    totalUsage,
    carried.turns,
    cx.signal,
  );
  if (!baseline.ok) {
    cx.stream = onStream;
    return finishCombinator(cx, executor, task, baseline.failure);
  }
  let currentBest = baseline.value;

  let stagnation = 0;
  let iteration = 1;

  // #125: HillClimbing owns a budget scope for its optimization loop. POLICY is
  // the task's global turn ceiling (`total_steps`); this REPLACES the ad-hoc
  // `turnCap` / `carried.turns >= turnCap` gate that #124 used. Behavior is its
  // CONFIGURED `behavior` (default `escalate`) (#129).
  const hcBehavior = c.behavior ?? defaultBudgetBehavior();
  pushBudget(cx, combinatorBudgetPolicy(task.budget.max_turns), hcBehavior, "hill_climbing");

  for (;;) {
    // #125: charge-based budget gate before the iteration's agent turn. A spent
    // `total_steps` cap surfaces this node's OWN typed `budget_exhausted`
    // (partial = best candidate + score), resolving its behavior — replacing the
    // legacy `budget_exceeded` Failure.
    const gateCharge = chargeCurrent(cx, 1);
    if (!gateCharge.ok) {
      const resolution = resolveCurrent(cx);
      // #129: a granted `continue` resets the scope and KEEPS ITERATING the climb
      // IN-PROCESS (do NOT pop; the refreshed allowance lets the next charge
      // pass). NO serialization (AC3). `max_continues` bounds the loop.
      if (resolution === "continue") {
        continue;
      }
      await executor.hillWriteTsv(taskId, rows);
      const partial = hillClimbingPartialJson(currentBest);
      popBudget(cx);
      cx.scratch.task = task;
      cx.stream = onStream;
      switch (resolution) {
        case "fail":
          return promoteBudgetExhausted(gateCharge.error, undefined);
        // #130: under `surface_to_human`, PAUSE with a `budget_exhausted` request
        // instead of propagating up. Combinators offer
        // [continue_with_budget, skip, fail]. (#129: `continue` handled above as
        // an in-process iterate.)
        case "escalate":
          if (executor.escalationMode().kind === "surface_to_human") {
            const waiting = promoteBudgetExhaustedToHuman(
              gateCharge.error,
              partial,
              combinatorEscalationActions(gateCharge.error),
              sessionId,
              task,
              { ...carried },
              carried.turns,
            );
            return recordTerminal(cx, waiting);
          }
          return promoteBudgetExhausted(gateCharge.error, partial);
      }
    }
    const overrun = executor.budgetExceeded(task.budget, carried);
    if (overrun != null) {
      await executor.hillWriteTsv(taskId, rows);
      const result: RunResult = {
        kind: "failure",
        reason: { kind: "budget_exceeded", limit_type: overrun },
        session_id: sessionId,
        usage: totalUsage,
        turns: carried.turns,
      };
      popBudget(cx);
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, result);
    }

    // ── One agent turn proposes a change: recurse `runStrategy(c.inner, cx)`.
    const iterTask: Task = {
      id: task.id,
      instruction: task.instruction,
      session_id: sessionId,
      budget: task.budget,
      loop_strategy: c.inner,
    };
    cx.scratch.task = iterTask;
    const iterState = emptySessionState();
    await executor.appendUserMessage(iterState, task.instruction);
    cx.scratch.runSession = iterState;
    cx.scratch.runBudget = { ...carried };
    const iterOutcome = await runStrategy(c.inner, cx);
    // #125 rule 4/7: a child's `budget_exhausted` reaches HillClimbing as a
    // `StrategyOutcome`, never auto-cascaded. Surface this node's own typed
    // `budget_exhausted` (partial = best candidate + score).
    if (iterOutcome.kind === "budget_exhausted") {
      await executor.hillWriteTsv(taskId, rows);
      const partial = hillClimbingPartialJson(currentBest);
      const err = currentBudgetExhausted(cx, {
        policy: combinatorBudgetPolicy(task.budget.max_turns),
        behavior: hcBehavior,
        stepsTaken: carried.turns,
        continuesUsed: 0,
        phase: "hill_climbing",
      });
      popBudget(cx);
      cx.scratch.task = task;
      cx.stream = onStream;
      return promoteBudgetExhausted(err, partial);
    }
    const turnResult: RunResult = takeChildOverride(cx) ?? {
      kind: "failure",
      reason: { kind: "budget_exceeded", limit_type: "turns" },
      session_id: sessionId,
      usage: emptyAggregateUsage(),
      turns: carried.turns,
      session_state: emptySessionState(),
    };
    foldUsageInto(totalUsage, carried, turnResult);

    // A paused / escalated turn propagates verbatim.
    if (
      turnResult.kind === "waiting_for_human" ||
      turnResult.kind === "consult" ||
      turnResult.kind === "escalate"
    ) {
      await executor.hillWriteTsv(taskId, rows);
      popBudget(cx);
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, turnResult);
    }

    // ── Evaluate the metric + keep/revert decision.
    const { currentBest: best, nonImprovement } = await executor.hillIteration(
      evaluator,
      sessionId,
      taskId,
      iteration,
      direction,
      revert,
      minDelta,
      currentBest,
      rows,
      spanSeq,
      cx.signal,
    );
    currentBest = best;
    if (nonImprovement) {
      stagnation += 1;
    } else {
      stagnation = 0;
    }

    if (maxStagnation != null && stagnation >= maxStagnation) {
      await executor.hillWriteTsv(taskId, rows);
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
      popBudget(cx);
      cx.stream = onStream;
      return finishCombinator(cx, executor, task, result);
    }

    iteration += 1;
  }
}

// ── EscalationMode (HITL-vs-AFK config knob, #120) ──────────────────────────

/**
 * The HITL-vs-AFK escalation knob (PRD goal #7: local vs. prod differ only by
 * config). Selects whether budget escalation surfaces to a human or proceeds
 * autonomously. Stored on {@link HarnessConfig} this slice; consumed in #130.
 *
 * Adjacently tagged on `kind` (`snake_case`) for symmetry with the other
 * harness enums:
 *   - `{ "kind": "surface_to_human" }` — pauses and surfaces to a human (HITL).
 *   - `{ "kind": "autonomous" }` — proceeds autonomously (AFK / prod).
 *
 * No baked-in default value (mirrors the budget-types discipline); the
 * {@link HarnessBuilder} picks an explicit default ({@link surfaceToHuman}).
 * NOT placed on the serialized {@link Task} — there is no fixture for it.
 */
export type EscalationMode = { kind: "surface_to_human" } | { kind: "autonomous" };

export const EscalationModeSchema: z.ZodType<EscalationMode> = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("surface_to_human") }),
  z.object({ kind: z.literal("autonomous") }),
]);

/** Budget escalation pauses and surfaces to a human (HITL). */
export const surfaceToHuman: EscalationMode = { kind: "surface_to_human" };
/** Budget escalation proceeds autonomously (AFK / prod). */
export const autonomous: EscalationMode = { kind: "autonomous" };

// ── HarnessError (registry resolution errors, #120) ─────────────────────────

/** Discriminant tags for {@link HarnessError}. PascalCase to match the Rust
 *  `#[serde(tag = "kind")]` wire shape (see `fixtures/harness/registry_errors.json`). */
export type HarnessErrorKind = "InvalidConfiguration" | "StrategyNotFound" | "UnresolvedHandle";

/**
 * Typed errors surfaced by the {@link ExecutionRegistry} (issue #120) and the
 * harness configuration path. Mirrors the Rust `HarnessError` enum byte-for-byte
 * (`#[serde(tag = "kind")]`, PascalCase variant tags — see
 * `fixtures/harness/registry_errors.json`).
 *
 * Each variant is a class extending `Error` with a discriminant `kind` so
 * call-sites can exhaustively `switch` on `err.kind`.
 */
export abstract class HarnessError extends Error {
  abstract readonly kind: HarnessErrorKind;

  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }

  /** JSON wire shape — matches Rust `#[serde(tag = "kind")]`. */
  abstract toJSON(): Record<string, unknown>;
}

/**
 * Invalid harness configuration (Rust `HarnessError::InvalidConfiguration`).
 * Not part of #120's registry-resolution additions and not fixture-covered; the
 * `detail` is carried for the `Error.message` but the registry never emits this
 * variant.
 */
export class InvalidConfiguration extends HarnessError {
  readonly kind = "InvalidConfiguration" as const;
  constructor(readonly detail: string) {
    super(`invalid configuration: ${detail}`);
  }
  toJSON() {
    return { kind: this.kind, detail: this.detail };
  }
}

/**
 * A `StrategyRef` `custom` key referenced a custom strategy that is not
 * registered in {@link ExecutionRegistry}'s custom map. RECOVERABLE — returned,
 * never thrown across the resolution boundary (same pattern as a missing agent
 * handle). Issue #120.
 */
export class StrategyNotFound extends HarnessError {
  readonly kind = "StrategyNotFound" as const;
  constructor(readonly key: string) {
    super(`custom strategy not found: ${key}`);
  }
  toJSON() {
    return { kind: this.kind, key: this.key };
  }
}

/**
 * A serializable handle ({@link AgentRef}/{@link ToolsetRef}/{@link SchemaRef})
 * referenced an entry absent from the {@link ExecutionRegistry}. The
 * STARTUP-validation error: surfaced before the first turn. Issue #120.
 *
 * The handle category is the `handleKind` field, which serializes as
 * `handle_kind` to avoid colliding with the enum's `kind` discriminant tag.
 */
export class UnresolvedHandle extends HarnessError {
  readonly kind = "UnresolvedHandle" as const;
  constructor(
    readonly handleKind: string,
    readonly key: string,
  ) {
    super(`unresolved ${handleKind} handle: ${key}`);
  }
  toJSON() {
    return { kind: this.kind, handle_kind: this.handleKind, key: this.key };
  }
}

/** Parse a JSON value into a {@link HarnessError}, rejecting unknown variants. */
export function harnessErrorFromJson(value: unknown): HarnessError {
  if (typeof value !== "object" || value === null) {
    throw new TypeError("HarnessError must be a JSON object");
  }
  const obj = value as Record<string, unknown>;
  switch (obj.kind) {
    case "InvalidConfiguration":
      return new InvalidConfiguration(String(obj.detail ?? ""));
    case "StrategyNotFound":
      return new StrategyNotFound(String(obj.key));
    case "UnresolvedHandle":
      return new UnresolvedHandle(String(obj.handle_kind), String(obj.key));
    default:
      throw new TypeError(`unknown HarnessError kind: ${String(obj.kind)}`);
  }
}

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

/**
 * A one-shot task from just an instruction: a fresh {@link SessionId} and a
 * default `react` loop with a `per_loop` budget of 8. Use {@link newTask} when
 * you need to control the session id (e.g. multi-turn) or the loop strategy.
 *
 * Mirrors `Task::simple` in `rust/crates/spore-core/src/harness.rs`.
 */
export function simpleTask(instruction: string): Task {
  return newTask(instruction, SessionId.generate(), reactPerLoop(8));
}

// ============================================================================
// Streaming events emitted by the harness
// ============================================================================

/**
 * The kind of content block a {@link HarnessStreamEvent} `block_start` opens
 * (issue #103, resolved spec decision **Q2**: a single generic frame marker
 * carrying a `BlockKind` rather than typed-per-kind markers).
 */
export type BlockKind = "text" | "reasoning" | "tool_use";

/**
 * Events the harness surfaces on {@link StreamSink}.
 *
 * ## Delta-level streaming (issue #103)
 *
 * The harness maps each raw model `StreamEvent` produced by the agent through
 * `mapModelStreamEvent` into zero or more of the delta/frame variants below,
 * alongside the existing coarse lifecycle events. Resolution notes:
 *
 *   - **Q2**: frame markers are the generic `block_start` / `block_stop`
 *     carrying a {@link BlockKind}, NOT typed-per-kind markers.
 *   - **Q3**: model `message_start` / `message_stop` are DROPPED at the
 *     harness boundary. `turn_start` / `turn_end` already cover this.
 *   - **Q5**: the coarse `tool_call` now also carries the final `args`, and
 *     `tool_result` the result `content`. Both new fields are optional so
 *     pre-#103 serialized events round-trip.
 *
 * Tool lifecycle ordering per call:
 *   `tool_call_start` → `tool_args_delta`* → (`block_stop`) → coarse `tool_call`.
 */
export type HarnessStreamEvent =
  | { kind: "turn_start"; turn: number }
  | { kind: "turn_end"; turn: number }
  | {
      kind: "tool_call";
      call_id: string;
      name: string;
      /** Final, fully-accumulated tool-call arguments (issue #103, Q5). */
      args?: unknown;
    }
  | {
      kind: "tool_result";
      call_id: string;
      is_error: boolean;
      /** The tool result content (issue #103, Q5). */
      content?: string;
    }
  | { kind: "final_response"; content: string }
  | { kind: "budget_warning"; limit_type: BudgetLimitType }
  /**
   * Emitted by the harness loop when it dispatches the `send_message` tool
   * (issue #81 — `SendMessageTool`). The loop surfaces the message content as
   * this prominent event instead of collapsing it into a normal tool result;
   * rendering it prominently is the architect's UI concern. A minimal success
   * tool result is still recorded in history so the loop continues.
   */
  | { kind: "user_message"; content: string }
  // ── Delta-level streaming (issue #103) ──────────────────────────────────
  /** Streamed text fragment (model `content_block_delta`). */
  | { kind: "text_delta"; content: string }
  /** Streamed reasoning/thinking fragment (model `thinking_delta`). Q4. */
  | { kind: "reasoning_delta"; content: string }
  /**
   * Streamed tool-argument JSON fragment (model `tool_use_delta`), correlated
   * to a `call_id` via the open-block index.
   */
  | { kind: "tool_args_delta"; call_id: string; partial_json: string }
  /** A content block opened (Q2). Emitted on the first delta for an index. */
  | { kind: "block_start"; index: number; block: BlockKind }
  /** A content block closed (model `content_block_stop`). Q2. */
  | { kind: "block_stop"; index: number }
  /**
   * A tool-use block opened (issue #103). Emitted so consumers can correlate
   * the subsequent `tool_args_delta` fragments and the final coarse
   * `tool_call` by `call_id`. The `name` may be empty when the underlying
   * model stream does not surface it before args (a documented limitation —
   * the name is recovered on the coarse `tool_call`).
   */
  | { kind: "tool_call_start"; index: number; call_id: string; name: string };

export type StreamSink = (event: HarnessStreamEvent) => void;

/**
 * Per-turn state threaded through `mapModelStreamEvent` (issue #103). Tracks
 * which block indices are open and their kind so `tool_use_delta` /
 * `content_block_stop` events correlate back to a `call_id`, and so each
 * block's `block_start` is emitted exactly once.
 */
export class TurnStreamState {
  /** Open block index → its {@link BlockKind}. */
  readonly openBlocks = new Map<number, BlockKind>();
  /** Tool-use block index → its derived `call_id` (`call_{index}`). */
  readonly toolCalls = new Map<number, string>();

  static callIdFor(index: number): string {
    return `call_${index}`;
  }
}

/**
 * Map one raw model `StreamEvent` to zero or more harness
 * {@link HarnessStreamEvent}s (issue #103), threading {@link TurnStreamState}
 * so blocks and tool calls are correlated across events.
 *
 * Rules:
 *   - Q2: a block's `block_start` is emitted exactly once, the first time a
 *     delta for that index is observed; `content_block_stop` maps to
 *     `block_stop`.
 *   - Q3: model `message_start` / `message_stop` map to nothing (dropped).
 *   - A tool-use block additionally emits `tool_call_start` on open, then each
 *     fragment as `tool_args_delta` keyed by the derived `call_id`.
 */
export function mapModelStreamEvent(
  event: ModelStreamEvent,
  state: TurnStreamState,
): HarnessStreamEvent[] {
  switch (event.type) {
    // Q3: dropped at the harness boundary.
    case "message_start":
    case "message_stop":
      return [];
    case "content_block_delta": {
      const out: HarnessStreamEvent[] = [];
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "text");
        out.push({ kind: "block_start", index: event.index, block: "text" });
      }
      out.push({ kind: "text_delta", content: event.delta });
      return out;
    }
    case "thinking_delta": {
      const out: HarnessStreamEvent[] = [];
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "reasoning");
        out.push({ kind: "block_start", index: event.index, block: "reasoning" });
      }
      out.push({ kind: "reasoning_delta", content: event.delta });
      return out;
    }
    case "tool_use_start": {
      const out: HarnessStreamEvent[] = [];
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "tool_use");
        // Use the real call id from the model; consumers correlate subsequent
        // tool_args_delta by it.
        state.toolCalls.set(event.index, event.id);
        out.push({ kind: "block_start", index: event.index, block: "tool_use" });
        out.push({
          kind: "tool_call_start",
          index: event.index,
          call_id: event.id,
          name: event.name,
        });
      }
      return out;
    }
    case "tool_use_delta": {
      const out: HarnessStreamEvent[] = [];
      // Fallback: if a stream omitted tool_use_start, open the block here with a
      // synthesized id and empty name so args still surface.
      if (!state.openBlocks.has(event.index)) {
        state.openBlocks.set(event.index, "tool_use");
        const callId = TurnStreamState.callIdFor(event.index);
        state.toolCalls.set(event.index, callId);
        out.push({ kind: "block_start", index: event.index, block: "tool_use" });
        out.push({ kind: "tool_call_start", index: event.index, call_id: callId, name: "" });
      }
      const callId = state.toolCalls.get(event.index) ?? TurnStreamState.callIdFor(event.index);
      out.push({ kind: "tool_args_delta", call_id: callId, partial_json: event.partial_json });
      return out;
    }
    case "content_block_stop":
      state.openBlocks.delete(event.index);
      return [{ kind: "block_stop", index: event.index }];
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
}

// ============================================================================
// Forward-declared sibling component types
// ============================================================================

/**
 * Tool Escalation Protocol — the typed channel by which a tool signals the
 * harness to terminate cleanly and pass a *structural* state change up to its
 * caller (issue #80).
 *
 * The harness is a pure intermediary: it never acts on a signal itself. Mode
 * switching, plan approval, and graceful abort are the caller's concern. The
 * harness terminates cleanly, surfaces the signal via the `escalate`
 * {@link RunResult}, and the caller (CLI, chat UI, REST API, parent harness)
 * owns the orchestration. This mirrors the `waiting_for_human` model — the
 * harness does not resume itself either.
 *
 * Variants:
 * - `enter_plan_mode` — agent requests entry into plan mode, carrying
 *   accumulated context as a seed for the planning harness.
 * - `exit_plan_mode` — planning agent's terminal signal, carrying the produced
 *   {@link PlanArtifact} for human approval before an execution harness is
 *   instantiated.
 * - `switch_mode` — agent requests a mode switch; carries the target
 *   {@link Mode} (the EXISTING mode enum — there is no separate `HarnessMode`).
 * - `abort` — agent requests a graceful, intentional stop with a reason.
 *   Distinct from a `HaltReason` `agent_error` — it surfaces as an `escalate`
 *   {@link RunResult}, NOT a `failure`.
 *
 * Wire format: tagged on `kind`, `snake_case`, byte-identical across the four
 * language implementations. Round-tripped by
 * `fixtures/harness/escalation_signals.json`.
 */
export const HarnessSignalSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("enter_plan_mode"), context: z.string() }),
  z.object({
    kind: z.literal("exit_plan_mode"),
    plan: z.object({ tasks: z.array(z.string()), rationale: z.string() }),
  }),
  z.object({
    kind: z.literal("switch_mode"),
    // `"yolo"` is intentionally absent — it is a dangerous-only mode (issue
    // #34) and cannot be named through the default build, mirroring the Rust
    // `SwitchMode { mode: Mode }` where `Mode::Yolo` does not exist without the
    // `dangerous` feature.
    mode: z.enum(["always_ask", "auto_edit", "plan", "safe_auto"]),
  }),
  z.object({ kind: z.literal("abort"), reason: z.string() }),
]);
export type HarnessSignal =
  | { kind: "enter_plan_mode"; context: string }
  | { kind: "exit_plan_mode"; plan: PlanArtifact }
  | { kind: "switch_mode"; mode: Mode }
  | { kind: "abort"; reason: string };

/**
 * Tool dispatch output. Full type lives in issue #4/#5; this covers loop routing.
 *
 * Prefer the {@link toolOutput} constructors over the object literals — they
 * spell out the common cases (`success` / recoverable `error` / `fatal`) and
 * document the field semantics below in one place.
 *
 *   - `truncated` (on `success`) — `true` ONLY when the tool itself clipped its
 *     output to fit an inline budget (large outputs routed through
 *     {@link SandboxProvider.handleLargeOutput} set this). Plain tool authors
 *     should leave it `false` (omit it) — use {@link toolOutput.success}.
 *   - `recoverable` (on `error`) — `true` if the agent may sensibly retry or
 *     adapt: the loop appends the error as a tool result and continues. `false`
 *     halts the run. Most tool failures are recoverable — prefer
 *     {@link toolOutput.error}; reach for {@link toolOutput.fatal} only when
 *     continuing is pointless.
 */
export type ToolOutput =
  | { kind: "success"; content: string; truncated?: boolean }
  | { kind: "error"; message: string; recoverable: boolean }
  | { kind: "waiting_for_human"; child_state: ChildPausedState; request: HumanRequest }
  /**
   * Tool requests a structural state change from the harness's parent (#80).
   * The harness terminates cleanly and passes the signal to the caller via the
   * `escalate` {@link RunResult}. NOT appended to message history.
   */
  | { kind: "escalate"; signal: HarnessSignal }
  /**
   * Tool asks the user a clarifying question (issue #81, Q4b —
   * `AskUserQuestionTool`). UNLIKE the subagent `waiting_for_human` path there
   * is NO {@link ChildPausedState}: the loop builds a {@link PausedState}
   * directly with `human_request` set to {@link HumanRequest} `clarification`,
   * preserves the clarifying call as the head of `pending_tool_calls`, and
   * returns a `waiting_for_human` {@link RunResult}. On resume the human's
   * answer text is injected as the tool RESULT for that clarifying call.
   */
  | { kind: "awaiting_clarification"; question: string; options?: string[] }
  /**
   * Mid-loop consult signal (issue #114). A worker-side tool returns it with
   * `child_state` ABSENT; the worker harness pauses and returns
   * {@link RunResult} `consult` with the consult call preserved as the head of
   * `pending_tool_calls` and `human_request` absent. At the subagent boundary,
   * {@link "@spore/tools".SubagentTool} populates `child_state` — but in the A1
   * mediation seam it consumes the signal itself rather than bubbling it, so a
   * parent orchestrator never observes this variant on the happy path. Mirrors
   * `waiting_for_human`: the optional {@link ChildPausedState} keeps the common
   * (worker-emitted) case cheap. NOT appended to message history.
   */
  | { kind: "consult"; child_state?: ChildPausedState; request: ConsultRequest };

/**
 * Ergonomic constructors for the common {@link ToolOutput} cases. Mirrors Rust's
 * `ToolOutput::success` / `error` / `fatal` — see the field semantics on
 * {@link ToolOutput}.
 */
export const toolOutput = {
  /**
   * A successful, non-truncated result — the common case for a tool that
   * returns its full output. Saves spelling out `truncated: false`.
   */
  success(content: string): ToolOutput {
    return { kind: "success", content, truncated: false };
  },

  /**
   * A **recoverable** error: the harness loop appends it as a tool result and
   * lets the agent adapt or retry. The right default for almost every tool
   * failure (bad arguments, missing file, transient I/O).
   */
  error(message: string): ToolOutput {
    return { kind: "error", message, recoverable: true };
  },

  /**
   * A **fatal** error: continuing is pointless, so the run halts. Reserve for
   * genuinely unrecoverable conditions; prefer {@link toolOutput.error} when the
   * agent could reasonably do something different next turn.
   */
  fatal(message: string): ToolOutput {
    return { kind: "error", message, recoverable: false };
  },

  /**
   * A **worker-side consult** signal (issue #114): the tool asks for mid-loop
   * help. `child_state` is omitted — the harness loop builds the
   * {@link RunResult} `consult` pause; only {@link "@spore/tools".SubagentTool}
   * populates `child_state` at the boundary.
   */
  consult(request: ConsultRequest): ToolOutput {
    return { kind: "consult", request };
  },
} as const;

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

/**
 * Discriminated isolation mode (default build).
 *
 * `{ kind: "none" }` (no path enforcement) is a named safety footgun and is
 * **not** part of this type. It is reachable only through the dangerous opt-in
 * entry point (`@spore/core/dangerous`), which re-exposes it as
 * {@link DangerousIsolationMode}. This mirrors the Rust `dangerous` Cargo
 * feature that removes `IsolationMode::None` from the default build (issue #34).
 * The wire tag for the dangerous mode stays `"none"`.
 */
export type IsolationMode =
  | { kind: "workspace_scoped" }
  | { kind: "bubblewrap"; profile: BwrapProfile }
  | { kind: "docker"; image: string; network: NetworkPolicy };

/**
 * The dangerous-only isolation mode: no isolation, no path enforcement. Its
 * wire value is `{ kind: "none" }`, but the type is **branded**: a value can
 * only be minted by the dangerous opt-in entry point (`@spore/core/dangerous`).
 * A default-build caller cannot forge it by writing `{ kind: "none" }`, so the
 * dangerous sandbox constructor that consumes it is unreachable without
 * importing the dangerous module. Idiomatic TS analogue of the Rust `dangerous`
 * Cargo feature gating `IsolationMode::None` (issue #34).
 */
export type DangerousIsolationMode = { kind: "none" } & {
  readonly __sporeDangerous: unique symbol;
};

/**
 * Internal union of every isolation mode, including the dangerous
 * {@link DangerousIsolationMode}. Used by the sandbox switch bodies and by the
 * dangerous entry point. NOT part of the default public API — default callers
 * use {@link IsolationMode}, which cannot name the dangerous mode.
 */
export type AnyIsolationMode = IsolationMode | DangerousIsolationMode;

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

  /**
   * Current isolation mode. Returns {@link AnyIsolationMode} so a sandbox
   * constructed through the dangerous opt-in can report `{ kind: "none" }`;
   * default-built sandboxes only ever return a safe {@link IsolationMode}.
   */
  isolationMode?(): AnyIsolationMode;

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

/**
 * The operator's choice on a budget-exhaustion {@link HumanRequest}
 * `budget_exhausted` pause (#130). Tagged on `kind`, snake_case. The
 * data-carrying variant uses a NAMED `steps` field (mirroring {@link BudgetPolicy}'s
 * `value` convention — fork A); the wire form is
 * `{"kind":"continue_with_budget","steps":N}`.
 *
 *   - `continue_with_budget` — grant `steps` ADDITIONAL steps to the node's
 *     budget scope and resume from the checkpoint.
 *   - `skip` — mark the current task skipped; a combinator's outer loop advances
 *     to its sibling tasks. Offered only by combinators (fork C).
 *   - `fail` — abort the node and propagate `budget_exceeded` (partial
 *     discarded, mirroring the `fail` resolution contract).
 */
export const EscalationActionSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("continue_with_budget"), steps: u32 }),
  z.object({ kind: z.literal("skip") }),
  z.object({ kind: z.literal("fail") }),
]);
export type EscalationAction = z.infer<typeof EscalationActionSchema>;

export const HumanRequestSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("tool_approval"),
    calls: z.array(ToolCallSchema),
    risk_level: RiskLevelSchema,
  }),
  z.object({
    kind: z.literal("clarification"),
    question: z.string(),
    // Optional (#81, Q4b): multiple-choice options carried with the
    // clarification. Absent on old `clarification` blobs (back-compat
    // deserialization) and on free-form questions.
    options: z.array(z.string()).nullable().optional(),
  }),
  z.object({ kind: z.literal("review"), content: z.string() }),
  // A node's budget scope resolved to `escalate` under `surface_to_human`
  // (#130): the run pauses and surfaces the exhaustion to the operator. Carries
  // the node's `phase`, its `policy`, the `steps_taken` / `continues_used`
  // counters (so `resumeInner` can reconstruct the node's budget context — fork
  // E), any `partial_output` produced before exhaustion, and the ADVISORY
  // `available_actions` the author offers (fork C/D). The operator answers with
  // {@link HumanResponse} `escalate`. Existing variants are UNCHANGED.
  z.object({
    kind: z.literal("budget_exhausted"),
    phase: z.string(),
    policy: BudgetPolicySchema,
    steps_taken: u32,
    continues_used: u32,
    // Optional — omitted from the wire when absent (a `fail`-resolved partial is
    // discarded by contract).
    partial_output: z.string().nullable().optional(),
    available_actions: z.array(EscalationActionSchema),
  }),
]);
export type HumanRequest = z.infer<typeof HumanRequestSchema>;

export type HumanResponse =
  | { kind: "allow" }
  | { kind: "allow_with_modification"; calls: ToolCall[] }
  | { kind: "deny"; reason: string }
  | { kind: "halt" }
  | { kind: "answer"; text: string }
  | { kind: "approve_with_feedback"; feedback: string }
  | { kind: "reject"; reason: string }
  /**
   * The operator's resolution of a {@link HumanRequest} `budget_exhausted`
   * pause (#130): the chosen {@link EscalationAction}. Distinct from
   * `allow`/`halt`/`deny` so the budget-escalation resume path is unambiguous
   * (fork B).
   */
  | { kind: "escalate"; action: EscalationAction };

export const HumanResponseSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("allow") }),
  z.object({ kind: z.literal("allow_with_modification"), calls: z.array(ToolCallSchema) }),
  z.object({ kind: z.literal("deny"), reason: z.string() }),
  z.object({ kind: z.literal("halt") }),
  z.object({ kind: z.literal("answer"), text: z.string() }),
  z.object({ kind: z.literal("approve_with_feedback"), feedback: z.string() }),
  z.object({ kind: z.literal("reject"), reason: z.string() }),
  z.object({ kind: z.literal("escalate"), action: EscalationActionSchema }),
]) satisfies z.ZodType<HumanResponse>;

// ============================================================================
// Mid-loop consult primitive (issue #114)
// ============================================================================
//
// A third pause/resume path that stops at the ORCHESTRATOR instead of bubbling
// to the human. A worker-side tool returns {@link ToolOutput} `consult`; the
// worker harness pauses and returns {@link RunResult} `consult`. At the
// subagent boundary, {@link "@spore/tools".SubagentTool} mediates it
// deterministically (seam A1): it routes by `kind` to a registered
// {@link ConsultHandlerEntry}, enforces the per-kind budget, runs the handler
// (the orchestrator's DIRECT child — depth-1), and resumes the worker via
// {@link "./interface.js".Harness.resumeConsult} with a {@link ConsultResponse}.
// The orchestrator's model is never involved.
//
// All wire shapes are byte-identical to `fixtures/harness/consult.json`:
// `kind`-tagged unions in snake_case; `ConsultRequest` has NO defaults (a
// partial request fails to parse rather than silently defaulting).

/**
 * The worker's free-form ask when it pauses mid-loop to consult a
 * parent-spawned helper (issue #114). `kind` selects the handler; `situation`,
 * `attempts`, and `question` carry the context the handler needs. All fields
 * are REQUIRED — there are deliberately no schema defaults.
 */
export const ConsultRequestSchema = z.object({
  /** Routing key — selects the {@link ConsultHandlerEntry}. */
  kind: z.string(),
  /** Free-form description of where the worker is stuck. */
  situation: z.string(),
  /** How many times the worker has already tried (advisory; the per-kind budget
   *  is enforced independently by the mediator). */
  attempts: z.number().int().nonnegative(),
  /** The concrete question the worker wants answered. */
  question: z.string(),
});
export type ConsultRequest = z.infer<typeof ConsultRequestSchema>;

/**
 * The resume input handed back to a paused worker after a consult (issue #114).
 * Parallel to {@link HumanResponse}; tagged on `kind`, snake_case.
 *
 *   - `answer` — the handler produced an answer; `text` is injected as the tool
 *     RESULT for the pending consult call.
 *   - `budget_exhausted` — the per-kind budget is exhausted under a `soft_fail`
 *     overflow policy: the worker is resumed with this message and finishes
 *     with what it has.
 */
export const ConsultResponseSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("answer"), text: z.string() }),
  z.object({ kind: z.literal("budget_exhausted"), message: z.string() }),
]);
export type ConsultResponse = z.infer<typeof ConsultResponseSchema>;

/**
 * Per-kind budget-overflow behaviour (issue #114). Tagged on `kind`, snake_case.
 *
 *   - `soft_fail` — resume the worker with {@link ConsultResponse} `budget_exhausted`
 *     so it finishes without further help.
 *   - `escalate_to_human` — convert the over-budget consult into a
 *     {@link RunResult} `waiting_for_human` so the host decides.
 */
export const ConsultOverflowPolicySchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("soft_fail") }),
  z.object({ kind: z.literal("escalate_to_human") }),
]);
export type ConsultOverflowPolicy = z.infer<typeof ConsultOverflowPolicySchema>;

/**
 * A registered consult handler (issue #114): the helper {@link Harness} to run,
 * the per-kind budget (max consults of this kind before overflow), and the
 * overflow policy. Held by `kind` in {@link HarnessConfig.consultHandlers}.
 *
 * The `handler` is run by {@link "@spore/tools".SubagentTool} as the
 * ORCHESTRATOR's direct child (depth-1), never nested under the worker.
 */
export interface ConsultHandlerEntry {
  /** The helper harness run on the {@link ConsultRequest}. */
  handler: Harness;
  /** Max number of consults of this kind before the overflow policy applies. */
  budget: number;
  /** What to do once the budget is exhausted. */
  overflow: ConsultOverflowPolicy;
}

/** Per-kind consult handlers (issue #114), keyed by {@link ConsultRequest.kind}. */
export type ConsultHandlerMap = Map<string, ConsultHandlerEntry>;

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
  /**
   * Optional (#80): a `waiting_for_human` pause always sets it; an `escalate`
   * pause omits it (the escalation carries no human request). Old
   * `waiting_for_human` blobs still deserialize (field present); escalation
   * blobs omit it (undefined/null on the wire).
   */
  human_request?: HumanRequest;
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
  /**
   * Optional (#80): a `waiting_for_human` pause always sets it; an `escalate`
   * pause omits it (the escalation carries no human request). The `escalate`
   * {@link RunResult} preserves the full {@link PausedState} with this field
   * absent.
   */
  human_request?: HumanRequest;
  task: Task;
  budget_used: BudgetSnapshot;
  child_state: ChildPausedState | null;
}

/**
 * The SHARED durable checkpoint round-trip (#129, AC1). Both cross-process
 * `continue` (a `HumanRequest::BudgetExhausted` pause whose request carries
 * `continues_used`) and Ralph's pause-propagation hand the SAME
 * {@link PausedState} to the caller for persistence; this is the one seam they
 * share. It is JUST the `PausedState` serialize/deserialize — NOT a unification
 * of their CONTEXT policies (Q2): a `continue` resumes preserving
 * `session_state.messages`; Ralph re-seeds a fresh window from its filesystem
 * `.spore/progress.json` checkpoint, which stays Ralph-specific.
 *
 * Produces the durable blob the caller persists. Mirrors Rust
 * `PausedState::serialize_checkpoint`.
 */
export function serializeCheckpoint(state: PausedState): string {
  return JSON.stringify(state);
}

/**
 * Restore a {@link PausedState} from a durable checkpoint blob (#129, AC1). The
 * resume side of {@link serializeCheckpoint}. Re-hydrates the {@link SessionId} /
 * {@link TaskId} newtypes and re-validates the embedded {@link Task} (and its
 * strategy tree) and {@link HumanRequest} via their zod schemas, so the restored
 * value is structurally EQUAL to the original. Mirrors Rust
 * `PausedState::load_checkpoint`.
 */
export function loadCheckpoint(blob: string): PausedState {
  const raw = JSON.parse(blob) as Record<string, unknown>;
  const hydrateChild = (c: Record<string, unknown>): ChildPausedState => ({
    session_id: new SessionId(c.session_id as string),
    task_id: new TaskId(c.task_id as string),
    turn_number: c.turn_number as number,
    session_state: SessionStateSchema.parse(c.session_state),
    pending_tool_calls: (c.pending_tool_calls as ToolCall[]) ?? [],
    approved_results: (c.approved_results as ToolResultRecord[]) ?? [],
    human_request: c.human_request == null ? undefined : HumanRequestSchema.parse(c.human_request),
    task: TaskSchema.parse(c.task),
    budget_used: BudgetSnapshotSchema.parse(c.budget_used),
    parent_tool_call_id: c.parent_tool_call_id as string,
  });
  return {
    session_id: new SessionId(raw.session_id as string),
    task_id: new TaskId(raw.task_id as string),
    turn_number: raw.turn_number as number,
    session_state: SessionStateSchema.parse(raw.session_state),
    pending_tool_calls: (raw.pending_tool_calls as ToolCall[]) ?? [],
    approved_results: (raw.approved_results as ToolResultRecord[]) ?? [],
    human_request:
      raw.human_request == null ? undefined : HumanRequestSchema.parse(raw.human_request),
    task: TaskSchema.parse(raw.task),
    budget_used: BudgetSnapshotSchema.parse(raw.budget_used),
    child_state:
      raw.child_state == null ? null : hydrateChild(raw.child_state as Record<string, unknown>),
  };
}

// ============================================================================
// Halt reasons / RunResult
// ============================================================================

export type HaltReason =
  | { kind: "budget_exceeded"; limit_type: BudgetLimitType }
  | { kind: "termination_policy_halt"; reason: string }
  | { kind: "middleware_halt"; hook: HookPoint; reason: string }
  | { kind: "agent_error"; error: AgentError }
  /**
   * A {@link ContextError} surfaced by the {@link ContextManager} during
   * assembly halts the run (e.g. a cache-hash mismatch — both Block 1
   * `"static"` and, as of #32, Block 2 `"per_session"` halt mid-session). This
   * is the routing type; mirrors `agent_error`. The live {@link StandardHarness}
   * loop does not yet trigger it because its placeholder `ContextManager`
   * assemble is infallible pending the #7 migration.
   */
  | { kind: "context_error"; error: ContextError }
  | { kind: "sandbox_violation"; violation: SandboxViolation }
  | { kind: "unrecoverable_tool_error"; tool: string; error: string }
  | { kind: "human_halted" }
  | { kind: "stagnation_limit_reached"; iterations: number; best_metric: number }
  | { kind: "strategy_not_yet_implemented"; strategy: string }
  /**
   * Returned by {@link StandardHarness} for the `plan_execute` strategy (issue
   * #59) when the accepted plan parsed into an EMPTY task list (`tasks: []`).
   * Per Q3, an empty plan is a failure — the run does NOT silently succeed.
   */
  | { kind: "empty_plan" }
  /**
   * Returned by {@link StandardHarness} for the `plan_execute` strategy (issue
   * #59) when an execute step's bounded ReAct sub-loop errored or the agent
   * returned a blocked/failed outcome (Q5). A plan is a dependency chain by
   * assumption, so the whole run aborts at the failing step — execution does NOT
   * continue to the next task. Carries the failing step's positional index, its
   * instruction, and a human-readable reason derived from the underlying
   * {@link HaltReason}.
   */
  | { kind: "step_failed"; task_index: number; task: string; reason: string }
  /**
   * The `plan_execute` plan phase (issue #70) failed before producing an
   * artifact: the planner's response was unparseable, the planner requested a
   * tool call in the one-shot turn, or the agent returned an error. Carries the
   * underlying {@link "../plan/index.js".PlanPhaseError} detail.
   */
  | { kind: "plan_phase_failed"; error: PlanPhaseErrorKind }
  /**
   * Returned by {@link StandardHarness} for the `self_verifying` strategy
   * (issue #61, D4) when the build↔evaluate loop ran out of the verifier's
   * `maxIterations` round-trips without an explicit `passed` verdict. A RUNTIME
   * limit — the work was attempted in good faith but never verified; a caller
   * might retry with a different task decomposition. Carries the number of
   * round-trips run and the last failure reason the verifier gave. PEER to
   * {@link self_verify_misconfigured} (NOT a sub-case of it).
   */
  | { kind: "self_verify_exhausted"; iterations: number; last_reason: string }
  /**
   * Returned by {@link StandardHarness} for the `self_verifying` strategy
   * (issue #61, D4) when the strategy cannot run because it is misconfigured —
   * e.g. `config.verifier` is absent. Likely a BUILD-TIME bug in the caller's
   * wiring. Surfaced as a typed halt, NOT a throw. PEER to
   * {@link self_verify_exhausted} (NOT a sub-case of it).
   */
  | { kind: "self_verify_misconfigured"; reason: string }
  /**
   * Returned by {@link StandardHarness} for the `ralph` strategy (issue #58, B3)
   * when the multi-context-window continuation loop reached its `maxResets`
   * outer-loop cap with tasks still incomplete (the external completion check —
   * `.spore/progress.json` + `.spore/feature_list.json` — never passed). Carries
   * the number of context windows run and the last incompletion reason. PEER to
   * {@link self_verify_exhausted}.
   */
  | { kind: "ralph_completion_unmet"; iterations: number; last_reason: string }
  /**
   * Returned by {@link StandardHarness} for the `hill_climbing` strategy
   * (issue #60, Decision 6) when the strategy cannot run because it is
   * misconfigured — `config.metricEvaluator` is absent — OR the iteration-0
   * baseline evaluation itself errored (Decision 7: there is no `current_best`
   * to climb from, so a failed baseline is a misconfiguration of the experiment,
   * NOT a stagnation increment). Likely a BUILD-TIME bug in the caller's wiring.
   * Surfaced as a typed halt, NOT a throw.
   */
  | { kind: "hill_climbing_misconfigured"; reason: string }
  /**
   * Returned by the PlanExecute DAG executor (#126, decision A) at drain: a task
   * failed terminally (unrecoverable error or `BudgetExhausted` resolving to
   * `fail`) and its transitive dependents were cascade-`blocked`, while unrelated
   * branches still completed. The run as a whole is a `failure`, but the full
   * partition is reported: which tasks `completed`, which were `blocked` by the
   * cascade, the `failed_task` that triggered it, and the human-readable `reason`.
   * (A run where EVERY task completes is a `success`, as before.)
   */
  | {
      kind: "tasks_blocked_by_failure";
      completed: number[];
      blocked: number[];
      failed_task: number;
      reason: string;
    }
  /**
   * Returned by the PlanExecute DAG executor (#126, AC5) when the persisted task
   * graph contains a directed cycle, re-checked at EXECUTE ENTRY as defense in
   * depth (`add_task` already rejects cycles, but the `task_list` tool path could
   * in principle persist a cyclic graph out of band). No task is run. Carries a
   * human-readable description.
   */
  | { kind: "task_graph_cycle"; reason: string }
  /**
   * Returned by {@link StandardHarness} when {@link ExecutionRegistry.validate}
   * fails at run entry (issue #120): a handle referenced by the task's strategy
   * tree is unresolved against the configured {@link ExecutionRegistry}, or a
   * `StrategyRef` `custom` key is missing. A STARTUP error surfaced before the
   * first turn. Carries the underlying {@link HarnessError}. Validation only
   * fires when the registry is populated, so legacy callers (Option B, the
   * deprecated single-collaborator fields) are unaffected.
   */
  | { kind: "configuration_error"; error: HarnessError };

export type RunResult =
  | {
      kind: "success";
      output: string;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
      /**
       * The post-run conversation history (issue #102). Carries the full
       * {@link SessionState.messages} the loop produced — assistant tool-call
       * turns and tool-result turns included — so an in-process caller can
       * resume losslessly via {@link HarnessRunOptions.session_state} without
       * reconstructing history from `output`. Optional so old serialized
       * `RunResult` blobs (and other languages mid-migration) still parse — the
       * TS analogue of Rust's `#[serde(default)]`; read it via
       * {@link runResultSessionState}, which defaults absence to an empty
       * {@link SessionState}.
       */
      session_state?: SessionState;
    }
  | {
      kind: "failure";
      reason: HaltReason;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
      /**
       * The post-run conversation history at the point of failure (issue #102).
       * Same contract as on the `success` variant: lossless resume, optional for
       * back-compat. Read via {@link runResultSessionState}.
       */
      session_state?: SessionState;
    }
  | { kind: "waiting_for_human"; state: PausedState; request: HumanRequest }
  /**
   * Harness terminated cleanly due to a tool escalation signal (#80). The
   * caller handles the signal and decides whether to resume, instantiate a new
   * harness, or present UI. `state` preserves the full {@link PausedState}
   * (with `human_request` absent) so the original harness can be resumed; the
   * signal is NOT stored in `state`, so it is discarded on resume — the harness
   * never re-acts on it.
   */
  | {
      kind: "escalate";
      signal: HarnessSignal;
      state: PausedState;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
    }
  /**
   * Worker paused mid-loop to consult a parent-spawned helper (issue #114).
   * Sibling of `waiting_for_human`, but it stops at the ORCHESTRATOR (via the
   * SubagentTool A1 mediation), not the human. `state` preserves the full
   * {@link PausedState} with `human_request` absent and the consult call as the
   * head of `pending_tool_calls`, so `harness.resumeConsult(state, response)`
   * continues the worker. With no consult handlers registered, a standalone
   * worker simply returns this unchanged to its caller (R6 graceful degradation).
   */
  | {
      kind: "consult";
      request: ConsultRequest;
      state: PausedState;
      session_id: SessionId;
      usage: AggregateUsage;
      turns: number;
    };

/**
 * The post-run {@link SessionState} carried by a terminal {@link RunResult}
 * (issue #102). For `success`/`failure` returns the carried history, defaulting
 * absence to an empty {@link SessionState} (back-compat with pre-#102 blobs). For
 * `waiting_for_human`/`escalate` returns the {@link PausedState.session_state}
 * those variants already carry. The one place to read post-run history from any
 * terminal/paused result.
 */
export function runResultSessionState(result: RunResult): SessionState {
  switch (result.kind) {
    case "success":
    case "failure":
      return result.session_state ?? emptySessionState();
    case "waiting_for_human":
    case "escalate":
    case "consult":
      return result.state.session_state;
  }
}

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
