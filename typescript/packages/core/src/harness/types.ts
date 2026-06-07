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
import type { Context } from "../agent/types.js";
import type { AgentError } from "../agent/errors.js";
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
}

/** SelfVerifying combinator: run `inner`, then judge it against `evaluator`. */
export interface SelfVerifyingConfig {
  kind: "self_verifying";
  inner: LoopStrategy;
  evaluator: SchemaRef;
}

/**
 * Ralph combinator: re-run `inner` under a fixed `agent` across context-window
 * resets.
 */
export interface RalphConfig {
  kind: "ralph";
  inner: LoopStrategy;
  agent: AgentRef;
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
      agent: AgentRefSchema,
      toolset: ToolsetRefSchema,
      output: SchemaRefSchema.optional(),
    }),
    z.object({
      kind: z.literal("plan_execute"),
      plan: LoopStrategySchema,
      execute: LoopStrategySchema,
      plan_model: ModelConfigSchema.optional(),
    }),
    z.object({
      kind: z.literal("self_verifying"),
      inner: LoopStrategySchema,
      evaluator: SchemaRefSchema,
    }),
    z.object({
      kind: z.literal("ralph"),
      inner: LoopStrategySchema,
      agent: AgentRefSchema,
    }),
    z.object({
      kind: z.literal("hill_climbing"),
      inner: LoopStrategySchema,
      direction: HillClimbingDirectionSchema,
      max_stagnation: u32,
      revert_on_no_improvement: z.boolean(),
      min_improvement_delta: z.number(),
      evaluator: AgentRefSchema,
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
      const out: Record<string, unknown> = {
        kind: "react",
        budget: budgetPolicyToJson(strategy.budget),
        agent: strategy.agent,
        toolset: strategy.toolset,
      };
      if (strategy.output !== undefined) out.output = strategy.output;
      return out;
    }
    case "plan_execute": {
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
      return out;
    }
    case "self_verifying":
      return {
        kind: "self_verifying",
        inner: loopStrategyToJson(strategy.inner),
        evaluator: strategy.evaluator,
      };
    case "ralph":
      return {
        kind: "ralph",
        inner: loopStrategyToJson(strategy.inner),
        agent: strategy.agent,
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

// ── RunStrategy composition seam (#119) ─────────────────────────────────────

/**
 * Placeholder for #119; full shape owned by #123 (StrategyOutcome +
 * ExecutionContext runtime scaffold). Intentionally near-empty — do NOT
 * pre-build #123's budget-stack / aggregate-usage scaffolding here.
 */
export interface ExecutionContext {
  readonly __executionContext?: never;
}

/**
 * Placeholder for #119; full shape owned by #123. The stub strategy run bodies
 * return `{ kind: "pending" }` so the seam exists without a throw; per-variant
 * outcomes land with the executor slice (#124).
 */
export type StrategyOutcome = { kind: "pending" };

/**
 * The runtime composition seam: every strategy node knows how to run itself
 * given an {@link ExecutionContext}. The TS runtime-polymorphism idiom is one
 * `run(cx)` method whose single dispatch is {@link runStrategy}.
 */
export interface RunStrategy {
  run(cx: ExecutionContext): Promise<StrategyOutcome>;
}

/**
 * The single dispatch site for the strategy tree (#119). One `switch` over the
 * `kind` discriminant delegates to per-config run logic. Each per-config body
 * is a STUB returning {@link StrategyOutcome} `pending` (it does NOT throw);
 * real bodies land in #124.
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

// Per-config run STUBS (#124 lands the real bodies). They never throw.
async function runReactConfig(_c: ReactConfig, _cx: ExecutionContext): Promise<StrategyOutcome> {
  return { kind: "pending" };
}
async function runPlanExecuteConfig(
  _c: PlanExecuteConfig,
  _cx: ExecutionContext,
): Promise<StrategyOutcome> {
  return { kind: "pending" };
}
async function runSelfVerifyingConfig(
  _c: SelfVerifyingConfig,
  _cx: ExecutionContext,
): Promise<StrategyOutcome> {
  return { kind: "pending" };
}
async function runRalphConfig(_c: RalphConfig, _cx: ExecutionContext): Promise<StrategyOutcome> {
  return { kind: "pending" };
}
async function runHillClimbingConfig(
  _c: HillClimbingConfig,
  _cx: ExecutionContext,
): Promise<StrategyOutcome> {
  return { kind: "pending" };
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
