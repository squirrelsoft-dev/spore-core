/**
 * TerminationPolicy — canonical types (spore-core issue #13).
 *
 * Mirrors `rust/crates/spore-core/src/termination.rs` byte-for-byte on the
 * wire: tagged unions use a `kind` discriminator in `snake_case`; struct
 * fields are `snake_case`. Same fixture, same outcome — see
 * `/fixtures/README.md`.
 *
 * After every turn, the harness asks the policy what to do next:
 *   - Continue
 *   - HaltSuccess { summary }
 *   - HaltFailure { reason }                       (typed reason)
 *   - HaltBudgetExceeded { limit_type, used, limit } (hard stop)
 *
 * Rules enforced by {@link StandardTerminationPolicy}:
 *   1. Budget is checked first — hard stop regardless of `agent_claims_done`.
 *   2. If `!agent_claims_done`, the policy returns `continue`.
 *   3. Any sensor result with outcome `halt` becomes an
 *      `unrecoverable_sensor_halt` failure.
 *   4. The injected {@link CompletionCheck} decides success vs continue:
 *        - `null` (complete)  ⇒ `halt_success` with `agent_response` summary
 *          (empty string if absent).
 *        - non-null (reason)  ⇒ `continue` — the harness re-injects the
 *          reason into the next turn's context.
 *
 * The `human_halted` failure variant is reserved for the harness — the
 * policy itself never produces it. It is included on
 * {@link TerminationFailureReason} for completeness of the wire schema.
 */

import type { AgentError } from "../agent/errors.js";
import type {
  BudgetLimits,
  BudgetLimitType,
  BudgetSnapshot,
  SessionId,
  SessionState,
  TaskId,
} from "../harness/types.js";
import type { HookPoint } from "../middleware/types.js";
import type { SensorId, SensorResult } from "../sensor/types.js";

// ============================================================================
// BudgetValue
// ============================================================================

/**
 * The measured budget quantity at the moment of halt — carried on
 * {@link TerminationDecision} variant `halt_budget_exceeded` so observers
 * can compute overshoot against the configured limit. Duration is wire-
 * encoded as whole seconds.
 */
export type BudgetValue =
  | { kind: "turns"; value: number }
  | { kind: "tokens"; value: number }
  | { kind: "duration"; value: number }
  | { kind: "usd"; value: number };

export const BudgetValue = {
  turns(v: number): BudgetValue {
    return { kind: "turns", value: v };
  },
  tokens(v: number): BudgetValue {
    return { kind: "tokens", value: v };
  },
  /** `seconds` — whole-second resolution, mirroring the Rust `Duration` wire form. */
  duration(seconds: number): BudgetValue {
    return { kind: "duration", value: seconds };
  },
  usd(v: number): BudgetValue {
    return { kind: "usd", value: v };
  },
} as const;

// ============================================================================
// SessionStateSnapshot
// ============================================================================

/**
 * Read-only snapshot of session state handed to {@link CompletionCheck.check}.
 * Wraps {@link SessionState} so the check can identify the source session
 * and task — completion checks frequently key into per-session scratchpads.
 */
export interface SessionStateSnapshot {
  session_id: SessionId;
  task_id: TaskId;
  state: SessionState;
}

export function newSessionStateSnapshot(
  sessionId: SessionId,
  taskId: TaskId,
  state: SessionState,
): SessionStateSnapshot {
  return { session_id: sessionId, task_id: taskId, state };
}

// ============================================================================
// TerminationFailureReason
// ============================================================================

/**
 * Typed reason carried by `halt_failure`. Wire schema mirrors the Rust
 * `#[serde(tag = "kind", rename_all = "snake_case")]` enum byte-for-byte.
 *
 * `human_halted` is set by the harness directly when `HumanResponse::Halt`
 * is received; the {@link TerminationPolicy} itself never produces it.
 */
export type TerminationFailureReason =
  | { kind: "completion_check_failed"; detail: string }
  | { kind: "max_retries_exhausted"; tool: string; attempts: number }
  | { kind: "unrecoverable_sensor_halt"; sensor_id: SensorId; detail: string }
  | { kind: "middleware_halt"; hook: HookPoint; reason: string }
  | { kind: "agent_error"; error: AgentError }
  | { kind: "policy_violation"; detail: string }
  | { kind: "human_halted" };

// ============================================================================
// TerminationDecision
// ============================================================================

export type TerminationDecision =
  | { kind: "continue" }
  | { kind: "halt_success"; summary: string }
  | { kind: "halt_failure"; reason: TerminationFailureReason }
  | {
      kind: "halt_budget_exceeded";
      limit_type: BudgetLimitType;
      used: BudgetValue;
      limit: BudgetValue;
    };

// ============================================================================
// TerminationInput
// ============================================================================

export interface TerminationInput {
  session_id: SessionId;
  task_id: TaskId;
  turn_number: number;
  agent_claims_done: boolean;
  agent_response?: string | null;
  budget_used: BudgetSnapshot;
  budget_limits: BudgetLimits;
  sensor_results: SensorResult[];
  session_state: SessionStateSnapshot;
}

// ============================================================================
// CompletionCheck
// ============================================================================

/**
 * Pluggable domain-specific completion check. Injected at construction —
 * the policy is otherwise domain-agnostic.
 *
 * Returns `null` if complete, or a string reason if not yet done. The
 * harness injects the reason into the next turn's context when non-null.
 */
export interface CompletionCheck {
  check(state: SessionStateSnapshot, signal?: AbortSignal): Promise<string | null>;
  description(): string;
}

/** Always-complete check. Causes the policy to halt with success the moment
 *  the agent claims done. */
export class NullCompletionCheck implements CompletionCheck {
  async check(_state: SessionStateSnapshot): Promise<string | null> {
    return null;
  }
  description(): string {
    return "null (always complete)";
  }
}

/** Test/fixture completion check that returns a configured outcome. */
export class FixedCompletionCheck implements CompletionCheck {
  private constructor(
    private readonly outcome: string | null,
    private readonly label: string,
  ) {}

  static complete(): FixedCompletionCheck {
    return new FixedCompletionCheck(null, "fixed:complete");
  }

  static incomplete(reason: string): FixedCompletionCheck {
    return new FixedCompletionCheck(reason, "fixed:incomplete");
  }

  async check(_state: SessionStateSnapshot): Promise<string | null> {
    return this.outcome;
  }
  description(): string {
    return this.label;
  }
}

// ============================================================================
// TerminationPolicy interface
// ============================================================================

export interface TerminationPolicy {
  evaluate(input: TerminationInput, signal?: AbortSignal): Promise<TerminationDecision>;

  /** Cheap budget poll — does not require sensor results or a completion
   *  check. Returns a `halt_budget_exceeded` decision if any limit is
   *  breached, otherwise `null`. */
  checkBudget(snapshot: BudgetSnapshot, limits: BudgetLimits): TerminationDecision | null;
}

// ============================================================================
// checkBudgetDefault — exposed for direct use by the harness loop.
// ============================================================================

/**
 * Default budget check used by {@link StandardTerminationPolicy} and
 * exposed as a free function so callers can poll it cheaply every turn
 * before assembling the rest of {@link TerminationInput}.
 *
 * Limit order mirrors Rust: turns, input_tokens, output_tokens, wall_time,
 * cost_usd. The first breach short-circuits.
 */
export function checkBudgetDefault(
  snapshot: BudgetSnapshot,
  limits: BudgetLimits,
): TerminationDecision | null {
  if (limits.max_turns != null && snapshot.turns >= limits.max_turns) {
    return {
      kind: "halt_budget_exceeded",
      limit_type: "turns",
      used: BudgetValue.turns(snapshot.turns),
      limit: BudgetValue.turns(limits.max_turns),
    };
  }
  if (limits.max_input_tokens != null && snapshot.input_tokens >= limits.max_input_tokens) {
    return {
      kind: "halt_budget_exceeded",
      limit_type: "input_tokens",
      used: BudgetValue.tokens(snapshot.input_tokens),
      limit: BudgetValue.tokens(limits.max_input_tokens),
    };
  }
  if (limits.max_output_tokens != null && snapshot.output_tokens >= limits.max_output_tokens) {
    return {
      kind: "halt_budget_exceeded",
      limit_type: "output_tokens",
      used: BudgetValue.tokens(snapshot.output_tokens),
      limit: BudgetValue.tokens(limits.max_output_tokens),
    };
  }
  if (limits.max_wall_time != null) {
    const used = snapshot.wall_time ?? 0;
    if (used >= limits.max_wall_time) {
      return {
        kind: "halt_budget_exceeded",
        limit_type: "wall_time",
        used: BudgetValue.duration(used),
        limit: BudgetValue.duration(limits.max_wall_time),
      };
    }
  }
  if (limits.max_cost_usd != null && snapshot.cost_usd >= limits.max_cost_usd) {
    return {
      kind: "halt_budget_exceeded",
      limit_type: "cost_usd",
      used: BudgetValue.usd(snapshot.cost_usd),
      limit: BudgetValue.usd(limits.max_cost_usd),
    };
  }
  return null;
}
