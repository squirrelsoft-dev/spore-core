/**
 * StandardTerminationPolicy — canonical {@link TerminationPolicy} (spore-core
 * issue #13).
 *
 * Algorithm (see `docs/harness-engineering-concepts.md` § TerminationPolicy):
 *   1. Budget check (unconditional, before everything else).
 *   2. If `!agent_claims_done` ⇒ `continue`.
 *   3. Any sensor result with outcome `halt` ⇒ `halt_failure` carrying
 *      `unrecoverable_sensor_halt`.
 *   4. Run the injected {@link CompletionCheck}:
 *        - `null` (complete) ⇒ `halt_success` with `agent_response` summary
 *          (empty string when absent).
 *        - non-null (reason) ⇒ `continue` — the harness re-injects the
 *          reason into the next turn's context.
 */

import type { BudgetLimits, BudgetSnapshot } from "../harness/types.js";

import {
  NullCompletionCheck,
  checkBudgetDefault,
  type CompletionCheck,
  type TerminationDecision,
  type TerminationInput,
  type TerminationPolicy,
} from "./types.js";

export class StandardTerminationPolicy implements TerminationPolicy {
  constructor(private readonly completion: CompletionCheck) {}

  /** Convenience constructor using {@link NullCompletionCheck}. */
  static withNullCheck(): StandardTerminationPolicy {
    return new StandardTerminationPolicy(new NullCompletionCheck());
  }

  /** The injected completion check. Exposed for diagnostics / observability. */
  completionCheck(): CompletionCheck {
    return this.completion;
  }

  checkBudget(snapshot: BudgetSnapshot, limits: BudgetLimits): TerminationDecision | null {
    return checkBudgetDefault(snapshot, limits);
  }

  async evaluate(input: TerminationInput, signal?: AbortSignal): Promise<TerminationDecision> {
    // 1. Budget — unconditional hard stop.
    const overrun = this.checkBudget(input.budget_used, input.budget_limits);
    if (overrun != null) return overrun;

    // 2. If agent does not claim done, always continue.
    if (!input.agent_claims_done) return { kind: "continue" };

    // 3. Sensor halt overrides any success path.
    const halted = input.sensor_results.find((r) => r.outcome === "halt");
    if (halted != null) {
      return {
        kind: "halt_failure",
        reason: {
          kind: "unrecoverable_sensor_halt",
          sensor_id: halted.sensor_id,
          detail: halted.detail,
        },
      };
    }

    // 4. Domain-specific completion check.
    const reason = await this.completion.check(input.session_state, signal);
    if (reason == null) {
      return { kind: "halt_success", summary: input.agent_response ?? "" };
    }
    return { kind: "continue" };
  }
}
