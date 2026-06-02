/**
 * `CompleteOnFinalResponse` — the harness-loop {@link TerminationPolicy} that
 * lets the loop complete as soon as the agent produces a final response (always
 * `continue`, which the harness interprets as "accept the final response and
 * succeed").
 *
 * This is the default policy wired by
 * {@link "./standard.js".HarnessBuilder.conversational} — a tool-less chat agent
 * halts naturally on its first final response, with no extra completion criteria
 * to satisfy.
 *
 * Mirrors `rust/crates/spore-core/src/harness.rs` `CompleteOnFinalResponse`.
 */

import type {
  BudgetSnapshot,
  SessionState,
  TerminationDecision,
  TerminationPolicy,
} from "./types.js";

export class CompleteOnFinalResponse implements TerminationPolicy {
  async evaluate(
    _session: SessionState,
    _budgetUsed: BudgetSnapshot,
  ): Promise<TerminationDecision> {
    return { kind: "continue" };
  }
}
