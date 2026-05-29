/**
 * Plan phase / plan artifact (spore-core issue #70 — PlanExecute, phase 1 of 2).
 *
 * This module owns the *capture* half of the PlanExecute plan phase: turning a
 * planner model's `final_response` text into a structured {@link PlanArtifact}.
 * The *phase driver* itself (`runPlanPhase`) lives on {@link StandardHarness}
 * because it needs the harness's turn machinery; this module supplies the
 * deterministic, total text→artifact step and the phase error type.
 *
 * # Public surface
 * - {@link PlanArtifact} — re-exported from `../hooks`; the existing,
 *   serializable contract (`{ tasks: string[]; rationale: string }`) that is the
 *   payload of the `on_plan_created` hook (shipped with #69). This issue REUSES
 *   it rather than defining a competing type. It is the contract consumed by
 *   #72 / #59.
 * - {@link PlanPhaseError} — error class for the plan phase, with a discriminant
 *   `kind` field for `switch` exhaustiveness (per CONVENTIONS.md).
 * - {@link capturePlanArtifact} — the model-text → {@link PlanArtifact} capture
 *   function. Deterministic and total: never throws on malformed input; instead
 *   it returns a {@link PlanPhaseError} of kind `unparseable_plan`.
 *
 * # Resolved spec decisions (issue #70 — match the Rust reference)
 * - **Q1 (model routing):** `HarnessConfig.plannerAgent` (optional) plus a
 *   `HarnessBuilder.plannerAgent` setter. When the strategy is `plan_execute`
 *   and `plannerAgent` is set, the plan turn runs on it; otherwise on the
 *   default agent. `plan_model` stays DESCRIPTIVE metadata only.
 * - **Q2 (HITL):** The plan phase ALWAYS runs to completion. It fires
 *   `on_plan_created` synchronously (the hook may rewrite the artifact); the
 *   stored artifact reflects any mutation. No pause / no `waiting_for_human`.
 * - **Q3 (capture grammar):** JSON-in-response. Trim ASCII whitespace; strip a
 *   single leading ``` / ```json fence line and a single trailing ``` fence if
 *   present; parse a JSON object with `tasks` (required array of strings, kept
 *   verbatim, may be empty) and `rationale` (optional string, default `""`).
 *   Any failure → {@link PlanPhaseError} `unparseable_plan`.
 * - **Q4 (terminal RunResult):** After producing, firing `on_plan_created` on,
 *   and storing the artifact, the plan phase hands off to the execute phase
 *   (issue #59), which parses the artifact into a `TaskList` and loops over the
 *   steps. (Historically — before #59 — the `plan_execute` arm halted here with
 *   an `execute_phase_not_implemented` marker; that variant has been removed now
 *   that the execute phase exists.)
 */

export type { PlanArtifact } from "../hooks/index.js";

/**
 * Key under which the produced {@link PlanArtifact} is stored in
 * `SessionState.extras` (serialized JSON value). Stable across all four
 * languages.
 */
export const PLAN_EXECUTE_EXTRAS_KEY = "plan_execute";

/**
 * Errors raised by the plan phase. A domain error class with a discriminant
 * `kind` field (per CONVENTIONS.md error handling). Wire-tagged on `kind` in
 * snake_case to match the Rust `PlanPhaseError` enum.
 */
export type PlanPhaseErrorKind =
  /**
   * The planner's response text could not be parsed into a {@link PlanArtifact}
   * under the Q3 grammar (not valid JSON, not a JSON object, or `tasks` absent /
   * not an array / containing a non-string element).
   */
  | { kind: "unparseable_plan"; message: string }
  /**
   * The plan turn errored or did not produce a `final_response` (e.g. the
   * planner requested a tool call in the one-shot turn — R2 — or the agent
   * returned an error).
   */
  | { kind: "planning_turn_failed"; message: string };

export class PlanPhaseError extends Error {
  override readonly name = "PlanPhaseError";
  readonly kind: PlanPhaseErrorKind["kind"];
  readonly detail: PlanPhaseErrorKind;

  constructor(detail: PlanPhaseErrorKind) {
    super(detail.message);
    this.kind = detail.kind;
    this.detail = detail;
  }

  static unparseablePlan(message: string): PlanPhaseError {
    return new PlanPhaseError({ kind: "unparseable_plan", message });
  }

  static planningTurnFailed(message: string): PlanPhaseError {
    return new PlanPhaseError({ kind: "planning_turn_failed", message });
  }
}
