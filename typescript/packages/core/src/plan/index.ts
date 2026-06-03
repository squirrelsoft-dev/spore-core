/**
 * Public re-exports for the `plan` component (spore-core issue #70).
 *
 * The phase driver (`runPlanPhase`) and the `plan_execute` arm live on
 * {@link "../harness/standard.js".StandardHarness}; this module owns the
 * deterministic, total text→artifact capture step plus the phase error type.
 */

export {
  PLAN_EXECUTE_EXTRAS_KEY,
  PlanPhaseError,
  type PlanArtifact,
  type PlanPhaseErrorKind,
} from "./types.js";
export { capturePlanArtifact, stripCodeFence, type CaptureResult } from "./capture.js";
