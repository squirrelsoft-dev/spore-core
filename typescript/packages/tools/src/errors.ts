/**
 * `ToolExecutionError` — typed error class for tool implementations.
 *
 * Mirrors `rust/crates/spore-core/src/tools/error.rs`. Tools convert these
 * to {@link ToolOutput} via {@link toolExecutionErrorToOutput} so the registry
 * stays in its happy path and never sees thrown exceptions.
 *
 * Error → {@link ToolOutput} mapping:
 *   - `invalid_parameters` → `error { recoverable: true }`
 *   - `execution_failed`   → `error { recoverable }` (as given)
 *   - `sandbox_violation`  → `sandbox_violation { violation }` (TYPED, not
 *     flattened — the HARNESS decides recoverable-vs-halt via its
 *     `SandboxViolationPolicy`, by default recoverable; the boundary still holds
 *     either way)
 *   - `timeout`            → `error { recoverable: true }`
 */

import type { SandboxViolation, ToolOutput } from "@spore/core";

export type ToolExecutionError =
  | { kind: "invalid_parameters"; reason: string }
  | { kind: "execution_failed"; reason: string; recoverable: boolean }
  | { kind: "sandbox_violation"; violation: SandboxViolation }
  | { kind: "timeout"; afterMs: number };

export function toolExecutionErrorToOutput(e: ToolExecutionError): ToolOutput {
  switch (e.kind) {
    case "invalid_parameters":
      return {
        kind: "error",
        message: `invalid parameters: ${e.reason}`,
        recoverable: true,
      };
    case "execution_failed":
      return { kind: "error", message: e.reason, recoverable: e.recoverable };
    case "sandbox_violation":
      // Carry the TYPED violation to the harness, which applies the configured
      // `SandboxViolationPolicy` (recoverable feedback by default; halt on
      // opt-in). The tool does NOT pre-decide recoverability — keeping the
      // violation typed all the way to the harness is what makes the policy
      // uniform across every tool and both surfacing paths (this conversion and
      // the pre-dispatch `validate` check). See `harness/types.ts`.
      return { kind: "sandbox_violation", violation: e.violation };
    case "timeout":
      return {
        kind: "error",
        message: `timed out after ${Math.round(e.afterMs / 1000)}s`,
        recoverable: true,
      };
  }
}
