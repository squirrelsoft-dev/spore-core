/**
 * `ToolExecutionError` — typed error class for tool implementations.
 *
 * Mirrors `rust/crates/spore-core/src/tools/error.rs`. Tools convert these
 * to {@link ToolOutput} via {@link toolExecutionErrorToOutput} so the registry
 * stays in its happy path and never sees thrown exceptions.
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
      return {
        kind: "error",
        message: `sandbox violation: ${e.violation.kind}`,
        recoverable: false,
      };
    case "timeout":
      return {
        kind: "error",
        message: `timed out after ${Math.round(e.afterMs / 1000)}s`,
        recoverable: true,
      };
  }
}
